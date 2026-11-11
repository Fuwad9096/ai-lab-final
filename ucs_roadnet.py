import argparse
import csv
import gzip
import heapq
import os
import random
import time
import urllib.request
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

URL = "https://snap.stanford.edu/data/roadNet-CA.txt.gz"
FILE = "roadNet-CA.txt.gz"

# Official numbers from the SNAP page (used to verify our loader)
OFFICIAL_NODES = 1_965_206
OFFICIAL_EDGES = 2_766_607
OFFICIAL_WCC_NODES = 1_957_027
OFFICIAL_WCC_EDGES = 2_760_388

# 1. Loading
def download_if_missing(path=FILE):
    if not os.path.exists(path):
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, path)
    return path


def load_edges(path):
    with gzip.open(path, "rt") as f:
        edges = np.loadtxt(f, dtype=np.int64, comments="#")
    return edges


def make_demo_edges(side=300, seed=0):
    rng = np.random.default_rng(seed)
    edges = []
    for r in range(side):
        for c in range(side):
            n = r * side + c
            if c + 1 < side and rng.random() > 0.15:
                edges.append((n, n + 1))
            if r + 1 < side and rng.random() > 0.15:
                edges.append((n, n + side))
    return np.array(edges, dtype=np.int64)


def build_graph(raw_edges, weight_mode="unit", seed=42, verify=True):
    stats = {}
    stats["raw_lines"] = len(raw_edges)

    # self-loops
    loops = raw_edges[:, 0] == raw_edges[:, 1]
    stats["self_loops"] = int(loops.sum())
    edges = raw_edges[~loops]

    # treat graph as undirected: sort each pair, then drop duplicates
    edges = np.sort(edges, axis=1)
    before = len(edges)
    edges = np.unique(edges, axis=0)
    stats["duplicate_edges_removed"] = before - len(edges)

    # remap IDs to 0..N-1
    ids, inv = np.unique(edges.ravel(), return_inverse=True)
    edges = inv.reshape(-1, 2)
    n = len(ids)
    stats["nodes"] = n
    stats["edges"] = len(edges)

    if verify:
        print("\n--- Verification against SNAP page ---")
        print(f"Nodes: {n:,}  (official {OFFICIAL_NODES:,})  match={n == OFFICIAL_NODES}")
        print(f"Edges: {len(edges):,}  (official {OFFICIAL_EDGES:,})  match={len(edges) == OFFICIAL_EDGES}")

    # weights (one weight per undirected edge, same in both directions)
    rng = np.random.default_rng(seed)
    if weight_mode == "random":
        w = rng.integers(1, 11, size=len(edges)).astype(np.int64)
    else:
        w = np.ones(len(edges), dtype=np.int64)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([w, w])
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    # largest connected component
    n_comp, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    biggest = sizes.argmax()
    mask = labels == biggest
    stats["components"] = int(n_comp)
    stats["largest_wcc_nodes"] = int(mask.sum())
    A = A[mask][:, mask].tocsr()
    stats["largest_wcc_edges"] = int(A.nnz // 2)

    if verify:
        print(f"Components: {n_comp:,}")
        print(f"Largest WCC nodes: {stats['largest_wcc_nodes']:,} (official {OFFICIAL_WCC_NODES:,})")
        print(f"Largest WCC edges: {stats['largest_wcc_edges']:,} (official {OFFICIAL_WCC_EDGES:,})")

    degrees = np.diff(A.indptr)
    stats["avg_degree"] = float(degrees.mean())
    stats["max_degree"] = int(degrees.max())
    return A, degrees, stats



# Uniform-Cost Search
def ucs(indptr, indices, weights, src, dst):
    """
    Uniform-Cost Search: always expand the node with the smallest path cost g(n).
    Returns (cost, path, nodes_expanded). Cost is None if no path exists.
    """
    INF = float("inf")
    dist = {src: 0}
    parent = {src: -1}
    pq = [(0, src)]
    settled = set()
    expanded = 0

    while pq:
        d, u = heapq.heappop(pq)
        if u in settled:
            continue                      # stale queue entry
        settled.add(u)
        expanded += 1

        if u == dst:                      # goal test when node is EXPANDED (keeps UCS optimal)
            path = []
            while u != -1:
                path.append(u)
                u = parent[u]
            return d, path[::-1], expanded

        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            if v in settled:
                continue
            nd = d + weights[i]
            if nd < dist.get(v, INF):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return None, [], expanded


def bfs_hops(indptr, indices, src, dst):
    if src == dst:
        return 0
    seen = {src}
    q = deque([(src, 0)])
    while q:
        u, d = q.popleft()
        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            if v not in seen:
                if v == dst:
                    return d + 1
                seen.add(v)
                q.append((v, d + 1))
    return None


# Benchmark
def benchmark(A, n_queries, seed, unit_weights):
    # lists are much faster than numpy arrays for element-by-element loops
    indptr = A.indptr.tolist()
    indices = A.indices.tolist()
    weights = A.data.tolist()
    n = A.shape[0]

    rng = random.Random(seed)
    results = []
    mismatches = 0

    print(f"\nRunning {n_queries} queries (seed={seed}) ...")
    for q in range(n_queries):
        s, t = rng.randrange(n), rng.randrange(n)
        while t == s:
            t = rng.randrange(n)

        t0 = time.perf_counter()
        cost, path, expanded = ucs(indptr, indices, weights, s, t)
        elapsed = time.perf_counter() - t0

        correct = ""
        if unit_weights:
            ref = bfs_hops(indptr, indices, s, t)
            correct = (ref == cost)
            if not correct:
                mismatches += 1

        results.append({
            "query": q, "source": s, "target": t,
            "cost": cost, "hops": len(path) - 1,
            "nodes_expanded": expanded,
            "time_sec": round(elapsed, 5),
            "matches_bfs": correct,
        })
        if (q + 1) % 10 == 0:
            print(f"  {q + 1}/{n_queries} done")

    return results, mismatches


def summarize(results, mismatches, unit_weights):
    t = np.array([r["time_sec"] for r in results])
    e = np.array([r["nodes_expanded"] for r in results])
    c = np.array([r["cost"] for r in results], dtype=float)
    print("\n=========== UCS RESULTS ===========")
    print(f"Queries:               {len(results)}")
    print(f"Query time (s):        mean {t.mean():.3f} | median {np.median(t):.3f} | p95 {np.percentile(t, 95):.3f}")
    print(f"Nodes expanded:        mean {e.mean():,.0f} | median {np.median(e):,.0f} | max {e.max():,}")
    print(f"Path cost:             mean {c.mean():.1f} | max {c.max():.0f}")
    if unit_weights:
        print(f"Correct vs BFS:        {len(results) - mismatches}/{len(results)}")


# Output: CSV + poster plots
def save_csv(results, path="ucs_results.csv"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"Saved {path}")


def make_plots(results, degrees):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots (pip install matplotlib)")
        return

    # Degree distribution
    vals, counts = np.unique(degrees, return_counts=True)
    plt.figure(figsize=(6, 4))
    plt.bar(vals, counts)
    plt.yscale("log")
    plt.xlabel("Degree (roads at an intersection)")
    plt.ylabel("Number of intersections (log)")
    plt.title("Degree distribution")
    plt.tight_layout()
    plt.savefig("plot_degree.png", dpi=200)
    plt.close()

    # Nodes expanded vs path length
    hops = [r["hops"] for r in results]
    exp = [r["nodes_expanded"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.scatter(hops, exp, s=12)
    plt.xlabel("Shortest path length (hops)")
    plt.ylabel("Nodes expanded by UCS")
    plt.title("Search effort vs path length")
    plt.tight_layout()
    plt.savefig("plot_expanded_vs_length.png", dpi=200)
    plt.close()

    # Query time vs path length
    tm = [r["time_sec"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.scatter(hops, tm, s=12)
    plt.xlabel("Shortest path length (hops)")
    plt.ylabel("Query time (s)")
    plt.title("UCS query time vs path length")
    plt.tight_layout()
    plt.savefig("plot_time_vs_length.png", dpi=200)
    plt.close()
    print("Saved plot_degree.png, plot_expanded_vs_length.png, plot_time_vs_length.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", choices=["unit", "random"], default="unit",
                    help="unit = every road has length 1 (default); random = synthetic lengths 1..10")
    ap.add_argument("--demo", action="store_true", help="use a small synthetic graph")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.demo:
        raw = make_demo_edges()
        verify = False
    else:
        raw = load_edges(download_if_missing())
        verify = True
    print(f"Loaded {len(raw):,} raw edges in {time.perf_counter() - t0:.1f}s")

    A, degrees, stats = build_graph(raw, args.weights, args.seed, verify)
    print("\n--- Cleaning report ---")
    for k, v in stats.items():
        print(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v:.3f}")
    print(f"Preprocessing finished in {time.perf_counter() - t0:.1f}s")

    unit = args.weights == "unit"
    results, mismatches = benchmark(A, args.queries, args.seed, unit)
    summarize(results, mismatches, unit)
    save_csv(results)
    make_plots(results, degrees)


if __name__ == "__main__":
    main()