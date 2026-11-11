import argparse
import csv
import heapq
import os
import random
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra

import ucs_roadnet as base

INF = float("inf")

# Algorithms (all assume unit edge weights: distance = number of hops)
# Each returns (cost, path, nodes_expanded, explored_groups)
# explored_groups is only filled when trace=True (used for the map pictures)

def _path(parent, u):
    p = []
    while u != -1:
        p.append(u)
        u = parent[u]
    return p[::-1]

# UCS Search /// Expands the node with the smallest cost so far, g(n).
def ucs_search(indptr, indices, s, t, trace=False):
    dist, parent, pq = {s: 0}, {s: -1}, [(0, s)]
    settled, order, expanded = set(), [], 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in settled:
            continue
        settled.add(u)
        expanded += 1
        if trace:
            order.append(u)
        if u == t:
            return d, _path(parent, u), expanded, [order]
        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            if v in settled:
                continue
            nd = d + 1
            if nd < dist.get(v, INF):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return None, [], expanded, [order]

# Bidirectional BFS

# One search starts from the start, one from goal
# Always expands the smaller frontier, one full level at a time
# After a level in which the searches touched, we stop.
# The best meeting found in that level is exact
def bidirectional_bfs(indptr, indices, s, t, trace=False):
    if s == t:
        return 0, [s], 1, [[s], []]
    dist = [{s: 0}, {t: 0}]
    parent = [{s: -1}, {t: -1}]
    frontier = [[s], [t]]
    order = [[], []]
    best, meet, expanded = INF, None, 0

    while frontier[0] and frontier[1]:
        side = 0 if len(frontier[0]) <= len(frontier[1]) else 1
        mine, other, par = dist[side], dist[1 - side], parent[side]
        new = []
        for u in frontier[side]:
            expanded += 1
            if trace:
                order[side].append(u)
            du = mine[u]
            for i in range(indptr[u], indptr[u + 1]):
                v = indices[i]
                if v not in mine:
                    mine[v] = du + 1
                    par[v] = u
                    new.append(v)
                ov = other.get(v)
                if ov is not None:
                    total = du + 1 + ov
                    if total < best:
                        best, meet = total, v
        frontier[side] = new
        if best < INF:
            break

    if meet is None:
        return None, [], expanded, order
    left = _path(parent[0], meet)                 # start ... meet
    right, x = [], parent[1][meet]                # meet's backward parent ... goal
    while x != -1:
        right.append(x)
        x = parent[1][x]
    return best, left + right, expanded, order

# A* with Triangular Inequality (A* + ATL)

# A* search with the ALT heuristic.
# lm = list of landmark distance tables (lm[i][n] = hops from landmark i to node n).
# h(n) = max over landmarks of |d(L,t) - d(L,n)|   (never overestimates, so A* stays exact)
def astar_alt(indptr, indices, lm, s, t, trace=False):
    k = len(lm)
    dt = [lm[i][t] for i in range(k)]

    def h(n):
        b = 0
        for i in range(k):
            x = dt[i] - lm[i][n]
            if x < 0:
                x = -x
            if x > b:
                b = x
        return b

    g, parent = {s: 0}, {s: -1}
    pq = [(h(s), 0, s)]                           # (f, -g, node): ties prefer larger g
    settled, order, expanded = set(), [], 0
    while pq:
        f, neg_g, u = heapq.heappop(pq)
        if u in settled:
            continue
        settled.add(u)
        expanded += 1
        if trace:
            order.append(u)
        gu = -neg_g
        if u == t:
            return gu, _path(parent, u), expanded, [order]
        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            if v in settled:
                continue
            ng = gu + 1
            if ng < g.get(v, INF):
                g[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + h(v), -ng, v))
    return None, [], expanded, [order]

# Landmarks (preprocessing for ALT)
# Hop distance from src to every node (scipy, unweighted). Graph must be connected.
def bfs_distances(A, src):
    d = dijkstra(A, directed=True, indices=int(src), unweighted=True)
    if not np.isfinite(d).all():
        raise ValueError("Graph is not connected; use the largest component only.")
    return d.astype(np.int32)

# strategy 'random'   : k random nodes
# strategy 'farthest' : start from the node farthest from a random node, then
#                       repeatedly add the node farthest from all chosen landmarks.
# Returns (landmark node ids, distance table D of shape (k, n)).
# Both strategies are prefix-consistent: the first j rows are the j-landmark set.
def build_landmarks(A, k, strategy, seed):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    D = np.empty((k, n), dtype=np.int32)
    if strategy == "random":
        picks = rng.choice(n, size=k, replace=False)
        for i, p in enumerate(picks):
            D[i] = bfs_distances(A, p)
        return picks, D

    mind = bfs_distances(A, int(rng.integers(n)))          # only used to seed the choice
    cur, picks = int(mind.argmax()), []
    for i in range(k):
        D[i] = bfs_distances(A, cur)
        picks.append(cur)
        mind = D[i].copy() if i == 0 else np.minimum(mind, D[i])
        cur = int(mind.argmax())
    return np.array(picks), D

# Zero-copy row views that index quickly from plain Python.
def as_views(D):
    return [memoryview(D[i]) for i in range(D.shape[0])]

# Benchmark helpers
# Same sampling as ucs_roadnet.py, so the queries are identical.
def sample_queries(n, count, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        s, t = rng.randrange(n), rng.randrange(n)
        while t == s:
            t = rng.randrange(n)
        out.append((s, t))
    return out


def run_queries(name, fn, queries, every=50):
    rows = []
    for q, (s, t) in enumerate(queries):
        t0 = time.perf_counter()
        cost, path, expanded, _ = fn(s, t)
        rows.append({"cost": cost, "hops": len(path) - 1,
                     "expanded": expanded, "time": time.perf_counter() - t0})
        if (q + 1) % every == 0:
            print(f"  {name}: {q + 1}/{len(queries)}")
    return rows


def load_ucs_csv(path, queries):
    """Read earlier UCS results; accept them only if the queries are identical."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ok = all((int(r["source"]), int(r["target"])) == queries[int(r["query"])]
             for r in rows if int(r["query"]) < len(queries))
    if not ok:
        print("WARNING: queries in the CSV differ from the generated ones "
              "(different seed or graph?). Ignoring the CSV.")
        return None
    out = {}
    for r in rows:
        q = int(r["query"])
        if q < len(queries):
            out[q] = {"cost": int(r["cost"]), "hops": int(r["hops"]),
                      "expanded": int(r["nodes_expanded"]), "time": float(r["time_sec"])}
    print(f"Re-using {len(out)} UCS results from {path} (queries verified identical).")
    return out


def stats(rows, idx=None):
    sel = rows if idx is None else [rows[i] for i in idx]
    t = np.array([r["time"] for r in sel])
    e = np.array([r["expanded"] for r in sel])
    return {"t_mean": t.mean(), "t_med": np.median(t), "t_p95": np.percentile(t, 95),
            "e_mean": e.mean(), "e_med": np.median(e)}

# Outputs

def save_plots(table, all_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in table]
    colors = {"UCS": "#f4a261", "Bidirectional BFS": "#4361ee", "A* + ALT": "#2a9d8f"}
    cols = [colors.get(n, "#888") for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

    ax = axes[0, 0]
    vals = [r["e_mean"] for r in table]
    ax.bar(names, vals, color=cols)
    ax.set_yscale("log")
    ax.set_ylabel("Mean nodes expanded (log)")
    ax.set_title("Search effort per query (lower is better)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 1]
    vals = [r["t_mean"] for r in table]
    ax.bar(names, vals, color=cols)
    ax.set_yscale("log")
    ax.set_ylabel("Mean query time, seconds (log)")
    ax.set_title("Query time (preprocessing not included)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}s", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    for name, rows in all_rows.items():
        ax.scatter([r["hops"] for r in rows], [r["expanded"] for r in rows],
                   s=9, alpha=0.55, color=colors.get(name, "#888"), label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Shortest path length (hops)")
    ax.set_ylabel("Nodes expanded (log)")
    ax.set_title("Effort vs path length")
    ax.legend()

    ax = axes[1, 1]
    base_row = table[0]
    sp = [base_row["e_mean_common"] / r["e_mean_common"] for r in table]
    ax.bar(names, sp, color=cols)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylabel("Reduction in nodes expanded vs UCS (x)")
    ax.set_title("How many times fewer nodes than UCS")
    for i, v in enumerate(sp):
        ax.text(i, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def landmark_study(A, indptr, indices, queries, ks, seed, outdir):
    """Effect of the number of landmarks and the selection strategy on A* + ALT."""
    kmax = max(ks)
    results = []
    for strategy in ("random", "farthest"):
        print(f"\nLandmark study: building {kmax} '{strategy}' landmarks ...")
        t0 = time.perf_counter()
        _, D = build_landmarks(A, kmax, strategy, seed)
        per_lm = (time.perf_counter() - t0) / kmax
        for k in ks:
            lm = as_views(D[:k])
            rows = run_queries(f"{strategy} k={k}",
                               lambda s, t: astar_alt(indptr, indices, lm, s, t),
                               queries, every=10 ** 9)
            st = stats(rows)
            results.append({"strategy": strategy, "landmarks": k,
                            "mean_expanded": round(st["e_mean"]),
                            "mean_time_s": round(st["t_mean"], 4),
                            "preprocessing_s": round(per_lm * k, 1),
                            "memory_mb": round(k * A.shape[0] * 4 / 1e6, 1)})
            print(f"  {strategy:9s} k={k:3d}: mean expanded {st['e_mean']:>10,.0f}, "
                  f"mean time {st['t_mean']:.3f}s")

    with open(os.path.join(outdir, "landmark_study.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    for strategy, c in (("random", "#e76f51"), ("farthest", "#2a9d8f")):
        sel = [r for r in results if r["strategy"] == strategy]
        axes[0].plot([r["landmarks"] for r in sel], [r["mean_expanded"] for r in sel],
                     "o-", color=c, label=strategy)
        axes[1].plot([r["landmarks"] for r in sel], [r["mean_time_s"] for r in sel],
                     "o-", color=c, label=strategy)
    axes[0].set_ylabel("Mean nodes expanded")
    axes[1].set_ylabel("Mean query time (s)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.set_xlabel("Number of landmarks")
        ax.grid(alpha=0.3)
        ax.legend(title="selection")
    axes[0].set_title("More landmarks, less search")
    axes[1].set_title("...but each query also costs more per node")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "landmark_study.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved landmark_study.csv and landmark_study.png in {outdir}")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42, help="use the SAME seed as ucs_roadnet.py")
    ap.add_argument("--landmarks", type=int, default=16)
    ap.add_argument("--landmark-strategy", choices=["farthest", "random"], default="farthest")
    ap.add_argument("--ucs-csv", default=None, help="ucs_results.csv from ucs_roadnet.py")
    ap.add_argument("--ucs-queries", type=int, default=30,
                    help="if no CSV: run UCS itself on this many queries")
    ap.add_argument("--landmark-study", action="store_true")
    ap.add_argument("--study-queries", type=int, default=100)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--demo", action="store_true", help="synthetic graph, for testing")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- data ----------
    if args.demo:
        raw, verify = base.make_demo_edges(side=150), False
    else:
        raw, verify = base.load_edges(base.download_if_missing()), True
    A, _, _ = base.build_graph(raw, "unit", args.seed, verify)
    n = A.shape[0]
    indptr, indices = A.indptr.tolist(), A.indices.tolist()
    queries = sample_queries(n, args.queries, args.seed)

    # ---------- UCS: reuse CSV or run on a subset ----------
    ucs_rows = {}
    if args.ucs_csv and os.path.exists(args.ucs_csv):
        loaded = load_ucs_csv(args.ucs_csv, queries)
        if loaded:
            ucs_rows = loaded
    if not ucs_rows:
        m = min(args.ucs_queries, len(queries))
        print(f"\nRunning UCS on the first {m} queries ...")
        rows = run_queries("UCS", lambda s, t: ucs_search(indptr, indices, s, t),
                           queries[:m], every=10)
        ucs_rows = {i: r for i, r in enumerate(rows)}
    common = sorted(ucs_rows)                                  # queries where UCS has data

    # ---------- Bidirectional BFS ----------
    print("\nRunning Bidirectional BFS ...")
    bi_rows = run_queries("Bidirectional BFS",
                          lambda s, t: bidirectional_bfs(indptr, indices, s, t), queries)

    # ---------- A* + ALT ----------
    print(f"\nPreprocessing: {args.landmarks} '{args.landmark_strategy}' landmarks ...")
    t0 = time.perf_counter()
    _, D = build_landmarks(A, args.landmarks, args.landmark_strategy, args.seed)
    pre_time = time.perf_counter() - t0
    lm = as_views(D)
    print(f"Landmark preprocessing: {pre_time:.1f}s, memory {D.nbytes / 1e6:.0f} MB")
    print("\nRunning A* + ALT ...")
    alt_rows = run_queries("A* + ALT", lambda s, t: astar_alt(indptr, indices, lm, s, t), queries)

    # ---------- correctness ----------
    agree_bi_alt = sum(b["cost"] == a["cost"] for b, a in zip(bi_rows, alt_rows))
    agree_ucs = sum(ucs_rows[i]["cost"] == bi_rows[i]["cost"] == alt_rows[i]["cost"] for i in common)

    # ---------- summary table ----------
    ucs_list = [ucs_rows[i] for i in common]
    ucs_st = stats(ucs_list)
    table = []
    spec = [("UCS", ucs_list, ucs_list, 0.0, 0.0),
            ("Bidirectional BFS", bi_rows, [bi_rows[i] for i in common], 0.0, 0.0),
            ("A* + ALT", alt_rows, [alt_rows[i] for i in common], pre_time, D.nbytes / 1e6)]
    for name, rows, rows_common, pre, mem in spec:
        st = stats(rows)                       # all queries this algorithm ran
        st_common = stats(rows_common)         # only the queries UCS also ran (fair speed-up)
        table.append({"name": name, "queries": len(rows), "pre": pre, "mem": mem,
                      "e_mean_common": st_common["e_mean"], "t_mean_common": st_common["t_mean"],
                      **st})
    for r in table:
        r["speed_t"] = table[0]["t_mean_common"] / r["t_mean_common"]
        r["speed_e"] = table[0]["e_mean_common"] / r["e_mean_common"]

    print("\n" + "=" * 100)
    print(f"{'Algorithm':<20}{'Queries':>8}{'Preproc(s)':>12}{'Extra MB':>10}{'Mean t(s)':>11}"
          f"{'P95 t(s)':>10}{'Mean expanded':>15}{'% graph':>9}{'Fewer nodes':>13}")
    print("-" * 100)
    for r in table:
        print(f"{r['name']:<20}{r['queries']:>8}{r['pre']:>12.1f}{r['mem']:>10.0f}"
              f"{r['t_mean']:>11.3f}{r['t_p95']:>10.3f}{r['e_mean']:>15,.0f}"
              f"{100 * r['e_mean'] / n:>8.1f}%{r['speed_e']:>12.1f}x")
    print("=" * 100)
    print(f"Costs agree: Bidirectional vs ALT {agree_bi_alt}/{len(queries)}; "
          f"all three vs UCS {agree_ucs}/{len(common)}")
    print("'Fewer nodes' compares each method with UCS on the same queries.")

    # ---------- files ----------
    with open(os.path.join(args.outdir, "comparison_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "source", "target", "hops", "ucs_expanded", "ucs_time",
                    "bidir_expanded", "bidir_time", "alt_expanded", "alt_time"])
        for i, (s, t) in enumerate(queries):
            u = ucs_rows.get(i)
            w.writerow([i, s, t, alt_rows[i]["hops"],
                        u["expanded"] if u else "", round(u["time"], 5) if u else "",
                        bi_rows[i]["expanded"], round(bi_rows[i]["time"], 5),
                        alt_rows[i]["expanded"], round(alt_rows[i]["time"], 5)])
    with open(os.path.join(args.outdir, "comparison_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "queries", "preprocessing_s", "extra_memory_mb", "mean_time_s",
                    "median_time_s", "p95_time_s", "mean_expanded", "median_expanded",
                    "percent_of_graph", "times_fewer_nodes_than_ucs", "times_faster_than_ucs"])
        for r in table:
            w.writerow([r["name"], r["queries"], round(r["pre"], 1), round(r["mem"]),
                        round(r["t_mean"], 4), round(r["t_med"], 4), round(r["t_p95"], 4),
                        round(r["e_mean"]), round(r["e_med"]), round(100 * r["e_mean"] / n, 2),
                        round(r["speed_e"], 2), round(r["speed_t"], 2)])
    print(f"Saved comparison_results.csv and comparison_summary.csv in {args.outdir}")

    try:
        all_rows = {"UCS": ucs_list, "Bidirectional BFS": bi_rows, "A* + ALT": alt_rows}
        save_plots(table, all_rows, os.path.join(args.outdir, "comparison_plots.png"))
    except ImportError:
        print("matplotlib not installed: skipping plots")

    if args.landmark_study:
        landmark_study(A, indptr, indices, queries[:args.study_queries],
                       [4, 8, 16, 32], args.seed, args.outdir)


if __name__ == "__main__":
    main()