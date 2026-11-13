"""Run plain causal-learn GES (standard BIC) on the saved Mục 3 replays.

Protocol notes (read before citing):
  * The authors' eval_time notebooks run GES with a PATCHED BIC score that
    adds a simulated BART-cost block per score evaluation and time ONLY
    that simulation block ("SLA elapsed score-computation time"). That is
    not a real GES runtime. This script follows the precedent set for
    PC/FCI in muc3/results/*/metrics.json: plain algorithm, full
    wall-clock time. GES parameters otherwise match the authors' call
    exactly: score_func="local_score_BIC", maxP=None, parameters=None
    (=> lambda_value 0.5, standard BIC), node_names set.
  * Input matrices match the PC/FCI matrices in the metrics scripts:
    CartPole [states(4), action]; LunarLander states only (8 cols,
    INCLUDE_ACTION=False per the timing notebook); PongDuel the greedy
    numerically-independent 10-column state basis from
    pongduel_metrics.py (same rank procedure, tol=1e-10). No
    standardisation anywhere (same as PC/FCI). GES/BIC assumes
    linear-Gaussian data; binary/one-hot columns (actions, contacts)
    violate that assumption exactly as they do for Fisher-Z PC/FCI --
    recorded as a protocol deviation, not hidden.
  * MicroRTS has no PC/FCI precedent and no timing notebook. Two runs:
    (a) states_t, all 4924 rows, constant column light2 dropped (zero
    variance breaks the BIC regression); (b) acted rows only
    (n_valid>0, 3200 rows), [states_t, action_label], same constant
    drop. Both are contemporaneous [state(, action)] matrices like the
    gym runs, NOT lagged DBN data; the BART global graph (lagged
    differences) is a different object and edge counts are not
    directly comparable.
  * PC/FCI are re-run in the same session/version on the same matrices
    for a same-version timing comparison (times only; the metrics
    scripts discard their graphs, and so does this script). Quoted
    metrics.json times are preserved separately in the output JSON.

Usage (from the repo root, with replay .npz files in place):
    python muc3/scripts/run_ges.py [--microrts-npz PATH] [--outdir DIR]
Output: <env>/ges.json per environment under muc3/results/ (or --outdir).
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

import causallearn


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_edges(record, node_names):
    """Classify GES CPDAG edges from the adjacency matrix.

    causal-learn encoding (ges docstring): G.graph[j, i] == 1 and
    G.graph[i, j] == -1 means i --> j; both -1 means i --- j (undirected).
    Any other endpoint pair is recorded verbatim as 'other'.
    """
    g = record["G"]
    adj = np.asarray(g.graph)
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = int(adj[i, j]), int(adj[j, i])
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1:
                kind, src, dst = "directed", i, j  # i --> j
            elif a == 1 and b == -1:
                kind, src, dst = "directed", j, i  # j --> i
            elif a == -1 and b == -1:
                kind, src, dst = "undirected", i, j
            else:
                kind, src, dst = f"other({a},{b})", i, j
            edges.append({"source": node_names[src], "target": node_names[dst],
                          "source_index": int(src), "target_index": int(dst),
                          "kind": kind})
    return edges


def run_ges(matrix, node_names, reps=3):
    from causallearn.search.ScoreBased.GES import ges
    times, edge_sets, scores = [], [], []
    last_record = None
    for _ in range(reps):
        t0 = time.perf_counter()
        record = ges(matrix, score_func="local_score_BIC", maxP=None,
                     parameters=None, node_names=node_names)
        times.append(time.perf_counter() - t0)
        edges = extract_edges(record, node_names)
        edge_sets.append(json.dumps(edges, sort_keys=True))
        scores.append(float(record["score"]))
        last_record = record
    return {"times_s": times, "median_time_s": float(np.median(times)),
            "edges": extract_edges(last_record, node_names),
            "edge_sets_identical_across_reps": len(set(edge_sets)) == 1,
            "scores": scores}


def run_pc_fci(matrix):
    """Same-session PC/FCI wall-clock on the same matrix (times only)."""
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ConstraintBased.PC import pc
    t0 = time.perf_counter()
    pc(matrix, alpha=0.05, indep_test="fisherz", stable=False, show_progress=False)
    pc_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    fci(matrix, independence_test_method="fisherz", alpha=0.05, verbose=False)
    fci_s = time.perf_counter() - t0
    return {"pc_fisherz_wall_clock_s": pc_s, "fci_fisherz_wall_clock_s": fci_s}


def col_summary(matrix, node_names):
    out = []
    for j, name in enumerate(node_names):
        col = matrix[:, j]
        out.append({"column": name, "index": j, "n_unique": int(len(np.unique(col))),
                    "std": float(np.std(col)), "min": float(np.min(col)),
                    "max": float(np.max(col))})
    return out


def env_versions():
    import importlib.metadata
    import scipy, sklearn
    try:
        cl_version = importlib.metadata.version("causal-learn")
    except Exception:
        cl_version = getattr(causallearn, "__version__", "unknown")
    return {"python": platform.python_version(),
            "platform": platform.platform(),
            "causal_learn": cl_version,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "cpu_count": os.cpu_count()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="repo root (muc3/results/<env>/replay.npz below it)")
    ap.add_argument("--microrts-npz", default=None,
                    help="path to microrts_merged.npz (Drive artifact, not in repo)")
    ap.add_argument("--outdir", default=None,
                    help="output base dir; default <root>/muc3/results")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    results_root = os.path.join(args.root, "muc3", "results")
    out_base = args.outdir or results_root
    versions = env_versions()
    print("versions:", json.dumps(versions), flush=True)

    jobs = []
    # CartPole: [states, action], same as cartpole_metrics.py structure_data.
    p = os.path.join(results_root, "cartpole", "replay.npz")
    d = np.load(p)
    jobs.append({"env": "cartpole", "replay_path": p, "replay_sha256": sha256(p),
                 "matrix": np.concatenate((d["states"], d["actions"].astype(int).reshape(-1, 1)), axis=1).astype(float),
                 "node_names": ["cart_position", "cart_velocity", "pole_angle", "pole_velocity", "push"],
                 "representation": "[states(4), action] float64, no standardisation; identical matrix to cartpole_metrics.py PC/FCI."})
    # LunarLander: states only, INCLUDE_ACTION=False.
    p = os.path.join(results_root, "lunarlander", "replay.npz")
    d = np.load(p)
    jobs.append({"env": "lunarlander", "replay_path": p, "replay_sha256": sha256(p),
                 "matrix": d["states"].astype(float),
                 "node_names": ["x", "y", "vx", "vy", "angle", "angular_velocity", "left_contact", "right_contact"],
                 "representation": "states only (8 cols), no standardisation; INCLUDE_ACTION=False per eval_time_Lunar Lander notebook and lunarlander_metrics.py."})
    # PongDuel: greedy independent state basis, verbatim from pongduel_metrics.py.
    p = os.path.join(results_root, "pongduel", "replay.npz")
    d = np.load(p)
    states = d["states"]
    kept, rank = [], 0
    for column in range(states.shape[1]):
        candidate = states[:, kept + [column]]
        candidate_rank = np.linalg.matrix_rank(candidate - candidate.mean(axis=0), tol=1e-10)
        if candidate_rank > rank:
            kept.append(column)
            rank = candidate_rank
    jobs.append({"env": "pongduel", "replay_path": p, "replay_sha256": sha256(p),
                 "matrix": states[:, kept].astype(float),
                 "node_names": [f"state_{i}" for i in kept],
                 "representation": f"independent state basis, kept columns {kept} by the greedy rank procedure (tol=1e-10) from pongduel_metrics.py; no standardisation.",
                 "kept_columns": kept})
    # MicroRTS (optional): states_t; drop constant columns (zero variance).
    microrts_jobs = []
    if args.microrts_npz:
        d = np.load(args.microrts_npz)
        feats = ["base1", "barrack1", "worker1", "light1", "heavy1", "ranged1",
                 "base2", "barrack2", "worker2", "light2", "heavy2", "ranged2", "timestep"]
        st = d["states_t"].astype(float)
        const_cols = [j for j in range(st.shape[1]) if len(np.unique(st[:, j])) < 2]
        keep = [j for j in range(st.shape[1]) if j not in const_cols]
        microrts_jobs.append({"env": "microrts", "variant": "states_only_all_rows",
                              "matrix": st[:, keep],
                              "node_names": [feats[j] for j in keep],
                              "representation": f"states_t, all {st.shape[0]} rows; dropped constant column(s) {[feats[j] for j in const_cols]} (zero variance; BIC regression undefined). No standardisation.",
                              "dropped_constant_columns": [feats[j] for j in const_cols]})
        acted = np.where(d["n_valid"] > 0)[0]
        lab = d["action_label"].astype(float)
        m2 = np.concatenate([st[acted][:, keep], lab[acted].reshape(-1, 1)], axis=1)
        microrts_jobs.append({"env": "microrts", "variant": "states_plus_action_acted_rows",
                              "matrix": m2,
                              "node_names": [feats[j] for j in keep] + ["action_label"],
                              "representation": f"[states_t, action_label] on acted rows only (n_valid>0, {len(acted)} rows); dropped constant column(s) {[feats[j] for j in const_cols]}. No standardisation.",
                              "dropped_constant_columns": [feats[j] for j in const_cols]})

    for job in jobs:
        m = job["matrix"]
        print(f"[{job['env']}] matrix {m.shape}, running GES x{args.reps} ...", flush=True)
        res = run_ges(m, job["node_names"], reps=args.reps)
        pc_fci = run_pc_fci(m)
        edges = res["edges"]
        out = {"environment": job["env"],
               "method": "GES (causal-learn), score local_score_BIC, maxP=None, parameters=None (lambda_value default 0.5, standard BIC), plain score (NOT the authors' patched simulated-cost score)",
               "date": "2026-11-12", "versions": versions,
               "replay_file": {"path": os.path.relpath(job["replay_path"], args.root),
                               "sha256": job["replay_sha256"]},
               "representation": job["representation"],
               "columns": job["node_names"], "n_rows": int(m.shape[0]),
               "n_columns": int(m.shape[1]),
               "column_summary": col_summary(m, job["node_names"]),
               "standardisation": "none",
               "ges_wall_clock_s": res["times_s"],
               "ges_wall_clock_median_s": res["median_time_s"],
               "ges_score": res["scores"],
               "edge_sets_identical_across_reps": res["edge_sets_identical_across_reps"],
               "n_edges": len(edges),
               "n_directed": sum(1 for e in edges if e["kind"] == "directed"),
               "n_undirected": sum(1 for e in edges if e["kind"] == "undirected"),
               "n_other": sum(1 for e in edges if e["kind"] not in ("directed", "undirected")),
               "edges": edges,
               "same_session_pc_fci_wall_clock_s": pc_fci,
               "protocol_deviations": [
                   "Timing is full GES wall-clock, not the authors' simulated-cost timer in the eval_time notebooks (same precedent as PC/FCI in metrics.json).",
                   "BIC assumes linear-Gaussian data; binary action/contact columns violate it, as they do for Fisher-Z PC/FCI."]}
        if "kept_columns" in job:
            out["kept_columns"] = job["kept_columns"]
        od = os.path.join(out_base, job["env"])
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "ges.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{job['env']}] edges={len(edges)} (dir {out['n_directed']}, undir {out['n_undirected']}), "
              f"median={res['median_time_s']:.3f}s, PC={pc_fci['pc_fisherz_wall_clock_s']:.3f}s, "
              f"FCI={pc_fci['fci_fisherz_wall_clock_s']:.3f}s", flush=True)

    if microrts_jobs:
        variants = []
        for job in microrts_jobs:
            m = job["matrix"]
            print(f"[microrts/{job['variant']}] matrix {m.shape}, running GES x{args.reps} ...", flush=True)
            res = run_ges(m, job["node_names"], reps=args.reps)
            pc_fci = run_pc_fci(m)
            edges = res["edges"]
            variants.append({"variant": job["variant"],
                             "representation": job["representation"],
                             "columns": job["node_names"], "n_rows": int(m.shape[0]),
                             "n_columns": int(m.shape[1]),
                             "column_summary": col_summary(m, job["node_names"]),
                             "standardisation": "none",
                             "ges_wall_clock_s": res["times_s"],
                             "ges_wall_clock_median_s": res["median_time_s"],
                             "ges_score": res["scores"],
                             "edge_sets_identical_across_reps": res["edge_sets_identical_across_reps"],
                             "n_edges": len(edges),
                             "n_directed": sum(1 for e in edges if e["kind"] == "directed"),
                             "n_undirected": sum(1 for e in edges if e["kind"] == "undirected"),
                             "n_other": sum(1 for e in edges if e["kind"] not in ("directed", "undirected")),
                             "edges": edges,
                             "same_session_pc_fci_wall_clock_s": pc_fci,
                             "dropped_constant_columns": job["dropped_constant_columns"]})
            print(f"[microrts/{job['variant']}] edges={len(edges)}, median={res['median_time_s']:.3f}s", flush=True)
        out = {"environment": "microrts",
               "method": "GES (causal-learn), score local_score_BIC, maxP=None, parameters=None (lambda_value default 0.5, standard BIC), plain score",
               "date": "2026-11-12", "versions": versions,
               "source_file": {"path": args.microrts_npz, "sha256": sha256(args.microrts_npz),
                               "note": "microrts_merged.npz, 100 episodes, Drive artifact (not in repo); same file used for the BART global graph and fidelity."},
               "variants": variants,
               "protocol_deviations": [
                   "No PC/FCI precedent or timing notebook exists for MicroRTS; representations follow the gym conventions (states-only like LunarLander; states+action like CartPole).",
                   "Contemporaneous [state(, action)] matrix, NOT the lagged state-difference data of the BART global graph; edge counts are not directly comparable with the BART graph.",
                   "Constant column(s) dropped: zero-variance columns make the BIC regression undefined.",
                   "BIC assumes linear-Gaussian data; binary/count columns violate it, as for Fisher-Z PC/FCI."]}
        od = os.path.join(out_base, "microrts")
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "ges.json"), "w") as f:
            json.dump(out, f, indent=2)
    print("GES DONE", flush=True)


if __name__ == "__main__":
    main()
