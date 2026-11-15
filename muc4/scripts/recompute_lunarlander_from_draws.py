#!/usr/bin/env python3
"""Phase 2: recompute LunarLander CI decisions + edge weights from SAVED draws.

Pure numpy -- no R, no BART refit. Reads the per-target .npz files saved
by run_lunarlander_ci_sensitivity.py (500 ITE posterior draws per
(target, treatment-column) pair + the X_est + the HTE cluster labels
computed in-process) and:

1. Re-derives the HTE decision at 90%/95%/99% from the saved draws +
   saved cluster labels (pooled per-cluster quantiles, edge iff any
   cluster CI excludes 0) and checks it against the in-process
   decisions in lunarlander_ci_sensitivity.json. Any difference means
   the saved artifacts do not reproduce the run and must be reported.
2. Computes edge weights for every (target, column) pair from the
   saved draws:
   * mean_abs_ite: mean of |ITE| over all 500 draws x N points.
   * ci95_mean_abs_ite: quantiles of the per-draw mean |ITE| (500
     values) at 2.5%/97.5% -- a posterior interval for the mean
     absolute effect. (90%/99% analogues included.)
   * mean_ite (signed) + ci95 of the per-draw mean ITE, and the
     fraction of (draw, point) cells with ITE > 0, for direction.
   * margin95: for the pooled 95% cluster CIs, the largest exclusion
     distance from 0 across clusters (0 when a cluster CI covers 0).
     A 95% edge with a small margin rests on a cluster whose CI barely
     excludes 0.
   CAVEAT (also stated in edge_weights.md): the weight is mean |ITE|
   from BART on standardised inputs -- NOT a causal coefficient with
   physical units, and NOT comparable across targets/columns with
   different outcome scales. BART has run-to-run noise; weights from
   one fit are descriptive, not population values.
3. Compares the 95% edge set with the Muc 3 graph (metrics.json
   bart_targets) for context only -- a different unseeded fit, so
   differences are expected BART/split noise, not errors.

Usage:
    python3 recompute_lunarlander_from_draws.py \
        --draws-dir <dir with target_*.npz> \
        --sensitivity-json <lunarlander_ci_sensitivity.json> \
        --muc3-metrics <muc3/results/lunarlander/metrics.json> \
        --output-dir <dir>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FEATURES = [
    "x", "y", "vx", "vy", "angle", "angular_velocity",
    "left_contact", "right_contact",
    "do_nothing", "fire_left_engine", "fire_main_engine", "fire_right_engine",
]
N_STATES = 8
CI_LEVELS = (0.90, 0.95, 0.99)


def edge_name(c, i_repr):
    prefix_t = "action" if c >= N_STATES else "state"
    prefix_tp1 = "action" if i_repr >= N_STATES else "state"
    return f"{prefix_t}_{FEATURES[c]}_t -> {prefix_tp1}_{FEATURES[i_repr]}_t+1"


def pooled_decisions(draws, clusters):
    """Per-level HTE decision + per-cluster CIs from saved draws/labels."""
    out, cis = {}, {}
    for level in CI_LEVELS:
        alpha = 1.0 - level
        sig, cis_l = {}, {}
        for cid in np.unique(clusters):
            pooled = draws[:, clusters == cid]
            lci, uci = np.quantile(pooled, [alpha / 2.0, 1.0 - alpha / 2.0])
            sig[int(cid)] = bool(lci > 0 or uci < 0)
            cis_l[int(cid)] = (float(lci), float(uci))
        out[level] = any(sig.values())
        cis[level] = cis_l
    return out, cis


def pair_weights(draws):
    """Edge-weight summary from one (500, N) ITE draw matrix."""
    abs_draws = np.abs(draws)
    per_draw_mean_abs = abs_draws.mean(axis=1)
    per_draw_mean = draws.mean(axis=1)
    out = {
        "mean_abs_ite": float(abs_draws.mean()),
        "mean_ite": float(draws.mean()),
        "frac_cells_gt0": float((draws > 0).mean()),
        "n_draws": int(draws.shape[0]),
        "n_points": int(draws.shape[1]),
    }
    for level in CI_LEVELS:
        alpha = 1.0 - level
        lo, hi = np.quantile(per_draw_mean_abs,
                             [alpha / 2.0, 1.0 - alpha / 2.0])
        out[f"ci{int(level*100)}_mean_abs_ite"] = [float(lo), float(hi)]
        lo, hi = np.quantile(per_draw_mean,
                             [alpha / 2.0, 1.0 - alpha / 2.0])
        out[f"ci{int(level*100)}_mean_ite"] = [float(lo), float(hi)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws-dir", type=Path, required=True)
    ap.add_argument("--sensitivity-json", type=Path, required=True)
    ap.add_argument("--muc3-metrics", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    sens = json.loads(args.sensitivity_json.read_text())
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = {}
    recompute_mismatches = []
    n_checked = 0
    for i_repr in range(len(FEATURES)):
        npz_path = args.draws_dir / f"target_{i_repr}.npz"
        z = np.load(npz_path)
        draws_all = z["ite_draws"]  # (12, 500, N)
        clusters_all = z["clusters"]  # (12, N)
        assert draws_all.shape[0] == len(FEATURES)
        assert draws_all.shape[1] == 500, draws_all.shape
        for c in range(len(FEATURES)):
            draws = draws_all[c]
            clusters = clusters_all[c]
            decisions, cis = pooled_decisions(draws, clusters)
            # margin95: strongest cluster exclusion distance from 0.
            margin95 = 0.0
            n_sig95 = 0
            for cid, (lci, uci) in cis[0.95].items():
                if lci > 0 or uci < 0:
                    n_sig95 += 1
                    margin95 = max(margin95, min(abs(lci), abs(uci)))
            w = pair_weights(draws)
            # Compare with the in-process decision.
            detail = sens["targets"][FEATURES[i_repr]]["detail"][FEATURES[c]]
            for level in CI_LEVELS:
                n_checked += 1
                if decisions[level] != detail[str(level)]["significant"]:
                    recompute_mismatches.append({
                        "target": FEATURES[i_repr], "column": FEATURES[c],
                        "level": level,
                        "in_process": detail[str(level)]["significant"],
                        "from_saved_draws": bool(decisions[level])})
            pairs[f"{FEATURES[c]} -> {FEATURES[i_repr]}"] = {
                "treatment": FEATURES[c], "target": FEATURES[i_repr],
                "edge": edge_name(c, i_repr),
                "decisions": {str(level): bool(decisions[level])
                              for level in CI_LEVELS},
                "repo_95": bool(detail["repo_95"]),
                "n_clusters": int(len(np.unique(clusters))),
                "n_sig_clusters_95": n_sig95,
                "margin95": round(float(margin95), 4),
                **{k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in w.items()},
                "ci95_mean_abs_ite": [round(v, 4) for v in
                                      w["ci95_mean_abs_ite"]],
                "ci95_mean_ite": [round(v, 4) for v in w["ci95_mean_ite"]],
            }
            # Keep full precision for the 90/99 CIs too (rounded above
            # only for the two 95% entries re-written here).
            pairs[f"{FEATURES[c]} -> {FEATURES[i_repr]}"].update({
                "ci90_mean_abs_ite": [round(v, 4) for v in
                                      w["ci90_mean_abs_ite"]],
                "ci99_mean_abs_ite": [round(v, 4) for v in
                                      w["ci99_mean_abs_ite"]],
                "ci90_mean_ite": [round(v, 4) for v in w["ci90_mean_ite"]],
                "ci99_mean_ite": [round(v, 4) for v in w["ci99_mean_ite"]],
            })

    edges95 = sorted(p["edge"] for p in pairs.values()
                     if p["decisions"]["0.95"])
    edges90 = sorted(p["edge"] for p in pairs.values()
                     if p["decisions"]["0.9"])
    edges99 = sorted(p["edge"] for p in pairs.values()
                     if p["decisions"]["0.99"])
    ranked = sorted(
        (p for p in pairs.values() if p["decisions"]["0.95"]),
        key=lambda p: -p["mean_abs_ite"])
    non_edges = sorted(
        (p for p in pairs.values() if not p["decisions"]["0.95"]),
        key=lambda p: -p["mean_abs_ite"])

    muc3 = None
    if args.muc3_metrics and args.muc3_metrics.is_file():
        m = json.loads(args.muc3_metrics.read_text())
        muc3_edges = set()
        for t in m["bart_targets"]:
            ti = t["target_index"]
            for ci in t["cause_indices"]:
                muc3_edges.add(edge_name(ci, ti))
        ours = set(edges95)
        muc3 = {
            "n_edges": len(muc3_edges),
            "ours_n_edges": len(ours),
            "jaccard": round(len(ours & muc3_edges) / len(ours | muc3_edges), 4)
                       if (ours | muc3_edges) else 1.0,
            "only_ours": sorted(ours - muc3_edges),
            "only_muc3": sorted(muc3_edges - ours),
            "note": ("Context only: Muc 3 is a different unseeded fit "
                     "(isolated process per target, no split seed). "
                     "Differences are expected BART/split noise."),
        }

    result = {
        "source": "saved 500-draw ITE matrices, no refit",
        "n_pairs": len(pairs),
        "recompute_check": {
            "n_checked": n_checked,
            "n_mismatches": len(recompute_mismatches),
            "mismatches": recompute_mismatches,
        },
        "edges": {"0.9": edges90, "0.95": edges95, "0.99": edges99,
                  "n_90": len(edges90), "n_95": len(edges95),
                  "n_99": len(edges99)},
        "ranked_95_edges": [
            {"rank": k + 1, "edge": p["edge"],
             "mean_abs_ite": p["mean_abs_ite"],
             "ci95_mean_abs_ite": p["ci95_mean_abs_ite"],
             "mean_ite": p["mean_ite"],
             "ci95_mean_ite": p["ci95_mean_ite"],
             "survives_99": p["decisions"]["0.99"],
             "n_sig_clusters_95": p["n_sig_clusters_95"],
             "margin95": p["margin95"]}
            for k, p in enumerate(ranked)],
        "strongest_non_edges_95": [
            {"edge": p["edge"], "mean_abs_ite": p["mean_abs_ite"],
             "ci95_mean_abs_ite": p["ci95_mean_abs_ite"],
             "appears_at_90": p["decisions"]["0.9"]}
            for p in non_edges[:10]],
        "muc3_context": muc3,
        "pairs": pairs,
    }
    (out_dir / "edge_weights.json").write_text(json.dumps(result, indent=2))
    print(f"pairs={len(pairs)} edges 90/95/99 = "
          f"{len(edges90)}/{len(edges95)}/{len(edges99)}; "
          f"recompute mismatches: {len(recompute_mismatches)}/{n_checked}")
    if recompute_mismatches:
        print("MISMATCHES:", json.dumps(recompute_mismatches, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
