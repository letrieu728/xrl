"""Compare DQN vs heuristic CartPole counterfactuals on identical metrics.

Reads the two result JSONs (DQN run + heuristic multi-query run on the
remote) and computes, with the SAME code path for both: verified-flip
rate, final CFs per query, number of changed features per CF, L1/L2
distance normalised by each feature's replay std, and changed-feature
counts. The heuristic JSON does not store replay std, so the heuristic
replay is regenerated here with the authors' own play() (gym
CartPole-v0, seed 99, 200 episodes, notebook heuristic) -- no BART, no
XRL build -- and checked against the recorded 6,115 transitions /
action counts before its std is used. The DQN replay std is read from
the DQN JSON (recorded at run time).

Usage:
    <step3 venv>/bin/python compare_counterfactual_dqn_vs_heuristic.py \
        --dqn-json <counterfactual_dqn_cartpole.json> \
        --heur-json <counterfactual_cartpole_multi.json> \
        --out <comparison.json>

Prints a summary table and writes the full comparison JSON.
"""
import argparse
import json
import os
import sys

import numpy as np

FEATS = ["cart_position", "cart_velocity", "pole_angle", "pole_velocity"]


def heuristic_policy(state):
    return 1 if 0.4 * (state[0] / 2.4) + 0.6 * (state[2] / 2.095) >= 0 else 0


def cf_metrics(queries, feat_std):
    """Per-CF metrics from query/CF states; identical code for both runs."""
    rows = []
    for q in queries:
        if "counterfactuals" not in q:
            continue
        qv = np.array([q["query"][f] for f in FEATS], dtype=float)
        for c in q["counterfactuals"]:
            cf = c["cf"]
            cv = np.array([cf[f] for f in FEATS], dtype=float)
            deltas = cv - qv
            changed = [FEATS[i] for i in range(4)
                       if abs(deltas[i]) > 1e-9]
            rows.append({
                "query_row": q["query_row"],
                "flips": bool(c["flips"]),
                "n_changed": len(changed),
                "changed_features": changed,
                "deltas": {FEATS[i]: float(deltas[i]) for i in range(4)},
                "norm_l1": float(np.sum(np.abs(deltas) / feat_std)),
                "norm_l2": float(np.sqrt(np.sum((deltas / feat_std) ** 2))),
            })
    return rows


def summarize(name, data, feat_std, transitions):
    queries = [q for q in data["queries"] if "counterfactuals" in q]
    failed = [q for q in data["queries"] if "counterfactuals" not in q]
    rows = cf_metrics(queries, feat_std)
    n_cf = len(rows)
    n_flip = sum(1 for r in rows if r["flips"])
    n_changed = [r["n_changed"] for r in rows]
    l1 = [r["norm_l1"] for r in rows]
    feat_counts = {f: sum(1 for r in rows if f in r["changed_features"])
                   for f in FEATS}
    return {
        "run": name,
        "transitions": transitions,
        "n_queries": len(data["queries"]),
        "n_queries_failed": len(failed),
        "n_queries_with_verified_flip": sum(
            1 for q in queries if q.get("n_verified_flip", 0) >= 1),
        "n_final_cf": n_cf,
        "n_verified_flip": n_flip,
        "verified_flip_rate_cf": (n_flip / n_cf) if n_cf else None,
        "final_cf_per_query_mean": (n_cf / len(queries)) if queries else None,
        "n_raw_dice_per_query": [q.get("n_raw_dice") for q in queries],
        "n_changed_per_cf": n_changed,
        "n_changed_mean": float(np.mean(n_changed)) if n_changed else None,
        "n_changed_median": float(np.median(n_changed)) if n_changed else None,
        "norm_l1_list": [round(v, 4) for v in l1],
        "norm_l1_mean": float(np.mean(l1)) if l1 else None,
        "norm_l1_median": float(np.median(l1)) if l1 else None,
        "norm_l1_min": float(np.min(l1)) if l1 else None,
        "norm_l1_max": float(np.max(l1)) if l1 else None,
        "changed_feature_counts": feat_counts,
        "replay_feature_std": {f: float(s) for f, s in
                               zip(FEATS, feat_std)},
        "per_cf": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dqn-json", required=True)
    ap.add_argument("--heur-json", required=True)
    ap.add_argument("--causal-xrl-copy", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.dqn_json) as f:
        dqn = json.load(f)
    with open(args.heur_json) as f:
        heur = json.load(f)

    # Regenerate the heuristic replay (authors' play(), no BART) to get
    # its feature std, and verify it reproduces the recorded replay.
    cxrl_copy = (args.causal_xrl_copy or
                 "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
    sys.path.insert(0, cxrl_copy)
    import gym
    from cxrl.lib.experience import play
    np.random.seed(99)
    env = gym.make("CartPole-v0")
    env.seed(99)
    states, actions, _, _ = play(env, pi=heuristic_policy, nepisodes=200)
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int)
    heur_std = states.std(axis=0, ddof=0)
    replay_check = {
        "regenerated_transitions": int(len(states)),
        "recorded_transitions": int(heur["transitions"]),
        "regenerated_action_counts": {
            "0": int((actions == 0).sum()), "1": int((actions == 1).sum())},
        "recorded_action_counts": {"0": 3139, "1": 2976},
        "match": bool(len(states) == heur["transitions"]
                      and int((actions == 0).sum()) == 3139
                      and int((actions == 1).sum()) == 2976),
    }
    print("heuristic replay regeneration check:", replay_check, flush=True)

    dqn_std = np.array([dqn["replay_feature_std"][f] for f in FEATS])
    s_heur = summarize("heuristic (6 queries, remote JSON)",
                       heur, heur_std, heur["transitions"])
    s_dqn = summarize("DQN run 9", dqn, dqn_std, dqn["transitions"])
    out = {"heuristic_replay_check": replay_check,
           "heuristic": s_heur, "dqn": s_dqn}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}", flush=True)

    def fmt(s):
        rate = s["verified_flip_rate_cf"]
        return (f"{s['run']}: queries with >=1 verified flip "
                f"{s['n_queries_with_verified_flip']}/{s['n_queries']}; "
                f"CF verified {s['n_verified_flip']}/{s['n_final_cf']} "
                f"({rate:.3f} mean rate)" if rate is not None else
                f"{s['run']}: no final CFs")
    print(fmt(s_heur))
    print(fmt(s_dqn))
    for s in (s_heur, s_dqn):
        print(f"  {s['run']}: n_changed mean={s['n_changed_mean']} "
              f"median={s['n_changed_median']}; norm_L1 mean="
              f"{s['norm_l1_mean']:.4f} median={s['norm_l1_median']:.4f} "
              f"min={s['norm_l1_min']:.4f} max={s['norm_l1_max']:.4f}; "
              f"changed counts={s['changed_feature_counts']}")


if __name__ == "__main__":
    main()
