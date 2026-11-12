"""Group 2c: CartPole counterfactuals, multiple queries (fixed XRL code).

Same setup as step 3 (gym CartPole-v0, seed 99, heuristic policy from the
authors' notebook, 200 episodes, repr_integration=False, BART via
R/dbarts), but SIX queries instead of one: replay rows 6, 100, 500,
1000, 2000, 3000 (row 6 repeats step 3 as a regression check). One XRL
build serves all queries. Each query: DiCE seeds reset to 99 immediately
before counterfactual() (the authors' call passes no random_seed), then
the same-seed diagnostic regeneration counts raw DiCE CFs before
simplify+dominating filtering. Every final CF is re-verified with the
RAW policy function (not via DiCE/DummySklearn). A failed query is
recorded with its traceback and does NOT abort the remaining queries.

Outputs: counterfactual_cartpole_multi.json + .csv (all CFs, with
query_row column) in --outdir.
"""
import json
import os
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd

if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = pd.DataFrame._append

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
import sklearn.tree._tree as _sk_tree
if not hasattr(_sk_tree, "DTYPE"):
    _sk_tree.DTYPE = np.float32

import gym
from cxrl.lib.xrl import XRL

SEED = 99
QUERY_ROWS = [6, 100, 500, 1000, 2000, 3000]
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/home/hatch/workspace/step3-scratch/group2/cf2c"


def policy(state):
    x0 = state[0] / 2.4
    x2 = state[2] / 2.095
    return 1 if x0 * 0.4 + x2 * 0.6 >= 0 else 0


def clean(obj):
    """JSON-safe: NaN/inf -> None (score can be NaN if 0 CFs survive)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    np.random.seed(SEED)
    env = gym.make("CartPole-v0")
    env.seed(SEED)
    feature_names = ["cart_position", "cart_velocity", "pole_angle", "pole_velocity", "push"]
    t0 = time.time()
    xrl = XRL(env, pi=policy, repr_names=feature_names, verbose=False,
              repr_integration=False, nepisodes=200)
    print(f"XRL built in {time.time()-t0:.1f}s; pi n_classes={xrl.pi.n_classes}", flush=True)
    states, actions, rewards, states_new = xrl.replay_buffer
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int)
    print(f"replay: {len(states)} transitions; actions: 0={int((actions==0).sum())} "
          f"1={int((actions==1).sum())}", flush=True)
    print("causalG edges:", sorted(xrl.causalG.edges()), flush=True)
    feats = xrl.state_features

    results = []
    all_cf_rows = []
    for row_idx in QUERY_ROWS:
        q = states[row_idx]
        qa = int(actions[row_idx])
        query = pd.DataFrame([list(q) + [qa]], columns=feats + ["action"])
        desired = 1 - qa
        rec = {"query_row": row_idx,
               "query": query.to_dict("records")[0],
               "observed_action": qa, "desired_action": desired}
        print(f"\n--- query row {row_idx}: observed={qa} desired={desired} ---", flush=True)
        print("query:", rec["query"], flush=True)
        try:
            random.seed(SEED)
            np.random.seed(SEED)
            t1 = time.time()
            cf_df, score = xrl.counterfactual(query, desired)
            rec["cf_time_s"] = round(time.time() - t1, 2)
            random.seed(SEED)
            np.random.seed(SEED)
            e1 = xrl.exp.generate_counterfactuals(
                query.drop(["action"], axis=1), total_CFs=40, desired_range=None,
                desired_class=desired, posthoc_sparsity_param=None,
                permitted_range=None, features_to_vary="all")
            rec["n_raw_dice"] = int(len(e1.cf_examples_list[0].final_cfs_df))
            rec["n_final"] = int(len(cf_df))
            rec["score"] = {k: float(v) for k, v in score.items()}
            cfs = []
            for _, r in cf_df.iterrows():
                a_raw = policy(r[feats].to_numpy(dtype=float))
                cfs.append({"cf": r.to_dict(), "raw_policy_action": int(a_raw),
                            "flips": bool(a_raw == desired and int(r["action"]) == desired)})
            rec["counterfactuals"] = cfs
            rec["n_verified_flip"] = int(sum(1 for c in cfs if c["flips"]))
            print(f"raw DiCE={rec['n_raw_dice']} final={rec['n_final']} "
                  f"verified={rec['n_verified_flip']}/{rec['n_final']} score={rec['score']}",
                  flush=True)
            for c in cfs:
                row = dict(c["cf"])
                row["query_row"] = row_idx
                row["raw_policy_action"] = c["raw_policy_action"]
                all_cf_rows.append(row)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            rec["traceback"] = traceback.format_exc()[-1500:]
            print(f"QUERY row {row_idx} FAILED: {rec['error']}", flush=True)
        results.append(rec)

    # Regression check: row 6 vs step-3 result file.
    reg = None
    try:
        with open("/home/hatch/workspace/step3-scratch/counterfactual_result.json") as f:
            old = json.load(f)
        new6 = next(r for r in results if r["query_row"] == 6)
        if "error" not in new6:
            old_cfs = old["counterfactuals"]
            new_cfs = [c["cf"] for c in new6["counterfactuals"]]
            same_n = len(old_cfs) == len(new_cfs)
            maxdiff = 0.0
            if same_n:
                for oc, nc in zip(old_cfs, new_cfs):
                    for k in feats:
                        maxdiff = max(maxdiff, abs(float(oc[k]) - float(nc[k])))
            reg = {"same_n": same_n, "max_abs_diff_state_values": maxdiff,
                   "old_score": old["score"], "new_score": new6["score"]}
            print("REGRESSION row6 vs step3:", reg, flush=True)
    except Exception as e:
        reg = {"error": f"{type(e).__name__}: {e}"}
    out = {"seed": SEED, "episodes": 200, "transitions": int(len(states)),
           "query_rows": QUERY_ROWS,
           "causal_edges": sorted([list(e) for e in xrl.causalG.edges()]),
           "queries": clean(results), "regression_row6_vs_step3": clean(reg)}
    with open(os.path.join(OUTDIR, "counterfactual_cartpole_multi.json"), "w") as f:
        json.dump(out, f, indent=2)
    if all_cf_rows:
        pd.DataFrame(all_cf_rows).to_csv(
            os.path.join(OUTDIR, "counterfactual_cartpole_multi.csv"), index=False)
    print("saved multi-query results; DONE", flush=True)


if __name__ == "__main__":
    main()
