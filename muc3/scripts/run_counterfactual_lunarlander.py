"""Group 2c: LunarLander counterfactual attempt (fixed XRL code + multiclass fix).

Setup: gym 0.21 LunarLander-v2, seed 99, PD heuristic policy copied
verbatim from the authors' eval_fidelity_Lunar Lander.ipynb, 30 episodes
(the notebook's NEPISODES), repr_integration=False, one_hot_actions=False
(the DiCE explainer needs a single action column), BART via R/dbarts.
DummySklearn multiclass fix (Group 2) is active in the copy: predict_proba
returns one column per class (4).

Queries: first replay row with each observed action present, desired =
(observed + 2) % 4. Each query is attempted with the stock
counterfactual() path (attempt A). If attempt A raises, attempt B builds
a second DiCE explainer with the action outcome cast to str (DiCE
categorical handling) and runs generate + simplify + dominating manually;
both attempts are recorded with full tracebacks. Every CF is re-verified
with the RAW PD policy. Nothing is asserted to succeed: the output JSON
records exactly what happened per query.

Outputs: counterfactual_lunarlander.json (+ .csv if any CFs) in --outdir.
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
NEPISODES = 30
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/home/hatch/workspace/step3-scratch/group2/cf2c"


def policy(state):
    """PD heuristic copied verbatim from eval_fidelity_Lunar Lander.ipynb."""
    x, y, vx, vy, angle, angular_velocity, left_contact, right_contact = state
    target_angle = 0.5 * x + 1.0 * vx
    target_angle = np.clip(target_angle, -0.4, 0.4)
    target_y = 0.55 * abs(x)
    angle_control = 0.5 * (target_angle - angle) - 1.0 * angular_velocity
    hover_control = 0.5 * (target_y - y) - 0.5 * vy
    if left_contact or right_contact:
        angle_control = 0.0
        hover_control = -0.5 * vy
    if hover_control > abs(angle_control) and hover_control > 0.05:
        return 2
    elif angle_control < -0.05:
        return 3
    elif angle_control > 0.05:
        return 1
    else:
        return 0


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def attempt_b_str_outcome(xrl, query, desired, states, actions, feats):
    """Attempt B: DiCE explainer with str outcome; manual simplify+dominating."""
    import dice_ml
    from dice_ml import Dice
    df = pd.DataFrame(np.concatenate([states, actions[:, None]], axis=1),
                      columns=feats + ["action"])
    df["action"] = df["action"].astype(int).astype(str)
    dice_data = dice_ml.Data(dataframe=df, continuous_features=feats,
                             outcome_name="action")
    dice_model = dice_ml.Model(model=xrl.pi, backend="sklearn")
    exp_b = Dice(dice_data, dice_model, method="random")
    random.seed(SEED)
    np.random.seed(SEED)
    e1 = exp_b.generate_counterfactuals(
        query.drop(["action"], axis=1), total_CFs=40, desired_range=None,
        desired_class=str(desired), posthoc_sparsity_param=None,
        permitted_range=None, features_to_vary="all")
    raw_df = e1.cf_examples_list[0].final_cfs_df.copy()
    n_raw = int(len(raw_df))
    cf_df = raw_df.copy()
    # simplify/dominating expect the query without action for state cols;
    # they compare predictions (ints) to desired (int) -- fine.
    cf_df = xrl._XRL__simplify_counterfactuals(cf_df, query.drop(["action"], axis=1), desired)
    cf_df = xrl._XRL__dominating_counterfactuals(cf_df, query.drop(["action"], axis=1))
    return cf_df, n_raw


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    np.random.seed(SEED)
    env = gym.make("LunarLander-v2")
    env.seed(SEED)
    state_names = ["x", "y", "vx", "vy", "angle", "angular_velocity",
                   "left_contact", "right_contact"]
    feature_names = state_names + ["action"]
    t0 = time.time()
    xrl = XRL(env, pi=policy, repr_names=feature_names, verbose=False,
              repr_integration=False, nepisodes=NEPISODES)
    print(f"XRL built in {time.time()-t0:.1f}s; pi n_classes={xrl.pi.n_classes}", flush=True)
    states, actions, rewards, states_new = xrl.replay_buffer
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int)
    print(f"replay: {len(states)} transitions; action dist: "
          f"{dict(zip(*np.unique(actions, return_counts=True)))}", flush=True)
    print("causalG edges:", sorted(xrl.causalG.edges()), flush=True)
    feats = xrl.state_features

    # Queries: first row of each observed action, in action order.
    queries = []
    for a in sorted(np.unique(actions).tolist()):
        row_idx = int(np.where(actions == a)[0][0])
        queries.append((row_idx, int(a), (int(a) + 2) % 4))
    print("queries (row, observed, desired):", queries, flush=True)

    results = []
    all_cf_rows = []
    for row_idx, qa, desired in queries:
        query = pd.DataFrame([list(states[row_idx]) + [qa]], columns=feats + ["action"])
        rec = {"query_row": row_idx, "query": query.to_dict("records")[0],
               "observed_action": qa, "desired_action": desired}
        print(f"\n--- query row {row_idx}: observed={qa} desired={desired} ---", flush=True)
        # Attempt A: stock counterfactual()
        try:
            random.seed(SEED)
            np.random.seed(SEED)
            t1 = time.time()
            cf_df, score = xrl.counterfactual(query, desired)
            random.seed(SEED)
            np.random.seed(SEED)
            e1 = xrl.exp.generate_counterfactuals(
                query.drop(["action"], axis=1), total_CFs=40, desired_range=None,
                desired_class=desired, posthoc_sparsity_param=None,
                permitted_range=None, features_to_vary="all")
            rec["attempt"] = "A_stock_counterfactual"
            rec["cf_time_s"] = round(time.time() - t1, 2)
            rec["n_raw_dice"] = int(len(e1.cf_examples_list[0].final_cfs_df))
            rec["n_final"] = int(len(cf_df))
            rec["score"] = {k: float(v) for k, v in score.items()}
            cfs = []
            for _, r in cf_df.iterrows():
                a_raw = policy(r[feats].to_numpy(dtype=float))
                cfs.append({"cf": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                   for k, v in r.to_dict().items()},
                            "raw_policy_action": int(a_raw),
                            "flips": bool(a_raw == desired)})
            rec["counterfactuals"] = cfs
            rec["n_verified_flip"] = int(sum(1 for c in cfs if c["flips"]))
            print(f"A OK: raw={rec['n_raw_dice']} final={rec['n_final']} "
                  f"verified={rec['n_verified_flip']}/{rec['n_final']}", flush=True)
            for c in cfs:
                row = dict(c["cf"])
                row["query_row"] = row_idx
                row["raw_policy_action"] = c["raw_policy_action"]
                all_cf_rows.append(row)
        except Exception as e:
            rec["attempt_A_error"] = f"{type(e).__name__}: {str(e)[:500]}"
            rec["attempt_A_traceback"] = traceback.format_exc()[-2000:]
            print(f"attempt A FAILED: {rec['attempt_A_error']}", flush=True)
            # Attempt B: str outcome
            try:
                cf_df, n_raw = attempt_b_str_outcome(xrl, query, desired, states, actions, feats)
                rec["attempt"] = "B_str_outcome_manual"
                rec["n_raw_dice"] = n_raw
                rec["n_final"] = int(len(cf_df))
                cfs = []
                for _, r in cf_df.iterrows():
                    a_raw = policy(r[feats].to_numpy(dtype=float))
                    cfs.append({"cf": {k: str(v) for k, v in r.to_dict().items()},
                                "raw_policy_action": int(a_raw),
                                "flips": bool(a_raw == desired)})
                rec["counterfactuals"] = cfs
                rec["n_verified_flip"] = int(sum(1 for c in cfs if c["flips"]))
                print(f"B OK: raw={n_raw} final={rec['n_final']} "
                      f"verified={rec['n_verified_flip']}/{rec['n_final']}", flush=True)
                for c in cfs:
                    row = dict(c["cf"])
                    row["query_row"] = row_idx
                    row["raw_policy_action"] = c["raw_policy_action"]
                    all_cf_rows.append(row)
            except Exception as e2:
                rec["attempt_B_error"] = f"{type(e2).__name__}: {str(e2)[:500]}"
                rec["attempt_B_traceback"] = traceback.format_exc()[-2000:]
                print(f"attempt B FAILED: {rec['attempt_B_error']}", flush=True)
        results.append(rec)

    out = {"seed": SEED, "episodes": NEPISODES, "transitions": int(len(states)),
           "action_dist": {int(k): int(v) for k, v in zip(*np.unique(actions, return_counts=True))},
           "causal_edges": sorted([list(e) for e in xrl.causalG.edges()]),
           "queries": clean(results)}
    with open(os.path.join(OUTDIR, "counterfactual_lunarlander.json"), "w") as f:
        json.dump(out, f, indent=2)
    if all_cf_rows:
        pd.DataFrame(all_cf_rows).to_csv(
            os.path.join(OUTDIR, "counterfactual_lunarlander.csv"), index=False)
    print("saved LunarLander results; DONE", flush=True)


if __name__ == "__main__":
    main()
