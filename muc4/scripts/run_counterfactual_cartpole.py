"""Fallback B: CartPole counterfactuals with the FIXED XRL class.

Setup (matches Muc 3 / author CartPole notebook exactly):
- env: gym CartPole-v0, seed 99 (env.seed + np.random.seed)
- policy: notebook heuristic  1 if 0.4*(x/2.4) + 0.6*(theta/2.095) >= 0 else 0
  (identical to muc3/scripts/cartpole_collect.py and muc4 heuristic_policy)
- repr_names: ['cart_position','cart_velocity','pole_angle','pole_velocity','push']
- nepisodes=200, repr_integration=False (notebook defaults)
- BART via R/dbarts through rpy2 (R_LIBS must point at ~/workspace/R-lib)
- DiCE seeds: random.seed(99) + np.random.seed(99) immediately before
  counterfactual(), because the authors' generate_counterfactuals call
  passes no random_seed (DiCE 0.7.2 DiceRandom then draws from the
  global RNGs).

Query: replay row 6 (as in the authors' xrl_test.py, which uses
replay[6:7]), built from xrl.replay_buffer, with the observed action
column included so counterfactual()'s assert path is exercised.
desired_action = 1 - observed.

Outputs: counterfactual dataframe + score, saved as JSON/CSV; every
counterfactual is re-checked with the raw policy function.
"""
import sys, time, random, json
import numpy as np
import pandas as pd
# COMPAT SHIM (runner only, no library edited): dice-ml 0.7.2 calls
# DataFrame.append (dice_random.py:124, public_data_interface.py:568),
# removed in pandas 2.0. pandas' own private _append has the same
# signature/semantics for DataFrame inputs; restore the public name.
if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = pd.DataFrame._append

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
# sklearn compat shim (same as muc4 smoke_bart.py): repo BART code uses
# sklearn.tree._tree.DTYPE, removed in sklearn>=1.6. Restore it so the
# repo code runs unchanged.
import sklearn.tree._tree as _sk_tree
if not hasattr(_sk_tree, "DTYPE"):
    _sk_tree.DTYPE = np.float32

import gym
from cxrl.lib.xrl import XRL

SEED = 99

def policy(state):
    x0 = state[0] / 2.4
    x2 = state[2] / 2.095
    return 1 if x0 * 0.4 + x2 * 0.6 >= 0 else 0

def main():
    np.random.seed(SEED)
    env = gym.make("CartPole-v0")
    env.seed(SEED)
    feature_names = ["cart_position", "cart_velocity", "pole_angle", "pole_velocity", "push"]
    t0 = time.time()
    xrl = XRL(env, pi=policy, repr_names=feature_names, verbose=False,
              repr_integration=False, nepisodes=200)
    print(f"XRL built in {time.time()-t0:.1f}s", flush=True)
    states, actions, rewards, states_new = xrl.replay_buffer
    states = np.asarray(states, dtype=float); actions = np.asarray(actions, dtype=int)
    print(f"replay: {len(states)} transitions; actions: 0={int((actions==0).sum())} 1={int((actions==1).sum())}", flush=True)
    print("state_features:", xrl.state_features, "| exp:", type(xrl.exp).__name__, "| pi:", type(xrl.pi).__name__, flush=True)
    print("causalG edges:", sorted(xrl.causalG.edges()), flush=True)

    # query = replay row 6, with observed action (authors' test convention)
    q = states[6]; qa = int(actions[6])
    query = pd.DataFrame([list(q) + [qa]], columns=xrl.state_features + ["action"])
    print("query row 6:", query.to_dict("records")[0], flush=True)
    desired = 1 - qa
    print("desired_action:", desired, flush=True)

    random.seed(SEED); np.random.seed(SEED)
    t0 = time.time()
    cf_df, score = xrl.counterfactual(query, desired)
    print(f"counterfactual() in {time.time()-t0:.1f}s: {len(cf_df)} rows after simplify+dominating", flush=True)

    # Diagnostic (same seeds, same call as inside counterfactual): how many
    # raw CFs did DiCE return before simplify+dominating filtering?
    random.seed(SEED); np.random.seed(SEED)
    e1 = xrl.exp.generate_counterfactuals(query.drop(["action"], axis=1), total_CFs=40,
                                          desired_range=None, desired_class=desired,
                                          posthoc_sparsity_param=None, permitted_range=None,
                                          features_to_vary="all")
    raw_df = e1.cf_examples_list[0].final_cfs_df
    print(f"DiCE raw: {len(raw_df)} counterfactuals before simplify+dominating", flush=True)
    print("score:", score, flush=True)
    print(cf_df.to_string(), flush=True)

    # re-verify every CF with the RAW policy function (not via DiCE/DummySklearn)
    feats = xrl.state_features
    ok = []
    for _, row in cf_df.iterrows():
        a_raw = policy(row[feats].to_numpy(dtype=float))
        ok.append((int(row["action"]) == desired) and (a_raw == desired))
    print(f"re-verified with raw policy: {sum(ok)}/{len(ok)} CFs flip to {desired}", flush=True)
    assert all(ok), "some counterfactuals do not flip under the raw policy"

    out = {
        "seed": SEED, "episodes": 200, "transitions": len(states),
        "query": query.to_dict("records")[0], "desired_action": desired,
        "n_counterfactuals": len(cf_df), "score": score,
        "counterfactuals": cf_df.to_dict("records"),
        "causal_edges": sorted([list(e) for e in xrl.causalG.edges()]),
    }
    with open("/home/hatch/workspace/step3-scratch/counterfactual_result.json", "w") as f:
        json.dump(out, f, indent=2)
    cf_df.to_csv("/home/hatch/workspace/step3-scratch/counterfactual_result.csv", index=False)
    print("saved counterfactual_result.json/.csv; FALLBACK DONE", flush=True)

if __name__ == "__main__":
    main()
