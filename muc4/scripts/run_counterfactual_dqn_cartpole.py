"""Counterfactuals for the TRAINED DQN policy on CartPole (fixed XRL code).

Same setup as the heuristic multi-query run
(`run_counterfactual_cartpole_multi.py`, step 3 + group 2c) except the
policy: gym CartPole-v0, seed 99, 200 episodes, the same repr_names,
repr_integration=False, real BART via R/dbarts, DiCE method "random"
with total_CFs=40 (the authors' original call, untouched). The policy
here is the trained Double DQN (run 9, MLP 4 -> 128 -> 128 -> 2) loaded
from `muc4/results/cartpole-dqn/dqn_checkpoint.pt` (binary file, kept
on Drive, never committed). Six queries: replay rows 6, 100, 500,
1000, 2000, 3000 -- the same row indices as the heuristic run (row
contents differ because the DQN replay differs).

Every final counterfactual is re-verified with a DIRECT forward pass
of the DQN network (argmax Q), never trusted from DiCE. Q-values and
the decision margin are recorded for every query and counterfactual,
plus per-CF change metrics (changed features, raw deltas, L1 distance
normalised by each feature's replay std) so the DQN results can be
compared with the heuristic baseline on equal footing.

Success criteria were pre-registered BEFORE this run in
`muc4/counterfactual-dqn-criteria.md` (primary: >= 5/6 queries with
>= 1 verified-flipping final CF). Failed queries are recorded with
their traceback and do NOT abort the remaining queries.

Environment: the step-3 venv (gym 0.21.0, dice-ml 0.7.2, torch CPU)
with R_LIBS pointing at the R library holding dbarts 0.9.34, e.g.:

    R_LIBS=~/workspace/R-lib <venv>/bin/python \
        run_counterfactual_dqn_cartpole.py \
        --checkpoint <path/to/dqn_checkpoint.pt> --outdir <outdir>

Outputs: counterfactual_dqn_cartpole.json + .csv in --outdir.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback

# rpy2 initialises R on first use; R_LIBS must be set before that.
# Only fill in the known local default when the caller did not set it.
if "R_LIBS" not in os.environ:
    _default_rlib = os.path.expanduser("~/workspace/R-lib")
    if os.path.isdir(os.path.join(_default_rlib, "dbarts")):
        os.environ["R_LIBS"] = _default_rlib

import numpy as np
import pandas as pd
import torch

if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = pd.DataFrame._append

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cartpole_policies import QNetwork  # noqa: E402  (repo file, same dir)

SEED = 99
QUERY_ROWS = [6, 100, 500, 1000, 2000, 3000]
FEATURE_NAMES = ["cart_position", "cart_velocity", "pole_angle",
                 "pole_velocity", "push"]


def clean(obj):
    """JSON-safe: NaN/inf -> None (score can be NaN if 0 CFs survive)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def load_dqn(checkpoint_path):
    """Load the checkpoint exactly as cartpole_policies does; return
    (pi, qvalues, metadata). pi/qvalues take one state (array-like)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu",
                      weights_only=False)
    hidden = int(ckpt.get("hyperparams", {}).get("hidden", 128))
    net = QNetwork(hidden=hidden)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    def qvalues(state):
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(state, dtype=np.float32)
                                ).unsqueeze(0)
            return net(s).squeeze(0).numpy().astype(float)

    def pi(state):
        return int(np.argmax(qvalues(state)))

    meta = {"hyperparams": ckpt.get("hyperparams", {}),
            "env": ckpt.get("env"),
            "framework": ckpt.get("framework"),
            "best_eval_episode": ckpt.get("best_eval_episode"),
            "best_eval": ckpt.get("best_eval"),
            "state_dict_shapes": {k: list(v.shape)
                                  for k, v in ckpt["state_dict"].items()}}
    return pi, qvalues, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="dqn_checkpoint.pt (default: "
                    "../results/cartpole-dqn/dqn_checkpoint.pt, repo layout)")
    ap.add_argument("--causal-xrl-copy", default=None,
                    help="patched causal-xrl copy dir (default: "
                    "/home/hatch/workspace/step3-scratch/causal-xrl-copy)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.checkpoint or os.path.join(
        script_dir, "..", "results", "cartpole-dqn", "dqn_checkpoint.pt")
    cxrl_copy = (args.causal_xrl_copy or
                 "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
    outdir = args.outdir or os.path.join(
        os.path.expanduser("~"), "workspace", "xrl-g4a-work", "run1")
    os.makedirs(outdir, exist_ok=True)
    torch.set_num_threads(2)

    with open(ckpt_path, "rb") as f:
        ckpt_md5 = hashlib.md5(f.read()).hexdigest()
    ckpt_bytes = os.path.getsize(ckpt_path)
    pi, qvalues, ckpt_meta = load_dqn(ckpt_path)
    print(f"checkpoint: {ckpt_path} ({ckpt_bytes} bytes, md5={ckpt_md5})",
          flush=True)
    print(f"checkpoint meta: best_eval_episode="
          f"{ckpt_meta['best_eval_episode']} best_eval="
          f"{ckpt_meta['best_eval']}", flush=True)

    sys.path.insert(0, cxrl_copy)
    import sklearn.tree._tree as _sk_tree
    if not hasattr(_sk_tree, "DTYPE"):
        _sk_tree.DTYPE = np.float32
    import gym
    from cxrl.lib.xrl import XRL

    seed = args.seed
    np.random.seed(seed)
    env = gym.make("CartPole-v0")
    env.seed(seed)
    t0 = time.time()
    xrl = XRL(env, pi=pi, repr_names=FEATURE_NAMES, verbose=False,
              repr_integration=False, nepisodes=args.episodes)
    build_s = time.time() - t0
    print(f"XRL built in {build_s:.1f}s; pi n_classes={xrl.pi.n_classes}",
          flush=True)
    states, actions, rewards, states_new = xrl.replay_buffer
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int)
    print(f"replay: {len(states)} transitions; actions: "
          f"0={int((actions == 0).sum())} 1={int((actions == 1).sum())}",
          flush=True)
    print("causalG edges:", sorted(xrl.causalG.edges()), flush=True)
    feats = xrl.state_features
    feat_std = states.std(axis=0, ddof=0)
    print("replay feature std:", {f: float(s) for f, s in
                                  zip(feats, feat_std)}, flush=True)
    # Sanity: the replay actions must equal the DQN policy on the replay
    # states (they were collected by this same pi).
    check_idx = np.arange(0, len(states), max(1, len(states) // 200))
    agree = sum(1 for i in check_idx if pi(states[i]) == int(actions[i]))
    print(f"replay/policy agreement on {len(check_idx)} sampled states: "
          f"{agree}/{len(check_idx)}", flush=True)

    results = []
    all_cf_rows = []
    for row_idx in QUERY_ROWS:
        q = states[row_idx]
        qa = int(actions[row_idx])
        query = pd.DataFrame([list(q) + [qa]], columns=feats + ["action"])
        desired = 1 - qa
        q_qvals = qvalues(q)
        rec = {"query_row": row_idx,
               "query": query.to_dict("records")[0],
               "observed_action": qa, "desired_action": desired,
               "query_qvalues": [float(v) for v in q_qvals],
               "query_margin_q_obs_minus_q_des": float(
                   q_qvals[qa] - q_qvals[desired]),
               "query_pi_action": int(pi(q))}
        print(f"\n--- query row {row_idx}: observed={qa} desired={desired} "
              f"Q={np.round(q_qvals, 4).tolist()} margin="
              f"{rec['query_margin_q_obs_minus_q_des']:.4f} ---", flush=True)
        print("query:", rec["query"], flush=True)
        try:
            random.seed(seed)
            np.random.seed(seed)
            t1 = time.time()
            cf_df, score = xrl.counterfactual(query, desired)
            rec["cf_time_s"] = round(time.time() - t1, 2)
            random.seed(seed)
            np.random.seed(seed)
            e1 = xrl.exp.generate_counterfactuals(
                query.drop(["action"], axis=1), total_CFs=40,
                desired_range=None, desired_class=desired,
                posthoc_sparsity_param=None, permitted_range=None,
                features_to_vary="all")
            rec["n_raw_dice"] = int(len(e1.cf_examples_list[0].final_cfs_df))
            rec["n_final"] = int(len(cf_df))
            rec["score"] = {k: float(v) for k, v in score.items()}
            cfs = []
            for _, r in cf_df.iterrows():
                cf_state = r[feats].to_numpy(dtype=float)
                cf_qvals = qvalues(cf_state)
                a_raw = int(np.argmax(cf_qvals))
                deltas = cf_state - q
                changed = [feats[i] for i in range(len(feats))
                           if abs(deltas[i]) > 1e-9]
                norm_l1 = float(np.sum(np.abs(deltas) / feat_std))
                norm_l2 = float(np.sqrt(np.sum((deltas / feat_std) ** 2)))
                cfs.append({"cf": r.to_dict(),
                            "raw_policy_action": a_raw,
                            "cf_qvalues": [float(v) for v in cf_qvals],
                            "cf_margin_q_des_minus_q_obs": float(
                                cf_qvals[desired] - cf_qvals[qa]),
                            "flips": bool(a_raw == desired and
                                          int(r["action"]) == desired),
                            "changed_features": changed,
                            "n_changed": len(changed),
                            "deltas": {feats[i]: float(deltas[i])
                                       for i in range(len(feats))},
                            "norm_l1": norm_l1,
                            "norm_l2": norm_l2})
            rec["counterfactuals"] = cfs
            rec["n_verified_flip"] = int(sum(1 for c in cfs if c["flips"]))
            print(f"raw DiCE={rec['n_raw_dice']} final={rec['n_final']} "
                  f"verified={rec['n_verified_flip']}/{rec['n_final']} "
                  f"score={rec['score']}", flush=True)
            for c in cfs:
                row = dict(c["cf"])
                row["query_row"] = row_idx
                row["raw_policy_action"] = c["raw_policy_action"]
                row["flips"] = c["flips"]
                row["n_changed"] = c["n_changed"]
                row["changed_features"] = ";".join(c["changed_features"])
                row["norm_l1"] = c["norm_l1"]
                row["norm_l2"] = c["norm_l2"]
                all_cf_rows.append(row)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            rec["traceback"] = traceback.format_exc()[-1500:]
            print(f"QUERY row {row_idx} FAILED: {rec['error']}", flush=True)
        results.append(rec)

    n_ok = sum(1 for r in results
               if r.get("n_verified_flip", 0) >= 1)
    out = {"policy": "DQN run 9 (Double DQN, MLP 4->128->128->2)",
           "checkpoint_path": ckpt_path,
           "checkpoint_md5": ckpt_md5,
           "checkpoint_bytes": ckpt_bytes,
           "checkpoint_meta": clean(ckpt_meta),
           "seed": seed, "episodes": args.episodes,
           "env": "gym CartPole-v0 (gym 0.21.0)",
           "dice_method": "random", "total_CFs": 40,
           "transitions": int(len(states)),
           "action_counts": {"0": int((actions == 0).sum()),
                             "1": int((actions == 1).sum())},
           "replay_feature_std": {f: float(s) for f, s in
                                  zip(feats, feat_std)},
           "replay_policy_agreement_sampled": {
               "n_checked": int(len(check_idx)), "n_agree": int(agree)},
           "xrl_build_s": round(build_s, 1),
           "query_rows": QUERY_ROWS,
           "causal_edges": sorted([list(e) for e in xrl.causalG.edges()]),
           "success_criteria_preregistered": (
               "muc4/counterfactual-dqn-criteria.md: C1 >= 5/6 queries "
               "with >= 1 verified-flipping final CF"),
           "c1_queries_with_verified_flip": int(n_ok),
           "c1_total_queries": int(len(QUERY_ROWS)),
           "c1_met": bool(n_ok >= 5),
           "queries": clean(results)}
    with open(os.path.join(outdir,
                           "counterfactual_dqn_cartpole.json"), "w") as f:
        json.dump(out, f, indent=2)
    if all_cf_rows:
        pd.DataFrame(all_cf_rows).to_csv(
            os.path.join(outdir, "counterfactual_dqn_cartpole.csv"),
            index=False)
    print(f"\nsaved DQN counterfactual results; C1: {n_ok}/{len(QUERY_ROWS)} "
          f"queries with >=1 verified flip -> "
          f"{'MET' if n_ok >= 5 else 'NOT MET'}; DONE", flush=True)


if __name__ == "__main__":
    main()
