#!/usr/bin/env python3
"""Collect LunarLander rollouts for the XRL trained-vs-heuristic comparison.

Same collection rule as run_xrl_comparison.py (CartPole): first reset
seeded, later resets unseeded; a transition is recorded only when the
step is NOT terminal. Policies from lunarlander_policies.py (repo PD
heuristic + DQN checkpoint). Saves replay_<policy>.npz (states,
actions, rewards, next_states) + rollout_<policy>.json per policy.

The step-4 run used: --episodes 30 --seed 0 --checkpoint
results/lunarlander-dqn/best_checkpoint.pt (group-2 checkpoint, best @
ep 850). Replay .npz files are working data, not pushed to the repo;
per_run_details.json in the results dir carries the rollout summaries.

Usage:
    .venv-ablation/bin/python -B scripts/collect_lunarlander.py \\
        --episodes 30 --seed 0 --checkpoint <best_checkpoint.pt> \\
        --outdir <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lunarlander_policies import heuristic_policy, load_dqn_policy  # noqa: E402
import gymnasium as gym  # noqa: E402


def collect(policy_fn, nepisodes, seed, env_name="LunarLander-v3"):
    env = gym.make(env_name)
    states, actions, rewards, states_new = [], [], [], []
    ep_lengths, ep_rewards = [], []
    t0 = time.perf_counter()
    for ep in range(nepisodes):
        obs, _ = env.reset(seed=seed) if ep == 0 else env.reset()
        state_t = np.asarray(obs, dtype=float)
        done, steps, total = False, 0, 0.0
        while not done:
            a = int(policy_fn(state_t))
            obs2, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            total += float(r)
            state_tp1 = np.asarray(obs2, dtype=float)
            if not done:
                states.append(state_t)
                actions.append(a)
                rewards.append(float(r))
                states_new.append(state_tp1)
            state_t = state_tp1
            steps += 1
        ep_lengths.append(steps)
        ep_rewards.append(total)
    env.close()
    return (np.array(states), np.array(actions), np.array(rewards),
            np.array(states_new)), time.perf_counter() - t0, \
        ep_lengths, ep_rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", required=True,
                    help="LunarLander DQN checkpoint .pt")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--env", default="LunarLander-v3")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dqn = load_dqn_policy(args.checkpoint)
    for name, pi in [("heuristic", heuristic_policy), ("dqn", dqn)]:
        (s, a, r, sn), wall, lens, rews = collect(
            pi, args.episodes, args.seed, args.env)
        np.savez(outdir / f"replay_{name}.npz", states=s, actions=a,
                 rewards=r, next_states=sn)
        summary = {
            "policy": name, "env": args.env, "episodes": args.episodes,
            "seed": args.seed, "n_transitions": int(len(a)),
            "rollout_s": round(wall, 2),
            "ep_lengths": lens, "ep_rewards": [round(x, 2) for x in rews],
            "mean_ep_len": round(float(np.mean(lens)), 2),
            "mean_ep_reward": round(float(np.mean(rews)), 2),
            "std_ep_reward": round(float(np.std(rews)), 2),
            "min_ep_reward": round(float(np.min(rews)), 2),
            "max_ep_reward": round(float(np.max(rews)), 2),
            "action_counts": np.bincount(a, minlength=4).tolist(),
            "action_names": ["do_nothing", "fire_left_engine",
                             "fire_main_engine", "fire_right_engine"],
        }
        if name == "dqn":
            ck = dqn.checkpoint
            summary["checkpoint"] = str(args.checkpoint)
            summary["checkpoint_best_eval_episode"] = ck.get(
                "best_eval_episode")
            summary["checkpoint_best_eval"] = ck.get("best_eval")
            summary["checkpoint_hyperparams"] = ck.get("hyperparams")
        (outdir / f"rollout_{name}.json").write_text(
            json.dumps(summary, indent=2))
        print(name, "transitions=", len(a),
              "mean_len=", summary["mean_ep_len"],
              "mean_rew=", summary["mean_ep_reward"],
              "actions=", summary["action_counts"], f"wall={wall:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
