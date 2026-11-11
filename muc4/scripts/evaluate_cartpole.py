#!/usr/bin/env python3
"""Evaluate the trained DQN policy and the CausalXRL heuristic policy on
CartPole-v1, 100 episodes each, greedy/deterministic, one fresh env seed
per episode (seeds 10000..10099, same scheme as train_dqn_cartpole.py).

Usage:
    python evaluate_cartpole.py --checkpoint ../results/cartpole-dqn/dqn_checkpoint.pt \
        --out ../results/cartpole-dqn/eval_log.txt [--episodes 100]
"""
import argparse
import sys
import os

import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cartpole_policies import heuristic_policy, load_dqn_policy


def evaluate(env_name, pi, n_episodes, seed_offset=10000):
    env = gym.make(env_name)
    rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed_offset + i)
        total = 0.0
        done = False
        while not done:
            state, reward, terminated, truncated, _ = env.step(
                int(pi(np.asarray(state, dtype=np.float32))))
            total += reward
            done = terminated or truncated
        rewards.append(total)
    env.close()
    r = np.array(rewards, dtype=np.float64)
    return {"n": int(n_episodes), "mean": float(r.mean()),
            "std": float(r.std()), "min": float(r.min()),
            "max": float(r.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dqn_pi = load_dqn_policy(args.checkpoint)
    dqn = evaluate(args.env, dqn_pi, args.episodes)
    heur = evaluate(args.env, heuristic_policy, args.episodes)

    lines = [
        "CartPole-v1 evaluation (100 episodes each, greedy/deterministic, "
        "env seeds 10000..10099)",
        f"checkpoint: {args.checkpoint}",
        "",
        "DQN (trained, this work):",
        f"  mean = {dqn['mean']:.2f}",
        f"  std  = {dqn['std']:.2f}",
        f"  min  = {dqn['min']:.0f}",
        f"  max  = {dqn['max']:.0f}",
        "",
        "Heuristic (CausalXRL repo, hand-written):",
        f"  mean = {heur['mean']:.2f}",
        f"  std  = {heur['std']:.2f}",
        f"  min  = {heur['min']:.0f}",
        f"  max  = {heur['max']:.0f}",
        "",
        f"DQN - heuristic mean difference: {dqn['mean'] - heur['mean']:+.2f}",
        "",
        "Note: training used gymnasium (not gym 0.20, incompatible with "
        "Python 3.12); the trained policy is a plain pi(state)->action "
        "function, so it plugs into the authors' XRL(env, pi) interface.",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
