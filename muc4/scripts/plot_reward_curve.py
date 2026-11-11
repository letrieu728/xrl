#!/usr/bin/env python3
"""Plot the DQN training reward curve from training_log.csv.

Usage:
    python plot_reward_curve.py --csv ../results/cartpole-dqn/training_log.csv \
        --out ../results/cartpole-dqn/reward_curve.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    ep = data["episode"]
    rew = data["reward"]
    window = 100
    rolling = np.convolve(rew, np.ones(window) / window, mode="valid")
    rolling_ep = ep[window - 1:]

    plt.figure(figsize=(9, 5))
    plt.plot(ep, rew, color="#b0c4de", linewidth=0.6, label="episode reward")
    plt.plot(rolling_ep, rolling, color="#1f4e79", linewidth=2,
             label=f"{window}-episode rolling mean")
    plt.axhline(500, color="#c00000", linestyle="--", linewidth=1,
                label="CartPole-v1 max (500)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on CartPole-v1 — training reward (seed 0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"wrote {args.out} (episodes={int(ep[-1])}, "
          f"final {window}-ep mean={rolling[-1]:.1f})")


if __name__ == "__main__":
    main()
