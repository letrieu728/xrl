#!/usr/bin/env python3
"""Plot DQN training reward + periodic greedy-eval curve.

Reads training_log.csv (episode reward, rolling mean) and eval_log.csv
(episode, eval_mean/std/min/max) written by train_dqn_cartpole.py
--eval-every, and draws both on one figure with the 500 target line
and a marker for the best periodic eval.

Usage:
    python plot_eval_curve.py --train-csv ../results/cartpole-dqn/run7/training_log.csv \
        --eval-csv ../results/cartpole-dqn/run7/eval_log.csv \
        --out ../results/cartpole-dqn/run7/eval_curve.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="DQN on CartPole-v1 — training reward + periodic greedy eval (seed 0)")
    args = ap.parse_args()

    data = np.genfromtxt(args.train_csv, delimiter=",", names=True)
    ep = data["episode"]
    rew = data["reward"]
    window = 100
    rolling = np.convolve(rew, np.ones(window) / window, mode="valid")
    rolling_ep = ep[window - 1:]

    ev = np.genfromtxt(args.eval_csv, delimiter=",", names=True)
    ev_ep = np.atleast_1d(ev["episode"])
    ev_mean = np.atleast_1d(ev["eval_mean"])
    ev_std = np.atleast_1d(ev["eval_std"])
    ev_min = np.atleast_1d(ev["eval_min"])
    ev_max = np.atleast_1d(ev["eval_max"])
    best_i = int(np.argmax(ev_mean))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(ep, rew, color="#b0c4de", linewidth=0.6, label="episode reward (train, eps-greedy)")
    ax.plot(rolling_ep, rolling, color="#1f4e79", linewidth=2,
            label=f"{window}-episode rolling mean (train)")
    ax.errorbar(ev_ep, ev_mean, yerr=ev_std, fmt="o-", color="#c00000",
                capsize=3, linewidth=1.5, markersize=4,
                label="periodic greedy eval (50 ep, mean ± std)")
    ax.plot([ev_ep[best_i]], [ev_mean[best_i]], marker="*", markersize=16,
            color="#c00000", markeredgecolor="black", markeredgewidth=0.7,
            linestyle="None",
            label=f"best periodic eval: {ev_mean[best_i]:.1f} @ ep {int(ev_ep[best_i])}")
    ax.axhline(500, color="#2e7d32", linestyle="--", linewidth=1, label="CartPole-v1 max (500)")
    ax.axhline(450, color="#2e7d32", linestyle=":", linewidth=1, label="target lower bound (450)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(args.title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out} (train episodes={int(ep[-1])}, "
          f"periodic evals={len(ev_ep)}, best eval={ev_mean[best_i]:.2f} "
          f"@ ep {int(ev_ep[best_i])}, eval range "
          f"{ev_mean.min():.1f}..{ev_mean.max():.1f})")


if __name__ == "__main__":
    main()
