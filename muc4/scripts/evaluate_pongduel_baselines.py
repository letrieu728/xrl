#!/usr/bin/env python3
"""Evaluate PongDuel baselines: repo heuristic (strong/weak) + random.

Same eval protocol as train_dqn_pongduel.evaluate(): 100 episodes,
one fresh env seed per episode (10000..10099), wrapper reward_mode
agent1. Random uses a dedicated random.Random(12345) for joint-action
choices (recorded; env seeds still vary per episode). Run BEFORE
training so the success criterion is set against measured baselines.
"""
import json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pongduel_wrapper import make_pongduel_env, heuristic_policy


def evaluate_policy(name, pi_fn, n_episodes=100, seed_offset=10000):
    env = make_pongduel_env(reward_mode="agent1")
    r1, r2, lens, acts = [], [], [], []
    for i in range(n_episodes):
        s = env.reset(seed=seed_offset + i)
        t1 = t2 = 0.0; steps = 0; done = False
        while not done:
            a = int(pi_fn(s))
            s, r, done, info = env.step(a)
            t1 += float(r); t2 += float(info["original_reward_n"][1])
            steps += 1; acts.append(a)
        r1.append(t1); r2.append(t2); lens.append(steps)
    env.close()
    a1, a2 = np.array(r1), np.array(r2)
    out = {"policy": name, "n": n_episodes, "seed_offset": seed_offset,
           "agent1": {"mean": round(float(a1.mean()), 4),
                      "std": round(float(a1.std()), 4),
                      "min": float(a1.min()), "max": float(a1.max()),
                      "rewards": [float(x) for x in r1]},
           "agent2": {"mean": round(float(a2.mean()), 4),
                      "std": round(float(a2.std()), 4),
                      "min": float(a2.min()), "max": float(a2.max())},
           "mean_ep_len": round(float(np.mean(lens)), 2),
           "joint_action_counts": np.bincount(acts, minlength=9).tolist()}
    print(json.dumps({k: v for k, v in out.items() if k != "agent1"} | {
        "agent1": {k: v for k, v in out["agent1"].items()
                   if k != "rewards"}}, indent=2))
    return out


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    results["heuristic_strong_weak"] = evaluate_policy(
        "heuristic_strong_weak", heuristic_policy)
    rng = random.Random(12345)
    results["random"] = evaluate_policy(
        "random", lambda s: rng.randint(0, 8))
    (outdir / "baselines_eval.json").write_text(
        json.dumps(results, indent=2))
    print("wrote", outdir / "baselines_eval.json")


if __name__ == "__main__":
    main()
