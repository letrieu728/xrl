#!/usr/bin/env python3
"""Train a DQN agent on PongDuel-v0 (CausalXRL extension D, step 5).

Adapted from scripts/train_dqn_lunarlander.py (itself a copy of
train_dqn_cartpole.py run-9 recipe) with ONLY these changes, all
documented:

  * Environment: the repo notebook's SingleAgentPongDuelWrapper
    (pongduel_wrapper.py) over ma-gym PongDuel-v0, gym 0.21 API
    (reset(seed) / 4-tuple step) instead of gymnasium. Network dims
    read from the env: MLP 20 -> 128 -> 128 -> 9 (flat 20-dim obs,
    9 joint actions). Reward = agent1's reward (wrapper default,
    notebook cell 3); agent2 reward + episode length are logged
    alongside for every episode/eval.
  * Defaults: --episodes 2000 (HARD BUDGET for step 5 — do not
    exceed; if the run is killed by a VM restart, report the partial
    run from the saved best checkpoint, do NOT restart training),
    --eps-decay-episodes 1300 (same 65% ratio as run 9 / LunarLander).
  * training_log.csv gains reward_agent2 + ep_len columns (episode /
    reward(agent1) / epsilon / loss columns keep their positions).

Everything else (Double DQN, lr 1e-3, gamma 0.99, batch 64, buffer
50000, eps 1.0->0.02, target update every 10 episodes, warmup 1000,
MSE, periodic greedy eval + best-checkpoint selection, eval seed
protocol) is identical to the run-9 / LunarLander recipe. No
hyperparameter tuning was done for PongDuel: the point of step 5 is
to test whether the SAME recipe learns anything on the joint-action
PongDuel reduction within 2,000 episodes, not to squeeze PongDuel.

Self-contained PyTorch (no stable-baselines3). CPU-only, $0.

Usage:
    step3-scratch/venv/bin/python train_dqn_pongduel.py --seed 0 \\
        --episodes 2000 --double-dqn --eps-decay-episodes 1300 \\
        --eval-every 50 --eval-episodes 50 --outdir <dir>
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pongduel_wrapper import make_pongduel_env  # noqa: E402


class QNetwork(nn.Module):
    def __init__(self, state_dim=20, hidden=128, action_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def greedy_policy_from_net(net, device):
    def pi(state):
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(state, dtype=np.float32),
                                device=device).unsqueeze(0)
            return int(torch.argmax(net(s), dim=1).item())
    return pi


def evaluate(pi, n_episodes, seed_offset=10000, reward_mode="agent1"):
    """Greedy evaluation, one fresh env seed per episode (deterministic).

    Returns agent1 (primary) + agent2 reward stats and ep lengths.
    """
    env = make_pongduel_env(reward_mode=reward_mode)
    r1, r2, lens = [], [], []
    for i in range(n_episodes):
        state = env.reset(seed=seed_offset + i)
        total1, total2, steps = 0.0, 0.0, 0
        done = False
        while not done:
            action = pi(np.asarray(state, dtype=np.float32))
            state, reward, done, info = env.step(int(action))
            total1 += float(reward)
            total2 += float(info["original_reward_n"][1])
            steps += 1
        r1.append(total1)
        r2.append(total2)
        lens.append(steps)
    env.close()
    a1, a2 = np.array(r1, dtype=float), np.array(r2, dtype=float)

    def stats(a):
        return {"n": int(len(a)), "mean": float(a.mean()),
                "std": float(a.std()), "min": float(a.min()),
                "max": float(a.max()), "rewards": [float(x) for x in a]}
    return {"agent1": stats(a1), "agent2": stats(a2),
            "mean_ep_len": float(np.mean(lens)),
            "ep_lengths": [int(x) for x in lens]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--buffer-size", type=int, default=50000)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.02)
    ap.add_argument("--eps-decay-episodes", type=int, default=1300)
    ap.add_argument("--target-update-every", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--grad-clip", type=float, default=0.0)
    ap.add_argument("--double-dqn", action="store_true")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--eval-seed-offset", type=int, default=10000)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    if args.episodes > 2000:
        raise ValueError("step-5 budget is MAX 2000 episodes; "
                         f"got --episodes {args.episodes}")
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    hyper = vars(args)
    print(f"seed={args.seed} episodes={args.episodes} env=PongDuel-v0 "
          f"(wrapper, reward agent1) torch={torch.__version__} device=cpu",
          flush=True)

    env = make_pongduel_env(reward_mode="agent1")
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    print(f"env dims: state_dim={state_dim} action_dim={action_dim}",
          flush=True)
    policy_net = QNetwork(state_dim=state_dim, hidden=args.hidden,
                          action_dim=action_dim).to(device)
    target_net = QNetwork(state_dim=state_dim, hidden=args.hidden,
                          action_dim=action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    log_path = os.path.join(args.outdir, "training_log.csv")
    eval_log_path = os.path.join(args.outdir, "eval_log.csv")
    best_eval = None
    eval_history = []
    with open(eval_log_path, "w", newline="") as ef:
        ew = csv.writer(ef)
        ew.writerow(["episode", "eval_mean_agent1", "eval_std_agent1",
                     "eval_min_agent1", "eval_max_agent1",
                     "eval_mean_agent2", "mean_ep_len", "eval_n"])
    t0 = time.time()
    rewards_hist = []
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "epsilon", "loss",
                         "reward_agent2", "ep_len"])
        for ep in range(1, args.episodes + 1):
            eps = max(args.eps_end,
                      args.eps_start - (args.eps_start - args.eps_end)
                      * (ep - 1) / max(1, args.eps_decay_episodes))
            state = env.reset(seed=args.seed * 100000 + ep)
            total, total2, steps = 0.0, 0.0, 0
            ep_losses = []
            done = False
            while not done:
                if random.random() < eps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        s = torch.as_tensor(np.asarray(state, dtype=np.float32),
                                            device=device).unsqueeze(0)
                        action = int(torch.argmax(policy_net(s), dim=1).item())
                next_state, reward, done, info = env.step(int(action))
                buffer.push(state, action, reward, next_state, float(done))
                state = next_state
                total += float(reward)
                total2 += float(info["original_reward_n"][1])
                steps += 1
                if len(buffer) >= args.warmup:
                    s_b, a_b, r_b, s2_b, d_b = buffer.sample(args.batch_size)
                    s_t = torch.as_tensor(s_b, device=device)
                    a_t = torch.as_tensor(a_b, device=device).unsqueeze(1)
                    r_t = torch.as_tensor(r_b, device=device)
                    s2_t = torch.as_tensor(s2_b, device=device)
                    d_t = torch.as_tensor(d_b, device=device)
                    q = policy_net(s_t).gather(1, a_t).squeeze(1)
                    with torch.no_grad():
                        if args.double_dqn:
                            next_a = policy_net(s2_t).argmax(dim=1, keepdim=True)
                            q_next = target_net(s2_t).gather(1, next_a).squeeze(1)
                        else:
                            q_next = target_net(s2_t).max(dim=1)[0]
                        target = r_t + args.gamma * q_next * (1.0 - d_t)
                    if args.loss == "huber":
                        loss = nn.functional.smooth_l1_loss(q, target)
                    else:
                        loss = nn.functional.mse_loss(q, target)
                    optimizer.zero_grad()
                    loss.backward()
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(policy_net.parameters(),
                                                 args.grad_clip)
                    optimizer.step()
                    ep_losses.append(float(loss.item()))
            rewards_hist.append(total)
            if ep % args.target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())
            mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            writer.writerow([ep, round(float(total), 2), round(eps, 4),
                             "" if not ep_losses else round(mean_loss, 6),
                             round(float(total2), 2), steps])
            f.flush()
            if args.eval_every > 0 and ep % args.eval_every == 0:
                pi_eval = greedy_policy_from_net(policy_net, device)
                ev = evaluate(pi_eval, args.eval_episodes,
                              seed_offset=args.eval_seed_offset)
                a1 = ev["agent1"]
                ev_rec = {"episode": ep, "mean": a1["mean"], "std": a1["std"],
                          "min": a1["min"], "max": a1["max"], "n": a1["n"],
                          "mean_agent2": ev["agent2"]["mean"],
                          "mean_ep_len": ev["mean_ep_len"]}
                eval_history.append(ev_rec)
                with open(eval_log_path, "a", newline="") as ef:
                    ew = csv.writer(ef)
                    ew.writerow([ep, round(a1["mean"], 2), round(a1["std"], 2),
                                 int(a1["min"]), int(a1["max"]),
                                 round(ev["agent2"]["mean"], 2),
                                 round(ev["mean_ep_len"], 1), a1["n"]])
                improved = best_eval is None or a1["mean"] > best_eval["mean"]
                if improved:
                    best_eval = ev_rec
                    torch.save({
                        "state_dict": policy_net.state_dict(),
                        "hyperparams": hyper,
                        "env": "PongDuel-v0 (SingleAgentPongDuelWrapper, "
                               "reward_mode=agent1)",
                        "framework": f"torch {torch.__version__}, gym 0.21 + "
                                     "ma-gym (training only)",
                        "best_eval_episode": ep,
                        "best_eval": {k: v for k, v in a1.items()
                                      if k != "rewards"},
                        "train_mean_last100_at_save": float(
                            np.mean(rewards_hist[-100:])),
                    }, os.path.join(args.outdir, "best_checkpoint.pt"))
                print(f"  [periodic eval ep {ep}] greedy {a1['n']}ep agent1: "
                      f"mean={a1['mean']:.2f} std={a1['std']:.2f} "
                      f"min={a1['min']:.0f} max={a1['max']:.0f} "
                      f"(agent2 mean={ev['agent2']['mean']:.2f}) "
                      f"{'NEW BEST' if improved else ''} "
                      f"(best: {best_eval['mean']:.2f} @ ep "
                      f"{best_eval['episode']})", flush=True)
            if ep % 50 == 0 or ep == 1:
                last100 = float(np.mean(rewards_hist[-100:]))
                print(f"ep {ep:4d} reward_agent1={total:5.1f} "
                      f"mean_last100={last100:5.2f} eps={eps:.3f} "
                      f"loss={mean_loss:.4f} elapsed={time.time()-t0:.0f}s",
                      flush=True)
    env.close()
    last100 = float(np.mean(rewards_hist[-100:]))
    train_time = time.time() - t0
    print(f"TRAIN DONE: episodes={args.episodes} "
          f"mean_last100={last100:.2f} time={train_time:.1f}s", flush=True)
    if best_eval is not None:
        print(f"BEST PERIODIC EVAL: agent1 mean={best_eval['mean']:.2f} "
              f"@ episode {best_eval['episode']} "
              f"({best_eval['n']} eval episodes)", flush=True)

    final_net_eval = None
    if best_eval is not None:
        pi_final = greedy_policy_from_net(policy_net, device)
        final_net_eval = evaluate(pi_final, 100)
        a1 = final_net_eval["agent1"]
        print(f"FINAL-NET EVAL (100 ep, greedy, last-episode net): agent1 "
              f"mean={a1['mean']:.2f} std={a1['std']:.2f} "
              f"min={a1['min']:.0f} max={a1['max']:.0f}", flush=True)
        best_ckpt = torch.load(os.path.join(args.outdir, "best_checkpoint.pt"),
                               map_location="cpu", weights_only=False)
        best_net = QNetwork(state_dim=state_dim, hidden=args.hidden,
                            action_dim=action_dim).to(device)
        best_net.load_state_dict(best_ckpt["state_dict"])
        best_net.eval()
        pi = greedy_policy_from_net(best_net, device)
    else:
        pi = greedy_policy_from_net(policy_net, device)
    final_eval = evaluate(pi, 100)
    a1 = final_eval["agent1"]
    print(f"FINAL EVAL (100 ep, greedy): agent1 mean={a1['mean']:.2f} "
          f"std={a1['std']:.2f} min={a1['min']:.0f} max={a1['max']:.0f} "
          f"(agent2 mean={final_eval['agent2']['mean']:.2f})", flush=True)

    ckpt = {
        "state_dict": policy_net.state_dict(),
        "hyperparams": hyper,
        "env": "PongDuel-v0 (SingleAgentPongDuelWrapper, reward_mode=agent1)",
        "framework": f"torch {torch.__version__}, gym 0.21 + ma-gym "
                     "(training only)",
        "train_mean_last100": last100,
        "train_time_s": train_time,
        "final_eval_100ep": {k: v for k, v in
                             (final_net_eval["agent1"]
                              if final_net_eval is not None
                              else final_eval["agent1"]).items()
                             if k != "rewards"},
    }
    torch.save(ckpt, os.path.join(args.outdir, "final_checkpoint.pt"))
    if best_eval is not None:
        import shutil
        shutil.copyfile(os.path.join(args.outdir, "best_checkpoint.pt"),
                        os.path.join(args.outdir, "dqn_checkpoint.pt"))
        print("checkpoint saved: best_checkpoint.pt promoted to "
              "dqn_checkpoint.pt; last-episode net kept as "
              "final_checkpoint.pt", flush=True)
    else:
        torch.save(ckpt, os.path.join(args.outdir, "dqn_checkpoint.pt"))
    with open(os.path.join(args.outdir, "train_summary.json"), "w") as f:
        json.dump({"train_mean_last100": last100,
                   "train_time_s": train_time,
                   "best_periodic_eval": best_eval,
                   "periodic_evals": eval_history,
                   "final_eval_100ep": {k: v for k, v in
                                        final_eval["agent1"].items()
                                        if k != "rewards"},
                   "final_eval_100ep_agent2": {k: v for k, v in
                                               final_eval["agent2"].items()
                                               if k != "rewards"},
                   "final_eval_100ep_mean_ep_len":
                       final_eval["mean_ep_len"],
                   "final_net_eval_100ep": ({k: v for k, v in
                                             final_net_eval["agent1"].items()
                                             if k != "rewards"}
                                            if final_net_eval is not None
                                            else None),
                   "dqn_checkpoint_is": ("best_checkpoint (by periodic "
                                         "eval)" if best_eval is not None
                                         else "final net"),
                   "hyperparams": hyper}, f, indent=2)
    print(f"summary saved: {os.path.join(args.outdir, 'train_summary.json')}",
          flush=True)


if __name__ == "__main__":
    main()
