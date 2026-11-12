#!/usr/bin/env python3
"""Train a DQN agent on LunarLander-v3 (CausalXRL extension D, Group 2).

Copied from train_dqn_cartpole.py (run9 recipe) with ONLY these changes,
all documented:
  * state/action dims read from the env (8 -> 128 -> 128 -> 4) instead of
    the hard-coded CartPole dims (4 -> 128 -> 128 -> 2);
  * defaults: --env LunarLander-v3, --episodes 1000, --eps-decay-episodes
    650 (same 65% ratio as run9: 1300/2000);
  * rewards logged as floats (LunarLander rewards are shaped floats, can
    be negative; CartPole script truncated to int).
Everything else (Double DQN, lr, gamma, batch, buffer, eps schedule shape,
target update, warmup, MSE, periodic greedy eval + best-checkpoint
selection, eval seed protocol) is identical to the run9 invocation.

Self-contained, written from scratch in PyTorch (no stable-baselines3).
Q-network: MLP 4 -> 128 -> 128 -> 2, replay buffer, epsilon-greedy,
target network, Adam.

Training uses modern gymnasium (NOT gym 0.20, which is incompatible with
Python 3.12). This is acceptable for TRAINING ONLY: the trained policy is
a plain function pi(state) -> action, so it can later plug into the
authors' XRL(env, pi) interface, which only needs env + pi.

Usage:
    python train_dqn_lunarlander.py --seed 0 --episodes 1000 --double-dqn \
        --eps-decay-episodes 650 --eval-every 50 --eval-episodes 50 \
        --outdir ../results/lunarlander-dqn
"""
import argparse
import csv
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class QNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden=128, action_dim=2):
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


def evaluate(env_name, pi, n_episodes, seed_offset=10000):
    """Greedy evaluation, one fresh env seed per episode (deterministic)."""
    env = gym.make(env_name)
    rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed_offset + i)
        total = 0.0
        done = False
        while not done:
            action = pi(np.asarray(state, dtype=np.float32))
            state, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            done = terminated or truncated
        rewards.append(total)
    env.close()
    r = np.array(rewards, dtype=np.float64)
    return {"n": int(n_episodes), "mean": float(r.mean()), "std": float(r.std()),
            "min": float(r.min()), "max": float(r.max()),
            "rewards": [float(x) for x in rewards]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--buffer-size", type=int, default=50000)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.02)
    ap.add_argument("--eps-decay-episodes", type=int, default=650,
                    help="episodes over which epsilon decays linearly to eps-end")
    ap.add_argument("--target-update-every", type=int, default=10,
                    help="episodes between hard target-network updates")
    ap.add_argument("--warmup", type=int, default=1000,
                    help="steps before learning starts")
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="max grad norm for clipping, 0 = off")
    ap.add_argument("--double-dqn", action="store_true",
                    help="Double DQN (run5): action selection by policy net, "
                    "evaluation by target net: target = r + gamma * "
                    "Q_target(s', argmax_a Q_policy(s', a)) * (1-done)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="episodes between periodic greedy evaluations "
                    "(0 = off). When on, the best checkpoint BY PERIODIC "
                    "EVAL (not the last-episode net) is saved to "
                    "best_checkpoint.pt and promoted to dqn_checkpoint.pt. "
                    "Run7+: eval every 50 episodes, 50 eval episodes.")
    ap.add_argument("--eval-episodes", type=int, default=50,
                    help="greedy episodes per periodic evaluation")
    ap.add_argument("--eval-seed-offset", type=int, default=10000,
                    help="env seed offset for periodic eval (same seeds as "
                    "the final 100-ep eval, whose offset is 10000)")
    ap.add_argument("--env", default="LunarLander-v3")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    hyper = vars(args)
    print(f"seed={args.seed} episodes={args.episodes} env={args.env} "
          f"torch={torch.__version__} device=cpu", flush=True)

    env = gym.make(args.env)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    print(f"env dims: state_dim={state_dim} action_dim={action_dim}", flush=True)
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
    best_eval = None  # {"episode": int, "mean": float, ...}
    eval_history = []
    if args.eval_every > 0:
        with open(eval_log_path, "w", newline="") as ef:
            ew = csv.writer(ef)
            ew.writerow(["episode", "eval_mean", "eval_std", "eval_min",
                         "eval_max", "eval_n"])
    t0 = time.time()
    rewards_hist = []
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "epsilon", "loss"])
        for ep in range(1, args.episodes + 1):
            eps = max(args.eps_end,
                      args.eps_start - (args.eps_start - args.eps_end)
                      * (ep - 1) / max(1, args.eps_decay_episodes))
            state, _ = env.reset(seed=args.seed * 100000 + ep)
            total = 0.0
            ep_losses = []
            done = False
            while not done:
                if random.random() < eps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        s = torch.as_tensor(state, dtype=torch.float32,
                                            device=device).unsqueeze(0)
                        action = int(torch.argmax(policy_net(s), dim=1).item())
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.push(state, action, reward, next_state, float(done))
                state = next_state
                total += reward

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
                             "" if not ep_losses else round(mean_loss, 6)])
            f.flush()
            if args.eval_every > 0 and ep % args.eval_every == 0:
                pi_eval = greedy_policy_from_net(policy_net, device)
                ev = evaluate(args.env, pi_eval, args.eval_episodes,
                              seed_offset=args.eval_seed_offset)
                ev_rec = {"episode": ep, "mean": ev["mean"],
                          "std": ev["std"], "min": ev["min"],
                          "max": ev["max"], "n": ev["n"]}
                eval_history.append(ev_rec)
                with open(eval_log_path, "a", newline="") as ef:
                    ew = csv.writer(ef)
                    ew.writerow([ep, round(ev["mean"], 2), round(ev["std"], 2),
                                 round(ev["min"], 2), round(ev["max"], 2), ev["n"]])
                improved = best_eval is None or ev["mean"] > best_eval["mean"]
                if improved:
                    best_eval = ev_rec
                    torch.save({
                        "state_dict": policy_net.state_dict(),
                        "hyperparams": hyper,
                        "env": args.env,
                        "framework": f"torch {torch.__version__}, "
                                     "gymnasium (training only)",
                        "best_eval_episode": ep,
                        "best_eval": {k: v for k, v in ev.items()
                                      if k != "rewards"},
                        "train_mean_last100_at_save": float(
                            np.mean(rewards_hist[-100:])),
                    }, os.path.join(args.outdir, "best_checkpoint.pt"))
                print(f"  [periodic eval ep {ep}] greedy {ev['n']}ep: "
                      f"mean={ev['mean']:.2f} std={ev['std']:.2f} "
                      f"min={ev['min']:.0f} max={ev['max']:.0f} "
                      f"{'NEW BEST' if improved else ''} "
                      f"(best so far: "
                      f"{best_eval['mean']:.2f} @ ep {best_eval['episode']})",
                      flush=True)
            if ep % 50 == 0 or ep == 1:
                last100 = float(np.mean(rewards_hist[-100:]))
                print(f"ep {ep:4d} reward={total:7.1f} "
                      f"mean_last100={last100:6.1f} eps={eps:.3f} "
                      f"loss={mean_loss:.4f} elapsed={time.time()-t0:.0f}s",
                      flush=True)
    env.close()

    last100 = float(np.mean(rewards_hist[-100:]))
    train_time = time.time() - t0
    print(f"TRAIN DONE: episodes={args.episodes} "
          f"mean_last100={last100:.2f} time={train_time:.1f}s", flush=True)
    if best_eval is not None:
        print(f"BEST PERIODIC EVAL: mean={best_eval['mean']:.2f} "
              f"std={best_eval['std']:.2f} min={best_eval['min']:.0f} "
              f"max={best_eval['max']:.0f} @ episode {best_eval['episode']} "
              f"({best_eval['n']} eval episodes)", flush=True)

    # Final greedy evaluation over 100 episodes (deterministic seeds).
    # With periodic eval on, the FINAL EVAL is run on the BEST checkpoint
    # (the net promoted to dqn_checkpoint.pt); the last-episode net is
    # also evaluated for the record, so the selection gain is visible
    # and not hidden.
    final_net_eval = None
    if best_eval is not None:
        pi_final = greedy_policy_from_net(policy_net, device)
        final_net_eval = evaluate(args.env, pi_final, 100)
        print(f"FINAL-NET EVAL (100 ep, greedy, last-episode net): "
              f"mean={final_net_eval['mean']:.2f} "
              f"std={final_net_eval['std']:.2f} "
              f"min={final_net_eval['min']:.0f} "
              f"max={final_net_eval['max']:.0f}", flush=True)
        best_ckpt = torch.load(os.path.join(args.outdir, "best_checkpoint.pt"),
                               map_location="cpu", weights_only=False)
        best_net = QNetwork(state_dim=state_dim, hidden=args.hidden, action_dim=action_dim).to(device)
        best_net.load_state_dict(best_ckpt["state_dict"])
        best_net.eval()
        pi = greedy_policy_from_net(best_net, device)
    else:
        pi = greedy_policy_from_net(policy_net, device)
    final_eval = evaluate(args.env, pi, 100)
    print(f"FINAL EVAL (100 ep, greedy): mean={final_eval['mean']:.2f} "
          f"std={final_eval['std']:.2f} min={final_eval['min']:.0f} "
          f"max={final_eval['max']:.0f}", flush=True)

    ckpt = {
        "state_dict": policy_net.state_dict(),
        "hyperparams": hyper,
        "env": args.env,
        "framework": f"torch {torch.__version__}, gymnasium (training only)",
        "train_mean_last100": last100,
        "train_time_s": train_time,
        # final_checkpoint.pt holds the LAST-episode net, so its
        # final_eval_100ep must describe that net (final_net_eval when
        # periodic eval is on); the best net's 100-ep eval lives in
        # train_summary.json's final_eval_100ep.
        "final_eval_100ep": {k: v for k, v in
                             (final_net_eval if final_net_eval is not None
                              else final_eval).items() if k != "rewards"},
    }
    torch.save(ckpt, os.path.join(args.outdir, "final_checkpoint.pt"))
    if best_eval is not None:
        # Promote the best-by-periodic-eval checkpoint to the canonical
        # dqn_checkpoint.pt (downstream scripts load this path).
        import shutil
        shutil.copyfile(os.path.join(args.outdir, "best_checkpoint.pt"),
                        os.path.join(args.outdir, "dqn_checkpoint.pt"))
        print("checkpoint saved: best_checkpoint.pt promoted to "
              "dqn_checkpoint.pt; last-episode net kept as "
              "final_checkpoint.pt", flush=True)
    else:
        torch.save(ckpt, os.path.join(args.outdir, "dqn_checkpoint.pt"))
        print(f"checkpoint saved: {os.path.join(args.outdir, 'dqn_checkpoint.pt')}",
              flush=True)
    with open(os.path.join(args.outdir, "train_summary.json"), "w") as f:
        json.dump({"train_mean_last100": last100,
                   "train_time_s": train_time,
                   "best_periodic_eval": best_eval,
                   "periodic_evals": eval_history,
                   # final_eval_100ep = eval of the PROMOTED checkpoint
                   # (best net when periodic eval is on, else the final
                   # net); final_net_eval_100ep = eval of the
                   # last-episode net. final_checkpoint.pt's own
                   # final_eval_100ep describes the last-episode net.
                   "final_eval_100ep": {k: v for k, v in final_eval.items()
                                        if k != "rewards"},
                   "final_net_eval_100ep": ({k: v for k, v in
                                             final_net_eval.items()
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
