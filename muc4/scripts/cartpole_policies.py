#!/usr/bin/env python3
"""CartPole policies for the CausalXRL extension (Chapter 4).

Contains BOTH policies so they can be compared head-to-head and plugged
into the authors' XRL(env, pi) interface, which only needs a plain
function pi(state) -> action:

* heuristic_policy(state) -- the hand-written heuristic from the
  CausalXRL repo (github.com/yanmluk/causal-xrl), CartPole branch:
      return 1 if 0.4*(state[0]/2.4) + 0.6*(state[2]/2.095) >= 0 else 0
* load_dqn_policy(checkpoint_path) -- loads the DQN trained by
  scripts/train_dqn_cartpole.py and returns pi(state) -> int (0 or 1).

state is a numpy array of 4 floats [x, x_dot, theta, theta_dot];
action is 0 (push left) or 1 (push right).

Smoke test:  python cartpole_policies.py [checkpoint_path]
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Must match the network in train_dqn_cartpole.py (4 -> 128 -> 128 -> 2)."""

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


def heuristic_policy(state):
    """Hand-written heuristic from the CausalXRL repo (CartPole)."""
    state = np.asarray(state, dtype=np.float64)
    return 1 if 0.4 * (state[0] / 2.4) + 0.6 * (state[2] / 2.095) >= 0 else 0


def load_dqn_policy(checkpoint_path):
    """Load a DQN checkpoint and return pi(state) -> int."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("hyperparams", {}).get("hidden", 128))
    net = QNetwork(hidden=hidden)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    def pi(state):
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            return int(torch.argmax(net(s), dim=1).item())

    pi.checkpoint = ckpt  # attach metadata for callers that want it
    return pi


if __name__ == "__main__":
    # --- smoke test: heuristic policy (exact repo formula) ---
    assert heuristic_policy(np.zeros(4)) == 1, "0.0 >= 0 must give 1"
    assert heuristic_policy(np.array([2.4, 0.0, 0.0, 0.0])) == 1
    assert heuristic_policy(np.array([-2.4, 0.0, 0.0, 0.0])) == 0
    assert heuristic_policy(np.array([0.0, 0.0, 2.095, 0.0])) == 1
    assert heuristic_policy(np.array([0.0, 0.0, -2.095, 0.0])) == 0
    print("heuristic_policy: 5/5 assertions passed")

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "results", "cartpole-dqn", "dqn_checkpoint.pt")
    if os.path.exists(ckpt_path):
        pi = load_dqn_policy(ckpt_path)
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        state, _ = env.reset(seed=0)
        rewards = []
        for _ in range(5):
            total = 0.0
            done = False
            while not done:
                a = pi(state)
                assert a in (0, 1), f"action must be 0/1, got {a}"
                state, r, term, trunc, _ = env.step(a)
                total += r
                done = term or trunc
            rewards.append(total)
            state, _ = env.reset(seed=len(rewards))
        env.close()
        print(f"load_dqn_policy: loaded {ckpt_path}, "
              f"5-episode smoke rewards = {rewards}")
        print("SMOKE TEST PASSED")
    else:
        print(f"no checkpoint at {ckpt_path} -- DQN smoke test skipped "
              "(run train_dqn_cartpole.py first)")
        print("SMOKE TEST PASSED (heuristic only)")
