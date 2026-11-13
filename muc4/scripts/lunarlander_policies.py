#!/usr/bin/env python3
"""LunarLander policies for the CausalXRL extension (Chapter 4, step 4).

* heuristic_policy(state) -- PD controller from the CausalXRL repo
  (github.com/yanmluk/causal-xrl, cxrl/jupyter/examples/LunarLander.ipynb
  cell 5, verbatim; same formula in eval_fidelity_Lunar Lander.ipynb).
* load_dqn_policy(checkpoint_path) -- loads the LunarLander DQN trained
  by scripts/train_dqn_lunarlander.py (MLP 8 -> 128 -> 128 -> 4) and
  returns pi(state) -> int in {0,1,2,3}.

state is a numpy array of 8 floats
[x, y, vx, vy, angle, angular_velocity, left_contact, right_contact];
action: 0 do_nothing, 1 fire_left, 2 fire_main, 3 fire_right.
"""
import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Must match train_dqn_lunarlander.py (8 -> 128 -> 128 -> 4)."""

    def __init__(self, state_dim=8, hidden=128, action_dim=4):
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
    """PD controller from the CausalXRL repo (LunarLander.ipynb)."""
    state = np.asarray(state, dtype=np.float64)
    x, y, vx, vy, angle, angular_velocity, left_contact, right_contact = state
    target_angle = 0.5 * x + 1.0 * vx
    target_angle = np.clip(target_angle, -0.4, 0.4)
    target_y = 0.55 * abs(x)
    angle_control = 0.5 * (target_angle - angle) - 1.0 * angular_velocity
    hover_control = 0.5 * (target_y - y) - 0.5 * vy
    if left_contact or right_contact:
        angle_control = 0.0
        hover_control = -0.5 * vy
    if hover_control > abs(angle_control) and hover_control > 0.05:
        return 2
    elif angle_control < -0.05:
        return 3
    elif angle_control > 0.05:
        return 1
    else:
        return 0


def load_dqn_policy(checkpoint_path):
    """Load a LunarLander DQN checkpoint and return pi(state) -> int."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyperparams", {})
    net = QNetwork(state_dim=8, hidden=int(hp.get("hidden", 128)),
                   action_dim=4)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    def pi(state):
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            return int(torch.argmax(net(s), dim=1).item())

    pi.checkpoint = ckpt
    return pi
