"""Verify the S3 agent-3.pt state dict loads into the causal-xrl Agent class
(architecture copied verbatim from cxrl_microRTS/lib/gym_microrts_rl/gym_microrts_rl.py
lines 70-85, with layer_init from env_tools.py lines 134-137). No gym_microrts
import: this checks weights/architecture compatibility only."""
import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, mapsize=16 * 16, nvec_sum=78):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        self.nvec_sum = nvec_sum
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 6 * 6, 256)),
            nn.ReLU(),)
        self.actor = layer_init(nn.Linear(256, self.mapsize * nvec_sum), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2)))


sd = torch.load("/home/hatch/workspace/step3-scratch/agent-3.pt", map_location="cpu", weights_only=True)
print("type:", type(sd).__name__, "| keys:", len(sd))
for k in sorted(sd):
    print(f"  {k}: {tuple(sd[k].shape)} {sd[k].dtype}")
agent = Agent()
missing, unexpected = agent.load_state_dict(sd, strict=False), None
res = agent.load_state_dict(sd, strict=True)
print("strict load_state_dict: OK (no error)")
agent.eval()
x = torch.zeros(1, 16, 16, 27)
with torch.no_grad():
    h = agent(x)
    logits = agent.actor(h)
    value = agent.critic(h)
print("forward: hidden", tuple(h.shape), "actor logits", tuple(logits.shape), "critic", tuple(value.shape))
print("logits finite:", bool(torch.isfinite(logits).all()), "| value:", float(value.item()))
print("total params:", sum(p.numel() for p in agent.parameters()))
