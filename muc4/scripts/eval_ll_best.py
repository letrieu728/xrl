"""Final 100-ep greedy eval of the saved BEST LunarLander checkpoint.

Replicates exactly the end-of-run final eval in train_dqn_lunarlander.py
(evaluate(env, pi, 100), default seed_offset=10000, greedy), which never
ran because a VM restart killed training at episode 873/1000. The
last-episode net was never saved, so only the best-checkpoint eval is
reproducible; the final-net eval is reported as unavailable, not made up.
"""
import importlib.util
import json
import os
import sys

import torch

OUTDIR = "/home/hatch/workspace/goals/cschttt-new-presentation-paper-selection/files/causalxrl-muc4-mo-rong/results/lunarlander-dqn"
SCRIPT = "/home/hatch/workspace/step3-scratch/group2/train_dqn_lunarlander.py"

spec = importlib.util.spec_from_file_location("ll_trainer", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ckpt = torch.load(os.path.join(OUTDIR, "best_checkpoint.pt"),
                  map_location="cpu", weights_only=False)
print("checkpoint keys:", sorted(ckpt.keys()))
print("best_eval_episode:", ckpt.get("best_eval_episode"))
print("best_eval:", ckpt.get("best_eval"))
print("env:", ckpt.get("env"))

device = torch.device("cpu")
net = mod.QNetwork(state_dim=8, hidden=128, action_dim=4).to(device)
net.load_state_dict(ckpt["state_dict"])
net.eval()
pi = mod.greedy_policy_from_net(net, device)
ev = mod.evaluate("LunarLander-v3", pi, 100)  # default seed_offset=10000
print(f"BEST-CHECKPOINT FINAL EVAL (100 ep, greedy, seeds 10000..10099): "
      f"mean={ev['mean']:.2f} std={ev['std']:.2f} "
      f"min={ev['min']:.2f} max={ev['max']:.2f}")

with open(os.path.join(OUTDIR, "final_eval_best.json"), "w") as f:
    json.dump({"checkpoint": "best_checkpoint.pt",
               "best_eval_episode": ckpt.get("best_eval_episode"),
               "periodic_best_eval": {k: v for k, v in ckpt.get("best_eval", {}).items()},
               "final_eval_100ep": {k: v for k, v in ev.items() if k != "rewards"},
               "eval_protocol": "greedy, 100 episodes, fresh env seeds 10000..10099, "
                                "same evaluate() as train_dqn_lunarlander.py",
               "training_status": "INTERRUPTED by VM restart at episode 873/1000 "
                                  "(last training_log row 2026-11-12T09:52:57Z); "
                                  "final-net eval unavailable (last-episode net "
                                  "was never saved)"}, f, indent=2)
with open(os.path.join(OUTDIR, "final_eval_best_rewards.csv"), "w") as f:
    f.write("episode,seed,reward\n")
    for i, r in enumerate(ev["rewards"]):
        f.write(f"{i},{10000 + i},{r:.2f}\n")
print("saved final_eval_best.json + final_eval_best_rewards.csv")
