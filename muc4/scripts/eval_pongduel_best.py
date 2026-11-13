#!/usr/bin/env python3
"""Final 100-ep greedy eval of a saved PongDuel DQN checkpoint.

Same protocol as train_dqn_pongduel.evaluate(): seeds 10000..10099,
wrapper reward_mode agent1. Used for the killed step-5 run, whose only
saved net is best_checkpoint.pt (ep 350): the last-episode net was
never saved (final_checkpoint.pt is written only at normal script end)
and is NOT evaluated. Writes final_eval_best.json +
final_eval_best_rewards.csv to --outdir.

Usage:
    step3-scratch/venv/bin/python eval_pongduel_best.py \\
        --checkpoint <best_checkpoint.pt> --outdir <dir>
"""
import argparse, csv, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dqn_pongduel import QNetwork, greedy_policy_from_net, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-offset", type=int, default=10000)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = QNetwork(state_dim=20,
                   hidden=int(ckpt["hyperparams"]["hidden"]), action_dim=9)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    pi = greedy_policy_from_net(net, torch.device("cpu"))
    ev = evaluate(pi, args.episodes, seed_offset=args.seed_offset)
    a1, a2 = ev["agent1"], ev["agent2"]
    out = {"checkpoint": str(args.checkpoint),
           "best_eval_episode": ckpt.get("best_eval_episode"),
           "best_eval_50ep": ckpt.get("best_eval"),
           "final_eval_100ep_agent1": {k: v for k, v in a1.items()
                                       if k != "rewards"},
           "final_eval_100ep_agent2": {k: v for k, v in a2.items()
                                       if k != "rewards"},
           "mean_ep_len": ev["mean_ep_len"],
           "protocol": "greedy, 100 episodes, fresh env seeds "
                       "10000..10099, same evaluate() as "
                       "train_dqn_pongduel.py"}
    (outdir / "final_eval_best.json").write_text(json.dumps(out, indent=2))
    with open(outdir / "final_eval_best_rewards.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "seed", "reward_agent1", "reward_agent2",
                    "ep_len"])
        for i, (x, y, ln) in enumerate(zip(a1["rewards"], a2["rewards"],
                                           ev["ep_lengths"])):
            w.writerow([i, args.seed_offset + i, x, y, ln])
    print("agent1:", {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in a1.items() if k != "rewards"})
    print("wrote", outdir / "final_eval_best.json")


if __name__ == "__main__":
    main()
