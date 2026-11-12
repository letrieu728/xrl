"""Infer one one-hot LunarLander DBN target in an isolated Python/R process."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cxrl.lib.causal.infer_network import infer_causes_hte


FEATURES = [
    "x", "y", "vx", "vy", "angle", "angular_velocity", "left_contact", "right_contact",
    "do_nothing", "fire_left_engine", "fire_main_engine", "fire_right_engine",
]
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "lunarlander"


def checkpoint(target, stage):
    target_dir = OUT_DIR / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"target_{target}.progress.json").write_text(
        json.dumps({"target_index": target, "stage": stage}), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=int, choices=range(len(FEATURES)))
    args = parser.parse_args()
    replay = np.load(OUT_DIR / "replay.npz")
    states = replay["states"]
    actions = replay["actions"].astype(int)
    one_hot_actions = np.eye(4, dtype=float)[actions]
    current = np.concatenate((states, one_hot_actions), axis=1)
    replay_t, replay_tp1 = current[:-1], current[1:]

    checkpoint(args.target, "replay_loaded")
    started = time.perf_counter()
    checkpoint(args.target, "before_infer_causes_hte")
    causes, _ = infer_causes_hte(replay_t, replay_tp1[:, args.target])
    result = {
        "target_index": args.target,
        "target": FEATURES[args.target],
        "causes": [FEATURES[index] for index in causes],
        "cause_indices": causes,
        "method": "BART heterogeneous treatment effects",
        "one_hot_actions": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    target_dir = OUT_DIR / "targets"
    (target_dir / f"target_{args.target}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    checkpoint(args.target, "result_written")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
