"""Infer one CartPole DBN target in an isolated Python/R/dbarts process."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cxrl.lib.causal.infer_network import infer_causes_hte


FEATURES = ["cart_position", "cart_velocity", "pole_angle", "pole_velocity", "push"]
ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = ROOT / "results" / "cartpole" / "replay.npz"
TARGET_DIR = ROOT / "results" / "cartpole" / "targets"


def checkpoint(target, stage):
    """Persist the last reached native-R boundary for post-mortem diagnosis."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / f"target_{target}.progress.json").write_text(
        json.dumps({"target_index": target, "stage": stage}), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=int, choices=range(len(FEATURES)))
    args = parser.parse_args()

    replay = np.load(REPLAY_PATH)
    states = replay["states"]
    actions = replay["actions"].reshape(-1, 1)
    replay_t = np.concatenate((states, actions), axis=1)[:-1]
    replay_tp1 = np.concatenate((states, actions), axis=1)[1:]

    started = time.perf_counter()
    checkpoint(args.target, "replay_loaded")
    checkpoint(args.target, "before_infer_causes_hte")
    causes, _ = infer_causes_hte(replay_t, replay_tp1[:, args.target])
    checkpoint(args.target, "infer_causes_hte_completed")
    result = {
        "target_index": args.target,
        "target": FEATURES[args.target],
        "causes": [FEATURES[index] for index in causes],
        "cause_indices": causes,
        "method": "BART heterogeneous treatment effects",
        "elapsed_seconds": time.perf_counter() - started,
    }
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    target_path = TARGET_DIR / f"target_{args.target}.json"
    target_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    checkpoint(args.target, "result_written")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
