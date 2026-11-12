"""Persist one LunarLander action SCM's local-cause mask for fidelity evaluation."""

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
OUT_DIR = ROOT / "results" / "lunarlander" / "fidelity_masks"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=int, choices=range(8, 12))
    args = parser.parse_args()
    replay = np.load(ROOT / "results" / "lunarlander" / "replay.npz")
    current = np.concatenate((replay["states"], np.eye(4)[replay["actions"].astype(int)]), axis=1)
    x, y = current[:-1], current[1:, args.target]
    started = time.perf_counter()
    causes, model = infer_causes_hte(x, y)
    # The evaluation notebook passes raw queries to local_treatments; preserve that behavior.
    local = model.local_treatments(x).astype(np.uint8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"action_{args.target}.npz", local_causes=local)
    result = {
        "action_index": args.target, "action": FEATURES[args.target],
        "global_causes": [FEATURES[index] for index in causes],
        "global_cause_indices": causes,
        "samples_with_more_than_one_nonself_local_cause": int(
            sum((row.sum() - row[args.target]) > 1 for row in local)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT_DIR / f"action_{args.target}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
