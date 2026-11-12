"""Generate one action's local-BART fidelity score vector in an isolated process."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from cxrl.lib.causal.infer_network import infer_causes_hte
from cxrl.lib.models import BARTRegressor


FEATURES = [
    "x", "y", "vx", "vy", "angle", "angular_velocity", "left_contact", "right_contact",
    "do_nothing", "fire_left_engine", "fire_main_engine", "fire_right_engine",
]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lunarlander"
MASKS = OUT / "fidelity_masks"
SCORES = OUT / "fidelity_scores"


def split_indices(labels, masks):
    valid = np.any(
        np.column_stack([(mask.sum(axis=1) - mask[:, action]) > 1 for action, mask in masks.items()]),
        axis=1,
    )
    valid_indices = np.flatnonzero(valid)
    valid_labels = labels[valid_indices]
    train, test = train_test_split(
        np.arange(len(valid_indices)), test_size=0.4, random_state=10,
        stratify=valid_labels if len(np.unique(valid_labels)) > 1 else None,
    )
    return valid_indices, train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=int, choices=range(8, 12))
    args = parser.parse_args()
    replay = np.load(OUT / "replay.npz")
    actions = replay["actions"].astype(int)
    current = np.concatenate((replay["states"], np.eye(4)[actions]), axis=1)
    x, action_next = current[:-1], actions[1:]
    masks = {
        action: np.load(MASKS / f"action_{action}.npz")["local_causes"].astype(bool)
        for action in range(8, 12)
    }
    valid_indices, train_positions, test_positions = split_indices(action_next, masks)
    train_indices, test_indices = valid_indices[train_positions], valid_indices[test_positions]
    x_train, x_test = x[train_indices], x[test_indices]
    y_train = (action_next[train_indices] == args.action - 8).astype(float)

    started = time.perf_counter()
    global_causes, action_scm = infer_causes_hte(x, (action_next == args.action - 8).astype(float))
    causes = [cause for cause in global_causes if cause != args.action]
    covariates = [column for column in range(len(FEATURES)) if column not in causes]
    predictions = {}
    for treatment in causes:
        columns = [treatment] + covariates
        surrogate = BARTRegressor()
        surrogate.fit(x_train[:, columns], y_train)
        predictions[treatment] = np.asarray(surrogate.predict(x_test[:, columns]))

    score = np.zeros(len(test_indices), dtype=float)
    total_weight = np.zeros(len(test_indices), dtype=float)
    local = masks[args.action][test_indices]
    for treatment in causes:
        active = local[:, treatment]
        if not np.any(active):
            continue
        effects = np.asarray(
            action_scm.ITE(x_test[active], int(treatment), additive=treatment < 6)
        ).mean(axis=0)
        weight = np.abs(effects)
        score[active] += predictions[treatment][active] * weight
        total_weight[active] += weight
    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SCORES / f"action_{args.action}.npz", score=score, total_weight=total_weight,
        test_indices=test_indices, valid_indices=valid_indices, action_next=action_next[test_indices],
    )
    result = {
        "action_index": args.action, "action": FEATURES[args.action],
        "global_causes": [FEATURES[cause] for cause in causes],
        "valid_samples": int(len(valid_indices)), "test_samples": int(len(test_indices)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (SCORES / f"action_{args.action}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
