"""Compute LunarLander baselines and PC/FCI wall-clock timings from checkpoints."""

import json
import time
from pathlib import Path

import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "lunarlander"


def metrics(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
    }


def main():
    replay = np.load(OUT_DIR / "replay.npz")
    states, actions = replay["states"], replay["actions"].astype(int)
    current = np.concatenate((states, np.eye(4)[actions]), axis=1)
    x, y = current[:-1], actions[1:]
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.4, random_state=10, stratify=y
    )
    baselines = {
        "logistic_regression": metrics(LogisticRegression(random_state=1, max_iter=1000), x[train], x[test], y[train], y[test]),
        "decision_tree": metrics(DecisionTreeClassifier(random_state=1), x[train], x[test], y[train], y[test]),
    }
    # Evaluation timing notebook sets INCLUDE_ACTION=False; retain that representation.
    pc_started = time.perf_counter()
    pc(states, alpha=0.05, indep_test="fisherz", stable=False, show_progress=False)
    pc_seconds = time.perf_counter() - pc_started
    fci_started = time.perf_counter()
    fci(states, independence_test_method="fisherz", alpha=0.05, verbose=False)
    fci_seconds = time.perf_counter() - fci_started
    targets = [json.loads((OUT_DIR / "targets" / f"target_{index}.json").read_text()) for index in range(12)]
    result = {
        "environment": "LunarLander-v2", "seed": 99, "episodes": 30,
        "evaluation_scope": "heuristic PD-controller evaluation only; no RL training",
        "baselines": baselines,
        "structure_seconds": {
            "bart_hte_isolated_target_sum": sum(item["elapsed_seconds"] for item in targets),
            "pc_fisherz_wall_clock_8_state_columns": pc_seconds,
            "fci_fisherz_wall_clock_8_state_columns": fci_seconds,
        },
        "bart_targets": targets,
        "timing_note": "BART time sums 12 isolated processes over state and one-hot action targets. PC/FCI follow the timing notebook's state-only representation and report full wall-clock time.",
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
