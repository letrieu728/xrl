"""Compute CartPole baselines and PC/FCI wall-clock timings on the saved replay."""

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


FEATURES = ["cart_position", "cart_velocity", "pole_angle", "pole_velocity", "push"]
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cartpole"


def score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
    }


def main():
    replay = np.load(OUT_DIR / "replay.npz")
    states = replay["states"]
    actions = replay["actions"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        states, actions, test_size=0.2, random_state=99, stratify=actions
    )
    baselines = {
        "logistic_regression": score(
            LogisticRegression(random_state=99, max_iter=1000),
            x_train, x_test, y_train, y_test,
        ),
        "decision_tree": score(
            DecisionTreeClassifier(random_state=99), x_train, x_test, y_train, y_test
        ),
    }

    # Time the algorithms' complete wall-clock execution, not a simulated CI block.
    structure_data = np.concatenate((states, actions.reshape(-1, 1)), axis=1)
    pc_started = time.perf_counter()
    pc(structure_data, alpha=0.05, indep_test="fisherz", stable=False, show_progress=False)
    pc_seconds = time.perf_counter() - pc_started
    fci_started = time.perf_counter()
    fci(structure_data, independence_test_method="fisherz", alpha=0.05, verbose=False)
    fci_seconds = time.perf_counter() - fci_started

    target_results = []
    for index, target in enumerate(FEATURES):
        data = json.loads((OUT_DIR / "targets" / f"target_{index}.json").read_text())
        target_results.append(data)
    bart_seconds = sum(item["elapsed_seconds"] for item in target_results)
    result = {
        "environment": "CartPole-v0",
        "seed": 99,
        "episodes": 200,
        "evaluation_scope": "heuristic-policy evaluation only; no RL training",
        "baselines": baselines,
        "structure_seconds": {
            "bart_hte_isolated_target_sum": bart_seconds,
            "pc_fisherz_wall_clock": pc_seconds,
            "fci_fisherz_wall_clock": fci_seconds,
        },
        "bart_targets": target_results,
        "timing_note": "BART is the sum of five isolated fresh Python/R processes. PC/FCI are full algorithm wall-clock times on the same replay; they are not the paper's simulated CI-test timer.",
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
