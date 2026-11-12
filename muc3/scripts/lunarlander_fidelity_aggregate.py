"""Aggregate isolated one-vs-rest BART action score checkpoints."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lunarlander"
SCORES = OUT / "fidelity_scores"


def main():
    items = [np.load(SCORES / f"action_{action}.npz") for action in range(8, 12)]
    expected = items[0]["test_indices"]
    if any(not np.array_equal(item["test_indices"], expected) for item in items[1:]):
        raise RuntimeError("Action score checkpoints used different test splits.")
    action_next = items[0]["action_next"].astype(int)
    priors = np.bincount(action_next, minlength=4) / len(action_next)
    scores = np.column_stack([
        np.divide(item["score"], item["total_weight"], out=np.full(len(expected), priors[offset]), where=item["total_weight"] > 0)
        for offset, item in enumerate(items)
    ])
    predicted = scores.argmax(axis=1)
    result = {
        "environment": "LunarLander-v2",
        "method": "repository local BART weighted-treatment fidelity logic, isolated per action",
        "valid_local_action_samples": int(len(items[0]["valid_indices"])),
        "test_samples": int(len(action_next)),
        "accuracy": float(accuracy_score(action_next, predicted)),
        "f1_macro": float(f1_score(action_next, predicted, average="macro", zero_division=0)),
        "note": "Each one-vs-rest action score was computed in a separate Python/R process; rows without an active local cause use that action's test-set class prior.",
    }
    (OUT / "bart_fidelity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
