"""Run the CartPole BART local-action fidelity calculation from the eval notebook."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from cxrl.lib.causal.infer_network import infer_causes_hte
from cxrl.lib.models import BARTRegressor


FEATURES = ["cart_position", "cart_velocity", "pole_angle", "pole_velocity", "push"]
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cartpole"


def main():
    replay = np.load(OUT_DIR / "replay.npz")
    current = np.concatenate((replay["states"], replay["actions"].reshape(-1, 1)), axis=1)
    x = current[:-1]
    y = current[1:, 4].astype(int)

    # Rebuild only the action SCM in this process so its local HTE clusters remain available.
    action_causes, action_scm = infer_causes_hte(x, y)
    valid = []
    for index, query in enumerate(current[:-1]):
        local_causes = action_scm.local_treatments(query.reshape(1, -1))
        non_self = [cause for cause in np.nonzero(local_causes[0])[0] if cause != 4]
        if len(non_self) > 1:
            valid.append(index)
    if not valid:
        raise RuntimeError("No valid local-action samples; fidelity cannot be computed.")

    valid = np.asarray(valid)
    eval_x = current[valid]
    eval_y = current[valid + 1, 4].astype(int)
    indices = np.arange(len(valid))
    train_idx, test_idx = train_test_split(indices, test_size=0.4, random_state=10, stratify=eval_y)
    x_train, x_test = eval_x[train_idx], eval_x[test_idx]
    y_train, y_test = eval_y[train_idx], eval_y[test_idx]

    # This is the same reduced-model construction as eval_fidelity_Cart Pole.ipynb.
    cause_set = list(action_causes)
    covariates = [i for i in range(len(FEATURES)) if i not in cause_set]
    reduced_predictions = {}
    for treatment in cause_set:
        columns = [treatment] + covariates
        model = BARTRegressor()
        model.fit(x_train[:, columns], y_train)
        reduced_predictions[treatment] = np.asarray(model.predict(x_test[:, columns]))

    weighted_predictions = []
    for prediction_index, eval_index in enumerate(test_idx):
        query = eval_x[eval_index : eval_index + 1]
        local_causes = action_scm.local_treatments(query)
        numerator, denominator = 0.0, 0.0
        for treatment in np.nonzero(local_causes[0])[0]:
            if treatment not in reduced_predictions:
                continue
            weight = abs(np.asarray(action_scm.ITE(query, int(treatment), additive=treatment < 4)).mean())
            numerator += reduced_predictions[treatment][prediction_index] * weight
            denominator += weight
        if denominator == 0:
            raise RuntimeError("A selected local action model has zero total treatment weight.")
        weighted_predictions.append(numerator / denominator)

    predicted = (np.asarray(weighted_predictions) > 0.5).astype(int)
    result = {
        "environment": "CartPole-v0",
        "method": "repository local BART weighted-treatment fidelity logic",
        "valid_local_action_samples": int(len(valid)),
        "test_samples": int(len(y_test)),
        "action_global_causes": [FEATURES[index] for index in cause_set],
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
    }
    (OUT_DIR / "bart_fidelity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
