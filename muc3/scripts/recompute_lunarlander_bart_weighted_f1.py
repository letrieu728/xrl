"""Recompute LunarLander BART fidelity WITH per-sample predictions saved.

The original Mục 3 run (Codex, Windows, seed 99) saved only aggregate
accuracy + macro-F1 in ``muc3/results/lunarlander/bart_fidelity.json``.
Per-sample predictions were not saved -- but the per-action BART score
checkpoints WERE saved in ``muc3/results/lunarlander/fidelity_scores/``
(``action_{8..11}.npz``: ``score``, ``total_weight``, ``test_indices``,
``valid_indices``, ``action_next``). The aggregate prediction is a
deterministic function of those checkpoints (see
``lunarlander_fidelity_aggregate.py``), so this script reconstructs the
exact predictions of the ORIGINAL BART run -- no BART refit -- and adds
the paper's multiclass metric, weighted F1 (arXiv:2610.42028v1, §5.2).

Aggregation logic (identical to ``lunarlander_fidelity_aggregate.py``):
for each action, the score is ``score / total_weight``; rows whose
total weight is zero fall back to that action's TEST-set class prior
(the group's implementation; the author notebook uses TRAIN-set priors
-- see the sensitivity check below, which found zero changed
predictions on this run). The prediction is the argmax over the four
normalised action scores.

Verification built in:
- the four checkpoints must share identical ``test_indices``,
  ``valid_indices`` and ``action_next`` (the aggregate script enforces
  the same for ``test_indices``);
- recomputed accuracy and macro-F1 must match the stored
  ``bart_fidelity.json`` values to 1e-12, otherwise the script fails:
  the checkpoints would not be the ones behind the stored numbers;
- when ``replay.npz`` and the ``fidelity_masks/`` checkpoints are
  present, the valid/test split is independently reconstructed with
  the ``split_indices`` logic from
  ``lunarlander_fidelity_action_scores.py`` and must match the
  checkpoints exactly.

Why no fresh BART rerun is used for the reported numbers: the author
code path has no seed anywhere (``infer_causes_hte`` calls
``train_test_split`` without ``random_state``; ``bart_train`` in
``bart_regressor.py`` passes no seed to dbarts). Separate
``infer_causes_hte`` calls on the same target already gave different
cause sets in the original run (e.g. action 9: targets ``[2,3,4,5,6,9,10]``,
masks ``[3,4,5,9,10]``, scores ``[4,5,10]`` + self). A fresh rerun is a
new stochastic realisation and would NOT reproduce 82.80% / 64.10%;
its weighted F1 would belong to that new run, not to the reported one.
A fresh rerun was run separately as a variability check only; its
results are kept apart from this file's outputs.

Usage (from the repository root)::

    python muc3/scripts/recompute_lunarlander_bart_weighted_f1.py

Outputs (default ``--out-dir``: ``muc3/results/lunarlander/``):
``bart_fidelity_weighted.json`` and
``bart_fidelity_weighted_predictions.npz`` (``y_true``, ``y_pred``,
``scores``, ``test_indices``, ``valid_indices``). The stored
``bart_fidelity.json`` is read, never overwritten.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

import sklearn

FEATURES = [
    "x", "y", "vx", "vy", "angle", "angular_velocity", "left_contact", "right_contact",
    "do_nothing", "fire_left_engine", "fire_main_engine", "fire_right_engine",
]
ACTION_NAMES = FEATURES[8:]
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def detect_bart_environment():
    """Best-effort record of the R/dbarts/rpy2 versions available here.

    This script itself uses none of them (no BART refit); the versions
    are recorded so the environment that COULD rerun BART is on file.
    Never fails: anything undetectable is reported as such.
    """
    info = {"used_by_this_script": False}
    try:
        import rpy2
        info["rpy2"] = rpy2.__version__
    except Exception:
        info["rpy2"] = "not importable in this Python environment"
    rscript = shutil.which("Rscript")
    if rscript is None:
        info["r"] = "Rscript not found"
        info["dbarts"] = "not detected (no Rscript)"
        return info
    try:
        out = subprocess.run(
            [rscript, "-e", 'cat("R", as.character(getRversion()), "\\n");'
             'cat("dbarts", if (requireNamespace("dbarts", quietly=TRUE)) '
             'as.character(packageVersion("dbarts")) else "NOT-INSTALLED", "\\n")'],
            capture_output=True, text=True, timeout=60)
        for line in out.stdout.splitlines():
            key, _, value = line.partition(" ")
            if key in ("R", "dbarts"):
                info[key.lower() if key == "R" else key] = value.strip()
    except Exception as exc:  # noqa: BLE001 - record, never fail
        info["detection_error"] = str(exc)
    return info


def split_indices(labels, masks):
    """Verbatim copy of split_indices() in lunarlander_fidelity_action_scores.py."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="muc3 directory (default: parent of this script's directory)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: <root>/results/lunarlander)")
    args = parser.parse_args()

    out_dir = args.root / "results" / "lunarlander"
    scores_dir = out_dir / "fidelity_scores"
    masks_dir = out_dir / "fidelity_masks"
    dest = args.out_dir or out_dir
    dest.mkdir(parents=True, exist_ok=True)

    # --- Load the four saved per-action BART score checkpoints. ---
    items, checkpoint_info = [], {}
    for action in range(8, 12):
        path = scores_dir / f"action_{action}.npz"
        data = np.load(path)
        items.append(data)
        info = {"path": f"muc3/results/lunarlander/fidelity_scores/action_{action}.npz",
                "bytes": int(path.stat().st_size), "sha256": sha256_of(path),
                "keys": {k: [int(d) for d in data[k].shape] for k in data.files}}
        sidecar = scores_dir / f"action_{action}.json"
        if sidecar.exists():
            info["elapsed_seconds"] = float(load_json(sidecar)["elapsed_seconds"])
            info["global_causes"] = load_json(sidecar)["global_causes"]
        checkpoint_info[FEATURES[action]] = info

    expected_test = items[0]["test_indices"]
    expected_valid = items[0]["valid_indices"]
    expected_next = items[0]["action_next"]
    for item in items[1:]:
        if not np.array_equal(item["test_indices"], expected_test):
            raise RuntimeError("Action score checkpoints used different test splits.")
        if not np.array_equal(item["valid_indices"], expected_valid):
            raise RuntimeError("Action score checkpoints used different valid sets.")
        if not np.array_equal(item["action_next"], expected_next):
            raise RuntimeError("Action score checkpoints disagree on test labels.")

    y_true = expected_next.astype(int)
    test_priors = np.bincount(y_true, minlength=4) / len(y_true)
    scores = np.column_stack([
        np.divide(item["score"], item["total_weight"],
                  out=np.full(len(y_true), test_priors[offset]),
                  where=item["total_weight"] > 0)
        for offset, item in enumerate(items)
    ])
    y_pred = scores.argmax(axis=1)

    # --- Metrics. ---
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    result_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": [
            {"action": ACTION_NAMES[i], "label": i, "precision": float(precision[i]),
             "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i in range(4)
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
        "confusion_matrix_rows_true_cols_pred": ACTION_NAMES,
        "test_label_counts": {ACTION_NAMES[i]: int(c) for i, c in
                              enumerate(np.bincount(y_true, minlength=4))},
        "test_priors_used_as_fallback": [float(p) for p in test_priors],
        "zero_total_weight_rows_per_action": {
            ACTION_NAMES[o]: int((item["total_weight"] == 0).sum())
            for o, item in enumerate(items)},
        "rows_with_any_zero_total_weight": int(
            np.any(np.column_stack([item["total_weight"] == 0 for item in items]), axis=1).sum()),
    }

    # --- Verify against the stored aggregate numbers (exact match required). ---
    stored = load_json(out_dir / "bart_fidelity.json")
    acc_diff = abs(result_metrics["accuracy"] - float(stored["accuracy"]))
    macro_diff = abs(result_metrics["f1_macro"] - float(stored["f1_macro"]))
    verification = {
        "stored_bart_fidelity_json": {
            "accuracy": float(stored["accuracy"]), "f1_macro": float(stored["f1_macro"]),
            "valid_local_action_samples": int(stored["valid_local_action_samples"]),
            "test_samples": int(stored["test_samples"])},
        "accuracy_absolute_difference": float(acc_diff),
        "f1_macro_absolute_difference": float(macro_diff),
        "checkpoints_share_test_valid_labels": True,
    }
    if acc_diff > 1e-12 or macro_diff > 1e-12:
        raise RuntimeError(
            f"Recomputed accuracy/macro-F1 differ from stored bart_fidelity.json "
            f"(acc diff {acc_diff:.3e}, macro diff {macro_diff:.3e}); the saved "
            f"checkpoints are not the ones behind the stored numbers. Refusing "
            f"to report a weighted F1 for a different run.")
    if int(stored["valid_local_action_samples"]) != len(expected_valid) or \
            int(stored["test_samples"]) != len(y_true):
        raise RuntimeError("Stored sample counts disagree with the checkpoints.")

    # --- Optional: independently reconstruct the split from replay + masks. ---
    split_check, replay_info, prior_sensitivity = None, None, None
    replay_path = out_dir / "replay.npz"
    if replay_path.exists():
        replay = np.load(replay_path)
        actions = replay["actions"].astype(int)
        replay_info = {
            "path": "muc3/results/lunarlander/replay.npz",
            "bytes": int(replay_path.stat().st_size), "sha256": sha256_of(replay_path),
            "transitions": int(len(actions)),
            "keys": {k: [int(d) for d in replay[k].shape] for k in replay.files},
        }
        mask_paths = [masks_dir / f"action_{a}.npz" for a in range(8, 12)]
        if all(p.exists() for p in mask_paths):
            masks = {a: np.load(p)["local_causes"].astype(bool) for a, p in zip(range(8, 12), mask_paths)}
            valid_idx, train_pos, test_pos = split_indices(actions[1:], masks)
            split_check = {
                "reconstructed_valid_matches_checkpoints": bool(np.array_equal(valid_idx, expected_valid)),
                "reconstructed_test_matches_checkpoints": bool(
                    np.array_equal(valid_idx[test_pos], expected_test)),
                "valid_is_all_candidates": bool(np.array_equal(valid_idx, np.arange(len(actions) - 1))),
                "n_train": int(len(train_pos)), "n_test": int(len(test_pos)),
            }
            if not (split_check["reconstructed_valid_matches_checkpoints"]
                    and split_check["reconstructed_test_matches_checkpoints"]):
                raise RuntimeError("Split reconstructed from replay+masks does not match checkpoints.")
            # Sensitivity: author notebook uses TRAIN-set priors as fallback.
            train_priors = np.bincount(actions[1:][valid_idx[train_pos]], minlength=4) / len(train_pos)
            scores_train_prior = np.column_stack([
                np.divide(item["score"], item["total_weight"],
                          out=np.full(len(y_true), train_priors[offset]),
                          where=item["total_weight"] > 0)
                for offset, item in enumerate(items)
            ])
            pred_train_prior = scores_train_prior.argmax(axis=1)
            prior_sensitivity = {
                "notebook_train_priors": [float(p) for p in train_priors],
                "predictions_changed_vs_test_priors": int((pred_train_prior != y_pred).sum()),
                "accuracy_with_train_priors": float(accuracy_score(y_true, pred_train_prior)),
                "f1_macro_with_train_priors": float(
                    f1_score(y_true, pred_train_prior, average="macro", zero_division=0)),
                "f1_weighted_with_train_priors": float(
                    f1_score(y_true, pred_train_prior, average="weighted", zero_division=0)),
            }

    # --- Original BART runtimes (from the saved per-action JSON sidecars). ---
    original_runtime = {}
    for subdir, key in ((masks_dir, "fidelity_masks"), (scores_dir, "fidelity_scores")):
        elapsed = []
        for action in range(8, 12):
            sidecar = subdir / f"action_{action}.json"
            if sidecar.exists():
                elapsed.append(float(load_json(sidecar)["elapsed_seconds"]))
        if elapsed:
            original_runtime[key] = {"per_action_seconds": elapsed, "sum_seconds": float(sum(elapsed))}
    if len(original_runtime) == 2:
        original_runtime["total_seconds"] = float(
            original_runtime["fidelity_masks"]["sum_seconds"]
            + original_runtime["fidelity_scores"]["sum_seconds"])

    payload = {
        "environment": "LunarLander-v2",
        "method": "predictions reconstructed from the saved per-action local-BART "
                  "weighted-treatment score checkpoints (no BART refit); "
                  "aggregation identical to lunarlander_fidelity_aggregate.py",
        "data_source": "original replay.npz (seed 99, 30 episodes, PD controller) and "
                       "the original run's fidelity_scores checkpoints; BART was NOT "
                       "rerun for these numbers",
        "paper_metric": "weighted F1 for multiclass (arXiv:2610.42028v1, section 5.2)",
        "valid_local_action_samples": int(len(expected_valid)),
        "test_samples": int(len(y_true)),
        "split": "valid = any action SCM with >1 non-self local cause (all 7,381 "
                 "candidates kept); train_test_split(test_size=0.4, random_state=10, "
                 "stratify=labels) -- author notebook cell 15",
        "metrics": result_metrics,
        "verification_vs_stored_bart_fidelity_json": verification,
        "split_reconstruction_check": split_check,
        "prior_fallback_sensitivity": prior_sensitivity,
        "score_checkpoints": checkpoint_info,
        "replay_file": replay_info,
        "original_bart_runtime_seconds": original_runtime,
        "bart_refit": False,
        "caveat": "Weighted F1 belongs to the ORIGINAL BART run whose accuracy / "
                  "macro-F1 are stored in bart_fidelity.json. A fresh BART rerun "
                  "would give different numbers: the author path sets no seed "
                  "(unseeded sample split in infer_causes_hte, unseeded dbarts "
                  "fit), so cause sets and scores vary run to run.",
        "script": "muc3/scripts/recompute_lunarlander_bart_weighted_f1.py",
        "computed_at": datetime.now().astimezone().isoformat(),
        "environment_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "r_dbarts": "not used by this script (no BART refit)",
        },
        "bart_environment_available": detect_bart_environment(),
    }

    (dest / "bart_fidelity_weighted.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        dest / "bart_fidelity_weighted_predictions.npz",
        y_true=y_true, y_pred=y_pred, scores=scores,
        test_indices=expected_test, valid_indices=expected_valid)
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {dest / 'bart_fidelity_weighted.json'} and "
          f"{dest / 'bart_fidelity_weighted_predictions.npz'}", file=sys.stderr)


if __name__ == "__main__":
    main()
