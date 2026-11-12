"""Recompute CartPole LR/DT baselines on the BART run's exact test set.

Split finding (read from the committed scripts, verified here by
reconstruction):

- BART (``muc3/scripts/cartpole_bart_fidelity.py``): valid rows are the
  t -> t+1 candidates whose local action model has more than one
  non-self cause; the stored ``bart_fidelity.json`` reports 6,114 valid
  samples, equal to the full candidate count (6,115 transitions - 1),
  so the valid set IS every candidate row, in ascending index order.
  The split is ``train_test_split(arange(6114), test_size=0.4,
  random_state=10, stratify=eval_y)`` -- STRATIFIED. Features are the
  full row at t including the action (5 columns); the target is the
  action at t+1.
- Notebook-protocol baselines (``muc3/results/baselines_recomputed.json``,
  primary): the same 6,114 pairs and seed, but the author notebook
  (``eval_fidelity_Cart Pole.ipynb``, cell 15) splits WITHOUT stratify.
  Same sizes (train 3,668 / test 2,446), DIFFERENT test rows.

So the published baseline numbers and the BART numbers were computed
on different test rows. This script evaluates LR/DT on the BART split
exactly (same rows, same features, same target), keeping the notebook
model settings (``DecisionTreeClassifier(random_state=1)``,
``LogisticRegression(random_state=1)`` with notebook defaults,
``max_iter=100``; notebook cells 17/19). It does NOT rerun BART; the
BART values are copied by reference from ``bart_fidelity.json``.

The aligned numbers are the ones comparable with the group's BART
run. The unstratified primary numbers in ``baselines_recomputed.json``
stay the notebook-protocol comparison. Both are reported side by
side in the output so the two comparisons are never conflated.

Assumption, stated plainly: the per-step valid mask was not saved, so
the valid set is established by the count argument above (stored
valid count == candidate count, valid rows are a subset of the
candidates, hence equal to all of them) plus the ascending-order loop
in the BART script. If the stored count were wrong, the alignment
claim would fail; the script therefore asserts the count equality
and, when ``baselines_recomputed.json`` is present, asserts that the
recomputed splits reproduce its stored primary AND stratified-variant
numbers exactly.

Usage (from the repository root)::

    python muc3/scripts/align_cartpole_baselines_to_bart_split.py

Outputs (default ``--out-dir``: ``muc3/results/``):
``baselines_aligned.json`` and ``baselines_aligned.md``. Existing
``baselines_recomputed.json`` / ``.md`` are read, never overwritten.
"""

import argparse
import hashlib
import json
import platform
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import sklearn

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["cart_position", "cart_velocity", "pole_angle", "pole_velocity", "push"]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def counts(labels):
    values, freqs = np.unique(np.asarray(labels), return_counts=True)
    return {int(v): int(f) for v, f in zip(values, freqs)}


def score_model(model, x_train, x_test, y_train, y_test):
    caught = []
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
        caught = [str(w.message) for w in found]
    predicted = model.predict(x_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=[0, 1], zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1_binary": float(f1_score(y_test, predicted, zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "per_class": [
            {"label": int(label), "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate([0, 1])],
        "warnings": caught,
    }
    if hasattr(model, "n_iter_"):
        result["n_iter"] = [int(v) for v in np.atleast_1d(model.n_iter_)]
        result["max_iter"] = int(model.max_iter)
    return result


def evaluate(x_eval, y_eval, train_pos, test_pos, label, split_desc):
    return {
        "label": label,
        "split": split_desc,
        "n_eval": int(len(y_eval)),
        "n_train": int(len(train_pos)),
        "n_test": int(len(test_pos)),
        "train_label_counts": counts(y_eval[train_pos]),
        "test_label_counts": counts(y_eval[test_pos]),
        "test_indices": [int(i) for i in test_pos],
        "logistic_regression": score_model(
            LogisticRegression(random_state=1),
            x_eval[train_pos], x_eval[test_pos], y_eval[train_pos], y_eval[test_pos]),
        "decision_tree": score_model(
            DecisionTreeClassifier(random_state=1),
            x_eval[train_pos], x_eval[test_pos], y_eval[train_pos], y_eval[test_pos]),
    }


def pct(value):
    return f"{100 * value:.2f}%"


def render_markdown(payload):
    aligned = payload["aligned_to_bart_split"]
    primary = payload["notebook_protocol_split_recomputed"]
    bart = payload["bart_fidelity_from_json_not_recomputed"]
    lines = [
        "# CartPole baselines aligned to the BART test split (2026-11-12)",
        "",
        "LR/DT baselines recomputed on the EXACT test rows of the group's BART",
        "fidelity run, so BART-vs-baseline comparisons on CartPole are",
        "like-for-like. **BART was not rerun.** BART values below are copied",
        "from the existing `muc3/results/cartpole/bart_fidelity.json`.",
        "",
        "## Why alignment was needed",
        "",
        "- BART (`muc3/scripts/cartpole_bart_fidelity.py`): valid rows are the",
        "  t -> t+1 candidates with more than one non-self local cause. The",
        "  stored run kept all 6,114 candidates, and the split is",
        "  `train_test_split(arange(6114), test_size=0.4, random_state=10,",
        "  stratify=eval_y)` -- stratified.",
        "- Notebook-protocol baselines (`baselines_recomputed.json`, primary):",
        "  same 6,114 pairs, same seed, but the author notebook",
        "  (`eval_fidelity_Cart Pole.ipynb`, cell 15) splits WITHOUT stratify.",
        "  Same sizes (train 3,668 / test 2,446), different test rows:",
        f"  {payload['split_comparison']['shared_test_rows']:,} of 2,446 test rows",
        f"  are shared (Jaccard {payload['split_comparison']['test_jaccard']:.3f}).",
        "- The valid set equals all candidates by count: the stored valid",
        "  count (6,114) equals the candidate count, and valid rows are a",
        "  subset of the candidates. The per-step mask itself was not saved;",
        "  the count argument is the evidence, and the stored",
        "  `bart_fidelity.json` counts are trusted as written by the BART",
        "  script itself.",
        "",
        f"- Replay SHA-256: `{payload['replay_file']['sha256']}`.",
        f"  Recompute environment: scikit-learn {payload['environment_versions']['scikit_learn']}, "
        f"NumPy {payload['environment_versions']['numpy']}.",
        "",
        "## CartPole-v0 -- BART-split baselines (aligned)",
        "",
        "- Task: [state_t, action_t] -> action_{t+1}; 6,114 pairs; train 3,668 /",
        "  test 2,446 (test_size=0.4, random_state=10, stratified -- the BART split).",
        "- Models: `DecisionTreeClassifier(random_state=1)`,",
        "  `LogisticRegression(random_state=1)` (notebook defaults).",
        "- Paper-comparable F1 for binary CartPole is binary F1 (push=1).",
        "",
        "| Model | Accuracy | F1 binary | F1 weighted | F1 macro |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("logistic_regression", "Logistic Regression (aligned)"),
                       ("decision_tree", "Decision Tree (aligned)")):
        m = aligned[key]
        lines.append(f"| {label} | {pct(m['accuracy'])} | {pct(m['f1_binary'])} | "
                     f"{pct(m['f1_weighted'])} | {pct(m['f1_macro'])} |")
    lines += [
        f"| BART (from `bart_fidelity.json`, NOT recomputed) | {pct(bart['accuracy'])} | "
        f"{pct(bart['f1'])} (binary F1) | not available | n/a |",
        "",
        "## For contrast -- notebook-protocol split (unstratified, as published)",
        "",
        "Same task, same seed, unstratified split. These are the primary numbers",
        "in `baselines_recomputed.json`, recomputed here and verified identical.",
        "Use them for the notebook-protocol comparison; use the aligned table",
        "above for the BART comparison. Do not mix the two splits in one table.",
        "",
        "| Model | Accuracy | F1 binary | F1 weighted | F1 macro |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("logistic_regression", "Logistic Regression (notebook split)"),
                       ("decision_tree", "Decision Tree (notebook split)")):
        m = primary[key]
        lines.append(f"| {label} | {pct(m['accuracy'])} | {pct(m['f1_binary'])} | "
                     f"{pct(m['f1_weighted'])} | {pct(m['f1_macro'])} |")
    lines += [
        "",
        "## Verification",
        "",
        f"- Aligned split reproduces the stored stratified variant in "
        f"`baselines_recomputed.json` exactly: "
        f"{payload['verification']['aligned_matches_stored_stratified_variant']}.",
        f"- Notebook split reproduces the stored primary numbers exactly: "
        f"{payload['verification']['notebook_matches_stored_primary']}.",
        "- BART test rows are exactly the aligned test rows by construction:",
        "  the same index array, seed, test size and stratify labels as the",
        "  BART script, over the same ordered candidate rows.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="muc3 directory (default: parent of this script's directory)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: <root>/results)")
    args = parser.parse_args()

    results_dir = args.root / "results"
    cart_dir = results_dir / "cartpole"
    dest = args.out_dir or results_dir
    dest.mkdir(parents=True, exist_ok=True)

    replay_path = cart_dir / "replay.npz"
    replay = np.load(replay_path)
    states = np.asarray(replay["states"], dtype=float)
    actions = np.asarray(replay["actions"]).astype(int)
    metadata = load_json(cart_dir / "collection_metadata.json")
    bart = load_json(cart_dir / "bart_fidelity.json")

    # Features/target exactly as the BART script builds them.
    current = np.concatenate((states, actions.reshape(-1, 1)), axis=1)
    x_all, y_all = current[:-1], actions[1:]
    n_candidates = len(y_all)

    # Count proof: valid rows are a subset of the candidates; the stored
    # valid count equals the candidate count, so valid == all candidates.
    bart_valid = int(bart["valid_local_action_samples"])
    if bart_valid != n_candidates:
        raise RuntimeError(
            f"bart_fidelity.json valid count {bart_valid} != candidate count "
            f"{n_candidates}; the valid mask was not saved and the BART split "
            f"cannot be reconstructed without rerunning BART. Aborting.")
    valid = np.arange(n_candidates)  # ascending order, as the BART script appends
    eval_x, eval_y = current[valid], current[valid + 1, 4].astype(int)
    assert np.array_equal(eval_x, x_all) and np.array_equal(eval_y, y_all)

    indices = np.arange(len(valid))
    # BART split, verbatim construction from cartpole_bart_fidelity.py.
    bart_train, bart_test = train_test_split(
        indices, test_size=0.4, random_state=10, stratify=eval_y)
    # Notebook-protocol split (author notebook cell 15: no stratify).
    nb_train, nb_test = train_test_split(indices, test_size=0.4, random_state=10)
    assert int(bart["test_samples"]) == len(bart_test) == len(nb_test) == 2446

    aligned = evaluate(eval_x, eval_y, bart_train, bart_test,
                       "BART split (stratified)",
                       {"test_size": 0.4, "random_state": 10, "stratify": "labels"})
    primary = evaluate(eval_x, eval_y, nb_train, nb_test,
                       "notebook protocol (unstratified)",
                       {"test_size": 0.4, "random_state": 10})

    shared = len(set(bart_test.tolist()) & set(nb_test.tolist()))
    union = len(set(bart_test.tolist()) | set(nb_test.tolist()))

    # Verify against the stored recomputation, when available.
    verification = {"aligned_matches_stored_stratified_variant": None,
                    "notebook_matches_stored_primary": None}
    stored_path = results_dir / "baselines_recomputed.json"
    if stored_path.exists():
        stored = load_json(stored_path)["environments"]["cartpole"]
        variant = stored["variants"]["stratified_same_split_as_bart_fidelity"]
        prim = stored["primary"]

        def matches(detail, stored_detail):
            return all(
                abs(detail[k]["accuracy"] - stored_detail[k]["accuracy"]) < 1e-12
                and abs(detail[k]["f1_binary"] - stored_detail[k]["f1_binary"]) < 1e-12
                for k in ("logistic_regression", "decision_tree"))
        verification["aligned_matches_stored_stratified_variant"] = matches(aligned, variant)
        verification["notebook_matches_stored_primary"] = matches(primary, prim)
        if not all(verification.values()):
            raise RuntimeError(f"Recomputed numbers disagree with stored "
                               f"baselines_recomputed.json: {verification}")

    payload = {
        "title": "CartPole baselines aligned to the BART fidelity test split",
        "date": "2026-11-12",
        "finding": "The BART run used a stratified split; the published "
                   "notebook-protocol baselines used an unstratified split. "
                   "Same sizes, different test rows. Aligned baselines below "
                   "use the BART split exactly.",
        "task": "features at t (including action at t) -> action at t+1",
        "features": FEATURES,
        "valid_steps": {
            "candidates": int(n_candidates),
            "used": int(len(valid)),
            "evidence": "bart_fidelity.json valid_local_action_samples (6,114) "
                        "equals the candidate count; valid rows are a subset "
                        "of the candidates, hence all of them, in ascending "
                        "index order as built by cartpole_bart_fidelity.py. "
                        "The per-step mask was not saved.",
            "bart_fidelity_valid_samples": bart_valid,
            "bart_fidelity_test_samples": int(bart["test_samples"]),
        },
        "models": {
            "logistic_regression": {"class": "LogisticRegression", "random_state": 1},
            "decision_tree": {"class": "DecisionTreeClassifier", "random_state": 1},
        },
        "aligned_to_bart_split": aligned,
        "notebook_protocol_split_recomputed": primary,
        "split_comparison": {
            "shared_test_rows": int(shared),
            "test_rows_each": int(len(bart_test)),
            "test_jaccard": float(shared / union),
            "bart_test_label_counts": counts(eval_y[bart_test]),
            "notebook_test_label_counts": counts(eval_y[nb_test]),
        },
        "bart_fidelity_from_json_not_recomputed": bart,
        "replay_file": {
            "path": "muc3/results/cartpole/replay.npz",
            "bytes": int(replay_path.stat().st_size),
            "sha256": sha256_of(replay_path),
            "transitions": int(len(actions)),
        },
        "collection": metadata,
        "verification": verification,
        "bart_rerun": False,
        "script": "muc3/scripts/align_cartpole_baselines_to_bart_split.py",
        "computed_at": datetime.now().astimezone().isoformat(),
        "environment_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    (dest / "baselines_aligned.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (dest / "baselines_aligned.md").write_text(
        render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
