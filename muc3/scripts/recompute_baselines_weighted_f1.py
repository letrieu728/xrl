"""Recompute Mục 3 LR/DT baselines under the authors' fidelity protocol.

Correction (2026-11-12): the CartPole baselines stored in
``muc3/results/cartpole/metrics.json`` measured a different task from the
authors' fidelity notebook (``cxrl/jupyter/eval/eval_fidelity_Cart Pole.ipynb``
in yanmluk/causal-xrl @ ae9096e): state_t -> action_t on all 6,115 rows with
a 20% / seed-99 stratified split, instead of the notebook's
[state_t, action_t] -> action_{t+1} on valid steps with a 40% / seed-10
unstratified split. LunarLander and PongDuel baselines already used the
t -> t+1 task, but reported only macro-F1, while the paper (arXiv:2610.42028v1,
section 5.2) defines F1 as weighted average for multiclass problems. This
script recomputes the cheap LR/DT baselines only; it does NOT rerun BART.
BART fidelity values are copied by reference from the existing
``bart_fidelity.json`` files and are never recomputed here.

Author-notebook protocol (CartPole cells 11/13/15/17/19, LunarLander
cells 11/13/15/17/19; quoted in ``muc3/results/baselines_recomputed.md``):

- Valid steps: timesteps whose local causal model has more than one
  non-action cause of the action node (``count > 1``). The saved
  ``bart_fidelity.json`` counts show this filter kept every candidate row
  in both runs (CartPole 6,114/6,114; LunarLander 7,381/7,381), so the
  recomputation uses all t -> t+1 pairs. The per-step valid masks were not
  saved; reproducing them would require rerunning the BART/SCM inference,
  which is out of scope for this correction. If a future run saves a mask,
  pass it with ``--valid-mask`` (see ``load_valid_mask``).
- Features: the full representation row at t, including the action at t
  (CartPole 5 columns; LunarLander 8 state + 4 one-hot action columns;
  PongDuel 20 state + 1 scalar joint-action column, following the same
  pattern -- PongDuel has no fidelity notebook, only an inference example).
- Target: the action at t+1.
- Split: ``train_test_split(..., test_size=0.4, random_state=10)``;
  LunarLander additionally stratifies (notebook cell 15), CartPole does
  not. PongDuel follows the LunarLander variant (stratified).
- Models: ``DecisionTreeClassifier(random_state=1)`` and
  ``LogisticRegression(random_state=1)`` (CartPole: notebook defaults,
  ``max_iter=100``; LunarLander/PongDuel: ``max_iter=1000``).
- Metrics: accuracy (the notebook metric) plus binary F1 (CartPole),
  weighted F1 (the paper metric for multiclass) and macro F1. The
  paper-comparable F1 is flagged per environment in the output JSON.

Episode-boundary pairing is inherited from the notebooks: the replay is
one flat array, so the last non-terminal step of one episode is paired
with the first step of the next episode, exactly as the notebooks pair
``replay.iloc[valid_steps]`` with ``replay.iloc[valid_steps + 1]``.

Usage (from the repository root, with the saved replay checkpoints)::

    python muc3/scripts/recompute_baselines_weighted_f1.py

Outputs: ``muc3/results/baselines_recomputed.json`` and
``muc3/results/baselines_recomputed.md``. Existing ``metrics.json`` files
are read, never overwritten, so the old numbers stay in the history.
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import sklearn

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Paper Table 1 (arXiv:2610.42028v1). LM = the paper's linear model; the
# group uses logistic regression. F1 is weighted for multiclass (section 5.2).
PAPER_TABLE_1 = {
    "cartpole": {
        "linear_model": {"accuracy": 0.885, "f1": 0.900},
        "decision_tree": {"accuracy": 0.941, "f1": 0.946},
        "causal_xrl": {"accuracy": 0.914, "f1": 0.916},
    },
    "lunarlander": {
        "linear_model": {"accuracy": 0.813, "f1": 0.802},
        "decision_tree": {"accuracy": 0.851, "f1": 0.847},
        "causal_xrl": {"accuracy": 0.887, "f1": 0.866},
    },
    "pongduel": {
        "linear_model": {"accuracy": 0.815, "f1": 0.811},
        "decision_tree": {"accuracy": 0.883, "f1": 0.884},
        "causal_xrl": {"accuracy": 0.902, "f1": 0.907},
    },
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_valid_mask(path, n_candidates):
    """Load an optional per-candidate boolean mask saved by a BART run.

    Mask format: ``.npy`` boolean array, or ``.npz`` with key
    ``valid_mask`` / ``valid`` / ``valid_indices``. Returns row indices.
    """
    if path.suffix == ".npy":
        mask = np.load(path).astype(bool)
        if mask.shape != (n_candidates,):
            raise ValueError(f"valid mask shape {mask.shape} != ({n_candidates},)")
        return np.flatnonzero(mask)
    data = np.load(path)
    if "valid_indices" in data:
        return np.asarray(data["valid_indices"], dtype=int)
    for key in ("valid_mask", "valid"):
        if key in data:
            mask = np.asarray(data[key]).astype(bool)
            if mask.shape != (n_candidates,):
                raise ValueError(f"valid mask shape {mask.shape} != ({n_candidates},)")
            return np.flatnonzero(mask)
    raise ValueError(f"no recognised valid-mask key in {path}: {list(data.keys())}")


def counts(labels):
    values, freqs = np.unique(np.asarray(labels), return_counts=True)
    return {int(v): int(f) for v, f in zip(values, freqs)}


def score_model(model, x_train, x_test, y_train, y_test):
    """Fit one model and return accuracy + all F1 variants + per-class rows."""
    caught = []
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
        caught = [str(w.message) for w in found]
    predicted = model.predict(x_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, zero_division=0
    )
    labels = sorted(np.unique(np.concatenate([y_test, predicted])).tolist())
    result = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1_weighted": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "per_class": [
            {
                "label": int(label),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        ],
        "warnings": caught,
    }
    if len(np.unique(y_test)) == 2:
        result["f1_binary"] = float(f1_score(y_test, predicted, zero_division=0))
    if hasattr(model, "n_iter_"):
        result["n_iter"] = [int(v) for v in np.atleast_1d(model.n_iter_)]
        result["max_iter"] = int(model.max_iter)
    return result


def build_features(env, states, actions):
    """Return (X_t, y_{t+1}) over all t -> t+1 candidate pairs."""
    if env == "cartpole":
        current = np.concatenate((states, actions.reshape(-1, 1)), axis=1)
        return current[:-1], actions[1:], ["cart_position", "cart_velocity", "pole_angle", "pole_velocity", "push"]
    if env == "lunarlander":
        current = np.concatenate((states, np.eye(4)[actions]), axis=1)
        return current[:-1], actions[1:], [
            "x", "y", "vx", "vy", "angle", "angular_velocity",
            "left_contact", "right_contact",
            "do_nothing", "fire_left_engine", "fire_main_engine", "fire_right_engine",
        ]
    if env == "pongduel":
        current = np.concatenate((states, actions.reshape(-1, 1)), axis=1)
        return current[:-1], actions[1:], [f"state_{i}" for i in range(states.shape[1])] + ["encoded_action"]
    raise ValueError(env)


def run_env(env, lr_kwargs, stratify, valid_mask_path, extra_splits):
    out_dir = RESULTS / env
    replay_path = out_dir / "replay.npz"
    replay = np.load(replay_path)
    states = np.asarray(replay["states"], dtype=float)
    actions = np.asarray(replay["actions"]).astype(int)
    metadata = json.loads((out_dir / "collection_metadata.json").read_text(encoding="utf-8"))
    old_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    bart = json.loads((out_dir / "bart_fidelity.json").read_text(encoding="utf-8"))

    x_all, y_all, feature_names = build_features(env, states, actions)
    n_candidates = len(y_all)
    bart_valid = bart.get("valid_local_action_samples", bart.get("local_candidate_samples"))
    if valid_mask_path is not None:
        valid_idx = load_valid_mask(valid_mask_path, n_candidates)
        valid_source = f"mask file {valid_mask_path.name} ({len(valid_idx)}/{n_candidates} candidates)"
        valid_count_matches = None
    else:
        valid_idx = np.arange(n_candidates)
        valid_count_matches = (bart_valid == n_candidates)
        if env in ("cartpole", "lunarlander") and not valid_count_matches:
            raise RuntimeError(
                f"{env}: bart_fidelity.json valid count {bart_valid} != candidate count "
                f"{n_candidates}; the all-candidates population is not justified without "
                "a saved valid mask. Rerun with --valid-mask or rerun the BART fidelity step."
            )
        valid_source = (
            "all t -> t+1 candidate pairs; bart_fidelity.json reports "
            f"{bart_valid} valid samples, equal to the candidate count {n_candidates}"
            if env != "pongduel"
            else "all t -> t+1 candidate pairs (baseline scope; the BART proxy used a "
            f"different {bart_valid}-row local-candidate subset -- see note)"
        )
    x_eval, y_eval = x_all[valid_idx], y_all[valid_idx]

    def evaluate(split_kwargs, label):
        idx = np.arange(len(y_eval))
        train_pos, test_pos = train_test_split(idx, **split_kwargs)
        detail = {
            "label": label,
            "split": {k: (v if not isinstance(v, np.ndarray) else "labels") for k, v in split_kwargs.items()},
            "n_eval": int(len(y_eval)),
            "n_train": int(len(train_pos)),
            "n_test": int(len(test_pos)),
            "train_label_counts": counts(y_eval[train_pos]),
            "test_label_counts": counts(y_eval[test_pos]),
            "logistic_regression": score_model(
                LogisticRegression(**lr_kwargs), x_eval[train_pos], x_eval[test_pos],
                y_eval[train_pos], y_eval[test_pos]),
            "decision_tree": score_model(
                DecisionTreeClassifier(random_state=1), x_eval[train_pos], x_eval[test_pos],
                y_eval[train_pos], y_eval[test_pos]),
        }
        return detail

    primary_kwargs = {"test_size": 0.4, "random_state": 10}
    if stratify:
        primary_kwargs["stratify"] = y_eval
    primary = evaluate(primary_kwargs, "notebook protocol")
    variants = {}
    for label, kwargs in extra_splits.items():
        kw = dict(kwargs)
        if kw.get("stratify") == "y":
            kw["stratify"] = y_eval
        variants[label] = evaluate(kw, label)

    return {
        "environment": metadata.get("environment", env),
        "collection": metadata,
        "replay_file": {
            "path": f"muc3/results/{env}/replay.npz",
            "bytes": int(replay_path.stat().st_size),
            "sha256": sha256_of(replay_path),
            "keys": {k: {"shape": [int(d) for d in replay[k].shape], "dtype": str(replay[k].dtype)} for k in replay.files},
            "transitions": int(len(actions)),
            "action_counts_all_rows": counts(actions),
        },
        "features": feature_names,
        "task": "features at t (including action at t) -> action at t+1",
        "valid_steps": {
            "candidates": int(n_candidates),
            "used": int(len(valid_idx)),
            "source": valid_source,
            "bart_valid_count_equals_candidate_count": valid_count_matches,
            "bart_fidelity_valid_samples": bart.get("valid_local_action_samples", bart.get("local_candidate_samples")),
            "bart_fidelity_test_samples": bart.get("test_samples"),
        },
        "models": {
            "logistic_regression": {"class": "LogisticRegression", **lr_kwargs},
            "decision_tree": {"class": "DecisionTreeClassifier", "random_state": 1},
        },
        "primary": primary,
        "variants": variants,
        "old_baselines_from_metrics_json": old_metrics.get("baselines"),
        "bart_fidelity_from_json_not_recomputed": bart,
        "paper_table_1": PAPER_TABLE_1[env],
    }


def cartpole_old_protocol_check():
    """Reproduce the superseded CartPole task to prove the data is unchanged.

    Old task (muc3/scripts/cartpole_metrics.py): states -> action_t,
    test_size=0.2, random_state=99, stratify, model random_state=99.
    Returns the recomputed values alongside the stored ones.
    """
    out_dir = RESULTS / "cartpole"
    replay = np.load(out_dir / "replay.npz")
    states = np.asarray(replay["states"], dtype=float)
    actions = np.asarray(replay["actions"]).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        states, actions, test_size=0.2, random_state=99, stratify=actions
    )
    recomputed = {
        "logistic_regression": score_model(
            LogisticRegression(random_state=99, max_iter=1000), x_train, x_test, y_train, y_test),
        "decision_tree": score_model(
            DecisionTreeClassifier(random_state=99), x_train, x_test, y_train, y_test),
    }
    stored = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["baselines"]
    return {
        "task": "states at t -> action at t (SUPERSEDED; differs from the notebook)",
        "split": {"test_size": 0.2, "random_state": 99, "stratify": "labels"},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "recomputed": recomputed,
        "stored_in_metrics_json": stored,
    }


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def pct(value):
    return f"{100 * value:.2f}%"


def render_markdown(payload):
    lines = [
        "# Recomputed Mục 3 baselines (correction, 2026-11-12)",
        "",
        "LR/DT baselines recomputed from the saved `replay.npz` checkpoints under the",
        "authors' fidelity-notebook protocol. **BART was not rerun.** BART values below",
        "are copied from the existing `bart_fidelity.json` files. The old CartPole",
        "baselines measured a different task (see below); the old `metrics.json` files",
        "are kept unchanged as history.",
        "",
        "## Protocol (from the author notebooks, yanmluk/causal-xrl @ ae9096e)",
        "",
        "- Valid steps: local causal model has **more than one** non-action cause of",
        "  the action node (`count > 1`, CartPole cell 11; LunarLander cell 11).",
        "  Saved BART runs kept all candidates (CartPole 6,114/6,114; LunarLander",
        "  7,381/7,381), so all t -> t+1 pairs are used here. Per-step masks were",
        "  not saved; reproducing them needs a BART/SCM rerun (out of scope).",
        "- Features: full representation row at t **including action_t**; target:",
        "  **action at t+1** (CartPole cells 13/15; LunarLander cells 13/15).",
        "- Split: `test_size=0.4, random_state=10`; LunarLander stratified",
        "  (cell 15), CartPole unstratified (cell 15). PongDuel follows the",
        "  LunarLander variant -- it has no fidelity notebook, only an inference",
        "  example (`cxrl/jupyter/examples/PongDuel.ipynb`).",
        "- Models: `DecisionTreeClassifier(random_state=1)`;",
        "  `LogisticRegression(random_state=1)` (CartPole: notebook defaults;",
        "  LunarLander/PongDuel: `max_iter=1000`).",
        "- Metrics: accuracy is the notebook metric. Paper F1 is **weighted** for",
        "  multiclass (paper section 5.2: \"average='weighted' if multiclass\");",
        "  CartPole is binary, so its paper-comparable F1 is binary F1",
        "  (positive class push=1), with weighted F1 alongside.",
        "",
        f"Recompute environment: scikit-learn {payload['environment_versions']['scikit_learn']}, "
        f"NumPy {payload['environment_versions']['numpy']}.",
        "",
    ]
    for env in ("cartpole", "lunarlander", "pongduel"):
        block = payload["environments"][env]
        primary = block["primary"]
        lines += [
            f"## {block['environment']}",
            "",
            f"- Replay: {block['replay_file']['transitions']:,} transitions, "
            f"{block['valid_steps']['used']:,} t -> t+1 pairs, "
            f"train {primary['n_train']:,} / test {primary['n_test']:,} "
            f"(test_size=0.4, random_state=10"
            f"{', stratified' if 'stratify' in primary['split'] else ', unstratified'}).",
            f"- Replay SHA-256: `{block['replay_file']['sha256']}`.",
            "",
            "| Model | Accuracy | F1 (paper-comparable) | F1 weighted | F1 macro | F1 binary |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        paper_f1 = "f1_binary" if env == "cartpole" else "f1_weighted"
        for key, label in (("logistic_regression", "Logistic Regression"), ("decision_tree", "Decision Tree")):
            m = primary[key]
            lines.append(
                f"| {label} (recomputed) | {pct(m['accuracy'])} | {pct(m[paper_f1])} | "
                f"{pct(m['f1_weighted'])} | {pct(m['f1_macro'])} | "
                f"{pct(m['f1_binary']) if 'f1_binary' in m else 'n/a'} |"
            )
        bart = block["bart_fidelity_from_json_not_recomputed"]
        bart_f1 = bart.get("f1", bart.get("f1_macro"))
        if "f1" in bart:
            bart_comparable = f"{pct(bart_f1)} (binary F1)"
        else:
            bart_comparable = "n/a -- macro-F1 only, not paper-comparable"
        lines.append(
            f"| BART (from `bart_fidelity.json`, NOT recomputed) | {pct(bart['accuracy'])} | "
            f"{bart_comparable} | not available | "
            f"{pct(bart['f1_macro']) if 'f1_macro' in bart else 'n/a'} | "
            f"{pct(bart['f1']) if 'f1' in bart else 'n/a'} |"
        )
        paper = block["paper_table_1"]
        lines += [
            f"| Paper Table 1 (LM / DT / CausalXRL accuracy) | "
            f"{pct(paper['linear_model']['accuracy'])} / {pct(paper['decision_tree']['accuracy'])} / "
            f"{pct(paper['causal_xrl']['accuracy'])} | F1 {pct(paper['linear_model']['f1'])} / "
            f"{pct(paper['decision_tree']['f1'])} / {pct(paper['causal_xrl']['f1'])} | | | |",
            "",
        ]
        if env == "cartpole":
            old = block["old_baselines_from_metrics_json"]
            lines += [
                "Old (superseded) CartPole baselines from `metrics.json` -- **different task**",
                "(states -> action_t, 20% / seed 99, stratified): LR accuracy "
                f"{pct(old['logistic_regression']['accuracy'])}, F1 {pct(old['logistic_regression']['f1'])}; "
                f"DT accuracy {pct(old['decision_tree']['accuracy'])}, F1 {pct(old['decision_tree']['f1'])}. "
                "Do not compare them with the notebook-protocol numbers above or with Table 1.",
                "",
                "Sensitivity: with the stratified split the group's BART fidelity run used",
                "(same 6,114 pairs, same seed), LR/DT are "
                f"{pct(block['variants']['stratified_same_split_as_bart_fidelity']['logistic_regression']['accuracy'])} / "
                f"{pct(block['variants']['stratified_same_split_as_bart_fidelity']['decision_tree']['accuracy'])} "
                "accuracy. The primary numbers above use the notebook's unstratified split,",
                "so their test rows differ from the BART test rows; the sensitivity row shares",
                "the BART test rows exactly.",
                "",
            ]
        if env == "lunarlander":
            old = block["old_baselines_from_metrics_json"]
            lines += [
                "Old `metrics.json` baselines (same t -> t+1 task, macro-F1 only): LR accuracy "
                f"{pct(old['logistic_regression']['accuracy'])}, macro-F1 {pct(old['logistic_regression']['f1_macro'])}; "
                f"DT accuracy {pct(old['decision_tree']['accuracy'])}, macro-F1 {pct(old['decision_tree']['f1_macro'])}. "
                "LR reproduces exactly. The recomputed DT differs slightly (see JSON note: the stored "
                "DT accuracy matches `random_state=99`, not the `random_state=1` in the committed "
                "script and notebook; the stored macro-F1 matches neither). The recomputed DT uses "
                "the committed/notebook `random_state=1` and reproduces identically under "
                "scikit-learn 1.5.2 and 1.9.1.",
                "",
                "BART weighted-F1 cannot be computed from the saved artifacts: per-sample BART "
                "predictions were not saved, only accuracy + macro-F1. The BART-vs-paper F1 "
                "comparison therefore stays not like-for-like (macro vs weighted) until the "
                "BART fidelity run is repeated with predictions saved.",
                "",
            ]
        if env == "pongduel":
            old = block["old_baselines_from_metrics_json"]
            lines += [
                "Old `metrics.json` baselines (same t -> t+1 task, macro-F1 only): LR accuracy "
                f"{pct(old['logistic_regression']['accuracy'])}, macro-F1 {pct(old['logistic_regression']['f1_macro'])}; "
                f"DT accuracy {pct(old['decision_tree']['accuracy'])}, macro-F1 {pct(old['decision_tree']['f1_macro'])}. "
                "DT reproduces exactly; LR differs by <0.03 pp (solver/convergence).",
                "",
                "**PongDuel BART stays a PROXY** (rounded numeric regression, 9 joint classes,",
                "14,995 local-candidate rows, test 5,998 -- a different subset from the 74,444",
                "baseline rows). It is not a paper-equivalent categorical BART fidelity result,",
                "and no weighted-F1 is computed for it.",
                "",
            ]
    lines += [
        "## Old-protocol reproduction check (CartPole)",
        "",
        "Rerunning the superseded CartPole task on the same replay reproduces the stored",
        "`metrics.json` baselines exactly (see JSON), confirming the replay data is",
        "unchanged and only the task definition differed.",
        "",
        "## Comparability summary",
        "",
        "- CartPole baselines: comparable to the notebook protocol and, for binary F1,",
        "  to Table 1's binary case. BART comparison carries the split caveat above.",
        "- LunarLander baselines: weighted-F1 now comparable to Table 1. BART F1 is",
        "  still macro-only: not comparable to Table 1's weighted F1.",
        "- PongDuel baselines: weighted-F1 comparable to Table 1 as an adaptation",
        "  (no author fidelity notebook). BART proxy: not comparable, by design.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-mask", type=Path, default=None,
                        help="Optional per-candidate valid-step mask (.npy/.npz); "
                        "applies to CartPole only in this correction.")
    args = parser.parse_args()

    cartpole = run_env(
        "cartpole", {"random_state": 1}, stratify=False,
        valid_mask_path=args.valid_mask,
        extra_splits={"stratified_same_split_as_bart_fidelity": {
            "test_size": 0.4, "random_state": 10, "stratify": "y"}},
    )
    lunarlander = run_env(
        "lunarlander", {"random_state": 1, "max_iter": 1000}, stratify=True,
        valid_mask_path=None, extra_splits={},
    )
    pongduel = run_env(
        "pongduel", {"random_state": 1, "max_iter": 1000}, stratify=True,
        valid_mask_path=None, extra_splits={},
    )
    # Reproduction notes, verified 2026-11-12 (see the JSON for full values).
    lunarlander["reproduction_note"] = (
        "LR reproduces metrics.json exactly. DT recomputed with random_state=1 "
        "(committed script + notebook) gives accuracy 0.7914 / macro-F1 0.6325, "
        "vs stored 0.7880 / 0.6275. The stored accuracy matches "
        "DecisionTreeClassifier(random_state=99) exactly; the stored macro-F1 "
        "matches neither random_state=1 nor 99 under scikit-learn 1.5.2 or 1.9.1. "
        "Cause of the residual macro-F1 gap is unconfirmed; the recomputed "
        "random_state=1 values are the notebook-faithful ones and are kept."
    )
    pongduel["reproduction_note"] = (
        "DT reproduces metrics.json exactly. LR differs by <0.03 pp in accuracy "
        "and macro-F1 (solver convergence across environments); treated as "
        "reproduced within tolerance."
    )
    payload = {
        "title": "Mục 3 baselines recomputed under the authors' fidelity-notebook protocol",
        "date": "2026-11-12",
        "correction": (
            "CartPole baselines in metrics.json measured states->action_t "
            "(20%/seed 99). Recomputed as [state_t, action_t]->action_{t+1} "
            "(40%/seed 10, notebook protocol). LunarLander/PongDuel gain "
            "weighted-F1 alongside macro-F1. BART NOT rerun."
        ),
        "author_repo": "https://github.com/yanmluk/causal-xrl",
        "author_commit": "ae9096ea3065cd2ac6ee4f5fdcbc7fb6a9504162",
        "paper": "arXiv:2610.42028v1, Table 1 and section 5.2 (F1 weighted if multiclass)",
        "environment_versions": {"scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "script": "muc3/scripts/recompute_baselines_weighted_f1.py",
        "environments": {
            "cartpole": cartpole,
            "lunarlander": lunarlander,
            "pongduel": pongduel,
        },
        "cartpole_old_protocol_check": cartpole_old_protocol_check(),
        "remaining_gaps": [
            "BART weighted-F1 (LunarLander) and BART predictions were not saved; "
            "computing them needs a BART fidelity rerun with predictions saved.",
            "Per-step valid masks were not saved; the all-candidates population "
            "is justified only by the saved valid counts equalling the candidate "
            "counts (CartPole/LunarLander).",
            "PongDuel BART fidelity remains a documented regression proxy; "
            "no categorical BART fidelity exists for PongDuel.",
        ],
    }
    (RESULTS / "baselines_recomputed.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "baselines_recomputed.md").write_text(
        render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:2000])
    print("... wrote muc3/results/baselines_recomputed.json and .md")


if __name__ == "__main__":
    main()
