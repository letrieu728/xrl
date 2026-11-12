# Recomputed Mục 3 baselines (correction, 2026-11-12)

LR/DT baselines recomputed from the saved `replay.npz` checkpoints under the
authors' fidelity-notebook protocol. **BART was not rerun.** BART values below
are copied from the existing `bart_fidelity.json` files. The old CartPole
baselines measured a different task (see below); the old `metrics.json` files
are kept unchanged as history.

## Protocol (from the author notebooks, yanmluk/causal-xrl @ ae9096e)

- Valid steps: local causal model has **more than one** non-action cause of
  the action node (`count > 1`, CartPole cell 11; LunarLander cell 11).
  Saved BART runs kept all candidates (CartPole 6,114/6,114; LunarLander
  7,381/7,381), so all t -> t+1 pairs are used here. Per-step masks were
  not saved; reproducing them needs a BART/SCM rerun (out of scope).
- Features: full representation row at t **including action_t**; target:
  **action at t+1** (CartPole cells 13/15; LunarLander cells 13/15).
- Split: `test_size=0.4, random_state=10`; LunarLander stratified
  (cell 15), CartPole unstratified (cell 15). PongDuel follows the
  LunarLander variant -- it has no fidelity notebook, only an inference
  example (`cxrl/jupyter/examples/PongDuel.ipynb`).
- Models: `DecisionTreeClassifier(random_state=1)`;
  `LogisticRegression(random_state=1)` (CartPole: notebook defaults;
  LunarLander/PongDuel: `max_iter=1000`).
- Metrics: accuracy is the notebook metric. Paper F1 is **weighted** for
  multiclass (paper section 5.2: "average='weighted' if multiclass");
  CartPole is binary, so its paper-comparable F1 is binary F1
  (positive class push=1), with weighted F1 alongside.

Recompute environment: scikit-learn 1.9.1, NumPy 2.5.4.

## CartPole-v0

- Replay: 6,115 transitions, 6,114 t -> t+1 pairs, train 3,668 / test 2,446 (test_size=0.4, random_state=10, unstratified).
- Replay SHA-256: `da5e73d17dc3ce749273e7c7f75a2b42f9183b63ed6c4f8f8a7c756a5fa89f09`.

| Model | Accuracy | F1 (paper-comparable) | F1 weighted | F1 macro | F1 binary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression (recomputed) | 89.62% | 89.26% | 89.61% | 89.60% | 89.26% |
| Decision Tree (recomputed) | 93.83% | 93.64% | 93.83% | 93.82% | 93.64% |
| BART (from `bart_fidelity.json`, NOT recomputed) | 91.62% | 91.50% (binary F1) | not available | n/a | 91.50% |
| Paper Table 1 (LM / DT / CausalXRL accuracy) | 88.50% / 94.10% / 91.40% | F1 90.00% / 94.60% / 91.60% | | | |

Old (superseded) CartPole baselines from `metrics.json` -- **different task**
(states -> action_t, 20% / seed 99, stratified): LR accuracy 90.76%, F1 90.43%; DT accuracy 98.20%, F1 98.16%. Do not compare them with the notebook-protocol numbers above or with Table 1.

Sensitivity: with the stratified split the group's BART fidelity run used
(same 6,114 pairs, same seed), LR/DT are 90.43% / 95.42% accuracy. The primary numbers above use the notebook's unstratified split,
so their test rows differ from the BART test rows; the sensitivity row shares
the BART test rows exactly.

## LunarLander-v2

- Replay: 7,382 transitions, 7,381 t -> t+1 pairs, train 4,428 / test 2,953 (test_size=0.4, random_state=10, stratified).
- Replay SHA-256: `11ef1ea62efc6824a31994cb787e1a157b1869f5be92769f7c3c780bb628e03d`.

| Model | Accuracy | F1 (paper-comparable) | F1 weighted | F1 macro | F1 binary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression (recomputed) | 77.21% | 76.24% | 76.24% | 59.91% | n/a |
| Decision Tree (recomputed) | 79.14% | 79.06% | 79.06% | 63.25% | n/a |
| BART (from `bart_fidelity.json`, NOT recomputed) | 82.80% | 81.52% (weighted F1, recovered 2026-11-12 -- see note below) | 81.52% | 64.10% | n/a |
| Paper Table 1 (LM / DT / CausalXRL accuracy) | 81.30% / 85.10% / 88.70% | F1 80.20% / 84.70% / 86.60% | | | |

Old `metrics.json` baselines (same t -> t+1 task, macro-F1 only): LR accuracy 77.21%, macro-F1 59.91%; DT accuracy 78.80%, macro-F1 62.75%. LR reproduces exactly. The recomputed DT differs slightly (see JSON note: the stored DT accuracy matches `random_state=99`, not the `random_state=1` in the committed script and notebook; the stored macro-F1 matches neither). The recomputed DT uses the committed/notebook `random_state=1` and reproduces identically under scikit-learn 1.5.2 and 1.9.1.

CORRECTION 2026-11-12 (supersedes the original statement below): BART weighted-F1 **was** recovered from the saved artifacts after all. The per-action BART score checkpoints (`fidelity_scores/action_{8..11}.npz`) were saved, and re-aggregating them reproduces accuracy 82.80% and macro-F1 64.10% with difference 0.0, giving **weighted F1 81.52%** (per-class F1 in `lunarlander/bart_fidelity_weighted.json`; script `scripts/recompute_lunarlander_bart_weighted_f1.py`). The BART-vs-paper F1 comparison is therefore like-for-like after all (82.80% / 81.52% vs the paper's 88.70% / 86.60%). Original (superseded) statement, kept for history: "BART weighted-F1 cannot be computed from the saved artifacts: per-sample BART predictions were not saved, only accuracy + macro-F1."

## PongDuel-v0

- Replay: 74,445 transitions, 74,444 t -> t+1 pairs, train 44,666 / test 29,778 (test_size=0.4, random_state=10, stratified).
- Replay SHA-256: `4244ab7a81a7a66c981eeee998118be6dc936ee6aa2b92eb475b498e39f81f1d`.

| Model | Accuracy | F1 (paper-comparable) | F1 weighted | F1 macro | F1 binary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression (recomputed) | 81.79% | 81.38% | 81.38% | 62.65% | n/a |
| Decision Tree (recomputed) | 96.21% | 96.21% | 96.21% | 90.21% | n/a |
| BART (from `bart_fidelity.json`, NOT recomputed) | 81.73% | n/a -- macro-F1 only, not paper-comparable | not available | 19.51% | n/a |
| Paper Table 1 (LM / DT / CausalXRL accuracy) | 81.50% / 88.30% / 90.20% | F1 81.10% / 88.40% / 90.70% | | | |

Old `metrics.json` baselines (same t -> t+1 task, macro-F1 only): LR accuracy 81.77%, macro-F1 62.62%; DT accuracy 96.21%, macro-F1 90.21%. DT reproduces exactly; LR differs by <0.03 pp (solver/convergence).

**PongDuel BART stays a PROXY** (rounded numeric regression, 9 joint classes,
14,995 local-candidate rows, test 5,998 -- a different subset from the 74,444
baseline rows). It is not a paper-equivalent categorical BART fidelity result,
and no weighted-F1 is computed for it.

## Old-protocol reproduction check (CartPole)

Rerunning the superseded CartPole task on the same replay reproduces the stored
`metrics.json` baselines exactly (see JSON), confirming the replay data is
unchanged and only the task definition differed.

## Comparability summary

- CartPole baselines: comparable to the notebook protocol and, for binary F1,
  to Table 1's binary case. BART comparison carries the split caveat above.
- LunarLander baselines: weighted-F1 now comparable to Table 1. BART
  weighted-F1 (81.52%) was recovered 2026-11-12 from the saved score
  checkpoints, so BART F1 is comparable to Table 1's weighted F1 as well.
- PongDuel baselines: weighted-F1 comparable to Table 1 as an adaptation
  (no author fidelity notebook). BART proxy: not comparable, by design.
