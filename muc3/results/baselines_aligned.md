# CartPole baselines aligned to the BART test split (2026-11-12)

LR/DT baselines recomputed on the EXACT test rows of the group's BART
fidelity run, so BART-vs-baseline comparisons on CartPole are
like-for-like. **BART was not rerun.** BART values below are copied
from the existing `muc3/results/cartpole/bart_fidelity.json`.

## Why alignment was needed

- BART (`muc3/scripts/cartpole_bart_fidelity.py`): valid rows are the
  t -> t+1 candidates with more than one non-self local cause. The
  stored run kept all 6,114 candidates, and the split is
  `train_test_split(arange(6114), test_size=0.4, random_state=10,
  stratify=eval_y)` -- stratified.
- Notebook-protocol baselines (`baselines_recomputed.json`, primary):
  same 6,114 pairs, same seed, but the author notebook
  (`eval_fidelity_Cart Pole.ipynb`, cell 15) splits WITHOUT stratify.
  Same sizes (train 3,668 / test 2,446), different test rows:
  936 of 2,446 test rows
  are shared (Jaccard 0.237).
- The valid set equals all candidates by count: the stored valid
  count (6,114) equals the candidate count, and valid rows are a
  subset of the candidates. The per-step mask itself was not saved;
  the count argument is the evidence, and the stored
  `bart_fidelity.json` counts are trusted as written by the BART
  script itself.

- Replay SHA-256: `da5e73d17dc3ce749273e7c7f75a2b42f9183b63ed6c4f8f8a7c756a5fa89f09`.
  Recompute environment: scikit-learn 1.9.1, NumPy 2.5.4.

## CartPole-v0 -- BART-split baselines (aligned)

- Task: [state_t, action_t] -> action_{t+1}; 6,114 pairs; train 3,668 /
  test 2,446 (test_size=0.4, random_state=10, stratified -- the BART split).
- Models: `DecisionTreeClassifier(random_state=1)`,
  `LogisticRegression(random_state=1)` (notebook defaults).
- Paper-comparable F1 for binary CartPole is binary F1 (push=1).

| Model | Accuracy | F1 binary | F1 weighted | F1 macro |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression (aligned) | 90.43% | 90.22% | 90.43% | 90.43% |
| Decision Tree (aligned) | 95.42% | 95.32% | 95.42% | 95.42% |
| BART (from `bart_fidelity.json`, NOT recomputed) | 91.62% | 91.50% (binary F1) | not available | n/a |

## For contrast -- notebook-protocol split (unstratified, as published)

Same task, same seed, unstratified split. These are the primary numbers
in `baselines_recomputed.json`, recomputed here and verified identical.
Use them for the notebook-protocol comparison; use the aligned table
above for the BART comparison. Do not mix the two splits in one table.

| Model | Accuracy | F1 binary | F1 weighted | F1 macro |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression (notebook split) | 89.62% | 89.26% | 89.61% | 89.60% |
| Decision Tree (notebook split) | 93.83% | 93.64% | 93.83% | 93.82% |

## Verification

- Aligned split reproduces the stored stratified variant in `baselines_recomputed.json` exactly: True.
- Notebook split reproduces the stored primary numbers exactly: True.
- BART test rows are exactly the aligned test rows by construction:
  the same index array, seed, test size and stratify labels as the
  BART script, over the same ordered candidate rows.
