# Ablation blockers — CausalXRL extension A (CartPole, surrogate estimator)

SURROGATE ESTIMATOR (RandomForest, 200 trees), NOT BART/dbarts — stability
check only, to be re-run with real BART when available.

## Blockers: none

All 6 runs (nepisodes 50/100/200 × method hte/ate) completed; CI
sensitivity and graph comparison are in the outputs. No run failed
and no part of the task was left undone.

Notes (not blockers):
- R / rpy2 / dbarts were NOT installed, per the task (another agent
  is trying that route in parallel). The repo's `BARTRegressor` was
  replaced by the RandomForest surrogate via monkeypatching, with
  `rpy2` stubbed in `sys.modules` for import only; the repo clone at
  `/tmp/causal-xrl-muc4` was left unmodified (`git status` clean).
- The surrogate's numbers are NOT BART/paper numbers and must be
  re-run with real BART when available. Known surrogate limits are
  documented in `scripts/surrogate_bart.py`: per-tree draws instead
  of a BART posterior, +1 SD vs −1 SD numeric contrast (repo: +1 vs
  +0), no extrapolation beyond the training range, and hte is
  conservative (it under-detects vs the true CartPole physics).
- CPU contention from the parallel DQN-training agent slowed fits
  (load avg ~6 at times); thread caps in `run_ablation.py` kept the
  grid to ~5 minutes total. See `notes/ablation-errors.md`.
