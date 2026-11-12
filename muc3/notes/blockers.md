# Blockers

## CartPole BART pipeline — 2026-11-11

**Status:** blocked after three targeted fixes; per the handoff rule, do not make a fourth fix without a new hypothesis or owner direction.

The CartPole heuristic rollout completed at 200 episodes (6,115–6,139 transitions), and a standalone `BARTRegressor.fit()` through Python → rpy2 → R → dbarts succeeded.  The full `XRL(...)->infer_DBN()` process then exited during structure inference without a Python traceback or persisted output, before fidelity and PC/FCI timing could run.

Attempts:

1. Initialized R RNG with `set.seed(99)` in `BARTRegressor`; this fixed `object '.Random.seed' not found`.
2. Changed dbarts from four chains/four threads to one chain/one thread to avoid a native rpy2/dbarts concurrency failure on Windows.
3. Used the repository's `method="expectation"` path and ran Python with `-X faulthandler`; the process still ended after the rollout without a graph or native trace.

**Next decision required:** permit a new, independently justified diagnosis (for example, running each DBN target in a separate Python/R process with a checkpoint), or explicitly approve the documented DecisionTree fallback. No group metric has been produced and no paper number has been copied.
