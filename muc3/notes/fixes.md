# Fixes

- `cxrl/lib/models/bart_regressor.py`: initialize R's random-number state with `set.seed(99)` before creating dbarts functions. This fixes dbarts' `.Random.seed` initialization error.
- `cxrl/lib/models/bart_regressor.py`: retain dbarts' required four chains (4 × 125 = 500 posterior draws) and limit only the number of threads to one. This preserves the repository HTE implementation's posterior-draw invariant while avoiding multithreaded Windows execution.
- `cxrl/build_box2d.cmd`: added a repeatable Windows C++ environment setup for building `box2d-py`; Box2D imports successfully.
- Approved independent diagnostic: run each CartPole DBN target in a fresh Python/R process against one fixed replay checkpoint. This tests whether the prior silent termination is caused by accumulated `rpy2`/`dbarts` state rather than the rollout or a single target.
