# CartPole checkpoint

This directory contains an evaluation-only run with the repository's hand-written linear CartPole policy, seed 99, 200 episodes and 6,115 non-terminal transitions. No RL agent was trained.

- BART local-action fidelity: accuracy 91.62%, F1 91.50% (`bart_fidelity.json`).
- Logistic Regression: accuracy 89.62%, binary F1 89.26%; Decision Tree: accuracy 93.83%, binary F1 93.64% (notebook protocol, `../baselines_recomputed.md`). The baselines in `metrics.json` (90.76%/98.20%) measured a different task (states -> action_t) and are superseded; the file is kept as history.
- BART structure learning used five fresh Python/R/dbarts processes and took 242.19 seconds in total. PC and FCI timings are full Fisher-Z wall-clock runs (0.076 s and 0.061 s); they are not directly comparable to the paper's simulated CI-test timer.

The individual target result files are checkpoints for recovery and audit.
