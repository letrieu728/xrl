# CartPole checkpoint

This directory contains an evaluation-only run with the repository's hand-written linear CartPole policy, seed 99, 200 episodes and 6,115 non-terminal transitions. No RL agent was trained.

- BART local-action fidelity: accuracy 91.62%, F1 91.50% (`bart_fidelity.json`).
- Logistic Regression: accuracy 90.76%, F1 90.43%; Decision Tree: accuracy 98.20%, F1 98.16% (`metrics.json`).
- BART structure learning used five fresh Python/R/dbarts processes and took 242.19 seconds in total. PC and FCI timings are full Fisher-Z wall-clock runs (0.076 s and 0.061 s); they are not directly comparable to the paper's simulated CI-test timer.

The individual target result files are checkpoints for recovery and audit.
