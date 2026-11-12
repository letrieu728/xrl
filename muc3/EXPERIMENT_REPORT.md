# CausalXRL evaluation-only reproduction

## Scope and conditions

- No RL agent was trained. CartPole uses the repository's hand-written linear policy; LunarLander uses its hand-written PD controller; PongDuel uses the repository's strong/weak heuristic joint policy.
- Seed: 99. CartPole: 200 episodes / 6,115 non-terminal transitions. LunarLander: 30 episodes / 7,382 transitions. PongDuel: 200 episodes / 74,445 transitions (14,995 local candidate samples used for fidelity).
- BART runs use `dbarts` through `rpy2`, four posterior chains and one thread. DBN targets were evaluated in isolated Python/R processes so each completed target is independently checkpointed.
- MicroRTS was intentionally excluded: its old TensorFlow/Torch/gym-microrts stack, missing `stratvis`, and unavailable data are outside this evaluation scope.

## Results

| Environment | Causal BART fidelity | Logistic Regression | Decision Tree | BART structure time | PC time | FCI time |
| --- | --- | --- | --- | --- | --- | --- |
| CartPole | accuracy 91.62%, F1 91.50% | accuracy 90.76%, F1 90.43% | accuracy 98.20%, F1 98.16% | 242.19 s | 0.076 s | 0.061 s |
| LunarLander | accuracy 82.80%, macro-F1 64.10% | accuracy 77.21%, macro-F1 59.91% | accuracy 78.80%, macro-F1 62.75% | 1,086.40 s | 0.197 s | 0.799 s |
| PongDuel | accuracy 81.73%, macro-F1 19.51%* | accuracy 81.77%, macro-F1 62.62% | accuracy 96.21%, macro-F1 90.21% | 35,531.20 s | 0.997 s | 0.957 s |

`*` PongDuel has no fidelity notebook and its joint action is a nine-class categorical value while the repository exposes BART regression only. Its reported fidelity is therefore a documented numeric-action BART proxy: regression prediction rounded to 0--8. It is **not** a paper-equivalent categorical BART fidelity result.

## Comparison with the author-reported Table 1

| Environment | Author BART accuracy / F1 | This run | Accuracy difference | F1 difference | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| CartPole | 91.4% / 91.6% | 91.62% / 91.50% | +0.22 pp | -0.10 pp | Close reproduction of fidelity. |
| LunarLander | 88.7% / 86.6% | 82.80% / 64.10% | -5.90 pp | -22.50 pp | Directionally plausible but materially lower, especially macro-F1. |
| PongDuel | 90.2% / 90.7% | 81.73% / 19.51% proxy | N/A | N/A | Not a valid like-for-like comparison. |

The author-reported Decision Tree CartPole accuracy is 94.1%; this run obtained 98.20%. This difference is expected because the policy, seed, rollout distribution and package versions are not identical.

### Reasons for the differences

1. **Policies are heuristic, not trained checkpoints.** The repository contains no RL-training code. The rollouts here use the provided linear CartPole policy, PD LunarLander controller and strong/weak PongDuel heuristics. Different state/action distributions directly alter surrogate fidelity and causal tests.
2. **Different sample budgets and split conditions.** LunarLander uses the evaluation notebook's 30 episodes because it is expensive; train/test rows, episode terminations and random split outcomes need not match the authors' setup.
3. **BART runtime configuration differs.** This Windows run uses `dbarts` through `rpy2`, four chains and one thread, with separate Python/R processes per target to prevent native-state failures. The original experiment's in-memory execution, CPU and package versions are unknown.
4. **Metric definition is consequential for multiclass actions.** LunarLander reports macro-F1 across four actions; rare actions lower macro-F1 sharply. PongDuel has nine joint actions but no repository fidelity notebook or BART classifier, so its rounded numeric-regression proxy must not be compared with the paper's categorical fidelity.
5. **Timing measurements are not like-for-like.** The author timings appear to measure an in-memory algorithmic path; this run sums isolated BART target processes. PongDuel PC/FCI also require an independent state basis because the raw observation has deterministic duplicate/one-hot columns and Fisher-Z otherwise fails with a singular correlation matrix.

## Timing comparability

- BART timing is the sum of isolated target-process elapsed times, not the paper's single in-memory run.
- CartPole PC/FCI use full Fisher-Z wall-clock on the replay representation; LunarLander follows its timing notebook's eight state columns.
- PongDuel's raw twenty state columns are singular due to duplicate/one-hot dependent columns. PC/FCI therefore ran on a numerically independent ten-column state basis. This is explicitly not directly comparable to the paper's reported timing.
- The paper numbers should be used only as reference: policy heuristics, seed, platform, package versions and timing instrumentation differ.

## Audit artifacts

- [CartPole results](./results/cartpole/)
- [LunarLander results](./results/lunarlander/)
- [PongDuel results](./results/pongduel/)
- [Errors](./notes/errors.md), [fixes](./notes/fixes.md), [blockers](./notes/blockers.md)
