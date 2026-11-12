# CausalXRL evaluation-only reproduction

## Correction 2026-11-12 (Step 1: baseline comparability; BART not rerun)

The baseline numbers in the Results table below are the **corrected** values.
The CartPole baselines first stored in `results/cartpole/metrics.json`
(LR accuracy 90.76%, DT accuracy 98.20%) measured a **different task** from the
authors' fidelity notebook: state_t -> action_t on all rows, 20% / seed-99
split. They were recomputed under the notebook protocol
([state_t, action_t] -> action_{t+1}, valid steps, 40% / seed-10 split).
LunarLander and PongDuel baselines already used the t -> t+1 task; they gained
weighted-F1 alongside macro-F1, because the paper defines F1 as weighted for
multiclass (section 5.2). BART fidelity values are **unchanged** (no BART rerun).
Old numbers are preserved in the original `metrics.json` files and recorded in
the "Superseded numbers" section below. Full detail, including per-class F1,
split counts and replay hashes: [recomputed baselines](./results/baselines_recomputed.md)
(JSON: [baselines_recomputed.json](./results/baselines_recomputed.json)),
script: [recompute_baselines_weighted_f1.py](./scripts/recompute_baselines_weighted_f1.py).

## Scope and conditions

- No RL agent was trained. CartPole uses the repository's hand-written linear policy; LunarLander uses its hand-written PD controller; PongDuel uses the repository's strong/weak heuristic joint policy.
- Seed: 99. CartPole: 200 episodes / 6,115 non-terminal transitions. LunarLander: 30 episodes / 7,382 transitions. PongDuel: 200 episodes / 74,445 transitions (14,995 local candidate samples used for fidelity).
- BART runs use `dbarts` through `rpy2`, four posterior chains and one thread. DBN targets were evaluated in isolated Python/R processes so each completed target is independently checkpointed.
- MicroRTS was intentionally excluded: its old TensorFlow/Torch/gym-microrts stack, missing `stratvis`, and unavailable data are outside this evaluation scope.

## Fidelity protocol (authors' notebooks, yanmluk/causal-xrl @ ae9096e)

- Valid steps: timesteps whose local causal model has more than one non-action cause of the action node (`count > 1`; `eval_fidelity_Cart Pole.ipynb` cell 11, `eval_fidelity_Lunar Lander.ipynb` cell 11). The saved BART runs kept every candidate row (CartPole 6,114/6,114; LunarLander 7,381/7,381), so the recomputed baselines use all t -> t+1 pairs; per-step valid masks were not saved, and reproducing them would require rerunning the BART/SCM inference.
- Features: the full representation row at t, including the action at t. Target: the action at t+1.
- Split: `test_size=0.4, random_state=10`; LunarLander stratified, CartPole unstratified (notebook cell 15 in each). PongDuel follows the LunarLander variant as an adaptation: PongDuel has no fidelity notebook, only an inference example.
- Models: `DecisionTreeClassifier(random_state=1)`, `LogisticRegression(random_state=1)` (CartPole: notebook defaults; LunarLander/PongDuel: `max_iter=1000`).
- Metrics: accuracy is the notebook metric. F1 follows the paper (section 5.2: "average='weighted' if multiclass"): binary F1 for CartPole, weighted-F1 for LunarLander/PongDuel. Macro-F1 is kept alongside for the multiclass environments. Note the group's CartPole BART fidelity used a stratified split while the notebook split is unstratified, so BART and recomputed-baseline test rows differ; a stratified-split sensitivity for the baselines is in `baselines_recomputed.md` (LR 90.43%, DT 95.42% accuracy, same test rows as BART).

## Results

| Environment | Causal BART fidelity | Logistic Regression | Decision Tree | BART structure time | PC time | FCI time |
| --- | --- | --- | --- | --- | --- | --- |
| CartPole | accuracy 91.62%, binary F1 91.50% | accuracy 89.62%, binary F1 89.26% | accuracy 93.83%, binary F1 93.64% | 242.19 s | 0.076 s | 0.061 s |
| LunarLander | accuracy 82.80%, macro-F1 64.10% (weighted-F1 not available) | accuracy 77.21%, weighted-F1 76.24%, macro-F1 59.91% | accuracy 79.14%, weighted-F1 79.06%, macro-F1 63.25% | 1,086.40 s | 0.197 s | 0.799 s |
| PongDuel | accuracy 81.73%, macro-F1 19.51%* | accuracy 81.79%, weighted-F1 81.38%, macro-F1 62.65% | accuracy 96.21%, weighted-F1 96.21%, macro-F1 90.21% | 35,531.20 s | 0.997 s | 0.957 s |

`*` PongDuel has no fidelity notebook and its joint action is a nine-class categorical value while the repository exposes BART regression only. Its reported fidelity is therefore a documented numeric-action BART proxy: regression prediction rounded to 0--8. It is **not** a paper-equivalent categorical BART fidelity result.

Sample counts (t -> t+1 pairs / train / test): CartPole 6,114 / 3,668 / 2,446; LunarLander 7,381 / 4,428 / 2,953; PongDuel 74,444 / 44,666 / 29,778. The PongDuel BART proxy used a different subset (14,995 local candidates / 5,998 test).

### Superseded numbers (kept for history; do not quote as results)

- CartPole baselines in `results/cartpole/metrics.json` (states -> action_t, 20% / seed 99, stratified): LR accuracy 90.76%, F1 90.43%; DT accuracy 98.20%, F1 98.16%. Rerunning that old task on the same replay reproduces these values exactly, confirming the data is unchanged and only the task definition differed. The old DT value (98.20%) in particular must not be compared with the paper's DT accuracy (94.1%): it classified the action taken *at* each state, which the linear policy makes almost deterministic from the state alone.
- LunarLander DT in `results/lunarlander/metrics.json` (same t -> t+1 task): accuracy 78.80%, macro-F1 62.75%, vs recomputed 79.14% / 63.25%. LR reproduces exactly; the stored DT accuracy matches `DecisionTreeClassifier(random_state=99)`, not the `random_state=1` in the committed script and notebook, and the stored macro-F1 matches neither under scikit-learn 1.5.2 or 1.9.1. The recomputed `random_state=1` values are kept as the notebook-faithful ones.
- PongDuel LR in `results/pongduel/metrics.json`: accuracy 81.77%, macro-F1 62.62%, vs recomputed 81.79% / 62.65% (difference <0.03 pp, solver convergence). DT reproduces exactly.

## Comparison with the author-reported Table 1

### BART fidelity vs Table 1

| Environment | Author BART accuracy / F1 | This run | Accuracy difference | F1 difference | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| CartPole | 91.4% / 91.6% | 91.62% / 91.50% (binary F1) | +0.22 pp | -0.10 pp | Close reproduction of fidelity. |
| LunarLander | 88.7% / 86.6% (weighted F1) | 82.80% / 64.10% (macro-F1) | -5.90 pp | N/A | F1 not like-for-like: macro vs weighted. BART weighted-F1 cannot be computed from saved artifacts (predictions not saved). |
| PongDuel | 90.2% / 90.7% | 81.73% / 19.51% proxy | N/A | N/A | Not a valid like-for-like comparison. |

The previously reported LunarLander F1 difference of -22.50 pp compared macro-F1 with the paper's weighted-F1 and is withdrawn as a like-for-like figure.

### Baselines vs Table 1 (paper-comparable F1: binary for CartPole, weighted otherwise)

| Environment | LR accuracy / F1 (paper) | LR this run | DT accuracy / F1 (paper) | DT this run |
| --- | --- | --- | --- | --- |
| CartPole | 88.5% / 90.0% | 89.62% / 89.26% (+1.12 pp / -0.74 pp) | 94.1% / 94.6% | 93.83% / 93.64% (-0.27 pp / -0.96 pp) |
| LunarLander | 81.3% / 80.2% | 77.21% / 76.24% (-4.09 pp / -3.96 pp) | 85.1% / 84.7% | 79.14% / 79.06% (-5.96 pp / -5.64 pp) |
| PongDuel | 81.5% / 81.1% | 81.79% / 81.38% (+0.29 pp / +0.28 pp) | 88.3% / 88.4% | 96.21% / 96.21% (+7.91 pp / +7.81 pp, adaptation) |

As in the paper, Decision Tree beats BART fidelity on CartPole (93.83% vs 91.62% accuracy on the notebook split; 95.42% vs 91.62% on the shared stratified test rows). The PongDuel DT result exceeds the paper's because the baseline protocol is an adaptation (no author fidelity notebook) and the heuristic joint policy is highly predictable from the state.

### Reasons for the differences

1. **Policies are heuristic, not trained checkpoints.** The repository contains no RL-training code. The rollouts here use the provided linear CartPole policy, PD LunarLander controller and strong/weak PongDuel heuristics. Different state/action distributions directly alter surrogate fidelity and causal tests.
2. **Different sample budgets and split conditions.** LunarLander uses the evaluation notebook's 30 episodes because it is expensive; train/test rows, episode terminations and random split outcomes need not match the authors' setup. CartPole BART fidelity additionally used a stratified split where the notebook uses an unstratified one.
3. **BART runtime configuration differs.** This Windows run uses `dbarts` through `rpy2`, four chains and one thread, with separate Python/R processes per target to prevent native-state failures. The original experiment's in-memory execution, CPU and package versions are unknown.
4. **Metric definition is consequential for multiclass actions.** LunarLander reports macro-F1 across four actions; rare actions lower macro-F1 sharply. Weighted-F1 (the paper's definition) is 76.24% / 79.06% for LR/DT, vs macro-F1 59.91% / 63.25%. BART weighted-F1 is unavailable (predictions not saved), so the BART F1 comparison with Table 1 remains open. PongDuel has nine joint actions but no repository fidelity notebook or BART classifier, so its rounded numeric-regression proxy must not be compared with the paper's categorical fidelity.
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
- [Recomputed baselines (correction 2026-11-12)](./results/baselines_recomputed.md) ([JSON](./results/baselines_recomputed.json), [script](./scripts/recompute_baselines_weighted_f1.py))
- [Errors](./notes/errors.md), [fixes](./notes/fixes.md), [blockers](./notes/blockers.md)
