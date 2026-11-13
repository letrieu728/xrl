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

## Follow-up 2026-11-12 (Group 1: recovered metrics, aligned split, graphs)

Three follow-ups landed after the correction above; BART was not refit for
the reported numbers except where stated:

- **LunarLander BART weighted-F1 recovered: 81.52%.** The per-sample
  predictions were recovered exactly from the saved per-action score
  checkpoints (re-aggregation reproduces accuracy 82.80% and macro-F1
  64.10% with difference 0.0), so the BART-vs-Table-1 F1 comparison is
  now like-for-like. See [below](#lunarlander-bart-fidelity--weighted-f1-recovered-2026-11-12)
  and `results/lunarlander/bart_fidelity_weighted.json`.
- **CartPole baselines aligned to the BART test split.** The BART run used
  a stratified split while the notebook protocol is unstratified; on the
  shared BART test rows LR is 90.43% / 90.22% and DT 95.42% / 95.32%
  (accuracy / binary F1). See [below](#cartpole-baselines-aligned-to-the-bart-test-split-2026-11-12)
  and `results/baselines_aligned.json` / `baselines_aligned.md`. The
  notebook-split numbers in the Results table remain the primary
  notebook-protocol comparison; the two splits must not be mixed.
- **Global causal graphs plotted and a Fig. 4-style local case study.**
  Plots of the learned graphs for all three environments and one examined
  CartPole state are in the
  [Global causal graphs](#global-causal-graphs-bart-hte-plotted-from-the-learned-cause-lists)
  and [Local case study](#local-case-study-cartpole-fig-4-style) sections
  below. The PNG files live on Drive (same relative paths as the repo)
  because the GitHub connector accepts text files only; the plotting
  scripts are committed in `scripts/`.

## Follow-up 2026-11-12 (Group 3a: GES structure learning)

GES, the third structure-learning baseline in the authors' `eval_time`
notebooks (alongside PC/FCI), was run on the same saved replays with
plain causal-learn BIC and full wall-clock timing — the same precedent
as PC/FCI in `metrics.json`, rejecting the notebooks' simulated-cost
timer (see the [GES section](#ges-structure-learning-2026-11-12)).
Median of 3 runs (identical edge sets every run): CartPole 0.018 s /
7 edges, LunarLander 0.090 s / 18 edges, PongDuel 0.346 s / 34 edges;
MicroRTS 0.563 s / 50 edges (states only) and 0.685 s / 53 edges
(states + action). No old number was changed. Full edge lists, scores,
representations and same-session PC/FCI re-runs:
`results/cartpole/ges.json`, `results/lunarlander/ges.json`,
`results/pongduel/ges.json`, `results/microrts/ges.json`;
script: [run_ges.py](./scripts/run_ges.py).

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
- Metrics: accuracy is the notebook metric. F1 follows the paper (section 5.2: "average='weighted' if multiclass"): binary F1 for CartPole, weighted-F1 for LunarLander/PongDuel. Macro-F1 is kept alongside for the multiclass environments. Note the group's CartPole BART fidelity used a stratified split while the notebook split is unstratified, so BART and recomputed-baseline test rows differ; the full aligned-split baseline comparison (LR 90.43% / 90.22%, DT 95.42% / 95.32%, same test rows as BART) is in `results/baselines_aligned.md` ([JSON](./results/baselines_aligned.json)).

## Results

| Environment | Causal BART fidelity | Logistic Regression | Decision Tree | BART structure time | PC time | FCI time |
| --- | --- | --- | --- | --- | --- | --- |
| CartPole | accuracy 91.62%, binary F1 91.50% | accuracy 89.62%, binary F1 89.26% | accuracy 93.83%, binary F1 93.64% | 242.19 s | 0.076 s | 0.061 s |
| LunarLander | accuracy 82.80%, weighted-F1 81.52%, macro-F1 64.10% | accuracy 77.21%, weighted-F1 76.24%, macro-F1 59.91% | accuracy 79.14%, weighted-F1 79.06%, macro-F1 63.25% | 1,086.40 s | 0.197 s | 0.799 s |
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
| LunarLander | 88.7% / 86.6% (weighted F1) | 82.80% / 81.52% (weighted-F1) | -5.90 pp | -5.08 pp | Like-for-like weighted-F1 comparison; BART weighted-F1 recovered from the saved per-action score checkpoints (see below). Macro-F1 64.10% is kept in the Results table. |
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
4. **Metric definition is consequential for multiclass actions.** LunarLander reports macro-F1 across four actions; rare actions lower macro-F1 sharply. Weighted-F1 (the paper's definition) is 76.24% / 79.06% for LR/DT, vs macro-F1 59.91% / 63.25%. BART weighted-F1 was recovered from the saved per-action score checkpoints (81.52%; see below), so the BART F1 comparison with Table 1 is now like-for-like. PongDuel has nine joint actions but no repository fidelity notebook or BART classifier, so its rounded numeric-regression proxy must not be compared with the paper's categorical fidelity.
5. **Timing measurements are not like-for-like.** The author timings appear to measure an in-memory algorithmic path; this run sums isolated BART target processes. PongDuel PC/FCI also require an independent state basis because the raw observation has deterministic duplicate/one-hot columns and Fisher-Z otherwise fails with a singular correlation matrix.

## Recovered metrics and aligned split (2026-11-12)

### LunarLander BART fidelity — weighted F1 recovered (2026-11-12)

The original BART fidelity run stored only accuracy and macro-F1
(`muc3/results/lunarlander/bart_fidelity.json`: accuracy 82.80%,
macro-F1 64.10%, test n = 2,953). Its per-sample predictions can
nevertheless be recovered exactly: the per-action BART score
checkpoints (`muc3/results/lunarlander/fidelity_scores/action_{8..11}.npz`)
were saved, and the aggregate prediction is a deterministic function
of them (normalised score per action, test-set prior fallback where a
row has zero total weight, argmax over the four actions — the logic of
`lunarlander_fidelity_aggregate.py`). Re-aggregating those checkpoints
reproduces accuracy 82.80% and macro-F1 64.10% with difference 0.0, and
gives the paper's multiclass metric (arXiv:2610.42028v1, §5.2):
**weighted F1 81.52%**. Per-class F1: do_nothing 87.51% (support 1,479),
fire_left_engine 59.68% (159), fire_main_engine 83.06% (1,185),
fire_right_engine 26.14% (130). The gap between weighted F1 (81.52%)
and macro-F1 (64.10%) is the rare-action effect: fire_right_engine
recall is only 15.38% (20/130), and it carries 1/4 of the macro
average but only 4.4% of the weighted average. On the same test rows
the recomputed baselines are LR 77.21% accuracy / 76.24% weighted F1
and DT 79.14% / 79.06%, so the ordering BART > DT > LR holds for
weighted F1 as well as for accuracy; the paper's Table 1 CausalXRL
figures are 88.70% accuracy / 86.60% F1.

Caveats. BART was NOT refit for this number: the weighted F1 belongs
to the original run, and the stored `bart_fidelity.json` is unchanged
history. A fresh BART rerun would give different numbers — the author
code path sets no seed (unseeded sample split in `infer_causes_hte`,
unseeded dbarts fit), so cause sets and scores vary run to run; the
original run's own repeated inferences already disagreed (e.g. three
different cause sets for the fire_left_engine target). A fresh full
rerun on the same replay and the same test rows (2026-11-12, this VM)
confirms the scale of that variability: accuracy 82.56%, macro-F1
67.80%, weighted F1 81.74% — accuracy/weighted-F1 within 0.25 pp of
the reported run, macro-F1 +3.71 pp via the rare fire_right_engine
class; 118 of 2,953 predictions changed. The rerun is a variability
check only and does not replace the numbers above. The group's
prior-fallback choice (test-set priors, where the author notebook
uses train-set priors) was checked and changes zero of the 2,953
predictions, so it does not affect any reported metric. Original BART
runtime: 1,008.19 s (masks 201.20 s + action scores 806.99 s).
Predictions and scores: [`bart_fidelity_weighted_predictions.npz` (Drive)](https://drive.google.com/file/d/1RmlwK1RsBkbNjQj9ZUYH9nfg6BXtzsCu/view?usp=drivesdk);
recomputation: `muc3/scripts/recompute_lunarlander_bart_weighted_f1.py`.
This supersedes the statement in `baselines_recomputed.md` that BART
weighted F1 "cannot be computed from the saved artifacts".

### CartPole baselines aligned to the BART test split (2026-11-12)

The published CartPole baselines and the BART fidelity numbers were
computed on different test rows. The BART script
(`muc3/scripts/cartpole_bart_fidelity.py`) splits with
`train_test_split(arange(6114), test_size=0.4, random_state=10,
stratify=eval_y)` — stratified — while the author notebook
(`eval_fidelity_Cart Pole.ipynb`, cell 15) splits without stratify,
which is what the primary numbers in `baselines_recomputed.json`
follow. Same sizes (train 3,668 / test 2,446), but only 936 of the
2,446 test rows are shared (Jaccard 0.237). The valid set in both
cases is all 6,114 t -> t+1 candidates: the stored BART run kept
6,114/6,114, so the valid rows are every candidate row (the per-step
mask itself was not saved; the count equality is the evidence).

Recomputed on the BART split exactly (same rows, same
[state_t, action_t] -> action_{t+1} task, notebook model settings):
**logistic regression 90.43% accuracy / 90.22% binary F1; decision
tree 95.42% / 95.32%**. BART on the same rows: 91.62% / 91.50%
(from `bart_fidelity.json`, not recomputed). On the aligned split the
ordering is DT > BART > LR, the same ordering as on the notebook
split (DT 93.83% / LR 89.62%), with different gaps. The aligned
numbers are the like-for-like BART comparison; the unstratified
primary numbers remain the notebook-protocol comparison. The two
splits must not be mixed in one table. Details and test indices:
`muc3/results/baselines_aligned.json` and `baselines_aligned.md`;
script: `muc3/scripts/align_cartpole_baselines_to_bart_split.py`.

## Global causal graphs (BART HTE, plotted from the learned cause lists)

The three figures linked below plot the learned global causal graphs from the
per-target cause lists, one node per variable (state + action) and one
arrow per learned cause relation `cause(t) -> target(t+1)`. The plots
add no information: every arrow is a cause entry from the source files,
and no edge weights are shown because none were stored.

| Environment | Learned cause relations | Drawn arrows | Persistence relations | Source |
| --- | ---: | ---: | ---: | --- |
| CartPole | 13 | 8 | 5 | `results/cartpole/metrics.json` (`bart_targets`) |
| LunarLander | 44 | 32 | 12 | `results/lunarlander/metrics.json` (`bart_targets`, one-hot actions) |
| PongDuel | 53 | 36 | 17 | `results/pongduel/targets/target_0..20.json` (per-target checkpoints) |

[CartPole global causal graph (PNG, Drive)](https://drive.google.com/file/d/19PicmWOAUWkNyxgcSUc2Hfwpxc7EVCgr/view?usp=drivesdk)

[LunarLander global causal graph (PNG, Drive)](https://drive.google.com/file/d/1HSkl-QycUNTrHpGM8cA0hUbFGjQhRE1o/view?usp=drivesdk)

[PongDuel global causal graph (PNG, Drive)](https://drive.google.com/file/d/168rTQwuo08v4uA_Agaqx00UlQQ1X3rsX/view?usp=drivesdk)

Plotting script: [plot_muc3_graphs.py](./scripts/plot_muc3_graphs.py)
(networkx + matplotlib; reruns from the JSON files above).

Two conventions need stating, because they decide what the figures do
and do not show. First, a *persistence* relation — a variable's own
value at `t` learned as a cause of its value at `t+1` (cause == target)
— is not drawn as a self-loop. The node label carries a `(p)` marker
and the figure header prints the persistence count, following the
authors' `DBN_to_causal` collapse
(`cxrl/lib/causal/infer_network.py` @ ae9096e), which drops the
`t -> t+1` self-edge when collapsing the DBN to one layer. The table
above therefore always gives the total alongside the drawn arrows, so
no learned relation is hidden: CartPole 13 = 8 + 5, LunarLander
44 = 32 + 12, PongDuel 53 = 36 + 17. Second, variables with no learned
relation at all are still drawn (grey): in PongDuel, `agent1_col` and
`agent2_col` have no learned cause and no learned effect, and
`dir_NW2` and `encoded_action` are the only variables with learned
causes but no persistence relation.

PongDuel completeness was checked, not assumed: `metrics.json` stores
no per-target causes for PongDuel, so the 21 checkpoint files under
`results/pongduel/targets/` are the source, and their summed
`elapsed_seconds` equal the `bart_hte_isolated_target_sum` in
`metrics.json` exactly (35,531.20 s), confirming the set is complete
(21/21 targets). The same check holds for CartPole (242.19 s) and
LunarLander (1,086.40 s). The PNG files are stored on Drive under the same relative paths as this report ([CartPole graphs folder](https://drive.google.com/drive/folders/1hld8QccmurkgjnRI5tLzg3-PVe76JIL-), [LunarLander graphs folder](https://drive.google.com/drive/folders/1DJ4SGPOE17ql-glB-TrVKiInCqczAbXV), [PongDuel graphs folder](https://drive.google.com/drive/folders/1XbobDlq9cs05NeEgZv4Y33DBTIgf4U8w)), because the GitHub connector accepts text files only. All three figures state the environment,
method (BART HTE), episode / transition counts and seed 99 in their
headers: CartPole 200 episodes / 6,115 transitions, LunarLander
30 / 7,382, PongDuel 200 / 74,445.

## Local case study (CartPole, Fig. 4 style)

Following the paper's Figure 4 (p. 7) — a sampled game state next to
its local causal model, with signed edge weights — one CartPole state
from the Mục 3 replay was examined with the authors' local-causal
procedure. The local model needs the fitted BART structural causal
models, which Mục 3 did not save (only cause lists), so the five BART
HTE models were refit on the same replay with the same per-target
`infer_causes_hte` call, seeded (seed 99), in 173.8 s with real
`dbarts` 0.9.34 via `rpy2`. The refit reproduces the stored global
causes exactly for `cart_velocity` and `push`, and differs by one
cause on each of the other three targets (`cart_position` loses
`pole_angle`; `pole_angle` trades `cart_velocity` for `pole_velocity`;
`pole_velocity` loses `pole_angle`). The stored lists above remain the
Section 3 global result; the refit exists only to obtain the local
model, and both cause sets are recorded in
[local_case_study.json](./results/cartpole/local_case_study.json).
Script: [local_case_study_cartpole.py](./scripts/local_case_study_cartpole.py).

[CartPole local causal case study figure (PNG, Drive)](https://drive.google.com/file/d/1f3hK-ss1JdSzLjvYTFE9dPT7IObewEyn/view?usp=drivesdk)

The sampled state is replay row 1131 of 6,115, chosen by a fixed rule:
among the 5,086 of 6,114 scanned rows whose local graph has more than
one non-action local cause of `push` (the fidelity notebook's validity
rule), and whose local `push` causes include both inputs of the
hand-written linear policy, the row with the largest `|pole_angle|`.
The state is close to failure: the pole leans right at 0.2075 rad
(11.89°), within 0.002 rad of the 12° termination threshold, while the
cart sits left of centre and moves right.

| Quantity | Value |
| --- | --- |
| `cart_position` / `cart_velocity` | −0.1543 / +0.1748 |
| `pole_angle` / `pole_velocity` | +0.2075 rad (+11.89°) / −0.1255 |
| Policy action at this state | push right (1) |
| Policy score `0.4·(x/2.4) + 0.6·(θ/2.095)` | +0.0337 (position term −0.0257, angle term +0.0594) |
| Recorded next action / next `pole_angle` | 1 / 0.2049 rad |

At this state the local causal model (raw query row, notebook
convention; see the caveat below) has 10 local edges. The local
parents of `push` are `cart_position`, `pole_angle` and `push` itself,
with posterior-mean ITE weights (500 draws; 95% intervals in brackets):

| Local edge into `push` | ITE mean | 95% interval |
| --- | ---: | --- |
| `cart_position -> push` | +2.35 | (+1.54, +3.44) |
| `pole_angle -> push` | +4.16 | (+3.41, +5.00) |
| `push -> push` | +1.51 | (+1.04, +2.02) |

The remaining local edges are `cart_velocity -> cart_position` (+0.02),
`push -> cart_velocity` (+0.39), `push -> pole_velocity` (−0.58), and
local self-effects on all five variables (+0.08, +0.80, +0.10, +1.05,
+1.51 in feature order). In words: the local model says the agent
pushes right here mainly because the pole leans right — the angle term
is what tips the linear policy's score positive against the
left-of-centre cart position — and both policy inputs are locally
causal with positive effects, so moving either further right would
strengthen the push-right tendency. The two outgoing local edges from
`push` are physically coherent: pushing right increases the cart's
rightward velocity (+0.39) and reduces the pole's angular velocity
(−0.58), i.e. it drives the cart under the falling pole. The small
`cart_velocity -> cart_position` weight (+0.02) matches the 0.02 s
integration step: one unit of extra velocity moves the cart about
0.02 units in one step. Neither velocity variable is a local parent
of `push` at this state, which matches the policy: the linear rule
uses only position and angle. Note that `pole_velocity` has a
non-zero single-row ITE for `push` (+1.07) yet is not locally causal;
local causality is a cluster-level call in `local_treatments`, not a
single-row ITE test.

Caveats, in order of importance. (1) The local graph is sensitive to
the query scale. The HTE trees were fitted on standardized data while
the notebook queries them with the raw row; a diagnostic that
standardizes the query row instead (global scaler, numeric columns
only) changes the local model on all five targets — `push` keeps only
`cart_position` and `push`, losing `pole_angle`, and three targets
lose all local causes. The case study above follows the notebook's
raw-query convention, which is also what the fidelity evaluation uses,
but the local explanation must not be read as scale-robust. The
diagnostic is recorded in full in `local_case_study.json`.
(2) ITE contrasts are +1 raw unit per state column (0 -> 1 for `push`),
so magnitudes are not unit-free importances across columns with
different scales, and the local self-effect magnitudes in particular
should not be given a physical reading. (3) No BART action prediction
is reported for this state: the SCM was fitted on standardized data,
and predicting on the raw query row would mix scales; only the
recorded replay actions are quoted. (4) As the paper states (§4.3.3),
the local model is a behavioural surrogate of the heuristic policy
over collected trajectories, not the policy's internal reasoning.

## GES structure learning (2026-11-12)

The authors' `eval_time` notebooks run three baselines — PC, FCI and
GES — but the earlier `metrics.json` files recorded only PC/FCI. GES
was run now on the same saved replays. The replays were re-downloaded
and verified by SHA-256 against `baselines_recomputed.json` (all three
match exactly) before running.

**What was run.** causal-learn 0.1.4.8,
`ges(X, score_func="local_score_BIC", maxP=None, parameters=None,
node_names=...)` — the authors' call, with the standard BIC penalty
(`lambda_value` default 0.5), no maximum-parent limit, no
standardisation, and the graph's edge list extracted from the returned
CPDAG (directions cross-checked against the graph's own edge API).
Three repetitions per environment; edge sets and scores were identical
across repetitions, so the median wall-clock time is reported. Python
3.12.3, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, local CPU only.
Script: [run_ges.py](./scripts/run_ges.py); run log:
`results/ges_run.log`.

**The one deliberate protocol deviation: real timing, not the
simulated timer.** The notebooks do not time plain GES. They monkeypatch
the BIC score with a block that simulates BART cost (random 500-draw
arrays and quantiles per score evaluation) and time *only that
simulation block* ("SLA elapsed score-computation time"). That is not a
GES runtime, so it was not reproduced — the same precedent already set
for PC/FCI in `metrics.json` ("full algorithm wall-clock times ... not
the paper's simulated CI-test timer"). GES timings below are full
`ges()` wall-clock times and must not be compared with any simulated
timer figure.

**Input matrices** match the PC/FCI matrices in the metrics scripts
exactly: CartPole `[states(4), action]` (6,115 x 5); LunarLander states
only (7,382 x 8, `INCLUDE_ACTION=False` per the timing notebook);
PongDuel the greedy numerically-independent 10-column state basis from
`pongduel_metrics.py` (same rank procedure, `tol=1e-10`; the reproduced
kept columns `[0, 2, 3, 4, 5, 6, 7, 8, 9, 10]` match `metrics.json`
exactly). MicroRTS has no timing notebook or PC/FCI precedent: two
variants on `microrts_merged.npz` (100 episodes) — states only
(4,924 x 12) and states + action label on acted rows (3,200 x 13) —
both dropping the constant `light2` column (zero variance; the BIC
regression is undefined for it). No conversion beyond the PongDuel
basis and the MicroRTS constant-column drop was needed; GES takes the
same float matrices as PC/FCI.

### Timing comparison (seconds)

BART/PC/FCI "quoted" columns are unchanged values from
`results/*/metrics.json` (earlier runs, different sessions/machines).
"Same-session" PC/FCI columns are re-runs with causal-learn 0.1.4.8 on
the same matrices in the same session as GES, included because the
quoted timings are not same-session with GES. Same-session PC/FCI are
consistently faster than the quoted ones, which shows how
machine/version-dependent sub-second timings are; only the order of
magnitude (GES/PC/FCI under 1 s vs BART in the hundreds to tens of
thousands of seconds) should be read as a finding.

| Environment | BART structure time (quoted) | PC time (quoted) | FCI time (quoted) | GES time (median of 3, new) | PC time (same-session re-run) | FCI time (same-session re-run) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CartPole | 242.19 | 0.076 | 0.061 | **0.018** | 0.014 | 0.017 |
| LunarLander | 1,086.40 | 0.197 | 0.799 | **0.090** | 0.040 | 0.104 |
| PongDuel | 35,531.20 | 0.997 | 0.957 | **0.346** | 0.273 | 0.274 |
| MicroRTS (states only) | 498.30 (all 14 BART targets incl. action, §8 of the MicroRTS report) | — (never run) | — (never run) | **0.563** | 0.165 | 0.520 |
| MicroRTS (states + action) | — (action target: 21.21 s for 1 cause; different object) | — | — | **0.685** | 0.286 | 1.009 |

### Edge counts (different objects — read the caveat)

| Environment | BART DBN edges (t -> t+1) | GES edges (contemporaneous CPDAG) |
| --- | ---: | --- |
| CartPole | 13 (sum of per-target causes, `metrics.json`) | 7 (0 directed, 7 undirected) |
| LunarLander | 44 (sum of per-target causes, `metrics.json`) | 18 (18 directed, 0 undirected) |
| PongDuel | 53 (sum of per-target causes, 21 target files; time sum reproduces `metrics.json` 35,531.20 s) | 34 (32 directed, 2 undirected) |
| MicroRTS (states only) | 23 DBN edges (state targets, `microrts_graph.json`) | 50 (47 directed, 3 undirected) |
| MicroRTS (states + action) | 1 cause for the separate action target | 53 (53 directed) |

PC/FCI edge counts were never recorded (the metrics scripts discard
their graphs) and are not invented here. Caveats, in order of
importance: (1) **GES/PC/FCI learn a contemporaneous graph** over
`[state(, action)]` at time t, while **BART learns a DBN** over t -> t+1
(state differences for MicroRTS). Edge counts across the two columns
are counts of different objects and must not be ranked against each
other. (2) The CartPole GES result is a fully unoriented CPDAG (no
v-structure was identified), so all 7 adjacencies are undirected; the
LunarLander and MicroRTS graphs are fully (or almost fully) oriented.
(3) BIC assumes linear-Gaussian data; binary action/contact columns and
count columns violate that assumption exactly as they do for Fisher-Z
PC/FCI. (4) GES scores (higher is better, BIC scale): CartPole
43,942.03; LunarLander 49,162.55; PongDuel 1,932,199.17; MicroRTS
states-only and states+action scores are in `ges.json`. Scores are not
comparable across environments (different n, variables and scales).

## Timing comparability

- BART timing is the sum of isolated target-process elapsed times, not the paper's single in-memory run.
- CartPole PC/FCI use full Fisher-Z wall-clock on the replay representation; LunarLander follows its timing notebook's eight state columns.
- PongDuel's raw twenty state columns are singular due to duplicate/one-hot dependent columns. PC/FCI therefore ran on a numerically independent ten-column state basis. This is explicitly not directly comparable to the paper's reported timing.
- GES (2026-11-12) uses full plain-BIC wall-clock on the same representations, not the notebooks' simulated-cost timer; see the GES section. MicroRTS GES drops the constant `light2` column and is contemporaneous, not a DBN.
- The paper numbers should be used only as reference: policy heuristics, seed, platform, package versions and timing instrumentation differ.

## Audit artifacts

- [CartPole results](./results/cartpole/)
- [LunarLander results](./results/lunarlander/)
- [PongDuel results](./results/pongduel/)
- [Recomputed baselines (correction 2026-11-12)](./results/baselines_recomputed.md) ([JSON](./results/baselines_recomputed.json), [script](./scripts/recompute_baselines_weighted_f1.py))
- [LunarLander BART weighted-F1 (recovered 2026-11-12)](./results/lunarlander/bart_fidelity_weighted.json) ([script](./scripts/recompute_lunarlander_bart_weighted_f1.py), [per-sample predictions, Drive](https://drive.google.com/file/d/1RmlwK1RsBkbNjQj9ZUYH9nfg6BXtzsCu/view?usp=drivesdk))
- [CartPole baselines aligned to the BART split](./results/baselines_aligned.md) ([JSON](./results/baselines_aligned.json), [script](./scripts/align_cartpole_baselines_to_bart_split.py))
- [Global causal graph plots (Drive)](https://drive.google.com/drive/folders/1_bpPrzqfPPgjwxBRsK8xZWuQkvoj0Kcx) ([plotting script](./scripts/plot_muc3_graphs.py)); [CartPole local case study data](./results/cartpole/local_case_study.json) ([script](./scripts/local_case_study_cartpole.py))
- [GES structure learning (2026-11-12)](./results/cartpole/ges.json): [CartPole](./results/cartpole/ges.json), [LunarLander](./results/lunarlander/ges.json), [PongDuel](./results/pongduel/ges.json), [MicroRTS](./results/microrts/ges.json) ([script](./scripts/run_ges.py), [run log](./results/ges_run.log))
- [Errors](./notes/errors.md), [fixes](./notes/fixes.md), [blockers](./notes/blockers.md)
