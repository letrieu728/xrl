# Ablation errors log — CausalXRL extension A (CartPole, surrogate estimator)

SURROGATE ESTIMATOR (HistGradientBoosting), NOT BART/dbarts — stability
check only, to be re-run with real BART when available.

## Error 1 — 2026-11-11, smoke run (50 ep, hte)
- **Symptom:** `ValueError: X.dtype should be np.float32, got float64` in
  `SurrogateRegressor.is_causal_HTE`, at `tree.tree_.apply(...)`.
- **Cause:** sklearn 1.9 `Tree.apply` requires float32 input. The repo's
  own code uses `X.astype(sklearn.tree._tree.DTYPE)` (float32) in
  `decision_tree_cluster` / `HTE.cluster`; the surrogate passed float64.
- **Fix:** cast to `np.float32` in both `tree_.apply` calls
  (`is_causal_HTE` and `is_state_causal` in `scripts/surrogate_bart.py`).
- **Status:** fixed; smoke run re-started.

## Error 2 — 2026-11-11, smoke run (50 ep, hte), second attempt
- **Symptom:** run stuck inside `HistGradientBoostingRegressor.fit`
  (`grower.py: _compute_best_split_and_push`) for >4 min on 1099×5
  data; killed. A timed repro showed a single fit taking >40 s.
- **Cause:** CPU contention, not data. `uptime` showed load average
  5.90 on a 4-CPU / 1.5-shared-CPU VM (a parallel agent is training a
  DQN on CPU at the same time). Uncapped OpenMP threads in HistGBR
  thrashed under that load. Re-timed later: ~2.4 s per fit with
  `OMP_NUM_THREADS=2`.
- **Fix:** `scripts/run_ablation.py` now sets `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` to
  2 (via `os.environ.setdefault`, before numpy/sklearn import). Base
  learner kept as default `HistGradientBoostingRegressor`
  (`random_state=0`), per the task spec.
- **Status:** fixed; smoke run re-started.

## Error 3 — 2026-11-11, smoke run (50 ep, hte), first surrogate design
- **Symptom:** smoke run completed but returned the COMPLETE graph,
  25/25 edges, every column causal for every target. The ablation
  would have been vacuous (Jaccard 1.0 everywhere, nothing to compare).
- **Cause:** design flaw, not a crash. The first surrogate used
  HistGradientBoosting + bootstrap CIs of the MEAN ITE over the
  estimation points. A CI of the mean captures only sampling variation
  of the mean, not model uncertainty, so any systematic learned effect
  -- however small -- is "significant". The repo's BART math instead
  quantifies uncertainty through posterior draws, which a point
  bootstrap cannot mimic.
- **Fix:** switched base learner to `RandomForestRegressor`
  (n_estimators=200, allowed by the task spec: "HistGradientBoosting
  ... (or RandomForest)"). The 200 trees play the role of BART's 500
  posterior draws: `ITE` returns (200, N) per-tree draws and `ATE` /
  `is_causal_HTE` apply the repo's draw-based math verbatim (per-draw
  ATE quantiles; HTE pooled draw quantiles per leaf cluster). Same
  50-episode data then gives hte: 6 edges, ate: 14 edges (vs 16
  direct-cause edges in the true CartPole-v1 physics) -- the surrogate
  now discriminates. Rewrote `scripts/surrogate_bart.py` (v2) and
  updated `scripts/run_ablation.py` (`--n-draws`, per-tree-draw
  wording, HTE cluster table). The HistGBR version is NOT kept.
- **Status:** fixed; full 6-run grid re-started with the RF surrogate.

## Error 4 — 2026-11-11, graph plotting polish (after the grid finished)
- **Symptom (a):** first redraw attempt failed with
  `KeyError: 'state_cart_position_t+1'` in `draw_networkx_labels` --
  the label-position dict only held t-nodes but the call defaulted to
  labelling all graph nodes.
- **Fix attempt (b):** passed `nodelist=` -- rejected:
  `TypeError: draw_networkx_labels() got an unexpected keyword
  argument 'nodelist'` (networkx 3.7 takes a `labels` dict instead).
- **Fix (c):** passed `labels={n: n for n in ...}` for each side.
  Then right-side labels were clipped at the figure edge, so the
  x-limits were widened (-0.55..1.55) and all 6 PNGs regenerated from
  `per_run_details.json` (no re-running of inference; edge sets
  unchanged). Verified visually: `graph_hte_200ep.png` fully legible.
- **Status:** fixed. (Errors 1-3 are above; this was the 4th fix
  overall, 2nd for plotting. Per-task limit is 3 fixes per error --
  each distinct error here needed at most 3 attempts.)
