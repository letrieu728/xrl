# Repeat measurement — hte / 200 episodes (REAL BART)

> **BART (R/dbarts via rpy2) - real CausalXRL estimator**

Two INDEPENDENT processes ran the identical config (hte, 200 episodes,
heuristic policy, seed 0, same script `run_ablation_bart.py`, same
data-collection code): a pilot run first (`run_label=pilot-hte200`) and
the grid run second (`run_label=main`). The pilot output directory
(`results/ablation-bart-pilot/`) is local only and is NOT pushed; all
pilot numbers needed for audit are embedded in `repeat_hte200.json`
(also embedded in `per_run_details.json` under
`repeat_hte200_pilot_vs_grid`).

## 95% edge sets

| run | n_edges | infer_s | push parents |
|---|---|---:|---|
| pilot | 12 | 92.68 | push, x, θ (3) |
| grid (main) | 13 | 88.87 | push, x, θ (3) |

- Jaccard = 0.7857, symmetric difference = 3 edges.
- Only in grid: `v→w`, `θ→w`.
- Only in pilot: `θ→x`.
- Push parents IDENTICAL in both runs.

## Sensitivity curves (same two runs, recomputed from ITE draws)

| CI level | pilot n_edges | grid n_edges | Jaccard | symdiff |
|---|---:|---:|---:|---:|
| 90% | 15 | 13 | 0.7500 | 4 |
| 95% | 12 | 13 | 0.7857 | 3 |
| 99% | 11 | 11 | 0.8333 | 2 |

## Cross-check against the earlier XRL run-9 heuristic graphs

Same protocol and seed, independent earlier run
(`results/xrl-trained-vs-heuristic/`, heuristic policy, 200 episodes):

| method | grid n_edges | XRL run-9 n_edges | Jaccard | symdiff |
|---|---:|---:|---:|---:|
| hte | 13 | 14 | 0.8000 | 3 |
| ate | 15 | 15 | 1.0000 | 0 |

- hte: only in grid `v→w`; only in XRL run-9 `v→θ`, `w→v`.
- ate: the two edge sets are IDENTICAL — ate at 200 episodes reproduced
  exactly across independent runs.

## Conclusion

hte at 200 episodes varies by 1 edge in count (12–14 across three
independent runs: pilot 12, grid 13, XRL run-9 14) with a 3-edge
symmetric difference between runs. Cross-episode differences of 2–4
edges in the ablation grid are therefore within/near run noise for hte
and must not be over-interpreted. Only differences larger than this
band — e.g. the BART-vs-surrogate symmetric differences of 6–9 edges —
are estimator effects. ate at 200 episodes was exactly reproducible in
the two latest runs, but the wider historical range (15–19) still
stands from earlier runs (see REPORT.md).
