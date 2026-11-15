# Edge weights — LunarLander (REAL BART, hte) + CartPole status

> **BART (R/dbarts via rpy2) - real CausalXRL estimator**

> Weights computed from the SAME 500 saved ITE posterior draws as `lunarlander_ci_sensitivity.md` (same fit, same estimation splits, no refit). Data: the existing Muc 3 replay (`muc3/results/lunarlander/replay.npz`, 30 episodes / 7,382 transitions, PD controller, collection seed 99); split seed 0; BART R seed 99 hardcoded in the repo.

## Definition and caveat

- **Weight** of a (treatment, target) pair = mean of |ITE| over all 500 posterior draws x N estimation points (N = 2,215 per target). **CI** = quantiles of the per-draw mean |ITE| (500 values). Signed mean ITE + its CI and the fraction of (draw, point) cells with ITE > 0 are reported alongside for direction/heterogeneity. Full per-pair numbers (all 144 pairs, 90/95/99% CIs): `edge_weights.json`.

- **CAVEAT — read before quoting any weight.** The weight is mean |ITE| from BART on standardised inputs. It is NOT a causal coefficient with physical units, NOT comparable across targets with different outcome scales (e.g. a binary contact target vs a continuous velocity target), and NOT a significance measure: a large weight with a CI covering 0 in every cluster is not an edge, and a tiny weight with a tight CI can be a 99% edge (see examples below). BART has run-to-run noise (see the repeatability note in `lunarlander_ci_sensitivity.md`); weights from this single fit are descriptive, not population values. Binary-target pairs can show large extrapolated ITE values because BART regression predictions are unbounded and counterfactual inputs can fall off-manifold (e.g. the contact self-loops below) — report the number, do not interpret it as a probability change.

- **Stability flags (scale-free).** `survives_99` = edge still present at 99% CI. `margin95` = largest exclusion distance of a pooled 95% cluster CI from 0, in the target's outcome units — only meaningful WITHIN one target, never across targets.

## Ranked 95% edges (39, by mean |ITE|)

| rank | edge | mean \|ITE\| | CI95 of mean \|ITE\| | mean ITE | CI95 of mean ITE | survives 99% | sig. clusters (95%) | margin95 |
|---:|---|---|---:|---:|---|---:|---:|---:|
| 1 | `state_right_contact_t -> state_right_contact_t+1` | 3.5805 | [3.1175, 4.0293] | +3.5805 | [+3.1175, +4.0293] | yes | 7/7 | 3.4196 |
| 2 | `state_left_contact_t -> state_left_contact_t+1` | 3.2646 | [2.8777, 3.5711] | +3.2646 | [+2.8777, +3.5711] | yes | 7/7 | 3.2213 |
| 3 | `action_fire_left_engine_t -> action_fire_left_engine_t+1` | 1.2838 | [0.6804, 2.0375] | +1.2585 | [+0.5815, +2.0375] | yes | 5/7 | 0.8440 |
| 4 | `action_fire_right_engine_t -> action_fire_right_engine_t+1` | 1.1335 | [0.5320, 1.7742] | +1.1211 | [+0.4514, +1.7742] | yes | 4/6 | 0.5046 |
| 5 | `state_left_contact_t -> action_do_nothing_t+1` | 1.0396 | [0.7113, 1.3857] | +1.0393 | [+0.7113, +1.3857] | yes | 7/7 | 0.7046 |
| 6 | `action_do_nothing_t -> action_do_nothing_t+1` | 1.0057 | [0.6571, 1.5768] | +0.9055 | [+0.3196, +1.5768] | yes | 3/7 | 0.9413 |
| 7 | `state_left_contact_t -> action_fire_main_engine_t+1` | 0.9790 | [0.6773, 1.3426] | -0.9737 | [-1.3426, -0.6563] | yes | 5/8 | 1.0422 |
| 8 | `state_y_t -> state_left_contact_t+1` | 0.8888 | [0.5550, 1.3254] | -0.8405 | [-1.3196, -0.4960] | yes | 4/7 | 1.0501 |
| 9 | `action_fire_main_engine_t -> action_fire_main_engine_t+1` | 0.7901 | [0.5603, 1.0045] | +0.0109 | [-0.6204, +0.6607] | no | 2/8 | 0.1326 |
| 10 | `action_fire_main_engine_t -> action_do_nothing_t+1` | 0.7863 | [0.4020, 1.3261] | +0.6681 | [+0.0386, +1.3076] | yes | 2/7 | 0.3159 |
| 11 | `state_vy_t -> action_fire_main_engine_t+1` | 0.7384 | [0.5988, 0.9167] | -0.4896 | [-0.7008, -0.2518] | yes | 3/8 | 0.7569 |
| 12 | `state_left_contact_t -> state_right_contact_t+1` | 0.6437 | [0.2603, 1.0378] | +0.6417 | [+0.2487, +1.0378] | yes | 7/7 | 0.2075 |
| 13 | `state_y_t -> state_right_contact_t+1` | 0.6427 | [0.3422, 1.0360] | -0.5995 | [-0.9920, -0.2638] | yes | 3/7 | 0.5435 |
| 14 | `state_vy_t -> action_do_nothing_t+1` | 0.6420 | [0.5147, 0.8112] | +0.2766 | [+0.0460, +0.5246] | no | 2/8 | 0.3231 |
| 15 | `action_fire_main_engine_t -> action_fire_right_engine_t+1` | 0.5965 | [0.2675, 1.2218] | +0.5257 | [-0.0905, +1.2218] | no | 2/6 | 0.1876 |
| 16 | `state_right_contact_t -> action_do_nothing_t+1` | 0.5702 | [0.2647, 0.8630] | +0.5665 | [+0.2277, +0.8630] | no | 8/8 | 0.1332 |
| 17 | `state_angular_velocity_t -> action_fire_right_engine_t+1` | 0.5274 | [0.3787, 0.6739] | +0.5232 | [+0.3761, +0.6695] | no | 4/7 | 0.1682 |
| 18 | `state_angular_velocity_t -> action_fire_left_engine_t+1` | 0.5105 | [0.3736, 0.6686] | -0.4984 | [-0.6592, -0.3637] | yes | 3/7 | 0.3811 |
| 19 | `state_right_contact_t -> state_left_contact_t+1` | 0.5097 | [0.2073, 0.8647] | +0.5071 | [+0.1821, +0.8647] | yes | 7/7 | 0.1141 |
| 20 | `state_right_contact_t -> action_fire_main_engine_t+1` | 0.4852 | [0.2290, 0.7938] | -0.4653 | [-0.7938, -0.1257] | yes | 3/8 | 0.2504 |
| 21 | `state_angle_t -> action_fire_right_engine_t+1` | 0.4722 | [0.3163, 0.6746] | +0.4295 | [+0.2372, +0.6348] | no | 2/7 | 0.1834 |
| 22 | `state_y_t -> state_y_t+1` | 0.4119 | [0.3922, 0.4297] | +0.4118 | [+0.3922, +0.4295] | yes | 5/6 | 0.4106 |
| 23 | `state_vx_t -> action_fire_left_engine_t+1` | 0.3772 | [0.2672, 0.5293] | +0.2848 | [+0.1551, +0.4106] | no | 1/6 | 0.1064 |
| 24 | `state_vx_t -> action_fire_right_engine_t+1` | 0.3600 | [0.2388, 0.5009] | -0.2820 | [-0.4145, -0.1517] | no | 1/7 | 0.0709 |
| 25 | `state_right_contact_t -> action_fire_left_engine_t+1` | 0.3541 | [0.1234, 0.6972] | -0.3016 | [-0.6972, +0.1809] | no | 1/7 | 0.0099 |
| 26 | `state_vx_t -> state_vx_t+1` | 0.3394 | [0.3363, 0.3422] | +0.3389 | [+0.3346, +0.3422] | yes | 7/7 | 0.3241 |
| 27 | `state_vy_t -> state_vy_t+1` | 0.3043 | [0.2887, 0.3269] | +0.3043 | [+0.2887, +0.3269] | yes | 7/7 | 0.2926 |
| 28 | `state_vy_t -> action_fire_left_engine_t+1` | 0.3034 | [0.1693, 0.5001] | +0.1981 | [-0.0044, +0.4324] | no | 1/8 | 0.0125 |
| 29 | `state_x_t -> action_fire_main_engine_t+1` | 0.2597 | [0.1659, 0.3636] | +0.1046 | [-0.0202, +0.2229] | no | 1/7 | 0.0233 |
| 30 | `state_x_t -> state_x_t+1` | 0.2551 | [0.2515, 0.2589] | +0.2304 | [+0.2268, +0.2348] | yes | 6/7 | 0.2338 |
| 31 | `state_angle_t -> state_angle_t+1` | 0.1883 | [0.1855, 0.1913] | +0.1883 | [+0.1855, +0.1913] | yes | 5/5 | 0.1765 |
| 32 | `state_angular_velocity_t -> state_angular_velocity_t+1` | 0.1500 | [0.1440, 0.1562] | +0.1494 | [+0.1435, +0.1555] | yes | 5/5 | 0.1189 |
| 33 | `state_right_contact_t -> state_angular_velocity_t+1` | 0.0956 | [0.0726, 0.1228] | -0.0752 | [-0.1139, -0.0162] | no | 2/7 | 0.0142 |
| 34 | `state_right_contact_t -> state_vy_t+1` | 0.0231 | [0.0162, 0.0316] | +0.0179 | [+0.0022, +0.0293] | no | 2/8 | 0.0106 |
| 35 | `state_left_contact_t -> state_vx_t+1` | 0.0184 | [0.0078, 0.0467] | -0.0136 | [-0.0440, -0.0011] | no | 1/8 | 0.0027 |
| 36 | `state_right_contact_t -> state_vx_t+1` | 0.0105 | [0.0036, 0.0190] | -0.0008 | [-0.0082, +0.0056] | no | 1/6 | 0.0003 |
| 37 | `state_y_t -> state_vy_t+1` | 0.0101 | [0.0057, 0.0170] | -0.0080 | [-0.0150, -0.0034] | no | 1/7 | 0.0013 |
| 38 | `state_angular_velocity_t -> state_angle_t+1` | 0.0067 | [0.0060, 0.0076] | +0.0066 | [+0.0059, +0.0075] | yes | 5/7 | 0.0039 |
| 39 | `state_vx_t -> state_x_t+1` | 0.0047 | [0.0031, 0.0075] | +0.0027 | [-0.0010, +0.0050] | yes | 2/7 | 0.0013 |

## Fragile vs stable 95% edges

- **23 edges survive at 99%** (stable under a stricter threshold).
- **16 edges are lost at 99%** (fragile — their pooled cluster CIs touch/cover 0 at 99%):
  - `action_fire_main_engine_t -> action_fire_main_engine_t+1` (mean |ITE| 0.7901, sig. clusters 2, margin95 0.1326)
  - `state_vy_t -> action_do_nothing_t+1` (mean |ITE| 0.6420, sig. clusters 2, margin95 0.3231)
  - `action_fire_main_engine_t -> action_fire_right_engine_t+1` (mean |ITE| 0.5965, sig. clusters 2, margin95 0.1876)
  - `state_right_contact_t -> action_do_nothing_t+1` (mean |ITE| 0.5702, sig. clusters 8, margin95 0.1332)
  - `state_angular_velocity_t -> action_fire_right_engine_t+1` (mean |ITE| 0.5274, sig. clusters 4, margin95 0.1682)
  - `state_angle_t -> action_fire_right_engine_t+1` (mean |ITE| 0.4722, sig. clusters 2, margin95 0.1834)
  - `state_vx_t -> action_fire_left_engine_t+1` (mean |ITE| 0.3772, sig. clusters 1, margin95 0.1064)
  - `state_vx_t -> action_fire_right_engine_t+1` (mean |ITE| 0.3600, sig. clusters 1, margin95 0.0709)
  - `state_right_contact_t -> action_fire_left_engine_t+1` (mean |ITE| 0.3541, sig. clusters 1, margin95 0.0099)
  - `state_vy_t -> action_fire_left_engine_t+1` (mean |ITE| 0.3034, sig. clusters 1, margin95 0.0125)
  - `state_x_t -> action_fire_main_engine_t+1` (mean |ITE| 0.2597, sig. clusters 1, margin95 0.0233)
  - `state_right_contact_t -> state_angular_velocity_t+1` (mean |ITE| 0.0956, sig. clusters 2, margin95 0.0142)
  - `state_right_contact_t -> state_vy_t+1` (mean |ITE| 0.0231, sig. clusters 2, margin95 0.0106)
  - `state_left_contact_t -> state_vx_t+1` (mean |ITE| 0.0184, sig. clusters 1, margin95 0.0027)
  - `state_right_contact_t -> state_vx_t+1` (mean |ITE| 0.0105, sig. clusters 1, margin95 0.0003)
  - `state_y_t -> state_vy_t+1` (mean |ITE| 0.0101, sig. clusters 1, margin95 0.0013)
- Smallest margins among 95% edges (closest to 0 within their own target scale — see caveat): `state_right_contact_t -> state_vx_t+1` (0.0003); `state_y_t -> state_vy_t+1` (0.0013); `state_vx_t -> state_x_t+1` (0.0013); `state_left_contact_t -> state_vx_t+1` (0.0027); `state_angular_velocity_t -> state_angle_t+1` (0.0039).

## Borderline non-edges (present at 90% only, 16)

These pairs are significant at 90% but not at 95% — the closest calls on the non-edge side:
- `action_fire_main_engine_t -> action_fire_left_engine_t+1`
- `action_fire_right_engine_t -> state_angular_velocity_t+1`
- `action_fire_right_engine_t -> state_right_contact_t+1`
- `state_angle_t -> action_do_nothing_t+1`
- `state_angle_t -> action_fire_left_engine_t+1`
- `state_angle_t -> state_right_contact_t+1`
- `state_angle_t -> state_vx_t+1`
- `state_left_contact_t -> action_fire_left_engine_t+1`
- `state_left_contact_t -> state_angle_t+1`
- `state_left_contact_t -> state_angular_velocity_t+1`
- `state_right_contact_t -> state_angle_t+1`
- `state_right_contact_t -> state_y_t+1`
- `state_vy_t -> action_fire_right_engine_t+1`
- `state_vy_t -> state_left_contact_t+1`
- `state_vy_t -> state_right_contact_t+1`
- `state_x_t -> action_do_nothing_t+1`

## Weight is not significance

- **Strongest non-edges at 95%** (large mean |ITE| but no cluster CI excludes 0 at 95%):
  - `state_y_t -> action_fire_main_engine_t+1`: mean |ITE| 0.6574, CI95 [0.5005, 0.8644], present at 90%: no
  - `action_fire_right_engine_t -> state_right_contact_t+1`: mean |ITE| 0.5858, CI95 [0.1115, 1.2534], present at 90%: yes
  - `action_do_nothing_t -> action_fire_main_engine_t+1`: mean |ITE| 0.5320, CI95 [0.3579, 0.8124], present at 90%: no
  - `state_y_t -> action_do_nothing_t+1`: mean |ITE| 0.5068, CI95 [0.3516, 0.6976], present at 90%: no
  - `action_fire_right_engine_t -> action_fire_left_engine_t+1`: mean |ITE| 0.4910, CI95 [0.0860, 1.4057], present at 90%: no
  - `action_fire_main_engine_t -> action_fire_left_engine_t+1`: mean |ITE| 0.4867, CI95 [0.1973, 1.1246], present at 90%: yes
  - `action_do_nothing_t -> action_fire_left_engine_t+1`: mean |ITE| 0.4411, CI95 [0.1332, 0.8583], present at 90%: no
  - `action_fire_main_engine_t -> state_left_contact_t+1`: mean |ITE| 0.4353, CI95 [0.0964, 1.0042], present at 90%: no
  - `action_fire_right_engine_t -> action_fire_main_engine_t+1`: mean |ITE| 0.4330, CI95 [0.0982, 1.1855], present at 90%: no
  - `action_fire_right_engine_t -> action_do_nothing_t+1`: mean |ITE| 0.4264, CI95 [0.1353, 1.0163], present at 90%: no
- **Heterogeneous sign:** `action_fire_main_engine_t -> action_fire_main_engine_t+1` has mean |ITE| 0.7901 but signed mean ITE +0.0109 (47.2% of cells > 0) — the effect flips sign across points/clusters; significant at 95% via 2 clusters, lost at 99%.
- **Small but stable:** `state_angular_velocity_t -> state_angle_t+1` (mean |ITE| 0.0067) and `state_vx_t -> state_x_t+1` (0.0047) survive at 99% — tiny, consistent effects with tight CIs.
- **Large extrapolated values:** the contact self-loops (`state_right_contact_t -> state_right_contact_t+1`, mean |ITE| 3.5805, 100% of cells > 0; `state_left_contact_t -> state_left_contact_t+1`, 3.2646) exceed the 0/1 range of the binary target — a BART regression-extrapolation artefact on counterfactual inputs, reported as computed, not a probability change (see caveat).

## CartPole edge weights — NOT DONE (no saved draws exist)

Per the task instruction, CartPole weights were to be computed ONLY from already-saved 500-draw ITE matrices, with no new BART run for CartPole. No such draws were found:
- **Repo:** `muc4/results/ablation-bart/per_run_details.json` stores only per-pair cluster significance at 90/95/99% (no ITE magnitudes, no draws); `run_ablation_bart.py` never wrote draws to disk. No `.npz` ITE-draw file exists anywhere in the repo (the only `.npz` files are Muc 3 replays and LunarLander fidelity masks/scores).
- **Drive:** name searches for `ite` and `draws` returned nothing; the Drive `ablation-bart` folder contains only `graphs/`; every `.npz` on Drive is MicroRTS batches or the LunarLander fidelity-predictions file — none is CartPole ITE draws. (Searches run 2026-11-15; see run log.)
- **Consequence:** CartPole edge weights cannot be computed without a new BART refit that saves draws, which the task forbids here. If needed later, re-run the CartPole ablation with draw saving added (same pattern as `run_lunarlander_ci_sensitivity.py`), and compute weights from the saved draws with the phase-2 pattern in `recompute_lunarlander_from_draws.py`.

## Reproducibility check

- Phase-2 recompute from the SAVED `.npz` draws alone (pure numpy, no R, script `recompute_lunarlander_from_draws.py`): decisions at 90/95/99% for all 144 pairs reproduced exactly — **0 mismatches / 432 checks**. Edge counts from saved draws: 90% = 55, 95% = 39, 99% = 23.

