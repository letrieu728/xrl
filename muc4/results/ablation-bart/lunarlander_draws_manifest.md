# ITE-draw files for the LunarLander CI sensitivity run (NOT pushed as text)

These 12 per-target `.npz` files hold the 500 saved ITE posterior draws behind
`lunarlander_ci_sensitivity.md` and `edge_weights.md`. They live in the work dir and
must be uploaded to Drive by the coordinator (Drive folder `XRL - file nhi phan`,
under `muc4/results/ablation-bart/ite_draws_lunarlander/`); do NOT push them via the
GitHub connector (binary, 1.2 GB total).

Each file: `ite_draws` float64 (12 columns, 500 draws, 2215 points), `X_est`
(2215, 12), `clusters` (12, 2215), `features`, `num_cols`, `causes95`,
`target_index`, `split_seed`. Run: 2026-11-15, split seed 0, R 4.3.3 / dbarts 0.9.34.

| file | target | bytes | sha256 |
|---|---|---:|---|
| target_0.npz | x | 106748190 | `8b2c3117a6c3a90bb51a473a9343e2e6a6f9ef60463e988162d8b10608dd6c46` |
| target_1.npz | y | 106748182 | `4341e3738fed30936754901a1be6aab48aad6e78bf9d17617fd8abc22c13d64c` |
| target_2.npz | vx | 106748198 | `0dbc4c2897f569fc699fc4eed5157cee544c34e7b31104c0a5b9fdcf8959b71b` |
| target_3.npz | vy | 106748198 | `87ad6563d389d3a4395f5b84989cb3dbebc688fbea0274b3d831a257fbcb8315` |
| target_4.npz | angle | 106748190 | `5d6babfa307642aa96695ff227fc45e91445ac4c4b991780e9700b32aedc7ee5` |
| target_5.npz | angular_velocity | 106748190 | `96889544edfa34bbb7ae6b3e4be962cf9592dc353777d4cf682a1ab778b087b7` |
| target_6.npz | left_contact | 106748198 | `50523c56bfa3dd36f646e8edfe807b249a81a88b8c95d8eb9f2176b0dcc542bc` |
| target_7.npz | right_contact | 106748198 | `5f482c2a29931a665023f011a52458cf5220a8b68d230cb04e1de6f67f9e42f3` |
| target_8.npz | do_nothing | 106748214 | `3e15a3f42dae1335e913371cfe7e09f769c4aa641f04f954848834ef71ba3026` |
| target_9.npz | fire_left_engine | 106748214 | `529391e28dc59495773b5c5cdadd76b7a7b00a7efc37ca74fb5a514a005961ad` |
| target_10.npz | fire_main_engine | 106748214 | `cead7b50b3e7d4ff19ae9691a2900c773931601358ba9d9f39a44b29bfc9341b` |
| target_11.npz | fire_right_engine | 106748214 | `3f4a459dc61bdbd3135668ff56da3248266f2983d3deb135fdfc0e41069b38a9` |

Total: 1280978400 bytes (1.28 GB), 12 files.
