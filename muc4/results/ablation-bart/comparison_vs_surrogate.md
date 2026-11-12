# REAL BART vs surrogate (RandomForest) — same configs

> BART: **BART (R/dbarts via rpy2) - real CausalXRL estimator**. Surrogate: RandomForest, 200 trees (`results/ablation/`, seed 0, same data-collection code and same fidelity protocol). Jaccard is over DBN edge sets (t -> t+1) for the same (method, episodes).

| config | BART n_edges | surrogate n_edges | Jaccard | symdiff | BART push parents | surrogate push parents |
|---|---:|---:|---:|---:|---|---|
| ate_50 | 15 | 12 | 0.5882 | 7 | `action_push_t; state_cart_position_t; state_pole_angle_t` | `state_cart_position_t` |
| ate_100 | 15 | 14 | 0.5263 | 9 | `action_push_t; state_cart_position_t; state_pole_angle_t` | `state_cart_position_t; state_pole_angle_t` |
| ate_200 | 15 | 14 | 0.5263 | 9 | `action_push_t; state_cart_position_t; state_pole_angle_t; state_pole_velocity_t` | `state_cart_position_t; state_pole_velocity_t` |
| hte_50 | 12 | 5 | 0.3077 | 9 | `action_push_t; state_cart_position_t; state_pole_angle_t` | `(none)` |
| hte_100 | 13 | 7 | 0.5385 | 6 | `action_push_t; state_cart_position_t; state_pole_angle_t; state_pole_velocity_t` | `(none)` |
| hte_200 | 13 | 7 | 0.5385 | 6 | `action_push_t; state_cart_position_t; state_pole_angle_t` | `(none)` |

## Edge differences per config

- **ate_50**: only in BART: `action_push_t -> action_push_t+1; state_pole_angle_t -> action_push_t+1; state_pole_angle_t -> state_cart_position_t+1; state_pole_velocity_t -> state_cart_velocity_t+1; state_pole_velocity_t -> state_pole_angle_t+1`; only in surrogate: `state_cart_velocity_t -> state_pole_velocity_t+1; state_pole_velocity_t -> state_cart_position_t+1`
- **ate_100**: only in BART: `action_push_t -> action_push_t+1; state_cart_position_t -> state_cart_position_t+1; state_cart_velocity_t -> state_pole_angle_t+1; state_pole_angle_t -> state_cart_position_t+1; state_pole_velocity_t -> state_cart_velocity_t+1`; only in surrogate: `state_cart_position_t -> state_pole_velocity_t+1; state_cart_velocity_t -> state_pole_velocity_t+1; state_pole_angle_t -> state_pole_velocity_t+1; state_pole_velocity_t -> state_cart_position_t+1`
- **ate_200**: only in BART: `action_push_t -> action_push_t+1; state_cart_velocity_t -> state_cart_position_t+1; state_pole_angle_t -> action_push_t+1; state_pole_angle_t -> state_cart_position_t+1; state_pole_velocity_t -> state_cart_velocity_t+1`; only in surrogate: `state_cart_position_t -> state_cart_velocity_t+1; state_cart_velocity_t -> state_pole_velocity_t+1; state_pole_velocity_t -> state_cart_position_t+1; state_pole_velocity_t -> state_pole_angle_t+1`
- **hte_50**: only in BART: `action_push_t -> action_push_t+1; action_push_t -> state_cart_velocity_t+1; action_push_t -> state_pole_velocity_t+1; state_cart_position_t -> action_push_t+1; state_cart_velocity_t -> state_cart_position_t+1; state_cart_velocity_t -> state_pole_angle_t+1; state_pole_angle_t -> action_push_t+1; state_pole_velocity_t -> state_pole_angle_t+1`; only in surrogate: `state_cart_velocity_t -> state_pole_velocity_t+1`
- **hte_100**: only in BART: `action_push_t -> action_push_t+1; state_cart_position_t -> action_push_t+1; state_cart_velocity_t -> state_cart_position_t+1; state_pole_angle_t -> action_push_t+1; state_pole_velocity_t -> action_push_t+1; state_pole_velocity_t -> state_pole_angle_t+1`; only in surrogate: `—`
- **hte_200**: only in BART: `action_push_t -> action_push_t+1; state_cart_position_t -> action_push_t+1; state_cart_velocity_t -> state_cart_position_t+1; state_pole_angle_t -> action_push_t+1; state_pole_angle_t -> state_pole_velocity_t+1; state_pole_velocity_t -> state_pole_angle_t+1`; only in surrogate: `—`

