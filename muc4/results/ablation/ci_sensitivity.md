# CI-threshold sensitivity — 200-episode runs

> **SURROGATE ESTIMATOR (RandomForest, 200 trees), NOT BART/dbarts - stability check only, to be re-run with real BART when available**

> Edges at 90% vs 95% CI recomputed from the SAME stored per-tree draws (200 trees); no re-run.

## hte (200 episodes)

- **95% CI**: 7 edges: `action_push_t -> state_cart_velocity_t+1; action_push_t -> state_pole_velocity_t+1; state_cart_position_t -> state_cart_position_t+1; state_cart_velocity_t -> state_cart_velocity_t+1; state_cart_velocity_t -> state_pole_velocity_t+1; state_pole_angle_t -> state_pole_angle_t+1; state_pole_velocity_t -> state_pole_velocity_t+1`
- **90% CI**: 8 edges: `action_push_t -> state_cart_velocity_t+1; action_push_t -> state_pole_velocity_t+1; state_cart_position_t -> state_cart_position_t+1; state_cart_velocity_t -> state_cart_velocity_t+1; state_cart_velocity_t -> state_pole_velocity_t+1; state_pole_angle_t -> state_pole_angle_t+1; state_pole_velocity_t -> state_pole_angle_t+1; state_pole_velocity_t -> state_pole_velocity_t+1`
- Edges gained at 90% (not at 95%): `state_pole_velocity_t -> state_pole_angle_t+1`
- Edges lost at 90% (present at 95%): `—`

## ate (200 episodes)

- **95% CI**: 14 edges: `action_push_t -> state_cart_velocity_t+1; action_push_t -> state_pole_velocity_t+1; state_cart_position_t -> action_push_t+1; state_cart_position_t -> state_cart_position_t+1; state_cart_position_t -> state_cart_velocity_t+1; state_cart_velocity_t -> state_cart_velocity_t+1; state_cart_velocity_t -> state_pole_velocity_t+1; state_pole_angle_t -> state_cart_velocity_t+1; state_pole_angle_t -> state_pole_angle_t+1; state_pole_angle_t -> state_pole_velocity_t+1; state_pole_velocity_t -> action_push_t+1; state_pole_velocity_t -> state_cart_position_t+1; state_pole_velocity_t -> state_pole_angle_t+1; state_pole_velocity_t -> state_pole_velocity_t+1`
- **90% CI**: 14 edges: `action_push_t -> state_cart_velocity_t+1; action_push_t -> state_pole_velocity_t+1; state_cart_position_t -> action_push_t+1; state_cart_position_t -> state_cart_position_t+1; state_cart_position_t -> state_cart_velocity_t+1; state_cart_velocity_t -> state_cart_velocity_t+1; state_cart_velocity_t -> state_pole_velocity_t+1; state_pole_angle_t -> state_cart_velocity_t+1; state_pole_angle_t -> state_pole_angle_t+1; state_pole_angle_t -> state_pole_velocity_t+1; state_pole_velocity_t -> action_push_t+1; state_pole_velocity_t -> state_cart_position_t+1; state_pole_velocity_t -> state_pole_angle_t+1; state_pole_velocity_t -> state_pole_velocity_t+1`
- Edges gained at 90% (not at 95%): `—`
- Edges lost at 90% (present at 95%): `—`

