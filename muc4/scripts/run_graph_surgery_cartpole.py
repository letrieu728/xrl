#!/usr/bin/env python3
"""Graph-surgery simulation on CartPole (ONE step only, narrowed scope).

Pipeline (see PREREGISTRATION.md in the output directory — criteria were
fixed BEFORE this script was first run):

1. Load the Muc 3 replay (seed 99, 200 episodes, 6,115 transitions) and
   rebuild within-episode t -> t+1 pairs (5,915; cross-episode pairs
   excluded on purpose — see preregistration).
2. Episode-grouped train/test split (160/40 episodes).
3. Fit per-node RandomForest surrogates using ONLY the BART graph
   parents, plus an all-features baseline (same family, same seeds).
   Report held-out FIDELITY before interpreting any intervention.
4. Sample 80 intervention queries from TEST rows only (40 state, 40
   action; near-fall states |theta| >= 0.15 included by stratification).
5. Verify every query against the REAL gymnasium CartPole env in ONE
   step, with the authors' heuristic as the real policy.
6. Evaluate the pre-registered criteria; write per-query JSONs.

MicroRTS: NOT run here (see graph_surgery.py). DQN: NOT part of the
criteria; an optional separate diagnostic script
(``dqn_agreement_diagnostic.py``) compares DQN vs heuristic agreement
if a checkpoint is available locally. It is never committed.

Cost: $0 — local CPU only. No Modal, no paid API.

Usage (from a repo checkout)::

    python muc4/scripts/run_graph_surgery_cartpole.py \
        --replay muc3/results/cartpole/replay.npz \
        --out-dir muc4/results/graph-surgery-simulation

``--replay`` defaults to the repo-relative path above; the Drive copy of
``replay.npz`` must be placed there first (binary files are not in git).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_surgery import (  # noqa: E402
    AllFeaturesSurrogate,
    CausalGraph,
    PerNodeSurrogate,
    dbn_from_cartpole_json,
    load_columns_for_dbn,
)

# --- Pre-registered constants (PREREGISTRATION.md; do not tune) --------
SPLIT_SEED = 123
MODEL_SEED = 99
QUERY_SEED = 20261115
N_TREES = 300
NEAR_THRESHOLD_THETA = 0.15  # rad; env failure at |theta| > 0.20944
MIN_DELTA = {"cart_position": 0.05, "pole_angle": 0.05,
             "cart_velocity": 0.5, "pole_velocity": 0.5}
INTERVENTION_VALUES = {
    "cart_position": [-0.15, -0.08, 0.0, 0.08, 0.15],
    "cart_velocity": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "pole_angle": [-0.19, -0.10, 0.0, 0.10, 0.19],
    "pole_velocity": [-2.4, -1.2, 0.0, 1.2, 2.4],
}
CRITERIA = {
    "F1_push_accuracy_min": 0.85,
    "F2_regressor_r2_min": 0.85,
    "C_A1_graph_match_min": 0.80,
    "C_A2_noninferiority_margin": 0.05,
    "C_A3_baseline_flip_min": 0.10,
    "C_B1_mae_ratio_max": 1.10,
    "C_B3_mae_ratio_max": 1.10,
}
STATE_NAMES = ["cart_position", "cart_velocity", "pole_angle", "pole_velocity"]
SHORT = {"cart_position": "x", "cart_velocity": "v",
         "pole_angle": "theta", "pole_velocity": "w", "push": "push"}


def heuristic_policy(state) -> int:
    """Authors' hand-written CartPole heuristic (exact repo formula)."""
    state = np.asarray(state, dtype=np.float64)
    return 1 if 0.4 * (state[0] / 2.4) + 0.6 * (state[2] / 2.095) >= 0 else 0


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reconstruct_episodes(states, next_states):
    """Split replay rows into episodes: a boundary is where the next
    state of row i is not the state of row i+1 (terminal transitions
    were excluded at collection time, so this is exact)."""
    linked = np.all(next_states[:-1] == states[1:], axis=1)
    episodes, start = [], 0
    for i, ok in enumerate(linked):
        if not ok:
            episodes.append(np.arange(start, i + 1))
            start = i + 1
    episodes.append(np.arange(start, len(states)))
    return episodes


def model_factory(node: str):
    from sklearn.ensemble import (RandomForestClassifier,
                                  RandomForestRegressor)
    # n_jobs=2: training parallelism only; fitted trees are identical
    # for any n_jobs given the same random_state (per-tree seeds are
    # drawn up front). After fitting, predict-time n_jobs is set to 1
    # (see fit_and_score_fidelity) to avoid per-call dispatch overhead.
    if node == "push_next":
        return RandomForestClassifier(
            n_estimators=N_TREES, random_state=MODEL_SEED, n_jobs=2)
    return RandomForestRegressor(
        n_estimators=N_TREES, random_state=MODEL_SEED, n_jobs=2)


def _set_predict_jobs_1(model):
    for est in model.models.values():
        est.n_jobs = 1


def fit_and_score_fidelity(graph, train_cols, test_cols, test_pairs):
    """Fit graph + baseline surrogates; score held-out fidelity."""
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
    from sklearn.metrics import mean_squared_error, r2_score

    graph_model = PerNodeSurrogate(graph, model_factory,
                                   classifiers={"push_next"}).fit(train_cols)
    print("fitted graph surrogate", flush=True)
    base_model = AllFeaturesSurrogate(graph, model_factory,
                                      classifiers={"push_next"}).fit(train_cols)
    print("fitted baseline surrogate", flush=True)
    # Predict-time n_jobs=1: identical predictions, no per-call
    # joblib dispatch overhead (the per-row loop with n_jobs=-1 was
    # pathologically slow; batch prediction below is exact).
    _set_predict_jobs_1(graph_model)
    _set_predict_jobs_1(base_model)

    root_cols = {name: np.asarray(test_cols[name], dtype=float)
                 for name in graph.roots}
    pred_g = graph_model.predict_targets_batch(root_cols)
    pred_b = base_model.predict_targets_batch(root_cols)
    print("scored held-out batch", flush=True)

    fidelity = {}
    for target in graph.targets:
        y = np.asarray(test_cols[target], dtype=float)
        entry = {"n_test": int(len(y))}
        if target == "push_next":
            for label, pred in (("graph", pred_g[target]),
                                ("baseline", pred_b[target])):
                entry[label] = {
                    "accuracy": float(accuracy_score(y, pred)),
                    "f1": float(f1_score(y, pred, zero_division=0)),
                }
            persist = np.asarray(test_cols["push"], dtype=float)
            entry["persistence"] = {
                "accuracy": float(accuracy_score(y, persist)),
                "f1": float(f1_score(y, persist, zero_division=0)),
            }
        else:
            root = target[: -len("_next")]
            for label, pred in (("graph", pred_g[target]),
                                ("baseline", pred_b[target])):
                entry[label] = {
                    "mae": float(mean_absolute_error(y, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                    "r2": float(r2_score(y, pred)),
                }
            persist = np.asarray(test_cols[root], dtype=float)
            entry["persistence"] = {
                "mae": float(mean_absolute_error(y, persist)),
                "rmse": float(np.sqrt(mean_squared_error(y, persist))),
                "r2": float(r2_score(y, persist)),
            }
        fidelity[target] = entry
    return graph_model, base_model, fidelity


def main() -> int:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser.add_argument("--replay", type=Path,
                        default=repo_root / "muc3" / "results" / "cartpole"
                        / "replay.npz")
    parser.add_argument("--graph", type=Path,
                        default=repo_root / "muc4" / "results"
                        / "graph-surgery-simulation"
                        / "cartpole_graph_bart_muc3.json")
    parser.add_argument("--out-dir", type=Path,
                        default=repo_root / "muc4" / "results"
                        / "graph-surgery-simulation")
    args = parser.parse_args()
    t_start = time.time()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- sanity: heuristic formula smoke checks (from cartpole_policies.py)
    assert heuristic_policy(np.zeros(4)) == 1
    assert heuristic_policy(np.array([2.4, 0.0, 0.0, 0.0])) == 1
    assert heuristic_policy(np.array([-2.4, 0.0, 0.0, 0.0])) == 0
    assert heuristic_policy(np.array([0.0, 0.0, 2.095, 0.0])) == 1
    assert heuristic_policy(np.array([0.0, 0.0, -2.095, 0.0])) == 0

    import gymnasium
    import sklearn

    replay = np.load(args.replay)
    states = np.asarray(replay["states"], dtype=float)
    actions = np.asarray(replay["actions"], dtype=float)
    next_states = np.asarray(replay["next_states"], dtype=float)
    replay_sha = sha256_of(args.replay)
    print(f"replay: {args.replay} ({len(states)} transitions, "
          f"sha256={replay_sha[:16]}...)")

    # Observed domain (marginals) — intervention values must stay inside.
    domain = {name: {"min": float(states[:, j].min()),
                     "max": float(states[:, j].max())}
              for j, name in enumerate(STATE_NAMES)}
    for name, values in INTERVENTION_VALUES.items():
        assert all(domain[name]["min"] <= v <= domain[name]["max"]
                   for v in values), f"intervention value outside domain: {name}"

    # Heuristic must reproduce the logged actions exactly (collection check).
    logged = np.array([heuristic_policy(s) for s in states], dtype=float)
    assert float((logged == actions).mean()) == 1.0, \
        "heuristic does not reproduce replay actions"

    graph = dbn_from_cartpole_json(args.graph)
    print(f"graph: {len(graph.edges())} edges (targets: {graph.targets})")

    episodes = reconstruct_episodes(states, next_states)
    print(f"episodes reconstructed: {len(episodes)} "
          f"(sizes {min(len(e) for e in episodes)}..{max(len(e) for e in episodes)})")
    assert len(episodes) == 200

    # Within-episode pairs only: base row i with successor i+1 in same episode.
    pair_rows = np.concatenate(
        [ep[:-1] for ep in episodes if len(ep) > 1])
    ep_of_row = {}
    for ep_id, ep in enumerate(episodes):
        for r in ep:
            ep_of_row[int(r)] = ep_id

    rng_split = np.random.default_rng(SPLIT_SEED)
    ep_ids = rng_split.permutation(len(episodes))
    test_ep = set(int(e) for e in ep_ids[:40])
    train_pairs = np.array([r for r in pair_rows if ep_of_row[int(r)] not in test_ep])
    test_pairs = np.array([r for r in pair_rows if ep_of_row[int(r)] in test_ep])
    print(f"pairs: {len(pair_rows)} within-episode "
          f"(train {len(train_pairs)}, test {len(test_pairs)})")

    def columns_for(rows):
        return load_columns_for_dbn(
            states[rows], actions[rows], next_states[rows],
            actions[rows + 1], STATE_NAMES)

    train_cols = columns_for(train_pairs)
    test_cols = columns_for(test_pairs)
    graph_model, base_model, fidelity = fit_and_score_fidelity(
        graph, train_cols, test_cols, test_pairs)
    print("fidelity (held-out episodes):", flush=True)
    print(json.dumps(fidelity, indent=1), flush=True)
    # Checkpoint fidelity immediately so it survives any later failure.
    (out_dir / "fidelity.json").write_text(
        json.dumps(fidelity, indent=2), encoding="utf-8")

    # --- real-env sanity: 1 env step must reproduce replay next_states.
    env = gymnasium.make("CartPole-v1")
    env.reset(seed=0)

    def step_from(state, action):
        # Direct state assignment (no reset: reset would randomize the
        # state). Also clear the terminated bookkeeping so a previous
        # query's terminal step cannot trigger the "step after
        # terminated" warning; the physics below depends only on the
        # assigned state, and the returned flag is recomputed from the
        # new state, so this changes no number (verified: rerun with
        # this line gives byte-identical query JSONs).
        env.unwrapped.state = np.array(state, dtype=np.float64)
        env.unwrapped.steps_beyond_terminated = None
        ns, _, term, trunc, _ = env.step(int(action))
        return np.asarray(ns, dtype=float), bool(term or trunc)

    rng_sanity = np.random.default_rng(0)
    sanity_rows = rng_sanity.choice(pair_rows, size=100, replace=False)
    sanity_diffs = np.array([
        np.abs(step_from(states[r], actions[r])[0] - next_states[r])
        for r in sanity_rows])
    sanity = {"n": 100,
              "max_abs_diff_per_coord": sanity_diffs.max(axis=0).tolist(),
              "mean_abs_diff_per_coord": sanity_diffs.mean(axis=0).tolist()}
    print(f"env sanity (CartPole-v1 vs replay next_states): {sanity}")

    # --- sample queries from TEST rows only.
    rng_q = np.random.default_rng(QUERY_SEED)
    near = np.array([abs(states[r][2]) >= NEAR_THRESHOLD_THETA
                     for r in test_pairs])
    test_near = test_pairs[near]
    print(f"test rows: {len(test_pairs)}, near-threshold: {len(test_near)}")

    def pick_value(feature, current):
        values = list(INTERVENTION_VALUES[feature])
        rng_q.shuffle(values)
        for v in values:
            if abs(v - current) >= MIN_DELTA[feature]:
                return float(v)
        return None

    def sample_row(pool, used):
        for _ in range(50):
            r = int(rng_q.choice(pool))
            if r not in used:
                return r
        return int(rng_q.choice(pool))

    queries_a, used_a = [], set()
    qid = 0
    for feature in STATE_NAMES:
        j = STATE_NAMES.index(feature)
        for stratum, pool, count in (("random", test_pairs, 5),
                                     ("near_threshold", test_near, 5)):
            for _ in range(count):
                for _attempt in range(50):
                    r = sample_row(pool, used_a)
                    value = pick_value(feature, float(states[r][j]))
                    if value is not None:
                        break
                else:
                    raise RuntimeError("could not sample a valid query")
                used_a.add(r)
                qid += 1
                queries_a.append({
                    "query_id": f"A{qid:03d}", "type": "state",
                    "replay_row": r, "episode": ep_of_row[r],
                    "stratum": stratum,
                    "near_threshold": bool(abs(states[r][2]) >= NEAR_THRESHOLD_THETA),
                    "intervened_feature": feature,
                    "intervened_value": value,
                    "original_value": float(states[r][j]),
                })
    queries_b, used_b = [], set()
    for stratum, pool, count in (("random", test_pairs, 20),
                                 ("near_threshold", test_near, 20)):
        for _ in range(count):
            r = sample_row(pool, used_b)
            used_b.add(r)
            qid += 1
            queries_b.append({
                "query_id": f"B{qid:03d}", "type": "action",
                "replay_row": r, "episode": ep_of_row[r],
                "stratum": stratum,
                "near_threshold": bool(abs(states[r][2]) >= NEAR_THRESHOLD_THETA),
                "intervened_action": int(1 - actions[r]),
                "original_action": int(actions[r]),
            })
    print(f"queries: {len(queries_a)} state + {len(queries_b)} action = "
          f"{len(queries_a) + len(queries_b)}")

    # --- run queries with real-env verification (1 env step each + 1
    # unintervened reference step each).
    def run_query(q):
        r = q["replay_row"]
        s_t = states[r]
        a_t = int(actions[r])
        values = {name: float(s_t[j]) for j, name in enumerate(STATE_NAMES)}
        values["push"] = float(a_t)
        if q["type"] == "state":
            interventions = {q["intervened_feature"]: q["intervened_value"]}
            s_do = s_t.copy()
            s_do[STATE_NAMES.index(q["intervened_feature"])] = q["intervened_value"]
            env_state, env_action = s_do, a_t
        else:
            interventions = {"push": float(q["intervened_action"])}
            env_state, env_action = s_t, q["intervened_action"]
        pred_g_do = graph_model.predict_targets(values, interventions)
        pred_b_do = base_model.predict_targets(values, interventions)
        pred_g_ref = graph_model.predict_targets(values)
        pred_b_ref = base_model.predict_targets(values)
        s_real, terminated = step_from(env_state, env_action)
        s_real_ref, _ = step_from(s_t, a_t)
        a_real = heuristic_policy(s_real)
        a_real_ref = heuristic_policy(s_real_ref)
        out = dict(q)
        out.update({
            "base_state": [float(v) for v in s_t],
            "base_action": a_t,
            "graph_pred_next_state": [float(pred_g_do[f"{n}_next"]) for n in STATE_NAMES],
            "graph_pred_push_next": int(pred_g_do["push_next"]),
            "baseline_pred_next_state": [float(pred_b_do[f"{n}_next"]) for n in STATE_NAMES],
            "baseline_pred_push_next": int(pred_b_do["push_next"]),
            "graph_pred_next_state_unintervened": [float(pred_g_ref[f"{n}_next"]) for n in STATE_NAMES],
            "graph_pred_push_next_unintervened": int(pred_g_ref["push_next"]),
            "baseline_pred_next_state_unintervened": [float(pred_b_ref[f"{n}_next"]) for n in STATE_NAMES],
            "baseline_pred_push_next_unintervened": int(pred_b_ref["push_next"]),
            "real_next_state": [float(v) for v in s_real],
            "real_next_state_unintervened": [float(v) for v in s_real_ref],
            "real_next_action": a_real,
            "real_next_action_unintervened": a_real_ref,
            "terminated_after_step": terminated,
        })
        if q["type"] == "state":
            out["policy_action_at_intervened_state"] = heuristic_policy(s_do)
            out["policy_action_at_intervened_state_flipped"] = bool(
                heuristic_policy(s_do) != a_t)
        out["graph_push_match"] = bool(out["graph_pred_push_next"] == a_real)
        out["baseline_push_match"] = bool(out["baseline_pred_push_next"] == a_real)
        out["graph_push_flipped"] = bool(
            out["graph_pred_push_next"] != out["graph_pred_push_next_unintervened"])
        out["baseline_push_flipped"] = bool(
            out["baseline_pred_push_next"] != out["baseline_pred_push_next_unintervened"])
        out["real_push_flipped"] = bool(a_real != a_real_ref)
        return out

    results_a = [run_query(q) for q in queries_a]
    print("verified 40 type-A queries", flush=True)
    results_b = [run_query(q) for q in queries_b]
    print("verified 40 type-B queries", flush=True)
    env.close()

    # --- summarize + evaluate pre-registered criteria.
    def mae(rows, pred_key, real_key="real_next_state", coords=None):
        coords = coords if coords is not None else range(4)
        errs = [abs(rows[i][pred_key][j] - rows[i][real_key][j])
                for i in range(len(rows)) for j in coords]
        return float(np.mean(errs))

    f1_pass = fidelity["push_next"]["graph"]["accuracy"] >= CRITERIA["F1_push_accuracy_min"]
    r2s = {t: fidelity[t]["graph"]["r2"] for t in graph.targets if t != "push_next"}
    f2_pass = all(v >= CRITERIA["F2_regressor_r2_min"] for v in r2s.values())

    match_g_a = float(np.mean([r["graph_push_match"] for r in results_a]))
    match_b_a = float(np.mean([r["baseline_push_match"] for r in results_a]))
    vw = [r for r in results_a if r["intervened_feature"] in
          ("cart_velocity", "pole_velocity")]
    xt = [r for r in results_a if r["intervened_feature"] in
          ("cart_position", "pole_angle")]
    ca3 = {
        "n_vw_queries": len(vw),
        "graph_flip_rate_vw": float(np.mean([r["graph_push_flipped"] for r in vw])),
        "baseline_flip_rate_vw": float(np.mean([r["baseline_push_flipped"] for r in vw])),
        "graph_match_vw": float(np.mean([r["graph_push_match"] for r in vw])),
        "baseline_match_vw": float(np.mean([r["baseline_push_match"] for r in vw])),
        "real_flip_rate_vw": float(np.mean([r["real_push_flipped"] for r in vw])),
        "graph_flip_rate_xtheta": float(np.mean([r["graph_push_flipped"] for r in xt])),
        "baseline_flip_rate_xtheta": float(np.mean([r["baseline_push_flipped"] for r in xt])),
        "real_flip_rate_xtheta": float(np.mean([r["real_push_flipped"] for r in xt])),
    }
    c_a1 = match_g_a >= CRITERIA["C_A1_graph_match_min"]
    c_a2 = match_g_a >= match_b_a - CRITERIA["C_A2_noninferiority_margin"]
    c_a3 = (ca3["baseline_flip_rate_vw"] >= CRITERIA["C_A3_baseline_flip_min"]
            and ca3["graph_match_vw"] >= ca3["baseline_match_vw"]
            - CRITERIA["C_A2_noninferiority_margin"])

    mae_g_b = mae(results_b, "graph_pred_next_state")
    mae_b_b = mae(results_b, "baseline_pred_next_state")
    mae_g_xt = mae(results_b, "graph_pred_next_state", coords=[0, 2])
    mae_b_xt = mae(results_b, "baseline_pred_next_state", coords=[0, 2])
    mae_g_vw = mae(results_b, "graph_pred_next_state", coords=[1, 3])
    mae_b_vw = mae(results_b, "baseline_pred_next_state", coords=[1, 3])
    c_b1 = mae_g_b <= mae_b_b * CRITERIA["C_B1_mae_ratio_max"]
    c_b2 = mae_g_xt <= mae_b_xt
    c_b3 = mae_g_vw <= mae_b_vw * CRITERIA["C_B3_mae_ratio_max"]

    # Predicted vs real change under action intervention, per coordinate.
    delta = {}
    for j, name in enumerate(STATE_NAMES):
        real_d = np.mean([abs(r["real_next_state"][j]
                              - r["real_next_state_unintervened"][j])
                          for r in results_b])
        g_d = np.mean([abs(r["graph_pred_next_state"][j]
                           - r["graph_pred_next_state_unintervened"][j])
                       for r in results_b])
        b_d = np.mean([abs(r["baseline_pred_next_state"][j]
                           - r["baseline_pred_next_state_unintervened"][j])
                       for r in results_b])
        delta[name] = {"mean_abs_real_change": float(real_d),
                       "mean_abs_graph_predicted_change": float(g_d),
                       "mean_abs_baseline_predicted_change": float(b_d)}

    criteria_eval = {
        "F1": bool(f1_pass), "F2": bool(f2_pass),
        "C-A1": bool(c_a1), "C-A2": bool(c_a2), "C-A3": bool(c_a3),
        "C-B1": bool(c_b1), "C-B2": bool(c_b2), "C-B3": bool(c_b3),
        "overall_validated": bool(f1_pass and f2_pass and c_a1 and c_a2
                                  and c_b1 and c_b2 and c_b3),
    }
    summary = {
        "experiment": "graph-surgery-simulation/cartpole/1-step",
        "preregistration": "PREREGISTRATION.md (criteria fixed before running)",
        "replay": {"path": str(args.replay), "sha256": replay_sha,
                   "transitions": int(len(states)), "episodes": 200,
                   "seed": 99, "within_episode_pairs": int(len(pair_rows)),
                   "train_pairs": int(len(train_pairs)),
                   "test_pairs": int(len(test_pairs))},
        "graph": {"path": str(args.graph), "edges": graph.edges(),
                  "n_edges": len(graph.edges())},
        "seeds": {"split": SPLIT_SEED, "model": MODEL_SEED,
                  "query": QUERY_SEED},
        "models": {"family": "sklearn RandomForest",
                   "n_estimators": N_TREES,
                   "graph_features": "parents only",
                   "baseline_features": "all 5 (x,v,theta,w,push)"},
        "domain": domain,
        "intervention_values": INTERVENTION_VALUES,
        "env": {"name": "CartPole-v1", "gymnasium": gymnasium.__version__,
                "policy": "authors heuristic (exact repo formula)",
                "sanity_vs_replay": sanity},
        "versions": {"python": platform.python_version(),
                     "sklearn": sklearn.__version__,
                     "numpy": np.__version__},
        "fidelity": fidelity,
        "type_a": {"n": len(results_a),
                   "graph_push_match": match_g_a,
                   "baseline_push_match": match_b_a,
                   "graph_minus_baseline_pp": (match_g_a - match_b_a) * 100,
                   "near_threshold_n": int(sum(r["near_threshold"] for r in results_a)),
                   "policy_action_at_intervened_state_flip_rate": float(np.mean(
                       [r["policy_action_at_intervened_state_flipped"]
                        for r in results_a])),
                   "detail": ca3},
        "type_b": {"n": len(results_b),
                   "graph_mae": mae_g_b, "baseline_mae": mae_b_b,
                   "graph_mae_xtheta": mae_g_xt, "baseline_mae_xtheta": mae_b_xt,
                   "graph_mae_vw": mae_g_vw, "baseline_mae_vw": mae_b_vw,
                   "near_threshold_n": int(sum(r["near_threshold"] for r in results_b)),
                   "delta_per_coord": delta},
        "criteria": CRITERIA,
        "criteria_eval": criteria_eval,
        "elapsed_seconds": time.time() - t_start,
    }
    (out_dir / "fidelity.json").write_text(
        json.dumps(fidelity, indent=2), encoding="utf-8")
    (out_dir / "queries_type_a.json").write_text(
        json.dumps(results_a, indent=2), encoding="utf-8")
    (out_dir / "queries_type_b.json").write_text(
        json.dumps(results_b, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"criteria_eval": criteria_eval,
                      "type_a_match": {"graph": match_g_a, "baseline": match_b_a},
                      "type_b_mae": {"graph": mae_g_b, "baseline": mae_b_b}},
                     indent=1))
    print(f"wrote outputs to {out_dir} in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
