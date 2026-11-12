#!/usr/bin/env python3
"""CausalXRL extension A -- ablation / stability check with REAL BART (CartPole).

REAL ESTIMATOR: the repo's BARTRegressor (R dbarts via rpy2). This is the
re-run of scripts/run_ablation.py (RandomForest surrogate) with the real
estimator. Every output is labelled BART; surrogate numbers are only used
in the comparison section, clearly marked as surrogate.

What it does
------------
1. Collects CartPole-v1 rollouts with the repo's exact heuristic policy
   (CartPole.ipynb): ``1 if 0.4*(x/2.4) + 0.6*(theta/2.095) >= 0 else 0``.
   Same collection code and recording rule as run_ablation.py
   (gymnasium; a transition is recorded only when the step is NOT
   terminal). Rollouts for fewer episodes are NOT a prefix re-collection
   here either: each episode count is collected fresh with the same seed
   sequence as run_ablation.py (env seeded 0 on episode 0), so the
   transition counts must match the surrogate run (1572 / 3183 / 6330)
   and are asserted in the log (warning, not fatal, if they differ).
2. Runs the repo's causal-discovery pipeline (commit ae9096e) with the
   REAL BARTRegressor. No repo file is edited. The per-target runners
   below replicate ``infer_causes_hte`` / ``infer_causes_ate`` from
   ``cxrl/lib/causal/infer_network.py`` line-for-line for the split,
   standardisation and fit, and call the repo's own decision methods
   (``bart.is_causal_HTE`` / ``bart.ATE``) for the 95% baseline -- so the
   95% edge sets ARE the repo pipeline's output. The only reason for the
   replication (instead of calling ``infer_causes_*`` directly) is that
   the repo functions do not return the estimation split ``X_est`` or
   keep it anywhere, and CI sensitivity needs ``X_est`` + the fitted
   model to recompute decisions from the SAME ITE posterior draws
   without refitting.
   * method ``hte`` -> repo ``infer_causes_hte`` logic (70/30 split,
     StandardScaler standardisation).
   * method ``ate`` -> repo ``infer_causes_ate`` logic (50/50 split,
     per-split mean/std standardisation, including the repo quirk that
     X_est is standardised by its OWN mean/std).
3. Ablation grid: nepisodes in [50, 100, 200] x method in [hte, ate].
4. Behaviour-prediction check per run: IDENTICAL protocol to
   run_ablation.py -- DecisionTree + LogisticRegression trained on
   (state -> action), 80/20 split (random_state=seed, stratify),
   accuracy/F1. Labelled the same way; comparable across estimators.
5. CI-threshold sensitivity (Appendix D style) on the 200-episode runs,
   BOTH methods: edges at 90% / 95% / 99% CI, recomputed from the SAME
   fitted models and SAME X_est via ITE posterior draws (500 draws).
   The repo hardcodes 95% in two places and offers NO level parameter:
   ``bart_regressor.py`` R function ``ate`` (quantile probs
   c(0.025, 0.975)) and ``HTE.__init__`` (np.quantile q=[0.025, 0.975]).
   No repo file is edited. Instead:
   * ATE: per-draw ATE = row-mean of the ITE draw matrix (exactly the R
     ``ate`` math: rowMeans(ite), mean, quantile over draws), quantiles
     recomputed at 0.05/0.95, 0.025/0.975, 0.005/0.995. The recomputed
     95% decision is VALIDATED against the repo ``bart.ATE`` decision
     for every (target, column); mismatches are logged and stored.
   * HTE: the repo ``HTE`` clustering is rebuilt from the same ITE
     draws (DecisionTreeRegressor min_samples_leaf=int(N*0.1),
     max_depth=4, no random_state -- exactly the repo defaults) and the
     per-cluster pooled quantiles are recomputed at the three levels.
     The recomputed 95% decision is VALIDATED against the repo
     ``is_causal_HTE`` decision for every (target, column).
6. Cross-episode-count graph comparison per method (Jaccard vs 200 ep),
   and a direct BART-vs-surrogate comparison per config (edge counts,
   Jaccard, push parents) read from the surrogate run's
   ``per_run_details.json``.

Reproducibility: seed 0 everywhere (numpy global RNG seeded once before
each run's 5-target loop because the repo's ``train_test_split`` calls
use ``random_state=None``; BART's R seed is fixed at 99 inside the repo's
R code). Incremental save: ``partial_results.json`` is rewritten after
every run, so a VM restart loses at most the run in progress.

Memory: fitted BART models are dropped (``del _bart_machine`` + gc)
after each run, except that 200-episode models are kept until their
sensitivity recompute finishes. Heuristic data is small (<=6330 rows);
this mirrors the memory fixes in run_xrl_comparison.py.

Usage:
    .venv-ablation/bin/python -B scripts/run_ablation_bart.py \
        [--episodes 50 100 200] [--methods hte ate] [--seed 0] \
        [--output-dir results/ablation-bart] [--repo repo-causal-xrl]
"""

import argparse
import csv
import gc
import json
import logging
import os
# rpy2 needs R_HOME; default it so the script works without the caller
# exporting it (R is at /usr/lib/R on this VM). Must be set before any
# rpy2 import (which happens when the repo modules are imported).
os.environ.setdefault("R_HOME", "/usr/lib/R")
# Limit BLAS/OpenMP threads (see run_ablation.py for why).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # do not write __pycache__ into the repo copy

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
LABEL = "BART (R/dbarts via rpy2) - real CausalXRL estimator"

FEATURE_NAMES = ["cart_position", "cart_velocity", "pole_angle",
                 "pole_velocity", "push"]
N_FEATURES = len(FEATURE_NAMES)

# Expected transition counts from the surrogate run (same collection
# code + seed); checked in the log, warning only if different.
EXPECTED_TRANSITIONS = {50: 1572, 100: 3183, 200: 6330}

CI_LEVELS = (0.90, 0.95, 0.99)


def heuristic_policy(state):
    """Exact heuristic from the repo's CartPole.ipynb."""
    state = np.asarray(state, dtype=np.float64)
    return 1 if 0.4 * (state[0] / 2.4) + 0.6 * (state[2] / 2.095) >= 0 else 0


# ----------------------------------------------------------------------
# repo loading (real BART; DTYPE shim only, no repo file edited)
# ----------------------------------------------------------------------
def load_repo_infer_network(repo_dir):
    """Import the repo's infer_network with the REAL BARTRegressor.

    Compat shim (no repo file edited): the repo's bart_regressor.py uses
    sklearn.tree._tree.DTYPE, removed in sklearn >= 1.6. Old DTYPE was
    np.float32; restore the attribute so the repo code runs unchanged.
    Same shim as smoke_bart.py / run_xrl_comparison.py.
    """
    import sklearn.tree._tree as _sk_tree
    if not hasattr(_sk_tree, "DTYPE"):
        _sk_tree.DTYPE = np.float32
    sys.path.insert(0, str(repo_dir))
    import cxrl.lib.causal.infer_network as infer_network  # noqa: E402
    return infer_network


# ----------------------------------------------------------------------
# data collection (identical to run_ablation.py)
# ----------------------------------------------------------------------
def collect_cartpole(nepisodes, seed=0):
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    states, actions, rewards, states_new = [], [], [], []
    ep_lengths = []
    t0 = time.perf_counter()
    for ep in range(nepisodes):
        if ep == 0:
            obs, _ = env.reset(seed=seed)
        else:
            obs, _ = env.reset()
        state_t = np.asarray(obs, dtype=float)
        done = False
        steps = 0
        while not done:
            action = heuristic_policy(state_t)
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            state_tp1 = np.asarray(obs2, dtype=float)
            if not done:
                states.append(state_t)
                actions.append(action)
                rewards.append(reward)
                states_new.append(state_tp1)
            state_t = state_tp1
            steps += 1
        ep_lengths.append(steps)
    env.close()
    rollout_s = time.perf_counter() - t0
    buffer = (states, actions, rewards, states_new)
    return buffer, rollout_s, ep_lengths


# ----------------------------------------------------------------------
# graph helpers (same conventions as run_ablation.py)
# ----------------------------------------------------------------------
def dbn_edges_to_strings(G):
    return sorted(f"{u} -> {v}" for u, v in G.edges())


def collapsed_edges(G):
    edges = set()
    for u, v in G.edges():
        target = v[:-2] if v.endswith("t+1") else v
        if u == target:
            continue
        edges.add(f"{u} -> {target}")
    return sorted(edges)


def per_target_causes(G):
    out = {name: [] for name in FEATURE_NAMES}
    for u, v in G.edges():
        base = v[:-4] if v.endswith("_t+1") else v
        for feat in FEATURE_NAMES:
            if base.endswith(feat):
                out[feat].append(u)
    return {k: sorted(v) for k, v in out.items()}


def draw_dbn(G, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    t_nodes = [f"{'action' if f == 'push' else 'state'}_{f}_t"
               for f in FEATURE_NAMES]
    tp1_nodes = [f"{'action' if f == 'push' else 'state'}_{f}_t+1"
                 for f in FEATURE_NAMES]
    pos = {}
    for i, n in enumerate(t_nodes):
        pos[n] = (0.0, len(t_nodes) - 1 - i)
    for i, n in enumerate(tp1_nodes):
        pos[n] = (1.0, len(tp1_nodes) - 1 - i)
    H = nx.DiGraph()
    H.add_nodes_from(t_nodes + tp1_nodes)
    H.add_edges_from(G.edges())
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(H, pos, nodelist=t_nodes, node_color="#cfe8ff",
                           node_size=2600, ax=ax)
    nx.draw_networkx_nodes(H, pos, nodelist=tp1_nodes, node_color="#ffe4c2",
                           node_size=2600, ax=ax)
    nx.draw_networkx_edges(H, pos, arrows=True, arrowsize=18,
                           arrowstyle="-|>", connectionstyle="arc3,rad=0.08",
                           ax=ax)
    pos_left = {n: (x - 0.06, y) for n, (x, y) in pos.items()
                if n in t_nodes}
    pos_right = {n: (x + 0.06, y) for n, (x, y) in pos.items()
                 if n in tp1_nodes}
    nx.draw_networkx_labels(H, pos_left, labels={n: n for n in t_nodes}, font_size=8, horizontalalignment="right", ax=ax)
    nx.draw_networkx_labels(H, pos_right, labels={n: n for n in tp1_nodes}, font_size=8, horizontalalignment="left", ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.55, 1.55)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# fidelity check (IDENTICAL protocol to run_ablation.py)
# ----------------------------------------------------------------------
def fidelity_check(states, actions, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    X = np.asarray(states, dtype=float)
    y = np.asarray(actions, dtype=int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    dt = DecisionTreeClassifier(random_state=seed)
    dt.fit(X_tr, y_tr)
    dt_pred = dt.predict(X_te)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, random_state=seed))
    lr.fit(X_tr, y_tr)
    lr_pred = lr.predict(X_te)
    return {
        "dt_acc": float(accuracy_score(y_te, dt_pred)),
        "dt_f1": float(f1_score(y_te, dt_pred, zero_division=0)),
        "lr_acc": float(accuracy_score(y_te, lr_pred)),
        "lr_f1": float(f1_score(y_te, lr_pred, zero_division=0)),
        "n_test": int(len(y_te)),
        "action1_frac": float(y.mean()),
    }


# ----------------------------------------------------------------------
# replay construction (identical to repo infer_DBN)
# ----------------------------------------------------------------------
def build_replay(replay_buffer):
    from cxrl.lib.utils import convert_and_expand
    states, actions, rewards, states_new = replay_buffer
    states = convert_and_expand(states).astype(float)
    actions = convert_and_expand(actions).astype(float)
    rewards = convert_and_expand(rewards).astype(float)
    states_new = convert_and_expand(states_new).astype(float)
    n_states = states.shape[1]
    replay_t = np.concatenate((states, actions), axis=1)[:-1, :]
    replay_tp1 = np.concatenate((states, actions), axis=1)[1:, :]
    return replay_t, replay_tp1, n_states


def edge_name(c, i_repr, n_states):
    prefix_t = "action" if c >= n_states else "state"
    prefix_tp1 = "action" if i_repr >= n_states else "state"
    return (f"{prefix_t}_{FEATURE_NAMES[c]}_t",
            f"{prefix_tp1}_{FEATURE_NAMES[i_repr]}_t+1")


def drop_fitted_model(model):
    """Free a fitted BART model's R object (memory hygiene)."""
    for _attr in ("_bart_machine",):
        if hasattr(model, _attr):
            try:
                delattr(model, _attr)
            except Exception:  # noqa: BLE001
                pass
    gc.collect()


# ----------------------------------------------------------------------
# inference runners: repo logic replicated to retain X_est + model.
# 95% decisions come from the repo's own methods (is_causal_HTE / ATE).
# ----------------------------------------------------------------------
def run_hte(infer_network, replay_buffer, seed, log):
    """Repo infer_causes_hte per target, retaining (model, X_est).

    Split/scale/fit replicate infer_network.infer_causes_hte verbatim:
    train_test_split(test_size=0.3, random_state=None), num_cols = cols
    with >2 unique values in the FULL replay, StandardScaler fit on
    X_train and applied to both splits with only num_cols replaced.
    """
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    import networkx as nx
    replay_t, replay_tp1, n_states = build_replay(replay_buffer)
    G = nx.DiGraph()
    targets = {}  # i_repr -> {model, X_est, num_cols, causes95}
    np.random.seed(seed)
    random.seed(seed)
    t0 = time.perf_counter()
    for i_repr in range(replay_t.shape[1]):
        target = replay_tp1[:, i_repr]
        M = replay_t.shape[1]
        bart = infer_network.BARTRegressor()
        X_train, X_est, y_train, y_est = train_test_split(
            replay_t, target, test_size=0.3)
        num_cols = [c for c in range(M)
                    if len(np.unique(replay_t[:, c])) > 2]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_train[:, num_cols] = X_train_scaled[:, num_cols]
        X_est_scaled = scaler.transform(X_est)
        X_est[:, num_cols] = X_est_scaled[:, num_cols]
        bart.fit(X_train, y_train)
        causes = []
        for c in range(M):
            if bart.is_causal_HTE(X_est, c, additive=(c in num_cols)):
                causes.append(c)
        log.info("hte target %s: causes95=%s (X_est n=%d)",
                 FEATURE_NAMES[i_repr],
                 [FEATURE_NAMES[c] for c in causes], X_est.shape[0])
        targets[i_repr] = {"model": bart, "X_est": X_est.copy(),
                           "num_cols": list(num_cols), "causes95": causes}
        for c in causes:
            G.add_edge(*edge_name(c, i_repr, n_states))
    return G, targets, time.perf_counter() - t0


def run_ate(infer_network, replay_buffer, seed, log):
    """Repo infer_causes_ate per target, retaining (model, X_est).

    Split/scale/fit replicate infer_network.infer_causes_ate verbatim:
    train_test_split(test_size=0.5, random_state=None), num_cols = cols
    with >2 unique values in the FULL replay, each split standardised
    by its OWN mean/std (repo quirk, kept).
    """
    from sklearn.model_selection import train_test_split
    import networkx as nx
    replay_t, replay_tp1, n_states = build_replay(replay_buffer)
    G = nx.DiGraph()
    targets = {}
    np.random.seed(seed)
    random.seed(seed)
    t0 = time.perf_counter()
    for i_repr in range(replay_t.shape[1]):
        target = replay_tp1[:, i_repr]
        M = replay_t.shape[1]
        bart = infer_network.BARTRegressor()
        X_train, X_est, y_train, y_est = train_test_split(
            replay_t, target, test_size=0.5)
        num_cols = [c for c in range(M)
                    if len(np.unique(replay_t[:, c])) > 2]
        X_train[:, num_cols] = (X_train[:, num_cols]
                                - X_train[:, num_cols].mean()) \
            / X_train[:, num_cols].std()
        X_est[:, num_cols] = (X_est[:, num_cols]
                              - X_est[:, num_cols].mean()) \
            / X_est[:, num_cols].std()
        bart.fit(X_train, y_train)
        causes = []
        ate95 = {}
        for c in range(M):
            ate, lci, uci = bart.ATE(X_est, c, additive=(c in num_cols))
            ate95[c] = (float(ate), float(lci), float(uci))
            if lci > 0 or uci < 0:
                causes.append(c)
        log.info("ate target %s: causes95=%s (X_est n=%d)",
                 FEATURE_NAMES[i_repr],
                 [FEATURE_NAMES[c] for c in causes], X_est.shape[0])
        targets[i_repr] = {"model": bart, "X_est": X_est.copy(),
                           "num_cols": list(num_cols), "causes95": causes,
                           "ate95": ate95}
        for c in causes:
            G.add_edge(*edge_name(c, i_repr, n_states))
    return G, targets, time.perf_counter() - t0


# ----------------------------------------------------------------------
# CI sensitivity from ITE posterior draws (same model, same X_est)
# ----------------------------------------------------------------------
def ate_levels_from_draws(ite_draws):
    """(ate, {level: (lci, uci, significant)}) from an ITE draw matrix.

    ite_draws: (500 draws, N points), as returned by BARTRegressor.ITE.
    Mirrors the repo R ``ate`` math: per-draw ATE = rowMeans(ite),
    ATE = mean over draws, CI = quantiles over per-draw ATEs.
    """
    pmean = np.asarray(ite_draws, dtype=float).mean(axis=1)
    ate = float(pmean.mean())
    out = {}
    for level in CI_LEVELS:
        alpha = 1.0 - level
        lci, uci = np.quantile(pmean, [alpha / 2.0, 1.0 - alpha / 2.0])
        out[level] = (float(lci), float(uci), bool(lci > 0 or uci < 0))
    return ate, out


def hte_levels_from_draws(X_est, ite_draws):
    """Rebuild repo HTE clustering; per-level causal decision + details.

    Clustering replicates HTE.__init__ exactly: mean ITE over draws,
    DecisionTreeRegressor(min_samples_leaf=int(N*0.1), max_depth=4)
    with default random_state=None, clusters via tree_.apply. Only the
    pooled-quantile levels vary. Returns (per-level bool,
    n_clusters, cluster_sizes, per-level per-cluster significance).
    """
    import sklearn.tree
    draws = np.asarray(ite_draws, dtype=float)
    assert draws.shape[0] == 500, f"expected 500 draws, got {draws.shape}"
    ite_mean = draws.mean(axis=0)
    n = X_est.shape[0]
    min_samples = int(n * 0.1)
    tree = sklearn.tree.DecisionTreeRegressor(
        min_samples_leaf=min_samples, max_depth=4)
    tree.fit(X_est, ite_mean)
    clusters = tree.tree_.apply(
        np.asfortranarray(X_est.astype(sklearn.tree._tree.DTYPE)))
    sizes = {int(cid): int((clusters == cid).sum())
             for cid in np.unique(clusters)}
    per_level = {}
    per_cluster = {}
    for level in CI_LEVELS:
        alpha = 1.0 - level
        sig = {}
        for cid in np.unique(clusters):
            pooled = draws[:, clusters == cid]
            lci, uci = np.quantile(pooled, [alpha / 2.0, 1.0 - alpha / 2.0])
            sig[int(cid)] = bool(lci > 0 or uci < 0)
        per_cluster[level] = sig
        per_level[level] = any(sig.values())
    return per_level, len(sizes), sizes, per_cluster


def ci_sensitivity(targets, method, n_states, log):
    """Edges at 90/95/99% from the same fitted models + X_est.

    Returns (sensitivity dict, validation dict, seconds). Validation:
    the recomputed 95% decision must equal the repo 95% decision for
    every (target, column); any mismatch is logged + recorded.
    """
    t0 = time.perf_counter()
    out = {level: {"edges": set(), "per_target": {}} for level in CI_LEVELS}
    validation = {"mismatches_95": [], "n_checked": 0}
    detail = {}  # target -> col -> per-level numbers
    for i_repr in range(N_FEATURES):
        rec = targets[i_repr]
        model, X_est, num_cols = rec["model"], rec["X_est"], rec["num_cols"]
        detail[FEATURE_NAMES[i_repr]] = {}
        for c in range(N_FEATURES):
            additive = c in num_cols
            ite = model.ITE(X_est, c, additive=additive)
            ite = np.asarray(ite, dtype=float)
            if ite.shape[0] != 500:
                log.warning("ITE draws shape %s != (500, N) for target=%s col=%s",
                            ite.shape, FEATURE_NAMES[i_repr], FEATURE_NAMES[c])
            repo_sig_95 = c in rec["causes95"]
            validation["n_checked"] += 1
            if method == "ate":
                ate, levels = ate_levels_from_draws(ite)
                detail[FEATURE_NAMES[i_repr]][FEATURE_NAMES[c]] = {
                    "ate": round(ate, 4),
                    **{str(level): {"lci": round(v[0], 4),
                                    "uci": round(v[1], 4),
                                    "significant": v[2]}
                       for level, v in levels.items()},
                    "repo_ate95": [round(v, 4)
                                   for v in rec["ate95"][c]],
                }
                recomputed_95 = levels[0.95][2]
                sig = {level: v[2] for level, v in levels.items()}
            else:
                per_level, n_cl, sizes, per_cluster = hte_levels_from_draws(
                    X_est, ite)
                detail[FEATURE_NAMES[i_repr]][FEATURE_NAMES[c]] = {
                    "n_clusters": n_cl,
                    "cluster_sizes": {str(k): v for k, v in sizes.items()},
                    **{str(level): {"significant": per_level[level],
                                    "cluster_sig": {str(k): v for k, v in
                                                    per_cluster[level].items()}}
                       for level in CI_LEVELS},
                }
                recomputed_95 = per_level[0.95]
                sig = per_level
            if recomputed_95 != repo_sig_95:
                validation["mismatches_95"].append({
                    "target": FEATURE_NAMES[i_repr],
                    "column": FEATURE_NAMES[c],
                    "repo_95": bool(repo_sig_95),
                    "recomputed_95": bool(recomputed_95)})
                log.warning("95%% VALIDATION MISMATCH target=%s col=%s repo=%s recomputed=%s",
                            FEATURE_NAMES[i_repr], FEATURE_NAMES[c],
                            repo_sig_95, recomputed_95)
            for level in CI_LEVELS:
                if sig[level]:
                    u, v = edge_name(c, i_repr, n_states)
                    out[level]["edges"].add(f"{u} -> {v}")
        for level in CI_LEVELS:
            per_target = {name: [] for name in FEATURE_NAMES}
            for e in out[level]["edges"]:
                u, v = e.split(" -> ")
                base = v[:-4] if v.endswith("_t+1") else v
                for feat in FEATURE_NAMES:
                    if base.endswith(feat):
                        per_target[feat].append(u)
            out[level]["per_target"] = {k: sorted(v)
                                        for k, v in per_target.items()}
    sens = {}
    for level in CI_LEVELS:
        sens[str(level)] = {"edges": sorted(out[level]["edges"]),
                            "n_edges": len(out[level]["edges"]),
                            "per_target": out[level]["per_target"]}
        log.info("CI sensitivity %s level=%.2f n_edges=%d edges=%s",
                 method, level, len(out[level]["edges"]),
                 sorted(out[level]["edges"]))
    log.info("CI sensitivity %s validation: %d checked, %d mismatches at 95%%",
             method, validation["n_checked"],
             len(validation["mismatches_95"]))
    return {"levels": sens, "validation": validation, "detail": detail}, \
        time.perf_counter() - t0


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--methods", nargs="+", default=["hte", "ate"],
                    choices=["hte", "ate"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--surrogate-details", type=Path, default=None)
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="skip the 200ep CI sensitivity recompute (repeat runs)")
    ap.add_argument("--run-label", type=str, default="main",
                    help="label stored in JSON to identify repeat runs")
    args = ap.parse_args()

    repo_dir = args.repo or (BASE_DIR / "repo-causal-xrl")
    if not repo_dir.is_dir():
        repo_dir = Path("/tmp/causal-xrl-muc4")
    out_dir = args.output_dir or (BASE_DIR / "results" / "ablation-bart")
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    surr_path = args.surrogate_details or (
        BASE_DIR / "results" / "ablation" / "per_run_details.json")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "ablation_bart_run.log", mode="w"),
                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("ablation-bart")
    log.info(LABEL)
    log.info("run_label=%s repo=%s seed=%s episodes=%s methods=%s "
             "skip_sensitivity=%s", args.run_label, repo_dir, args.seed,
             args.episodes, args.methods, args.skip_sensitivity)
    commit = None
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                capture_output=True, text=True,
                                timeout=30).stdout.strip()
        log.info("repo commit=%s", commit)
    except Exception as e:  # noqa: BLE001
        log.warning("could not read repo commit: %s", e)

    import sklearn, gymnasium, networkx, matplotlib  # noqa: E402
    import pandas, scipy  # noqa: E402
    versions = {
        "sklearn": sklearn.__version__, "gymnasium": gymnasium.__version__,
        "networkx": networkx.__version__,
        "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "pandas": pandas.__version__, "scipy": scipy.__version__,
        "python": sys.version.split()[0],
    }
    log.info("versions: %s", versions)
    # rpy2 / R / dbarts versions (same check pattern as run_xrl_comparison)
    try:
        import rpy2  # noqa: E402
        import rpy2.robjects as _ro  # noqa: E402
        versions["rpy2"] = rpy2.__version__
        _ro.r('library("dbarts")')
        versions["dbarts"] = list(
            _ro.r('as.character(packageVersion("dbarts"))'))[0]
        versions["R"] = list(_ro.r('as.character(getRversion())'))[0]
        log.info("rpy2=%s R=%s dbarts=%s", versions["rpy2"],
                 versions["R"], versions["dbarts"])
    except Exception as e:  # noqa: BLE001
        log.error("rpy2/dbarts check failed: %s", e)
        raise

    infer_network = load_repo_infer_network(repo_dir)

    rows = []
    details = {"label": LABEL, "estimator": "bart", "run_label": args.run_label,
               "seed": args.seed, "repo_commit": commit,
               "repo_dir": str(repo_dir), "versions": versions,
               "runs": {}}
    sensitivity_all = {}

    for nep in args.episodes:
        buffer, rollout_s, ep_lengths = collect_cartpole(nep, seed=args.seed)
        states, actions, rewards, states_new = buffer
        n_trans = len(states)
        expected = EXPECTED_TRANSITIONS.get(nep)
        if expected is not None and n_trans != expected:
            log.warning("transitions=%d != surrogate-run expected %d "
                        "(same seed/collection code) -- investigate",
                        n_trans, expected)
        log.info("episodes=%d transitions=%d rollout_s=%.2f "
                 "mean_ep_len=%.1f action1_frac=%.3f",
                 nep, n_trans, rollout_s, float(np.mean(ep_lengths)),
                 float(np.mean(actions)))
        fid = fidelity_check(states, actions, seed=args.seed)
        log.info("fidelity nep=%d DT acc=%.4f F1=%.4f | LR acc=%.4f F1=%.4f",
                 nep, fid["dt_acc"], fid["dt_f1"], fid["lr_acc"], fid["lr_f1"])
        for method in args.methods:
            if method == "hte":
                G, targets, infer_s = run_hte(
                    infer_network, buffer, args.seed, log)
            else:
                G, targets, infer_s = run_ate(
                    infer_network, buffer, args.seed, log)
            edges = dbn_edges_to_strings(G)
            edges_c = collapsed_edges(G)
            causes = per_target_causes(G)
            log.info("run nep=%d method=%s n_edges=%d infer_s=%.2f edges=%s",
                     nep, method, len(edges), infer_s, edges)
            log.info("per-target causes nep=%d method=%s: %s",
                     nep, method, json.dumps(causes))
            png = graphs_dir / f"graph_{method}_{nep}ep.png"
            draw_dbn(G, png,
                     f"CartPole global causal graph ({method}, {nep} episodes)\n"
                     f"{LABEL}")
            key = f"{method}_{nep}"
            run_rec = {
                "nepisodes": nep, "method": method,
                "n_transitions": n_trans, "rollout_s": round(rollout_s, 2),
                "mean_ep_len": round(float(np.mean(ep_lengths)), 1),
                "action1_frac": round(float(np.mean(actions)), 4),
                "infer_s": round(infer_s, 2),
                "edges": edges, "edges_collapsed": edges_c,
                "n_edges": len(edges),
                "per_target_causes": causes,
                "fidelity": {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in fid.items()},
            }
            if method == "ate":
                run_rec["ate_table_95"] = {
                    FEATURE_NAMES[i]: {
                        FEATURE_NAMES[c]: [round(v, 4) for v in
                                           targets[i]["ate95"][c]]
                        for c in range(N_FEATURES)}
                    for i in range(N_FEATURES)}
            # ---- sensitivity (200ep only, unless skipped) ----
            if nep == 200 and not args.skip_sensitivity:
                _, replay_tp1, n_states = build_replay(buffer)
                sens, sens_s = ci_sensitivity(targets, method, n_states, log)
                sensitivity_all[method] = sens
                run_rec["sensitivity_s"] = round(sens_s, 2)
                run_rec["sensitivity_n_edges"] = {
                    level: sens["levels"][level]["n_edges"]
                    for level in sens["levels"]}
                log.info("sensitivity %s done in %.1fs", method, sens_s)
            details["runs"][key] = run_rec
            rows.append({
                "nepisodes": nep, "method": method,
                "n_transitions": n_trans, "n_edges": len(edges),
                "edges": "; ".join(edges),
                "edges_collapsed": "; ".join(edges_c),
                "per_target_causes": json.dumps(causes),
                "time_s": round(infer_s, 2), "rollout_s": round(rollout_s, 2),
                "dt_acc": round(fid["dt_acc"], 4), "dt_f1": round(fid["dt_f1"], 4),
                "lr_acc": round(fid["lr_acc"], 4), "lr_f1": round(fid["lr_f1"], 4),
                "sensitivity_s": run_rec.get("sensitivity_s"),
            })
            # free fitted models before the next run
            for rec in targets.values():
                drop_fitted_model(rec["model"])
            del targets
            gc.collect()
            # incremental save after every run
            (out_dir / "partial_results.json").write_text(
                json.dumps({"rows": rows, "details": details,
                            "sensitivity": sensitivity_all}, indent=2,
                           default=str))
            log.info("incremental save: %s (run %s done)",
                     out_dir / "partial_results.json", key)

    if sensitivity_all:
        details["ci_sensitivity_200ep"] = sensitivity_all

    # ---- cross-episode comparison per method (BART) ----
    comparison = {}
    for method in args.methods:
        by_ep = {r["nepisodes"]: set(r["edges"].split("; ")) if r["edges"]
                 else set() for r in rows if r["method"] == method}
        if not by_ep:
            continue
        ref = by_ep[max(by_ep)]
        comp = {}
        for nep in sorted(by_ep):
            comp[str(nep)] = {
                "n_edges": len(by_ep[nep]),
                "jaccard_vs_200": round(jaccard(by_ep[nep], ref), 4),
                "symdiff_vs_200": len(by_ep[nep] ^ ref),
                "edges_only_here": sorted(by_ep[nep] - ref),
                "edges_missing_vs_200": sorted(ref - by_ep[nep]),
            }
        union = set().union(*by_ep.values())
        inter = set.intersection(*by_ep.values()) if by_ep else set()
        comp["stable_edges_all_counts"] = sorted(inter)
        comp["unstable_edges"] = sorted(union - inter)
        comparison[method] = comp
        log.info("comparison %s: stable=%s unstable=%s",
                 method, comp["stable_edges_all_counts"],
                 comp["unstable_edges"])
    details["comparison_vs_200ep"] = comparison

    # ---- BART vs surrogate comparison (same config) ----
    bart_vs_surr = None
    if surr_path.is_file():
        try:
            surr = json.loads(surr_path.read_text())
            bart_vs_surr = {}
            for r in rows:
                key = f"{r['method']}_{r['nepisodes']}"
                srec = surr.get(key)
                if not srec:
                    continue
                bedges = set(r["edges"].split("; ")) if r["edges"] else set()
                sedges = set(srec["edges"])
                bart_vs_surr[key] = {
                    "bart_n_edges": len(bedges),
                    "surrogate_n_edges": len(sedges),
                    "jaccard_bart_vs_surrogate": round(jaccard(bedges, sedges), 4),
                    "symdiff": len(bedges ^ sedges),
                    "only_bart": sorted(bedges - sedges),
                    "only_surrogate": sorted(sedges - bedges),
                    "bart_push_parents": json.loads(
                        r["per_target_causes"])["push"],
                    "surrogate_push_parents":
                        srec["per_target_causes"]["push"],
                    "bart_infer_s": r["time_s"],
                    "surrogate_infer_s": srec["infer_s"],
                }
                log.info("BART-vs-surrogate %s: bart=%d surr=%d "
                         "Jaccard=%.4f push bart=%s surr=%s",
                         key, len(bedges), len(sedges),
                         bart_vs_surr[key]["jaccard_bart_vs_surrogate"],
                         bart_vs_surr[key]["bart_push_parents"],
                         bart_vs_surr[key]["surrogate_push_parents"])
            details["bart_vs_surrogate"] = bart_vs_surr
        except Exception as e:  # noqa: BLE001
            log.warning("surrogate comparison failed (%s): %s", surr_path, e)
    else:
        log.warning("surrogate details not found at %s; comparison skipped",
                    surr_path)

    # ---- write outputs ----
    fieldnames = ["nepisodes", "method", "n_transitions", "n_edges", "edges",
                  "edges_collapsed", "per_target_causes", "time_s",
                  "rollout_s", "dt_acc", "dt_f1", "lr_acc", "lr_f1",
                  "sensitivity_s"]
    with open(out_dir / "ablation_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "per_run_details.json").write_text(
        json.dumps(details, indent=2, default=str))

    md = []
    md.append("# CausalXRL extension A — ablation / stability check with REAL BART (CartPole)\n")
    md.append(f"> **{LABEL}**\n")
    md.append("> Data: gymnasium CartPole-v1 rollouts with the repo's exact "
              "heuristic policy (repo `play()` uses gym 0.20, incompatible "
              "with Python 3.12, so gymnasium is used for data collection "
              "only; the repo's transition-recording rule is kept). "
              "Seed 0. `hte` = repo `infer_causes_hte` logic (70/30 split); "
              "`ate` = repo `infer_causes_ate` logic (50/50 split); 95% "
              "decisions from the repo's own `is_causal_HTE` / `ATE` "
              "methods. Fidelity = DecisionTree / LogisticRegression "
              "predicting the next action from the state (80/20 split) — "
              "the SAME protocol as the surrogate ablation run, so the "
              "DT/LR numbers are directly comparable.\n")
    md.append("| nepisodes | method | n_trans | n_edges | time_s (infer) | "
              "DT acc / F1 | LR acc / F1 | edges |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        md.append(f"| {r['nepisodes']} | {r['method']} | {r['n_transitions']} "
                  f"| {r['n_edges']} | {r['time_s']} | "
                  f"{r['dt_acc']} / {r['dt_f1']} | {r['lr_acc']} / {r['lr_f1']} "
                  f"| {r['edges']} |")
    md.append("")
    md.append("## Per-target causes\n")
    for r in rows:
        md.append(f"- **{r['method']} / {r['nepisodes']} ep**: "
                  f"`{r['per_target_causes']}`")
    md.append("")
    md.append("## Graph comparison vs the 200-episode graph (same method, BART)\n")
    for method, comp in comparison.items():
        md.append(f"### {method}\n")
        for nep in sorted([k for k in comp if k.isdigit()], key=int):
            c = comp[nep]
            md.append(f"- **{nep} ep**: n_edges={c['n_edges']}, "
                      f"Jaccard vs 200 ep = {c['jaccard_vs_200']}, "
                      f"symmetric difference = {c['symdiff_vs_200']} edges; "
                      f"only here: {c['edges_only_here'] or '—'}; "
                      f"missing vs 200 ep: {c['edges_missing_vs_200'] or '—'}")
        md.append(f"- **Stable edges (present at 50, 100 and 200 ep)**: "
                  f"{comp['stable_edges_all_counts'] or '—'}")
        md.append(f"- **Unstable edges (appear/disappear across counts)**: "
                  f"{comp['unstable_edges'] or '—'}")
        md.append("")
    (out_dir / "ablation_table.md").write_text("\n".join(md) + "\n")

    if sensitivity_all:
        ci_md = ["# CI-threshold sensitivity — 200-episode runs (REAL BART)\n",
                 f"> **{LABEL}**\n",
                 "> Edges at 90% / 95% / 99% CI recomputed from the SAME "
                 "fitted BART models and the SAME estimation split via the "
                 "500 ITE posterior draws; no refit. The repo hardcodes 95% "
                 "(`bart_regressor.py` R `ate`: quantile probs c(0.025, "
                 "0.975); `HTE.__init__`: np.quantile q=[0.025, 0.975]) and "
                 "offers no level parameter; no repo file was edited. "
                 "Recomputed 95% decisions were validated against the repo "
                 "95% decisions for every (target, column) — see validation "
                 "lines below and `per_run_details.json`.\n"]
        for method, sens in sensitivity_all.items():
            lv = sens["levels"]
            ci_md.append(f"## {method} (200 episodes)\n")
            for level in ("0.9", "0.95", "0.99"):
                e = lv[level]["edges"]
                ci_md.append(f"- **{float(level)*100:.0f}% CI**: "
                             f"{len(e)} edges: `{'; '.join(e)}`")
            e90, e95, e99 = set(lv["0.9"]["edges"]), set(lv["0.95"]["edges"]), set(lv["0.99"]["edges"])
            ci_md.append(f"- 90% → 95%: {len(e90 - e95)} edge(s) lost: "
                         f"`{'; '.join(sorted(e90 - e95)) or '—'}`; "
                         f"{len(e95 - e90)} gained: "
                         f"`{'; '.join(sorted(e95 - e90)) or '—'}`")
            ci_md.append(f"- 95% → 99%: {len(e95 - e99)} edge(s) lost: "
                         f"`{'; '.join(sorted(e95 - e99)) or '—'}`; "
                         f"{len(e99 - e95)} gained: "
                         f"`{'; '.join(sorted(e99 - e95)) or '—'}`")
            v = sens["validation"]
            ci_md.append(f"- Validation: {v['n_checked']} (target, column) "
                         f"pairs checked at 95%, "
                         f"{len(v['mismatches_95'])} mismatch(es) "
                         f"between recomputed and repo decisions"
                         + (f": {v['mismatches_95']}" if v["mismatches_95"] else " — recompute matches the repo exactly.")
                         )
            ci_md.append("")
        (out_dir / "ci_sensitivity.md").write_text("\n".join(ci_md) + "\n")
        log.info("wrote %s", out_dir / "ci_sensitivity.md")

    if bart_vs_surr:
        cmp_md = ["# REAL BART vs surrogate (RandomForest) — same configs\n",
                  f"> BART: **{LABEL}**. Surrogate: RandomForest, 200 trees "
                  "(`results/ablation/`, seed 0, same data-collection code "
                  "and same fidelity protocol). Jaccard is over DBN edge "
                  "sets (t -> t+1) for the same (method, episodes).\n",
                  "| config | BART n_edges | surrogate n_edges | Jaccard | "
                  "symdiff | BART push parents | surrogate push parents |",
                  "|---|---:|---:|---:|---:|---|---|"]
        for key in sorted(bart_vs_surr,
                          key=lambda k: (k.split("_")[0], int(k.split("_")[1]))):
            b = bart_vs_surr[key]
            cmp_md.append(f"| {key} | {b['bart_n_edges']} | "
                          f"{b['surrogate_n_edges']} | "
                          f"{b['jaccard_bart_vs_surrogate']} | {b['symdiff']} | "
                          f"`{'; '.join(b['bart_push_parents']) or '(none)'}` | "
                          f"`{'; '.join(b['surrogate_push_parents']) or '(none)'}` |")
        cmp_md.append("")
        cmp_md.append("## Edge differences per config\n")
        for key in sorted(bart_vs_surr,
                          key=lambda k: (k.split("_")[0], int(k.split("_")[1]))):
            b = bart_vs_surr[key]
            cmp_md.append(f"- **{key}**: only in BART: "
                          f"`{'; '.join(b['only_bart']) or '—'}`; "
                          f"only in surrogate: "
                          f"`{'; '.join(b['only_surrogate']) or '—'}`")
        cmp_md.append("")
        (out_dir / "comparison_vs_surrogate.md").write_text(
            "\n".join(cmp_md) + "\n")
        log.info("wrote %s", out_dir / "comparison_vs_surrogate.md")

    log.info("wrote %s", out_dir / "ablation_table.csv")
    log.info("wrote %s", out_dir / "ablation_table.md")
    log.info("wrote %s", out_dir / "per_run_details.json")
    log.info("DONE")


if __name__ == "__main__":
    main()
