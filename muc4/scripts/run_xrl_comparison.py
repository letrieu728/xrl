#!/usr/bin/env python3
"""CausalXRL extension D — XRL trained-vs-heuristic comparison (CartPole).

Compares the repo's XRL pipeline (global causal graph + fidelity) run on
rollouts from TWO policies:

* ``heuristic`` — the hand-written heuristic from the CausalXRL repo
  (CartPole.ipynb): ``1 if 0.4*(x/2.4) + 0.6*(theta/2.095) >= 0 else 0``.
* ``dqn`` — the DQN trained by scripts/train_dqn_cartpole.py
  (``--checkpoint``; the best checkpoint available at run time, recorded
  in the outputs).

For each policy, 200 episodes (seed 0) are collected with gymnasium
(the repo's ``play()`` uses gym 0.20, incompatible with Python 3.12;
the repo's transition-recording rule is kept: a transition is recorded
only when the step is NOT terminal), then the repo's causal-discovery
pipeline from ``cxrl/lib/causal/infer_network.py`` (commit ae9096e,
read-only clone at /tmp/causal-xrl-muc4) is run with BOTH methods:

* ``hte`` — repo ``infer_DBN(method='ate')`` -> ``infer_causes_hte``
  (70/30 split).
* ``ate`` — repo ``infer_causes_ate`` per target (50/50 split), called by
  a loop replicating ``infer_DBN``'s replay construction and edge
  building (``infer_DBN`` itself has no ``ate`` dispatch).

Three comparisons (the three items of extension D):

1. Behaviour-prediction (fidelity) accuracy/F1 of the next action,
   eval-notebook style (``cxrl/jupyter/eval/eval_fidelity_Cart Pole.ipynb``):
   steps where the local causal graph has >1 cause for ``push`` are kept;
   60/40 split (random_state=10); DecisionTree, LogisticRegression, and
   BART weighted by |ITE| over the local causes of the action (parents of
   ``push`` in the learned hte graph are used as the global-cause set,
   instead of the notebook's hard-coded ``['cart_position', 'pole_angle',
   'push']`` — the learned set is reported in the outputs).
   The BART-weighted prediction is capped at 200 test steps (each step
   needs one R ``predict(type='bart')`` per local cause); DT/LR use the
   full held-out set. The cap is recorded in the outputs.
2. Learned causal graphs — PNG per policy/method, edge lists, edge
   symmetric difference and Jaccard between the two policies.
3. Structure-learning time (infer seconds) per policy/method.

Estimator (``--estimator``):

* ``bart`` (default) — the repo's real ``BARTRegressor`` (R ``dbarts``
  via ``rpy2``). Outputs are labelled ``BART (R/dbarts via rpy2)``.
* ``surrogate`` — ``scripts/surrogate_bart.py`` (RandomForest, 200 trees)
  monkeypatched in place of ``BARTRegressor`` (same fallback as
  ``run_ablation.py``). Every output is labelled
  ``SURROGATE ESTIMATOR (RandomForest, 200 trees), NOT BART/dbarts``.

Known repo bugs avoided (see CausalXRL-Phan-tich-ma-nguon.md):
* ``infer_causes_ate_barl`` imports non-existent ``ai_analyzer`` — never called.
* ``XRL.counterfactual`` calls ``self.pi.predict`` while ``pi`` is a plain
  function — counterfactuals are NOT run here (out of scope for this
  comparison); ``DummySklearn`` is therefore not needed.
* The repo ``HTE`` class asserts exactly 500 posterior draws; real BART
  gives 125 ndpost x 4 chains = 500, so the assert holds.

Reproducibility: seed 0 everywhere (numpy global RNG seeded before each
inference call because the repo's ``train_test_split`` calls use
``random_state=None``; BART's R seed is fixed at 99 inside the repo's R
code; RandomForest ``random_state=0`` for the surrogate).

Usage:
    .venv-ablation/bin/python -B scripts/run_xrl_comparison.py \
        --checkpoint results/cartpole-dqn/dqn_checkpoint.pt \
        [--estimator bart|surrogate] [--episodes 200] [--seed 0] \
        [--output-dir results/xrl-trained-vs-heuristic]
"""

import argparse
import csv
import gc
import json
import logging
import os
# rpy2 needs R_HOME; default it so the script works without the caller
# exporting it (R is at /usr/lib/R on this VM).
os.environ.setdefault("R_HOME", "/usr/lib/R")
# Limit BLAS/OpenMP threads (see run_ablation.py for why).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import random
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_DIR = Path("/tmp/causal-xrl-muc4")

FEATURE_NAMES = ["cart_position", "cart_velocity", "pole_angle",
                 "pole_velocity", "push"]
N_FEATURES = len(FEATURE_NAMES)
NUM_COLS = [0, 1, 2, 3]  # state columns are numeric; col 4 (push) is binary

BART_LABEL = "BART (R/dbarts via rpy2) - real CausalXRL estimator"
SURROGATE_LABEL = ("SURROGATE ESTIMATOR (RandomForest, 200 trees), NOT "
                   "BART/dbarts - comparison only, to be re-run with real "
                   "BART when available")


# ----------------------------------------------------------------------
# data collection (gymnasium; same recording rule as repo play())
# ----------------------------------------------------------------------
def collect_cartpole(policy_fn, nepisodes, seed=0):
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
            action = int(policy_fn(state_t))
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
    return (states, actions, rewards, states_new), rollout_s, ep_lengths


# ----------------------------------------------------------------------
# estimator loading
# ----------------------------------------------------------------------
def _install_rpy2_stub():
    """Dummy rpy2 modules so the repo's bart_regressor imports when R is
    absent. Only used with --estimator surrogate; never called."""
    if "rpy2" in sys.modules:
        return
    rpy2 = types.ModuleType("rpy2")
    robjects = types.ModuleType("rpy2.robjects")
    numpy2ri = types.ModuleType("rpy2.robjects.numpy2ri")
    packages = types.ModuleType("rpy2.robjects.packages")
    vectors = types.ModuleType("rpy2.robjects.vectors")
    numpy2ri.activate = lambda: None
    packages.importr = lambda name: types.SimpleNamespace()
    packages.isinstalled = lambda name: True
    vectors.StrVector = list
    robjects.r = lambda code: None
    robjects.globalenv = {}
    robjects.numpy2ri = numpy2ri
    robjects.packages = packages
    robjects.vectors = vectors
    rpy2.robjects = robjects
    sys.modules["rpy2"] = rpy2
    sys.modules["rpy2.robjects"] = robjects
    sys.modules["rpy2.robjects.numpy2ri"] = numpy2ri
    sys.modules["rpy2.robjects.packages"] = packages
    sys.modules["rpy2.robjects.vectors"] = vectors


def load_repo_infer_network(estimator, repo_dir=REPO_DIR):
    """Import the repo's infer_network with the chosen estimator.

    ``bart``: real BARTRegressor (requires R + dbarts + rpy2).
    ``surrogate``: monkeypatched SurrogateRegressor (rpy2 stubbed).
    ``repo_dir``: the repo checkout to import from (default
    /tmp/causal-xrl-muc4; --repo overrides it — /tmp is wiped on VM
    replacement, so the persistent workspace copy is preferred)."""
    # Compat shim (no repo file edited): the repo's bart_regressor.py uses
    # sklearn.tree._tree.DTYPE, removed in sklearn >= 1.6 (this venv has
    # 1.9.1). Old DTYPE was np.float32; restore the attribute so the repo
    # code (HTE.cluster, decision_tree_cluster) runs unchanged. Needed for
    # BOTH estimators because infer_network imports bart_regressor.
    import sklearn.tree._tree as _sk_tree
    if not hasattr(_sk_tree, "DTYPE"):
        _sk_tree.DTYPE = np.float32
    sys.path.insert(0, str(SCRIPTS_DIR))
    if estimator == "surrogate":
        from surrogate_bart import SurrogateRegressor  # noqa: E402
        _install_rpy2_stub()
        sys.path.insert(0, str(repo_dir))
        import cxrl.lib.causal.infer_network as infer_network  # noqa: E402
        infer_network.BARTRegressor = SurrogateRegressor
        return infer_network
    sys.path.insert(0, str(repo_dir))
    import cxrl.lib.causal.infer_network as infer_network  # noqa: E402
    return infer_network


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
    nx.draw_networkx_labels(H, pos_left, labels={n: n for n in t_nodes},
                            font_size=8, horizontalalignment="right", ax=ax)
    nx.draw_networkx_labels(H, pos_right, labels={n: n for n in tp1_nodes},
                            font_size=8, horizontalalignment="left", ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.55, 1.55)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------------
# inference runners (repo pipeline)
# ----------------------------------------------------------------------
def run_hte(infer_network, replay_buffer, seed):
    """Per-target repo infer_causes_hte, replicating infer_DBN's replay
    construction and edge building (infer_DBN with method='ate' calls
    infer_causes_hte per target).

    MEMORY (fix for the silent deaths in runs 3/5): the repo's infer_DBN
    keeps all 5 fitted BART models in its scm dict; on DQN data (~36k
    train rows) each model is hundreds of MB, so the process grew to
    ~5 GB and died silently mid-run (no traceback — consistent with
    OOM). Only TWO things are ever needed from the hte scm afterwards:
      * local_treatments() — uses only the small HTE decision trees
        (hte_per_treat), never the fitted model;
      * ITE() on the ACTION target only (index 4) — needs that target's
        fitted model (for the BART-weighted fidelity).
    So the heavy fitted model (real BART: _bart_machine; surrogate:
    _model) is dropped for every target EXCEPT the action target, right
    after that target's causes are extracted. Peak during hte is then
    ~1 fitted model instead of 5. The RNG call order (one seed, then
    one train_test_split per target) is identical to infer_DBN, so the
    splits — and hence the results — match the repo path up to the usual
    run-to-run variability.
    """
    from cxrl.lib.utils import convert_and_expand
    states, actions, rewards, states_new = replay_buffer
    states = convert_and_expand(states).astype(float)
    actions = convert_and_expand(actions).astype(float)
    rewards = convert_and_expand(rewards).astype(float)
    states_new = convert_and_expand(states_new).astype(float)
    n_states = states.shape[1]
    replay_t = np.concatenate((states, actions), axis=1)[:-1, :]
    replay_tp1 = np.concatenate((states, actions), axis=1)[1:, :]
    import networkx as nx
    G = nx.DiGraph()
    scm = {}
    np.random.seed(seed)
    random.seed(seed)
    t0 = time.perf_counter()
    for i_repr in range(replay_t.shape[1]):
        target = replay_tp1[:, i_repr]
        causes, model = infer_network.infer_causes_hte(replay_t, target,
                                                        verbose=False)
        scm[i_repr] = model
        for c in causes:
            prefix_t = "action" if c >= n_states else "state"
            prefix_tp1 = "action" if i_repr >= n_states else "state"
            G.add_edge(f"{prefix_t}_{FEATURE_NAMES[c]}_t",
                       f"{prefix_tp1}_{FEATURE_NAMES[i_repr]}_t+1")
        if i_repr != 4:  # action target keeps its fitted model (ITE)
            for _attr in ("_bart_machine", "_model"):
                if hasattr(model, _attr):
                    try:
                        delattr(model, _attr)
                    except Exception:  # noqa: BLE001
                        pass
            gc.collect()
    return G, scm, time.perf_counter() - t0


def run_ate(infer_network, replay_buffer, seed):
    """Per-target repo infer_causes_ate, replicating infer_DBN's replay
    construction and edge building (infer_DBN has no 'ate' dispatch).

    MEMORY: each fitted BART model holds a full R bartMachine (on DQN
    data ~34k train rows, one model is hundreds of MB). The models are
    NOT needed after the causes are extracted (only the hte scm is used
    for fidelity), so each model is dropped right after its target is
    done. Returns (G, {}, infer_s, ate_table) where ate_table is only
    populated for estimators that store per-column ATE points
    (surrogate); real BART models are dropped without re-querying ATE.
    """
    from cxrl.lib.utils import convert_and_expand
    states, actions, rewards, states_new = replay_buffer
    states = convert_and_expand(states).astype(float)
    actions = convert_and_expand(actions).astype(float)
    rewards = convert_and_expand(rewards).astype(float)
    states_new = convert_and_expand(states_new).astype(float)
    n_states = states.shape[1]
    replay_t = np.concatenate((states, actions), axis=1)[:-1, :]
    replay_tp1 = np.concatenate((states, actions), axis=1)[1:, :]
    import networkx as nx
    G = nx.DiGraph()
    ate_table = {}
    np.random.seed(seed)
    random.seed(seed)
    t0 = time.perf_counter()
    for i_repr in range(replay_t.shape[1]):
        target = replay_tp1[:, i_repr]
        causes, model = infer_network.infer_causes_ate(replay_t, target,
                                                        verbose=False)
        if hasattr(model, "_ate_point"):
            ate_table[FEATURE_NAMES[i_repr]] = {
                FEATURE_NAMES[c]: [round(v, 4) for v in
                                   model._ate_point[c]]
                for c in sorted(model._ate_point)}
        for c in causes:
            prefix_t = "action" if c >= n_states else "state"
            prefix_tp1 = "action" if i_repr >= n_states else "state"
            G.add_edge(f"{prefix_t}_{FEATURE_NAMES[c]}_t",
                       f"{prefix_tp1}_{FEATURE_NAMES[i_repr]}_t+1")
        del model
        gc.collect()
    return G, {}, time.perf_counter() - t0, ate_table


# ----------------------------------------------------------------------
# fidelity, eval-notebook style (uses the hte run's scm)
# ----------------------------------------------------------------------
def local_causal_network(scm, query_row):
    """Replicates XRL.local_causal_network for one (state+action) row.

    Returns the set of (source_feature, target_feature) edges. The repo
    version builds a nx.DiGraph from a 1-row DataFrame; here the query is
    a 1x5 numpy array and the column order is FEATURE_NAMES."""
    import numpy as np
    edges = set()
    for i in range(N_FEATURES):
        if i not in scm:
            continue
        local = scm[i].local_treatments(np.asarray(query_row, dtype=float))
        for treat in np.nonzero(np.asarray(local)[0])[0]:
            edges.add((FEATURE_NAMES[int(treat)], FEATURE_NAMES[i]))
    return edges


def fidelity_eval_notebook(infer_network, scm_hte, G_hte, states, actions,
                           estimator, log, bart_test_cap=200):
    """DT / LR / BART-weighted action prediction, eval-notebook style.

    Keeps steps whose local causal graph has >1 cause for 'push'; 60/40
    split (random_state=10, as in the notebook); DT and LR on the full
    held-out set; BART-weighted on at most ``bart_test_cap`` test steps
    (each needs one R predict per local cause). Returns a dict with the
    metrics plus the counts and the global-cause set used."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    replay = np.concatenate(
        (np.asarray(states, dtype=float),
         np.asarray(actions, dtype=float).reshape(-1, 1)), axis=1)
    log.info("fidelity: replay rows=%d, finding valid steps "
             "(local graph >1 cause for push)...", replay.shape[0])
    t0 = time.perf_counter()
    valid_steps = []
    for step in range(replay.shape[0]):
        edges = local_causal_network(scm_hte, replay[step:step + 1])
        n_push_causes = sum(1 for (s, t) in edges
                            if t == "push" and s != "push")
        if n_push_causes > 1:
            valid_steps.append(step)
    log.info("fidelity: valid_steps=%d (%.1f%% of %d), local-scan took %.1fs",
             len(valid_steps), 100.0 * len(valid_steps) / max(replay.shape[0], 1),
             replay.shape[0], time.perf_counter() - t0)
    if not valid_steps:
        return {"valid_steps": 0, "note": "no step had >1 local cause for push"}
    if valid_steps[-1] == replay.shape[0] - 1:
        valid_steps = valid_steps[:-1]
    valid_tp1 = (np.array(valid_steps) + 1).tolist()
    eval_t = replay[valid_steps]
    eval_tp1 = replay[valid_tp1]
    action_index = 4
    X = eval_t
    y = eval_tp1[:, action_index].astype(int)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(valid_steps)), test_size=0.4, random_state=10)
    dt = DecisionTreeClassifier(random_state=1)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    lr = LogisticRegression(random_state=1, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    out = {
        "valid_steps": int(len(valid_steps)),
        "n_test_dt_lr": int(len(y_test)),
        "dt_acc": float(accuracy_score(y_test, dt_pred)),
        "dt_f1": float(f1_score(y_test, dt_pred, zero_division=0)),
        "lr_acc": float(accuracy_score(y_test, lr_pred)),
        "lr_f1": float(f1_score(y_test, lr_pred, zero_division=0)),
    }
    log.info("fidelity: DT acc=%.4f F1=%.4f | LR acc=%.4f F1=%.4f (n_test=%d)",
             out["dt_acc"], out["dt_f1"], out["lr_acc"], out["lr_f1"],
             out["n_test_dt_lr"])

    # --- BART-weighted prediction (notebook cells 20-25) ---
    # Global-cause set: parents of 'push' in the learned hte DBN graph
    # (notebook hard-codes ['cart_position','pole_angle','push']).
    global_cause = sorted({u for (u, v) in G_hte.edges()
                           if v == "action_push_t+1"})
    global_cause_feats = [u.replace("state_", "").replace("action_", "")
                          .replace("_t", "") for u in global_cause]
    out["global_cause_push_parents"] = global_cause_feats
    log.info("fidelity: global parents of push (hte graph) = %s",
             global_cause_feats)
    if not global_cause_feats:
        out["bart_note"] = "no learned parent of push; BART-weighted skipped"
        return out
    causes_idx = [FEATURE_NAMES.index(f) for f in global_cause_feats]
    covars_idx = [i for i in range(N_FEATURES) if i not in causes_idx]
    # MEMORY: the hte scm holds one fitted BART model per target; on DQN
    # data each is hundreds of MB (the process died silently — almost
    # certainly OOM — while fitting these fidelity models on top of the
    # full scm). From here on only two things are needed from the scm:
    #   * local_treatments() — uses only the small HTE decision trees
    #     (hte_per_treat), never the fitted model;
    #   * ITE() on the ACTION target only (action_index) — needs that
    #     target's fitted model.
    # So drop the heavy fitted model from every target except the action
    # target before fitting the fidelity BARTs. Real BART stores it as
    # _bart_machine; the surrogate as _model.
    for _i, _m in list(scm_hte.items()):
        if _i == action_index:
            continue
        for _attr in ("_bart_machine", "_model"):
            if hasattr(_m, _attr):
                try:
                    delattr(_m, _attr)
                except Exception:  # noqa: BLE001
                    pass
        del _m
    import gc as _gc
    _gc.collect()
    log.info("fidelity: dropped fitted models for scm targets != action "
             "(kept target %d for ITE)", action_index)
    bart_cls = infer_network.BARTRegressor
    hash_y_bart = {}
    t0 = time.perf_counter()
    for treat in causes_idx:
        cols = [treat] + covars_idx
        bart = bart_cls()
        bart.fit(X_train[:, cols], y_train)
        hash_y_bart[treat] = np.asarray(
            bart.predict(X_test[:, cols]), dtype=float).ravel()
        # MEMORY: drop the fitted model right after its predictions are
        # extracted; only the prediction vectors are needed below.
        del bart
        gc.collect()
    log.info("fidelity: fitted %d BART models for global causes in %.1fs",
             len(causes_idx), time.perf_counter() - t0)
    # Cap the per-step ITE loop: each test step needs one R predict per
    # local cause. Keep the first bart_test_cap test steps (deterministic).
    test_steps = idx_test
    capped = len(test_steps) > bart_test_cap
    use_idx = test_steps[:bart_test_cap]
    out["bart_test_cap"] = int(bart_test_cap)
    out["bart_test_capped"] = bool(capped)
    out["n_test_bart"] = int(len(use_idx))
    y_pred_bart = np.zeros(len(use_idx))
    t0 = time.perf_counter()
    for j, step in enumerate(use_idx):
        query = replay[step:step + 1]
        local = scm_hte[action_index].local_treatments(query)
        total_weight = 0.0
        y_bart_pred = 0.0
        # position of this test step inside X_test (test split order)
        pos_in_test = int(np.where(idx_test == step)[0][0])
        for treat in np.nonzero(np.asarray(local)[0])[0]:
            treat = int(treat)
            if treat not in hash_y_bart:
                continue
            ite = scm_hte[action_index].ITE(
                query, treat, additive=(treat in NUM_COLS))
            weight = float(abs(np.asarray(ite, dtype=float).mean()))
            total_weight += weight
            y_bart_pred += hash_y_bart[treat][pos_in_test] * weight
        y_pred_bart[j] = (y_bart_pred / total_weight) if total_weight > 0 else 0.5
    y_test_bart = y_test[np.isin(idx_test, use_idx)]
    out["bart_acc"] = float(accuracy_score(
        (y_pred_bart > 0.5).astype(int), y_test_bart))
    out["bart_f1"] = float(f1_score(
        (y_pred_bart > 0.5).astype(int), y_test_bart, zero_division=0))
    log.info("fidelity: BART-weighted acc=%.4f F1=%.4f (n_test=%d, capped=%s, "
             "ITE loop %.1fs)", out["bart_acc"], out["bart_f1"],
             out["n_test_bart"], capped, time.perf_counter() - t0)
    return out


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="DQN checkpoint .pt (best available at run time)")
    ap.add_argument("--estimator", choices=["bart", "surrogate"],
                    default="bart")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["hte", "ate"],
                    choices=["hte", "ate"])
    ap.add_argument("--policies", nargs="+", default=["heuristic", "dqn"],
                    choices=["heuristic", "dqn"],
                    help="policies to run (default: both)")
    ap.add_argument("--bart-test-cap", type=int, default=200)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=REPO_DIR)
    args = ap.parse_args()

    label = BART_LABEL if args.estimator == "bart" else SURROGATE_LABEL
    out_dir = args.output_dir or (BASE_DIR / "results" /
                                  "xrl-trained-vs-heuristic")
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "comparison_run.log", mode="w"),
                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("xrl-comparison")
    log.info(label)
    log.info("repo=%s seed=%s episodes=%s methods=%s estimator=%s",
             args.repo, args.seed, args.episodes, args.methods,
             args.estimator)
    log.info("checkpoint=%s", args.checkpoint)
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo,
                                capture_output=True, text=True,
                                timeout=30).stdout.strip()
        log.info("repo commit=%s", commit)
    except Exception as e:  # noqa: BLE001
        log.warning("could not read repo commit: %s", e)

    import sklearn, gymnasium, networkx, matplotlib  # noqa: E402
    log.info("versions: sklearn=%s gymnasium=%s networkx=%s matplotlib=%s "
             "numpy=%s pandas=%s", sklearn.__version__, gymnasium.__version__,
             networkx.__version__, matplotlib.__version__, np.__version__,
             __import__("pandas").__version__)
    if args.estimator == "bart":
        try:
            import rpy2.robjects as _ro  # noqa: F401
            log.info("rpy2 import OK")
            # NOTE: ro.r('requireNamespace(...)')[0] returns None in rpy2
            # (invisible return); use library() + packageVersion instead.
            _ro.r('library("dbarts")')
            _ver = list(_ro.r('as.character(packageVersion("dbarts"))'))[0]
            log.info("R dbarts OK, version=%s", _ver)
        except Exception as e:  # noqa: BLE001
            log.error("rpy2/dbarts check failed: %s", e)
            raise

    infer_network = load_repo_infer_network(args.estimator, args.repo)
    if args.estimator == "surrogate":
        import surrogate_bart  # noqa: E402
        surrogate_bart.N_DRAWS_DEFAULT = 200
        log.info("surrogate: SurrogateRegressor (RandomForest, n_draws=200)")

    sys.path.insert(0, str(SCRIPTS_DIR))
    from cartpole_policies import heuristic_policy, load_dqn_policy  # noqa: E402
    dqn_pi = load_dqn_policy(args.checkpoint)
    ckpt_meta = getattr(dqn_pi, "checkpoint", {}) or {}
    ckpt_eval = ckpt_meta.get("final_eval_100ep", {})
    log.info("DQN checkpoint hyperparams=%s final_eval_100ep=%s",
             ckpt_meta.get("hyperparams", {}), ckpt_eval)

    policies = [("heuristic", heuristic_policy), ("dqn", dqn_pi)]
    policies = [(n, p) for (n, p) in policies if n in args.policies]
    log.info("policies=%s", [n for n, _ in policies])
    rows = []
    details = {"label": label, "estimator": args.estimator,
               "checkpoint": str(args.checkpoint),
               "checkpoint_hyperparams": ckpt_meta.get("hyperparams", {}),
               "checkpoint_final_eval_100ep": ckpt_eval,
               "episodes": args.episodes, "seed": args.seed,
               "repo_commit": commit if "commit" in dir() else None,
               "runs": {}}
    scm_hte_by_policy = {}
    edges_by_policy_method = {}

    for pname, pi in policies:
        buffer, rollout_s, ep_lengths = collect_cartpole(
            pi, args.episodes, seed=args.seed)
        states, actions, rewards, states_new = buffer
        n_trans = len(states)
        log.info("policy=%s episodes=%d transitions=%d rollout_s=%.2f "
                 "mean_ep_len=%.1f action1_frac=%.3f",
                 pname, args.episodes, n_trans, rollout_s,
                 float(np.mean(ep_lengths)), float(np.mean(actions)))
        details["runs"][pname] = {
            "n_transitions": n_trans, "rollout_s": round(rollout_s, 2),
            "mean_ep_len": round(float(np.mean(ep_lengths)), 1),
            "action1_frac": round(float(np.mean(actions)), 4),
            "methods": {},
        }
        for method in args.methods:
            ate_table = None
            if method == "hte":
                G, scm, infer_s = run_hte(infer_network, buffer, args.seed)
            else:
                G, scm, infer_s, ate_table = run_ate(
                    infer_network, buffer, args.seed)
            edges = dbn_edges_to_strings(G)
            edges_c = collapsed_edges(G)
            causes = per_target_causes(G)
            log.info("run policy=%s method=%s n_edges=%d infer_s=%.2f "
                     "edges=%s", pname, method, len(edges), infer_s, edges)
            log.info("per-target causes policy=%s method=%s: %s",
                     pname, method, json.dumps(causes))
            png = graphs_dir / f"graph_{pname}_{method}_{args.episodes}ep.png"
            draw_dbn(G, png,
                     f"CartPole global causal graph ({pname}, {method}, "
                     f"{args.episodes} episodes)\n{label}")
            key = f"{pname}_{method}"
            edges_by_policy_method[key] = set(edges)
            details["runs"][pname]["methods"][method] = {
                "n_edges": len(edges), "edges": edges,
                "edges_collapsed": edges_c, "per_target_causes": causes,
                "infer_s": round(infer_s, 2),
            }
            if ate_table:
                details["runs"][pname]["methods"][method]["ate_table"] = \
                    ate_table
            rows.append({
                "policy": pname, "method": method,
                "n_transitions": n_trans, "n_edges": len(edges),
                "edges": "; ".join(edges),
                "edges_collapsed": "; ".join(edges_c),
                "per_target_causes": json.dumps(causes),
                "time_s": round(infer_s, 2),
                "rollout_s": round(rollout_s, 2),
            })
            if method == "hte":
                scm_hte_by_policy[pname] = (G, scm, buffer)
            # Incremental save after EVERY method (not just after the
            # policy): if the process dies mid-policy, the completed
            # methods are already on disk in partial_results.json.
            (out_dir / "partial_results.json").write_text(
                json.dumps({"rows": rows, "details": details}, indent=2,
                           default=str))

        # ---- fidelity for this policy NOW (hte scm is still in memory;
        # incremental save right after, so a later interruption loses at
        # most the current policy, not everything) ----
        G_hte, scm_hte, buffer = scm_hte_by_policy[pname]
        states, actions, _, _ = buffer
        fid = fidelity_eval_notebook(
            infer_network, scm_hte, G_hte, states, actions,
            args.estimator, log, bart_test_cap=args.bart_test_cap)
        details["runs"][pname]["fidelity_eval_notebook"] = fid
        for r in rows:
            if r["policy"] == pname and r["method"] == "hte":
                r.update({
                    "fid_valid_steps": fid.get("valid_steps"),
                    "fid_n_test_dt_lr": fid.get("n_test_dt_lr"),
                    "dt_acc": round(fid["dt_acc"], 4) if "dt_acc" in fid else None,
                    "dt_f1": round(fid["dt_f1"], 4) if "dt_f1" in fid else None,
                    "lr_acc": round(fid["lr_acc"], 4) if "lr_acc" in fid else None,
                    "lr_f1": round(fid["lr_f1"], 4) if "lr_f1" in fid else None,
                    "bart_acc": round(fid["bart_acc"], 4) if "bart_acc" in fid else None,
                    "bart_f1": round(fid["bart_f1"], 4) if "bart_f1" in fid else None,
                    "bart_n_test": fid.get("n_test_bart"),
                    "bart_test_capped": fid.get("bart_test_capped"),
                    "global_cause_push_parents": json.dumps(
                        fid.get("global_cause_push_parents")),
                })
        (out_dir / "partial_results.json").write_text(
            json.dumps({"rows": rows, "details": details}, indent=2,
                       default=str))
        log.info("incremental save: %s (policy %s done)",
                 out_dir / "partial_results.json", pname)
        # MEMORY: drop this policy's hte scm (5 fitted BART models on
        # DQN data are GBs) before the next policy; its numbers are
        # already in rows/details/partial_results.json.
        del scm_hte_by_policy[pname]
        del G_hte, scm_hte, buffer
        gc.collect()

    # ---- trained-vs-heuristic graph comparison per method ----
    comparison = {}
    for method in args.methods:
        h = edges_by_policy_method.get(f"heuristic_{method}", set())
        d = edges_by_policy_method.get(f"dqn_{method}", set())
        comparison[method] = {
            "heuristic_n_edges": len(h), "dqn_n_edges": len(d),
            "jaccard": round(jaccard(h, d), 4),
            "symdiff": len(h ^ d),
            "only_heuristic": sorted(h - d),
            "only_dqn": sorted(d - h),
            "common": sorted(h & d),
        }
        log.info("comparison %s: Jaccard=%.4f symdiff=%d only_heur=%s "
                 "only_dqn=%s", method, comparison[method]["jaccard"],
                 comparison[method]["symdiff"],
                 comparison[method]["only_heuristic"],
                 comparison[method]["only_dqn"])
    details["comparison_trained_vs_heuristic"] = comparison

    # ---- write outputs ----
    fieldnames = ["policy", "method", "n_transitions", "n_edges", "edges",
                  "edges_collapsed", "per_target_causes", "time_s",
                  "rollout_s", "fid_valid_steps", "fid_n_test_dt_lr",
                  "dt_acc", "dt_f1", "lr_acc", "lr_f1", "bart_acc",
                  "bart_f1", "bart_n_test", "bart_test_capped",
                  "global_cause_push_parents"]
    with open(out_dir / "comparison_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "per_run_details.json").write_text(
        json.dumps(details, indent=2, default=str))

    md = []
    md.append("# CausalXRL extension D — XRL trained-vs-heuristic "
              "comparison (CartPole)\n")
    md.append(f"> **{label}**\n")
    md.append(f"> Checkpoint DQN: `{args.checkpoint}` "
              f"(final eval 100ep from checkpoint metadata: {ckpt_eval}). "
              f"Episodes: {args.episodes} per policy, seed {args.seed}. "
              "Data: gymnasium CartPole-v1 rollouts (repo `play()` uses gym "
              "0.20, incompatible with Python 3.12, so gymnasium is used "
              "for data collection only; the repo's transition-recording "
              "rule is kept). `hte` = repo `infer_causes_hte` (70/30 "
              "split); `ate` = repo `infer_causes_ate` (50/50 split). "
              "Fidelity is eval-notebook style: steps whose local causal "
              "graph has >1 cause for `push`, 60/40 split "
              "(random_state=10); DT/LR on the full held-out set, "
              "BART-weighted on at most "
              f"{args.bart_test_cap} test steps (see per-run details). "
              "Global-cause set for BART-weighted = learned parents of "
              "`push` in the hte graph (not the notebook's hard-coded "
              "list). Time = structure-learning seconds (infer only, "
              "excluding rollout and fidelity).\n")
    md.append("| policy | method | n_trans | n_edges | time_s (infer) | "
              "DT acc / F1 | LR acc / F1 | BART acc / F1 (n_test) | edges |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        dt = (f"{r.get('dt_acc')} / {r.get('dt_f1')}"
              if r.get("dt_acc") is not None else "—")
        lr = (f"{r.get('lr_acc')} / {r.get('lr_f1')}"
              if r.get("lr_acc") is not None else "—")
        bart = (f"{r.get('bart_acc')} / {r.get('bart_f1')} "
                f"(n={r.get('bart_n_test')})"
                if r.get("bart_acc") is not None else "—")
        md.append(f"| {r['policy']} | {r['method']} | {r['n_transitions']} "
                  f"| {r['n_edges']} | {r['time_s']} | {dt} | {lr} | {bart} "
                  f"| {r['edges']} |")
    md.append("")
    md.append("## Trained (DQN) vs heuristic — graph differences\n")
    for method, comp in comparison.items():
        md.append(f"### {method}\n")
        md.append(f"- Jaccard = {comp['jaccard']}, symmetric difference = "
                  f"{comp['symdiff']} edges "
                  f"(heuristic {comp['heuristic_n_edges']} edges, DQN "
                  f"{comp['dqn_n_edges']} edges).")
        md.append(f"- Only in heuristic: "
                  f"`{'; '.join(comp['only_heuristic']) or '—'}`")
        md.append(f"- Only in DQN: `{'; '.join(comp['only_dqn']) or '—'}`")
        md.append(f"- Common: `{'; '.join(comp['common']) or '—'}`")
        md.append("")
    md.append("## Per-target causes\n")
    for r in rows:
        md.append(f"- **{r['policy']} / {r['method']}**: "
                  f"`{r['per_target_causes']}`")
    (out_dir / "comparison_table.md").write_text("\n".join(md) + "\n")
    log.info("wrote %s", out_dir / "comparison_table.csv")
    log.info("wrote %s", out_dir / "comparison_table.md")
    log.info("wrote %s", out_dir / "per_run_details.json")
    log.info("DONE")


if __name__ == "__main__":
    main()
