#!/usr/bin/env python3
"""CausalXRL extension A -- ablation / stability check on CartPole.

SURROGATE ESTIMATOR (RandomForest, 200 trees), NOT BART/dbarts
-- stability check only, to be re-run with real BART when available.

What it does
------------
1. Collects CartPole-v1 rollouts with the repo's exact heuristic policy
   (CartPole.ipynb):  ``1 if 0.4*(x/2.4) + 0.6*(theta/2.095) >= 0 else 0``.
   The repo's ``play()`` uses the old gym 0.20 API (``env.reset()`` ->
   state, ``env.step()`` -> 4 values), which is incompatible with
   Python 3.12, so data collection uses gymnasium here (same
   transition-recording rule as ``play()``: a transition is recorded
   only when the step is NOT terminal). This substitution is stated
   on every output.
2. Runs the repo's causal-discovery pipeline from
   ``cxrl/lib/causal/infer_network.py`` (commit ae9096e, read-only
   clone at /tmp/causal-xrl-muc4) with ``BARTRegressor`` monkeypatched
   to the surrogate in ``surrogate_bart.py``. No repo file is edited.
   ``rpy2`` is stubbed in ``sys.modules`` (R is not installed) so the
   repo modules import; the stub is never called because the real
   ``BARTRegressor`` is never instantiated.
   * method ``hte`` -> repo ``infer_DBN(method='ate')``, which calls
     ``infer_causes_hte`` (70/30 split).
   * method ``ate`` -> repo ``infer_causes_ate`` (50/50 split), called
     per target by a loop that replicates ``infer_DBN``'s replay
     construction and edge building (``infer_DBN`` itself has no
     ``ate`` dispatch -- it only dispatches ``hte`` and
     ``expectation``/CITestFast).
3. Ablation grid: nepisodes in [50, 100, 200] x method in [hte, ate].
4. Behaviour-prediction (fidelity-style) check per run: DecisionTree
   + LogisticRegression trained on (state -> action) to predict the
   next action, held-out accuracy/F1. This mirrors the paper's
   Table 1 fidelity idea and is labelled surrogate-pipeline fidelity,
   NOT the paper's BART numbers.
5. CI-threshold sensitivity on the 200-episode ``hte`` run (and, as a
   bonus, the 200-episode ``ate`` run): edges at 90% vs 95% CI,
   recomputed from the SAME stored per-tree draws.
6. Cross-episode-count graph comparison per method: Jaccard and
   edge symmetric difference vs the 200-episode graph; stable vs
   unstable edges in words.

Reproducibility: seed 0 everywhere (numpy global RNG seeded before
each inference call because the repo's ``train_test_split`` calls use
``random_state=None``; RandomForest ``random_state=0``;
per-tree draws (200 trees) stand in for BART posterior draws;
gymnasium env seeded 0 on episode 0). Rollouts for
fewer episodes are a prefix of the 200-episode rollout.

Usage:
    .venv-ablation/bin/python -B scripts/run_ablation.py \
        [--episodes 50 100 200] [--methods hte ate] [--seed 0] \
        [--n-draws 200] [--output-dir results/ablation]
"""

import argparse
import json
import logging
import os
# Limit BLAS/OpenMP threads: the VM has ~1.5 shared CPUs and a parallel
# agent may be training on CPU; uncapped OpenMP threads in
# HistGradientBoostingRegressor (first surrogate attempt) thrashed badly under load (one fit took
# >40 s at load avg ~6, vs ~2.5 s with threads capped). Must be set
# before numpy/sklearn are imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import random
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # do not write __pycache__ into the repo clone

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_DIR = Path("/tmp/causal-xrl-muc4")
LABEL = ("SURROGATE ESTIMATOR (RandomForest, 200 trees), NOT BART/dbarts"
         " - stability check only, to be re-run with real BART when available")

FEATURE_NAMES = ["cart_position", "cart_velocity", "pole_angle",
                 "pole_velocity", "push"]
N_FEATURES = len(FEATURE_NAMES)


def heuristic_policy(state):
    """Exact heuristic from the repo's CartPole.ipynb."""
    state = np.asarray(state, dtype=np.float64)
    return 1 if 0.4 * (state[0] / 2.4) + 0.6 * (state[2] / 2.095) >= 0 else 0


# ----------------------------------------------------------------------
# rpy2 stub (R is not installed; the real BARTRegressor is never used)
# ----------------------------------------------------------------------
def _install_rpy2_stub():
    """Install dummy rpy2 modules so ``cxrl.lib.models.bart_regressor``
    can be imported. Only the import-time names are provided; no R code
    is ever executed because the surrogate replaces BARTRegressor."""
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


def load_repo_infer_network():
    """Import the repo's infer_network with BARTRegressor monkeypatched
    to the surrogate. Returns the module. Does not modify repo files."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    from surrogate_bart import SurrogateRegressor  # noqa: E402
    _install_rpy2_stub()
    sys.path.insert(0, str(REPO_DIR))
    import cxrl.lib.causal.infer_network as infer_network  # noqa: E402
    infer_network.BARTRegressor = SurrogateRegressor
    return infer_network, SurrogateRegressor


# ----------------------------------------------------------------------
# data collection (gymnasium; same recording rule as repo play())
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
# graph helpers
# ----------------------------------------------------------------------
def dbn_edges_to_strings(G):
    return sorted(f"{u} -> {v}" for u, v in G.edges())


def collapsed_edges(G):
    """Collapse a DBN (t -> t+1) to a same-time causal graph.

    Same rule as the repo's ``DBN_to_causal`` (strip the '+1' from the
    target node name), except the repo's self-loop exclusion compares
    against ``f"state_{n}_t+1"`` where ``n`` already carries the
    ``state_`` prefix, so it never fires; here the self-loop check is
    done correctly (source == stripped target).
    """
    edges = set()
    for u, v in G.edges():
        target = v[:-2] if v.endswith("t+1") else v
        if u == target:
            continue
        edges.add(f"{u} -> {target}")
    return sorted(edges)


def per_target_causes(G):
    """{target_node_t+1: [source_node_t, ...]} from DBN edges."""
    out = {name: [] for name in FEATURE_NAMES}
    for u, v in G.edges():
        base = v[:-4] if v.endswith("_t+1") else v  # 'state_X_t+1' -> 'state_X'
        # map back to feature name via suffix match
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
    # labels offset outside the node circles so they stay readable
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
# fidelity check
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
# inference runners
# ----------------------------------------------------------------------
def run_hte(infer_network, replay_buffer, seed):
    """Repo infer_DBN(method='ate') -> infer_causes_hte per target."""
    np.random.seed(seed)
    random.seed(seed)
    t0 = time.perf_counter()
    G, scm = infer_network.infer_DBN(
        replay_buffer, FEATURE_NAMES, method="ate",
        one_hot_actions=False, repr_integration=False, verbose=False)
    return G, scm, time.perf_counter() - t0


def run_ate(infer_network, replay_buffer, seed):
    """Per-target repo infer_causes_ate, replicating infer_DBN's replay
    construction and edge building (infer_DBN has no 'ate' dispatch)."""
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
        causes, model = infer_network.infer_causes_ate(replay_t, target,
                                                        verbose=False)
        scm[i_repr] = model
        for c in causes:
            prefix_t = "action" if c >= n_states else "state"
            prefix_tp1 = "action" if i_repr >= n_states else "state"
            G.add_edge(f"{prefix_t}_{FEATURE_NAMES[c]}_t",
                       f"{prefix_tp1}_{FEATURE_NAMES[i_repr]}_t+1")
    return G, scm, time.perf_counter() - t0


# ----------------------------------------------------------------------
# CI sensitivity from stored per-tree draws
# ----------------------------------------------------------------------
def ci_sensitivity(scm, method):
    """Edges at 90% vs 95% CI from the same stored per-tree draws."""
    out = {}
    for level in (0.90, 0.95):
        edges = set()
        per_target = {}
        for i_repr in range(N_FEATURES):
            model = scm[i_repr]
            causes = []
            for c in range(N_FEATURES):
                if method == "hte":
                    sig, _ = model.hte_from_draws(c, level=level)
                else:
                    _, _, _, sig = model.ate_from_draws(c, level=level)
                if sig:
                    causes.append(c)
            per_target[FEATURE_NAMES[i_repr]] = [
                FEATURE_NAMES[c] for c in causes]
            for c in causes:
                prefix_t = "action" if c >= 4 else "state"
                prefix_tp1 = "action" if i_repr >= 4 else "state"
                edges.add(f"{prefix_t}_{FEATURE_NAMES[c]}_t"
                          f" -> {prefix_tp1}_{FEATURE_NAMES[i_repr]}_t+1")
        out[level] = {"edges": sorted(edges), "per_target": per_target}
    return out


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
    ap.add_argument("--n-draws", type=int, default=200,
                    help="RandomForest trees = ITE draws per run (default 200)")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=REPO_DIR)
    args = ap.parse_args()

    out_dir = args.output_dir or (BASE_DIR / "results" / "ablation")
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "ablation_run.log", mode="w"),
                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("ablation")
    log.info(LABEL)
    log.info("repo=%s seed=%s n_draws=%s", args.repo, args.seed,
             args.n_draws)
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

    infer_network, SurrogateRegressor = load_repo_infer_network()
    import surrogate_bart  # noqa: E402
    # infer_network instantiates BARTRegressor() with no args, so the
    # draw count is passed via the module default (read at
    # construction time by SurrogateRegressor.__init__).
    surrogate_bart.N_DRAWS_DEFAULT = args.n_draws
    log.info("surrogate: %s (RandomForest, n_draws=%s)",
             SurrogateRegressor.__name__, args.n_draws)

    rows = []
    details = {}
    scm_200 = {}
    for nep in args.episodes:
        buffer, rollout_s, ep_lengths = collect_cartpole(nep, seed=args.seed)
        states, actions, rewards, states_new = buffer
        n_trans = len(states)
        log.info("episodes=%d transitions=%d rollout_s=%.2f "
                 "mean_ep_len=%.1f action1_frac=%.3f",
                 nep, n_trans, rollout_s, float(np.mean(ep_lengths)),
                 float(np.mean(actions)))
        fid = fidelity_check(states, actions, seed=args.seed)
        log.info("fidelity nep=%d DT acc=%.4f F1=%.4f | LR acc=%.4f F1=%.4f",
                 nep, fid["dt_acc"], fid["dt_f1"], fid["lr_acc"], fid["lr_f1"])
        for method in args.methods:
            if method == "hte":
                G, scm, infer_s = run_hte(infer_network, buffer, args.seed)
            else:
                G, scm, infer_s = run_ate(infer_network, buffer, args.seed)
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
            extra = {"per_target_causes": causes}
            if method == "ate":
                ate_table = {}
                for i_repr in range(N_FEATURES):
                    model = scm[i_repr]
                    ate_table[FEATURE_NAMES[i_repr]] = {
                        FEATURE_NAMES[c]: [round(v, 4) for v in
                                           model._ate_point[c]]
                        for c in sorted(model._ate_point)}
                extra["ate_table"] = ate_table
            else:
                hte_table = {}
                for i_repr in range(N_FEATURES):
                    model = scm[i_repr]
                    hte_table[FEATURE_NAMES[i_repr]] = {
                        FEATURE_NAMES[c]: {
                            "n_clusters": len(np.unique(
                                model._hte_clusters[c])),
                            "cluster_sizes": model._hte_cluster_size.get(c, {}),
                        } for c in sorted(model._hte_clusters)}
                extra["hte_cluster_table"] = hte_table
            details[key] = {
                "nepisodes": nep, "method": method,
                "n_transitions": n_trans, "rollout_s": round(rollout_s, 2),
                "mean_ep_len": round(float(np.mean(ep_lengths)), 1),
                "action1_frac": round(float(np.mean(actions)), 4),
                "infer_s": round(infer_s, 2),
                "edges": edges, "edges_collapsed": edges_c,
                "n_edges": len(edges),
                "fidelity": {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in fid.items()},
                **extra,
            }
            rows.append({
                "nepisodes": nep, "method": method,
                "n_transitions": n_trans, "n_edges": len(edges),
                "edges": "; ".join(edges),
                "edges_collapsed": "; ".join(edges_c),
                "per_target_causes": json.dumps(causes),
                "time_s": round(infer_s, 2), "rollout_s": round(rollout_s, 2),
                "dt_acc": round(fid["dt_acc"], 4), "dt_f1": round(fid["dt_f1"], 4),
                "lr_acc": round(fid["lr_acc"], 4), "lr_f1": round(fid["lr_f1"], 4),
            })
            if nep == 200:
                scm_200[method] = scm

    # ---- CI sensitivity (200-ep runs, same stored draws) ----
    sensitivity = {}
    for method, scm in scm_200.items():
        sensitivity[method] = ci_sensitivity(scm, method)
        for level in (0.90, 0.95):
            log.info("CI sensitivity 200ep %s level=%.2f n_edges=%d edges=%s",
                     method, level, len(sensitivity[method][level]["edges"]),
                     sensitivity[method][level]["edges"])
    details["ci_sensitivity_200ep"] = {
        method: {str(level): v for level, v in sens.items()}
        for method, sens in sensitivity.items()}

    # ---- cross-episode comparison per method ----
    comparison = {}
    for method in args.methods:
        by_ep = {r["nepisodes"]: set(r["edges"].split("; ")) if r["edges"]
                 else set() for r in rows if r["method"] == method}
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

    # ---- write outputs ----
    import csv
    fieldnames = ["nepisodes", "method", "n_transitions", "n_edges", "edges",
                  "edges_collapsed", "per_target_causes", "time_s",
                  "rollout_s", "dt_acc", "dt_f1", "lr_acc", "lr_f1"]
    with open(out_dir / "ablation_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "per_run_details.json").write_text(
        json.dumps(details, indent=2, default=str))

    md = []
    md.append("# CausalXRL extension A — ablation / stability check (CartPole)\n")
    md.append(f"> **{LABEL}**\n")
    md.append("> Data: gymnasium CartPole-v1 rollouts with the repo's exact "
              "heuristic policy (repo `play()` uses gym 0.20, incompatible "
              "with Python 3.12, so gymnasium is used for data collection "
              "only; the repo's transition-recording rule is kept). "
              "Seed 0. `hte` = repo `infer_causes_hte` (70/30 split); "
              "`ate` = repo `infer_causes_ate` (50/50 split). "
              "Numeric ATE contrast is +1 SD vs −1 SD (see "
              "`scripts/surrogate_bart.py` deviation #1). Fidelity = "
              "DecisionTree / LogisticRegression predicting the next action "
              "from the state (80/20 split) — surrogate-pipeline fidelity, "
              "NOT the paper's Table 1 BART numbers.\n")
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
    md.append("## Graph comparison vs the 200-episode graph (same method)\n")
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

    ci_md = ["# CI-threshold sensitivity — 200-episode runs\n",
             f"> **{LABEL}**\n",
             "> Edges at 90% vs 95% CI recomputed from the SAME stored "
             "per-tree draws (200 trees); no re-run.\n"]
    for method, sens in sensitivity.items():
        e90 = sens[0.90]["edges"]
        e95 = sens[0.95]["edges"]
        ci_md.append(f"## {method} (200 episodes)\n")
        ci_md.append(f"- **95% CI**: {len(e95)} edges: `{'; '.join(e95)}`")
        ci_md.append(f"- **90% CI**: {len(e90)} edges: `{'; '.join(e90)}`")
        ci_md.append(f"- Edges gained at 90% (not at 95%): "
                     f"`{'; '.join(sorted(set(e90) - set(e95))) or '—'}`")
        ci_md.append(f"- Edges lost at 90% (present at 95%): "
                     f"`{'; '.join(sorted(set(e95) - set(e90))) or '—'}`")
        ci_md.append("")
    (out_dir / "ci_sensitivity.md").write_text("\n".join(ci_md) + "\n")
    log.info("wrote %s", out_dir / "ablation_table.csv")
    log.info("wrote %s", out_dir / "ablation_table.md")
    log.info("wrote %s", out_dir / "ci_sensitivity.md")
    log.info("wrote %s", out_dir / "per_run_details.json")
    log.info("DONE")


if __name__ == "__main__":
    main()
