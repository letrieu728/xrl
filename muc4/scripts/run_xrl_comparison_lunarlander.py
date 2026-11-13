#!/usr/bin/env python3
"""CausalXRL extension D — XRL trained-vs-heuristic comparison (LunarLander).

Adaptation of scripts/run_xrl_comparison.py (CartPole) to LunarLander,
following the author's eval notebook
(cxrl/jupyter/eval/eval_fidelity_Lunar Lander.ipynb) and the Mục 3
LunarLander protocol. Differences from CartPole, all explicit:

* Representation: 8 state variables + 4 one-hot actions = 12 columns /
  12 DBN targets (CartPole: 4 + 1 binary = 5). Fixed 4-column one-hot
  encoding (notebook cell 8), NOT pd.get_dummies, so the columns always
  match feature_names even if a run missed an action (all 4 actions do
  occur in both policies here; the fixed encoding is equivalent to
  get_dummies in that case).
* Policies: heuristic = PD controller from LunarLander.ipynb (see
  lunarlander_policies.py); dqn = LunarLander DQN checkpoint
  (MLP 8 -> 128 -> 128 -> 4, best_checkpoint.pt @ ep 850).
* Episodes: 30 per policy, seed 0 (author eval notebook NEPISODES=30;
  Mục 3 also 30). LunarLander episodes are ~10x longer than CartPole
  heuristic episodes, so 30 episodes give a comparable transition
  volume (~7.6k heuristic / ~13.1k DQN) to CartPole's 200 episodes
  (6.3k heuristic). 200 LunarLander episodes were ruled out by time:
  the Mục 3 anchor (7,382 rows -> 1,086 s HTE) scales to ~2 h per
  policy-method at 200 episodes, plus the OOM risk seen at ~100k rows
  in the CartPole DQN runs.
* Fidelity is MULTICLASS (4 actions): valid steps = any action SCM with
  >1 non-self local cause (notebook cell 11); 60/40 split,
  random_state=10, stratify=y (cell 15); DT/LR/BART-weighted report
  accuracy + weighted F1 (paper section 5.2 metric) + macro F1 +
  per-class scores. BART-weighted follows notebook cells 21-23: global
  causes = predecessors of the action node in the collapsed causal
  graph (repo DBN_to_causal, self excluded); one BART fit per
  (action, global cause); per test row, ITE-weighted score per action,
  train-prior fallback when total weight is 0, argmax over 4 actions.
  NO test cap (CartPole capped BART-weighted at 200 test steps): a
  timing probe measured ITE at ~7 ms/row, so the full test set is
  feasible here; the cap difference is recorded in the outputs.
* KNOWN QUIRK (author's, kept, not silently fixed): the HTE trees are
  fit on STANDARDIZED estimation data (infer_causes_hte standardizes
  num_cols), but local_treatments queries are RAW replay rows, exactly
  as the author notebook does. Local-cause validity therefore reflects
  the author's implementation, including this scale mismatch.
* STOCHASTICITY: repo bart_train() passes NO seed to dbarts (only
  bart_sample has seed=99), and setup_R sets no R seed, so every BART
  fit is a fresh stochastic realisation even with numpy seeded. Graph
  causes and fidelity-stage causes (a separate HTE refit, see below)
  may differ by a few edges between fits; both are recorded.
* REPLAY CONSTRUCTION (repo infer_DBN, kept): replay_t/replay_tp1 are
  built by shifting the CONCATENATED transition list, so the last
  transition of one episode pairs with the first of the next at
  episode boundaries (same quirk as the CartPole script).

Checkpointing (VM restarts are frequent): this script runs in stages,
each independently resumable:

* ``graphs`` — one process per (policy, method); loops the 12 targets
  in index order, writing target_<policy>_<method>_<idx>.json
  IMMEDIATELY after each target (causes, elapsed, run id, timestamp).
  Completed targets are skipped on re-run; numpy RNG state is replayed
  for skipped targets (dummy train_test_split calls with the same
  test_size) so the remaining targets get the same splits as an
  uninterrupted run (R fits still differ — see stochasticity note).
  Each fitted model is dropped right after its target (memory).
* ``fidelity`` — one process per policy; refits HTE models (same
  replay, same seed protocol) because fidelity needs the fitted
  models for local_treatments/ITE, which the per-target graph
  checkpoints do not store (same separation as Mục 3's
  targets/masks/scores stages). Writes fidelity_<policy>.json (+ a
  scores .npz in the work dir) only when complete; a killed run is
  re-run whole (fresh stochastic fit, disclosed).
* ``assemble`` — no BART; builds graphs, PNGs, comparison tables and
  per-run details from the checkpoints.

Usage (from the muc4 working dir; --repo points at the read-only repo
copy, commit ae9096e; --replay at the .npz from collect_lunarlander.py):

    R_LIBS=~/workspace/R-lib .venv-ablation/bin/python -B \\
        scripts/run_xrl_comparison_lunarlander.py graphs \\
        --policy heuristic --method hte --replay <replay_heuristic.npz> \\
        --ckpt-dir <dir> --repo <repo-causal-xrl>
    ... (hte/ate x heuristic/dqn) ...
    R_LIBS=~/workspace/R-lib .venv-ablation/bin/python -B \\
        scripts/run_xrl_comparison_lunarlander.py fidelity \\
        --policy heuristic --replay <replay_heuristic.npz> \\
        --out-dir <dir> --repo <repo-causal-xrl>
    ... (both policies) ...
    .venv-ablation/bin/python -B scripts/run_xrl_comparison_lunarlander.py \\
        assemble --ckpt-dir <dir> --out-dir <results-dir>
"""

import argparse
import csv
import gc
import json
import logging
import os
os.environ.setdefault("R_HOME", "/usr/lib/R")
os.environ.setdefault("R_LIBS", os.path.expanduser("~/workspace/R-lib"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

FEATURE_NAMES = ["x", "y", "vx", "vy", "angle", "angular_velocity",
                 "left_contact", "right_contact", "do_nothing",
                 "fire_left_engine", "fire_main_engine", "fire_right_engine"]
STATE_NAMES = FEATURE_NAMES[:8]
ACTION_NAMES = FEATURE_NAMES[8:]
N_STATES = 8
N_FEATURES = 12
ACTION_INDICES = list(range(8, 12))
# Notebook cell 23: additive ITE only for the 6 continuous state cols.
NUM_COLS_FIDELITY = [0, 1, 2, 3, 4, 5]

BART_LABEL = "BART (R/dbarts via rpy2) - real CausalXRL estimator"


def build_replay(replay_path):
    """Load a collect_lunarlander.py .npz; return (replay_t, replay_tp1,
    replay_full, meta). Fixed 4-col one-hot actions (notebook cell 8)."""
    r = np.load(replay_path)
    states = np.asarray(r["states"], dtype=float)
    actions = np.asarray(r["actions"]).astype(int)
    assert states.shape[1] == 8, states.shape
    assert set(np.unique(actions)) <= {0, 1, 2, 3}
    onehot = np.zeros((len(actions), 4), dtype=float)
    onehot[np.arange(len(actions)), actions] = 1.0
    full = np.concatenate((states, onehot), axis=1)
    return full[:-1], full[1:], full, {
        "n_transitions": int(len(actions)),
        "action_counts": np.bincount(actions, minlength=4).tolist()}


def node_name(idx, tplus1):
    prefix = "action" if idx >= N_STATES else "state"
    return f"{prefix}_{FEATURE_NAMES[idx]}_t+1" if tplus1 else \
        f"{prefix}_{FEATURE_NAMES[idx]}_t"


def build_dbn(causes_by_target):
    """DBN graph with repo infer_DBN edge naming from {target: [causes]}."""
    import networkx as nx
    G = nx.DiGraph()
    for tgt, causes in causes_by_target.items():
        for c in causes:
            G.add_edge(node_name(c, False), node_name(tgt, True))
    return G


def dbn_edges_to_strings(G):
    return sorted(f"{u} -> {v}" for u, v in G.edges())


def collapsed_edges(G):
    """Feature-level edges (self-loops KEPT: x->x autoregression and
    action->action persistence are headline findings)."""
    edges = set()
    for u, v in G.edges():
        src = u.replace("state_", "").replace("action_", "")[:-2]
        tgt = v.replace("state_", "").replace("action_", "")[:-4]
        edges.add(f"{src} -> {tgt}")
    return sorted(edges)


def per_target_causes(G):
    # Exact feature match after stripping the prefix: suffix matching
    # would confuse x/vx and y/vy.
    out = {name: [] for name in FEATURE_NAMES}
    for u, v in G.edges():
        base = v[:-4] if v.endswith("_t+1") else v
        if base.startswith("state_"):
            feat = base[len("state_"):]
        elif base.startswith("action_"):
            feat = base[len("action_"):]
        else:
            continue
        if feat in out:
            out[feat].append(u)
    return {k: sorted(v) for k, v in out.items()}


def draw_dbn(G, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    t_nodes = [node_name(i, False) for i in range(N_FEATURES)]
    tp1_nodes = [node_name(i, True) for i in range(N_FEATURES)]
    pos = {}
    for i, n in enumerate(t_nodes):
        pos[n] = (0.0, len(t_nodes) - 1 - i)
    for i, n in enumerate(tp1_nodes):
        pos[n] = (1.0, len(tp1_nodes) - 1 - i)
    H = nx.DiGraph()
    H.add_nodes_from(t_nodes + tp1_nodes)
    H.add_edges_from(G.edges())
    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_nodes(H, pos, nodelist=t_nodes, node_color="#cfe8ff",
                           node_size=2200, ax=ax)
    nx.draw_networkx_nodes(H, pos, nodelist=tp1_nodes, node_color="#ffe4c2",
                           node_size=2200, ax=ax)
    nx.draw_networkx_edges(H, pos, arrows=True, arrowsize=16,
                           arrowstyle="-|>", connectionstyle="arc3,rad=0.08",
                           ax=ax)
    pos_left = {n: (x - 0.06, y) for n, (x, y) in pos.items()
                if n in t_nodes}
    pos_right = {n: (x + 0.06, y) for n, (x, y) in pos.items()
                 if n in tp1_nodes}
    nx.draw_networkx_labels(H, pos_left, labels={n: n for n in t_nodes},
                            font_size=7.5, horizontalalignment="right", ax=ax)
    nx.draw_networkx_labels(H, pos_right, labels={n: n for n in tp1_nodes},
                            font_size=7.5, horizontalalignment="left", ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.62, 1.62)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def load_infer_network(repo_dir):
    import sklearn.tree._tree as _sk_tree
    if not hasattr(_sk_tree, "DTYPE"):
        _sk_tree.DTYPE = np.float32
    sys.path.insert(0, str(repo_dir))
    import cxrl.lib.causal.infer_network as infer_network  # noqa: E402
    return infer_network


def check_bart(log):
    import rpy2.robjects as ro  # noqa: F401
    log.info("rpy2 import OK")
    ro.r('library("dbarts")')
    ver = list(ro.r('as.character(packageVersion("dbarts"))'))[0]
    log.info("R dbarts OK, version=%s", ver)
    return ver


def replay_rng_for_skipped(replay_t, target_col, test_size):
    """Advance numpy RNG exactly as infer_causes_*'s train_test_split
    would, so resumed runs keep the uninterrupted split sequence."""
    from sklearn.model_selection import train_test_split
    train_test_split(replay_t, target_col, test_size=test_size)


# ----------------------------------------------------------------------
# graphs stage
# ----------------------------------------------------------------------
def cmd_graphs(args, log):
    run_id = uuid.uuid4().hex[:8]
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    infer_network = load_infer_network(args.repo)
    check_bart(log)
    replay_t, replay_tp1, _, meta = build_replay(args.replay)
    log.info("policy=%s method=%s replay_t=%s n_transitions=%d "
             "action_counts=%s run_id=%s", args.policy, args.method,
             replay_t.shape, meta["n_transitions"], meta["action_counts"],
             run_id)
    test_size = 0.3 if args.method == "hte" else 0.5
    infer_fn = (infer_network.infer_causes_hte if args.method == "hte"
                else infer_network.infer_causes_ate)
    np.random.seed(args.seed)
    random.seed(args.seed)
    targets = args.targets
    for idx in range(N_FEATURES):
        ckpt_path = ckpt_dir / f"target_{args.policy}_{args.method}_{idx:02d}.json"
        if ckpt_path.exists():
            log.info("target %d (%s): SKIP (checkpoint exists)", idx,
                     FEATURE_NAMES[idx])
            if idx in targets:
                pass  # explicit --targets still respects checkpoints
            replay_rng_for_skipped(replay_t, replay_tp1[:, idx], test_size)
            continue
        if idx not in targets:
            replay_rng_for_skipped(replay_t, replay_tp1[:, idx], test_size)
            continue
        t0 = time.perf_counter()
        causes, model = infer_fn(replay_t, replay_tp1[:, idx], verbose=False)
        elapsed = time.perf_counter() - t0
        rec = {
            "policy": args.policy, "method": args.method,
            "target_index": idx, "target": FEATURE_NAMES[idx],
            "causes": [int(c) for c in causes],
            "cause_names": [FEATURE_NAMES[c] for c in causes],
            "elapsed_seconds": round(elapsed, 2),
            "n_replay_rows": int(replay_t.shape[0]),
            "test_size": test_size, "seed": args.seed,
            "run_id": run_id,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "replay": str(args.replay),
        }
        ckpt_path.write_text(json.dumps(rec, indent=2))
        log.info("target %d (%s): causes=%s elapsed=%.1fs -> %s",
                 idx, FEATURE_NAMES[idx], rec["cause_names"], elapsed,
                 ckpt_path.name)
        del model
        gc.collect()
    log.info("graphs stage done policy=%s method=%s", args.policy, args.method)


# ----------------------------------------------------------------------
# fidelity stage
# ----------------------------------------------------------------------
def cmd_fidelity(args, log):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"fidelity_{args.policy}.json"
    if out_json.exists() and not args.force:
        log.info("fidelity JSON exists, skipping (use --force to redo): %s",
                 out_json)
        return
    infer_network = load_infer_network(args.repo)
    dbarts_ver = check_bart(log)
    replay_t, replay_tp1, replay_full, meta = build_replay(args.replay)
    log.info("policy=%s replay_t=%s n_transitions=%d action_counts=%s",
             args.policy, replay_t.shape, meta["n_transitions"],
             meta["action_counts"])

    # --- HTE refit (fidelity-stage scm; see module docstring) ---
    np.random.seed(args.seed)
    random.seed(args.seed)
    scm = {}
    fid_causes = {}
    fit_times = {}
    t_fit0 = time.perf_counter()
    for idx in range(N_FEATURES):
        t0 = time.perf_counter()
        causes, model = infer_network.infer_causes_hte(
            replay_t, replay_tp1[:, idx], verbose=False)
        fit_times[FEATURE_NAMES[idx]] = round(time.perf_counter() - t0, 2)
        fid_causes[idx] = [int(c) for c in causes]
        scm[idx] = model
        log.info("fidelity HTE target %d (%s): causes=%s (%.1fs)", idx,
                 FEATURE_NAMES[idx], [FEATURE_NAMES[c] for c in causes],
                 fit_times[FEATURE_NAMES[idx]])
        if idx not in ACTION_INDICES:
            # local_treatments needs only the small HTE trees, never the
            # fitted bart machine; drop it (CartPole memory fix).
            for _attr in ("_bart_machine", "_model"):
                if hasattr(model, _attr):
                    try:
                        delattr(model, _attr)
                    except Exception:  # noqa: BLE001
                        pass
            gc.collect()
    log.info("fidelity HTE refit total %.1fs", time.perf_counter() - t_fit0)

    # --- valid steps (notebook cell 11) ---
    t0 = time.perf_counter()
    valid_steps = []
    for step in range(replay_full.shape[0] - 1):
        query = replay_full[step:step + 1]
        for ai in ACTION_INDICES:
            local = scm[ai].local_treatments(query)
            n_local = sum(1 for c in np.nonzero(np.asarray(local)[0])[0]
                          if c != ai)
            if n_local > 1:
                valid_steps.append(step)
                break
    log.info("valid_steps=%d / %d (%.1f%%), scan %.1fs", len(valid_steps),
             replay_full.shape[0] - 1,
             100.0 * len(valid_steps) / max(replay_full.shape[0] - 1, 1),
             time.perf_counter() - t0)
    if not valid_steps:
        raise RuntimeError("no valid steps (no action SCM with >1 "
                           "non-self local cause)")

    valid_tp1 = (np.array(valid_steps) + 1).tolist()
    eval_t = replay_full[valid_steps]
    eval_tp1 = replay_full[valid_tp1]
    y = eval_tp1[:, ACTION_INDICES].argmax(axis=1).astype(int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        eval_t, y, np.arange(len(valid_steps)), test_size=0.4,
        random_state=10, stratify=y)
    y_train_oh = np.eye(4)[y_train]
    y_test_oh = np.eye(4)[y_test]
    log.info("split: train=%d test=%d train_counts=%s test_counts=%s",
             len(y_train), len(y_test),
             np.bincount(y_train, minlength=4).tolist(),
             np.bincount(y_test, minlength=4).tolist())

    def clf_metrics(name, y_true, y_pred):
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted",
                                          zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                       zero_division=0)),
            "per_class": [
                {"action": ACTION_NAMES[i], "label": i,
                 "precision": float(prec[i]), "recall": float(rec[i]),
                 "f1": float(f1[i]), "support": int(sup[i])}
                for i in range(4)],
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
        }

    dt = DecisionTreeClassifier(random_state=1)
    dt.fit(X_train, y_train)
    dt_m = clf_metrics("dt", y_test, dt.predict(X_test))
    lr = LogisticRegression(random_state=1, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_m = clf_metrics("lr", y_test, lr.predict(X_test))
    log.info("DT acc=%.4f F1w=%.4f | LR acc=%.4f F1w=%.4f (n_test=%d)",
             dt_m["accuracy"], dt_m["f1_weighted"], lr_m["accuracy"],
             lr_m["f1_weighted"], len(y_test))

    # --- global causes per action (notebook cell 21, via repo DBN_to_causal) ---
    G_fid = build_dbn(fid_causes)
    causalG = infer_network.DBN_to_causal(G_fid)

    def feature_index_from_node(node):
        if node.startswith("state_") and node.endswith("_t"):
            name = node[len("state_"):-len("_t")]
        elif node.startswith("action_") and node.endswith("_t"):
            name = node[len("action_"):-len("_t")]
        else:
            return None
        try:
            return FEATURE_NAMES.index(name)
        except ValueError:
            return None

    action_global_causes = {}
    for ai, aname in zip(ACTION_INDICES, ACTION_NAMES):
        target_node = f"action_{aname}_t"
        if target_node in causalG:
            causes = sorted(
                idx for idx in (feature_index_from_node(p)
                                for p in causalG.predecessors(target_node))
                if idx is not None and idx != ai)
        else:
            causes = []
        action_global_causes[ai] = causes
        log.info("global causes %s <- %s", aname,
                 [FEATURE_NAMES[i] for i in causes])

    # --- fidelity BART fits, one per (action, global cause) ---
    bart_cls = infer_network.BARTRegressor
    hash_y_bart = {}
    t0 = time.perf_counter()
    for ai, aname in zip(ACTION_INDICES, ACTION_NAMES):
        causes = action_global_causes[ai]
        covars = [i for i in range(N_FEATURES) if i not in causes]
        y_train_action = y_train_oh[:, ai - N_STATES]
        for treat in causes:
            cols = [treat] + covars
            bart = bart_cls()
            bart.fit(X_train[:, cols], y_train_action)
            hash_y_bart[(ai, treat)] = np.asarray(
                bart.predict(X_test[:, cols]), dtype=float).ravel()
            del bart
            gc.collect()
    fit_s = time.perf_counter() - t0
    log.info("fidelity BART fits: %d (action,cause) pairs in %.1fs",
             len(hash_y_bart), fit_s)

    # --- ITE-weighted prediction (notebook cells 22-23), FULL test set ---
    action_priors = y_train_oh.mean(axis=0)
    y_pred_scores = np.zeros((len(y_test), 4))
    fallback_counts = [0, 0, 0, 0]
    t0 = time.perf_counter()
    for row_index, step in enumerate(idx_test):
        query = replay_full[valid_steps[step]:valid_steps[step] + 1]
        for offset, ai in enumerate(ACTION_INDICES):
            local = scm[ai].local_treatments(query)
            total_weight = 0.0
            y_bart_pred = 0.0
            for treat in np.nonzero(np.asarray(local)[0])[0]:
                treat = int(treat)
                if (ai, treat) not in hash_y_bart:
                    continue
                ite = scm[ai].ITE(query, treat,
                                  additive=(treat in NUM_COLS_FIDELITY))
                weight = float(abs(np.asarray(ite, dtype=float).mean()))
                total_weight += weight
                y_bart_pred += hash_y_bart[(ai, treat)][row_index] * weight
            if total_weight > 0:
                y_pred_scores[row_index, offset] = y_bart_pred / total_weight
            else:
                y_pred_scores[row_index, offset] = action_priors[offset]
                fallback_counts[offset] += 1
    ite_s = time.perf_counter() - t0
    y_pred_bart = y_pred_scores.argmax(axis=1)
    bart_m = clf_metrics("bart", y_test, y_pred_bart)
    log.info("BART-weighted acc=%.4f F1w=%.4f F1m=%.4f (n_test=%d, "
             "ITE loop %.1fs, fallbacks=%s)", bart_m["accuracy"],
             bart_m["f1_weighted"], bart_m["f1_macro"], len(y_test), ite_s,
             fallback_counts)

    np.savez(out_dir / f"fidelity_scores_{args.policy}.npz",
             y_true=y_test, y_pred_bart=y_pred_bart,
             y_pred_dt=dt.predict(X_test), y_pred_lr=lr.predict(X_test),
             scores=y_pred_scores, test_indices=idx_test,
             valid_indices=np.array(valid_steps))
    result = {
        "label": BART_LABEL, "policy": args.policy,
        "dbarts_version": dbarts_ver, "seed": args.seed,
        "replay": str(args.replay),
        "n_transitions": meta["n_transitions"],
        "action_counts_replay": meta["action_counts"],
        "valid_steps": int(len(valid_steps)),
        "valid_fraction": float(len(valid_steps) / (replay_full.shape[0] - 1)),
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "train_action_counts": np.bincount(y_train, minlength=4).tolist(),
        "test_action_counts": np.bincount(y_test, minlength=4).tolist(),
        "split": "60/40, random_state=10, stratify=y (author notebook cell 15)",
        "valid_rule": "any action SCM (indices 8-11) with >1 non-self local "
                      "cause (author notebook cell 11)",
        "bart_test_cap": None,
        "bart_test_note": "full test set, no cap (ITE probe ~7 ms/row); "
                          "CartPole comparison capped BART-weighted at 200",
        "decision_tree": dt_m, "logistic_regression": lr_m,
        "bart_weighted": bart_m,
        "global_causes_per_action": {
            ACTION_NAMES[ai - N_STATES]: [FEATURE_NAMES[i] for i in
                                          action_global_causes[ai]]
            for ai in ACTION_INDICES},
        "fidelity_stage_hte_causes": {
            FEATURE_NAMES[idx]: [FEATURE_NAMES[c] for c in fid_causes[idx]]
            for idx in range(N_FEATURES)},
        "fidelity_stage_hte_fit_seconds": fit_times,
        "fidelity_bart_pairs": int(len(hash_y_bart)),
        "fidelity_bart_fit_seconds": round(fit_s, 2),
        "ite_loop_seconds": round(ite_s, 2),
        "prior_fallback_counts_per_action": fallback_counts,
        "action_priors_train": [float(x) for x in action_priors],
    }
    out_json.write_text(json.dumps(result, indent=2))
    log.info("wrote %s", out_json)


# ----------------------------------------------------------------------
# assemble stage
# ----------------------------------------------------------------------
def cmd_assemble(args, log):
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    details = {"label": BART_LABEL, "episodes": args.episodes,
               "seed": args.seed, "runs": {}}
    rows = []
    edges_by_pm = {}
    for policy in ["heuristic", "dqn"]:
        fid_path = Path(args.fid_dir) / f"fidelity_{policy}.json"
        fid = json.loads(fid_path.read_text()) if fid_path.exists() else None
        details["runs"][policy] = {"fidelity_eval_notebook": fid,
                                   "methods": {}}
        for method in ["hte", "ate"]:
            causes_by_target = {}
            total_s = 0.0
            target_recs = []
            for idx in range(N_FEATURES):
                p = ckpt_dir / f"target_{policy}_{method}_{idx:02d}.json"
                if not p.exists():
                    raise FileNotFoundError(f"missing checkpoint {p}")
                rec = json.loads(p.read_text())
                causes_by_target[idx] = rec["causes"]
                total_s += rec["elapsed_seconds"]
                target_recs.append(rec)
            G = build_dbn(causes_by_target)
            edges = dbn_edges_to_strings(G)
            edges_c = collapsed_edges(G)
            causes = per_target_causes(G)
            png = graphs_dir / f"graph_{policy}_{method}_{args.episodes}ep.png"
            draw_dbn(G, png,
                     f"LunarLander global causal graph ({policy}, {method}, "
                     f"{args.episodes} episodes)\n{BART_LABEL}")
            edges_by_pm[f"{policy}_{method}"] = set(edges)
            n_trans = target_recs[0]["n_replay_rows"] + 1
            details["runs"][policy]["methods"][method] = {
                "n_edges": len(edges), "edges": edges,
                "edges_collapsed": edges_c, "per_target_causes": causes,
                "infer_s": round(total_s, 2),
                "target_checkpoints": target_recs,
                "n_transitions": n_trans,
            }
            row = {"policy": policy, "method": method,
                   "n_transitions": n_trans, "n_edges": len(edges),
                   "edges": "; ".join(edges),
                   "edges_collapsed": "; ".join(edges_c),
                   "per_target_causes": json.dumps(causes),
                   "time_s": round(total_s, 2)}
            if method == "hte" and fid:
                row.update({
                    "fid_valid_steps": fid["valid_steps"],
                    "fid_n_test": fid["n_test"],
                    "dt_acc": round(fid["decision_tree"]["accuracy"], 4),
                    "dt_f1_weighted": round(fid["decision_tree"]["f1_weighted"], 4),
                    "dt_f1_macro": round(fid["decision_tree"]["f1_macro"], 4),
                    "lr_acc": round(fid["logistic_regression"]["accuracy"], 4),
                    "lr_f1_weighted": round(fid["logistic_regression"]["f1_weighted"], 4),
                    "lr_f1_macro": round(fid["logistic_regression"]["f1_macro"], 4),
                    "bart_acc": round(fid["bart_weighted"]["accuracy"], 4),
                    "bart_f1_weighted": round(fid["bart_weighted"]["f1_weighted"], 4),
                    "bart_f1_macro": round(fid["bart_weighted"]["f1_macro"], 4),
                    "bart_n_test": fid["n_test"],
                    "global_causes_per_action": json.dumps(
                        fid["global_causes_per_action"]),
                })
            rows.append(row)
            log.info("%s/%s: %d edges, %.1fs", policy, method, len(edges),
                     total_s)
    comparison = {}
    for method in ["hte", "ate"]:
        h = edges_by_pm[f"heuristic_{method}"]
        d = edges_by_pm[f"dqn_{method}"]
        comparison[method] = {
            "heuristic_n_edges": len(h), "dqn_n_edges": len(d),
            "jaccard": round(jaccard(h, d), 4),
            "symdiff": len(h ^ d),
            "only_heuristic": sorted(h - d),
            "only_dqn": sorted(d - h),
            "common": sorted(h & d),
        }
    details["comparison_trained_vs_heuristic"] = comparison
    fieldnames = ["policy", "method", "n_transitions", "n_edges", "edges",
                  "edges_collapsed", "per_target_causes", "time_s",
                  "fid_valid_steps", "fid_n_test", "dt_acc",
                  "dt_f1_weighted", "dt_f1_macro", "lr_acc",
                  "lr_f1_weighted", "lr_f1_macro", "bart_acc",
                  "bart_f1_weighted", "bart_f1_macro", "bart_n_test",
                  "global_causes_per_action"]
    with open(out_dir / "comparison_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "per_run_details.json").write_text(
        json.dumps(details, indent=2, default=str))
    md = []
    md.append("# CausalXRL extension D — XRL trained-vs-heuristic "
              "comparison (LunarLander)\n")
    md.append(f"> **{BART_LABEL}**\n")
    md.append(f"> Episodes: {args.episodes} per policy, seed {args.seed}. "
              "Data: gymnasium LunarLander-v3 rollouts (repo `play()` uses "
              "gym 0.20, incompatible with Python 3.12; the repo's "
              "transition-recording rule is kept). Representation: 8 state "
              "variables + 4 one-hot actions = 12 columns/targets. `hte` = "
              "repo `infer_causes_hte` (70/30 split); `ate` = repo "
              "`infer_causes_ate` (50/50 split). Fidelity is author "
              "eval-notebook style, multiclass: valid steps = any action "
              "SCM with >1 non-self local cause; 60/40 split "
              "(random_state=10, stratify=y); DT/LR/BART-weighted on the "
              "FULL test set (no cap — unlike CartPole's 200-step cap; "
              "see per-run details). F1 reported is weighted F1 (paper "
              "§5.2) plus macro F1. Global-cause sets for BART-weighted = "
              "predecessors of each action node in the collapsed causal "
              "graph from the FIDELITY-stage HTE refit (a separate "
              "stochastic fit from the graph-stage checkpoints, because "
              "bart_train has no seed — see per-run details). Time = "
              "structure-learning seconds (sum over 12 per-target "
              "checkpoints; infer only, excluding rollout and fidelity).\n")
    md.append("| policy | method | n_trans | n_edges | time_s (infer) | "
              "DT acc / F1w / F1m | LR acc / F1w / F1m | "
              "BART acc / F1w / F1m (n_test) | edges |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        def _fmt(p):
            return (f"{r.get(p + '_acc')} / {r.get(p + '_f1_weighted')} / "
                    f"{r.get(p + '_f1_macro')}"
                    if r.get(p + "_acc") is not None else "—")
        bart = (_fmt("bart") + f" (n={r.get('bart_n_test')})"
                if r.get("bart_acc") is not None else "—")
        md.append(f"| {r['policy']} | {r['method']} | {r['n_transitions']} "
                  f"| {r['n_edges']} | {r['time_s']} | {_fmt('dt')} | "
                  f"{_fmt('lr')} | {bart} | {r['edges']} |")
    md.append("")
    md.append("## Trained (DQN) vs heuristic — graph differences\n")
    for method, comp in comparison.items():
        md.append(f"### {method}\n")
        md.append(f"- Jaccard = {comp['jaccard']}, symmetric difference = "
                  f"{comp['symdiff']} edges (heuristic "
                  f"{comp['heuristic_n_edges']} edges, DQN "
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
    log.info("wrote comparison tables + per_run_details.json to %s", out_dir)
    log.info("DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("graphs")
    g.add_argument("--policy", required=True, choices=["heuristic", "dqn"])
    g.add_argument("--method", required=True, choices=["hte", "ate"])
    g.add_argument("--replay", required=True)
    g.add_argument("--ckpt-dir", required=True)
    g.add_argument("--repo", required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--targets", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    f = sub.add_parser("fidelity")
    f.add_argument("--policy", required=True, choices=["heuristic", "dqn"])
    f.add_argument("--replay", required=True)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--repo", required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--force", action="store_true")
    a = sub.add_parser("assemble")
    a.add_argument("--ckpt-dir", required=True)
    a.add_argument("--fid-dir", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--episodes", type=int, default=30)
    a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("xrl-ll")
    if args.stage == "graphs":
        args.targets = {int(x) for x in args.targets.split(",") if x != ""}
        cmd_graphs(args, log)
    elif args.stage == "fidelity":
        cmd_fidelity(args, log)
    else:
        cmd_assemble(args, log)


if __name__ == "__main__":
    main()
