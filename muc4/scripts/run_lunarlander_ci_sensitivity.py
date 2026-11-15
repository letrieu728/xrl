#!/usr/bin/env python3
"""LunarLander CI-threshold sensitivity + ITE-draw saving with REAL BART (hte).

Companion to scripts/run_ablation_bart.py (CartPole). That script has CI
sensitivity (90/95/99%) for CartPole only; this script does the same for
LunarLander, on the EXISTING Muc 3 replay (muc3/results/lunarlander/
replay.npz: 30 episodes / 7,382 transitions, PD controller, collection
seed 99) -- no new rollouts.

Method (identical in structure to run_ablation_bart.py, hte branch):
1. Replay construction EXACTLY as muc3/scripts/lunarlander_bart_target.py:
   states + one-hot actions (4) -> 12 columns; replay_t, replay_tp1 =
   current[:-1], current[1:]. Episode boundaries are NOT respected --
   this is the Muc 3 convention, kept for comparability with the Muc 3
   graph (44 edges, metrics.json bart_targets).
2. Per target (12): replicate infer_causes_hte verbatim (70/30 split,
   StandardScaler fit on X_train, num_cols = cols with >2 unique values
   in the FULL replay) and take the 95% decisions from the repo's own
   bart.is_causal_HTE. No repo file is edited.
3. For every (target, treatment-column) pair (144): call bart.ITE once
   more on the SAME fitted model + SAME X_est and SAVE the 500 posterior
   ITE draws (500 x N_est) to a per-target .npz in --draws-dir. ITE
   prediction is deterministic given the fitted trees, so these are the
   same draws the repo's HTE decision used.
4. From those SAME draws (no refit), recompute the HTE decision at
   90%/95%/99% by rebuilding the repo HTE clustering
   (DecisionTreeRegressor min_samples_leaf=int(N*0.1), max_depth=4,
   pooled per-cluster quantiles) and validate the recomputed 95%
   decision against the repo 95% decision for all 144 pairs.

Seeds (three different seeds -- do not conflate):
* Data-collection seed 99: baked into the existing replay.npz (Muc 3).
* Split seed (--seed, default 0): numpy global RNG seeded ONCE before
  the target loop, exactly as run_ablation_bart.py does, so the 70/30
  splits are reproducible. Muc 3's per-target scripts set NO split seed
  (each ran in an isolated unseeded process), so there is no "Muc 3
  split seed" to match; seed 0 follows the CartPole ablation
  convention. The 95% validation is internal to this fit and does not
  depend on the split seed.
* BART R seed 99: hardcoded inside the repo's R code (bart(...,
  seed=99)); unchanged.
RNG-order note: unlike run_ablation_bart.py (all fits first, then one
sensitivity pass), this script processes one target at a time (fit ->
repo decisions -> ITE draws -> rebuild) and drops each fitted model
before the next target, to bound memory with 12 targets. The HTE
rebuild trees therefore consume global RNG between splits. Splits are
still fully reproducible given this script + seed, but they are NOT
the same splits a two-pass replication would draw. This is documented,
not hidden.

Time guard: --deadline-minutes (default 60) stops the loop gracefully
before a target that would start after the deadline; partial per-target
files + log are kept and the exit code is 2. ATE is NOT run by this
script (hte only, per the task).

Outputs:
* --draws-dir/target_<i>.npz: ite_draws (12 cols, 500, N), X_est,
  clusters (12, N), feature names, num_cols, causes95. These are LARGE
  (~100 MB each, float64) and stay in the work dir / Drive -- never
  pushed as text.
* --output-dir/lunarlander_per_target/target_<i>.json: incremental
  per-target summary (repo 95% decisions, per-level decisions, cluster
  sizes, timing).
* --output-dir/lunarlander_ci_sensitivity.json + .md: aggregated
  sensitivity + validation.
* Phase 2 (scripts/recompute_lunarlander_from_draws.py, pure numpy, no
  R) re-derives the decisions and edge weights from the SAVED .npz
  files alone and checks them against this run's in-process decisions.

Usage (from the XRL repo root, with the ablation venv):
    .venv-ablation/bin/python -B muc4/scripts/run_lunarlander_ci_sensitivity.py \
        [--seed 0] [--repo repo-causal-xrl] [--deadline-minutes 60]
"""

import argparse
import gc
import json
import logging
import os

os.environ.setdefault("R_HOME", "/usr/lib/R")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

BASE_DIR = Path(__file__).resolve().parent.parent  # muc4/
REPO_ROOT = BASE_DIR.parent
LABEL = "BART (R/dbarts via rpy2) - real CausalXRL estimator"

FEATURES = [
    "x", "y", "vx", "vy", "angle", "angular_velocity",
    "left_contact", "right_contact",
    "do_nothing", "fire_left_engine", "fire_main_engine", "fire_right_engine",
]
N_FEATURES = len(FEATURES)
N_STATES = 8
CI_LEVELS = (0.90, 0.95, 0.99)
R_LIB_WORKSPACE = Path.home() / "workspace" / "R-lib"


def edge_name(c, i_repr):
    prefix_t = "action" if c >= N_STATES else "state"
    prefix_tp1 = "action" if i_repr >= N_STATES else "state"
    return (f"{prefix_t}_{FEATURES[c]}_t",
            f"{prefix_tp1}_{FEATURES[i_repr]}_t+1")


def load_repo_infer_network(repo_dir, log):
    """Import the repo's infer_network with the REAL BARTRegressor.

    Two compat steps, no repo file edited: (1) restore
    sklearn.tree._tree.DTYPE (removed in sklearn >= 1.6); (2) prepend
    ~/workspace/R-lib to R's .libPaths when dbarts is not visible, so
    the repo's library("dbarts") finds dbarts 0.9.34 after VM restarts.
    """
    import sklearn.tree._tree as _sk_tree
    if not hasattr(_sk_tree, "DTYPE"):
        _sk_tree.DTYPE = np.float32
    import rpy2.robjects as _ro
    try:
        _ro.r('library("dbarts")')
        log.info("dbarts found on default R .libPaths")
    except Exception:  # noqa: BLE001
        log.warning("dbarts not on default .libPaths; prepending %s",
                    R_LIB_WORKSPACE)
        _ro.r(f'.libPaths(c("{R_LIB_WORKSPACE}", .libPaths()))')
        _ro.r('library("dbarts")')
        log.info("dbarts loaded from %s", R_LIB_WORKSPACE)
    sys.path.insert(0, str(repo_dir))
    import cxrl.lib.causal.infer_network as infer_network  # noqa: E402
    return infer_network


def build_replay(replay_path):
    """EXACTLY muc3/scripts/lunarlander_bart_target.py replay construction."""
    replay = np.load(replay_path)
    states = replay["states"]
    actions = replay["actions"].astype(int)
    one_hot_actions = np.eye(4, dtype=float)[actions]
    current = np.concatenate((states, one_hot_actions), axis=1)
    replay_t, replay_tp1 = current[:-1], current[1:]
    return replay_t, replay_tp1, {
        "states_shape": list(states.shape),
        "n_rows_current": int(current.shape[0]),
        "n_pairs": int(replay_t.shape[0]),
        "action_counts": [int((actions == a).sum()) for a in range(4)],
    }


def hte_levels_from_draws(X_est, ite_draws):
    """Rebuild repo HTE clustering; per-level decision + cluster labels.

    Replicates HTE.__init__ exactly: mean ITE over draws,
    DecisionTreeRegressor(min_samples_leaf=int(N*0.1), max_depth=4)
    with default random_state=None, clusters via tree_.apply. Only the
    pooled-quantile levels vary. Returns (per-level bool, clusters,
    sizes, per-level per-cluster significance, per-level per-cluster
    pooled CI bounds).
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
    per_level, per_cluster, per_cluster_ci = {}, {}, {}
    for level in CI_LEVELS:
        alpha = 1.0 - level
        sig, cis = {}, {}
        for cid in np.unique(clusters):
            pooled = draws[:, clusters == cid]
            lci, uci = np.quantile(pooled, [alpha / 2.0, 1.0 - alpha / 2.0])
            sig[int(cid)] = bool(lci > 0 or uci < 0)
            cis[int(cid)] = (float(lci), float(uci))
        per_cluster[level] = sig
        per_cluster_ci[level] = cis
        per_level[level] = any(sig.values())
    return per_level, clusters, sizes, per_cluster, per_cluster_ci


def drop_fitted_model(model):
    for _attr in ("_bart_machine",):
        if hasattr(model, _attr):
            try:
                delattr(model, _attr)
            except Exception:  # noqa: BLE001
                pass
    gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--replay", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--draws-dir", type=Path, default=None)
    ap.add_argument("--targets", type=int, nargs="+", default=None,
                    help="subset of target indices (pilot use only)")
    ap.add_argument("--deadline-minutes", type=float, default=60.0)
    args = ap.parse_args()

    repo_dir = args.repo or (REPO_ROOT / "repo-causal-xrl")
    if not repo_dir.is_dir():
        repo_dir = Path("/tmp/causal-xrl-muc4")
    replay_path = args.replay or (
        REPO_ROOT / "muc3" / "results" / "lunarlander" / "replay.npz")
    out_dir = args.output_dir or (BASE_DIR / "results" / "ablation-bart")
    draws_dir = args.draws_dir or (out_dir / "ite_draws_lunarlander")
    per_target_dir = out_dir / "lunarlander_per_target"
    out_dir.mkdir(parents=True, exist_ok=True)
    draws_dir.mkdir(parents=True, exist_ok=True)
    per_target_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "lunarlander_ci_run.log",
                                      mode="w"),
                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("lunarlander-ci")
    t_start = time.perf_counter()
    deadline_s = args.deadline_minutes * 60.0
    log.info(LABEL)
    log.info("repo=%s replay=%s seed(split)=%s deadline_min=%.1f",
             repo_dir, replay_path, args.seed, args.deadline_minutes)
    commit = None
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                capture_output=True, text=True,
                                timeout=30).stdout.strip()
        log.info("repo commit=%s", commit)
    except Exception as e:  # noqa: BLE001
        log.warning("could not read repo commit: %s", e)
    import hashlib
    replay_sha = hashlib.sha256(Path(replay_path).read_bytes()).hexdigest()
    log.info("replay sha256=%s", replay_sha)

    import sklearn  # noqa: E402
    versions = {"sklearn": sklearn.__version__, "numpy": np.__version__,
                "python": sys.version.split()[0]}
    try:
        import rpy2  # noqa: E402
        import rpy2.robjects as _ro  # noqa: E402
        versions["rpy2"] = rpy2.__version__
        versions["R"] = list(_ro.r('as.character(getRversion())'))[0]
        log.info("rpy2=%s R=%s", versions["rpy2"], versions["R"])
    except Exception as e:  # noqa: BLE001
        log.error("rpy2 check failed: %s", e)
        raise

    infer_network = load_repo_infer_network(repo_dir, log)
    try:
        import rpy2.robjects as _ro
        versions["dbarts"] = list(
            _ro.r('as.character(packageVersion("dbarts"))'))[0]
        log.info("dbarts=%s", versions["dbarts"])
    except Exception as e:  # noqa: BLE001
        log.error("dbarts version check failed: %s", e)
        raise
    log.info("versions: %s", versions)

    replay_t, replay_tp1, replay_info = build_replay(replay_path)
    log.info("replay: %s", replay_info)
    num_cols = [c for c in range(N_FEATURES)
                if len(np.unique(replay_t[:, c])) > 2]
    log.info("num_cols (>2 unique in full replay_t): %s (%s)",
             num_cols, [FEATURES[c] for c in num_cols])

    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    target_indices = args.targets if args.targets is not None \
        else list(range(N_FEATURES))
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_targets = {}
    validation = {"mismatches_95": [], "n_checked": 0}
    stopped_early = False
    for i_repr in target_indices:
        elapsed = time.perf_counter() - t_start
        if elapsed > deadline_s:
            log.error("DEADLINE: %.1fs > %.1fs before target %d (%s); "
                      "stopping gracefully with %d/%d targets done",
                      elapsed, deadline_s, i_repr, FEATURES[i_repr],
                      len(all_targets), N_FEATURES)
            stopped_early = True
            break
        t_t0 = time.perf_counter()
        target = replay_tp1[:, i_repr]
        bart = infer_network.BARTRegressor()
        X_train, X_est, y_train, y_est = train_test_split(
            replay_t, target, test_size=0.3)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_train[:, num_cols] = X_train_scaled[:, num_cols]
        X_est_scaled = scaler.transform(X_est)
        X_est[:, num_cols] = X_est_scaled[:, num_cols]
        t_fit0 = time.perf_counter()
        bart.fit(X_train, y_train)
        fit_s = time.perf_counter() - t_fit0
        causes95 = []
        repo_decision = {}
        for c in range(N_FEATURES):
            sig = bart.is_causal_HTE(X_est, c, additive=(c in num_cols))
            repo_decision[c] = bool(sig)
            if sig:
                causes95.append(c)
        log.info("target %s: fit_s=%.1f causes95=%s (X_est n=%d)",
                 FEATURES[i_repr], fit_s,
                 [FEATURES[c] for c in causes95], X_est.shape[0])
        # ITE draws + sensitivity recompute from the SAME draws.
        t_ite0 = time.perf_counter()
        draws_all = np.empty((N_FEATURES, 500, X_est.shape[0]),
                             dtype=np.float64)
        clusters_all = np.empty((N_FEATURES, X_est.shape[0]), dtype=np.int64)
        detail = {}
        for c in range(N_FEATURES):
            ite = np.asarray(
                bart.ITE(X_est, c, additive=(c in num_cols)), dtype=float)
            assert ite.shape == (500, X_est.shape[0]), (
                f"ITE shape {ite.shape} != (500, {X_est.shape[0]}) "
                f"target={FEATURES[i_repr]} col={FEATURES[c]}")
            draws_all[c] = ite
            per_level, clusters, sizes, per_cluster, per_cluster_ci = \
                hte_levels_from_draws(X_est, ite)
            clusters_all[c] = clusters
            validation["n_checked"] += 1
            if per_level[0.95] != repo_decision[c]:
                validation["mismatches_95"].append({
                    "target": FEATURES[i_repr], "column": FEATURES[c],
                    "repo_95": bool(repo_decision[c]),
                    "recomputed_95": bool(per_level[0.95])})
                log.warning("95%% VALIDATION MISMATCH target=%s col=%s "
                            "repo=%s recomputed=%s",
                            FEATURES[i_repr], FEATURES[c],
                            repo_decision[c], per_level[0.95])
            detail[FEATURES[c]] = {
                "repo_95": bool(repo_decision[c]),
                "n_clusters": len(sizes),
                "cluster_sizes": {str(k): v for k, v in sizes.items()},
                **{str(level): {
                    "significant": bool(per_level[level]),
                    "cluster_sig": {str(k): v for k, v in
                                    per_cluster[level].items()},
                    "cluster_ci": {str(k): [round(v[0], 4), round(v[1], 4)]
                                   for k, v in
                                   per_cluster_ci[level].items()}}
                   for level in CI_LEVELS},
            }
        ite_s = time.perf_counter() - t_ite0
        npz_path = draws_dir / f"target_{i_repr}.npz"
        np.savez(npz_path,
                 ite_draws=draws_all, X_est=X_est, clusters=clusters_all,
                 features=np.array(FEATURES),
                 num_cols=np.array(num_cols, dtype=int),
                 causes95=np.array(causes95, dtype=int),
                 target_index=np.array(i_repr),
                 split_seed=np.array(args.seed))
        target_rec = {
            "target_index": i_repr, "target": FEATURES[i_repr],
            "X_est_n": int(X_est.shape[0]), "X_train_n": int(X_train.shape[0]),
            "num_cols": num_cols,
            "causes95": [FEATURES[c] for c in causes95],
            "cause_indices95": causes95,
            "fit_s": round(fit_s, 2), "ite_s": round(ite_s, 2),
            "target_total_s": round(time.perf_counter() - t_t0, 2),
            "npz": str(npz_path),
            "npz_bytes": int(npz_path.stat().st_size),
            "detail": detail,
        }
        (per_target_dir / f"target_{i_repr}.json").write_text(
            json.dumps(target_rec, indent=2))
        all_targets[FEATURES[i_repr]] = target_rec
        log.info("target %s done: ite_s=%.1f total_s=%.1f npz=%s (%.1f MB)",
                 FEATURES[i_repr], ite_s, target_rec["target_total_s"],
                 npz_path.name, target_rec["npz_bytes"] / 1e6)
        drop_fitted_model(bart)
        del bart, draws_all, clusters_all
        gc.collect()

    total_s = time.perf_counter() - t_start
    log.info("targets done: %d/%d in %.1fs; validation: %d checked, "
             "%d mismatches at 95%%",
             len(all_targets), N_FEATURES, total_s,
             validation["n_checked"], len(validation["mismatches_95"]))

    # Aggregate edges per level from the recomputed decisions.
    sens = {}
    for level in CI_LEVELS:
        edges = set()
        per_target = {name: [] for name in FEATURES}
        for i_repr in range(N_FEATURES):
            rec = all_targets.get(FEATURES[i_repr])
            if rec is None:
                continue
            for c in range(N_FEATURES):
                if rec["detail"][FEATURES[c]][str(level)]["significant"]:
                    u, v = edge_name(c, i_repr)
                    edges.add(f"{u} -> {v}")
                    per_target[FEATURES[i_repr]].append(u)
        sens[str(level)] = {"edges": sorted(edges), "n_edges": len(edges),
                            "per_target": {k: sorted(v)
                                           for k, v in per_target.items()}}
        log.info("CI level=%.2f n_edges=%d edges=%s",
                 level, len(edges), sorted(edges))

    out = {
        "label": LABEL, "method": "hte", "environment": "LunarLander-v2",
        "data": {"replay": str(replay_path), "sha256": replay_sha,
                 "collection_seed": 99, "episodes": 30,
                 "policy": "repository notebook hand-written PD controller",
                 **replay_info},
        "split_seed": args.seed, "repo_commit": commit,
        "repo_dir": str(repo_dir), "versions": versions,
        "num_cols": num_cols, "targets_completed": len(all_targets),
        "targets_total": N_FEATURES, "stopped_early": stopped_early,
        "total_s": round(total_s, 2),
        "levels": sens, "validation": validation,
        "targets": all_targets,
    }
    (out_dir / "lunarlander_ci_sensitivity.json").write_text(
        json.dumps(out, indent=2, default=str))
    log.info("wrote %s", out_dir / "lunarlander_ci_sensitivity.json")

    if len(all_targets) == N_FEATURES:
        lv = sens
        e90, e95, e99 = (set(lv["0.9"]["edges"]), set(lv["0.95"]["edges"]),
                         set(lv["0.99"]["edges"]))
        md = [
            "# CI-threshold sensitivity — LunarLander (REAL BART, hte)\n",
            f"> **{LABEL}**\n",
            "> Edges at 90% / 95% / 99% CI recomputed from the SAME fitted "
            "BART models and the SAME estimation splits via the 500 ITE "
            "posterior draws; no refit. Data: the existing Muc 3 replay "
            "(`muc3/results/lunarlander/replay.npz`, 30 episodes / 7,382 "
            "transitions, PD controller, collection seed 99). Split seed 0 "
            "(numpy RNG seeded once before the target loop, as in "
            "`run_ablation_bart.py`); BART R seed 99 hardcoded in the repo. "
            "The repo hardcodes 95% and offers no level parameter; no repo "
            "file was edited. Recomputed 95% decisions were validated "
            "against the repo 95% decisions for every (target, column).\n",
            "## hte (30 episodes, 12 targets)\n",
        ]
        for level in ("0.9", "0.95", "0.99"):
            e = lv[level]["edges"]
            md.append(f"- **{float(level)*100:.0f}% CI**: {len(e)} edges: "
                      f"`{'; '.join(e)}`")
        md.append(f"- 90% → 95%: {len(e90 - e95)} edge(s) lost: "
                  f"`{'; '.join(sorted(e90 - e95)) or '—'}`; "
                  f"{len(e95 - e90)} gained: "
                  f"`{'; '.join(sorted(e95 - e90)) or '—'}`")
        md.append(f"- 95% → 99%: {len(e95 - e99)} edge(s) lost: "
                  f"`{'; '.join(sorted(e95 - e99)) or '—'}`; "
                  f"{len(e99 - e95)} gained: "
                  f"`{'; '.join(sorted(e99 - e95)) or '—'}`")
        md.append(f"- Validation: {validation['n_checked']} (target, column) "
                  f"pairs checked at 95%, {len(validation['mismatches_95'])} "
                  f"mismatch(es) between recomputed and repo decisions"
                  + (f": {validation['mismatches_95']}"
                     if validation["mismatches_95"]
                     else " — recompute matches the repo exactly."))
        md.append("")
        md.append("## Per-target causes at 95% (recomputed = repo)\n")
        for name in FEATURES:
            md.append(f"- **{name}**: "
                      f"`{'; '.join(sens['0.95']['per_target'][name]) or '(none)'}`")
        md.append("")
        (out_dir / "lunarlander_ci_sensitivity.md").write_text(
            "\n".join(md) + "\n")
        log.info("wrote %s", out_dir / "lunarlander_ci_sensitivity.md")
    else:
        log.warning("INCOMPLETE: %d/%d targets; aggregate .md not written",
                    len(all_targets), N_FEATURES)

    log.info("DONE in %.1fs", total_s)
    return 2 if stopped_early else 0


if __name__ == "__main__":
    sys.exit(main())
