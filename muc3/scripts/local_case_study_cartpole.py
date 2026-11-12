#!/usr/bin/env python3
"""Fig. 4-style local causal case study for CartPole (CausalXRL, Muc 3).

Replicates the authors' local-causal-model logic on the Mục 3 CartPole
replay and draws a two-panel figure in the style of the paper's Figure 4
(arXiv:2610.42028, p. 7): (a) the sampled state drawn as a schematic,
(b) the local causal model with signed ITE edge weights.

WHY A REFIT IS NEEDED
---------------------
Mục 3 stored only per-target *cause lists* (``metrics.json`` /
``targets/``); the fitted BART structural causal models (SCM) were not
saved, and the local model needs them: ``XRL.local_causal_network``
(``cxrl/lib/xrl/xrl.py`` @ ae9096e) calls ``scm[i].local_treatments``
for every target ``i``. This script therefore refits the five BART HTE
models on the SAME replay with the SAME per-target call the Mục 3
scripts used (``infer_causes_hte`` on ``replay_t`` -> target at
``t+1``; see ``cartpole_bart_target.py``). The refit is seeded
(``--seed``, default 99, one ``numpy`` seed before the target loop, as
in the Mục 4 comparison script) so the train/test splits are
reproducible, but ``infer_causes_hte`` uses the global RNG and the R
BART sampler, so refit global causes may differ slightly from the
stored Mục 3 lists. Both are recorded in the output JSON and any
difference is reported, never silently overwritten: the stored lists
remain the Section 3 global result; the refit exists only to obtain
the local model.

LOCAL- MODEL CONVENTION (authors' code, read, not re-derived)
-------------------------------------------------------------
* Query: one RAW replay row ``[state_t, action_t]`` (1x5), exactly as
  the fidelity notebook queries ``xrl.local_causal_network`` /
  ``scm[action].local_treatments`` (``eval_fidelity_Cart Pole.ipynb``
  cells 11 and 23). No rescaling is applied to the query, matching the
  notebook and the Mục 4 replication. NOTE: the HTE decision trees
  were fitted on STANDARDIZED estimation data inside
  ``infer_causes_hte``, while the query is raw-scale; this scale
  mismatch is the authors' own behaviour and is kept for fidelity. A
  standardized-query diagnostic is computed for the chosen state and
  recorded in the JSON (``diagnostic_standardized_query``) but is NOT
  used for the figure or the selection.
* Valid step (notebook cell 11): the local graph has MORE THAN ONE
  non-action cause of the action node (``push``).
* Edge weight (notebook cell 23): ``abs(mean(ITE))`` where ``ITE`` is
  the 500 posterior draws from ``scm[push].ITE(query, treat,
  additive=(treat in num_cols))``. For the figure, the SIGNED posterior
  mean is shown (Fig. 4 colours positive/negative relationships), with
  the 95% interval of the draws in the JSON. Contrast: +1 RAW unit for
  the four state columns (``additive=True`` on the raw query row),
  0 -> 1 for ``push``. Because columns have different raw scales, ITE
  magnitudes are only comparable as the notebook's relative weights,
  not as unit-free importances; the report text must say so.

STATE SELECTION (deterministic, documented)
-------------------------------------------
Scan replay rows in order (excluding the last row, which has no t+1).
Among valid rows whose local ``push`` causes include BOTH
``cart_position`` and ``pole_angle`` (the two inputs of the hand-written
linear policy), choose the row with the largest ``|pole_angle|`` (the
most tilted state: the clearest visual case). Fallbacks, in order:
valid rows (any local causes), then rows with >= 1 non-action local
``push`` cause. ``--row-index`` overrides the selection (still scans
for statistics; the override is recorded).

DATA
----
Default replay: ``../results/cartpole/replay.npz`` relative to this
script (the XRL repo layout; the file is already committed there).
Override with ``--replay``. The script verifies the replay against the
hand-written linear policy from ``cartpole_collect.py`` and records
the file size/SHA-256. NO rollout is recreated: if the replay file is
missing the script fails instead of inventing data.

AUTHOR REPO
-----------
``--repo`` must point at a checkout of github.com/yanmluk/causal-xrl
at commit ``ae9096e`` (the commit Mục 3 used). If omitted, the script
looks in ``$CXRL_REPO``, ``/tmp/causal-xrl-auth``, the Mục 4 persistent
copy, and ``./repo-causal-xrl``, in that order. No repo file is edited;
the sklearn ``_tree.DTYPE`` compatibility shim from the Mục 4 scripts
is applied in-process only.

OUTPUTS (``--out-dir``, default ``../results/cartpole``)
--------------------------------------------------------
    local_case_study.json
    graphs/local_case_study.png

REQUIREMENTS: the Mục 4 venv (rpy2 + R ``dbarts`` 0.9.34 + gymnasium
is NOT needed here), e.g.::

    .venv-ablation/bin/python muc3/scripts/local_case_study_cartpole.py \\
        --repo /tmp/causal-xrl-auth \\
        --replay muc3/results/cartpole/replay.npz
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# rpy2 needs R_HOME; default it (R is at /usr/lib/R on this VM).
os.environ.setdefault("R_HOME", "/usr/lib/R")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
FEATURES = ["cart_position", "cart_velocity", "pole_angle",
            "pole_velocity", "push"]
NUM_COLS = [0, 1, 2, 3]  # state columns are numeric; col 4 (push) is binary
ACTION_INDEX = 4
POLICY_INPUTS = {"cart_position", "pole_angle"}


def heuristic_policy(state):
    """Hand-written linear policy from muc3/scripts/cartpole_collect.py."""
    return int((state[0] / 2.4) * 0.4 + (state[2] / 2.095) * 0.6 >= 0)


def resolve_repo(cli_repo):
    candidates = []
    if cli_repo:
        candidates.append(Path(cli_repo))
    if os.environ.get("CXRL_REPO"):
        candidates.append(Path(os.environ["CXRL_REPO"]))
    candidates += [
        Path("/tmp/causal-xrl-auth"),
        Path.home() / "workspace" / "goals" /
        "cschttt-new-presentation-paper-selection" / "files" /
        "causalxrl-muc4-mo-rong" / "repo-causal-xrl",
        Path.cwd() / "repo-causal-xrl",
    ]
    for cand in candidates:
        if (cand / "cxrl" / "lib" / "causal" / "infer_network.py").is_file():
            return cand
    raise SystemExit(
        "author repo checkout not found. Clone it and pass --repo:\n"
        "  git clone https://github.com/yanmluk/causal-xrl.git /tmp/causal-xrl-auth\n"
        "  cd /tmp/causal-xrl-auth && git checkout ae9096e")


def repo_commit(repo_dir):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - non-git copy; record as unknown
        pass
    return "unknown (not a git checkout)"


def local_edges_for_row(scm, query_row):
    """Replicate XRL.local_causal_network for one (state+action) row.

    Returns {target_index: [treatment indices]} using each target
    model's local_treatments() on the RAW query row (notebook
    convention). ``query_row`` is a 1x5 numpy array.
    """
    out = {}
    for i in range(len(FEATURES)):
        local = scm[i].local_treatments(np.asarray(query_row, dtype=float))
        out[i] = [int(t) for t in np.nonzero(np.asarray(local)[0])[0]]
    return out


def ite_stats(model, query_row, treat_idx):
    """Posterior ITE draws for one treatment at one query row.

    Returns (mean, q025, q975, n_draws) following the notebook's ITE
    call: additive contrast for numeric columns, 0->1 for push.
    """
    draws = np.asarray(model.ITE(np.asarray(query_row, dtype=float),
                                 int(treat_idx),
                                 additive=(int(treat_idx) in NUM_COLS)),
                       dtype=float).ravel()
    return (float(np.mean(draws)),
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
            int(draws.shape[0]))


def draw_case_study_figure(out_path, state, action, local_by_target,
                           ite_by_edge, row_index, n_rows, meta):
    """Two-panel Fig. 4-style figure: (a) state schematic, (b) local graph.

    Local self-effects (source == target) are NOT drawn as loops: the
    node label carries a "\u21bb" marker and the weights are listed in
    the caption block under panel (b). Only non-self local edges are
    drawn as arrows. Weight-label offsets are hand-tuned per edge for
    this 5-node layout and fall back to a generic offset for any
    unexpected edge.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch, Rectangle

    x, x_dot, theta, theta_dot = [float(v) for v in state]
    action_name = "push right (1)" if action == 1 else "push left (0)"

    fig = plt.figure(figsize=(14, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.16)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.16)

    # ---- panel (a): state schematic (drawn from the recorded values) ----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-1.15, 2.05)
    ax.axis("off")
    ax.set_title("(a) Sampled state (schematic, not to scale)",
                 fontsize=12, pad=8)
    # ground / track
    ax.plot([-2.4, 2.4], [0, 0], color="#333333", lw=2)
    for tick in (-2.4, 0.0, 2.4):
        ax.plot([tick, tick], [-0.06, 0], color="#333333", lw=1)
    ax.text(0, -0.30, "track  x \u2208 [\u22122.4, 2.4]", ha="center",
            fontsize=9, color="#555555")
    # cart (width 0.5, height 0.3), pole pivot at cart top centre
    cart_x = max(min(x, 2.4), -2.4)
    cart = Rectangle((cart_x - 0.25, 0.02), 0.5, 0.3, facecolor="#cfe8ff",
                     edgecolor="#333333", lw=1.5, zorder=3)
    ax.add_patch(cart)
    pivot = (cart_x, 0.32)
    ax.plot(*pivot, marker="o", color="#333333", markersize=5, zorder=4)
    # pole: length 1.1, angle theta from vertical, positive theta to the right
    pole_len = 1.1
    tip = (pivot[0] + pole_len * np.sin(theta),
           pivot[1] + pole_len * np.cos(theta))
    ax.plot([pivot[0], tip[0]], [pivot[1], tip[1]], color="#b3541e",
            lw=3, solid_capstyle="round", zorder=4)
    ax.plot(*tip, marker="o", color="#b3541e", markersize=7, zorder=5)
    # cart-velocity arrow (horizontal, above the cart)
    if abs(x_dot) > 1e-9:
        ax.annotate("", xy=(cart_x + 0.55 * np.sign(x_dot), 0.62),
                    xytext=(cart_x - 0.55 * np.sign(x_dot), 0.62),
                    arrowprops=dict(arrowstyle="-|>", color="#2ca02c",
                                    lw=2))
        ax.text(cart_x, 0.80, f"cart_velocity = {x_dot:+.3f}",
                ha="center", fontsize=9, color="#2ca02c")
    # values block BELOW the track (never overlapping the pole)
    ax.text(0, -0.58,
            f"cart_position = {x:+.4f}      cart_velocity = {x_dot:+.4f}\n"
            f"pole_angle = {theta:+.4f} rad ({np.degrees(theta):+.1f}\u00b0)"
            f"      pole_velocity = {theta_dot:+.4f}\n"
            f"policy action at this state: {action_name}",
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#999999"))

    # ---- panel (b): local causal graph with signed ITE weights ----
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title("(b) Local causal model (notebook convention)",
                  fontsize=12, pad=8)
    pos = {
        "cart_position": (0.0, 1.0),
        "cart_velocity": (1.0, 1.0),
        "pole_angle": (0.0, 0.0),
        "pole_velocity": (1.0, 0.0),
        "push": (2.0, 0.5),
    }
    self_effects = {s: ite_by_edge[(s, s)]["ite_mean"]
                    for (s, t) in ite_by_edge if s == t}
    edges = [(FEATURES[s], FEATURES[tgt])
             for tgt, treats in local_by_target.items()
             for s in treats if FEATURES[s] != FEATURES[tgt]]
    involved = {n for e in edges for n in e} | set(self_effects)
    G = nx.DiGraph()
    G.add_nodes_from(FEATURES)
    colors = []
    for n in FEATURES:
        if n == "push":
            colors.append("#ffe4c2")
        elif n in involved:
            colors.append("#cfe8ff")
        else:
            colors.append("#e8e8e8")
    nx.draw_networkx_nodes(G, pos, nodelist=FEATURES, node_color=colors,
                           node_size=1500, edgecolors="#333333",
                           linewidths=1.2, ax=ax2)
    labels = {n: (n + " \u21bb" if n in self_effects else n)
              for n in FEATURES}
    label_pos = {
        "cart_position": (0.0, 1.16),
        "cart_velocity": (1.0, 1.16),
        "pole_angle": (0.0, -0.16),
        "pole_velocity": (1.0, -0.16),
        "push": (2.18, 0.5),
    }
    for n in ("cart_position", "cart_velocity"):
        nx.draw_networkx_labels(G, {n: label_pos[n]}, labels={n: labels[n]},
                                font_size=11, horizontalalignment="center",
                                verticalalignment="bottom", ax=ax2)
    for n in ("pole_angle", "pole_velocity"):
        nx.draw_networkx_labels(G, {n: label_pos[n]}, labels={n: labels[n]},
                                font_size=11, horizontalalignment="center",
                                verticalalignment="top", ax=ax2)
    nx.draw_networkx_labels(G, {"push": label_pos["push"]},
                            labels={"push": labels["push"] + " (action)"},
                            font_size=11, horizontalalignment="left",
                            verticalalignment="center", ax=ax2)
    max_abs = max((abs(v["ite_mean"]) for v in ite_by_edge.values()),
                  default=0.0)
    # Hand-tuned curvature + weight-label offsets for this layout.
    style = {
        ("cart_velocity", "cart_position"): {"rad": 0.30, "d": (0.0, 0.20)},
        ("cart_position", "push"): {"rad": -0.10, "d": (-0.25, -0.03)},
        ("pole_angle", "push"): {"rad": 0.10, "d": (-0.25, 0.05)},
        ("push", "cart_velocity"): {"rad": 0.30, "d": (0.20, 0.18)},
        ("push", "pole_velocity"): {"rad": -0.10, "d": (0.30, -0.02)},
    }
    for (u, v) in edges:
        w = ite_by_edge[(u, v)]["ite_mean"]
        color = "#2ca02c" if w > 0 else ("#d62728" if w < 0 else "#444444")
        width = 1.2 + (3.0 * abs(w) / max_abs if max_abs > 0 else 0.0)
        st = style.get((u, v), {"rad": 0.05, "d": (0.0, 0.14)})
        arr = FancyArrowPatch(
            pos[u], pos[v], arrowstyle="-|>", mutation_scale=18,
            linewidth=width, color=color,
            connectionstyle=f"arc3,rad={st['rad']}",
            shrinkA=24, shrinkB=26, zorder=3)
        ax2.add_patch(arr)
        mx = (pos[u][0] + pos[v][0]) / 2 + st["d"][0]
        my = (pos[u][1] + pos[v][1]) / 2 + st["d"][1]
        ax2.text(mx, my, f"{w:+.2f}", ha="center", va="center",
                 fontsize=11, color=color, zorder=4,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                           alpha=0.9))
    ax2.set_xlim(-0.7, 3.1)
    ax2.set_ylim(-0.55, 1.6)
    self_line = "Local self-effects (\u21bb): " + ", ".join(
        f"{n} {self_effects[n]:+.2f}" for n in FEATURES if n in self_effects)
    ax2.text(0.5, -0.07,
             "Green = positive ITE, red = negative ITE (Fig. 4 colours). "
             "Width \u221d |ITE|.",
             ha="center", fontsize=9, color="#555555",
             transform=ax2.transAxes)
    ax2.text(0.5, -0.14, self_line, ha="center", fontsize=9,
             color="#555555", transform=ax2.transAxes)

    fig.suptitle(
        f"CartPole local causal model \u2014 Fig. 4-style case study "
        f"(replay row {row_index} of {n_rows})\n"
        f"Local parents of push: "
        f"{', '.join(FEATURES[s] for s in local_by_target[ACTION_INDEX])} "
        f"\u00b7 policy action: {action_name} \u00b7 BART HTE refit on "
        f"M\u1ee5c 3 replay (seed {meta['seed']})",
        fontsize=12, y=0.97)
    fig.text(0.5, 0.01,
             "Local edges: XRL.local_causal_network / local_treatments on "
             "refit BART SCM (yanmluk/causal-xrl @ ae9096e), raw query row, "
             "notebook convention. Weights: posterior-mean ITE, +1 raw unit "
             "for state columns, 0\u21921 for push; 95% intervals in "
             "local_case_study.json. The refit global causes are recorded "
             "in the JSON and may differ slightly from the stored M\u1ee5c 3 "
             "lists.",
             ha="center", va="bottom", fontsize=8, wrap=True,
             bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none"))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=None,
                    help="author repo checkout (yanmluk/causal-xrl @ ae9096e)")
    ap.add_argument("--replay", type=Path, default=None,
                    help="CartPole replay.npz (default: ../results/cartpole/replay.npz)")
    ap.add_argument("--metrics", type=Path, default=None,
                    help="stored metrics.json for cause comparison "
                         "(default: ../results/cartpole/metrics.json)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: ../results/cartpole)")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--row-index", type=int, default=None,
                    help="override the automatic state selection")
    ap.add_argument("--from-json", type=Path, default=None,
                    help="redraw the figure from an existing "
                         "local_case_study.json without refitting BART "
                         "(figure iteration / reruns)")
    args = ap.parse_args()

    if args.from_json is not None:
        result = json.loads(args.from_json.read_text())
        sel = result["selected_state"]
        state = [sel["state"][f] for f in FEATURES[:4]]
        local_by_target = {
            FEATURES.index(t): [FEATURES.index(s) for s in cs]
            for t, cs in result["local_causal_model"][
                "local_causes_by_target"].items()}
        ite_by_edge = {(e["source"], e["target"]): e for e in
                       result["local_causal_model"]["local_edges"]}
        out_dir = args.out_dir or (HERE.parent / "results" / "cartpole")
        graphs_dir = out_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        png_path = graphs_dir / "local_case_study.png"
        draw_case_study_figure(
            png_path, state, sel["action_at_t"], local_by_target,
            ite_by_edge, sel["row_index"], result["data"]["transitions"],
            {"seed": result["method"]["seed"]})
        print(f"redrew {png_path} from {args.from_json} (no BART refit)")
        return

    repo_dir = resolve_repo(args.repo)
    print(f"author repo: {repo_dir} (commit {repo_commit(repo_dir)})")

    # Compat shim (no repo file edited): the repo's bart_regressor.py uses
    # sklearn.tree._tree.DTYPE, removed in sklearn >= 1.6. Old DTYPE was
    # np.float32; restore the attribute so the repo code runs unchanged.
    import sklearn.tree._tree as _sk_tree
    if not hasattr(_sk_tree, "DTYPE"):
        _sk_tree.DTYPE = np.float32
    sys.path.insert(0, str(repo_dir))
    from cxrl.lib.causal.infer_network import infer_causes_hte

    replay_path = args.replay or (HERE.parent / "results" / "cartpole" / "replay.npz")
    if not replay_path.is_file():
        raise SystemExit(f"replay file not found: {replay_path} "
                         f"(no rollout is recreated; pass --replay)")
    replay_bytes = replay_path.read_bytes()
    replay = np.load(replay_path)
    states = np.asarray(replay["states"], dtype=float)
    actions = np.asarray(replay["actions"], dtype=float)
    next_states = np.asarray(replay["next_states"], dtype=float)
    n_rows = states.shape[0]
    print(f"replay: {replay_path} ({len(replay_bytes)} bytes, "
          f"sha256={hashlib.sha256(replay_bytes).hexdigest()[:16]}..., "
          f"{n_rows} transitions)")
    policy_pred = np.array([heuristic_policy(s) for s in states], dtype=float)
    policy_match = int((policy_pred == actions).sum())
    print(f"policy check: {policy_match}/{n_rows} actions match the "
          f"hand-written linear policy")

    # EXACTLY the Mục 3 target-script construction (cartpole_bart_target.py):
    # current = [states, actions]; replay_t = current[:-1],
    # replay_tp1 = current[1:].
    current = np.concatenate((states, actions.reshape(-1, 1)), axis=1)
    replay_t = current[:-1]
    replay_tp1 = current[1:]

    # ---- refit: one BART HTE model per target (Mục 3 call, seeded) ----
    np.random.seed(args.seed)
    random.seed(args.seed)
    scm = {}
    refit_causes = {}
    refit_seconds = {}
    t_refit0 = time.perf_counter()
    for target in range(len(FEATURES)):
        t0 = time.perf_counter()
        causes, model = infer_causes_hte(replay_t, replay_tp1[:, target])
        scm[target] = model
        refit_causes[FEATURES[target]] = [FEATURES[c] for c in causes]
        refit_seconds[FEATURES[target]] = time.perf_counter() - t0
        print(f"  refit target {FEATURES[target]}: "
              f"causes={refit_causes[FEATURES[target]]} "
              f"({refit_seconds[FEATURES[target]]:.1f}s)")
    refit_total = time.perf_counter() - t_refit0
    print(f"refit total: {refit_total:.1f}s")

    # ---- compare with the stored Mục 3 cause lists (never overwrite) ----
    stored_causes = None
    metrics_path = args.metrics or (HERE.parent / "results" / "cartpole" / "metrics.json")
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        stored_causes = {t["target"]: list(t["causes"])
                         for t in metrics["bart_targets"]}
        for name in FEATURES:
            mark = "SAME" if stored_causes[name] == refit_causes[name] else "DIFFERS"
            print(f"  stored-vs-refit {name}: {mark} "
                  f"(stored={stored_causes[name]})")
    else:
        print(f"  no stored metrics.json at {metrics_path}; skipping comparison")

    # ---- scan all rows for notebook-valid local states ----
    t_scan0 = time.perf_counter()
    scan = []
    for row in range(n_rows - 1):  # last row has no t+1
        local = local_edges_for_row(scm, current[row:row + 1])
        push_causes = local[ACTION_INDEX]
        n_non_action = sum(1 for s in push_causes if s != ACTION_INDEX)
        scan.append({
            "row": row,
            "local_push_causes": [FEATURES[s] for s in push_causes],
            "n_non_action_push_causes": n_non_action,
            "pole_angle": float(states[row][2]),
        })
    scan_seconds = time.perf_counter() - t_scan0
    valid = [s for s in scan if s["n_non_action_push_causes"] > 1]
    preferred = [s for s in valid
                 if POLICY_INPUTS <= set(s["local_push_causes"])]
    print(f"scan: {len(scan)} rows in {scan_seconds:.1f}s; "
          f"valid (>1 non-action local push cause): {len(valid)}; "
          f"with both policy inputs: {len(preferred)}")

    # ---- selection ----
    if args.row_index is not None:
        if not (0 <= args.row_index < n_rows - 1):
            raise SystemExit(f"--row-index out of range: {args.row_index}")
        chosen = scan[args.row_index]
        rule = "explicit --row-index override"
    elif preferred:
        chosen = max(preferred, key=lambda s: abs(s["pole_angle"]))
        rule = ("valid row with both policy inputs as local push causes, "
                "largest |pole_angle|")
    elif valid:
        chosen = max(valid, key=lambda s: abs(s["pole_angle"]))
        rule = "valid row (fallback: no row had both policy inputs), largest |pole_angle|"
    else:
        with_one = [s for s in scan if s["n_non_action_push_causes"] >= 1]
        if not with_one:
            raise SystemExit("no replay row has any non-action local cause "
                             "for push; cannot build a non-trivial case study")
        chosen = max(with_one, key=lambda s: abs(s["pole_angle"]))
        rule = ("fallback: no notebook-valid row; row with >=1 non-action "
                "local push cause, largest |pole_angle|")
    row_index = chosen["row"]
    print(f"selected row {row_index} by rule: {rule}")
    print(f"  state={states[row_index].tolist()} action={int(actions[row_index])}")

    query = current[row_index:row_index + 1]
    local_by_target = local_edges_for_row(scm, query)
    local_edges = {}
    ite_by_edge = {}
    for tgt, treats in local_by_target.items():
        names = [FEATURES[s] for s in treats]
        local_edges[FEATURES[tgt]] = names
        for s in treats:
            mean, lci, uci, n_draws = ite_stats(scm[tgt], query, s)
            key = (FEATURES[s], FEATURES[tgt])
            ite_by_edge[key] = {
                "ite_mean": mean, "ite_lci_95": lci, "ite_uci_95": uci,
                "abs_ite": abs(mean), "n_draws": n_draws,
                "contrast": ("+1 raw unit" if s in NUM_COLS else "0 -> 1"),
            }
    # All-treatment ITEs for the push target at this state (transparency:
    # shows the local/non-local contrast for every candidate treatment).
    push_all_ite = {}
    for s in range(len(FEATURES)):
        mean, lci, uci, n_draws = ite_stats(scm[ACTION_INDEX], query, s)
        push_all_ite[FEATURES[s]] = {
            "ite_mean": mean, "ite_lci_95": lci, "ite_uci_95": uci,
            "abs_ite": abs(mean), "n_draws": n_draws,
            "locally_causal": s in local_by_target[ACTION_INDEX],
            "contrast": ("+1 raw unit" if s in NUM_COLS else "0 -> 1"),
        }

    # Diagnostic: same query, globally standardized (NOT the notebook
    # convention; recorded only to expose the raw-vs-standardized scale
    # behaviour of local_treatments).
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(replay_t)
    query_std = query.copy()
    query_std[:, NUM_COLS] = scaler.transform(query)[:, NUM_COLS]
    diag_local = local_edges_for_row(scm, query_std)
    diagnostic = {
        "query_standardization": "StandardScaler fit on all replay_t rows; "
                                 "numeric columns only (col 4 unchanged)",
        "local_causes_by_target": {
            FEATURES[t]: [FEATURES[s] for s in treats]
            for t, treats in diag_local.items()},
        "agrees_with_raw_query": {
            FEATURES[t]: sorted(local_by_target[t]) == sorted(diag_local[t])
            for t in range(len(FEATURES))},
        "note": "Diagnostic only; the figure and the case study use the "
                "RAW query row (notebook convention).",
    }

    # dbarts version via rpy2 (proves the real estimator was used).
    try:
        import rpy2.robjects as robjects
        dbarts_version = str(robjects.r(
            'as.character(packageVersion("dbarts"))')[0])
    except Exception as exc:  # noqa: BLE001
        dbarts_version = f"unknown ({exc})"
    import sklearn
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "dbarts": dbarts_version,
    }

    result = {
        "case_study": "CartPole local causal model, Fig. 4 style",
        "paper": "arXiv:2610.42028, Figure 4 (p. 7)",
        "data": {
            "replay_file": str(replay_path),
            "replay_bytes": len(replay_bytes),
            "replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
            "transitions": n_rows,
            "environment": "CartPole-v0",
            "seed_collection": 99,
            "policy": "repository notebook hand-written linear rule",
            "policy_match_rows": policy_match,
            "replay_source": "muc3/results/cartpole/replay.npz in "
                             "letrieu728/XRL @ main (downloaded via GitHub, "
                             "NOT a recreated rollout)",
        },
        "method": {
            "estimator": "BART heterogeneous treatment effects "
                         "(R/dbarts via rpy2, repo BARTRegressor)",
            "repo": "github.com/yanmluk/causal-xrl",
            "repo_commit": repo_commit(repo_dir),
            "repo_path_used": str(repo_dir),
            "functions": [
                "cxrl.lib.causal.infer_network.infer_causes_hte",
                "cxrl.lib.models.bart_regressor.BARTRegressor.local_treatments",
                "cxrl.lib.models.bart_regressor.BARTRegressor.ITE",
                "cxrl.lib.xrl.xrl.XRL.local_causal_network (replicated)",
            ],
            "bart_r_settings": "repo defaults: nskip=200, ndpost=125, "
                               "nchain=4, nthread=4, R seed=99",
            "split": "train_test_split(test_size=0.3) per target, "
                     "numpy global RNG seeded once before the target loop",
            "seed": args.seed,
            "query_convention": "RAW replay row [state_t, action_t] "
                                "(fidelity-notebook convention; HTE trees "
                                "were fitted on standardized data — see "
                                "diagnostic_standardized_query)",
            "ite_contrast": "+1 raw unit for state columns, 0 -> 1 for push",
            "versions": versions,
            "refit_seconds_total": refit_total,
            "refit_seconds_per_target": refit_seconds,
            "scan_seconds": scan_seconds,
        },
        "global_causes": {
            "stored_muc3": stored_causes,
            "refit_this_run": refit_causes,
            "note": "Stored lists remain the Section 3 global result. "
                    "The refit exists only to obtain the SCM for the "
                    "local model; differences reflect split/BART sampling "
                    "variability, not a correction.",
        },
        "scan": {
            "rows_scanned": len(scan),
            "valid_rows_gt1_non_action_push_cause": len(valid),
            "valid_rows_with_both_policy_inputs": len(preferred),
            "selection_rule": rule,
        },
        "selected_state": {
            "row_index": row_index,
            "state": {
                "cart_position": float(states[row_index][0]),
                "cart_velocity": float(states[row_index][1]),
                "pole_angle": float(states[row_index][2]),
                "pole_velocity": float(states[row_index][3]),
            },
            "pole_angle_degrees": float(np.degrees(states[row_index][2])),
            "action_at_t": int(actions[row_index]),
            "action_name": "push right (1)" if actions[row_index] == 1
                           else "push left (0)",
            "policy_action_matches_replay": bool(
                heuristic_policy(states[row_index]) == actions[row_index]),
            "next_state": {
                "cart_position": float(next_states[row_index][0]),
                "cart_velocity": float(next_states[row_index][1]),
                "pole_angle": float(next_states[row_index][2]),
                "pole_velocity": float(next_states[row_index][3]),
            },
            "action_at_t_plus_1": int(actions[row_index + 1]),
            "note": "action_at_t_plus_1 is the recorded replay action at "
                    "the next row; no BART action prediction is reported "
                    "because the SCM was fitted on standardized data and "
                    "predicting on the raw query row would mix scales.",
        },
        "local_causal_model": {
            "local_causes_by_target": local_edges,
            "local_edges": [
                {"source": s, "target": t, **ite_by_edge[(s, t)]}
                for (s, t) in sorted(ite_by_edge)],
            "n_local_edges": len(ite_by_edge),
            "local_push_causes": local_edges["push"],
            "n_non_action_local_push_causes": sum(
                1 for s in local_by_target[ACTION_INDEX]
                if s != ACTION_INDEX),
            "push_target_all_treatment_ite": push_all_ite,
        },
        "diagnostic_standardized_query": diagnostic,
        "caveats": [
            "Local edges come from a fresh BART refit on the Mục 3 "
            "replay, not from the stored Mục 3 cause lists (the SCM "
            "was not saved). Refit global causes are recorded above "
            "for comparison.",
            "The query row is raw-scale while the HTE trees were fitted "
            "on standardized data (authors' notebook behaviour). The "
            "standardized-query diagnostic in this file shows whether "
            "the local model changes under a standardized query.",
            "ITE contrasts are +1 raw unit per state column, so "
            "magnitudes are not unit-free importances across columns "
            "with different scales.",
            "The local model is a behavioural surrogate of the "
            "heuristic policy over collected trajectories, not the "
            "policy's internal reasoning (paper §4.3.3).",
        ],
    }

    out_dir = args.out_dir or (HERE.parent / "results" / "cartpole")
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "local_case_study.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    png_path = graphs_dir / "local_case_study.png"
    draw_case_study_figure(
        png_path, states[row_index], int(actions[row_index]),
        local_by_target, ite_by_edge, row_index, n_rows,
        {"seed": args.seed})
    print(f"wrote {png_path}")
    print(f"local edges ({len(ite_by_edge)}):")
    for (s, t) in sorted(ite_by_edge):
        w = ite_by_edge[(s, t)]
        print(f"  {s} -> {t}: ITE mean {w['ite_mean']:+.4f} "
              f"95% CI ({w['ite_lci_95']:+.4f}, {w['ite_uci_95']:+.4f})")


if __name__ == "__main__":
    main()
