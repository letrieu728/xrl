#!/usr/bin/env python3
"""Plot global causal graphs for CausalXRL Section 3 (Muc 3) from the
learned per-target cause lists stored in the XRL repo.

DATA SOURCE (read, never invented)
----------------------------------
Fetched 2026-11-12 from GitHub repo ``letrieu728/XRL``, branch ``main``,
via the GitHub connector (``get_file_contents``). The JSON files read by
this script are verbatim copies kept next to it in the repo layout::

    muc3/results/cartpole/metrics.json          (GitHub blob SHA 642816c...)
    muc3/results/lunarlander/metrics.json       (GitHub blob SHA a487cbe...)
    muc3/results/pongduel/targets/target_*.json (21 checkpoint files;
        muc3/results/pongduel/metrics.json stores NO per-target causes,
        only timings/baselines, so the checkpoint files are the source)
    muc3/results/<env>/collection_metadata.json (episodes/transitions/seed)

The cause lists were learned on Codex's Windows run (seed 99) with the
authors' ``infer_causes_hte`` (BART heterogeneous treatment effects,
``cxrl/lib/causal/infer_network.py`` @ yanmluk/causal-xrl ae9096e), one
isolated Python/R process per target. Verification performed at fetch
time: the summed per-target ``elapsed_seconds`` equal the
``bart_hte_isolated_target_sum`` in each ``metrics.json`` exactly
(CartPole 242.19 s, LunarLander 1086.40 s, PongDuel 35531.20 s),
confirming the PongDuel checkpoint set is complete (21/21 targets).

GRAPH CONVENTION
----------------
Each learned cause relation is a DBN edge from a variable at time ``t``
to a variable at time ``t+1``: ``cause (t) -> target (t+1)``. The
figures collapse the two time slices into one layer (one node per
variable), following the authors' ``DBN_to_causal`` collapse in the
same file: a persistence relation ``X(t) -> X(t+1)`` (cause == target)
is NOT drawn as a self-loop; instead the node label carries a ``(p)``
marker and the figure states how many persistence relations were
learned. Drawn arrows are therefore the non-self cause relations only,
and every figure prints BOTH counts (total learned relations, drawn
arrows, persistence relations) so nothing is hidden. Variables with no
learned relation at all are still drawn (grey, dashed) so the node set
is complete. No edge weights were stored in the source files, so all
arrows are drawn uniformly; no weights are shown or implied.

OUTPUTS (repo layout, under ``muc3/results/<env>/graphs/``)
-----------------------------------------------------------
    global_graph_cartpole.png
    global_graph_lunarlander.png
    global_graph_pongduel.png

USAGE
-----
Run with a Python that has networkx + matplotlib, e.g. the Mục 4 venv::

    .venv-ablation/bin/python muc3/scripts/plot_muc3_graphs.py [--env ENV]

``--env`` may be repeated; default is all three environments. The
script resolves data paths relative to its own location (``../results``),
so it reruns standalone inside ``xrl-g1d`` and after merging into the
XRL repo (where the same JSON files already live).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

STATE_COLOR = "#cfe8ff"   # same state/action colours as Mục 4 draw_dbn
ACTION_COLOR = "#ffe4c2"
ISOLATED_COLOR = "#e8e8e8"
EDGE_COLOR = "#444444"

# Variable order and action columns, exactly as in the Mục 3 target
# scripts (cartpole_bart_target.py / lunarlander_bart_target.py /
# pongduel_bart_target.py).
ENVS = {
    "cartpole": {
        "title_env": "CartPole-v0",
        "features": ["cart_position", "cart_velocity", "pole_angle",
                     "pole_velocity", "push"],
        "actions": {"push"},
        "source": "muc3/results/cartpole/metrics.json (bart_targets)",
    },
    "lunarlander": {
        "title_env": "LunarLander-v2",
        "features": ["x", "y", "vx", "vy", "angle", "angular_velocity",
                     "left_contact", "right_contact",
                     "do_nothing", "fire_left_engine",
                     "fire_main_engine", "fire_right_engine"],
        "actions": {"do_nothing", "fire_left_engine",
                    "fire_main_engine", "fire_right_engine"},
        "source": ("muc3/results/lunarlander/metrics.json (bart_targets; "
                   "one-hot actions)"),
    },
    "pongduel": {
        "title_env": "PongDuel-v0",
        "features": ["agent1_row", "agent1_col", "ball_row1", "ball_col1",
                     "dir_NW1", "dir_W1", "dir_SW1", "dir_SE1", "dir_E1",
                     "dir_NE1", "agent2_row", "agent2_col", "ball_row2",
                     "ball_col2", "dir_NW2", "dir_W2", "dir_SW2",
                     "dir_SE2", "dir_E2", "dir_NE2", "encoded_action"],
        "actions": {"encoded_action"},
        "source": ("muc3/results/pongduel/targets/target_0..20.json "
                   "(per-target checkpoints; metrics.json has no causes)"),
    },
}


def load_env(env):
    """Return (causes dict target -> [causes], metadata dict)."""
    cfg = ENVS[env]
    env_dir = RESULTS / env
    meta = json.loads((env_dir / "collection_metadata.json").read_text())
    if env == "pongduel":
        targets = []
        for i in range(len(cfg["features"])):
            targets.append(json.loads(
                (env_dir / "targets" / f"target_{i}.json").read_text()))
    else:
        metrics = json.loads((env_dir / "metrics.json").read_text())
        targets = metrics["bart_targets"]
    causes = {}
    for t in targets:
        assert t["target"] in cfg["features"], t["target"]
        for c in t["causes"]:
            assert c in cfg["features"], c
        causes[t["target"]] = list(t["causes"])
    assert set(causes) == set(cfg["features"]), "missing target(s)"
    return causes, meta


def split_edges(features, causes):
    """Split learned relations into drawn arrows and persistence."""
    arrows = set()
    persistence = set()
    for target, cs in causes.items():
        for c in cs:
            if c == target:
                persistence.add(target)
            else:
                arrows.add((c, target))
    total = sum(len(cs) for cs in causes.values())
    assert total == len(arrows) + len(persistence)
    return sorted(arrows), sorted(persistence), total


def header_text(cfg, meta, total, n_arrows, n_persist):
    return (
        f"{cfg['title_env']} — global causal graph (BART HTE, DBN t→t+1, "
        f"collapsed to one layer)\n"
        f"BART heterogeneous treatment effects (dbarts via rpy2) · "
        f"{meta['episodes']} episodes · {meta['transitions']:,} transitions "
        f"· seed {meta['seed']}\n"
        f"{total} learned cause relations = {n_arrows} drawn arrows + "
        f"{n_persist} persistence relations (p)"
    )


FOOTNOTE = (
    "Edges are learned cause relations cause(t) → target(t+1) from the "
    "per-target cause lists; no edge weights were stored, so arrows are "
    "unweighted. “(p)” on a node means the variable's own value at t was "
    "learned as a cause of its value at t+1 (persistence); shown as a "
    "marker, not a loop, following the authors' DBN_to_causal collapse "
    "(yanmluk/causal-xrl @ ae9096e, cxrl/lib/causal/infer_network.py)."
)


def draw_edges(ax, pos, arrows, node_size, width=1.4, mutation_scale=16,
               line_sides=None):
    """Draw arrows with FancyArrowPatch so arrowheads stop at node borders.

    Nodes must be drawn BEFORE calling this. Curvature is deterministic:
    bidirectional pairs curve in opposite directions; edges sharing a
    row/column fan out toward the ``line_sides`` direction for that line
    (a unit bulge vector keyed by ("x", x) / ("y", y)), with capped
    bulges so long edges do not sweep across labels; other edges get a
    slight curve.
    """
    import math
    from matplotlib.patches import FancyArrowPatch

    radius = math.sqrt(node_size / math.pi)  # node radius in points
    line_sides = line_sides or {}
    arrow_set = set(arrows)
    line_count = {}
    rads = {}
    for (u, v) in sorted(arrows):
        if (v, u) in arrow_set:
            rads[(u, v)] = 0.25 if (u, v) < (v, u) else -0.25
        elif pos[u][0] == pos[v][0] or pos[u][1] == pos[v][1]:
            key = ("x", pos[u][0]) if pos[u][0] == pos[v][0] else ("y", pos[u][1])
            k = line_count.get(key, 0)
            line_count[key] = k + 1
            span = math.dist(pos[u], pos[v])
            bulge = [0.15, 0.28, 0.41, 0.54][k % 4]
            rad = 2.0 * bulge / max(span, 1e-9)
            desired = line_sides.get(key)
            if desired is not None:
                dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
                # arc3 positive rad bulges to the LEFT of travel (-dy, dx)
                left = (-dy / span, dx / span)
                if left[0] * desired[0] + left[1] * desired[1] < 0:
                    rad = -rad
            elif k % 2 == 1:
                rad = -rad
            rads[(u, v)] = rad
        else:
            rads[(u, v)] = 0.05
    for (u, v) in arrows:
        arr = FancyArrowPatch(
            pos[u], pos[v], arrowstyle="-|>", mutation_scale=mutation_scale,
            linewidth=width, color=EDGE_COLOR,
            connectionstyle=f"arc3,rad={rads[(u, v)]}",
            shrinkA=radius + 2, shrinkB=radius + 3, zorder=3)
        ax.add_patch(arr)


def plot_cartpole(causes, meta, out_path):
    cfg = ENVS["cartpole"]
    features = cfg["features"]
    arrows, persistence, total = split_edges(features, causes)
    isolated = [n for n in features
                if not causes[n] and all(n not in cs for cs in causes.values())]
    # Manual layout: states in a 2x2 block, action to the right.
    pos = {
        "cart_position": (0.0, 1.0),
        "cart_velocity": (1.0, 1.0),
        "pole_angle": (0.0, 0.0),
        "pole_velocity": (1.0, 0.0),
        "push": (2.0, 0.5),
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    G = nx.DiGraph()
    G.add_nodes_from(features)
    colors = [ACTION_COLOR if n in cfg["actions"] else STATE_COLOR
              for n in features]
    nx.draw_networkx_nodes(G, pos, nodelist=features, node_color=colors,
                           node_size=1500, edgecolors="#555555",
                           linewidths=1.0, ax=ax)
    draw_edges(ax, pos, arrows, node_size=1500, width=1.6, mutation_scale=20,
               # Sides are inverted vs. the nominal bulge direction (see
               # the PongDuel note); these values bow the arcs OUTWARD.
               line_sides={("y", 1.0): (0, -1), ("y", 0.0): (0, 1),
                           ("x", 0.0): (1, 0)})
    labels = {n: (n + " (p)" if n in persistence else n) for n in features}
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
                                verticalalignment="bottom", ax=ax)
    for n in ("pole_angle", "pole_velocity"):
        nx.draw_networkx_labels(G, {n: label_pos[n]}, labels={n: labels[n]},
                                font_size=11, horizontalalignment="center",
                                verticalalignment="top", ax=ax)
    nx.draw_networkx_labels(G, {"push": label_pos["push"]},
                            labels={"push": labels["push"]}, font_size=11,
                            horizontalalignment="left",
                            verticalalignment="center", ax=ax)
    ax.set_title(header_text(cfg, meta, total, len(arrows),
                             len(persistence)), fontsize=11, pad=12)
    fig.text(0.5, 0.01, f"Source: {cfg['source']} + collection_metadata.json "
             f"(letrieu728/XRL @ main). {FOOTNOTE}",
             ha="center", va="bottom", fontsize=8, wrap=True,
             bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none"))
    ax.set_xlim(-0.7, 2.7)
    ax.set_ylim(-0.55, 1.55)
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return total, len(arrows), len(persistence)


def plot_lunarlander(causes, meta, out_path):
    cfg = ENVS["lunarlander"]
    features = cfg["features"]
    arrows, persistence, total = split_edges(features, causes)
    isolated = [n for n in features
                if not causes[n] and all(n not in cs for cs in causes.values())]
    states = [f for f in features if f not in cfg["actions"]]
    actions = [f for f in features if f in cfg["actions"]]
    pos = {}
    for i, n in enumerate(states):
        pos[n] = (0.0, len(states) - 1 - i)
    for i, n in enumerate(actions):
        pos[n] = (1.0, 6.0 - 2.0 * i)
    label_pos = {}
    for n in states:
        label_pos[n] = (pos[n][0] - 0.05, pos[n][1])
    for n in actions:
        label_pos[n] = (pos[n][0] + 0.05, pos[n][1])
    fig, ax = plt.subplots(figsize=(12, 9))
    # Labels sit outside the columns, aligned outward.
    colors = [ACTION_COLOR if n in cfg["actions"] else STATE_COLOR
              for n in features]
    G = nx.DiGraph()
    G.add_nodes_from(features)
    nx.draw_networkx_nodes(G, pos, nodelist=features, node_color=colors,
                           node_size=1100, edgecolors="#555555",
                           linewidths=1.0, ax=ax)
    draw_edges(ax, pos, arrows, node_size=1100, width=1.3, mutation_scale=16,
               line_sides={("x", 0.0): (1, 0), ("x", 1.0): (-1, 0)})
    labels = {n: (n + " (p)" if n in persistence else n) for n in features}
    nx.draw_networkx_labels(
        G, {n: label_pos[n] for n in states},
        labels={n: labels[n] for n in states}, font_size=10,
        horizontalalignment="right", ax=ax)
    nx.draw_networkx_labels(
        G, {n: label_pos[n] for n in actions},
        labels={n: labels[n] for n in actions}, font_size=10,
        horizontalalignment="left", ax=ax)
    ax.set_title(header_text(cfg, meta, total, len(arrows),
                             len(persistence)), fontsize=11, pad=12)
    fig.text(0.5, 0.01, f"Source: {cfg['source']} + collection_metadata.json "
             f"(letrieu728/XRL @ main). Action nodes are the one-hot action "
             f"columns. {FOOTNOTE}",
             ha="center", va="bottom", fontsize=8, wrap=True,
             bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none"))
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.7, 7.7)
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return total, len(arrows), len(persistence)


def plot_pongduel(causes, meta, out_path):
    cfg = ENVS["pongduel"]
    features = cfg["features"]
    arrows, persistence, total = split_edges(features, causes)
    isolated = [n for n in features
                if not causes[n] and all(n not in cs for cs in causes.values())]
    # Two rows: player-1/ball-1 variables on top, player-2/ball-2 below,
    # so each "1" variable sits above its "2" counterpart; the joint
    # action sits at the right. Isolated columns stay in index order.
    top = features[0:10]
    bottom = features[10:20]
    pos = {}
    for i, n in enumerate(top):
        pos[n] = (float(i), 1.0)
    for i, n in enumerate(bottom):
        pos[n] = (float(i), 0.0)
    pos["encoded_action"] = (10.5, 0.5)
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = []
    for n in features:
        if n in isolated:
            colors.append(ISOLATED_COLOR)
        elif n in cfg["actions"]:
            colors.append(ACTION_COLOR)
        else:
            colors.append(STATE_COLOR)
    G = nx.DiGraph()
    G.add_nodes_from(features)
    nx.draw_networkx_nodes(G, pos, nodelist=features, node_color=colors,
                           node_size=950, edgecolors="#555555",
                           linewidths=1.0, ax=ax)
    draw_edges(ax, pos, arrows, node_size=950, width=1.2, mutation_scale=15,
               # NOTE: arc3 bulges opposite to the nominal "desired" vector
               # on this anisotropic axes, verified visually 2026-11-12;
               # these sides bow the row arcs INTO the middle band.
               line_sides={("y", 1.0): (0, 1), ("y", 0.0): (0, -1)})
    labels = {n: (n + " (p)" if n in persistence else n) for n in features}
    top_pos = {n: (pos[n][0], pos[n][1] + 0.14) for n in top}
    nx.draw_networkx_labels(G, top_pos, labels={n: labels[n] for n in top},
                            font_size=9, horizontalalignment="center",
                            verticalalignment="bottom", ax=ax)
    bot_pos = {n: (pos[n][0], pos[n][1] - 0.14) for n in bottom}
    nx.draw_networkx_labels(G, bot_pos,
                            labels={n: labels[n] for n in bottom},
                            font_size=9, horizontalalignment="center",
                            verticalalignment="top", ax=ax)
    nx.draw_networkx_labels(
        G, {"encoded_action": (10.5, 0.68)},
        labels={"encoded_action": "encoded_action"},
        font_size=9, horizontalalignment="center",
        verticalalignment="bottom", ax=ax)
    ax.set_title(header_text(cfg, meta, total, len(arrows),
                             len(persistence)), fontsize=12, pad=12)
    grey_note = (f"Grey node(s) ({', '.join(isolated)}) have no learned "
                 f"cause or effect. " if isolated else "")
    fig.text(0.5, 0.01, f"Source: {cfg['source']} + collection_metadata.json "
             f"(letrieu728/XRL @ main). {grey_note}{FOOTNOTE}",
             ha="center", va="bottom", fontsize=8, wrap=True,
             bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none"))
    ax.set_xlim(-0.7, 11.4)
    ax.set_ylim(-0.65, 1.65)
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return total, len(arrows), len(persistence)


PLOTTERS = {
    "cartpole": plot_cartpole,
    "lunarlander": plot_lunarlander,
    "pongduel": plot_pongduel,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", action="append", choices=list(ENVS),
                    default=None, help="environment(s) to plot (default: all)")
    args = ap.parse_args()
    envs = args.env or list(ENVS)
    for env in envs:
        causes, meta = load_env(env)
        out_dir = RESULTS / env / "graphs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"global_graph_{env}.png"
        total, n_arrows, n_persist = PLOTTERS[env](causes, meta, out_path)
        print(f"{env}: {total} learned relations "
              f"({n_arrows} arrows + {n_persist} persistence) -> {out_path}")
        for target in ENVS[env]["features"]:
            print(f"    {target}: causes={causes[target]}")


if __name__ == "__main__":
    main()
