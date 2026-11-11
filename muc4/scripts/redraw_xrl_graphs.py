#!/usr/bin/env python3
"""Redraw DBN graph PNGs from the merged per_run_details.json edge lists.

Used after merging, because the PNGs saved during the runs came from
earlier runs whose edge sets differ by 1-3 edges from the merged table
(BART run-to-run variability). The DQN hte PNG is already correct
(copied from the DQN-only run); this redraws the other three so every
PNG matches comparison_table.md exactly.
"""
import json
import sys
from pathlib import Path

import networkx as nx

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "xrl-trained-vs-heuristic"
sys.path.insert(0, str(BASE / "scripts"))
from run_xrl_comparison import draw_dbn  # noqa: E402

LABEL = "BART (R/dbarts via rpy2) - real CausalXRL estimator"

details = json.loads((OUT / "per_run_details.json").read_text())
for pname, pruns in details["runs"].items():
    for method, m in pruns["methods"].items():
        png = OUT / "graphs" / f"graph_{pname}_{method}_200ep.png"
        G = nx.DiGraph()
        for e in m["edges"]:
            u, v = e.split(" -> ")
            G.add_edge(u, v)
        draw_dbn(G, png,
                 f"CartPole global causal graph ({pname}, {method}, "
                 f"200 episodes)\n{LABEL}")
        print(f"redrew {png.name} ({m['n_edges']} edges)")
