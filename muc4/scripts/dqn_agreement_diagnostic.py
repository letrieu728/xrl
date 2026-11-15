#!/usr/bin/env python3
"""OPTIONAL diagnostic: DQN (run 9) vs heuristic agreement on the
graph-surgery queries. NOT part of the pre-registered success criteria.

Reads the per-query JSONs written by ``run_graph_surgery_cartpole.py``
and a locally downloaded DQN checkpoint, and reports:

* DQN-vs-heuristic agreement at the intervened states (type A) and at
  the real next states (types A and B);
* agreement between the graph model's predicted ``push'`` and the DQN
  action at the real next state (for context only — the criteria use
  the heuristic, the authors' policy).

The checkpoint (.pt) is NEVER committed to the repo: download it from
the Drive folder "XRL - file nhi phan" into a local work directory and
pass its path with ``--checkpoint``.

Requires torch (CPU). The existing ``~/workspace/dqn-venv`` provides
it; run with that venv's python::

    ~/workspace/dqn-venv/bin/python muc4/scripts/dqn_agreement_diagnostic.py \\
        --results-dir muc4/results/graph-surgery-simulation \\
        --checkpoint /path/to/work/dqn_run9_checkpoint.pt

Network architecture MUST match ``train_dqn_cartpole.py``
(4 -> 128 -> 128 -> 2, hidden size read from the checkpoint).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_dqn_policy(checkpoint_path):
    import torch
    import torch.nn as nn

    class QNetwork(nn.Module):
        def __init__(self, state_dim=4, hidden=128, action_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )

        def forward(self, x):
            return self.net(x)

    ckpt = torch.load(checkpoint_path, map_location="cpu",
                      weights_only=False)
    hidden = int(ckpt.get("hyperparams", {}).get("hidden", 128))
    net = QNetwork(hidden=hidden)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    def pi(state):
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            return int(torch.argmax(net(s), dim=1).item())

    return pi, ckpt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pi, ckpt = load_dqn_policy(args.checkpoint)
    print("checkpoint keys:", sorted(ckpt.keys()))
    hp = ckpt.get("hyperparams", {})
    print("hyperparams:", hp)

    rows = []
    for fname in ("queries_type_a.json", "queries_type_b.json"):
        rows.extend(json.loads((args.results_dir / fname).read_text()))
    print(f"queries: {len(rows)}")

    out_rows = []
    for r in rows:
        if r["type"] == "state":
            s_do = list(r["base_state"])
            j = ["cart_position", "cart_velocity", "pole_angle",
                 "pole_velocity"].index(r["intervened_feature"])
            s_do[j] = r["intervened_value"]
        else:
            s_do = list(r["base_state"])
        dqn_at_state = pi(s_do)
        dqn_at_next = pi(r["real_next_state"])
        out_rows.append({
            "query_id": r["query_id"], "type": r["type"],
            "dqn_action_at_query_state": dqn_at_state,
            "dqn_action_at_real_next_state": dqn_at_next,
            "heuristic_next_action": r["real_next_action"],
            "graph_pred_push_next": r["graph_pred_push_next"],
            "dqn_matches_heuristic_at_next": bool(
                dqn_at_next == r["real_next_action"]),
            "graph_matches_dqn_at_next": bool(
                r["graph_pred_push_next"] == dqn_at_next),
        })
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_hyperparams": {k: hp.get(k) for k in
                                   ("hidden", "seed", "episodes")},
        "n_queries": len(out_rows),
        "dqn_vs_heuristic_at_real_next_state_agreement": float(np.mean(
            [o["dqn_matches_heuristic_at_next"] for o in out_rows])),
        "graph_pred_vs_dqn_at_real_next_state_agreement": float(np.mean(
            [o["graph_matches_dqn_at_next"] for o in out_rows])),
        "note": "Diagnostic only; not part of the pre-registered criteria.",
        "per_query": out_rows,
    }
    out = args.out or (args.results_dir / "dqn_diagnostic.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_query"},
                     indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
