"""Group 2a: merge per-target checkpoints into the global graph + dashboard SCM.

Inputs: --ckptdir with target_*.json (+ bart_*.pkl / scaler_*.pkl),
        --merged npz (for counts/provenance).
Outputs:
  * microrts_graph.json -- per-target results, DBN edges, causal edges
    (DBN_to_causal equivalent, same logic as the authors' function),
    total learning time. State targets only for the DBN (13 vars);
    the action target is reported separately (its causes live in the
    14-var pair space, not the dashboard's 13-var space).
  * scm_hte.pkl / scaler_hte.pkl -- dicts keyed by state column index
    0..12 (None for constant/error targets; no scaler entry for those),
    in the exact format app.py's /compute_global_causal loads
    (filenames on disk for the dashboard are scm500_hte.pickle /
    scaler500_hte.pickle; the "500" in the authors' names is kept by
    the dashboard route, NOT claimed as our episode count -- see the
    2b report for the naming caveat).
  * strategy_states.npz -- key 'strategy_states': merged states_t
    (float64), the array /compute_global_causal loads as
    data/strategy_state_epwt.npz.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
from cxrl_microRTS.lib.xrl_causal.xrl_causal import DBN_to_causal  # noqa: E402

FEATURES = ["base1", "barrack1", "worker1", "light1", "heavy1", "ranged1",
            "base2", "barrack2", "worker2", "light2", "heavy2", "ranged2", "timestep"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckptdir", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    d = np.load(args.merged)
    per_target = {}
    total_time = 0.0
    G = nx.DiGraph()
    scm, scalers = {}, {}
    for col, name in enumerate(FEATURES):
        with open(os.path.join(args.ckptdir, f"target_{name}.json")) as f:
            rec = json.load(f)
        per_target[name] = rec
        total_time += float(rec.get("time_s", 0.0))
        if rec["status"] == "ok":
            for c in rec["cause_idx"]:
                G.add_edge(f"state_{FEATURES[c]}_t", f"state_{name}_t+1")
            with open(os.path.join(args.ckptdir, f"bart_{name}.pkl"), "rb") as f:
                scm[col] = pickle.load(f)
            with open(os.path.join(args.ckptdir, f"scaler_{name}.pkl"), "rb") as f:
                scalers[col] = pickle.load(f)
        else:
            scm[col] = None
    # action target (reported, not part of the 13-var DBN)
    action_rec = None
    apath = os.path.join(args.ckptdir, "target_action.json")
    if os.path.exists(apath):
        with open(apath) as f:
            action_rec = json.load(f)
        total_time += float(action_rec.get("time_s", 0.0))
    cg = DBN_to_causal(G)
    out = {
        "n_transitions": int(d["states_t"].shape[0]),
        "episodes": int(len(np.unique(d["ep_id"]))),
        "episode_ids": [int(x) for x in np.unique(d["ep_id"]).tolist()],
        "method": "hte",
        "state_integration": True,
        "features": FEATURES,
        "per_target": per_target,
        "action_target": action_rec,
        "dbn_edges": sorted([list(e) for e in G.edges()]),
        "causal_edges": sorted([list(e) for e in cg.edges()]),
        "total_learn_time_s": total_time,
        "n_dbn_edges": int(G.number_of_edges()),
        "n_causal_edges": int(cg.number_of_edges()),
    }
    with open(os.path.join(args.outdir, "microrts_graph.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.outdir, "scm_hte.pkl"), "wb") as f:
        pickle.dump(scm, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.outdir, "scaler_hte.pkl"), "wb") as f:
        pickle.dump(scalers, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez(os.path.join(args.outdir, "strategy_states.npz"),
             strategy_states=d["states_t"].astype(np.float64))
    print(f"DBN edges={out['n_dbn_edges']} causal edges={out['n_causal_edges']} "
          f"total_time={total_time:.1f}s", flush=True)
    print(f"scm keys={sorted(scm.keys())} fitted={sum(1 for v in scm.values() if v is not None)} "
          f"scaler keys={sorted(scalers.keys())}", flush=True)
    print("MERGE DONE", flush=True)


if __name__ == "__main__":
    main()
