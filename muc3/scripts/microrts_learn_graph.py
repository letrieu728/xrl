"""Stage S6c2: per-target global graph with the authors' infer_causes_hte.

infer_DBN aborts the WHOLE graph when one target crashes BART (observed:
heavy2, sigma estimate 0 on near-constant target). This driver calls the
authors' infer_causes_hte once per target (same function infer_DBN
calls for method='hte', same state_integration=True differencing) and
records per-target outcome: causes found / constant target (skipped by
the same rule infer_DBN uses) / BART error (message recorded, target
marked NOT COMPUTED, never silently filled with []).

Inputs: merged trajectories from S6b (2 ep) + S6b2 (8 ep).
"""
import sys, time, json, traceback
import numpy as np
import sklearn.tree._tree as t
if not hasattr(t, "DTYPE"):
    t.DTYPE = np.float32

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
from cxrl_microRTS.lib.xrl_causal.xrl_causal import infer_causes_hte
from cxrl_microRTS.lib.utils.utils import convert_and_expand
import networkx as nx

FEATURES = ["base1", "barrack1", "worker1", "light1", "heavy1", "ranged1",
            "base2", "barrack2", "worker2", "light2", "heavy2", "ranged2", "timestep"]

d1 = np.load("/home/hatch/workspace/step3-scratch/microrts_traj.npz")
d2 = np.load("/home/hatch/workspace/step3-scratch/microrts_traj8.npz")
states = np.vstack([d1["states_t"], d2["states_t"]]).astype(float)
states_new = np.vstack([d1["states_tp1"], d2["states_tp1"]]).astype(float)
states = convert_and_expand(states).astype(float)
states_new = convert_and_expand(states_new).astype(float)
print(f"merged: {states.shape[0]} transitions from 10 episodes", flush=True)

G = nx.DiGraph()
results = {}
t_all = time.time()
for col in range(states.shape[1]):
    name = FEATURES[col]
    target = states_new[:, col] - states[:, col]
    nz = int((target != 0).sum())
    if len(np.unique(target)) < 2:
        print(f"[{name}] CONSTANT target -> skipped (infer_DBN rule), causes=[]", flush=True)
        results[name] = {"status": "constant", "nonzero": nz, "causes": []}
        continue
    t0 = time.time()
    try:
        causes, bart, scaler = infer_causes_hte(states, target)
        cnames = [FEATURES[c] for c in causes]
        print(f"[{name}] nonzero={nz} causes={cnames} ({time.time()-t0:.1f}s)", flush=True)
        results[name] = {"status": "ok", "nonzero": nz, "causes": cnames}
        for c in causes:
            G.add_edge(f"state_{FEATURES[c]}_t", f"state_{name}_t+1")
    except Exception as e:
        print(f"[{name}] nonzero={nz} ERROR: {type(e).__name__}: {str(e)[:200]}", flush=True)
        results[name] = {"status": "error", "nonzero": nz,
                         "error": f"{type(e).__name__}: {str(e)[:300]}",
                         "causes": None}
print(f"total {time.time()-t_all:.1f}s; DBN edges={G.number_of_edges()}", flush=True)

# DBN_to_causal equivalent (same logic as xrl_causal.DBN_to_causal)
cg = nx.DiGraph()
for n in [n for n in G.nodes() if "t+1" not in n]:
    for e in G.edges(n):
        if e[1] != f"state_{n}_t+1":
            cg.add_edge(n, e[1][:-2])
print(f"causal edges ({cg.number_of_edges()}):", sorted(cg.edges()), flush=True)
out = {"n_transitions": int(states.shape[0]), "episodes": 10, "method": "hte",
       "state_integration": True, "features": FEATURES,
       "per_target": results,
       "dbn_edges": sorted([list(e) for e in G.edges()]),
       "causal_edges": sorted([list(e) for e in cg.edges()])}
with open("/home/hatch/workspace/step3-scratch/microrts_graph10.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved microrts_graph10.json; S6c2 DONE", flush=True)
import os; os._exit(0)
