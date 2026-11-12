"""Stage S6c: learn a global MicroRTS causal graph with the authors'
xrl_causal.infer_DBN (method='hte', as in the dashboard's scm500_hte),
from the live trajectories collected in S6b. Saves edges as JSON."""
import sys, time, json
import numpy as np
import sklearn.tree._tree as t
if not hasattr(t, "DTYPE"):
    t.DTYPE = np.float32

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
from cxrl_microRTS.lib.xrl_causal.xrl_causal import infer_DBN, DBN_to_causal

FEATURES = ["base1", "barrack1", "worker1", "light1", "heavy1", "ranged1",
            "base2", "barrack2", "worker2", "light2", "heavy2", "ranged2", "timestep"]
d = np.load("/home/hatch/workspace/step3-scratch/microrts_traj.npz")
states, states_new = d["states_t"], d["states_tp1"]
print(f"input: states={states.shape} states_new={states_new.shape}", flush=True)
t0 = time.time()
G, scm, G_scaler = infer_DBN(states, states_new, state_features=FEATURES,
                             method="hte", state_integration=True)
print(f"infer_DBN(hte) in {time.time()-t0:.1f}s: DBN nodes={G.number_of_nodes()} edges={G.number_of_edges()}", flush=True)
for e in sorted(G.edges()):
    print("  DBN edge:", e, flush=True)
cg = DBN_to_causal(G)
print(f"causal graph: nodes={cg.number_of_nodes()} edges={cg.number_of_edges()}", flush=True)
for e in sorted(cg.edges()):
    print("  causal edge:", e, flush=True)
out = {"n_transitions": int(states.shape[0]), "method": "hte",
       "state_integration": True, "features": FEATURES,
       "dbn_edges": sorted([list(e) for e in G.edges()]),
       "causal_edges": sorted([list(e) for e in cg.edges()])}
with open("/home/hatch/workspace/step3-scratch/microrts_graph.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved microrts_graph.json; S6c DONE", flush=True)
import os; os._exit(0)
