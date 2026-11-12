import json, os, pickle, sys
import numpy as np
import sklearn.tree._tree as t
if not hasattr(t, "DTYPE"):
    t.DTYPE = np.float32
sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
import rpy2.robjects.numpy2ri; rpy2.robjects.numpy2ri.activate()
from cxrl_microRTS.lib.models.bart_regressor import BARTRegressor  # noqa: F401
G = "/home/hatch/workspace/step3-scratch/group2"; CK = os.path.join(G, "microrts100/ckpt")
d = np.load(os.path.join(G, "microrts100/microrts_merged.npz"))
st = d["states_t"].astype(float); lab = d["action_label"].astype(float)
nv = d["n_valid"]; ep = d["ep_id"]
FEATS = ["base1","barrack1","worker1","light1","heavy1","ranged1","base2","barrack2","worker2","light2","heavy2","ranged2","timestep","action"]
def load(name):
    with open(os.path.join(CK, f"bart_{name}.pkl"), "rb") as f: b = pickle.load(f)
    with open(os.path.join(CK, f"scaler_{name}.pkl"), "rb") as f: s = pickle.load(f)
    with open(os.path.join(CK, f"num_cols_{name}.json")) as f: nc = json.load(f)["num_cols"]
    return b, s, nc
# action BART: which columns ever flag?
b, s, nc = load("action")
qA = []
for e in np.unique(ep):
    idx = np.where(ep == e)[0]
    for a, bb in zip(idx[:-1], idx[1:]):
        if nv[a] > 0 and nv[bb] > 0:
            qA.append(np.concatenate([st[a], [lab[a]]]))
qA = np.array(qA); qs = qA.copy(); qs[:, nc] = s.transform(qA[:, nc])
flagcols = {}
for i in range(qs.shape[0]):
    local = np.asarray(b.local_treatments(qs[i:i+1]))[0]
    for c in np.nonzero(local)[0]:
        flagcols[int(c)] = flagcols.get(int(c), 0) + 1
print("action BART flag counts by column:", {FEATS[c]: n for c, n in sorted(flagcols.items())})
# worker1 control (scaled)
b2, s2, nc2 = load("worker1")
qs2 = st.copy(); qs2[:, nc2] = s2.transform(st[:, nc2])
cnt = []
for i in range(0, 1000):
    local = np.asarray(b2.local_treatments(qs2[i:i+1]))[0]
    cnt.append(int(np.count_nonzero(local)))
cnt = np.array(cnt)
print(f"worker1 control scaled n=1000 flagged dist={dict(zip(*np.unique(cnt, return_counts=True)))} rows>1={int((cnt>1).sum())}")
print("DIAG2 DONE")
