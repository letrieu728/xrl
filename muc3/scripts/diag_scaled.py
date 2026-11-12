import json, os, pickle, sys
import numpy as np
import sklearn.tree._tree as t
if not hasattr(t, "DTYPE"):
    t.DTYPE = np.float32
sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from cxrl_microRTS.lib.models.bart_regressor import BARTRegressor  # noqa: F401
G = "/home/hatch/workspace/step3-scratch/group2"
CK = os.path.join(G, "microrts100/ckpt")
d = np.load(os.path.join(G, "microrts100/microrts_merged.npz"))
st = d["states_t"].astype(float); lab = d["action_label"].astype(float)
nv = d["n_valid"]; ep = d["ep_id"]
with open(os.path.join(CK, "bart_action.pkl"), "rb") as f:
    bart_action = pickle.load(f)
with open(os.path.join(CK, "scaler_action.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(CK, "num_cols_action.json")) as f:
    num_cols = json.load(f)["num_cols"]
def scale(q):
    qs = q.copy(); qs[:, num_cols] = scaler.transform(q[:, num_cols]); return qs
qA = []
for e in np.unique(ep):
    idx = np.where(ep == e)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if nv[a] > 0 and nv[b] > 0:
            qA.append(np.concatenate([st[a], [lab[a]]]))
qA = np.array(qA)
keep = np.where(nv > 0)[0]
qB = np.concatenate([st[keep], lab[keep][:, None]], axis=1)
for name, q in [("A", qA), ("B", qB)]:
    qs = scale(q)
    cnt = []
    for i in range(qs.shape[0]):
        local = np.asarray(bart_action.local_treatments(qs[i:i+1]))[0]
        cnt.append(int(np.count_nonzero(np.delete(local, 13))))
    cnt = np.array(cnt)
    print(f"[{name} SCALED] n={qs.shape[0]} flagged(excl13) dist={dict(zip(*np.unique(cnt, return_counts=True)))} valid>1={int((cnt>1).sum())}")
ite = bart_action.ITE(scale(qA[0:1]), 10, additive=(10 in num_cols))
print("ITE(heavy2) row0:", np.asarray(ite, dtype=float).ravel()[:5], "shape", np.asarray(ite).shape)
print("DIAG SCALED DONE")
