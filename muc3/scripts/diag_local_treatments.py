"""Diagnostic: local_treatments flag-count distribution for the action BART.

Answers: is fidelity valid=0 a bug or a genuine outcome of the notebook
protocol (>1 local incoming edge into the action node)? Checks the action
BART on all A/B queries, plus the worker1 BART as a sanity control (a
target with 5 global causes should flag multiple treatments if the
mechanism works at all).
"""
import json
import os
import pickle
import sys

import numpy as np
import sklearn.tree._tree as t

if not hasattr(t, "DTYPE"):
    t.DTYPE = np.float32

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
from cxrl_microRTS.lib.models.bart_regressor import BARTRegressor  # noqa: E402,F401

G = "/home/hatch/workspace/step3-scratch/group2"
CK = os.path.join(G, "microrts100/ckpt")
d = np.load(os.path.join(G, "microrts100/microrts_merged.npz"))
st = d["states_t"].astype(float)
lab = d["action_label"].astype(float)
nv = d["n_valid"]
ep = d["ep_id"]

with open(os.path.join(CK, "bart_action.pkl"), "rb") as f:
    bart_action = pickle.load(f)

# Construction A queries
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
    counts, counts_no13 = [], []
    for i in range(q.shape[0]):
        local = np.asarray(bart_action.local_treatments(q[i:i + 1]))[0]
        counts.append(int(np.count_nonzero(local)))
        counts_no13.append(int(np.count_nonzero(
            np.delete(local, 13) if local.shape[0] > 13 else local)))
    counts = np.array(counts)
    counts_no13 = np.array(counts_no13)
    print(f"[{name}] n={q.shape[0]} flagged(all) dist={dict(zip(*np.unique(counts, return_counts=True)))}")
    print(f"[{name}] flagged(excl col13) dist={dict(zip(*np.unique(counts_no13, return_counts=True)))}")
    print(f"[{name}] rows with >1 non-action flag: {int((counts_no13 > 1).sum())}")

# Sanity control: worker1 BART (5 global causes) on raw state rows
with open(os.path.join(CK, "bart_worker1.pkl"), "rb") as f:
    bart_w = pickle.load(f)
counts = []
for i in range(0, 500):
    local = np.asarray(bart_w.local_treatments(st[i:i + 1]))[0]
    counts.append(int(np.count_nonzero(local)))
counts = np.array(counts)
print(f"[worker1 control] n=500 flagged dist={dict(zip(*np.unique(counts, return_counts=True)))}")
print(f"[worker1 control] rows with >1 flag: {int((counts > 1).sum())}")

# ITE magnitudes for the action BART on a few rows, treatment=heavy2 (col 10)
with open(os.path.join(CK, "num_cols_action.json")) as f:
    num_cols_action = json.load(f)["num_cols"]
print("num_cols_action:", num_cols_action)
for i in [0, 100, 500]:
    q = qA[i:i + 1]
    ite = bart_action.ITE(q, 10, additive=(10 in num_cols_action))
    print(f"row {i}: ITE(heavy2)={np.asarray(ite, dtype=float).ravel()}")
print("DIAG DONE")
