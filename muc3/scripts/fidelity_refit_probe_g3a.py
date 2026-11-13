"""G3a Step 3 probe D: is the action target's single global cause seed/protocol-fragile?

Refits the action BART on the SAME 2832 pairs with (a) the MicroRTS
infer_causes_hte code path but a different split seed, and (b) the gym
(cxrl) infer_causes_hte variant: test_size=0.3, ONE scaler fit on
X_train applied to both halves (seeded with 99; the gym code itself is
unseeded). Diagnostic ONLY: a different cause count here does NOT
justify switching protocol -- it only measures fragility.
"""
import argparse, sys, time
import numpy as np
import sklearn.tree._tree as t
if not hasattr(t, "DTYPE"):
    t.DTYPE = np.float32
sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

G = "/home/hatch/workspace/step3-scratch/group2"
FEATS = ["base1","barrack1","worker1","light1","heavy1","ranged1","base2","barrack2","worker2","light2","heavy2","ranged2","timestep","action"]

def build_pairs():
    d = np.load(f"{G}/microrts100/microrts_merged.npz")
    st = d["states_t"].astype(float); lab = d["action_label"].astype(float)
    nv = d["n_valid"]; ep = d["ep_id"]
    Xr, yr = [], []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if nv[a] > 0 and nv[b] > 0:
                Xr.append(np.concatenate([st[a], [lab[a]]]))
                yr.append(float(lab[b]))
    return np.array(Xr), np.array(yr)

def run(variant):
    from cxrl_microRTS.lib.models.bart_regressor import BARTRegressor
    X, y = build_pairs()
    M = X.shape[1]
    num_cols = [c for c in range(M) if len(np.unique(X[:, c])) > 2]
    t0 = time.time()
    if variant == "microrts-seed1":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
        scaler_train = preprocessing.StandardScaler().fit(X_train[:, num_cols])
        X_train_s = np.copy(X_train); X_train_s[:, num_cols] = scaler_train.transform(X_train[:, num_cols])
        scaler_test = preprocessing.StandardScaler().fit(X_test[:, num_cols])
        X_test_s = np.copy(X_test); X_test_s[:, num_cols] = scaler_test.transform(X_test[:, num_cols])
    elif variant == "gym-style-seed99":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_s = np.copy(X_train); X_train_s[:, num_cols] = scaler.transform(X_train)[:, num_cols]
        X_test_s = np.copy(X_test); X_test_s[:, num_cols] = scaler.transform(X_test)[:, num_cols]
    else:
        raise ValueError(variant)
    bart = BARTRegressor()
    bart.fit(X_train_s, y_train)
    causes = []
    for c in range(M):
        if bart.is_causal_HTE(X_test_s, c, additive=(c in num_cols)):
            causes.append(c)
    print(f"[{variant}] causes={[FEATS[c] for c in causes]} cause_idx={causes} time={time.time()-t0:.1f}s", flush=True)
    # significant-cluster counts per column (same HTE objects)
    for c in range(M):
        hte = bart.hte_per_treat[c]
        sig = [int(k) for k, v in hte.is_cls_causal.items() if v]
        print(f"  col {FEATS[c]}: sig_clusters={sig}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["microrts-seed1", "gym-style-seed99"])
    run(ap.parse_args().variant)
