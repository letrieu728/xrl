"""G3a Step 3: full fidelity diagnosis (single script, JSON + printout).

Answers, with counts at every filter step:
  1. HTE internals of the saved action BART: per treatment column, how many
     HTE-tree clusters exist, which are significant, how many training
     (X_test_scaled) rows fall in significant clusters.
  2. Whether scaling choice changes the verdict: train-scaler (saved),
     test-scaler (reconstructed), raw.
  3. How well the action BART fits at all (predict on the pairs).
  4. Whether ANY state signal predicts the next action (cheap DT probe).
"""
import json, os, pickle, sys
import numpy as np
import sklearn.tree._tree as t
if not hasattr(t, "DTYPE"):
    t.DTYPE = np.float32
sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from cxrl_microRTS.lib.models.bart_regressor import BARTRegressor  # noqa: F401
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

G = "/home/hatch/workspace/step3-scratch/group2"
CK = os.path.join(G, "microrts100/ckpt")
FEATS = ["base1","barrack1","worker1","light1","heavy1","ranged1","base2","barrack2","worker2","light2","heavy2","ranged2","timestep","action"]
d = np.load(os.path.join(G, "microrts100/microrts_merged.npz"))
st = d["states_t"].astype(float); lab = d["action_label"].astype(float)
nv = d["n_valid"]; ep = d["ep_id"]

# ---- rebuild the action-pair design EXACTLY as microrts_graph_target.py ----
X_rows, y_rows = [], []
for e in np.unique(ep):
    idx = np.where(ep == e)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if nv[a] > 0 and nv[b] > 0:
            X_rows.append(np.concatenate([st[a], [lab[a]]]))
            y_rows.append(float(lab[b]))
X = np.array(X_rows); y = np.array(y_rows)
print(f"pairs: X={X.shape} y uniques={np.unique(y, return_counts=True)}")

with open(os.path.join(CK, "bart_action.pkl"), "rb") as f:
    bart = pickle.load(f)
with open(os.path.join(CK, "scaler_action.pkl"), "rb") as f:
    scaler_train = pickle.load(f)
with open(os.path.join(CK, "num_cols_action.json")) as f:
    num_cols = json.load(f)["num_cols"]
with open(os.path.join(CK, "target_action.json")) as f:
    action_rec = json.load(f)
print("global causes:", action_rec["causes"], "| num_cols:", num_cols)

# ---- 1. reconstruct the exact HTE training split (MicroRTS infer_causes_hte) ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=99)
scaler_test = preprocessing.StandardScaler().fit(X_test[:, num_cols])
X_test_scaled = np.copy(X_test); X_test_scaled[:, num_cols] = scaler_test.transform(X_test[:, num_cols])
X_train_scaled = np.copy(X_train); X_train_scaled[:, num_cols] = scaler_train.transform(X_train[:, num_cols])
print("scaler_train mean:", np.round(scaler_train.mean_, 4))
print("scaler_test  mean:", np.round(scaler_test.mean_, 4))
print("scaler_train scale:", np.round(scaler_train.scale_, 4))
print("scaler_test  scale:", np.round(scaler_test.scale_, 4))
print("max |mean diff|:", float(np.max(np.abs(scaler_train.mean_ - scaler_test.mean_))),
      "max |scale diff|:", float(np.max(np.abs(scaler_train.scale_ - scaler_test.scale_))))

hte_report = {}
for c in range(X.shape[1]):
    hte = bart.hte_per_treat[c]
    cids = hte.cluster(X_test_scaled)
    uniq, counts = np.unique(cids, return_counts=True)
    sig = [int(k) for k, v in hte.is_cls_causal.items() if v]
    sig_rows = int(sum(int(n) for cid, n in zip(uniq, counts) if int(cid) in sig))
    hte_report[FEATS[c]] = {"n_clusters": int(len(uniq)),
                            "significant_clusters": sig,
                            "cluster_sizes": {int(k): int(v) for k, v in zip(uniq, counts)},
                            "rows_in_significant_clusters": sig_rows,
                            "n_test_rows": int(X_test_scaled.shape[0])}
    print(f"HTE col {c} {FEATS[c]}: clusters={len(uniq)} sig={sig} sig_rows={sig_rows}/{X_test_scaled.shape[0]}")

# ---- 2. screening under three scalings ----
def build_queries():
    qA = []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if nv[a] > 0 and nv[b] > 0:
                qA.append(np.concatenate([st[a], [lab[a]]]))
    keep = np.where(nv > 0)[0]
    qB = np.concatenate([st[keep], lab[keep][:, None]], axis=1)
    return np.array(qA), qB
qA, qB = build_queries()

def screen(q, scaler, label):
    if scaler is None:
        qs = q
    else:
        qs = q.copy(); qs[:, num_cols] = scaler.transform(q[:, num_cols])
    dist = {}; flagcols = {}
    for i in range(qs.shape[0]):
        local = np.asarray(bart.local_treatments(qs[i:i+1]))[0]
        flagged = [int(c) for c in np.nonzero(local)[0] if int(c) != 13]
        dist[len(flagged)] = dist.get(len(flagged), 0) + 1
        for c in flagged:
            flagcols[FEATS[c]] = flagcols.get(FEATS[c], 0) + 1
    valid = sum(n for k, n in dist.items() if k > 1)
    print(f"[{label}] n={qs.shape[0]} flag-count dist={dict(sorted(dist.items()))} valid>1={valid} flagcols={flagcols}")
    return {"n": int(qs.shape[0]), "flag_count_dist": {str(k): int(v) for k, v in dist.items()},
            "valid_gt1": int(valid), "flag_cols": flagcols}

screens = {}
for name, q in [("A", qA), ("B", qB)]:
    screens[f"{name}_train_scaler"] = screen(q, scaler_train, f"{name} train-scaler")
    screens[f"{name}_test_scaler"] = screen(q, scaler_test, f"{name} test-scaler")
    screens[f"{name}_raw"] = screen(q, None, f"{name} raw")

# ---- 3. BART fit quality on the pairs (train-scaler space, as fitted) ----
qs_all = X.copy(); qs_all[:, num_cols] = scaler_train.transform(X[:, num_cols])
pred = np.asarray(bart.predict(qs_all), dtype=float).ravel()
ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
r2 = 1 - ss_res / ss_tot
corr = float(np.corrcoef(y, pred)[0, 1])
pred_cls = np.clip(np.rint(pred), 0, 5).astype(int)
acc = float(np.mean(pred_cls == y.astype(int)))
print(f"BART fit on pairs: R2={r2:.4f} corr={corr:.4f} rounded-acc={acc:.4f} (majority={float(np.max(np.unique(y, return_counts=True)[1])/len(y)):.4f})")
print(f"  y mean={y.mean():.3f} std={y.std():.3f}; pred mean={pred.mean():.3f} std={pred.std():.3f} range=({pred.min():.3f},{pred.max():.3f})")

# ---- 4. cheap signal probe: DT on construction A, raw, 60/40 seed 10 ----
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
Xtr, Xte, ytr, yte = train_test_split(X, y.astype(int), test_size=0.4, random_state=10)
dt = DecisionTreeClassifier(random_state=1).fit(Xtr, ytr)
pdt = dt.predict(Xte)
print(f"DT probe A (raw, all 14 cols): acc={accuracy_score(yte, pdt):.4f} majority={float(np.max(np.unique(yte, return_counts=True)[1])/len(yte)):.4f}")
imp = sorted(zip(FEATS, dt.feature_importances_), key=lambda z: -z[1])
print("  importances:", [(n, round(float(v), 3)) for n, v in imp])

out = {"pairs_n": int(X.shape[0]),
       "global_causes_action": action_rec["causes"],
       "num_cols_action": num_cols,
       "hte_per_column": hte_report,
       "scaler_train_mean": scaler_train.mean_.tolist(), "scaler_test_mean": scaler_test.mean_.tolist(),
       "scaler_train_scale": scaler_train.scale_.tolist(), "scaler_test_scale": scaler_test.scale_.tolist(),
       "screens": screens,
       "bart_fit": {"r2": r2, "corr": corr, "rounded_acc": acc},
       "dt_probe": {"acc": float(accuracy_score(yte, pdt)),
                    "importances": {n: float(v) for n, v in imp}}}
with open("/tmp/g3a_diag_full.json", "w") as f:
    json.dump(out, f, indent=2)
print("DIAG FULL DONE -> /tmp/g3a_diag_full.json")
