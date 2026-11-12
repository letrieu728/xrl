"""Group 2a: MicroRTS behavior-prediction fidelity, eval-notebook style.

Mirrors the authors' eval_fidelity_Cart Pole.ipynb protocol, adapted to
MicroRTS where the paper gives no action encoding (see protocol notes in
the output JSON and in muc3/microRTS-thu-nghiem.md):

  * Action representation: mode of PPO unit action types at the step
    (0 none, 1 move, 2 harvest, 3 return, 4 produce, 5 attack).
  * Construction A (primary, notebook structure): within each episode,
    consecutive acted transitions (i -> i+1); X = [state_t(i), label(i)]
    (14 cols), y = label(i+1). Construction B (secondary, same-row):
    X = state_t (13 cols), y = label(i), acted transitions only.
  * Valid rows: local treatments of the ACTION-target BART flag >1
    non-action cause for this row (notebook: >1 incoming local edge
    into the action node, excluding the action itself). Queries to
    local_treatments/ITE are SCALED with the action scaler on
    num_cols_action: infer_causes_hte fits BART on scaled data and the
    HTE trees live in scaled space (the dashboard does the same).
    Raw queries were tried first (fidelity_v1_rawquery.json) and are
    WRONG for this BART wrapper; scaled screening is the definitive
    protocol-faithful run. --valid-mode nofilter keeps ALL candidate
    rows (deviation, labelled in the output) when the notebook filter
    yields too few rows.
  * Split 60/40, random_state=10. LM = LogisticRegression(random_state=1),
    DT = DecisionTreeClassifier(random_state=1), both on all features.
  * Ours: per global cause of the action, one BART fit on
    [cause + non-cause covariates] (train split); per test row, weights
    are |mean ITE| from the action-target SCM; prediction = weighted
    average rounded to the nearest action code (notebook's 0.5 threshold
    is the binary special case). Queries are passed SCALED to
    local_treatments/ITE (see above). If total weight is 0, fall back to the
    unweighted mean of the available per-cause predictions (counted).
  * Metrics: accuracy + F1 weighted (paper Sec. 5.2) + macro F1 (extra)
    + majority-class baseline accuracy (extra, for context).

Inputs: --merged npz, --ckptdir (target_action.json, bart_action.pkl,
num_cols_action.json). Output: fidelity JSON (+ printed table).
"""
import argparse
import gc
import json
import pickle
import sys
import time

import numpy as np
import sklearn.tree._tree as t

if not hasattr(t, "DTYPE"):
    t.DTYPE = np.float32

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
import rpy2.robjects.numpy2ri

# Unpickled BARTRegressor models need numpy<->R conversion for ITE;
# __init__ activates it, but unpickling does not construct the class.
rpy2.robjects.numpy2ri.activate()
from cxrl_microRTS.lib.models.bart_regressor import BARTRegressor  # noqa: E402

FEATURES = ["base1", "barrack1", "worker1", "light1", "heavy1", "ranged1",
            "base2", "barrack2", "worker2", "light2", "heavy2", "ranged2", "timestep"]


def screen_valid(bart_action, queries14):
    """Valid rows: >1 flagged non-action treatment for the action outcome."""
    valid = []
    t0 = time.time()
    for i in range(queries14.shape[0]):
        local = np.asarray(bart_action.local_treatments(queries14[i:i + 1]))[0]
        flagged = [int(c) for c in np.nonzero(local)[0] if int(c) != 13]
        if len(flagged) > 1:
            valid.append(i)
    return valid, time.time() - t0


def run_construction(name, X, y, queries14, bart_action, num_cols_action,
                     global_causes14, valid_mode="notebook"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    out = {"construction": name, "n_candidate": int(X.shape[0]),
           "valid_mode": valid_mode}
    valid, t_screen = screen_valid(bart_action, queries14)
    out["screen_time_s"] = round(t_screen, 1)
    out["n_valid_notebook_filter"] = int(len(valid))
    out["valid_fraction_notebook_filter"] = float(len(valid) / max(1, X.shape[0]))
    print(f"[{name}] candidates={X.shape[0]} notebook-valid={len(valid)} "
          f"({out['valid_fraction_notebook_filter']:.3f}) screen={t_screen:.1f}s",
          flush=True)
    if valid_mode == "nofilter":
        valid = list(range(X.shape[0]))
        print(f"[{name}] nofilter mode: using ALL {len(valid)} candidate rows",
              flush=True)
    out["n_valid"] = int(len(valid))
    out["valid_fraction"] = float(len(valid) / max(1, X.shape[0]))
    if len(valid) < 20:
        out["error"] = "too few valid rows for a 60/40 split"
        print(f"[{name}] ABORT: {out['error']}", flush=True)
        return out
    Xv, yv = X[valid], y[valid]
    qv = queries14[valid]
    labels, counts = np.unique(yv, return_counts=True)
    out["valid_label_dist"] = {int(k): int(v) for k, v in zip(labels, counts)}
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        Xv, yv, np.arange(len(valid)), test_size=0.4, random_state=10)
    out["n_train"] = int(len(y_train))
    out["n_test"] = int(len(y_test))
    test_labels, test_counts = np.unique(y_test, return_counts=True)
    out["test_label_dist"] = {int(k): int(v) for k, v in zip(test_labels, test_counts)}
    maj = float(np.max(test_counts) / len(y_test))
    out["majority_baseline_acc"] = maj

    # --- LM / DT on all features ---
    t0 = time.time()
    dt = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)
    p_dt = dt.predict(X_test)
    lr = LogisticRegression(random_state=1, max_iter=1000).fit(X_train, y_train)
    p_lr = lr.predict(X_test)
    out["dt"] = {"acc": float(accuracy_score(y_test, p_dt)),
                 "f1_weighted": float(f1_score(y_test, p_dt, average="weighted", zero_division=0)),
                 "f1_macro": float(f1_score(y_test, p_dt, average="macro", zero_division=0))}
    out["lm"] = {"acc": float(accuracy_score(y_test, p_lr)),
                 "f1_weighted": float(f1_score(y_test, p_lr, average="weighted", zero_division=0)),
                 "f1_macro": float(f1_score(y_test, p_lr, average="macro", zero_division=0))}
    print(f"[{name}] DT acc={out['dt']['acc']:.4f} F1w={out['dt']['f1_weighted']:.4f} | "
          f"LM acc={out['lm']['acc']:.4f} F1w={out['lm']['f1_weighted']:.4f} | "
          f"majority={maj:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # --- Ours: BART-weighted ---
    n_feat = X.shape[1]
    causes = [c for c in global_causes14 if c < n_feat]
    out["global_causes_used"] = [int(c) for c in causes]
    out["global_causes_skipped_no_feature"] = [int(c) for c in global_causes14 if c >= n_feat]
    covars = [c for c in range(n_feat) if c not in causes]
    hash_pred = {}
    fit_errors = {}
    t0 = time.time()
    for c in causes:
        cols = [c] + covars
        try:
            bart = BARTRegressor()
            bart.fit(X_train[:, cols], y_train.astype(float))
            hash_pred[c] = np.asarray(
                bart.predict(X_test[:, cols]), dtype=float).ravel()
            del bart
            gc.collect()
        except Exception as e:
            fit_errors[str(c)] = f"{type(e).__name__}: {str(e)[:200]}"
    out["bart_fit_errors"] = fit_errors
    out["bart_fit_time_s"] = round(time.time() - t0, 1)
    print(f"[{name}] per-cause BART fits: {len(hash_pred)} ok, "
          f"{len(fit_errors)} failed, {out['bart_fit_time_s']}s", flush=True)
    if not hash_pred:
        out["bart"] = {"error": "no per-cause BART fit succeeded"}
        return out
    # ITE-weighted aggregation over ALL test rows (no cap; timed).
    y_pred = np.zeros(len(y_test))
    n_fallback = 0
    t0 = time.time()
    test_pos = {int(s): j for j, s in enumerate(idx_test)}
    for j, step in enumerate(idx_test):
        query = qv[step:step + 1]
        local = np.asarray(bart_action.local_treatments(query))[0]
        total_w, y_acc = 0.0, 0.0
        for treat in np.nonzero(local)[0]:
            treat = int(treat)
            if treat not in hash_pred:
                continue
            ite = bart_action.ITE(query, treat, additive=(treat in num_cols_action))
            w = float(abs(np.asarray(ite, dtype=float).mean()))
            total_w += w
            y_acc += hash_pred[treat][j] * w
        if total_w > 0:
            y_pred[j] = y_acc / total_w
        else:
            n_fallback += 1
            y_pred[j] = float(np.mean([hash_pred[c][j] for c in hash_pred]))
    out["ite_loop_time_s"] = round(time.time() - t0, 1)
    out["n_zero_weight_fallback"] = int(n_fallback)
    lo, hi = float(np.min(y_train)), float(np.max(y_train))
    y_pred_cls = np.clip(np.rint(y_pred), lo, hi).astype(int)
    out["bart"] = {"acc": float(accuracy_score(y_test, y_pred_cls)),
                   "f1_weighted": float(f1_score(y_test, y_pred_cls, average="weighted", zero_division=0)),
                   "f1_macro": float(f1_score(y_test, y_pred_cls, average="macro", zero_division=0)),
                   "n_test": int(len(y_test))}
    print(f"[{name}] BART acc={out['bart']['acc']:.4f} F1w={out['bart']['f1_weighted']:.4f} "
          f"fallback={n_fallback} ITEloop={out['ite_loop_time_s']}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--ckptdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid-mode", choices=["notebook", "nofilter"],
                    default="notebook")
    args = ap.parse_args()
    import os
    d = np.load(args.merged)
    with open(os.path.join(args.ckptdir, "target_action.json")) as f:
        action_rec = json.load(f)
    with open(os.path.join(args.ckptdir, "num_cols_action.json")) as f:
        num_cols_action = json.load(f)["num_cols"]
    with open(os.path.join(args.ckptdir, "bart_action.pkl"), "rb") as f:
        bart_action = pickle.load(f)
    with open(os.path.join(args.ckptdir, "scaler_action.pkl"), "rb") as f:
        scaler_action = pickle.load(f)
    global_causes14 = action_rec["cause_idx"]
    print(f"action global causes: {[ (FEATURES+['action'])[c] for c in global_causes14 ]} "
          f"status={action_rec['status']}", flush=True)

    st = d["states_t"].astype(float)
    lab = d["action_label"].astype(float)
    nv = d["n_valid"]
    ep = d["ep_id"]
    # Construction A pairs
    XA, yA, qA = [], [], []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if nv[a] > 0 and nv[b] > 0:
                XA.append(np.concatenate([st[a], [lab[a]]]))
                yA.append(lab[b])
                qA.append(np.concatenate([st[a], [lab[a]]]))
    XA = np.array(XA); yA = np.array(yA); qA = np.array(qA)
    # Construction B rows
    keep = np.where(nv > 0)[0]
    XB = st[keep]
    yB = lab[keep]
    qB = np.concatenate([st[keep], lab[keep][:, None]], axis=1)
    print(f"A pairs={XA.shape[0]}  B rows={XB.shape[0]}  total transitions={st.shape[0]}",
          flush=True)

    def scale_queries(q):
        qs = q.copy()
        qs[:, num_cols_action] = scaler_action.transform(q[:, num_cols_action])
        return qs

    resA = run_construction("A_next_action_notebook_style", XA, yA, scale_queries(qA),
                            bart_action, num_cols_action, global_causes14,
                            valid_mode=args.valid_mode)
    resB = run_construction("B_same_row_state_to_action", XB, yB, scale_queries(qB),
                            bart_action, num_cols_action, global_causes14,
                            valid_mode=args.valid_mode)
    out = {
        "valid_mode": args.valid_mode,
        "queries_scaled_with": "scaler_action.pkl on num_cols_action",
        "n_transitions": int(st.shape[0]),
        "episodes": int(len(np.unique(ep))),
        "n_acted_transitions": int(len(keep)),
        "action_label_dist_all": {int(k): int(v) for k, v in
                                  zip(*np.unique(d["action_label"], return_counts=True))},
        "action_target_status": action_rec["status"],
        "global_causes_action": [(FEATURES + ["action"])[c] for c in global_causes14],
        "num_cols_action": num_cols_action,
        "construction_A": resA,
        "construction_B": resB,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("FIDELITY DONE ->", args.out, flush=True)


if __name__ == "__main__":
    main()
