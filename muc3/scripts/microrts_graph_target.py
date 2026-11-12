"""Group 2a: ONE target of the MicroRTS global graph (separate process).

Calls the authors' infer_causes_hte (cxrl_microRTS/lib/xrl_causal) for a
single target, exactly as infer_DBN would, and checkpoints the result so
one target's failure can never lose the others. Run once per target by
run_graph_targets.sh; merge_graph_targets.py combines the checkpoints.

Targets:
  * 13 state targets: X = states_t (13 cols), y = states_tp1 - states_t
    (state_integration=True, the dashboard/scm500_hte convention).
  * 1 action target ("action"): pairs of consecutive transitions inside
    each episode, both with n_valid>0. X = [states_t(i), label(i)]
    (14 cols), y = label(i+1). state_integration=False (the action level
    itself, as in the eval notebooks' replay construction).

Constant targets (fewer than 2 unique y values) follow infer_DBN's rule:
causes=[], no model, no scaler -- recorded as status "constant", never
as an error and never silently filled.

Outputs in --outdir:
  target_<name>.json   {target, status, causes, n_rows, nonzero, time_s, error?}
  bart_<name>.pkl      fitted BARTRegressor (only if fitted)
  scaler_<name>.pkl    fitted StandardScaler (only if fitted)
  num_cols_<name>.json {num_cols: [...]} -- columns with >2 uniques in X,
                       i.e. the columns the scaler was fit on.
"""
import argparse
import gc
import json
import pickle
import sys
import time
import traceback

import numpy as np
import sklearn.tree._tree as t

if not hasattr(t, "DTYPE"):
    t.DTYPE = np.float32

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
from cxrl_microRTS.lib.xrl_causal.xrl_causal import infer_causes_hte  # noqa: E402
from cxrl_microRTS.lib.utils.utils import convert_and_expand  # noqa: E402

FEATURES = ["base1", "barrack1", "worker1", "light1", "heavy1", "ranged1",
            "base2", "barrack2", "worker2", "light2", "heavy2", "ranged2", "timestep"]


def build_action_pairs(d):
    """Consecutive-transition pairs within each episode (both acted)."""
    st, lab, nv, ep = d["states_t"], d["action_label"], d["n_valid"], d["ep_id"]
    X_rows, y_rows = [], []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if nv[a] > 0 and nv[b] > 0:
                X_rows.append(np.concatenate([st[a].astype(float), [float(lab[a])]]))
                y_rows.append(float(lab[b]))
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    help="one of the 13 state names, or 'action'")
    ap.add_argument("--merged", required=True, help="merged .npz from merge_batches")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    d = np.load(args.merged)
    t0 = time.time()
    rec = {"target": args.target}
    try:
        if args.target == "action":
            X, y = build_action_pairs(d)
            names = FEATURES + ["action"]
            rec["n_rows"] = int(X.shape[0])
            rec["n_pairs_note"] = ("consecutive transitions within episode, "
                                   "both with n_valid>0")
        else:
            col = FEATURES.index(args.target)
            X = convert_and_expand(d["states_t"]).astype(float)
            Xn = convert_and_expand(d["states_tp1"]).astype(float)
            y = Xn[:, col] - X[:, col]
            names = FEATURES
            rec["n_rows"] = int(X.shape[0])
        rec["nonzero"] = int((y != 0).sum())
        rec["y_unique"] = sorted(np.unique(y).tolist())
        num_cols = [c for c in range(X.shape[1]) if len(np.unique(X[:, c])) > 2]
        with open(os.path.join(args.outdir, f"num_cols_{args.target}.json"), "w") as f:
            json.dump({"num_cols": num_cols, "columns": names}, f, indent=2)
        rec["num_cols"] = num_cols
        if len(np.unique(y)) < 2:
            rec.update({"status": "constant", "causes": [], "time_s": time.time() - t0})
            print(f"[{args.target}] CONSTANT -> causes=[]", flush=True)
        else:
            causes, bart, scaler = infer_causes_hte(X, y)
            cnames = [names[c] for c in causes]
            with open(os.path.join(args.outdir, f"bart_{args.target}.pkl"), "wb") as f:
                pickle.dump(bart, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(args.outdir, f"scaler_{args.target}.pkl"), "wb") as f:
                pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
            rec.update({"status": "ok", "causes": cnames,
                        "cause_idx": [int(c) for c in causes],
                        "time_s": time.time() - t0})
            print(f"[{args.target}] n={rec['n_rows']} nonzero={rec['nonzero']} "
                  f"causes={cnames} ({rec['time_s']:.1f}s)", flush=True)
            del bart
            gc.collect()
    except Exception as e:
        rec.update({"status": "error", "time_s": time.time() - t0,
                    "error": f"{type(e).__name__}: {str(e)[:500]}",
                    "traceback": traceback.format_exc()[-2000:]})
        print(f"[{args.target}] ERROR: {rec['error']}", flush=True)
    with open(os.path.join(args.outdir, f"target_{args.target}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[{args.target}] checkpoint written; status={rec['status']}", flush=True)


if __name__ == "__main__":
    main()
