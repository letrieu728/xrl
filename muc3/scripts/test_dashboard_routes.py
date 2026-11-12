"""Group 2b: dashboard route verification via Flask test_client.

Run from cxrl_microRTS/ as CWD (app.py loads data/ relative to CWD),
under Xvfb (DISPLAY set; /get_board_state creates a real env).
Records, for each route, the HTTP status + a content summary or the
exact error. Prints a summary table at the end and writes it as JSON.
A route is only reported 200 if test_client actually returned 200.

Order matters: /compute_global_causal must run before /get_local_causal
(the latter reads globals set by the former -- the authors' design).
"""
import json
import os
import sys
import traceback

import numpy as np

np.int = int
COPY = "/home/hatch/workspace/step3-scratch/causal-xrl-copy"
os.chdir(os.path.join(COPY, "cxrl_microRTS"))
sys.path.insert(0, COPY)
sys.path.insert(0, os.path.join(COPY, "cxrl_microRTS"))

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/hatch/workspace/step3-scratch/group2/dashboard_routes.json"


def main():
    import app as m
    m.app.config["TESTING"] = False  # keep 500s as responses, not raises
    client = m.app.test_client()
    results = {}

    def record(name, resp, extra=None):
        rec = {"status": int(resp.status_code)}
        if extra:
            rec.update(extra)
        if resp.status_code != 200:
            body = resp.data[:500].decode("utf-8", "replace")
            rec["body_head"] = body
        results[name] = rec
        print(f"{name}: {rec}", flush=True)

    # 1. index
    r = client.get("/")
    record("GET /", r, {"n_bytes": len(r.data)} if r.status_code == 200 else None)
    # 2. pure compute routes (step-3 baseline)
    r = client.post("/utt_to_uht", json={"utt": [[2] * 16] * 16})
    record("POST /utt_to_uht", r)
    xml = open(os.path.join(COPY, "cxrl_microRTS", "maps", "case1.xml")).read()
    r = client.post("/read_xml_map", json={"content": xml})
    record("POST /read_xml_map", r)
    # 3. fixed route: board state
    try:
        r = client.get("/get_board_state")
        extra = None
        if r.status_code == 200:
            j = r.get_json()
            utt = j["utt"]
            n_units = sum(len(v) for v in utt.values())
            extra = {"utt_keys": sorted(utt.keys()), "n_unit_cells": n_units,
                     "uot_shape": [len(j["uot"]), len(j["uot"][0])],
                     "uht_nonzero": int(np.count_nonzero(np.array(j["uht"])))}
        record("GET /get_board_state", r, extra)
    except Exception as e:
        results["GET /get_board_state"] = {"status": "EXCEPTION",
                                           "error": f"{type(e).__name__}: {e}",
                                           "traceback": traceback.format_exc()[-1500:]}
        print("GET /get_board_state EXCEPTION:", e, flush=True)
    # 4. fixed route: global causal (needs data/)
    try:
        r = client.get("/compute_global_causal")
        extra = None
        if r.status_code == 200:
            extra = {"json": r.get_json()}
        record("GET /compute_global_causal", r, extra)
    except Exception as e:
        results["GET /compute_global_causal"] = {"status": "EXCEPTION",
                                                 "error": f"{type(e).__name__}: {e}",
                                                 "traceback": traceback.format_exc()[-1500:]}
        print("GET /compute_global_causal EXCEPTION:", e, flush=True)
    # 5. local causal (depends on 4)
    for step_id in (0, 10, 100):
        try:
            r = client.post("/get_local_causal", json={"step_id": step_id})
            extra = None
            if r.status_code == 200:
                j = r.get_json()
                extra = {"n_nodes": len(j.get("nodes", [])),
                         "n_edges": len(j.get("edges", [])),
                         "n_effects": len(j.get("effect", []))}
            record(f"POST /get_local_causal step_id={step_id}", r, extra)
        except Exception as e:
            results[f"POST /get_local_causal step_id={step_id}"] = {
                "status": "EXCEPTION", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1500:]}
            print(f"/get_local_causal step {step_id} EXCEPTION:", e, flush=True)
    # 6. episode playback (needs ep_play_front_data.npz)
    try:
        board = np.zeros((5, 16, 16), dtype=int).tolist()
        r = client.post("/run_env_episode",
                        json={"board_state": board, "player_urt_mat": [[0] * 16] * 16})
        extra = None
        if r.status_code == 200:
            j = r.get_json()
            extra = {"n_rawobs": len(j.get("rawobs_log", [])),
                     "n_gamesteps": len(j.get("unique_gamesteps", []))}
        record("POST /run_env_episode", r, extra)
    except Exception as e:
        results["POST /run_env_episode"] = {"status": "EXCEPTION",
                                            "error": f"{type(e).__name__}: {e}",
                                            "traceback": traceback.format_exc()[-1500:]}
        print("POST /run_env_episode EXCEPTION:", e, flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("DASHBOARD ROUTE CHECK DONE ->", OUT, flush=True)
    print("SUMMARY:", json.dumps({k: v.get("status") for k, v in results.items()},
                                 indent=2), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
