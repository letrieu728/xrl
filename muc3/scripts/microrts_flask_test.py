"""Stage S5: Flask dashboard test via Flask test_client (no live server needed
to prove the app serves a page; a live `flask run` smoke test is separate).
- import app module (exercises ALL stratvis-fixed imports + map_path.txt)
- GET / -> expect 200 + index.html content
- POST /utt_to_uht (pure-numpy route, no env) -> expect 200
- POST /get_board_state -> documents the RLAgent() no-arg bug (expect 500)
NOTE: importing app.py triggers the gym_microrts_rl import chain including
the checkpoint load. Run with CWD=cxrl_microRTS so the relative
checkpoint path resolves, under xvfb-run (JVM render only if a route
runs the env; these routes do not, except get_board_state which fails
before touching the env).
"""
import sys
import numpy as np
np.int = int  # gym-microrts 0.3.2 compat shim
sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
# app.py lives INSIDE cxrl_microRTS/ and is imported as a top-level module
# (Flask's `flask run` from that directory does the same).
sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy/cxrl_microRTS")
import app as flask_app_module
print("app module imported OK; routes:", sorted(str(r) for r in flask_app_module.app.url_map.iter_rules()), flush=True)

client = flask_app_module.app.test_client()

r = client.get("/")
print("GET / ->", r.status_code, "len=", len(r.data), flush=True)
print("  title snippet:", r.data[:300], flush=True)

r2 = client.post("/utt_to_uht", json={"utt": [[2]*16]*16})
print("POST /utt_to_uht ->", r2.status_code, r2.get_json(), flush=True)

r3 = client.post("/get_board_state")
print("POST /get_board_state ->", r3.status_code, flush=True)
print("  body:", r3.data[:500], flush=True)
print("S5 DONE", flush=True)
