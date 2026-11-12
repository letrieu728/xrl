"""Stage S5b: more dashboard routes (test_client)."""
import sys
import numpy as np
np.int = int
sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy/cxrl_microRTS")
import app as m
client = m.app.test_client()

r = client.get("/get_board_state")
print("GET /get_board_state ->", r.status_code, flush=True)
print("  body:", r.data[:400], flush=True)

xml = open("/home/hatch/workspace/step3-scratch/causal-xrl-copy/cxrl_microRTS/maps/case1.xml").read()
r = client.post("/read_xml_map", json={"content": xml})
print("POST /read_xml_map ->", r.status_code, flush=True)
if r.status_code == 200:
    j = r.get_json()
    print("  raw_obs shape:", len(j["raw_obs"]), "x", len(j["raw_obs"][0]),
          "| player_urt_mat:", len(j["player_urt_mat"]), "x", len(j["player_urt_mat"][0]), flush=True)
else:
    print("  body:", r.data[:400], flush=True)

r = client.get("/compute_global_causal")
print("GET /compute_global_causal ->", r.status_code, flush=True)
print("  body:", r.data[:400], flush=True)
print("S5b DONE", flush=True)
import os; os._exit(0)
