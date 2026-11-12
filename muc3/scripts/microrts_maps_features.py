"""Stage S1+S2: numpy/lxml-only MicroRTS modules on the fixed copy.
- xml_to_obs on the 5 shipped maps -> raw_obs (5,16,16) + player_urt_mat
- rawobs_to_state -> 12 strategy counts per map
- obs_gen_xml roundtrip: raw_obs -> XML file -> xml_to_obs again, compare
Run from the repo-copy root so `cxrl_microRTS.*` resolves.
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")

from cxrl_microRTS.lib.gym_microrts_rl.xml_to_obs import xml_to_obs
from cxrl_microRTS.lib.gym_microrts_rl.utils import rawobs_to_state, get_players_rsc, players_rsc_to_urtmat
from cxrl_microRTS.lib.gym_microrts_rl.obs_gen_xml import obs_to_xml
print("S1 imports OK: xml_to_obs, utils, obs_gen_xml")

FEATURES = ["base1", "barrack1", "worker1", "light1", "heavy1", "ranged1",
            "base2", "barrack2", "worker2", "light2", "heavy2", "ranged2"]
maps_dir = Path("/home/hatch/workspace/step3-scratch/causal-xrl-copy/cxrl_microRTS/maps")
rows = {}
for name in ["case1", "case2", "case3", "case4", "default_map"]:
    xml = (maps_dir / f"{name}.xml").read_text()
    raw_obs, player_urt_mat = xml_to_obs(xml)
    assert raw_obs.shape == (5, 16, 16), raw_obs.shape
    assert player_urt_mat.shape == (16, 16)
    state = rawobs_to_state(raw_obs)
    rows[name] = state.tolist()
    nunits = int((raw_obs[3] != 0).sum())
    print(f"{name}: raw_obs={raw_obs.shape} units={nunits} "
          f"p1_rsc={get_players_rsc(player_urt_mat, raw_obs[3], raw_obs[2])} "
          f"state={dict(zip(FEATURES, state.tolist()))}")

out = Path("/home/hatch/workspace/step3-scratch/map_strategy_states.json")
out.write_text(json.dumps({"features": FEATURES, "maps": rows}, indent=2))
print("wrote", out)

# S2 roundtrip on case1
xml = (maps_dir / "case1.xml").read_text()
raw_obs, player_urt_mat = xml_to_obs(xml)
players_rsc = get_players_rsc(player_urt_mat, raw_obs[3], raw_obs[2])
tmp = "/home/hatch/workspace/step3-scratch/roundtrip_case1.xml"
obs_to_xml(raw_obs, players_rsc, tmp)
raw2, urt2 = xml_to_obs(Path(tmp).read_text())
print("roundtrip raw_obs identical:", bool(np.array_equal(raw_obs, raw2)),
      "| player_urt_mat identical:", bool(np.array_equal(player_urt_mat, urt2)))
print("S1+S2 DONE")
