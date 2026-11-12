"""Stage S4d: run the authors' RLAgent on the shipped case1.xml custom map,
following app.py runenv_steps conversion exactly:
  xml -> xml_to_obs -> board_state -> my_obs (1,16,16,27) + players_rsc
  -> obs_to_xml to MAP_PATH/custom.xml -> RLAgent(mappath='maps/16x16/custom.xml')
  -> run_num_steps(3 unique states)
Also saves the rawobs log + strategy states for the xrl_causal attempt.
"""
import sys, time, json
import numpy as np
np.int = int  # gym-microrts 0.3.2 compat shim, see stage_s4c.py
import torch

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
from cxrl_microRTS.lib.gym_microrts_rl.xml_to_obs import xml_to_obs
from cxrl_microRTS.lib.gym_microrts_rl.obs_gen_xml import obs_to_xml
from cxrl_microRTS.lib.gym_microrts_rl.utils import get_players_rsc, rawobs_to_state
from cxrl_microRTS.lib.gym_microrts_rl.gym_microrts_rl import RLAgent, MAP_PATH

MAP = "case1"
xml = open(f"/home/hatch/workspace/step3-scratch/causal-xrl-copy/cxrl_microRTS/maps/{MAP}.xml").read()
raw_obs, player_urt_mat = xml_to_obs(xml)
board_state = raw_obs  # app.py uses np.array(board_state) with [1]=urt [2]=uot [3]=utt [4]=uat
urt = np.array(board_state[1]); uot = np.array(board_state[2])
utt = np.array(board_state[3]); uat = np.array(board_state[4])
players_rsc = get_players_rsc(player_urt_mat, utt, uot)
print("players_rsc:", players_rsc, "MAP_PATH:", MAP_PATH, flush=True)

my_obs = np.zeros((1, 16, 16, 27))
my_obs[0, :, :, 13:21] = np.eye(8)[utt]
my_obs[0, :, :, 10:13] = np.eye(3)[uot]
uh1t_obs = np.logical_or(np.logical_or(utt == 1, utt == 4), utt == 7).astype(int)
uh4t_obs = np.logical_and(utt != 0, uh1t_obs != 1).astype(int) * 4
uht_obs = uh1t_obs + uh4t_obs
my_obs[0, :, :, :5] = np.eye(5)[uht_obs]
urt_obs = np.clip(urt, 0, 4)
my_obs[0, :, :, 5:10] = np.eye(5)[urt_obs]
my_obs[0, :, :, 21:] = np.eye(6)[uat]

xml_map_path = MAP_PATH + "/custom.xml"
obs_to_xml(raw_obs, players_rsc, xml_map_path)
print("wrote", xml_map_path, flush=True)

t0 = time.time()
agent = RLAgent(obs_=torch.Tensor(my_obs), rawobs_=raw_obs, mappath_="maps/16x16/custom.xml")
print(f"RLAgent(custom.xml) constructed in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
logs, steps, urt_log = agent.run_num_steps(3, player_urt_mat, False)
print(f"run_num_steps(3) in {time.time()-t0:.1f}s: rawobs_log={logs.shape} gamesteps={steps.tolist()}", flush=True)
states = np.array([rawobs_to_state(r) for r in logs])
print("strategy states:\n", states, flush=True)
np.savez("/home/hatch/workspace/step3-scratch/case1_run.npz",
         rawobs_log=logs, gamesteps=steps, urt_log=urt_log, strategy_states=states)
print("saved case1_run.npz; S4d DONE", flush=True)
# NOTE: normal interpreter exit hangs in jpype JVM shutdown (non-daemon
# Java render/client threads; observed in S4c probe: stuck in
# jpype._core._JTerminate). Results are already saved+flushed, so force
# exit. This is a harness workaround, not a change to author code.
import os
os._exit(0)
