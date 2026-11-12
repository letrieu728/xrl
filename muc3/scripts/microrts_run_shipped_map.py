"""Stage S4c: real MicroRTS env run with the authors' RLAgent + S3 checkpoint.
Map: shipped basesWorkers16x16.xml (no custom map yet).
Steps: construct RLAgent -> manual_get_valid_action (reset + 1 policy query)
-> run_num_steps(3 unique states).
"""
import sys, time
import numpy as np
# COMPAT SHIM (test script only, no dependency file edited): gym-microrts 0.3.2
# uses `np.int`, removed in NumPy 1.24. numpy.org guidance: `int` is the
# exact replacement. 3 occurrences in gym_microrts (vec_env/global/local),
# all as dtype=np.int.
np.int = int
import torch

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
from cxrl_microRTS.lib.gym_microrts_rl.gym_microrts_rl import RLAgent

t0 = time.time()
obs = torch.zeros(1, 16, 16, 27)
raw = np.zeros((5, 16, 16), dtype=int)
print("constructing RLAgent (JVM start + JNIVecClient)...", flush=True)
agent = RLAgent(obs_=obs, rawobs_=raw, mappath_="maps/16x16/basesWorkers16x16.xml")
print(f"RLAgent constructed in {time.time()-t0:.1f}s; envs={type(agent.envs).__name__}", flush=True)
print("action_space nvec:", agent.envs.action_space.nvec if hasattr(agent.envs, "action_space") else "(via wrapper)", flush=True)

t0 = time.time()
valid = agent.manual_get_valid_action()
print(f"manual_get_valid_action in {time.time()-t0:.1f}s: shape={valid.shape}", flush=True)
print("first 3 valid actions:\n", valid[:3], flush=True)

t0 = time.time()
urt = np.zeros((16, 16), dtype=int)
logs, steps, urt_log = agent.run_num_steps(3, urt, False)
print(f"run_num_steps(3) in {time.time()-t0:.1f}s: rawobs_log={logs.shape} gamesteps={steps.tolist()} urt_log={urt_log.shape}", flush=True)
print("S4c DONE", flush=True)
