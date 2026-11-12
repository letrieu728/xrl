"""Stage S6b: collect MicroRTS (t, t+1) strategy-state trajectories with the
authors' own RLAgent.run_episodes(), on the shipped basesWorkers16x16 map,
PPO agent (S3 checkpoint) vs CoacAI. Saves .npz for infer_DBN."""
import sys, time
import numpy as np
np.int = int  # gym-microrts 0.3.2 compat shim
import torch

sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")
from cxrl_microRTS.lib.gym_microrts_rl.gym_microrts_rl import RLAgent

NEP = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/hatch/workspace/step3-scratch/microrts_traj.npz"
obs = torch.zeros(1, 16, 16, 27)
raw = np.zeros((5, 16, 16), dtype=int)
agent = RLAgent(obs_=obs, rawobs_=raw, mappath_="maps/16x16/basesWorkers16x16.xml")
print("RLAgent ready; collecting", NEP, "episodes...", flush=True)
t0 = time.time()
st, stp1, gs = agent.run_episodes(NEP)
el = time.time() - t0
print(f"run_episodes({NEP}) in {el:.1f}s: states_t={st.shape} states_tp1={stp1.shape}", flush=True)
print("gamesteps per episode (lengths):", [len(g) for g in gs], flush=True)
print("states_t head:\n", st[:5], flush=True)
print("unique values per column (t):", [sorted(set(st[:, c].tolist())) for c in range(st.shape[1])], flush=True)
np.savez(OUT, states_t=st, states_tp1=stp1)
print("saved", OUT, "; S6b DONE", flush=True)
import os; os._exit(0)
