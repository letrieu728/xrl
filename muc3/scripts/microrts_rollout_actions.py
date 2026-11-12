"""Group 2a: MicroRTS rollout with action recording (PPO vs CoacAI).

Faithful re-implementation of the authors' RLAgent.run_episodes() loop
(cxrl_microRTS/lib/gym_microrts_rl/gym_microrts_rl.py), with ONE addition:
at every env step we also record the PPO policy's chosen unit actions
(valid_actions, as computed by the author's own code) and summarise them
as an action-type histogram + mode label for the transition that the step
produces.

Why a re-implementation instead of calling run_episodes()? The author's
function returns only (states_t, states_tp1, gamesteps) and discards the
actions. The fidelity protocol of the paper (Sec. 5.1.1 / Table 1)
requires an action representation, which the MicroRTS code never records.
Every other line of the loop (render, get_action, Java action packing,
step, strategy-state-change detection, termination) is copied from
run_episodes() unchanged. The diff between this loop and run_episodes()
is only the recording lines, marked with [G2-REC].

Action label definition (documented, deterministic):
  valid_actions columns: [src_cell, action_type, ...]; action_type in
  {0 none, 1 move, 2 harvest, 3 return, 4 produce, 5 attack}
  (gym_microrts/types.py). For a step, hist = bincount of action_type
  over all valid actions; label = argmax(hist) (lowest type wins ties);
  if there are no valid actions, label = -1 and hist = zeros.
  A transition (s_t -> s_tp1) is emitted exactly when the strategy state
  changes after a step (author's rule); its label/hist are those of the
  step that produced s_tp1.

Outputs (one .npz per batch, absolute path via --out):
  states_t (N,13) int, states_tp1 (N,13) int  -- 12 counts + gamestep,
  action_label (N,) int, action_hist (N,6) int, n_valid (N,) int,
  ep_id (N,) int (global episode index), gamestep_t/tp1 are columns 12
  of states_t/states_tp1, ep_n_transitions (n_episodes,) int,
  ep_n_steps (n_episodes,) int (env steps executed per episode),
  ep_outcome (n_episodes,) int (1 if terminated by done flag, else 0).

Usage (run under Xvfb, DISPLAY set; the script chdirs to cxrl_microRTS
because the author's checkpoint path is CWD-relative):
  python -u microrts_rollout_actions.py --episodes 10 --start-ep 0 \
      --out /abs/path/batch00.npz
The process ends with os._exit(0): JVM shutdown hangs (step-3 finding);
all results are saved and flushed before that.
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

np.int = int  # gym-microrts 0.3.2 compat shim (step 3, runner only)
import torch

COPY = "/home/hatch/workspace/step3-scratch/causal-xrl-copy"
sys.path.insert(0, COPY)
# Author's checkpoint path is relative to CWD; resolve it as step 3 did.
os.chdir(os.path.join(COPY, "cxrl_microRTS"))

from cxrl_microRTS.lib.gym_microrts_rl.gym_microrts_rl import RLAgent, device  # noqa: E402
from cxrl_microRTS.lib.gym_microrts_rl.utils import rawobs_to_state  # noqa: E402

MAP = "maps/16x16/basesWorkers16x16.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--start-ep", type=int, default=0)
    ap.add_argument("--seed", type=int, default=5,
                    help="RNG seed for this batch. The author's module seeds "
                    "random/numpy/torch with 5 at import, so every fresh "
                    "process replays the SAME episode sequence (verified: "
                    "pilot eps 0-1 byte-identical to batch00 eps 2-3, and "
                    "step 3's 2-ep file == first 109 rows of its 8-ep file). "
                    "Distinct per-batch seeds are REQUIRED for independent "
                    "episodes; the seed is saved in the .npz.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Reseed AFTER the author's module import (which fixed seed=5).
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[rollout] episodes={args.episodes} start_ep={args.start_ep} out={args.out} "
          f"map={MAP} device={device} seed={args.seed}", flush=True)
    t00 = time.time()
    obs = torch.zeros(1, 16, 16, 27)
    raw = np.zeros((5, 16, 16), dtype=int)
    agent = RLAgent(obs_=obs, rawobs_=raw, mappath_=MAP)
    print(f"[rollout] RLAgent ready in {time.time()-t00:.1f}s", flush=True)

    all_t, all_tp1, all_lab, all_hist, all_nv, all_ep = [], [], [], [], [], []
    ep_n_trans, ep_n_steps, ep_done = [], [], []

    from jpype.types import JArray, JInt

    for local_ep in range(args.episodes):
        ep_id = args.start_ep + local_ep
        t_ep = time.time()
        next_obs, raw_obs_ext_t = agent.envs.reset()
        raw_obs_t = np.squeeze(raw_obs_ext_t)
        strategy_state_t = rawobs_to_state(raw_obs_t)
        next_obs = torch.Tensor(next_obs).to(device)
        torch.zeros(1).to(device)
        gamestep = 0
        state_ws_t = np.concatenate((strategy_state_t, np.array([gamestep])))
        n_steps = 0
        done_flag = 0
        ep_trans = 0
        while True:
            agent.envs.render()
            with torch.no_grad():
                # [G2-HARNESS] The author's Agent.get_action prints a
                # multi_categoricals line on EVERY step (~1000/episode).
                # Silence only that call's stdout; no author code changed,
                # no result affected.
                with contextlib.redirect_stdout(io.StringIO()):
                    action, logproba, _, invalid_action_mask = agent.agent.get_action(
                        next_obs, envs=agent.envs)
            real_action = torch.cat([torch.stack(
                [torch.arange(0, agent.mapsize, device=device)
                 for _ in range(agent.envs.num_envs)]).unsqueeze(2), action], 2)
            real_action = real_action.cpu().numpy()
            valid_actions = real_action[invalid_action_mask[:, :, 0].bool().cpu().numpy()]
            # [G2-REC] record the action mix of THIS step (before env.step).
            nv = int(valid_actions.shape[0])
            if nv > 0:
                hist = np.bincount(valid_actions[:, 1].astype(int), minlength=6).astype(int)
                label = int(np.argmax(hist))
            else:
                hist = np.zeros(6, dtype=int)
                label = -1
            valid_actions_counts = invalid_action_mask[:, :, 0].sum(1).long().cpu().numpy()
            java_valid_actions = []
            valid_action_idx = 0
            for env_idx, valid_action_count in enumerate(valid_actions_counts):
                java_valid_action = []
                for c in range(valid_action_count):
                    java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
                    valid_action_idx += 1
                java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
            java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)
            gamestep += 1
            n_steps += 1
            next_obs, rs, ds, infos, raw_obs_ext_tp1 = agent.envs.step(java_valid_actions)
            next_obs = torch.Tensor(next_obs).to(device)
            raw_obs_tp1 = np.squeeze(raw_obs_ext_tp1)
            strategy_state_tp1 = rawobs_to_state(raw_obs_tp1)
            if not np.array_equal(strategy_state_t, strategy_state_tp1):
                state_ws_tp1 = np.concatenate((strategy_state_tp1, np.array([gamestep])))
                all_t.append(state_ws_t.copy())
                all_tp1.append(state_ws_tp1.copy())
                all_lab.append(label)
                all_hist.append(hist)
                all_nv.append(nv)
                all_ep.append(ep_id)
                ep_trans += 1
                strategy_state_t = strategy_state_tp1
                state_ws_t = np.concatenate((strategy_state_t, np.array([gamestep])))
            if ds[0] >= True:
                done_flag = 1
                break
        ep_n_trans.append(ep_trans)
        ep_n_steps.append(n_steps)
        ep_done.append(done_flag)
        print(f"[rollout] ep {ep_id}: {ep_trans} transitions, {n_steps} env steps, "
              f"done={done_flag}, {time.time()-t_ep:.1f}s", flush=True)

    states_t = np.array(all_t, dtype=int)
    states_tp1 = np.array(all_tp1, dtype=int)
    np.savez(args.out,
             states_t=states_t, states_tp1=states_tp1,
             action_label=np.array(all_lab, dtype=int),
             action_hist=np.array(all_hist, dtype=int),
             n_valid=np.array(all_nv, dtype=int),
             ep_id=np.array(all_ep, dtype=int),
             seed=np.array([args.seed], dtype=int),
             ep_n_transitions=np.array(ep_n_trans, dtype=int),
             ep_n_steps=np.array(ep_n_steps, dtype=int),
             ep_done=np.array(ep_done, dtype=int))
    labs, counts = np.unique(all_lab, return_counts=True)
    print(f"[rollout] SAVED {args.out}: {states_t.shape[0]} transitions from "
          f"{args.episodes} episodes in {time.time()-t00:.1f}s", flush=True)
    print(f"[rollout] action_label histogram: {dict(zip(labs.tolist(), counts.tolist()))}", flush=True)
    print(f"[rollout] n_valid per transition: min={min(all_nv)} max={max(all_nv)} "
          f"mean={float(np.mean(all_nv)):.2f}", flush=True)
    print("[rollout] DONE", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
