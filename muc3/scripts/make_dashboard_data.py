"""Group 2b: build the dashboard's data/ directory from Group-2a artifacts.

The authors' app.py loads, relative to the process CWD:
  data/strategy_state_epwt.npz   (key 'strategy_states', float64)
  ./data/scm500_hte.pickle       (dict {col_idx: BARTRegressor|None})
  ./data/scaler500_hte.pickle    (dict {col_idx: StandardScaler})
None of these exist in the author repo (step-3 finding). This script
copies the artifacts produced by merge_graph_targets.py into
cxrl_microRTS/data/ under EXACTLY those filenames (app.py hard-codes
them), then verifies each file loads and has the expected structure.

NAMING CAVEAT (read before citing): the "500" in scm500_hte/scaler500_hte
is the AUTHORS' filename, kept because app.py requires it. It does NOT
mean our SCM was learned from 500 episodes -- our episode count is in
the merge manifest and the experiment report. Do not cite the filename
as evidence of scale.

Optional: --with-ep-play runs ONE real episode via the authors'
RLAgent.run_an_episode() and writes data/ep_play_front_data.npz
(keys unique_rawobs_log, unique_gamesteps, player_urt_log), which
/run_env_episode loads. Requires Xvfb (DISPLAY set) + JVM.
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys

# Unpickling the SCM needs the BARTRegressor class importable.
sys.path.insert(0, "/home/hatch/workspace/step3-scratch/causal-xrl-copy")

import numpy as np

COPY = "/home/hatch/workspace/step3-scratch/causal-xrl-copy"
DATA_DIR = os.path.join(COPY, "cxrl_microRTS", "data")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True,
                    help="dir with scm_hte.pkl, scaler_hte.pkl, strategy_states.npz")
    ap.add_argument("--with-ep-play", action="store_true")
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = {}
    pairs = [("strategy_states.npz", "strategy_state_epwt.npz"),
             ("scm_hte.pkl", "scm500_hte.pickle"),
             ("scaler_hte.pkl", "scaler500_hte.pickle")]
    for src_name, dst_name in pairs:
        src = os.path.join(args.artifacts, src_name)
        dst = os.path.join(DATA_DIR, dst_name)
        shutil.copyfile(src, dst)
        manifest[dst_name] = {"bytes": os.path.getsize(dst), "sha256": sha256(dst)}
        print(f"copied {src_name} -> data/{dst_name} "
              f"({manifest[dst_name]['bytes']} bytes)", flush=True)
    # Verify: npz key + shapes; pickle dicts + key sets.
    z = np.load(os.path.join(DATA_DIR, "strategy_state_epwt.npz"))
    assert "strategy_states" in z.files, f"npz keys: {z.files}"
    ss = z["strategy_states"]
    print(f"strategy_states: shape={ss.shape} dtype={ss.dtype}", flush=True)
    manifest["strategy_states_shape"] = [int(x) for x in ss.shape]
    with open(os.path.join(DATA_DIR, "scm500_hte.pickle"), "rb") as f:
        scm = pickle.load(f)
    with open(os.path.join(DATA_DIR, "scaler500_hte.pickle"), "rb") as f:
        scalers = pickle.load(f)
    assert isinstance(scm, dict) and isinstance(scalers, dict)
    print(f"scm keys={sorted(scm.keys())} fitted="
          f"{sum(1 for v in scm.values() if v is not None)}; "
          f"scaler keys={sorted(scalers.keys())}", flush=True)
    manifest["scm_keys"] = [int(k) for k in sorted(scm.keys())]
    manifest["scaler_keys"] = [int(k) for k in sorted(scalers.keys())]
    manifest["scm_fitted"] = int(sum(1 for v in scm.values() if v is not None))
    if args.with_ep_play:
        print("running one real episode for ep_play_front_data.npz ...", flush=True)
        ep_path = run_ep_play()
        manifest["ep_play_front_data.npz"] = {"bytes": os.path.getsize(ep_path),
                                              "sha256": sha256(ep_path)}
    with open(os.path.join(DATA_DIR, "MANIFEST_group2.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("data/ READY:", json.dumps(manifest, indent=2), flush=True)


def run_ep_play():
    """One real episode via RLAgent.run_an_episode; save frontend npz.

    Follows app.py::runenv_steps exactly: case1.xml -> xml_to_obs ->
    obs_to_xml into MAP_PATH/custom.xml -> RLAgent on custom.xml ->
    run_an_episode(player_urt_mat from the same xml).
    """
    np.int = int
    import torch
    sys.path.insert(0, COPY)
    os.chdir(os.path.join(COPY, "cxrl_microRTS"))
    from cxrl_microRTS.lib.gym_microrts_rl.gym_microrts_rl import RLAgent, MAP_PATH
    from cxrl_microRTS.lib.gym_microrts_rl.obs_gen_xml import obs_to_xml
    from cxrl_microRTS.lib.gym_microrts_rl.utils import get_players_rsc
    from cxrl_microRTS.lib.gym_microrts_rl.xml_to_obs import xml_to_obs
    xml = open(os.path.join(COPY, "cxrl_microRTS", "maps", "case1.xml")).read()
    raw_obs, player_urt_mat = xml_to_obs(xml)
    board_state = raw_obs
    urt = np.array(board_state[1]); uot = np.array(board_state[2])
    utt = np.array(board_state[3]); uat = np.array(board_state[4])
    players_rsc = get_players_rsc(player_urt_mat, utt, uot)
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
    obs_to_xml(raw_obs, players_rsc, MAP_PATH + "/custom.xml")
    agent = RLAgent(obs_=torch.Tensor(my_obs), rawobs_=raw_obs,
                    mappath_="maps/16x16/custom.xml")
    logs, gamesteps, urt_log = agent.run_an_episode(player_urt_mat, False)
    out = os.path.join(DATA_DIR, "ep_play_front_data.npz")
    np.savez(out, unique_rawobs_log=np.asarray(logs),
             unique_gamesteps=np.asarray(gamesteps),
             player_urt_log=np.asarray(urt_log))
    print(f"ep_play: rawobs_log={np.asarray(logs).shape} "
          f"gamesteps={len(gamesteps)} urt_log={np.asarray(urt_log).shape}", flush=True)
    return out


if __name__ == "__main__":
    main()
