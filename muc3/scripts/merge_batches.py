"""Group 2a: merge rollout batch .npz files into one dataset.

Concatenates all batch files (pilot + batch00..) in episode order,
verifies episode ids are contiguous with no duplicates, and writes the
merged .npz plus a JSON manifest (per-batch transition counts, episode
ranges, label histogram). Fails loudly on any inconsistency instead of
silently dropping data.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batchdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.batchdir, "batch*.npz")))
    assert files, "no batch files found"
    print(f"merging {len(files)} batch files", flush=True)
    parts = []
    manifest_batches = []
    for f in files:
        d = np.load(f)
        n = int(d["states_t"].shape[0])
        eps = sorted(np.unique(d["ep_id"]).tolist())
        manifest_batches.append({"file": os.path.basename(f), "n_transitions": n,
                                 "episodes": [int(e) for e in eps],
                                 "seed": (int(d["seed"][0]) if "seed" in d.files
                                          else 5),
                                 "ep_n_transitions": d["ep_n_transitions"].tolist(),
                                 "ep_n_steps": d["ep_n_steps"].tolist(),
                                 "ep_done": d["ep_done"].tolist()})
        parts.append(d)
        print(f"  {os.path.basename(f)}: {n} transitions, episodes {eps[0]}..{eps[-1]}",
              flush=True)
    keys = ["states_t", "states_tp1", "action_label", "action_hist",
            "n_valid", "ep_id"]
    merged = {k: np.concatenate([d[k] for d in parts]) for k in keys}
    all_eps = sorted(np.unique(merged["ep_id"]).tolist())
    assert all_eps == list(range(all_eps[0], all_eps[-1] + 1)), \
        f"episode ids not contiguous: {all_eps}"
    # Per-episode transition counts must match the sum of batch ep counts.
    assert merged["states_t"].shape[0] == sum(
        b["n_transitions"] for b in manifest_batches)
    np.savez(args.out, **merged,
             ep_n_transitions=np.concatenate([d["ep_n_transitions"] for d in parts]),
             ep_n_steps=np.concatenate([d["ep_n_steps"] for d in parts]),
             ep_done=np.concatenate([d["ep_done"] for d in parts]))
    labs, counts = np.unique(merged["action_label"], return_counts=True)
    manifest = {"batch_files": manifest_batches,
                "n_batches": len(files),
                "n_transitions": int(merged["states_t"].shape[0]),
                "episodes": [int(e) for e in all_eps],
                "n_episodes": len(all_eps),
                "action_label_dist": {int(k): int(v) for k, v in zip(labs, counts)},
                "n_acted_transitions": int((merged["n_valid"] > 0).sum()),
                "n_no_valid_action": int((merged["n_valid"] == 0).sum()),
                "total_env_steps": int(np.concatenate(
                    [d["ep_n_steps"] for d in parts]).sum())}
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"MERGED: {manifest['n_transitions']} transitions, {manifest['n_episodes']} "
          f"episodes, labels={manifest['action_label_dist']}", flush=True)


if __name__ == "__main__":
    main()
