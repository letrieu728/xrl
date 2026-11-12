#!/bin/bash
# Group 2a: run one BART target per process (crash isolation + checkpoints).
# Usage: run_graph_targets.sh <merged.npz> <ckptdir>
# Each target writes target_<name>.json (+ pickles) into <ckptdir>; a failed
# target is logged and does NOT stop the remaining targets.
set -u
MERGED="$1"
CKPT="$2"
SCRATCH=/home/hatch/workspace/step3-scratch
mkdir -p "$CKPT"
for T in base1 barrack1 worker1 light1 heavy1 ranged1 base2 barrack2 worker2 light2 heavy2 ranged2 timestep action; do
  echo "===== target $T ($(date -u +%FT%TZ)) ====="
  $SCRATCH/venv/bin/python -u $SCRATCH/group2/microrts_graph_target.py \
      --target "$T" --merged "$MERGED" --outdir "$CKPT" \
      > "$CKPT/target_${T}.log" 2>&1
  rc=$?
  echo "target $T EXIT:$rc"
  tail -n 3 "$CKPT/target_${T}.log"
done
echo "===== summary ====="
$SCRATCH/venv/bin/python -c "
import json, glob
for f in sorted(glob.glob('$CKPT/target_*.json')):
    r = json.load(open(f))
    print(r['target'], r['status'], r.get('causes'), f\"{r.get('time_s', 0):.1f}s\")
"
