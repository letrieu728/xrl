#!/bin/bash
# Group 2a chain: merge batches -> per-target BART graph -> merge graph/SCM
# -> fidelity. Run AFTER all rollout batches finish. Logs each stage.
set -u
SCRATCH=/home/hatch/workspace/step3-scratch
G2=$SCRATCH/group2
OUT=$G2/microrts100
# dbarts 0.9.34 lives in ~/workspace/R-lib (survives VM restarts; the
# /usr/local R library does not). rpy2 inherits R_LIBS.
export R_LIBS=/home/hatch/workspace/R-lib
mkdir -p "$OUT"
echo "===== STAGE merge ($(date -u +%FT%TZ)) ====="
$SCRATCH/venv/bin/python -u $G2/merge_batches.py --batchdir $G2/batches \
  --out $OUT/microrts_merged.npz --manifest $OUT/merge_manifest.json \
  2>&1 | tee $OUT/merge.log
echo "===== STAGE graph targets ($(date -u +%FT%TZ)) ====="
bash $G2/run_graph_targets.sh $OUT/microrts_merged.npz $OUT/ckpt \
  2>&1 | tee $OUT/graph_targets.log
echo "===== STAGE merge graph ($(date -u +%FT%TZ)) ====="
$SCRATCH/venv/bin/python -u $G2/merge_graph_targets.py --ckptdir $OUT/ckpt \
  --merged $OUT/microrts_merged.npz --outdir $OUT/artifacts 2>&1 | tee $OUT/merge_graph.log
echo "===== STAGE fidelity ($(date -u +%FT%TZ)) ====="
$SCRATCH/venv/bin/python -u $G2/microrts_fidelity.py --merged $OUT/microrts_merged.npz \
  --ckptdir $OUT/ckpt --out $OUT/fidelity.json 2>&1 | tee $OUT/fidelity.log
echo "===== CHAIN DONE ($(date -u +%FT%TZ)) ====="
