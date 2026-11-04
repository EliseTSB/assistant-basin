#!/bin/bash
# Timing pilot driver: one process per configuration (so peak RSS is per configuration).
# usage: pilot_runs.sh SNAPSHOT_DIR CORPUS_PARQUET OUT_ROOT TAG
set -u
SNAP=$1; CORPUS=$2; ROOT=$3; TAG=$4
HERE=$(cd "$(dirname "$0")" && pwd)
for dtype in fp32 bf16; do
  for bs in 4 8; do
    name=$TAG-$dtype-bs$bs
    python "$HERE/olmo_atlas_collect.py" collect --model-dir "$SNAP" --corpus "$CORPUS" \
      --out "$ROOT/$name" --docs 40 --passes 1 --dtype $dtype --batch-size $bs \
      --checkpoint-every 20 --snapshot-every 20 --threads 4 > "$ROOT/$name.log" 2>&1
    grep RESULT "$ROOT/$name.log" | sed "s/^RESULT/$name/"
  done
done
