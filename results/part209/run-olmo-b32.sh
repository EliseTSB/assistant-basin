#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
env HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part209/run209.py --stage instruct --pilot --r2 --batch 32 --out results/part209/pilot/olmo-b32 --models-json results/part209/pilot/models.json
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
