#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
env HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part206/record206.py --stage instruct --out results/part206 --batch 8 && env HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part206/record206.py --stage instruct --out results/part206 --pass gt
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
