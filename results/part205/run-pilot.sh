#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
/home/elisepc/neuron-probe/.venv/bin/python -I -B part205/run205.py --stage instruct --prompts part205/prompts205-pilot.json --out results/part205/pilot --check-pool --models-json results/part205/pilot/models.json
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
