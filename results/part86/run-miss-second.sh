#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
cd . && /usr/bin/python3 -I -B results/part86/second-reader/calib86.py --version 2 --label luna --items <local>/part86-miss86/items/miss-items-second.jsonl --out results/part86/miss86/labels_luna_v2_second.jsonl && /usr/bin/python3 -I -B results/part86/second-reader/calib86.py --version 2 --label luna --items <local>/part86-miss86/items/miss-items-second.jsonl --out results/part86/miss86/labels_luna_v2_second.jsonl; echo CHAIN-RC=$?
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
