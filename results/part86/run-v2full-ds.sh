#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
cd . && export OPENCODE_GO_API_KEY="$(grep -m1 "^OPENCODE_GO_API_KEY=" <local> | cut -d= -f2- | tr -d "\"")" && /usr/bin/python3 -I results/part86/second-reader/calib86.py --version 2 --label ds --items ~/jlens-persona-neurons/data/qwen3.5-4b/labels/items.jsonl --out results/part86/second-reader/labels_ds_v2_full.jsonl --workers 8; echo CHAIN-RC=$?
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
