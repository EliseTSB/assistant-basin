#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
cd . && export OPENCODE_GO_API_KEY="$(grep -m1 "^OPENCODE_GO_API_KEY=" <local> | cut -d= -f2- | tr -d "\"")" && L="/usr/bin/python3 -I -B part92b/label92b.py" && $L --sample && $L --gate-a --reader ds && $L --gate-a --reader luna && $L --reader ds --out results/part92b/labels-ds.jsonl && $L --reader ds --out results/part92b/labels-ds.jsonl && $L --reader luna --out results/part92b/labels-luna.jsonl && $L --reader luna --out results/part92b/labels-luna.jsonl; echo CHAIN-RC=$?
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
