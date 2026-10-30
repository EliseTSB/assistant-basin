#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
cd . && export OPENCODE_GO_API_KEY="$(grep -m1 "^OPENCODE_GO_API_KEY=" <local> | cut -d= -f2- | tr -d "\"")" && M="/usr/bin/python3 -I -B part86/miss86.py label --items <local>/part86-miss86/items/miss-items.jsonl --out <local>/part86-miss86/labels/labels_ds_v2.jsonl --usage <local>/part86-miss86/labels/labels_ds_v2.usage.jsonl" && $M && $M; echo CHAIN-RC=$?
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
