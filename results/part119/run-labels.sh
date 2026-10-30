#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
export OPENCODE_GO_API_KEY="$(grep -m1 "^OPENCODE_GO_API_KEY=" /home/elisepc/green/DiscordBot/.env | cut -d= -f2- | tr -d "\"")" && L="/usr/bin/python3 -I -B part119/label119.py" && $L --gate --reader ds && $L --gate --reader luna && $L --reader ds --out results/part119/labels-ds.jsonl && $L --reader luna --out results/part119/labels-luna.jsonl
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
