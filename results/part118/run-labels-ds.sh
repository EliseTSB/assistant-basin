#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
cd /home/elisepc/green/worktrees/ab-part67 && export OPENCODE_GO_API_KEY="$(grep -m1 "^OPENCODE_GO_API_KEY=" /home/elisepc/green/DiscordBot/.env | cut -d= -f2- | tr -d "\"")" && /usr/bin/python3 -I -B part118/label118.py --reader ds --answers results/part118/answers.jsonl --neurons results/part92/step2/neurons.json --out results/part118/labels-ds.jsonl --usage-out results/part118/labels-ds.usage.jsonl; echo CHAIN-RC=$?
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
