#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
/usr/bin/python3 -I -B part118/label118.py --reader primary --answers results/part118/answers.jsonl --neurons results/part92/step2/neurons.json --out results/part118/labels.jsonl --usage-out results/part118/labels.usage.jsonl
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
