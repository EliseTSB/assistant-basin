#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
cd . && export SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt && python3 part92/label92.py --answers results/part117/items.jsonl --neurons results/part117/personas.json --out results/part117/labels.jsonl && python3 part92/label92.py --answers results/part117/items.jsonl --neurons results/part117/personas.json --out results/part117/labels.jsonl && python3 part117/second117.py --label && python3 part117/second117.py --agree && python3 part117/score117.py > results/part117/score.log 2>&1; echo CHAIN-RC=$? >> results/part117/score.log
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
