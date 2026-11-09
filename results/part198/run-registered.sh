#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
bash results/part198/chain-registered.sh
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
