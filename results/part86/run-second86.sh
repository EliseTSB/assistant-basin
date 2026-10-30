#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
cd . && export SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt && /usr/bin/python3 -I results/part86/second-reader/second86.py --label && /usr/bin/python3 -I results/part86/second-reader/second86.py --agree; echo CHAIN-RC=$?
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
