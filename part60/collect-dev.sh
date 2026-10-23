#!/bin/bash
# collect-dev.sh: polls and collects the six Part 60 dev batches (M1,M2,M4 primary; M3 selection; M5 round1; M6 primary) every 60 s until all are collected (80 rounds). Idempotent: a collect that fails (batch not ended) is retried; one that succeeds is not repeated.
# WHAT IT MEASURES: nothing; it only runs `gen60.py collect`. WHAT IT GOT WRONG THE FIRST TIME: nothing yet. NOTE: it is a background loop and dies at a cleancontinue boundary: rerun it.
cd /home/elisepc/assistant-basin
set -a; . ~/.anthropic-credits.env; set +a
export SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt
pending="M1:primary M2:primary M4:primary M3:selection M5:round1 M6:primary"
for round in $(seq 1 80); do
  next=""
  for spec in $pending; do m=${spec%%:*}; p=${spec##*:}
    if timeout 120 python3 part60/gen60.py collect --method $m --split dev --phase $p > /tmp/claude-1000/col-$m-$p.out 2>&1; then echo "$(date +%H:%M) collected $m $p"; else next="$next $spec"; fi
  done
  pending=$(echo $next)
  [ -z "$pending" ] && { echo "ALL COLLECTED $(date +%H:%M)"; exit 0; }
  sleep 60
done
echo "TIMEOUT pending: $pending"; exit 1
