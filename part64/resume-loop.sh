#!/bin/bash
# resume-loop.sh — drive a paid gen64 run (Part 64's pipelined writer) to completion, unattended.
#
# WHAT IT DOES: runs `gen64.py resume` until run.json says complete. It retries only a network-shaped failure
# (transport error, reset, remote close, timeout), after a 90 s pause, up to 25 passes. Anything else (a chunk that
# needs a person, a stage-cap stop, a probe stop, a possible duplicate) ends the loop with exit 3 for a human look.
#
# WHAT IT GOT WRONG FIRST (2026-10-23):
#  1. The first version grepped the log for "RecoveryRequired", which gen64 never prints (it prints "gen64: <msg>"),
#     so it retried every error, including ones that need a person. Each retry is a fresh process, which re-lists
#     the account's batches and re-downloads every ended batch's results (~40 MB) for an unknown-POST chunk.
#  2. The second version's network pattern missed "Remote end closed connection without response" and stopped a
#     healthy run for 37 minutes. Widen the pattern from a real log line, never from memory.
# Run it detached (setsid nohup) so a session boundary does not kill it; it reads the credit key from
# ~/.anthropic-credits.env and unsets the proxy variables (direct API calls only).
W="${GEN64_WORKTREE:-/home/elisepc/green/worktrees/ab-part64}"; cd "$W" || exit 2
P=/home/elisepc/neuron-probe/.venv/bin/python
set -a; . /home/elisepc/.anthropic-credits.env; set +a
for i in $(seq 1 25); do
  env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy $P part64/gen64.py resume --collection results/part64/collection.jsonl --root results/part64/gen64 >> results/part64/resume.log 2>&1 < /dev/null
  rc=$?
  st=$(python3 -c "import json; print(json.load(open('results/part64/gen64/run.json')).get('status'))")
  last=$(tail -n 1 results/part64/resume.log)
  echo "$(date +%H:%M:%S) pass $i rc=$rc status=$st :: ${last:0:200}"
  [ "$st" = complete ] && exit 0
  if ! printf '%s' "$last" | grep -qE "API transport failed|timed out|Connection reset|Temporary failure|Network is unreachable|Remote end closed|RemoteDisconnected|Connection aborted|EOF occurred|IncompleteRead|URLError"; then
    echo "$(date +%H:%M:%S) stopping: not a network failure"; exit 3
  fi
  sleep 90
done
echo "loop bound reached"; exit 4
