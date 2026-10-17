#!/bin/bash
# climb-opencode.sh <name> <opencode model> — a free-model climber for Part 1e (Claude, 2026-10-16).
# opencode has no /goal, so this loop is the goal: it runs `opencode run` passes in work-<name> until 07:45 CT or until
# FINAL.md exists, each pass continuing the same session. Every pass runs inside codex's Linux sandbox: writes only to
# the work folder and opencode's own state folders, network on (Luna via 127.0.0.1:8317, Jev via the relay on :8341).
set -u
NAME=$1; MODEL=$2; W=/home/elisepc/neuron-climb/work-$NAME; cd "$W" || exit 2
END=$(date -d '2026-10-17 07:45 CDT' +%s)
SB=(codex sandbox -c 'sandbox_mode="workspace-write"' -c 'sandbox_workspace_write.network_access=true'
    -c "sandbox_workspace_write.writable_roots=[\"$HOME/.local/share/opencode\",\"$HOME/.cache/opencode\",\"$HOME/.local/state/opencode\",\"$HOME/.config/opencode\"]")
FIRST="Your task is in ./GOAL.md (a copy of /home/elisepc/neuron-climb/GOAL.md). Read it now and follow it exactly. Your working directory is $W; the current best is ./best. Start by writing STATUS.md, then run experiments. Keep STATUS.md current: this session will be continued in later passes until 07:45 America/Chicago."
NEXT="Continue the climb in ./GOAL.md. Check the time (TZ=America/Chicago date), read STATUS.md and the tail of ledger.jsonl, then run the next experiments and update STATUS.md. If it is 07:30 or later, write the deliverables (final/ and FINAL.md) and stop."
pass=0
while [ "$(date +%s)" -lt "$END" ] && [ ! -f FINAL.md ]; do
  pass=$((pass + 1)); echo "=== pass $pass $(TZ=America/Chicago date '+%F %T %Z')"
  if [ "$pass" = 1 ]; then P=$FIRST; C=(); else P=$NEXT; C=(--continue); fi
  env -u HTTPS_PROXY -u HTTP_PROXY -u NODE_EXTRA_CA_CERTS CLIMB_NAME=$NAME "${SB[@]}" opencode run -m "$MODEL" "${C[@]}" "$P"
  echo "=== pass $pass ended rc=$? $(TZ=America/Chicago date '+%T')"; sleep 20
done
echo "=== climber $NAME done $(TZ=America/Chicago date '+%F %T')"
