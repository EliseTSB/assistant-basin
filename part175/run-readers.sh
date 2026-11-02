#!/usr/bin/env bash
# Run after Part 175 has written Stage A, answers and generation gates.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 9
PY="${PART175_PYTHON:-python3}"
OUT=results/part175
LABELS=$OUT/labels

if ! "$PY" -I -B -c 'import numpy' >/dev/null 2>&1; then
  echo "Part 175 reader run stopped: PART175_PYTHON='$PY' cannot import numpy under Python -I." >&2
  exit 12
fi

step() {
  "$@" | tail -c 800
  rc=${PIPESTATUS[0]}
  echo
  echo "step_rc=$rc $(date +%H:%M:%S)"
  return "$rc"
}

echo "start $(date +%H:%M:%S)"
if "$PY" -I -B -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "stopped" else 1)' "$OUT/stage_a.json"; then
  step "$PY" -I -B part175/score175.py --answers "$OUT/answers.jsonl" --gates "$OUT/gates.json" \
    --labels-dir "$LABELS" --out-dir "$OUT" --stage-a "$OUT/stage_a.json" \
    --stage-a-answers "$OUT/stage_a_answers.jsonl" || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
  echo "ALL DONE rc=0"
  exit 0
fi

# If a generation gate failed, preserve the scorer's explicit unread/fallback output without calling readers.
if ! step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --stage check-gates; then
  step "$PY" -I -B part175/score175.py --answers "$OUT/answers.jsonl" --gates "$OUT/gates.json" \
    --labels-dir "$LABELS" --out-dir "$OUT" --stage-a "$OUT/stage_a.json" \
    --stage-a-answers "$OUT/stage_a_answers.jsonl"
  rc=$?
  echo "ALL DONE rc=$rc"
  exit "$rc"
fi
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --reader ds --stage known || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --reader haiku --stage known || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --reader ds --stage first || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
# The one retry pass runs before the first-reader error cap is inspected.
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --reader ds --stage retry-failed || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --stage error-cap || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --reader haiku --stage second || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part175/score175.py --answers "$OUT/answers.jsonl" --gates "$OUT/gates.json" \
  --labels-dir "$LABELS" --out-dir "$OUT" --stage-a "$OUT/stage_a.json" \
  --stage-a-answers "$OUT/stage_a_answers.jsonl" || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
echo "ALL DONE rc=0"
exit 0
