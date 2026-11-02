#!/usr/bin/env bash
# Run after Part 174 has written its answers and gates.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 9
PY="${PART174_PYTHON:-python3}"
OUT=results/part174
LABELS=$OUT/labels

step() {
  "$@" | tail -c 800
  rc=${PIPESTATUS[0]}
  echo
  echo "step_rc=$rc $(date +%H:%M:%S)"
  return "$rc"
}

echo "start $(date +%H:%M:%S)"
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --stage check-gates || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --reader ds --stage known || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --reader haiku --stage known || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --reader ds --stage first || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
# The one retry pass runs before the first-reader error cap is inspected.
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --reader ds --stage retry-failed || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --stage error-cap || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" --reader haiku --stage second || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/score174.py --answers "$OUT/answers.jsonl" --gates "$OUT/gates.json" --labels-dir "$LABELS" --out-dir "$OUT" || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
rc=0
echo "ALL DONE rc=$rc"
exit "$rc"
