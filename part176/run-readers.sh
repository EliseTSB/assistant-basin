#!/usr/bin/env bash
# Part 176 reader order: known checks, first labels, network-only retries, second reader, score, audit.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 9
PY="${PART176_PYTHON:-python3}"
OUT=results/part176
LABELS=$OUT/labels

if ! "$PY" -I -B -c 'import numpy' >/dev/null 2>&1; then
  echo "Part 176 reader run stopped: PART176_PYTHON='$PY' cannot import numpy under Python -I." >&2
  exit 12
fi

step() {
  "$@" | tail -c 1200
  rc=${PIPESTATUS[0]}
  echo
  echo "step_rc=$rc $(date +%H:%M:%S)"
  return "$rc"
}

echo "start $(date +%H:%M:%S)"
if ! step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" \
  --stage check-gates; then
  echo "Part 176 generation gates failed; scoring the unread result without starting readers." >&2
  step "$PY" -I -B part176/score176.py --answers "$OUT/answers.jsonl" --gates "$OUT/gates.json" \
    --labels-dir "$LABELS" --out-dir "$OUT"
  rc=$?
  echo "ALL DONE rc=$rc"
  exit "$rc"
fi
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" \
  --reader ds --stage known || { rc=$?; \
    step "$PY" -I -B part176/score176.py --answers "$OUT/answers.jsonl" --gates "$OUT/gates.json" \
      --labels-dir "$LABELS" --out-dir "$OUT"; \
    echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" \
  --reader haiku --stage known || { rc=$?; \
    step "$PY" -I -B part176/score176.py --answers "$OUT/answers.jsonl" --gates "$OUT/gates.json" \
      --labels-dir "$LABELS" --out-dir "$OUT"; \
    echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" \
  --reader ds --stage first || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part176/retry_net.py --reader ds --answers "$OUT/answers.jsonl" \
  --labels-dir "$LABELS"
retry_rc=$?
if [ "$retry_rc" -ne 0 ]; then
  echo "Part 176 DNS retry budget ended; remaining errors continue to the reader cap." >&2
fi

# The cap check keeps every remaining model/transport failure in the denominator.
step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" \
  --reader ds --stage error-cap
cap_rc=$?
if [ "$cap_rc" -ne 0 ]; then
  echo "Part 176 first-reader error cap failed; skipping the second reader and audit." >&2
  step "$PY" -I -B part176/score176.py --answers "$OUT/answers.jsonl" --gates "$OUT/gates.json" \
    --labels-dir "$LABELS" --out-dir "$OUT"
  rc=$?
  echo "ALL DONE rc=$rc"
  exit "$rc"
fi

step "$PY" -I -B part174/label174.py --answers "$OUT/answers.jsonl" --out-dir "$LABELS" \
  --reader haiku --stage second || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part176/retry_net.py --reader haiku --answers "$OUT/answers.jsonl" \
  --labels-dir "$LABELS"
retry_rc=$?
if [ "$retry_rc" -ne 0 ]; then
  echo "Part 176 Haiku DNS retry budget ended; remaining errors stay visible to scoring." >&2
fi
step "$PY" -I -B part176/score176.py --answers "$OUT/answers.jsonl" --gates "$OUT/gates.json" \
  --labels-dir "$LABELS" --out-dir "$OUT" || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
step "$PY" -I -B part176/audit176.py --answers "$OUT/answers.jsonl" \
  --labels-dir "$LABELS" --out-dir "$LABELS" || { rc=$?; echo "ALL DONE rc=$rc"; exit "$rc"; }
echo "ALL DONE rc=0"
exit 0
