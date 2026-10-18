#!/usr/bin/env bash
# Part 1f WSL runbook. Run `record`, the Windows GPU commands in REPORT.md, then `replay` and `compare`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/elisepc/neuron-probe/.venv/bin/python}"
CLIMB_HOME="${CLIMB_HOME:-/home/elisepc/neuron-climb}"
RUN_DIR="${RUN_DIR:-/home/elisepc/neuron-climb-1f}"
PART1_DIR="${PART1_DIR:-/home/elisepc/assistant-basin/results/part1-qwen3-0.6b}"
SPLIT_JSON="${SPLIT_JSON:-$ROOT/results/part1e-climb-split.json}"
BASELINE="${BASELINE:-$ROOT/climb/baseline}"
PIPELINE="${PIPELINE:-$CLIMB_HOME/work-luna/final/sim}"
JUDGE="${JUDGE:-$CLIMB_HOME/work-luna/final/judge}"
CACHE_LUNA_FROM="${CACHE_LUNA_FROM:-$CLIMB_HOME/work-test}"
RECORD_WORK="$RUN_DIR/work-clef"
REPLAY_WORK="$RUN_DIR/work-clef-replay"
REQUESTS="$RUN_DIR/requests.jsonl"
ANSWERS="${ANSWERS:-$RUN_DIR/answers.jsonl}"
RECORD_PORT="${RECORD_PORT:-8342}"
REPLAY_PORT="${REPLAY_PORT:-8343}"

fail() { printf 'run_1f.sh: %s\n' "$*" >&2; exit 2; }
check_python() {
  [[ -x "$PYTHON" ]] || fail "Python interpreter not found: $PYTHON"
}
check_inputs() {
  check_python
  [[ -f "$SPLIT_JSON" ]] || fail "TEST split not found: $SPLIT_JSON"
  [[ -f "$PART1_DIR/units.json" && -f "$PART1_DIR/key.json" && -f "$PART1_DIR/heldout.json" \
     && -f "$PART1_DIR/jev-sim.json" && -f "$PART1_DIR/verdicts-gpt-6-luna-high-full.json" ]] \
    || fail "PART1_DIR must contain units.json, key.json, heldout.json, jev-sim.json, and verdicts-gpt-6-luna-high-full.json: $PART1_DIR"
  [[ -f "$BASELINE/candidate.json" && -f "$PIPELINE/candidate.json" && -f "$JUDGE/candidate.json" ]] \
    || fail 'baseline, frozen c23 pipeline, and frozen c45 judge candidate files are required'
  [[ -d "$CACHE_LUNA_FROM/cache/luna" ]] || fail "cached Luna descriptions not found under $CACHE_LUNA_FROM/cache/luna"
}
prepare_work() {
  local target="$1"
  rm -rf "$target"
  mkdir -p "$target/cache/jev"
  cp -a "$CACHE_LUNA_FROM/cache/luna" "$target/cache/luna"
}
run_test_score() {
  local target="$1" score_path="$2" rows_path="$3" url="$4"
  CLIMB_WORK="$target" CLIMB_NAME=claude-test JEV_URL="$url" \
    "$PYTHON" "$ROOT/climb/climb_test.py" "$PART1_DIR" "$SPLIT_JSON" \
    "$BASELINE" "$PIPELINE" "$JUDGE" --out "$score_path" --rows-out "$rows_path"
}
start_stub() {
  local mode="$1" port="$2" log="$3"
  shift 3
  "$PYTHON" "$ROOT/jev_stub.py" --mode "$mode" --host 127.0.0.1 --port "$port" "$@" >"$log" 2>&1 &
  STUB_PID=$!
  trap stop_stub EXIT INT TERM
  for _ in $(seq 1 100); do
    if ! kill -0 "$STUB_PID" 2>/dev/null; then cat "$log" >&2; fail 'Jev stub exited before becoming ready'; fi
    if "$PYTHON" -c 'import socket,sys; s=socket.socket(); s.settimeout(.2); r=s.connect_ex(("127.0.0.1",int(sys.argv[1]))); s.close(); sys.exit(r)' "$port"; then
      return
    fi
    sleep 0.1
  done
  fail "Jev stub did not listen on port $port"
}
stop_stub() {
  if [[ -n "${STUB_PID:-}" ]]; then
    kill "$STUB_PID" 2>/dev/null || true
    wait "$STUB_PID" 2>/dev/null || true
    unset STUB_PID
  fi
}

action="${1:-}"
case "$action" in
  record)
    check_inputs
    mkdir -p "$RUN_DIR"
    prepare_work "$RECORD_WORK"
    rm -f "$REQUESTS" "$RUN_DIR/record-stub.log"
    start_stub record "$RECORD_PORT" "$RUN_DIR/record-stub.log" --requests "$REQUESTS"
    run_test_score "$RECORD_WORK" "$RECORD_WORK/record-test-scores.json" \
      "$RECORD_WORK/record-test-rows.json" "http://127.0.0.1:$RECORD_PORT/v1/systemone"
    stop_stub
    printf 'recorded requests: %s\n' "$REQUESTS"
    ;;
  replay)
    check_inputs
    [[ -f "$REQUESTS" ]] || fail "recorded requests are missing: $REQUESTS"
    [[ -f "$ANSWERS" ]] || fail "Clef answers are missing: $ANSWERS"
    prepare_work "$REPLAY_WORK"
    rm -f "$RUN_DIR/replay-stub.log"
    start_stub replay "$REPLAY_PORT" "$RUN_DIR/replay-stub.log" --answers "$ANSWERS"
    run_test_score "$REPLAY_WORK" "$REPLAY_WORK/clef-test-scores.json" \
      "$REPLAY_WORK/clef-test-rows.json" "http://127.0.0.1:$REPLAY_PORT/v1/systemone"
    stop_stub
    printf 'Clef TEST scores: %s\n' "$REPLAY_WORK/clef-test-scores.json"
    ;;
  compare)
    check_python
    [[ -f "$CLIMB_HOME/work-test/test-scores.json" ]] || fail 'original Jev TEST scores are missing'
    [[ -f "$REPLAY_WORK/clef-test-scores.json" && -f "$REPLAY_WORK/clef-test-rows.json" \
       && -f "$REQUESTS" && -f "$ANSWERS" && -f "$RUN_DIR/clef-full.log" ]] \
      || fail "replay or GPU outputs are missing under $RUN_DIR"
    "$PYTHON" "$ROOT/compare_1f.py" \
      --jev-scores "$CLIMB_HOME/work-test/test-scores.json" \
      --clef-scores "$REPLAY_WORK/clef-test-scores.json" \
      --rows "$REPLAY_WORK/clef-test-rows.json" \
      --requests "$REQUESTS" --answers "$ANSWERS" \
      --jev-cache "$CLIMB_HOME/work-test/cache/jev" \
      --gpu-log "$RUN_DIR/clef-full.log"
    ;;
  *)
    cat >&2 <<EOF
Usage: $0 record|replay|compare

Run record first. Then copy requests.jsonl to Windows, run clef_batch_gpu.py
with --limit 32 (smoke), then run it again without --limit (resuming the same
answers file). Copy answers.jsonl and the full-run log back into $RUN_DIR,
then run replay and compare.
EOF
    exit 2
    ;;
esac
