#!/usr/bin/env bash
# Run only on the rented scoring machine after uploading the local generation package.
set -u

RUN_ROOT=${PART60_HOME:-/root/part60}
LOG_DIR="$RUN_ROOT/logs"
RETURN_READY_PATH=${PART60_RETURN_PATH:-/root/part60-return.tgz}
RETURN_INCOMPLETE_PATH=${PART60_INCOMPLETE_PATH:-/root/part60-return-INCOMPLETE.tgz}
MODEL_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
MAX_RUNTIME_SECONDS=${PART60_MAX_RUNTIME_SECONDS:-14400}
mkdir -p "$LOG_DIR" "$RUN_ROOT/results/part60/acts" "$RUN_ROOT/results/part60/reproduction"
cd "$RUN_ROOT" || exit 2
RUN_START_SECONDS=$SECONDS

# A failed rerun must never leave an older success archive at the path an operator copies.
rm -f "$RETURN_READY_PATH" "$RETURN_INCOMPLETE_PATH" "$RUN_ROOT/part60-return.tmp.tgz"

declare -A phase_status=()
failures=()
run_status=0

record_status() {
  local phase=$1 rc=$2
  echo "$phase exit=$rc"
  phase_status[$phase]=$rc
  if [ "$rc" -ne 0 ]; then
    run_status=1
    failures+=("$phase")
  fi
}

run_checked() {
  local phase=$1
  shift
  echo "=== $phase $(date +%H:%M:%S)"
  local elapsed=$((SECONDS - RUN_START_SECONDS))
  local remaining=$((MAX_RUNTIME_SECONDS - elapsed))
  if [ "$remaining" -le 0 ]; then
    printf 'runtime limit reached before %s (%s seconds)\n' "$phase" "$MAX_RUNTIME_SECONDS" > "$LOG_DIR/$phase.log"
    record_status "$phase" 124
    return 0
  fi
  timeout --signal=TERM --kill-after=30 "${remaining}s" "$@" > "$LOG_DIR/$phase.log" 2>&1
  local rc=$?
  record_status "$phase" "$rc"
  return 0
}

skip_phase() {
  local phase=$1 reason=$2
  printf 'skipped: %s\n' "$reason" | tee "$LOG_DIR/$phase.log"
  phase_status[$phase]=125
}

if [ "${PART60_STUB:-0}" = "1" ]; then
  echo "stubbed run mode"
  run_checked package-setup true
  run_checked model-prefetch true
  run_checked score-dev true
  run_checked verify-dev true
  if [ -f "$RUN_ROOT/results/part60/out/test/M0.jsonl" ]; then
    run_checked score-test true
    run_checked verify-test true
  else
    skip_phase score-test "no registered test package was uploaded"
    skip_phase verify-test "no registered test package was uploaded"
  fi
  printf 'stubbed\n' > "$RUN_ROOT/versions-run.txt"
else
  required=(
    part60/score60.py part60/gen60.py
    results/part56-pilot/pilot.jsonl results/part56-pilot/label/atlas-labels.jsonl
    results/part58/ids.all.json results/part58/units.all.json
    results/part58/sent.all.jsonl results/part58/acts.all.npz
    results/part58/gen.all.jsonl results/part58/gen.dev.jsonl results/part58/gen.test.jsonl
    results/part60/thresholds.json
    results/part60/out/dev/M0.jsonl results/part60/out/dev/M1.jsonl
    results/part60/out/dev/M2.jsonl results/part60/out/dev/M3.jsonl
    results/part60/out/dev/M4.jsonl results/part60/out/dev/M5.jsonl
    results/part60/out/dev/M6.jsonl
  )
  missing=()
  for path in "${required[@]}"; do
    [ -f "$RUN_ROOT/$path" ] || missing+=("$path")
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    printf 'missing required input: %s\n' "${missing[*]}"
    record_status preflight 1
  else
    record_status preflight 0
  fi

  if [ "${phase_status[preflight]}" -eq 0 ]; then
    run_checked package-setup python3 -m pip install transformers==5.20.0 accelerate huggingface_hub
    if [ "${phase_status[package-setup]}" -eq 0 ]; then
      run_checked version-record bash -c '
        version_file=$1
        if python3 -m pip freeze > "$version_file"; then exit 0; fi
        if python3 -c '\''import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'\'' > "$version_file"; then
          echo "pip freeze failed; recorded torch/transformers fallback" >&2
          exit 0
        fi
        exit 1
      ' bash "$RUN_ROOT/versions-run.txt"
      run_checked gpu-inventory nvidia-smi
      run_checked python-inventory python3 -c 'import torch, transformers; print("torch",torch.__version__,"transformers",transformers.__version__,"cuda",torch.version.cuda,torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda")'
    else
      skip_phase version-record "package setup failed"
      skip_phase gpu-inventory "package setup failed"
      skip_phase python-inventory "package setup failed"
    fi
  else
    skip_phase package-setup "preflight failed"
    skip_phase version-record "preflight failed"
    skip_phase gpu-inventory "preflight failed"
    skip_phase python-inventory "preflight failed"
  fi

  if [ "${phase_status[preflight]:-125}" -eq 0 ] && [ "${phase_status[package-setup]:-125}" -eq 0 ]; then
    run_checked model-prefetch python3 -c 'from pathlib import Path; import sys; from huggingface_hub import snapshot_download; Path(sys.argv[1]).write_text(snapshot_download("Qwen/Qwen3-1.7B", revision=sys.argv[2]))' "$RUN_ROOT/model-path.txt" "$MODEL_REVISION"
    prefetch_rc=${phase_status[model-prefetch]}
  else
    skip_phase model-prefetch "package setup failed"
    prefetch_rc=125
  fi

  if [ "$prefetch_rc" -eq 0 ]; then
    model_dir=$(cat "$RUN_ROOT/model-path.txt")
    dev_methods=M0,M1,M2,M3,M4,M5,M6
    run_checked score-dev python3 -u part60/score60.py --root results/part60 score --split dev --methods "$dev_methods" --model "$model_dir" --device cuda --dtype bf16 --batch-size 8
    if [ "${phase_status[score-dev]}" -eq 0 ]; then
      run_checked verify-dev python3 -u part60/score60.py --root results/part60 verify --split dev --methods "$dev_methods" --model "$model_dir" --device cuda --dtype bf16 --batch-size 8
    else
      skip_phase verify-dev "dev scoring failed"
    fi

    if [ -f "$RUN_ROOT/results/part60/out/test/M0.jsonl" ]; then
      selection_file="$RUN_ROOT/results/part60/analysis/test-selection.json"
      if [ ! -f "$selection_file" ]; then
        record_status test-gate 1
      else
        test_status=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("status",""))' "$selection_file")
        candidate=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("candidate") or "")' "$selection_file")
        test_methods=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(d.get("methods", [])))' "$selection_file")
        if [ "$test_status" != "ready" ] || [ -z "$test_methods" ] || [ "$test_methods" != "M0,M1" ] && [ "$test_methods" != "M0,M1,$candidate" ]; then
          record_status test-gate 1
        else
          test_missing=()
          IFS=',' read -r -a selected_methods <<< "$test_methods"
          for method in "${selected_methods[@]}"; do
            if [ "$method" != "M0" ] && [ ! -f "$RUN_ROOT/results/part60/out/test/$method.jsonl" ]; then
              test_missing+=("results/part60/out/test/$method.jsonl")
            fi
          done
          if [ "${#test_missing[@]}" -gt 0 ]; then
            printf 'test candidate outputs missing: %s\n' "${test_missing[*]}"
            record_status test-gate 1
          else
            record_status test-gate 0
            run_checked score-test python3 -u part60/score60.py --root results/part60 score --split test --methods "$test_methods" --model "$model_dir" --device cuda --dtype bf16 --batch-size 8
            if [ "${phase_status[score-test]}" -eq 0 ]; then
              run_checked verify-test python3 -u part60/score60.py --root results/part60 verify --split test --methods "$test_methods" --model "$model_dir" --device cuda --dtype bf16 --batch-size 8
            else
              skip_phase verify-test "test scoring failed"
            fi
          fi
        fi
      fi
    else
      skip_phase test-gate "test split was not prepared after dev analysis"
      skip_phase score-test "test split was not prepared after dev analysis"
      skip_phase verify-test "test split was not prepared after dev analysis"
    fi
  else
    skip_phase score-dev "model prefetch failed"
    skip_phase verify-dev "model prefetch failed"
    skip_phase test-gate "model prefetch failed"
    skip_phase score-test "model prefetch failed"
    skip_phase verify-test "model prefetch failed"
  fi
fi

date +%s > "$RUN_ROOT/run60.end"
archive_tmp="$RUN_ROOT/part60-return.tmp.tgz"
archive_paths=(results/part60/acts results/part60/reproduction results/part60/thresholds.json logs versions-run.txt run60.end)
present=()
for path in "${archive_paths[@]}"; do
  [ -e "$RUN_ROOT/$path" ] && present+=("$path")
done
mkdir -p "$(dirname "$RETURN_READY_PATH")" "$(dirname "$RETURN_INCOMPLETE_PATH")"
tar -C "$RUN_ROOT" -czf "$archive_tmp" "${present[@]}"
archive_rc=$?
if [ "$archive_rc" -ne 0 ]; then
  record_status archive "$archive_rc"
elif [ "$run_status" -eq 0 ]; then
  mv -f "$archive_tmp" "$RETURN_READY_PATH"
  record_status archive-rename $?
else
  mv -f "$archive_tmp" "$RETURN_INCOMPLETE_PATH"
  record_status incomplete-archive-rename $?
fi

if [ "$run_status" -eq 0 ]; then
  echo "RETURN_READY"
  exit 0
fi
printf 'RUN_FAILED failing_phases='
if [ "${#failures[@]}" -eq 0 ]; then
  echo unknown
else
  printf '%s' "${failures[0]}"
  for ((i=1; i<${#failures[@]}; i++)); do printf ',%s' "${failures[$i]}"; done
  echo
fi
exit 1
