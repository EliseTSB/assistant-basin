#!/bin/bash
# Part 64 rented-GPU collection/scoring runner. The same invocation can be
# repeated after local generation: it reuses a validated collection artifact.
set -u

RUN_ROOT="${PART64_ROOT:-/root/assistant-basin}"
RETURN_DIR="${PART64_RETURN_DIR:-/root}"
CORPUS="${PART64_CORPUS:-/mnt/c/Users/alexa/neuron-probe-gpu/qwen35/pile10k.jsonl}"
GENERATED="${PART64_GENERATED:-$RUN_ROOT/results/part64/generated.jsonl}"
RESULTS="$RUN_ROOT/results/part64"
LOGS="$RUN_ROOT/logs/part64"
MODEL_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_DIR="/root/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/$MODEL_REVISION"
FRESH="$RESULTS/fresh.jsonl"
COLLECTION="$RESULTS/collection.jsonl"
MANIFEST="$COLLECTION.manifest.json"
SCORED="$RESULTS/scored.jsonl"

cd "$RUN_ROOT" || exit 2
mkdir -p "$RESULTS" "$LOGS" "$RETURN_DIR"
touch "$RUN_ROOT/versions-pip.txt"
date +%s > "$RUN_ROOT/run64.start"
export PART60_BASIN="$RUN_ROOT"
export PART64_BASIN="$RUN_ROOT"

run_status=0
failures=()
declare -A phase_status=()

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
  "$@" > "$LOGS/$phase.log" 2>&1
  local rc=$?
  record_status "$phase" "$rc"
  return 0
}

skip_phase() {
  local phase=$1 reason=$2
  printf 'skipped: %s\n' "$reason" | tee "$LOGS/$phase.log"
  record_status "$phase" 125
  return 0
}

note_skip() {
  local phase=$1 reason=$2
  printf 'skipped: %s\n' "$reason" | tee "$LOGS/$phase.log"
  phase_status[$phase]=0
  return 0
}

run_phase() {
  local phase=$1 dependency=$2
  shift 2
  if [ "${phase_status[$dependency]:-125}" -ne 0 ]; then
    skip_phase "$phase" "dependency $dependency did not exit 0"
    return 0
  fi
  run_checked "$phase" "$@"
}

required_files=(
  part64/collect64_model.py part64/collect64.py part64/score64.py
  part60/gen60.py part60/score60.py part60/analyze60.py
  results/part56-pilot/pilot.jsonl
  results/part58/acts.all.npz results/part58/ids.all.json
  results/part58/units.all.json results/part58/sent.all.jsonl
)
missing_files=()
for path in "${required_files[@]}"; do
  if [ ! -f "$RUN_ROOT/$path" ]; then
    missing_files+=("$path")
  fi
done
if [ "${#missing_files[@]}" -gt 0 ]; then
  printf 'missing required input: %s\n' "${missing_files[@]}" > "$LOGS/preflight-files.log"
  cat "$LOGS/preflight-files.log"
  record_status preflight-files 1
else
  record_status preflight-files 0
fi

if [ "${phase_status[preflight-files]}" -eq 0 ]; then
  run_checked package-install python -m pip install transformers==5.20.0 flash-linear-attention==0.5.2 accelerate huggingface_hub
  if [ "${phase_status[package-install]}" -eq 0 ]; then
    python -m pip uninstall -y causal-conv1d > "$LOGS/optional-uninstall.log" 2>&1 || true
    run_checked package-version-check python -c 'import importlib.metadata as m; names=("torch", "transformers", "flash-linear-attention", "accelerate", "huggingface_hub"); versions={name:m.version(name) for name in names}; print("\n".join(f"{name}=={version}" for name,version in versions.items())); assert versions["transformers"] == "5.20.0", versions["transformers"]; assert versions["flash-linear-attention"] == "0.5.2", versions["flash-linear-attention"]'
    if [ "${phase_status[package-version-check]}" -eq 0 ]; then
      run_checked package-version-record bash -c 'python -m pip freeze > "$1/versions-pip.txt"' _ "$RUN_ROOT"
    else
      skip_phase package-version-record "package-version-check did not exit 0"
    fi
  else
    skip_phase package-version-check "package-install did not exit 0"
    skip_phase package-version-record "package-install did not exit 0"
  fi
else
  skip_phase package-install "preflight-files did not exit 0"
  skip_phase package-version-check "preflight-files did not exit 0"
  skip_phase package-version-record "preflight-files did not exit 0"
fi

if [ "${phase_status[package-version-record]:-125}" -eq 0 ]; then
  run_checked gpu-inventory nvidia-smi
  run_checked gpu-inventory-query nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  run_checked python-inventory python -c 'import torch, transformers; print("torch",torch.__version__,"cuda",torch.version.cuda,"transformers",transformers.__version__,torch.cuda.get_device_name(0))'
else
  skip_phase gpu-inventory "package-version-record did not exit 0"
  skip_phase gpu-inventory-query "package-version-record did not exit 0"
  skip_phase python-inventory "package-version-record did not exit 0"
fi

fetch_model() {
  local repo=$1 revision=$2 attempt rc
  for attempt in 1 2 3; do
    if HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - "$repo" "$revision" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir='/root/hf/hub'))
PY
    then
      return 0
    else
      rc=$?
    fi
    echo "Hugging Face fetch attempt $attempt/3 failed for $repo (exit $rc)" >&2
    if [ "$attempt" -lt 3 ]; then sleep 1; fi
  done
  return "$rc"
}

if [ "${phase_status[python-inventory]:-125}" -eq 0 ]; then
  run_checked prefetch-qwen fetch_model Qwen/Qwen3-1.7B "$MODEL_REVISION"
else
  skip_phase prefetch-qwen "package or GPU inventory setup did not complete"
fi
export HF_HOME=/root/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_phase corpus-verify prefetch-qwen env PYTHONPATH="$RUN_ROOT" python -c 'import json,sys; from pathlib import Path; from part64.collect64 import verify_corpus; print(json.dumps(verify_corpus(Path(sys.argv[1])), sort_keys=True))' "$CORPUS"

if [ "${phase_status[corpus-verify]:-125}" -eq 0 ]; then
  if [ -e "$COLLECTION" ] || [ -e "$MANIFEST" ]; then
    run_checked collection-cache-check env PYTHONPATH="$RUN_ROOT" python -c 'import json,sys; from pathlib import Path; from part64.collect64 import verify_corpus; corpus=verify_corpus(Path(sys.argv[1])); out=Path(sys.argv[2]); manifest=Path(sys.argv[3]); data=json.loads(manifest.read_text()); ok=out.is_file() and data.get("subset_verified") is True and data.get("unit_count")==500 and data.get("corpus",{}).get("sha256")==corpus["sha256"] and data.get("median_fresh_top12_match",0)>=10; print("validated existing collection" if ok else "existing collection cache is incomplete or mismatched"); raise SystemExit(0 if ok else 2)' "$CORPUS" "$COLLECTION" "$MANIFEST"
  else
    run_phase collection-model prefetch-qwen env PYTHONPATH="$RUN_ROOT" python -u part64/collect64_model.py --data "$CORPUS" --out "$FRESH" --cache-dir /root/hf/hub
    run_phase collection-assembly collection-model env PYTHONPATH="$RUN_ROOT" python -u part64/collect64.py --fresh "$FRESH" --out "$COLLECTION" --corpus "$CORPUS"
  fi
else
  skip_phase collection-model "corpus verification did not exit 0"
  skip_phase collection-assembly "corpus verification did not exit 0"
fi

GEN_INPUT_ARGS=()
if [ -f "$GENERATED" ]; then
  GEN_INPUT_ARGS+=(--input "$GENERATED")
else
  for writer in haiku opus; do
    for sample in 1x 3x; do
      for unit_type in neuron control; do
        generated_group="$RUN_ROOT/results/part64/gen64/outputs/$writer-$sample-$unit_type.jsonl"
        if [ -f "$generated_group" ]; then
          GEN_INPUT_ARGS+=(--input "$generated_group")
        fi
      done
    done
  done
fi

if [ "${#GEN_INPUT_ARGS[@]}" -gt 0 ]; then
  if [ "${phase_status[collection-cache-check]:-${phase_status[collection-assembly]:-125}}" -eq 0 ]; then
    run_checked scoring python -u part64/score64.py "${GEN_INPUT_ARGS[@]}" --out "$SCORED" --part60-root "$RUN_ROOT/results/part60" --model "$MODEL_DIR" --device cuda
  else
    skip_phase scoring "collection did not validate or complete"
  fi
else
  if [ "${phase_status[collection-cache-check]:-${phase_status[collection-assembly]:-125}}" -eq 0 ]; then
    note_skip scoring "generated sentences are absent; expected on the collection return before local generation"
  else
    skip_phase scoring "collection did not validate or complete"
  fi
fi

echo "=== archive $(date +%H:%M:%S)"
date +%s > "$RUN_ROOT/run64.end"
archive_tmp="$RETURN_DIR/part64-return.tmp.tgz"
archive_paths=(results/part64 logs/part64 versions-pip.txt run64.start run64.end)
tar -C "$RUN_ROOT" -czf "$archive_tmp" "${archive_paths[@]}"
tar_rc=$?
if [ "$tar_rc" -ne 0 ]; then
  record_status archive "$tar_rc"
else
  if [ "$run_status" -eq 0 ]; then
    mv "$archive_tmp" "$RETURN_DIR/part64-return.tgz"
    mv_rc=$?
    if [ "$mv_rc" -ne 0 ]; then record_status archive-rename "$mv_rc"; fi
  else
    mv "$archive_tmp" "$RETURN_DIR/part64-return-INCOMPLETE.tgz"
    mv_rc=$?
    if [ "$mv_rc" -ne 0 ]; then record_status incomplete-archive-rename "$mv_rc"; fi
  fi
fi

if [ "$run_status" -eq 0 ]; then
  echo "RETURN_READY"
else
  printf 'RUN_FAILED failing_phases='
  if [ "${#failures[@]}" -gt 0 ]; then
    printf '%s' "${failures[0]}"
    for ((index = 1; index < ${#failures[@]}; index++)); do printf ',%s' "${failures[$index]}"; done
  else
    printf 'unknown'
  fi
  printf '\n'
fi
echo "run_status=$run_status"
exit "$run_status"
