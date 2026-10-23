#!/bin/bash
# Part 61 unattended runner. Copy the return archive before destroying the rented box.
set -u

RUN_ROOT="${PART61_ROOT:-/root/part61}"
RETURN_DIR="${PART61_RETURN_DIR:-/root}"
cd "$RUN_ROOT" || exit 2
mkdir -p "$RUN_ROOT/results/part61" "$RUN_ROOT/logs" "$RETURN_DIR"
touch "$RUN_ROOT/versions-pip.txt"
date +%s > "$RUN_ROOT/run61.start"

run_status=0
failures=()
declare -A phase_status=()
last_phase_rc=0

record_status() {
  local phase=$1 rc=$2
  echo "$phase exit=$rc"
  phase_status[$phase]=$rc
  last_phase_rc=$rc
  if [ "$rc" -ne 0 ]; then
    run_status=1
    failures+=("$phase")
  fi
}

run_checked() {
  local phase=$1
  shift
  echo "=== $phase $(date +%H:%M:%S)"
  "$@" > "$RUN_ROOT/logs/$phase.log" 2>&1
  local rc=$?
  record_status "$phase" "$rc"
  return 0
}

skip_phase() {
  local phase=$1 reason=$2
  printf 'skipped: %s\n' "$reason" | tee "$RUN_ROOT/logs/$phase.log"
  record_status "$phase" 125
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
  part61/run61.py part61/analyze61.py part61/run61.sh part61/RUNBOOK-61.md part61/test_part61.py
  part26.py part20.py part20c.py part20d.py part20f.py neuron_report.py
  results/part26/source.json results/part26/source.npz
  results/part26/target-olmo2-1b.npz results/part26/target-olmo2-1b.json
  results/part54a/source-pile-maxima-7c19d0120476.json
)
missing_files=()
for path in "${required_files[@]}"; do
  if [ ! -f "$RUN_ROOT/$path" ]; then
    missing_files+=("$path")
  fi
done
if [ "${#missing_files[@]}" -gt 0 ]; then
  printf 'missing required input: %s\n' "${missing_files[@]}"
  record_status preflight-files 1
else
  record_status preflight-files 0
fi

if [ "${phase_status[preflight-files]}" -eq 0 ]; then
  run_checked package-install python -m pip install transformers==5.20.0 flash-linear-attention==0.5.2 accelerate huggingface_hub
  if [ "${phase_status[package-install]}" -eq 0 ]; then
    python -m pip uninstall -y causal-conv1d > "$RUN_ROOT/logs/optional-uninstall.log" 2>&1 || true
    run_checked package-version-check python -c 'import importlib.metadata as m; names=("torch", "transformers", "flash-linear-attention", "accelerate", "huggingface_hub"); versions={name:m.version(name) for name in names}; print("\n".join(f"{name}=={version}" for name, version in versions.items())); assert versions["transformers"] == "5.20.0", versions["transformers"]; assert versions["flash-linear-attention"] == "0.5.2", versions["flash-linear-attention"]'
  else
    skip_phase package-version-check "package-install did not exit 0"
  fi
  if [ "${phase_status[package-version-check]}" -eq 0 ]; then
    run_checked package-version-record bash -c 'python -m pip freeze > "$1/versions-pip.txt"' _ "$RUN_ROOT"
  else
    skip_phase package-version-record "package-version-check did not exit 0"
  fi
else
  skip_phase package-install "preflight-files did not exit 0"
  skip_phase package-version-check "preflight-files did not exit 0"
  skip_phase package-version-record "preflight-files did not exit 0"
fi

fetch_model() {
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir='/root/hf/hub'))
PY
}

if [ "${phase_status[package-version-record]:-125}" -eq 0 ]; then
  run_checked gpu-inventory nvidia-smi
  run_checked gpu-inventory-query nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  run_checked python-inventory python -c 'import torch, transformers; print("torch",torch.__version__,"cuda",torch.version.cuda,"transformers",transformers.__version__,torch.cuda.get_device_name(0))'
else
  skip_phase gpu-inventory "package-version-record did not exit 0"
  skip_phase gpu-inventory-query "package-version-record did not exit 0"
  skip_phase python-inventory "package-version-record did not exit 0"
fi

if [ "${phase_status[python-inventory]:-125}" -eq 0 ]; then
  run_checked prefetch-qwen fetch_model Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  run_checked prefetch-olmo fetch_model allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e
else
  skip_phase prefetch-qwen "package or GPU inventory setup did not complete"
  skip_phase prefetch-olmo "package or GPU inventory setup did not complete"
fi
export HF_HOME=/root/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ "${phase_status[prefetch-qwen]:-125}" -eq 0 ]; then
  run_checked qwen61 python -u part61/run61.py --model qwen35-4b --clamp-unit 22:2742 --device cuda --batch-size 1 --out results/part61
  run_phase qwen61-analysis qwen61 python -u part61/analyze61.py results/part61/61-qwen35-4b-L22u2742.npz
else
  skip_phase qwen61 "prefetch-qwen did not exit 0"
  skip_phase qwen61-analysis "qwen61 did not run because prefetch-qwen failed"
fi
if [ "${phase_status[prefetch-olmo]:-125}" -eq 0 ]; then
  run_checked olmo61 python -u part61/run61.py --model olmo2-1b --clamp-unit 10:6978 --device cuda --batch-size 1 --out results/part61
  run_phase olmo61-analysis olmo61 python -u part61/analyze61.py results/part61/61-olmo2-1b-L10u6978.npz
else
  skip_phase olmo61 "prefetch-olmo did not exit 0"
  skip_phase olmo61-analysis "olmo61 did not run because prefetch-olmo failed"
fi
if [ "${phase_status[qwen61-analysis]:-125}" -eq 0 ] && [ "${phase_status[olmo61-analysis]:-125}" -eq 0 ]; then
  run_checked part61-evaluation python -u part61/analyze61.py --combined \
    results/part61/61-qwen35-4b-L22u2742.npz results/part61/61-olmo2-1b-L10u6978.npz
else
  skip_phase part61-evaluation "both model analyses must exit 0"
fi

echo "=== archive $(date +%H:%M:%S)"
date +%s > "$RUN_ROOT/run61.end"
archive_tmp="$RETURN_DIR/part61-return.tmp.tgz"
archive_paths=(results/part61 logs versions-pip.txt run61.start run61.end)
tar -C "$RUN_ROOT" -czf "$archive_tmp" "${archive_paths[@]}"
tar_rc=$?
if [ "$tar_rc" -ne 0 ]; then
  record_status archive "$tar_rc"
else
  if [ "$run_status" -eq 0 ]; then
    mv "$archive_tmp" "$RETURN_DIR/part61-return.tgz"
    mv_rc=$?
    if [ "$mv_rc" -ne 0 ]; then record_status archive-rename "$mv_rc"; fi
  else
    mv "$archive_tmp" "$RETURN_DIR/part61-return-INCOMPLETE.tgz"
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
