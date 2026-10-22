#!/bin/bash
# Part 59 unattended runner; upload the registered input tarball first.
cd /root/part59 || exit 2
mkdir -p /root/hf /root/part59/results/part59a /root/part59/results/part59b /root/part59/logs
date +%s > /root/part59/run59.start

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
  "$@" > "/root/part59/logs/$phase.log" 2>&1
  local rc=$?
  record_status "$phase" "$rc"
  return 0
}

skip_phase() {
  local phase=$1 reason=$2
  printf 'skipped: %s\n' "$reason" | tee "/root/part59/logs/$phase.log"
  record_status "$phase" 125
  return 0
}

run_phase() {
  local phase=$1
  local dependency=$2
  shift 2
  if [ "${phase_status[$dependency]:-125}" -ne 0 ]; then
    skip_phase "$phase" "dependency $dependency did not exit 0"
    return 0
  fi
  run_checked "$phase" "$@"
}

required_files=(
  part59/run59a.py part59/run59b.py part59/analyze59b.py part59/blind_setting.py
  part59/test_part59.py part59/prompts59a.json part26.py part20.py part20c.py part20d.py
  part20f.py neuron_report.py judge_client.py
  results/part26/source.json results/part26/source.npz
  results/part26/target-olmo2-1b.npz results/part26/target-olmo2-1b.json
  results/part54a/source-pile-maxima-7c19d0120476.json
)
missing_files=()
for path in "${required_files[@]}"; do
  if [ ! -f "/root/part59/$path" ]; then
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
    python -m pip uninstall -y causal-conv1d > /root/part59/logs/optional-uninstall.log 2>&1 || true
    run_checked package-version-check python -c 'import importlib.metadata as m; names=("torch", "transformers", "flash-linear-attention", "accelerate", "huggingface_hub"); versions={name:m.version(name) for name in names}; print("\n".join(f"{name}=={version}" for name, version in versions.items())); assert versions["transformers"] == "5.20.0", versions["transformers"]; assert versions["flash-linear-attention"] == "0.5.2", versions["flash-linear-attention"]'
  else
    skip_phase package-version-check "package-install did not exit 0"
  fi
  if [ "${phase_status[package-version-check]}" -eq 0 ]; then
    run_checked package-version-record bash -c 'python -m pip freeze > /root/part59/versions-pip.txt'
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

qwen_prefetch_rc=125
olmo_prefetch_rc=125
if [ "${phase_status[python-inventory]:-125}" -eq 0 ] && [ "$run_status" -eq 0 ]; then
  run_checked prefetch-qwen fetch_model Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  qwen_prefetch_rc=$last_phase_rc
else
  skip_phase prefetch-qwen "setup did not complete successfully"
fi
if [ "$run_status" -eq 0 ]; then
  run_checked prefetch-olmo fetch_model allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e
  olmo_prefetch_rc=$last_phase_rc
else
  skip_phase prefetch-olmo "run_status is nonzero after prefetch-qwen or setup"
fi
export HF_HOME=/root/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ "$qwen_prefetch_rc" -eq 0 ]; then
  run_checked qwen59a python -u part59/run59a.py --model qwen35-4b --clamp-unit 22:2742 --device cuda --seed 0 --out results/part59a
else
  skip_phase qwen59a "prefetch-qwen did not exit 0"
fi
if [ "$olmo_prefetch_rc" -eq 0 ]; then
  run_checked olmo59a python -u part59/run59a.py --model olmo2-1b --clamp-unit 10:6978 --device cuda --seed 0 --out results/part59a
else
  skip_phase olmo59a "prefetch-olmo did not exit 0"
fi
if [ "$qwen_prefetch_rc" -eq 0 ]; then
  run_checked qwen59b python -u part59/run59b.py --model qwen35-4b --clamp-unit 22:2742 --device cuda --batch-size 1 --seed 0 --out results/part59b
  run_phase qwen59b-analysis qwen59b python -u part59/analyze59b.py results/part59b/59b-qwen35-4b-L22u2742.npz --device cuda
else
  skip_phase qwen59b "prefetch-qwen did not exit 0"
  skip_phase qwen59b-analysis "qwen59b did not run because prefetch-qwen failed"
fi
if [ "$olmo_prefetch_rc" -eq 0 ]; then
  run_checked olmo59b python -u part59/run59b.py --model olmo2-1b --clamp-unit 10:6978 --device cuda --batch-size 1 --seed 0 --out results/part59b
  run_phase olmo59b-analysis olmo59b python -u part59/analyze59b.py results/part59b/59b-olmo2-1b-L10u6978.npz --device cuda
else
  skip_phase olmo59b "prefetch-olmo did not exit 0"
  skip_phase olmo59b-analysis "olmo59b did not run because prefetch-olmo failed"
fi

echo "=== archive $(date +%H:%M:%S)"
date +%s > /root/part59/run59.end
archive_tmp=/root/part59-return.tmp.tgz
archive_paths=(results/part59a results/part59b logs versions-pip.txt run59.start run59.end)
tar -C /root/part59 -czf "$archive_tmp" "${archive_paths[@]}"
tar_rc=$?
if [ "$tar_rc" -ne 0 ]; then
  record_status archive "$tar_rc"
else
  if [ "$run_status" -eq 0 ]; then
    mv "$archive_tmp" /root/part59-return.tgz
    mv_rc=$?
    if [ "$mv_rc" -ne 0 ]; then record_status archive-rename "$mv_rc"; fi
  else
    mv "$archive_tmp" /root/part59-return-INCOMPLETE.tgz
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
