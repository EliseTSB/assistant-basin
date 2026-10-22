#!/bin/bash
# Part 54 unattended driver: built from RUNBOOK-54.md blocks 1-7 (2026-10-22). No set -e: a failed model (e.g. gemma 12B OOM) must not stop the rest.
cd /root/part54 2>/dev/null || true
date +%s > /root/part54.start

echo "=== block 1 $(date +%H:%M:%S)"
mkdir -p /root/part54 /root/hf
tar -xzf /root/part54-inputs.tgz -C /root/part54
cd /root/part54
python -m pip install transformers==5.20.0 flash-linear-attention==0.5.2 accelerate huggingface_hub
python -m pip uninstall -y causal-conv1d || true
export HF_HOME=/root/hf
nvidia-smi
python -c "import torch, transformers; print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -m pip freeze > versions-pip.txt

echo "=== block 2 $(date +%H:%M:%S)"
fetch_model() {
  HF_HUB_OFFLINE=0 python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir='/root/hf/hub'))
PY
}
discard_model() {
  repo_cache=${1//\//--}
  rm -rf "/root/hf/hub/models--$repo_cache"
}

echo "=== block 3 $(date +%H:%M:%S)"
prefetch() { ( fetch_model "$1" "$2" > "/root/prefetch-${1//\//_}.log" 2>&1 & ) ; }
fetch_model Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
prefetch allenai/OLMo-2-1124-7B-Instruct 470b1fba1ae01581f270116362ee4aa1b97f4c84
mkdir -p results/part54a
HF_HUB_OFFLINE=1 python -u part26.py ladder --model qwen35-4b --clamp-unit 22:2742 --alien-frames results/part33-r2/natural-firing.json --source-record-dir results/part26 --corpus qwen35/pile10k.jsonl --out results/part54a --device cuda --samples 8 --max-new 64 --control-count 3 --seed 0 > results/part54a/run.log 2>&1
discard_model Qwen/Qwen3.5-4B

echo "=== block 4 $(date +%H:%M:%S)"
mkdir -p results/part54b
cp results/part26/source.json results/part26/source.npz results/part54b/
cp results/part26/target-smollm3-3b.json results/part26/target-smollm3-3b.npz results/part26/battery-smollm3-3b.json results/part54b/
mkdir -p results/part26-keep
cp results/part26/clamp-smollm3-3b.json results/part26-keep/

echo "=== block 5 $(date +%H:%M:%S)"
run_search_model() {
  repo_id=$1
  revision=$2
  short_name=$3
  fetch_model "$repo_id" "$revision"   # returns at once when the background prefetch already finished it
  HF_HUB_OFFLINE=1 python -u part26.py --steps target,battery --targets "$short_name" --corpus qwen35/pile10k.jsonl --out results/part54b --device cuda --batch-size 4 --write-chunk 256 > "results/part54b/$short_name.log" 2>&1
  discard_model "$repo_id"
}
prefetch Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c
run_search_model allenai/OLMo-2-1124-7B-Instruct 470b1fba1ae01581f270116362ee4aa1b97f4c84 olmo2-7b
prefetch microsoft/Phi-4-mini-instruct cfbefacb99257ffa30c83adab238a50856ac3083
run_search_model Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c qwen3-4b
prefetch ibm-granite/granite-3.3-2b-instruct 707f574c62054322f6b5b04b6d075f0a8f05e0f0
run_search_model microsoft/Phi-4-mini-instruct cfbefacb99257ffa30c83adab238a50856ac3083 phi4-mini
prefetch google/gemma-4-E2B-it 3e22461f65e89153144f8adb70e3b8c2cc9845a7
run_search_model ibm-granite/granite-3.3-2b-instruct 707f574c62054322f6b5b04b6d075f0a8f05e0f0 granite3-2b
prefetch google/gemma-4-E4B-it ee0ef6023621cff504d758262d4e04895a5af4a2
run_search_model google/gemma-4-E2B-it 3e22461f65e89153144f8adb70e3b8c2cc9845a7 gemma4-e2b-it
run_search_model google/gemma-4-E4B-it ee0ef6023621cff504d758262d4e04895a5af4a2 gemma4-e4b-it
run_search_model google/gemma-4-12B-it 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 gemma4-12b-it

echo "=== block 6 $(date +%H:%M:%S)"
python - <<'PY' > results/part54b/twins.tsv
import glob, json, part26
for path in sorted(glob.glob('results/part54b/battery-*.json')):
    data = json.load(open(path, encoding='utf-8'))
    if data['short_name'] not in {'olmo2-7b', 'qwen3-4b', 'phi4-mini', 'granite3-2b',
                                  'gemma4-e2b-it', 'gemma4-e4b-it', 'gemma4-12b-it'}:
        continue
    for row in data['candidates']:
        if part26.passes_part26_twin_checks(row):
            print(data['short_name'], f"{row['layer']}:{row['unit']}", data['model'], data['revision'], sep='\t')
PY
cat results/part54b/twins.tsv

echo "=== block 7 $(date +%H:%M:%S)"
run_ladder_model() {
  short_name=$1
  unit=$2
  repo_id=$3
  revision=$4
  fetch_model "$repo_id" "$revision"
  HF_HUB_OFFLINE=1 python -u part26.py ladder --model "$short_name" --revision "$revision" --clamp-unit "$unit" --out results/part54b --device cuda --samples 8 --max-new 64 --control-count 3 --seed 0 > "results/part54b/ladder-$short_name-$unit.log" 2>&1
  discard_model "$repo_id"
}
run_ladder_model smollm3-3b 23:1019 HuggingFaceTB/SmolLM3-3B a07cc9a04f16550a088caea529712d1d335b0ac1
while IFS=$'\t' read -r short_name unit repo_id revision; do
  run_ladder_model "$short_name" "$unit" "$repo_id" "$revision"
done < results/part54b/twins.tsv

echo "=== done $(date +%H:%M:%S)"; date +%s > /root/part54.end; tar -C /root/part54 -czf /root/part54-return.tgz results/part54a results/part54b versions-pip.txt; echo RETURN_READY
