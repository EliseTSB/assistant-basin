#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_REVISION="48d788eca847d4d7548f375ad03d3c9312f6139e"
MODEL_REPO="allenai/OLMo-2-0425-1B-Instruct"
MODEL_DIR="$ROOT/model/olmo-2-0425-1b-instruct-$MODEL_REVISION"

cd "$ROOT"
python -m pip install "transformers==5.13.0" accelerate huggingface_hub safetensors

python - <<'PY'
import torch
import transformers

if transformers.__version__ != "5.13.0":
    raise SystemExit(f"expected transformers 5.13.0, found {transformers.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA bf16 is not supported on this instance")
name = torch.cuda.get_device_name(0)
if "3090" not in name:
    raise SystemExit(f"expected an RTX 3090, found {name}")
print(f"Using {name}, CUDA bf16, transformers {transformers.__version__}", flush=True)
PY

MODEL_DIR="$MODEL_DIR" MODEL_REPO="$MODEL_REPO" MODEL_REVISION="$MODEL_REVISION" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(repo_id=os.environ["MODEL_REPO"],
                  revision=os.environ["MODEL_REVISION"],
                  local_dir=os.environ["MODEL_DIR"])
PY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +e
rc127=0; rc125=0
python -I -B part127/run127.py \
  --stage pilot --model "$MODEL_DIR" --out results/part127/pilot-b8 \
  --batch 8 --device cuda \
&& python -I -B part127/run127.py \
  --stage main --model "$MODEL_DIR" --out results/part127/main \
  --pilot-dir results/part127/pilot-b8 --batch 8 --device cuda
rc127=$?
echo "PART127 rc=$rc127"

# Part 125, Amendment 2: pilot, then the main stage its own projection allows.
python -I -B part125/run125.py --stage pilot --model "$MODEL_DIR" \
  --out results/part125/main --device cuda --max-new-tokens 120
rc125=$?
if [ $rc125 -eq 0 ]; then
  python - <<'PY'
import json
d = json.load(open("results/part125/main/pilot-meta.json"))["pilot_size_decision"]
print("PART125 pilot projection minutes", d["projected_minutes"], "over 90:", d["over_90_minutes"], flush=True)
raise SystemExit(3 if d["over_90_minutes"] else 0)
PY
  over=$?
  if [ $over -eq 3 ]; then
    python -I -B part125/run125.py --stage main --model "$MODEL_DIR" \
      --out results/part125/main --device cuda --max-new-tokens 120 --slot-only
  else
    python -I -B part125/run125.py --stage main --model "$MODEL_DIR" \
      --out results/part125/main --device cuda --max-new-tokens 120
  fi
  rc125=$?
fi
echo "PART125 rc=$rc125"
[ $rc127 -eq 0 ] && [ $rc125 -eq 0 ]
