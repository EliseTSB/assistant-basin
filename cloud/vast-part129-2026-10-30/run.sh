#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NL_COMMIT="07fdf45"
CHAT_REV="b968826d9c46dd6066d109eabc6255188de91218"
BASE_REV="49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
CHAT_DIR="$ROOT/models/qwen3-8b-$CHAT_REV"
BASE_DIR="$ROOT/models/qwen3-8b-base-$BASE_REV"
NL_ROOT="$ROOT/neuron-lens"
OUT="$ROOT/results/part129"
mkdir -p "$OUT"
cd "$ROOT"
CORPUS_DOCS="$(PYTHONPATH="$ROOT" python -B -c 'from part129.config129 import CORPUS_DOCS; print(",".join(f"{name}={count}" for name,count in CORPUS_DOCS.items()))')"
python -m pip install "transformers==5.20.0" "datasets" "pyarrow" "accelerate" "safetensors"
mkdir -p results/part129 && python -m pip freeze > results/part129/pip-freeze.txt || true
python - <<'PY'
import torch, transformers
if transformers.__version__ != "5.20.0": raise SystemExit(f"expected transformers 5.20.0, got {transformers.__version__}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 2: raise SystemExit("two CUDA devices are required")
for i in (0,1):
    name=torch.cuda.get_device_name(i)
    if "3090" not in name: raise SystemExit(f"expected RTX 3090 at cuda:{i}, found {name}")
    if not torch.cuda.is_bf16_supported(): raise SystemExit("CUDA bf16 is not supported")
    print(f"cuda:{i}: {name}", flush=True)
PY
CHAT_DIR="$CHAT_DIR" BASE_DIR="$BASE_DIR" python - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo, rev, path in (("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", os.environ["CHAT_DIR"]),
                        ("Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4", os.environ["BASE_DIR"])):
    snapshot_download(repo_id=repo, revision=rev, local_dir=path)
PY
if [ ! -d "$NL_ROOT/.git" ]; then
  git clone https://github.com/stdlogits/neuron-lens.git "$NL_ROOT"
  git -C "$NL_ROOT" checkout "$NL_COMMIT"
fi
test "$(git -C "$NL_ROOT" rev-parse --short HEAD)" = "$NL_COMMIT"
mkdir -p "$NL_ROOT/data/corpus"
CHAT_DIR="$CHAT_DIR" NL_ROOT="$NL_ROOT" python - <<'PY'
import json, os
from huggingface_hub import HfApi
repo=os.environ["NL_ROOT"]
rev=HfApi().dataset_info("HuggingFaceFW/dclm_100BT-shuffled").sha
open(repo+"/data/corpus/dclm.provenance.json","w",encoding="utf-8").write(json.dumps({"revision":rev})+"\n")
PY
cd "$NL_ROOT"
PYTHONPATH="$NL_ROOT" NEURON_LENS_MODEL="$CHAT_DIR" python scripts/05_act_corpus.py --docs "$CORPUS_DOCS"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CORPUS=("$NL_ROOT/data/corpus/act-dclm.jsonl" "$NL_ROOT/data/corpus/act-zh.jsonl" "$NL_ROOT/data/corpus/act-code.jsonl" "$NL_ROOT/data/corpus/act-chat-qwen3-8b.jsonl")
for batch in 16 32 64; do
  python -I -B part129/collect129.py --stage pilot --batch "$batch" --chat-model "$CHAT_DIR" --base-model "$BASE_DIR" --corpus "${CORPUS[@]}" --prepared "$OUT/prepared129.pt" --out "$OUT/pilot-b$batch"
done
for batch in 16 64; do
  python -I -B part129/collect129.py --stage pilot --batch "$batch" --chat-model "$CHAT_DIR" --base-model "$CHAT_DIR" --corpus "${CORPUS[@]}" --prepared "$OUT/prepared129.pt" --out "$OUT/gate2-b$batch"
done
printf 'Pilot timing captured. Main stage not started.\n'
