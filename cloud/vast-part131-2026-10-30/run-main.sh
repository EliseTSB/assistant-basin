#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p results/part131
python -m pip freeze > results/part131/pip-freeze.txt
download_run() {
  local tag="$1" repo="$2" revision="$3" frame="$4"
  local model="model/${tag}-${revision}"
  MODEL_DIR="$model" MODEL_REPO="$repo" MODEL_REVISION="$revision" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ['MODEL_REPO'], revision=os.environ['MODEL_REVISION'], local_dir=os.environ['MODEL_DIR'])
PY
  python -I -B part131/run131.py --stage main --tag "$tag" --model "$model" --revision "$revision" --frame "$frame" --out results/part131/main --batch 32
  rm -rf "$model"
}
download_run base unsloth/Meta-Llama-3.1-8B e9a141a2 plain
download_run instruct unsloth/Llama-3.1-8B-Instruct 4699cc75 chat
download_run tulu-sft allenai/Llama-3.1-Tulu-3-8B-SFT f2a0b46b chat
download_run tulu-dpo allenai/Llama-3.1-Tulu-3-8B-DPO a7beb67e chat
download_run tulu-final allenai/Llama-3.1-Tulu-3-8B 66694379 chat
download_run hermes NousResearch/Hermes-3-Llama-3.1-8B 896ea440 chat
echo 'Main stage complete.'
