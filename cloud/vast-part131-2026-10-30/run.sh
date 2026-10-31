#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python -m pip install "transformers==5.20.0" datasets accelerate safetensors "brotli>=1.2"
mkdir -p results/part131
python -m pip freeze > results/part131/pip-freeze.txt
python - <<'PY'
import torch, transformers
if transformers.__version__ != '5.20.0': raise SystemExit(f"unexpected transformers {transformers.__version__}")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise SystemExit('CUDA bf16 is unavailable')
if '3090' not in torch.cuda.get_device_name(0): raise SystemExit('expected RTX 3090')
print('Pilot host:', torch.cuda.get_device_name(0), 'torch', torch.__version__, 'transformers', transformers.__version__, flush=True)
PY
download_run() {
  local tag="$1" repo="$2" revision="$3" frame="$4"
  local model="model/${tag}-${revision}"
  MODEL_DIR="$model" MODEL_REPO="$repo" MODEL_REVISION="$revision" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ['MODEL_REPO'], revision=os.environ['MODEL_REVISION'], local_dir=os.environ['MODEL_DIR'])
PY
  python -I -B part131/run131.py --stage pilot --tag "$tag" --model "$model" --revision "$revision" --frame "$frame" --out results/part131/pilot --batch 32
  rm -rf "$model"
}
download_run base unsloth/Meta-Llama-3.1-8B e9a141a2 plain
download_run tulu-final allenai/Llama-3.1-Tulu-3-8B 66694379 chat
echo 'Pilot complete. Review results/part131/pilot before starting run-main.sh.'
