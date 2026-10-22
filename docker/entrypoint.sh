#!/usr/bin/env bash
# Print the stack, refresh the repo if the network allows, then run the given command or idle.
set -u

python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available(), 'transformers', transformers.__version__)" \
  || echo "entrypoint: could not import torch/transformers"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "entrypoint: nvidia-smi failed"
else
  echo "entrypoint: nvidia-smi not found"
fi

REPO=/opt/assistant-basin
if [ -d "$REPO/.git" ]; then
  if timeout 60 git -C "$REPO" pull --ff-only; then
    echo "entrypoint: repo at $(git -C "$REPO" rev-parse --short HEAD)"
  else
    echo "entrypoint: git pull skipped or failed; using baked code at $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
  fi
fi

if [ "$#" -gt 0 ]; then
  exec "$@"
fi
exec sleep infinity
