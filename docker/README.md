# GPU container

`ghcr.io/elisetsb/assistant-basin-gpu:latest` is the environment from RUNBOOK-54.md, built by
`.github/workflows/container.yml` on GitHub Actions. Each build is also tagged with its git commit sha.

## Launch on Vast.ai

1. Create a template (or edit the instance config) with image `ghcr.io/elisetsb/assistant-basin-gpu:latest`,
   no registry login (the package must be public), launch mode **SSH** with **direct SSH**, and 40 GB disk.
2. Rent one RTX 3090 (24 GB) with that template.
3. SSH to the instance's public IP and mapped port: `ssh -i ~/.ssh/vast_ed25519 -p <SSH_PORT> root@<PUBLIC_IP>`.
4. In SSH mode Vast runs its own startup instead of the image entrypoint, so start with
   `/usr/local/bin/entrypoint.sh true` (prints torch/CUDA/Transformers versions and `nvidia-smi`, pulls the code),
   then `cd /opt/assistant-basin`.

## What is inside

- Base `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` (torch 2.6.0+cu124, CUDA 12.4, cuDNN 9, Python 3.11).
- `transformers==5.20.0`, `flash-linear-attention==0.5.2`, `accelerate`, `huggingface_hub`, `numpy`, the
  `jlens` package from github.com/anthropics/jacobian-lens, plus `safetensors`, `psutil`, `peft`, `pyarrow`
  and `pytest`, which some scripts and tests import.
- No pip `causal-conv1d` (ABI mismatch with this torch image; the build fails if it is present).
- apt: `git`, `curl`, `ca-certificates`, `tmux`.
- This repository cloned at `/opt/assistant-basin` (build arg `REPO_REF`, default `main`), the working directory.
- `HF_HOME=/root/hf`. No model weights, data from outside the repo or credentials are in the image.

## Refresh code and download a model

Code: `cd /opt/assistant-basin && git pull --ff-only` (the entrypoint does this too when the network allows).
Files that are not in git, such as `qwen35/pile10k.jsonl`, still have to be copied up with `scp` as in RUNBOOK-54.md.

Models: fetch one pinned public snapshot at a time, using the revision from the `MODELS` table in `part26.py`,
then run with `HF_HUB_OFFLINE=1`:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='Qwen/Qwen3.5-4B',
                        revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                        cache_dir='/root/hf/hub'))
PY
rm -rf /root/hf/hub/models--Qwen--Qwen3.5-4B   # when done, to keep the 40 GB disk free
```
