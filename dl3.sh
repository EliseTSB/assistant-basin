#!/bin/bash
# dl3.sh — download the OLMo 3 7B ladder to E: one at a time (Claude fork, 2026-10-16).
cd /home/elisepc/neuron-probe
for R in allenai/Olmo-3-1025-7B allenai/Olmo-3-7B-Instruct-SFT allenai/Olmo-3-7B-Instruct-DPO allenai/Olmo-3-7B-Instruct allenai/Olmo-3-7B-RL-Zero-General; do
  env -u HTTPS_PROXY -u HTTP_PROXY -u SSL_CERT_FILE HF_HUB_DISABLE_XET=1 HF_HOME=/mnt/e/neuron-probe/hf \
    .venv/bin/python -I -c "from huggingface_hub import snapshot_download as s; s('$R', allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*'])" \
    && echo "$(date +%H:%M:%S) $R downloaded" >> dl3.log || echo "$(date +%H:%M:%S) $R FAILED" >> dl3.log
done
echo "$(date +%H:%M:%S) dl3 done" >> dl3.log
