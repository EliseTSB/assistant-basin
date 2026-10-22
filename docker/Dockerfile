# GPU image for this repo's experiments on a rented Vast.ai RTX 3090.
# Reproduces the environment RUNBOOK-54.md installs by hand. No model weights are baked in.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

LABEL org.opencontainers.image.source=https://github.com/elisetsb/assistant-basin

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates tmux \
 && rm -rf /var/lib/apt/lists/*

# Pins from RUNBOOK-54.md. torch stays the image's 2.6.0+cu124 (no torch reinstall).
# The extra unpinned packages (safetensors, psutil, peft, pyarrow, pytest) are imported by some scripts and tests.
RUN python -m pip install \
        transformers==5.20.0 \
        flash-linear-attention==0.5.2 \
        accelerate \
        huggingface_hub \
        numpy \
        safetensors \
        psutil \
        peft \
        pyarrow \
        pytest \
        "git+https://github.com/anthropics/jacobian-lens" \
 && (python -m pip uninstall -y causal-conv1d || true) \
 && python -c "import torch, transformers; assert torch.__version__.startswith('2.6.0'), torch.__version__; assert transformers.__version__ == '5.20.0', transformers.__version__" \
 && ! python -m pip show causal-conv1d >/dev/null 2>&1

ARG REPO_URL=https://github.com/elisetsb/assistant-basin.git
ARG REPO_REF=main
RUN git clone "$REPO_URL" /opt/assistant-basin \
 && git -C /opt/assistant-basin checkout "$REPO_REF" \
 && mkdir -p "$HF_HOME"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /opt/assistant-basin
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
