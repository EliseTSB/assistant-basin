set -e
cd /root && mkdir -p work infer && tar xzf part43-bundle.tgz -C work && tar xzf infer-bundle.tgz -C infer
cd work && sha256sum -c /root/bundle.sha256
pip install -q "transformers==5.20.0" "flash-linear-attention==0.5.2" accelerate safetensors huggingface_hub 2>&1 | tail -2
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', cache_dir='/root/hf-cache')
snapshot_download('Qwen/Qwen3-1.7B', cache_dir='/root/hf-cache')
print('models ok')
PY
python -c "import transformers,torch,fla;print('versions',transformers.__version__,torch.__version__,fla.__version__)"
echo SETUP DONE
