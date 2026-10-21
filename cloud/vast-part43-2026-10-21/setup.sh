set -e
mkdir -p /root/p43 /root/hf/hub && cd /root/p43 && tar xzf /root/p43.tgz
pip -q install transformers==5.20.0 flash-linear-attention==0.5.2 accelerate 2>&1 | grep -v WARNING | tail -2
python -c "
from huggingface_hub import snapshot_download
p=snapshot_download('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', cache_dir='/root/hf/hub')
print('model', p)"
python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version(),'transformers',transformers.__version__,torch.cuda.get_device_name(0))" > /root/p43/versions.txt
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader >> /root/p43/versions.txt
echo SETUP_OK
