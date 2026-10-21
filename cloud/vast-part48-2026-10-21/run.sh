date +%s > run.start
python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version(),'transformers',transformers.__version__,'tf32_matmul',torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn',torch.backends.cudnn.allow_tf32,'bf16_reduced',torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'fp16_reduced',torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction);print(torch.cuda.get_device_name(0))" > versions.txt 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> versions.txt 2>&1
python -u part48.py --models pythia-410m,pythia-1.4b,olmo2-1b,qwen3-1.7b-base,smollm2-1.7b --device cuda --dtype fp32 --out results/part48-3090 > run.log 2>&1
echo rc=$? >> run.log
date +%s > run.end
