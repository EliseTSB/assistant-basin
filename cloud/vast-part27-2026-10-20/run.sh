date +%s > run.start
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u part27.py --steps costs,base --corpus qwen35/pile10k.jsonl --out results/part27 --device cuda --max-gpu-gb 22 --batch-size 2 > run.log 2>&1
echo rc=$? >> run.log
date +%s > run.end
