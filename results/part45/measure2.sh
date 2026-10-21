set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A="--device cuda --dtype bf16 --cache-dir /root/hf-cache --out results/part45"
echo "$(date -u +%FT%TZ) patch: part45.py command_measure wraps _measure_rows in torch.no_grad() (the first full measure ran with autograd on: CUDA OOM, then numpy() on a grad tensor); batch_size back to 8"
echo "$(date -u +%FT%TZ) start measure"; python part45.py measure $A
echo "$(date -u +%FT%TZ) start score"; python part45.py score $A
echo "$(date -u +%FT%TZ) FULL DONE"
