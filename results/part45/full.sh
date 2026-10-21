set -e
A="--device cuda --dtype bf16 --cache-dir /root/hf-cache --out results/part45"
echo "$(date -u +%FT%TZ) start full"; python part45.py run --full $A
echo "$(date -u +%FT%TZ) start measure"; python part45.py measure $A
echo "$(date -u +%FT%TZ) start score"; python part45.py score $A
echo "$(date -u +%FT%TZ) FULL DONE"
