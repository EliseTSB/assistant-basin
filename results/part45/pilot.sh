set -e
A="--device cuda --dtype bf16 --cache-dir /root/hf-cache --out results/part45"
echo "$(date -u +%FT%TZ) start dirs"; python part45.py dirs $A
echo "$(date -u +%FT%TZ) start pilot"; python part45.py run --pilot $A
echo "$(date -u +%FT%TZ) start measure"; python part45.py measure $A
echo "$(date -u +%FT%TZ) start score"; python part45.py score $A
mkdir -p results/part45-pilot && cp results/part45/raw.jsonl results/part45/measure.json results/part45/score.json results/part45/score.md results/part45-pilot/
echo "$(date -u +%FT%TZ) PILOT DONE"
