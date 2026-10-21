cd /root/work
A="--device cuda --dtype bf16 --cache-dir /root/hf-cache --out results/part43"
echo "$(date -u +%FT%TZ) start manifest"; python -u part43.py manifest $A; echo "manifest rc=$?"
echo "$(date -u +%FT%TZ) start pilot"; python -u part43.py pilot $A --candidate-pool-size 1024; echo "pilot rc=$?"
echo "$(date -u +%FT%TZ) P43 DONE"
