#!/bin/bash
# Part 209 pilot step 1b: OLMo 2 1B Instruct R2 timing on the CPU, batch 16 then batch 32, one at a time.
cd /home/elisepc/green/worktrees/ab-part203 || exit 1
P='env HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part209/run209.py --stage instruct --pilot --r2'
for B in 16 32; do
  OUT=$(python3 -I -B tools/launch_arc.py 209 --cpu "$P --batch $B --out results/part209/pilot/olmo-b$B --models-json results/part209/pilot/models.json" --mem 8G --tag olmo-b$B 2>&1)
  U=$(echo "$OUT" | grep -oE 'unit part209-olmo-b[0-9]+-[0-9]+' | head -1 | cut -d' ' -f2)
  echo "$(date +%H:%M:%S) batch $B unit=$U"
  [ -z "$U" ] && { echo "$OUT" | tail -5; exit 2; }
  while systemctl --user is-active --quiet "$U"; do sleep 20; done
  echo "$(date +%H:%M:%S) batch $B ended: $(systemctl --user show -p Result --value "$U" 2>/dev/null)"
  tail -2 results/part209/olmo-b$B.log
done
echo ALL-DONE
