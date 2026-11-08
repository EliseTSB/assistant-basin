#!/bin/bash
# Part 198 pilot, part 2: AlphaEdit, MEMIT-b10, AlphaEdit-b10, after Windows has >= 11 GB free (gotcha: wait for 11 GB).
cd "$HOME/edit-detect/part198"
for i in $(seq 1 120); do
  sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  free_gb=$(/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" 2>/dev/null | tr -d '\r')
  echo "$(date +%T) windows free ${free_gb} GB"
  if awk "BEGIN{exit !($free_gb >= 11)}"; then break; fi
  [ "$i" = 120 ] && { echo "GAVE-UP"; exit 3; }
  sleep 30
done
systemd-run --user --scope --collect --unit=part198-pilot2 -p MemoryMax=11G -p MemorySwapMax=0 \
  nice -n 10 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$HOME/edit-detect/.venv/bin/python" -I pilot198.py \
  --out pilot-part2.json --methods AlphaEdit,MEMIT-b10,AlphaEdit-b10
echo "PART2-RC=$?"
