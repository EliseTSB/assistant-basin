#!/bin/bash
# Part 198 pilot, part 3: MEMIT-b10 and AlphaEdit-b10, ONE METHOD PER PROCESS (part 2's MEMIT-b10 hit the 11G cap with
# AlphaEdit's caches still held; gotcha). Each waits for Windows >= 11 GB free first.
cd "$HOME/edit-detect/part198"
for m in MEMIT-b10 AlphaEdit-b10; do
  for i in $(seq 1 120); do
    sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    free_gb=$(/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" 2>/dev/null | tr -d '\r')
    echo "$(date +%T) $m windows free ${free_gb} GB"
    if awk "BEGIN{exit !($free_gb >= 11)}"; then break; fi
    [ "$i" = 120 ] && { echo "GAVE-UP $m"; exit 3; }
    sleep 30
  done
  systemd-run --user --scope --collect --unit=part198-pilot3-$m -p MemoryMax=11G -p MemorySwapMax=0 \
    nice -n 10 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$HOME/edit-detect/.venv/bin/python" -I pilot198.py \
    --out pilot-part3-$m.json --methods $m
  echo "RC-$m=$?"
done
echo "PART3-DONE"
