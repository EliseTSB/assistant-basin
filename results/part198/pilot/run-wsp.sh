#!/bin/bash
# Part 198 pilot, wsp on Qwen3.5-0.8B: waits for pilot part 3 to finish, then for Windows >= 8 GB free, then runs once.
cd "$HOME/edit-detect/part198"
while systemctl --user is-active -q part198-pilot3-wait.scope; do sleep 30; done
for i in $(seq 1 120); do
  sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  free_gb=$(/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" 2>/dev/null | tr -d '\r')
  echo "$(date +%T) wsp windows free ${free_gb} GB"
  if awk "BEGIN{exit !($free_gb >= 8)}"; then break; fi
  [ "$i" = 120 ] && { echo "GAVE-UP wsp"; exit 3; }
  sleep 30
done
systemd-run --user --scope --collect --unit=part198-wsp -p MemoryMax=8G -p MemorySwapMax=0 \
  nice -n 10 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$HOME/neuron-probe/.venv/bin/python" -I wsp198.py --out wsp-pilot.json
echo "RC-wsp=$?"
