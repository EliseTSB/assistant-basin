#!/bin/bash
# Part 198 exploration: re-run ROME, MEMIT (one process) and AlphaEdit (own process) single edits saving their updates,
# then read every c_proj's singular directions through GPT-2 XL's unembedding. Each model step waits for 11 GB free.
cd "$HOME/edit-detect/part198"
waitmem() {
  for i in $(seq 1 120); do
    sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    free_gb=$(/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" 2>/dev/null | tr -d '\r')
    echo "$(date +%T) $1 windows free ${free_gb} GB"
    if awk "BEGIN{exit !($free_gb >= $2)}"; then return 0; fi
    sleep 30
  done
  echo "GAVE-UP $1"; exit 3
}
run() { systemctl --user reset-failed "$1" 2>/dev/null; systemd-run --user --scope --collect --unit="$1" -p MemoryMax=11G -p MemorySwapMax=0 nice -n 10 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$HOME/edit-detect/.venv/bin/python" -I "${@:2}"; echo "RC-$1=$?"; }
waitmem rome-memit 11; run part198-sd1 pilot198.py --out pilot-sd.json --methods ROME,MEMIT --save-delta
waitmem alphaedit 11; run part198-sd2 pilot198.py --out pilot-sd-ae.json --methods AlphaEdit --save-delta
waitmem explore 7; run part198-xg explore_gpt2.py --out explore-gpt2.json pilot-sd.json.ROME.delta.pt pilot-sd.json.MEMIT.delta.pt pilot-sd-ae.json.AlphaEdit.delta.pt
echo "EXPLORE-GPT2-DONE"
