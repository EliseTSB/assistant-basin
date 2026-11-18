#!/usr/bin/env bash
# Run timecut.py once Part 187's Arc run has ended, the AST copy is complete, Elise's game is closed and Windows has
# 6 GB free. Started detached (survives a session boundary); log: timecut_after_arc.log.
set -u
R=/home/elisepc/green/Journal/notes/reports/embeddinggemma2-chicken-neurons-2026-11-03
L=/mnt/c/Users/alexa/neuron-probe-gpu/progress.log
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
for i in $(seq 1 600); do tail -n +969 "$L" | grep -q "step 1 rc=" && break; sleep 30; done
echo "$(date +%H:%M:%S) arc: $(tail -n +969 "$L" | grep "step 1 rc=" | tail -1)"
for i in $(seq 1 120); do [ -s /mnt/e/neuron-probe/ast-audioset/model.safetensors ] && [ "$(stat -c %s /mnt/e/neuron-probe/ast-audioset/model.safetensors)" -gt 300000000 ] && break; sleep 30; done
ok=0; for i in $(seq 1 600); do r=$($PS -NoProfile -Command "\$f=[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1); \$g=@(Get-Process deadlock -ErrorAction SilentlyContinue).Count; \"\$f \$g\"" 2>/dev/null | tr -d '\r'); f=${r% *}; g=${r#* }; if [ "$g" = "0" ] && awk "BEGIN{exit !($f>=6)}"; then ok=$((ok+1)); else ok=0; fi; [ $ok -ge 2 ] && break; sleep 60; done
echo "$(date +%H:%M:%S) gate ok=$ok free=$f game=$g"
[ $ok -ge 2 ] || { echo NOT-STARTED; exit 0; }
cd "$R" && HF_HUB_OFFLINE=1 PYTHONPATH=/home/elisepc/eg2-extra systemd-run --user --scope --quiet -p MemoryMax=5G /home/elisepc/neuron-probe/.venv/bin/python -B timecut.py > timecut.log 2>&1
echo "$(date +%H:%M:%S) timecut rc=$?"
tail -30 timecut.log
