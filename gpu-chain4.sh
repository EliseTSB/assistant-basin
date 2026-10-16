#!/bin/bash
# gpu-chain4.sh — Part 8: collect the Qwen2.5-Coder harness three-way on the A770 as each download lands (Claude,
# 2026-10-16). Pattern from gpu-chain3.sh. A model is ready when dlq.log says "<repo> downloaded" (main pass) or
# "<repo> complete" (retry pass); it is given up on after the retry pass's third FAILED. Instruct and the two agent
# models also get sysgen8_gpu.py (48 answers per condition, plus raw_nosys). The base comes last (its main pass
# failed on a DNS blip at 02:08 and only the retry pass, which starts after the main unit, completes it).
G=/mnt/c/Users/alexa/neuron-probe-gpu; S=/mnt/c/Windows/System32/schtasks.exe; cd /home/elisepc/neuron-probe
log() { echo "$(date +%H:%M:%S) $*" >> gpu-chain4.log; }
wait_job() { until [ -f $G/job.exit ]; do sleep 10; done; }
run() {  # $1 = steps text (one or more lines)
  wait_job; printf '%s\n' "$1" > $G/steps.txt; rm -f $G/job.exit $G/job.log $G/progress.log $G/step-*.log
  $S /run /tn NeuronProbeGPU > /dev/null 2>&1; sleep 5; wait_job; rc=$(cat $G/job.exit)
  log "job rc=$rc: $(echo "$1" | tr '\n' ';')"; cp $G/progress.log "gpu4-progress-$(date +%H%M%S).log" 2>/dev/null; return $rc
}
ready() {  # 0 = ready, 1 = given up; waits otherwise
  until grep -q -E "$1 (downloaded|complete)" dlq.log || grep -q "$1 FAILED (retry pass, try 3)" dlq.log; do sleep 30; done
  grep -q -E "$1 (downloaded|complete)" dlq.log
}
log "chain4 start"
for R in Qwen/Qwen2.5-Coder-7B-Instruct SWE-bench/SWE-agent-LM-7B OpenHands/openhands-lm-7b-v0.1 Qwen/Qwen2.5-Coder-7B; do
  ready "$R" || { log "$R download given up"; continue; }
  slug=${R//\//__}
  steps="collect_gpu.py $R out"
  [ "$R" != Qwen/Qwen2.5-Coder-7B ] && steps="$steps
sysgen8_gpu.py $R out\\sysgen8-$slug.json --samples 16 --raw-nosys"
  run "$steps"; rc=$?
  [ -f $G/out/$slug.npz ] && cp $G/out/$slug.npz out/ && log "$R activations copied"
  [ -f "$G/out/sysgen8-$slug.json" ] && cp "$G/out/sysgen8-$slug.json" out/ && log "$R self-naming copied"
  [ $rc = 0 ] || log "$R job FAILED rc=$rc (see the gpu4-progress log and $G/step-*.log)"
done
log "chain4 done"
