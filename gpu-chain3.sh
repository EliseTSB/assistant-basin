#!/bin/bash
# gpu-chain3.sh — collect the OLMo 3 7B ladder on the A770 as each download lands (Claude fork, 2026-10-16).
# For each model: wait for dl3.log to say it downloaded, wait for the previous GPU job to finish, write steps.txt,
# start the NeuronProbeGPU task, wait for job.exit, copy the npz into out/. Logs to gpu-chain3.log.
G=/mnt/c/Users/alexa/neuron-probe-gpu; S=/mnt/c/Windows/System32/schtasks.exe; cd /home/elisepc/neuron-probe
log() { echo "$(date +%H:%M:%S) $*" >> gpu-chain3.log; }
wait_job() { until [ -f $G/job.exit ]; do sleep 10; done; }
run() {  # $1 = steps text
  wait_job; printf '%s\n' "$1" > $G/steps.txt; rm -f $G/job.exit $G/job.log $G/progress.log $G/step-*.log
  $S /run /tn NeuronProbeGPU > /dev/null 2>&1; sleep 5; wait_job; rc=$(cat $G/job.exit); log "job rc=$rc: $1"; return $rc
}
wait_job
[ -f $G/out/allenai__Olmo-3-1025-7B.npz ] && cp $G/out/allenai__Olmo-3-1025-7B.npz out/ && log "base collected"
for R in allenai/Olmo-3-7B-Instruct-SFT allenai/Olmo-3-7B-Instruct-DPO allenai/Olmo-3-7B-Instruct allenai/Olmo-3-7B-RL-Zero-General; do
  until grep -q -E "$R (downloaded|FAILED)" dl3.log; do sleep 20; done
  grep -q "$R FAILED" dl3.log && { log "$R download FAILED"; continue; }
  slug=${R//\//__}
  run "collect_gpu.py $R out" && cp $G/out/$slug.npz out/ && log "$R collected" || log "$R collect FAILED"
done
log "collection done"
