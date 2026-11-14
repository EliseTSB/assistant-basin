#!/bin/bash
# Written 2026-11-14 about 05:00 CT; relaunched after Amendment 3 (see chain-q.log for the time) by Claude (Opus 5.5). Family Q under Amendment 2 item 4: O's cost gate failed (A0
# 2026-11-13, gap 0.0191 < 0.194), so both Qwen models run the controls arms only (--arms controls). One A770 job at a
# time, each only once Windows has the RAM (arc-steps-when-ram.sh), results copied back after each step:
#   1. the A770 G1 check (steps.txt step 2), one attempt: a failed gate is not retried;
#   2. qwen-chat, controls. Kernel check: results/part183/kernel-check.json (Qwen3.5-4B 851bf6e8, xpu, bf16, eager)
#      passes part_check kernel against results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json (21/21, rerun at about 05:00 before the first launch);
#   3. Qwen3.5-4B-Base's kernel check, which did not exist: the CPU float32-upcast reference on the Windows CPU, then the
#      card candidate (eager bf16; it can exit 1 at an exact bf16 tie, so the record is judged, not the exit code), then
#      part_check kernel; qwen-base runs only if that prints KERNEL CHECK: PASS;
#   4. qwen-base, controls.
# The design predates the launcher's Registered/Total/Pilot/Kernel-check line formats and is public, so each launch
# carries an --override (ARC_OVERRIDE) naming the check that was done instead.
set -u
cd /home/elisepc/green/worktrees/ab-part201 || exit 1
export ARC_TREE=/home/elisepc/green/worktrees/ab-part201 LAUNCH_ARC_CONFIG=/home/elisepc/green/worktrees/ab-part67/tools/local.json
ARC=/home/elisepc/green/Journal/scripts/research/arc-steps-when-ram.sh
J=/mnt/c/Users/alexa/neuron-probe-gpu
BASE="Registered design (a91f73bd; Amendments 1 and 2 public) predates the launcher's Registered/Total/Pilot/Kernel-check line formats."
COPY_Q=(part205/run205.py part205/common205.py part205/config.json part205/prompts205.json part205/models.json neuron_report.py)

merge_models() {  # copy the job folder's models.json back only if it holds every row the worktree's has, plus $1
  python3 -I - "$J/part205/models.json" part205/models.json "$1" <<'PY' || return 1
import json, sys, os
win, wt, stage = sys.argv[1], sys.argv[2], sys.argv[3]
w, t = json.load(open(win)), json.load(open(wt))
# "stages" is a dict keyed by stage name (run205.write_models_json); every worktree stage must be unchanged
assert all(w["stages"].get(k) == v for k, v in t["stages"].items()), "job folder models.json lost or changed a worktree row"
assert stage in w["stages"], f"no {stage} row"
tmp = wt + ".tmp"; json.dump(w, open(tmp, "w"), indent=1); os.replace(tmp, wt); print("models.json merged:", stage)
PY
}

# 1. G1 (skipped when a passing g1.json from this card is already here: it passed at 05:12 on 11-14, and the
#    lock fix does not touch g1check205.py's path)
if python3 -I -c "import json,sys; sys.exit(0 if json.load(open('results/part205/g1/g1.json'))['verdict']['pass'] else 1)" 2>/dev/null; then
  echo "$(date +%H:%M:%S) G1 already passed (results/part205/g1/g1.json); skipped"
else
ARC_OVERRIDE="$BASE Step 2 (the A770 G1 check). Kernel check: results/part183/kernel-check.json passes against the Qwen3.5-4B CPU float32 reference (same revision, xpu, bf16, eager)." \
  $ARC 205 11 1 "-I -B part205/g1check205.py --stage qwen-chat --out results/part205/g1" \
  -- part205/run205.py part205/common205.py part205/score205.py part205/config.json part205/prompts205.json part205/g1check205.py neuron_report.py \
  || { echo "PART205 G1 FAILED OR DID NOT RUN"; exit 3; }
mkdir -p results/part205/g1 && cp "$J/results/part205/g1/g1.json" results/part205/g1/ || { echo "COPY FAILED: g1.json"; exit 5; }
python3 -I -c "import json,sys; v=json.load(open('results/part205/g1/g1.json'))['verdict']; print('G1', v); sys.exit(0 if v['pass'] else 1)" \
  || { echo "PART205 G1 VERDICT FAIL: family Q held"; exit 4; }
fi

# 2. qwen-chat, controls
ARC_OVERRIDE="$BASE Family Q, qwen-chat, --arms controls (Amendment 2 item 4: O's cost gate failed). Kernel check: results/part183/kernel-check.json passes against the Qwen3.5-4B CPU float32 reference (same revision, xpu, bf16, eager)." \
  $ARC 205 11 2 "-I -B part205/run205.py --stage qwen-chat --out results/part205 --device xpu --check-pool --arms controls" \
  -- "${COPY_Q[@]}" || { echo "PART205 QWEN-CHAT FAILED"; exit 6; }
rm -rf results/part205/qwen-chat && cp -r "$J/results/part205/qwen-chat" results/part205/ || { echo "COPY FAILED: qwen-chat"; exit 5; }
merge_models qwen-chat || { echo "MODELS.JSON MERGE FAILED: qwen-chat"; exit 5; }
echo "$(date +%H:%M:%S) PART205-QWEN-CHAT-DONE"

# 3. Qwen3.5-4B-Base kernel check
REV=1001bb4d826a52d1f399e183466143f4da7b741b
REF=results/kernel-reference/qwen-qwen3.5-4b-base-cpu-float32.json
CAND=results/part205/kernel-check-base.json
KCOPY=(tools/kernel_reference.py tools/part_check.py neuron_report.py)
ARC_OVERRIDE="$BASE Kernel check for Qwen3.5-4B-Base before family Q's base stage (PROCEDURE.md section 6): the CPU float32-upcast reference." \
  $ARC 205 12 1 "-I -B tools/kernel_reference.py --model Qwen/Qwen3.5-4B-Base --revision $REV --device cpu --dtype float32-upcast --attn-implementation eager --out $REF" \
  -- "${KCOPY[@]}" || { echo "BASE KERNEL REFERENCE FAILED"; exit 7; }
mkdir -p results/kernel-reference && cp "$J/$REF" "$REF" || { echo "COPY FAILED: base reference"; exit 5; }
ARC_OVERRIDE="$BASE Kernel check for Qwen3.5-4B-Base: the card candidate (xpu, bf16, eager)." \
  $ARC 205 11 1 "-I -B tools/kernel_reference.py --model Qwen/Qwen3.5-4B-Base --revision $REV --device xpu --dtype bfloat16 --attn-implementation eager --batch-sizes 1,2,8 --reference $REF --out $CAND" \
  -- "${KCOPY[@]}" || echo "candidate step rc!=0 (expected at an exact bf16 tie); judging the record"
cp "$J/$CAND" "$CAND" || { echo "COPY FAILED: base candidate (no record written)"; exit 5; }
python3 -I -B tools/part_check.py kernel --reference "$REF" --candidate "$CAND" > results/part205/kernel-check-base.txt 2>&1
tail -3 results/part205/kernel-check-base.txt
grep -q "KERNEL CHECK: PASS" results/part205/kernel-check-base.txt || { echo "BASE KERNEL CHECK DID NOT PASS: qwen-base held"; exit 8; }

# 4. qwen-base, controls
ARC_OVERRIDE="$BASE Family Q, qwen-base, --arms controls (Amendment 2 item 4: O's cost gate failed). Kernel check: $CAND passes part_check kernel against $REF (run 2026-11-14 in this chain, results/part205/kernel-check-base.txt)." \
  $ARC 205 11 2 "-I -B part205/run205.py --stage qwen-base --out results/part205 --device xpu --check-pool --arms controls" \
  -- "${COPY_Q[@]}" || { echo "PART205 QWEN-BASE FAILED"; exit 6; }
rm -rf results/part205/qwen-base && cp -r "$J/results/part205/qwen-base" results/part205/ || { echo "COPY FAILED: qwen-base"; exit 5; }
merge_models qwen-base || { echo "MODELS.JSON MERGE FAILED: qwen-base"; exit 5; }
echo "$(date +%H:%M:%S) PART205-Q-ALL-DONE"
