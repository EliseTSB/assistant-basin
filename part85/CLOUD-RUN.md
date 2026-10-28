# Part 85 cloud CPU run

Run one checkpoint at a time on a Claude cloud machine. Outputs default to
`results/part85/<checkpoint>/`. Resolve and pin branches at run time; each script
refuses a full commit that does not match the registered prefix.

## Install and prepare the reference

From the repository root:

```bash
python3 -m pip install --upgrade torch transformers datasets huggingface_hub safetensors numpy
mkdir -p results/part85
```

Run the Instruct checkpoint's step 0 and step 1 first. Step 0 supplies the fixed
weight reference, and step 1 creates the shared WikiText token IDs and activation
reference used by every other checkpoint:

```bash
python3 part85/step0_weights.py --checkpoint instruct
python3 part85/step1_acts.py --checkpoint instruct --device cpu
```

The WikiText-103 test split is downloaded through `datasets` on the first step 1
run. Its 200 x 128 token IDs and SHA-256 are saved to
`results/part85/neutral-token-ids.json`; subsequent checkpoints load the same
IDs. Keep that file with the results.

## Run one checkpoint's steps 0 and 1

For any other registered checkpoint, pass the Instruct step 0 result and run the
checkpoint's activation step after Instruct:

```bash
python3 part85/step0_weights.py --checkpoint pretrain_4T \
  --reference results/part85/instruct/step0.json
python3 part85/step1_acts.py --checkpoint pretrain_4T --device cpu \
  --reference-acts results/part85/instruct/step1.json
```

Substitute one key from `python3 part85/checkpoints.py --keys ...` for
`pretrain_4T`. The six pretraining, three mid-training, SFT, DPO, and Instruct
keys are listed in that module. `--limit-rate BYTES_PER_SECOND` throttles step 0
HTTP range reads if needed. A local safetensors directory can be passed to step
0 with `--local PATH` for reader checks.

## Pilot gate, then main cells

Run the pilot for end-of-pretraining and Instruct after each has step 1. It
produces four 24-answer cells (96 answers total):

```bash
python3 part85/part85.py --mode pilot --checkpoint pretrain_4T --device cpu
python3 part85/part85.py --mode pilot --checkpoint instruct --device cpu
python3 part85/score85.py prepare --mode pilot --attempt 1 --results-root results/part85 \
  --items results/part85/pilot-items.jsonl --key results/part85/pilot-key.json
```

Have one blind reader label every item in `pilot-items.jsonl` with the
`blind_origin.RUBRIC` categories (`origin`: `EARTH`, `OFFWORLD`, or `NEITHER`;
`form`: `OK` or `BROKEN`) and save one JSON object per line with `id`, `origin`,
and `form`. For the 24 end-of-pretraining unclamped items, add a boolean
`responsive` field for whether the answer responds to its question rather than
continuing a list or unrelated text; this makes the pilot's quality gate
machine-checkable. Apply Part 84's registered note: a company name that contains a
space-related word (such as “Alienware”) remains an ordinary company origin
unless the speaker itself claims to be alien or from space; mixed Chinese and
English alone is not BROKEN. Keep the answer key away from both readers. Then
score the pilot:

```bash
python3 part85/score85.py score --mode pilot --key results/part85/pilot-key.json \
  --labels results/part85/pilot-labels.jsonl --out results/part85/pilot-score.json
```

If the quality condition fails, write the revised frame to a UTF-8 file with
one `{q}` placeholder, then rerun both pilot checkpoints once with `--attempt 2`
and the same `--plain-template-file` path:

```bash
python3 part85/part85.py --mode pilot --checkpoint pretrain_4T --attempt 2 \
  --plain-template-file results/part85/pilot-frame-v2.txt --device cpu
python3 part85/part85.py --mode pilot --checkpoint instruct --attempt 2 \
  --plain-template-file results/part85/pilot-frame-v2.txt --device cpu
python3 part85/score85.py prepare --mode pilot --attempt 2 --results-root results/part85 \
  --items results/part85/pilot-items-v2.jsonl --key results/part85/pilot-key-v2.json
```

Label attempt 2, then score it before continuing. Attempts use separate answer
files; a failed second quality gate does not authorize a third pilot or the main
run.

If the gate says `plain_frame_pass`, run the selected checkpoint's main cells
with that score file. Each run writes 4 plain cells; SFT, DPO, and Instruct also
write 4 chat-template cells. Each cell contains 48 answers:

```bash
python3 part85/part85.py --mode main --checkpoint pretrain_4T \
  --pilot-score results/part85/pilot-score.json --device cpu
```

When the passed pilot used a revised frame, pass that same file to every main
run with `--plain-template-file`; the script checks its SHA-256 against the
pilot score before generating plain-frame cells.

If the gate says `chat_only_posttrained`, run only `sft`, `dpo`, and `instruct`;
the script runs their chat cells and refuses pretraining checkpoints. If the
quality gate asks for `amend_frame_and_rerun_once`, do not launch the main run
until the revised frame has been written down and a new pilot has been scored.
Main cells require both independent reader label files before scoring:

```bash
python3 part85/score85.py prepare --mode main --results-root results/part85 \
  --items results/part85/main-items.jsonl --key results/part85/main-key.json
python3 part85/score85.py score --mode main --key results/part85/main-key.json \
  --labels results/part85/reader-1.jsonl --labels2 results/part85/reader-2.jsonl \
  --results-root results/part85 --out results/part85/main-score.json
```

`prepare` exposes only random `id` and answer `text`; keep the generated key
separate from both readers. It filters pilot attempts and pilot/main rows, so pilot
answers do not enter the main blind set. Reruns skip only cells with a complete
answer file and verified prompt and generated-position hook counts.
