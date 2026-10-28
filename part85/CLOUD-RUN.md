# Part 85 cloud CPU run

Use one cloud session per checkpoint. Keep that session open through steps 0,
1, and 2 so the same pinned local snapshot is reused and the checkpoint is not
downloaded twice. The Instruct session runs first: its `step0.json` and
`step1.json` are the references for all other checkpoints. The pilot needs the
Instruct and end-of-pretraining (`pretrain_4T`) sessions; collect both pilot
cells before scoring the gate.

Outputs default to `results/part85/<checkpoint>/`. Resolve each registered
branch to its pinned full commit before downloading the snapshot. Run commands
from the repository root.

## Install

```bash
python3 -m pip install --upgrade torch transformers datasets huggingface_hub safetensors numpy
mkdir -p results/part85
```

## Start the Instruct session first

Resolve and download Instruct once. Keep `SNAPSHOT_DIR` in this session and pass
it to each Part 85 script:

```bash
SNAPSHOT_DIR="$(python3 -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("instruct"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))')"
python3 part85/step0_weights.py --checkpoint instruct --snapshot-dir "$SNAPSHOT_DIR"
python3 part85/step1_acts.py --checkpoint instruct --snapshot-dir "$SNAPSHOT_DIR" --device cpu
python3 part85/part85.py --mode pilot --checkpoint instruct --snapshot-dir "$SNAPSHOT_DIR" --device cpu
```

Step 0 supplies the fixed weight reference. Step 1 creates the shared WikiText
token IDs and activation reference. WikiText-103 is pinned to the registered
dataset commit in `step1_acts.py`; its 200 x 128 token IDs and SHA-256 are saved
to `results/part85/neutral-token-ids.json`. Keep that file with the results.

## End-of-pretraining session and pilot gate

Start a separate `pretrain_4T` session, then run its steps 0 and 1 against the
Instruct references. The same local snapshot is also used for its pilot:

```bash
SNAPSHOT_DIR="$(python3 -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("pretrain_4T"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))')"
python3 part85/step0_weights.py --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" \
  --reference results/part85/instruct/step0.json
python3 part85/step1_acts.py --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" --device cpu \
  --reference-acts results/part85/instruct/step1.json
python3 part85/part85.py --mode pilot --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" --device cpu
```

The two pilot sessions produce four 24-answer cells (96 answers total). Prepare
the blind set and key:

```bash
python3 part85/score85.py prepare --mode pilot --attempt 1 --results-root results/part85 \
  --items results/part85/pilot-items.jsonl --key results/part85/pilot-key.json
```

Every item includes its question and answer. One blind reader labels every
item with the `blind_origin.RUBRIC` categories (`origin`: `EARTH`, `OFFWORLD`,
or `NEITHER`; `form`: `OK` or `BROKEN`) and saves one JSON object per line with
`id`, `origin`, and `form`. The reader also adds a boolean `responsive` to all
96 pilot labels for whether the answer responds to its displayed question. The
gate counts only `OK` and responsive end-of-pretraining unclamped answers.
Apply Part 84's registered note: a company name that contains a space-related
word (such as “Alienware”) remains an ordinary company origin unless the
speaker itself claims to be alien or from space; mixed Chinese and English
alone is not BROKEN. Keep the answer key away from the reader. Then score:

```bash
python3 part85/score85.py score --mode pilot --key results/part85/pilot-key.json \
  --items results/part85/pilot-items.jsonl --labels results/part85/pilot-labels.jsonl \
  --out results/part85/pilot-score.json
```

If quality fails, write the revised frame to a UTF-8 file with one `{q}`
placeholder, then rerun both pilot checkpoints once with `--attempt 2` and the
same `--plain-template-file` path. Resume each checkpoint in its existing
session and use its existing `SNAPSHOT_DIR`:

```bash
python3 part85/part85.py --mode pilot --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" \
  --attempt 2 --plain-template-file results/part85/pilot-frame-v2.txt --device cpu
```

Run the same command in the Instruct session with `--checkpoint instruct` and
that session's snapshot path. Prepare and score attempt 2 before continuing.
A failed second quality gate does not authorize another pilot or the main run.

## Main run, one checkpoint per session

For each remaining checkpoint, start one session, resolve and download its
snapshot once, then keep using that directory for all three steps. The Instruct
and `pretrain_4T` sessions already have steps 0 and 1 and their snapshots from
the pilot phase; resume those sessions for step 2 after the gate is scored.

If the gate says `plain_frame_pass`, run all four plain cells for every
checkpoint. SFT, DPO, and Instruct also run four chat-template cells. Each
cell has 48 answers:

```bash
SNAPSHOT_DIR="$(python3 -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("pretrain_21B"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))')"
python3 part85/step0_weights.py --checkpoint pretrain_21B --snapshot-dir "$SNAPSHOT_DIR" \
  --reference results/part85/instruct/step0.json
python3 part85/step1_acts.py --checkpoint pretrain_21B --snapshot-dir "$SNAPSHOT_DIR" --device cpu \
  --reference-acts results/part85/instruct/step1.json
python3 part85/part85.py --mode main --checkpoint pretrain_21B --snapshot-dir "$SNAPSHOT_DIR" \
  --pilot-score results/part85/pilot-score.json --device cpu
```

Substitute one registered key for `pretrain_21B`. When the pilot used a revised
frame, pass that same `--plain-template-file` to every main command. The script
checks its SHA-256 against the passed pilot score.

If the gate says `chat_only_posttrained`, run only `sft`, `dpo`, and `instruct`;
the script runs their chat cells and refuses pretraining checkpoints. If the
quality gate asks for `amend_frame_and_rerun_once`, do not launch main until the
revised frame has been written down and a new pilot has been scored.

After all authorized answer cells are complete, prepare and score the main
blind set. Main scoring requires the complete 2,880 answers (or all cells in
the pilot-authorized chat-only scope), all twelve step 0 and step 1 files, and
both independent reader label files:

```bash
python3 part85/score85.py prepare --mode main --results-root results/part85 \
  --items results/part85/main-items.jsonl --key results/part85/main-key.json
python3 part85/score85.py score --mode main --key results/part85/main-key.json \
  --items results/part85/main-items.jsonl --labels results/part85/reader-1.jsonl \
  --labels2 results/part85/reader-2.jsonl --results-root results/part85 \
  --out results/part85/main-score.json
```

`prepare` exposes a cryptographically random ID, the question, and answer text;
keep the generated key separate from both readers. It filters pilot attempts and
pilot/main rows, and refuses to overwrite existing prepared files unless
`--force` is passed. A scoring run checks the items file against its recorded
SHA-256. Reruns skip only cells with a complete answer file, matching template,
matching step 1 clamp value, and verified prompt and generated-position hook
counts. `--partial` is only for an explicitly incomplete main analysis; it is
the only scoring mode that permits `NOT READ` verdicts.
