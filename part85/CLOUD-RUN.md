# Part 85 cloud run and session handoff

Each session starts from a fresh clone at the same pinned repository commit. The pilot is one session that runs Instruct and then `pretrain_4T`; each main session handles one checkpoint. The sessions push resumable results to `part85-run/pilot` or `part85-run/<key>` after each result file or answer cell.

Set `PINNED_COMMIT` to the commit chosen by the researcher after the pilot references have been checked in. Run commands from the repository root.

## Install the pinned environment

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple 'torch==2.14.1+cpu'
python3 -m pip install \
  'transformers==5.20.0' \
  'datasets==5.1.0' \
  'numpy==2.5.4' \
  'safetensors==0.8.0' \
  'huggingface_hub==2.2.0'
python3 -m pip check
```

The reader code uses the Hugging Face Datasets `load_dataset` API with the dataset repository, configuration, split and pinned `revision`. ⚠ `datasets==5.0.1` (the first pin) cannot install beside `huggingface_hub==2.2.0`: it requires `huggingface-hub<2.0` (PyPI metadata, checked 2026-10-27 23:2x CT), so pip stops with ResolutionImpossible. `datasets==5.1.0` allows hub 2.x. Claude ran the real load on the host with exactly these pins on 2026-10-27: `create_neutral_sequences` with the Instruct tokenizer (revision 48d788ec) gave shape (200, 128) and sha256 `96699282f18fc53d5b5467a51803241aab479e23523b92980bdb83546ae15d87`. The Instruct session's `neutral-token-ids.json` must carry that sha256; stop if it does not. The load also builds the train split (about 1.8M rows), which takes time but no extra steps. Before downloading the model snapshot, check the Hub load in this session:

```bash
python3 - <<'PY'
from datasets import load_dataset
rows = load_dataset(
    'Salesforce/wikitext', 'wikitext-103-v1', split='test',
    revision='f776294184f13b8ff2337b3841cf9269a6216d1e',
)
print('WikiText pinned split:', len(rows), 'rows; first text:', repr(rows[0]['text'][:80]))
PY
```

Stop if that command fails. The test suite uses a stub for this API call; the host check above is the real one. The dataset revision remains the registered immutable commit.

## Pilot session: Instruct, then `pretrain_4T`

Clone and check out the pinned repository state. Restore the pilot branch before preflight; a not-yet-created branch is normal on the first run.

```bash
git clone <REPO_URL> part85-pilot
cd part85-pilot
git checkout --detach "$PINNED_COMMIT"
python3 part85/handoff85.py restore --branch part85-run/pilot --results-root results/part85
python3 part85/handoff85.py preflight --checkpoint instruct --phase pilot --results-root results/part85
```

Instruct is the reference checkpoint. Step 1 creates `neutral-token-ids.json`; each subsequent file is pushed to the pilot branch.

```bash
SNAPSHOT_DIR="$(python3 -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("instruct"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))')"
python3 part85/step0_weights.py --checkpoint instruct --snapshot-dir "$SNAPSHOT_DIR" \
  --results-root results/part85 --push-branch part85-run/pilot
python3 part85/step1_acts.py --checkpoint instruct --snapshot-dir "$SNAPSHOT_DIR" --device cpu \
  --results-root results/part85 --sequence-ids results/part85/neutral-token-ids.json \
  --push-branch part85-run/pilot
python3 part85/part85.py --mode pilot --checkpoint instruct --snapshot-dir "$SNAPSHOT_DIR" \
  --results-root results/part85 --device cpu --push-branch part85-run/pilot
rm -rf "$SNAPSHOT_DIR"
```

Preflight `pretrain_4T` before its snapshot download. It requires the Instruct step files and shared token IDs created above.

```bash
python3 part85/handoff85.py preflight --checkpoint pretrain_4T --phase pilot --results-root results/part85
SNAPSHOT_DIR="$(python3 -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("pretrain_4T"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))')"
python3 part85/step0_weights.py --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" \
  --reference results/part85/instruct/step0.json --results-root results/part85 \
  --push-branch part85-run/pilot
python3 part85/step1_acts.py --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" --device cpu \
  --reference-acts results/part85/instruct/step1.json \
  --sequence-ids results/part85/neutral-token-ids.json --results-root results/part85 \
  --push-branch part85-run/pilot
python3 part85/part85.py --mode pilot --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" \
  --results-root results/part85 --device cpu --push-branch part85-run/pilot
rm -rf "$SNAPSHOT_DIR"
```

Keep the answer key separate from the blind reader. Assemble both pilot checkpoints locally, prepare attempt 1, label every item, then score:

```bash
python3 part85/handoff85.py assemble --branches part85-run/pilot --out results/part85
python3 part85/score85.py prepare --mode pilot --attempt 1 --results-root results/part85 \
  --items results/part85/pilot-items.jsonl --key results/part85/pilot-key.json
# The reader labels all 96 items, including responsive, and saves pilot-labels.jsonl.
python3 part85/score85.py score --mode pilot --key results/part85/pilot-key.json \
  --items results/part85/pilot-items.jsonl --labels results/part85/pilot-labels.jsonl \
  --out results/part85/pilot-score.json
```

The pilot gate counts end-of-pretraining unclamped answers that are both `OK` and responsive. Follow its recorded action. A `quality_failed_after_rerun` result ends the experiment; it does not authorize another attempt.

### One registered attempt-2 rerun

If attempt 1 says `amend_frame_and_rerun_once`, save the revised UTF-8 template at `results/part85/pilot-frame-v2.txt`. Restore the pilot branch in the same session (or a new clone at the same pinned commit), then preflight and download each checkpoint snapshot in turn. The step files and token IDs from attempt 1 are reused. Run step 2 for Instruct, delete its snapshot, then preflight and download `pretrain_4T` and run step 2 for it:

```bash
python3 part85/handoff85.py restore --branch part85-run/pilot --results-root results/part85
python3 part85/handoff85.py preflight --checkpoint instruct --phase pilot --results-root results/part85
SNAPSHOT_DIR="$(python3 -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("instruct"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))')"
python3 part85/part85.py --mode pilot --checkpoint instruct --snapshot-dir "$SNAPSHOT_DIR" \
  --attempt 2 --plain-template-file results/part85/pilot-frame-v2.txt \
  --results-root results/part85 --device cpu --push-branch part85-run/pilot
rm -rf "$SNAPSHOT_DIR"
python3 part85/handoff85.py preflight --checkpoint pretrain_4T --phase pilot --results-root results/part85
SNAPSHOT_DIR="$(python3 -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("pretrain_4T"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))')"
python3 part85/part85.py --mode pilot --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" \
  --attempt 2 --plain-template-file results/part85/pilot-frame-v2.txt \
  --results-root results/part85 --device cpu --push-branch part85-run/pilot
rm -rf "$SNAPSHOT_DIR"
python3 part85/handoff85.py assemble --branches part85-run/pilot --out results/part85
python3 part85/score85.py prepare --mode pilot --attempt 2 --results-root results/part85 \
  --items results/part85/attempt-2/pilot-items.jsonl \
  --key results/part85/attempt-2/pilot-key.json
# The reader labels all 96 attempt-2 items at attempt-2/pilot-labels.jsonl.
python3 part85/score85.py score --mode pilot \
  --key results/part85/attempt-2/pilot-key.json \
  --items results/part85/attempt-2/pilot-items.jsonl \
  --labels results/part85/attempt-2/pilot-labels.jsonl \
  --out results/part85/attempt-2/pilot-score.json
mkdir -p results/part85/attempt-1
cp results/part85/pilot-score.json results/part85/attempt-1/pilot-score.json
cp results/part85/attempt-2/pilot-score.json results/part85/pilot-score.json
```

Score attempt 2 before starting main. Only the canonical `results/part85/pilot-score.json` is the main-run gate input. Keep attempt 1 files and labels unchanged.

## Pin and publish the references

After scoring the accepted pilot, assemble the pilot branch into the local result tree. Generate a SHA-256 manifest, then commit `part85/references.json` and every listed reference file at the repository root. This commit becomes `PINNED_COMMIT` for main sessions. The manifest contains Instruct step 0 and step 1, shared token IDs, pilot score, and the custom plain template if the pilot used one.

```bash
python3 part85/handoff85.py assemble --branches part85-run/pilot --out results/part85
python3 part85/handoff85.py refs-manifest --results-root results/part85 --out part85/references.json
# Check in part85/references.json and the files it names, then set PINNED_COMMIT to that commit.
```

## Main sessions: one checkpoint per session

Each session clones the pinned commit, restores its own branch, and runs main preflight before downloading a model. For Instruct or `pretrain_4T`, restore the pilot branch first to reuse its step files and pilot cells; then restore the checkpoint's main branch so an interrupted main run takes precedence.

```bash
git clone <REPO_URL> part85-<key>
cd part85-<key>
git checkout --detach "$PINNED_COMMIT"
python3 part85/handoff85.py restore --branch part85-run/pilot --results-root results/part85
python3 part85/handoff85.py restore --branch part85-run/<key> --results-root results/part85
python3 part85/handoff85.py preflight --checkpoint <key> --phase main --results-root results/part85
```

For Instruct and `pretrain_4T`, skip step 0 or step 1 if its restored file already exists. Other checkpoints run both steps against the pinned Instruct references. The step scripts verify a supplied snapshot's full commit: an HF cache snapshot must live at `snapshots/<full-commit>`, or a copied directory must have `config.json` with the matching `_commit_hash`. Do not use an unprovenanced directory.

```bash
SNAPSHOT_DIR="$(python3 -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("<key>"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))')"
python3 part85/step0_weights.py --checkpoint <key> --snapshot-dir "$SNAPSHOT_DIR" \
  --reference results/part85/instruct/step0.json --results-root results/part85 \
  --push-branch part85-run/<key>
python3 part85/step1_acts.py --checkpoint <key> --snapshot-dir "$SNAPSHOT_DIR" --device cpu \
  --reference-acts results/part85/instruct/step1.json \
  --sequence-ids results/part85/neutral-token-ids.json --results-root results/part85 \
  --push-branch part85-run/<key>
```

For a checkpoint restored with its step files already present, omit those step commands. Then run step 2 with the same pinned snapshot and gate. If the pilot passed the plain frame, all registered checkpoints run plain cells; SFT, DPO, and Instruct also run chat cells. If the gate says `chat_only_posttrained`, run only SFT, DPO, and Instruct; their scope contains chat cells only. A pretraining checkpoint is refused in chat-only scope. If the pilot requests amendment, do not start main.

```bash
python3 part85/part85.py --mode main --checkpoint <key> --snapshot-dir "$SNAPSHOT_DIR" \
  --pilot-score results/part85/pilot-score.json --results-root results/part85 \
  --device cpu --push-branch part85-run/<key>
# Add only when the accepted pilot used its revised template:
# --plain-template-file results/part85/pilot-frame-v2.txt
rm -rf "$SNAPSHOT_DIR"
```

A completed checkpoint pushes its result directory after each step file and after every answer cell. A failed push fetches and rebases onto the checkpoint branch, then retries up to three times. A missing restore branch is normal for a new checkpoint. Do not delete a snapshot until that checkpoint has completed its requested steps and answer cells.

## Local scoring and assembly

For a plain-frame pilot pass, assemble `part85-run/pilot` and all twelve checkpoint branches. For a chat-only pilot, assemble the pilot branch and the `sft`, `dpo`, and `instruct` branches. Identical copies of a checkpoint's step files are allowed; conflicting bytes or a branch containing another checkpoint's directory are refused.

```bash
python3 part85/handoff85.py assemble --branches \
  part85-run/pilot part85-run/pretrain_21B part85-run/pretrain_210B \
  part85-run/pretrain_1T part85-run/pretrain_2T part85-run/pretrain_3T \
  part85-run/pretrain_4T part85-run/midtrain_13B part85-run/midtrain_26B \
  part85-run/midtrain_51B part85-run/sft part85-run/dpo part85-run/instruct \
  --out results/part85
python3 part85/score85.py prepare --mode main --results-root results/part85 \
  --items results/part85/main-items.jsonl --key results/part85/main-key.json
python3 part85/score85.py score --mode main --key results/part85/main-key.json \
  --items results/part85/main-items.jsonl --labels results/part85/reader-1.jsonl \
  --labels2 results/part85/reader-2.jsonl --results-root results/part85 \
  --out results/part85/main-score.json
```

For a chat-only pilot, replace the branch list with the in-scope set:

```bash
python3 part85/handoff85.py assemble --branches \
  part85-run/pilot part85-run/sft part85-run/dpo part85-run/instruct \
  --out results/part85
```

`prepare` refuses existing item/key files. `score` refuses to replace an existing output; pass `--force` only when intentionally replacing it. Main scoring checks answer counts, unique question/sample pairs, registered seeds, checkpoint identities, and step-file checkpoint identities. Use `--partial` only for an explicitly incomplete analysis; it is the only mode that permits `NOT READ` verdicts.
