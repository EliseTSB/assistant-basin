# Part 85 cloud run and single-writer handoff

Every cloud session writes to one owned result directory:

| Session | Branch | Owned directory |
|---|---|---|
| Pilot attempt 1 | `part85-run/pilot-a1` | `results/part85/pilot-a1/` |
| Pilot attempt 2 | `part85-run/pilot-a2` | `results/part85/pilot-a2/` |
| Main checkpoint `<key>` | `part85-run/main-<key>` | `results/part85/main/<key>/` |

The Instruct pilot session creates `neutral-token-ids.json` once at
`results/part85/pilot-a1/neutral-token-ids.json`. The accepted pilot references live in
`part85/references/` in the later references commit. A result path is never written by two sessions or branches.
Each output records its code commit and pinned checkpoint commit.

## Test gate before any paid session

Run the committed mutation gate and require every mutant to be killed. A skipped test, a survived mutant, or a
nonzero exit stops the launch.

```bash
/home/elisepc/neuron-probe/.venv/bin/python -I -B part85/mutate85.py --jobs 2
```

The runner creates an isolated copy per mutation and invokes `test_part85.py` with `-I -B`. It uses temporary bare
Git remotes only. It limits OpenMP/BLAS threads to one per child test process to keep parallel runs from oversubscribing
the host. Do not download OLMo checkpoints for this gate.

## Build and pin code before the pilot

Commit the Part 85 code and tests first, before any run results or pilot references are added. Call that commit
`CODE_COMMIT`. Pilot sessions clone only this code commit. The pilot outputs record `CODE_COMMIT`.

Install the pinned environment in the session:

```bash
PY=/home/elisepc/neuron-probe/.venv/bin/python
$PY -I -B -m pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple 'torch==2.14.1+cpu'
$PY -I -B -m pip install \
  'transformers==5.20.0' \
  'datasets==5.1.0' \
  'numpy==2.5.4' \
  'safetensors==0.8.0' \
  'huggingface_hub==2.2.0'
$PY -I -B -m pip check
```

`datasets==5.0.1` cannot install with `huggingface_hub==2.2.0` because it requires hub `<2.0`; 5.1.0 accepts hub 2.x.
The real pinned WikiText test split was loaded with these packages and the Instruct tokenizer on 2026-10-27. Its
200 x 128 token matrix has SHA-256
`96699282f18fc53d5b5467a51803241aab479e23523b92980bdb83546ae15d87`. The Instruct step-1 session must produce
that digest. Before downloading a model, confirm the pinned dataset is reachable:

```bash
$PY -I -B - <<'PY'
from datasets import load_dataset
rows = load_dataset(
    'Salesforce/wikitext', 'wikitext-103-v1', split='test',
    revision='f776294184f13b8ff2337b3841cf9269a6216d1e',
)
print('WikiText pinned split:', len(rows), 'rows; first text:', repr(rows[0]['text'][:80]))
PY
```

Stop if it fails. The test suite stubs this API; this is the live data check. The dataset revision is immutable.

## Pilot attempt 1

Clone the code-only commit and restore the attempt's own branch before writing. On the first run it will be absent.
Run all commands from the repository root with the required Python interpreter:

```bash
PY=/home/elisepc/neuron-probe/.venv/bin/python
git clone <REPO_URL> part85-pilot-a1
cd part85-pilot-a1
git checkout --detach "$CODE_COMMIT"
export PART85_CODE_COMMIT="$CODE_COMMIT"
$PY -I -B part85/handoff85.py restore --branch part85-run/pilot-a1 --results-root results/part85
$PY -I -B part85/handoff85.py preflight --checkpoint instruct --phase pilot --attempt 1 --results-root results/part85
```

Run Instruct first. It writes the shared token IDs into the pilot-a1 directory; every step output and answer cell is
pushed only after its atomic file write completes.

```bash
SNAPSHOT_DIR="$( $PY -I -B -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("instruct"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))' )"
$PY -I -B part85/step0_weights.py --checkpoint instruct --snapshot-dir "$SNAPSHOT_DIR" \
  --results-root results/part85 --push-branch part85-run/pilot-a1
$PY -I -B part85/step1_acts.py --checkpoint instruct --snapshot-dir "$SNAPSHOT_DIR" --device cpu \
  --results-root results/part85 --sequence-ids results/part85/pilot-a1/neutral-token-ids.json \
  --push-branch part85-run/pilot-a1
$PY -I -B part85/part85.py --mode pilot --checkpoint instruct --snapshot-dir "$SNAPSHOT_DIR" \
  --results-root results/part85 --device cpu --push-branch part85-run/pilot-a1
CACHE_ROOT="$( $PY -I -B -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)' )"
MODEL_ID="$( $PY -I -B -c 'from part85.checkpoints import resolve_checkpoint; print(resolve_checkpoint("instruct")["repo_id"])' )"
rm -rf "$CACHE_ROOT/models--${MODEL_ID//\//--}"
```

The Instruct token digest must match the registered SHA above. Stop if it does not. Before downloading
`pretrain_4T`, run pilot preflight. It verifies the Instruct step files and token digest.

```bash
$PY -I -B part85/handoff85.py preflight --checkpoint pretrain_4T --phase pilot --attempt 1 --results-root results/part85
SNAPSHOT_DIR="$( $PY -I -B -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("pretrain_4T"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))' )"
$PY -I -B part85/step0_weights.py --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" \
  --reference results/part85/pilot-a1/instruct/step0.json --results-root results/part85 \
  --push-branch part85-run/pilot-a1
$PY -I -B part85/step1_acts.py --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" --device cpu \
  --reference-acts results/part85/pilot-a1/instruct/step1.json \
  --sequence-ids results/part85/pilot-a1/neutral-token-ids.json --results-root results/part85 \
  --push-branch part85-run/pilot-a1
$PY -I -B part85/part85.py --mode pilot --checkpoint pretrain_4T --snapshot-dir "$SNAPSHOT_DIR" \
  --results-root results/part85 --device cpu --push-branch part85-run/pilot-a1
MODEL_ID="$( $PY -I -B -c 'from part85.checkpoints import resolve_checkpoint; print(resolve_checkpoint("pretrain_4T")["repo_id"])' )"
rm -rf "$CACHE_ROOT/models--${MODEL_ID//\//--}"
```

The cleanup removes the Hugging Face cache directory `models--<org>--<name>`, which frees the downloaded model.
Deleting `SNAPSHOT_DIR` would remove only its snapshot link. If step 0, step 1, or a run cell already exists with the
same checkpoint, code, and clamp provenance, its command skips the computation and retries the push.

Assemble the pilot branch locally. Prepare attempt 1, keep the answer key away from the blind reader, label all 96
items including `responsive`, then score:

```bash
$PY -I -B part85/handoff85.py assemble --branches part85-run/pilot-a1 --out results/part85
$PY -I -B part85/score85.py prepare --mode pilot --attempt 1 --results-root results/part85 \
  --items results/part85/pilot-items.jsonl --key results/part85/pilot-key.json
# Reader writes results/part85/pilot-labels.jsonl.
$PY -I -B part85/score85.py score --mode pilot --key results/part85/pilot-key.json \
  --items results/part85/pilot-items.jsonl --labels results/part85/pilot-labels.jsonl \
  --out results/part85/pilot-score.json
```

The gate counts end-of-pretraining unclamped answers that are both `OK` and responsive. If it requests the one
registered frame rerun, do that attempt before starting main. If attempt 2 has `quality_failed_after_rerun`, stop the
experiment.

## Optional attempt 2

Write the revised template to `results/part85/pilot-a2/pilot-frame-v2.txt`. Start a fresh session cloned from
`CODE_COMMIT`; restore `part85-run/pilot-a2` before writing. Do not base the attempt-2 branch on the attempt-1 branch.
Fetch the attempt-1 branch as read-only input, copy its Instruct steps and token file into the attempt-2 owned
directory, and keep the new branch's result tree limited to `pilot-a2/`:

```bash
PY=/home/elisepc/neuron-probe/.venv/bin/python
git checkout --detach "$CODE_COMMIT"
export PART85_CODE_COMMIT="$CODE_COMMIT"
git fetch origin refs/heads/part85-run/pilot-a1
$PY -I -B part85/handoff85.py restore --branch part85-run/pilot-a2 --results-root results/part85
mkdir -p results/part85/pilot-a2/instruct
git show FETCH_HEAD:results/part85/pilot-a1/instruct/step0.json > results/part85/pilot-a2/instruct/step0.json
git show FETCH_HEAD:results/part85/pilot-a1/instruct/step1.json > results/part85/pilot-a2/instruct/step1.json
git show FETCH_HEAD:results/part85/pilot-a1/neutral-token-ids.json > results/part85/pilot-a2/neutral-token-ids.json
```

Run step 2 for Instruct and `pretrain_4T` only. Restore/check preflight and download each pinned snapshot before use.
Use the revised template on both commands. Delete the model's Hugging Face `models--<org>--<name>` cache directory
after each checkpoint finishes.

```bash
PY=/home/elisepc/neuron-probe/.venv/bin/python
CACHE_ROOT="$( $PY -I -B -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)' )"
$PY -I -B part85/handoff85.py preflight --checkpoint instruct --phase pilot --attempt 2 --results-root results/part85
SNAPSHOT_DIR="$( $PY -I -B -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("instruct"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))' )"
$PY -I -B part85/part85.py --mode pilot --checkpoint instruct --attempt 2 \
  --plain-template-file results/part85/pilot-a2/pilot-frame-v2.txt --snapshot-dir "$SNAPSHOT_DIR" \
  --results-root results/part85 --device cpu --push-branch part85-run/pilot-a2
MODEL_ID="$( $PY -I -B -c 'from part85.checkpoints import resolve_checkpoint; print(resolve_checkpoint("instruct")["repo_id"])' )"
rm -rf "$CACHE_ROOT/models--${MODEL_ID//\//--}"
$PY -I -B part85/handoff85.py preflight --checkpoint pretrain_4T --phase pilot --attempt 2 --results-root results/part85
SNAPSHOT_DIR="$( $PY -I -B -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("pretrain_4T"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))' )"
$PY -I -B part85/part85.py --mode pilot --checkpoint pretrain_4T --attempt 2 \
  --plain-template-file results/part85/pilot-a2/pilot-frame-v2.txt --snapshot-dir "$SNAPSHOT_DIR" \
  --results-root results/part85 --device cpu --push-branch part85-run/pilot-a2
MODEL_ID="$( $PY -I -B -c 'from part85.checkpoints import resolve_checkpoint; print(resolve_checkpoint("pretrain_4T")["repo_id"])' )"
rm -rf "$CACHE_ROOT/models--${MODEL_ID//\//--}"
```

Run the same command for `pretrain_4T` after pilot preflight. Assemble `part85-run/pilot-a2`, prepare attempt 2 under
`results/part85/attempt-2/`, and score it. Keep attempt 1's files unchanged. Copy the accepted score to
`results/part85/pilot-score.json` for the reference-pinning step.

## Pin references after the accepted pilot

After the accepted pilot is scored, assemble its result branch into `results/part85`. The accepted score must be at
`results/part85/pilot-score.json`; for an attempt-2 pass, the custom template remains at
`results/part85/pilot-a2/pilot-frame-v2.txt`. Run `refs-manifest`. It copies the Instruct step files, token IDs,
accepted score, and (when used) custom template into `part85/references/`, then records each SHA-256 in
`part85/references.json`.

```bash
$PY -I -B part85/handoff85.py assemble --branches part85-run/pilot-a1 --out results/part85
# For an accepted attempt 2, assemble pilot-a2 as well; keep the attempt-1 artifacts for the shared references.
$PY -I -B part85/handoff85.py refs-manifest --results-root results/part85 --out part85/references.json
```

Commit `part85/references.json` and every file under `part85/references/`. This is the **references commit**. Main
sessions clone this later commit, not `CODE_COMMIT`. This avoids the circular pin: pilot uses code-only; then the
researcher pins references and uses that new commit for main. Each session's `restore` fetches and checks out its own
branch tip before writing. A branch that is absent is normal on its first run.

## Main: one checkpoint per session

Each session clones the references commit, restores only its own `part85-run/main-<key>` branch, and runs preflight
before downloading a snapshot. Preflight checks every manifest SHA-256, the pinned neutral-token digest, and any
step files already present in `results/part85/main/<key>/`. Missing step files are expected before that checkpoint's
step commands run. A present file must identify this checkpoint and its registered commit.

```bash
git clone <REPO_URL> part85-<key>
cd part85-<key>
PY=/home/elisepc/neuron-probe/.venv/bin/python
CACHE_ROOT="$( $PY -I -B -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)' )"
git checkout --detach "$REFERENCES_COMMIT"
export PART85_CODE_COMMIT="$REFERENCES_COMMIT"
$PY -I -B part85/handoff85.py restore --branch part85-run/main-<key> --results-root results/part85
$PY -I -B part85/handoff85.py preflight --checkpoint <key> --phase main --results-root results/part85
```

Run step 0 and step 1 against the pinned Instruct references. Instruct's neutral IDs are read from
`part85/references/neutral-token-ids.json`; outputs go only to the main checkpoint directory. If a step output was
restored and has matching provenance, the step skips it and retries its push.

```bash
SNAPSHOT_DIR="$( $PY -I -B -c 'from part85.checkpoints import resolve_checkpoint; from huggingface_hub import snapshot_download; cp=resolve_checkpoint("<key>"); print(snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"]))' )"
$PY -I -B part85/step0_weights.py --checkpoint <key> --snapshot-dir "$SNAPSHOT_DIR" \
  --reference part85/references/instruct/step0.json --results-root results/part85 \
  --push-branch part85-run/main-<key>
$PY -I -B part85/step1_acts.py --checkpoint <key> --snapshot-dir "$SNAPSHOT_DIR" --device cpu \
  --reference-acts part85/references/instruct/step1.json \
  --sequence-ids part85/references/neutral-token-ids.json --results-root results/part85 \
  --push-branch part85-run/main-<key>
```

Run step 2 with the accepted pinned pilot gate. For `plain_frame_pass`, every registered checkpoint runs plain cells
and SFT, DPO, and Instruct also run chat cells. For `chat_only_posttrained`, run SFT, DPO, and Instruct chat cells
only; pretraining checkpoints are out of scope. If the gate requests amendment or reports quality failure, do not
start main. When the pilot used a revised frame, pass the pinned template in main mode too.

```bash
$PY -I -B part85/part85.py --mode main --checkpoint <key> --snapshot-dir "$SNAPSHOT_DIR" \
  --pilot-score part85/references/pilot-score.json --results-root results/part85 \
  --device cpu --push-branch part85-run/main-<key>
# Add this argument only when the accepted pilot used a revised frame:
# --plain-template-file part85/references/pilot-frame-v2.txt
MODEL_ID="$( $PY -I -B -c 'from part85.checkpoints import resolve_checkpoint; print(resolve_checkpoint("<key>")["repo_id"])' )"
rm -rf "$CACHE_ROOT/models--${MODEL_ID//\//--}"
```

Each completed answer cell pushes only its complete checkpoint directory. If a push fails after a file was written,
the next run validates completed cells and retries pushing even when no new files were staged. Push verifies that the
remote tip equals local `HEAD`; retries are bounded and use `--force-with-lease` against that branch's observed tip.

## Score and assemble

Run assembly from a coordinator clone with the references commit checked out. `assemble` fetches each named result branch
from `origin`, so that clone does not need to have run or restored those branches previously.

For a plain-frame pass, assemble the accepted pilot branch and all twelve main branches. For a chat-only pilot,
assemble the pilot branch and the three in-scope main branches. Assembly copies each branch's one owned directory;
it refuses foreign result paths, mismatched branch names, missing claimed answer files, and malformed ownership.

```bash
$PY -I -B part85/handoff85.py assemble --branches \
  part85-run/pilot-a1 part85-run/main-pretrain_21B part85-run/main-pretrain_210B \
  part85-run/main-pretrain_1T part85-run/main-pretrain_2T part85-run/main-pretrain_3T \
  part85-run/main-pretrain_4T part85-run/main-midtrain_13B part85-run/main-midtrain_26B \
  part85-run/main-midtrain_51B part85-run/main-sft part85-run/main-dpo part85-run/main-instruct \
  --out results/part85
$PY -I -B part85/score85.py prepare --mode main --results-root results/part85 \
  --items results/part85/main-items.jsonl --key results/part85/main-key.json
# Two readers label the blinded items and write reader-1.jsonl and reader-2.jsonl.
$PY -I -B part85/score85.py score --mode main --key results/part85/main-key.json \
  --items results/part85/main-items.jsonl --labels results/part85/reader-1.jsonl \
  --labels2 results/part85/reader-2.jsonl --results-root results/part85 \
  --out results/part85/main-score.json
```

For chat-only scope, replace the branch list with `part85-run/pilot-a1 part85-run/main-sft part85-run/main-dpo
part85-run/main-instruct`. Main scoring checks the pinned template hash as well as checkpoint identities, counts,
question/sample pairs, and registered seeds. `prepare` refuses existing item/key files. `score` refuses to replace
an existing output; pass `--force` only when intentionally replacing it. Use `--partial` only for an explicitly
incomplete analysis; it is the only mode that permits `NOT READ` verdicts.
