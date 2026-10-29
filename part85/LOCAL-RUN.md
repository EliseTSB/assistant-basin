# Part 85 local run on the A770 (replaces CLOUD-RUN.md for this run)

Written by Claude (Opus 5.5), 2026-10-28, for decision `part85-after-r4-2026-10-28` option 1. `CLOUD-RUN.md` stays as
the record of the cloud path; nothing in it applies here. No branches, no pushes: results live on the GPU box under
`C:\Users\alexa\neuron-probe-gpu\results\part85\` and are copied into the repository after scoring.

## 0. Stage and gate

Commit the Part 85 source and tests in the repository first. From that checkout, get the full commit with
`git rev-parse HEAD` and write its 40-hex output, on one line, to the staged `part85\CODE_COMMIT`. The staged folder is
not a git checkout; `handoff85.py` reads this file to stamp results with the code commit. Copy `part85\` (every `.py`,
`passages.json`, `CODE_COMMIT`, and `references\` if present) to `C:\Users\alexa\neuron-probe-gpu\part85\`, and the
repository-level `observe.py`, `neuron_report.py`, and `PART85-DESIGN.md` beside it in
`C:\Users\alexa\neuron-probe-gpu\`. Production and audit read the registered plain frame from that design file.
`observe.py` must stay byte-identical to the one Part 86 staged (`sha256sum` both); `observe85.py` wraps it and adds
raw token IDs and forward-pass records.

Keep a full Git checkout of the committed repository accessible to the Windows process as `G`. `provenance85.py`
uses `git show` from `G`; the staged GPU folder has no `.git` and is not an authority source. `L` below is the job
runner folder containing its `step-N.log` files. The `--commit` for provenance is the full commit that contains the
currently registered design and `part85\references.json`; `CODE_COMMIT` remains the source-code commit stamped into
production rows.

From `C:\Users\alexa\neuron-probe-gpu`, with the Windows venv `C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe`:

```text
python -I -B part85\test_part85.py
python -I -B part85\test_download85.py
python -I -B part85\mutate85.py --jobs 2
```

Proceed only when the suites print `OK` and the mutation line says `44/44 killed`.

## 1. Download (overnight, Elise's yes 2026-10-28 10:51)

Instruct is already cached at `E:\neuron-probe\hf\hub\models--allenai--OLMo-2-0425-1B-Instruct\snapshots\48d788eca847d4d7548f375ad03d3c9312f6139e`.
The other eleven are 59.5 GB (measured with `--dry-run` 2026-10-28: nine 5.95 GB two-shard safetensors checkpoints and
SFT and DPO at 2.98 GB as `pytorch_model.bin`), about 3.3 h at 5 MB/s. Run it from a Windows prompt, not from WSL, and
not inside the GPU job runner (it is network-bound and needs the Hub, while the runner sets `HF_HUB_OFFLINE=1`):

```text
python -I -B part85\download85.py --dry-run
python -I -B part85\download85.py --rate 5000000
```

It is resumable: rerun the same line after any interruption. It refuses on a hash mismatch (exit 1, the bad bytes kept
beside the file as `.bad-<sha prefix>`); do not continue past a refusal. It finishes by writing
`part85\commits.local.json` (all twelve checkpoint commits, resolved through the Hub API and checked against the
registered prefixes by `write_local_commits()`), which `resolve_checkpoint` reads when the runner is offline. Preserve
both `CODE_COMMIT` and `commits.local.json` when restaging `part85\` after the download.

## 2. Steps through the job runner

The runner (`job.py`) reads `steps.txt` one line per step, prepends the interpreter, runs each from the job folder with
`HF_HOME=E:\neuron-probe\hf` and `HF_HUB_OFFLINE=1`, and stops at the first nonzero exit. `steps.txt` is CRLF: read
it with `newline=''`. Paths are bare, no quotes. **Never delete a finished line: replace it with `-c pass`** so the step
numbers (and the logs `step-N.log`) keep pointing at the same work. Before every model load the free Windows RAM must
be at least **12 GB** (`drop_caches` in WSL first if needed); the runner's own 2 GB guard is a last stop, not the floor.
Each scheduled task is killed at six hours; a group below must project under that.

Every model-loading step runs under the observer. `S` below is
`E:\neuron-probe\hf\hub\models--allenai--OLMo-2-0425-1B<suffix>\snapshots\<full commit>` for that checkpoint (suffix
empty, `-SFT`, `-DPO` or `-Instruct`; full commits in `part85\commits.local.json`); `R` is `results\part85`. Step 0/1
calibration is stored under `R\main\KEY\` and its neutral sequence file is `R\neutral-token-ids.json`. Pilot answers
are under `R\pilot-aN\KEY\`; their replay reads calibration records from `R\main\KEY\observe\step1.jsonl`. Main
answers and calibration for each checkpoint are both under `R\main\KEY\`.

Install the missing WikiText dependency into the Windows venv before creating neutral sequences:

```text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -m pip install datasets
```

### 2a. Steps 0 and 1 for all twelve checkpoints, Instruct first (round-4 #7: these run whatever the pilot decides)

```text
part85\step0_weights.py --checkpoint instruct --snapshot-dir S --results-root R
part85\step1_acts.py --checkpoint instruct --snapshot-dir S --device xpu --results-root R --sequence-ids R\neutral-token-ids.json
```

Instruct runs without `--reference` (round-4 #3). Its step 1 builds `R\neutral-token-ids.json` from WikiText-103 and
must reproduce the pinned digest. That one step needs the network (the runner is offline), so run Instruct's two lines
from a Windows prompt with `set HF_HOME=E:\neuron-probe\hf` after the download has written `commits.local.json`;
every later step finds the token file and runs in the job runner. Then for each other key (pretrain_21B ... dpo):

```text
part85\step0_weights.py --checkpoint KEY --snapshot-dir S --results-root R --reference R\main\instruct\step0.json
part85\step1_acts.py --checkpoint KEY --snapshot-dir S --device xpu --results-root R --sequence-ids R\neutral-token-ids.json --reference-acts R\main\instruct\step1.json
```

Audit every step 1 before anything reads it (Instruct first; the others name Instruct's audited file):

```text
part85\audit85.py R\main\KEY --checkpoint KEY --step1 --tokenizer S --token-ids R\neutral-token-ids.json --reference R\main\instruct\step1.json
```

Require `AUDIT85 KEY step1: PASS` for all twelve. Step 0 is a pure weight read with no forward pass; check it
independently with `part85\check0.py S --reference <Instruct S>` (cosines must match step 0 exactly; the readout
variant `post_and_final` matches Amendment 1.2).

### 2a.1 Commit pre-pilot references

Pilot replay also requires a committed references manifest, even though the accepted pilot score does not exist yet.
Before pilot generation, copy the audited Instruct calibration and neutral token IDs into the repository checkout:

```text
Copy-Item R\main\instruct\step0.json part85\references\instruct\step0.json
Copy-Item R\main\instruct\step1.json part85\references\instruct\step1.json
Copy-Item R\neutral-token-ids.json part85\references\neutral-token-ids.json
```

Create `part85\references.json` with those three files' SHA-256 values (the manifest has `schema_version: 2` and a
`files` list of `{path, sha256}` entries using the three repository-relative paths above). Commit this manifest,
the pinned files, and the current `PART85-DESIGN.md` before pilot replay. This bootstrap manifest is for pilot
provenance; after the pilot is accepted, section 2c replaces it with the final manifest including `pilot-score.json`.
From the repository root, this standard-library command writes the bootstrap manifest:

```text
python -I -B -c "import hashlib,json,pathlib; names=['part85/references/instruct/step0.json','part85/references/instruct/step1.json','part85/references/neutral-token-ids.json']; root=pathlib.Path('.'); files=[{'path':n,'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest()} for n in names]; (root/'part85/references.json').write_text(json.dumps({'schema_version':2,'files':files},indent=1,sort_keys=True)+'\n',encoding='utf-8')"
```

Resolve the new commit's full id with `git -C G rev-parse HEAD`, set it as `REFERENCES_COMMIT`, and copy the committed
manifest and directory to the staged GPU folder before proceeding:

```text
New-Item -ItemType Directory -Force C:\Users\alexa\neuron-probe-gpu\part85\references | Out-Null
Copy-Item part85\references.json C:\Users\alexa\neuron-probe-gpu\part85\references.json
Copy-Item part85\references\* C:\Users\alexa\neuron-probe-gpu\part85\references\ -Recurse -Force
```

For each checkpoint, after its step-1 audit has printed PASS, run provenance once before that checkpoint's replay.
Use `AUDIT_LOG_KEY` for the `step-N.log` containing that exact audit line; provenance checks the full repository blobs,
the snapshot files against Hugging Face metadata, and the byte hashes of `step1.json` and `observe\step1.jsonl`.
Run this outside the offline job runner so the Hub metadata request can complete:

```text
part85\provenance85.py --repo-root G --commit REFERENCES_COMMIT --checkpoint KEY --snapshot S_KEY --step1-dir R\main\KEY --audit-log L\AUDIT_LOG_KEY --out R\provenance\KEY.json
```

### 2b. Pilot (plain frame, end of pretraining and Instruct, 96 answers)

```text
part85\part85.py --mode pilot --attempt 1 --checkpoint pretrain_4T --snapshot-dir S --results-root R --device xpu --step1 R\main\pretrain_4T\step1.json
part85\part85.py --mode pilot --attempt 1 --checkpoint instruct --snapshot-dir S --results-root R --device xpu --step1 R\main\instruct\step1.json
part85\audit85.py R\pilot-a1\pretrain_4T --checkpoint pretrain_4T --mode pilot --attempt 1 --tokenizer S --step1-json R\main\pretrain_4T\step1.json
part85\audit85.py R\pilot-a1\instruct --checkpoint instruct --mode pilot --attempt 1 --tokenizer S --step1-json R\main\instruct\step1.json
```

`--step1` points both attempts at the one audited step 1. Require both audit passes before replay. Do not open pilot
answer JSONL files or run `score85.py prepare`/`score` until both checkpoints print `REPLAY85: PASS` and both negative
controls confirm changed answers. Only then read and score the pilot to choose its gate.

Pilot answers live under `R\pilot-a1\KEY`; step-1 calibration records live under `R\main\KEY\observe\step1.jsonl`.
Run each full replay from `steps.txt` (it must exit 0 and print PASS). `S_PRETRAIN_4T` and `S_INSTRUCT` are each
checkpoint's full snapshot path defined above:

```text
part85\replay85.py --cp-dir R\pilot-a1\pretrain_4T --step1-dir R\main\pretrain_4T --checkpoint pretrain_4T --mode pilot --snapshot S_PRETRAIN_4T --device xpu --references part85\references.json --provenance R\provenance\pretrain_4T.json --code-commit part85\CODE_COMMIT
part85\replay85.py --cp-dir R\pilot-a1\instruct --step1-dir R\main\instruct --checkpoint instruct --mode pilot --snapshot S_INSTRUCT --device xpu --references part85\references.json --provenance R\provenance\instruct.json --code-commit part85\CODE_COMMIT
```

Run the negative controls from a Windows prompt, outside `steps.txt`: the current replay returns 3 when a negative
control succeeds, and `job.py` stops on any nonzero exit. Require its message to say clamped answers changed; stop if it
prints `NOTHING CHANGED`. The replay itself selects the registered clamped answers for this control.

```text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part85\replay85.py --cp-dir R\pilot-a1\pretrain_4T --step1-dir R\main\pretrain_4T --checkpoint pretrain_4T --mode pilot --snapshot S_PRETRAIN_4T --perturb-value 1.01 --device xpu --references part85\references.json --provenance R\provenance\pretrain_4T.json --code-commit part85\CODE_COMMIT
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part85\replay85.py --cp-dir R\pilot-a1\instruct --step1-dir R\main\instruct --checkpoint instruct --mode pilot --snapshot S_INSTRUCT --perturb-value 1.01 --device xpu --references part85\references.json --provenance R\provenance\instruct.json --code-commit part85\CODE_COMMIT
```

If gate 1 fails, register the revised frame in `PART85-DESIGN.md` as the Amendment 3 line, with the SHA-256 of its
UTF-8 bytes using LF line endings. Commit that design change **before the rerun**, then copy the committed design file
beside the staged `part85\` directory on the GPU box, including the current references manifest and pinned files in
that commit. Set `REFERENCES_COMMIT` to this new commit and copy the committed references files to the GPU staging
folder. Regenerate each checkpoint's provenance using that design-and-references commit before replay. Production and
audit read attempt 2's frame from that line.
Run production and audit for both pilot checkpoints:

```text
part85\part85.py --mode pilot --attempt 2 --checkpoint pretrain_4T --snapshot-dir S_PRETRAIN_4T --results-root R --device xpu --step1 R\main\pretrain_4T\step1.json
part85\part85.py --mode pilot --attempt 2 --checkpoint instruct --snapshot-dir S_INSTRUCT --results-root R --device xpu --step1 R\main\instruct\step1.json
part85\audit85.py R\pilot-a2\pretrain_4T --checkpoint pretrain_4T --mode pilot --attempt 2 --tokenizer S_PRETRAIN_4T --step1-json R\main\pretrain_4T\step1.json
part85\audit85.py R\pilot-a2\instruct --checkpoint instruct --mode pilot --attempt 2 --tokenizer S_INSTRUCT --step1-json R\main\instruct\step1.json
```

Replay both attempt-2 runs and run the negative controls outside `steps.txt`. Do not open either answer file until all
four commands pass. Replay reads the registered attempt and frame; there is no per-run frame argument.

```text
part85\replay85.py --cp-dir R\pilot-a2\pretrain_4T --step1-dir R\main\pretrain_4T --checkpoint pretrain_4T --mode pilot --attempt 2 --snapshot S_PRETRAIN_4T --device xpu --references part85\references.json --provenance R\provenance\pretrain_4T.json --code-commit part85\CODE_COMMIT
part85\replay85.py --cp-dir R\pilot-a2\instruct --step1-dir R\main\instruct --checkpoint instruct --mode pilot --attempt 2 --snapshot S_INSTRUCT --device xpu --references part85\references.json --provenance R\provenance\instruct.json --code-commit part85\CODE_COMMIT
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part85\replay85.py --cp-dir R\pilot-a2\pretrain_4T --step1-dir R\main\pretrain_4T --checkpoint pretrain_4T --mode pilot --attempt 2 --snapshot S_PRETRAIN_4T --perturb-value 1.01 --device xpu --references part85\references.json --provenance R\provenance\pretrain_4T.json --code-commit part85\CODE_COMMIT
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part85\replay85.py --cp-dir R\pilot-a2\instruct --step1-dir R\main\instruct --checkpoint instruct --mode pilot --attempt 2 --snapshot S_INSTRUCT --perturb-value 1.01 --device xpu --references part85\references.json --provenance R\provenance\instruct.json --code-commit part85\CODE_COMMIT
```

### 2c. Pin the references

`part85\handoff85.py refs-manifest --results-root R --out part85\references.json` copies Instruct's calibration from
`R\main\instruct\step0.json` and `step1.json`, neutral IDs from `R\neutral-token-ids.json`, and the accepted pilot score
from `R\pilot-score.json` into `part85\references\` with their hashes. For an attempt-2 pilot, the frame export is
written from the committed design line's JSON string to `R\pilot-a2\pilot-frame-v2.txt` using UTF-8 and LF line
endings. The export is canonicalized and checked against the pilot score; it does not define the production frame.
Commit the manifest and files as the references commit, then restage `part85\` and `PART85-DESIGN.md` while keeping
`CODE_COMMIT` and the downloaded `commits.local.json`. Resolve the references commit id with `git -C G rev-parse HEAD`
and use it as `REFERENCES_COMMIT` for main provenance records.

### 2d. Main run

The main production and audit commands read the plain frame from the committed `PART85-DESIGN.md` and verify that its
hash matches the accepted pilot. The main scope comes from the accepted pilot score. For the `plain_frame_pass` gate,
run these for all twelve keys:

```text
part85\part85.py --mode main --checkpoint KEY --snapshot-dir S --results-root R --device xpu --pilot-score part85\references\pilot-score.json
part85\audit85.py R\main\KEY --checkpoint KEY --mode main --tokenizer S
```

For the `chat_only_posttrained` gate, use the same production command for SFT, DPO and Instruct, and add `--chat-only`
to each audit command:

```text
part85\part85.py --mode main --checkpoint KEY --snapshot-dir S --results-root R --device xpu --pilot-score part85\references\pilot-score.json
part85\audit85.py R\main\KEY --checkpoint KEY --mode main --tokenizer S --chat-only
```

Measured rate first: run Instruct alone and project the rest before grouping. Require `AUDIT85 KEY main: PASS` for every
key in the gate's scope before preparing the main items.

### 2e. Replay before reading

Pilot and main answers are not read until full replay passes and the perturbed negative control changes an answer in a
clamped cell. Each main run's answer records and step-1 observer file both live under `R\main\KEY`. For every key in
scope, `S_KEY` is that checkpoint's full snapshot path. Before replay, create its provenance record with the final
references commit and the log containing the checkpoint's `AUDIT85 KEY step1: PASS` line, as in section 2a.1. Replay
checks `part85\references.json`, the design, the full snapshot hashes, and the audited step-1 hashes against that
record:

```text
part85\replay85.py --cp-dir R\main\KEY --step1-dir R\main\KEY --checkpoint KEY --mode main --snapshot S_KEY --device xpu --references part85\references.json --provenance R\provenance\KEY.json --code-commit part85\CODE_COMMIT
part85\replay85.py --cp-dir R\main\KEY --step1-dir R\main\KEY --checkpoint KEY --mode main --snapshot S_KEY --perturb-value 1.01 --device xpu --references part85\references.json --provenance R\provenance\KEY.json --code-commit part85\CODE_COMMIT
```

For `plain_frame_pass`, run both commands for all twelve keys. For `chat_only_posttrained`, run them for SFT, DPO and
Instruct only. The manifest supplies the selection in either branch, so these commands carry no scope or frame option.
Run the first command for each key from `steps.txt`; each must print `REPLAY85: PASS` for every registered answer. Run
the second command from the Windows prompt, outside `steps.txt`, because a successful negative control currently exits
3; require it to report changed clamped answers. Only then run `score85.py prepare` or read main answers.

## 3. Blind reading and publication (round-4 #8)

1. After the replay gate, `score85.py prepare` writes items and the key; move the key to `R\private\` at once.
2. Two independent Claude Haiku 5.5 readers, each in a tool-less session, get the registered Part 26.8 rubric plus the
   Part 84 company-name note (as registered) and their own item chunks; neither sees the key or the other's labels.
3. Commit both label files in the repository (reader 1, reader 2) before the key is opened.
4. Only then run `score85.py score` with both label files, and commit the score.
5. Publish (push) the references commit and the label commit before any result is written up, and check from a fresh
   clone that both commits exist on the remote.

**Limit (Claude, 2026-10-28):** replay certifies a run against inputs pinned by a separate provenance record; it does not protect against an operator replacing that record. Elise called the checking over-engineered at 18:51; no more review rounds on it.
