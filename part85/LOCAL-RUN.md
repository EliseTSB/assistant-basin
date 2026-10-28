# Part 85 local run on the A770 (replaces CLOUD-RUN.md for this run)

Written by Claude (Opus 5.5), 2026-10-28, for decision `part85-after-r4-2026-10-28` option 1. `CLOUD-RUN.md` stays as
the record of the cloud path; nothing in it applies here. No branches, no pushes: results live on the GPU box under
`C:\Users\alexa\neuron-probe-gpu\results\part85\` and are copied into the repository after scoring.

## 0. Stage and gate

Copy `part85\` (every `.py`, `passages.json`, `references\` if present) to `C:\Users\alexa\neuron-probe-gpu\part85\`, and
the repository-level `observe.py` and `neuron_report.py` beside it in `C:\Users\alexa\neuron-probe-gpu\`. `observe.py`
must stay byte-identical to the one Part 86 staged (`sha256sum` both); `observe85.py` wraps it and adds raw token IDs and
forward-pass records. Write the code commit (the full 40-hex SHA of the staged `part85\` tree) into
`part85\CODE_COMMIT`: the job runner starts each step with its own environment, so `PART85_CODE_COMMIT` cannot be set
per line, and the staged folder is not a git checkout.

From `C:\Users\alexa\neuron-probe-gpu`, with the Windows venv `C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe`:

```text
python -I -B part85\test_part85.py
python -I -B part85\test_download85.py
python -I -B part85\mutate85.py --jobs 2
```

Proceed only when the suites print `OK` and the mutation line says `42/42 killed`.

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
`part85\commits.local.json` (each checkpoint's full commit), which `resolve_checkpoint` reads when the runner is offline.

## 2. Steps through the job runner

The runner (`job.py`) reads `steps.txt` one line per step, prepends the interpreter, runs each from the job folder with
`HF_HOME=E:\neuron-probe\hf` and `HF_HUB_OFFLINE=1`, and stops at the first nonzero exit. `steps.txt` is CRLF: read
it with `newline=''`. Paths are bare, no quotes. **Never delete a finished line: replace it with `-c pass`** so the step
numbers (and the logs `step-N.log`) keep pointing at the same work. Before every model load the free Windows RAM must
be at least **12 GB** (`drop_caches` in WSL first if needed); the runner's own 2 GB guard is a last stop, not the floor.
Each scheduled task is killed at six hours; a group below must project under that.

Every model-loading step runs under the observer. `S` below is
`E:\neuron-probe\hf\hub\models--allenai--OLMo-2-0425-1B<suffix>\snapshots\<full commit>` for that checkpoint (suffix
empty, `-SFT`, `-DPO` or `-Instruct`; full commits in `part85\commits.local.json`); `R` is `results\part85`.

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

### 2b. Pilot (plain frame, end of pretraining and Instruct, 96 answers)

```text
part85\part85.py --mode pilot --attempt 1 --checkpoint pretrain_4T --snapshot-dir S --results-root R --device xpu --step1 R\main\pretrain_4T\step1.json
part85\part85.py --mode pilot --attempt 1 --checkpoint instruct --snapshot-dir S --results-root R --device xpu --step1 R\main\instruct\step1.json
part85\audit85.py R\pilot-a1\pretrain_4T --checkpoint pretrain_4T --mode pilot --attempt 1 --tokenizer S --step1-json R\main\pretrain_4T\step1.json
part85\audit85.py R\pilot-a1\instruct --checkpoint instruct --mode pilot --attempt 1 --tokenizer S --step1-json R\main\instruct\step1.json
```

`--step1` points both attempts at the one audited step 1 (round-4 #4: attempt 2 needs no copies and no fetched
branches). Require both `PASS` before blinding. Then `score85.py prepare` and `score` exactly as in `CLOUD-RUN.md`'s
pilot section, with `--results-root R`. Attempt 2, only if gate 1 fails: write the amended frame to
`R\pilot-a2\pilot-frame-v2.txt` first, rerun the two `part85.py` lines with `--attempt 2 --plain-template-file` that file,
and audit with `--attempt 2 --template-file` that file.

### 2c. Pin the references

`part85\handoff85.py refs-manifest --results-root R --out part85\references.json` copies Instruct's step files, the token
ids, the accepted pilot score (and template) into `part85\references\` with their hashes. Commit them in the repository
as the references commit and restage `part85\` (with its `CODE_COMMIT`) before the main run.

### 2d. Main run

```text
part85\part85.py --mode main --checkpoint KEY --snapshot-dir S --results-root R --device xpu --pilot-score part85\references\pilot-score.json
part85\audit85.py R\main\KEY --checkpoint KEY --mode main --tokenizer S
```

All twelve keys for a plain-frame pass (192 answers each, 384 for SFT, DPO and Instruct); for `chat_only_posttrained`
only SFT, DPO and Instruct with `--chat-only` on the audit. Measured rate first: run Instruct alone and project the
rest before grouping. Require `AUDIT85 KEY main: PASS` for every key before preparing the main items.

### 2e. Replay before reading

Claude writes `replay85.py` from the design (not in this fold). Main results are read only after it prints PASS on every
answer, with its perturbed negative control last in the same `steps.txt` (it must exit 1 with answers changed).

## 3. Blind reading and publication (round-4 #8)

1. `score85.py prepare` writes items and the key; move the key to `R\private\` at once.
2. Two independent Claude Haiku 5.5 readers, each in a tool-less session, get the registered Part 26.8 rubric plus the
   Part 84 company-name note (as registered) and their own item chunks; neither sees the key or the other's labels.
3. Commit both label files in the repository (reader 1, reader 2) before the key is opened.
4. Only then run `score85.py score` with both label files, and commit the score.
5. Publish (push) the references commit and the label commit before any result is written up, and check from a fresh
   clone that both commits exist on the remote.
