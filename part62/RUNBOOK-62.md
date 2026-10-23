# Part 62 runbook

Part 62 compares a finer dose ladder for the Qwen3.5-4B and OLMo-2-1B target
units while clamping both prompt and generated-token positions separately. The
registered ladder, questions, model revisions, and thresholds are constants in
`run62.py` and `analyze62.py`. The generation matrix is 2 models × 10 doses × 3
phases × 3 questions × 8 samples, plus 2 models × 2 control doses × 3 questions
× 8 samples: 1,536 answers total. The two models are loaded sequentially, in
fp32, to keep the peak memory below the combined model footprint.
The same-layer control units were selected before the run with seed 62062 and
are fixed in the generator: Qwen L22 unit 8870 and OLMo L10 unit 7884. The
generator checks the registered widths (9216 and 8192) before using them.
Part 62's phase definitions follow Amendment 1: `prompt` clamps every prompt
position except the last; `gen` clamps the last prompt position and every
decode step; `both` clamps all positions. The hook wrapper is in `run62.py` and
uses the shared `neuron_report.NeuronHook` without editing it.

## Inputs

The project root must contain the Part 26 records in `results/part26/`, the
Part 33 record at `results/part33-r2/natural-firing.json`, and locally cached
model snapshots at the pinned revisions in `run62.py`. The generator sets the
Hugging Face offline flags through the Part 26 loader. It does not fetch model
files. OLMo's alien-frame maximum is recomputed on the 30 selected Part 33
passages with Part 26's tokenizer and `scale_frames` capture method; Qwen's
maximum is recorded in the same way and the run stops unless it is within 0.1%
of the registered 24.69 reference. The dose itself remains in Part 26
Pile-maximum units. Generation runs in fp32, but the OLMo Pile maximum comes
from a bf16 Part 26 record; each manifest records the anchor's source dtype so
the fp32 generation claim is not mistaken for an fp32 remeasurement of that
anchor.

## Generate and archive

From the project checkout root on the rented GPU box, run:

```sh
cd /root/assistant-basin
bash part62/run62.sh
```

The runner itself downloads both pinned snapshots into `/root/hf/hub` before
setting offline mode; no model revision is selected at runtime.

The runner writes one atomic JSON file per model, dose, phase, question, or
scaled-control cell under `results/part62/`. It writes and updates
`part62-manifest.json` after each successful cell. A complete manifest records
each condition seed, target and control set values, both down-projection
column norms, the alien-frame maximum, and the generated answers. For each
model, dose, question, and sample index, the `both`, `gen`, and `prompt` phases
share one recorded seed; different doses use independent derived seeds. Samples
are generated one at a time to preserve this pairing. After a kill,
rerun the same command into the same output directory. Resume checks the run
fingerprint and every existing cell fingerprint and file hash; a different
seed, device, input, code, cell spec, or runtime measurement is refused without
overwriting that cell. A valid atomically written cell whose manifest update
was interrupted is verified and added back to the manifest. If any command
fails, the shell prints `RUN_FAILED`, archives partial artifacts to
`/root/part62-return-INCOMPLETE.tgz`, and exits without deleting files or
terminating the instance. After generation, the script verifies the
manifest, creates `/root/part62-return.tgz`, checks the archive, and only then
prints `RETURN_READY` and exits zero. There is no automatic instance teardown
in this script.

Before generation, `run62.sh` fetches both pinned snapshots with
`snapshot_download(..., cache_dir='/root/hf/hub')`: Qwen revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and OLMo revision
`48d788eca847d4d7548f375ad03d3c9312f6139e`. It then sets
`HF_HOME=/root/hf`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.
On failure it preserves the box and writes
`/root/part62-return-INCOMPLETE.tgz`; success writes
`/root/part62-return.tgz` and ends with `RETURN_READY`.

`run62.sh` also writes keyword-proxy counts before the reader runs. Those
counts are exploratory and do not substitute for blind labels.

## Blind labels

After retrieving the archive, run the paid reader from the project root:

```sh
python3 part62/blind62.py run-all --folder results/part62 --review-dir results/part62-review
```

The default reader is Claude Haiku 5.5 (`claude-haiku-5-5`). A calibration-only
batch runs first on 120 seeded, balanced as class availability allows, already-labelled
origin answers from Parts 54a, 54b, and 26.8. Part 59a has no eligible origin
questions. The selected source answer ids, seed, and realized EARTH/OFFWORLD/NEITHER counts are stored with the result.
Together with the 1,536 experiment answers, this makes 1,656 reader requests.
Haiku must agree with the historical Opus labels at Cohen's kappa >= 0.8, with
no unlabelled or truncated calibration responses. A provider error, missing
label, or truncated calibration response records an incomplete calibration,
keeps both readers locked from answers, and does not authorize Opus. Retry that
calibration explicitly with `--retry-calibration`; a complete 120-row calibration
below the kappa threshold is a recorded failure and may authorize explicit Opus
fallback. Every request uses
`max_tokens=1024` and omits `temperature`.

All Part 62 reader attempts use `paidaction.DEFAULT_LEDGER_PATH`, shared across
review directories and parts, with the library's $8 aggregate ceiling. Haiku's
reservations are additionally constrained to its $4 `part62-haiku` kind-tag
sub-cap. If Haiku's recorded calibration fails, an operator may explicitly
select the reference reader with `--reader opus`; Opus skips calibration and
labels answers directly. It is never selected implicitly by `run-all`. Missing,
pending, interrupted, ambiguous, or passed Haiku calibration does not authorize
Opus. Requests are submitted in sequential chunks of 200.
Each chunk reserves its full `max_tokens` worst case, persists the batch id
before result collection, and settles from returned input/output usage before
the next chunk is reserved. Raw result rows are saved per batch by their stable
custom IDs. Ambiguous attempts remain counted. `run-all` stores
`part62-run-all.json` for Haiku and `part62-run-all-opus.json` for Opus, with one
durable record and status per calibration or answer attempt. Its lock serializes
reruns. A rerun collects and settles a saved batch id before advancing. A
definitively rejected answer POST with a released reserve is safely retried on
the next `run-all`; it receives a new attempt and reserve. A rejected
calibration POST requires `--retry-calibration` before a new attempt is made.
The raw POST response is fsynced before its id is parsed. If saving that response
fails, the accepted batch id is persisted in the chunk record before more work;
if that record also cannot be saved, the raw response is printed to stderr for
recovery. If the response file was saved but the normal run-record update failed,
the chunk is marked `response-saved`; resume `run-all` to collect it. Retirement
refuses that state, and `--retry-answers` cannot resubmit it. If the process dies
before either the response or its batch id can be persisted, the outcome is
unknown and the reserve stays counted. After checking the provider and ledger,
an operator can retire that attempt:

```sh
python3 part62/blind62.py retire ATTEMPT_ID --who OPERATOR \
  --reason 'verified outcome unknown' --review-dir results/part62-review
```

The ledger records retirement while retaining the reserve. The retire command
accepts only `part62-haiku` or `part62-opus` attempts whose submission id maps to
the matching Part 62 run chunk. It refuses a chunk whose batch id is already
recorded; resume `run-all` to collect that batch instead. A retired calibration does not authorize Opus. To retry a
retired or incomplete calibration, run the explicit command printed by the
refusal; it appends a new calibration attempt with a new reserve while preserving
the prior attempt:

```sh
python3 part62/blind62.py run-all --folder results/part62 \
  --review-dir results/part62-review --reader haiku \
  --ledger-path /home/elisepc/assistant-basin/paidaction/paid-ledger.jsonl \
  --retry-calibration
```

A retired answer chunk also remains counted. After reviewing the unknown outcome,
the explicit `--retry-answers` command printed by the refusal submits that chunk
again under a new reserve. For settled answer chunks with errored, expired,
truncated, or unlabelled rows, `--retry-failed-answers` submits only those rows
that have not already used one `--retry-failed-answers` retry per answer row
in this persistent run. The retry chunk records the custom ids, so repeating
the flag cannot re-pay the same row.
Successful rows from the earlier batch remain in the final labels. If an answer POST was rejected but its reserve is still active, retire it
using the printed command first, then rerun with `--retry-answers`.

To inspect a known batch before rerunning:

```sh
python3 part62/blind62.py status BATCH_ID --review-dir results/part62-review
```

A missing, pending, interrupted, ambiguous, or incomplete calibration blocks answers from
both readers and prints the Haiku command to run or resume next. Haiku answer
chunks require a recorded passing calibration. Opus answer chunks require an
explicit `--reader opus` command and a recorded failed Haiku calibration; Opus
does not run its own calibration. Reader requests omit `temperature`.

The submitter accepts only the complete Part 62 manifest and exactly the files
listed by the writer. It shuffles answers with a fixed seed and sends answer
text and the origin rubric without model, question, phase, dose, or control
metadata. It records reader, sample ids, stop reasons, usage, and the shared
ledger path with the labels. Source the credit key without printing it and
confirm the CA bundle and review directory are present before submission.

## Analysis

To print the keyword proxies before reader labels exist:

```sh
python3 part62/analyze62.py --folder results/part62 --out results/part62-analysis.json
```

After a successful calibration and complete label collection:

```sh
python3 part62/analyze62.py \
  --folder results/part62 \
  --labels results/part62-review/part62-labels-BATCH_ID.json \
  --review-dir results/part62-review \
  --out results/part62-analysis.json
```

The analyzer verifies that the local label key matches the same shuffled rows
read back from the real writer files. For 62a it reports the exact paired
OFFWORLD count difference and a conservative 95% difference interval formed
from Bonferroni-adjusted exact binomial intervals for the two proportions.
The registered `within 3 of 24` rule is held only when every registered pair
meets that limit, and is reported as “held within sampling noise.” It prints
the review's three known-noisy
keyword proxies as exploratory views, six binomial logistic fits (slope,
Pile-dose midpoint, 10–90% width, and alien-frame midpoint), the registered
62a–62g results, using `NOT-EVALUABLE` when evidence is incomplete and
`HELD WITHIN SAMPLING NOISE` for 62a only when the registered paired count
condition holds, along with both registered decision-rule outcomes. If
complete separation prevents a finite logistic fit, it reports a
10–90 bracket from the last dose below 10% to the first dose at or above 90%;
62c evaluates its threshold on that bracket width, which is an upper bound.
62d requires a finite logistic fit because a bracket width is only an upper
bound and cannot establish that the true width exceeds 0.8. A non-estimable
prediction is reported as `NOT-EVALUABLE`.

The stored `target_equivalent_set_value_norm` is the norm of the requested
activation set value times the target down-projection column norm. The shared
hook sets an activation to that value, so this is not the actual change from
the unmodified activation; the actual delta also depends on the baseline
activation at each position.

## Local stub tests

From the project root, run:

```sh
python3 -m unittest discover -s part62 -p 'test_part62.py' -v
```

The tests use a CPU-only stub model and do not load checkpoints, contact an
API, or require a GPU. They call the real NeuronHook through Part 62's position
adapter, exercise writer and resume paths with stubs, and cover the separated
fit and 62c boundary. Copy-based mutation checks must make the phase hook,
62c evaluator, and writer parameter computation tests go red.
