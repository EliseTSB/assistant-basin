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

From any working directory on the rented GPU box, run:

```sh
bash part62/run62.sh
```

The runner writes one atomic JSON file per model, dose, phase, question, or
scaled-control cell under `results/part62/`. It writes and updates
`part62-manifest.json` after each successful cell. A complete manifest records
each condition seed, target and control set values, both down-projection
column norms, the alien-frame maximum, and the generated answers. After a kill,
rerun the same command into the same output directory. Resume checks the run
fingerprint and every existing cell fingerprint and file hash; a different
seed, device, input, code, cell spec, or runtime measurement is refused without
overwriting that cell. A valid atomically written cell whose manifest update
was interrupted is verified and added back to the manifest. If any command
fails, the shell prints `RUN_FAILED_KEEP_BOX` and exits without deleting files
or terminating the instance. After generation, the script verifies the
manifest, creates `/root/part62-return.tgz`, checks the archive, and only then
prints `RETURN_READY` and exits zero. There is no automatic instance teardown
in this script.

`run62.sh` also writes keyword-proxy counts before the reader runs. Those
counts are exploratory and do not substitute for blind labels.

## Blind labels

After retrieving the archive, run the reader from the project root:

```sh
python3 part62/blind62.py submit --folder results/part62 --review-dir results/part62-review
python3 part62/blind62.py status BATCH_ID --review-dir results/part62-review
python3 part62/blind62.py collect BATCH_ID --review-dir results/part62-review
```

The registered default reader is Claude Haiku 5.5 (`claude-haiku-5-5`) with a
separate hard cumulative cap of $4. If and only if collection writes a failed
Haiku calibration record, an operator may explicitly submit the fallback with
`blind62.py submit --reader opus`; that path uses Claude Opus 5.5 and its own
separate $8 cap and ledger. It never starts automatically. Each batch, labels
record, analysis result, and printed status identifies the reader/model.
Both readers require `max_tokens >= 1024`. The 34 hand-labelled Part 59A texts
are reused with mapped origin labels (EARTH-REAL→EARTH, SCIFI→OFFWORLD, and
FANTASY/OTHER→NEITHER), retaining the OK/BROKEN judgments; more than two
errors or any unverified/truncated calibration response fails the hard gate.

Before submitting, source the credit key without printing it and confirm the CA
bundle and review directory are present. The code checks `ANTHROPIC_CREDITS_KEY`,
the `ANTHROPIC_CA_BUNDLE` path (default `~/.mitmproxy/ca-bundle-combined.crt`),
the complete writer files, and review-directory writability before reserving.
For example:

```sh
set -a
. ~/.anthropic-credits.env
set +a
test -n "$ANTHROPIC_CREDITS_KEY"
```

The project design lists the current credit key as expiring 2026-10-24. Plan the
reader step before then.

The submitter accepts only the complete Part 62 manifest and exactly the files
listed by the writer. It shuffles answers with a fixed seed, sends answer text
and the origin rubric without model, question, phase, dose, or control metadata,
and keeps the mapping in local review files. It uses the Part 54 origin labels
(`EARTH` / `OFFWORLD` / `NEITHER`, plus `OK` / `BROKEN`) and the Part 59A
transport and calibration-error threshold. Calibration reuses all 34 hand-
labelled Part 59A rows mapped to the origin categories; the maximum of two
errors is a hard gate. The reader request has `max_tokens=1024`. Reserve
estimation uses request bytes and 150 expected output tokens per request,
rather than the full output-token ceiling. Collection records each response's
`stop_reason` and output token usage and accepts
only `end_turn` or `stop_sequence`; a `max_tokens` stop, missing stop reason, or
other completion status blocks scoring. It also refuses to score if more than
two of the 34 origin calibration items are wrong or if any generated answer is
unlabelled.

Each submission attempt gets a random attempt id before the call and has its own
key map, intent record, and durable reserve in a reader-specific submission
ledger. The ledger lock is released automatically if the process dies. A successful response's batch
id is written to the ledger immediately; repeating submit then returns that id
until it is collected. A timeout, 429, 5xx, or other ambiguous result keeps its
reserve counted; a later submit creates a new attempt and a new reserve, and no
batch is searched for or adopted. A reserve is released only for a 4xx response
carrying a valid Anthropic error body. Separate ledgers enforce independent
$4 Haiku and $8 Opus hard caps.

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
read back from the real writer files. It prints the review's three known-noisy
keyword proxies as exploratory views, six binomial logistic fits (slope,
Pile-dose midpoint, 10–90% width, and alien-frame midpoint), the registered
62a–62g results as `HELD` or `FAILED`, and both registered decision-rule
outcomes. If complete separation prevents a finite logistic fit, it reports a
10–90 bracket from the last dose below 10% to the first dose at or above 90%;
62c and 62d evaluate their thresholds on that width. A non-estimable fit with
no bracket remains `FAILED`.

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
