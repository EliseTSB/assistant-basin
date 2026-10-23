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

## Inputs

The project root must contain the Part 26 records in `results/part26/`, the
Part 33 record at `results/part33-r2/natural-firing.json`, and locally cached
model snapshots at the pinned revisions in `run62.py`. The generator sets the
Hugging Face offline flags through the Part 26 loader. It does not fetch model
files. OLMo's alien-frame maximum is recomputed on the 30 selected Part 33
passages with Part 26's tokenizer and `scale_frames` capture method; Qwen's
maximum is recorded in the same way for a direct check against the registered
24.69 reference. The dose itself remains in Part 26 Pile-maximum units.

## Generate and archive

From any working directory on the rented GPU box, run:

```sh
bash part62/run62.sh
```

The runner writes one atomic JSON file per model, dose, phase, question, or
scaled-control cell under `results/part62/`. It writes and updates
`part62-manifest.json` after each successful cell. A complete manifest records
each condition seed, target and control activation values, both down-projection
column norms, the alien-frame maximum, and the generated answers. The output
directory must be empty at the start; the runner refuses to overwrite an
existing run. If any command fails, the shell prints `RUN_FAILED_KEEP_BOX` and
exits without deleting files or terminating the instance. After generation,
the script verifies the manifest, creates `/root/part62-return.tgz`, checks the
archive, and only then prints `RETURN_READY` and exits zero. There is no
automatic instance teardown in this script.

`run62.sh` also writes keyword-proxy counts before the reader runs. Those
counts are exploratory and do not substitute for blind labels.

## Blind labels

After retrieving the archive, run the reader from the project root:

```sh
python3 part62/blind62.py submit --folder results/part62 --review-dir results/part62-review
python3 part62/blind62.py status BATCH_ID --review-dir results/part62-review
python3 part62/blind62.py collect BATCH_ID --review-dir results/part62-review
```

The submitter accepts only the complete Part 62 manifest and exactly the files
listed by the writer. It shuffles answers with a fixed seed, sends answer text
and the origin rubric without model, question, phase, dose, or control metadata,
and keeps the mapping in local review files. It uses the Part 54 origin labels
(`EARTH` / `OFFWORLD` / `NEITHER`, plus `OK` / `BROKEN`) and the Part 59A
transport and calibration-error threshold. The reader request has
`max_tokens=1024`. Collection records each response's `stop_reason` and accepts
only `end_turn` or `stop_sequence`; a `max_tokens` stop, missing stop reason, or
other completion status blocks scoring. It also refuses to score if more than
two of the 22 origin calibration items are wrong or if any generated answer is
unlabelled.

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
read back from the real writer files. It prints the review's three keyword
proxies, six binomial logistic fits (slope, Pile-dose midpoint, 10–90% width,
and alien-frame midpoint), and the registered 62a–62g results as `HELD` or
`FAILED`. If a phase has no finite two-parameter logistic maximum-likelihood
fit, its fit is marked not estimable and predictions that require that fit are
reported `FAILED`; thresholds are not adjusted to force a fit.

## Local stub tests

From the project root, run:

```sh
python3 -m unittest discover -s part62 -p 'test_part62.py' -v
```

The tests use a CPU-only stub model and do not load checkpoints, contact an
API, or require a GPU. Two mutation checks alter the code object of the actual
phase-hook and prediction functions, run the affected test, and require each
mutant to produce a red assertion before restoring the original function.
