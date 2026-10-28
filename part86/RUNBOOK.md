# Part 86 A770 runbook

Copy `screen86.py`, `common86.py`, `triage86.py`, `blind86.py`, `verify86.py`, `test86.py`, `READER-BRIEF.md`,
`READER-BRIEF-VERIFY.md`, and `neuron_report.py` to `C:\Users\alexa\neuron-probe-gpu\part86\`. Record the copied
`neuron_report.py` SHA-256 in the run notes; every screen and verification config also stores SHA-256 hashes for all
`part86/*.py` plus `neuron_report.py`, torch and transformers versions, actual dtype, dose, and layer residual median.
The calibration file remains at `C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt`; supply it with
`--calib`.

The job runner reads each line of `steps.txt` and invokes Python with that line as arguments. Keep paths bare and do
not add quotes. It logs to `progress.log`, holds a single-run lock, stops a step below 2 GB available physical RAM,
and disables the Level Zero copy engine by default. Check available RAM before launch. Each scheduled task is killed at
six hours. The `--reset-layer N` option deletes that layer's metadata, final output, and `.batches` fragments together;
use it when a layer needs a clean restart. Without reset, orphan batch fragments are refused.

## 0. Tests on the GPU box

Run from `C:\Users\alexa\neuron-probe-gpu\part86` before any screen work:

```text
set PART86_GPU_STRICT=1
python -I test86.py
```

With `PART86_GPU_STRICT=1`, a missing torch or transformers import raises an error instead of allowing model tests to
skip. The test suite is CPU-only and uses a tiny Qwen-style model; it does not download or load Qwen3.5-4B.

## 1. Balanced layer 22 timing pilot

In `C:\Users\alexa\neuron-probe-gpu\steps.txt`, put this one line:

```text
C:\Users\alexa\neuron-probe-gpu\part86\screen86.py --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\pilot-time --layers 22 --limit 256 --batch-size 16 --device xpu
```

Start the normal `job.py` task. Confirm after one minute that the timing step is still running and `progress.log` has
no `rc=` line for it. Screen throughput times only the generation calls, after model load; model load, calibration,
canaries, and file writes are excluded. The pilot screens 256 neurons and 256 random directions (512 answers). The
rate must be at least `165376 / 43200 = 3.8282 answers/second` to project all 17 layers under 12 GPU-hours. If it is
below that threshold, stop before the full sweep and amend dose, token budget, or triage in writing. Each four-layer
group has 38,912 answers and must project below six hours (`rate > 1.8015 answers/second`). Recheck the calculation
with the observed generation-only rate before each launch.

## 2. Full layer 22 pilot gate

After timing passes, use a fresh output directory and set this one-line `steps.txt`:

```text
C:\Users\alexa\neuron-probe-gpu\part86\screen86.py --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\screen --layers 22 --batch-size 16 --device xpu
```

The full pilot is 9,216 neurons plus 512 random directions, 9,728 answers in total. At run start the code asserts the
rendered no-system prompt and empty thinking block. It also checks that layer 22's residual median is within 5% of
Part 84's 24.8, and runs a runtime canary with zero-push and two pushed rows in a batch of 16. The zero-push answer
must triage as kept and name Qwen. It compares the first two pushed answers with batch-1 answers and records the exact
match flags and texts in `runtime-canaries.json`; this comparison is diagnostic because the design notes that bf16
batching can change greedy text slightly.

After the full screen finishes, inspect `pilot-gate.json` and `layer_22.jsonl`. `pilot-gate.json` logs N2742's raw
answer. Gate 1 passes only if that **answer text** claims an off-world origin. Gate 2 is the generation-only throughput
projection from the timing pilot. Gate 3 is 20 to 2,000 changed answers **out of the full 9,728 layer-22 answers**
(9,216 real and 512 random). Recompute triage with:

```text
python C:\Users\alexa\neuron-probe-gpu\part86\triage86.py --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --out C:\Users\alexa\neuron-probe-gpu\p86\triage.json
```

If any pilot gate fails, do not launch the other layers; amend the design in writing first. A limited `--limit` screen
cannot be used for verdict scoring: the scoring command asserts 9,216 real and 512 random answers in every layer.

## 3. Remaining layers in under-six-hour groups

Use the same output directory so each layer's metadata and completed batches can resume. Before each job, estimate
`38912 / measured_generation_answers_per_second`; proceed only if it is below six hours. The 12-hour gate rate implies
about 2.82 hours per four-layer group.

```text
C:\Users\alexa\neuron-probe-gpu\part86\screen86.py --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\screen --layers 10-13 --batch-size 16 --device xpu
```

```text
C:\Users\alexa\neuron-probe-gpu\part86\screen86.py --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\screen --layers 14-17 --batch-size 16 --device xpu
```

```text
C:\Users\alexa\neuron-probe-gpu\part86\screen86.py --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\screen --layers 18-21 --batch-size 16 --device xpu
```

```text
C:\Users\alexa\neuron-probe-gpu\part86\screen86.py --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\screen --layers 23-26 --batch-size 16 --device xpu
```

If code, library versions, dtype, dose, or median changes, resume refuses. Use `--reset-layer N` only when intentionally
discarding that layer's partial and completed data, and then rerun its full command.

After all groups finish, regenerate triage counts and prepare blind items. Keep the key in a separate private folder:

```text
python C:\Users\alexa\neuron-probe-gpu\part86\triage86.py --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --out C:\Users\alexa\neuron-probe-gpu\p86\triage.json
python C:\Users\alexa\neuron-probe-gpu\part86\blind86.py prepare --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --triage C:\Users\alexa\neuron-probe-gpu\p86\triage.json --items C:\Users\alexa\neuron-probe-gpu\p86\reader-input\blind-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\private\blind-key.json
```

The key is written before the reader items. Give the reader only the reader brief and the item texts pasted inline into a
tool-less session with no file tools. Do not give the reader the key, layer/neuron mapping, filesystem, or access to the
screen directory. Save returned JSONL labels as `blind-labels.jsonl`, then score once the labels are complete:

```text
python C:\Users\alexa\neuron-probe-gpu\part86\blind86.py score --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --items C:\Users\alexa\neuron-probe-gpu\p86\reader-input\blind-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\private\blind-key.json --labels C:\Users\alexa\neuron-probe-gpu\p86\blind-labels.jsonl --out C:\Users\alexa\neuron-probe-gpu\p86\blind-results.json
```

The scorer re-runs triage, validates the item-to-screen answer mapping, checks the manifest hashes, and refuses limited
or incomplete layer denominators. Verdicts 86.2 and 86.4 use Amendment 1 as recorded in `PART86-DESIGN.md` and
`PREDICTIONS.md`; the original readings are superseded.

## 4. Verification

After a second blind pass selects 20 distinct personas, save `picked20.json` with objects
`{ "layer": 22, "neuron": 2742, "persona": "..." }`. Verification validates every pick against a real screen neuron
whose answer triaged changed. Run a one-neuron timing chunk first; it prints the measured per-neuron time and projected
total for 20. Each neuron is stored as `L<layer>_N<neuron>.json` in the output directory and is skipped on rerun only
with matching code, library, dtype, dose, calibration, screen median, and config provenance.

```text
C:\Users\alexa\neuron-probe-gpu\part86\verify86.py run --picked C:\Users\alexa\neuron-probe-gpu\p86\picked20.json --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\verify86 --max-neurons 1 --device xpu
```

If the projection is below six hours, the remaining neurons may run in one command by omitting `--max-neurons`. For a
safer resumable sequence, repeat the same `--max-neurons 1` command in separate tasks until all 20 files are complete;
each task performs one neuron's 264 answers and the command automatically skips completed files. The final output is
20 neuron files, 5,280 answers total.

Prepare blind persona-claim items into a reader folder with the key in a separate private folder, paste the item texts
inline into the tool-less reader, and do not share the key:

```text
C:\Users\alexa\neuron-probe-gpu\part86\verify86.py prepare --input C:\Users\alexa\neuron-probe-gpu\p86\verify86 --items C:\Users\alexa\neuron-probe-gpu\p86\reader-input\verify-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\private\verify-key.json
C:\Users\alexa\neuron-probe-gpu\part86\verify86.py score --items C:\Users\alexa\neuron-probe-gpu\p86\reader-input\verify-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\private\verify-key.json --labels C:\Users\alexa\neuron-probe-gpu\p86\verify-labels.jsonl --out C:\Users\alexa\neuron-probe-gpu\p86\verify-results.json
```

Blind and score outputs refuse to overwrite existing paths unless `--force` is supplied. Keep `manifest.json`, metadata,
screen outputs, keys, items, and final results together with the run archive; the manifest records code, config, screen,
and item SHA-256 hashes.
