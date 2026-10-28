# Part 86 A770 runbook

Copy `screen86.py`, `common86.py`, `triage86.py`, `blind86.py`, `verify86.py`, `test86.py`, `mutate86.py`,
`golden-verdicts.json`, `READER-BRIEF.md`, and `READER-BRIEF-VERIFY.md` to
`C:\Users\alexa\neuron-probe-gpu\part86\`. The repository-level `neuron_report.py` and `part84` package must also be
present beside `part86` for the isolated mutation runner. Screen configs hash only generation-path files
(`common86.py`, `screen86.py`, and `neuron_report.py`), calibration text, versions, dose, dtype, median, and effective
generation settings. Blind scoring and verification write separate manifests with their own code hashes. The
calibration file remains at `C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt`; supply it with `--calib`.

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

Before any GPU time, run the mutation audit from the `part86` directory:

```text
python -I -B mutate86.py --jobs 4
```

Proceed only when the final line says `MUTATE86: 29/29 killed; all killed`. A stale exact replacement, a skipped test,
or a surviving mutation exits nonzero and blocks the launch.

## 1. Balanced layer 22 timing pilot

In `C:\Users\alexa\neuron-probe-gpu\steps.txt`, put this one line:

```text
C:\Users\alexa\neuron-probe-gpu\part86\screen86.py --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\pilot-time --layers 22 --limit 256 --batch-size 16 --device xpu
```

Start the normal `job.py` task. Confirm after one minute that the timing step is still running and `progress.log` has
no `rc=` line for it. Screen throughput times only the generation calls, after model load; model load, calibration,
canaries, and file writes are excluded. The pilot screens 256 neurons and 256 random directions (512 answers). The
rate must be at least `165376 / 43200 = 3.82815 answers/second` to project all 17 layers under 12 GPU-hours. If it is
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
Part 84's 24.8, then runs a runtime canary. The canary measures the same unsteered prompt at batch size 16 and alone,
and records both answers, token ids, and exact-match status. In the batch-16 call, one zero-push row must exactly equal a
second unsteered row of the same batch shape; that equality is a hard gate. The zero-push answer must also triage as
kept and name Qwen. Two pushed rows are compared with batch-1 answers and recorded as a diagnostic for bf16 batch
effects.

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
python C:\Users\alexa\neuron-probe-gpu\part86\blind86.py prepare --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --triage C:\Users\alexa\neuron-probe-gpu\p86\triage.json --items C:\Users\alexa\neuron-probe-gpu\p86\reader-input\blind-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\private\blind-key.json --chunk-size 500
```

Preparation includes all changed answers plus seed-8686 samples of up to 200 kept and 200 broken answers from each of
the real and random conditions, drawn across all 17 layers. It writes reader chunks beside the master items file:
`blind-items-reader1-chunks/` holds all items in chunks of at most 500; `blind-items-reader2-chunks/` holds a seed-8687
random 10% overlap. Each chunk includes its opaque ids. Give the first reader the reader brief and reader-1 chunks
pasted one at a time into a tool-less session. Give the second reader only reader-2 chunks. Do not give either reader the
key, layer/neuron mapping, filesystem, or access to the screen directory. Save the first labels as `blind-labels.jsonl`
and the second labels as `blind-labels-2.jsonl`.

The key is written before the reader items. The scorer reports the coherent-persona share in each kept/broken sample
beside the layer rates; these miss rates do not change verdicts. It reports Cohen's kappa for coherent persona versus
not before the verdicts. If kappa is below 0.6, verdicts 86.2-86.6 are `NOT READ` until a full second reading is
completed. For that reread, have reader two label every reader-1 chunk and pass the full label file as `--labels2`.
Score only after both passes are complete:

```text
python C:\Users\alexa\neuron-probe-gpu\part86\blind86.py score --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --items C:\Users\alexa\neuron-probe-gpu\p86\reader-input\blind-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\private\blind-key.json --labels C:\Users\alexa\neuron-probe-gpu\p86\blind-labels.jsonl --labels2 C:\Users\alexa\neuron-probe-gpu\p86\blind-labels-2.jsonl --out C:\Users\alexa\neuron-probe-gpu\p86\blind-results.json
```

The scorer re-runs triage, validates the item-to-screen answer mapping, checks generation-path provenance, and refuses
limited or incomplete layer denominators. Verdicts 86.2 and 86.4 follow the pure count rules in `golden-verdicts.json`
and Amendments 1 and 2. The 86.1 automatic off-world match is a convenience only; hand-read L22 N2742's raw answer and
record that decision in the run results.

## 4. Verification

### Picker input and pick-to-neuron mapping (finding #7)

Build `picker-items.jsonl` by joining primary blind items to `blind-labels.jsonl` by opaque id, then retaining only key
rows whose hidden source is real and whose primary label is a coherent persona. Give the picker only
`{ "id": "...", "persona": "...", "text": "..." }` rows; omit layer, neuron, condition, and key data. Have the
picker return 20 distinct ids and persona phrases. Resolve those ids in the private blind key, confirm each points to a
real `neuron` row and to the same coherent-persona primary label, then write `picked20.json` rows
`{ "layer": 22, "neuron": 2742, "persona": "..." }`.

Verification rechecks every pick against its primary blind label and a real screen neuron whose answer triaged changed.
Run a one-neuron timing chunk first; it prints the measured per-neuron time and projected total for 20. Each neuron is
stored as `L<layer>_N<neuron>.json` in the output directory and is skipped on rerun only with matching code, library,
dtype, dose, calibration, screen median, and config provenance.

```text
C:\Users\alexa\neuron-probe-gpu\part86\verify86.py run --picked C:\Users\alexa\neuron-probe-gpu\p86\picked20.json --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --blind-key C:\Users\alexa\neuron-probe-gpu\p86\private\blind-key.json --blind-labels C:\Users\alexa\neuron-probe-gpu\p86\blind-labels.jsonl --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\verify86 --max-neurons 1 --device xpu
```

If the projection is below six hours, the remaining neurons may run in one command by omitting `--max-neurons`. For a
safer resumable sequence, repeat the same `--max-neurons 1` command in separate tasks until all 20 files are complete;
each task performs one neuron's 264 answers and the command automatically skips completed files. The final output is
20 neuron files, 5,280 answers total.

Prepare blind persona-claim items into a reader folder with the key in a separate private folder, paste the item texts
inline into the tool-less reader, and do not share the key:

```text
C:\Users\alexa\neuron-probe-gpu\part86\verify86.py prepare --input C:\Users\alexa\neuron-probe-gpu\p86\verify86 --items C:\Users\alexa\neuron-probe-gpu\p86\reader-input\verify-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\private\verify-key.json --chunk-size 500
C:\Users\alexa\neuron-probe-gpu\part86\verify86.py score --items C:\Users\alexa\neuron-probe-gpu\p86\reader-input\verify-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\private\verify-key.json --labels C:\Users\alexa\neuron-probe-gpu\p86\verify-labels.jsonl --labels2 C:\Users\alexa\neuron-probe-gpu\p86\verify-labels-2.jsonl --out C:\Users\alexa\neuron-probe-gpu\p86\verify-results.json
```

Verification reader items also have 500-row chunks and a seed-8687 10% second-reader sample. The second reader labels
that sample with the same persona-claim schema. Kappa is printed and stored before 86.5; if it is below 0.6, read all
verification chunks a second time and rescore with the full second labels file.

Blind and score outputs refuse to overwrite existing paths unless `--force` is supplied. Keep manifests, metadata,
screen outputs, keys, reader chunks, labels, and final results together with the run archive. Screen, blind-score, and
verification manifests record their own code and input SHA-256 hashes.
