# Part 86 A770 runbook

Copy `screen86.py`, `common86.py`, `triage86.py`, `blind86.py`, `verify86.py`, `READER-BRIEF.md`, and `READER-BRIEF-VERIFY.md` to `C:\Users\alexa\neuron-probe-gpu\part86\`. The calibration file remains at `C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt`; it is supplied with `--calib` and is not copied into the repository.

The job runner reads each line of `steps.txt` and invokes Python with that line as arguments. Keep paths bare and do not add quotes. It logs to `progress.log`, holds a single-run lock, stops a step below 2 GB available physical RAM, and disables the Level Zero copy engine by default. Check available RAM before launch. Each scheduled task is killed at six hours; the four-layer groups below have a comfortable margin if the pilot gate passes.

## 1. Balanced layer 22 timing pilot

In `C:\Users\alexa\neuron-probe-gpu\steps.txt`, put this one line:

```text
C:\Users\alexa\neuron-probe-gpu\part86\screen86.py --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\pilot-time --layers 22 --limit 256 --batch-size 16 --device xpu
```

Start the normal `job.py` task. Confirm after one minute that the timing step is still running and `progress.log` has no `rc=` line for it. When it finishes, use the last Part 86 line in `p86\pilot-time\progress.log`; its `answers_per_second` is the conservative measured throughput including load and calibration overhead. The pilot generated 256 neurons and 256 random directions (512 answers).

The rate must be at least `165376 / 43200 = 3.8282 answers/second` to project all 17 layers under 12 GPU-hours. If it is below that threshold, stop before the full sweep and amend dose, token budget, or triage in writing. The group projections below use this measured rate: each group has 38,912 answers and must project below six hours (`rate > 1.8015 answers/second`). Recheck the calculation with the observed rate before each launch.

## 2. Full layer 22 pilot gate

After timing passes, use a fresh output directory and set the following one-line `steps.txt`:

```text
C:\Users\alexa\neuron-probe-gpu\part86\screen86.py --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\screen --layers 22 --batch-size 16 --device xpu
```

This screens all 9,216 neurons and 512 random directions in L22. It is resumable by completed batches. After it finishes, run these from an ordinary Windows terminal:

```text
python C:\Users\alexa\neuron-probe-gpu\part86\triage86.py --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --out C:\Users\alexa\neuron-probe-gpu\p86\triage.json
```

Read `layer_22.jsonl` to confirm L22 N2742's greedy answer claims an off-world origin. Confirm L22 has 20 to 2,000 changed answers using `triage.json`. If either pilot gate fails, do not launch the other layers; amend the design in writing first.

## 3. Remaining layers in under-six-hour groups

Use the same output directory (`p86\screen`) so each layer's sidecar and completed batches can resume. Before each job, estimate `38912 / measured_answers_per_second`; proceed only if it is below six hours. The 12-hour gate rate implies about 2.82 hours per four-layer group.

For each group, put the single command on the only line of `steps.txt`, launch `job.py`, and confirm after one minute that the step remains running:

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

After all groups finish, regenerate triage counts:

```text
python C:\Users\alexa\neuron-probe-gpu\part86\triage86.py --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --out C:\Users\alexa\neuron-probe-gpu\p86\triage.json
```

Then prepare all changed answers for the blind reader:

```text
python C:\Users\alexa\neuron-probe-gpu\part86\blind86.py prepare --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --triage C:\Users\alexa\neuron-probe-gpu\p86\triage.json --items C:\Users\alexa\neuron-probe-gpu\p86\blind-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\blind-key.json
```

Give only `blind-items.jsonl` and `READER-BRIEF.md` to the screen reader. Save its JSONL labels to `blind-labels.jsonl`, then score:

```text
python C:\Users\alexa\neuron-probe-gpu\part86\blind86.py score --screen-dir C:\Users\alexa\neuron-probe-gpu\p86\screen --items C:\Users\alexa\neuron-probe-gpu\p86\blind-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\blind-key.json --labels C:\Users\alexa\neuron-probe-gpu\p86\blind-labels.jsonl --out C:\Users\alexa\neuron-probe-gpu\p86\blind-results.json
```

The screen items contain only random id and answer text. Keep the key private until labels are complete.

## 4. Verification

After a second blind pass picks 20 distinct personas, save its JSON array as `picked20.json` with objects `{ "layer": 22, "neuron": 2742, "persona": "..." }`. Run:

```text
C:\Users\alexa\neuron-probe-gpu\part86\verify86.py run --picked C:\Users\alexa\neuron-probe-gpu\p86\picked20.json --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --out C:\Users\alexa\neuron-probe-gpu\p86\verify86.jsonl --device xpu
```

Prepare blind persona-claim items, give the reader those items and `READER-BRIEF-VERIFY.md`, and save the returned JSONL as `verify-labels.jsonl`:

```text
C:\Users\alexa\neuron-probe-gpu\part86\verify86.py prepare --input C:\Users\alexa\neuron-probe-gpu\p86\verify86.jsonl --items C:\Users\alexa\neuron-probe-gpu\p86\verify-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\verify-key.json
```

```text
C:\Users\alexa\neuron-probe-gpu\part86\verify86.py score --items C:\Users\alexa\neuron-probe-gpu\p86\verify-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p86\verify-key.json --labels C:\Users\alexa\neuron-probe-gpu\p86\verify-labels.jsonl --out C:\Users\alexa\neuron-probe-gpu\p86\verify-results.json
```

Verification uses 24 neuron samples and 24 samples for each of ten matched directions per neuron: 5,280 answers total.
