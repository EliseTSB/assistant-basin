# Part 65 runbook

This run uses the pinned local Qwen3-1.7B snapshot on the Intel Arc A770, then ranks and judges the returned write summaries on the host. `job.py` runs each non-comment line of the root `steps.txt` as `python <line>` with its working directory set to `C:\Users\alexa\neuron-probe-gpu`. It stops at the first nonzero result, and writes `step-N.log` plus `progress.log` there. Its RAM guard and copy-engine-off defaults remain active.

## Stage the Windows basin

From WSL, copy the code from this checkout and the frozen Part 64 collection into `C:\Users\alexa\neuron-probe-gpu\p65basin`. `atlas_collect_fast2.py` imports `part38.py`; those are the only extra project modules used by steering to reproduce the canonical random controls. The Windows Python environment must already have the pinned model cached and the project dependencies installed. Do not use a command that downloads the model.

```bash
GPU_ROOT=/mnt/c/Users/alexa/neuron-probe-gpu
mkdir -p "$GPU_ROOT/p65basin/results/part64" "$GPU_ROOT/p65basin/results/part65"
cp -a /home/elisepc/green/worktrees/ab-part65/part60 "$GPU_ROOT/p65basin/"
cp -a /home/elisepc/green/worktrees/ab-part65/part64 "$GPU_ROOT/p65basin/"
cp -a /home/elisepc/green/worktrees/ab-part65/part65 "$GPU_ROOT/p65basin/"
cp /home/elisepc/green/worktrees/ab-part65/atlas_collect_fast2.py "$GPU_ROOT/p65basin/"
cp /home/elisepc/green/worktrees/ab-part65/part38.py "$GPU_ROOT/p65basin/"
cp /home/elisepc/green/worktrees/ab-part64/results/part64/collection.jsonl \
  "$GPU_ROOT/p65basin/results/part64/collection.jsonl"
```

## Median, pilot, check, and full run

Put these four lines, in this order, into `C:\Users\alexa\neuron-probe-gpu\steps.txt`. They use paths relative to the runner's working directory. The median step measures the registered 2,000 corpus-token residual norms and writes `layer-medians.json`. The pilot reuses those medians for a stratified 20-unit sample. The check gates the full run on observed +0.8 changes, the separately generated dose-zero comparison, hook activity/layer, and equal push norms. Its equal-norm gate re-reads the run-wide check recorded in the pilot rows; it does not recompute run-wide norms from the pilot sample. If it fails, `job.py` stops before the full run.

```text
p65basin\part65\steer65.py --corpus qwen35\pile10k.jsonl --medians p65basin\results\part65\layer-medians.json --device xpu --measure-medians-only
p65basin\part65\steer65.py --collection p65basin\results\part64\collection.jsonl --corpus qwen35\pile10k.jsonl --medians p65basin\results\part65\layer-medians.json --out p65basin\results\part65\pilot.jsonl --units 20 --device xpu
p65basin\part65\steer65.py --pilot-check p65basin\results\part65\pilot.jsonl
p65basin\part65\steer65.py --collection p65basin\results\part64\collection.jsonl --corpus qwen35\pile10k.jsonl --medians p65basin\results\part65\layer-medians.json --out p65basin\results\part65\steer.jsonl --device xpu
```

Inspect `step-1.log` through `step-4.log`, `progress.log`, and `p65basin\results\part65\pilot.jsonl` before transferring. The full run selects all 500 frozen units because `--units` is omitted. If Windows' RAM guard kills a step, inspect available RAM and the named step log, then rerun `job.py`; steering resumes completed unit rows from the same output file. Do not use the pilot output as the full-run output because they have different unit manifests.

## Copy results back

From WSL, copy the whole Part 65 result directory back to this worktree. It contains the medians, pilot and full steering JSONL files, manifests, and logs if saved alongside them.

```bash
mkdir -p /home/elisepc/green/worktrees/ab-part65/results/part65
cp -a /mnt/c/Users/alexa/neuron-probe-gpu/p65basin/results/part65/. \
  /home/elisepc/green/worktrees/ab-part65/results/part65/
```

Run the pilot check again on the copied pilot artifact, then rank the full output:

```bash
python part65/steer65.py --pilot-check results/part65/pilot.jsonl
python part65/rank65.py --steer results/part65/steer.jsonl --out results/part65/ranked.jsonl
```

## Judge on the host

Set `ANTHROPIC_CREDITS_KEY` (preferred) or `ANTHROPIC_API_KEY` in the host shell. When `~/.mitmproxy/ca-bundle-combined.crt` exists, the judge trusts that bundle; otherwise it uses Python's default trust store. First run the one-request probe and inspect its saved answer and chunk state before authorizing the full 1,000-trial plan:

```bash
python part65/judge65.py \
  --collection results/part64/collection.jsonl \
  --read-dir results/part64/gen64/outputs \
  --ranked results/part65/ranked.jsonl \
  --sentence-test results/part64/scored.jsonl \
  --out-dir results/part65/judge \
  --probe
```

The probe answer is at `results/part65/judge/probe/answers.jsonl`. Continue only after it settles and contains a parsed A/B result. Submit the full plan with the same inputs and output directory, omitting `--probe`:

```bash
python part65/judge65.py \
  --collection results/part64/collection.jsonl \
  --read-dir results/part64/gen64/outputs \
  --ranked results/part65/ranked.jsonl \
  --sentence-test results/part64/scored.jsonl \
  --out-dir results/part65/judge
```

The full answers are at `results/part65/judge/answers.jsonl`. If a known provider batch is still pending, use the same command with `--recover`; do not resubmit an ambiguous no-ID POST. A released chunk is final for that output directory: the judge prints the chunk path and its error, then exits nonzero. Fix the underlying input or request and rerun the whole judge with the full-plan inputs into a new `--out-dir`. Do not resubmit the released attempt or combine answers from the old and new output directories. The judge reserves through the shared paidaction ledger and retains its $1 Part 65 sub-cap.

## Analyze

After all ten judge chunks settle, run the registered analysis. Unparseable trials are excluded; units with no valid trial are counted and omitted. Analysis stops with a clear error if more than 5% of trials are unparseable.

```bash
python part65/analyze65.py \
  --answers results/part65/judge/answers.jsonl \
  --collection results/part64/collection.jsonl \
  --read-neuron results/part64/gen64/outputs/haiku-1x-neuron.jsonl \
  --read-control results/part64/gen64/outputs/haiku-1x-control.jsonl \
  --ranked results/part65/ranked.jsonl \
  --out results/part65/analysis-65.json
```

This writes JSON and Markdown with 65a-65g, the 65f registered observed-cosine difference and its null-adjusted counterpart, plus the three exploratory depth-band estimates and intervals for 65c/65g.
