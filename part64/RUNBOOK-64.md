# Part 64 runbook

Part 64 compares Haiku-1x, Haiku-3x, Opus-1x, and Opus-3x on the same 500 Part 60 dev units. Use only the dev split. The scorer and pass rule are imported from Part 60; failed or unparseable generations remain non-passes in the denominator. Do not open Part 60's test split unless an arm meets the registered advancement rule in the final analysis.

## 1. Probe the rented 3090

Stage the project and Part 60 source artifacts on the box, then run a non-generative preflight:

```bash
nvidia-smi
python -c 'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.get_device_name(0))'
python -c 'import sys; from pathlib import Path; sys.path.insert(0, "/root/assistant-basin"); from part64.collect64 import verify_corpus; print(verify_corpus(Path(sys.argv[1])))' /path/to/pile10k.jsonl
```

The runner pins the same Qwen3-1.7B snapshot and the same `transformers==5.20.0` / `flash-linear-attention==0.5.2` packages as the Part 61 runner. It needs `results/part56-pilot/pilot.jsonl` plus Part 58's `acts.all.npz`, `ids.all.json`, `units.all.json`, and `sent.all.jsonl` so Part 60 can reconstruct the canonical units and thresholds.

If the probe fails, inspect Python/CUDA versions and the corpus report, stage the missing registered inputs, and repeat the probe before collection.

## 2. Collect windows on Vast

Run the wrapper once before local generation:

```bash
PART64_ROOT=/root/assistant-basin \
PART64_CORPUS=/path/to/pile10k.jsonl \
bash part64/run64.sh
```

The runner fetches the pinned model snapshot, checks the uploaded corpus against `collect64.py`, runs targeted fresh collection, assembles the 1x-in-3x samples, and returns `/root/part64-return.tgz`. The archive contains `results/part64/collection.jsonl`, its manifest, logs, package versions, and run timestamps. Copy the archive back and inspect `logs/part64/collection-model.log`, `collection-assembly.log`, and `corpus-verify.log` if the wrapper reports `RUN_FAILED`.

The wrapper reuses a prior collection only when its manifest records all 500 units, a verified subset, median top-12 reproduction of at least 10/12, and the same corpus SHA-256. To recover from a failed collection, correct the logged input or staging problem and rerun the wrapper. It will reuse a valid completed collection; an incomplete or mismatched cache fails closed.

## 3. Generate locally

Unpack the collection archive so `results/part64/collection.jsonl` is available locally. Keep the Part 60 M1 prompt and writer API defaults unchanged. Run the two paid single-unit probes, inspect both results, then run the four arms:

```bash
python part64/gen64.py probe \
  --collection results/part64/collection.jsonl \
  --root results/part64/gen64
python part64/gen64.py run \
  --collection results/part64/collection.jsonl \
  --root results/part64/gen64
```

After both probes settle, `probe` prints each writer's projected stage cost using that probe's billed input and output tokens, the complete planned request count for that writer (including its probe), and batch prices. If the Opus projection prints `STOP` because it exceeds $36, stop and report the projection; do not run generation.

The generator maintains a durable plan, writer ledgers in the single shared directory `results/part64/writer-stage/`, and settled group files under the run root. The shared ledger directory is resolved against the repository root and is independent of `--root`, so a new run root does not reset a writer's stage cap. It writes eight files under `results/part64/gen64/outputs/`: one per writer, sample size, and unit type. Opus 1x groups use three ledgers and Opus 3x groups use six; every ledger share's complete max-token reserve stays below the paidaction $8 ledger cap. Each request chunk reserves at most $1. Before every reserve, a stage guard sums settled plus reserved spend across all ledgers for that writer and refuses reservations past the registered cap: Haiku $4 and Opus $40. Rows retain `metadata.writer`, `metadata.sample`, `metadata.unit_key`, `metadata.type`, `parse_status`, and the parsed M1 tiers. `score64.py` reads this native format directly; no manual sentence edits or reformatting are needed. Keep failed rows so each arm has the fixed 500-unit denominator.

The 64f comparison is a cross-configuration check, not a pure reproduction: Part 64's 1x text matches Part 60 M1, while Part 64 uses `max_tokens=8192` and API-default thinking and Part 60 M1 used `max_tokens=1100` with thinking explicitly disabled. This caveat does not change the registered prediction.

Check the durable run and output files before transfer:

```bash
python part64/gen64.py status --root results/part64/gen64
find results/part64/gen64/outputs -maxdepth 1 -type f -name '*.jsonl' -print
```

If the API run stops, recover its existing batches rather than resubmitting them:

```bash
python part64/gen64.py status --root results/part64/gen64
python part64/gen64.py resume \
  --collection results/part64/collection.jsonl \
  --root results/part64/gen64
```

If the POST response was lost, resume checks the provider listing. Once a matching run-specific request ID appears in an ended batch, the generator adopts that batch and settles its results without another POST. If the provider batch is still processing, rerun resume after it ends. Retirement is allowed only after a complete listing proves that no matching batch exists.

If a settled probe failed validation, review its saved text and usage, then an operator may authorize one new attempt with a recorded reason:

```bash
python part64/gen64.py reprobe haiku --root results/part64/gen64 \
  --who "operator" --reason "first probe returned invalid M1 JSON"
```

Retiring an unknown POST requires a complete provider batch listing. If a batch may contain the chunk's run-specific request IDs, recover that batch; the tool will refuse retirement and resubmission.

Stage all eight `outputs/*.jsonl` files at `results/part64/gen64/outputs/` on the GPU box. If only some group outputs exist, the scorer rejects the incomplete arm/unit grid instead of treating it as a smaller experiment.

## 4. Score on Vast

Rerun the wrapper after staging the generated group outputs:

```bash
PART64_ROOT=/root/assistant-basin \
PART64_CORPUS=/path/to/pile10k.jsonl \
bash part64/run64.sh
```

A valid collection is reused. With generated rows present, the runner discovers the eight group files and invokes `score64.py` against the pinned local snapshot and Part 60 thresholds. It returns `/root/part64-return.tgz` with `scored.jsonl`, its manifest, logs, and versions. Inspect `logs/part64/scoring.log` on failure and rerun after fixing the stated issue. The scored JSONL is written atomically, so a failed scoring process does not replace a previously complete file. Any row stopped by `max_tokens` remains a non-pass, and analysis reports the truncation count for every arm.

## 5. Analyze locally

Unpack the returned artifacts into the project results tree and run the analysis against Part 60's completed dev analysis:

```bash
python part64/analyze64.py \
  --input results/part64/scored.jsonl \
  --part60-analysis results/part60/analysis/analysis-60.json \
  --out results/part64/analysis-64.json
```

This writes JSON and Markdown with arm pass rates, real-minus-control gaps, 64a-64f, the unit-paired bootstrap, and arm advancement decisions. The script rejects incomplete paired arms and missing Part 60 dev M1 rates. If analysis fails, inspect the named row or missing artifact, repair the upstream handoff, then rerun the same analysis command.

The advancement rule is unchanged: an arm advances only when its full-rule gap is at least 5 percentage points above Haiku-1x and the paired bootstrap 95% interval is strictly above zero. 64e is reported as a direction only because the registered design says it is underpowered. The analysis itself reads only the dev files and never opens test data.
