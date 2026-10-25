# Part 37 build and runbook

The worker builds the pipeline; the generation, capture, and stated readout run later on the Windows A770. Do not download models. Each command uses the pinned revision with `local_files_only=True`. The commands below run one model per process, so the model is freed when that line exits before the next model is loaded.

## Stage the basin

From WSL, stage the Part 37 source under `C:\Users\alexa\neuron-probe-gpu\p37basin`. The corpus is expected at `C:\Users\alexa\neuron-probe-gpu\qwen35\pile10k.jsonl`; confirm it is present before starting.

```bash
GPU_ROOT=/mnt/c/Users/alexa/neuron-probe-gpu
mkdir -p "$GPU_ROOT/p37basin/part37/results/pilot" "$GPU_ROOT/p37basin/part37/results/full"
cp -a /home/elisepc/green/worktrees/ab-part37/part37 "$GPU_ROOT/p37basin/"
```

The Windows environment needs Python 3.11+, PyTorch with XPU support, Transformers, NumPy, and scikit-learn installed. The three exact pinned model snapshots must already be cached locally. The `job.py` runner executes each non-comment line in the root `steps.txt` as `python <line>`, from `C:\Users\alexa\neuron-probe-gpu`, stopping at the first nonzero exit code.

## Pilot, then full run

Put these lines in `C:\Users\alexa\neuron-probe-gpu\steps.txt` in this order. The first section collects the first 20 questions, screens them, builds both readers' records, captures Stage A, and checks the registered stop rule. Exit code 3 from the pilot check is an intentional stop: inspect `pilot-analysis-37.json` and do not run the full section. Exit code 0 means continue into the full section.

```text
p37basin\part37\collect37.py generate --model qwen --questions p37basin\part37\questions.json --out p37basin\part37\results\pilot\qwen-gens.jsonl --limit 20 --device xpu
p37basin\part37\collect37.py generate --model olmo --questions p37basin\part37\questions.json --out p37basin\part37\results\pilot\olmo-gens.jsonl --limit 20 --device xpu
p37basin\part37\collect37.py generate --model smol --questions p37basin\part37\questions.json --out p37basin\part37\results\pilot\smol-gens.jsonl --limit 20 --device xpu
p37basin\part37\collect37.py screen --gens p37basin\part37\results\pilot\qwen-gens.jsonl p37basin\part37\results\pilot\olmo-gens.jsonl p37basin\part37\results\pilot\smol-gens.jsonl --out p37basin\part37\results\pilot\kept.json
p37basin\part37\collect37.py build-records --reader qwen --gens p37basin\part37\results\pilot\qwen-gens.jsonl p37basin\part37\results\pilot\olmo-gens.jsonl p37basin\part37\results\pilot\smol-gens.jsonl --kept p37basin\part37\results\pilot\kept.json --questions p37basin\part37\questions.json --pile qwen35\pile10k.jsonl --out p37basin\part37\results\pilot\qwen-records.jsonl
p37basin\part37\collect37.py build-records --reader olmo --gens p37basin\part37\results\pilot\qwen-gens.jsonl p37basin\part37\results\pilot\olmo-gens.jsonl p37basin\part37\results\pilot\smol-gens.jsonl --kept p37basin\part37\results\pilot\kept.json --questions p37basin\part37\questions.json --pile qwen35\pile10k.jsonl --out p37basin\part37\results\pilot\olmo-records.jsonl
p37basin\part37\collect37.py capture --reader qwen --records p37basin\part37\results\pilot\qwen-records.jsonl --out p37basin\part37\results\pilot\qwen-capture --device xpu
p37basin\part37\collect37.py capture --reader olmo --records p37basin\part37\results\pilot\olmo-records.jsonl --out p37basin\part37\results\pilot\olmo-capture --device xpu
p37basin\part37\analyze37.py --capture-qwen p37basin\part37\results\pilot\qwen-capture --capture-olmo p37basin\part37\results\pilot\olmo-capture --pilot-check p37basin\part37\pilot-analysis-37.json
p37basin\part37\collect37.py generate --model qwen --questions p37basin\part37\questions.json --out p37basin\part37\results\full\qwen-gens.jsonl --device xpu
p37basin\part37\collect37.py generate --model olmo --questions p37basin\part37\questions.json --out p37basin\part37\results\full\olmo-gens.jsonl --device xpu
p37basin\part37\collect37.py generate --model smol --questions p37basin\part37\questions.json --out p37basin\part37\results\full\smol-gens.jsonl --device xpu
p37basin\part37\collect37.py screen --gens p37basin\part37\results\full\qwen-gens.jsonl p37basin\part37\results\full\olmo-gens.jsonl p37basin\part37\results\full\smol-gens.jsonl --out p37basin\part37\results\full\kept.json
p37basin\part37\collect37.py build-records --reader qwen --gens p37basin\part37\results\full\qwen-gens.jsonl p37basin\part37\results\full\olmo-gens.jsonl p37basin\part37\results\full\smol-gens.jsonl --kept p37basin\part37\results\full\kept.json --questions p37basin\part37\questions.json --pile qwen35\pile10k.jsonl --out p37basin\part37\results\full\qwen-records.jsonl
p37basin\part37\collect37.py build-records --reader olmo --gens p37basin\part37\results\full\qwen-gens.jsonl p37basin\part37\results\full\olmo-gens.jsonl p37basin\part37\results\full\smol-gens.jsonl --kept p37basin\part37\results\full\kept.json --questions p37basin\part37\questions.json --pile qwen35\pile10k.jsonl --out p37basin\part37\results\full\olmo-records.jsonl
p37basin\part37\collect37.py capture --reader qwen --records p37basin\part37\results\full\qwen-records.jsonl --out p37basin\part37\results\full\qwen-capture --device xpu
p37basin\part37\collect37.py capture --reader olmo --records p37basin\part37\results\full\olmo-records.jsonl --out p37basin\part37\results\full\olmo-capture --device xpu
p37basin\part37\collect37.py stated --reader qwen --records p37basin\part37\results\full\qwen-records.jsonl --out p37basin\part37\results\full\qwen-stated.jsonl --device xpu
p37basin\part37\collect37.py stated --reader olmo --records p37basin\part37\results\full\olmo-records.jsonl --out p37basin\part37\results\full\olmo-stated.jsonl --device xpu
```

The pilot stop rule is true only when, for both readers, every mean-log-prob-regressed authorship AUROC at both readout positions is within 0.05 of its per-layer shuffled-label 95th percentile, and each of the two within-author contrasts has best AUROC within 0.05 of chance at both positions. The pilot is Stage A only and does not run stated readouts. The full run regenerates all 100 questions and records the full screened set; the first 20 are intentionally regenerated under the same pinned settings.

Each capture uses one forward pass per record and saves float16 shards with an index and per-span token log-probabilities and entropies. Sampling seeds 1 and 2 are reset separately for every generated response. SmolLM3 receives `/no_think` in its system prompt; Qwen3 receives `enable_thinking=False`.

## Copy back and analyze on the host

Copy the full artifacts from Windows into `part37/results/full` in the worktree:

```bash
mkdir -p /home/elisepc/green/worktrees/ab-part37/part37/results/full
cp -a /mnt/c/Users/alexa/neuron-probe-gpu/p37basin/part37/results/full/. \
  /home/elisepc/green/worktrees/ab-part37/part37/results/full/
```

From the repository root, run the host analysis (no model or GPU needed). It writes `part37/analysis-37.json` and `part37/analysis-37.md`:

```bash
python part37/analyze37.py \
  --capture-qwen part37/results/full/qwen-capture \
  --capture-olmo part37/results/full/olmo-capture \
  --stated-qwen part37/results/full/qwen-stated.jsonl \
  --stated-olmo part37/results/full/olmo-stated.jsonl \
  --out-dir part37
```

The analysis prints each scored prediction beside its exact threshold rule, including 37.5, 37.6 and 37.10 as HELD and 37.7 as NOT RUN. It reports every stated result with its two-token mass and flags cells whose median mass is below 0.5.
