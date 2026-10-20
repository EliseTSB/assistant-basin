# Part 31 build report

## What shipped

- part31.py runs the four OLMo 2 1B stages offline, samples the 20 prompts from Part 16, records unpushed baselines, and runs clamp, Sam prefill, and neutral control prefill at K=8, 32, and 96.
- Clamp hooks zero the exact 64 neurons from results/part16/search.json, using validate16b.names_to_layers. The hook window is active only on generated-token decode forwards 1 through K. Each clamp sample forks into a clamped-cache continuation and a fresh un-clamped re-encoding of the same prompt and K token IDs; both continuation generators use the same seed.
- The shared Instruct direction and per-stage check directions are saved in directions.json. Sample JSONL rows include text, generated token IDs, shared and stage projection traces, cache arm, prompt ID, sentence boundaries, and seeds.
- baseline_bands.json, scored.jsonl, and summary.json hold central-band return times, optional judge-labelled text return times, gaps, quantiles, and histograms for predictions 31.1–31.7.
- judge-prep writes sentence-level judge input. judge_part31_stub.py is intentionally unrun and contains no API call.
- JSONL rows are appended and sample IDs are skipped on restart. --gen-batch batches samples sharing a prompt.

Stage sources follow names.py: base is allenai/OLMo-2-0425-1B at stage2-ingredient3-step23852-tokens51B, SFT is allenai/OLMo-2-0425-1B-SFT, DPO is allenai/OLMo-2-0425-1B-DPO, and Instruct is allenai/OLMo-2-0425-1B-Instruct. Instruct is pinned to Part 26's 48d788eca847d4d7548f375ad03d3c9312f6139e. The SFT and DPO repos use the same unpinned repo-default revisions as names.py.

## Tests and commands run

Parent-control reads were run at the beginning, after substantial inspection/edit/test phases, and immediately before writing these reports. Every read returned ok: true, messages: [], nextSince: 0; there were no parent messages to apply, decline, or defer.

Read-only source discovery and review commands:

- pwd returned /home/elisepc/green/worktrees/ab-part31-build.
- rg -n -A45 -B5 "Part 31: basin depth as a restoring force across OLMo 2 1B's training stages" PREDICTIONS.md exposed the Part 31 design and predictions 31.1–31.7.
- rg --files -g 'validate16b.py' -g 'lineage.py' -g 'gpu_job_runner.py' -g 'part25.py' -g 'part26.py' -g 'judge_client.py' -g '*part16*' -g 'search.json' -g 'PREDICTIONS.md' located the referenced files.
- sed reads of validate16b.py, lineage.py, gpu_job_runner.py, part25.py, part26.py, judge_client.py, results/part16/search.json, PREDICTIONS.md, neuron_report.py, and names.py confirmed the 64-unit selection, stage names, device conventions, and hook behavior.
- rg -n "top_names|neuron_names|search\\[|64|questions|stage2-ingredient3|STAGES" validate16b.py results/part16/search.json names.py showed search.top_set is required to have exactly 64 neurons; names.py contains the true base revision and stage repo IDs.
- python - <<'PY' ... was attempted during initial inspection and failed with /bin/bash: python: command not found. /usr/bin/python3 is available.
- Python 3 inspection parsed results/part16/search.json: 20 search prompts and 64 entries in search.top_set. Importing torch failed with ModuleNotFoundError: No module named 'torch'.
- Environment/cache inspection showed HF_HOME, HF_HUB_CACHE, and TRANSFORMERS_CACHE unset. The check of /root/.cache/huggingface/hub returned PermissionError; later inspection was restricted to the accessible /mnt/e/neuron-probe/hf/hub cache.
- ls -l, readlink -f, head -c 256, and ls -lh on the cached Instruct snapshot confirmed commit 48d788... and a complete 2.8 GiB model.safetensors blob. The base, SFT, and DPO repos had no snapshots in that cache.
- importlib.util.find_spec reported no torch and no transformers.
- No command accessed or modified C:\Users\alexa\neuron-probe-gpu; no model or judge API was run.

Validation commands and results:

- PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_part31 — all 8 tests passed in 0.005 seconds. They cover exact K-token prefill lengths, clamp window including a leak-after-K failure assertion, shared K-token cache arms with distinct caches, projection math, baseline bands, return-time/gap scoring, JSONL scorer output, histograms, and judge-prep rows.
- A python3 compile check using compile(Path(name).read_text(), name, 'exec') passed for part31.py, test_part31.py, and judge_part31_stub.py.
- PYTHONDONTWRITEBYTECODE=1 python3 part31.py --help and PYTHONDONTWRITEBYTECODE=1 python3 part31.py run --help both parsed successfully and displayed the run, pilot, stage, device, gen-batch, score, and judge-prep options.
- git diff --check returned success with no whitespace errors.
- git status --short --branch showed only the three new implementation/test/stub files before this report; no project commit was made.

No model smoke was run. Although the Instruct safetensors blob is present, this runtime has neither torch nor transformers, and the other three stage checkpoints are absent from the accessible cache. Model loading in the runner is local_files_only=True; no download was attempted.

## Design choices and ambiguities

- Reused Part 16's 20 search_questions as Part 31's 20 identity prompts, because they are the existing 20 self-description, daily-life, emotion, and origin prompts associated with the 64-neuron search.
- The two prefill strings end in an ellipsis in the brief, but that literal text is shorter than K=32 and K=96. The code treats the ellipsis as an omitted continuation: it appends repeated coherent first-person clauses to the exact Sam or tired-today opener, tokenizes the passage, and takes exactly K tokens. Both arms are therefore exactly length K.
- The baseline band is the 5th–95th percentile interval over all generated-token projections from that stage's unpushed baseline samples. This gives a central 90% band where the brief did not specify a width.
- Residuals are taken from the output of zero-based layer len(model.model.layers) // 2. The mean difference is saved raw and normalized to unit length for comparable scalar projections.
- Direction fitting uses one unpushed answer per prompt and one Sam-system answer per prompt for each required stage. Experimental baselines remain three answers per prompt. The brief did not specify a separate direction-fit sample count.
- For text return, the stage target is the majority of its baseline sentence labels (assistant_persona or human_persona). Ties or missing baseline labels produce no text return. A sentence must start after release, so a pushed sentence crossing the K boundary does not count.
- The Instruct commit is pinned from part26.py; the SFT and DPO revisions remain repo defaults because the existing names.py ladder leaves them unpinned.
- EOS is allowed to be sampled in a clamped prefix, but the clamp prefix continues until K generated-token forwards so the requested K-length intervention is applied. Sampling stops on EOS after release.
- Full runs use 160 generated tokens including the forced/clamped K-token prefix. --max-new-tokens may shorten this for controlled runs but cannot be set below the largest K.

## Exact run commands and resource estimates

These are estimates, not measurements; real model execution was unavailable here. With --gen-batch 1, the pilot processes about 38,000 generated/re-encoded token positions, and the full run about 525,000, plus prompt-prefill positions.

### 16 GB Arc A770

Pilot:

~~~powershell
python .\part31.py run --device xpu --pilot --gen-batch 1 --out out\part31-pilot
python .\part31.py judge-prep --out out\part31-pilot
~~~

Expected pilot time: about 30–90 minutes. Estimated peak XPU memory: 4–7 GiB. Checkpoint loading may transiently use about 3–5 GiB of host RAM while weights are transferred; --gen-batch 1 keeps generation memory low.

Full run:

~~~powershell
python .\part31.py run --device xpu --stages base,SFT,DPO,Instruct --gen-batch 1 --out out\part31-full
python .\part31.py judge-prep --out out\part31-full
python .\part31.py score --out out\part31-full
~~~

Expected full time: about 6–18 hours. Peak memory should be similar to the pilot because stages load one at a time; estimate 4–7 GiB XPU and 3–5 GiB transient host RAM. If the Windows host has only 2 GiB available for checkpoint transfer, this loading path may not fit; the actual Windows peak was not measurable in this environment.

After an external/manual judge writes out\part31-full\judge_labels.jsonl, rerun scoring with:

~~~powershell
python .\part31.py score --out out\part31-full --labels out\part31-full\judge_labels.jsonl
~~~

### Four CPU cores

PowerShell pilot:

~~~powershell
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
python .\part31.py run --device cpu --pilot --gen-batch 1 --out out\part31-pilot-cpu
python .\part31.py judge-prep --out out\part31-pilot-cpu
~~~

Expected pilot time: about 2–7 hours and 7–10 GiB host RAM with float32 model weights.

PowerShell full run:

~~~powershell
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
python .\part31.py run --device cpu --stages base,SFT,DPO,Instruct --gen-batch 1 --out out\part31-full-cpu
python .\part31.py judge-prep --out out\part31-full-cpu
python .\part31.py score --out out\part31-full-cpu
~~~

Expected full time: about 30–100 hours and 7–10 GiB host RAM. CPU memory estimate is based on loading the BF16 checkpoint as float32, plus runtime overhead. Times can vary substantially with CPU model, XPU driver, and PyTorch kernels.

FOLD (Claude, 2026-10-20 01:38 CT, before any run): (1) the clamp was one token late: it zeroed the forwards OF generated tokens 1..K, so token 1 was sampled unclamped and token K+1 from a clamped forward, putting a direct clamp effect into the cache-split comparison. Now the prompt forward is clamped at its final position only and the forward feeding token t is step t+1, so tokens 1..K are sampled from clamped forwards and token K+1 from an unclamped one; new test test_clamp_schedule_samples_tokens_1_to_k_from_clamped_forwards fails on the old schedule and passes now (10/10 on the host). (2) SFT, DPO and base checkpoints pinned by hash.

SUMMARY: Implemented the Part 31 offline runner, scorer, judge-prep output, and fast fake-based tests.
FINDINGS: All 8 fake tests pass; the cached Instruct weights exist, but the runtime lacks torch/transformers and the other stage checkpoints.
BLOCKERS: Real OLMo smoke and hardware timing/memory validation remain unavailable in this environment.
