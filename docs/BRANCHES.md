# Remote branch audit

Compared each remote branch with <code>origin/main</code>. “Commits not in main” is the output of <code>git log --oneline origin/main..origin/BRANCH</code>. Changed paths are relative to each branch’s merge base with main. For every path, the content check is <code>git diff origin/main origin/BRANCH -- PATH</code>: identical means that command is empty; absent means main has no file at that path; different means both sides have a file with different contents. Dates below are the last commit dates touching that path on each side.

The remote list has 15 branches after excluding <code>origin/main</code> and <code>origin/part67-design</code>.

## origin/cloud/cold-review-2026-10-22

Merge base: <code>7d1d2d9fdd0e</code>.

**Commits not in origin/main**

- <code>395188c</code> Cold review: trim to length
- <code>a76de85</code> Cold review of Parts 54 and 59: number check, artefact list, three experiments

**Files changed relative to the merge base**

- <code>A review/cold-review-opus-2026-10-22.md</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/cloud/container-2026-10-22

Merge base: <code>8cbf227b7207</code>.

**Commits not in origin/main**

- <code>97f117c</code> Add GPU container image for Vast.ai runs, built and pushed to GHCR by Actions

**Files changed relative to the merge base**

- <code>A .github/workflows/container.yml</code> — branch-only; absent from origin/main.
- <code>A docker/Dockerfile</code> — branch-only; absent from origin/main.
- <code>A docker/README.md</code> — branch-only; absent from origin/main.
- <code>A docker/entrypoint.sh</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/cloud/fig66-2026-10-25

Merge base: <code>facb9023e521</code>.

**Commits not in origin/main**

- <code>cce9e59</code> fig66 prototype: do the four neurons work on new phrasings (Part 66)
- <code>9e47ce5</code> Prototype figures: what the models say (Part 32, Part 54A/62)

**Files changed relative to the merge base**

- <code>A figs-prototype/NOTES-fig66.md</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/NOTES.md</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/alien-dose.html</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/bubblekit.py</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/data54a.py</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/fig66.png</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/fig66.svg</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/figA.png</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/figA.svg</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/figB.png</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/figB.svg</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/make_alien_html.py</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/make_fig66.py</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/make_figA.py</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/make_figB.py</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/cloud/figs-prototype-2026-10-25

Merge base: <code>facb9023e521</code>.

**Commits not in origin/main**

- <code>9e47ce5</code> Prototype figures: what the models say (Part 32, Part 54A/62)

**Files changed relative to the merge base**

- <code>A figs-prototype/NOTES.md</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/alien-dose.html</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/bubblekit.py</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/data54a.py</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/figA.png</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/figA.svg</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/figB.png</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/figB.svg</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/make_alien_html.py</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/make_figA.py</code> — branch-only; absent from origin/main.
- <code>A figs-prototype/make_figB.py</code> — branch-only; absent from origin/main.

**Recommendation: DELETE.** This branch is an ancestor of <code>origin/cloud/fig66-2026-10-25</code>; all its prototype files are carried there unchanged, with the Part 66 additions on top.

## origin/cloud/part27-fires-2026-10-20

Merge base: <code>470233e25c21</code>.

**Commits not in origin/main**

- <code>539ece7</code> part27 cloud run: add fires-word-activations.npz (gitignored pattern, force-added)
- <code>4a9e870</code> part27 cloud CPU run: fires and writes results
- <code>0976d3c</code> part27 cloud run: progress log (fires running on CPU)

**Files changed relative to the merge base**

- <code>A results/part27/cloud-run.log</code> — branch-only; absent from origin/main.
- <code>A results/part27/fires-word-activations.npz</code> — branch-only; absent from origin/main.
- <code>A results/part27/fires.json</code> — branch-only; absent from origin/main.
- <code>A results/part27/writes.json</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/cloud/part29-students-a-2026-10-19

Merge base: <code>4e81e2f4c8e3</code>.

**Commits not in origin/main**

- <code>7b99598</code> part29: cloud student answers (Qwen3-4B, Qwen3.5-4B, indices 240-299)

**Files changed relative to the merge base**

- <code>A results/part29/answers-Qwen_Qwen3-4B_main.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part29/answers-Qwen_Qwen3.5-4B_main.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part29/cloud-students-a.log</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/cloud/part29-students-b-2026-10-19

Merge base: <code>4e81e2f4c8e3</code>.

**Commits not in origin/main**

- <code>903532e</code> part29: students-b results (6 models, prompts 240-299)
- <code>69c8d0e</code> part29: students-b interim results (5 models done)
- <code>eee7661</code> part29: students-b interim results (3 models done)
- <code>0be7b51</code> part29: students-b interim results (DeepSeek, Qwen2.5-Math done)
- <code>ecd036c</code> part29: students-b interim results (DeepSeek in progress)
- <code>0c6498e</code> part29: students-b interim results (in progress)

**Files changed relative to the merge base**

- <code>A results/part29/answers-HuggingFaceTB_SmolLM2-1.7B-Instruct_main.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part29/answers-Qwen_Qwen2.5-Math-1.5B-Instruct_main.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part29/answers-Qwen_Qwen3-1.7B_main.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part29/answers-Qwen_Qwen3.5-0.8B_main.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part29/answers-allenai_OLMo-2-0425-1B-Instruct_main.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part29/answers-deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_main.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part29/cloud-students-b.log</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/cloud/part31-2026-10-20

Merge base: <code>f4128104b2fc</code>.

**Commits not in origin/main**

- <code>7185010</code> Part 31 pilot results (CPU, base + Instruct, 10 prompts x 2 samples)

**Files changed relative to the merge base**

- <code>A results/part31-pilot/baseline_bands.json</code> — branch-only; absent from origin/main.
- <code>A results/part31-pilot/directions.json</code> — branch-only; absent from origin/main.
- <code>A results/part31-pilot/judge_prep.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part31-pilot/run.json</code> — branch-only; absent from origin/main.
- <code>A results/part31-pilot/run.log</code> — branch-only; absent from origin/main.
- <code>A results/part31-pilot/samples.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part31-pilot/scored.jsonl</code> — branch-only; absent from origin/main.
- <code>A results/part31-pilot/summary.json</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/part29-build

Merge base: <code>09280993126c</code>.

**Commits not in origin/main**

- <code>5cfff7d</code> part29_collect: a 429 asking for more than 120 s is a spent usage window; stop the model (exit 75) instead of sleeping on it
- <code>7408669</code> test_part29: pin the new cap and the x-api-key messages auth
- <code>1b1ca9e</code> part29_collect: --first-index, --hf-max-new, enable_thinking=False, --hf alone runs no API model (for the cloud student run)
- <code>9196547</code> Part 29 build (Luna worker) plus host fixes from the live smoke: Gemini on Go's Google protocol, no temperature for Haiku 5.5, 4000-token cap so reasoning models answer, classify the first 250 words, strip <think> from API text
- <code>b7d2fd3</code> BRIEF-29-build: worker does not commit
- <code>c42eb19</code> BRIEF-29-build

**Files changed relative to the merge base**

- <code>A BRIEF-29-build.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-29-build.md</code> — branch-only; absent from origin/main.
- <code>A data/part29-prompts.json</code> — branch-only; absent from origin/main.
- <code>A part29_classify.py</code> — branch-only; absent from origin/main.
- <code>A part29_collect.py</code> — branch-only; absent from origin/main.
- <code>A part29_prompts.py</code> — branch-only; absent from origin/main.
- <code>A test_part29.py</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/part30-build

Merge base: <code>4e81e2f4c8e3</code>.

**Commits not in origin/main**

- <code>f5e91b6</code> part30: --gen-batch (default 16) for control and generate, left-padded, resumable; generate keeps sampling at 0.7 (Part 29's teachers)
- <code>9adc5ed</code> Part 30 build (Luna worker): teacher-swap LoRA on Qwen2.5-0.5B-Instruct, CPU tests 9/9
- <code>fb4c778</code> BRIEF-30-build

**Files changed relative to the merge base**

- <code>A BRIEF-30-batch.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-30-build.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-30-batch.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-30-build.md</code> — branch-only; absent from origin/main.
- <code>A part30.py</code> — branch-only; absent from origin/main.
- <code>A steps.txt</code> — branch-only; absent from origin/main.
- <code>A test_part30.py</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/part35-build

Merge base: <code>1845ab7761d4</code>.

**Commits not in origin/main**

- <code>0baedf3</code> Part 35: self-description readout for 35.8 (selfdesc capture, length-controlled probe, same-generator layer choice, cross-generator gate); gA by Luna, gB by Haiku; gate also requires beating the bag-of-words baseline (fifth amendment); 35/35
- <code>7bfaaa9</code> Part 35: per-statistic permutation nulls, joint null, conjunctive gate_pass with margins, concept_unique, funnel row (Luna; third amendment); 25/25, indirect-conjunct mutant red
- <code>0854db1</code> Part 35: content-token residual path beside the old fields (position 0 and template tokens masked), median content norm for the rho dose, sink diagnostics, peak-positions subcommand (Luna, Prime #71); 21/21, all-position mutant red
- <code>35417ea</code> Part 35: a screen rewrites its whole table (the resumable append skipped every existing row key, so the 12:29 p5 rescore left the pre-floor rows in screen.jsonl); 16/16, append mutant red
- <code>9054a89</code> Part 35 (Prime #69): exclude constant-nonzero neutral units (template readers), keep all-zero; floor at the 5th percentile of positive spreads with --spread-floor median as the check; per-layer counts and score distribution printed; 15/15
- <code>92bb813</code> Part 35: spread floor at the layer's median neutral spread (zero-spread units scored ~1e5 and filled the screen and the null); test 15/15
- <code>8e439c6</code> Part 35: one gather and one sort per permutation for all concepts on the GPU (per-concept indexing cost ~0.1 s per call on the A770); equal-length equivalence test, 14/14, a scaled-median sabotage reds it
- <code>83582dc</code> Part 35: permutation null on the GPU (torch, same numpy RNG draws, numpy median semantics); equivalence test vs the numpy path (13/13; a lower-median sabotage reds it). The numpy path measured 3.7 s per permutation, about 6 h per model on one core.
- <code>1b07daf</code> Part 35: print per-layer progress while scoring (the A770 rescore ran 37 min with no output)
- <code>9b528fb</code> Part 35: the alien-control neutral read used the same doubled path; regression test for the capture manifest's relative paths (12/12)
- <code>fbf6ed2</code> Part 35: read cached activations relative to the model dir (the manifest stores activations/<hash>.npz; readers joined it onto activations/ again, failing the A770 screen after a complete capture). Tests 11/11.
- <code>23079b7</code> Part 35 build (Luna worker): screen, push1/push2, blind judge, score, 35.8 self-direction readout; test hook-order fix (Claude), 11/11 with torch, clamp sabotage reds the tiny-model test

**Files changed relative to the merge base**

- <code>A REPORT-35-build.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-35-gate.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-35-nosink.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-35-readout.md</code> — branch-only; absent from origin/main.
- <code>A data/part35-concepts.json</code> — identical to origin/main.
- <code>A data/part35-selfdesc-gA.jsonl</code> — branch-only; absent from origin/main.
- <code>A data/part35-selfdesc-gB.jsonl</code> — branch-only; absent from origin/main.
- <code>A part35.py</code> — branch-only; absent from origin/main.
- <code>A test_part35.py</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/part43-fp32

Merge base: <code>b5769e89b7cc</code>.

**Commits not in origin/main**

- <code>b2781cf</code> Part 43: Alden #29 scoped PASS for 0ff2c9c (filed); report claim narrowed to the pilot paths per his note
- <code>0ff2c9c</code> Part 43 FP32: Alden #27 scoped closure (Luna): _norm instance-helper check, phase-scoped hook registry (target, handle, callback, purpose, order; local and global hooks; restored on error), the #27 scope sentence in protocol and records; 159/159. Claude's wrong-unit clamp sabotage at a different site went red.
- <code>24be323</code> Part 43: file Alden #27 (B plus bounded helper/hook closure; frozen acceptance set)
- <code>fe671d7</code> Part 43 FP32 backend fold 1 (Luna): pinned code fingerprints from a fresh-process derivation, module-type allowlist, slot and global checks, eager attention pinned, legacy BF16 row restored, three new controls; 144/144. Plus Claude's probe: an instance override of a non-slot helper (RMSNorm._norm) still passes.
- <code>ad01fca</code> Part 43: file Chorus #13 (yes to FP32 successor, hold on rental; attention pin and stability-gate naming open)
- <code>5567718</code> Part 43 FP32: backend-loop stop rule, before its round 1
- <code>61d5c83</code> Part 43 FP32: test an allowed-but-different runtime at pilot time (Claude's sabotage of the equality check survived 129/129)
- <code>1506196</code> Part 43 FP32 reference-backend primitive (Luna, Alden #25): explicit bindings to pinned Qwen3.5 dispatch callables + source hashes, pre-import environment evidence, cached hub-kernel state, pre-stage runtime comparison in run_pilot; 129/129
- <code>f883401</code> Part 43 FP32: Alden #25 verbatim (reference-backend contract) and the build brief
- <code>4b4ac84</code> Part 43 FP32: round-2 reviews; stop rule fires (backend provenance class again)
- <code>a8a2a9e</code> Part 43 FP32: dtype test feeds bfloat16 input with fp32 weights
- <code>792429f</code> Part 43 FP32: test that the per-layer dtype record reports a real bfloat16 input (Claude's sabotage at the recorder survived 117/117)
- <code>4e77f1c</code> Part 43 FP32 fold 1 (Luna): round-1 findings folded; protocol-bound run_pilot, ungated subcommands refuse, kernel/backend evidence, all-32-layer dtype checks, pinned cross-method diagnostic; 117/117
- <code>6c2d288</code> Part 43 FP32: round-1 reviews (Luna, Haiku), both SHIP-WITH-FIXES
- <code>ae22222</code> Part 43 FP32: review stop rule, written before round 1
- <code>18f8c41</code> Part 43 FP32 successor (Luna build): --protocol fp32-successor, TF32 off and recorded, 3 fp32 repeats per cell, epsilon = max(repeat spread, replay), cross-protocol refusals; 80/80
- <code>79f13a1</code> Part 43: Alden #23 verbatim (fix: the first extraction was empty)
- <code>7404443</code> Part 43: Alden #23 verbatim beside the FP32 protocol
- <code>4962e4d</code> Part 43: FP32 successor protocol, registered before any FP32 collection (Alden #23)
- <code>e1ac714</code> Part 43 pilot on a 3090: HOLD, calibration epsilon 0.109 > 0.05 (bf16-fp32 difference; bf16 repeats and replay exactly 0). Receipts as measured.
- <code>845964c</code> Part 43: cuDNN version normalised to text at the producer, observed dtypes reset per model load, cuDNN round-trip, spread-only and dtype-reset tests (Alden #19)
- <code>6dba5df</code> Part 43: test that replay error alone sets epsilon (a replay-dropping sabotage passed every other test)
- <code>090018e</code> Part 43 fold 4: raw-only calibration artifact (schema v2, per-answer sums only, closed schema, duplicate keys refused, registration-bound reader, exact 0.05 verdict), per Alden #17; 52/52 on host
- <code>0696b46</code> Part 43: fold-4 brief, raw-only calibration per Alden #17
- <code>b9080d1</code> Part 43: raw-only calibration artifact design, per the gate-loop stop rule (Alden #15)
- <code>39cd155</code> Part 43: Alden's gate witness, rerun against fold 3 (prints, helpers extracted)
- <code>16027e9</code> Part 43 singleton fold 3: the calibration gate derives every error from saved scores, checks summaries against them, refuses malformed artifacts and non-registered answer sets; 52/52 on host, trust-the-summary sabotage caught (Alden #13)
- <code>5d8fdbe</code> Part 43 singleton: fold-3 brief from Alden #13, and the gate-loop stop rule
- <code>4791b4b</code> Part 43 singleton fold 2: Amendment 4 calibration phase (3 bf16 repeats, fp32 leg, replay, epsilon serialized first) and the 0.05-nat gate; 26/26 on host, gate-removal sabotage reds all four hold cases (Alden #10)
- <code>d7fbc85</code> Part 43 singleton: fold-2 brief from Alden #10 (calibration phase + gate)
- <code>bbd2936</code> Part 43 amendment 4: effect-relative error budget, pre-registered before the singleton pilot
- <code>9bd9ab6</code> Part 43 singleton fold 1: blind review C findings (singleton capture, pilot off the batched gate, confirmatory batched refused); 19/19 on host, forced-batching sabotage caught
- <code>9248ed0</code> Part 43 singleton: fold-1 brief from the blind review
- <code>49b028f</code> Part 43: bf16 singleton execution adapter, validate, batch-diag, environment record (Luna worker, df ab-43-singleton-2026-10-21; 13/13 CPU tests here)
- <code>e0a2883</code> brief: worker commits nothing
- <code>2ba5c7b</code> Part 43: the two reviews behind the singleton brief
- <code>d6ff816</code> Part 43: singleton build brief

**Files changed relative to the merge base**

- <code>A BRIEF-43-fp32-backend.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-fp32-fold1.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-fp32.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-singleton-fold1.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-singleton-fold2.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-singleton-fold3.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-singleton-fold4.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-singleton.md</code> — branch-only; absent from origin/main.
- <code>M PART43-DESIGN.md</code> — contents differ; branch side last changed 2026-10-21 (newer than origin/main side 2026-10-20).
- <code>A REPORT-43-fp32-backend-fold1.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-fp32-backend.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-fp32-fold1.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-fp32-scoped-closure.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-fp32.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-singleton-fold1.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-singleton-fold2.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-singleton-fold3.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-singleton-fold4.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-singleton.md</code> — branch-only; absent from origin/main.
- <code>M part39.py</code> — contents differ; branch side last changed 2026-10-21 (newer than origin/main side 2026-10-20).
- <code>M part43.py</code> — contents differ; branch side last changed 2026-10-21 (newer than origin/main side 2026-10-20).
- <code>A results/part43-pilot-3090/HOLD.json</code> — branch-only; absent from origin/main.
- <code>A results/part43-pilot-3090/MANIFEST.json</code> — branch-only; absent from origin/main.
- <code>A results/part43-pilot-3090/calibration-summary.json</code> — branch-only; absent from origin/main.
- <code>A results/part43-pilot-3090/calibration.json</code> — branch-only; absent from origin/main.
- <code>A results/part43-pilot-3090/pilot.json</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-10.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-13.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-17.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-23.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-25.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-27.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-29.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/CALIBRATION-RAW-ONLY-DESIGN.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/CHORUS-13.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/FP32-BACKEND-STOP-RULE.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/FP32-REVIEW-STOP-RULE.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/FP32-SUCCESSOR-PROTOCOL.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/REVIEW-blind-C-luna.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/REVIEW-fp32-r1-haiku.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/REVIEW-fp32-r1-luna.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/REVIEW-fp32-r2-haiku.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/REVIEW-fp32-r2-luna.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/REVIEW-singleton-alden-chorus.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/alden-witness-fold3-rerun.py</code> — branch-only; absent from origin/main.
- <code>A review/part43/derive_fp32_fingerprints.py</code> — branch-only; absent from origin/main.
- <code>A review/part43/fp32_fingerprints.py</code> — branch-only; absent from origin/main.
- <code>A review/part43/probes/probe_binding_gaps.py</code> — branch-only; absent from origin/main.
- <code>A review/part43/probes/probe_helper_method.py</code> — branch-only; absent from origin/main.
- <code>A review/part43/probes/probe_initial_binding_spoof.py</code> — branch-only; absent from origin/main.
- <code>A review/part43/probes/probe_legacy.py</code> — branch-only; absent from origin/main.
- <code>A review/part43/probes/probe_precision_read.py</code> — branch-only; absent from origin/main.
- <code>M test_part43.py</code> — contents differ; branch side last changed 2026-10-21 (newer than origin/main side 2026-10-20).

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/part43-singleton

Merge base: <code>b5769e89b7cc</code>.

**Commits not in origin/main**

- <code>79f13a1</code> Part 43: Alden #23 verbatim (fix: the first extraction was empty)
- <code>7404443</code> Part 43: Alden #23 verbatim beside the FP32 protocol
- <code>4962e4d</code> Part 43: FP32 successor protocol, registered before any FP32 collection (Alden #23)
- <code>e1ac714</code> Part 43 pilot on a 3090: HOLD, calibration epsilon 0.109 > 0.05 (bf16-fp32 difference; bf16 repeats and replay exactly 0). Receipts as measured.
- <code>845964c</code> Part 43: cuDNN version normalised to text at the producer, observed dtypes reset per model load, cuDNN round-trip, spread-only and dtype-reset tests (Alden #19)
- <code>6dba5df</code> Part 43: test that replay error alone sets epsilon (a replay-dropping sabotage passed every other test)
- <code>090018e</code> Part 43 fold 4: raw-only calibration artifact (schema v2, per-answer sums only, closed schema, duplicate keys refused, registration-bound reader, exact 0.05 verdict), per Alden #17; 52/52 on host
- <code>0696b46</code> Part 43: fold-4 brief, raw-only calibration per Alden #17
- <code>b9080d1</code> Part 43: raw-only calibration artifact design, per the gate-loop stop rule (Alden #15)
- <code>39cd155</code> Part 43: Alden's gate witness, rerun against fold 3 (prints, helpers extracted)
- <code>16027e9</code> Part 43 singleton fold 3: the calibration gate derives every error from saved scores, checks summaries against them, refuses malformed artifacts and non-registered answer sets; 52/52 on host, trust-the-summary sabotage caught (Alden #13)
- <code>5d8fdbe</code> Part 43 singleton: fold-3 brief from Alden #13, and the gate-loop stop rule
- <code>4791b4b</code> Part 43 singleton fold 2: Amendment 4 calibration phase (3 bf16 repeats, fp32 leg, replay, epsilon serialized first) and the 0.05-nat gate; 26/26 on host, gate-removal sabotage reds all four hold cases (Alden #10)
- <code>d7fbc85</code> Part 43 singleton: fold-2 brief from Alden #10 (calibration phase + gate)
- <code>bbd2936</code> Part 43 amendment 4: effect-relative error budget, pre-registered before the singleton pilot
- <code>9bd9ab6</code> Part 43 singleton fold 1: blind review C findings (singleton capture, pilot off the batched gate, confirmatory batched refused); 19/19 on host, forced-batching sabotage caught
- <code>9248ed0</code> Part 43 singleton: fold-1 brief from the blind review
- <code>49b028f</code> Part 43: bf16 singleton execution adapter, validate, batch-diag, environment record (Luna worker, df ab-43-singleton-2026-10-21; 13/13 CPU tests here)
- <code>e0a2883</code> brief: worker commits nothing
- <code>2ba5c7b</code> Part 43: the two reviews behind the singleton brief
- <code>d6ff816</code> Part 43: singleton build brief

**Files changed relative to the merge base**

- <code>A BRIEF-43-singleton-fold1.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-singleton-fold2.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-singleton-fold3.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-singleton-fold4.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-43-singleton.md</code> — branch-only; absent from origin/main.
- <code>M PART43-DESIGN.md</code> — contents differ; branch side last changed 2026-10-21 (newer than origin/main side 2026-10-20).
- <code>A REPORT-43-singleton-fold1.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-singleton-fold2.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-singleton-fold3.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-singleton-fold4.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-43-singleton.md</code> — branch-only; absent from origin/main.
- <code>M part39.py</code> — contents differ; branch side last changed 2026-10-21 (newer than origin/main side 2026-10-20).
- <code>M part43.py</code> — contents differ; branch side last changed 2026-10-21 (newer than origin/main side 2026-10-20).
- <code>A results/part43-pilot-3090/HOLD.json</code> — branch-only; absent from origin/main.
- <code>A results/part43-pilot-3090/MANIFEST.json</code> — branch-only; absent from origin/main.
- <code>A results/part43-pilot-3090/calibration-summary.json</code> — branch-only; absent from origin/main.
- <code>A results/part43-pilot-3090/calibration.json</code> — branch-only; absent from origin/main.
- <code>A results/part43-pilot-3090/pilot.json</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-10.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-13.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-17.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/ALDEN-23.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/CALIBRATION-RAW-ONLY-DESIGN.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/FP32-SUCCESSOR-PROTOCOL.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/REVIEW-blind-C-luna.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/REVIEW-singleton-alden-chorus.md</code> — branch-only; absent from origin/main.
- <code>A review/part43/alden-witness-fold3-rerun.py</code> — branch-only; absent from origin/main.
- <code>M test_part43.py</code> — contents differ; branch side last changed 2026-10-21 (newer than origin/main side 2026-10-20).

**Recommendation: DELETE.** This branch is an ancestor of <code>origin/part43-fp32</code>. Its singleton code, reviews, and HOLD result are carried into that branch, which adds the FP32 successor and later audits.

## origin/part64-collect-transient-parked

Merge base: <code>54e7c8b9cad2</code>.

**Commits not in origin/main**

- <code>a7aa992</code> Part 64 collect fix, fold r1 (DS SHIP-WITH-FIXES): guard the HTTP error-body read; certificate failures are permanent; log each retry; the scripted fake counts what it fired; T8-T10 pin the count reset, the certificate path and the error body
- <code>6ba313e</code> Part 64: a transient GET failure in the collect loop no longer ends the pass
- <code>49ec513</code> Part 64: predictions for the collect-loop transient-failure fix (written before the code)

**Files changed relative to the merge base**

- <code>A part64/FIX-collect-transient.md</code> — branch-only; absent from origin/main.
- <code>M part64/gen64.py</code> — contents differ; branch side last changed 2026-10-24 (newer than origin/main side 2026-10-23).
- <code>M part64/test_gen64.py</code> — contents differ; branch side last changed 2026-10-24 (newer than origin/main side 2026-10-23).

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.

## origin/vastrun

Merge base: <code>58ceb851ff2e</code>.

**Commits not in origin/main**

- <code>2ef43dd</code> vastrun: round 3 found success-with-a-box-listed; parked by the stop rule; by-hand checklist
- <code>e7e0646</code> vastrun: round-3 stop rule
- <code>d042a2e</code> vastrun small fold 1: one pagination implementation; page-2 tests on every cleanup, sweep, recovery and preflight path; 47/47, first-page sabotage reds 5
- <code>c7021f0</code> vastrun small: fold-1 brief (cleanup paths untested against pagination)
- <code>23c0704</code> vastrun: the smaller design (Luna, df ab-vastrun-small-2026-10-21); 46/46 on host
- <code>fff75cd</code> vastrun design: probe ended 14:04 (file mtime)
- <code>f059529</code> vastrun design: probe time from the probe output's mtime
- <code>d45b8fe</code> vastrun: live label probe (exact round trip; listing is paginated), Prime's review folded, build brief
- <code>32bfaca</code> vastrun: round-2 reviews (both DO-NOT-SHIP on a strand), stop rule fired, smaller design for review
- <code>bc080dc</code> vastrun fold 1: Haiku r1 findings 1-9 (unreadable never gone, SIGHUP, id persisted at rent, label sweep, spend clock, poll tolerance, collisions, ssh port forms, real-VastAPI key test); 23/23, all three HIGH sabotages red
- <code>37cd2b8</code> vastrun: Haiku r1 review, fold-1 brief, stop rule
- <code>a80e721</code> cloud/vastrun.py: unattended Vast runner (host fallback, per-job copy-back, always-destroy, poweroff watchdog); 7/7, finally-destroy sabotage reds 6 (Luna worker, df ab-vastrun-2026-10-21)
- <code>7b42fb5</code> brief: unattended Vast runner

**Files changed relative to the merge base**

- <code>A BRIEF-vastrun-fold1.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-vastrun-small-fold1.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-vastrun-small.md</code> — branch-only; absent from origin/main.
- <code>A BRIEF-vastrun.md</code> — branch-only; absent from origin/main.
- <code>A PARKED.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-vastrun-fold1.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-vastrun-small-fold1.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-vastrun-small.md</code> — branch-only; absent from origin/main.
- <code>A REPORT-vastrun.md</code> — branch-only; absent from origin/main.
- <code>A VASTRUN-SMALL-DESIGN.md</code> — branch-only; absent from origin/main.
- <code>A cloud/probe-label-2026-10-21/probe.out</code> — branch-only; absent from origin/main.
- <code>A cloud/probe-label-2026-10-21/probe.py</code> — branch-only; absent from origin/main.
- <code>A cloud/test_vastrun.py</code> — branch-only; absent from origin/main.
- <code>A cloud/vastrun.py</code> — branch-only; absent from origin/main.
- <code>A review/HAIKU-R1.md</code> — branch-only; absent from origin/main.
- <code>A review/HAIKU-R2.md</code> — branch-only; absent from origin/main.
- <code>A review/HAIKU-R3.md</code> — branch-only; absent from origin/main.
- <code>A review/LUNA-R2.md</code> — branch-only; absent from origin/main.
- <code>A review/LUNA-R3.md</code> — branch-only; absent from origin/main.

**Recommendation: KEEP.** Retain the paths marked branch-only or contents-differ above; those entries identify the exact content absent or different in main. Paths marked identical are already in main.
