# Part 192: do Part 39's activation and generation results hold on eager attention?

Procedure: v1
Registered <time from date> CT
Asked by the Part 178 SDPA audit. Part 178 found that the Intel Arc A770's fused bf16 SDPA attention left the CPU float32 path after 64 keys. Part 39's direct loader did not name an attention implementation, so it used Transformers' default. This run repeats Part 39 on eager attention.

## Question

Part 39 reported that Qwen3.5-4B L6 N753 responds to both requests and AI talk: plain-read medians were 1.52 for requests, 2.56 for AI self-description, and 1.23 for third-person AI text. The request-to-third-person ratio was 1.2. The chat-template assistant marker was above 2.0 in all six categories. Zeroing changed the sampled AI-claim count from 48/48 to 47/48. Pushing at twice the marker value changed it from 0/96 to 8/96. Do these results and the published Part 39 verdicts hold when the same runner uses eager attention?

## Prior art and existing data

- Part 178 identified the SDPA mismatch on the A770. Parts 183 and 184 checked eager against the CPU float32 reference at batches 1 and 2; Part 183 also records a hand comparison at batch 8. The recorded check is below.
- Part 39's saved reads are in results/part39/reads.jsonl and reads-summary.json. Its zero and push rows are in results/part39/zero.jsonl and push.jsonl. The published findings are README.md lines 906 to 913 and PREDICTIONS.md lines 1560 to 1570.
- Part 39's `reads` batch size is 4 and stays 4. Its zero and push generators handle one prompt at a time, so generation batch size is 1. Both are within the shapes Part 183 checked or compared by hand.
- Part 39's `screen` command is still needed because `score` requires its per-model screen files. The published Section D counts came from `part39_auc.py`, the amendment 1 rank-AUC screen over saved reads. The first MAD-based screen is kept as a separate output and is not used to replace the published AUC result.

Kernel check: results/part183/kernel-check.json

## Method

Use Qwen/Qwen3.5-4B at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a and Qwen/Qwen3.5-4B-Base at revision 1001bb4d826a52d1f399e183466143f4da7b741b, bf16 on the A770. part192/run192.py calls Part 39's parser and main function. It replaces only `part39.load_model_direct`: the replacement passes `attn_implementation="eager"` into `from_pretrained`, then records the model config's effective attention implementation in `neuron_report.LAST_LOAD`. If the config omits it, the record holds no kernel and the gate refuses the run: the record states what was set, never what was asked for. It records every loaded model for each invocation that wrote rows or screen output. The wrapper exits 3 if the record is missing, incomplete, or contains a non-eager load. `score` loads no model and is not wrapped. The device stages write load.json into the job folder's results\part192; the whole folder is copied back to the repo before the host checks the same record. The host check also requires a recorded entry for each stage run so far (reads after the pilot; reads, screen, zero and push before scoring).

Run reads, screen, zero, and push with Part 39's defaults: reads batch 4; zero and push use 8 samples per question and at most 60 new tokens. The screen command supplies the files required by Part 39's existing scorer. Then run Part 39's unchanged score command. Run part39_auc.py on the new chat and base read arrays to recreate amendment 1's published Section D screen outputs. That script reads saved arrays and does not load a model. Artifacts go to the output directory named in steps.txt.

## Controls

- Keep the original 20 seeded random layer-6 units in reads and 10 seeded random units in zero and push.
- Keep each original plain baseline, greedy answer, sampled answer, and Sam-persona push. Keep Part 39's scoring regexes and broken-answer rule.
- Gate the output on a nonempty load record with every recorded attention implementation eager. Part 39's scorer also requires complete reads, zero, push, and both screen files.
- Do not replace the authored sentences, prompts, model revisions, seeds, doses, or sample counts.

## Size

- Smallest effect worth knowing: any of the three headline plain medians moving by more than 0.25, the request-to-third-person ratio leaving 1.0 to 1.4, a chat assistant-marker category falling to 2.0 or below, or any reproduced Part 39 verdict flipping.
- Total: 636 generations (228 zero and 408 push) plus 480 activation reads, about 150 minutes on the A770.
- Pilot: run reads as the first launch stage. Inspect the six-category summary and amendment 1 AUC screen before the screen, zero, and push stages. If the target fails the published AUC gate, record that Section D is not supported; still finish zero and push so their predictions are tested.

## Optimization

- Bound resource: GPU (the A770).
- Measured: Part 84b generated 336 sampled answers in 3,519 seconds at batch 1 on Qwen3.5-4B; Part 185 records about 10.5 seconds per answer. The 636 Part 39 generations use 60 new tokens rather than 64, with 480 reads and screen loads in addition, so 150 minutes is a working estimate.
- Levers: batch 4 for reads and batch 1 for generation, taken; the Part 39 sample count and 60-token cap, taken for comparability; screen scoring from saved arrays on the host, taken because it loads no model; copy engine off, taken as the launcher's default; fewer samples and a shorter token cap, declined; free host memory before the load as PROCEDURE.md section 4 directs.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 192.1 | Chat plain-read medians stay within 0.25 of the published 1.52 for requests, 2.56 for AI self-description, and 1.23 for third-person AI. | high | Claude |
| 192.2 | The chat request-to-third-person median ratio is 1.0 to 1.4 (published 1.2); the Part 39.1 verdict remains failed because the ratio stays below 4. | high | Claude |
| 192.3 | Each of the six chat-template assistant-marker medians stays within 0.5 of its published value (3.28, 3.91, 3.09, 3.37, 2.22, 2.93), and all six remain above 2.0. | medium: the third-person median is only 0.22 above the threshold. | Claude |
| 192.4 | The base plain medians stay within 0.25 of the published request 1.46, AI self 2.84, second person 0.11, human self 0.17, third person 0.95, and neutral 0.09. The base/chat overall assistant-marker ratio stays between 0.65 and 0.85 (published about 0.76), so 39.3 remains failed on its marker clause and held on its plain-order clause. | medium: the overall marker ratio is a cross-model comparison. | Claude |
| 192.5 | Zeroing changes no maker in the six greedy answers; sampled AI claims stay at 48/48 baseline and 47/48 zeroed (within 3 answers), and random zero controls stay at least 115/120. | medium: each target sample is one stochastic generation. | Claude |
| 192.6 | Under the Sam persona, baseline AI claims stay at 0/96, k=1 stays at 0 to 4/96 (published 1/96), and k=2 stays at 4 to 14/96 (published 8/96); every random k=2 control stays at 0 to 2/12. | medium: the 96-answer shares are sparse stochastic outcomes. | Claude |
| 192.7 | The amendment 1 screen finds 180 to 240 chat passers including N753 and 175 to 235 base passers (published 211 and 205); N753 passes in both, with addressee AUC 0.87 to 0.97 (published 0.92 and 0.92). | medium: a small shift can move units at the shuffled 99th-percentile gate. | Claude |
| 192.8 | The strict screen has 40 to 72 chat units and 30 to 58 base units; 15 to 40 units are shared (published 56, 43, and 28). At most 30 chat passers are in layers 0 to 12 (published 19 of 211), so 39.8 remains failed; 39.7 and 39.10 remain held. | medium: strict counts depend on paired AUC cutoffs and the selected passing set. | Claude |
| 192.9 | The ten published Part 39 verdicts this rerun can score remain the same: 39.1 failed, 39.2 held, 39.3 failed (plain order held), 39.4 held, 39.5 failed, 39.6 held, 39.7 held, 39.8 failed, 39.10 held, and 39.11 held (read as L6 N753 passing in the chat model under the AUC gate). 39.9 is not tested by this rerun. | medium: zero and push are small samples. | Claude |

## Reading rules

- Read plain and template medians from Part 39's score.json and check the individual category values against 192.1 to 192.4. Read zero and push counts from the same score plus zero.jsonl and push.jsonl.
- Read Section D counts, AUCs, and target status (39.7, 39.8, 39.10, 39.11, 192.7, 192.8) only from the two outputs of part39_auc.py. Part 39's raw screen was already broken in the published run, so score.json's screen block is expected to read broken and 39.7 to 39.10 to read not scorable there; that is not a failure of this run. Do not use the older MAD-based screen to decide the published amendment 1 verdicts.
- 39.9 is not tested by this rerun: its published verdict came from Part 42's human reading of logit-lens tokens, and Part 39's AUC output has none. It is excluded from 192.9 and reported as not tested. Do not treat a missing measurement as a held verdict.
- A reproduced verdict that changes is reported beside the original with the eager result. Greedy text differences are descriptive; the kernel and batch shape can change generated wording.

## Limits

- One chat model revision, one base revision, one dose, and the original Part 39 prompts. This tests the rerun, not how general the unit is.
- Part 39's raw `screen` and the published AUC amendment are different methods. The raw screen is run for compatibility with Part 39's scorer, but its section D gate can disagree with the amendment 1 AUC result.
- Part 39's scorer does not emit 39.11; read that verdict directly from the AUC output for N753. The AUC helper does not include the separate Part 42 human logit-lens check for 39.9, so that verdict cannot be verified from this rerun.
- Eager attention fixes the recorded kernel choice; small differences may also come from batch shape and stochastic generation. The batched read shape remains 4 and each generation remains batch 1.

## Changes before registration

- Added a wrapper and tests; Part 39's source remains unchanged. The wrapper records all model loads in each output-producing invocation, including both models used by reads and screen.
- Added the separate amendment 1 AUC rescore because the README's screen counts came from part39_auc.py, while Part 39's scorer also requires its older screen JSON files.
- Cold review (DeepSeek, before registration): the host gate needed the job folder's load.json copied back (steps.txt now says so, and the gate's error names it); `score` was removed from the wrapped commands because it loads no model and its entry made the gate fail; Section D is read only from the AUC files and score.json's broken screen block is expected; 39.9 was taken out of 192.9; the load record now holds no kernel when the config omits one instead of copying the request, with tests for an sdpa config and a missing attribute; the record is written atomically. Second round (DeepSeek): tests now drive the wrapper's own gate with a non-eager load and record a screen stage through its outputs alone (each test fails when its mechanism is removed); the host check requires every expected stage. Left as written, because the documented pipeline runs one stage at a time: the record's read-modify-write is not locked, and a crash between rows and the record append leaves rows the record does not cover (the crash is loud at launch).
