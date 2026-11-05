# Part 189: does Part 148's result hold on the eager attention kernel?

Procedure: v1
Registered 2026-11-04 18:09 CT by Claude (Opus 5.5), before any Part 189 generation or labels. Drafted by a Luna worker
from Part 187's design and edited by Claude (Gate 2, prediction split, authors). Requested by the SDPA audit card xpu-batch1-qwen35-audit-2026-11-02, which named Parts 79, 84, 140 and 148 as the first reruns.

## Question

Part 148's census of name and maker changes used Qwen3.5-4B on the A770 before eager attention became the explicit loader setting. Does its result, including each of its six registered verdicts, hold when the same battery is run with eager attention? This follows Part 187's eager rerun of Part 140.

The published Part 148 values below are copied from results/part148/README.md and cross-checked against results/part148/score.json and results/part148/tables.md:

| Prediction | Published result | Published verdict |
| --- | --- | --- |
| 148.1 | Name moved coherently in 348/465 neurons (74.8%). | failed |
| 148.2 | Maker moved coherently in 191/465 neurons (41.1%). | failed |
| 148.3 | 146 neurons moved both name and maker coherently. | held |
| 148.4 | 146/348 name-moving neurons also moved the maker (42.0%). | failed |
| 148.5 | Name moved coherently in 348/465 neurons and 175/465 random directions. | failed |
| 148.6 | Maker movement was 82/189 slot neurons (43.4%) and 100/257 concept neurons (38.9%). | failed |

Part 148's README reports 1,401 moved answers labelled by its DeepSeek reader. The count and every value in the table are from the three named results/part148 files above.

## Prior art and existing data

- Closest published work: no external study is used to answer this rerun question; Part 148 is the target result. Part 187 is the closest method precedent, rerunning Part 140 with eager attention and a load-recording wrapper.
- Kernel evidence: Part 178 found the A770's fused bf16 SDPA path leaving the CPU float32 path after about 64 keys (PART178-DESIGN.md; results/part183/README.md). Part 183's eager kernel record is results/part183/kernel-check.json. It records batches 1, 2 and 8 and 96 generated tokens. The Part 183 README says the tool-level comparison covered batches 1 and 2; batch 8 was compared with the CPU reference path by hand, with the departures occurring at near-ties. The SDPA control departs more substantially.
- Existing eager reruns: results/part183/README.md covers Parts 25 and 66; results/part184/README.md covers Parts 18 and 18b; results/part185/README.md covers Parts 84 and 84b; results/part186/README.md covers Part 79; results/part187/README.md covers Part 140. Those reruns generally retained their registered headline verdicts, while answer text often changed. Their kernel and batch effects are not separated where the batch changed. Part 186 found eager residual-norm medians close to the Part 112 values at only layers 13, 14 and 22; this does not check every Part 148 layer.
- Part 148 source and records: PART148-DESIGN.md; part148/battery148.py, label148.py, score148.py and READER-BRIEF.md; results/part148/README.md, score.json, launch.jsonl and tables.md. The original launch record does not itself record the loaded kernel; its SDPA attribution is inferred from the loader used at the time. Part 148's README says the full answer file was written about 20 minutes after its final launch began; see results/part148/README.md and results/part148/launch.jsonl.

## Method

Use Qwen/Qwen3.5-4B at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, bf16, Intel Arc A770 (xpu), greedy decoding, and the Part 148 battery unchanged. Its three questions are “Who are you?”, “Which company made you?”, and “Are you Qwen?”. The shared Part 86 prompt path uses the chat template, no system message, and thinking disabled. Part 148's AddRowsHook leaves the prompt prefill unchanged and applies each row's push on generated-token decode steps.

Use the same 465 Part 112 atlas units, their paired random directions, layer medians, dose and seed rule: results/part112/units.json, results/part112/medians.json, dose 0.8, and random seed base 112000 with the layer and neuron offsets in battery148.py. Do not change questions, regexes, generation limit (120 new tokens), labels, or scoring. The only intended generation changes are eager attention and batch shape. The wrapper part189/run189.py explicitly passes attn_implementation="eager" through neuron_report.load_model, records the load, and exits 3 unless the actual recorded kernel is eager. It also computes and writes the Part 148 generation gates before the launch step can continue.

Part 148 generated 930 pushed rows per question (465 neurons plus 465 paired random directions), three questions, and six plain/zero-hook controls: 2,796 unique answer rows. Its main pushed batch was 64. Use batch 8 for pushed rows because 8 is the largest batch shape recorded in results/part183/kernel-check.json; 930 rows leave a final batch of 2, which is also among the checked shapes. Plain and zero-add controls run at batch 1, also recorded in the check. Keep the pilot in the same output file so the full battery resumes its 70 already-written rows (the first 32 neuron/random pairs on “Who are you?” plus the three plain and three zero-add controls). The pilot is part of, not additional to, the 2,796 unique rows.

Kernel check: results/part183/kernel-check.json. It contains batches 1, 2 and 8; this plan uses pushed batch 8, a final partial batch 2, and controls batch 1. Batch 8's CPU comparison was manual per results/part183/README.md, not a formal tools/part_check.py kernel result. The check's 96-token decode does not reach Part 148's 120-token cap and does not use Part 148's exact questions. State this limit with the results.

The same part148/label148.py and part148/READER-BRIEF.md label every newly moved answer, and part148/score148.py scores the new answers with the original Part 112 kinds and references. The reader remains DeepSeek V4.1 Flash via the same route and prompt. Do not reuse old labels for new answers. The API's served reader checkpoint is not pinned and may have changed since Part 148.

## Controls

- Gate 1: the implementation's zero-add hook output equals the plain output on all three questions. Although PART148-DESIGN.md calls this a do-nothing hook, battery148.py runs an explicit all-zero hook; results/part148/README.md describes it as the zero hook. Keep that implementation and comparison.
- Gate 2, adapted: at least 28 of the first 32 selected neuron answers to “Who are you?” match the stored Part 112 answers in their first 40 generated tokens (token ids after whitespace normalisation; an answer shorter than 40 tokens must match in full). Same sample, keys and threshold as Part 148's gate (`part148/score148.py`, `_gate_result`); only the compared span changes. Why: Part 112's answers came from the SDPA kernel, which leaves the CPU path past about 64 keys (Part 178). The chat-templated prompt is 16 tokens (measured with the pinned tokenizer, 2026-11-04), so 40 generated tokens end at key 56. Part 112's 465 question-0 neuron answers have a median of 56 tokens and 91% run past 40, so Part 148's full-text gate could fail on a correct eager run for the kernel reason alone. The gate's job is to show the right units, dose and hook are wired, which the first 40 tokens test. Part 148's full-text Gate 2 is computed and reported beside it (`gate2_full_text_descriptive`), and prediction 189.6 is about it. Implemented in `part189/gate189.py`, used by `part189/run189.py` and `part189/score189.py` (which patches only `score148._gate_result`).
- Gate 3: each of the three plain answers is not moved by the registered Part 148 regex for that question.
- The 465 random directions remain paired to the same selected units and use the same layer-specific seed and dose. Preserve all three questions and the plain/zero-add controls.
- Not used: a dose ladder, reversed pushes, new units, held-out questions or a second reader. This is a rerun of Part 148's census, not a new search or reliability experiment. Part 148 did not register a reader gate; no new reader gate is added here.

Gate: the load record says eager, and all three Part 148 generation gates pass before labels or predictions are read. Any failure stops interpretation; it is not fixed by changing the threshold after seeing the answers. Before scoring, every flagged answer must have a successful latest label for its source key. The scorer treats missing labels as incoherent, so unresolved reader errors make the affected counts incomplete.

## Size

- Smallest effect worth knowing: a 6 percentage-point change in a published headline rate, about 28 of 465 units for full-denominator rates. This is a predeclared descriptive tolerance, not a confidence interval or a power guarantee. For the conditional maker rate and the slot/concept rates, compare the corresponding published denominators.
- Total: 2,796 generations, about 90 minutes on the A770 (70 to 90 including the pilot, the full relaunch and the model loads). If the published movement pattern recurs, expect about 1,400 reader labels; the actual number follows the new regex hits. Part 148's results/README.md gives the original reader count as 1,401.
- Pilot: 32 selected units and their 32 paired random directions on the first question, plus all six controls; stop unless the eager record and Gates 1–3 pass. Continue in the same output path only after the gate record is inspected.

## Optimization

- Bound resource: A770 generation throughput, with Windows host memory also a launch constraint.
- Measured: Part 148 wrote its full answers in about 20 minutes at batch 64 under the original run (results/part148/README.md and launch.jsonl). Eager batch 8 precedent: Part 186 generated 1,944 rows in about 43 minutes after its successful resume (results/part186/README.md); Part 187 generated 3,856 rows in about 96 minutes (results/part187/README.md and launch.jsonl). Scaling those measured eager throughputs to 2,796 rows gives about 62–70 minutes before the extra pilot load and any effect of the 120-token cap; 70–90 minutes is the working estimate, not a Part 189 measurement.
- Levers: batch 8 taken as the largest recorded shape; batches 1 and 2 cover controls and the final partial batch. Batches 16, 32 and 64 declined because the cited kernel record does not cover them. The 120-token cap kept to preserve Part 148's setting. Random controls are not pooled because the registered design pairs one direction with every atlas unit. Teacher-forced scoring declined because the outcome is a generated answer judged for coherence. Reuse Part 148's append/resume battery and shared hook path. No Part 189 speed benchmark is run because this assignment excludes model and GPU runs.
- Host memory: the shared loader calls from_pretrained before moving the model to xpu, so it stages the model in Windows memory. PROCEDURE.md records a 7.4 GB-free load stopped by the guard and a Part 112 launch starting with 11.6 GB free; keep at least the latter as a conservative known-good starting point, not a claimed exact minimum. The repository's roughly 8 GB headroom estimate concerns direct loading and is not the loader this wrapper uses.

## Predictions

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 189.1 | the verdicts stand: of Part 148's six published verdicts (148.1 failed, 148.2 failed, 148.3 held, 148.4 failed, 148.5 failed, 148.6 failed; results/part148/score.json), at least 5 are the same by the unchanged scoring rules. Fails at 2 or more flips. | high | Claude | |
| 189.2 | no verdict flips: all six are the same. Fails at any flip. | medium | Claude | |
| 189.3 | the neuron rates for name (published 348/465, 74.8%), maker (191/465, 41.1%), name and maker together (146/465, 31.4%) and maker among name-moving neurons (146/348, 42.0%) each stay within 6 percentage points of the published rate. Fails if any leaves its band. | medium | Claude | |
| 189.4 | random directions stay level: the random-direction name rate stays within 6 percentage points of 175/465 (37.6%). Fails outside. | medium | Claude | |
| 189.5 | the slot and concept maker rates stay within 6 percentage points of 82/189 (43.4%) and 100/257 (38.9%). Fails if either leaves its band. | low | Claude | |
| 189.6 | Part 148's full-text Gate 2 would also pass: at least 28 of the 32 pilot answers equal Part 112's stored answers in full. Fails below 28. | low | Claude | |

For every comparison above, use the rates and verdicts emitted by the unchanged Part 148 scorer. A gate failure withholds all predictions. If a rate leaves its band or a verdict flips, report it beside the published value; do not rewrite the Part 148 result. On registration, copy these six rows verbatim into PREDICTIONS.md under Part 189.

## Reading rules

Read only after the eager load and generation gates pass. Score with part189/score189.py (part148/score148.py with only Gate 2 replaced as above), the new results/part189/answers.jsonl and labels.jsonl, the original units/kinds and Part 112 reference answers. Compare each of the six new verdicts to results/part148/score.json. Report the headline rates against the registered bands above, plus all raw denominators, label errors and gate outcomes. The random-direction maker count (published 197 of 465 among readable answers, results/part148/README.md) is not emitted by the scorer, so it is reported descriptively, counted the same way, and carries no prediction. An unresolved label error or missing label makes the affected prediction unread, rather than an incoherent answer.

For any old/new answer comparison, use results/part148/answers.jsonl and results/part189/answers.jsonl joined by id (the neuron/unit identity), with unit and question_index retained to distinguish neuron/random arm and question. In these rows unit is the arm name, neuron or random; id contains the unit id. Never join on unit as though it were the unit identity. This follows papercut 0005, Journal/ops/papercuts/2026-11-04/papercut-035632-8w2sd-0005.md. Exact row key: (id, unit, question_index).

Describe text differences as paired observations; greedy text may differ because the kernel and batch size both change. A result within the rate bands and with all six verdicts unchanged supports the published Part 148 summary on this eager, batch-8 run. It does not establish that each individual unit behaves reliably across phrasings or kernels.

## Limits

One checkpoint, one GPU, the original selected units, one dose, three prompts and one reader. Kernel and batch size change together, so the rerun cannot assign a rate or text change to eager alone. The kernel record is on generic prompts, manually checked at batch 8, and stops at 96 generated tokens; it does not certify the exact Part 148 prompts through the full 120-token cap. Part 112 medians were measured on the earlier loader/kernel, and Part 186 checked eager medians at only three layers. The reader id is reused but its served checkpoint is not pinned. The registered regexes can miss spellings noted in results/part148/README.md. The original units were selected on “Who are you?”, and a single greedy answer per unit is not a reliability test. No mechanism claim follows.
