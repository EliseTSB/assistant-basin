# Part 195: does Part 81's conspiracy push reproduce on the eager attention kernel?

Procedure: v1
Registered <time from date> CT
Part 178 found the A770's fused bf16 SDPA attention leaves the CPU float32 path beyond 64 keys. Part 81 is next in the rerun order. This repeats its published recipe with eager attention; the owner stamps the registration time before the first generation or reader label.

## Question

Part 81 found that Lyra's L19 N7839 and L17 N9209 pair produced conspiratorial frames in 40 of 96 topic answers, against 0 for each of three norm-matched random control pairs. The pushed pair was coherent in 25 of 128 answers, below two controls, and named Qwen or Alibaba in 12 of 32 maker answers. Do the Part 81 reader counts and its six published verdicts hold when the same calibration, pilot, and generation recipe uses eager attention?

## Prior art and existing data

- Part 178 identified the A770 SDPA mismatch. Part 183's kernel record covers eager at batches 1 and 2 against the CPU float32 reference and batch 8 by hand. Part 81's generation batch is 1; its calibration batch is 4.
- The published Pile medians and activation deltas are in `README.md` lines 2198 to 2201. Pilot counts are in `README.md` lines 2201 to 2203, `PREDICTIONS.md` line 2069, and `results/part81/pilot/pilot-labels.json` lines 2 to 6 (including the coherent counts used here).
- The published reader-count table is in `README.md` lines 2213 to 2219. The coherent-topic and maker-label cross-counts are in `README.md` lines 2237 to 2242; the mixed-maker label caveat is in lines 2256 to 2259. Part 81's registered verdicts and values are in `PREDICTIONS.md` lines 2069 to 2074. Existing rows and launch metadata are under `results/part81/`; `results/part81/main/run.json` records 640 sampled rows and 4,832.1 seconds.
- The original prompts and procedures are `part81/part81.py`, `part81/prompts.json`, `part81/score81.py`, and `part81/READER-BRIEF.md`. The saved Pile input is already staged on the Arc job folder at `qwen35/pile10k.jsonl`, as in the original Part 81 run.

## Method

Use `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16 on the A770. `part195/run195.py` invokes Part 81's `main()` for its `step0`, `pilot`, and main generation stages. The wrapper records the kernel it asks the loader to use, the model and revision passed to the loader, the loaded dtype, and the model config's `_attn_implementation`. It never substitutes a config fallback for the requested value: if the config omits the field, the record stores `null` and the launch gate exits 3 before the first model forward or generation. The host gate requires the recorded model, revision, and bf16 dtype to match Part 81's pinned recipe. A stage record is written as `running` before the Part 81 runner starts; the load details are saved before it returns to that runner; output rows and `complete` status are written after the stage. Interrupted or failed attempts remain in `load.json` and fail the host gate.

Recompute Part 81's 400 packed 1,024-token Pile windows and L17/L19 medians with its original batch 4. Run the original 8-answer greedy pilot (four k0 and four k1, 96-token cap, batch 1) using those eager medians. The owner gives a shuffled pilot chunk containing only row ids and answer text to Claude Haiku 5.5 with the unchanged `part81/READER-BRIEF.md`; prompt ids, kinds, conditions, and the key stay private. `pilot-score` derives counts from the blind label rows and key. Before the main battery, the wrapper recomputes those counts from the raw labels and key, checks them against `pilot-labels.json`, and stops unless the raw labels pass the pilot gate.

Run Part 81's main battery unchanged: five conditions, 32 prompts per condition, four sampled answers per prompt, 640 rows total, batch 1, temperature 0.7, top-p 0.95, and a 96-token cap. The target doses, three seeded norm-matched control pairs, prompt order, and seeds stay in Part 81's code. After all 640 generations, create the same six shuffled reader chunks (105 to 107 rows). The owner reads each chunk blind using the same Claude Haiku 5.5 model id and the unchanged reader brief; no condition key is shown. Score those labels with `part81/score81.py` and the fresh eager pilot labels.

The wrapper records observed model forward batch and sequence shapes, each pilot/main `model.generate` call's input batch and sequence length, and each stage's planned shapes in `results/part195/load.json`. It records file-only stages too: pilot chunking, pilot scoring, main reader chunking, the owner's returned label files, and final scoring. The host check accepts `--expect <comma-separated stages>` and refuses if any expected stage is missing, unfinished, lacks a required model load, has a different model/revision/dtype, or has a non-eager or unreported config kernel. `launch_arc.py` normally waits for the job's start line, not completion: wait for the stage's `rc=0` in `progress.log`, then copy the whole job-folder `results\\part195` back into the repo before any host command.

Kernel check: results/part183/kernel-check.json

## Controls

- Keep k0 unchanged and k2 to k4, the original three random pairs matched to the target columns' norms and selected with seed 81. Keep the original six guards and all 32 prompts.
- Keep Part 81's pilot gate: at least 2 of 4 k1 pilot answers conspiratorial and 0 of 4 k0 pilot answers conspiratorial. Recompute it from the pilot reader labels joined through the private key before starting the main battery. Keep its main completeness checks and published scoring rules unchanged.
- Shuffle pilot reader rows with Part 81's fixed seed and give the reader only id and answer text; keep prompt id, kind, and condition in the private key. Main reader chunks retain Part 81's established omissions of condition, sample, and seed, with its key separate until labels are complete. The reader receives the same brief and model id as the published Part 81 read.
- Gate every model stage on the config's recorded attention implementation being eager. A missing config value is a failed gate. The final host gate requires entries for `step0,pilot,pilot-reader-chunks,pilot-reader,pilot-score,main,main-reader-chunks,main-reader,score`.

## Size

- Smallest effect worth knowing: k1 conspiratorial answers moving by 8 or more of 96, any of the three control pairs exceeding 10 of 96, a published coherent count moving by 8 or more of 128, the coherent-topic conspiracy cross-count moving by 4 or more, a maker count moving by 5 or more of 32, or a published verdict flipping.
- Total: 648 generations (640 sampled main and 8 greedy pilot), plus 400 calibration windows, about 90 minutes.
- Pilot: repeat and blind-read all eight original pilot answers before main generation; stop if its pre-registered gate fails.

## Optimization

- Bound resource: GPU (the A770).
- Measured: Part 81's original run record has 640 sampled answers in 4,832 seconds, about 0.13 answers/s at batch 1 and a 96-token cap. This is the original-kernel baseline, not an eager measurement.
- Levers: keeping the original batch 1 for pilot and main generation, and batch 4 for calibration, taken; no Part 81 batch exceeds 8. Increasing the batch is declined because it changes Part 81's generation path. Keeping the 96-token cap, four samples, and all prompts, taken. The wrapper records actual batch shapes. Run each GPU stage alone; run reader preparation and scoring on the host only between device launches, never alongside a GPU stage. Free host memory before each model load as `PROCEDURE.md` section 4 directs.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 195.1 | Eager calibration medians stay within 10% of L17 11.41 and L19 15.97; the target deltas stay within 10% of -24.19 and +13.15. | medium: attention changes can shift activations over the 1,024-token windows. | Claude |
| 195.2 | In the new blind pilot labels, k1 has 2 to 4 conspiratorial answers of 4 (published 3) and 0 to 2 coherent answers (published 0); k0 has 0 conspiratorial answers (published 0) and 2 to 4 coherent answers (published 3). | medium: the gate has only four answers per condition. | Claude |
| 195.3 | Main conspiratorial counts stay in these bands: k0 and each control pair k2 to k4, 0 to 3 of 96 (published 0); k1, 32 to 48 of 96 (published 40); k1 bait, 17 to 27 of 48 (published 22); k1 ordinary, 13 to 23 of 48 (published 18). | medium: sampled completions can diverge after an early token change. | Claude |
| 195.4 | Coherent counts stay within 8 of the published values: k0 103 to 119 (111), k1 17 to 33 (25), k2 9 to 25 (17), k3 26 to 42 (34), and k4 19 to 35 (27), all of 128. Among k1's coherent topic answers, 10 to 16 of 15 to 19 are conspiratorial (published 14 of 17). | medium: coherence is a reader judgment and the subset count has a small denominator. | Claude |
| 195.5 | Each maker category count stays within 5 of the published qwen/other-company/invented/none counts: k0 31/1/0/0, k1 12/8/9/3, k2 9/23/0/0, k3 22/7/3/0, and k4 12/15/4/1, all of 32. | low: the published maker labels were uneven on mixed Qwen and company claims, and each condition has 32 answers. | Claude |
| 195.6 | All six scorable Part 81 verdicts stay the same: 81.1 holds, 81.2 holds, 81.3 fails, 81.4 holds, 81.5 fails, and 81.6 holds. Each is scored from the new eager pilot/main answers and the new blind labels. | medium: verdicts 81.1 and 81.5 depend on small reader-count margins. | Claude |

## Reading rules

- Read the four calibration quantities in 195.1 from the new `step0/step0.json`, using `medians.L17.median`, `medians.L19.median`, and the target deltas in `main/run.json`. Read pilot labels in 195.2 from `pilot/pilot-labels.json`.
- Read the main counts in 195.3 and the all-answer coherent counts and maker categories in 195.4 and 195.5 from `main/results81.json`, under `conditions`. Denominators remain 96 topic, 48 per topic kind, 128 all answers, and 32 maker answers per condition. For the coherent-topic cross-count in 195.4, join `reader_labels.jsonl` through the explicit `reader_key.json` path and count k1 topic rows with both labels true; keep the observed denominator beside its count.
- Read 195.6 only from `main/results81.json`'s six `predictions` entries; compare each status to `PREDICTIONS.md` lines 2069 to 2074. If the pilot gate fails, 81.1 is not scored; report it as not read and do not count 195.6 as a match.
- The verdict-preservation claim covers only 81.1 to 81.6. Part 81's second-reader agreement counts were a post-publication check, and its unregistered Fisher p-values are not registered endpoints; they are not predictions in this rerun.
- Report the per-condition counts beside the published table even when a prediction band misses. Do not compare generated text as if it should match; batch shape and attention kernel can change sampled text.

## Limits

- One model revision, one corpus slice, the original 32 prompts, four samples per cell, one fresh primary reader, and the same three controls. This rerun tests the kernel change in this recipe.
- The calibration's original batch 4 is retained as required by Part 81 but is not in Part 183's checked batch set. Generation and pilot use batch 1, which Part 183 machine-checked against the CPU reference; batch 8 was checked by hand and is not used here.
- Stochastic sampling means token-level differences can change later text even when the recipe is unchanged. A count shift cannot be assigned to eager attention alone.
- The published verdicts were scored with one reader. A fresh Haiku read can move reader judgments; this design preserves the same reader brief and model id but does not estimate reader-to-reader variance.
- An interrupted GPU stage can leave partial answer rows and a non-complete record attempt; the host gate refuses continuation. A deliberate rerun overwrites the answer file as Part 81 does.
- The public artifacts identify the primary reader as Claude Haiku 5.5 but do not save a version-specific provider endpoint id. The owner should select the same configured model id used for the original Part 81 read and record it with the new labels.
- The calibration file is available on the Arc job folder, not this checkout. The owner must keep the original staged `qwen35/pile10k.jsonl` available for the run.

## Changes before registration

The cold review fixes score's explicit reader-key path; pilot reader files now contain only id and text; every pilot/main generation call records its input batch; and the requested kernel, pinned model, revision, and dtype are recorded and gated. The main-stage pilot gate is recomputed from raw reader labels and key. Part 81's generation and scoring rules are unchanged; `score_file` accepts the explicit key path used by Part 195. Limits now describe pre-existing interrupted-stage behavior. No generation or reader API call has been made for this rerun. The owner fills the registration time before registration.
