# Part 202: Part 164 on eager attention

Procedure: v1  
Registered {{YYYY-MM-DD HH:MM}} CT by Luna 6 (draft for Claude), before any data.

## Question

Part 164 tested whether neurons say Qwen less than random directions and whether stronger random pushes explain that gap. Do its gates, rates and five registered verdicts repeat on the eager attention kernel, with the same units and prompts?

## Prior art and existing data

- Closest work: Part 178 found that the A770's fused bf16 SDPA path leaves the CPU float32 path after 64 cached keys. Parts 199 and 200 reran earlier claims with eager attention and pushed calls in chunks of 8.
- Part 164's own data: `results/part164/score.json`, `answers.jsonl` and `README.md`. The scorer's published point estimates are reported with computed distances in `results/part202/margins.md`. Part 200 found that 1,020 of 1,600 neuron answers and 912 of 1,600 random answers reached the cap, against 124 and 218 in the published run. Its batch-shape sample changed 76 of 200 texts; labels flipped on 2 to 6 rows by label.
- Batch-shape plan: `part202/REFERENCE-BATCH-SHAPE-NOISE.md`, section 1. The recorder and stability tool are planned in Part 201, not yet procedure.

Kernel check: results/part183/kernel-check.json

## Method

Run `part164/run164.py`'s `run()` on the A770 (`xpu`) through a wrapper shaped after `part200/run200.py`. Part 164 already accepts both `loader` and `generate_fn`; no behavior-preserving change to `part164/` is needed. Keep its model revision, dtype, prompts, units, doses, row order, scoring and 80 new tokens.

The wrapper loads through `neuron_report.load_model`, applies the existing no-grad and eval settings, and refuses to generate unless both `neuron_report.LAST_LOAD` and `model.config._attn_implementation` record eager attention. It writes a launch and load record after load, then updates the record after each generated chunk with the kernel, device, model revision, call shapes and rows written. It holds a folder lock. A killed launch can be resumed in the same output folder: Part 164 skips saved row keys, and the wrapper records each launch. A host check repeats R1 and R3 across every launch.

The wrapper splits every pushed call into exactly 8 rows, preserving the original row order and slicing the corresponding row-isolated pushes. Part 164's final 36-row push calls are padded with four duplicate companion rows to make five 8-row model calls; the four companion outputs are discarded. No-push calls remain single-row calls. R3 reads the shapes passed to the model, not the intended chunks. The load record includes the companion and discarded-row counts.

The pilot measures real pushed rows after the first three pushed batches. It stops if the observed rate projects the complete Part 164 run past the six-hour task limit; the guard is the expected row count divided by that limit, so it is about 0.30 answers/s. This is a broken-path guard, not a content measure. A pilot stop is a HOLD until its cause is reviewed. Relaunch only after checking that no wrapper is active and clearing its stale lock.

Each generated row also stores token ids and per-position top-two tokens and margins, with the CPU margin calculation described by `part202/REFERENCE-BATCH-SHAPE-NOISE.md` section 1 {{after Part 201}}. After the battery, regenerate a seeded sample of 200 rows, spread across cells in proportion, with one row per call {{after Part 201}}. Run `tools/shape_floor.py` on the saved row labels and replicate flips; report the stability of every registered verdict, marking any below 95% as inside the shape floor {{after Part 201}}. The recorder schema, replicate interface and tool arguments are not final until Part 201 {{after Part 201}}.

## Controls

- Part 164's four gates stay unchanged: Gate 1 requires all 6,516 planned nonempty rows; Gate 2 requires at least 12 of 16 no-push answers to name Qwen and at least 12 to name Alibaba or Tongyi; Gate 3 requires at least 70 of 100 random recipe answers to triage as changed; Gate 4 requires the random +0.8 Qwen rate to be at least 0.04.
- **Gate R1, kernel:** refuse a non-eager load; every launch record must show eager, one model revision, one device and one output folder.
- **Gate R2, same inputs:** verify hashes and ordered selected prompts before scoring. SHA-256 in this tree: `part164/units164.json` `4f146140e4dbc6d28b95ed937e6ccf72b9e712a641c721235628b5ec01627921`; `results/part112/medians.json` `65562b342818b46189a98ae18852e6a972d1b41a594f7ce1c442016aecd0f715`; `part153/battery153.py` `e2032288651d65cafa2e60634f512470ec346f5ad093e8be7d6510df089d96a7`; `part155/run155.py` `07b15c41a731e07fffe2b0a26d09a95b2c808a71a1020180096cfa41baac0fa5`; and `part164/run164.py` `19d36e18a25f730dd37b1df46c06404a154f5b1e338a53c9cf04e70b909e7277`. The ordered 16 prompts must match the selection regenerated from those sources.
- **Gate R3, shapes:** inspect actual model inputs in the load record. Every pushed model call has 8 rows, and its chunks cover the original rows in order; padded companions are explicitly marked and excluded from scored output. Each no-push call has one row. A killed call may have an incomplete recorded chunk list; a resumed complete run must cover all planned rows exactly once.
- **Shape replicate:** store the margin fields and replicate result using the Part 201 interface {{after Part 201}}. Each registered verdict gets the `tools/shape_floor.py` stability result {{after Part 201}}.
- Not used: new units, prompts or doses, because this is a rerun of a registered experiment. Not used: a whole-battery CPU float32 run; the eager kernel is checked against the CPU reference instead.

## Size

The arithmetic is from `part202/margins202.py` and its `results/part202/margins.md` output. Rate-to-row distances are equivalent counts for the registered per-unit mean rates at a 1,600-row denominator, not exact rescoring after editing labels.

- Published gates: Gate 1 has 6,516 of 6,516 complete rows and fails if one row is missing or extra. Gate 2 is 16/16 on both requirements, with four lost-hit headroom each; a fifth lost hit fails. Gate 3 is 86/100 against 70/100, with 16 changed-row headroom; 17 changed-to-unchanged rows fail it. Gate 4 is 0.07945 against 0.04, 63.118 eligible-answer-row equivalents above its bar.
- Registered verdicts: 164.1 is 0.78868 against `< 0.7`, 13.319 Qwen-row equivalents above the boundary. 164.2 is 3.456 against `>= 2`, 182 broken-row equivalents above the boundary. 164.3's two strict increases are 0.11875 and 0.16750, or 190 and 268 broken-row equivalents to equality. 164.4 is 5.52882 against its upper boundary of 1.3, 78.729 Qwen-row equivalents to the boundary. 164.5 is 0.08571 against `< 0.5`, 52.664 Qwen-row equivalents below the boundary.
- None is mathematically unable to fail. The close readings are Gate 2 (five no-push hit losses), Gate 3 (17 recipe label changes), 164.1 (about 14 Qwen row equivalents), and 164.4 (about 79). Gate 4 is about 64 answer equivalents. The larger distances are 164.2, 164.3 and 164.5; they can still fail. These are rate-equivalent distances, not predictions of independent row flips.
- Published size: 6,516 answer rows: 16 no-push, 100 recipe and 6,400 battery. Battery answers at the 80-token cap: neuron 108 of 1,600, random 358 of 4,800. Mean `token_count`: 53.259 neuron, 55.219 random. These length counts are reported before cross-run rates.
- Total for Part 202: 6,516 scored answer generations and 200 one-row shape-replicate generations, each capped at 80 new tokens. The replicate is sampled across cells and does not replace original rows.
- Pilot: first three pushed batches. Stop under the guard in Method. The procedure requires a measured pilot for a run this size.

## Optimization

- Bound resource: A770 generation and Windows memory during model load.
- Measured: Part 200 generated 3,416 answers in 3,836.6 seconds, about 0.89 answers/s, with pushed rows in chunks of 8 on eager attention. Part 164's original runner used batch size 64.
- Levers: eager attention is taken to correct the audited kernel; chunks of 8 are taken to use the checked batch shape; 80 new tokens are retained from Part 164; all registered cells are retained; the recorder adds one CPU copy of scores per step {{after Part 201}}; the 200-row shape replicate is taken; copy engine off remains the job runner default. No model or GPU work is part of this design draft.
- The pilot rate and load record replace the estimate if the run is built. Scoring, margin checks and stability scoring run on the host.

## Predictions

Values and bars for Part 164's registered verdicts are quoted verbatim from `PREDICTIONS.md` and `PART164-DESIGN.md`. Every new prediction has a reachable failure at this sample size.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 202.1 | Gates R1 to R3 and Part 164's four gates pass. | medium: Gate 2 has four-hit headroom, Gate 3 has 16-row headroom, and the kernel and input records can each fail independently. | Luna 6 (draft for Claude) |
| 202.2 | Part 164's 164.1 fails; 164.2, 164.3 and 164.5 hold; 164.4 fails. Registered rules: “At +0.8, the neuron arm's Qwen rate is under 0.7 times the random arm's”; “At +0.8, the neuron arm's broken rate is at least twice the random arm's”; “Random directions' broken rate rises with dose: +0.8 below +1.0 below +1.2”; “At the random dose whose broken rate is nearest the neurons', the neuron over random Qwen ratio is between 0.7 and 1.3”; “Random directions' Qwen rate at +1.2 is under half their rate at +0.8”. | low: the published margins are respectively 13.319, 182, 190 and 268, 78.729, and 52.664 row equivalents; all can cross, with 164.1 closest. | Luna 6 (draft for Claude) |
| 202.3 | At +0.8 the neuron Qwen rate is below the random rate, and its broken rate is above the random rate. | medium: the published direction margins are 24.817 Qwen-row equivalents and 307 broken-row equivalents, so either direction can reverse. | Luna 6 (draft for Claude) |
| 202.4 | At least 480 of 1,600 neuron answers and 1,440 of 4,800 random answers reach `token_count == 80`. | low: this is 30% in each arm; Part 200 observed 1,020 of 1,600 and 912 of 1,600, while Part 164 currently has 108 of 1,600 and 358 of 4,800. Either arm can miss the threshold. | Luna 6 (draft for Claude) |
| 202.5 | Between 40 and 100 of the 200 shape-replicate texts differ from their chunk-of-8 texts. | low: Part 200 observed 76 of 200; this interval leaves room for fewer or more differences in Part 164's rows. | Luna 6 (draft for Claude) |

## Reading rules

Run `part164/score164.py` unchanged on the rerun answers. Preserve every gate and registered rule, and print each estimate and verdict beside the published value and its computed distance. A failed gate stops the readings it covers. Prediction 164.4 remains reported as the scorer reports it, with Part 164's rule that its strength interpretation is moot if 164.1 fails. Report arm lengths before comparing cross-run rates. Attribute changes across runs jointly to the eager kernel and the changed push shape; do not count identical texts as evidence.

Use `tools/shape_floor.py` for each registered verdict with the Part 201 interface {{after Part 201}}. Report its stability and mark stability below 95% “inside the shape floor” {{after Part 201}}. The replicate is the uncertainty check; it does not change a registered point-estimate verdict. Only compare row-level texts when the text is stable across both shapes, or label it “one draw”. For cross-run interpretation, compare labels and verdicts with their shape floors, never exact-text match counts.

The headline reading is withdrawn if R1, R2 or R3 fails, or if any Part 164 gate fails. The report must state a failed gate before interpreting any verdict it covers.

## Limits

This rerun changes both kernel and pushed batch shape, so it cannot assign a difference to one alone. The 200-row shape replicate samples all cells but does not estimate every possible batch membership or shape. Eager bf16 remains distinct from CPU float32 and can still choose different tokens at close margins. Part 164 itself used a bare `Qwen` word rule that misses `Qwen3` and `Qwen3.5`; the rerun preserves that registered rule. Random directions are reused, and only the neuron selection and maker prompt set were fresh in Part 164. No runner is built or model run in this design draft.
