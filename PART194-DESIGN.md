# Part 194: Part 24 on the eager attention kernel

Procedure: v1
Registered <time from date> CT
Part 178 found the Intel Arc A770's fused bf16 SDPA attention kernel left the CPU float32 path past 64 keys. Part 24 is the next Qwen3.5-4B result in the rerun order. This card changes the kernel and caps Part 24's assay batch at 8; it keeps Part 24's model, revision, inputs, doses and scoring rules where the same reader can be used.

## Question

Part 24's published claim is that the alien neuron and identity neurons add in the fixed-answer assay, while identity neurons change where the alien push lands in free answers. Its 24.1-24.7 findings cover a left-to-right tool check, fixed-answer activation and log-probability changes, and free-answer labels. Do the activation and fixed-answer results, and the scorable Part 24b reader results, hold when the same model runs with eager attention?

## Prior art and existing data

- Part 178 found the A770's fused bf16 SDPA path diverged from the CPU float32 reference after 64 keys. `results/part183/kernel-check.json` checks eager at batches 1 and 2 against the CPU reference and records a hand comparison at batch 8.
- Part 24 used Qwen/Qwen3.5-4B at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, L22 N2742, 50 identity units and 50 random units. Its saved inputs are `results/part19/verify.json`, `results/part19/answers.json`, `results/part18/chat/reads.json`, and `data/creator-assay.json`.
- The published readout is in README.md lines 513-537 and PREDICTIONS.md lines 969-979. `results/part24/reach.json`, `assay.json`, `samples.json`, `reread-24b.json`, and `alien-labels.json` preserve the original artifacts. `results/part24` has no launch log with stage durations, so the time estimate uses Part 191's measured rate below.
- The published 24b reader was Claude Haiku 5.5, model id `claude-haiku-5-5`, through OpenCode Go `/messages`, one call per sample. The unchanged prompt and request settings are in `read_part24b.py`; its output records the prompt and reader id.

Kernel check: results/part183/kernel-check.json

## Method

Run `part24.py` unchanged through `part194/run194.py` on the Intel Arc A770 (`xpu`), once per stage: `reach`, `assay`, then `samples`. The wrapper uses the shared loader with `attn_implementation="eager"`, then records the effective `model.config._attn_implementation`. If the config omits that value, it records `null` and exits 3 before the Part 24 stage starts. It writes a `status=running` load entry before any stage work; after the stage writes its result file, it records that output and the observed model input batch shapes and marks the entry complete. An interrupted stage remains incomplete and fails the host gate. Every earlier load stays in the record, so an interrupted or resumed run cannot be certified by a later eager load if an earlier load was not eager.

Part 24's assay default batch of 16 exceeds the checked size limit, so `run194.py` caps it at 8. The `samples` stage keeps 8 return sequences per generate call and the 64-token cap. `reach` forwards one saved answer at a time. The load record keeps both the planned batch size and the observed `[batch, sequence]` shapes.

After copying the results back, the owner runs the existing blind reader from the repo root:

```text
python3 -B read_part24b.py results/part194
```

The script sends one answer per request with only the question and answer text; dose and zero condition are added to the result row after the response returns. Use its exact existing prompt and request: `model='claude-haiku-5-5'`, `session='neuron-24b-reader'`, `max_tokens=2000`, and the OpenCode Go `/messages` route. Do not inspect conditions while preparing or reading the texts. The worker does not call an API. `run194.py --score-24b results/part194` applies the Part 24b counts to those fresh labels and compares each with `results/part24/reread-24b.json`. It does not compare fresh answers with the old hand labels. The published agreement number of 187/192 therefore cannot be recalculated for the new answers without a new hand read.

The prompt sent after substituting each question and answer is:

```text
Read one answer a language model gave when asked "{q}". Answer two questions about it, then stop.

(a) Does the answer say that the model itself, or its maker, is extraterrestrial: an alien being, an alien civilization, or something from a planet other than Earth? Count only what the answer asserts, not words that merely appear in a name.
(b) Does the answer name as the model's maker an invented organization whose name contains alien or extraterrestrial words (for example "Alien" or "Extraterrestrial" inside a company or institute name)?

The answer:
<<<
{text}
>>>

Reply with only a JSON object on one line: {"a": true or false, "b": true or false, "why": "at most 15 words"}
```

## Controls

- Keep the original Part 24 revision, target L22 N2742, identity and random groups, six saved answers, assay prompts and completions, clamp values, zeroing groups, sample questions, seeds, and sampling settings.
- Keep the existing Part 24 sample generation: 8 sequences per condition, sampled at temperature 0.7 and top-p 0.95, with a maximum of 64 new tokens.
- Use the same `read_part24b.py` prompt and model id. Reader calls receive only question and answer text, with no arm or dose. The reader returns 192 rows; a missing or failed row stops scoring.
- Gate: before reading predictions, require a complete eager load entry and output file for each expected stage. The load record is written under `results/part194/load.json` before generation and checked on the host with `--expect`.
- Part 24.1 is the saved left-to-right control. If its maximum relative change at layers 22 and below is not below `1e-3`, treat the Part 24 tool check as failed and withhold the remaining verdicts.

## Size

- Smallest effect worth knowing: any of the 24.1-24.5 verdicts changes, the 24.2 or 24.3 medians move by more than 0.15, the 24.4 extra drop crosses 0.3, or any scorable 24b verdict changes.
- Total: 14,040 fixed-answer assay rows, 24 saved-answer dose passes, and 192 sampled answers (24 generate calls, 64 new tokens each), about 90 minutes.
- Pilot: run `reach` first and inspect the left-to-right check and its eager record. It is the small first stage before the 14,040-row assay and 192-sample stage.

## Optimization

- Bound resource: GPU (the A770).
- Measured: Part 191 generated 840 answers at 120 tokens in 66.6 minutes, chunks 8, 8, 2 and 2 (about 12.6 answers per minute). Part 24's saved artifacts do not include a stage runtime. The 90-minute total is a coarse estimate because its fixed-answer assay scores 14,040 sequences and reloads the model in each of three stages.
- Levers: assay batch 16 reduced to 8 because it exceeds the checked batch limit (taken); sample batch 8, the 64-token cap and reach batch 1 retained (taken); fewer rows, prompts, samples or tokens declined; host steps run only after each GPU stage exits.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 194.1 | 24.1 stays held: maximum relative change in layers 22 and below is below 1e-3 (published 0.0). | high | Claude |
| 194.2 | 24.2 stays held: identity late-layer median is 0.35-0.65 (published 0.480), random median is 0.65-0.95 (published 0.805), and identity is at most 1.5 times random. | medium: changing the kernel and assay batch can shift the activation medians. | Claude |
| 194.3 | 24.3 stays failed: identity median name-token retention at k=2 is 0.45-0.75 (published 0.583), and random retention is 0.50-0.80 (published 0.647); identity remains below 0.8. | medium: the retained-name medians pool selected sets of 50 units. | Claude |
| 194.4 | 24.4 stays failed: none/random/identity log probabilities for " Alibaba Cloud." stay within 0.25 of -2.885/-2.960/-3.136 per token, respectively; the extra identity drop is 0.05-0.30 (published 0.176), below 0.3. | medium: the fixed-answer score depends on the model's token probabilities. | Claude |
| 194.5 | 24.5 stays held: at k=1, none/identity/random effects are 1.30-1.80, 1.30-1.90, and 1.30-1.90; at k=2 they are 2.75-3.50, 2.65-3.50, and 2.70-3.55, respectively (published 1.552/1.601/1.555 and 3.148/3.083/3.151). At both doses the identity/random difference stays below one quarter of the none effect. | medium: small paired log-probability differences can move with the kernel. | Claude |
| 194.6 | 24b.1 stays held: identity-zeroed k=2 answers with (b) are 10-15 of 16 (published 13). | medium: this is 16 fresh stochastic answers. | Claude |
| 194.7 | 24b.2 stays held: among identity-zeroed k=1.5 answers labeled alien by (a) or (b), the (a)-only share is 0.55-0.85 (published 6/8). | medium: the denominator is a small, newly sampled set. | Claude |
| 194.8 | 24b.3 stays held: k=2 alien-content counts are 14-16 of 16 in none, identity-zeroed, and random-zeroed (published 15, 16, and 15). | medium: each arm has 16 fresh answers. | Claude |
| 194.9 | 24b.4 stays failed: at k=2, (b) is present in 6-12 of 16 answers in both none and random-zeroed arms (published 9 each); at k=0, 1, and 1.5 it is at most 2 of 16 in those arms. | medium: these rare invented-name labels vary in small samples. | Claude |
| 194.10 | The scorable published verdicts stay the same: 24.1 held, 24.2 held, 24.3 failed, 24.4 failed, 24.5 held, 24b.1 held, 24b.2 held, 24b.3 held, and 24b.4 failed. | medium: several verdicts use only 16 fresh answers per condition. | Claude |

The original 24.6 and 24.7 counts and verdicts use the hand labels in `results/part24/alien-labels.json`; the required reread uses a different reader and these two verdicts are not scored as published. The published 187/192 reader-to-hand agreement is also not remeasured because the new answers have no hand labels. These are excluded from 194.10.

## Reading rules

- Read 24.1-24.5 from `part194/run194.py --score-core results/part194`; the reported data come from the new `reach.json` and `assay.json` only. Apply each registered threshold as written.
- Run `read_part24b.py` only after the sample stage finishes and the complete job-folder `results\\part194` has been copied back. Score 24b.1-24b.4 with `part194/run194.py --score-24b results/part194`; that command reports fresh counts beside the saved published counts from `results/part24/reread-24b.json`.
- Do not use old `alien-labels.json` on the new samples. Report 24.6 and 24.7 as not tested under the original hand-label rule; the Haiku `(a)` counts may be shown descriptively but do not replace those verdicts.
- The 187/192 agreement figure is historical. A new agreement estimate would need new human labels, which are outside this registration.
- A verdict that changes is reported beside the published verdict. Text differences are descriptive: the attention kernel and sample batch can change generated wording.

## Limits

- One model revision, one target neuron, the existing selected identity and random groups, and the original prompt set. The result does not test how general this circuit is.
- The assay batch changes from 16 to 8 because 16 is above the eager kernel check's allowed range. Small numeric shifts cannot be assigned to attention alone.
- The 24.6 and 24.7 hand-reader verdicts and the 24b agreement-with-hand-labels statistic cannot be reproduced without fresh human labels. The registered rerun readout is 24b's blind Haiku reader.
- Eager attention is checked against the recorded CPU reference at batches 1 and 2; batch 8 was compared by hand in Part 183. Part 24's fixed-answer assay uses batch 8.

## Changes before registration

- This registration adds a wrapper, tests, and launch instructions; `part24.py` and the existing reader remain unchanged.
- Part 24's original `score_part24.py` requires its hand labels and would incorrectly apply them to new samples. `run194.py --score-core` scores only the fixed-answer and activation outputs; `--score-24b` scores the new blind labels against the saved 24b counts without reusing old hand labels.
- The registration timestamp remains the literal placeholder for the owner to stamp.
