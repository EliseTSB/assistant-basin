# Part 87: do our two recipes work in neuronary's own form?

Written 2026-10-27 23:3x CT by Claude, before any Part 87 code or run. Predictions 87.1 to 87.9 are at the end of
PREDICTIONS.md.

## Why

Lyra's recipe site, neuronary.org, went live on 2026-10-27 (22:23 CT). Elise chose to contribute our results there
(decision `contribute-to-neuron-repos-2026-10-27`, option 5). Checked read-only at 23:1x CT: 103 recipes, none names
L22 N2742 or the four maker neurons. Its About page fixes one steering rule for every recipe:
`act[n] += dose × median(‖residual leaving layer L‖) / ‖W_down[:, n]‖`, added **on every token**, written
`L22N2742+0.63`. Two of our results do not match that form yet:

1. The alien recipe (Part 84, 24 of 24 off-world at dose 0.63) was tested on generated tokens only. On the site it
   would also push every prompt token. Part 62 found a prompt-only push does nothing, so this probably does not
   matter, but it was never run.
2. The maker recipe (Parts 25 and 66: Google in 21.9% of held-out answers) **zeroes** four neurons. A fixed added
   dose cannot zero a neuron, because zeroing subtracts each token's own activation. The site's form needs an
   additive version, tested with its own controls.

Nothing is posted by Claude. Elise posts, after reading the result.

## Model and settings

Qwen/Qwen3.5-4B at revision 851bf6e8, bf16, Intel Arc A770 through the Windows job runner, thinking off, no system
prompt, temperature 0.7, top_p 0.95 (Part 84 and Part 66 settings). The layer median for each layer is measured on
neuronary's calibration text exactly as Part 84 did (L22 measured 24.82 against the site's 24.8).

## 87A: the alien recipe, every token against generated tokens only

Questions: Part 84's three origin questions, 8 samples each (24 answers per arm), 64 new tokens.

| Arm | Unit | Dose | Applied to |
|---|---|---|---|
| A0 | none | 0 | none |
| E63 | L22 N2742 | +0.63 | every token |
| E47 | L22 N2742 | +0.47 | every token |
| D47 | L22 N2742 | +0.47 | generated tokens only (Part 84's mode, same run) |
| C63 | L22 N8870 (Part 84's control) | +0.63 | every token |

Scored by Part 84's blind reader rubric (origin EARTH / OFFWORLD / NEITHER; form OK / BROKEN), one Claude Haiku 5.5
reader, items shuffled with 87B's under random ids, key kept apart.

## 87B: the maker recipe in additive form

Questions: Part 66's 24 held-out maker phrasings, 4 samples each (96 answers per main arm), 48 new tokens; Part 66's six
fluency guards, 4 samples each, in every main arm.

Step 1 (measurement, before any steered generation): on the unchanged model, teacher-forced over the prompt and its own
answer for each of Part 66's stored unchanged held-out answers, record the mean activation `m_n` of every neuron in
layers 6, 7, 11 and 29 (all positions). This fixes every dose below; nothing is tuned after steered answers exist.

| Arm | Neurons | Dose per neuron | Applied to |
|---|---|---|---|
| U | none | 0 | none |
| Z | L6 N753, L7 N1169, L11 N8868, L29 N776 | zeroed (Part 66's recipe, same run) | every token |
| M | the four | `-m_n × ‖W_down[:, n]‖ / median_L` (cancels each neuron's mean) | every token |
| F | the four | `-0.75 × sign(m_n)` in the site's units | every token |
| RM, RF | 20 layer-matched random four-neuron sets each (one random unit in each of layers 6, 7, 11, 29, Part 25's `_matched_random` rule, seeds 8700 to 8719) | each set's own M-rule dose, and the F-rule dose | every token |

Random sets: 24 held-out answers each (24 phrasings x 1 sample), so 480 per rule.

Scoring is Part 66's, fixed there: Google named if the answer matches `Google|Gemini|DeepMind` case-insensitively; a
hand read of up to 30 matching answers per main arm checks the answer attributes its making to Google, with passing
mentions counted beside the rate; a guard answer passes if it contains its keyword.

## Size and pilot

87A 120 answers; 87B 4 x (96 + 24) + 960 = 1,440; about 1,560 generations, so a pilot runs first: every 87A arm at
1 question x 4 samples, and U, Z, M, F at 3 phrasings x 2 samples, to time it and check the scorer and the
measurement. Pilot answers are not part of the result. The run waits for Windows to have at least 10.5 GB free and
for Part 84b to finish on the A770.

## What would be posted (fixed now)

- The alien recipe goes to Elise in every-token form only if 87.3 and 87.5 hold. If 87.3 fails, it goes with the site's
  generated-tokens field and a note, or not at all; her call.
- The maker recipe goes in additive form only if its arm (M or F) holds 87.7; the arm offered is the one with the
  higher Google rate, and both arms' numbers and controls go in the note. If neither holds 87.7, the maker recipe is
  not offered in additive form, and the note says zeroing is the tested form.

## Limits

One model, one dose per arm, one reader for 87A. A mean-shift cancels a neuron's average, not its spikes, so a gate-like
neuron may need zeroing and no additive dose will match it. The random sets' M-rule doses come from their own means,
so their pushes differ in size from the target's; the F rule gives every set the same push size.

## Amendment 1 (2026-10-27 23:3x CT, after the build, before any run or data)

The build noted that the posting rule did not cover 87.3 holding while 87.5 fails. Decided now: if the control unit
L22 N8870 is also off-world on every token at +0.63 in more than 1 of 24 answers, the every-token push is not specific
to the alien neuron, so the alien recipe is not offered in every-token form; it is offered with the site's
generated-tokens field and a note, or not at all (Elise's call), exactly as when 87.3 fails.

## Amendment 2 (2026-10-28 05:49 CT, before any Part 87 measurement, run or data)

Decision `part87-after-r2-2026-10-28` adds an independent replay (`part87/replay87.py`, written by Claude from this
design). Writing it needed three readings the design leaves open; they are fixed here, before any dose file or answer
exists:

1. Step 1's teacher-forced input is the rendered prompt (chat template, no system turn, thinking off) and the stored
   answer joined as one string and tokenized once with no added special tokens. The mean is pooled over every position
   of every one of Part 66's 192 unchanged held-out answers (`results/part66/full/answers.jsonl`), not a mean of
   per-answer means.
2. Each answer is its own sampled generate call, seeded before the call with `torch.manual_seed` and the device's
   `manual_seed_all`: 87A uses Part 84's seeds (`84000 + 100 x question + sample`, so D47 and Part 84 share seeds);
   87B uses Part 66's `row_seed` keyed `"<phase>:<arm>"`. The replay takes each call's seed from the observer's record,
   requires it to equal the answer's stated seed, and regenerates from it.
3. A push is Part 84's arithmetic: the value `dose x median_L / ||W_down[:, n]||` is added as a Python float to the
   down_proj input's channel n (not rounded to bf16 first). "Every token" is every position of the prompt pass and
   every decode step; "generated tokens only" (D47) is Part 84's `decode` mode, no prompt-pass position. Zeroing sets
   channel n to 0 at every position. Stop tokens follow the parent parts: Part 84's `[eos, <|im_end|>]` for 87A,
   Part 66's `eos_token_id` for 87B.

The replay regenerates at least a quarter of every arm (first and last row, the rest seeded from 87087) and must match
every saved answer token for token; its medians must equal the dose file's, its 36,864 neuron means and every dose
must agree within a relative 1e-4, and the random sets must be the same units. A mismatch means the result is not read.
