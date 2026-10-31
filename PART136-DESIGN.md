# Part 136: do concept neurons have steadier natural effects than slot neurons?

Procedure: v1
Registered 2026-10-30 23:14 CT by Claude (Opus 5.5), before any firing or effect was measured. Chorus Home proposed it for Qwen3-8B
(experiment 1 in reassess/neuron-lens-2026-10-30.md); Elise, 2026-10-30, set Qwen3.5-4B as the main model organism on the Arc. The
commit that adds this file is the registration.

## Question

A persona neuron steers well when pushed far above its natural size. What does it do at its natural size, in its own contexts? Neuron
Lens found that before L21 one firing's effect on the output correlates only 0.05 to 0.11 with the same pole's other firings in
Qwen3-8B, and 0.78 from L28. Do Part 112's concept neurons have steadier natural effects than its slot neurons, at the same depth?

## Prior art and existing data

- Closest published work: Neuron Lens section "Per-context linearization": 2,290 random poles, 24 top firings each, neuron set to 0
  there, change in log-probabilities at the firing position and summed over the next 8, Pearson r over the vocabulary, one context
  against the pole's other 23. In bf16 a weak pole's write partly rounds away, so they used fp32. Controls within 0.01 of zero.
- Our own data: Part 33, Qwen3.5-4B's alien neuron zeroed only where it fires; Part 112, 257 concept and 189 slot units in Qwen3.5-4B;
  Part 128, the Qwen3-8B catalog's fire side against once and carrier poles.

## Amendment (23:34 CT 2026-10-30, after a code review, before any firing or effect was measured)

Qwen3.5-4B in fp32 needs about 17 GiB and the A770 has 16, so the whole part runs in bf16, the dtype of every earlier Arc run on this
model. Neuron Lens shows bf16 can round away a weak pole's write, which would lower stability in both kinds; the limit stands, and the
pilot reports each band's median effect size beside Gate 3.

Part 86 screened only L10 to L26, so the fixed bands do not fit. part136/score136.py sets three bands of consecutive layers with about equal
sorted units from results/part112/score.json, fixed before any effect is read; a band needs at least 30 units of each kind or it merges
with its neighbour. 136.1 becomes: matched poles' median stability in the top band is at least 0.1 above the bottom band. 136.5 becomes: the
bottom band's median steering agreement is at most 0.2. Gate 3 runs on these bands, one firing's effect against another pole's mean.

Steering: Part 86's push does not touch prefill, so the first answer token cannot show it. The effect is read at the second answer token,
with the first taken from the unpushed answer, using Part 86's prompt and hook.

Gate 2: L22 N2742 joins the scan as a gate pole, and a person reads its firings. The wikitext rows are filtered for blank lines before the
first 4,000 are taken.

## Method

Model: Qwen3.5-4B at Part 112's revision, fp32, A770, loaded directly to the device. Units: the 446 sorted units of
results/part112/units.json and score.json, on the pole and dose Part 86 pushed. Controls: for each sorted unit, one matched pole in the
same layer that is not in Part 86's atlas, closest in firing rate on the scan.

Natural firings: part136/scan136.py reads 4,000 wikitext-103 documents and 4,000 UltraChat rows rendered with the model's chat template,
up to 128 tokens each, from the files on E:, and keeps each pole's top 24 firings at the down_proj input, from different documents, at
position 1 or later. Fixed before scoring: a pole needs 24 firings above a quarter of its own top value, or it is left out and counted.

Effects: part136/remove136.py, for each pole and firing, runs the document to the firing position plus 8 tokens with and without the
neuron set to 0 at that position only. Effect A: change in log-probability over the vocabulary at the firing position. Effect B: summed
over the next 8 positions. Stability: Pearson r between one firing's effect and the mean of the pole's other 23, averaged over the 24.
Steering: change in log-probability at the first answer token of "Who are you?" under Part 86's push; steering agreement is its r
against the pole's mean firing effect.

Comparison: concept against slot, stratified by layer band (L0-7, L8-15, L16-23, L24-31), by a permutation of labels within bands, 10,000
times, seed 136.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing hook, matched random poles, poles as separate units, several controls. Not used: both
directions, because removal has one direction, and dose ladder, because the natural size is the dose. Off-target: Effect A's r
against another pole's effect in the same band.

Gate 1: the do-nothing hook reproduces the plain model bit for bit in one batch. Gate 2, known unit: at least 12 of L22 N2742's 24 top firings
are on alien, extraterrestrial or a legal alien, read by hand. Gate 3: the r of one pole's effect against another pole's mean effect
has an absolute median of at most 0.02 in every band. Gate 4: at least 75% of sorted units have 24 firings. A failed gate stops
the reading.

## Size

- Smallest effect worth knowing: a difference of 0.05 in median stability between concept and slot, within bands.
- Total: 892 poles, 24 firings each, 21,408 removal passes and 21,408 shared baselines, plus 8,000 scan documents and 446 steering
  passes, about 45 minutes on the A770 if forward passes run at about 15 per second.
- Pilot: 24 units, 12 of each kind plus their matched poles, timing batch 8, 16 and 32. Stopping rule: over 90 minutes projected, cut the
  matched poles to half by amendment before the main stage.

## Optimization

- Bound resource: GPU
- Measured: 3.8 documents/s at batch 16 on Part 129's main run on a 3090, two models; the pilot replaces this with a rate on the Arc.
- Levers: baselines shared per firing, taken; batch size timed in the pilot, taken; copy engine off, taken; bf16 declined, Neuron Lens
  shows weak writes round away; the prefix of a document is cached by truncating at firing plus 8, taken.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 136.1 | Median stability (Effect A) of matched poles rises with depth: L24-31 is at least 0.3 above L0-15. | high | Claude |
| 136.2 | Within bands, concept units' median stability is at least 0.05 above slot units' (Effect A). | low | Claude |
| 136.3 | Sorted units' median stability is at least 0.05 above their matched poles' within bands. | low | Claude |
| 136.4 | Within bands, concept units' steering agreement is at least 0.05 above slot units'. | medium | Claude |
| 136.5 | In L0-15 the median steering agreement of sorted units is at most 0.2. | medium | Claude |
| 136.6 | Effect B gives the same direction as Effect A on 136.2 and 136.4. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 136 (registered 2026-10-30, PART136-DESIGN.md): do concept neurons have steadier
natural effects than slot neurons?".

## Reading rules

part136/score136.py computes every number. A permutation p below 0.05 goes beside each difference, and a difference under 0.05 is
reported as such whatever its p. Part 112's sort rests on one Haiku reader, and the same comparison is repeated on units DeepSeek's
version 2 also calls personas; a verdict that flips is reported. A failed Gate 3 means effects carry a shared direction, and the
comparison is withdrawn.

## Limits

One model, two corpora, 128 tokens, top firings only, so natural here means the strongest natural firings. Removal is one neuron at one
token. Effects are on next-token probabilities. Concept and slot come from answers under a push, one sort. Qwen3.5-4B is not the model
of the Neuron Lens catalog, so bands are compared by relative depth at most.
