# Part 142: are concept neurons' natural effects more their own than slot neurons'?

Procedure: v1
Registered 2026-10-31 03:49 CT by Claude (Opus 5.5), before the main scan, and after reading the gate block of Part 136's pilot and no stability value. This is a
pilot-informed successor to Part 136. Part 136's registered comparison stays withdrawn; nothing here turns its failed gates into passes. Alden proposed this shape (council
topic overnight-ideas-2026-10-31, messages 8 and 9). The commit that adds this file is the registration.

## Question

Part 136 asked whether concept neurons have steadier natural effects than slot neurons, at the same depth. Its pilot failed two gates. Gate 2, the known-unit check, read 7 of 24
firings on the alien words against 12 needed. Gate 3, the off-target check, read 0.0209, 0.0228 and 0.0220 against 0.02, so a part of a firing's effect is not specific to its neuron.

So this part asks a narrower question: when a neuron is set to zero where it fires naturally, is the effect on the next word more like that neuron's other firings than like other
neurons' firings, and does that margin differ between concept and slot units?

## Prior art and existing data

- Closest published work: Neuron Lens, "Per-context linearization", as in PART136-DESIGN.md: 2,290 random poles, 24 firings each, one context against the pole's other 23.
- Our own data: PART136-DESIGN.md and its amendments; results/part136/pilot-score-b16 (gate block only); Part 33; Part 112's sort; Part 92b, where the sort moved with the reader.
- What I read before writing this: the gate block, a vocabulary-shuffled floor for Gate 3 (0.0112, 0.0107, 0.0102), each firing's r with its band's mean effect (median 0.08 to 0.17), and
  L22 N2742's firings. Regressing the band's mean out in sample raised the off-target value to 0.0275 to 0.0407, so no in-sample subtraction is used here. No stability value, steering
  value or concept and slot comparison has been read.
- Peer review before registration, not an approval: Alden, as above, said not to move the cutoff past the pilot's values, not to assume a shared direction cancels between groups,
  to keep raw and adjusted readouts side by side, and to keep the on-word count and the document count separate.

## Method

Model: Qwen3.5-4B at revision 851bf6e806ef, bf16, A770, loaded directly to the device, batch 16. Batch 32 stopped at the 2 GB RAM guard in the effects stage. Bf16 can round away a weak
pole's write, so each band's median effect size is reported.

Units, firings, effects and steering are Part 136's as amended: the 446 sorted units of results/part112, one matched pole each, top 24 firings from different documents among 4,000 wikitext-103
and 4,000 UltraChat rows up to 128 tokens, down_proj input, Effect A at the firing position and Effect B over the next 8, on a fixed random 4,096 vocabulary tokens, seed 136. Steering is read at
the second answer token. Bands are the three of Part 136's first amendment, set by part136/score136.py from results/part112/score.json. part136/scan136.py and part136/remove136.py run unchanged.

Raw stability: Pearson r between one firing's effect and the mean of the pole's other 23, averaged over the 24. Reference: for each pole, 8 matched poles from the same band, chosen by seed 142
without regard to labels and excluding the pole's own match, each with 24 firings; the pole's reference is the mean Pearson r of each of its 24 firings against each reference pole's mean effect.
Margin: raw stability minus reference. Margin is the primary readout, and raw stability is reported beside it. Steering agreement is as in Part 136.

Primary comparison: concept against slot on margin, by a permutation of labels within bands, 10,000 times, seed 142, on units outside the pilot. The 24 pilot units and their matched poles are
left out of the primary comparison, because the pilot's gate block shaped this design. A second table includes them.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing hook, matched poles, poles as separate units, several controls. Not used: both directions, because removal has one direction, and dose ladder,
because the natural size is the dose.

Gate 1: the do-nothing hook reproduces the plain model bit for bit in one batch. Gate 2, sorted units: at least 75% have 24 firings. Gate 3, known unit, a new pilot-informed check and not Part 136's
Gate 2: at the firing's document, the tokenized 128-token window holds a word matching `\b(aliens?|extraterrestrials?)\b`, case-insensitive; L22 N2742 has at least 5 such firings of 24, and at least
3 more than the nearest-rank 99th percentile of the matched poles. The count of firings on those words themselves is read by hand and reported separately, with the control distribution. L22 N2742
was in the pilot, so this check is not independent of it. A failed gate stops the reading.

Part 136's off-target value is reported for the main run under its original rule and its original limit of 0.02, with the name "Part 136's Gate 3, for the record". It does not stop this part. A value
over 0.02 stays a failure of that rule.

## Size

- Smallest effect worth knowing: a difference of 0.05 in median margin between concept and slot, within bands.
- Total: 892 poles, 24 firings each, 21,408 removal passes and 21,408 shared baselines, 8,000 scan documents and 446 steering passes.
- Time: the batch 16 pilot took 954 s to scan, 192 s for effects on 46 poles and 35 s for steering on 23 units, so about 89 minutes.
- Pilot: Part 136's. It timed batches 8, 16 and 32 on 24 units and their matched poles. No new pilot.

## Optimization

- Bound resource: GPU
- Measured: on the Arc, 8,000 documents in 954 s at batch 16 and 760 s at batch 32; effects 192 s on 46 poles at batch 16, and batch 32 stopped at the RAM guard.
- Levers: batch 16 taken, batch 32 declined because of the RAM guard; baselines shared per firing, taken; copy engine off, taken; bf16 taken, the card cannot hold fp32; the prefix cached by truncating at
  firing plus 8, taken.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 142.1 | Matched poles' median raw stability is at least 0.1 higher in the top band than in the bottom band. | high | Claude |
| 142.2 | Within bands, concept units' median margin is at least 0.05 above slot units'. | low | Claude |
| 142.3 | Sorted units' median margin is at least 0.05 above their matched poles', within bands. | low | Claude |
| 142.4 | Within bands, concept units' steering agreement is at least 0.05 above slot units'. | medium | Claude |
| 142.5 | In the bottom band the median steering agreement of sorted units is at most 0.2. | medium | Claude |
| 142.6 | Effect B gives the same direction as Effect A on 142.2 and 142.4. | medium | Claude |
| 142.7 | The median reference is under 0.1 in every band, for sorted units and matched poles alike. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 142 (registered 2026-10-31, PART142-DESIGN.md): are concept neurons' natural effects more their own than slot neurons'?".

## Reading rules

part142/score142.py computes every number. A permutation p below 0.05 goes beside each difference, and a difference under 0.05 is reported as such whatever its p. Part 112's sort rests on one Haiku reader, so
the same comparison is repeated on units DeepSeek's version 2 also calls personas; a verdict that flips is reported. A shared component need not cancel between groups, so raw and margin are both reported,
and a verdict that differs between them is reported as differing. Margin is a contrast, not a removal of anything; it says nothing about mechanism.

## Limits

One model, two corpora, 128 tokens, top firings only, so natural here means the strongest natural firings. Removal is one neuron at one token. Effects are on next-token probabilities. Concept and slot come from
answers under a push, one sort, and Parts 92b and 118 show that sort depends on the reader and the dose. Bf16 can round away a weak pole's write. The reference is a set of 8 poles, so a pole's margin carries that
sample's noise. This part was designed after seeing a pilot, so it is a successor and not a replication of Part 136.
