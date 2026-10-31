# Part 142: are concept neurons' natural effects more their own than slot neurons'?

On Qwen3.5-4B in bf16, the primary comparison excludes 22 of the 24 pilot units and their matched poles. Among the remaining 364 units, the concept-minus-slot median margin difference is 0.056 (permutation p=0.0001). This holds for Part 112’s Haiku sort under the registered threshold. It fails with pilot units included: the difference is 0.047 (p=0.0005). It also fails on DeepSeek v2’s 43-persona subset: 0.004 (p=0.909). Raw stability and margin give the same threshold verdict for the primary comparison.

A firing is one of a unit's 24 highest-activation positions, selected from different documents in the 8,000-document scan. A pole is a sorted unit or one of its matched control units. Effect A is the change in sampled next-token log probabilities when the firing unit is set to zero at its firing position. Effect B is the summed change over the next eight positions. Raw stability is the mean correlation between each firing's effect vector and the mean vector of that pole's other defined firings. The reference is the mean correlation with eight same-band matched poles. Margin is raw stability minus that reference. Margin is a contrast; it does not remove a shared effect or identify a mechanism. Steering agreement is the correlation between the mean zeroing-effect vector and the change from pushing the unit at Part 86's dose. Since zeroing and pushing act in opposite directions, a negative steering agreement is the sign one would expect under direct reversal. That sign alone does not establish a mechanism.

## Gates

1. **Gate 1, do-nothing hook:** passed. The hook reproduced the plain model bit for bit in one batch.
2. **Gate 2, firing coverage:** passed. 386 of 446 sorted units had all 24 firings (86.55%, against a 75% minimum).
3. **Gate 3, known unit L22 N2742:** passed. The tokenizer-decoded 128-token windows for 9 of its 24 firing documents contained `alien(s)` or `extraterrestrial(s)`. The nearest-rank 99th percentile among matched-pole document counts was 1; the unit's count is at least 5 and at least 3 above that percentile. By hand, 7 of the 24 firing tokens themselves were on those words: five `alien`, one `ater` (inside *extraterrestrial*), and one `ens` (inside *Aliens*). The document/window count and on-word firing-token count differ. The gate is pilot-informed because L22 N2742 was in Part 136's pilot. The matched-pole distribution of on-word token counts was not recomputed; the local tokenizer snapshot was unavailable.
4. **Gate 4, finite effects:** passed after the first effects run failed. The rerun had 0 non-finite rows of 18,672. Gate 4 was added after the failure and stops the reading if any effect row is non-finite.

The gate pole's 24 firing positions, tokens decoded from the local Qwen tokenizer snapshot, and short contexts are below. “Chat” means UltraChat; “wiki” means wikitext. Contexts are brief excerpts around the firing position.

| # | Source | Token | Short context |
| ---: | --- | --- | --- |
| 1 | chat | `ater` | advanced extraterrestrial civilization |
| 2 | chat | `of` | some kind of physical phenomenon |
| 3 | chat | `alien` | save the world from an alien invasion |
| 4 | chat | `alien` | save his daughter from an imminent alien invasion |
| 5 | chat | `ens` | fight against Aliens or Zombies |
| 6 | chat | `.` | societal structures. Overall, the film |
| 7 | chat | `alien` | humans who discover an alien civilization |
| 8 | wiki | `alien` | fictional alien. This was truly a solo album |
| 9 | wiki | `1` | broadcast October 30, 1938 |
| 10 | chat | `0` | at least 1,000 words about a character |
| 11 | wiki | `of` | search of planets |
| 12 | chat | `alien` | haunted by supernatural entities or alien creatures |
| 13 | wiki | `,` | a girl from Venus, who lands on Earth |
| 14 | chat | `on` | planet they've landed on has been masquerading |
| 15 | chat | `Black` | Men in Black 3 |
| 16 | chat | `stellar` | interstellar travel |
| 17 | chat | `an` | discovers that an asteroid is on a collision course |
| 18 | wiki | `a` | artifact was indeed a vase |
| 19 | wiki | `that` | scientists that closely orbiting Hot Jupiter planets |
| 20 | chat | `space` | blasting into space like an astronaut |
| 21 | wiki | `a` | was indeed a planet, designated WASP-44b |
| 22 | chat | `if` | astronomers have wondered if the sun's core spins |
| 23 | wiki | `4` | WASP-4b |
| 24 | chat | `a` | Superman has a deep compassion for humanity |

Part 136's Gate 3, for the record, uses its original off-target rule and 0.02 limit. It failed in every band: median absolute correlation was 0.02292 in L0-22 (1,901,808 firing-to-other-pole comparisons across 282 poles), 0.02167 in L23-23 (852,768 comparisons, 189 poles), and 0.02150 in L24-31 (2,254,608 comparisons, 307 poles). This does not stop Part 142's reading.

## Predictions

Differences under 0.05 are reported as such even when the permutation p is small. Permutations shuffle concept and slot labels within bands 10,000 times with seed 142. The 142.3 matched-pole contrast has no permutation p in the scorer.

| # | Prediction (registered wording) | Confidence | Result | Verdict |
| --- | --- | --- | --- | --- |
| 142.1 | Matched poles' median raw stability is at least 0.1 higher in the top band than in the bottom band. | high | Difference 0.124 (155 top-band and 141 bottom-band matched poles); no permutation p computed for this contrast. | held |
| 142.2 | Within bands, concept units' median margin is at least 0.05 above slot units'. | low | Primary, with pilot units excluded: difference 0.056, p=0.0001, n=364. It fails with pilot units included: 0.047, p=0.0005, n=386. It fails on DeepSeek v2's persona subset: 0.004, p=0.909, n=43. | held |
| 142.3 | Sorted units' median margin is at least 0.05 above their matched poles', within bands. | low | Difference 0.004; sorted units n=364 and matched poles n=369 across bands; no permutation p computed for this contrast. Sorted units barely differ from matched poles. | failed |
| 142.4 | Within bands, concept units' steering agreement is at least 0.05 above slot units'. | medium | Primary difference -0.024, p=0.012, n=364. The p value is for a difference in the opposite direction from the prediction. DeepSeek's repeat is +0.010 (sign flips). Concept minus slot in steering-agreement magnitude is +0.024; not registered. | failed |
| 142.5 | In the bottom band the median steering agreement of sorted units is at most 0.2. | medium | -0.316. | held |
| 142.6 | Effect B gives the same direction as Effect A on 142.2 and 142.4. | medium | Effect B margin difference 0.083, p=0.0001; steering difference -0.041, p=0.0003. Both directions match Effect A. | held |
| 142.7 | The median reference is under 0.1 in every band, for sorted units and matched poles alike. | medium | Sorted-unit/matched-pole medians: L0-22 0.00132/0.00031; L23-23 0.00021/-0.00056; L24-31 0.00043/0.00048. | held |

The comparison on the DeepSeek v2 subset has only 43 units with a unique persona match, compared with 364 units in the primary comparison. The all-unit margin difference is 0.047, so including the pilot changes the result below the registered threshold. The results table includes pilot units; 22 of the 24 pilot units were excluded from the primary comparison. Steering agreement is negative in general because zeroing and pushing act in opposite directions. The registered prediction that concept steering agreement would be at least 0.05 above slot agreement assumed the wrong sign. For 142.3, the scorer's band-weighted difference uses 133/133 sorted/matched poles in L0-22, 87/88 in L23-23, and 144/148 in L24-31.

## Results by band

Values are medians. Raw stability, margin, and steering agreement are shown for concept and slot units. Effect sizes are median L2 norms of the effect vectors over the firing rows in each band.

| Band | Kind | n | Raw stability | Margin | Steering agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| L0-22 | concept | 78 | 0.53636 | 0.53435 | -0.33351 |
| L0-22 | slot | 62 | 0.50505 | 0.51040 | -0.28961 |
| L23-23 | concept | 49 | 0.61039 | 0.60749 | -0.37196 |
| L23-23 | slot | 45 | 0.55611 | 0.55218 | -0.37093 |
| L24-31 | concept | 92 | 0.68147 | 0.68524 | -0.44754 |
| L24-31 | slot | 60 | 0.62252 | 0.62191 | -0.42660 |

| Band | Effect A median L2 | Effect B median L2 |
| --- | ---: | ---: |
| L0-22 | 6.36552 | 6.06642 |
| L23-23 | 6.53887 | 5.73695 |
| L24-31 | 7.18284 | 5.41165 |

Effect A had no undefined firing vectors. Effect B had 176 undefined firing vectors. These are firings at a window's end with an empty sum, not bf16 rounding. Across 1,720 rows, Effect B sums fewer than eight positions. Counts by band and kind:

| Band | Kind | Effect A undefined | Effect B undefined |
| --- | --- | ---: | ---: |
| L0-22 | concept | 0 | 21 |
| L0-22 | slot | 0 | 19 |
| L0-22 | matched pole | 0 | 33 |
| L23-23 | concept | 0 | 11 |
| L23-23 | slot | 0 | 7 |
| L23-23 | matched pole | 0 | 28 |
| L24-31 | concept | 0 | 26 |
| L24-31 | slot | 0 | 12 |
| L24-31 | matched pole | 0 | 19 |

The raw concept-minus-slot difference is 0.057 (p=0.0001); the margin difference is 0.056 (p=0.0001). Both exceed 0.05, so raw stability and margin agree on the primary threshold verdict. The small change from raw to margin does not imply that shared effects cancel between groups. The DeepSeek v2 subset gives a margin difference of 0.004, below 0.05.

## Amendment 2 and run history

The first main effects run was void: 17,152 of 18,672 effect rows had NaN in every Effect A position. It stopped twice with `UR_RESULT_ERROR_UNKNOWN`. Its prediction scores were NaN, and no finite margin or difference from that run was read. Steering was separate and had no NaNs in its 778 rows.

The effects code changed to send only read positions through `lm_head` and to raise on any non-finite log probability. The scorer now treats non-finite vectors as undefined. Gate 4, requiring zero non-finite effect rows, was added after the failure. The effects stage reran from its first row at batch 16. It took 50 minutes, ended with no error, and had zero non-finite rows. `results/part142/repeat-check.json` records that all 960 of 960 old finite Effect A rows matched on the repeat check, with median Pearson r=1.0.

The cause of the NaNs is not established. Memory pressure is a guess: the old full-logit path had an estimated peak near the GPU's available memory, but the NaNs occurred without an error and no GPU reset was recorded. The fix stopped the NaNs in the rerun; it did not prove why they occurred.

The design estimated about 89 minutes from pilot timings. Batch 16 was used; batch 32 had stopped at a 2 GB RAM guard in the earlier effects pilot. The run processed 778 poles, with 18,672 removal passes and 18,672 shared baselines. It left out 116 poles with fewer than 24 firings, as listed in `results/part142/scan/left_out.json`. It used 8,000 wikitext-103 and UltraChat documents capped at 128 tokens, the top 24 firings from different documents, down-projection input zeroing, a fixed random sample of 4,096 vocabulary tokens, and steering at the second answer token.

## Method and limits

The bands are L0-22, L23-23, and L24-31, fixed from Part 112's score. The model was Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, loaded directly on the A770 in bf16 at batch 16. The primary concept-versus-slot comparison excludes the 24 pilot units and their matched poles. Of the 24 pilot units, 22 had data and were excluded from the primary; the results table above includes those units. The permutation test shuffles labels within bands 10,000 times with seed 142. The reference for each unit uses eight same-band matched poles chosen without labels.

The closest published work named in the design is Neuron Lens, “Per-context linearization”: 2,290 random poles, 24 firings per pole, and one context compared with the pole's other 23 firings.

This is one model, two corpora, 128-token windows, and the top firings only. “Natural” here means the strongest natural firings found in these documents. Removal is one unit at one token, and the measured effects are on next-token probabilities. The concept and slot labels came from answers to a push and one sort; Parts 92b and 118 show that the sort depends on the reader and dose. Bf16 can round away a weak pole's write. Each reference uses eight poles, so its margin includes noise from that sample. The part was designed after a pilot and is a successor to Part 136, not a replication.

## Files

The score and its gates come from `results/part142/score/score142.json` and `results/part142/score/SCORE.md`. Its `poles_with_fewer_than_12_defined_firings` key includes the 116 left-out poles; the true count of poles failing that defined-firing threshold is 0. The scan count and firing contexts come from `results/part142/scan/scan_meta.json` and `results/part142/scan/firings.jsonl`. The gate hook record and steering rows are in `results/part142/effects/gates.json` and `results/part142/effects/steering.jsonl`. `part142/score142.py` defines the score calculations; `part136/remove136.py` defines the effect and steering calculations. The large the effects file is not copied here.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; nine findings folded, and effect sizes and permutation p values could not be recomputed without the large file.
