# Part 152: what do the 3,253 neurons on the screen page read, and does it match what they write?

Procedure: v1
Registered 2026-10-31 12:17 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1566111754616971415): "I want to explore L19 N4979 more, use the lenses." She picks neurons by reading the Part 86 answers page, and Part 151b showed that read windows change what a neuron looks like. This puts read windows beside every neuron on that page and tests the read-write match at a size five units could not.

## Question

Part 151b's five units hit their own write lists on 2 of 5 units against a registered 3, too few to say whether reading matches writing in general. Across every neuron on the Part 86 page, does a neuron's J-lens word list hit its own top windows more than another neuron's list does? And on Qwen3.5-4B, do concept neurons read their persona's words where slot neurons do not, as Part 128 found on Qwen3-8B?

## Prior art and existing data

- Closest published work: Neuron Lens (stdlogits), read and write catalogs for Qwen3-8B, a different model. We know of no read catalog for Qwen3.5-4B. Part 128 joined that catalog to Part 91's sort: carriers matched on the fire side for 6 of 11, once neurons for 0 or 1 of 16.
- Our own data: Part 151b, six units on documents 2,000 to 9,999; Part 124, a phrase test on the atlas units with a shuffled null; Part 112's sort of 465 atlas units; the J-lens lists in scripts/research/screen-page/lens.json, computed before this data.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, no chat frame, all 10,000 documents of the atlas's Pile sample, each cut to 256 tokens, batch 8, right padding. Units: the 3,253 neurons in lens.json, layers 10 to 26, plus L22 N2742 if absent. Activation is the input of the down projection. Each document gives one highest and one lowest token per unit. Top 20 and bottom 20 per unit, 30 tokens before and 12 after. Collection is vectorized per layer; the top lists update only where a document beats a unit's current 20th value.

Lists. A unit's J list is its 12 J-lens words from lens.json, stripped and lowercased, keeping words of 4 or more letters with letters only. A unit with fewer than 3 such words is left out of 152.1 and 152.2 and counted. A unit's persona words are the words of Part 112's persona text, lowercased, letters only, 4 or more letters, minus: language, model, large, assistant, that, with, from, this, your, about, have, what, developed, created, trained.

A window hits a list when the marked token or a word in the 12 tokens after it starts with a list word, Part 151's rule. Shuffles pair each unit with another unit's list within the same layer, for J lists, and within the same kind, for persona words. 1,000 shuffles, seed 152. p counts shuffles at or above the real value, with one added to top and bottom.

## Controls

Applied from STANDARD-CONTROLS.md: positive controls, the alien neuron and Part 151b's two readings; a shuffled pairing of units and lists within layer or kind, which keeps each list's commonness and each unit's windows; a split into the two sorted kinds. Not used: pushes, doses and random directions, this part reads and does not push; a second model, the question is about one model; a blind reader, every count is by rule.

Gate 1: 10,000 documents read. Gate 2: 20 top and 20 bottom windows for every unit. Gate 3: at least 5 of L22 N2742's 20 top windows contain alien, extraterrestrial, ufo or martian. Gate 4: at least 12 of L19 N4979's 20 top windows mark " and" or " or", and at least 12 of L24 N8542's 20 meet Part 151b's award rule; Part 151b measured 17 and 17 on 8,000 of these documents, so a failure means the rig broke. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a real mean own-list hit count twice the shuffled mean.
- Total: 10,000 documents, about 2,240,000 tokens, 3,253 units, about 22 minutes on the A770, about 45 MB of windows.
- Pilot: 200 documents on all units, timed, with Part 151's windows for the same units compared to the top values that fall inside those documents.

## Optimization

- Bound resource: GPU
- Measured: Part 151b, 8,000 documents and 1,790,088 tokens in 924 s on the A770 with the copy engine off, six units. So about 19 minutes for the forward passes here, plus column transfers measured in the pilot.
- Levers: vectorized top lists, taken, so no per-unit Python per document; one maximum per document, taken; only the wanted columns cross to the CPU, taken; no per-value item calls, taken; copy engine off, the runner's default, taken; batch size kept at 8 so windows match Part 151b's, declined to change.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 152.1 | Across all units, the mean number of top windows hitting a unit's own J list is at least twice the mean under within-layer shuffles, p below 0.01. | medium | Claude |
| 152.2 | At least 10% of units hit their own J list on at least 3 of 20 top windows. | medium | Claude |
| 152.3 | Among Part 112's concept units, the mean number of top windows hitting a unit's persona words is at least twice the mean under within-kind shuffles, p below 0.01. | medium | Claude |
| 152.4 | Among Part 112's slot units, that mean is less than half the concept units' mean. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 152 (registered 2026-10-31, PART152-DESIGN.md): what do the 3,253 neurons on the screen page read, and does it match what they write?".

## Reading rules

part152/score152.py computes every number. Verdicts use point estimates. If 152.1 fails, the reading that neurons here read what they write is withdrawn at this size too. If 152.3 fails, Part 128's carrier reading did not transfer to this model on this rule. Bottom windows are stored and not scored here; any reading of them is exploratory. Neuron windows go on the page only after the cold check, with this part's limits named.

## Limits

One corpus sample, one model, 256 tokens per document, one maximum per document. The J lists have 12 words, and persona words are few, so hit rates are low. A prefix rule can match unrelated words. The page's neurons were selected for changing "Who are you?" when pushed, so 152.1 and 152.2 describe a selected set. A window shows where a unit fires hardest, not what it is for.
