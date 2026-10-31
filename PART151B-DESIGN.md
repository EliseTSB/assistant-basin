# Part 151b: do Part 150's five units read on new documents what the first 2,000 documents suggested?

Procedure: v1
Registered 2026-10-31 11:15 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1566111754616971415): "I want to explore L19 N4979 more, use the lenses." and (1566112825909190868): "L19 N5975 L24 N8542 L24 N6835 L23 N116 explore these ones too". Part 151 failed its control gate, so its predictions are untestable and the readings I gave Elise from its windows are not results.

## Question

Part 151 read these units on the first 2,000 documents and its alien control reached 2 of 20 windows against a registered 5, because those documents hold few aliens. Does the read side hold on documents neither Part 151 nor I have seen? The two readings Part 151's windows suggested, L19 N4979 on " and" or " or" and L24 N8542 on awards and nominations, are fixed here as predictions before any new window exists.

## Prior art and existing data

- Closest published work: Neuron Lens (stdlogits), read catalogs for Qwen3-8B, a different model. No Qwen3.5-4B read catalog is known to us.
- Our own data: results/part151, windows on documents 0 to 1,999, read by me and described to Elise as reading, not result. By my reading 18 of L19 N4979's top 20 mark " and" or " or", and 14 of L24 N8542's top 20 mention awards or nominations. Those counts were made after seeing the windows, which is why this part uses other documents.
- Counted before registration, text only, no activations: 13 documents after number 2,000 name alien, extraterrestrial, UFO or martian in their first 1,100 characters, 2 before it.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, no chat frame, documents 2,000 to 9,999 of the atlas's Pile sample, none read in Part 151, each cut to 256 tokens, batch 8, right padding. part151/read151.py gains a --start option; nothing else in it changes. Activation is the input of the down projection, each document's single highest and lowest token per unit. Top 20 and bottom 20 per unit, with 30 tokens before and 12 after. Units: L19 N4979, L19 N5975, L23 N116, L24 N6835, L24 N8542, and L22 N2742 as the positive control.

A window hits a list when the marked token or a word in the 12 tokens after it starts with a word from the list. Lists: Part 150's lens_words.json, fixed before this data. For 151b.6 a window counts when its marked token, stripped and lowercased, is "and" or "or". For 151b.7 a window counts when its before, marked and after text matches award, nominat, oscar, prize, emmy, grammy, golden globe, winner or winning, ignoring case.

## Controls

Applied from STANDARD-CONTROLS.md: a positive control, the alien neuron; each unit's windows scored against the other four units' lists, and 151b.6 and 151b.7 counted on the other four units too, the control for how often any rule hits. Not used: pushes, doses and random directions, this part reads and does not push; a second model, this asks about one model.

Gate 1: 8,000 documents read. Gate 2: 20 top windows for every unit. Gate 3: at least 5 of L22 N2742's 20 top windows contain alien, extraterrestrial, ufo or martian, the same gate as Part 151. It can fail: 13 documents is few, and Part 151's other top windows were about space. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: 5 of 20 windows on the unit's own list, against about 0 to 2 on other lists.
- Total: 8,000 documents, about 1,780,000 tokens, about 16 minutes on the A770.
- Pilot: not needed, the same script read 2,000 documents in Part 151 and passed gates 1 and 2.

## Optimization

- Bound resource: GPU
- Measured: Part 151, 2,000 documents and 445,795 tokens in 239 s on the A770 with the copy engine off, so about 16 minutes here.
- Levers: only the six unit columns cross to the CPU per batch, taken; no per-value item calls, taken; copy engine off, the runner's default, taken; one maximum per document, taken; batch size kept at 8 for the same windows as Part 151, declined to change because the run is 16 minutes.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 151b.1 | At least 3 of the 5 units hit their own list on more top windows than on any other unit's list. | medium | Claude |
| 151b.2 | L24 N8542 hits its own list on at least 5 of 20 top windows. | high | Claude |
| 151b.3 | L24 N6835 hits its own list on at least 5 of 20 top windows. | high | Claude |
| 151b.4 | L19 N4979 hits its own list on at least 5 of 20 top windows. | low | Claude |
| 151b.5 | L19 N5975 hits its own list on at least 5 of 20 top windows. | medium | Claude |
| 151b.6 | At least 10 of L19 N4979's 20 top windows mark " and" or " or", more than on any of the other four units. | medium | Claude |
| 151b.7 | At least 10 of L24 N8542's 20 top windows mention awards or nominations by the Method's rule, more than on any of the other four units. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 151b (registered 2026-10-31, PART151B-DESIGN.md): do Part 150's five units read on new documents what the first 2,000 documents suggested?".

## Reading rules

part151b/score151b.py computes every number. Verdicts use point estimates. All windows are published and read by a person before any sentence about what a unit reads. If 151b.1 fails, the reading that these units read what they write is withdrawn for this corpus. If 151b.6 or 151b.7 fails, the matching reading I gave Elise is withdrawn, and I tell her so.

## Limits

One corpus sample, one model, 256 tokens per document, one maximum per document. A list that is short or common can hit by chance, which the other units' lists are there to show. Five hand-picked units. A window shows where a unit fires hardest, not what it is for. The rules for 151b.6 and 151b.7 came from reading Part 151's windows, so they are tested on new documents and not on new units.
