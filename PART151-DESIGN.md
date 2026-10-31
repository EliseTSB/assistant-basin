# Part 151: what do Part 150's five units read on Qwen3.5-4B itself?

Procedure: v1
Registered 2026-10-31 10:52 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1566111754616971415): "I want to explore L19 N4979 more, use the lenses." Part 150's read guesses came from a Qwen3-1.7B atlas, so this is the read side on the right model. The commit that adds this file is the registration.

## Question

Part 150 found that four of these five units write their lens words when pushed. Do they also fire on those words in ordinary text, with no push? Part 128 found that, on Qwen3-8B, units carrying a theme across questions tended to read it, and once-only units did not.

## Prior art and existing data

- Closest published work: Neuron Lens (stdlogits), read catalogs for Qwen3-8B, a different model, and the read-versus-write split in Part 128 of this repo.
- Our own data: results/part150, the write side; the atlas-full files, Qwen3-1.7B, not this model. No Qwen3.5-4B read catalog exists on disk.
- No peer review before registration: about 10 minutes on the A770.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, no chat frame, the first 2,000 documents of the atlas's Pile sample, each cut to 256 tokens, batch 8, right padding. Activation is the input of the down projection, each document's single highest and lowest token per unit. Top 20 and bottom 20 per unit, with 30 tokens before and 12 after. Units: L19 N4979, L19 N5975, L23 N116, L24 N6835, L24 N8542, and L22 N2742 as the positive control.

A window hits a list when the marked token or a word in the 12 tokens after it starts with a word from the list. Lists: Part 150's lens_words.json, fixed before this data.

## Controls

Applied from STANDARD-CONTROLS.md: a positive control, the alien neuron; each unit's windows scored against the other four units' lists, the control for how often any list hits. Not used: pushes, doses and random directions, this part reads and does not push; a blind reader, the score is a word match and the windows are published.

Gate 1: 2,000 documents read. Gate 2: 20 top windows for every unit. Gate 3: at least 5 of L22 N2742's 20 top windows contain alien, extraterrestrial, ufo or martian. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: 5 of 20 windows on the unit's own list, against about 0 to 2 on other lists.
- Total: 2,000 documents, about 500,000 tokens, about 10 minutes on the A770.
- Pilot: not needed, forward passes only, under one GPU-hour.

## Optimization

- Bound resource: GPU
- Measured: Part 86's atlas collection, 80 batches in 73 s at the copy-engine-off setting on the A770.
- Levers: only the six unit columns cross to the CPU per batch, taken; no per-value item calls, taken; copy engine off, the runner's default, taken; one maximum per document, taken, it also keeps windows from one document from filling a list; batch size inherited from the atlas, not timed, declined because the run is about 10 minutes.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 151.1 | At least 3 of the 5 units hit their own list on more top windows than on any other unit's list. | medium | Claude |
| 151.2 | L24 N8542 hits its own list on at least 5 of 20 top windows. | high | Claude |
| 151.3 | L24 N6835 hits its own list on at least 5 of 20 top windows. | high | Claude |
| 151.4 | L19 N4979 hits its own list on at least 5 of 20 top windows. | low | Claude |
| 151.5 | L19 N5975 hits its own list on at least 5 of 20 top windows. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 151 (registered 2026-10-31, PART151-DESIGN.md): what do Part 150's five units read on Qwen3.5-4B itself?".

## Reading rules

part151/score151.py computes every number. Verdicts use point estimates. All windows are published and read by a person before any sentence about what a unit reads. If 151.1 fails, the reading that these units read what they write is withdrawn for this corpus.

## Limits

One corpus sample, 2,000 documents, 256 tokens each, one model, one maximum per document. A list that is short or common can hit by chance, which the other units' lists are there to show. Five hand-picked units.
