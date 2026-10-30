# Part 122: do slot and concept neurons read their persona, and in which frame?

Procedure: v1
Registered 2026-10-30 10:15 CT by Claude (Opus 5.5), before any data. Elise asked on 2026-10-30 (Discord message
1565744696012185754): "Do we have any research into how slot vs concept neurons work on read vs write?" We do on write and
hardly at all on read. The commit that adds this file is the registration.

## Question

Part 92 sorted OLMo 2 1B's persona neurons by what a push writes: a concept neuron's theme leaks into most answers, a slot
neuron's persona shows only when the model describes itself. Through the J-lens, 11 of 18 concept neurons match their
persona against 3 of 21 slot neurons (exploratory, jlens-persona-neurons research_log). Nobody has asked what these neurons
read. Does a concept neuron fire on text about its theme? Does a slot neuron fire on text about its persona at all, or only
when the assistant itself is the persona? If the second, that is what "slot" means on the read side.

## Prior art and existing data

- Closest published work: Bryson Tang's and Lyra's neuron explorers compare what a neuron reads with what it writes in single
  neurons. Part 65 compared read labels with write summaries across a sample, on Qwen. None sorts by slot and concept. See Search below.
- Our own data: Part 95 (OLMo L2 N2466, which Part 92 sorts as a slot unit with the persona Chicken): by peak activation
  space passages beat matched ordinary ones in 11 of 16 pairs, and chicken passages were no higher than ordinary ones. Part
  94 (Qwen3.5-4B L24 N3659): dark passages beat matched ones in 14 of 16 pairs. These are the only read tests, on two neurons.
  Part 92's exploratory table: neither kind's input or output weights moved more in post-training.

Search: one search on 2026-10-30 found a Big Five steering study (The Steerable Self, ICML 2026) where a probe that reads a trait did not write it reliably, and DPN-LE, which picks neurons by a read-side contrast and writes through them. Both seen as search snippets only. Neither sorts neurons by how far their effect spreads across questions.

## Method

OLMo 2 1B Instruct at revision 48d788eca847d4d7548f375ad03d3c9312f6139e, bf16, the A770, Part 92's chat frame. The 39 units
of results/part92/step2/neurons.json sorted slot or concept by results/part92/step2/kinds.json. Read means the value of the
down_proj input at the unit's channel, the same value a push adds to, in the unit's layer. No push is applied.

For each unit, DeepSeek V4.1 Flash writes passages from the persona name and description only, with no kind, layer or neuron,
by part122/write_passages.py; they are published with this registration. Three frames, each with 8 matched pairs, where the
ordinary passage has the same structure and length with the persona replaced by an unrelated everyday subject:

- T, third person: a two-sentence passage about the persona, read as plain text.
- F, first person: a two-sentence passage in which the persona speaks, read as plain text.
- C, chat: the same first-person passage placed as the assistant's turn after "Who are you?" in the chat frame. Its ordinary
  pair is the same frame with the matched ordinary first-person passage.

Readout: for each passage, the maximum and the mean of the unit's activation over the passage's own tokens, excluding the
frame's tokens. A pair is a win when the persona passage's maximum is above the ordinary passage's. A unit reads in a frame
when it wins at least 7 of 8 pairs (one-sided sign test p = .035). Mean activation is reported beside it.

Part 92b found the concept sort reader-dependent, so the 8 units both readers call concept and the 21 both call slot are
reported beside the Haiku kinds.

## Controls

- Matched random controls: for each unit, three neurons drawn from its layer with a fixed seed, read on the same passages.
- Shuffled: each unit read on another unit's passages, with the permutation fixed by seed, in the same frame.
- Both directions: a win is tested one-sided; losses are reported, because a unit may be suppressed by its persona.
- Poles: the ordinary passage is the matched pole. Not used: a push, a do-nothing hook, a reader, broken answers, or a dose
  ladder, because nothing is pushed and no answer is generated.
- Off-target: the shuffled control. Select and test: the units were selected on writing, and this part tests reading.

Gate, before any prediction is read: (1) token counts of persona and ordinary passages within a pair differ by at most 2 in at
least 90% of pairs, read by eye on a seeded sample of 24 pairs before the run, with any persona passage that does not concern
its persona replaced; (2) Part 95's L2 N2466 replicates on its own passages (part95/passages95.json): space beats ordinary by
maximum in at least 9 of 16 pairs, so the capture reads the same channel; (3) the random-neuron read rate in each frame is at
most 15% of unit-frame cells. A miss stops the reading, and the tables stay visible under a heading saying so.

## Size

- Smallest effect worth knowing: 9 of 18 concept units against 3 of 21 slot units, the size of the J-lens gap, Fisher p about
  .02.
- Total: 39 units, 3 frames, 8 pairs, 2 passages: 936 forward passes, about 2 minutes on the A770. The controls read the same
  passes. 39 DeepSeek calls to write the passages.
- Pilot: not needed: under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: the A770.
- Measured: n/a: under 5 minutes.
- Levers: every channel read from one pass (taken: the controls add no passes); bf16 (taken: reading needs no fine dose);
  direct load to the device (taken); mid-training checkpoint (declined: a follow-up if the kinds differ).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 122.1 | In frame T, at least 9 of the 18 concept units read their persona. | medium | Claude |
| 122.2 | In frame T, at most 5 of the 21 slot units read their persona. | medium | Claude |
| 122.3 | In frame T, the concept rate is above the slot rate, with a two-sided Fisher p below .05. | medium | Claude |
| 122.4 | In frame C, at least 5 more of the 21 slot units read their persona than in frame T. | low | Claude |
| 122.5 | The gain from T to C is larger for slot units than for concept units, as a share of each kind. | low | Claude |
| 122.6 | In every frame, the shuffled control read rate is at most 10% of units. | medium | Claude |
| 122.7 | Of the 11 concept units whose persona the J-lens matches, at least 7 read in frame T. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 122 (registered 2026-10-30)".

## Reading rules

part122/score122.py computes every number. Gates first. (1) If 122.1, 122.2 and 122.3 hold and 122.4 holds: "Concept neurons
read their theme in plain text; slot neurons do not, and read their persona more when the assistant speaks as it". (2) If
122.1, 122.2 and 122.3 hold and 122.4 fails: "Concept neurons read their theme in plain text and slot neurons do not, in any
frame tested". (3) If 122.3 fails: "These kinds, sorted on writing, do not differ in reading their persona on these
passages". (4) Otherwise the numbers are reported and nothing is read. A read test can only show what a neuron responds to in
these passages: a null does not show that a neuron reads nothing.

## Limits

One model, 39 units, passages written by one model from a short description, 8 pairs per frame, so a unit's verdict is
coarse. Personas with a common word and personas that are phrases are not equally easy to write about. Early-layer units may
read a word, not a meaning, and this part does not separate them. Activation is one channel at one layer. The chat frame's
ordinary pair is still a first-person persona of an everyday subject, not the model's own answer.
