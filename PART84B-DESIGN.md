# Part 84b: below the ceiling, does steering the first answer token add to the alien neuron's effect?

Written 2026-10-27 by Claude, after Part 84 was scored and before any Part 84b code or run. Predictions 84b.1 to 84b.5
are in PREDICTIONS.md and are pushed before any code.

## Why

Part 84 found that Qwen3.5-4B's alien neuron (L22 N2742), added at dose 0.63 in Lyra's neuronary units, makes 24 of 24
answers claim an off-world origin whether the addition starts at the last prompt position ("gen") or only at the
second answer token ("decode", her `generated_tokens` mode). Both arms sat at the ceiling, so Part 84 could not say
whether steering the first answer token adds anything. At 0.47, decode gave 13 to 17 of 24 depending on the reading,
against Part 62's 20 of 24 for a clamp of the same size that included the last prompt position, but those differ in
more than the first token (clamp against addition, fp32 against bf16, seeds). Part 84b compares the two modes directly,
below the ceiling, with paired seeds.

**Prior art.** Part 62 and Part 84 (this repo); Lyra's research log reports her own live chat at +0.8 in her units.
None compares the two modes below the ceiling.

## Method

Identical to Part 84 (`part84/part84.py`: model, revision 851bf6e8, bf16, the A770, prompts, sampling, Lyra's sizing
rule on her calibration text kept out of the repo) except:

- **Cells:** A0 unchanged, and doses 0.40, 0.47 and 0.55, each in "decode" and "gen" mode, on L22 N2742.
- **Samples:** 16 per question, 48 per cell; seed = 84500 + 100 x question index + sample, the same in every cell, so
  each decode answer has a gen partner with the same seed. 336 answers.
- **Reading:** answers shuffled under random ids; labelled by a Claude Haiku 5.5 reader with Part 84's prompt
  (`results/part84/reading/reader-prompt.md`: the Part 26.8 rubric plus Part 84's note on company names), now
  registered as is. A second Haiku reader labels the same items independently with the same prompt; its labels are
  a check (84b.5), not a second vote. No hand adjudication.
- **Paired comparison:** for each dose, the number of seed pairs where gen is OFFWORLD and decode is not, against the
  reverse, with an exact sign test on the discordant pairs (reported, not a prediction threshold).

## Reading rule

- 84b.1 is a gate; if it fails, 84b.2 to 84b.4 are not read until it is explained.
- Counts are of 48 answers per cell, from the first reader.

## Limits known in advance

- One neuron, one model, three questions; bf16.
- The first token is usually "I" in both modes; any difference must come through the residual at the last prompt
  position, which later answer tokens attend to.
