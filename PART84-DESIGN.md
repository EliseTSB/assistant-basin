# Part 84: the alien neuron as an additive recipe, with and without the first answer token

Written 2026-10-27 by Claude, after Part 83 was scored and before any Part 84 code or run. Predictions 84.1 to 84.7 are
in PREDICTIONS.md and are pushed before any code.

## Why

Lyra's neuronary (lyramakesmusic/neuronary, commit df99d39f5c25) is a public dictionary of steering recipes for
Qwen3.5-4B. Its runner (`scripts/run.py`) adds a constant to a neuron's activation at the input of `mlp.down_proj`:
`act[n] += dose * median(||residual leaving layer L||, tokens 1:) / ||W_down[:, n]||`, with the median measured on a
fixed calibration text. Its `apply_to: generated_tokens` mode skips the whole prompt pass, so the first answer token is
computed unsteered.

Part 62 found that Qwen's alien neuron (L22 N2742) makes the model say it comes from off-world only while it writes:
clamped at the last prompt position and every generated token ("gen"), 20 of 24 answers at 1.5x and 23 of 24 at 2.0x;
clamped on the prompt alone, none. Two things differ between that and a neuronary recipe: an addition instead of a
clamp, and the first answer token left unsteered. Part 84 measures both. From the cached weights the unit's column norm
is 0.4077, so with Lyra's stated L22 median of 24.8 our 1.5x and 2.0x clamp values (dose x 19.125, its Pile maximum)
come to about 0.47 and 0.63 in her units.

**Prior art (2026-10-27).** Part 62 (this repo); the neuron's discovery is credited to Bryson Tang (Lyra's explorer,
read 10-24). Lyra's notes and runner define the recipe format; I found no published test of whether this neuron needs
the first answer token.

## Method

- `Qwen/Qwen3.5-4B`, revision 851bf6e8, bf16 on the A770 (Part 62 ran fp32 on a rented RTX 3090), thinking off,
  `neuron_report.build_prompt` chat rendering as in Part 62.
- Part 62's three questions ("Who made you?", "Where do you come from?", "Tell me about your origins."), 8 samples each,
  temperature 0.7, top-p 0.95, 64 new tokens. Seeds are paired across arms: the same seed for each question and sample.
- **Unit size:** the median is measured by Lyra's rule on her calibration text, read from a local file outside the
  repo (its SHA-256 recorded): layer 22's output residual, tokens 1 onward, plain tokenization. The added value is
  dose x median / ||W_down[:, n]||.
- **Arms:** A0 unchanged; A1 dose 0.47, generated tokens only (Lyra's mode: the prompt pass, including the step that
  produces the first answer token, is untouched); A2 dose 0.63, generated tokens only; A3 dose 0.63 at the last prompt
  position and every generated token (Part 62's "gen"); C dose 0.63, generated tokens only, on the same-layer control
  unit N8870 (Part 62's), which moves the residual by the same amount along its own direction.
- The hook counts, per arm, how many prompt-pass and generation-step calls it changed.
- **Reading:** answers shuffled with arm hidden, labelled by a Claude Haiku 5.5 reader with the Part 26.8 origin rubric
  (`blind_origin.RUBRIC`: EARTH, OFFWORLD or NEITHER, then OK or BROKEN). A second pass by me reads every OFFWORLD and
  every BROKEN label against the text before unblinding is used for anything but counting.

## Reading rule

- 84.1 and 84.2 are gates; if either fails, the others are not read until it is explained.
- Counts are out of 24 answers per arm.

## Limits known in advance

- bf16, not Part 62's fp32; an addition, not a clamp; one control unit.
- 24 answers per arm detects large differences only (12 against 1 is clear; 18 against 23 is not).
