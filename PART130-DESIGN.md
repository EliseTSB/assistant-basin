# Part 130: is the same pole a persona in all three Llama fine-tunes, against a defined chance?

Procedure: v1
Registered 2026-10-30 19:28 CT by Claude (Opus 5.5), before the overlap was computed. Elise on the call, 2026-10-30 18:34 CT, voice line
27: test Sauers' sibling neurons and see how they affect non-persona questions. This free count picks the units for that
test. The commit that adds this file is the registration.

## Question

Sauers (x.com/Sauers_/status/2116093664288178672, read from a figure) reports 3,681 neurons that are a persona in Llama 3.1
8B Instruct, Tulu 3 8B and Hermes 3 8B against 294 by chance, with no null defined. The library is public, so the count can
be redone with a stated null, per layer and per pair of models.

## Prior art and existing data

- Closest published work: that figure, built on Josh Fonseca Rivera's Neuron Persona Library (neuronpersonas.com).
- Our own data: Parts 90 and 90b, one neuron across Tulu's stages; Part 91, personas in Qwen3-8B-Base.

## Method

Library files fetched 2026-10-30. A pole counts when its row has category persona and a name. part130/overlap130.py counts
poles that are a persona in all three models, and in each pair. Null: rotate Tulu's and Hermes' flags within each layer by a
random offset, 1,000 times, seed 130. A reader judges whether the Llama and Hermes names mean the same concept on 150 random
triple poles, each beside a shuffled pair.

## Controls

Applied from STANDARD-CONTROLS.md: matched random controls, the rotation and the shuffled names. Not used: do-nothing hook,
broken answers and dose ladder, because no model runs here.

Gate: DeepSeek calls at most 10% of shuffled name pairs the same, and Luna on 15% of rows agrees at kappa 0.6 or more. It
fails if the reader is generous.

## Size

- Smallest effect worth knowing: a triple count twice the null, which the counts settle.
- Total: 300 reader calls plus 45 for Luna, about 3 minutes.
- Pilot: not needed: no generation and no GPU.

## Optimization

- Bound resource: API
- Measured: n/a: under 5 minutes.
- Levers: rotation is vectorized numpy; cheap readers taken.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 130.1 | The triple count is at least 5 times the rotation null's mean. | high | Claude |
| 130.2 | The ratio of observed to null is at least 1.5 times higher in L0-15 than in L16-31. | low | Claude |
| 130.3 | Among sampled triple poles, the reader calls the Llama and Hermes names the same concept in at least 50%. | medium | Claude |
| 130.4 | The three pairwise ratios of observed to null are within a factor of 1.5 of each other. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 130 (registered 2026-10-30, PART130-DESIGN.md): is the same pole a persona in all three Llama fine-tunes, against a defined chance?".

## Reading rules

Counts come from overlap130.json. 130.3 uses DeepSeek's labels; a verdict that flips on Luna's subset is reported. Error rows
are retried, and a row still missing is counted as missing.

## Limits

Library labels come from one prompt and one Haiku judge, at each neuron's strongest coherent push. A shared persona across
fine-tunes says nothing yet about Llama's base model; that is the next part.
