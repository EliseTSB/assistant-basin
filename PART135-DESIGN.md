# Part 135: are a persona neuron's two poles the same concept, opposites, or unrelated?

Procedure: v1
Registered 2026-10-30 22:46 CT by Claude (Opus 5.5), before any both-pole neuron was counted or read. Chorus Home proposed it
(experiment 5 in reassess/neuron-lens-2026-10-30.md), and Elise relayed it on 2026-10-30. The commit that adds this file is the
registration.

## Question

A neuron has two poles: push it up, or push it down. In Qwen3-8B's persona library some neurons make a persona on both. When they
do, are the two personas the same concept, opposing ends of one idea, or unrelated? Does the relation follow what the two poles
fire on in ordinary text, as the Neuron Lens catalog labels them?

## Prior art and existing data

- Closest published work: Neuron Lens section 5 reports paired promoting and suppressing signs of one high-kurtosis neuron. DPN-LE
  (arXiv 2604.27929) selects neurons exclusive to one trait direction from high and low trait examples. Neither sorts the two signs
  of single neurons across a persona library. A web search on 2026-10-30 found no paper that does.
- Our own data: Part 18, both signs of Qwen3.5-4B L22 N2742 (positive alien, negative maker wording); Part 130's reader for whether
  two persona names mean the same concept; Part 126b and the Part 125 family read the catalog's labels.

## Method

Library: neuronpersonas.com's Qwen3-8B files, fetched 2026-10-30, at the path Part 129 reads. Row i of a layer is neuron i//2, plus
pole when i is even and minus when odd. A pole is a persona pole when its row has category persona. Catalog: Neuron Lens labels at
repo commit 07fdf45 (`neurons/labels`).

part135/pairs135.py builds, with seed 135:

1. Both-pole neurons: both rows are persona poles. Per layer and in total, with the rotation null: rotate the minus-pole flags within
   each layer by a random offset, 1,000 times.
2. Persona pairs: up to 250 both-pole neurons drawn at random, each pole given as its library name and the one-sentence description
   in its row. Shuffled persona pairs: 250 pairs made the same way from a plus pole and a minus pole of two different neurons in the
   same layer, drawn from the same neurons.
3. Fire pairs: for the same 250 neurons, the two poles' catalog `fires` labels. Shuffled fire pairs: as above.

part135/label135.py: DeepSeek V4.1 Flash, max_tokens 4000, 16 workers, three attempts, reads every pair, blind to arm and neuron,
left and right in random order, items shuffled. It answers one word: same (one concept or close synonyms), opposite (two ends of one
idea, such as hero and villain), or unrelated. Luna reads a fixed 15% of rows, drawn before reading.

part135/score135.py counts every prediction.

## Controls

Applied from STANDARD-CONTROLS.md: both directions and poles are the question; matched random controls are the shuffled pairs, matched
on layer and pole sign, and the rotation. Not used: do-nothing hook, broken answers, dose ladder, answer-polarity split, because no
model runs here.

Gate 1, known examples: DeepSeek and Luna each get at least 11 of 12 hand-written pairs right, four of each relation. Written by Claude
in part135/known135.json, committed before reading. Gate 2, shuffled pairs: DeepSeek calls at most 10% of shuffled persona pairs same.
It fails if the reader is generous. Gate 3, second reader: three-way kappa between DeepSeek and Luna on the shared 15% is at least
0.60. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a relation's share 10 points away from the shuffled pairs. 250 pairs give about 4.4 points of 95%
  margin at a share of 25%.
- Total: 1,000 study pairs plus 12 known, about 1,012 DeepSeek calls and 162 Luna calls, about 25 minutes at the measured rate,
  no GPU.
- Pilot: not needed: 12 known examples run first and no generation or GPU is used.

## Optimization

- Bound resource: API
- Measured: 327 rows in about 7.5 minutes, DeepSeek V4.1 Flash at 8 workers on Part 134's labels, about 0.7 rows/s.
- Levers: 16 workers taken; a cheaper reader declined, DeepSeek is already the cheapest with a measured agreement here; batch API
  declined, DeepSeek has none on this route; shared quota estimated at under 2% of the Go window, 1,012 calls of about 2,500 tokens.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 135.1 | Both-pole neurons number at least 2 times the rotation null's mean. | medium | Claude |
| 135.2 | Both-pole share of persona poles is higher in L18-35 than in L0-17. | low | Claude |
| 135.3 | DeepSeek calls at least half of the persona pairs unrelated. | medium | Claude |
| 135.4 | The same share of persona pairs is at least 10 points above the shuffled pairs' same share. | low | Claude |
| 135.5 | The opposite share of persona pairs is at most 10 points above the shuffled pairs' opposite share. | medium | Claude |
| 135.6 | The same share of fire pairs is at most 5 points above the shuffled fire pairs' same share. | low | Claude |
| 135.7 | Three-way agreement between a neuron's persona relation and its fire relation is at most kappa 0.2. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 135 (registered 2026-10-30, PART135-DESIGN.md): are a persona neuron's two poles the
same concept, opposites, or unrelated?".

## Reading rules

score135.py computes every number. 135.3 to 135.7 use DeepSeek's labels; a verdict that flips on Luna's subset is reported. Error rows
are retried, and a row still missing is counted as missing. If fewer than 100 both-pole neurons exist, 135.3 to 135.7 are reported as
too few and not scored. If opposite is high in both persona pairs and shuffled pairs, the label carries no meaning and the
write-up says so.

## Limits

Library labels come from one prompt and one Haiku judge, at each neuron's strongest coherent push, and each pole's dose can differ. A
pair of names says nothing about what the neuron computes. No pushes run here, so no pole's persona is reproduced. Catalog labels are
one reader's summary of top contexts. Qwen3-8B only.
