# Part 56: a neuron atlas and a causal screen, with Haiku as the bulk labeller (registered 2026-10-22 01:30 CT, before any code or run)

Elise, 2026-10-22 01:26 CT: can we use a billion Haiku tokens in the next couple of days (the $89.09 grant expires 10-24)? Prior art asked of Muse
(council muse-part56-prior-art #2); folded in before scoring. Judge: claude-haiku-5-5 on the Batch API, cached system block, a short NOTE line (judge v2 lessons from Part 55: the
reply says nothing on an unremarkable neuron; no real item quoted in the examples). Prices: Haiku batch $0.05 in / $0.25 out per million (under 100K tokens a request).

## 56A the atlas (pilot first)
- Model Qwen3-1.7B @ the pinned revision of Part 49 (172,032 MLP neurons, 28 layers x 6,144). Corpus: the Pile 10k sample at /mnt/e/neuron-probe/corpus (docs truncated to 512 tokens).
  Per neuron keep the 20 documents with the highest activation (one best position per document) as a 64-token window with the firing token marked, plus 10 windows at random among positions with activation above 10% of the neuron's max
  (selectivity), plus its max and the share of positions above 10% of max.
- Pilot gate (cheap, before the full run): 2,000 random neurons, 2,000 random directions per the control below, and four known units (wizard L19:4826, magic L22:1610, and two more the build picks from Part 35's rank-1 list).
- Control: a random unit-norm direction in the same intermediate space for each real neuron sampled, run through the identical pipeline (its top windows, its description). A description is judged against what random directions get.
- Haiku reply per neuron (JSON): concept (at most 12 words), one_job 1 to 5, kind in {token, word-family, topic, syntax, format, identity-or-origin, persona-or-creator, other, none}, note (empty unless the neuron is unusual).
- Scale if the pilot gate passes: all 172,032 neurons plus 172,032 matched random directions. About 1.8K tokens each, so roughly 0.6B input tokens and $30 for the pair.

## 56B the causal screen (after 56A)
Take the neurons flagged identity-or-origin or persona-or-creator (expected a few hundred to a couple of thousand) plus an equal number of random unflagged neurons. Clamp each at -2, -1, +1, +2 times its largest activation on the three origin questions of Part 26.8, 4 samples, and have Haiku
(judge v2 with the Part 26.8 rubric and a NOTE) grade EARTH / OFFWORLD / NEITHER / BROKEN and name any maker mentioned. GPU-bound: roughly 2,000 x 4 x 3 x 4 = 96,000 generations, so a pilot of 200 neurons first.

## Predictions (Claude)
| id | prediction |
|---|---|
| 56a | Positive controls: Haiku's description of the wizard unit (L19:4826) names spells, magic or wizards, and of the magic unit (L22:1610) names spells or magic. 2 of 2. |
| 56b | Random directions: at most 20% get one_job 4 or 5; real neurons exceed that share by at least 10 points (earlier judges separated real from random by only 7 to 9 points). |
| 56c | Stability: re-describing 500 random neurons from the same windows in a different order gives the same kind in at least 80% and the same concept words (any shared content word) in at least 70%. |
| 56d | Flags: at most 1% of real neurons are flagged identity-or-origin or persona-or-creator, and random directions are flagged at under a third of that rate. |
| 56e | Causal precision (56B pilot): of flagged neurons, at most 25% move any of the label shares (EARTH, OFFWORLD, or a named maker) by 3 or more answers of 12 at some dose, against at most 5% of the random unflagged neurons. A description predicts a neuron's causal effect only weakly (Parts 18 to 24: firing is not driving). |
| 56f | The Part 26.8 twin (OLMo L10:6978) is flagged identity-or-origin or persona-or-creator by Haiku in an OLMo-2-1B pilot atlas of its own layer, i.e. the atlas would have found it. |
