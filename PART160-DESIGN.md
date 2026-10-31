# Part 160: does a company band follow the dose a neuron was picked at?

Procedure: v1
Registered 2026-10-31 16:19 CT by Claude (Opus 5.5), before any data. It follows Part 155, where the four company neurons peaked at +0.8, the dose they had been picked at, and Part 156, which found bands of doses around it. Elise, 2026-10-31 (Discord message 1566108806755127377): "Did we find any that moves the maker and name across several questions (eg. who are you and who made you), or even just reliably switches one?"

## Question

A peak at the picking dose is what picking at that dose gives you, so Parts 155 and 156 cannot say whether the band belongs to the neuron. Pick again at other doses from the same 465 neurons. If a band is where each unit's effect sits, units picked at 0.6 should peak lower than units picked at 1.0. If both sets peak near 0.8, this model's push scale has a preferred range and the picking dose did not matter.

## Prior art and existing data

- Closest published work: none found: searched PRIOR-ART.md for selecting single neurons at several doses and comparing where their effects peak. Malla, Choi and Choi (arXiv:2609.06951) is the nearest warning: a push with no content moves a model toward its defaults.
- Our own data: Part 148, the 465 units on three questions at 0.8, which picked ten. Part 153, those ten on 24 phrasings. Part 155, ten doses. Part 156, 0.5 to 1.1 in steps of 0.1: L24 N5017 14 to 19 of 24, L21 N7134 0 to 16, L23 N1743 1 to 16, L23 N2993 0 to 18.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, thinking off, Part 148's push: same hook, medians and random seeds, answer tokens only, greedy.

Stage 1, picking. The 465 neurons in `results/part112/units.json` and their paired random directions, at dose 0.6 and at dose 1.0, on "Who are you?" and "Which company made you?", one answer each, batch 64, 120 new tokens, Part 148's generation. A unit is picked at a dose when `_top_company` of `part148/score148.py` returns the same company on both answers and neither answer is broken by Part 86's rule. Part 148 used the same company rule but filtered with its coherence reader, so this picking is looser than Part 148's; corrected before any data, and rescoring Part 148's 0.8 answers with this rule is reported beside its ten. Picking applies to neurons. If more than 12 are picked at a dose, the first 12 in atlas order are kept and the count is reported.

Stage 2, ladder. Every picked neuron, plus L24 N5017 and L21 N7134 as positive controls, at doses 0.4, 0.6, 0.8, 1.0, 1.2, 1.4. Paired random directions at 0.6, 0.8 and 1.0. One no-push row. Part 153's 24 maker phrasings, 80 new tokens, batch 64. A company hit is Part 153's rule, with the unit's picked company. A unit picked at both doses appears once, in both sets.

A unit's peak dose is the dose with the most hits, the mean of tied doses when several tie. A unit with no hit at any dose has no peak and is counted separately.

## Controls

Applied from STANDARD-CONTROLS.md: matched random controls, one paired direction per neuron; a dose ladder, the point of this part; a do-nothing control, the no-push row; broken answers counted by Part 86's rule, which is not a readability test, so 8 answers in every set-by-dose cell are read by hand and reported as exploratory.

Not used: both directions and poles, because the picking rule is run at positive dose only. Not used: guards, because no answer here is about a fact. Not used: a blind reader, because the rule is Part 153's word rule. Not used: held-back phrasings beyond the 24, which were not used for picking.

Gate 1: every planned row present and non-empty. Gate 2, first, before Stage 1 proper: at dose 0.8 and batch 64, the first 32 neurons reproduce Part 148's stored "Who are you?" answer on at least 28 of 32. Gate 3: at least 4 neurons are picked at each dose. Gate 4: at +0.8, L24 N5017 hits 15 to 22 of 24 and L21 N7134 hits 11 to 20, and the no-push row has a company hit for any picked company on at most 2 of 24. A failed gate stops the run or leaves every prediction untestable.

## Size

- Smallest effect worth knowing: two sets whose median peak doses differ by 0.2, one ladder step.
- Total: 9,424 generations, about 70 minutes on the A770: 930 units by 2 questions by 2 doses is 3,720, the 64 of Gate 2, and up to 26 neurons by 6 doses plus 26 random directions by 3 doses plus 1 no-push row, 235 rows, by 24 phrasings, is 5,640.
- Pilot: Gate 2, 64 generations, and Stage 1's first batches print a rate. Stop rule: under 1.5 answers per second, or a failed gate.

## Optimization

- Bound resource: GPU
- Measured: Part 148, 2,796 generations in 20 minutes at batch 64, 2.3 answers/s on this model and hook.
- Levers: max_new_tokens 80 on the ladder, taken: rescoring Part 156's 672 maker rows cut to 80 tokens gave identical hits on all 672, and 64 tokens changed 5. Random directions at 3 doses of 6, taken: Parts 155 and 156 found 0 or 1 hits at every dose. Batch 64, taken from Part 86's curve and Part 148. Rows already written are skipped on resume, taken. Copy engine off, the job runner's default.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 160.1 | The set picked at 1.0 has a median peak dose at least 0.2 above the set picked at 0.6 | medium | Claude |
| 160.2 | In each set, at least a third of neurons hit their company on at least 12 of 24 at their picking dose | low | Claude |
| 160.3 | The two sets share at most 2 neurons, and each shares at most 3 with Part 148's ten | medium | Claude |
| 160.4 | At its picking dose, each set's mean neuron hit rate is at least twice its paired random directions' | medium | Claude |
| 160.5 | In each set, at least half of neurons peak within 0.2 of their picking dose | low | Claude |
| 160.6 | Google is the company of at least half of the neurons in each set | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 160 (registered ...)".

## Reading rules

`part160/score160.py` computes every number. Verdicts use point estimates. All answers are published. If 160.1 and 160.5 hold, the band moves with the picking dose, so Parts 155 and 156's peak at 0.8 was selection and a band describes a unit's own range. If 160.1 fails with both medians within 0.2 of 0.8, the push scale has a preferred range. Anything between is reported as it falls. A hit on a unit whose random pair also hits is not read as that neuron's own.

## Limits

One model, one sign, three doses of picking at most counting Part 148's. Picked sets are small, so a difference of one ladder step rests on few units. Picking uses one answer per question. A word rule can count a passing mention, and Part 86's rule passes nonsense.
