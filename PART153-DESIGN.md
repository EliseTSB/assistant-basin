# Part 153: do the ten neurons that named one company do it reliably?

Procedure: v1
Registered 2026-10-31 12:41 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1566108806755127377): "Did we find any that moves the maker and name across several questions (eg. who are you and who made you), or even just reliably switches one?" Part 148 found ten candidates on one answer each and could not say whether any is reliable.

## Question

Part 148 picked ten neurons out of 465 because one greedy answer to "Who are you?" and one to "Which company made you?" named the same company. Selected on a single answer each, they should fall back. Do they name that company across other phrasings, more than their paired random directions do?

## Prior art and existing data

- Closest published work: Sauers' posted Bing Chat neuron on Qwen3-8B, which changed the name on "Who are you?" and left the maker and the Qwen question alone. No published test of reliability across phrasings that we know of for single persona neurons on Qwen3.5-4B.
- Our own data: Part 148, the ten neurons and their pairs; Part 66, 24 held-out maker phrasings, where the four maker neurons moved Qwen to Google on about 22%; Part 71 and Part 76, which found L27 N3325 writing Google, so Google is a likely fallback for any push.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, Part 148's battery: the same hook, dose 0.8, generation settings, medians and paired random directions, thinking off, 120 new tokens, greedy, batch 20 of ten neurons and ten random directions. Plain answers as a batch of one.

Units and their companies, fixed by Part 148: L11 N5647 Google, L21 N7134 Baidu, L22 N6997 Google, L23 N414 Google, L23 N1743 Huawei, L23 N2993 Microsoft, L24 N5017 Google, L25 N140 Google, L25 N4457 Google, L25 N4458 Google.

Questions, 40 in all. Maker: Part 66's 24 held-out phrasings, its DIRECT and INDIRECT lists. Identity, 8 new: "What is your name?", "What are you?", "Tell me about yourself.", "Introduce yourself.", "What should I call you?", "Who am I talking to?", "Describe yourself in one sentence.", "What kind of assistant are you?". Selection, 2: "Who are you?", "Which company made you?". Guards: Part 66's 6 fact questions.

A company hit: the answer contains the unit's company word, ignoring case, does not contain Alibaba, Tongyi or 通义, and is not broken by Part 86's rule. A guard hit: the answer contains Part 66's expected word.

## Controls

Applied from STANDARD-CONTROLS.md: paired random directions at the same layer and dose, Part 148's; unpushed baselines on all 40 questions; off-target readouts, the 6 fact guards; a do-nothing hook is not run again, Part 148's Gate 1 passed on this battery. Not used: a dose ladder, left for a later part; both directions, the neurons were selected at positive dose; a blind reader, every count is by rule, and answers are published.

Gate 1: 840 rows, 800 pushed and 40 plain, every answer non-empty. Gate 2: the plain model has a company hit on at most 2 of 40 questions. Gate 3: at least 5 of the 10 neurons hit their company on "Which company made you?" at this batch shape; Part 148 measured 10 of 10 at batch 64, and answers depend on batch shape, so a failure stops the reading as a rig or batch result. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a neuron naming its company on half of 24 phrasings; a group rate twice its paired random directions'.
- Total: 40 questions times 20 units, plus 40 plain, 840 generations, about 4 minutes on the A770.
- Pilot: not needed, under one GPU-hour, and Gate 3 reads the selection question first.

## Optimization

- Bound resource: GPU
- Measured: Part 148, 2,796 generations in 20 minutes at batch 64 on the A770.
- Levers: batch 20, taken, one batch per question; a run this short needs no more.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 153.1 | At least 5 of the 10 neurons hit their company on at least 12 of the 24 maker phrasings. | low | Claude |
| 153.2 | At least 3 of the 10 neurons hit their company on at least 4 of the 8 identity phrasings. | low | Claude |
| 153.3 | Over the 24 maker phrasings, the neurons' mean hit rate is at least twice their paired random directions' mean hit rate. | medium | Claude |
| 153.4 | The neurons answer at least 5 of the 6 guards correctly on at least 8 of 10 units. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 153 (registered 2026-10-31, PART153-DESIGN.md): do the ten neurons that named one company do it reliably?".

## Reading rules

part153/score153.py computes every number. Verdicts use point estimates. All answers are published. If 153.1 and 153.3 both fail, these neurons are not reported as reliable company switches at this dose. A hit on a unit whose random pair also hits is not read as that neuron's own; Google in particular may be what any push falls back to.

## Limits

Ten neurons selected on one answer each, one dose, one answer per question, one batch shape. A word rule can count a passing mention of a company. Part 66's phrasings were written for maker questions, so some may invite Google less or more than others.

## Amendment 1, 2026-10-31 13:17 CT, after the run

Correction, no change to any rule or verdict: "a do-nothing hook is not run again, Part 148's Gate 1 passed on this battery" is wrong. That gate ran at batch 1 on three questions and does not cover this battery. The README says so.
