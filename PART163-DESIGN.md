# Part 163: picked the same way, do neurons move the model's name more than random directions?

Procedure: v1
Registered 2026-10-31 19:55 CT by Claude (Opus 5.5), before any data. It follows Part 161, where among reliable units at the picking dose 16.3% of neuron answers still said Qwen against 42.9% of random answers. That gap was unplanned, and the two arms differ in how they were chosen: the neurons came from Part 112's atlas, chosen for changing what the model calls itself, and the random directions did not. Elise, 2026-10-31 (Discord message 1566108806755127377): "Did we find any that moves the maker and name across several questions (eg. who are you and who made you".

## Question

Choose neurons and random directions by one rule, on one question, at one dose: the units whose push changed "Who are you?" in Part 86's screen. Then ask new phrasings. Do the neurons move the name and the maker together more often than random directions do, or was Part 161's gap the selection?

## Prior art and existing data

- Closest published work: Mayner on X (x.com/wmayner/status/2116628249153515740): on Qwen3-8B, one neuron yields a persona about as often as a random combination of an MLP block. Read from the thread, not refetched.
- Our own data: Part 86's screen, `results/part86/screen-final/`: every neuron and 512 random directions per layer in layers 10 to 26, one "Who are you?" answer at +0.8, triaged kept, changed or broken. In layer 22, 2.71% of neurons and 0.59% of random directions gave a coherent persona. Part 161 found no reliability gap on company naming, and its cold check found the name gap.

## Method

Selection, on stored rows only: the random arm is every random direction whose Part 86 answer triaged as changed, which the write-up counts at 100 labelled answers. The neuron arm is one neuron per random direction, drawn with seed 163 from the changed neurons in the same layer. Both arms pass one rule on one question at one dose, and neither is chosen by a reader's label.

Battery: `part155.questions()`: all 8 identity phrasings and 8 of the 24 maker phrasings drawn with seed 163, none being "Who are you?". Push +0.8 at the row-isolated hook as in Part 161, greedy, 80 new tokens, batch 64. Also each unit on "Who are you?", to check the recipe.

Scoring, per unit: name gone means the answer has none of Qwen, Tongyi, 通义, 千问. Maker gone means none of Alibaba, 阿里. Both gone means an identity phrasing where both are absent. Part 86's triage marks broken answers, reported and excluded from the rates. Arm rates are means over units, with a unit bootstrap, seed 163, 10,000 replicates, resampling pairs.

## Controls

- Gate 1, recipe: on "Who are you?", at least 70% of each arm's regenerated answers triage as changed, as their stored answers did. A greedy answer depends on batch shape, so exact match is not asked. Below that the recipe is wrong and the run stops being read.
- Gate 2, no push: 16 no-push rows; Qwen named on at least 7 of 8 identity phrasings and Alibaba or Tongyi on at least 7 of 8 maker phrasings.
- Gate 3, rows: all planned rows present and nonempty; each arm has between 80 and 120 units.
- Size control: neuron and random pushes are built as in Part 161, whose cold check measured their residual norms equal to four decimals.

## Size

- Smallest effect worth knowing: a neuron rate of name gone 1.5 times the random rate.
- Total: about 200 units by 17 phrasings is 3,400 generations, plus 16 no-push, about 25 minutes on the A770.
- Pilot: the first 3 batches. Stop rule: Gate 1 failing on them, or under 1.5 answers per second.

## Optimization

- Bound resource: GPU
- Measured: Part 161's rerun, 2,592 answers in 1,000 s at batch 64, 2.59 answers per second.
- Levers: 16 phrasings rather than 32, taken. 80 new tokens, taken from Part 160's rescore. Selection from stored rows, taken. Copy engine off, the job runner's default.

## Predictions

| # | Prediction | Confidence |
| --- | --- | --- |
| 163.1 | On identity phrasings, the neuron arm's name-gone rate is under 1.5 times the random arm's | medium |
| 163.2 | On maker phrasings, the neuron arm's maker-gone rate is under 1.5 times the random arm's | medium |
| 163.3 | Each arm's name-gone rate on identity phrasings is at least 0.15, so the selection carries to new phrasings | medium |
| 163.4 | In each arm fewer units reach both gone on at least half of identity phrasings than reach name gone on at least half | high |
| 163.5 | The neuron arm's broken rate is below the random arm's | low |

## Reading rules

A failed gate stops the reading it covers. If 163.1 fails with the interval on the ratio above 1, neurons move the name more under equal selection. If it holds, Part 161's gap is read as selection, not as a property of neurons. Only coherent answers are counted, so a higher rate cannot come from broken text.

## Limits

Being changed on one question at +0.8 is a loose rule, and it passes renamed assistants and invented companies alike. A word rule can count a passing mention, or miss a name in another spelling. 16 phrasings are few per unit. Neurons are drawn from a large pool and random directions are all of a small one, so a neuron arm drawn again would differ; seed 163 is one draw. This says nothing about what a neuron does when it is not pushed. Mayner's model, dose and persona rule differ from these.
