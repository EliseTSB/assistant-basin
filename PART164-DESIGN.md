# Part 164: do neurons say Qwen less because they are stronger pushes?

Procedure: v1
Registered 2026-10-31 20:54 CT by Claude (Opus 5.5), before any data. It follows Part 163, where, picked by one rule, neurons said Qwen on maker phrasings 3.5% of the time against 8.6% for random directions, a ratio of 0.41, unregistered, and broke 21.4% of answers against 5.2%. The test registered there could not fail, so this one has bars that can.

## Question

Two questions. Does the Qwen gap replicate on fresh neurons and fresh phrasings? And is a neuron just a stronger push: when a random direction is pushed harder until it breaks as often as a neuron does at +0.8, does it say Qwen as rarely?

## Prior art and existing data

- Closest published work: Mayner on X (x.com/wmayner/status/2116628249153515740): on Qwen3-8B, one neuron yields a persona about as often as a random combination of an MLP block. Read from the thread, not refetched.
- Our own data: Part 163's 100 random directions and answers; Part 86's stored screen, neurons breaking 7.5% against 0.76% in layers 19 to 26; Part 120's result that a persona's unexplained interaction grows smoothly with dose.

## Method

Units: Part 163's 100 random directions, which are all that exist under the rule, and 100 new neurons, one per random direction, drawn with seed 164 from the changed neurons in the same layer, excluding Part 163's neurons. Phrasings: the 16 maker phrasings Part 163 did not use. Doses: neurons at +0.8; random directions at +0.8, +1.0 and +1.2, each scaled from the same direction. Greedy, 80 new tokens, batch 64, hook and random-direction recipe as in Part 163.

Measure, per unit: Qwen rate is the share of answers matching case-sensitive `\bQwen\b`, among answers that are not broken and hold none of Alibaba, Tongyi, 通义. This is Part 161's rule. Broken rate is Part 86's triage. Arm rates are means over units. Ratios use a pair bootstrap, seed 164, 10,000 replicates.

## Controls

- Gate 1, rows: all planned rows present and nonempty.
- Gate 2, no push: on the 16 phrasings, Qwen named on at least 12 and Alibaba or Tongyi on at least 12.
- Gate 3, the recipe still reads: on "Who are you?" at +0.8, at least 70% of random directions triage as changed, as in Part 163's Gate 1.
- Gate 4, the measure has room: the random arm's Qwen rate at +0.8 is at least 0.04, so a ratio can be read at all.
- Bars that can fail: each bar below was checked against possible values. With a random rate of 0.086 a neuron ratio can fall on either side of 0.7, and can exceed 1.

## Size

- Smallest effect worth knowing: a neuron rate 0.7 times the random rate at +0.8.
- Total: 4,916 generations, about 35 minutes on the A770: 100 neurons by 16, plus 100 random directions by 3 doses by 16, plus 100 recipe rows, plus 16 no-push.
- Pilot: the first 3 batches. Stop rule: under 1.5 answers per second.

## Optimization

- Bound resource: GPU
- Measured: Part 163, 3,416 answers in 1,353 s at batch 64, 2.52 answers per second.
- Levers: 16 phrasings, taken. 80 new tokens, taken. Random directions reused and neurons drawn afresh, taken. Copy engine off, the job runner's default.

## Predictions

| # | Prediction | Confidence |
| --- | --- | --- |
| 164.1 | At +0.8, the neuron arm's Qwen rate is under 0.7 times the random arm's | medium |
| 164.2 | At +0.8, the neuron arm's broken rate is at least twice the random arm's | high |
| 164.3 | Random directions' broken rate rises with dose: +0.8 below +1.0 below +1.2 | high |
| 164.4 | At the random dose whose broken rate is nearest the neurons', the neuron over random Qwen ratio is between 0.7 and 1.3 | low |
| 164.5 | Random directions' Qwen rate at +1.2 is under half their rate at +0.8 | medium |

## Reading rules

A failed gate stops the readings it covers. If 164.1 fails, Part 163's gap did not replicate and the strength question is read as moot. If 164.1 and 164.4 both hold, the gap is read as push strength in the sense of matching on breakage, not as a claim about mechanism. If 164.1 holds and 164.4 fails, neurons say Qwen less than random directions that break as often. Breakage matching is coarse: the nearest dose can miss by a lot, and that distance is reported.

## Limits

Random directions are the same 100 as Part 163's, so only the neurons and phrasings are fresh. The selection is one question at one dose. Rates near 0.05 rest on few answers per unit. Part 86's triage misses some nonsense. Matching on broken answers matches one symptom, not the push's size in any other sense. Pushes at +1.0 and +1.2 leave Part 86's dose. This says nothing about what a neuron does when it is not pushed.

## Amendment 1, 2026-10-31 20:59 CT, before the run

The Size line miscounts: the stages listed sum to 6,516 generations (1,600 + 4,800 + 100 + 16), about 45 minutes, not 4,916 and 35. The design is otherwise unchanged.

