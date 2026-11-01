# Part 165: do neuron pushes break answers because they are concentrated?

Procedure: v1
Registered 2026-10-31 23:56 CT by Claude (Opus 5.5), before any data. It follows Part 164, whose one solid result was that neuron answers broke 3.5 times as often as random directions of the same write size, 27.0% against 7.8%, in all eight layers. Part 86's unselected screen showed 7.5% against 0.76% in layers 19 to 26.

## Question

Is the gap about the single neuron, or about how concentrated the push is? Between one neuron and a random direction over all neurons there is a ladder: 4, 16, 64, 256, 1,024 neurons. Does breakage fall along it? And does a write vector with no relation to the model's weights behave like the dense random direction?

The MLP's down projection is linear, so a push reaches the model only as the write vector W d. Sparse here therefore means a write vector close to one column of W. This part cannot say why a column differs.

## Prior art and existing data

- Closest published work: Mayner on X (x.com/wmayner/status/2116628249153515740), on Qwen3-8B: small sets of neurons reproduce persona vectors while top directions of a rotated basis do not, but one neuron yields a persona about as often as a random combination of an MLP block. It measures persona steering, not breakage. Read from the thread.
- Our own data: Part 164's breakage on picked units; Part 161, picked random directions as reliable as picked neurons; Part 86's screen.
- We know of no measurement of breakage against the number of neurons in a push at equal write size.

## Method

Model: Qwen3.5-4B, as in Part 164. Layers 19 to 26. Slots: 12 per layer, 96 in all. Each slot draws one permutation of the layer's neurons and one Gaussian vector, from `torch.Generator` on the CPU with seed 165000 + 100 × layer + slot, so all arms are nested and paired by slot. No selection: nothing about a neuron's behaviour is used.

Nine arms per slot, each scaled so that the write norm of W d is 0.8 times the layer's median from `results/part112/medians.json`:

| Arm | Push |
| --- | --- |
| n+ | the permutation's first neuron, positive, Part 86's neuron recipe |
| n± | the same neuron, with the sign of its Gaussian draw |
| k4, k16, k64, k256, k1024 | the first k neurons, with their Gaussian weights |
| full | all neurons, with their Gaussian weights, Part 86's random recipe with a new seed |
| iso | a write vector drawn Gaussian in the residual space, then d from the pseudoinverse of W, so it has no relation to W's columns beyond the solve |

Phrasings: the first 8 of Part 164's 16 maker phrasings, in that part's order. 80 new tokens. Answers are greedy. Within a phrasing the rows are shuffled with seed 165 before batching at 64, so each batch mixes arms and layers. Pushes go on the prompt and the answer, as in Part 164.

Measures, per slot and arm. Broken, the primary measure: Part 86's triage says broken, or the answer holds a three-word sequence three or more times, using triage's own tokenizer, so loops override kept. Part 164 showed triage alone lets loops through. Triage as it stands is reported beside it. Changed: the first 12 words differ from that phrasing's no-push answer. Arm rates are means over slots. Ratios use a paired slot bootstrap, seed 165, 10,000 replicates, keeping zero-denominator draws as infinite and reporting how many there were. Spearman is computed on arm rates against log k, with full taken as k equal to the layer's width.

## Controls

- Gate 1, rows: all planned rows present, unique and nonempty.
- Gate 2, no push: none of the 8 no-push answers is broken under the primary measure.
- Gate 3, size: every direction's write norm is within 1% of its target, measured on the real weights and saved, and the pseudoinverse's residual for iso is under 1% of the write norm.
- Gate 4, the measure has room: n+ has a broken rate of at least 0.03, so a ratio can be read at all.
- Random control: full and iso are random directions at the same write size, paired to the neuron arms by slot.
- A failed gate stops all readings.
- Bars that can fail: 165.1 fails if n+ is under twice full, which Part 164's 3.5 and Part 86's 10 both clear but unselected neurons on these phrasings need not. 165.2 and 165.6 fail on any flat or humped ladder. 165.3 fails if k16 sits nearer n±. 165.4 fails on either side. 165.5 fails if iso breaks as often as full.

## Size

- Smallest effect worth knowing: a ratio of 1.5 between adjacent ends of the ladder, at rates near 0.1. With 96 slots by 8 phrasings each arm has 768 answers.
- Total: 6,920 generations, about 46 minutes on the A770: 9 arms by 96 slots by 8 phrasings, plus 8 no-push.
- Pilot: the first 3 battery batches, timed after the no-push rows. Stop rule: under 1.5 answers per second.

## Optimization

- Bound resource: GPU
- Measured: Part 164, 6,336 answers in 2,370 s at batch 64, 2.67 answers per second.
- Levers: 8 phrasings rather than 16, taken, since slots carry the variance and Part 164 showed breakage varies by unit far more than by phrasing. 80 new tokens, taken. Nested paired arms, taken. Copy engine off, the job runner's default.
- Host work: selection needs no model. Pseudoinverses are computed once per layer on the CPU and cached.

## Predictions

| # | Prediction | Confidence |
| --- | --- | --- |
| 165.1 | n+ breaks at least twice as often as full | high |
| 165.2 | Broken rate falls with k: Spearman at or below -0.8 across n±, k4, k16, k64, k256, k1024, full | medium |
| 165.3 | k16's broken rate is nearer full's than n±'s | low |
| 165.4 | The n+ over n± broken ratio is between 0.67 and 1.5 | medium |
| 165.5 | iso's broken rate is under 0.8 times full's | low |
| 165.6 | Changed rate falls with k: Spearman at or below -0.6 across the same seven arms | medium |

## Reading rules

A failed gate stops all readings. If 165.1 fails, Part 164's gap needs selection, and the ladder is read as description only. If 165.1 and 165.2 hold, breakage tracks how close the write vector is to one column of W, and nothing is said about why. If 165.1 holds and 165.2 fails, the ladder's shape is reported and the difference is not read as gradual. If 165.2 holds and 165.6 fails, breakage falls while the amount of change does not, so concentrated pushes break answers rather than simply moving them more. 165.5 is read only if 165.1 holds.

## Limits

One model, one dose, 8 maker phrasings. Neurons are unselected, so most may do little. Sign and sparsity reach the model only through W d. A Gaussian mix of many columns leans toward W's top singular directions, which is why iso exists, and iso still depends on W through the solve. Triage is one rule with known misses, and the override may count deliberate lists. This says nothing about what a neuron does when it is not pushed.
