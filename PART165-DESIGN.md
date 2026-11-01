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

## Amendment 1, 2026-11-01 00:02 CT, before the run

The rows of phrasing q are shuffled with seed 165 + q rather than one seed 165, so each phrasing gets its own batch mix. Nothing else changes.

## Amendment 2, 2026-11-01 00:59 CT, before the run, on Weaver's review

Correction: the Method says pushes go on the prompt and the answer. They go on the answer only. Part 86's hook leaves the prompt pass untouched, and Part 164 ran that way, as this part does.

The run had already started when this was written, at 00:58, on the registered code, which is right on this point. Added, on Weaver's point that equal write norm need not mean equal effect: a separate effect run after the battery, `part165/effect165.py`, on the same directions, written before any battery result was read. For each slot, arm and phrasing, one row, generating 2 tokens with the same hook, so the second step is the first pushed one. Saved per row: the KL of the plain next-token distribution from the pushed one at that step, and the L2 change of the last layer's output at that step relative to the plain output's norm. Nothing else changes. These are 864 rows per phrasing at batch 64, about 4 minutes in all with the model load.

New prediction: 165.7, n+'s mean relative last-layer change is at least 1.5 times full's (medium). It fails if the arms' changes are about equal. If it holds, equal write norm did not give equal effect at the output, and the broken ratio is not read as a difference between pushes of equal effect; matching on effect becomes the next part. If it fails, Part 164's gap is not explained by size at the output. Arm means of both measures are reported for every arm, beside the broken rates. Part 164's plain model broke 0 of 16 no-push answers, so the floor there was 0.

## Amendment 3, 2026-11-01 01:06 CT, during the battery, on Alden's review, no answer read

Alden read the design and code at 02afd6f and found the timing error Amendment 2 fixed, and these. Nothing about the battery's generation changes.

Delivered dose. Gate 3 checks the intended float32 W d, but the hook casts d to bf16 and adds it to a bf16 activation, so the delivered write can differ, and sparse and dense pushes may differ in that. The effect run therefore also records, per row at the first pushed step, the realized write norm and its cosine with the intended write, on the real dtype and path, at a shared context, and the count of changed channels. If realized writes differ materially between arms, that is reported as a failure of equal dose and the ladder is not read as direction alone.

Geometry, per direction, no generation: the norm of d, the largest absolute cosine between W d and any column of W, and for each layer the numerical rank of W and its smallest singular value. So k is reported as support size, not as closeness to one column, and iso's isotropy is checked rather than assumed.

Scoring. Gate 3 requires exact unique coverage of layer, slot and arm, and finite values. The bootstrap is stratified by layer, which keeps the design's layer weights. Raw counts and rate differences are reported beside each ratio, and a ratio with a zero denominator is called undefined when both rates are zero. The share of primary-broken answers that are short answers under four words is reported per arm.

Reading rules. If 165.1 fails, this sample did not reproduce the gap; that alone does not show the gap needs selection. Changed is a binary indicator of output change, not an amount, so if 165.2 holds and 165.6 fails the reading is: lower breakage without the specified trend in this indicator. Gate 4 is about informativeness, so raw rates are kept whatever it says. A Spearman shows an ordinal trend; the shape is reported beside it.

## Amendment 4, 2026-11-01 01:15 CT, during the battery, on Muse's prior-art review

No answer has been read. Muse (council topic muse-part165-prior-art-2026-11-01, message 3) found no published measurement of degeneration against the concentration of a steering vector at fixed norm. The closest curve is SAIF (He et al., arXiv:2502.11356, read in part by Muse): steering performance over k SAE latents peaks at k = 15, then declines, with task accuracy as the readout, not degeneration. Azizi et al. (arXiv:2610.16084, abstract read by Muse) find a small sharp part of a vector drives a disproportionate share of the steered gap. I have not read either myself yet; both are cited through Muse until I do.

Muse's four traps against this design. Total write norm, not per-unit dose, is what this ladder holds fixed: already registered. Every rung is a random set at its count, so count is not confounded with unit quality: already registered. The curve may not be monotone: 165.2's Spearman stays the registered test, and a Spearman can miss an interior peak, so the ladder shape in `tables.md` is read beside it, as Amendment 3 says. The dense endpoint differs in distribution, not only sparsity, and Malla et al. (arXiv:2609.06951) predict it sends a model to its default answers, so repetition and default answers must be read apart from breakage.

Added for that: `part165/secondary165.py`, descriptive tables with no verdicts, scored after `score165.py`. Per arm: the mean fraction of distinct adjacent word pairs, Malla's repetition measure; and the share of answers whose first sentence holds a refusal opener (sorry, I cannot, I can't). No prediction uses them. They bear on how a broken rate is read: a rising broken rate with a falling pair fraction is repetition, and a rising refusal rate at `full` is default induction, which this part's primary rule does not count as broken.

