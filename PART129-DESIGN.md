# Part 129: neuron by neuron, does post-training change what Qwen3-8B computes?

Procedure: v1
Registered 2026-10-30 19:26 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-30 19:12 CT, Discord message 1565881287409934448:
"keep the research going. Lots to reassess now that there's new tools!" Her focus, 2026-10-27: following neurons across
training stages. The commit that adds this file is the registration.

## Question

Neuron Lens lists as an open question "the same catalog for Qwen3-8B-Base, as a neuron-by-neuron diff of what post-training
changed". Part 91 found that persona neurons' weights barely move (median down-column cosine 0.9968). Weights are not
function: small changes in early layers can reach every later neuron. Run both models on the same tokens and correlate
each neuron's activation between them, position by position. Which neurons still compute the same thing, on which text,
and are persona neurons among the ones that changed?


## Amendment (20:42 CT 2026-10-30, after the first pilot captures, before any correlation was read)

On 24 GB cards batch 32 runs out of memory (Qwen3-8B in bf16 takes about 16 GB of each card), and batch 64 was not tried. The
pilot measured 3.80 documents/s at batch 16, so 16,000 documents project to about 70 minutes, under the 90-minute rule, and
the main stage runs all of them at batch 16. Gate 2 runs at batch 8 and 16, in place of 16 and 64. Gate 1's pilot alignment
passed, 300 of 300 checks within 5%. The first attempts also fixed the weight cosines, which now run on the CPU.

## Amendment (21:41 CT 2026-10-30, during the main run, before any correlation was read)

Rented credit would run out before the run ended, with the chat slice last in the order. By the Size section's rule, the same share
comes from each slice: the run stopped at its 7,500-document checkpoint, all in the web slice, and resumed on the first 75% of each
slice, about 12,000 documents (web 7,510 lines, zh, code and chat 1,500 each). Documents already summed stay. Part 132 is not run here,
because Elise stopped further rental. The Size section's 16,000 is replaced by this count, and the score reports the documents read.

## Prior art and existing data

- Closest published work: Neuron Lens (stdlogits/neuron-lens, 2026-10-30) gives per-pole activations for Qwen3-8B only.
  H-Neurons (arXiv 2512.01797) carried instruct-trained probes to base models and measured weight cosines, without a
  neuron-by-neuron activation comparison. A search on 2026-10-30 found no direct comparison across base and chat models.
- Our own data: results/part91 (weights, personas in Base); Part 39 (210 "talking to the AI" units, 205 already in Base,
  another model); Parts 104 to 108 (what moves when an OLMo stage's weights are swapped); Part 110 (Qwen maker neurons'
  weights).

## Amendment (22:37 CT 2026-10-30, after Gates 1 and 3 failed and after every prediction's summary number had been seen)

This amendment is post hoc. Every prediction's summary was printed before it was written, so nothing read under it counts as a
confirmed prediction. The registered verdict stands: Gates 1 and 3 failed, and by the registered rule the reading stopped.

Gate 1 measured 364 checks, not 500, because the 21:41 cut removed the documents holding the rest. Of those 364, 363 were within
5%. That is 99.7% against the required 90%, on fewer checks than registered. The registered count was not met.

Gate 3 failed at L2 only, in all five groups (0.052 to 0.072 against 0.05). No other layer exceeds 0.040. L2's web median r is 0.939,
below L1 (0.968) and L3 (0.957), so r there is not high because of a shared mode, which is what the reading rule's withdrawal names.
The cause is not established. In the catalog, about a third of poles at L1 to L3 have their top firing on ten positions; 3,638 of
L2's 24,576 poles peak on one token, "You are on page 1⟦of⟧".

Under this amendment, part129/score129.py is rerun with --drop-layers 2 into results/part129/score129-amended.json. L2 leaves every
number. README reports the verdicts of both runs side by side, marks all of them post hoc, and PREDICTIONS.md says the same.

## Erratum (22:58 CT 2026-10-30, after a cold check of the README, correcting the two amendments above)

The 21:41 amendment says the run stopped at its 7,500-document checkpoint. It stopped at 8,000. Token counts from the catalog's
docs.parquet match the saved sums exactly: web is lines 0 to 7,999 (1,891,798 positions), zh is lines 490 to 1,499 (236,409), code and
chat are their first 1,500 lines. Resuming at document 8,000 on a corpus cut to 7,510 web lines skipped the first 490 zh lines. So the
slices are not an equal share: 8,000 web, 1,010 zh, 1,500 code, 1,500 chat, 12,009 documents after the length filter.

The 22:37 amendment says the cut removed the documents holding the rest of Gate 1's checks. The 136 unmeasured checks are all the zh
checks, in lines 0 to 99, which the skip removed. Gate 1 has no zh check on the main run; the pilot passed 93 zh checks, 300 of 300 in
all.

The 22:37 amendment also says L2's r is not high because of a shared mode. Its evidence does not show that: a lower r does not rule
out a shared mode, and Gate 3 is what measures one. The README says the withdrawal clause stands open at L2.

## Method

Models: Qwen/Qwen3-8B at b968826d9c46dd6066d109eabc6255188de91218 and Qwen/Qwen3-8B-Base at
49e3418fbbbca6ecbdf9608b4d22e5a407081db4 (Part 91's revisions), bf16, one per GPU on a rented machine with two RTX 3090s.
Tokens: Neuron Lens' activation corpus, built by their scripts/05_act_corpus.py at repo commit 07fdf45, tokenized as their
06_activations.py does, up to 256 tokens: the first quarter of each slice, 10,000 dclm, 2,000 zh, 2,000 code and 2,000 chat
(UltraChat rendered with Qwen3-8B's chat template, so both models read identical tokens). The hook sits at each layer's
down_proj input, in lock-step on both models: x is Qwen3-8B's activation, y Base's. Position 0 (the attention sink) and
padding are dropped. Groups: web (dclm), zh, code, chat-user (tokens in system and user turns) and chat-assistant (tokens in
assistant turns). Per neuron and group: n and the sums of x, y, x², y² and xy, in float64, giving r, the slope and both
variances. Per pole, the same on the rectified activations (max(±x, 0) against max(±y, 0)). Null: the same sums for x_n
against y of a fixed derangement within the layer (seed 129). Weights: per neuron, cosines of the down column and the gate
and up rows between the two checkpoints. No pushes, no generation.

## Controls

Applied from STANDARD-CONTROLS.md: both poles; a matched random control, the derangement and layer-matched poles. Not used:
do-nothing hook, because the hook only reads; broken-answer count and readers, because no text is generated; dose ladder,
because nothing is pushed.

Gate 1, alignment: for 500 random poles whose published top firing falls inside our documents, our Qwen3-8B activation at
that document and position matches the catalog's top_val within 5% for at least 90% of them. It fails if our tokens or
document order differ from theirs.
Gate 2, noise floor: in the pilot, Qwen3-8B against itself on both GPUs at different batch sizes gives a median r of at
least 0.999 at every layer. A lower value is the floor for every later number.
Gate 3, null: the median absolute r under the derangement is at most 0.05 at every layer.
A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a layer's median r moving by 0.02, and a persona-against-matched gap of 0.03. With 34,220
  library rows and about 3.8 million positions, sampling error is far below that; the limit is noise floor, not n.
- Total: 16,000 documents, two forward passes each, about 45 minutes on two RTX 3090s at about $0.27 an hour.
- Pilot: 400 documents, measuring documents per second at batch 16, 32 and 64. Stopping rule: if the main run projects over
  90 minutes, take the same share from each slice to fit 75 minutes, recorded as an amendment before the run.

## Optimization

- Bound resource: GPU
- Measured: Neuron Lens' own activation pass is 63,990 documents in about 70 minutes on one RTX 4090, 15 documents/s; our pilot times batch 16, 32 and 64 on real rows before the main stage.
- Levers: batch size taken, timed in the pilot; generation not needed; sums vectorized on the GPU, taken; resume from saved sums after each 500 documents, taken; one layer held at a time through a size-1 queue, taken; A770 copy engine not applicable; rented GPUs taken, because two 8B models do not fit on the A770.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 129.1 | On web text, the median r across all neurons is at least 0.85, and each layer's median is at least 0.75. | medium | Claude |
| 129.2 | The all-neuron median r on chat-assistant positions is at least 0.03 below the web median. | medium | Claude |
| 129.3 | On chat-assistant positions, the median of the L30-35 layer medians is at least 0.05 below that of L0-8. | medium | Claude |
| 129.4 | At 30 or more of 36 layers, the absolute Spearman correlation between 1 − r on chat-assistant positions and 1 − down-column cosine is at most 0.3. | medium | Claude |
| 129.5 | Across layers, library persona poles' median rectified r is within 0.03 of layer-matched non-persona poles, on both web and chat-assistant positions. | medium | Claude |
| 129.6 | Part 91's once poles have a chat-assistant rectified r at least 0.05 below its carrier poles. | low | Claude |
| 129.7 | Among the 442 neurons with the lowest chat-assistant r, catalog labels mention the chat register (regex in part129/score129.py) at least twice as often as among layer-matched random neurons. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 129 (registered 2026-10-30, PART129-DESIGN.md): neuron by neuron, does post-training change what Qwen3-8B computes?".

## Reading rules

part129/score129.py computes every number from the saved sums and the pinned catalog. Neurons with fewer than 100 positions
past a quarter of act_top in a group are left out of that group's r and counted. Library persona poles are rows of category
persona in the saved library (neuronpersonas.com, fetched 2026-10-30). Matched poles are drawn per layer from poles with no
persona row, at a similar rate (seed 129). If the noise floor exceeds the effect in 129.2 or 129.3, those predictions are
reported as unreadable. A result where r is high only because a shared mode dominates, shown by Gate 3 failing, withdraws
the headline reading.

## Limits

Activation correlation shows whether a neuron fires in the same places, not that its effect on the output is unchanged; the
down column moves too. A neuron can keep its firing and change partners. Chat text is rendered in Qwen3-8B's template, which
Base was not trained on, so a change on chat positions can also reflect Base's weaker model of that format. One pair of
checkpoints; the intermediate post-training stages are not published for Qwen.
