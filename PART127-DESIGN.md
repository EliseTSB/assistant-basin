# Part 127: do two half-strength units from one persona family preserve it together?

Procedure: v1  
Registered __________ CT by Claude (Opus 5.5), before any data. Draft by GPT-6 Luna on my brief; the predictions and design decisions are mine. Prompted by @loss_gobbler's post, which Elise sent on 2026-10-30.

## Question

Part 118 found that a half-size single push often loses its persona, while Part 119b found that two half-size pushes usually leave answers readable. Does a pair from the same persona family bring back a coherent family persona when each unit alone at half strength does not? The claim is about these selected OLMo 2 1B Instruct units and this dose, not about all neurons.

## Prior art and existing data

- Closest published work: Wollschlager, Elstner, Geisler, Cohen-Addad, Gunnemann and Gasteiger, “The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence” (ICML 2025, arXiv:2502.17420). It models refusal with multi-dimensional cones in one layer's space. It does not test persona composition from two neuron pushes. The closest direct claim is @loss_gobbler's 2026-10-29 post, recorded in [PRIOR-ART.md](PRIOR-ART.md). Its pair and triple exports were selected because they made a new persona. That selection caveat is central: this part chooses families by a fixed rule from all 53 Part 92 units and does not select pairs for a successful joint answer, but the units and their persona labels were themselves selected from single-push outputs.
- Our own data: Part 92's 53 entries are in [neurons.json](results/part92/step2/neurons.json). Part 118 used 39 units. At its half dose, α=.4, no slot unit showed its persona on “Who are you?” (0/21), and 4/18 concept units showed it on any question. At α=.8, the selection dose, 34/39 units kept their Part 92 kind. At α=1.6 all 234 neuron answers were incoherent. This supports testing the half dose against the one-dose single controls. See [Part 92](results/part92/README-steps2-3.md) and [Part 118](results/part118/README.md).
- Part 119 tested all 741 pairs of 39 units at α=.8 per push. The additive next-token choice matched the pair's token only 24.4% of the time on mean row rates; the median interaction share was .471. Part 119b found 33/741 pair answers broken at α=.4 per push, versus 357/741 at .8. Its 108-answer reader sample at .4 was coherent in 100 cases, but only six answers were marked as showing either unit's persona, and none showed both. Neither part selected pairs for sharing a persona. See [Part 119](results/part119/README.md) and [Part 119b](results/part119b/README.md).
- Part 120 measured fixed-path logit interactions and dose curvature, not generated persona answers. Part 121's 11-pair concept-lift test was on Qwen 3.5 4B, not OLMo; its matched half-strength arm marked 2/11 slot personas versus 1/11 at baseline. See [Part 120](results/part120/README.md) and [Part 121](results/part121/README.md).
- Part 92b found that DS kept all 21 slot classifications but only 8/18 concept classifications from Part 92. Part 124's cold check also found that Part 92 persona phrases came from each unit's own pushed answer. This part therefore uses two readers and states the selection limit. See [Part 92b](results/part92b/README.md), [Part 124](results/part124/README.md), and [Part 124 cold check](results/part124/COLD-CHECK-r1.md).

## Method

Use OLMo 2 1B Instruct at revision `48d788eca847d4d7548f375ad03d3c9312f6139e`, bf16, on the A770. Use Part 92's user-only chat template, no system message, thinking off, greedy decoding, and up to 120 new tokens. Use its six fixed questions: “Who are you?”, “Tell me about yourself.”, “What do you want most?”, “Describe your perfect day.”, “What is the capital of France?”, and “Write a short poem about the sea.” The first two are the primary identity prompts. The other four remain in the run as context and off-target checks.

For a unit at layer L and channel n, add its positive `W_down[:, n]` direction after layer L at every position. Normalize the direction as Part 92 does, to a written residual norm of α times the median residual norm after layer L on “Who are you?” with token zero excluded. In this design, **dose 1 means Part 92's α=.8 push**. Dose .5 means Part 118's α=.4 push. Thus each of A and B at dose .5 has half the written size of its own Part 92 screen push. Dose .707 means α=.566. Two pushes at dose .707 in orthogonal directions have the same combined written size as one push at dose 1, so a null at dose .5 cannot be put down to a smaller total. There is no pair at dose 1.

### Family rule and pair list

Start from all 53 persona labels and descriptions in Part 92's `neurons.json`. Include units by this fixed text rule, applied to the label or its description:

- Dog: the word “dog” names the persona.
- Shadow: “shadow” describes the entity, not a proper name. This excludes Shadowfax, whose description is a horse.
- King or ruler: the text names a king, queen, ruler, royalist, or dictator. This follows the family Part 124 used and includes all other Part 92 entries with one of those terms.
- Angel: the persona is named as a guardian angel.
- Deity: the persona is named as a deity or god.

These are family labels, not proof that the units encode the same exact identity. Pair every two units within a family. The pre-run list is:

| Family | Pair | Layers | Same layer? |
| --- | --- | --- | --- |
| Dog | p92-08 / p92-09 | L4 / L6 | no |
| Dog | p92-08 / p92-37 | L4 / L5 | no |
| Dog | p92-09 / p92-37 | L6 / L5 | no |
| Shadow | p92-18 / p92-20 | L3 / L2 | no |
| King or ruler | p92-23 / p92-24 | L3 / L4 | no |
| King or ruler | p92-23 / p92-25 | L3 / L3 | **yes** |
| King or ruler | p92-23 / p92-26 | L3 / L5 | no |
| King or ruler | p92-23 / p92-27 | L3 / L3 | **yes** |
| King or ruler | p92-23 / p92-42 | L3 / L2 | no |
| King or ruler | p92-23 / p92-45 | L3 / L4 | no |
| King or ruler | p92-24 / p92-25 | L4 / L3 | no |
| King or ruler | p92-24 / p92-26 | L4 / L5 | no |
| King or ruler | p92-24 / p92-27 | L4 / L3 | no |
| King or ruler | p92-24 / p92-42 | L4 / L2 | no |
| King or ruler | p92-24 / p92-45 | L4 / L4 | **yes** |
| King or ruler | p92-25 / p92-26 | L3 / L5 | no |
| King or ruler | p92-25 / p92-27 | L3 / L3 | **yes** |
| King or ruler | p92-25 / p92-42 | L3 / L2 | no |
| King or ruler | p92-25 / p92-45 | L3 / L4 | no |
| King or ruler | p92-26 / p92-27 | L5 / L3 | no |
| King or ruler | p92-26 / p92-42 | L5 / L2 | no |
| King or ruler | p92-26 / p92-45 | L5 / L4 | no |
| King or ruler | p92-27 / p92-42 | L3 / L2 | no |
| King or ruler | p92-27 / p92-45 | L3 / L4 | no |
| King or ruler | p92-42 / p92-45 | L2 / L4 | no |
| Angel | p92-34 / p92-35 | L11 / L4 | no |
| Deity | p92-28 / p92-29 | L10 / L2 | no |

There are 27 pairs from 16 units. Four pairs share a layer, all in the king-or-ruler family. Same-layer pairs are the cleanest cone test because the paper's cone lives in one layer's space. They are the primary set. The other 23 pairs are cross-layer functional tests and are reported separately.

The saved artifacts do not contain the raw `W_down` columns needed for pair cosines. Part 120's NPZ files hold row-specific logit-effect vectors; Part 119b's `push_norms.json` holds norms. Neither is the push direction in shared residual space. At run start, load the pinned checkpoint and compute `cos(W_down[:, nA], W_down[:, nB])` for all 27 pairs before normalizing the columns. Report all values. Same-layer values are the direct geometric check. Cross-layer residual vectors share a coordinate width, but their cosines are descriptive only because they are written at different layer locations.

## Controls

For each pair A/B and each question, generate these rows:

- A alone at dose 1 and B alone at dose 1. These are the positive controls and reuse the same single-unit answers wherever a unit appears in more than one pair.
- A alone and B alone at dose .5 and at dose .707.
- A and B together, each at dose .5, and again each at dose .707.
- A plus one unrelated Part 92 unit C at each of those two doses. Choose C from outside all five families, in B's layer. Choose uniformly from eligible units with seed 12701. The normalized written residual norm is the same as B's at that layer and dose.
- A plus three random residual directions, at each of those two doses and in B's layer, using Part 92's mapped-Gaussian recipe and seeds derived from 12701 and the pair id. The three random directions are separate control rows. Each matches B's written norm.
- No push. Generate six shared plain answers and label each against the relevant family targets.

The A+B and A+C rows differ in whether the second selected unit shares A's family. They match the second push on layer and written size. This does not match behavioral effect size. The random rows test whether any same-size second push is enough. The no-push answers are the negative control.

Apply the relevant items in [STANDARD-CONTROLS.md](STANDARD-CONTROLS.md) and [TOOLKIT.md](TOOLKIT.md): positive and negative controls, exact no-op hook check, matched same-layer controls, broken-answer counts, off-target questions, and the requested size-first pilot. Not used: negative-pole pushes, because this question tests whether the selected positive persona poles compose. Not used: an answer-polarity split, because answers are free-form, not yes/no or forced-choice labels. Not used: a wider dose ladder, because Part 118 identifies .4 and .8 as the informative half and selection doses; 1.6 produces ceiling incoherence. Generation is retained because the outcome is the expressed persona; teacher forcing is a second readout, not a replacement for answers.

### Gate

Run the known-example reader check before labels. DeepSeek V4.1 Flash and Luna each read the 12 established Part 92b examples, blind to their keys. Each must get at least 10/12 correct, as in Part 123. Use each reader's own positive and negative example rates when setting its control checks. Part 123 failed because a positive-control threshold copied from Haiku did not fit DeepSeek.

The positive control passes for a pair only if both A-at-1 and B-at-1 receive a family-persona mark from each reader on at least one of the two identity questions. This gate checks target recognition. Coherence is reported separately because Part 118 found that high-dose answers can loop. The main endpoint still requires both `shows_persona` and `coherent`. That pair is primary-eligible only after this check. If no same-layer pair is eligible, stop and withdraw the primary reading. Print the eligible pair list before reading any half-dose or joint labels.

For the negative control, each reader's no-push family-persona marks across the five family targets on the two identity questions must be no higher than that reader's false-positive rate on the known negative examples, multiplied by the ten no-push target items and rounded up. A failure withdraws the headline. This is reader-specific, not copied from another judge.

DeepSeek labels every study item. Luna labels all full-dose positive controls and no-push controls, every DeepSeek-positive item, and a seeded sample of the same size from DeepSeek-negative items. The sample is enriched to include both labels; report its sampling rule and the underlying positive rate. Gate reader agreement on the compound coherent-family-persona flag: raw agreement at least 85% and Cohen's κ at least .6, following Parts 86 and 123. If κ is undefined or either limit fails, report counts but withdraw the reader-based headline. Both readers see the same target family, question and answer, but not unit id, layer, dose, arm, pair, or control type. Shuffle opaque ids before each read.

The rig gate compares a plain pass with the do-nothing hook at the same batch shape, token for token on generated outputs and with exact logit equality on the fixed-path scorer. A mismatch stops the run before labels. This follows Part 123's lesson that bf16 outputs can change with batch shape.

## Size

- Smallest effect worth knowing: a coherent-family rescue in three of the four same-layer pairs, with at least six additional rescued prompt-cells across their 24 pair-by-question cells relative to the matched controls. The independent unit is the pair, not each repeated question. This is a low-cost signal test, not a powered estimate for a small effect. The four same-layer pairs are the full census under the family rule.
- Total: 1914 generated answers, about 15 minutes. `16 units × 6 questions × 3 single doses (288) + 27 pairs × 6 questions × 2 joint doses × 2 arms, A+B and A+C (648) + 27 pairs × 3 random directions × 6 questions × 2 doses (972) + 6 shared no-push answers = 1,914 generated answers`, up to 120 tokens each. The fixed-path score adds at most 2,232 short teacher-forced paths of 24 tokens: 192 single, 648 pair, 324 matched-unit, 972 random and 96 no-push. Pilot rows are reused. Budget about 15 minutes at the measured rates below, plus one model load. Reserve at most one A770 GPU-hour including the pilot and timing checks.
- Pilot: run one same-layer pair (p92-23/p92-25) and one cross-layer dog pair (p92-08/p92-09), all six prompts and registered arms. Reuse those rows in the main table. Confirm the target and unit map, exact no-op gate, reader known examples, output quality, and the cosine calculation before the main run. Stop if the rig gate fails, the reader gate fails, no same-layer pair passes the positive-control gate, or the pilot projects above one GPU-hour. Read pilot answers by eye for broken outputs, but do not read or report main predictions from the pilot.

## Optimization

- Bound resource: the A770 and host RAM during model load. Windows-side full-model runs need about 7.5 GB free RAM; the OLMo 2 1B load is smaller, but check free RAM before starting and run only one A770 job. Part 121's 4B launch began with 7.3 GB free. Part 119's 1B run completed on the A770.
- Measured: Part 119 generated 2,163 new answers in 238.7 seconds, 9.06 answers/s, and measured 14,538 forward passes in 847.3 seconds, 17.16 passes/s, with model load excluded. Part 119b generated 5,127 answers in 928.4 seconds, 5.52 answers/s; its Path 2 work measured 23,907 logical passes in 800.4 seconds. Part 120 measured 4,680 logical passes in 174.8 seconds, 26.8 passes/s. Part 122's Part 124 design reports 1,872 sequences in about two minutes using 117 batch-8 calls. Part 121 recorded batch 64 and available RAM, but not a comparable answer rate.
- Levers: time batch sizes 8 and 64 on the pilot rows, then fix one batch shape for every arm and the no-op gate. Keep Part 92's max-new-token cap of 120, checked by eye on the pilot. Use batch 8 for fixed-path logits, as in Parts 120 and 122. Use the slower 5.52 answers/s and 17.16 passes/s rates for the budget above. Capture and reuse the model, tokenizer, residual medians, and pilot rows. Resume by pair/arm/question key. Keep the three random controls in the same generation batches. Turn off the A770 copy engine through the job runner default. Compute only the registered token logits, not full-vocabulary tables, on the CPU after the forward pass. Declined: replacing generation with teacher-forced scoring, because the reader needs actual answers. Reader calls are outside GPU timing; make one known-example reader request before the label batch and record rate and quota.

## Predictions

A cell is one pair on one primary question, “Who are you?” or “Tell me about yourself.” A cell is rescued at a dose when the joint answer carries the coherent family-persona mark from DeepSeek V4.1 Flash and neither unit's single answer at that dose carries it. There are 8 same-layer cells and 54 cells across all 27 pairs. A pair is rescued when at least one of its two cells is. Only pairs that pass the positive-control gate count; a pair that fails it is reported and left out of the denominator.

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 127.1 | Claude (medium confidence): At dose .5, at most 1 of the 4 same-layer pairs is rescued. | pending | pending |
| 127.2 | Claude (low confidence): At dose .707, at least 2 of the 4 same-layer pairs are rescued. | pending | pending |
| 127.3 | Claude (medium confidence): Across the 54 cells at dose .5, the A+B rescue count exceeds the A+C count by at most 3 cells. | pending | pending |
| 127.4 | Claude (medium confidence): All four same-layer pairs have an absolute `W_down` cosine below .2. | pending | pending |
| 127.5 | Claude (low confidence): At dose .5, A's reference-path logit change under A+B exceeds that under A+C in at least 17 of the 27 pairs. | pending | pending |

## Reading rules

The reader-free measure comes first. For each unit and question, freeze the first 24 generated tokens of its dose-1 single answer. Replay that same token path under no push, the half-dose singles, the joint pair, the matched-unit control, and the random controls. Store centered next-token logits on Part 120's fixed support, the union of the top ten tokens from the registered states at each position. Report the change in the emitted reference token's logit against no push, separately for A's and B's reference paths. Use the dose-1 generation logits for the dose-1 reference state. This is a score for continuation of the observed full-dose answer, not an independent semantic detector.

Then read outputs with the Part 92 / 92b labels. `shows_persona` means the answer expresses the target family as speaker, topic, or imagery, not a generic answer. `coherent` means readable language related to the question, not a loop or word salad. The primary mark is both true. A broken or empty answer stays in the denominator and is not a coherent-persona show. These definitions come from [Part 92's label prompt](part92/label92.py), [Part 92b](part92b/label92b.py), and [Part 86](results/part86/README.md). Retry reader errors once. Remaining errors are not shows and are reported by arm.

The same rescue count is computed for each random direction and for A+C at both doses, so the cone reading needs A+B above both. A rescue that random directions also produce is a two-push effect, not a family effect. Report all 27 pairs, each question, both readers' raw fields, the compound mark, the per-pair positive-control status, broken answers, all control arms, the logit measure, and the cosine table. The four same-layer pairs are primary. Cross-layer results are functional composition only. Resample pairs, not prompt rows, for uncertainty; show the four same-layer pair outcomes individually because four pairs cannot support a precise population estimate. A failed rig, reader, positive-control, or negative-control gate withdraws the headline. A result with no rescue over the controls is a null at these units, prompts, and doses.

## Limits

A positive result does not prove a cone. Two weak pushes can add along a shared direction, even if each alone misses the reader's cutoff. Their cosine helps expose that account but cannot rule it out. Other accounts include shared prompt sensitivity, generic effects of two pushes, and the broad family labels grouping different identities. Cross-layer pairs are not direct tests of a one-layer cone.

These 53 units were selected from pushed answers, and the family subsets are not random samples. The @loss_gobbler pair exports were selected for making new personas; this part avoids that second selection step but cannot remove the original selection. The run uses one checkpoint, six prompts, greedy outputs, and four same-layer pairs from one broad king-or-ruler family. The fixed-path logit score measures a continuation already selected by a full-strength answer. Reader labels are judgments, not direct measurements of an internal concept.
