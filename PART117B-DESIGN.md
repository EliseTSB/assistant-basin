# Part 117b: why does a concept neuron pushed negative erase a slot neuron's self-description in Qwen3.5-4B?

Procedure: v1
Registered 2026-10-29 21:23 CT by Claude (Opus 5.5), before any data; the commit that adds this file is the registration.
Elise asked on 2026-10-29 for all five of the concept and slot questions and the interactions (Discord message
1565527967734571120: "I like all those questions. Im also really curious about further testing the interactions of slot
and concept neurons."). Part 117 answered the first round of interactions and left one cell without a control.
Reviewed before registration: a Luna 6 cold critique of this design and its code (results/part117b/design-review/
CRITIQUE.md, 8 findings, all folded) and Part 117's cold check. Alden was asked (council topic
alden-research-review-2026-10-29, posts 9 and 10) and had not answered at registration.

## Question

In Part 117, with a matched concept neuron at -1 beside its slot neuron at +1, the slot's self-description on "Who are
you?" was gone in 11 of 11 pairs (the slot alone shows it in 11 of 11), and the model described itself as a generic
assistant with an invented name ("Qianro", "Kushelle", "Cedar"). Part 117 had no control for that cell. Does the
erasure need the shared theme, does any concept neuron do it, or does any second push of that size do it? And what does
a concept neuron pushed to -1 do on its own?

## Prior art and existing data

- Closest published work: van der Weij, Poesio and Schoots (2024, arXiv:2403.05767, "Extending Activation Steering to
  Broad Skills and Multiple Behaviours") found that combining steering vectors for several behaviours into one vector
  was largely unsuccessful, while adding separate vectors at different places at once steered two behaviours. Activation
  addition (Turner et al., 2023) and contrastive activation addition (Rimsky et al., 2024) add vectors built from
  prompts. None found asks whether pushing one single unit erases another unit's effect, or controls that with a
  size-matched random push. Here two single neurons are pushed at their own layers.
- Our own data: Part 117 (results/part117/score.json and labels.jsonl). Every concept push tried beside the slot erased
  the slot's persona on "Who are you?": matched concept +1 (slot persona shown in 3 of 11 pairs), unrelated concept +1
  (0 of 11), matched concept -1 (0 of 11). At +1 the concept's persona took the slot's place; at -1 nothing did. So for
  the + sign the shared theme was already not needed; the - sign without the theme, and a random push of that size, are
  untested. Written vectors (part117b/cos117b.py, results/part117b/write-cosines.json, weights only): the slot's written
  vector is nearly orthogonal to its matched concept's (median cosine -0.0063 over the 11 pairs), its unrelated
  concept's (0.0005) and the random directions' below (the largest absolute value among all 55 is 0.06), so the concept
  push does not simply subtract the slot's vector. Parts 104 and 105 (OLMo 2 1B): a single greedy self-description
  shifted under any attention change of a given size, which is why a size-matched random push is the main control here.
  Part 109c (OLMo 2 1B dog pair): concept up with slot down gave no dog at all. In Part 117 the "Who are you?" answer
  lost "Qwen" in nearly every pushed condition, random-direction pairs included (1 of 11 kept it; the unpushed answer
  opens "I am **Qwen3.5**"), so the invented names are not specific to concept -1 (found by Part 117's cold check).

## Method

Qwen/Qwen3.5-4B at revision 851bf6e8, bf16, the A770, loaded straight to the device (Part 115b's loader); chat
template, no system prompt, thinking off; Part 92's six questions (Who are you?, Tell me about yourself., What do you
want most?, Describe your perfect day., What is the capital of France?, Write a short poem about the sea.); greedy, 120
new tokens, batch 64. Push: Part 112's (0.8 x the layer's median residual norm, results/part112/medians.json, added to
the down_proj input on generated tokens only) times a multiplier; components in one row add. The push and generation
code is Part 117's (part117b/rows117b.py reuses pairs117.pushes_for's arithmetic and battery112.generate; a test checks
that the cells shared with Part 117 get identical pushes). Units: Part 117's 11 pairs (results/part117/pairs.json).

Rows, per pair: slot +1; slot +0.85; slot +0.5; matched concept -1 with slot +1; matched concept -0.5 with slot +1; unrelated
concept -1 with slot +1; three random directions at -1 with slot +1, each written at the matched concept's push size in
its layer (index 0 is Part 112's paired random direction for that concept neuron, seed 112000 + 10000 x layer + neuron;
indices 1 and 2 add 1,000,000 and 2,000,000 to the seed); three neurons from the matched concept's layer that are not
among Part 112's 465 persona neurons, at -1 with Part 112's push size beside slot +1 (part117b/pick117b.py, seeded,
results/part117b/control-neurons.json, picked before any data). Plus each of the 20 distinct concept neurons (9 matched,
11 unrelated) alone at -1, and one unpushed row: 153 rows, 918 answers. Before the rows, a do-nothing check: each
question's greedy answer (40 new tokens, batch 1) with an all-zero push hook and with no hook installed at all.

Labels: Part 92's blind Haiku 5.5 reader (part92/label92.py, two passes, the second retrying errors) labels each answer
against each persona neuron its row pushes (slot, matched concept or unrelated concept; random directions and the
non-persona control neurons have none), giving shows_persona, self_claim and coherent; the unpushed row's answers are
labelled against all 31 personas. 1,296 labels. A second reader (Opus 5.5, the same prompt) labels a seeded 10%
sample (part117b/second117b.py); agreement on shows_persona is reported. String readout, by script:
whether the "Who are you?" answer contains "qwen" in any case.

## Controls

- Matched random controls, several: three random directions per pair, written at the matched concept's push size in
  its layer and pushed beside the slot exactly as the concept is (the property under test is that the second push is a
  concept neuron).
- Non-persona neuron controls: three real neurons per pair from the matched concept's layer that Part 112's atlas did
  not list as persona neurons, pushed the same way. A random direction may be one the model barely reads; a real
  neuron's write direction is one it learned to write, so this separates "a concept neuron" from "any real neuron".
  Both controls are matched on written norm and layer only, not on how much the model reads the direction, and
  "non-persona" means only "outside Part 112's 465-unit persona list".
- Unrelated concept: the same slot with a concept neuron of another theme (the property under test is the shared
  theme).
- Dose: matched concept -0.5 beside slot +1; slot +0.85 and +0.5 alone. A second push of the same size grows the
  residual stream, so after normalization the slot's share shrinks, by about 15% if the pushes and the residual are
  mutually orthogonal. That 15% is a heuristic, not a measurement; slot +0.85 is the dose matched to it, and slot +0.5
  extends the ladder.
- Both directions: the concept's +1 cells are Part 117's; this part runs the -1 side.
- Concept alone at -1: what the negative push does without the slot.
- Do-nothing hook: the zero-push hook against no hook, before the rows (above); gated.
- Unpushed row: base rates. It runs with the all-zero hook in the batches, and its six answers are compared with Part
  117's unpushed answers (reported, not gated).
- Broken answers: the reader's coherent label on "Who are you?".
- Not used: an activity-matched null, because nothing is zeroed; poles and the answer-polarity split, because the
  readout is whether a persona shows, not a choice between two answers; controls matched on downstream effect, because
  this part asks first whether size alone does it (a follow-up if the real-neuron and random controls disagree).

Gate: before any prediction is read, (1) the unpushed row must show none of the 31 personas in more than 2 of its 6
answers (else that unit is uninterpretable and its pairs are left out of the counts that use it); (2) the slot alone at
+1 must show its persona on "Who are you?" in at least 9 of the 11 pairs (Part 117: 11 of 11; else this run's pushes are
not Part 117's and nothing is read); (3) the unpushed "Who are you?" answer must contain "qwen" (else the string readout
means nothing); (4) the zero-push hook's six answers must equal the no-hook answers (else the hook is not inert and
nothing is read). A unit whose unlabelled baseline answers could carry it over the 2-of-6 line counts as failing (1).

## Size

- Smallest effect worth knowing: the slot persona on "Who are you?" in most pairs (about 8 of 11) in one condition
  against about none in its comparison. The unit of analysis is the pair (n = 11, one yes or no per pair), so
  differences of one or two pairs are noise; the random control has three rows per pair (33) so its rate does not rest
  on one draw per pair.
- Total: 918 greedy answers of up to 120 tokens plus 12 do-nothing answers of 40 (about 8.5 minutes on the A770 at
  Part 117's measured rate), 1,296 Haiku labels and about 130 Opus labels.
- Pilot: not needed: under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: the A770 for the answers (shared with Elise's game tonight); the API for the labels.
- Measured: Part 117, the same code, questions and batch size: 732 answers in 6 min 27 s including the model load
  (about 1.9 answers/s).
- Levers: batch 64 (taken: Part 117's batch size and prompt padding; the companions in each batch differ from Part
  117's, so its shared cells are not replays; timing two or three sizes declined, because the run is about 8.5 minutes
  and a timing pass would cost a model load of its own); 120 new tokens (taken: Part
  117's length, so its cells compare; the readouts use the opening of "Who are you?" and the coherent label, and a
  shorter cap would need the reader's agreement on cut answers measured first); rows leaving the batch when they finish
  (declined: not built in this code); loading straight to the device (taken, Part
  115b's loader, because Windows' free memory is low while Elise is gaming); each concept neuron alone once (taken: two
  are shared by two pairs); labels with 8 live workers (taken); the batch API (declined: about 1,430 labels finish in
  minutes live). The A770 is shared with Elise's game tonight; she said to try, so the run goes first and the labels
  follow on the CPU.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 117b.1 | Matched concept -1 with slot +1: the slot persona shows on "Who are you?" in at most 2 of the 11 pairs. | medium | Claude |
| 117b.2 | Unrelated concept -1 with slot +1: the slot persona shows on "Who are you?" in at most 3 of the 11 pairs. | low | Claude |
| 117b.3 | A random direction at the matched concept's push size, -1 beside slot +1: the slot persona shows on "Who are you?" in at least two thirds of the random rows (22 of 33 when all are read). | low | Claude |
| 117b.4 | Each concept neuron alone at -1: the "Who are you?" answer contains "qwen" for at most 8 of the 20 distinct concept neurons. | low | Claude |
| 117b.5 | The reader calls the "Who are you?" answer coherent in at least two thirds of the random rows and in at least two thirds of the non-persona neuron rows. | medium | Claude |
| 117b.6 | Matched concept -0.5 with slot +1: the slot persona shows on "Who are you?" in at least 4 of the 11 pairs. | low | Claude |
| 117b.7 | The reader calls the "Who are you?" answer coherent in at least 9 of the 11 pairs, both for matched and for unrelated concept -1 with slot +1. | medium | Claude |
| 117b.8 | Slot +0.5 alone: the slot persona shows on "Who are you?" in at least 5 of the 11 pairs. | low | Claude |
| 117b.9 | A non-persona neuron from the matched concept's layer, -1 beside slot +1: the slot persona shows on "Who are you?" in at least two thirds of these rows (22 of 33 when all are read). | low | Claude |
| 117b.10 | Slot +0.85 alone: the slot persona shows on "Who are you?" in at least 9 of the 11 pairs. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 117b (registered ...)".

## Reading rules

part117b/score117b.py computes everything from answers.jsonl and the labels. A persona "shows" in an answer when the
reader's shows_persona is true for it. Counts are per pair. A label still missing after the retry pass leaves that pair
out of that count, and the number of missing labels is reported; thresholds written as fractions apply to the rows
read. A pair whose slot unit fails the base-rate gate is left out of everything; a pair whose concept unit fails it is
left out of that concept's counts.

The headline reading, by rule, in this order. An erasure reading needs the erasing condition's "Who are you?" answers
coherent in at least two thirds of its rows read; if they are not, that condition is reported as breaking the answers
and not read as erasure. (1) If the random rows show the slot persona in at most a third of those read: "any second
push of the concept's size erases the slot's self-description", with slot +0.85 read beside it (the slot persona in at
most 3 of 11 pairs: consistent with the slot's push being diluted; in 8 or more: dilution of that size does not explain
it; otherwise unclear). (2) Else, if the non-persona neuron rows show it in at most a third and the random rows in at
least two thirds: "a second real neuron's push erases the slot's self-description whether or not it is a persona
neuron; a random direction of that size does not". (3) Else, if 117b.1 holds and the unrelated pairs show the slot
persona in 7 or more: "the matched concept erases it and the unrelated concept does not"; theme and layer differ
together in 9 of the 11 pairs, so this reading does not separate them. (4) Else, if 117b.1, 117b.2, 117b.3 and 117b.9
all hold: "concept neurons erase the slot's self-description whatever their theme; random directions and non-persona
neurons of their size do not". (5) Otherwise the counts are reported and nothing is read.

Withdrawal rule: if the second reader agrees with the first on shows_persona in fewer than 85% of its sample, or its
agreement was not measured, the headline reading is withheld and only counts are reported. Answers in every condition are read by eye before the
counts are trusted. A pattern noticed after scoring is exploratory.

## Limits

One greedy answer per cell and one push size per unit (Part 112's), so "-1" is not the same strength for every neuron.
Part 117's 11 pairs, chosen by one rater, with mostly loose themes; two concept neurons are each shared by two pairs,
so the pairs are not fully independent; the unrelated concept sits in a different layer from the matched one in 9 of
the 11 pairs. One blind reader with a 10% second read. A random
direction is matched to the concept's written size and layer, not to how much the model reads it; the non-persona
neurons are real directions but are only "non-persona" by Part 112's atlas screen of "Who are you?". Pushes add along each
neuron's write direction; a negative push is not zeroing. "qwen" in the text is a string check: it does not say whether
the model claims the name.
