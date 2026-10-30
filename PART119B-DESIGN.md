# Part 119b: is Part 119's failure to add a property of the dose where answers break?

Procedure: v1
Registered 2026-10-30 07:31 CT by Claude (Opus 5.5), before any data. Elise asked at 00:50 CT on 2026-10-30 to "keep trying
to refine your understanding of slot vs concept neurons and any other ways we can classify them, and try to figure out
how different types of neurons interact" (Discord message 1565603681804488839). Part 119's limit is the reason for this
part: at alpha 0.8 on both pushes, 357 of 741 real pair answers broke, against 4 of 39 single answers. The commit that
adds this file is the registration.

## Question

Part 119 pushed all 741 pairs of OLMo's 39 persona neurons at alpha 0.8 each. Adding the two single effects picked the
pair's next word 24.4% of the time, the median interaction share was 0.47, and real pairs interacted no more than a
neuron with a random second push of the same single effects (1.04 times). Half the pair answers broke, so those numbers
may describe a dose where the model is failing for any reason. This part asks: on the same pairs and the same positions,
does the interaction fall as both pushes shrink, and how fast? At a dose where answers do not break, do the effects add,
and does a real pair then interact more than a random one? Is breaking about the two neurons, or about how much is
pushed? Which single-unit number explains why concept units won Part 119's token contests (119.5)?

## Prior art and existing data

- Closest published work: a survey of representation engineering (Zhang et al., arXiv:2502.19649) reports that adding
  several steering vectors makes concepts interfere and lowers output quality; "Steering Under Compression"
  (arXiv:2609.06473) studies steering dose-response in quantized models. Both found by one search on 2026-10-30, read at
  the abstract or search-result level only. That search and a second one found no test of how two single-neuron
  pushes' interaction in next-word logits scales with dose against random controls.
- Our own data: Part 119 (all the 0.8 numbers here, and its answers, reused); Part 109c (OLMo dog pair: full strength plays
  a pet, half strength nothing); Part 117b (Qwen: a random second push removes the slot persona, and a slot alone at 0.85
  of its push keeps it in 2 of 11 pairs); Part 118 (dose ladder on six questions, labels due 2026-10-31, deliberately
  not read for this design); EPISTASIS-FRAMING.md (interaction definition, report two scales).

## Method

Same model, revision, device, chat frame, question, 39 units, random directions and push code as Part 119 (bf16, the A770,
"Who are you?", greedy, 60 new tokens). A dose alpha applies to every push in a row. Reused from Part 119 unchanged: the
plain answer, single and random rows at 0.8, real pair rows at 0.8, control rows at 0.8.

New answers (5,577). Real pairs A+B at 0.2, 0.4 and 0.6 (741 each). Controls A+R_B at 0.4 (1,482). Random pairs R_A+R_B,
one per unordered pair, at 0.4 and 0.8 (741 each). Single rows A and R_A at 0.2, 0.4, 0.6, 1.13 and 1.6 (78 each).

Fixed-path logits (Path 2), on Part 119's positions: unit A's own 0.8 answer, first 24 tokens, from p_A on, the same
tokens at every dose. At each of alpha 0.2, 0.4, 0.6 and 0.8: A alone once, then second push alone and both pushes for
each of the 38 real and 38 random second pushes; no push once per unit. 23,907 forward passes.

Shared support, so that a change with dose comes from the dose and not from a different scoring set:

- Token set: for each row and position, T is the union of the 10 highest tokens of the four passes at 0.8 (Part 119's T), and
  the same T is used at every dose.
- Resolved position: at a given dose, nA, n2 and nPair are all at least 1.0. bf16 logits near the top move in steps of
  about 1/8, an error of about 0.036 per logit; I sums four logits, so about 0.07 per token, about 0.36 over a T of 25
  tokens. A share at nPair 1.0 therefore carries a floor of about 0.13, and at nPair 5 about 0.005. Part 119's rows
  sit at nPair of about 10 to 40 at 0.8.
- Interaction share at a dose is read on positions resolved at all four doses. Coverage (positions and rows kept, of
  all) is reported at every dose, and all positions are reported beside it.

Own-path logits (Path 1): the 741 real pair rows at 0.4, first 24 tokens of each row's own answer, the four passes of Part
119. 2,964 forward passes.

Definitions are Part 119's and part119/score119.py computes them the same way: contested position, additive prediction
(top token of L_first + L_second - L_none on raw logits), margin winner-take-all, hit, row hit rate, unit bootstrap (2,000
draws), interaction I, interaction share, total variation second scale, effect-matched comparison (fitted separately at
each dose on that dose's control positions), token winner, dominance, Part 86's broken rule. New on Path 2:

- Pair-top agreement on Path 2. On a fixed path there is no generated token, so the target is the both-push pass's top
  token over the whole vocabulary. The comparison between two doses uses only positions contested at both doses.
  Report additive, winner-take-all and plain (the no-push top token) agreement on that shared mask, with its coverage.
  Agreement alone cannot show that both pushes contribute: if the second push did nothing, the additive top token would
  equal the pair's top token. That is what the slopes are for.
- Regression on T at each position: least squares without intercept of D_pair on D_A and D_2, giving b_A and b_2. A row's
  value is the median over resolved positions, and the summary the median over rows. Both pushes contribute and add when
  both are near 1.
- Kind gap of a slot x concept pair: the mean, over the pair's two ordered rows (one on each unit's path), of
  b_concept - b_slot.
- Effect size E(A): the median of nA over A's positions on its own Path 2 at 0.8.
- Bootstrap weighting: as in Part 119's scorer, a row's weight is the product of its two units' draw counts. Any interval
  a claim rests on is also given with the sum of the two counts.

Blind reader, because Part 119's cold check showed the triage rule alone undercounts: triage called 357 of 741 real pair
answers broken, and DeepSeek V4.1 Flash marked only 16 of its 108 sampled real pair answers coherent. Part 119's sample was
drawn with seed 11900. The same 108 real pairs and 40 controls, at 0.4, go to the same reader with the same prompt, in a
new seeded random order with opaque ids (seed 11904), the key held apart; Luna 6 reads a seeded 20% (30 items, seed 11905)
as second reader; the 12 known examples of part119/known119.jsonl go first. Only the coherent field is read. The 0.8 side of
the comparison is Part 119's labels, unchanged.

## Controls

- Matched random controls, several: A+R_B for every ordered pair (38 per unit), as in Part 119, at every dose on Path 2.
  They are compared with real rows on the same positions and by the effect-matched fit, not by raw means.
- Random pairs R_A+R_B: no neuron in the row at all, the row that can show breaking does not need a persona neuron.
- Both directions, poles and the answer-polarity split: not used, this part follows Part 119's two positive pushes.
- Do-nothing hook: a zero push against no hook, exact tokens (gated).
- Off-target questions: not used, every readout is on the path of this answer; Part 118 runs the six-question ladder.
- Dose ladder: this part is one, at 0.2, 0.4, 0.6 and 0.8 for pairs, and to 1.6 for singles.
- Broken answers: part86/triage86.py as written on every new answer (it marks repeats and answers under four words, and
  keeps answers naming Qwen, Tongyi or Alibaba or carrying an AI word in the first sentence), with the three-word repeat
  alone beside it. Answers at every dose are read by eye, and the reading recorded, before counts are trusted.
- Selection: the units were picked at alpha 0.8 on this question, so Path 2 is an answer the model gives at 0.8 and not at
  lower doses. That is the point of holding it fixed and also a limit (see Limits).
- A reader's coherent field beside the triage rule, on Part 119's own sample (see Method). The reader's "both" field is not
  read: at 0.4 most answers carry no persona to ask about.

Gate, before any prediction is read: (1) the zero push's answer equals the no-hook answer token for token; (2) the 0.8 run
of Path 2 reproduces Part 119's stored numbers: median row interaction share within 0.01 of 0.4707 (real) and 0.3686
(control), and median relative difference in nPair under 1%. This gate uses all positions, as Part 119 did. A miss stops
every Path 2 reading; (3) the instrument can report near-additivity with a real signal: at 0.2, at least 70% of real rows
keep at least one resolved position and the median row interaction share over them is at most 0.15. A miss stops the
reading of 119b.4, 119b.5, 119b.7 and 119b.8 as a dose effect, because a share that does not fall at a small push, or that
falls only because the effects fell below rounding, cannot be told from a floor in the measure.

## Size

- Smallest effect worth knowing: a 10-point difference in hit rate between doses, on the same 39 units; the bootstrap
  over units sets the interval. Eight agreed concept units give a kind comparison power only for a large effect.
- Total: 5,577 answers of up to 60 tokens (about 10 minutes at Part 119's 9.06 answers per second), 26,871 forward passes
  (about 26 minutes at its 17.2 passes per second), 202 reader calls (148 + 30 + 24 known), under 60K tokens of the shared
  Go window.
- Pilot: the plain row, singles at every dose, 20 real pairs at 0.4, 20 at 0.6, 20 random pairs at 0.8 (seed 11902), Path 2
  for 2 units with 8 second pushes each at 0.2 and 0.8. It checks gates 1 and 2 on those rows, times the batch choices
  below, and prints gate results and timing only: no share, hit rate or broken count from a row a prediction reads.
  Stopping rule: read the pilot's answers by eye; if the 0.4 pair answers are mostly broken, stop and amend the doses.

## Optimization

- Bound resource: the A770.
- Measured: Part 119's main run gave 9.06 answers per second at batch 64 and 17.2 forward passes per second; its pilot
  timed batch 64 and 128 at about the same speed; Path 2 ran its second and both passes one row at a time, and
  recomputed the no-push and first-push top ids for every row.
- Levers: reuse Part 119's answers and logits at 0.8 where the readout is the same (taken, except Path 2, which reruns for
  its new fields and doubles as gate 2); Path 1 on real rows only, since Part 119 reads control hit rates as description
  (taken, 5,928 passes saved); cache the no-push and first-push top ids per unit (taken); batch Path 2's second and both
  passes across a unit's rows (the pilot times 1 and 8, the faster used, and gate 2 must still pass); fewer second pushes
  at the middle doses (declined: about 8 minutes saved for a second row set to explain); answers at batch 64 (taken);
  A770 copy engine off (taken, the job default); top-k on the device (declined: the gotcha, repeated argmax as in Part
  119); resume by row key (taken).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 119b.1 | At 0.4 + 0.4, at most 10% of the 741 real pair answers are broken by the triage rule, and the reader marks at least 60% of the 108 sampled real pair answers coherent (16 of 108 at 0.8). | medium | Claude |
| 119b.2 | At 0.8 + 0.8, at least 20% of the 741 random pair answers R_A+R_B are broken. | low | Claude |
| 119b.3 | At alpha 1.13, the combined size of two orthogonal pushes of 0.8, at least 30% of the 39 single neuron answers are broken. | low | Claude |
| 119b.4 | On Path 2, the median row interaction share over the 1,482 real ordered pairs is at most 0.25 at 0.4. | medium | Claude |
| 119b.5 | On Path 2, the median row interaction share at 0.2 is at most half the share at 0.4. | medium | Claude |
| 119b.6 | On Path 2 at 0.4, both slopes add: the median row b_A and the median row b_2 over the 1,482 real ordered pairs both lie between 0.75 and 1.25. | medium | Claude |
| 119b.7 | On Path 2, on positions contested at both doses, the additive pair-top agreement at 0.4 minus at 0.8 is at least 20 points, with the unit bootstrap interval above 0, and the 0.4 agreement is at least 10 points above the plain baseline on the same positions. | medium | Claude |
| 119b.8 | On the same positions at 0.4, additive agreement is at least 5 points above winner-take-all's, with the interval above 0. | low | Claude |
| 119b.9 | Effect-matched at 0.4, the interval of the mean real residual lies inside plus or minus log 1.25. | low | Claude |
| 119b.10 | At 0.8, the kind gap is positive in at least 60% of the 168 slot x concept pairs of agreed units, and the median absolute gap at 0.4 is at most half that at 0.8. | low | Claude |
| 119b.11 | Across the 39 units, E and Part 119's dominance have a Spearman correlation of at least 0.5. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 119b (registered 2026-10-30)".

## Reading rules

part119b/score119b.py computes every number. Gates first, in the order above.

The headline, by rule, in this order. (1) If 119b.1, 119b.4 and 119b.6 all hold: "at 0.4, on prefixes from each neuron's
own 0.8 answer, the two pushes' effects on next-word scores mostly add, both pushes contribute, and the pair's own answers
stay intact; Part 119's failure to add does not extend to this dose". (2) Else, if the median share at 0.4 is at least 0.4:
"the failure to add extends to 0.4", with the 0.4 broken share beside it. (3) Otherwise the numbers are reported and
nothing is read.

If at least 70% of real positions are kept in the effect-matched comparison at 0.4 and 119b.9 fails with the interval
above log 1.25, the line is added: "at this dose real pairs interact more than a neuron and a random direction with the
same single effects", the one result here that would make interaction specific to real neurons. Below 70%, 119b.9 is
not read. If 119b.2 and 119b.3 both hold, the line is added: "breaking tracks how much is pushed more than which neurons
are pushed", with the 3% and 10% single rates, the measured combined norm of same-layer pairs, and the broken share of
same-layer against different-layer pairs beside it (same-layer vectors can add or cancel; different-layer pushes have no
single combined norm).

The reader's coherent count is withheld if a reader scores below 10 of 12 on the known examples, or if the DeepSeek and
Luna kappa on coherent is defined and below 0.6. If kappa is undefined because a reader gave one value throughout, raw
agreement is reported and the count is read.

Path 1 at 0.4 is description only, read beside Part 119's 24.4% with the number of rows that have a contested position. It
sits on different answers from Part 119's, so its difference from 24.4% is not read as a dose effect. 119b.11 says effect size tracks dominance across
units; it does not say effect size causes the concept wins, and no kind claim is made if the direction differs between
the agreed kinds and Part 92's Haiku kinds. Kind claims describe these units on this question. A pattern found after
scoring, by layer distance or by persona theme, is exploratory.

## Limits

One question, one model, greedy, four doses. Path 2's positions come from an answer the model gives at 0.8 and not at the
lower doses, so a small interaction there says the logits add on that prefix, not that a low dose produces a persona, and
the shared token set T was chosen at 0.8. Both pushes always share a dose. Rounding is bounded by the resolved-position rule,
not removed. Greedy answers on this device carry a small dependence on batch shape (Part 119's Amendment 2), so a single
row is one draw of that, and Path 2 keeps Part 119's batch-64 single answers as its fixed paths. A sum at the output can hide cancelling changes inside. Random directions are matched to size and layer, not
effect. Selection at 0.8 means a single unit's own answer is at a maximum there; the random second pushes carry the same
selection. Pair breaking is compared with singles at 1.13 and 1.6 only as a rough size comparison: most pairs' pushes sit
in different layers, where no one number is their combined size.
