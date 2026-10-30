# Part 119: when two of OLMo's persona neurons are pushed together, do their effects on the next-word scores add, and which one wins?

Procedure: v1
Drafted 2026-10-30 03:23 CT by Claude (Opus 5.5). Registered in commit 0d6d572 at 04:27:00 CT, before the pilot launch. Elise asked at 00:50 CT on 2026-10-30 to "keep trying to refine
your understanding of slot vs concept neurons and any other ways we can classify them, and try to figure out how
different types of neurons interact" (Discord message 1565603681804488839). A Luna 6 cold critique of this design before
registration (results/part119/design-review/CRITIQUE.md, 9 findings, all folded here) changed the winner-take-all rule,
added the interaction score and Path 2, and tightened gates 2 and 3 and predictions 119.5 to 119.9. The commit that adds
this file is the registration.

## Question

Parts 117 and 117b pushed pairs of Qwen neurons and read one greedy answer per cell: a second push of any kind took the
slot neuron's self-description away, and the reader could not tell an interaction from a threshold. Part 92b then showed
that Part 92's concept label depends on the reader for 10 of its 18 units. This part asks the interaction question with
readouts that need no reader. On "Who are you?" in OLMo 2 1B, with all 741 pairs of Part 92's 39 neurons: is the pair's
next word what adding the two neurons' single effects on the raw next-word scores predicts, or does the pair do
something the singles do not? Which neuron wins, and does its kind, or how hard its first word flips, predict that?

In the terms of EPISTASIS-FRAMING.md: the phenotype is the raw logit at one position, and the interaction is the pair's
logits minus the sum of the single effects.

## Prior art and existing data

- Closest published work: a9lim and treetowntree (X thread, 2026-10-27, x.com/_a9lim/status/2114902166804763073) steer
  several neurons at once in Qwen3-8B and report a rule for when the result is one coherent character: if A+B, B+C and
  A+C each give one, A+B+C gives a character that embodies all three, and A+B+C can be coherent when A+B alone is not.
  They read characters, with no random controls in the thread and no logit readout. Tzanakakis, Karatzikos and
  Georgakopoulos-Soares (2026, arXiv:2609.17599) define pairwise epistasis as the joint effect minus the sum of the two
  single effects and report non-additive damage among co-located rows, from ablations rather than pushes. van der Weij,
  Poesio and Schoots (2024, arXiv:2403.05767) found that one vector combining several behaviours mostly failed while
  separate vectors at different places worked. One search (2026-10-30) found nothing that tests additivity of two
  single-neuron pushes in next-word logits against random controls.
- Our own data: Part 92 (the 39 neurons and their kinds); Part 92b (21 slot units keep their kind under a DeepSeek
  reader, 8 of 18 concept units do); Part 117b (Qwen: random second pushes take the slot persona away in 30 of 33 rows;
  a slot alone at 0.85 of its push keeps it in 2 of 11 pairs); Part 109c (OLMo dog pair: full strength plays a pet, half
  strength nothing); Part 106 (log-probability of a persona sentence under its push); Part 49's interaction term
  (EPISTASIS-FRAMING.md); Part 53 and Parts 67 to 74 (single neurons driving others, activation readouts).

## Method

Registration: commit 0d6d572 at 04:27:00 CT on 2026-10-30, before the pilot launched at 04:27:21 CT.

OLMo 2 1B Instruct at Part 92's pinned revision, bf16, the A770, chat frame, no system prompt, thinking off, the one
question "Who are you?", greedy, 60 new tokens. Units: the 39 classified neurons of results/part92/step2/neurons.json
(Part 92's push: the neuron's W_down column scaled to alpha times its layer's median residual norm, added to the layer's
output at every position; part92/common92.py; alpha 0.8 unless a row says otherwise). Random directions: each unit's
paired random direction from Part 92 (its control_seed, its layer, its size), called R_A for unit A. A row with two
pushes hooks both layers; when the two share a layer the vectors add.

Answers: 1 plain row; 39 single rows (A); 39 random single rows (R_A); 741 pair rows (A+B, one per unordered pair); 1,482
control rows (A+R_B, every ordered pair with A and B different), so that each real second push has a random second push
of the same layer and size. 2,302 answers.

Logits, teacher-forced, in three sets. Path 1, each pair or control row's own answer, first 24 generated tokens: no push,
first push alone, second push alone, both pushes; 8,892 forward passes. Path 2, unit A's own answer, first 24 generated
tokens, the same path for every second push, chosen before any pair is read: no push and A alone once per unit, then second
push alone and both pushes for each of the 38 real second pushes B and the 38 random second pushes R_B; 6,006 forward
passes. A flip ladder: for each unit A, p_A is the first position where A's answer differs from the plain answer, t_A
A's token there and t_0 the plain token; with the plain answer as prefix up to p_A, one forward pass at each alpha in
0.1, 0.2, ... 1.0; 390 forward passes.

Definitions, all on token ids, all computed by part119/score119.py:

- A position on a row's own path is contested when the two single pushes' top tokens at that position differ (a for the
  first push, b for the second). The path stops at 24 tokens or the end token.
- Additive prediction: the top token of L_first + L_second - L_none (raw logits). A constant added to any of the three
  vectors does not change this top token.
- Winner-take-all prediction: the top token of the single push with the larger margin, a margin being a run's top logit
  minus its second-highest. A constant added to a run does not change its margin. An exact tie counts as a miss.
- A prediction hits when it equals the row's own next token. A row's hit rate is its hits over its contested positions.
  A summary is the mean over rows with at least one contested position, with a 95% bootstrap interval that resamples
  units (2,000 draws; a resampled unit brings all its rows).
- Interaction, on Path 2 at every position from p_A on. T is the union of the 10 highest tokens of each of the four
  passes. On T, each pass's logits are centered on their mean. D_A = L_A - L_0, D_2 = L_2 - L_0, D_pair = L_both - L_0, and
  I = D_pair - D_A - D_2, which is L_both - L_A - L_2 + L_0, the joint effect minus the sum of the single effects.
  Interaction share: |I|^2 / |D_pair|^2, norms over T; a row's value is the median over its positions, and the summary
  is the median over rows. Second scale, reported: the total variation distance between the softmax of the additive
  vector and the softmax of the both-push vector, over the whole vocabulary (EPISTASIS-FRAMING.md: report two scales).
- Effect-matched comparison. Over all control positions, least squares of log|I| on log|D_A| and log|D_2|. A real
  position's residual is its log|I| minus that fit; a row's value is the median over its positions, the summary the mean
  over rows with the unit bootstrap, and exp of it says how many times larger the interaction is in a real pair than in
  a control with the same single effects. Only real positions with log|D_2| inside the control positions' 1st to 99th
  percentile count, and the share of real positions kept is reported.
- Token winner of a real pair: at its contested positions, count tokens equal to a and tokens equal to b; the unit with
  more wins, equal counts or no such tokens is undecided.
- Dominance of a unit: wins over decided pairs, across its 38 real opponents.
- Robustness R(A): over the 38 rows A+R_B, the mean of (common prefix length of the row's answer and A's answer, minus
  p_A, floored at 0) divided by (24 minus p_A), both answers cut to 24 tokens. A unit whose answer does not differ from
  the plain answer in 24 tokens has no R and is reported.
- Flip alpha of A: the smallest ladder alpha at which t_A is the top token at p_A. Margin of A: L[t_A] - L[t_0] there at
  alpha 0.8.

Kinds: agreed slot (21 units, slot for both Part 92's Haiku reader and Part 92b's DeepSeek reader), agreed concept (8
units), disputed (10 units, concept for Haiku, slot or unclassified for DeepSeek). The kind claims below use the agreed
units, so that no claim rests on one reader, and the disputed units stay in the matrix as their own group. Every kind
number is also reported under Part 92's Haiku kinds (21 and 18). Pair classes among agreed units: slot x slot 210,
concept x concept 28, slot x concept 168; the other 335 pairs involve a disputed unit and are reported as their own
class.

Blind reader, for what token ids cannot say: DeepSeek V4.1 Flash (the reader of Parts 86 and 92b, 4,000-token cap, as
Part 118's Amendment 3 found necessary) reads a sample of answers against two persona descriptions, X and Y, assigned
to the two units in seeded random order. For each it answers whether the speaker describes itself as that persona or
clearly takes on its traits (a theme nearby does not count), and whether the answer is coherent. Sample, drawn with seed
11900 before any label: all 28 concept x concept pairs, 40 slot x concept pairs, 40 slot x slot pairs and 40 control
rows; a control row names B's persona as one of the two, although only R_B was pushed. 148 items, in seeded random order,
each with an opaque id, the class and units kept in a separate key that only the scorer reads. "Both" means both are yes. Luna 6
reads a seeded 20% (30 items) as second reader. 12 hand-written known examples go first, as in Part 92b.

## Controls

- Matched random controls, several: A+R_B for every ordered pair, the same layer and size as the real second push, 38
  random second pushes per unit. They are matched on layer and written size, not on effect, and a persona neuron's
  effect on this question is large by selection. So real and control are compared on the same path (Path 2, unit A's own
  answer) at the same positions, and by the effect-matched comparison above, not by hit rates on their own paths. Path 1
  hit rates on control rows are reported as description only: they sit on other paths and other contested positions.
- Both directions, poles and the answer-polarity split. Not used: this part asks about two positive pushes at the
  selection dose; Part 117b ran the negative side on Qwen, and a choice between two answers is not the readout.
- Do-nothing hook: a zero push against no hook, exact tokens (gated).
- Off-target questions. Not used: one question, because every readout here is on the path of this answer and Part 118 is
  running the dose ladder of the six-question battery.
- Dose ladder: the flip ladder above. Not used for the pairs: a half-dose pair row, because Part 109c saw nothing at half
  strength and this part first asks what the full-dose pair does; a follow-up if the additive test leaves it open.
- Broken answers: the reader's coherent field on the sample; share of rows with a three-word sequence repeated three
  times (part86/triage86.py's rule) on all rows.
- Selection: every persona here was selected at alpha 0.8 on this question at Instruct, so a unit's own answer is at a
  maximum for that unit. Every comparison of a pair against A's own answer carries that, which is why the random second
  pushes carry the same selection.
- Not used: non-persona neuron controls (Part 117b's did what random directions did); an activity-matched null, because
  nothing is zeroed.

Gate, before any prediction is read: (1) the zero push's answer equals the no-hook answer token for token; (2) the first
100 characters of the 39 single rows' decoded answers equal those of Part 118's alpha 0.8 "Who are you?" answers for at
least 33 of 39 units and at least 6 of the 8 agreed concept units (Part 118 stores decoded text, not token ids; the
same device and code path; no label is read); a miss on the concept count stops the kind claims; (3) on Path 1, the
teacher-forced both-push pass picks the row's own token at at least 98% of all positions and at least 97% of contested
positions of the real rows, with the control rows reported (else the logit analysis measures something other than the
generation, and nothing that uses logits is read). The first two can fail if the two-layer hook differs from Part 92's
single hook; the third can fail if the cached generation and the uncached forward pass differ on this device, most
likely at close calls, which is where contested positions sit.

## Size

- Smallest effect worth knowing: a 10-point gap between the additive and winner-take-all hit rates, and between real and
  control rows, across about 700 rows; the bootstrap over 39 units sets the interval. For kinds, 8 against 21 units gives
  a Mann-Whitney test about 80% power only for a large effect, so a kind difference smaller than about one standard
  deviation is out of reach.
- Total: 2,303 greedy answers of up to 60 tokens (the 2,302 rows and one no-hook answer for gate 1), 15,288 teacher-forced
  passes, 190 reader calls. About 10 minutes of generation and 3 of forward passes on the A770, from Part 118's rate
  (1,878 answers of 120 tokens in about 8 minutes).
- Pilot: the plain row, the 78 single rows, 20 pair rows and 40 control rows drawn with seed 11901, timed at batch 64 and
  128. Stopping rule: read all pilot answers by eye; if more than half the pair answers are broken, stop and amend the
  dose before the main run. Gates 1 to 3 are checked on the pilot first.

## Optimization

- Bound resource: the A770.
- Measured: 24 answers in 18.8 s at batch 24 on this model and device (Part 118's pilot), and 1,878 answers in about 8
  minutes in its main run; the pilot here measures two batch sizes on real rows before the main run.
- Levers: batch size (taken: 64 against 128 timed in the pilot, the better one used); max_new_tokens 60 (taken: every token
  readout uses the first 24, and the reader reads a self-description in the opening sentences, so a label-stability check
  on longer answers is declined); teacher-forced scoring (taken: everything but the
  answers and the reader's sample); all rows share one prompt, so no padding (taken); the shared prompt prefilled once
  (declined: pushes act on the prompt tokens, so each row needs its own prefill); rows leaving the batch when they finish
  (declined: few answers end before 60 tokens); pooled controls (taken: each unit's one paired random direction is
  reused across its 38 rows); A770 copy engine off (taken, the job runner's default); host memory (Part 118 loaded this
  model with room to spare); resume by row key (taken); reader live with 8 workers (taken: 190 calls), one known-example
  gate before the sample, under 60K tokens of the shared Go window.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 119.1 | Across the 741 real pairs, the additive prediction hits the pair's own token at at least 60% of contested positions (mean over rows). | low | Claude |
| 119.2 | Across the 741 real pairs, the additive prediction's hit rate is at least 5 points above winner-take-all's on the same positions, with the bootstrap interval of the difference above 0. | low | Claude |
| 119.3 | On Path 2, the median interaction share over the 1,482 real ordered pairs is at most 0.25. | low | Claude |
| 119.4 | Effect-matched, the interaction in real pairs is within 25% of a control's with the same single effects: the bootstrap interval of the mean residual lies inside plus or minus log 1.25. | low | Claude |
| 119.5 | In slot x concept pairs of agreed units, at least 84 of the 168 pairs are decided and the concept neuron is the token winner in at least 60% of decided pairs. | medium | Claude |
| 119.6 | Agreed concept units have higher robustness R than agreed slot units: median difference at least 0.10, one-sided Mann-Whitney p below 0.05. | medium | Claude |
| 119.7 | Agreed concept units have a lower flip alpha than agreed slot units: median difference at least 0.10, one-sided Mann-Whitney p below 0.05. | medium | Claude |
| 119.8 | Across the 39 units, dominance and flip alpha have a Spearman correlation of -0.4 or stronger (a neuron that flips early wins more often). | medium | Claude |
| 119.9 | The disputed units' median robustness R is closer to the agreed slot units' median than to the agreed concept units' median, by at least 0.05. | low | Claude |
| 119.10 | By the reader, "both" holds in at least 25% of the 28 concept x concept pairs, at most 10% of the 40 slot x slot pairs and at most 5% of the 40 control rows. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 119 (registered 2026-10-30)".

## Reading rules

part119/score119.py computes every number from the answers, the logit summaries and the labels. Gates first, in the
order above; a failed gate 3 stops every reading that uses logits (119.1 to 119.5, 119.7 and 119.8), and 119.6 and 119.9,
which use answers only, are still read.

A unit whose answer does not differ from the plain answer in 24 tokens has no Path 2 positions: such units are listed by
id, and 119.3 and 119.4 are read over the rows that have positions, with the count reported, unless more than 5% of the
1,482 real rows have none, when they are not read.

The headline, by rule, in this order. (1) If 119.1 to 119.4 all hold: "at this dose on this question the two pushes'
effects on next-word scores mostly add, and a real second neuron interacts no more than a random direction with the same
single effects". (2) Else, if the bootstrap interval of the mean residual lies above log 1.25: "real neuron pairs interact
more than a neuron and a random direction with the same single effects". (3) Else, if the additive hit rate is below 50%
or the median interaction share is above 0.5: "the pair's next word is mostly not predicted by adding single effects".
(4) Otherwise the numbers are reported and nothing is read. Any top-token result is stated as a result about the top
token, and the interaction share says how much of the change adds. The answer to "which one wins" is reported by class
with decided over total pairs; no kind claim is made if its direction differs between the agreed kinds and Part 92's
Haiku kinds. Kind claims describe these 8 agreed concept units and 21 slot units, selected on this question, not concept
neurons in general; 119.5 and 119.10 are also reported leaving out each concept unit in turn, with per-unit coverage.

The reader uses label92b's DS settings (temperature, thinking and cap as there, with Part 118's 4,000-token cap). Items
still in error after the retry pass are dropped from that count's denominator and reported. The reader's numbers are
withheld if it fails the 12 known examples below 10 of 12 or the second reader's agreement on "both" falls below kappa 0.6. A pattern found after scoring, by layer distance or by theme, is exploratory. Answers in
every pair class are read by eye before the counts are trusted.

## Limits

One question, one dose, greedy, one model. A contested position can be a style word with no persona in it; the token
winner counts tokens, not characters, which is why a reader's sample sits beside it. Robustness R and flip alpha are
measured on the question and dose the units were selected on: they are first-differing-token and string-stability
measures of these selected units, and a flip alpha cannot exceed 0.8 by construction. "Adds in logits" does not say the
neurons do not affect each other's activations; a sum at the output can hide cancelling changes inside. The pair is
pushed at every position, prompt included. Eight agreed concept units give little power; pairs share units, so the
bootstrap resamples units. Random directions are matched to the neuron's size and layer, not to its effect. Part 118's
dose results, due 2026-10-31, may change what "slot" means here.

## Amendment 1 (2026-10-30 04:33 CT, after the pilot, before the main run)

The pilot (139 answers, 60 logit rows, the flip ladder) passed gates 1 and 3 (98.1% of positions, 97.2% of the real rows'
contested positions, both close to their lines) and missed gate 2 by one: 32 of 39 units, with 6 of 8 agreed concept
units. All seven misses keep the same persona opening and part from Part 118's text at characters 25 to 86, at ordinary
close calls (for example "fool!" against "fool."). Part 119's rows and pilot-64 and pilot-128 are token-identical across
batch sizes (139 of 139), while Part 118's rows were left-padded in batches of mixed questions, so padding is the likely
difference and not the two-layer hook, which is what gate 2 exists to check.

Gate 2 changes to test that directly: part119/gate2b.py runs the 39 units through Part 92's own single-layer hook path
(common92.generate_batch) with the same question, no padding, 60 tokens, on the same device, and at least 38 of 39 decoded
answers must equal the single rows' decoded answers. The comparison with Part 118 stays reported, not gated. The concept
sub-gate moves to the same comparison.

Disclosure: to check gate 3 on the pilot I also printed the additive hit rate on the pilot's contested positions, 23.8% on
20 real pairs and 26.6% on 40 control rows, which is below the 60% in 119.1 and was not part of the gate check. No
prediction, threshold or reading rule changes because of it. Also seen: the flip ladder's spread (flip alpha 0.4 to 0.8,
no no-flip units), and the 20 pilot pair answers read by eye, as the pilot's stopping rule requires; about a quarter
look broken, under the one half that would stop the run.

Timing: batch 64 and 128 gave the same 1.5 answers/s on the pilot, because rows are grouped by their pair of layers and the
pilot's groups are small; 30 passes/s. Batch 64 is used. The run carries on in the pilot-64 folder, renamed results/part119/run,
so the pilot's rows are reused.

## Amendment 2 (2026-10-30 04:36 CT, during the main run, before any main score is read)

Gate 2 as amended read 35 of 39 against its 38. The four misses (p92-47, p92-48, p92-15, p92-32) keep the same opening and
part from the single rows at characters 36 to 70, at close calls such as "my honor!" against "and my honor.". Both paths ran
without padding, so what still differs is the batch: the single rows ran in groups of a layer's rows and gate2b ran
per layer. Logits here are bf16, in steps of 0.125 near the top, so a close call can turn on rounding that changes with
batch shape on this device. This is a hypothesis, not a finding.

The second miss is recorded as a miss. The gate's question stays the same, whether the two-layer row hook is Part 92's hook.
part119/gate2c.py now asks it with nothing else different: each of the 39 units at batch size 1, through common92.generate_batch
and through run119's row hook, in the same process, one after the other. At least 38 of 39 decoded answers must be equal, or
gate 2 fails and no prediction is read. The agreed concept sub-gate (at least 6 of 8) moves to the same comparison. The
gate2b comparison and the Part 118 comparison stay reported. If the hypothesis is right, greedy answers on this device
carry a small dependence on batch shape; the run's rows are not reshaped for that, and the size of it (4 of 39 answers
differing somewhere in 60 tokens) is reported as a limit on any single row.

## Note after scoring (2026-10-30 06:03 CT, written after the scores; no prediction or threshold changes)

A cold check of the write-up (results/part119/COLD-CHECK-r1.md) found 19 findings, grouped below as five corrections.

1. Registration. The 03:23 in the header is the draft time. The registration is commit 0d6d572 at 04:27:00 CT,
   before the pilot launched at 04:27:21.
2. 119.10. This design withholds only when agreement "falls below kappa 0.6"; undefined is not below 0.6. The readers
   agreed on all 30 shared items, and both got the known "both" example right. By this design wording, 119.10 is read
   and failed: 0 of 28 concept pairs against at least 25%. The scorer adds a non-null-kappa requirement, so its output
   says withheld / not read. The README records both readings; SCORE.md keeps the scorer's output as run.
3. Robustness thresholds. 119.6 and 119.9 asked for differences of 0.10 and 0.05 on a measure whose 39 values all lie
   between 0 and 0.058, so neither could hold. That was my registration error. The two failures say only the sign.
4. Bootstrap. The scorer weights a row by the product of its two units' draw counts. The design phrase "a resampled unit
   brings all its rows" does not specify how a row shared by two units is weighted. The README gives the sum-weighted
   intervals beside the scorer's product-weighted intervals.
5. Broken answers. The counts use part86/triage86.py as written: it keeps recognized identities or first-sentence A.I.,
   AI, assistant, language model, model, chatbot, or LLM text; otherwise it marks answers under four words or with a
   repeated three-word sequence occurring three times as broken. The README gives the full triage counts,
   three-word-repeat counts, and reader coherence by class.
