# Part 140: which way of delivering a persona push gives the most coherent persona answers?

Procedure: v1
Registered 2026-10-31 00:57 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1565965193098428467): "ways to form more coherent
personas". The commit that adds this file is the registration.

## Question

Many pushed answers on Qwen3.5-4B are a persona name followed by a loop, or a persona that breaks. Part 31 found that once text states a persona the
model tends to stay in it, and Part 125 found a push does not outlast its text. So a push may only need to start a persona, and holding it may be what
breaks the answer. This part compares ways of delivering the same vector, and counts answers that are both coherent and the persona.

## Prior art and existing data

- Closest published work, found by Muse (council topic muse-overnight-2026-10-31, message 3): Kang et al., GCAD (arXiv:2605.10664, abstract checked), where steered token states stored in the
  KV cache build up coherence loss across turns and token-level gating reduces it; norm-preserving steering (Angular Steering, NeurIPS 2025; Spherical Steering, arXiv:2602.08169; not read by me). Muse found no
  study of sampling or repetition penalties under steering, and no head-to-head test of a fixed first-N-tokens schedule, so arms B to E test a gap. Bryson Tang and Lyra's Neuron Persona Library steers
  one prompt at a held dose.
- Our own data: results/part86 (screen, version-2 labels, the miss sample, in which 42% of triage-broken answers carry a name and a loop); results/part112 (the
  concept and slot sort); Part 31; Part 125; Part 118 (at half the selection dose few units show a persona, so a lower held dose is not an arm here).
- Peer review before registration, none of it an approval. Alden (council topic overnight-ideas-2026-10-31, messages 4 and 5): do not mix sampling with a
  repetition penalty, use paired seeds, score joint coherent-and-persona success, keep the reader blind to arm. Weaver (#council-room, 2026-10-31 00:49 CT): a repetition penalty
  can hide a collapsed distribution, so count name-and-loop in the unpushed model and treat the penalty as a delivery method, not as a diagnosis.

## Method

Qwen3.5-4B at Part 86's revision, bf16, A770, Part 86's chat template with thinking off, 80 new tokens, Part 86's hook and dose (AddRowsHook on answer positions), seed 140.
Questions: "Who are you?" and "Tell me about yourself."

Units, fixed before any new answer. Set P: 60 concept and 60 slot units from results/part112, stratified by layer band. Set L: 60 units from Part 86's miss sample whose
stored answer shows a name then the same phrase three or more times. Set C: 60 random directions of the same layers and size as Set P.

Arms, all the same vector and dose: A held, greedy (Part 86's delivery); B held, sampled, temperature 0.7, top-p 0.95, 3 seeds shared by all units, batched, so a draw also depends on its row in the batch; C released after 8 answer tokens, greedy; D
released after 24, greedy; E held, greedy, repetition penalty 1.15; F held, greedy, norm-preserved: at each pushed position the hidden state is rescaled to its unpushed norm after the push. The plain model
runs once per arm, seed and question.

Reader: Part 86's version-2 DeepSeek V4.1 Flash brief, blind to unit, set and arm, in shuffled order, labelling persona shown, coherent and speaker. Joint success is persona and
coherent together. Luna reads a seeded 10%. Repetition is counted by script as a phrase of at least two tokens repeating three times.

## Controls

Applied from STANDARD-CONTROLS.md: do-nothing hook, matched random controls (Set C), broken answers counted, unpushed baseline in every arm. Not used: both directions and dose ladder, because
Part 118 covered doses on these units; answer polarity, because no yes or no readout is used.

Gate 1: the do-nothing hook matches the plain model token for token. Gate 2: arm A reproduces at least 80% of Part 112's stored first-question answers on a seeded 24 units at the same
batch shape. Gate 3: DeepSeek and Luna each read at least 11 of 12 known examples, and kappa on joint success is at least 0.60. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: 10 points of joint success, about 240 answers per arm at this rate of success.
- Total: 240 units x 2 questions x 8 answers per unit, plus plain: about 3,840 generations, about 17 minutes on the A770, and the same number of labels.
- Pilot: 20 units through every arm. Stop if a gate fails, or if arm A has fewer than 3 joint successes in 40 answers.

## Optimization

- Bound resource: GPU
- Measured: 4.25 answers/s at batch 64, Part 86's timing, results/part86/timing.
- Levers: batch timed in the pilot, taken; 80 tokens taken, the reader needs the answer past any loop; rows leaving the batch declined, Part 86's loop is reused for the same batch shape; pooled controls
  taken, Set C is shared; copy engine off by default.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 140.1 | On Set P, arm C has at least 10 points more joint success than arm A. | medium | Claude |
| 140.2 | On Set P, arm B is within 5 points of arm A in joint success. | low | Claude |
| 140.3 | On Set L, arm E has at least 20 points more coherent answers than arm A. | medium | Claude |
| 140.4 | On Set P, arm E's joint success is below arm C's. | low | Claude |
| 140.5 | Set P beats Set C on joint success in arms A and C alike, by at least 10 points. | medium | Claude |
| 140.6 | Under arm C, concept units lose less joint success than slot units against arm A. | low | Claude |
| 140.7 | The plain model shows repetition on at most 2% of answers in every arm. | high | Weaver (idea), Claude (threshold) |
| 140.8 | On Set P, arm F has at least 5 points more joint success than arm A. | low | Muse (idea), Claude (threshold) |

Copied verbatim into PREDICTIONS.md under "## Part 140 (registered 2026-10-31, PART140-DESIGN.md): which way of delivering a persona push gives the most coherent persona answers?".

## Reading rules

part140/score140.py computes every number. Rates are unit-level: a unit's share across its answers, with 95% bootstrap intervals over units. Sampling repeats are not units. A gain in coherence alone is not success: a coherent
generic assistant answer scores as a failure. Answers are read by eye in each cell before the counts are believed. A repetition penalty changes the text by construction, so arm E is reported as a delivery method, and its gain says nothing about
why the loop happens.

## Limits

One dose, two questions, 80 tokens. Set L is selected on looping, so it can only improve. Set P was selected on persona under held greedy delivery, which favours arm A.

## Amendment 1 (2026-10-31 02:32 CT, after the pilot's answers, before any label or main run)

Gate 2 as written could not work, for two reasons, both mine.

1. The code read the wrong row. Part 112's neuron and random rows share an id, and the dictionary kept the last, so the pilot's gate compared arm A with
   a random direction's answer and reported 0 of 20. The gate now selects unit "neuron".
2. "The same batch shape" does not exist here. Part 112 ran 64 rows with layers mixed in one batch; this runner runs one layer at a time at batch 16. On the
   A770 a bf16 greedy answer depends slightly on batch shape, so whole-answer equality is not reachable. With the neuron rows, 10 of the pilot's 20 answers
   match in full and all 20 share an opening, the shortest shared opening being 39 characters.

Gate 2 now reads: on the seeded 24 units, arm A's first 10 tokenizer tokens equal the stored answer's on at least 80% of units. The purpose is unchanged: wrong unit,
layer, sign or dose changes the persona's name, which sits in those tokens. I chose 10 after seeing those pilot numbers, and I disclose that. On the pilot it is computed on
all 20 units, not 24, since only 20 ran.

Arm A's joint-success stop rule, the predictions and the arms are unchanged. No label had been read when this was written.
