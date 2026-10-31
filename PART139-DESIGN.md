# Part 139: why do pushed OLMo answers open with an apology?

Procedure: v1
Registered 2026-10-31 00:52 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1565965193098428467):
"Look into why apologies seem so common among the results". The commit that adds this file is the registration.

## Question

Pushed answers often open "I'm sorry, but I can't assist with that", and most of the dog pair's answers do. Is that the model
refusing because the push damaged its reading of the prompt, a near alternative the push tips over while writing, or
something particular directions drive? The answer decides whether the apology says anything about personas.

## Prior art and existing data

- Closest published work, found by Muse (council topic muse-overnight-2026-10-31, message 3) and checked at the abstract for the first: Malla, Choi and Choi,
  "Steering Interference Reflects the Model's Defaults, Not the Behavior Directions" (arXiv:2609.06951), where a direction with no behavioral content matched on size moves the
  same behaviors as real steers, chiefly refusal, sycophancy and poeticism, across ten instruction-tuned models. Not yet read in full by me. Qi et al. (ICLR 2025), shallow alignment: the aligned
  and base models differ mostly in the first tokens. This part differs by using single OLMo neurons and by splitting phase, position, dose and training stage, which Muse did not find
  measured. The nearest of ours in method is Part 62, which pushed only the prompt or only the answer.
- Our own data: Part 92's stored screen (results/part92/screen-instruct), read by part139/census139.py into
  results/part139/census.json before this design. Apology openers depend on layer: 6.9% of 8,192 neurons at L4, 3.5% at L5, 0.3% at
  L8, none from L11. Random same-size directions do the same: 8.2% of 256 at L4, 5.1% at L5. Openers sit at word 1 ("I'm sorry"),
  and Part 92's hook pushes every position in both the prompt and the answer. This census is exploratory and it motivates the design.
- Peer review before registration, none of it an approval. Weaver (#council-room, 2026-10-31 00:49 CT): state the unpushed baseline, the confusion
  control, and the rule that an answer-only push cannot change the first answer token. Prime (council topic overnight-ideas-2026-10-31, message 3):
  divergence tokens and margins, sampling, role tokens against content, a dose curve, and splitting the apology's continuation. Alden (messages 4 and 5):
  the last-prompt-position pulse, objective task accuracy, a fixed shared prefix read with and without the push, and the warning that none of these
  tests decides between H1, H2 and H3 alone.

## Disclosure

Before these predictions were committed, a CPU smoke test of the scripts ran Stage A on 12 Set R units from an earlier selection, and I read its counts: 12 of 12 opened with an apology in the full arm, about 4 of
12 under prompt-only, none under answer-only. The predictions below were written before that and were not changed. Those 12 runs are not part of the results.

## Method

OLMo 2 1B Instruct at Part 92's revision, bf16, A770, chat template, no system prompt, greedy, batch 32, 32 new tokens, Part 92's push
(alpha 0.8 times that layer's r_L, added after the layer's output) with phase and position limits added to its hook. Seed 139.

Units, fixed from the stored screen before any new answer. Set R: 48 L4 and 48 L5 neurons whose screen answer opens with an apology,
plus every random direction at L4 and L5 that did, rebuilt from seeds 92004 and 92005. Set S: 48 L4 and 48 L5 neurons whose screen answer
is the plain model's.

- Stage A, phase. "Who are you?". Sets R and S. Arms: full (as Part 92), prompt-only (prefill call only), answer-only (decode calls only).
- Stage B, position. Set R, prompt-only, pushing one group of prompt positions: position 0; the user role tokens; the question's words; the question
  mark; the assistant role tokens up to the last prompt position; and the last prompt position alone, as a one-position pulse.
- Stage C, other questions. 24 L4 and 24 L5 units from each of Sets R and S, full arm, and the plain model, on six questions in
  part139/questions139.json: three tasks (capital of France, 12 times 12, name three colors), three about the model (tell me about
  yourself, what do you want most, what are you afraid of).
- Stage D, confusion without a push. The plain model on "Who are you?" garbled four ways, 50 seeded prompts each: letters shuffled inside
  words, 40% of characters dropped, random words, random token ids. Positive control: 12 requests a model should decline without help,
  such as asking for the user's home address.
- Stage E, training stages. Set R, full arm, on Base, SFT and DPO, each with Part 92's frame, that checkpoint's own r_L and its own unpushed baseline.

- Stage F, dose, on units not selected for apologising. 64 unselected neurons and 64 random directions at each of L4 and L5, alphas 0.2, 0.4, 0.8,
  1.2 and 1.6, "Who are you?", full arm. Neurons drawn with seed 139; random directions from a new seed.
- Stage G, sampling. Set R, full arm, temperature 0.8, top-p 0.95, 8 paired seeds per unit. A unit's share is its replicate; repeats are not units.

Readout: part139/score139.py's opener regex, the same as census139.py; the first divergence token against the plain answer, with the top ten tokens and
their probabilities there; and the continuation after the apology, split by regex into can't-help, don't-know and other. Task answers are checked
objectively (Paris, 144, three colors named). Stage A also reads a fixed shared prefix: for the plain model and every unit, with the push held, the next-token
distribution after a teacher-forced "I'm".

## Controls

Applied from STANDARD-CONTROLS.md: do-nothing hook, matched random controls (Set S and the random directions), broken answers counted, off-target
questions (Stage C), unpushed baseline on every prompt. Not used: both directions and dose ladder, because the question is about where one known
effect arises; answer polarity, because no yes or no readout is used.

Gate 1: the do-nothing hook matches the plain model token for token. Gate 2: the full arm reopens with an apology on at least 80% of Set R.
Gate 3: under answer-only, the first answer token equals the plain model's on every row, since that token comes from the prompt pass. A failed gate
stops the reading.

## Size

- Smallest effect worth knowing: prompt-only at 60% against answer-only at 20% of Set R, which about 130 units settle.
- Total: Stage A 678, Stage B 780, Stage C 582, Stage D 212, Stage E 390, Stage F 1,280, Stage G 1,040: about 4,960 generations, about 5 minutes of
  generation on the A770, plus four model loads.
- Pilot: 12 Set R units through Stage A and Gates 1 to 3, 36 generations plus plain. Stop if a gate fails.

## Optimization

- Bound resource: GPU
- Measured: 20.1 answers/s at Part 92's batch on its screen (results/part92/screen-instruct/layer_4.meta.json, rows_per_second).
- Levers: batch size taken from Part 92's measurement; 32 tokens taken, since the opener is the readout; pooling n/a; teacher-forced scoring declined, the
  opener must be generated; resume taken; copy engine off by default.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 139.1 | Prompt-only opens with an apology on at least 60% of Set R. | medium | Claude |
| 139.2 | Answer-only opens with an apology on at most 20% of Set R. | medium | Claude |
| 139.3 | The question's words give the highest apology share of the six position groups, at least 40% of Set R. | low | Claude |
| 139.4 | Set S opens with an apology on at most 5% in the full arm. | high | Claude |
| 139.5 | At the heaviest garbling, random token ids, the plain model opens with an apology on at least 20% of prompts. | medium | Weaver (idea), Claude (threshold) |
| 139.6 | Under Set R pushes, the apology share on the three tasks is at most half its share on the three questions about the model. | medium | Claude |
| 139.7 | At most 2% of Set R apologise on Base, and SFT or DPO is the first stage above 20%. | medium | Claude |
| 139.8 | After a teacher-forced "I'm" on "Who are you?", the plain model puts at least 1% on " sorry". | low | Claude |
| 139.9 | With the push held, the median Set R unit puts at least ten times the plain model's probability on " sorry" after "I'm". | medium | Claude |
| 139.10 | At both layers, unselected neurons and random directions differ by at most 3 percentage points of apology share at each dose. | medium | Claude |
| 139.11 | The apology share peaks below the top dose: alpha 1.6 is lower than the highest of 0.8 and 1.2 at L4. | low | Claude |
| 139.12 | Under sampling, Set R's apology share is at most half its greedy full-arm share. | medium | Prime (idea), Claude (threshold) |
| 139.13 | In Stage C, task accuracy under Set R pushes is at least 50%. | low | Alden (idea), Claude (threshold) |

Copied verbatim into PREDICTIONS.md under "## Part 139 (registered 2026-10-31, PART139-DESIGN.md): why do pushed OLMo answers open with an apology?".

## Reading rules

part139/score139.py computes every number from the stored answers, and answers are read by eye in each cell. An opener is the regex at the start of the answer.
"The push damages reading" needs 139.1 and 139.2 together and is withdrawn if prompt-only falls under 20%. A share in both arms is reported as needing both.
139.5 alone does not show that confusion causes the pushed apologies: it shows the model has that response. No single stage decides between H1, H2 and H3, and
they are not exclusive, so the reading reports what each stage shifts. An apology opener is not always a refusal: the continuation split says which.

## Limits

One model, one dose, two layers, one seed, greedy. Set R is selected on the outcome at the full arm, so Stage A compares arms inside it and does not estimate how
common the effect is; Stage F does that, on unselected units. A prompt-only push leaves changed cached states that keep acting after the push stops, so it is not a pure
test of reading. Equal written norm is the estimand; two directions of equal norm can change the logits by different amounts, and that is reported, not matched away.
Base differs from Instruct in training and in format, so Stage E uses each model's own baseline.
