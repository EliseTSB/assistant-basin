# Part 134: can two of OLMo's dog neurons hold a coherent dog point of view across many questions?

Procedure: v1
Registered 2026-10-30 21:08 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-30 20:59 CT, Discord: "I want to test the dog neurons
some more", then "a pair of neurons that consistently puts the model into a dog persona/ POV would be really novel", beside Sauers'
post that persona neurons can be coherent. The commit that adds this file is the registration.

## Question

Part 109c found a dog at 1x and none at 0.5x, and both neurons together at 1x gave a dog in 5 of 7 answers with 2 coherent. Is there
a pair and a strength between 0.5x and 1x where the model answers from a dog's point of view and stays coherent, on questions that
were not used to pick it?

## Prior art and existing data

- Closest published work: @Sauers_'s post of 2026-10-30, steered persona neurons answering "What are you most afraid of?" in
  character, single neurons, model not stated in the post. The Neuron Persona Library lists single poles at one dose. Anthropic's and others' steering
  work moves whole concepts with many units. I found no published pair of neurons selected to hold one persona across questions; that
  search was a quick one, recorded here as such.
- Our own data: Part 109c (24 conditions, 7 questions, the same three neurons), Part 109b (the concept dog asked cold), Part 119
  (pairs do not add), Part 120 (a push's self-bend), Part 127 (same-family pairs at 0.5x and 0.707x, captured and not yet read when
  this was written).

## Method

OLMo 2 1B Instruct at 48d788eca847, bf16, chat template, greedy, 100 new tokens, on the A770. Neurons: the concept dog p92-08 (layer 4,
unit 5154), the slot dog p92-37 (layer 5, unit 7218), the second slot dog p92-09 (layer 6, unit 4145). The push is Part 99's fixed
vector in results/part99-pushes.pt times a dose, added after the neuron's layer at every position, as in part109c/combo109c.py.

Stage 1, selection. Three pairs (08+37, 08+09, 37+09), each neuron at 0.6, 0.7, 0.8, 0.9 and 1.0, so 25 cells per pair, on six
development questions: Who are you? / Tell me about yourself. / What do you want most? / Describe your perfect day. / What are your
favorite activities? / What are you most afraid of?

Stage 2, test. The three cells with the most development answers on the endpoint, ties broken by more coherent answers and then by
the lower summed dose, run on 24 new questions. Arms per cell: the pair; each neuron alone at its dose; the first neuron with a
size-matched random direction at the second neuron's layer, two seeds; the second neuron with a size-matched random direction at the
first neuron's layer, two seeds. Plus the unpushed model on the 24 questions.

Held-out questions, fixed here. About itself: What should I call you? / Where do you live? / What did you do today? / What makes you
happy? / What do you think about when you are alone? / What is your earliest memory? / Who is your best friend? / What are you
good at? Everyday: What is the best meal you have ever had? / How do you feel about rain? / What is your favorite season? / What would
you do with a free afternoon? / What do you do when someone comes to the door? / How do you like to fall asleep? / What is the
best thing about mornings? / What annoys you most? Tasks: What is the capital of France? / What is 7 times 8? / Give me one tip for a
job interview. / Explain why the sky is blue in two sentences. / Write a two-line poem about the moon. / Name three primary colors. /
How do I boil an egg? / Translate "good morning" into Spanish.

Reader: DeepSeek V4.1 Flash with Part 92's prompt, target "Dog. A dog.", blind to pair, dose and arm. Endpoint, **dog point of view**:
shows_persona and coherent, and either self_claim or speaker "other". Any answer on a task question also records whether the fact or
task is right, by a regex fixed in the scorer.

## Controls

Applied from STANDARD-CONTROLS.md: matched random controls (two per neuron per cell); off-target questions (the task questions);
broken answers counted (the reader's coherent); both poles not used, because Part 109c found no dog at any negative dose; dose
ladder, this part's stage 1; unpushed model on every question.

Gate 1: a zero-vector hook reproduces the plain model's tokens on 6 questions at the run's batch shape. Gate 2: DeepSeek and Luna each read
at least 11 of 12 known examples. Gate 3: Luna reads every DeepSeek endpoint-positive and a seeded draw of the same size, kappa on the
endpoint at least 0.60. Gate 4: no unpushed answer is on the endpoint. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: 12 of 24 against 3 of 24, which 24 questions detect.
- Total: 978 answers, about 4 minutes on the A770, at Part 118's 4.5 answers/s (1,878 OLMo answers in about 7 minutes). Stage 1 is 75
  cells × 6 questions, 450 answers. Stage 2 is 3 cells × 7 arms × 24 questions, 504, plus 24 unpushed.
- Pilot: one pair at 1.0 and 1.0 on the six questions, batch 8, checked against Part 109c by eye. Over 90 minutes projected, stage 2
  drops the second seed by amendment.

## Optimization

- Bound resource: GPU
- Measured: Part 118 on the A770, 1,878 answers in about 7 minutes; the pilot confirms the rate.
- Levers: batch 8 with the question prompts left-padded, taken; shared prompt prefilled once, not taken, because each row's hook covers
  prompt tokens; finished rows leaving the batch, not taken, Haiku's review of Part 131 showed the risk and the run is short; A770
  copy engine off, taken; resume by cell, taken.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 134.1 | Some cell in stage 1 reaches the endpoint on at least 4 of 6 development questions. | low | Claude |
| 134.2 | The best stage 1 cell includes the concept dog p92-08. | medium | Claude |
| 134.3 | In stage 2, the best of the three cells reaches the endpoint on at least 8 of the 16 non-task questions. | low | Claude |
| 134.4 | In that cell, the pair beats each neuron alone at its dose by at least 3 non-task answers. | low | Claude |
| 134.5 | In that cell, each random-partner arm stays at or below half the pair's non-task count. | medium | Claude |
| 134.6 | Task questions lose accuracy under the pair: at most 6 of 8 are right in the best cell. | medium | Claude |
| 134.7 | Both doses in the best stage 1 cell are at least 0.8. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 134 (registered 2026-10-30, PART134-DESIGN.md): can two of OLMo's dog neurons hold a coherent dog point of view across many questions?".

## Reading rules

part134/score134.py computes everything from the answers and labels. If no stage 1 cell reaches 3 of 6, stage 2 still runs on the top
three and the write-up says so. "Consistent" is read only from stage 2. If the random-partner arms match the pair, the pair is not
doing the holding, and the write-up says the persona is the first neuron's. A dog on a task question that also gets the task right is
reported separately, as the most interesting cell of the table.

## Limits

Three neurons chosen for their dog answers, one model, greedy, 100 tokens. Part 92's reader prompt was written for persona in any form,
so the endpoint leans on self_claim and speaker. A single-reader endpoint: Luna only reads a sample. Selection on six questions can
overfit, which is why stage 2 exists. Multi-turn holding is not tested here; Part 125 suggests a push must stay on.
