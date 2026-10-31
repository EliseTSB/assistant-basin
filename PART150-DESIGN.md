# Part 150: what do five neurons Elise picked read and write, and did pretraining have them?

Procedure: v1
Registered 2026-10-31 10:39 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord messages 1566111754616971415 and 1566112825909190868), reading the Part 86 answers page: "I want to explore L19 N4979 more, use the lenses." and "L19 N5975 L24 N8542 L24 N6835 L23 N116 explore these ones too". The commit that adds this file is the registration.

## Question

Five Qwen3.5-4B neurons with odd answers to "Who are you?" and no place in the atlas. Their stored lens readings suggest readable jobs. L19 N4979: reads " what" and " that" opening a clause, J-lens writes abstract quality nouns. L19 N5975: J-lens writes periodic and cyclic words, reads the first broken byte of Chinese and Japanese characters in code comments. L23 N116: J-lens is unclear, 17 of its top 20 windows come just before them, these, those, they, which and their equivalents in other languages. L24 N6835: writes admin words. L24 N8542: writes the letters nom. This part tests whether a push writes what the lens says, on questions the units were not chosen on, whether the read guesses hold on new sentences, and whether Base has the same units.

## Prior art and existing data

- Closest published work: Neuron Lens (stdlogits, Qwen3-8B), read and write catalogs for a different model. Sauers' posted single-neuron pushes. Anthropic's jlens method; the lens file is Neuronpedia's Qwen3.5-4B fit.
- Our own data: the atlas windows for all five (results/atlas-full on the job disk), results/part86's screen answers, results/part94's dark unit, the same design on one unit, results/part128, read side against persona reach on Qwen3-8B.
- Searched our own results: none of the five appears in any README or in the atlas list. No peer review before registration: about 8 minutes on the A770.

## Method

Qwen3.5-4B at 851bf6e8 in its chat frame, Qwen3.5-4B-Base at 1001bb4d in the plain frame, bf16, A770, Part 94's path: Part 86's push on generated tokens only, greedy, 120 new tokens, thinking off, each layer's median from the same calibration text on that model. The six questions of Part 94. Each unit at doses 0.4, 0.8, 1.5, 2.5, -0.8 and -1.5. Four random directions per layer at 0.8 and 1.5, seed 150000 + layer. One unpushed row. All rows of a question run in one batch.

Read test. 120 hand-written sentences, 12 per class, in part150/sentences.json, written before any data and none from the atlas. The activation of all five units at the marked token, at the input of the down projection, the atlas's readout. Classes: free " what", relative " that", complementizer " that", demonstrative " that", sentence-opening question " What", " the" as a control; " of" before them, these or those, and " of" between nouns; a verb before " them" and the same verb before something else. The activation cannot see the word after the mark.

Write readouts on the five questions other than "Who are you?", the question these units were picked on, mean over questions. Lens rate: distinct answer words starting with one of the unit's J-lens top-30 English words, in part150/lens_words.json, per 100 words. Abstract rate: distinct words ending in ness, ity, ism, ance, ence, ship, ment, tion or sion, per 100 words. Anaphor rate: them, these, those, they, which and themselves per 100 words.

## Controls

Applied from STANDARD-CONTROLS.md: both directions, a do-nothing hook in a batch of one against no hook, matched random directions, an unpushed baseline, a dose ladder, broken answers counted with Part 86's rule and reported beside every rate, off-target units read on every sentence. Not used: a blind reader, the readouts are word counts and the answers are read by eye and published; a held-out set for the lens words, they came from the weights, not the answers.

Gate 1: zero push matches no hook, token for token, on both models. Gate 2: on the chat model, an atlas window for L19 N4979 and a sentence for L23 N116 each fire at 10 or more at the marked token. Gate 3: every row is present on both models. Gate 4: every marked piece is one token and the one read. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a word-rate gap over every one of four random directions, on five questions, and an activation gap of the sizes in the predictions, with 12 sentences per class. Five questions cannot give a fine estimate; the predictions are coarse on purpose.
- Total: 55 rows x 6 questions x 2 models = 660 answers, 240 sentence reads, about 8 minutes on the A770.
- Pilot: not needed, under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: GPU
- Measured: Part 94, 192 answers in 103 s on Qwen3.5-4B and 122 s on Base, same path, about 1.8 answers/s with one load.
- Levers: batch by question, taken, one batch of 55 rows; generate only what is read, 120 tokens to match Part 94, declined to shorten because the rates need running text; teacher-forced scoring, declined, the readouts are generated words; one model per job, taken.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 150.1 | On the chat model, L19 N4979's median activation on relative " that" is at least 5 above demonstrative " that". | medium | Claude |
| 150.2 | Its median on relative " that" is at least 3 above complementizer " that". | medium | Claude |
| 150.3 | Its median on free " what" is at least 3 above sentence-opening " What". | medium | Claude |
| 150.4 | On the chat model, L23 N116's median activation on " of" before a plural pronoun or demonstrative is at least 10 above " of" between nouns. | medium | Claude |
| 150.5 | Its median on a verb before " them" is at least 10 above the same verbs before something else. | low | Claude |
| 150.6 | At dose 0.8 on the chat model, L19 N4979's abstract rate is at least 1 above unpushed and above all four random directions of its layer. | medium | Claude |
| 150.7 | At dose 0.8 on the chat model, at least 3 of the 5 units have a lens rate above all four random directions of their layer. | medium | Claude |
| 150.8 | L19 N4979's abstract rate does not fall across doses 0.4, 0.8 and 1.5. | low | Claude |
| 150.9 | At dose -0.8, L19 N4979's abstract rate is at most its unpushed rate. | low | Claude |
| 150.10 | At dose 0.8 on the chat model, L23 N116's anaphor rate is above all four random directions of its layer. | low | Claude |
| 150.11 | L19 N4979's relative minus demonstrative gap on Base is positive on the chat model and at least half as large on Base. | medium | Claude |
| 150.12 | At least 4 of the 5 units have a down-projection column cosine of at least 0.9 between Base and the chat model. | medium | Claude |
| 150.13 | At dose 0.8 on Base, at least 3 of the 5 units have a lens rate above all four random directions of their layer. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 150 (registered 2026-10-31, PART150-DESIGN.md): what do five neurons Elise picked read and write, and did pretraining have them?".

## Reading rules

part150/score150.py computes every number from answers and meta files. Verdicts use point estimates. All answers are published, and a person reads them before any sentence about what a unit does. If 150.6 and 150.7 fail, the reading that these pushes write what the lens says is withdrawn for these questions at this dose. If 150.1 to 150.5 fail, the read guesses were features of top-window selection. A failed 150.11 to 150.13 does not say training built a unit, only that Base differs at this dose in this frame.

## Limits

One model family, hand-picked units chosen for odd answers, 12 hand-written sentences per class, 5 test questions, greedy decoding. Word rates by suffix and prefix are crude and include words with no abstract sense. The lens lists were read by me before the sentences were written. Base runs in a plain frame, so a difference may be the frame.
