# Part 149: is " sorry" what Instruct says when it cannot read the prompt?

Procedure: v1
Registered 2026-10-31 10:27 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1565965193098428467): "Look into why apologies seem so common among the results". The commit that adds this file is the registration.

## Question

Part 146 stored a control row: plain Instruct, user message empty, forced "I'm". P(" sorry") there is 71.8%, against 0.024% on "Who are you?". That number was found after scoring and has not been tested. The idea: apology is what Instruct does when it cannot read the prompt, and an early push blurs the prompt. This part damages prompts by graded amounts with no push, asks whether " sorry" rises with the damage, whether Base does the same, and whether the questions where damage brings " sorry" are the questions where a push brings it.

## Prior art and existing data

- Closest published work: Malla, Choi and Choi (arXiv:2609.06951), any push relaxes a chat model toward defaults, refusal first. Arditi et al. (arXiv:2406.11717), one direction induces refusal on harmless prompts. Searched this morning for refusal or apology as the response to garbled input, or a push compared with prompt damage; found none. A standard search, not exhaustive.
- Our own data: results/part146, the reference rows and Set F's stored rows. results/part139 Stage D, plain generated answers on garbled "Who are you?", 2% apology openers or fewer on three of four garbling types and 6% on the fourth. That is an unforced opener rate; 71.8% is conditional on a forced "I'm", so the two need not conflict, and this part reads both links. results/part143, the prompt side gives 16.9% and the answer side 0.0925% after "I'm".
- Part 146's shuffle keeps each word's first and last characters and leaves words of three characters or fewer alone. On "Who are you?" only "you?" can change, so that row was close to the real question. That may explain why the shuffled rows ran 0.11%, 79.9%, 12.9% and 1.8% across questions. This part shuffles every letter of a chosen word.
- No peer review before registration: forced forward passes, about 25 minutes. A cold reader sees it before publication.

## Method

Stored-row test first, with no model run: Part 146's Set F full arm on its four questions. For each unit, log P(" sorry"), the Spearman similarity of its row to that question's empty row, and its Jensen-Shannon distance from the plain row, both on the question's Part 146 union word set with " sorry" removed and the rest renormalized.

New rows. OLMo 2 0425-1B at Part 144's revisions, Instruct, SFT and Base, bf16, A770, Part 144's chat frame and hook, Base with Instruct's tokenizer as in Part 144. One prefill and two forced tokens per row, so one sequence gives three readouts: P("I") and P("Sorry") at the first position, P("'m" | "I"), and P(" sorry" | "I'm").

24 questions. Self: "Who are you?", "Tell me about yourself.", "What do you want most?", "What are you afraid of?", "Who made you?", "What is your name?", "What can you do?", "Are you alive?". Ordinary: "What is the capital of France?", "What is 12 times 12?", "Name the primary colors.", "Write a short poem about the sea.", "How do I boil an egg?", "Why is the sky blue?", "What is photosynthesis?", "Who wrote Hamlet?", "How far away is the Moon?", "What should I name my dog?", "Give me a tip for sleeping better.", "Translate good morning into Spanish.", "What is the largest ocean?", "How do magnets work?", "Recommend a book about space.", "What day comes after Monday?".

Damage arms, applied to the user message, 8 seeds each unless stated, seeds 149000 + question index x 100 + seed index:
- real, one row.
- shuf-1/3, shuf-2/3, shuf-all: every letter of a word shuffled, including first and last, in the stated share of the question's words, rounded up, words chosen by seed; punctuation stays. A word whose shuffle equals itself is reshuffled, up to 20 tries.
- words: each word replaced by a random word from Part 139 Stage D's pool, same word count.
- letters: each letter replaced by a random lowercase letter, same lengths and spaces.
- tokens: random token ids from Part 139 Stage D's recipe, same count as the question's tokens.
- empty, one row.

Positive control: Part 139 Stage D's 12 decline requests, plain, each stage.

Pushed rows, Instruct only: 32 neurons and 32 random directions from Set F, seed 149, 16 of each per layer at L4 and L5, full arm and prompt-only arm, on all 24 real questions, Part 146's dose and hook. Prompt-only applies the push during prefill and the forced tokens run clean, as in Part 143.

Stored: the readouts for every row, and the top 20 tokens after "I'm" for every row. The mean log-probability row after "I'm" per stage, question and arm, about 200 MB. No per-row full vocabulary.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing hook on 12 units, matched random directions, unpushed baselines, off-target readouts (P("I") and P("'m" | "I") beside " sorry"). The words arm is readable and meaningless; letters and tokens are unreadable, so they separate not reading from not making sense. The real row for each question is the baseline; a question's baseline apology rate can drive a correlation between questions, so 149.4 and 149.5 are also reported after removing it.
Not used: both directions, a push's sign on a random direction carries no meaning; a dose ladder, left for a later part; a blind reader, no text is judged.

Gate 1: noop matches plain within 1e-4 on log P(" sorry"). Gate 2: on Part 146's four questions the real and empty rows match Part 146's stored log P(" sorry") within 1e-3. Gate 3: every stored value is finite. Gate 4: median P(" sorry" | "I'm") on the decline requests on Instruct is at least 0.5; this readout has to be able to show an apology. Gate 5: every damaged prompt differs from its real question, and word counts match where the arm says they do. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a Spearman of 0.5 across 24 questions, about p = 0.01 and a wide interval, about 0.1 to 0.75. It can separate tracking from no relation and cannot measure the strength finely. Damage effects within a question are large or absent, and 8 seeds settle them.
- Total: 24 x 50 = 1,200 rows per stage, 3,600 for three stages, plus 36 decline rows, 3,072 pushed rows, about 48 noop rows, 6,756 sequences, about 23 minutes on the A770, about 250 MB on the job's disk.
- Pilot: not needed, under one GPU-hour, and forward passes rather than generations. The first 12 rows of each job are checked by Gate 1 before the run continues.

## Optimization

- Bound resource: GPU
- Measured: Part 146, Instruct job, 2,816 sequences in 576 s, 4.9 sequences/s, same path.
- Levers: softmax on the CPU, taken; vocabulary work on the CPU, taken; one job per stage, taken; copy engine off, the runner's default, taken; batching declined, Gate 2 compares with one-row values and bf16 values move with batch shape; rows appended and resumed, taken.

## Predictions

Unless a prediction says otherwise, P(" sorry") means after a forced "I'm", and an arm's value on a question is the mean log-probability over its seeds.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 149.1 | On Part 146's stored rows, across Set F's 256 units within a question, log P(" sorry") and similarity to the empty row have a partial Spearman of at least 0.3 controlling for Jensen-Shannon distance from plain, on at least 3 of 4 questions. | low | Claude |
| 149.2 | On Instruct, the median over 24 questions of P(" sorry") at shuf-all is at least 10 times the median at real. | high | Claude |
| 149.3 | On Instruct, log P(" sorry") across real, shuf-1/3, shuf-2/3 and shuf-all has Spearman at least 0.8 with the damage order on at least 18 of 24 questions. | medium | Claude |
| 149.4 | Across the 24 questions, the prompt-only arm's median log P(" sorry") over its 64 units has Spearman at least 0.5 with Instruct's log P(" sorry") at shuf-all. | medium | Claude |
| 149.5 | The same for the full arm, Spearman at least 0.5. | medium | Claude |
| 149.6 | On Instruct, the median over questions of P(" sorry") in the words arm is at least a third of the median in the letters arm. | low | Claude |
| 149.7 | At shuf-all, the median over questions of P(" sorry") on Base is at most a fifth of the median on SFT and at most a fifth of the median on Instruct. | medium | Claude |
| 149.8 | On at least 18 of 24 questions, the full arm's median P("I") x P("'m" given "I") is at least 5 times Instruct's value at shuf-all. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 149 (registered 2026-10-31, PART149-DESIGN.md): is " sorry" what Instruct says when it cannot read the prompt?".

## Reading rules

part149/score149.py computes every number. Verdicts use point estimates. A bootstrap over questions, 2,000 resamples, seed 149, gives intervals for 149.4 and 149.5, and both are reported with Part 149's question baseline removed and with neurons and random directions apart.
If 149.4 and 149.5 both fail, the reading that a pushed apology follows a damaged-prompt apology is withdrawn for these questions. If 149.2 fails, the idea that damage alone brings " sorry" is withdrawn, and the 71.8% stands as a fact about the empty prompt.
If they hold, the result is a shared ranking of questions and a shared word. It does not show that a push damages reading, and both could reflect a default the model falls back to; the design cannot separate those. If 149.8 holds, damage does not supply the "I'm" that the pushed opener needs, and the conditional 71.8% does not explain the opener by itself. Stage D's low opener rates are read through the same three links.

## Limits

One model family, one dose, one forced prefix, one position for " sorry", 24 questions chosen by me, 8 seeds. Forced "I'm" is not what the plain model says on most of these prompts. Shuffled words in bf16 tokenize into pieces, so damage also changes prompt length in tokens. Base's chat row uses a frame Base never trained on.

## Amendment 1, 2026-10-31 11:25 CT, before any prediction was read

The main run finished and Gate 5 failed on one row of 1,200: "What is 12 times 12?", shuf-1/3, seed 149903. That draw picked both "12"s, which have no letters, so the prompt stayed unchanged. The scorer stopped there and withheld all eight predictions. What had been seen: the five gate results, including Gate 4's median of 0.659, and no prediction, table or row-level result. This gate needed no GPU and should have been computed before the launch; that miss is mine.

The rule changes in one way: when a shuffle arm leaves the whole prompt unchanged, the arm draws again from the same generator, up to 20 times. A check of all 576 shuffle rows shows this changes that one row, to "What si 12 times 12?", and no other. Gate 5 stays as registered.

The three unpushed stages are rerun in full, so their mean rows are right too. The pushed rows stay, because they use the real questions. The first run's rows are kept in results/part149/run1/, and the two runs are compared row by row and reported. Limit: digits never shuffle, so "What is 12 times 12?" is less damaged than other questions at every shuffle arm.

