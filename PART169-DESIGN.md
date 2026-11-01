# Part 169: does a ban add loops by blocking what the model was going to say?

Procedure: v1
Registered 2026-11-01 05:29 CT by Claude (Opus 5.5), before any data for this part. Elise, 2026-11-01: "keep the research going, feel free to get the rest of the council involved" (Discord 1566330393123225731). Weaver's fifth ban and predictions are theirs, from #council-room, 2026-11-01, and their review is folded below. Alden was asked (council topic overnight-ideas-2026-11-01, message 16) and had not answered at registration; a later review becomes an amendment or a note on the results.

## Question

In Part 168 a list matched to the own list on mean probability mass raised loops by 16.5%, the opposite of the cut Weaver predicted, while the own list cut them by 31.4%. The mass list held 1.2 tokens of the plain answer per row and the own list 0.07. Does a ban add loops because it blocks tokens of the answer the model was going to give? And does the own ban's cut hold on new neurons and new phrasings, given that it cleared its bar by 1.4 points?

## Prior art and existing data

- Part 168 and its review (`results/part168/review168.txt`), exploratory: the top own-mass quartile with two or more plain-answer tokens on the mass list holds 281 of 1,344 rows and has 97, 126 and 44 loops under none, mass-matched and own. The other 1,063 rows have 91, 93 and 85. In the two own-mass tenths where both exist, rows with two or more plain tokens on the mass list have 36, 46 and 18 loops, and the rest have 20, 19 and 19. So the split is not simply own mass, though on few rows. Slot loop counts in n+ and k4 correlate at -0.002, though k4 holds its slot's n+ neuron.
- Part 168: mass lists hold about 1.5% function words and punctuation, so Weaver's function-word reading failed on stored data. Own induced 42 loops and mass-matched 75.
- Part 164: random directions break 7.8% of answers at this dose.
- Muse's search (council topic muse-part165-prior-art-2026-11-01, messages 5 and 7) found no study banning a fixed token list under greedy decoding and measuring loops by overlap with the unconstrained continuation. Closest, through Muse and not read here: Xu et al. (arXiv:2206.02369), repetition under greedy decoding self-reinforces, so a ban promotes the next token, which may sit on a loop; Holtzman et al. (arXiv:1904.09751), greedy loops with no ban, so the no-ban control is the experiment; Nguyen et al. (arXiv:2610.35328), masking is provably harmless only when the true continuation stays allowed; Jin and Han (arXiv:2605.10065), hard negative masks under greedy decoding on safety strings, with no loop measure; Welleck et al. (arXiv:1908.04319), unlikelihood training lowers repetition, which is not an inference ban. Their caution, taken here: mass-matched is not overlap-matched, so free, clear and hit vary overlap, and hit gives up the mass match.

## Method

Model: Qwen3.5-4B, same pin as Part 168. Layers 19 to 26, 12 slots per layer, seed base 169000 by Part 165's recipe, with Part 168's redraw, so no n+ neuron repeats one of Parts 165, 167 or 168 at that layer. Arms: n+ and k4 at Part 165's write norm. The full arm is dropped; Part 168 answered it. Phrasings 0 to 7, new to these bans, and all unbroken in Part 164's stored plain answers. That is 192 slots by 8 phrasings, 1,536 rows.

Stage 0, plain: each phrasing's plain answer to 80 tokens, greedy. Gate 2 is checked here first.

Stage 1, bias: Part 166's teacher-forced forward pass on the plain 40-token answer gives each row's mean logit bias and its top-200 tokens.

Stage 1b, lists, at Stage 1's contexts with q the row's mean pushed probability:

| List | Tokens | Calibration |
| --- | --- | --- |
| own | the row's top 50 bias tokens | none |
| free | Part 168's mass list: 50 tokens from the 20,000 most probable outside the own list and the next slot's own list, as in Part 168 | Part 168's Amendments 1 and 2, to own mass |
| clear | the same recipe, with every token of the phrasing's plain 80-token answer also removed from the pool | same, to own mass; a row out of reach is logged, not forced |
| hit | the free list's tokens that are in the plain 80-token answer | none; a row with no such token has no hit answer |
| plain50 | the 50 tokens with the largest mean probability on the plain 40-token path with no push, per phrasing | none; Weaver's fifth ban, a direct measure of the cost of blocking the continuation |

Special tokens, EOS and every added token in the tokenizer are removed from every list, which closes Part 168's `</think>` deviation. Each list is fixed before generation, from Stages 1 and 1b only, and seeded from arm, layer, slot, phrasing and list name. Logged per row: each list's mass, exponent, draws, in-tolerance flag, repaired flag, feasible flag, plain overlap, median rank under q, and count of tokens in the row's top 200 under q.

Two sets are frozen in `masses.jsonl` before generation. Paired rows: free and clear both feasible and in tolerance. High rows: own mass at least 0.0398, Part 168's upper-quartile cut, a fixed number and not this run's quartile.

Stage 2, generation: 80 tokens, greedy, same hook as Part 165.

| Ban | Applied to | Answers |
| --- | --- | --- |
| none, own, free, clear, plain50 | all 1,536 rows | 7,680 |
| hit | rows whose free list holds a plain token | about 700 |
| no push, free | the 768 n+ rows' free lists, hook off | 768 |
| no push, plain50 and none | each phrasing | 16 |

Measures: broken and loop by Part 165's rules, with Part 166's loop phrase. Each answer also logs every step where its ban changed the greedy choice, and the rank of the chosen token among the unbanned scores; this is exploratory and decides no verdict. An answer with no such step should equal its none answer, so the count that differ measures batch-shape noise and not a second channel. Induced: a loop where none had none, over rows with no loop under none. Ended, the converse. Length, EOS rate and joint success sit beside every loop rate. Coherence: a blinded hand read of 24 answers per ban on low-k rows.

## Controls

- Gate 1, rows: all bias rows and battery answers present, unique, finite and nonempty.
- Gate 2, no push: the 8 plain answers are unbroken by Part 165's primary rule and have no loop at 80 tokens. Checked before launch on Part 164's stored answers and at Stage 0, which stops the run on failure.
- Gate 3, size: every write norm within 1% of target, and realized dose logged as in Part 165.
- Gate 4, ban works: no banned answer holds a token on its row's list, on generated ids, and every stored list equals its row's list.
- Gate 5, room: the none arm's n+ broken rate and the pooled none loop rate are each at least 0.08.
- Gate 6, free lists: at least 90% of rows have a free list within tolerance. Checked after Stage 1b, before generation, and the run stops on failure.
- Gate 7, freshness: no n+ neuron equals one from Parts 165, 167 or 168 at its layer, or repeats here.
- Gate 8, contrast exists, checked before generation: no clear list holds a token of the plain 80-token answer; on high rows free lists hold a mean of at least 1 plain token; no list holds an added token; and at least 150 high rows are paired. If fewer are paired, the run still proceeds, and 169.4 and 169.5 are recorded as not testable, because clear lists cannot reach own mass where the mass is on plain tokens. That would itself be a result about these rows.
- A failed gate stops all readings.
- Bars that can fail: 169.1 fails under a 25% cut. 169.2 fails if the own cut is spread evenly. 169.3 fails if loops rise by under 5%. 169.4 and 169.5 fail if a list with no plain tokens adds loops as the free list does. 169.6 fails if a list adds loops with no push at all. 169.7 fails if the clear ban moves loops either way by over 15%. 169.8 fails if the rest of the list does the work.
- Not used: both directions and poles, since no neuron's meaning is claimed. Not used: dose ladder, since Part 165 fixed the dose. Not used: Muse's one-step ban, block once then release, which would separate self-sustaining loops from loops that need continuous blocking; it is a follow-up if 169.4 holds.

## Size

- Smallest effect worth knowing: a 0.02 gap in the loop rate between free and clear at a rate near 0.15, read on 1,536 paired answers per ban, and a 0.10 gap in induced rate on about 300 high rows.
- Total: 1,536 forward passes twice, plus about 9,164 generations, plus 8 plain, about 58 minutes on the A770.
- Pilot: the first 3 generation batches. Stop rule: under 1.5 answers per second.

## Optimization

- Bound resource: GPU.
- Measured: Part 168, 7,392 generations from 03:08:06 to 03:54:10 CST, about 2.7 answers per second with Stages 1 and 1b inside that time.
- Levers: full arm dropped, taken. matched ban dropped, since Part 168 showed it is close to no ban, taken. Stage 1 reuses Part 166's path, taken. One position's logits at a time, taken. Lists drawn on the CPU from the top 20,000 tokens, taken. Copy engine off, the job default. hit run only on rows that have a hit list, taken.
- Host work: lists after q reaches the CPU, and scoring, need no model.
- The block log adds one comparison per step on the device and two small CPU transfers. The pilot's rate covers it.

## Predictions

All pooled over n+ and k4. Verdicts are point estimates against the bars; whole-slot bootstrap intervals, stratified by layer, seed 169, 10,000 replicates, are reported beside them, with n+ and k4 each alone, since k4 contains its slot's n+ neuron.

| # | Prediction | Confidence |
| --- | --- | --- |
| 169.1 | Own ban cuts loop answers by at least 25% against none | medium |
| 169.2 | Own's relative cut on high rows is at least 0.25 above its relative cut on other rows | medium |
| 169.3 | Free ban raises loop answers by at least 5% against none | medium |
| 169.4 | On paired rows, clear's loop rate is at least 0.02 below free's | medium |
| 169.5 | On high paired rows with no loop under none, free's induced rate is at least 0.10 above clear's | medium |
| 169.6 | With no push, free lists give loops on under 5% of answers | medium |
| 169.7 | Clear's change in loop answers against none, on paired rows, is between -15% and 15% | low |
| 169.8 | On rows whose free list holds two or more plain tokens and with no loop under none, hit's induced rate is at least 0.6 times free's | medium |

Weaver's, from their review in #council-room, 2026-11-01. plain50 is built per phrasing, so its per-row dose varies, and they ask for the correlation rather than the average: "The average is uninteresting; the correlation is the test." 169.W1 registers that as a median split, with the Spearman correlation reported beside it. Their falsifier: "If plain50's induced count is at or below free's *and* does not correlate with per-row overlap, then 'cost of blocking the continuation' isn't a function of the continuation, and my decomposition collapses rather than adjusts."

| # | Prediction | Confidence |
| --- | --- | --- |
| 169.W1 | Among rows with no loop under none, plain50's induced rate on rows above the median overlap between plain50 and the row's none answer is at least 0.05 above its rate on the other rows | Weaver's |
| 169.W2 | plain50 raises low-k loop answers against none by at least a third of own's net cut in this run | Weaver's |

## Reading rules

A failed gate stops all readings. If 169.1 fails, Part 168's 31% did not hold at 25% on new neurons and phrasings. If 169.3 fails, its rise did not replicate, and 169.4, 169.5 and 169.8 read only as differences between bans. If 169.3, 169.4 and 169.5 hold, the rise goes with plain-answer tokens on the list and not with the mass removed. If 169.3 holds and 169.4 fails, a list raises loops whether or not it holds plain tokens, and the plain-token guess is wrong. If 169.8 holds, the plain tokens on the list reproduce most of the rise alone. If 169.6 fails, blocking raises loops with no push, and the generic account gains; if it holds, the rise needs a push. 169.4 is also read on a strength-matched subset, paired rows whose free and clear lists hold within 1 of the same count of tokens from the row's top 200 under q. If the gap there is under half of 169.4's, the contrast goes with how plausible the listed tokens are and not with plain tokens. Unpaired high rows are reported beside paired ones, and if they show the larger rise the test ran on the wrong half. Hit list sizes are reported with 169.8, and the absolute no-push rates with 169.6. If plain50's induced count is at or below free's and 169.W1 fails, Weaver's decomposition fails. Every reading is an effect of these defined masking policies. None identifies a lexical feedback mechanism, and no reading says why a push favours a word family. A failed superiority test is not an equivalence result.

## Limits

One model, one dose, 8 maker phrasings already used in Parts 165 and 166, greedy decoding. A clear list must reach own mass without the plain answer's tokens, so it holds near alternatives to them, and free and clear differ in exponent as well as content; both are logged. Where the mass sits on plain tokens a clear list cannot reach it, so mass and plain tokens cannot be separated on those rows. Plain tokens are those of one greedy answer, and a pushed answer leaves that path. The hit list carries most of the free list's mass on high rows, so 169.8 alone cannot separate tokens from mass.
