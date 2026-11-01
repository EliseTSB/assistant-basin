# Part 170: does blocking what the model was going to say raise loops without a push?

Procedure: v1
Registered 2026-11-01 08:34 CT by Claude (Opus 5.5), before any data for this part. Elise, 2026-11-01: "keep the research going, feel free to get the rest of the council involved" (Discord 1566330393123225731). Weaver asked for a powered no-push arm and for this to be framed as a method check, in #council-room on 2026-11-01. Muse's ban-size ladder and one-step release come from council topic muse-part165-prior-art-2026-11-01. Weaver's review of this design, in #council-room the same morning, is folded in below, and 170.W1 and 170.W2 are theirs. Alden could not be asked: their Codex process has not been running since about 02:52 CST, so three messages to them sit undelivered. Haiku 5.5 read the build cold in their place, as it did for Parts 168 and 169, and its findings are folded in.

## Question

In Part 169, blocking the 50 tokens most probable on a phrasing's plain answer took loop answers from 338 to 621 of 1,536 under a neuron push, while a list matched to the own list's mass changed nothing. With no push the same list gave 0 of 8, which is too few to say anything. Does blocking the plain answer raise loops on its own? If it does, every ban result in this study has a masking part that has nothing to do with neurons. If it needs a push, how does the rise grow with push size and with list size?

## Prior art and existing data

- Part 169: plain50 338 to 621 loop answers, 413 induced, 130 ended, a mean of 16.3 blocked steps per answer, first block at step 1 on 1,165 of 1,198 rows with no loop under no ban. Replacement tokens had a median rank of 2. In a 20-answer read of induced loops, 16 were cyclic repetition and 4 were prose repeating a phrase three times.
- Part 169, exploratory: induced rate fell as overlap between plain50 and the row's no-ban answer rose, 0.507 at under 10 tokens to 0.179 at 30 or more, and fell with no-ban answer length, 0.529 under 30 tokens to 0.310 at 70 or more (`results/part169/review169b.txt`). Overlap counts tokens in the pushed answer, so low overlap may mean the push already moved the answer off the plain vocabulary. That is a guess this part tests through dose.
- Part 169, no push: free lists gave 24 loops in 768 answers. Part 164: random directions break 7.8% of answers at this dose, and 1 of 16 plain answers loops at 80 tokens.
- Muse's search found no study of a ban-by-steering interaction or a ban-size curve for repetition (messages 7 and 9, read at source by Muse and not here). Xu et al. (arXiv:2206.02369): greedy repetition reinforces itself. Nguyen et al. (arXiv:2610.35328): masking is provably harmless only when the true continuation stays allowed, which plain50 breaks by construction. Jin and Han (arXiv:2605.10065): hard masks under greedy decoding on safety strings, with no loop measure. Tam et al. (arXiv:2408.02442): stricter format limits degrade reasoning, a constraint-strength gradient with accuracy as the readout. Muse's flag: overlap must be defined the same way across parts, so this part pins it.

## Method

Model: Qwen3.5-4B, same pin as Part 169. Layers 19 to 26, 6 slots per layer, 48 n+ neurons, seed base 170000 by Part 165's recipe, with no neuron repeating one of Parts 165, 167, 168 or 169 at that layer. k4 is dropped, since Part 169 showed the same shape and it holds its slot's neuron.

Doses: 0.25, 0.5 and 1 times Part 165's write norm, with the realized write logged and checked at each dose as in Part 165. Dose 0 is the no-push arm, hook off.

Prompts:

| Set | Prompts | Use |
| --- | --- | --- |
| P8 | phrasings 0 to 7, Part 169's | push arms and no push, so Part 169's result can be compared |
| M96 | 96 new paraphrases of who made you and who are you, in `part170/prompts170.py` | no push |
| Q96 | 96 ordinary short questions on facts and how-to, in `part170/prompts170.py` | no push |

Stage 0, plain: each prompt's plain answer to 80 tokens, greedy. Stage 0 runs the same prompts through the hook at zero write for P8, which must match plain bit for bit. Prompts whose plain answer loops or breaks are logged, frozen out of every comparison, and counted; P8's must pass as in Part 169.

Stage 1, lists, per prompt, with no push and before any generation: the mean probability of each token along the plain 40-token path, taken as in Part 169's plain50. The same pass logs each prompt's mean entropy along that path. Specials, EOS and added tokens are removed from every list.

| List | Tokens |
| --- | --- |
| b0 | no ban |
| b5, b50, b200 | the top 5, 50 and 200 tokens of the prompt's own list. b50 is Part 169's plain50 |
| rel50 | b50, released after the fifth generated token, Muse's one-step idea made a little longer so it can bite |
| other50 | for Q96 only, the top 50 of the next question's ranking in a fixed ring after removing this prompt's own b50, so a list of the same kind and exactly the same size that is not this answer. Its mass on this prompt's path is logged |

Stage 2, generation: 80 tokens, greedy, same hook as Part 165, block log as in Part 169. A push batch is one prompt, one list and one dose, 48 rows in a fixed order, and a no-push batch is one prompt with its lists.

| Arm | Rows | Lists | Answers |
| --- | --- | --- | --- |
| push, dose 0.25, 0.5, 1 | 48 neurons by P8, 384 rows | b0, b5, b50, b200 | 4,608 |
| push, dose 1 | the same 384 rows | rel50 | 384 |
| no push | P8, M96, Q96, 200 prompts | b0, b5, b50, b200, rel50 | 1,000 |
| no push | Q96 | other50 | 96 |

Measures: broken and loop by Part 165's rules, with Part 166's loop phrase. Rise: loop rate under a list minus loop rate under b0 on the same rows. Induced and ended as in Part 169, and net beside them, since Part 169 showed induced alone is churn. Length, EOS and joint success beside every loop rate. Share: the fraction of the distinct tokens in a row's b0 answer that are on its b50 list, the definition used from here on. Part 169's count is reported beside it. Mean share by dose is reported, as the direct look at whether a push moves answers off the plain vocabulary. Length quartile: quartiles of b0 token count among the rows in a reading.

Coherence: I read every no-push loop under b50 by hand, plus a blinded 24 answers from each of no push b0 and b50 and dose 1 b0 and b50.

## Controls

- Gate 1, rows: every planned answer present, unique, finite and nonempty.
- Gate 2, no push: the 8 P8 plain answers are unbroken and have no loop at 80 tokens, checked at Stage 0, which stops the run on failure. At most 15% of M96 and of Q96 may loop or break in plain; over that, the set's readings stop and the run goes on. On Q96 only a loop counts, since Part 165's broken rule was built for identity answers and calls any answer under four words broken. Frozen prompts are counted, frozen and surviving plain lengths are compared, and 170.1 and 170.2 are reported beside a version with frozen prompts kept in, because excluding them keeps the prompts least likely to tip.
- Gate 3, do-nothing hook: dose 0 through the hook equals plain token for token on P8. A build check: Part 20's identity clamp already measured 0.0 difference.
- Gate 4, size: every realized write within 1% of target at dose 1, 2% at dose 0.5 and 4% at dose 0.25, on one shared context. bf16 rounding of a single activation reached 0.6% at dose 1 in Part 165 (0.995 to 1.006 of target) and grows as the write shrinks.
- Gate 5, ban works: no banned answer holds a listed token, on generated ids. rel50 answers hold none in their first 5 tokens.
- Gate 6, room: b0 loop rate at dose 1 is at least 0.08, and b50's blocked steps at dose 1 average at least 5. The second half is a build check, since Part 169 measured 16.3.
- Gate 7, freshness: no neuron repeats one from Parts 165, 167, 168 or 169 at its layer, or repeats here.
- Gate 8, batch noise: b0 at dose 1 is generated twice in different batch order. Under 10% of answers may differ, and the count sets the noise floor beside every rise.
- A failed gate stops all readings.
- Bars that can fail: 170.1 and 170.2 fail if blocking the plain answer loops on its own by a quarter of the push rise. 170.3 fails if Part 169's rise does not recur. 170.4 fails if the rise does not grow with dose. 170.5's direction is nearly forced, since b5 is inside b50 and b50 inside b200 and greedy answers agree until a larger list blocks something, so only its magnitudes test anything. 170.6 fails if early blocks do more than their share of blocks. 170.7 fails if a same-size list that is not this answer raises loops. 170.8 fails if share does nothing inside a length quartile.
- Not used: both poles, since no neuron's meaning is claimed. Not used: random directions as pushes, since the neurons are only a source of a push here and Part 164 sized them; a follow-up if 170.4 holds.

## Size

- Smallest effect worth knowing: a no-push rise of 0.10, so 5% against 15% is separated, which is Weaver's bar as well. With about 100 prompts per set, a binomial standard error near 0.03 gives that at about three standard errors. A dose difference in rise of 0.10 on 384 rows.
- Total: 6,472 generations, about 55 minutes on the A770. That is 4,608 + 384 + 384 for the repeat + 1,000 + 96, plus 200 plain and 200 list passes. Push: 112 batches of 48 at about 18 seconds, 34 minutes. No-push: 200 prompts one batch each and 200 plain answers one at a time, about 20 minutes, since a batch holds one prompt.
- Pilot: the first 3 generation batches. Stop rule: under 1.5 answers per second.

## Optimization

- Bound resource: GPU.
- Measured: Part 169, 9,371 answers from 05:30:24 to 06:34:20 CST, about 2.4 answers per second with its list stages inside that time. Single-row and small-batch steps are estimated, not measured; the log will show them.
- Levers: k4, clear, free and hit dropped, taken. 6 slots instead of 12, taken, because the effect is large. One position's logits at a time, taken. Copy engine off, the job default. No-push arm runs first, so a Gate 2 failure costs under 5 minutes. Lists from one forward pass per prompt, taken.
- Host work: scoring needs no model.

## Predictions

Verdicts are point estimates against the bars. Push arms: whole-slot bootstrap, stratified by layer, seed 170, 10,000 replicates. No push: bootstrap over prompts, same seed and count. Rates are on prompts or rows frozen at Stage 0.

| # | Prediction | Confidence |
| --- | --- | --- |
| 170.1 | No push, identity prompts (P8 and M96): b50's rise in loop rate is under a quarter of its rise at dose 1, with the absolute rise reported beside it | medium |
| 170.2 | No push, Q96: the same bar, against the same dose 1 rise | medium |
| 170.3 | Dose 1: b50 raises loop answers by at least 50% against b0 | high |
| 170.4 | b50's rise at dose 1 is positive and at least twice its rise at dose 0.25 | medium |
| 170.5 | Dose 1: b200's loop rate is at least 0.05 above b50's, and b5's at least 0.05 below b50's | medium |
| 170.6 | Dose 1: rel50's rise as a share of b50's is under 1.5 times rel50's share of b50's blocked steps | low |
| 170.7 | No push, Q96: other50's change in loop rate is under 0.03 either way, read beside b5 and b50 | medium |
| 170.8 | Dose 1, rows with no loop under b0: within b0 length quartiles, weighted by rows, b50's induced rate below each quartile's median share is at least 0.05 above the rest. The raw median split is descriptive | low |

Weaver's, from their review in #council-room, 2026-11-01. They ask that their falsifiers be weighted over their forecasts. W1 predicts a small effect, against their own pattern. Between a third and a half on W2, they will call it inconclusive.

| # | Prediction | Confidence |
| --- | --- | --- |
| 170.W1 | No-push b50 rise is under 0.05 on both M96 and Q96, near zero on M96. Falsifier: a rise of 0.10 or more on either set, and then blocking alone supplies a substantial masking part | Weaver's |
| 170.W2 | Dose 1: rel50's rise is at least half of b50's. Falsifier: under a third, and then sustained blocking is what matters | Weaver's |

## Reading rules

Every push result in this study is on identity prompts, and Q96 has no push arm. So anything said here about a ban needing a push holds for identity prompts only, whose answers are unusually formulaic. A failed gate stops all readings. 170.1 and 170.2 are read separately and the two sets are never compared or pooled, since their base rates differ by answer length before any ban. If 170.1 and 170.2 both hold and 170.3 holds, the rise on these prompts is mostly a push rise, under the quarter bar; give the absolute no-push rise in the same sentence. If either fails, blocking the plain answer raises loops on its own, and own-ban and plain50 results here carry a masking part: say so in the same sentence as any ban result from this study. If 170.3 fails, Part 169's rise did not recur and 170.4 to 170.6 read only as differences between lists. If 170.4 holds, the rise grows with push size; two doses and a zero do not fix a curve's shape. 170.5's ordering is near-forced, so read its size and not its direction. If 170.6 holds, loop damage tracks how many steps were blocked; if it fails, early blocks do more than their share. 170.W2 reads the same ratio against fixed bars. If 170.7 fails, a list raises loops on prompts where it blocks little, and the size of b50's effect has to be read against that. Exploratory, decided by nothing: Q96 split at its median plain-path entropy, since Weaver expects a list to bite hardest where the mass sits on one answer, and rel50's blocked-step share. If 170.8 holds, it agrees with overlap standing in for push distance and does not show it. If only the raw split shows a gap, it was answer length. No reading says why loops form. Every reading is an effect of these defined masking policies on one model.

## Limits

One model, greedy decoding, three doses, 48 neurons as push sources. M96 and Q96 have no push arm, so the ban-by-push interaction is measured on identity prompts only. b50 is defined on the plain path, which a pushed answer leaves, so it blocks less of a pushed answer's actual continuation as dose rises; the block log records how much. The loop rule labels both cyclic repetition and prose repeating a phrase. Batch-shape noise is a property of this card and this dtype.
