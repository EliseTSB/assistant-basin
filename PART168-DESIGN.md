# Part 168: are a push's loops its own words, or just the probability a ban removes?

Procedure: v1
Registered 2026-11-01 02:46 CT by Claude (Opus 5.5), before any data for this part. Elise, 2026-11-01: "keep the research going, feel free to get the rest of the council involved" (Discord 1566330393123225731). It succeeds Part 167, which Gate 2 stopped before any reading.

## Question

Part 166 found that loops from one to four neuron pushes are mostly built from words the push favours, on the data that suggested the idea. Part 167 was built to test that on new directions and by banning those words, and its Gate 2 failed, so none of its predictions was read. This part asks the same question with Weaver's review folded in. Does the association hold on fresh directions? Does banning a push's top 50 tokens cut its loops more than banning a neighbour's top 50, and more than banning 50 random tokens that remove the same probability?

## Why Part 167 stopped

Gate 2 required the 8 plain answers to be unbroken. Phrasing 13's plain answer loops at 80 tokens ("a large language model developed by Qwen3.5, a large language model developed by Qwen3.5, a"), and has_trigram_loop counts it as broken. That answer was already on disk in `results/part164/answers.jsonl`, where Part 164's floor of 0 of 16 used triage alone, before Part 165 added the loop rule. One line of Python would have shown it before the run. The rule says a failed gate stops all readings, so Part 167's verdicts stay unfilled. I have not read its rates or answers, apart from the first words of two answers printed by a file check, and Gate 5's two floor values printed with the gates.

## Prior art and existing data

- Parts 165 and 166, as above. Part 158 and Part 159: a banned word can return through other tokens.
- Muse's search (council topic muse-part165-prior-art-2026-11-01, message 5) found no study banning a steering vector's favoured tokens and measuring loops. Closest, through Muse and not read here: Arad, Mueller and Belinkov (arXiv:2505.20063), a loop can be a collocate of the top token; Malla et al. (arXiv:2609.06951), a generic perturbation response as the rival account; Mahaut et al. (arXiv:2504.01100), natural repetition attending to low-information tokens.
- Part 167's Stage 1b log, which holds no outcome, measured Weaver's worry. On n+, the mean pushed mass on a row's own list has median 0.0093 and on its neighbour's list 0.0000, and the 95th percentile of overlap between the lists is 1 token. So the neighbour's ban removes about nothing and is close to no ban. Banning a list by size alone cannot separate words from removed mass.
- Plain model: 1 of 16 phrasings loops by itself at 80 tokens, so a plain loop is possible without any push.

## Method

Model: Qwen3.5-4B, same pin as Part 167. Layers 19 to 26, 12 slots per layer, seed base 168000 by Part 165's recipe, with one change: a slot whose n+ neuron equals an n+ neuron of Part 165 or Part 167 at that layer redraws, by adding 1,000,000 times the attempt number to the seed, so no n+ neuron repeats. Arms: n+, k4, full, at Part 165's write norm. Phrasings: Part 164's last 8 without phrasing 13, so 8, 9, 10, 11, 12, 14 and 15.

Stage 0, plain: each phrasing's plain answer to 80 tokens, greedy. Gate 2 is checked here, before anything else runs.

Stage 1, bias: Part 166's teacher-forced forward pass on the plain 40-token answer gives each row's mean logit bias, its top-200 tokens and its sharpness. 3 arms by 96 slots by 7 phrasings, 2,016 rows.

Stage 1b, lists: at Stage 1's contexts, the row's mean pushed probability q over the vocabulary. Own mass is the sum of q on the own list. The mass list is drawn for n+ and k4 rows from the 20,000 most probable tokens outside the own list, the matched list and the special tokens: 50 tokens without replacement, with weights q to the power a, where a is found by bisection so the expected mass equals own mass, and redrawn up to 200 times until the mass is within 10% of own mass. The row's generator is seeded from arm, layer, slot and phrasing. If no draw lands within 10%, the nearest is kept and counted against Gate 6. Logged for every row: own, matched and mass-list mass, overlaps, a, draws, and how many of the plain answer's tokens each list holds.

Stage 2, generation: 80 tokens, greedy, same hook as Part 165.

| Ban | Tokens blocked at every answer step | Arms |
| --- | --- | --- |
| none | nothing | n+, k4, full |
| own | the row's top 50 bias tokens | n+, k4, full |
| matched | the top 50 of the same arm, layer and phrasing at slot (s + 1) mod 12 | n+, k4, full |
| mass | the row's mass list | n+, k4 |

Special tokens and EOS are removed from every list. Lists are per row and fixed before generation, from Stages 1 and 1b only.

Measures: broken and loop by Part 165's rules; the loop phrase by Part 166's rule. Content words use the frozen common-word list from `results/part166/loop_tokens.json`. A paired transition counts when a row's none answer loops on a content word in its own top 50, the banned answer no longer loops on that word, and the banned answer loops on a different content word. It is sorted into three cells: rank 51 to 200 of the row's bias, which is migration; outside ranks 1 to 200, which supports the generic account; or default, where the phrase has only common words. A chance baseline for migration uses the other 11 slots' rank 51 to 200 sets. A new loop where none had none is induced, and its denominator is rows with no loop under none. The same word through other tokens is re-entry, tracked apart. Length, EOS rate and joint success, no loop and no primary break, are reported beside loop rates, since a loop cut by stopping early is not a repair. Coherence: a blinded hand read of 30 answers per ban in low-k arms.

## Controls

- Gate 1, rows: all 2,016 bias rows and 7,392 battery answers present, unique, finite and nonempty.
- Gate 2, no push: the 7 plain answers are unbroken by Part 165's primary rule at 80 tokens. Checked by the runner at Stage 0, which stops the run on failure, and checked before launch against Part 164's stored answers.
- Gate 3, size: every write norm within 1% of target, as Part 165.
- Gate 4, ban works: no banned answer holds a token on its row's list, for own, matched and mass, on generated ids only.
- Gate 5, room: the none arm's n+ broken rate is at least 0.08, and n+ and k4 pooled have a none loop rate of at least 0.08.
- Gate 6, mass match: at least 90% of n+ and k4 rows have a mass list within 10% of own mass. Checked after Stage 1b, before generation, and the run stops on failure.
- Gate 7, freshness: no n+ neuron equals one from Part 165 or Part 167 at its layer.
- A failed gate stops all readings.
- Bars that can fail: 168.1 fails if Part 166's association belonged to its phrasings or was found by looking. 168.3 and 168.4 fail if loops simply move to other words. 168.8 fails if the own ban's cut comes from the mass it removes.
- Not used: both directions and poles, since no neuron's meaning is claimed. Not used: dose ladder, since Part 165 fixed the dose.

## Size

- Smallest effect worth knowing: a 0.05 fall in the loop rate at a rate near 0.2, on 192 slots by 7 phrasings, so 1,344 answers per low-k cell.
- Total: 2,016 forward passes twice, plus 7,392 generations, plus 7 plain, about 50 minutes on the A770.
- Pilot: the first 3 generation batches. Stop rule: under 1.5 answers per second.

## Optimization

- Bound resource: GPU.
- Measured: Part 167, 6,920 generations in about 2,600 s, 2.8 answers per second, Stages 1 and 1b in about 4 minutes.
- Levers: mass ban on two arms instead of three, taken. Stage 1 reuses Part 166's path, taken. One position's logits at a time, taken. Mass list drawn on the CPU from the top 20,000 tokens, since top-k on the device stalls, taken. Copy engine off, the job default.
- Host work: lists after q reaches the CPU, and scoring, need no model.

## Predictions

| # | Prediction | Confidence |
| --- | --- | --- |
| 168.1 | In the none arm, pooled over n+ and k4, content-word own over shuffled is at least 5 | high |
| 168.2 | In the none arm, n+'s broken rate is at least 2 times full's | high |
| 168.3 | Own ban cuts loop answers by at least 30% against none, pooled over n+ and k4 | medium |
| 168.4 | Pooled over n+ and k4, the own loop rate is at least 0.05 below the matched rate | medium |
| 168.5 | The matched cut, pooled over n+ and k4, is between -15% and 15% | medium |
| 168.6 | Own ban's cut in full is under half its cut in n+ and k4 | low |
| 168.7 | At least a third of paired transitions are migrations, and above their baseline | low |
| 168.8 | Pooled over n+ and k4, the own loop rate is at least 0.05 below the mass-ban rate | medium |

Weaver's, recorded before the run:

| # | Prediction | Confidence |
| --- | --- | --- |
| 168.W1 | The mass ban cuts loop answers by 15% to 25% against none, pooled over n+ and k4 | Weaver's |
| 168.W2 | Among rows with no loop under none, the induced rate under own is at least 0.02 above matched | Weaver's |
| 168.W3 | Migrations are 15% to 30% of paired transitions | Weaver's |

Whole-slot bootstrap intervals, stratified by layer, seed 168, 10,000 replicates, are reported and do not decide verdicts. Each of n+ and k4 is also reported alone, since k4 contains its slot's n+ neuron.

## Reading rules

A failed gate stops all readings. If 168.1 holds, Part 166's association replicates on held-out directions. If 168.3, 168.4, 168.5 and 168.8 hold, banning a push's own words cuts its loops more than banning a neighbour's words or random words of the same mass, so the words themselves are part of what makes the loops. That compares one ban policy with another and names no mechanism. If 168.3 holds and 168.8 fails, the cut goes with the probability removed and says nothing about these words. If the mass ban cuts under 5%, Weaver's confound was wrong. If loops under own move outside ranks 1 to 200, the rival account of a generic response gains. No reading here says why a concentrated push favours one word family. After this part is scored, Part 167's rows may be tabulated beside it as exploratory, labelled as a stopped part with one looped plain phrasing.

## Limits

One model, one dose, 7 maker phrasings, all but held out from Part 166 only. The ban acts on tokens, so a word can return in pieces. Top 50 is a choice. Mass is measured on the plain path, and a pushed answer leaves that path. A mass list can hold ordinary tokens the answer needs, which is the cost it is there to measure. Greedy decoding only.
