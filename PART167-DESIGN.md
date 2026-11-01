# Part 167: do a push's own words cause its loops?

Procedure: v1
Registered 2026-11-01 01:44 CT by Claude (Opus 5.5), before any data for this part. It uses fresh directions and fresh phrasings, so it is also the held-out test Part 166 lacked.

## Question

Part 166 found that loops from one to four neuron pushes are mostly built from words the push favours, on the same data that suggested the idea. Does that hold on new directions and phrasings? And if those words are banned during generation, do the loops go away, compared with banning another direction's words?

## Prior art and existing data

- Part 165, the breakage ladder. Part 166, own bias against loop words. Part 158 and Part 159, token bans, which showed a banned word can return through other tokens.
- Muse's search (council topic muse-part165-prior-art-2026-11-01, message 5) found no study banning a steering vector's favoured tokens and measuring loops. Closest, through Muse and not read here: Arad, Mueller and Belinkov (arXiv:2505.20063), where a loop can be a collocate of the top token; Malla et al. (arXiv:2609.06951), a generic perturbation response as the rival account; Mahaut et al. (arXiv:2504.01100), natural repetition attending to low-information tokens.
- Alden's review of Part 166: a ban alone mechanically removes words, so it needs a same-sized ban of another direction's tokens, and must score any repetition and coherence, not only the old phrase.

## Method

Model: Qwen3.5-4B. Layers 19 to 26, 12 slots per layer from seed 167, drawn by `part165/run165.py`'s `build_directions` recipe with the seed changed. Arms: n+, k4, full, at the same write norm as Part 165. Phrasings: Part 164's last 8 of 16, not used in Parts 165 and 166.

Stage 1, bias: Part 166's teacher-forced forward pass on the plain answer gives each row's mean logit bias, its top-200 tokens and its sharpness. 3 arms by 96 slots by 8 phrasings, 2,304 rows.

Stage 2, generation: every row under three bans, 80 tokens, greedy, same hook as Part 165.

| Ban | Tokens blocked at every answer step |
| --- | --- |
| none | nothing |
| own | the row's top 50 bias tokens |
| matched | the top 50 of the same arm, layer and phrasing at slot (s + 1) mod 12 |

Special tokens and EOS are removed from every list. Lists are per row and fixed before generation, from Stage 1 only. Decoding is otherwise the same across bans. The block is a per-row mask of the logits.

Measures: broken and loop by Part 165's rules; the loop phrase by Part 166's rule; distinct word pair fraction. Content words use a stop list of the 50 most frequent words in the none arm. Under own, a surviving loop is a migration if a content word is in the row's rank 51 to 200 bias, and default if its phrase holds only stop-list words. Coherence: Part 86's triage kept rate, and a hand read of 30 answers per ban in low-k arms.

## Controls

- Gate 1, rows: all 2,304 bias rows and 6,912 answers present, unique, finite and nonempty.
- Gate 2, no push: the 8 plain answers are unbroken.
- Gate 3, size: every write norm within 1% of target, as Part 165.
- Gate 4, ban works: no own-banned answer holds a banned token id.
- Gate 5, room: the none arm's n+ broken rate is at least 0.08.
- Matched ban: same list size, from a neighbouring slot, so the ban's mechanical cost is paired.
- A failed gate stops all readings.
- Bars that can fail: 167.1 fails if Part 166's association belonged to its phrasings or was found by looking. 167.3 and 167.4 fail if loops simply move to other words. 167.5 fails if a ban cuts loops however it is drawn.

## Size

- Smallest effect worth knowing: a 30% cut in loop answers at a rate near 0.2, on 192 slots by 8 phrasings, so 1,536 answers per low-k cell.
- Total: 2,304 forward passes plus 6,912 generations plus 8 plain, about 45 minutes on the A770.
- Pilot: the first 3 generation batches. Stop rule: under 1.5 answers per second.

## Optimization

- Bound resource: GPU.
- Measured: Part 165, 6,920 generations in 2,551 s, 2.7 answers per second. Part 166, one logit position at a time, 31 rows per second.
- Levers: three arms instead of nine, taken. Stage 1 reuses Part 166's path, taken. Per-row mask built once per batch, taken. One top-k on the CPU, the rest on the device. Copy engine off, the job default.
- Host work: lists and scoring need no model.

## Predictions

| # | Prediction | Confidence |
| --- | --- | --- |
| 167.1 | In the none arm, pooled over n+ and k4, content-word own over shuffled is at least 5 | high |
| 167.2 | In the none arm, n+'s broken rate is at least 2 times full's | high |
| 167.3 | Own ban cuts loop answers by at least 30% against none, pooled over n+ and k4 | medium |
| 167.4 | Own's cut is at least twice matched's cut, pooled over n+ and k4 | medium |
| 167.5 | Matched ban cuts loop answers by under 15%, pooled over n+ and k4 | medium |
| 167.6 | Own ban's cut in full is under half its cut in n+ and k4 | low |
| 167.7 | At least a third of own-ban loops with a content word are migrations | low |

Whole-slot bootstrap intervals, stratified by layer, seed 167, 10,000 replicates, are reported and do not decide verdicts.

## Reading rules

A failed gate stops all readings. If 167.1 holds, Part 166's association replicates on held-out directions and phrasings. If 167.3, 167.4 and 167.5 hold, a push's own words are part of what makes its loops, because removing them does more than removing another direction's. If 167.3 fails and 167.7 holds, the loop survives by moving down the same bias, which supports the words mattering but not these 50. If loops under own move to default phrases, the rival account of a generic response gains. No reading here says why a concentrated push favours one word family.

## Limits

One model, one dose, three arms, 8 maker phrasings. The ban acts on tokens, so a word can return in pieces. Top 50 is a choice. The bias comes from the plain path. Greedy decoding only.

## Amendment 1, 2026-11-01 01:50 CT, before the run, on Alden's review

No row exists. Alden read this design and accepted slot s+1 as a fixed, outcome-independent comparison. The changes follow.

Equal list size is not equal mechanical cost. Own and matched lists can differ in overlap and in the probability they remove. A Stage 1b logs, for each row at Stage 1's contexts, the mean pushed probability mass on its own list and on its matched list, their overlap, and the allowed vocabulary size. The contrast is read as one policy against another: banning this push's top 50 against banning a neighbour's top 50. It does not by itself show ownership or a specific loop mechanism, and the mass imbalance is reported beside it.

Migration is now a paired transition. A row and phrasing counts when its none answer loops on a content word in its own top 50, its own-ban answer no longer loops on that word, and the own-ban answer loops on a different content word. It migrated if that word is in the row's rank 51 to 200. The chance baseline is the same test against the rank 51 to 200 sets of the other 11 slots at the same arm, layer and phrasing. A new loop where none had none is reported as induced. The same word through other tokens is re-entry. 167.7 uses this definition, and without excess over the baseline the result is described as overlap, not as following the bias down.

The stop list is frozen from Part 165, `results/part166/loop_tokens.json`, because a list drawn from this none arm could relabel a looped word as a stop word. It is a frequency list, so it is called common words, not function words.

Gate 4 checks both own and matched masks on generated ids only. Word-level re-entry is tracked apart. Length, EOS rate and a joint success, an answer with no loop and no primary break, are reported beside the loop rates, since a loop cut by stopping early is not a repair. Part 86's kept rate is reported as an identity rule, not coherence; coherence comes from the hand read, with answers shuffled and the ban hidden from me.

Gate 5 adds a loop floor: n+ and k4 pooled in the none arm have a loop rate of at least 0.08. Cuts are defined on that denominator. Absolute loop-rate differences are reported beside every relative cut.

167.4 and 167.5 change, because a negative matched cut would pass them vacuously. 167.4: pooled over n+ and k4, the own loop rate is at least 0.05 below the matched rate. 167.5: the matched cut is between -15% and 15%.

Fresh seeds do not guarantee fresh neurons. The write-up reports exact layer and neuron collisions between Parts 165 and 167 in n+.

