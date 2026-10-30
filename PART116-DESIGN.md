# Part 116: do the neurons most active where Qwen3.5-4B writes "Q", or those writing straight onto "Q", dislodge it with fewer than 200 zeroed?

Procedure: v1
Registered 2026-10-29 19:23 CT by Claude (Opus 5.5), before any data, from Elise's question at 19:21 CT: "How are you screening for
the "q" neurons? Can we trace the ones that are most active when it writes the Q in qwen?" The commit that adds this
file is the registration.

## Question

Part 115 ranked neurons by gradient attribution and needed the top 200 zeroed together to take "Q" off first place on
most phrasings. Do two other screens find a smaller set: the neurons most active at the position where "Q" is
predicted, and the neurons whose output writes most directly onto "Q"?

## Prior art and existing data

- Closest published work: direct logit attribution (projecting a component's output onto a token's unembedding) and
  activation-based neuron screening are standard; activation patching studies report that high activation and causal
  effect often disagree. No count of how many neurons each screen needs for one token was found.
- Our own data: Part 115 (gradient ranking: 3 of 16 at 100, 11 of 16 at 200, 16 of 16 at 500); Part 71 read L27 N3325's
  output weights straight onto "Google"; Part 114's mean |activation| file supplies the controls' matching.

## Method

Qwen/Qwen3.5-4B at revision 851bf6e8, bf16, the A770; Part 114's chat template, forced opening "I am **", 8 selection and
16 held-out phrasings, and readout. part116/rank116.py reads every MLP neuron's down-projection input at the last
position on the 8 selection phrasings and builds two rankings: activity (the absolute z-score against the same neuron
over every position of Part 115's four ordinary prompts) and direct write (that mean activation times the neuron's
down-projection column, scaled by the final norm, dotted with the unembedding row of "Q"). Each ranking is then zeroed
by part115/howmany115.py, unchanged: the top k for k = 1, 2, 5, 10, 20, 50, 100, 200 and 500 together, with greedy
continuations. Scoring: part116/score116.py, reusing part115/score115.py.

## Controls

Matched random controls: for each ranking and k, 10 random sets drawn exactly as in Part 115 (same layers, the 20
nearest in Part 114's mean |activation|, never one of that ranking's top 500). The comparison ranking is Part 115's
gradient result. Both directions: not used, zeroing has none. The four ordinary openings ride along as in Part 115, but
their baseline is known to put " **" first, so they are not read as a damage check.

Gate: Part 115's two gates run again inside howmany115.py for each ranking (an all-ones mask bit-identical to the plain
model; the unzeroed row within 0.1 of Part 114's). Either failure stops that ranking.

## Size

- Smallest effect worth knowing: a ranking needing 100 or fewer neurons where the gradient needed 200, read as 8 or more
  of 16 phrasings flipped against at most 2 of 16 for its random sets.
- Total: 4,000 zeroed forward passes (two rankings, 100 rows by 20 prompts) and 288 greedy continuations, about 5
  minutes on the A770, most of it three model loads.
- Pilot: not needed: under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: GPU.
- Measured: 2,000 zeroed rows in about 7 s at batch 50 (Part 115, same runner, model and device).
- Levers: reuse Part 115's runner and scorer unchanged (taken); rankings from one forward pass per prompt (taken);
  three model loads instead of one process (declined: about 2 minutes, and merging the steps would change Part 115's
  tested runner).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 116.1 | Ranked by activity, the smallest k whose set takes "Q" off first place on at least 8 of the 16 phrasings is larger than 200, or no k up to 500 does it. | medium | Claude |
| 116.2 | Ranked by direct write onto "Q", that smallest k is 200 or less. | low | Claude |
| 116.3 | The activity ranking's top 200 share fewer than 50 neurons with the gradient ranking's top 200. | medium | Claude |
| 116.4 | At k = 50, the activity ranking's set lowers the mean log-odds of "Q" less than the gradient ranking's set did in Part 115 (6.96). | low | Claude |
| 116.5 | For both rankings, no matched random set takes "Q" off first place on more than 2 of the 16 at any k. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 116 (registered ...)".

## Reading rules

part116/score116.py computes everything from the two results115.json files, Part 115's score.json and the three
ranking files. A flip is a held-out phrasing where "Q" is not the most likely next token. Withdrawal rule: if a
ranking's random sets flip 8 or more at the k where its own set does, nothing is read from that ranking. Patterns
noticed after scoring are reported as exploratory.

## Limits

One model, one forced opening (which contains "**", whose bold context Part 115 found entangled with these neurons).
The activity baseline is four short ordinary prompts. Direct write ignores everything after the neuron except the
final norm's scale.
