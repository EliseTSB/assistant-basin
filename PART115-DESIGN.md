# Part 115: how many of the strongest neurons, zeroed together, take the leading "Q" off Qwen3.5-4B's name?

Procedure: v1
Registered 2026-10-29 19:05 CT by Claude (Opus 5.5), before any data. It follows Elise's bet for Part 114 (2026-10-29 18:13 CT:
"I bet we can find a single neuron, maybe two, highly responsible for the the leading Q it really likes in identity
questions."), which failed: no single neuron and not the top pair moved "Q" off first place. The commit that adds this
file is the registration.

## Question

Part 114 found 20 neurons that each lower the log-odds of "Q" (the first token of "Qwen") more than matched controls,
and none that dislodges it. How many of the attribution's top neurons must be zeroed together before "Q" stops being
the most likely next token after "I am **", and is that specific to these neurons rather than to zeroing that many?

## Prior art and existing data

- Closest published work: the Hydra effect (McGrath et al., 2023) and self-repair (Rushing and Nanda, 2024): when a
  component is ablated, later components partly compensate, so joint ablations fall short of the sum of single ones.
  AtP* (Kramar et al., 2024) shows first-order attribution can miss components whose effects cancel. Neither counts
  how many neurons a single next-token preference needs.
- Our own data: Part 114's 20 single-neuron drops sum to 8.42 nats against a mean log-odds of 6.74 (results/
  part114/score.json); its one measured pair was close to additive (1.16 together against 1.29 summed). Part 83 found every set of attention heads it zeroed sub-additive. Part 114's attribution for every MLP
  neuron is saved (p114/attr114.pt on the job folder), so no new attribution pass is needed.

## Method

Qwen/Qwen3.5-4B at revision 851bf6e8, bf16, the A770; chat template, no system prompt, thinking off; the answer forced
to begin "I am **"; Part 114's 16 held-out phrasings and readout (the log-odds of the single token "Q", id 48, and
whether it is the most likely next token). Zeroing: the inputs of each chosen neuron's down projection set to 0 at
every position (a per-row mask). Sets: the top k of Part 114's attribution ranking for k = 1, 2, 5, 10, 20, 50, 100,
200 and 500, all zeroed together. Greedy 12-token continuations of "I am **" for the candidate sets. Off-target: four
ordinary forced openings in the same template ("The capital of France is" / " Paris", "2 + 2 =" / " 4", "The
opposite of hot is" / " cold", "Water is made of hydrogen and" / " oxygen"): the log-probability of the right next
token. Runner: part115/howmany115.py; scorer: part115/score115.py.

## Controls

Matched random controls: for each k, 10 random sets of the same size, each candidate swapped for a neuron of the same
layer drawn from the 20 nearest it in mean |activation| on the selection prompts, never one of the top 500 candidates
(toolkit I). Both directions: not used, because zeroing has no negative counterpart here. Broken answers: the ordinary
prompts show whether a set breaks the model rather than the name. Off-target questions: the four ordinary openings.

Gate: an all-ones mask on every layer must leave the logits bit-identical to the plain model, and the unzeroed row
must match Part 114's unzeroed held-out log-odds within a mean of 0.1. Either failure stops the run before any
zeroing is read.

## Size

- Smallest effect worth knowing: "Q" off first place on 8 of the 16 phrasings for a candidate set against at most 2 of
  16 for each matched set (one-sided Fisher p about 0.01); n is fixed at Part 114's 16 held-out phrasings.
- Total: 2,000 zeroed forward passes (100 rows by 20 prompts) and 144 greedy continuations of 12 tokens, about 4
  minutes on the A770 including the model load.
- Pilot: not needed: under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: GPU.
- Measured: about 46 zeroed forward rows/s at batch 64 on the A770 (Part 114: 6,736 rows in about 145 s, same model
  and prompts).
- Levers: reuse Part 114's attribution file (taken: saves the attribution pass); one mask multiply per layer instead
  of Part 114's Python loop over rows (taken); rows sharing a prompt batched, 50 at a time (taken); continuations for
  the candidate sets only, batched across k (taken); loading straight to the device (declined: untested, and Windows
  had 12 GB free at Part 114's launch); prefill once (declined: single forward passes share nothing to prefill).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 115.1 | The top 20 zeroed together lower the mean held-out log-odds of "Q" by less than the 8.42 their single drops add up to. | medium | Claude |
| 115.2 | The smallest k whose top-k set takes "Q" off first place on at least 8 of the 16 phrasings is between 10 and 100. | low | Claude |
| 115.3 | At every k, the top-k set lowers the mean log-odds of "Q" more than all 10 matched random sets of the same size. | medium | Claude |
| 115.4 | At that smallest k, no matched random set takes "Q" off first place on more than 2 of the 16. | medium | Claude |
| 115.5 | At that smallest k, the four ordinary openings' right-token log-probability falls by less than 1 nat on average. | low | Claude |
| 115.6 | At that smallest k, the most common first word of the continuations that do not start with "Q" is not a model or company name. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 115 (registered ...)".

## Reading rules

part115/score115.py computes everything from results115.json and Part 114's score.json. A flip is a phrasing where
"Q" is not the most likely next token. 115.4 to 115.6 are untestable if no k reaches 8 flips. 115.6 uses a fixed list
of model and company names in the scorer. Withdrawal rule: if at the smallest flipping k a matched random set also
flips 8 or more, the reading "these neurons hold the Q" is withdrawn in favour of "zeroing that many neurons breaks
the answer", and the ordinary openings decide which. Any pattern noticed after scoring is reported as exploratory.

## Limits

One forced opening and one next token. Zeroing is one kind of removal; mean-ablation could differ. The ranking is
Part 114's first-order attribution on 8 selection phrasings, so a different ranking could need fewer neurons. Four
ordinary openings are a coarse check of damage.
