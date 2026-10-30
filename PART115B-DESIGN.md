# Part 115b: do the neuron sets that take "Q" off Qwen3.5-4B's name after "I am **" also take it off without the "**"?

Procedure: v1
Registered 2026-10-29 20:07 CT by Claude (Opus 5.5), before any data; the commit that adds this file is the registration. Elise
asked for it on 2026-10-29 (Discord message 1565527521532051567: "Also I do want 115b"), after Part 115's cold check
called the bolding reading a hypothesis.

## Question

Part 115 zeroed the top 1 to 500 neurons of Part 114's gradient ranking and took "Q" off first place after the forced
opening "I am **" on 11 of 16 phrasings at 200 and all 16 at 500; Part 116's direct-write ranking needed 100. In both,
"Q" follows "**". Do the same neuron sets take the name's first token off first place when nothing is bolded? Three
openings separate the two changes: after 'My name is "' the target is the same token "Q" without "**"; after "My name is"
it is the different token " Q" (with a space).

## Prior art and existing data

- Closest published work: as for Part 115 (the Hydra effect, McGrath et al., 2023; self-repair, Rushing and Nanda, 2024;
  AtP*, Kramar et al., 2024). None asks whether a set found for one formatting context carries to another.
- Our own data: Part 115 (results/part115/score.json) and Part 116 (results/part116/score.json) give the bold-context
  thresholds. The baseline probe for this part (results/part115b/probe115b.json, part115b/probe115b.py, three passes)
  found that the chat model bolds its name after every unbolded opening tried ("My name is": " **" first on 12 of 16),
  so each phrasing here carries a plain-text instruction. With it, "My name is" gives " Q" first, 'My name is "' gives "Q"
  first and "My name is **" gives "Q" first, each on 16 of 16.

## Method

Qwen/Qwen3.5-4B at revision 851bf6e8, bf16, the A770; chat template, no system prompt, thinking off. Prompts: Part
114's 16 held-out phrasings, each followed by " Please answer in plain text, without markdown.", with three forced
openings: "My name is" (target " Q", id 1167), 'My name is "' (target "Q", id 48) and "My name is **" (target "Q").
Readout: the log-odds of the target and whether it is the most likely next token. Zeroing exactly as Part 115 (the
input of each chosen neuron's down projection set to 0 at every position), with Part 115's sets: the top k for k = 1,
2, 5, 10, 20, 50, 100, 200 and 500 of two rankings, Part 114's gradient attribution (p114/attr114.pt) and Part 116's
direct-write ranking (p116/rank_dla.pt), both computed for "Q" after "I am **". Greedy 12-token continuations for the
candidate sets in each opening. Off-target: six ordinary probes whose plain-model first choice is the right answer
(probe pass 2): "The capital of France is **" / "Paris", "2 + 2 = **" / "4", "The opposite of hot is **" / "cold",
"The chemical symbol for gold is **" / "Au", "On a clear day the sky is **" / "blue", and "Water is made of hydrogen
and" / " oxygen". Runner: part115b/howmany115b.py; scorer: part115b/score115b.py.

## Controls

Matched random controls: Part 115's 10 random sets per k for each ranking (same seeds, so the gradient ranking's
controls are Part 115's and the direct-write ranking's are Part 116's): each candidate swapped for a same-layer neuron
from the 20 nearest it in mean |activation|, never one of that ranking's top 500 (toolkit I). Within-prompt comparison:
the three openings share each question and instruction. Off-target: the six ordinary probes, five of them after "**".
Not used: both directions (zeroing has no negative counterpart); a dose ladder (the k ladder plays that role).

Gate: an all-ones mask on every layer must leave the logits bit-identical to the plain model, and in the unzeroed row
every target must be the most likely next token (all 16 phrasings in all three openings, and all six ordinary probes).
Either failure stops the run before any zeroing is read.

## Size

- Smallest effect worth knowing: as Part 115, the target off first place on 8 of 16 phrasings for a candidate set
  against at most 2 of 16 for each matched set (one-sided Fisher p about 0.01); n is fixed at the 16 held-out phrasings.
- Total: 2 rankings x 100 rows x 54 prompts (48 held-out, 6 ordinary) = 10,800 zeroed forward rows, plus 2 x 3 x 16
  greedy generations of 12 tokens over 9 candidate rows; about 6 minutes on the A770 including one model load.
- Pilot: not needed: under 1,000 generations and one GPU-hour; the gate and the three probe passes cover the baseline.

## Optimization

- Bound resource: GPU.
- Measured: about 46 zeroed forward rows/s at 50 rows per batch on the A770 (Part 114: 6,736 rows in about 145 s, same
  model and prompts; Part 115's runner, reused here, took 99.3 s for 2,000 rows and 144 continuations with its load).
- Levers: one model load for both rankings (taken); the saved rankings reused (taken, no attribution pass); one mask
  multiply per layer, rows sharing a prompt batched 50 at a time (taken, Part 115's code); continuations for the
  candidate rows only (taken); a larger batch (declined: timing it costs a second model load, about 1 minute, against a
  possible saving of about 1 minute); continuations batched across phrasings (declined: needs left padding, which
  changes positions against the forward rows); loading straight to the device (declined: untested; Windows had 11.5 GB
  free at the last probe).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 115b.1 | With the plain-text instruction and the opening "My name is **", the gradient ranking's smallest k whose set takes "Q" off first place on at least 8 of the 16 phrasings is 100, 200 or 500. | medium | Claude |
| 115b.2 | For the gradient ranking, the smallest such k after 'My name is "' is within one step of the "My name is **" one on the grid 1, 2, 5, 10, 20, 50, 100, 200, 500 (with "none up to 500" as the step after 500). | low | Claude |
| 115b.3 | For the gradient ranking, the smallest k that takes " Q" off first place on at least 8 of the 16 after "My name is" is larger than the 'My name is "' one, or no k up to 500 does it. | low | Claude |
| 115b.4 | The same holds for the direct-write ranking. | medium | Claude |
| 115b.5 | For both rankings and all three openings, no matched random set takes the target off first place on more than 2 of the 16 at any k. | medium | Claude |
| 115b.6 | For the gradient ranking, at the smallest k found under 115b.1, the six ordinary probes' right-answer log-probabilities fall by less than 1 nat on average. | low | Claude |
| 115b.7 | For the gradient ranking after "My name is" at k = 500, the most common first word of the continuations that do not start with "Q" is a model or company name. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 115b (registered ...)".

## Reading rules

part115b/score115b.py computes everything from results115b.json. A flip is a phrasing where the target is not the most
likely next token. The smallest flipping k is the smallest k on the grid whose candidate set flips at least 8 of 16;
"none up to 500" counts as larger than any k. 115b.2 to 115b.4 are untestable when the context they compare against
never flips; 115b.6 is untestable if 115b.1's context never flips; 115b.7 is untestable if no continuation at k = 500
starts with anything but "Q". First words and the fixed list of model and company names are Part 115's (score115.py,
AI_NAMES). Withdrawal rule: in any ranking and opening where a matched random set also flips 8 or more at the smallest
flipping k, the reading "these neurons hold the name's first token" is withdrawn there in favour of "zeroing that many
neurons breaks the answer", and the ordinary probes decide which. A pattern noticed after scoring is exploratory.

## Limits

One instruction wording and three openings. " Q" and "Q" are different tokens, so the plain opening changes the token
as well as removing the "**"; the quote opening holds the token fixed. The rankings were chosen for "Q" after "I am **",
so a set made for each opening could need fewer neurons. The instruction itself changes the prompt from Part 115's, so
the bold opening here is not Part 115's prompt. Zeroing is one kind of removal. Six ordinary probes are a coarse check.

## Amendment 1 (2026-10-29 20:10 CT, before any data)

The first launch (20:09 CT) was stopped 11 s in by the job runner's RAM guard: Windows' available memory fell from 9.5
GB to 1.5 GB while the shared loader staged the whole model in host memory. The run now loads the weights straight to
the A770 (`--direct-load`, `load_direct` in part115b/howmany115b.py: the same weights, dtype and device through
`device_map`), the lever this design had declined as untested. Its first try failed before any data: transformers'
allocator warm-up asks for one 7.8 GiB block, over the A770's single-allocation limit, so the warm-up is skipped (it
only pre-allocates; the weights are loaded the same way). Nothing else changes, and both gates still run first. If
the direct load fails or changes nothing, the run waits for more free memory instead.

### Amendment 1, note (2026-10-29 20:48 CT, from the cold check; the text above is kept as written)

The fallback written above ("If the direct load fails or changes nothing, the run waits for more free memory instead")
was not used. The direct load's first try failed on transformers' allocator warm-up, not on memory; the warm-up was
skipped (commit b2e90a0, 20:11:39) and the run relaunched at 20:11:42, before any data. The amendment should have said
so before the relaunch. The pre-registration probe for the quote opening (pass 3) is results/part115b/probe115b-pass3.json.

