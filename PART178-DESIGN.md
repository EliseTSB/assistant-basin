# Part 178: Which words open a thought in Qwen3.5's thinking, and which of them belong to thinking?

Procedure: v1
Registered 2026-11-02 22:10 CT by Claude (Opus 5.5), before any text from the 200 problems was generated or read. Elise, 2026-11-02: "Im not sure we know the words of power for any model we can run locally, although we could probably find them ourselves or in the literature" [message 1566947791207665815]. The commit that adds this file to public main will be the registration.

Seen before registering, and no more: Part 177's published tables; six hand-read windows on 2026-11-02, one reading "Result: $\cot x$." and then "Wait, I should check if the user meant"; the tokenisation of 28 candidate strings in Qwen3.5-0.8B's tokenizer. No opener count on any text. Weaver and Alden read earlier drafts, and neither scanned any text or label for it. Pilot and check runs on the 16 excluded problems printed speeds, memory, validity flags and token-agreement counts, and no text.

## Question

The words linked to changes of course in thinking are known from R1-style models and from leaked Claude traces. Nobody has listed them for the models we can run, and a list does not travel between tokenizers or training recipes. This part builds a list of candidate opener tokens for Qwen3.5-0.8B and Qwen3.5-4B from their own thinking under one sampling and format protocol, checks the imported list against it, and asks which openers go with thinking mode. It asks one small question about order: what opens the next thought after a selected literal hit that a reader judged asserted?

It picks candidates for Part 179, a fixed mid-thought insertion. It tests no causal claim. A word common in thinking may do nothing: one solution phase could produce both an answer and a checking phrase, a shared format could produce the same openers across models, and a mode could change register alone.

## Prior art and existing data

- Closest published work, from Muse's search of 2026-11-02, cited from Muse's report and not read by Claude: NoWait, Wang et al., arXiv:2506.08343, builds a list from chunk-initial words in QwQ-32B and expands it per tokenizer. DuP-PO, Ding et al., arXiv:2506.23840, counts wait at 37% of thinking tokens on one R1 distill. Nothing was found for Qwen3.5-0.8B or Qwen3.5-4B. The nearest checkpoint is Qwen3-4B-Thinking in Du et al., arXiv:2602.01999. Alden read NoWait section 3.1: its chunks are blank-line paragraphs, and the paper says 15 words but shows 17.
- Our own data: Part 177's stored texts and labels. `notes/reports/words-of-power-neurons-2026-11-02/` counted the words in a neuron catalog built on ordinary text, which says nothing about thinking.

## Method

The same 200 MATH-500 problems, the first 200 rows of Part 177. Six text sets:

- `U`, `E`: MIT's stored sample-0 texts for unedited and `p50c`, the files pinned in Part 177.
- `G`: new. One trace per problem from Qwen3.5-4B at snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on the Arc, bf16, thinking on, cap 2,048 new tokens.
- `R8`: new. Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17`, thinking on, cap 2,048, the same rig as `G`. This checks our rig against MIT's stored traces of the same model.
- `N4`, `N8`: new. Qwen3.5-4B and Qwen3.5-0.8B with thinking off, the same rig, cap 2,048. The non-thinking sets. All four generated sets have the same cap, cut from 4,096 for the thinking sets after the 0.8B pilot ran at about 100 tokens a second.

The rig uses MIT's settings from their `evaluate.py`, in vLLM's order: presence penalty 1.5 subtracted once from the logit of any token already generated in this row (not the prompt, and not more for repeats), then temperature 0.6, then top_k 20, then top_p 0.95, then a draw from a per-row generator seeded from the set and problem id, so a row's random draws do not depend on its batch. In bf16 on the Arc the logits themselves can differ slightly with batch shape, so a row reproduces exactly only at the same batch shape. Logits move to the CPU for these steps. The penalty and top-k run on the Arc and only the 20 survivors move to the CPU. No row is generated at batch 1: on the Arc, Qwen3.5's greedy tokens at batch 1 differed from batches 2 to 16, which agreed with each other and, on the 0.8B, with a CPU float32 reference (`part178/diag178d.py`, `diag178e.py`), so a lone last row runs beside a discarded companion. Generation runs through the model's own `generate` with this sampler as a logits processor, so the hybrid cache of Qwen3.5's linear-attention layers is handled by the library, and rows stop on either `<|im_end|>` or `<|endoftext|>`. Generated token ids are kept for every row. MIT's own non-thinking settings differ (temperature 0.7, top_p 0.8), and they are not used, so both modes share one sampler.

Discovery here is conditional on this protocol: presence penalty lowers exactly the repeats a count sees, so these rates are rates under that setting, and Part 179 uses the same one.

Definitions, by code in `part178/words178.py`:

- `thinking text`: text before `</think>`, or all of it if none.
- `prefix`: the first 2,048 tokens of a text. Stored texts are retokenised with the pinned tokenizer, which reconstructs an encoding and does not prove the ids emitted. Generated texts use their kept ids.
- `onset2k`: Part 177's line rule run on the prefix only, so every set has the same lookahead. A repeat past the prefix does not count, and the last 40 lines of a prefix are censored for a new onset.
- `window`: the first 2,000 tokens of a prefix, cut at `onset2k`. Primary, because looping text would repeat one word thousands of times. Each trace records token exposure, opener exposure and an end reason: think-close, onset, end of turn, cap or limit.
- `uncut`: the first 2,000 tokens with no onset cut, reported beside `window` as a sensitivity. A rate fixes the opportunity difference between traces and not a phase selected by the cut, so both are shown.
- `opener`: the first word of a line, or of a sentence after `.`, `?` or `!` and a space. Skipped first: spaces, list numbers, bullets, `*`, `_`, `#`, `>`, quotes and opening brackets. If the first thing left is math (inside `$...$`, `\(...\)`, `\[...\]`, or starting with a digit, backslash or operator), that line or sentence has no opener. Decimals and the abbreviations `e.g.`, `i.e.`, `etc.`, `vs.` end no sentence and are not words. A run of CJK characters counts as one word, a stated limit. Hyphens stay inside a word, so `double-check` is one word. Case-folded for the tables.
- `paragraph opener`: the same, for the first word of a blank-line paragraph only. NoWait's boundary.
- `rate(w)`: `w` per 1,000 openers, pooled over a set's windows, with an interval by trace. Ranking is by `rate(w)`, then by word for a tie. `traces(w)`, traces with `w` at least once, is reported beside it with the number of top-20 entries that differ.
- `mode ratio(w)`: `rate(w)` on `window` in `R8` over `N8` for Qwen3.5-0.8B, and in `G` over `N4` for Qwen3.5-4B, with an interval by trace. These are the within-rig pairs, and `U` is not a numerator here. It is a mode association and not proof of a thinking word, since a maths answer can reason too. A word with fewer than 5 occurrences as a control opener gets no ratio, and is reported with its counts.
- `imported list`: NoWait's 17 words as shown in the paper: wait, alternatively, hmm, but, however, alternative, another, check, double-check, oh, maybe, verify, other, again, now, ah, any.
- `reflection family`: wait, hmm, alternatively, actually, however, but. Fixed here, before any count.
- `token list`: for the top 30 openers per model and mode, one row per whole word span in context: the word, every constituent id, case, leading space or newline, count. `G`, `R8` and `N` rows come from kept ids, and `U` and `E` rows are marked reconstructed. Part 179 inserts a chosen token in a chosen context, not a lemma.

Follow-on, on `U`, using Part 177's key and label rows unchanged, on the full stored text. The eligible traces are those in `T` with a selected literal hit labelled `asserted`, and the target is the earliest such hit. This is the first selected reader-labelled hit, chosen among first, middle and last literal hits, and not the first semantic assertion, and the hit's end is not the assertion's end. `next opener`: the first opener after the hit ends, unavailable if none. `local baseline`: the share of reflection-family words among that trace's other openers within 500 tokens either side of the hit's start, truncated at the text's ends, excluding the next opener, unavailable if empty, not zero. The statistic is the equal-trace mean of `I(next opener is reflection family) - local baseline` over all eligible traces with both available, with a bootstrap by trace. That mean is the verdict. Three descriptive groups are reported beside it: before Part 177's full-text onset, after it, and no onset detected.

## Controls

- `N4` and `N8`, by model, for every opener in the table.
- `E` beside `U`, on the same problems. `R8` beside `U`, on the same model with our rig.
- Ordinary openers in the same table, for scale: the, so, we, let, then. Rare words `river` and `table` are reported as diagnostics and gate nothing, since maths can use them.
- Not used: pushes, poles, do-nothing hook, random directions, because nothing is pushed. A reader, because counts need none, and Part 177's labels are reused unchanged.

Gate 1: Part 177's text hashes and its full-text onset counts reproduce exactly for `U` and `E`. This is separate from `onset2k`.
Gate 2: the opener rule matches frozen expected results on fixtures: numbered and nested lists, a bullet, a sentence after `?`, `Wait,`, `**Wait**`, `$x$ Wait` (no opener), a formula-only line, prose after math on a new line, `3.5 and`, `e.g. wait`, `double-check`, and a soft-wrapped sentence, which is the rule's known sensitivity and is expected to add an opener. A deliberate break must turn them red.
Gate 3, mode validity: every thinking row's prompt ends with an open think tag and every non-thinking row's prompt ends with the template's empty think block, checked on ids, and no non-thinking row generates a think tag. Short valid answers stay. Adequacy: each set has at least 2,000 openers in its windows, or anything resting on that set is unread.
Gate 4, the sampler, on hand-worked fixtures: a token only in the prompt is not penalised; once and many repeats are penalised the same; penalty comes before top-k (seen Wait at 3.0, unseen A at 2.9 and B at 2.8, penalty 1.5, top_k 2 keeps A and B); temperature, top_k and top_p compose in that order; a row that ends stops drawing; a row's draws are the same when its batch-mates finish early.
Gate 5, a rank check only: over `U`'s top 20 openers, the Spearman of `rate(w)` between `R8` and `U` on `window` is at least 0.7. If not, comparisons of `G` with stored traces (178.4) are unread until explained. It checks the order of those 20 words and nothing else: it does not certify the sampler, the rates or the tokens, so absolute rates for the 20 and the count of them in `R8`'s top 20 are reported beside it. Mode ratios use within-rig pairs and do not depend on it.

If a gate fails, what rests on it is not read until it is explained: Gates 1 and 2 block everything, Gate 3 blocks the set that fails it, Gate 4 blocks every generated set, and Gate 5 blocks 178.4 only.

## Size

- Smallest effect worth knowing: a word at 20 against 5 per 1,000 openers, or a mode ratio of 2, on 200 traces per set, each with an interval by trace. A follow-on gap of 15 points on about 50 traces is a rate with its interval and no more.
- Total: 800 generations, about 200 minutes on the Arc if every row hit its cap, 1,638,400 tokens at most, 819,200 on each model. The schedule comes from the pilot.
- Pilot: the first 16 problems in dataset order outside the 200, excluded from all counts, repeated with new seeds to fill a batch. Each set runs at the full cap at the batch sizes the plan names. A first 0.8B pilot at batch 16 and cap 4,096 ran at about 100 tokens a second with the Arc about half busy, so the sampler bounds the small model. Before registering it printed throughput, peak memory, mode validity and the share of rows with empty thinking (0 in `G`). Its batch check compared batch 1 with batch 16 and found 15 of 128 tokens agreeing, which led to the batch-1 finding above. No text was read. After registering, its rows are read by eye before the main run is read. Stop and redesign if the 4B's thinking is empty, or if the projected run passes 4 hours.

## Optimization

- Bound resource: GPU, then CPU for sampling.
- Measured: on the excluded problems, 432 tokens/s at batch 64 for Qwen3.5-0.8B (432 to 471 over 1,024 steps after warm-up), and about 250 for the first pass in a process (`diag178c.py`, run in both orders); device memory peak 3.9 GB. Qwen3.5-4B at batch 16, 78 tokens a second on the first pass at the full cap; batch 32 reached 17.5 GB reserved, past the Arc's 15.6 GB, and spilled into Windows memory until the RAM guard stopped it, so the 4B runs at batch 16 and the 0.8B at 64. Projection, if every row ran to its cap: about 20 minutes for each 0.8B set and 80 for each 4B set, about 3 hours 20 minutes in all.
- Levers: batch size timed; logits only for the last position; penalty and top-k on the Arc (7 ms a step at batch 64 against 76 on the CPU, `diag178b.py`), one draw per row on the CPU; cached device blocks released every 64 steps, which held reserved memory at 3.8 GB against 5.2 with no measured speed cost; copy engine off; direct load; resume by appending whole batches. Declined: rows leaving the batch and new rows joining, because Qwen3.5's hybrid cache is handled by `generate`, which keeps a batch until its longest row ends.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 178.1 | In `U`, `wait` is among the top 10 openers by `rate(w)` | medium | Claude |
| 178.2 | In `G`, `wait` is among the top 10 openers by `rate(w)` | medium | Claude |
| 178.3 | `rate(wait)` differs by less than a factor of 1.5 between `U` and `E` on `window`; unread if either is zero | medium | Claude |
| 178.4 | At least 14 of `U`'s top 20 openers are in `G`'s top 20 | medium | Claude |
| 178.5 | At most 6 of the 17 imported words reach 5 per 1,000 paragraph openers in `U` | medium | Claude |
| 178.6 | The follow-on statistic on eligible traces is at least 0.15; not read below 40 eligible traces | low | Claude |
| 178.7 | At least 10 of `G`'s top 20 openers have a mode ratio under 2; unread if more than 5 of the 20 have no ratio | medium | Claude |
| 178.W1 | Ranking by `rate(w)` and by `traces(w)` disagree on at least 4 of the top 20 in `U`. Falsifier: 1 or fewer | medium | Weaver |
| 178.W2 | Among traces with both labels, the median token position of the first `asserted` selected hit exceeds the first `intermediate` selected hit's by at least 2,000 tokens. Falsifier: under 500 | medium | Weaver |
| 178.W3a | On Qwen3.5-0.8B, `wait` occurs at least 3 times more often per 1,000 openers in thinking than in non-thinking text, by `mode ratio` of `R8` over `N8`. Falsifier: ratio under 1.5. Unread under 500 control openers or with no ratio | medium | Weaver |
| 178.W3b | The same on Qwen3.5-4B, by `mode ratio` of `G` over `N4` | medium | Weaver |

Copied verbatim into PREDICTIONS.md under "## Part 178 (registered ...)".

## Reading rules

`part178/score178.py` writes every number and verdict. A verdict reads the point estimate against its bar, with an interval by trace. Weaver bars use `held` for the claim, `failed` for the stated falsifier and `inconclusive` otherwise, the rule from Part 177's Amendment 1. 178.W3a and 178.W3b are read per model, as Weaver ruled on 2026-11-02, with no joint verdict. The sentence that `wait` belongs to thinking and not to English needs both to hold; if one holds, Part 179 runs on that model and the write-up says the generality is not established. Answers to the math are not graded. The deliverable is called candidate opener tokens under this protocol. No sentence in the write-up may say a word steers, turns or corrects the model. Withdrawal, a one-way screen: if no reflection word reaches 5 per 1,000 openers in `U` or `G`, or none has a `mode ratio` of at least 2 on `G` over `N4` or `R8` over `N8`, the reading "these models have thinking openers of the R1 kind" is withdrawn, and Part 179 is redesigned from the table. Passing does not establish that reading.

## Limits

Two models, one benchmark, one sample per problem, MATH-500 only. `U` and `E` are mostly capped and looped traces, cut at a line rule that is not ground truth. Rates are conditional on presence penalty 1.5. `opener` is a text rule that depends partly on layout, and a word used mid-sentence is not counted. The non-thinking control is a maths answer and not all of ordinary prose. The follow-on is a temporal association, and the local baseline does not make it evidence of function. Counts cannot show what a word does.


## Amendment 1, 2026-11-03 05:59 CT

Made AFTER the original four generated sets were scored, so it is stated this way. It adds runs and changes no
prediction, rule or bar.

**Why.** A diagnostic (`part178/diag178f.py`, results in `results/part178/diag178f/`) compared the Arc's greedy
decodes with a CPU float32 reference on 8 short prompts. On the Arc, float32 matched the CPU on 8 of 8 prompts at batch
1 and 2 with both attention kernels. bf16 with the default SDPA kernel left the CPU's path at positions 65 to 73, where
the CPU's top-two margin was up to 5.25: on Qwen3.5-4B in 6 of 8 prompts, on Qwen3.5-0.8B in 2 of 8. bf16 with the eager
kernel differed only where the CPU's margin was at most 0.12, which is bf16 rounding. Single forward passes (prefill) at
lengths 56 to 132 matched at every confident position in both kernels. So the Arc's bf16 SDPA kernel changes decode
once the cache holds more than 64 keys. The four generated sets here were decoded that way, and nearly every counted
opener lies past position 64.

**What changes.** All four generated sets (`R8`, `N8`, `G`, `N4`) are generated again with
`--attn eager`, with everything else unchanged (same problems, sampler, seeds, cap, batch 16), into
`results/part178/eager/`, and scored with the same scorer and the same gates. The claim and the verdicts rest on the
eager runs. The original SDPA runs stay in `results/part178/` and are reported beside them as a comparison, with no
verdicts of their own. This is fixed now, before any eager row exists.

**Correction to Amendment 1, 2026-11-03 12:41 CT** (found by the cold check of the eager README, after the
eager runs were scored). "Everything else unchanged (... batch 16)" was wrong for the 0.8B sets: the SDPA runs of `R8`
and `N8` used batch 64 (`results/part178/launch.jsonl`), and the eager reruns used batch 16. `G` and `N4` used 16 in
both. So for the 0.8B sets the SDPA-to-eager comparison changes the batch as well as the kernel. The eager runs remain
the registered result; no rule, bar or prediction changes. A later batch-16 check (`part178/diag178g.py`) also showed
eager bf16 leaving the CPU float32 path at confident positions in 2 of 16 rows on the 0.8B, so "differed only where the
CPU's margin was at most 0.12" holds for the 8 prompts it was measured on, not in general.
