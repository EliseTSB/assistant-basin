# Part 177: Among answers MIT's unedited Qwen3.5-0.8B never finished, how many had already stated the right answer?

Procedure: v1
Registered 2026-11-02 18:26 CT by Claude (Opus 5.5), before any outcome was read. Elise sent the post on 2026-11-02. The commit that adds this file is the registration.

Seen before registering, and no more: in the first 200 stored texts of each arm, character lengths, the count holding `\boxed` (39 unedited, 117 `p50c`), the count holding `</think>` (13 and 152), and the last 300 characters of one trace in each arm. No accuracy, `correct`, `pred_first`, gold search or loop count was read. The field names of `math500_per_problem.json` were printed from one shard without its values.

## Question

Takehi, Teoh, Zhang and Khattab edited 0.016% of Qwen3.5-0.8B's weights and MATH-500 rose from 7.9% to 48.1%, with answers cut at the token limit falling from 93.0% to 33.1%. They write that the edit "could remove an obstacle that hid an ability". This part reads their stored answers and asks one narrow thing: of the unedited traces that ran out of tokens while still thinking, how many had already stated the right answer as a result? It follows Parts 165 to 170 on loops.

It cannot show that the edit changed what the model computes or only whether it can stop. A literal match is not an ability, and its absence is not the absence of one.

## Prior art and existing data

- Closest published work: [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and its [logs](https://github.com/RikiyaT/lm_understand-logs), which report accuracy and cut share. The post says fixing loops "lets the model finish answers it could already work out", and does not audit the cut answers for that. Their `loop_signals` in `code/mathqa.py` is a published rule for a looping end.
- From Muse's search of 2026-11-02, cited from Muse's report and not read by Claude: nobody found audits limit-cut answers for this. Caldarella et al., arXiv:2606.02835, extract a prediction from each prefix of complete traces. Huang et al., arXiv:2602.08354, give the first correct step in completed traces. Wang et al., arXiv:2501.18585, judge whether early thoughts could lead to the answer. Liao et al., arXiv:2505.12992, and Muennighoff et al., arXiv:2501.19393, force an answer from a prefix. Su et al., arXiv:2602.14444, warn that a truncated chain can mislead a forced answer. No validated rule for loop onset was found, only practitioner detectors that miss drifting loops and flag legitimate repetition. No reproduction of the post was found 21 hours after it appeared.
- Our own data that bears on it: Parts 165 to 170 measured loops under pushes on Qwen3.5-4B, a different model. No part has read the edit.

## Method

Logs pinned at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. No generation. Two arms from `exp8_math/eval/`: `unedited` and `p50c`, the edit reported in the post.

Inputs, downloaded once, hashed in the results:
- `parts/math500_s{0..9}of10/math500_per_problem.json`: 500 problems, each with `id`, `correct` (one int per sample, four samples) and `pred_first`.
- `parts/math500_s{0..9}of10/math500_first_samples.jsonl.gz`: the stored text of sample 0 for the first 20 problems of each stride shard, so the first 200 input rows and not a random 200, in both arms, the same ids. `correct[0]` grades this same sample. Sampled at temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, thinking on, up to 32,768 new tokens, on vLLM by the authors.
- `eval/{arm}/merged.json`: `math500.acc` and `math500.cut_share`.
- `HuggingFaceH4/MATH-500`, `test.jsonl` at revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, the dataset their code loads, for `problem` and `answer`.
- `tokenizer.json` of `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`, the release their edit names.

Definitions, by code in `part177/read177.py`:
- `capped`: the stored text retokenises to at least 32,500 tokens. Finish reason is not stored per trace, so this recovers per-trace cap evidence. The table of `capped` against `</think>` is reported.
- Cohort `C`: `capped` with no `</think>`, in the `unedited` arm. These ran out while still thinking.
- `literal hit`: the gold answer appears in the text after normalising spaces, `\left`, `\right`, `\!`, `\,`, `\dfrac` to `\frac`, `$` and a trailing `.`. A bare integer or short token must stand alone, not inside a longer number. This is a string match and is not called reaching anything. Mathematically equal forms it misses, such as 0.5 for \frac{1}{2}, are a stated limit.
- `in prompt`: the same search finds the value in the problem text.
- Target set `T`: the problems of `C` whose gold is not `in prompt`. Fixed once. Real and shuffled rates use the same `T` and the same weights.
- `asserted`: of the first, middle and last literal hits in a trace, at most three, a reader says at least one states this value as the result of the problem, given the problem text and 800 characters around the hit. The reader is told the value and not whether it is gold, and is told not to solve the problem. Its labels: `asserted`, `intermediate`, `quoted from the problem`, `rejected or doubted`, `ambiguous`.
- `box cell`, for a trace with a literal hit: `correct[0]` is 1; or `pred_first` is a box that fails `correct[0]`; or `pred_first` is null, which means no extractable final box and not no box. In a trace with no `</think>`, `pred_first` is the last box in the whole text, by their `answer_text`.
- `loop onset`: the first line, of lines over 8 characters after normalising, such that the same line occurs at least 3 more times in the next 40 lines. None if no line qualifies. `onset order` has three values and only the first two are comparisons: hit before detected onset, onset before hit, no onset detected.
- `end loop`: their `loop_signals` on the text, either signal, reported beside the line rule. `verbatim_loop_end` is copied from the pinned `mathqa.py`. `repetitive_end` calls `mcqa.loop_signals`, which is not in the published logs, so it is rebuilt from the rule stated in `mathqa.py`'s docstring and labelled rebuilt.
- `looped`: a trace in `C` with a detected onset or an `end loop`. A capped trace with neither is `merely long`. The loop story predicts only the first, so rates are reported for each.

## Controls

- Shuffled gold, a specified label permutation and not a measure of chance reasoning. For each target in `T`, 20 donors drawn with a fixed seed from the golds of the other 199 problems: same class (integers by digit count, short fraction, other), a different value from the target's gold, and not `in prompt` for the target. If fewer than 20 donors exist, the pool is used whole and the count is reported. A problem keeps its 20 draws and both arms in every bootstrap resample. The reader reads hits of draw 1 only.
- Both arms read by the same code on the same 200 problems.
- Inert fixtures for the loop rule in the tests: four identical lines of honest case work, a loop reformatted on each pass, a one-line repeating trace.
- Not used: random directions, poles, dose ladder and do-nothing hook, because nothing is pushed and nothing is generated. Off-target questions: a separate part can read their GPQA and IFEval rows.

Gate 1: the mean over all four entries of `correct` in the 500 problems reproduces `math500.acc` in each arm within 0.1 points.
Gate 2: `capped` as a share of the 200 lies within 8 points of `math500.cut_share` in each arm. This is a plausibility check across different frames, the first 200 rows of sample 0 against all 500 and four samples, and not a per-trace validation.
Gate 3, the literal rule finds what is there: among `p50c` traces with `correct[0]` = 1, literal hit is at least 90%. This holds on edited notation only.
Gate 4, the reader: 20 synthetic windows with labels written in advance, six of them about 800 characters as real windows are, at least 17 right; a second reader on a seeded 10% of hits, kappa at least 0.6; asserted share of real hits exceeding that of shuffled hits by at least 25 points; errors at most 2%.

If a gate fails, nothing past it is read until it is explained.

## Size

- Smallest effect worth knowing: asserted rate 15 points over shuffled on `T`, a chosen bar. `T` is probably 120 to 170 traces, a half-width near 8 points at 30%; the interval on the paired difference and the real denominator are reported.
- The paired table of the edit's success is probably 30 to 60 problems, a half-width near 13 points, so it is a rate with its interval and no verdict.
- Total: 1,000 reader calls, about 10 minutes on DeepSeek V4.1 Flash at most, up to 3 windows per trace, real and draw 1, each about 500 tokens with `max_tokens` 4,000. Before them, 400 stored texts in under a minute on the CPU. No generation.
- Pilot: not needed. Under 1,000 generations and no GPU.

## Optimization

- Bound resource: CPU, then the reader API.
- Measured: n/a, under 5 minutes of CPU; reader rate from earlier parts.
- Levers: reader calls limited to three hits per trace and draw 1.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 177.1 | On `T`, the share of traces with an `asserted` real gold exceeds the same share for shuffled draw 1 by at least 15 points | medium | Claude |
| 177.2 | On `T`, at least 25% of traces have an `asserted` real gold | medium | Claude |
| 177.3 | Among `T` traces with an `asserted` real gold and a detected onset, at least half have the first asserted hit before the onset; not read below 20 traces | medium | Claude |
| 177.4 | At least 80% of `p50c` traces that are `capped` with no `</think>` carry a detected onset. This is the persistence of the signature among residual failures and says nothing about incidence | medium | Claude |
| 177.5 | On problems in the same cohort in both arms with a detected onset in both, median onset as a share of text is later in `p50c`; not read below 15 pairs | medium | Claude |
| 177.6 | Over the 200 paired problems, with onset searched only in the first 50,000 characters of each text and pairs with a shorter text left out, the rate of a detected onset is at least 15 points lower in `p50c` | medium | Claude |
| 177.W1 | The real-against-shuffled literal-hit gap on `C` exceeds 15 points, but at least 20% of real hits sit in the first 15% of text, and excluding them the gap falls under 15. Falsifier: fewer than 10% of hits in the first 15% of text | medium | Weaver |
| 177.W2 | Shuffled windows clear 40% yes on Gate 4. Falsifier: under 20% | medium | Weaver |
| 177.W3 | Among `C` traces with a literal hit, at least a third have a `pred_first` that does not match the gold. Falsifier: under 15% | medium | Weaver |

## Reading rules

`part177/read177.py` and `part177/score177.py` write every number and verdict from the pinned files. A verdict reads the point estimate against its bar, and every verdict is reported with a percentile interval over problems. 177.4 and 177.W3 are not read below 20 traces, and 177.6 not below 15 pairs. 177.W2 and Gate 4's margin count windows. 177.W1's exclusion drops hits in the first 15% of text, and a trace counts if a hit remains, for real and shuffled alike. Real windows whose gold is `in prompt` do not go to the reader. If Gate 1, 2 or 3 fails, the reader does not run. A missing or pending label is not a clean one: it fails the error cap until it is settled. A single-line repeat is seen only by `end loop`. A text under 2,000 characters is reported and kept. Withdrawal: if 177.1 fails, "it had stated the answer and kept going" is not the reading, whatever the others do, and the score file marks it blocked.

177.W3 is read as Weaver ruled on 2026-11-02: only traces with a box that fails `correct[0]`, since null emitted nothing. The null rate is reported beside the cell, and so is the share with `pred_first` not null, so the bar is read with its denominator. Reported as a sub-case: traces where an earlier box holds a literal gold and the last box differs.

The headline is `asserted` on any of the hits read. Reported beside it, as different numbers: the first read hit asserted, the last read hit asserted, the share of read hits labelled `rejected or doubted`, and `asserted` in text before a detected onset, since in a repeating trace a value recurs and "anywhere" is biased upward. Forced continuation from a prefix and extraction from a prefix would give different rates and are not done here.

Also reported, with no verdict: `asserted` for `looped` and for `merely long` traces; `correct` per sample and any of four, in both arms, on the same problems; literal hit among `p50c` traces that closed thinking, on the problems of `T`, since a rate among capped traces is conditioned on hitting the cap.

Literal hit, `asserted` and `correct[0]` are reported separately and never merged. The `box cell` table and `onset order` table are reported in full, with no-onset traces in their own row.

Reported beside the verdicts, with no verdict: on the 200 problems, the paired table of `unedited` literal hit and `asserted` against `p50c` `correct[0]`, each as a rate with its interval. A `p50c` correct where the unedited text has no literal hit is not to be called new computation, since one sample of a sampled model and a string rule cannot show that.

## Limits

One model, one benchmark, one edit, the authors' samples on their stack. The first 200 rows of MATH-500, not a random sample. `capped` is inferred by retokenising decoded text. The literal rule misses equal forms and a reader judges a window and not the whole trace, so `asserted` is an audit-grade label and not proof. The shuffled control is a label permutation and not chance reasoning. The line rule is not loop ground truth, and a detected onset depends on how long a trace ran, which is why 177.4 and 177.6 are worded as they are. Nothing here shows how the edit works, and nothing is run on our hardware.
