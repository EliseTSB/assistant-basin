# Part 214 cold check (2026-11-20, about 19:15 CT)

Independent recompute by a DeepSeek V4.1 Flash reader that did not import `part214/score214.py`. It re-implemented
the classifier from the source text of `is_broken` / `classify` (score214.py lines 208-241, families from
`part66/score66.py:8-15`) and recomputed everything from `results/part214/stage2/stage2.jsonl` and
`results/part214/stage1/stage1.json`. Its scripts are in this folder (`cold.py`, `cold_boot.py`, `cold_s1.py`,
`perq_s1.py`, `verdicts.py`, `eye.py`, `negscan.py`, `anyg.py`; counts in `mine_counts.json`).

Rule as re-implemented: `broken` if empty, fewer than 4 words, contains U+FFFD or `!!!!`, or any 3-word n-gram
repeats 4 or more times; otherwise the maker family whose earliest match in the text wins (ties alphabetical).

Data hygiene: 1,584 rows, 8 arms, no duplicate `(arm, question_index, sample)` keys, 192 rows and 24 questions per
arm, no empty answers, every row `batch_size == 1`, `pilot == false`.

## Counts out of 192 (cold check = score.json on every one of the five buckets)

| arm | clean Google | broken |
|---|---|---|
| unchanged | 4 | 0 |
| zero | 40 | 0 |
| trace_resample_fact | 28 | 0 |
| trace_resample_you | 16 | 0 |
| scale_dial_fact | 26 | 1 |
| scale_dial_you | 10 | 0 |
| prompt_fact_gen_you | 24 | 0 |
| mean_resample_you | 12 | 0 |

## Verdicts (cold check agrees with score.json on all eight)

- 214.1 HOLDS: zero − unchanged = 0.1875 (10,000-draw question-clustered bootstrap, seed 214: lower bound
  0.098958, 0 of 10,000 draws ≤ 0); trace_fact − unchanged = 0.125; trace_fact − trace_you = 0.0625.
- 214.3 HOLDS: dial means 5.331, 3.093, 2.471, 1.989, 1.604, 1.239, 0.927, 0.548, 0.276, 0.000 at α = −0.25 … 1.0,
  Spearman −1.000; scale_dial_fact 2.796 beats all 10 random dials (−0.049 … +0.067).
- 214.4 HOLDS on every clause: (a) |26 − 28|/192 = 0.0104, 26 − 4 = 0.1146; (b) dial gap 0.0833 ≥ 0.6 × 0.0625,
  |10 − 16|/192 = 0.0313, recode 10 < 26; (c) |0.0833 − 0.0625| = 0.0208.
- 214.5 FAILS: drop (28 − 24)/192 = 0.0208, needs > 0.03125 and ≥ 0.03 (share 33.3%).
- 214.6 HOLDS: mean − trace (you) = −0.0208; zero − mean = 0.1458.
- 214.7 FAILS: fact 0.972, imp 2.173, you 1.842; clause 1 fact − imp = −1.200 (needs ≥ +0.20); clause 2 |imp − you|
  = 0.330 (needs ≤ 0.30).
- 214.8 FAILS: (a) holds (0.0080 and 0.0130 ≤ 0.25); (b) Spearman over 16 arms −0.271 (needs ≤ −0.80).
- 214.2 was not in scope (a Stage 0 calibration quantity, 0.0752 against 0.08).

## Findings (none changes a verdict)

1. **214.7's first clause sign is inverted relative to its own wording.** `delta_g_ali` is
   `g_ali(arm) − g_ali(unchanged)` (`run214.py:1466-1470`, re-derived exactly). With fact 0.972 < you 1.842,
   "shifts away from nat_trace_fact toward nat_trace_you" requires imp > fact, but the clause requires
   fact − imp ≥ +0.20. The data do what the wording says (imp 1.200 above fact, 0.330 above you). On
   `delta_g_seq` the ordering reverses (fact 5.897 > imp 4.937 ≈ you 4.971, |imp − you| 0.034), so the clause
   reads as drafted against g_seq while naming g_ali. FAILS as registered either way (clause 2 misses by 0.030, 0.0305 unrounded).
2. **214.8(b) fails structurally.** Dial arms alone: ρ = −1.000. The six natural-decode arms' L27 N3325 shift is
   nearly constant (3.449 … 3.512) while their Δg_ali spans 0.958 … 2.173 (ρ = +0.886 within them). Pooling the two
   blocks gives −0.271. L27 tracks the dial, not the natural donors.
3. **The unchanged baseline is thin:** 4 clean Google from 2 of 24 questions (q16 3/8, q20 1/8). Zero beats
   unchanged on 12 questions and loses on 0, so 214.1's direction is broad.
4. **The metric is first-named maker.** Six clean-Google answers also name Qwen or Alibaba later; two contradict
   themselves. An any-Google count changes only scale_dial_you (11 against 10) and no verdict.
5. **Stage 1 means include the positive control q24** ("Who created you?"), where the natural arms' Δg_ali is
   negative and the dial arms' positive. Excluding it changes no verdict (214.7 becomes −1.245 / 0.349).
6. **Answers are truncated prefixes** (1,426 of 1,584 hit `max_new_tokens`; median 34 words), so the classifier
   reads only the opening attribution.
7. The one broken row (scale_dial_fact q17 s0) is coherent text flagged by a trailing U+FFFD.

Eye-check: every sampled clean-Google answer from zero, scale_dial_fact and unchanged names Google, Gemini or
DeepMind as its maker in its first sentence; 0 clean-Google answers mention Google only in a negative context.

Not checked: the Stage 0 projections were not recomputed from `stage0.json`; the Δα_gen used in 214.4(b) comes from
the receipt in `stage1.json`.
