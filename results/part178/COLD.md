# Cold check: Part 178 results README (`results/part178/README.md`)

Reader: worker-01, cold. Repository read-only. Commands run from `/home/elisepc/green/worktrees/ab-part67`, Python via `/home/elisepc/neuron-probe/.venv/bin/python -I -B`.

## 1. Every number in README.md

Recomputed from `score.json` keys, from raw JSONL rows, or from scorer logic. All match to the displayed rounding unless noted.

| README number | Source / recomputation | Result |
|---|---|---|
| 86.22 (U `wait`), rank 2, rank CI 1–2 | `predictions.178.1.estimate` rate 86.2234, rank 2, rank_ci95 1–2; `tables.window.U` 1534/17791×1000 = 86.22 | match |
| 21.16 (G `wait`), rank 8, rank CI 6–10 | `predictions.178.2.estimate`; 349/16496×1000 = 21.16 | match |
| bar lengths 20 vs 5 | 21.16/86.22×20 ≈ 4.9 → 5 | match |
| factor 1.17, CI 1.09–1.25 | `178.3.estimate.max_factor` 1.1663; CI 1.0852, 1.2523 | match |
| overlap 14 of 20 | recomputed set intersection of `178.4.estimate.U_top20` and `G_top20` = 14 (let's, wait, no, the, so, if, this, is, let, yes, we, then, since, step) | match |
| "interval 18 to 20" (line 29) | `178.4.estimate.ci95_overlap` = 18–20, but it is Gate 5's U-vs-R8 bootstrap (see item 4) | value matches the key; the README correctly flags it as not a U–G interval |
| 4 imported words, CI 3–5 | `tables.paragraph.U.rates`: wait 74.12, alternative 22.36, verify 8.31, now 8.31 per 1,000 (denominator 1565); `178.5.estimate.count_ci95` 3–5 | match |
| 0.207, CI 0.101–0.306, 66 eligible | `178.6.estimate.point` 0.20677; ci95 0.10093, 0.30557; eligible_traces 66 (49+16+1 across onset groups) | match |
| 14 of 20 under 2, CI 13–17, 0 missing | `178.7.estimate.count_under_2` 14 (list recounted); `under_2_ci95` 13–17; `without_ratio` 0 | match |
| W1: 2 differing, CI 1–3 | `178.W1.estimate.different_top20_entries` 2; ci95 1–3 | match, convention issue (item 3) |
| W2: 742.5 tokens, CI −4,242.5 to 3,515.5, 40 traces | `178.W2.estimate`: difference 742.5 = 3975.0 − 3232.5; ci95; n_both_labels 40 | match |
| W3a ratio 4.08, CI 3.61–4.60, 10,601 control openers | `178.W3a.estimate`: 4.0830, 3.6123, 4.5986, 10601; recomputed 85.5 / 20.94 = 4.08 | match |
| W3b ratio 2.21, CI 1.81–2.83, 7,847 | `178.W3b.estimate`: 2.2135, 1.8094, 2.8277, 7847; recomputed 21.16 / 9.56 = 2.21 | match |
| Spearman 0.991, threshold 0.7, 20 of 20 | `read_gates.gate5`: spearman 0.99098, threshold 0.7, U_top20_in_R8_top20 20 | match |
| Cap counts G 171 / N4 46, EOT 29 / 154 | recounted from `G.jsonl` and `N4.jsonl` `finish_reason`: G cap 171, end-of-turn 29; N4 cap 46, end-of-turn 154; 200 rows each | match |
| Cap counts for R8 and N8 | not in README. Recounted: R8 cap 198 / EOT 2; N8 cap 103 / EOT 97 | **omitted** (item 2 of verdict list) |
| sampler: penalty 1.5, temp 0.6, top-k 20, top-p 0.95, cap 2,048 | every row of G, R8, N4, N8 carries these values | match |
| 200 problems per set | each set has 200 rows, 200 unique `problem_id` | match. Identity with Part 177's first 200 rows not checked by me |
| Gates 1–5 and adequacy passed | `score-run.log`: gate1–gate5 true, adequacy true | match |
| adequacy openers | `read_gates.adequacy`: U 17791, E 18288, G 16496, R8 17707, N4 7847, N8 10601 (all ≥ 2000) | match |
| batch sizes, bf16 | not recorded in rows; not verified | unverified |

Key errors in `README-NOTES.md` (not the README itself):
- Line 19 maps "interval 18–20" to `predictions.178.4.estimate.ci95.lower/upper`. No such key exists. The field is `estimate.ci95_overlap`, and it is Gate 5's U–R8 interval. The NOTES row presents it as a source for the U–G overlap, which contradicts the README body.

## 2. Verdicts, reapplying the registered rules

I applied each registered rule (`PREDICTIONS.md` Part 178 rows; `PART178-DESIGN.md` "Reading rules" and the rule strings in `score178.py`) to the point estimates.

- 178.1: rank 2 ≤ 10 → held. Correct.
- 178.2: rank 8 ≤ 10 → held. Correct.
- 178.3: max factor 1.166 < 1.5, both rates nonzero → held. Correct.
- 178.4: overlap 14 ≥ 14 → held. Correct (exactly at the bar).
- 178.5: count 4 ≤ 6 → held. Correct.
- 178.6: point 0.207 ≥ 0.15, 66 ≥ 40 eligible → held by the point rule. Correct per the rule, but the 95% interval 0.101–0.306 contains the bar. See fix list.
- 178.7: 14 ≥ 10 under 2, 0 missing ≤ 5 → held. Correct.
- 178.W1: different = 2; rule is held ≥ 4, failed ≤ 1 → inconclusive. Correct under the one-sided count (see item 3).
- 178.W2: difference 742.5; held ≥ 2000, failed < 500 → inconclusive. Correct.
- 178.W3a: ratio 4.08 ≥ 3, control openers 10601 ≥ 500 → held. Correct.
- 178.W3b: ratio 2.21, between 1.5 and 3 → inconclusive. Correct.

No verdict should change under the scorer's conventions. Two caveats: the W1 count convention could flip it to held under the symmetric-difference reading, and the generated-set verdicts (178.2, 178.4, 178.7, W3a, W3b) depend on the pending device check.

Also: `prediction_table.json` agrees with `score.json` on all 11 verdicts (checked). The README says this agrees with "the registered expected verdicts above". No expected verdicts were registered; the README table is itself taken from `score.json`. The comparison is internal consistency, not an independent check.

Withdrawal screen (`score.json` `withdrawal`): withdrawn = false. U wait 86.22, R8/N8 ratio 4.08, so the screen passes. The README does not say so.

## 3. Sentences that overclaim or soften

1. Title (line 1): "Under this protocol, `wait` ranked among the top ten openers in both thinking sets."
   U is MIT's stored text, generated with MIT's sampler (temperature 0.7, top_p 0.8), so it is not "under this protocol". Smallest fix: "In stored and generated thinking text, `wait` ranked among the top ten openers."
2. Line 29: "All seven predictions numbered 178.1 to 178.7 held."
   Accurate per the rule, but 178.6's interval includes its bar, and 178.2, 178.4, and 178.7 rest on generated sets whose device check is pending. Smallest fix: "All seven held on their point estimates. 178.6's interval (0.101–0.306) includes its 0.15 bar. 178.2, 178.4 and 178.7 rest on generated sets whose device check is pending."
3. Line 29: "all eleven verdicts match the registered expected verdicts above."
   No expected verdicts were registered. Smallest fix: "all eleven verdicts match the verdicts in `score.json`; both are scorer output."
4. Line 37: "the top-20 `rate(w)` ranking for `R8` against stored `U` had Spearman correlation 0.991."
   The statistic is over U's top-20 words: Spearman of rate(w) between R8 and U for those 20 words, not a ranking of R8. Smallest fix: "over U's top 20 openers, the Spearman correlation of `rate(w)` between `R8` and `U` was 0.991."
5. Line 49: "The different cap counts leave different amounts of generated text reaching the cap in those sets."
   Soft. It hides the size of the difference, and omits R8 (198/200 capped) and N8 (103/200). Smallest fix: "Cap hits: G 171/200, N4 46/200, R8 198/200, N8 103/200. The thinking sets are mostly capped and the non-thinking sets are not, so the mode ratios also compare text length."
6. Line 33 heading "Could `wait` just reflect format or task rather than thinking?" The section answers task (same problems) but not format. The thinking prompt ends in an open think tag and the non-thinking prompt in an empty think block; this is never addressed. Smallest fix: add "The two modes also differ in prompt format (open think tag vs empty think block); this comparison does not separate that from thinking."
7. Line 29, device check paragraph (line 55): "a CPU check found stored Arc generations of another part leaving the CPU's greedy path." The CPU run did not find this; it compared Arc rows from Part 140 with a cloud CPU reference (per `diag178f.py` docstring). Smallest fix: name Part 140 and say the diagnostic has not yet run (`results/part178/diag178f/` does not exist).

## 4. The 178.4 interval attachment

Confirmed.
- `part178/score178.py:700-705` builds 178.4 with `"ci95_overlap": tables["gate5"].get("overlap_ci95")` (line 702).
- `part178/score178.py:867` computes that value as `_bootstrap_overlap(trace_sets["U"], trace_sets["R8"], reps, seed + 120)`, which is U against R8, not U against G.
- `part178/score178.py:871` sets `metrics["gate5"] = gate5`, and line 927 passes `metrics` as `tables`, so `tables["gate5"]` is the Gate 5 record.
- `grep -n "_bootstrap_overlap("` returns only lines 380 and 867. No U–G bootstrap exists anywhere in the scorer, so the README's "no interval was computed for U against G" is correct.

The seed (298) and bounds (18–20) match the Gate 5 record in `score.json`.

## 5. Generated rows read by hand

I ran the scorer's `words178.opener_spans` on the full `generated_text` of five rows and compared the counted openers with the text.

- G row 0 (`test/precalculus/807.json`, cap). "The user wants..." counted as `the`. "1. **Identify the given information:**" counted as `identify` (list number and bold skipped). "3. **Calculate $r$:**" counted as `calculate`. Consistent.
- G row 100 (`test/prealgebra/1622.json`, end-of-turn). "Here's a thinking process..." counted as `here's`. "*   Goal: Find the perimeter..." counted as `goal`. Consistent.
- R8 row 0 (`test/precalculus/807.json`, cap). Its 7 "wait" tokens: 6 counted at line or sentence starts ("    Wait, what if..."). One uncounted: "*(Self-Correction/Double Check)*: Wait, let me re-read the prom…". The rule does not treat a colon as a boundary, so this is an undercount. It is a layout limit the README mentions only generally.
- N4 row 5 (`test/number_theory/1055.json`, cap, thinking off). "To find the smallest...", "### Step 1: Define Variables", "Let $n$ be...", "Since each bag...", "You then find..." all counted correctly. No think tag. Consistent.
- N8 row 7 (`test/number_theory/46.json`, end-of-turn, thinking off). "However, the problem statement..." (after a period) and "Since $6 < 9$" counted. Consistent.

Caveat: my counts are on full text, not the registered first-2000-token window, so I checked the rule's behaviour, not the registered numbers.

## 6. Questions a skeptical reader would ask that the README does not answer

A. **Cap and length confound (most important).** Full-text recomputation with `words178.opener_spans` (approximation: no 2,000-token window, no onset cut), split by `finish_reason`:
   - R8 cap 88.07 per 1,000 (198 traces), N8 cap 24.50 (103 traces): ratio 3.59.
   - G cap 22.73 (171 traces), N4 cap 17.85 (46 traces): ratio 1.27.
   - Full sets: R8 87.54 / N8 21.09 = 4.15; G 20.57 / N4 9.50 = 2.17. These track the registered 4.08 and 2.21, so the approximation is reasonable overall.
   Conditioning on finish reason is post-treatment, so these are a sensitivity check, not a replacement verdict. But on capped-only text the 4B ratio falls below the 1.5 falsifier. The README needs to say the mode ratios are length-confounded.

B. **Thinking rarely closes.** Only 41 of 200 G rows and 2 of 200 R8 rows contain `</think>` (`score`: `think_close_token_id` in `generated_ids`). For most thinking rows the "thinking text" is the whole capped output. The README does not report this.

C. **Control rates are missing.** The README gives only ratios. The non-thinking rates are N8 20.94 and N4 9.56 per 1,000 openers. Without them the reader cannot judge the ratio.

D. **Spread of `wait`.** `wait` occurs in 199/200 R8 traces, 173/200 G, 120/200 N8, 55/200 N4 (`window.*.trace_rates`). Non-thinking text also contains it in 60% of traces; this is not stated.

E. **Uncut sensitivity.** The design requires `uncut` beside `window`. Uncut ranks: U `wait` 2nd (86.91), G 8th (21.13), so the ranking holds. The README omits this.

F. **Follow-on groups.** The design requires three descriptive groups. `score.json` `178.6.by_onset_side`: before onset n=49, mean 0.163 (CI 0.042–0.283); after onset n=16, mean 0.358 (CI 0.169–0.551); no onset n=1, −0.059. The README gives only the pooled mean.

G. **178.4 held at exactly 14, no U–G interval.** It is one point from failing, and the README says no interval was computed. Smallest honest fix: either compute it or say the verdict depends on a single count.

H. **W1 convention.** `score178.py:738` counts `20 − |U-rate-top-20 ∩ U-trace-top-20|` = 2. The symmetric difference is 4 (rate-only: maybe, step; trace-only: analyze, here's). Under a symmetric-difference reading the verdict is held. The design does not specify, and the README does not say which reading is used.

I. **Part 179 and generality.** The design says that if only one model's W3 holds, Part 179 runs on that model and the write-up states that generality is not established. The README says 4B is inconclusive but does not state this consequence.

J. **Withdrawal screen.** Not reported. The screen passes (`withdrawn: false`).

K. **Batch sizes.** The design says the 0.8B sets run at batch 64, but its CPU-agreement evidence (`diag178d/e`) is at batches 2–16. The batch size used for each set is not recorded in the rows, so I could not check this.

## Device check dependency

The README's device-check paragraph is pending as expected. These verdicts depend on it: 178.2 (G), 178.4 (U vs G), 178.7 (G vs N4), 178.W3a (R8 vs N8), 178.W3b (inconclusive). Gate 5 (R8 vs U) also feeds 178.4's gate block. These do not depend on it: 178.1, 178.3, 178.5, 178.6, 178.W1, 178.W2 (all U/E only). The device-check paragraph should list these dependencies. The diagnostic output directory `results/part178/diag178f/` does not exist yet.

## Verdict

PUBLISHABLE AFTER FIXES: (1) title and line 29 "under this protocol" for stored U; (2) cap counts for all four sets, and a stated length/cap confound with the capped-only sensitivity (item 6A); (3) 178.6's bar inside its interval, and 178.2/178.4/178.7/W3a/W3b dependence on the device check; (4) "registered expected verdicts" wording; (5) Spearman wording at line 37; (6) state the W1 counting convention; (7) add control rates, uncut ranks, follow-on groups, withdrawal status and the Part 179 consequence; (8) correct the 178.4 key and source row in `README-NOTES.md`; (9) name the batch sizes or state they are unrecorded.
