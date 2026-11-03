# Cold check: Part 178 results README, eager version (`results/part178/README.md`)

Reader: cold worker (worker-01). I did not write the README or its scripts. Repository read-only except this folder. Paths below are relative to the repo root. "README line N" means the line number in `results/part178/README.md`. "score.json" means `results/part178/eager/score.json` unless stated. Raw-row checks used my own code, run with `/home/elisepc/neuron-probe/.venv/bin/python -I -B`. Opener counts used the scorer's own `part178/words178.py:opener_spans`, imported as is.

## 1. Every number in the README

Match means the README value equals the recomputed or source value to the README's rounding.

| README line | README number | Source or recomputation | Result |
|---|---|---|---|
| 1, 3 | `wait` rank 2 in U, rank 10 in G | score.json `predictions.178.1.estimate.rank` = 2; `predictions.178.2.estimate.rank` = 10 | match |
| 6-9 | U 86.22, G 19.59 per 1,000 | `tables.window.U/G.rates[wait].rate_per_1000` = 86.2234, 19.5924 | match |
| 11 | bar lengths 20 and 5 | 19.59 / 86.22 x 20 = 4.5, shown as 5 | match |
| 17 | 178.1 rate 86.22; rank interval 1 to 2 | `178.1.estimate.rate`, `.rank_ci95` lower 1.0, upper 2.0 | match |
| 18 | 178.2 rate 19.59; rank 10; interval 7 to 11 | `178.2.estimate.rate`, `.rank`, `.rank_ci95` 7.0 to 11.0 | match |
| 19 | 178.3 factor 1.17; interval 1.09 to 1.25 | `178.3.estimate.max_factor` 1.1663; `.max_factor_ci95` 1.0852 to 1.2523 | match |
| 20 | 178.4 overlap 14 of 20 | `178.4.estimate.overlap` = 14; I recounted the two top-20 lists by hand and got 14 | match |
| 21 | 178.5 count 4; words; interval 3 to 5 | `178.5.estimate.count` 4; `.words` wait, alternative, verify, now; `.count_ci95` 3.0 to 5.0 | match |
| 22 | 178.6 mean 0.207; interval 0.101 to 0.306; 66 traces | `178.6.estimate.point` 0.20677; `.ci95` 0.10093 to 0.30557; `.eligible_traces` 66 | match |
| 23 | 178.7 count 15; interval 13 to 16; 0 missing | `178.7.estimate.count_under_2` 15; `.under_2_ci95` 13 to 16; `.without_ratio` 0 | match |
| 24 | W1 count 2; interval 1 to 3 | `178.W1.estimate.different_top20_entries` 2; `.ci95` 1 to 3 | match |
| 24, 33 | W1 symmetric difference 4 | Set difference of `rate_top20` and `trace_top20`: rate-only maybe, step; trace-only analyze, here's; total 4. Also `readme_numbers.json.w1_top20_symmetric_difference.count` = 4 | match |
| 25 | W2 difference 742.5; interval -4,242.5 to 3,515.5; 40 traces | `178.W2.estimate.difference_tokens`, `.ci95`, `.n_both_labels` | match |
| 26 | W3a ratio 4.66; interval 4.13 to 5.33; 10,432 control openers | `178.W3a.estimate.wait_ratio` 4.6646; `.ci95` 4.1323 to 5.3301; `.control_openers` 10432. Also 86.298 / 18.501 | match |
| 27 | W3b ratio 1.96; interval 1.57 to 2.62; 7,813 control openers | `178.W3b.estimate.wait_ratio` 1.9625; `.ci95` 1.5697 to 2.6182; `.control_openers` 7813. Also 19.592 / 9.983 | match |
| 31 | 7 numbered predictions, 11 total | `readme_numbers.json.summary_counts` 7 and 11; I counted the keys in score.json | match |
| 35 | W3 held model count 1 | `summary_counts.w3_held_model_count` = 1; W3a held, W3b inconclusive | match |
| 41-46 | R8 86.30 (199/200); N8 18.50 (110/200); G 19.59 (168/200); N4 9.98 (52/200) | `readme_numbers.json.window_wait.<set>`; numerator/denominator 1533/17764, 193/10432, 324/16537, 78/7813 | match |
| 50 | Uncut: U 86.91 rank 2; G 19.85 rank 9 | `tables.uncut.U/G.rates[wait]` = 86.906, 19.846; ranks 2 and 9 from the sorted rates | match |
| 54 | Think-close: 3 of 200 R8 rows; 39 of 200 G rows | Raw JSONL recount: `think_close_token_id` in `generated_ids` = 3 (R8), 39 (G); the text `</think>` appears in the same rows | match |
| 56-61 | Cap / end-of-turn: R8 197/3; N8 97/103; G 175/25; N4 46/154 | Raw JSONL recount of `finish_reason`, 200 rows each | match |
| 69-72 | All 16 cells of the full-text table (openers, wait openers, rates, rows) | My recount using `opener_spans` on each `generated_text`: R8 cap 1617/18081 = 89.43, EOT 5/230 = 21.74, all 1622/18311 = 88.58; N8 178/7863 = 22.64, 29/3050 = 9.51, 207/10913 = 18.97; G 333/15703 = 21.21, 6/1752 = 3.42, 339/17455 = 19.42; N4 67/3433 = 19.52, 13/4500 = 2.89, 80/7933 = 10.08. Row counts 197/3, 97/103, 175/25, 46/154 | match, all 16 |
| 76 | Capped ratios 3.95 (0.8B) and 1.09 (4B); all-row ratios 4.67 and 1.93 | `readme_numbers.json.capped_sensitivity_full_text.thinking_to_control_wait_rate_ratios`; recomputed 89.43/22.64, 21.21/19.52, 88.58/18.97, 19.42/10.08 | match |
| 80-88 | Follow-on pooled 0.207; before onset 0.163 (0.042 to 0.283, n 49); after onset 0.358 (0.169 to 0.550, n 16); no onset -0.059 (n 1) | `178.6.estimate.by_onset_side`; copied in `readme_numbers.json.follow_on_by_onset_side` | match |
| 90 | Withdrawal false; wait 86.22 in U and 19.59 in G; 0.8B mode ratio at least 2 | `withdrawal.withdrawn` false; `.reflection_rates_U_G`; `.reflection_mode_ratios_G_N4_R8_N8.R8.wait` 4.66 | match |
| 94 | Eager attention; cap 2,048; temperature 0.6; top-k 20; top-p 0.95; presence penalty 1.5 | Every row of all four JSONL files has these values (checked row by row) | match |
| 94 | "Batch sizes are not recorded in the rows" | True: no batch field in the rows. But `results/part178/launch.jsonl` records the batch for every run (see Section 3, item C) | true, incomplete |
| 96 | Prefix 2,048; window 2,000 | `readme_numbers.json.definitions` | match |
| 98 | Window totals 7,813 to 18,288 openers | `read_gates.adequacy.by_set`: minimum N4 7813, maximum E 18288 | match |
| 98 | Gates 1-5 and adequacy passed | `read_gates.gate1`-`gate5` and `adequacy` all true; `eager/score-run.log` agrees | match |
| 98 | Spearman 0.986; threshold 0.7; 20 of 20 | `read_gates.gate5.spearman` 0.98609; `.threshold` 0.7; `.U_top20_in_R8_top20` 20 | match |
| 29 | 178.4 `ci95_overlap` is Gate 5's U-R8 interval, not a U-G interval | Code: `part178/score178.py:702` copies `tables["gate5"]["overlap_ci95"]`; `:867` computes it as `_bootstrap_overlap(U, R8, ...)`. Value 19 to 20, seed 298 | match |
| 106-117 | SDPA column: G 21.16, rank 8, 6 to 10; R8 85.50 (199/200); N8 20.94 (120/200); N4 9.56 (55/200); G uncut 21.13, rank 8; 178.7 count 14, 13 to 17; W3a 4.08, 3.61 to 4.60, 10,601; W3b 2.21, 1.81 to 2.83, 7,847; Gate 5 0.991, 20 | `results/part178/score.json` (SDPA), keys as the README cites. The uncut rank 8 is from the SDPA `tables.uncut.G.ranking` list | match |
| 117 | SDPA Gate 5 U-R8 interval 18 to 20; eager 19 to 20 | SDPA `read_gates.gate5.overlap_ci95` 18 to 20; eager 19 to 20 | match |
| 121 | All 11 verdicts identical in SDPA and eager; 178.1, 178.3, 178.5, 178.6, W1, W2 estimates unchanged | Compared every verdict and these estimates in both score files | match |

Summary: every number and interval I checked matches its source. I found no transcription error. The window-level token rates (for example 86.22) were checked against their numerators and denominators in score.json, not recomputed from raw tokens.

## 2. Verdicts re-derived from the registered rules

Rules are quoted from `predictions.<id>.rule` in score.json, which matches PART178-DESIGN.md and the Part 178 rows of PREDICTIONS.md (lines 3347-3361).

| Prediction | Rule | Point estimate | Derived | README | Agree |
|---|---|---|---|---|---|
| 178.1 | rank at most 10 | rank 2 | held | held | yes |
| 178.2 | rank at most 10 | rank 10 (at the bar) | held | held | yes |
| 178.3 | both rates nonzero; max factor below 1.5 | 1.166 | held | held | yes |
| 178.4 | overlap at least 14 | 14 (at the bar) | held | held | yes |
| 178.5 | count at most 6 | 4 | held | held | yes |
| 178.6 | mean at least 0.15 and at least 40 eligible | 0.207; 66 eligible | held | held | yes (point rule only; interval includes 0.15) |
| 178.7 | at least 10 with ratio under 2; at most 5 missing | 15; 0 missing | held | held | yes |
| 178.W1 | held at 4 or more; failed at 1 or fewer | 2 (one-sided count); 4 (symmetric) | inconclusive (one-sided) / held (symmetric) | inconclusive | yes, convention disclosed |
| 178.W2 | held at 2000 or more; failed under 500 | 742.5 | inconclusive | inconclusive | yes |
| 178.W3a | held at 3 or more; failed under 1.5; at least 500 control openers | 4.66; 10,432 | held | held | yes |
| 178.W3b | held at 3 or more; failed under 1.5 | 1.96 | inconclusive | inconclusive | yes |

No verdict changes under the scorer's conventions. The only convention-dependent verdict is W1, and the README says so. `prediction_table.json` agrees with score.json on all 11 verdicts (checked by code). The README's "agree with the registered expected verdicts" wording is gone; agreement is now described as scorer output, which is accurate.

## 3. Sentences that claim more or less than the numbers show

These are in order of importance.

A. **README line 100: "The eager reruns resolve the device check."** This claims more than the evidence. Evidence in the repo:
- `results/part178/diag178g/small-bf16-eager-b16-xpu.jsonl` against `small-fp32-sdpa-b16-cpu.jsonl`. These are 16 pilot problems, 256 greedy tokens, Qwen3.5-0.8B, batch 16, the batch the eager reruns used. Eager bf16 diverged from the CPU fp32 path in 12 of 16 rows. Three divergences were at CPU top-two margins above 0.12 (0.631 at position 33; 0.177 at position 0; 2.67 at position 146). Margins are defined at `part178/diag178g.py:13`, `:57`, `:85`. The Arc's fp32 run matched the CPU in 16 of 16 rows.
- Amendment 1 (`PART178-DESIGN.md:108-115`) says eager bf16 differed from the CPU "only where the CPU's margin was at most 0.12." That was measured on 8 short prompts at batch 1 and 2 (`diag178f`), not on batch 16.
- No CPU reference was run for Qwen3.5-4B at all. `diag178g` has no 4B CPU file.
Generated-set verdicts 178.2, 178.4, 178.7, W3a and W3b rest on the 4B and 0.8B eager sets at batch 16.
Smallest fix, in the README's style: "The device check is only partly closed. On 8 short prompts at batch 1 and 2, eager bf16 matched the CPU float32 path except where the CPU margin was at most 0.12 (PART178-DESIGN.md, Amendment 1). On 16 problems at batch 16 with Qwen3.5-0.8B, 12 of 16 greedy rows left the CPU path, 3 of them at margins above 0.12 (results/part178/diag178g/). No CPU check was run for Qwen3.5-4B. The generated-set verdicts 178.2, 178.4, 178.7, W3a and W3b depend on this."

B. **README line 100: "The diagnostic and rerun are recorded in PART178-DESIGN.md and eager/score-run.log."** Partly wrong. The design records only `diag178f` (Amendment 1). `diag178g` is not mentioned in the design, the README, or any file in `eager/`. `eager/score-run.log` holds only the gate and verdict JSON (one line). Smallest fix: "The diagnostics are in results/part178/diag178f/ and results/part178/diag178g/; Amendment 1 in PART178-DESIGN.md describes diag178f; the gate and verdict log is eager/score-run.log."

C. **Batch size changed for the 0.8B sets, and the README does not say so.** `results/part178/launch.jsonl` shows:
- SDPA runs: R8 at `--batch 64` (2026-11-02 22:11:51); N8 at `--batch 64` (22:11:51 and 22:34:43); G and N4 at `--batch 16`.
- Eager runs: R8 at `--batch 16` (06:00:46 and 06:41:25); N8 at 16 (07:31:36); G at 16 (08:20:49); N4 at 16 (10:04:10).
Amendment 1 (`PART178-DESIGN.md:117-121`) says the reruns kept "everything else unchanged (... batch 16)". That is wrong for R8 and N8. The design's optimization section (line 73) planned the 0.8B at batch 64. So the SDPA to eager change for the 0.8B sets (the rows in the README's SDPA table, lines 106-117) mixes kernel and batch, and the README attributes it all to the kernel. Smallest fix, under line 94: "The 0.8B SDPA sets ran at batch 64 and the eager reruns at batch 16; the 4B sets were at batch 16 in both (results/part178/launch.jsonl). Changes in the SDPA comparison therefore reflect kernel and batch together." Also correct Amendment 1 in the design, or add a dated note there.

D. **Title (line 1): "Stored and eager thinking text both place `wait` among the top ten openers."** Point-only. G's rank interval is 7 to 11 (line 18), and the 4B result is a single count at the bar. Smallest fix: "...both place `wait` among the top ten openers on point estimates." This matches the body.

E. **Missing: rates are conditional on the sampler (line 94 lists the settings but does not say so).** PART178-DESIGN.md line 30 says "these rates are rates under that setting." Smallest fix, after line 94: "Rates are conditional on presence penalty 1.5, which lowers repeats of the same word; stored U and E were generated with MIT's own sampler and are not on this protocol."

F. **Missing: limits (design lines 98-100).** The README has no limits paragraph. The design says: "Two models, one benchmark, one sample per problem, MATH-500 only. U and E are mostly capped and looped traces ... Counts cannot show what a word does." Smallest fix: add a Limits paragraph with these sentences. I did not re-verify the "mostly capped and looped" claim for U and E; it is the design's own statement.

G. **Line 39: "Controls are mathematical answers, not general prose."** Accurate but thin. Smallest fix: "Controls are non-thinking answers to the same maths problems, not general prose."

H. **Line 76: capped-only 4B ratio 1.09 "below the registered 1.5 falsifier."** Accurate. The README already says it is not a replacement verdict. No change needed, but the consequence should be explicit: "On capped rows alone the 4B ratio is 1.09, under the 1.5 falsifier; the registered verdict stays inconclusive."

I. **Line 33 (W1).** Now discloses the one-sided and symmetric counts. The registered text does not say which, and `part178/score178.py:738` uses the one-sided count. The README could add "the registered text does not specify which count." No other change.

J. **Line 29, 31 (178.6 and 178.4 wording).** Correct. 178.6's interval includes its bar (line 31). 178.4's overlap is "a single count of 14" (line 29). Both are honest.

K. **Line 35 (W3 consequence).** Correct per the design's Reading rules (line 96).

## 4. Fixes from the earlier COLD.md (`results/part178/COLD.md`, SDPA version)

| Earlier item | Status | Where |
|---|---|---|
| 1. Title and line 29 "under this protocol" for stored U | Applied | Title is now "Stored and eager thinking text ..."; "under this protocol" does not appear |
| 2. Cap counts for all four sets and the length/cap confound with capped-only sensitivity | Applied | Lines 54-76 |
| 3. 178.6 bar inside its interval; device-check dependence of 178.2, 178.4, 178.7, W3a, W3b | Partly applied | Bar: line 31 and 80. Dependence is listed in line 100, but the paragraph says the check is resolved (Section 3, item A) |
| 4. "Registered expected verdicts" wording | Applied | Line 31 |
| 5. Spearman wording | Applied | Line 98 |
| 6. State the W1 counting convention | Applied | Line 33 |
| 7. Control rates, uncut ranks, follow-on groups, withdrawal status, Part 179 consequence | Applied | Control rates lines 41-46; uncut line 50; groups 82-88; withdrawal 90; Part 179 line 35 |
| 8. 178.4 key and source row in `README-NOTES.md` | Applied | `README-NOTES.md`, "178.4's attached interval field" row, now cites `estimate.ci95_overlap` and `read_gates.gate5.overlap_ci95` and says it is the U-R8 interval |
| 9. Name batch sizes or state they are unrecorded | Partly applied | Line 94 says unrecorded. It does not give the batch sizes, and misses the change (Section 3, item C) |
| Section 3 fixes 1-6 (sentences) | Applied | Title, line 31, line 33, line 98, line 54-63, line 39 |
| Section 3 fix 7 (device-check paragraph: Part 140, "diag178f does not exist") | Replaced, with a new problem | The old paragraph is gone. `results/part178/diag178f/` now exists, and the new paragraph overclaims (Section 3, item A) |
| Question A (cap and length) | Applied | Lines 65-76 |
| Question B (thinking rarely closes) | Applied | Line 54 |
| Question C (control rates) | Applied | Lines 41-46 |
| Question D (spread of wait) | Applied | Column "Traces with `wait`" |
| Question E (uncut) | Applied | Line 50 |
| Question F (follow-on groups) | Applied | Lines 82-88 |
| Question G (178.4 at exactly 14, no U-G interval) | Applied | Lines 20, 29 |
| Question H (W1 convention) | Applied | Line 33 |
| Question I (Part 179 generality) | Applied | Line 35 |
| Question J (withdrawal screen) | Applied | Line 90 |
| Question K (batch sizes) | Partly applied | See item 9 above |
| Earlier "Key errors" in README-NOTES (18-20 mapping) | Applied | Now maps to `estimate.ci95_overlap` |

## 5. Questions a skeptical reader would still ask

1. **Did the 0.8B batch change matter?** R8 and N8 went from batch 64 (SDPA) to batch 16 (eager). The README's 0.8B comparisons do not separate that from the kernel change. Rows are per-seed, but bf16 logits can depend on batch shape (design line 28). The README needs a sentence on this (Section 3, item C).
2. **Does the 4B path match a CPU reference at batch 16?** No check exists (Section 3, item A).
3. **How many "wait" mentions does the opener rule miss?** I counted every case-insensitive `\bwait\b` in the full generated text: R8 1,830 mentions against 1,622 wait openers; N8 239 against 207; G 396 against 339; N4 94 against 80. So the rule misses about 11% (R8), 13% (N8), 14% (G) and 15% (N4) of mentions. The shares are similar across sets, so the mode ratio should move little. The design (line 100) names this limit; the README does not give the number.
4. **What sampler generated U and E?** The README says they were "not generated under the eager protocol" (line 94) but does not give their sampler. The design says MIT's non-thinking settings were 0.7 and 0.8 (line 28). The U-to-G comparison (178.1, 178.2, 178.4) mixes model, sampler, and mode.
5. **Do U and E reflect looping?** The design says they are mostly capped and looped (line 100). If so, the top-ranked `wait` in U could partly reflect repetition. The README does not check this.
6. **Which W1 count was registered?** The README discloses both counts, but a reader will ask which one was fixed before the run. The registered text does not say, so W1's verdict is convention-dependent.
7. **Are 178.2 and 178.4 stable?** Both are at the bar (rank 10; overlap 14). 178.4 has no U-G interval (line 20), so one trace could change the verdict. The README says so, but does not give a sensitivity check.
8. **The R8 rig check.** The design says R8 "checks our rig against MIT's stored traces of the same model" (line 25). The README gives R8 86.30 against U 86.22 (lines 43 and 17) but does not say that this is the check, and the only formal test is the Gate 5 rank check. One sentence would close this.
9. **Two R8 eager launches.** `launch.jsonl` shows R8 eager runs at 06:00:46 and 06:41:25. The final R8.jsonl has 200 rows with 200 unique problem ids, so there are no duplicates. The README does not say whether the second launch resumed or restarted.
10. **Is "thinking" the right word for G?** 161 of 200 G rows never contain `</think>` (line 54). For those, the "thinking text" is the whole output, which may include an answer. The README says so (line 54). A reader may still ask whether the 4B's window is mostly answer text.

## 6. Verdict

**PUBLISHABLE AFTER FIXES.**

Fixes, in priority order:
1. Rewrite line 100 so it does not say the device check is resolved. Scope it to 8 prompts at batch 1 and 2, state the batch-16 0.8B result (12 of 16 rows diverged, 3 at margins above 0.12), state that there is no 4B CPU check, and name the affected verdicts (Section 3, item A).
2. Correct line 100's source sentence: diag178g is not in the design or README; score-run.log holds only gate and verdict results (item B).
3. Disclose the batch change: 0.8B SDPA at batch 64, eager at batch 16, 4B at 16 in both. Add this to the SDPA section and to line 94. Correct Amendment 1's "batch 16 unchanged" wording (item C).
4. Add the sampler condition to line 94 ("Rates are conditional on presence penalty 1.5", item E).
5. Add a Limits paragraph: one benchmark, one sample per problem, two models, U and E mostly capped and looped, U's sampler not on this protocol (item F).
6. Add "on point estimates" to the title (item D).
7. Optional but recommended: give the opener rule's miss rate (about 11 to 15% of `wait` mentions, Section 5 item 3), and say the R8-to-U rate check is the rig check (Section 5 item 8).

Items 1 to 5 change what the README says about its evidence. Items 6 and 7 are wording or added context.

## 7. What I did not verify

- I did not re-tokenize U or E or recompute window-level rates from raw tokens. Window rates, intervals and verdicts were checked against score.json's numerators, denominators and point estimates, and against the registered rules.
- I did not re-run the scorer or the diagnostics. The gate results come from `eager/score-run.log` and score.json `read_gates`.
- I checked scorer lines 702, 738 and 867 by reading. I did not step through the code.
- The "mostly capped and looped" statement for U and E is the design's own, which I did not verify.
- The diag178g margin reading assumes the script's recorded top-two margin is the CPU reference's margin at each position (`part178/diag178g.py:13, 57, 85`).
