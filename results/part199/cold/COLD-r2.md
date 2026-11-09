# COLD-r2.md: round-2 check of the folds of `cold/COLD.md` (Part 199, commit 0d37036)

**Verdict: SHIP-WITH-FIXES.** All 13 round-1 findings are folded, and every number the folds added matches the files. What remains is four wording fixes (findings N1 to N4 below) and one unverified mechanism sentence (N3). None changes a registered number or verdict.

Scripts (in `worker-01/scripts/`, run as `/home/elisepc/neuron-probe/.venv/bin/python -I -B`): `r2_check.py` and `r2_p191.py`. They read `results/part170/answers.jsonl`, `results/part199/answers.jsonl`, both `lists.jsonl` files and `score.json` in place. Nothing was copied, nothing in the repo was written, and `report199.py` and `fig199.py` were not run.

## 1. Round-1 findings: folded or not

| # | Status | The sentence now (or what changed) |
| --- | --- | --- |
| 1 | Folded | "The prediction rested on a premise that does not hold: that the two kernels agree below 65 keys." The contradicting "neither is the 64-key boundary" is gone. 158 / 195 / 45 / 28 / 66 recompute. |
| 2 | Folded, residual N1, N3 | "199.2 fails under a set count as well: all three lists are the same set for 3 of 200 prompts." The table matches (b5 149 set and 133 ordered; b50 102 and 0; b200 8 and 0; b50 changed path 2, min 24). |
| 3 | Folded, residual N2 | "the b5 set differs for 39, the b50 set for 96 ... and the b200 set for 192" replaces "about a fifth". All three counts match. |
| 4 | Folded | Front page, Part 170 entry: "most answers on the SDPA kernel stopped early, many mid-phrase; on eager most ran to the cap." Matches 358/379/383 against 168/104/117. |
| 5 | Folded | "170.1's point estimate ... 0.174 to 0.329, past its 0.25 bar, both intervals cross the bar" is used in the same terms in the title, the answer, the Part 170 note, the front-page intro and the front-page Part 199 entry. Intervals 0.000 to 0.443 and 0.111 to 0.630 match `score.json`. |
| 6 | Folded | The length mechanism is removed. The README now says "This part does not identify why the eager rates are higher" and gives the pattern (eager loops at the cap; SDPA loops end early). The round-1 46% decomposition is not in the README. That was optional in the fix, so it is not a defect. |
| 7 | Folded | The README and the figure now use Part 191's rule. Early-stop mid-phrase counts are 119/271/268 on SDPA and 1/12/24 on eager, which match `r2_p191.py`. The Part 191 comparison (230 of 838, 2 of 588) matches `results/part191/README.md` line 71. |
| 8 | Folded | "Part 183 found no 64-key fault on eager at batches 1 and 2 (and at 8, by hand); the 5- and 6-row no-push shapes are unchecked." Part 183 does report eager divergences, but only at near-ties (margin 0.101 or less, at most 3 of 8 per shape), so "no 64-key fault" is a fair reading. |
| 9 | Folded | The premise is stated where the onset claim is made. The frozen-prompt attribution now rests on "none of the 200 plain answers does [loop or break]" on eager. Matches `stage0.json`. |
| 10 | Folded | The identity-92 table and the 0.372 ratio are added and flagged as exploratory. Recomputed: b0 0/92 and 0/92, b50 3/92 and 9/92, rise 0.0978 over dose-1 0.2630 = 0.372. |
| 11 | Folded | "The rise's point estimate is larger (0.263 against 0.188); no interval for that difference was computed." |
| 12 | Folded | Limits: "Gate 8 shuffles one batch at each shape and found 0 of 384 changed both times." |
| 13 | Folded | The 199.6 row adds "the eager rel50 rise is small and positive, 0.286 against 0.284." Matches the gate values in `report199.json`. |

## 2. Numbers the folds added (recomputed)

| Quantity | README | Recomputed | Match |
| --- | --- | --- | --- |
| Same first 40 tokens / identical whole / differing | 158 / 5 / 195 | 158 / 5 / 195 | yes |
| First differing key: min, median, max; at or below 64 | 28, 66, 72; 45 | 28, 66, 72; 45 | yes |
| b5 same order / same set (unchanged path) | 133 / 149, min shared 4 | 133 / 149, min 4 | yes |
| b50 unchanged: same set / min shared | 102 / 48 | 102 / 48 | yes |
| b50 changed: same set / min shared | 2 / 24 | 2 / 24 | yes |
| b200 unchanged: same set / min shared | 8 / 195 | 8 / 195 | yes |
| Sets differ over all 200: b5, b50, b200 | 39, 96, 192 | 39, 96, 192 | yes |
| Mean tokens differing, over all 200: b50, b200 | 2.05, 10.23 | 2.05, 10.23 | yes (over the 96 that differ: 4.27, 10.66; see N2) |
| All three lists equal as sets | 3 | 3 | yes |
| Dose-1 early stops, SDPA b0/b50/b200 | 358/379/383 | 358/379/383 | yes |
| Dose-1 early stops, eager b0/b50/b200 | 168/104/117 | 168/104/117 | yes |
| Part 191 mid-phrase, SDPA | 119/271/268 | 119/271/268 | yes |
| Part 191 mid-phrase, eager | 1/12/24 | 1/12/24 | yes |
| Loops ending early, SDPA b0/b50/b200 | 83/165/192 (of 96/168/193) | 83/165/192 (of 96/168/193) | yes |
| Loops ending early, eager b0/b50/b200 | 2/2/0 (of 109/210/223) | 2/2/0 (of 109/210/223) | yes |
| Dose-0.25 b0 loops: SDPA / eager | 10 / 0 | 10 / 0 | yes; all 10 placed past key 64 in `part199/onset/loop-onset-170.jsonl` (before 0, after 10) |
| Identity-92 no-push loops, b0 / b50, SDPA and eager | 0/3 and 0/9 | 0/3 and 0/9 | yes |
| 170.1 ratio on 92, eager | 0.372 | 0.0978 / 0.2630 = 0.372 | yes |
| 170.1 registered (104), eager | 0.329 | 0.32902 (`score.json`) | yes |

## 3. New problems the folds introduced or left

**N1 (fix, minor).** "Where the path agreed, the order did not." This is true for b50 and b200 (0 of 158 ordered), but b5 kept its order for 133 of 158 unchanged paths. Suggest: "For b50 and b200 the order never matched on the unchanged paths; b5 matched in order for 133 of 158."

**N2 (fix, minor).** "the b50 set for 96 (by 2.05 of 50 tokens on average)". The 2.05 is the mean over all 200 prompts, including the 104 with equal sets. Over the 96 that differ it is 4.27. Suggest: "(2.05 of 50 tokens on average over all 200 prompts; 4.3 on the 96 that differ)". The same applies to the b200 figure (10.23 over 200; 10.66 over the 192 that differ).

**N3 (fix, medium).** "The two kernels round the same forward pass differently, and the tail of a 50-token ranking holds many near-equal probabilities." This is stated as a fact. No file in the run records probabilities: `lists.jsonl` holds top-5, the lists and an entropy value. The sentence should be hedged ("may", "a candidate explanation, not tested here").

**N4 (fix, minor).** The heading "Why did no Stage 1 list match (199.2)?" is loose. Individual lists did match (b5 in order for 133 of 158). What is true is that no prompt had all three lists match in order (0 of 200). Suggest: "Why did no prompt's Stage 1 lists match in order?"

**N5 (minor, check).** Start time. The README says "07:23 to 09:55 CT". `load.json` records `started: 2026-11-09 07:24:03`, and `launch.jsonl` records the launcher step at 07:23:20. Either is defensible. Say which one is meant, or write "07:24" to match the load record.

**N6 (minor).** "Two smaller readings go the same way." The phrase is vague. Say what the two readings show (the dose-0.25 b0 loops vanish on eager; the 17.7 to 22.4 conditional range is exceeded).

## 4. Consistency across README, front page and Part 170 note

The Part 170 numbers (0.174, 0.329, 0.25; 25.0% to 43.8%; 28.4% to 54.7%) are the same in all three places. The "Four of nine" and "one of ten" counts match the predictions table. The front-page Part 199 entry, the Part 170 note and the Part 199 README do not contradict one another on the frozen-prompt attribution or on the kernel's stop pattern. The README's last line still reads "Cold check: pending", which is correct until this round is folded.

Not verified: Part 199's Method claims (for example the 8-hour guard and the 1.5 answers-per-second pilot rule), the `score170` bootstrap itself, and the figure's pixels.

SUMMARY: All 13 round-1 folds are in and every added number recomputes. Verdict SHIP-WITH-FIXES: four wording fixes (N1, N2, N4, and N3's hedge) and two minor checks (N5, N6). No registered number or verdict changes.
FINDINGS: 13 of 13 folded (none missing); 0 numbers wrong; 6 fold-residual or new items (N1 to N6), of which N3 is the only one about the strength of a claim.
BLOCKERS: none. The brief's cold/ folder was used for the report as asked; the copy in the output directory is the grader's file.
