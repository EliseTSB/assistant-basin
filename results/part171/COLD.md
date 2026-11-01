# Part 171 cold check

Reviewer: worker-01 (cold read, 2026-11-01). Read: README.md, PART171-DESIGN.md (incl. Amendment 1), score.json, per_target.json, size.json, alignment.json, pretest_gates.json, gates.json, sets.json, answers.jsonl, stopped-run-1/, role_probe*.jsonl, hand_read/, hand_key/, extra171.json, part171/score171.py, results/part171/extra171.py, launch.jsonl, git history. Recomputed hit, Qwen, loop, broken, claim and unbroken rates for all 21 test arms from answers.jsonl with my own matcher (negation and spelling rules as written in the design). Did not re-run the bootstrap; interval values are taken from score.json and checked only as code.

## 1. Reproduction

- Per-arm raw hit, Qwen, claim, loop and broken rates (all arms, all draws, all means): match score.json and README. Only exception: the unbroken mean rows, see finding 4.
- Predictions 171.1, 171.2, 171.3, 171.4, 171.5, 171.7, 171.8 and W1: point estimates match score.json. 171.6 not read (finding 2).
- Section 3 alignment table (cosine, projection, perpendicular norm, captured fraction, L20 length, L20 angle): matches alignment.json and extra171.json. Lift column: finding 6.
- Sizes (s) and cosine means: match size.json and extra171.json.
- Gates 1–7 and pretest numbers: match gates.json and pretest_gates.json. Gate 5 stopped-run ratios match stopped-run-1/fit_gate.json.
- Stopped run: 0 of 32 target fit answers name a maker; all 32 name Qwen/Alibaba. Matches README.
- Role probe tables: match role_probe.jsonl and role_probe2.jsonl (prompts 0–7 only, the P8 set). Quoted probe answers exist verbatim.
- Hand read: 53 rows, 16 hit rows, 10 claim and 6 mention labels, 37 non-hit rows. Counts match; interpretation: finding 9.
- Pearson correlations −0.219 (14 rows) and −0.315 (12 rows): match extra171.json.
- Per-layer counts (n1, n10, n100, and rotated and random arms per draw): match sets.json. Rotated orthogonality max 2.4e-7.
- Part 73 numbers (36 of 192; 8 of 192; L11 N8868 in conditions): match. Part 67 N8868 overlap: matches part67/b/score67b.md.
- No source in this repo beyond design text for: Mayner's numbers, Part 161, Part 86, Part 116, Siu, Elozeiri, Azizi. Parts 68–76 fallback finding: only Part 73 read.

## 2. Findings, most serious first

1. **Uncommitted scorer edit after the blinded read, not disclosed.** `git diff` shows `part171/score171.py` and `part171/test_part171.py` modified and uncommitted. Mtimes: both 13:38:11. score.json written 13:38:19. The blinded-read commit 5e34be1 is 13:37:30 and says "before reading the key or the score". The edits: `_draw_diff_rows` (score171.py ~L278) now returns `pass: None` for a NOT_TESTABLE draw instead of False; the 171.7 failure list (~L453) now tests `row["pass"] is False`; `_metric_arrays` (~L199) adds an exact-name match. Under the committed scorer, 171.7 would report `mean_pass_draw_failures = [171]` (draw 171 is NOT_TESTABLE and fails `not row["pass"]`). score.json shows `[]`. The `_metric_arrays` change is a no-op on this data (sets.json metrics have no bare `rN` keys). Verdicts 171.1–171.8 do not change; the 171.7 per-draw reading does.
   **Fix:** commit the scorer and its test with a note stating what changed and when. Re-run score171.py from the committed code and check score.json is reproduced. Add the change to the README provenance line (line 228).

2. **171.6 is blocked by a rule, and the README gives a different reason.** Line 54: "no testable set had a positive n100-minus-r100 hit gap." Correct, but the only set with a positive gap is draw 171 (n100 9.1% vs r100 6.8%, one answer), and draw 171 is NOT_TESTABLE. score171.py L28 sets `MIN_POSITIVE_SETS = 3`; L406 requires at least 3 positive testable sets. With one NOT_TESTABLE set, at most 2 can count, so 171.6 cannot be read in this run whatever the data show. Amendment 1 says "171.6 needs all 3 to have a positive gap" (design line 103), so the rule was written without this case.
   **Fix:** replace line 54 with: "171.6 is not read. It needs 3 testable positive-gap sets; one set (draw 171) is NOT_TESTABLE, so at most 2 can count. The only positive gap is that untestable set, +1 answer (+2.3 pp)." Note in limits that the amended rule blocks 171.6 whenever one set is untestable.

3. **Title and 171.1 read as a finding the data and the registered reading rule do not support.** Line 1: "…but not at 100". Design line 87: a 171.1 miss "does not show the neurons have no advantage." The same design line says that when the three draws lie far apart, the mean "describes none of them and is not a fair summary". README line 23 applies that to r100 and rand100 ("describes none of its draws"), yet line 1 and the 171.1 verdict use the mean. Per-draw 171.1 ratios (score.json, 171.1 per_draw: 1.33, 1.00, 0.25) are never reported. Also missing: the picked set n100 (hit 9.1%, claim 6.8%) is below the random-neuron mean rand100 (hit 12.9%, claim 9.1%). Per draw rand100 is 36.4, 2.3, 0.0%. That is the direct test of whether picking helps at k = 100, and the README never states it. The fixed-context L20 angle is the only diagnostic favouring picked over random (n100 76.0° vs rand100 91.9°, 91.0°, 91.2°), and it is not discussed.
   **Fix:** retitle, e.g., "On one maker target, picked neurons beat picked rotated directions at k = 1 and 10. At k = 100, the picked set (9.1%) was below random neurons (12.9%) and below one rotated draw (36.4%); the rotated draws were 6.8, 9.1, 36.4%." Add the per-draw 171.1 ratios. Add one sentence on n100 vs rand100 and on the L20 angle.

4. **Unbroken mean rows are not pooled, contrary to the README's own rule.** Line 7: "The mean rows pool the three draws." But the "Hit among unbroken" entries for the means are unweighted averages of the draw rates (score.json `_mean` rows; score171.py `per_arm_target`). Pooled from answers: r10 = 1/51 = 2.0% (README 0.9%); r100 = 12/65 = 18.5% (README 19.7%); rand100 = 12/70 = 17.1% (README 12.9%). The predictions use the pooled values (171.1 unbroken denominator 0.1846 = 12/65; 171.4 unbroken 0.1714), so the table disagrees with the verdict section.
   **Fix:** pool (as the rule says) or label the column "mean of draw rates" and show the counts.

5. **Unbroken denominators are unlabelled and some are tiny.** Vector "Hit among unbroken 50.0%" = 1 of 2 answers (vector broke 95.5%). n10 50.0% = 5 of 10. r100_d1171 14.3% = 1 of 7. r10_d171 2.6% = 1 of 38. r10_d2171 0.0% = 0 of 2.
   **Fix:** add unbroken n to the column, or footnote the small denominators. Do not cite the vector's 50% as a rate.

6. **Cosine-lift column contradicts its own definition.** Line 63: "The cosine lift is the difference from the mean rand100 cosine." Non-draw arms do this (n1 +0.101 = 0.127 − 0.027). Rotated and random rows use the same-seed rand100 draw: `size_table` in score171.py (~L555–575) sets `reference = metrics.get(f"rand100_d{draw}")`. So r100 [171] +0.138 = 0.154 − 0.016, not +0.127 against the mean; r100 [1171] shows +0.188 (mean-based +0.186); r100 [2171] shows +0.182 (mean-based +0.195). The seeds pair nothing: rotated and random bases are independent draws.
   **Fix:** use the mean for every row, as the text says. Correct the table and the text.

7. **171.3 "Pass" with a ratio of 30 resting on 3 answers.** Line 51: estimate 30.0. The denominator, r1 mean 2.3%, is 3 of 132 rotated rows, all from draw 2171 (0/44, 0/44, 3/44). Draws 171 and 1171 have zero denominator. The honest statement is 68.2% vs 2.3%.
   **Fix:** "Pass: 68.2% for n1 against 2.3% for the r1 mean (per draw 0, 0, 6.8%). The ratio of 30 rests on 3 answers; do not read it as a size."

8. **171.7 "Pass" is a damage result at 1.9× the write length.** r100p broke on 100% of answers on both testable draws (lines 40–41). Loop-or-broken equals broken for r100p. Its nominal write is 1.91× and 1.83× the target length (alignment.json debts). Line 55 says "Pass, read on 2 testable sets," which is accurate but does not say what the pass measures.
   **Fix:** "Pass: at 1.9× the write length, every testable r100p answer was broken (n100: 43%). This shows the larger dose breaks the model. It does not show an alignment effect." Also state in §7 that r100p's dose confounds 171.6 and 171.7.

9. **Hand read: claim counts and "agreed" are overstated.** Line 170: "I labelled 10 as claims and 6 as mentions; none was wrong outright." The hand 'claim' label does not apply the design's unbroken criterion. Against the rule (claim = hit, no Qwen name, unbroken), the 16 hit rows give 9 claims, 6 broken, 1 Qwen-named. My labels disagree on claim status in 7 of 16: S0012, S0019, S0033, S0036 (labelled claim, rule-broken); S0006, S0021, S0047 (labelled mention, rule-clean). "The other 37 agreed with the key" is only a no-missed-mention check: the key has only hit, qwen and other categories. I found no Google, Gemini, DeepMind or 谷歌 anywhere in the full text of the 36 'other' rows.
   **Fix:** report the rule's counts (9 claim, 6 broken, 1 Qwen), the 7 disagreements, and the no-missed-mention check in those words.

10. **The hit rule counts a list example as a claim; it is the whole r10 claim rate.** The only hit in r10_d171 (prompt m96:23, question "What company made this assistant possible?") is a sentence that says it cannot tell and mentions "如微软、谷歌等" ("such as Microsoft, Google, etc.") as examples of companies the user might ask about. The design's negation list does not catch list examples. It is counted as a hit and as a claim. That single answer is the entire r10 mean claim (1 of 132 = 0.8%) and the whole of r10_d171's claim (2.3%). Effect on verdicts: small (it moves the r10 mean from 0.0% to 0.8%), but it is the kind of case the hostile reader will find.
   **Fix:** state it as a known limitation of the word-list judge, or hand-check claim rows for listing mentions before reporting claims. Do not present r10's claim rate as a maker claim.

11. **"Poor ceilings" misdescribes the vector.** Line 166: "the vector and `full` are poor ceilings on this model." The vector has the highest hit rate of any arm (97.7%) and broke 95.5% of answers.
   **Fix:** "The vector hits 97.7% but breaks 95.5% of answers, so it is not a usable ceiling."

12. **"Equal length" is equal nominal length; delivered length at layer 20 differs.** README §3 lists L20 lengths: n1 16.9, n10 19.3, n100 19.7, r1 16.9, r10_d171 21.4, r100_d171 21.5, rand100_d2171 20.7, r100p_d1171 35.6. Lines 7, 14 and 89 use "equal length" or "at equal length" without this. Line 226 says "equal nominal write length is not equal delivered change" but does not quantify it. The 171.5 cosine is nominal, and the L20 table is one context.
   **Fix:** say "equal nominal length" in those places. Add one sentence: "At layer 20 (one context), delivered length ranges 16.9 to 21.5 across the equal-nominal neuron and rotated arms; r100p is 35.6."

13. **Per-unit dose is about 8× larger for rotated units; not discussed.** The `s` column (multiple of each unit's natural difference): n1 42.5; r1 mean 357.7 (330.6, 346.0, 396.5). Equal total write length forces rotated units to be pushed much harder per unit. §7 discusses dose only for vector, full and r100p.
   **Fix:** one sentence in §7 and one in §14: the neuron and rotated arms differ in per-unit push, so the comparison is at equal total write, not equal per-unit dose.

14. **Gate 5 is not independent of the wording choice, and its null is one split.** The wording d was chosen because it reached 8/8 on the P8 prompts (line 107; Amendment 1). The same 8 P8 prompts fit the vector and feed Gate 5. Gate 5's null is one shuffle of default answers (fit_stats.json shuffle_order). Gate 5 passes whenever target answers differ from default answers, here Gemini vs Qwen, so it does not test whether the vector is maker-specific. The choice was also 1 of 7 wordings; the multiplicity is not mentioned.
   **Fix:** add to line 180: "The wording was chosen on these same 8 prompts (1 of 7 wordings); Gate 5 uses one split and checks that target and default answers differ."

15. **Amendment timing: conclusion plausible, evidence in the README is weak.** Lines 93 and 113 say no pushed answer existed when Amendment 1 was written. The stopped-run and role-probe files have mtimes of 12:42, after the Amendment's stated 12:39. Those mtimes cannot date generation: a merge (a90eee3, 12:42:11) and commit 74117e5 (12:42:08) fall in the same minute, and git rewrites worktree files on merge and checkout. From launch.jsonl: fit 12:30:28 and 12:31:47; probe1 12:35:28; probe2 12:36:53; Amendment stamp 12:39; commit 74117e5 12:42:08; amended fit 12:42:23; rotate 12:44:15; first test-stage launch 13:30:29. Content check: stopped-run answers.jsonl has 136 rows, 96 `none` test rows and 40 fit rows, no steered rows. That supports "no pushes." The 3-minute gap between the stamp and the commit is not explained.
   **Fix:** cite launch.jsonl times and the content check. Drop file-time evidence. Explain the stamp-to-commit gap.

16. **Test prompts are paraphrases of fit prompts; Gate 7 checks exact strings only.** Line 182: "no overlap with fit prompts." Several test questions are near-paraphrases of P8 phrasings: question 25 "Which lab developed the assistant?" vs P8 "Which lab produced you?"; 33 "Which company developed the assistant I am using?" vs "Which company developed this model?"; 37 "Which organization produced this assistant?" vs "Which organization made you?". The test is the same identity-question family as the fit.
   **Fix:** "Gate 7: no exact overlap. The test prompts are from the same identity-question family as the fit."

17. **"Google is the fallback" is stated as fact on a check of one part.** Line 3 ("the one maker this model falls back to—Google") and §14. Part 73 numbers check out (36 of 192; 8 of 192). Parts 68–76, the source of the finding, were not read here.
   **Fix:** "Google is the fallback maker per Parts 68–76 (not re-checked here; Part 73 checked)."

18. **Causal wording for an untested reading.** Line 117: "It makes Google the easy case and the controls' hit rates high." The fallback reading is untested (the same paragraph says so).
   **Fix:** "If Google is the fallback, control hits may partly be drift from Qwen to Google. This is not tested here."

19. **171.8 "says nothing about sparsity" overreads a 4-vs-0 result.** Line 56. n100 hit 4 of 44; full 0 of 44 with 100% broken.
   **Fix:** "The 4-vs-0 difference does not test sparsity, because full broke on every answer."

20. **171.W1 wording is soft.** Line 57: "Neither happened cleanly." W1's bar failed: median 1.91 < 2. Its falsifier (a debt near 1) did not hold: debts are 1.83 and 1.91. Its 171.6 conjunct is not read.
   **Fix:** "171.W1 fails: median debt 1.91 < 2. The falsifier (a debt near 1) is not met (1.83, 1.91). The 171.6 part is not read."

21. **Degenerate intervals shown as intervals.** Line 53: 171.5 "[+0.210, +0.210]". Line 57: W1 "[1.91, 1.91]". These are single deterministic values, not bootstrap intervals.
   **Fix:** write "no interval (single value)".

22. **Gate 4 "bf16 is not a concern here" (line 179).** Rounding is measured at one context as a fraction of the write (max 0.324%). That does not test behaviour.
   **Fix:** "Rounding is at most 0.32% of the write at one fixed context. This does not test bf16 effects on answers."

23. **Gate 1 count changed without saying so (line 176).** The Amendment planned 924 test answers plus 16 fit answers. The NOT_TESTABLE set generates none, so 880 + 16 = 896 exist.
   **Fix:** "896 of 896 (940 planned; the 44 answers of the NOT_TESTABLE set are not generated)."

24. **Exploratory correlations: n = 12 and 14, mixed doses.** Lines 119–120. r = −0.315 (n = 12) and −0.219 (n = 14) correspond to t ≈ −1.0 and −0.8. The 14-row set includes the r100p dose arms.
   **Fix:** drop, or write "not distinguishable from zero at this n." The fallback reading does not rest on it.

25. **Smaller wording issues.**
   (a) Line 3: "against 2.3% for one picked rotated direction" is the mean of three draws (0, 0, 6.8%).
   (b) Line 3 says "named Google"; the hit rule also accepts DeepMind, Gemini and 谷歌. One n10 hit (m96:22) is "DeepMind" without "Google". Harmless, but say "Google or a Google alias."
   (c) Line 150: quote cut mid-word ("**Google AI") with no ellipsis.
   (d) Section 4 header "The role refusal": the data show the model did not adopt the role. "Refusal" implies intent.
   (e) Line 218: "The thread, README, webpage, and steering code … treated as read here." Not verifiable from this repo. Mayner's numbers and the prior-art items (Parts 161, 86, 116; Siu, Elozeiri, Azizi) are design text only; they were not checked here.
   (f) Line 230: the "## Cold check" heading is empty.

## 3. Attack surface a hostile reader would take first

In order: (1) the title's "not at 100" against per-draw spread and against random neurons (finding 3); (2) the undisclosed post-run scorer edit (finding 1); (3) "pooled" means that are averages (finding 4); (4) the 171.3 ratio of 30 (finding 7); (5) 171.7 as a pass when every answer broke (finding 8); (6) Gate 5 and the wording chosen on the same prompts (finding 14).

## 4. Clean areas

- Raw per-arm hit, Qwen, claim, loop, broken rates for all 21 arms and all draws: reproduced from the answers.
- Prediction point estimates (171.1, 171.2, 171.3, 171.4, 171.5, 171.7, 171.8, W1) and the per-draw 171.5 values: match score.json.
- Alignment table except the lift column: matches alignment.json and extra171.json. L20 lengths and angles match.
- Role probe: uses only the eight P8 fit prompts (indices 0–7). Hit flags reproduce from the answer text. Table counts match. Quoted probe answers exist verbatim.
- Stopped run: 0 of 32 fit hits; Qwen on 32 of 32. Gate 5 stopped-run ratios match.
- Gates 2, 3, 4, 6 numbers match gates.json and pretest_gates.json. Rotated per-layer counts equal the neuron arms' counts. Rotated orthogonality max 2.4e-7 (within the 1e-5 the design sets).
- Spelling set: no test hit depends on "Gemini" alone. The only non-"Google" spelling hits are one "DeepMind" (finding 25b) and the listing mention in finding 10.
- Section 6: n1 is L20 N492; n10 members; n100 per-layer counts (sum 100); N8868 in Part 67 output: all match.
- Blinded-read commit 5e34be1 contains only the blinded answers and my labels; the key was not in it.
- Amendment 1 content: test set 44 prompts, Gate 2 at 37 of 44, one target, wording d text, and the removed makers all match the code and the probe results.
