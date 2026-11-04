# Findings 1 and 7, rewritten: what changed and where each number comes from

Applied to `README.md` on branch `claude/findings-rewrite-parts-index` on 2026-11-04, after merging main at b0a1627 (the
completed reader correction, merge 4d00dae; the eager kernel default, caf592e; the known-issue note, 8c0034b). Written
by Claude, a Claude Code cloud session; no model ran. The first version of this file (2026-11-03) was a draft built on
the partial reader join; this version replaces it. Every number in the two findings was read from the file and line
named here, on the merged tree, not copied from a message.

## Short version

- Line 1 rewritten: adds the Part 176 clause and the known-issue sentence. Sources: 42 of 192, 1 of 192,
  `README.md` Part 66 section (line 1196 before this edit); 30 of 40 and median one broken, `results/part176/README.md`
  line 1 and `PREDICTIONS.md` row 176.1 (line 3322) and 176.5; the list membership of Parts 25 and 66, `README.md`
  known-issue note (lines 9 to 18).
- Line 7 rewritten: adds "both lower bounds", the Part 97 match and the Part 117b window. Sources:
  `results/part86/README.md` line 28; `results/part97/README.md` line 11; `results/part117b/README.md` line 82.

## Finding 1 (replaces the old Finding 1 and "What the four units hold back")

1. Title: "Four selected neurons hold Qwen3.5-4B's own maker answer, and the fallback they expose is the model's own
   second answer, reachable by many pushes." No number.
2. "The four neurons" paragraph: Part 25 one-minimal set and 0 of 20 random groups, `results/part25/search-google-removal.json`
   and `controls.json` (as in the old text); Part 66's 42 of 192, 1 of 192, 50 groups at most 2 of 24, positive control
   8 of 24 and 48 of 96, `README.md` Part 66 section (line 1196); Part 32's 16, 17 and 0 of 24, `PREDICTIONS.md` rows
   32.3 and 32.4 (lines 1376 and 1377).
3. Kernel line (new): the high-risk list and the lower-risk list, `README.md` lines 9 to 18; which parts are single
   forward passes, the same note's last sentence.
4. "What the fallback is": "Sam Altman" and "USA", `PREDICTIONS.md` line 1118; Base's OpenAI 5 and Alibaba 3 of 8,
   zeroed 6 and 2, 20 controls, `cloud/vast-part27-2026-10-20/base.json` (as in the old text); 38 to 4 of 192, 12.5% to
   37.5%, 92%, `README.md` Part 70 section (line 1362); Google first of 248,320 and Apple/Microsoft with Google at 36,
   Part 71 section (line 1387); 7% to 22% and 23 of 24, Part 72 section (line 1421).
5. "The writer route is dose-scoped": 8 and 17 of 192 against 36, Part 73 section (line 1456); 64%, 29%, 7%, Part 74
   section (line 1495); 15 matched units, Part 74b section (line 1539); -1.42 and -0.08, Part 76 section (line 1625);
   12 of 21 and p = .065, Part 77 section (line 1678); 96 to 46 of 168, 93 to 102, 53 and 52 of 72,
   `results/part154/README.md` (headline); 18 against 17 at d = 1 and 18 to 6 at 0.75, `PREDICTIONS.md` row 172.6
   (line 3260).
6. "The fallback needs no selected neuron": only Google adoptable, 68.2% against 2.3%, 47.7% against 5.3%, 9.1%
   against 12.9% and 17.4%, `results/part171/README.md` lines 1 to 3; 22, 26 and 36 of 44, `README.md` Part 173 summary
   and `PREDICTIONS.md` row 173.4 region (line 3277 for 173.5, Tencent 27 of 44); Meta 35 of 44,
   `results/part173/README.md` line 57; 28 of 80, 0 other-led, 7.2%, median 5 of 44, `results/part174/README.md` line 1
   and `PREDICTIONS.md` rows 174.1, 174.3 and 174.7 (lines 3290, 3292, 3296); 29.5 of 44 and 1 of 40,
   `results/part175/README.md` line 3 and row 175.5 (line 3312); 30 of 40, the support ladder, 22 of 34 within 0.025,
   `results/part176/README.md` line 1 and rows 176.1, 176.W1, 176.W3 (lines 3322, 3329, 3331); 63% against 11%,
   `results/part153/README.md` (headline); 33 of 47 against 25 of 37, `results/part161/README.md` line 3.
7. "Where a country feature may sit": 8 and 7 China-family tokens, 0.7% and 1.0%, 11 Google-family, no AI-family
   token for L6 N753, `results/part111/README.md` (headline and tables); 0.00272 against 0.00282,
   `results/part110/README.md` line 42. The reading is marked as a reading.
8. "Content specificity against privilege" (changed from the draft): 146 against 96 and 191 against 197,
   `results/part148/README.md` lines 22 and 20; the new subset counts 81 against 48 and 107 against 112,
   `results/reader-correction-2026-11-03/part148/v2_another_ai/score.json`, keys
   `exploratory.neurons.coherent_name_and_maker`, `exploratory.random_directions.coherent_name_and_maker` and
   `exploratory.*.coherent_by_question[1]`; 3.5% against 8.6%, 21.4% against 5.2%, `results/part163/README.md` line 3;
   6.4% against 8.1%, 27.0% against 7.8%, `results/part164/README.md` lines 1 to 3 and the row at line 15; 26%, 27%,
   15%, about 9%, `results/part165/README.md` line 3.
9. "The same pattern in other models": 94% to 68%, 6% to 18%, 94% to 76%, 20%, `PREDICTIONS.md` rows 16b.2 and 16b.4
   (lines 586 and 588); 16 of 23, `results/part9-think/name-counts.json` line 114, entry `Think|raw_nosys`; 109 of
   210,000 and about 1,180, `results/part15-data-counts/NOTES.md` line 19; 0 of 378,341, line 16; 4.2% against 44.4%
   and .037 against -.003, `results/ablate-smol-nameattr.json` (as in the old text).
10. Closing paragraph: no number; adds that the fallback is a property of the model's priors.

## Finding 7 (replaces the old Finding 7)

1. Title: "Single-neuron pushes give Qwen a coherent persona at rates near same-size random directions; what is
   neuron-specific is which persona appears, inside a narrow dose window, and the persona is not stored." No number.
2. Figure, caption and the three definitions: unchanged from the old text.
3. "The rate comparison": 250 of 9,216, 3 of 512, 2.713%, 0.586%, p = 0.000584, `results/part86/README.md` line 28
   and the layer table; 5.219%, line 18; 3.75, 2.7 to 6.4, 54 against 42, about 216, line 100. Part 92's rates,
   1,709 of 131,072 (1.30%), 45 of 4,096 (1.10%), 8.6% against 9.5% among changed answers, changed shares 15.2% against
   11.6%: computed by `review/part181/part92_screen_rates.py` (branch `claude/part181-review`) from
   `results/part92/screen-instruct/layer_*.jsonl` and `results/part92/labels-screen/` (key and labels); the counts are
   neurons 19,934 changed of 131,072 with 1,709 coherent personas, random 474 changed of 4,096 with 45. 101 of 343 and
   94 of 344, `results/part112/NUMBERS.md` lines 18 and 19.
4. Kernel line (new): the eight listed parts, `README.md` lines 9 to 18; Part 86's 48-token answers,
   `results/part86/screen-final/layer_22.meta.json` line 24.
5. "Reader dependence" (changed from the draft; every subset number is new): κ 0.814, 0.109, 0.051,
   `results/part86/README.md` line 44; 245, 141, 1 and 77 of 78, `results/part86/atlas-v2/ATLAS-v2.md` lines 6 and 8;
   21 slots and 8 of 18 concepts, κ 0.833, `results/part92b/README.md` line 3; Haiku 14 and DS 4 of 43,
   `results/part118/README.md` line 5; the complete join 465 of 465, 322 and 245,
   `results/reader-correction-2026-11-03/README.md` line 21 and `report.json` `universe_sizes`; 104 of 190 against
   154 of 329, line 47; 6.3% against 8.0% (0.0633 on 980 labels against 0.0796 on 1,860), line 49; 1.786, 0.226, 0.262
   and 2.105, 0.186, 0.440, lines 93 and 94; 148.5 held on the 245 units with 89 against 184, line 80; 148.6 failed
   everywhere, line 81; 117.2 failed on the 322 units with 7 of 10 against a bar of 8, lines 58 to 60 and
   `report.json` `parts.117.v2_all.predictions.117.2`; no confirmed pair on the 245 units, line 58 and line 129.
6. "What is neuron-specific: the content": 185 of 1,709, 28 of 1,709, 0 of 45, `results/part97/README.md` line 11
   and its shuffled row; 6 of 10 and 0 of 10, `results/part97/validation/score.json` line 4; 224 of 465 and 22 of 460,
   `PREDICTIONS.md` Part 98 table (row 98.J3, line 2317); 2.11, 0.19, 0.44, `results/part152/README.md` rows 152.3 and
   152.4 (line 26 and the row above it); 14 of 39 and median 4, `results/part124/README.md` line 1; 63% against 11%,
   `results/part153/README.md` (headline).
7. "The dose window": 11 of 11, 2 of 11, 0 of 11, `results/part117b/README.md` line 82; 11 of 11 to 1 of 11,
   `results/part121/README.md` line 3; 234, `results/part118/README.md` line 3; 22 of 34 within 0.025,
   `results/part176/README.md` line 1; 48.3%, 0%, 5.8%, 37.9%, 35.1%, 43.3%, `results/part140/README.md` lines 3, 39,
   30, 33, 27 and 36.
8. "Not a state": 2 of 29, 3 of 29, 11 of 29, `results/part125/README.md` line 5; the cache finding, `PREDICTIONS.md`
   row 31.5 (line 1353).
9. "What the push does instead": 0.99 and 0.49, `results/part145/README.md` (row 145.4 region, line 18); 0.77 to 0.82
   and 0.50 to 0.61, `results/part146/README.md` (table at line 32); 20.0%, 0%, 1.5%, 98.5%, `results/part139/README.md`
   lines 5 and 17; 0.121 to 0.822, `results/part149/README.md` line 20; 0.056 and 0.004, `results/part142/README.md`
   line 3.
10. Tally line: 18 held, 13 failed, 1 not scorable, `docs/tally_findings.py` run on the merged tree, 2026-11-04.

## Notes for the check

- The reader-correction sentence in Finding 7 is the one place the complete join changes the text against the
  2026-11-03 draft: 148.6 no longer flips, 148.5 flips on the 245 units, 117.2 flips on the 322 units, and Parts 117
  and 117b have no pair on the 245 units.
- The two kernel lines follow the front page's lists exactly. Part 86 is on neither list; the line says what its
  answer length was and nothing more.
- The reading in "Where a country feature may sit" is still marked as a reading; cut the paragraph after the Part
  110 sentence if readings should stay out of the findings.
- Part 98 has no results README; its counts come from the ledger table.
- Parts 25, 27, 66, 70 to 77 have no results README; their numbers come from their `README.md` sections at the
  line numbers above (as of the merge commit on this branch) and the result folders named there.
