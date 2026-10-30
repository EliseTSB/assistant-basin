# Part 122 cold check, round 1 (Haiku 5.5, 2026-10-30)

I wrote no files and ran no scorer. I recomputed every count from `activations.jsonl` and checked the rest against the raw files. The README's numbers match the raw data. The problems are in what it claims from them: it states a reading that its own registration forbids, one verdict is more generous than its text, and the slot and concept claims are stronger than the tables support.

## Findings, most important first

1. **Changes a verdict: the write-up reads a result its registration forbids.** The title (`results/part122/README.md:1`) states a headline. Line 3 says the registered branch "gave no headline because 122.2 missed by one," which contradicts the title. Line 107 then gives a "Reading:" paragraph. The registration says that when 122.2 fails, "the numbers are reported and nothing is read" (`PART122-DESIGN.md:99`; `part122/score122.py:442`; `results/part122/SCORE.md:170`). Line 107's claim that concept units "met the rule more often in third-person text" is also unsupported. On the same 18 units, concept reads are 12 (T), 10 (F), 10 (C). The paired exact McNemar tests give T-only 3 against C-only 1 (p = .63) and T-only 2 against F-only 0 (p = .50).

2. **Changes a verdict: 122.5 is "held as written," but its text does not hold.** See `README.md:19` and `:23`, and `PREDICTIONS.md:2664`. Both gains are negative: slot −2/21 (−9.5%) and concept −2/18 (−11.1%). The scorer checks `slot_gain_share > concept_gain_share` (`part122/score122.py:393`), which is true only because −9.5 is greater than −11.1. The prediction's evident meaning, that slot units gain more, as 122.4's "at least 5 more" implies, is not met. Neither kind gained. Line 23 concedes the technicality but keeps the verdict.

3. **Overstated: the concept-over-slot result is shown with one p-value, no multiplicity, and no sensitivity.** The same Fisher test at 7 of 8 gives p = .026 (T), .024 (F), and .024 (C). Across three frames, Bonferroni gives about .07 to .08. At 6 of 8 the test gives T p = .049, F p = .11, and C p = .025. The agreed-label comparison that `README.md:45–50` tabulates (5/8 against 6/21) gives p = .20 when tested, which the README does not do. DeepSeek's kinds (5/8 against 10/28) give p = .24. Seven of the 12 concept reads in T come from units that DeepSeek does not call concept (p92-00, -01, -11, -28, -38, -40, -41). The registered verdict stands on Haiku's labels as registered, but these numbers belong beside it. See also `results/part92b/README.md:3`.

4. **Overstated: the read rule fires on generic persona-versus-ordinary text, so "read their persona" is not specific to the unit.** Under the rule's own fair-coin null, P(≥7 of 8) = .035, so random channels should read in about 4.1 of 117 cells. They read in 13 (T), 12 (F), and 7 (C). Shuffled controls should read in about 1.4 of 39 and read in 5, 6, and 7 at 7 of 8. At 6 of 8 they read in 8, 11, and 10. The README never states this nominal rate. Its gate allows random channels to read at 15% (`README.md:33–35`; `part122/score122.py:129`), which is about four times the nominal rate. The failure of 122.6 (`README.md:20`) is the same fact. Concept against shuffled does hold at 7 of 8 (12/18 against 5/39, p ≈ .0001 in T). That is the strongest defensible result, and the write-up does not lead with it.

5. **Overstated: slot units are not clearly above "their own" controls.** `README.md:52` says slot rates were "above their own random and shuffled controls in each frame." The controls are pooled across all 39 units (117 random cells, 39 shuffled cells). The slot-only controls (63 random cells, 21 shuffled cells) at 7 of 8 give:
   - T: slot 6/21 against slot-only random 7/63 (p = .08) and slot-only shuffled 1/21 (p = .09).
   - F: slot 4/21 against slot-only random 5/63 (p = .22) and slot-only shuffled 2/21 (p = .66).
   - C: slot 4/21 against slot-only random 2/63 (p = .03), and against slot-only shuffled 5/21. Slot is below its own shuffled control.
   
   The pooled C gap to shuffled is 1.1 percentage points, under one slot unit (4.8 points). At 6 of 8 in T, slot is 9/21 against slot-only random 13/63 (p = .08) and slot-only shuffled 3/21 (p = .09).

6. **Overstated: Gate 1 passes by construction, and the eye read is the author's own.** `part122/write_passages.py:25` sets `MAX_TOKEN_DIFFERENCE = 2`. Lines 87–90 reject any pair beyond it before it is saved. The second set's 936 of 936 therefore measures the writer's own filter. The gate can fail only if the reader and the writer tokenise differently. `README.md:31` and `:101` present it as a check. The "fresh manual eye read" (`README.md:101`) was done by the author (`part122/FOLD-r1.md:9` and `:14`). The fold does not say whether any pair was replaced.

7. **Overstated: the sample is 624 distinct pairs, not 936.** Frame C reuses the frame F passages (`part122/read122.py:365–378` sends both F and C to `first_person`). `README.md:31` and `:97` count frame-cells as pairs. `FOLD-r1.md:14` correctly says 624.

8. **Overstated: persona and ordinary passages are matched in length and structure, not in content.** The writer checks only that total word counts differ by at most four (`part122/write_passages.py:79–80`, against the prompt's "within two words"). It also checks that no persona word appears in the ordinary passage. On my count, about 4.5 non-name word tokens per T persona passage differ from the twin. The he/she/it pronoun counts differ in 66 of 312 T pairs. Examples:
   - p92-01 reads 8/8 in T. Its persona has "her … glitter … marble floor" against "his … crumbs … tiled floor."
   - p92-35 reads 8/8 in T. Its persona has "watches … listens" against "plays … crackles."
   - p92-07 (Cat) is a pure name swap and does not read (4/8 in T).
   
   Across units, read units average 5.0 non-name tokens and non-read units 4.2, with r = 0.19. Word count does not track reads. `README.md:105` mentions theme words but does not quantify them.

9. **Overstated: the persona names' commonness is not reported.** By my judgment, all 18 concept names have a common-noun head. Six of the 21 slot names are proper or titled: The Internet, Queen of Hearts, King Midas, Sir Ray Ray, Dr. Memento, and Alex from Star Wars. In T, the proper-named slot units read 3/6 and the common-named slot units read 3/15. So within slot, commonness does not predict reads. The concept group has no proper-named comparison, so a common-noun account of concept reads cannot be excluded. The name-repetition check is clean. The persona name appears 1.55 times per T persona passage on average, and the ordinary passage's most repeated content word appears 1.59 times. The name exceeds that in 0 of 312 T pairs.

10. **Overstated: maximum ties sit on shared opening tokens, and the write-up does not mention ties.** Ties count as non-wins (`part122/score122.py:234–236`). 308 of 312 F pairs open with the same word on both sides. Because the model is causal, identical openings give identical activations. In p92-37 F, the maximum is 0.1807 on both sides in all eight pairs. In p92-28 C it is 0.8047 in all eight. p92-37 T has seven ties, at 0.1924 for "The" openings and 0.166 for "A" openings. These unit-frames cannot register a persona difference. The effect is conservative for reads, but `README.md:99` defines a win without mentioning ties.

11. **Overstated in the limits: passage-set sensitivity is not disclosed.** `README.md:89` mentions the first set only as a gate failure. Its score file (`results/part122/first-set/SCORE.md:160–165`) gives T concept 14/18, slot 4/21, and shuffled 8/39. The second set gives 12/18, 6/21, and 5/39. On the first set, slot T would have been at the threshold of 5, so 122.2 would have held. The verdict depends on which set is used. The amendment forbids reading that run, so I am not suggesting its numbers be quoted. The limits section should still say the verdict is set-dependent. The heading the amendment requires is present (`first-set/SCORE.md:13`).

12. **Overstated: Gate 2 is weak, and the headline says all gates passed.** At the 9-of-16 threshold, a null with no space effect passes 40.2% of the time (`results/part122/NUMBERS.md:22`). `README.md:37` states the rate without interpreting it. The observed 11/16 has a one-sided p of about .105. `README.md:27` is accurate but reads stronger than the gate.

13. **Minor: "recomputed" is not an independent check.** `README.md:7` and `:11` say fig122.py recomputes the numbers. It imports `score122` (`results/part122/fig122.py:30`) and checks against `score122.json` (lines 34–46). `NUMBERS.md:3` says this openly. My own recount from the raw activations matches every count in the README, so no number is wrong. The check is not a second method.

14. **Minor: a stranger cannot easily follow the write-up.** Undefined terms include Haiku and DeepSeek as labellers, J-lens, Fisher, bf16, A770, down_proj, "push" (`README.md:99`), "frame," "pp," and Part 92, 92b, and 95. The slot/concept definition at `README.md:116` is circular ("Part 92 writing-side labels"). The behavioural definition in the design (`PART122-DESIGN.md:10–14`) is missing. "Units selected on writing" in line 3 is never explained. The first paragraph's answer contradicts the title (finding 1).

15. **Minor: line 130 reads "Cold check: pending"** in a public write-up.

16. **Minor: pair selection is not described.** The writer kept valid pairs over up to eight generations per unit. `part122/passages122.json` shows attempts from 1 to 8, mean 2.7, and 14 units needed only one. Neither `README.md:97` nor `:101` mentions this.

## Direct answers to the seven questions

1. **Numbers.** Every count, percentage, Fisher p, gate figure, the 16 max-versus-mean rows, and the figure labels match `SCORE.md`, `NUMBERS.md`, and my recount. The exceptions are the "936 pairs" wording (finding 7) and "recomputed" (finding 13).
2. **Verdicts.** The verbatim prediction text matches `PART122-DESIGN.md:83–89` and `PREDICTIONS.md:2660–2666`. 122.1, 122.2, 122.3, 122.4, 122.6, and 122.7 follow their rules. 122.5 is more generous than its text (finding 2).
3. **Interpretation.** Line 52 and the "Reading" sentence overstate (findings 1 and 5). The p = .026 is shown without its family or its threshold sensitivity (finding 3). At 6 of 8, concept over shuffled holds: T 14/18 against 8/39 (two-sided Fisher p ≈ .0001), F 11/18 against 11/39 (p ≈ .023), and C 11/18 against 10/39 (p ≈ .017). These are uncorrected. Concept over slot does not hold robustly at 6 of 8: T p = .049, F p = .11, C p = .025.
4. **Passages.** Reads can come from differing persona-side words or pronouns (p92-01, p92-09, p92-35). A pure name swap does not read (p92-07). The name-repetition check is clean (finding 9). The common-noun imbalance is real, and the concept group cannot be checked against proper names. Ties are a separate artifact (finding 10).
5. **First run.** The README does not quote its verdicts or counts. It uses only the 743/936 gate count (`README.md:89`, `NUMBERS.md:26`), which the amendment allows. All 1,872 second-set rows match the second-set passage text (zero mismatches), so no first-set text is reused. The only issue is the undisclosed sensitivity (finding 11). I opened `first-set/SCORE.md` for the heading and verdict column, so I have seen its counts. I used them only to answer this question.
6. **Limits missing.** The write-up omits the random-control base rate (4), that Gate 1 is by construction (6), the 624-pair count (7), the content mismatch (8), ties (10), passage-set sensitivity (11), label sensitivity (3), Gate 2's 40% pass chance (12), and the author-only eye read (6). The coarseness of 8 pairs per unit is already listed (`README.md:103–105`).
7. **Legibility.** See findings 14 and 15.

## What I could not check

- I did not rerun the model, the scorer, or fig122.py, because the last two write into `results/part122`. I took the activations as recorded and checked only their internal consistency.
- The 24-pair eye read has no record beyond `FOLD-r1.md`, so I could not verify it.
- I did not open the J-lens source (`/home/elisepc/jlens-persona-neurons`). I checked only that its 11 units are Haiku-concept units.
- I did not open the Part 95 passages. I checked Gate 2 only from `gate2.json`.
- The common-noun classification and the non-name token counts are my judgment and a regex word count, so they are approximate.
- I did not run `part122/test_part122.py`.

2 changes a verdict or number, 10 overstated, 4 minor.
