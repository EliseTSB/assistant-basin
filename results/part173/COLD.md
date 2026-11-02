# Cold check: results/part173/README.md (Part 173)

Scope: repo the repository. Read PART173-DESIGN.md (including Reading rules and Limits), results/part173/README.md, score.json, prediction_table.json, family_table.json, cross_set_modal_table.json, cell_table.jsonl, readers.log, labels, scored_answers.jsonl, label173.py, run173.py, names173.py, and plot173.png. Recomputed from scored_answers.jsonl joined to labels-ds.jsonl with `-I -B`. Nothing in the repo was edited.

## Verdict

The README's numbers are, for the most part, faithful to its files. I recomputed the eligible-cell counts, the 138 cells, the 25 supported cells and their 23 stable ones, the 15 best cells (10 Google), the none/role/zero4 word and claim counts, the Gate 1 to 8 values, and the reader agreement figures. Each of them matches. All 12 registered predictions match the registered text in PREDICTIONS.md, and every verdict agrees with its bar in prediction_table.json. The figure matches cell_table.jsonl for all 138 cells. The two problems that matter are in the headline. The opening paragraph compares a count of sets with any supported replacement (7 of 12) against a count of Google replacements, and it says the 7 "did too". The Google-only count is 5 of 12. The README also leaves out a reading the design requires before the Google result can be interpreted: the cross-set and zero4 match. In that reading, 18 of 25 supported cells match zero4's modal entity, which is Google. Several smaller items need fixing: a process claim (43 token-limit failures) that no artifact supports, a 0.5% cap and retry that appear only in code and not in the registration, a Gate 4 push hash that comes from an in-memory fixture, an empty pooled chance simulation, and a handful of links whose target does not hold the number. Once these are fixed, the verdicts can stand as written.

## Findings

1. **[high] Opening paragraph: "[7 of 12](family_table.json) fit-built controls did too."**
   What is wrong: "did too" says the 7 produced a supported Google modal. The 7 is the count of fit-built sets with a supported replacement under any name (rotated 5 + fit-at-random-coordinates 2, per family_table.json). The Google-only count is 5 of 12: r100_d171, r10_d2171 and r100_d2171 among the rotated sets, and rand100_d171 and rand100_d2171 among the fit-at-random sets. This is two different counts put side by side as if they were the same thing.
   Fix: "3 of 6 random-valued, layer-profile-matched sets produced a supported Google modal; 5 of 12 fit-built controls did too, and 7 of 12 had a supported replacement under some other name."

2. **[high] Section "Do the random-valued controls lead, or only the fit-built sets?" (and Files / line 94 "A Google fallback name remains an open hypothesis").**
   What is wrong: the design's Cross-set reading requires, for each supported cell, whether its modal entity is also modal in zero4, and for each entity, how many sets have it modal at a supported cell. The README reports neither. From score.json and cross_set_modal_table.json, I get: zero4's modal entity is google in every cell (3 Google claims of 44 by the reader, matching the 3 Google word hits); modal_matches_zero4 is True for 18 of 25 supported cells and False for 7 (n1 at −0.5, n3 at +0.5, −0.5 and −1, r1_d171 at +1, r1_d1171 at −1, rand100_d171 at +0.75); Google is modal at a supported cell in 12 sets; Meta in 2; Tencent, Antibody and Q4 in 1 each. Without these, the README's claim that Google is "an open hypothesis" and that zero4 "does not establish why" is not checkable. The reader cannot see that one name is shared across most of the leads.
   Fix: add a paragraph after the family table: "Of the 25 supported cells, 18 have the same modal entity as zero4, which is Google on 3 of 44 zero4 claims, so that match rests on a small count. Google is modal at a supported cell in 12 of 23 sets; Meta in 2; Tencent, Antibody and Q4 in one each. The 7 cells that do not match zero4 are listed in cross_set_modal_table.json and cell_table.jsonl." Keep the open-hypothesis sentence, but state that the shared name is the pattern the design calls a fallback-name test, and that this run does not decide between a fallback and a real lead.

3. **[medium] Reading rules (4): placement of the layer-profile caveat and the "random-valued support" sentence.**
   What is wrong: the design says "173.8 and 173.W2 are read with that said first", meaning the n100 layer-profile point must come before those verdicts. In the README, the caveat appears only in the section after the prediction table (line 41 onward) and in Limits, so the 173.8 and 173.W2 rows come first with no caveat. The design also requires a sentence on what a random-valued support does and does not show: a supported replacement shows the behaviour is not exclusive to picked coordinates or to the Google fit's values, under these policies; none in the six would not show it is exclusive. The README has neither sentence in that form.
   Fix: before the registered-prediction table, add: "All six random-valued sets carry n100's layer profile, so 173.8 and 173.W2 test random values under that profile, not value-randomness alone. Three supported random-valued sets show the behaviour is not exclusive to picked coordinates or to the Google fit's values under these policies. That is not a rate for arbitrary pushes, and a count of 3 of 6 does not estimate one."

4. **[medium] Line 88: "[43](readers.log) ran out of the [4,000-token limit](readers.log)."**
   What is wrong: readers.log lists failed pair IDs and counts (47 in the first pass, 16 after the retry) but records no failure reason, and it contains no mention of a token limit. score.json does not contain the claim either. I searched run173.py, label173.py and score173.py for the same text and found none. The claim cannot be verified from the linked file or from anything in the run directory.
   Fix: delete the "43 ran out of the 4,000-token limit" clause, or cite the run log that records the reason and link to it. Otherwise say: "failure reasons were not recorded."

5. **[medium] Lines 88 and 84-88: "over the [0.5% error cap](score.json)", "The registered retry", "[16 of 4,483 pairs, or 0.36%](score.json)".**
   What is wrong: score.json contains no 0.005 cap value and no 0.36% figure. The cap is in run173.py's DESIGN dict (`reader_errors.maximum_fraction: 0.005`), and the retry is in label173.py. PART173-DESIGN.md states neither a cap nor a retry rule; its reader section says only that errors are reported per pair. So "registered retry" is not registered in the design document the README cites. The values are in readers.log (error_cap_fraction 0.005, error_fraction 0.003569, failures 16), not score.json.
   Fix: point the cap and the 16-of-4,483 figure to readers.log and run173.py (line 70). Either add the cap and the retry rule to PART173-DESIGN.md with their date, or say they were added to the code after registration and when. Do not call them registered unless they are.

6. **[medium] Gate 4 row: "Artifact hashes were recorded, not compared."**
   What is wrong: score.json gates.gate4.artifacts.pushes has source "in-memory test fixture" with sha256 22750d97…; the hook source path is a Windows path (C:\Users\alexa\neuron-probe-gpu\part171\run171.py); the model snapshot id is an E:\ path. So the recorded push hash does not identify Part 171's saved push artifact. The 44 of 44 token match is the only evidence that the replayed pushes were Part 171's. The README does not say this, and it reads as if the hashes tie the replay to Part 171's files.
   Fix: add: "The recorded push hash is from an in-memory test fixture, not a saved Part 171 file. The replay match, 44 of 44 for each set, is the only evidence that the pushes were Part 171's."

7. **[medium] The chance simulations are not mentioned, and one of them is empty.**
   What is wrong: PART173-DESIGN.md says score173.py reports two chance simulations, each with a probability and an expected count of the 138 cells. The README does not report either. In score.json, the none-frequency simulation gives probability 0.0 and expected cells 0.0. The random-valued pooled simulation has `entity_frequencies` equal to an empty dict, so its 0.0 comes from an empty pool and is not a chance result. The pool is probably built wrongly in score173.py.
   Fix: report the none simulation (0.0; 0.0 expected cells) in the README, and mark the pooled one as not computed because its pool is empty. Fix the pool construction in score173.py and rerun before quoting either figure.

8. **[low] Line 86: "DeepSeek V4.1 Flash read [4,483 unique clean answer pairs](score.json)."**
   What is wrong: 4,483 unique clean pairs is correct (recomputed from scored_answers.jsonl), but labels-ds.jsonl has 16 rows with no relation, the 16 unlabelled pairs that the README mentions on line 88. So 4,467 pairs were labelled and 16 were not. "Read" overstates this. The number 4,483 also does not appear in score.json.
   Fix: "DeepSeek V4.1 Flash was run on 4,483 unique clean answer pairs; 4,467 were labelled and 16 remain unlabelled ([readers.log](readers.log))."

9. **[low] Lines 7 and 94: "They broke many answers at larger doses; sparse pushes did not leave answers untouched."**
   What is wrong: this is not true of draw 1173. In score.json, u100s_d1173 breaks 44 of 44 answers at every dose, including +0.25 and −0.5, and the figure shows the same. Draw 173 is clean at +0.25 and has 1 broken at −0.5. Draw 2173 is clean at +0.25 and +0.5.
   Fix: "Draw 1173 broke all 44 answers at all six doses. Draws 173 and 2173 were clean at +0.25 and broke answers only at larger doses."

10. **[low] Line 86: "weighted replacement-primary agreement [96.6%](score.json) over [233 answer-weighted responses], with [225 matching responses]."**
   What is wrong: the reader gate in the design requires both the unweighted and the weighted agreement on pairs with a non-null primary from both readers, with at least 30 pairs. score.json reader_gate9.replacement_primary gives 223 of 231 comparable pairs matching unweighted (96.5%) and 225 of 233 weighted (96.6%). The README reports only the weighted figure and calls the counts "responses", which are pairs and answers. The gate passes either way, so no verdict changes.
   Fix: "96.5% of 231 comparable pairs agreed (223 matching) and 96.6% answer-weighted (225 of 233); both exceed the 80% bar." Use "pairs" and "answers" correctly.

11. **[low] Title and opening: "Three value-randomized sets lead with Google on these prompts."**
   What is wrong: "lead" suggests a dominant name. The three supported Google cells are pluralities of 22, 26 and 36 of 44 answers, with 0 broken, and the design itself says a modal count of 10 "is a plurality that can leave most answers elsewhere". The sentence in the opening paragraph ("produced a supported Google modal") is correct, so only the title overreaches.
   Fix: retitle: "Three value-randomized sets have a supported Google modal on these prompts."

12. **[low] Figure, plot173.png (cosmetic).**
   What is wrong: none of the figure's cell values differ from cell_table.jsonl. I checked all 138 cells' canonical fields against score.json; the 25 outlined cells are the eligible ones; the legend entries match PALETTE in plot173.py; the caption's statements are true. But labels overflow or are truncated: "Kun Chronic…" and "Deepmachine" run past their cell borders, and the n3 +1 cell shows "…" for a roughly 70-character string of 'g3333…', so the entity cannot be read.
   Fix: in plot173.py `display_name`, shorten names to fit the cell, e.g. cap at 10 characters with a tooltip-style footnote, and show "Other name (long)" for any name over a set length. No number changes.

## Checks that passed

- All 12 predictions: the registered text in README matches PREDICTIONS.md (lines 3273–3284), and each verdict matches prediction_table.json.
- Gates 1 to 9: counts match score.json. Gate 6 notes the omitted-dose probe with slope 1.0 against 0.5.
- Family counts (5/5, 4/5; 5/9, 3/9; 2/3, 2/3; 3/6, 3/6) match family_table.json.
- Best cells, 15 sets, 5 entities, 66.7% Google: match set_summary.
- 173.6 values (15, median 7, max 43) recomputed from per_set_maxima.
- 173.7 "1 qualifying pair, 0 different": consistent with the cells (n3 at ±0.5, both meta).
- Every relative link in the README resolves to an existing file.

## Limits of this check

The bootstrap intervals were not recomputed; I checked them only for internal consistency. I did not rerun score173.py. The zero4 modal count (3) and the Google claim counts were recomputed from labels-ds.jsonl. The second reader's labels were not recomputed; I took its agreement figures from score.json. I searched run logs only in the files named above for the 43 token-limit claim; a log outside this directory could contain it.

Count by severity: high 2, medium 5, low 5 (12 findings).
