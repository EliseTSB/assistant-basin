# Cold check: Part 174 README (results/part174/README.md)

## Verdict

The core numbers hold. I recomputed the 28 of 80 u100d supported count, the 0 of 40 f100d count, the 24 draws with a supported sign (4 both same, 20 one sign), the class counts, the modal counts and stability from labels-ds.jsonl and answers.jsonl (0 mismatches against scored_directions.jsonl), the Wilson intervals, the exact bounds (7.2% for 40 draws, 13.9% for 20), the floor counts (38, 28, 19), the question-level bootstrap (28.8 to 38.8%), the draw-cluster bootstrap (25 to 45%, and 45 to 75% either sign), the stage0 SDs (cf 0.02118, cr 0.02125, max |cf| 0.0581), the length deviation, kappa 0.964, the 2,972 pairs and 8 failures, and the descriptive raw-word correlations (0.140 and 0.104 pooled). All README links resolve, and each linked file holds the number it is cited for, apart from three values that are derived rather than stored (26, 88, 0.27%). The verdicts for 174.1 to 174.5, 174.7, 174.8, 174.10, W1, W2 and W3 follow from the registered bars. The problems are in how counts are described. The Google-claim histogram and figure treat 32 f100d directions that have no clean answers as labelled zeros, so the README's "113 labelled, 53 zero" is wrong (81 labelled, 21 zero). The 174.6 and 174.9 "not read" verdicts rest on a missing-data rule the design does not contain. The per-family Spearman figures the design requires are missing. The f100d relation row and the fallback status are under-specified. Nothing claims a fallback name, a rate outside d = 0.5, or a layer-profile cause, and Part 173 draws are not pooled.

## Findings (most serious first)

1. **medium.** README: "It covers 113 labelled signed directions: 53 have zero Google claims." Recomputed: google_claim_count is 0 (not null) for all 40 f100d directions, but 32 of those have zero clean answers (reader_available False), so there is nothing to label. Only 81 directions are labelled (73 u100d, 8 f100d). Zeros among labelled directions are 21. The largest bin is still zero, and 26 of 81 (32%) lie between 5 and 15, so the conclusion survives but the numbers do not. Fix: exclude no-clean directions from the histogram, report "81 labelled, 21 zero, 26 in 5 to 15, 32 with no clean answers", and correct the "over_120_directions" label in score.json.

2. **medium.** README 174.6 and 174.9: "Their missing D values make the registered correlations unread." The design's not-read conditions are repeated D over half the group, sd below 0.01, or nonfinite rho. Missing D is not among them, and the error cap (0.27% against 0.5%) did not trip, so the design does not unread these. On the 53 complete draws the registered-D rho is cf 0.157 (percentile CI about -0.11 to 0.41) and cr 0.033 (about -0.22 to 0.29). Both are under 0.30. Fix: either report those values with n = 53 and the 7 excluded draws named, or record that a missing-D rule is a judgment not in the registration and needs Weaver's ruling before the verdict stands.

3. **medium.** README paragraph on the Spearman check gives only the pooled descriptive figures (cf 0.140, cr 0.104). The design requires each to be shown for u100d and f100d apart beside the pooled figure. Recomputed u100d-only descriptive values: cf 0.179 (CI -0.156 to 0.479) and cr 0.141 (CI -0.164 to 0.426). f100d is not read because D is constant (0 on all 20 draws). Fix: add the u100d row and the f100d not-read row beside the pooled row.

4. **medium.** plot174.png and its caption: "Open diamonds mark missing reader counts, shown separately from zero." The 32 no-clean f100d directions are plotted as filled dots at Google claim count 0, indistinguishable from labelled zeros, so the figure shows observed zeros that were never read. The dots themselves match scored_directions.jsonl. Fix: draw the no-clean f100d directions as a separate marker (or drop them) and update the caption to name them.

5. **low.** README table "Relation labels: claim / none": f100d "0 / 28", "Google as primary name 0", "Google mentions 0". The u100d row covers 3,006 labelled answers out of 3,520, and the f100d row covers 28 of 1,760, so 1,732 f100d answers broke. The sentence "there were no Google primary names or mentions" is correct only for those 28. Fix: add the denominators to the row ("28 clean of 1,760 answers; 1,732 broke") and scope the sentence to clean answers.

6. **low.** README: "The floor file flags incomplete reader coverage in 7 directions, so these floor counts are incomplete." I checked: none of the 7 can change the 28 (10 floor), 38 (5 floor) or 19 (15 floor). Each has broken above 8 or a modal count too low to reach the floor even with its unlabelled answers. Fix: state that, so the headline 28 is not read as possibly undercounted.

7. **low.** README never states the fallback reading's status. Design: read only if 174.2 and 174.10 are both read and hold. 174.10 is not read, so the fallback reading is not read. Fix: add one sentence giving that status.

8. **low.** README: "The f100d family had 0 of 40 supported directions." Design requires each family's supported, Google-led and other-led counts reported together. For f100d that is 0, 0, 0, which is in score.json but not in the README. Fix: state all three.

9. **low.** README: "primary agreement 149 of 149 pairs and 171 of 171 answer-weighted." It does not say which pairs. Across all 222 pairs where both readers gave a claim, the resolved primary agrees on 218, so 4 disagreements sit outside the replacement sample. Fix: name the stratum, and state the 218 of 222 figure.

10. **low.** plot174.png: the vertical line is labelled "10-of-44 support floor" and the caption says it "marks the support floor." It marks only the modal-count floor. A direction to its right with more than 8 broken answers is not supported. Fix: relabel as "modal-count floor (support also needs at most 8 broken)".

11. **low.** score.json spearman_tests.cf.f100d and cr.f100d report standard_deviation 0.02118 and 0.02125, which are the pooled 60-draw SDs. The f100d-only 20-draw SDs are 0.02502 and 0.02484 (descriptive_fallback). No verdict changes. Fix: compute the family SD over the family's draws.

12. **low.** README last line is "Cold check: pending." Replace with the result once this check is accepted.

Counts by severity: 0 high, 4 medium, 8 low.
