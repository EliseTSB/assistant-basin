# Cold check: results/part175/README.md (Part 175)

Read-only check in the Part 175 worktree. No Arc, no paid API. Recomputation used the run-specified Python with `-I -B`. Read in full: PART175-DESIGN.md (including Reading rules, Limits, Amendments 1–3), results/part175/README.md, score.json, prediction_table.json, scored_directions.jsonl, stage_a.json, gates.json, layer_norms.json, readers.log, numbers175.json, numbers175.py, plot175.py, plot175.png, and the labels folder. Part 174 and 173 files consulted for the comparison figures.

Recomputation method: census per-direction broken, clean, labelled, unlabelled, Google claims and modal counts were rebuilt from answers.jsonl and labels/labels-ds.jsonl. Broken used Part 165's `primary_broken` (imported from part165/part86/part166); it matched the stored count in 120 of 120 census directions and 64 of 64 Stage A arms. Support used the design rule (modal count of at least 10, Qwen excluded from the modal, broken at most 8); it matched all 120 supported flags. Wilson intervals were computed directly. The bootstrap is my own percentile version (10,000 resamples, seed 175, draws resampled with both signs kept together).

## Verdict

The README's arithmetic reproduces. D* = 0.30 and the Stage A medians (6.5 at D*) match. Census counts are 1, 27 and 1 supported at 0.30, 0.225 and 0.15, with median breakage 29.5, 2 and 0. The Google shares are 5.9% (83/1,398), 33.1% (511/1,542) and 7.1% (50/705). All 27 supported directions at 0.225 are Google-led, and the sign split is 14 and 13. Draws supported: 17 of 20 at one sign, 10 of 20 at both. The layer values, the Part 174 and Part 173 comparison figures, the reader gates and all 19 link targets check out, and the verdicts match prediction_table.json and PREDICTIONS.md. The 14 reader failures do not change any verdict. The problems are in framing and disclosure. The title and first paragraph present 27 of 40 at 0.225 as a finding about flat pushes, but it is the best of three doses chosen on the same 20 draws it counts. The paragraph on survival below breakage reaches for a mechanism the design does not license. Several design-required items are missing from the README: intervals, the drift condition, and the inert-guard result. Two statements about the reader history do not match readers.log. Nothing here changes a verdict.

## Findings (most serious first)

1. **[high] Title and first paragraph.** "the registered dose still breaks; a lower selected dose names Google on these draws" and "At the lower 0.225 dose, 27 of 40 directions were supported ... That was the design's selected dose, the best of 3 ... after seeing them on Part 174's draws."
   - What is wrong: score.json `selected_rule` is "most definitely supported signed directions among the three doses, ties go to the higher dose". That rule was run on the census's own 20 draws, so 27 of 40 is the in-sample maximum over three doses. The design's Controls bullet "Select on one set, test on another" selects on four separate Stage A draws, and the design does not define "best", so the design's select/test split does not cover 0.225. The README says "chosen after seeing them on Part 174's draws" but never says the headline count is the one it was chosen on.
   - Fix: retitle, e.g. "...; in-sample, the best of three doses supported 27 of 40 flat directions on these 20 draws, all Google-led". Add to the first paragraph: "The best-of-three rule (most supported signed directions, ties to the higher dose; score.json `selected_rule`) was applied to these same 20 draws, so 27 of 40 is an in-sample maximum and not a test. The design does not state this rule." Drop "names Google" as a flat assertion.

2. **[medium] Band paragraph (Does the name survive below breakage?).** "Muse warned that a name appearing only in a narrow band just below breakage is most plausibly an off-manifold artefact. On this evidence, that is the more plausible reading here; the name does not survive well below breakage."
   - What is wrong: this picks a mechanism, which the Limits and the design's "no assumed cause" stance do not license. The warning is attributed to Muse, but "off-manifold" appears in no file in the repo except this README, and the design records Muse's review only as prior art. At 0.15 the README concludes absence, but the inert guard was run only at D*, so the README cannot separate "name absent" from "push too weak to name anything", which its own Limits concede.
   - Fix: delete the "more plausible reading" and "off-manifold" sentences, or cite Muse's written review by path. Replace the conclusion with: "At 0.15 (0.5 D*), where no direction breaks, 1 of 40 is supported and Google is 5.9% of labelled clean answers. The name is not present there. Whether that is absence or a push too weak to name anything was not tested; the inert guard was run only at D*."

3. **[medium] Registered predictions table, rows 175.2, 175.3, 175.4, 175.6.**
   - What is wrong: design lines 141–142 require 175.2 to 175.4 to "carry that condition beside them" when 175.5 fails. prediction_table.json carries "175.5 failed; predictions remain reported with the registered drift condition" for 175.2, 175.3 and 175.4. The README table omits it.
   - Fix: add the condition to the Verdict column or as a note under the table.

4. **[medium] No intervals anywhere in the README.**
   - What is wrong: the design's Reading rules (lines 144–145) require a Wilson interval at each sign on k of 20 draws, a pooled cluster bootstrap over draws (seed 175, 10,000 replicates), and the design's zero rule. The README gives only point counts. The prediction_table bounds are exact counts. Recomputed:
     - 0.225: positive 14/20, Wilson 48.1–85.5%; negative 13/20, Wilson 43.3–81.9%; pooled 27/40 = 67.5%, bootstrap 50.0–82.5%.
     - 0.15: positive 1/20, Wilson 0.9–23.6%; negative 0/20, Wilson 0–16.1%; pooled 1/40 = 2.5%, bootstrap 0–7.5%.
     - 0.30: positive 1/20, Wilson 0.9–23.6%; negative 0/20, Wilson 0–16.1%; pooled 1/40 = 2.5%, bootstrap 0–7.5%.
     - The design's zero rule (exact one-sided 95% bound, 1 − 0.05^(1/20) = 13.9%) is not applied anywhere. It should say which quantity it bounds.
   - Fix: add these intervals to the first section and the dose comparison, with the design's zero rule applied and labelled.

5. **[medium] Method, "The first reader pass ended with 45 of 3,782 failed pairs, all temporary failures in name resolution."**
   - What is wrong: readers.log's first reader step prints error_fraction 0.10629 = 402 of 3,782 (10.6%). The 45 is the result of the retry pass that ended at 10:01. Amendment 1 calls it the retry pass result, and the log's own fraction 0.01877 = 45/2,397 attempted. The README does not mention the 402 first-pass failures or what the retry pass re-ran. "All temporary failures in name resolution" is Amendment 1's statement about the 45 only.
   - Fix: "First pass: 402 of 3,782 failed (10.6%, readers.log). Retry pass: 45 still failed (Amendment 1's denominator 3,782; the log's denominator for that pass is 2,397), all name-resolution errors per Amendment 1." State how many of the 402 were labelled by the retry. I could not determine from readers.log which pairs the 10:01 pass re-ran; that needs the pass's input list.

6. **[medium] Method, "The final first-reader error count was 14 of 3,782," next to Amendment 2's "13 pairs remain failed for reasons other than DNS."**
   - What is wrong: all 14 final failures in labels-ds.jsonl carry `stop_reason: "max_tokens"`. None is a DNS error. Amendment 2's 13 plus 18 DNS retries gives 13, not 14, unless one retried DNS pair failed again with max_tokens. The README does not reconcile the two. I mapped the 14 to census answers: 11 directions are partly labelled, 8 at 0.15 (f100d_d174103−, 105±, 106+, 108±, 114−, 120−) and 3 at 0.225 (114+, 118+, 118−).
   - Verdict impact, checked: at 0.225 the three partly labelled directions have labelled modal counts of 10, 18 and 12, with 1, 1 and 2 unlabelled answers, and no other labelled name reaches 10, so each is supported and Google-led under every filling. At 0.15 each partly labelled direction has labelled modal plus unlabelled at most 5, below 10. The [1, 1] and [27, 27] bounds in score.json are therefore correct.
   - Fix: state that the 14 are max_tokens failures, reconcile 13 with 14, and state the bound check above so a reader can see 27 is robust.

7. **[medium] Amendment 3 paragraph and "Primary agreement was 188 of 189."**
   - What is wrong: Amendment 3 (written 10:20) records primary agreement of 183 of 189 before the retry. The retry moved five second-reader comparisons to agreement. The README reports only the post-retry 188 of 189 and says only that the retry "changed no first-reader label". Gate 9's agreement threshold is 0.8, so pass/fail is unchanged, but a reported statistic moved after the first scorer run and the README does not show it. What I could verify: the first-reader labels file hash (1b2771bc…) equals the hash frozen when the second-reader sample was drawn, so first-reader labels did not change after sampling. I could not check against a pre-retry snapshot.
   - Fix: "Before the Amendment 3 retry, primary agreement was 183 of 189; after it, 188 of 189 (score.json). Gate 9 passes at either value."

8. **[low] Line 17 "naming is weak at 0.15, strong at 0.225, and broken at 0.30," and the share paragraph.**
   - What is wrong: at 0.225 the Google share of labelled clean answers is 33.1% (511 of 1,542), which is not a majority, so "strong" overstates the share. The three shares are computed over different direction sets (32, 37 and 38 directions), so the cross-dose comparison is not like-for-like. Including labelled answers from partly labelled directions gives 5.3% at 0.15 (93 of 1,739). The conclusion holds, but the figure should say which definition it uses. At 0.30 the 7.1% comes from 38 directions with a median of 29.5 broken, so it is a different population.
   - Fix: say "33% of labelled clean answers name Google, and 27 of 40 directions are Google-led at the mode". State that the three shares use different direction sets, and give the all-labelled alternative for 0.15.

9. **[low] Inert guard not reported.**
   - What is wrong: design lines 67–69 require the share of the 1,760 answers at D* whose first 40 tokens differ from `none` to be computed, with under 15% meaning inert. I recomputed 1,760 of 1,760 (100%), so the census is not inert. score.json records this (`inert_guard`, `inert = False`), but the README never mentions it. No equivalent check exists at 0.15 or 0.225, which is the gap in finding 2.
   - Fix: add one sentence to the Method stating the guard passed (1,760 of 1,760 differ) and that it was run only at D*.

10. **[low] plot175.png and plot175.py.**
   - Checked: null counts are drawn in a separate column at x = 46 labelled "null", not at zero. Median labels (6.5, 29.5) and point positions match stage_a.json and scored_directions.jsonl.
   - What is wrong: the lower panel draws a dashed "10-of-44 modal floor" line on the Google-claim-count axis. That floor applies to the modal count, which can belong to another maker, and it also needs broken at most 8. A Google count of 10 does not mark support. The figure shows 0.225 directions unsupported at 18 Google claims (d174103−, broken 9), so the line misleads. The upper panel title is clipped at the top edge, the lower panel title is clipped at the right ("nulls are separa…"), and the Stage A legend overlaps the points at 44 broken near |d| = 0.45. The lower panel has no legend for the grey crosses; only the "null" tick explains them.
   - Fix: relabel or remove the floor line on the claim axis, fix margins and title clipping, and move the legend.

11. **[low] Layer section (lines 25–52).**
   - What is wrong: the table shows u100d written norm 0.000 at layers 1, 13 and 14, while the file records 21 written layers for both draws. The README does not explain the zeros, and the 30.999 total includes them. The section reports only f100d_d174101 at 0.30 and u100d_d174001 at 0.50. layer_norms.json has 185 rows (4 Stage A draws × 8 doses × 2 signs, 20 census draws × 3 doses × 2 signs, and one u100d row). The 0.225 dose the README headlines does not appear in the layer discussion. Written norm is linear in d, so the 0.225 values follow by scaling, but the README does not say so.
   - Verified: layer values at 0, 2 and 20, and totals 23.154 and 30.999, match numbers175.json and layer_norms.json. "The ratio is larger for the flat push in early layers" holds for layers 0 to 4 only; from layer 5 the u100d ratio is larger.
   - Fix: explain the zeros (check Part 174 first); add a 0.225 sentence with the linear scaling; say "layers 0 to 4".

12. **[low] readers.log does not record the final state.**
   - What is wrong: the last entry is the first scorer run (10:19:31, prediction_read_status "partial", with 175.2, 175.3, 175.4, 175.6 and 175.W2 "not read"). The Amendment 3 retry and the rerun that wrote the final score.json (read status "read") are not logged. "Gate 9 then passed" rests on score.json alone. Also, each step's output prints before its `step_rc=` line, so attributing output to a pass takes inference, which made finding 5 harder to check.
   - Fix: append the Amendment 3 retry and rerun to readers.log, and note the output-before-header ordering in the log header.

## Checked and correct (no finding)

- D* = 0.30 from the design's rule; Stage A pooled medians 0.10 to 0.45 reproduce (6.5 at 0.30; 31.0 at 0.35 shows the sharp edge).
- 175.W3: sign gaps are 1, 0, 0, 0, so 0 of 4 draws have a gap of at least 2 and the verdict "failed" is right.
- Census counts, medians, Google shares, the 14 and 13 sign split, and 17 of 20 and 10 of 20 draw counts.
- Verdicts 175.1, 175.2, 175.3 (not read), 175.4, 175.5, 175.6 (not read), 175.W1, 175.W2 and 175.W3 match the design's bars and the prediction_table.json.
- Part 174 figures: 28 of 80 (35%), median 5, range 0 to 35; f100d range 32 to 44 with median 44. Part 173 figures: Google 1, 9 and 0 at 0.25 with at most 1 broken; 8, 26 and 36 at 0.5 with at most 2 broken; broken 35, 10 and 44 at 0.75.
- Gates 1 to 8 pass in gates.json. Known-example checks are 20 of 20 for both readers.
- Amendments 1 and 2 are disclosed in order and before any count. Amendment 3 is correctly stated as made after the first scorer run.
- All 19 distinct link targets resolve, and each cited file contains the number linked from it.
- The 175.1 to 175.W3 wording in the README matches PREDICTIONS.md.

Count by severity: 1 high, 6 medium, 5 low (12 findings).
