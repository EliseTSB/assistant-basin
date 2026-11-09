# COLD.md: cold read of the Part 200 results README

**Verdict: SHIP-WITH-FIXES.** Every number I recomputed matches its cited file. The problems are omissions and overstatement: the README does not report lengths before comparing across runs (a registered reading rule), it does not attribute cross-run changes jointly to kernel and batch shape, and its headline says more than the verdicts support. None of these changes a registered verdict.

Scripts (in this folder, read repository files in place, run with `/home/elisepc/neuron-probe/.venv/bin/python -I -B`): `recompute.py` (counts, Qwen rule, batch-1 positions, label flips), `lengths_check.py` (cap-stratified broken counts, mid-phrase), `exact_check.py` (re-tokenization versus token_count), `handread_check.py` (excerpts and sampling order). `score-rerun/` is Part 163's scorer run on the rerun answers, output identical to `results/part200/score.json`.

## Findings

1. **Lengths are not reported before the cross-run comparison (README, 200.5 row and "Answer" paragraph; design reading rule "Lengths are reported before any rate is compared across the two runs").** Evidence (`lengths_check.py`): rerun broken answers at the cap: 366 of 387 neuron, 98 of 103 random. Broken among under-cap answers: published neuron 318/1,476 (21.5%) vs rerun 21/580 (3.6%); published random 72/1,382 (5.2%) vs rerun 5/688 (0.7%). Mean tokens published 55.1 / 59.2, rerun 68.8 / 68.3, rerun median 80 in both arms. The 200.5 "held" is correct as registered, but most of the rise in broken counts is answers that ran to the cap. Fix: add a length table beside 200.4 and 200.5, and say that the broken increase is concentrated in cap-hitting answers. Within-run arm ratio on under-cap answers is about 4.9 (3.6% vs 0.7%), so the arm gap survives the length split.

2. **Cross-run changes are not attributed jointly to kernel and batch shape (README, "Answer" paragraph and 200.3 / 200.5 rows).** The design says changes between runs go to "the kernel and the batch shape together, never to either alone". Only the at-cap section does this. The broken-count changes (343→387, 83→103) are presented with no attribution sentence. Fix: add the joint attribution sentence in the "Answer" paragraph and in the 200.5 row.

3. **A gate margin fell, and the README reports only "The gates passed" (README, first paragraph).** `results/part163/score.json` gate1 changed share was 0.88 in both arms; the rerun has 0.75 (neuron) and 0.71 (random), against a 0.70 floor. The random margin is 0.01. Fix: report the values beside "passed", and say what the drop means for the recipe dose effect on eager. I could not tell from the records whether the drop is kernel-driven.

4. **The headline overstates what the verdicts can show (title and first paragraph: "verdicts hold", "all five Part 163 verdicts stayed the same").** Per PREDICTIONS.md, 163.1 and 163.2 could not fail at these base rates (random name-gone 0.815, maker-gone 0.893). 163.3's bar is 0.15 against rates near 0.85. 163.4 is "nearly by construction" (Part 163 README) and the rerun margin is one unit (neuron 93 vs 92; random 89 vs 88). The live readings are 163.4 and 163.5, as the design says. The README says this only in the 200.2 parenthetical. Fix: say it in the first paragraph. (Unsure whether this is a defect or a wording choice; I lean to defect because the title is what gets quoted.)

5. **Two results that bear on the at-cap question are omitted (README, "Is the at-cap jump the kernel?").** (a) Text-level divergence: `compare.json` shows 390 of 3,200 battery answers identical to the published text (2,810 differ, 88%), and 4 of 16 no-push answers identical. A skeptic's first question is how much output changed, and the README reports only counts. (b) The exploratory under-cap mid-phrase count fell from 510/1,476 to 2/580 (neuron) and 409/1,382 to 1/688 (random) (`compare.json`, rule from `part199/report199.py`, recomputed in `lengths_check.py`). That collapse is the clearest evidence about answers stopping early. Fix: add both, labelled exploratory.

6. **The batch-1 first-difference method is sound for the classification but under-described (README, "Exploratory: batch-1 differences").** `recompute.py` and `exact_check.py`:
   - Re-tokenizing decoded text reproduces the generator's count: 3,178 of 3,416 rows match after the stop-token adjustment. The changed rows: 72 of 76 chunked and 72 of 76 batch-1 texts match exactly; the other 4 per side are mismatches.
   - `token_count` includes the stop token for under-cap answers (1,213 of 1,268 under-cap rows re-tokenize to exactly `token_count − 1`). The README should say this, since any length statement depends on it.
   - Position uncertainty is about ±1 token. The "64 keys" count is insensitive to it: 73 of 76 at thresholds 63, 64 and 65 (`recompute.py`); only three rows lie within 3 of 64 (positions 63, 63, 67).
   - Convention: positions are prompt length (17) plus the 0-based answer index. A row at exactly 64 would be ambiguous under Part 178's "more than 64 keys"; no row sits there, so the count is unaffected.
   - Interpretation gap (unsure): 73 of 76 changes occur at or below 64 keys, where Part 178 says the SDPA fault does not operate. That means batch shape changes eager output below the fault threshold. The README reports the counts and does not say what this implies. Fix: state the convention and the ±1 token uncertainty, and either interpret the sub-64 changes or say explicitly that you do not.

7. **Sampling of the hand read is not disclosed (`hand_read.md`; README "Exploratory" paragraph).** The five changed batch-1 pairs per arm are the first five changed rows in file order (`handread_check.py`: neuron L21N7991, L21N2550, L21N3124, L21N5297, L21N8377; random L19D219, L21D74, L21D284, L22D62, L22D150). The selection of the recipe and battery samples is also not stated. All 20 quoted recipe and battery excerpts match the stored answers, and the batch-1 phrases I spot-checked match. Fix: state how each sample was chosen.

8. **Run record and timing are unclear (`load.json`, `launch.jsonl`, README "Answer" paragraph).** (a) `load.json` has top-level `"part": 199` (a default in `part199/run199.py` line 247) and `chunk_size` / `batch_size` 64, while the calls ran in chunks of 8 (filed as a papercut). (b) The design says the pilot rate is recorded; neither `load.json` nor the README reports it. `summary.seconds` 3,836.6 for 3,416 answers is about 0.89 answers per second, against the 0.17 guard. (c) Unsure: the README's start and end timestamps (13:40:53 to 14:58:43) span about 4,670 s, against 3,837 s of generation; I did not find the reason for the gap.

9. **Process items (not README wording, papercuts filed).** The PREDICTIONS.md Part 200 rows have blank Result and Verdict columns. Part 163's README has no Part 200 headline update, which the design's reading rules require.

10. **Figure (`fig200.png`).** The left panel shows broken and at-cap counts only. Adding under-cap broken and mean length would show finding 1 directly. The right panel is fine.

## Checked and correct

- Predictions table (200.1 to 200.7): wording matches `PREDICTIONS.md` Part 200; results and verdicts match `report200.json`.
- Part 163 verdicts: `score163.py` re-run on the rerun answers gives output identical to `results/part200/score.json` (predictions, gates, rates). Published values match `results/part163/score.json`.
- Broken counts 387 / 103 (rerun), 343 / 83 (published); at-cap counts 1,020 / 912 and 124 / 218; mean and median lengths as in finding 1.
- Part 161 Qwen counts: 16/514 and 40/632 (rerun), 19/539 and 57/661 (published), reproduced with the case-sensitive rule.
- Batch-1: 200 rows, 76 changed (39 neuron, 37 random); first-difference counts 38 and 35 at or below 64 keys, one and two after; label flips name-gone 2, broken 6, Qwen 5.
- Units hash `2ed3bef4…dd710` matches the design; 3,416 rows; 3,200 battery; 200 batch-1; pushed calls are chunks of 8 (`load.json`).
- All 20 quoted excerpts in `hand_read.md` match the stored answers.
- Relative links in the README resolve to existing files.

## Unsure

- Finding 4: defect versus wording (I lean to defect).
- Finding 6: whether the sub-64 changes are worth interpreting in the README.
- Finding 8(c): the timing gap is not explained by any record I read.

SUMMARY: Part 200 README numbers all reproduce; SHIP-WITH-FIXES. Main fixes: report lengths before the cross-run comparison (broken rise is mostly at-cap answers), add joint kernel+shape attribution, state the gate-1 drop (0.88 to 0.71/0.75), and say only 163.4 and 163.5 are live readings.
FINDINGS: 10 numbered findings above; 4 of them (1, 2, 4, 5) should be fixed before publishing; no registered verdict changes.
BLOCKERS: none. Papercuts filed: load.json labels, PREDICTIONS.md blank rows, Part 163 README update missing.
