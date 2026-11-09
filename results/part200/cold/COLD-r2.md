# COLD-r2.md: cold check, round 2, Part 200 after the round-1 fold (commit 81c2310)

**Verdict: SHIP-WITH-FIXES.** Every number I recomputed matches its file, and the PREDICTIONS.md Part 200 rows match `score.json`, `compare.json` and `load.json`. The problems are new: one dangling cross-reference in the Part 163 update (it points to verdicts and counts that are not on the page), a wrong provenance and a wrong timestamp label in the Part 200 README, and some interpretive sentences that go past what the records show. None changes a registered verdict or a count. No fix needs a rerun.

Scripts (in this folder; they read repository files in place, run with `/home/elisepc/neuron-probe/.venv/bin/python -I -B`): `recompute.py`, `lengths_check.py`, `exact_check.py`, `handread_check.py` (round 1, rerun unchanged), `r2_order.py` (file order of sampled rows), `r2_load.py` (load-record counts). `report200.py` and `fig200.py` were not run in place.

## Round-1 findings: status after the fold

| R1 | Status | Evidence |
| --- | --- | --- |
| 1 lengths before rates | folded | Length table (README "Lengths before cross-run rates") matches `recompute.py`: mean 55.1/59.2 → 68.8/68.3; median 54/60 → 80/80; at-cap and under-cap split; broken at cap 25/11 → 366/98; broken under cap 318/72 → 21/5. Sums to 1,600 per cell. One wording problem: F4. |
| 2 joint attribution | folded | Answer paragraph, 200.3 and 200.5 rows now carry the kernel-plus-shape sentence. Undermined by F4. |
| 3 gate 1 margin | folded | 0.88 → 0.75 / 0.71 against 0.70 (`score.json`; published 0.88 in both arms). Wording: F11. |
| 4 headline overstatement | partly | Title still says "verdicts ... hold". First paragraph now names the live readings and says 163.1 and 163.2 could not fail. Missing: 163.4 "nearly by construction" (Part 163 README); 163.5 "live" is unargued: F7. |
| 5 identical text, mid-phrase | folded | 390 of 3,200 (`compare.json`; neuron 186, random 204); 4 of 16 no-push; mid-phrase 510/1,476 → 2/580 and 409/1,382 → 1/688. Labelled exploratory. |
| 6 first-difference method | folded, overreach | Convention (17 + 0-based index), 3,178 of 3,416 re-tokenize, 1,213 of 1,268 under-cap at `token_count − 1`, ±1 token, 73 of 76 at 63/64/65: all reproduced. Interpretation: F5. |
| 7 sampling | folded | Batch-1 picks are the first five changed rows in file order (neuron L21N7991, L21N2550, L21N3124, L21N5297, L21N8377; random L19D219, L21D74, L21D284, L22D62, L22D150), confirmed. Recipe and battery picks are not first-N in file order (positions 18 to 215), so "purposively chosen" is supported. The criterion is not recorded per row: F10. |
| 8 record and timing | partly | (a) "part": 199 labelled. The "chunk_size 64" provenance is wrong: F3. (b) Pilot rate still not reported separately; overall 0.89 answers per second and the 0.17 guard are. (c) "records do not explain the gap": F9. Start-time label: F2. |
| 9 PREDICTIONS blank rows | folded | Rows filled; checked against files (below). The Part 163 update exists but is defective: F1. |
| 10 figure | partly | Under-cap broken bars added (318, 21, 72, 5 correct). Mean length not added. Tick labels overlap: F8. |

## Findings (new text and remaining issues)

1. **Part 163 update: dangling references (`results/part163/README.md` line 3).** The update says "the five verdicts came out as below" and "1,020 and 912, against 124 and 218 below". Nothing rerun follows. `grep` finds 124 and 218 only in the update itself, and the Part 163 table (lines 10 to 15) shows published verdicts and published values only. This also breaks the design's reading rule that a headline update name which published numbers held and which did not, in the same sentence. Evidence: `grep -n "124\|218" results/part163/README.md` returns line 3 only. Fix: replace "came out as below" with "163.1 to 163.4 held and 163.5 failed, as published (163.4 and 163.5 the live readings; rerun values in Part 200)"; replace "against 124 and 218 below" with "against 124 and 218 published (Part 200 table)" and link `../part200/report200.md#battery-broken-and-at-cap-counts`.

2. **"The model load began at 13:40:53" is wrong (Part 200 README, first paragraph; front-page entry is unaffected).** `part200/run200.py` writes the load-record `started` field in `recording_loader` after `loader(device)` returns (about line 618), so 13:40:53 is when the model had loaded and the record was written. The launch is 13:40:08, so loading took about 45 seconds before that. Fix: "the load record was written at 13:40:53, after the model loaded".

3. **`chunk_size` provenance is wrong (Part 200 README, "Exploratory" paragraph; `report200.md` "Load record" line).** The README says "`chunk_size` and `batch_size` are 64 defaults inherited from run199". Run199's chunk size is 8 (`part199/run199.py:54`, `CHUNK = 8`). The `"chunk_size": 64` in `load.json` is the `repair` block, produced by `run200.repair_startup(folder, size=BATCH_SIZE)` (`run200.py:468`, `:484`). It is the batch size passed to the repair step, not a chunk size. `batch_size` 64 is in `summary` and `argv`. Every pushed chunk was 8: 68 pushed calls, 425 chunks, all of size 8 (`r2_load.py`). Fix: "`repair.chunk_size` holds 64, which is the batch size passed to the startup repair; every pushed chunk was 8 (425 of 425)". Also filed as a papercut.

4. **"The broken rise is a length change as much as anything" (Part 200 README, Answer paragraph).** This is in tension with the next sentence, which attributes cross-run changes to kernel and shape together and never to either alone. It also goes beyond the design's "may partly be a length difference". The numbers show a composition shift, not a rise. Net broken change is +44 neuron (343 → 387) and +20 random (83 → 103). Broken at cap rose by 341 and 87; broken under cap fell by 297 and 67. The title's "broken answers move to the 80-token cap" has the same problem. Fix: state the two net changes and the composition shift, then say "the design allows that part of the cross-run change is length; the records do not separate length from kernel and shape."

5. **"These are near-tie flips, not the SDPA fault" is asserted, not measured (Part 200 README, Exploratory paragraph; front-page entry).** No top-two margin was recorded at the 76 changed batch-1 rows. The README then says "This interpretation is a reading, not a test", which does not cover the declarative sentence before it. The design's reading rule also says a one-row decode's change "is not evidence about the 64-key fault". The README's use of "where Part 178's fault does not operate" leans on that evidence. Fix: delete "these are near-tie flips, not the SDPA fault", and write "consistent with near-tie flips; margins were not recorded". Drop the Part 178 clause or state the design rule beside it.

6. **Banner sentence "batch shape alone changed 76 of 200 eager answers" (front page, line 25).** Accurate: batch-1 and chunked rows both use `EagerLoader` (`run200.py` lines 13 and 258), so the only difference is batch shape. It reads as contradicting the attribution rule two lines later. Fix: "in the batch-1 comparison, both arms eager, 76 of 200 answers differed."

7. **"163.4 and 163.5 were the live ones" (Part 200 README first paragraph; Part 163 update; `PREDICTIONS.md` 200.2).** 163.4's margin is one unit (93 vs 92 neuron; 89 vs 88 random), and the README says so. 163.5 has no margin argument: neuron 0.242 (0.193 to 0.294) against random 0.064 (0.048 to 0.083), intervals far apart. Calling it live without stating the margin is unsupported; by the same logic as 163.1 it is not at realistic risk. Also missing from the Part 200 README is the Part 163 caveat that 163.4 "holds nearly by construction because both-gone is a subset of name-gone" (round-1 finding 4). Fix: state 163.5's gap (about 3.8 times, non-overlapping intervals) and add the 163.4 construction caveat in the first paragraph.

8. **Figure (`results/part200/fig200.png`, `part200/fig200.py`).** Under-cap broken bars are present and correct. Two problems on the rendered PNG: (a) the x tick labels "neuron broken under cap" and "random broken under cap" overlap; (b) mean length, which round-1 finding 10 asked for, is absent. The alt text still reads "broken and at-cap counts". Fix: wrap the under-cap labels (for example "neuron\nunder-cap broken") or widen the axis; add mean tokens as an annotation or a third panel; update the alt text in the README.

9. **"The records do not explain the gap" (Part 200 README, first paragraph and Exploratory paragraph).** The 4,670-second span runs from the record's `started` to `updated`. The 3,836.6-second figure is `summary.seconds`, timed inside `run163.run` (`part163/run163.py:141` and `:233`). `run200.py` line 630 runs the 200 batch-1 one-row calls after `run163.run` returns, and nothing times that stage. The roughly 833-second gap is therefore most likely the untimed batch-1 stage, at about 4.2 seconds per call. This is inferred from the code and cannot be confirmed from the records. Fix: "the gap is probably the batch-1 stage, which the record does not time; this is inferred from `run200.py`." Also filed as a papercut.

10. **Gate 1 sentence (Part 200 README, Answer paragraph).** "So the recipe dose effect still clears its eager gate narrowly" names the wrong thing. Gate 1 is the recipe changed-share check, at least 70% of each arm's regenerated "Who are you?" answers triaging as changed (`PART163-DESIGN.md` line 25). The random arm's 71 of 100 is one unit above the floor. Fix: "the random arm clears the 0.70 floor by one unit (71 of 100)."

11. **"Batch-8" wording (Part 200 README, "Does the batch shape matter" section).** "76 of the 200 batch-1 answers differ from their batch-8 answers". The comparison is against chunks of 8 from a pushed batch of 64. Fix: "from their chunked answers (chunks of 8)".

12. **Hand-read criterion (`hand_read.md`, README "Exploratory" paragraph).** The recipe and battery examples are described as "purposively chosen to show ordinary self-identification alongside malformed, repetitive, or otherwise pathological output". That is honest and supported (the picks are not first-N), but no per-row rule is recorded, so a reader cannot reproduce the choice. Fix: list the criterion per arm, or say "chosen by hand, no recorded rule".

## Checked and correct

- Lengths table: all 16 cells recomputed from `answers.jsonl` and `score.json` (`recompute.py`, `lengths_check.py`).
- Counts: broken 343/83 published, 387/103 rerun; at cap 124/218 published, 1,020/912 rerun; broken at cap 25/11 and 366/98; under-cap broken 318/72 and 21/5; under-cap rates 21.5%, 5.2%, 3.6%, 0.7%.
- Qwen rule (Part 161): 16/514 and 40/632 rerun; 19/539 and 57/661 published; case-sensitive rule.
- Gate 1: neuron 0.75 (75 of 100), random 0.71 (71 of 100), floor 0.70; published 0.88 each arm.
- Verdicts (`score.json`): 163.1 ratio 1.047 (0.973 to 1.128); 163.2 1.042 (0.990 to 1.101); 163.3 neuron 0.853, random 0.815; 163.4 93 vs 92 and 89 vs 88; 163.5 neuron 0.242, random 0.064, failed. Published values in the README table match `results/part163/score.json`.
- Batch-1: 200 rows, 76 changed (39 neuron, 37 random); first differences at or before 64 keys 38 and 35; 73 of 76 at 63, 64 and 65; re-tokenization 3,178 of 3,416 (`exact_check.py`); 1,213 of 1,268 under-cap at `token_count − 1`; rendered prompt 17 tokens.
- Exploratory counts: 390 of 3,200 battery answers identical (186 neuron, 204 random); 4 of 16 no-push identical; mid-phrase 510 → 2 and 409 → 1 (`compare.json`).
- Record: launch 13:40:08; `updated` 14:58:43; 68 pushed calls, 425 chunks, all of size 8; 3,416 answers; 200 batch-1 rows; `summary.seconds` 3,836.6 (0.89 answers per second).
- Hand-read excerpts: 20 of 20 recipe and battery excerpts and the changed batch-1 excerpts match the stored answers (`handread_check.py`).
- `PREDICTIONS.md` Part 200 rows, Result and Verdict: 200.1 HELD (gate 1 0.75 and 0.71; R1 and R3, 68 pushed calls in 425 chunks of 8); 200.2 HELD (1.047, 1.042, 163.3 held, 163.4 93 and 89, 163.5 failed as published); 200.3 HELD (387 vs 103, 3.76 times); 200.4 HELD (1,020 and 912 at cap, ≥ 480); 200.5 HELD (387 vs 343, 103 vs 83); 200.6 HELD (16/514 vs 40/632); 200.7 FAILED (76 of 200 vs at most 20). All match `score.json`, `compare.json` and `report200.json`.
- Front-page entry: numbers 0.88, 0.75, 0.71, 0.70, 16/514, 40/632, 387, 103, 1,020, 912, 76/200 and "six of seven predictions held" all match a file. The entry is placed after Part 199's.

## Unsure

- Finding 9: the batch-1 explanation is inferred from code; no batch-1 timing is recorded.
- Finding 7: whether "live" for 163.5 means something narrower than I read it (the design does not define it).
- Finding 5: whether the Part 178 clause counts as stating the design rule or as a new claim. I lean to new claim.

SUMMARY: Part 200's numbers, PREDICTIONS rows and front-page figures all reproduce. SHIP-WITH-FIXES. Fix first: the dangling "below" references in the Part 163 update (F1), the "load began" timestamp (F2), the `chunk_size` provenance (F3), and "the broken rise is a length change" (F4). The rest are wording and the figure.
FINDINGS: 12 numbered findings; F1 to F5 should be fixed before publishing; none changes a registered verdict or a count; round-1 items 1 to 3, 5, 6, 7 and 9 folded; 4, 8 and 10 partly.
BLOCKERS: none.
