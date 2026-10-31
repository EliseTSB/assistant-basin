# Part 129: two gates failed; the reading is not confirmed

Gate 1 and Gate 3 failed in the original run; after dropping L2, Gate 3 passes, while Gate 1 stays failed on count in both runs. The amendments were batch 16 for the main run and Gate 2 at batches 8 and 16 instead of 16 and 64; a mid-run cut for rented credit; and a post hoc drop of L2.

Nothing here is confirmed. The amended scores are post hoc summaries written after the prediction summaries had been seen. The registered rule stopped the original reading when a gate failed.

## Run and gates

The run recorded 12,009 documents after the length filter, not the planned 16,000. The erratum gives the actual slices: web lines 0 to 7,999; zh lines 490 to 1,499; and the first 1,500 lines of code and chat. The saved position totals are below; position 0 was dropped.

| Group | Positions |
| --- | ---: |
| web | 1,891,798 |
| zh | 236,409 |
| code | 373,102 |
| chat-user | 177,839 |
| chat-assistant | 204,299 |

The run resumed at document 8,000 on a corpus cut to 7,510 web lines. That skipped 490 zh lines, so the slices are not the equal share required by the Size rule. The zh numbers come from a smaller, later sample. The amended run drops L2 from score calculations; it does not change the captured documents.

Gate 1 measured 363 of 364 alignments within 5%, all on web, code, and chat. It still fails because the registered count was 500. All 136 unmeasured checks were zh checks in skipped lines 0 to 99. The pilot passed 300 of 300 checks, including 93 zh checks. The main run has no zh check. Gate 2 passed at batches 8 and 16, with minimum layer median r of 1.000 in each self-comparison.

Gate 3 failed in the original run: the limit was a median absolute null r of at most 0.05 at every layer and group. Only L2 exceeded it. After dropping L2, Gate 3 passes. Values below show L1 to L3 from the original run.

| Layer | web | zh | code | chat-user | chat-assistant |
| ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | 0.0210 | 0.0328 | 0.0395 | 0.0304 | 0.0279 |
| L2 | 0.0523 | 0.0627 | 0.0722 | 0.0624 | 0.0609 |
| L3 | 0.0132 | 0.0194 | 0.0198 | 0.0280 | 0.0201 |

At every other layer and group, the original null median is at most 0.040. The cause of L2's null is not established, and the design's withdrawal clause stands open at L2. The amendment cites the catalog's activation files, in its neurons folder: about a third of poles at L1 to L3 peak on ten positions; 3,638 of L2's 24,576 poles peak on one token in “You are on page 1 of”.

## Predictions under the two scoring runs

The scorer is [score129.py](../../part129/score129.py). The outputs are [score129.json](score129.json) and [score129-amended.json](score129-amended.json). The verdict column is post hoc. “Held” means the score met its numeric threshold, but these stopped and post hoc runs do not confirm it.

| # | Measure | Original score | L2 dropped | Post hoc verdict |
| --- | --- | --- | --- | --- |
| 129.1 | web r | median 0.9111; layer range 0.8475–0.9775 | median 0.9102; same non-L2 range | held |
| 129.2 | chat-assistant minus web | −0.017619 | −0.017680 | unreadable, not read |
| 129.3 | L30–35 minus L0–8 layer-median r | −0.050484 | −0.051900 | unreadable, not read |
| 129.4 | layers with absolute Spearman at most 0.3 | 34 of 36 | 33 of 35 scored layers | held |
| 129.5 | persona minus matched r, web; chat-assistant | +0.005240; +0.007086 | +0.005320; +0.007341 | held |
| 129.6 | once minus carrier, chat-assistant rectified r | −0.063157 | −0.063157 | held |
| 129.7 | low-r label hits; matched-control hits; ratio | 22/442; 3/442; 7.33 | 22/442; 8/442; 2.75 | unreadable, not read |

The design says 129.2 and 129.3 are unreadable if the null floor exceeds their effects, 0.03 and 0.05. For chat-assistant, the original L2 null is 0.0609, above both effects, so both are unreadable in the original run. After L2 is dropped, the largest remaining chat-assistant null is 0.0279 at L1, below both effects. The amended 129.2 score misses its 0.03 threshold. The amended 129.3 reduction is only 0.000484 beyond the 0.05 line in the original run; dropping L2 moves it by 0.001416. The change is larger than the margin, so 129.3 cannot be read as a pass or a failure.

For 129.7, one `random.Random(129)` serves all layers. Dropping L2 skips its draws, so every later layer redraws the control. The low-r set is also recomputed without L2: the two counts of 22 are not “the same” observations. The control hits changed from 3 to 8 of 442. A ratio threshold of 2 allows up to 11 control hits. A single draw cannot support a ratio.

## What r does and does not say

Both models read identical tokens in lock-step, and units are matched by index. r says that two units co-vary at the down_proj input across those positions. It does not show that the effect on the output is unchanged; the down column can differ while r stays high.

Null levels differ across depth. The chat-assistant null averages 0.0187 over L0–8 and 0.0084 over L30–35, so comparing r across layers carries different null levels. A chat gap may come from Base handling the chat template worse: the chat text uses Qwen3-8B's template, which Base was not trained on.

The results are Pearson correlations on position sums, not model outputs. Position 0 and padding were dropped. The run covers 12,009 documents, not 16,000. No pushes or generation were run. These are two checkpoints; intermediate Qwen post-training stages are not published. A neuron can keep firing in the same places and change partners or its effect on output. The scores do not establish a causal change in computation.

The main timing file divides 12,009 documents by only the post-resume time, so its `documents_per_second` is wrong. The resumed portion was 4,009 documents in 1,054.8 seconds, about 3.8 documents per second. The reported 11.4 documents per second is not the run rate.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-30; nine findings, all folded.
