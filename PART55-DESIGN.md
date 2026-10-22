# Part 55: which Claude tier is good enough for the bulk labelling in the neuron work? (registered 2026-10-22 00:44 CT, before running)

Elise, 2026-10-22 00:43 CT: how do Opus, Sonnet and Haiku benchmark on the bulk neuron work, with about $89 of promo credit expiring on 10-24.
Nothing has compared the tiers on our own tasks. The reference labels come from Opus 5.5 (the reader used so far), so this
measures agreement with it and on known-answer calibration texts, not truth; the calibration texts are the only items with known answers.
- Tasks, all from committed data: T1 the 454-item Part 26.8 origin batch (432 answers, 22 known-answer calibration texts; three-way
  EARTH / OFFWORLD / NEITHER plus OK / BROKEN; blind_origin.py's prompt); T2 the 1,812-item Part 49/49b broken-answer batch (1,792 answers, 20 calibration;
  BROKEN / OK; blind_reader.py's prompt). Both batches' Opus labels (default thinking setting) are the reference.
- Arms: claude-haiku-5-5, claude-sonnet-5-5, and claude-opus-5-5 again (the noise floor of Opus against its own earlier labels). Same prompt,
  same items, thinking explicitly disabled in all three (Haiku 5.5 thinks by default, so an unset field would confound the arms), max_tokens 1024,
  Message Batches API. One batch per arm. Cost recorded from the returned usage.
- Measures: agreement with the reference label per task (exact two-label match for T1); disagreement table; Cohen's kappa on T2; calibration accuracy.

| id | prediction (Claude) |
|---|---|
| 55a | T2: Haiku agrees with the Opus reference on at least 95% of the 1,792 answers and Sonnet on at least 97%. |
| 55b | T1: Haiku agrees on at least 85% of the 432 answers (both labels) and Sonnet on at least 92%. |
| 55c | Calibration: Sonnet and the Opus rerun get every calibration text right (22 of 22 and 20 of 20); Haiku gets at most one wrong in each task. |
| 55d | At least 60% of Haiku's T1 disagreements involve the NEITHER label on one side. |
| 55e | The Opus rerun with thinking disabled agrees with its own earlier labels on at least 97% of T1 and of T2 answers. |
| 55f | Per item cost ordering is Haiku < Sonnet < Opus with Haiku at most one fifth of Opus. |

## Amendment 1 (2026-10-22, before any scored result)
The first Sonnet and Opus submissions were rejected whole (2,266 errored each, not billed): Sonnet 5.5 turns thinking off with {"type": "between_tools"},
and Opus 5.5 does not accept "disabled". Haiku's batch (msgbatch_01VBbWg5MYZxyHbHRtHBCWsJ) ran with thinking disabled and is kept. Sonnet reruns with between_tools; Opus reruns
with adaptive thinking at the lowest effort ("low"), the closest it allows to off. So the arms are NOT identical in thinking setting; the Opus arm may think a little. Predictions unchanged.

## Amendment 2 (2026-10-22 00:5x CT, Elise: "lets try haiku and sonnet at low thinking effort too")
Two more arms, same items and prompt: haiku-low and sonnet-low, each thinking adaptive with output_config.effort "low" (a direct one-word probe showed both models accept it).
That makes five arms: haiku (thinking disabled), sonnet (between_tools), opus (adaptive, low), haiku-low, sonnet-low. Elise noted Haiku below 100K tokens per request is about a tenth of the price; the
cost line therefore uses the Console's own per-request figure where I can get it, and the batch discount is whatever the usage page shows. Predictions 55a-55f stay as written for the
original three arms; for the low arms I add 55g: low effort changes agreement with the reference by no more than 2 points against the same model with thinking off.

## Amendment 3 (2026-10-22 01:1x CT, Elise: cache the fixed instructions in a system prompt; let the judge say more than a choice)
New arm haiku-v2 (Haiku 5.5, thinking disabled, same items): the rubric, a taxonomy of breakage kinds, the reply format and eight worked examples (none from the study) go in ONE cached system block
per task (cache_control ephemeral, ttl 1h; measured by direct probe: both blocks cache, 962 and 744 tokens, and the cache minimum for Haiku 5.5 and Sonnet 5.5 is between 302 and 524 tokens, so the
old ~200 to 350 token rubrics could not have been cached); the user turn is only the text. The judge may add a second line "NOTE:" (at most 30 words) saying how a text is broken or
anything else worth a researcher's attention; the first line stays machine-readable. Caveats: the taxonomy quotes one real item pattern ("My name is AI, and I is a model"), and batch caching is best effort.
| id | prediction (Claude) |
|---|---|
| 55h | haiku-v2 label agreement with haiku (v1) is at least 97% on T2 and at least 92% on T1: the added instructions do not change what it calls broken. |
| 55i | Cache reads are at least 50% of the batch's input tokens (batch caching is best effort). |
| 55j | A NOTE line is present on at least 80% of items labelled BROKEN and at most 25% of items labelled OK. |
| 55k | Of 30 sampled notes on BROKEN items, at least 80% name a breakage type that reading the text supports. |
| 55l | Agreement of haiku-v2 with the Opus reference is no more than 2 points below haiku (v1). |

## Result, five of six arms (scored 2026-10-22 01:21 CT; the Opus rerun batch msgbatch_01DD7oe9cy7giozwX6WsUVmo is still processing, rows 55c and 55e wait for it)
Agreement with the Opus reference labels (T1 = 432 origin answers, both labels must match; T2 = 1,792 broken-answer answers), calibration in brackets:
| arm | T1 agree | T2 agree | T1 cal | T2 cal | input tokens | note |
|---|---|---|---|---|---|---|
| haiku (disabled) | 87.0% | 99.6% | 22/22 | 20/20 | 774,075 | |
| haiku-low | 90.7% | 99.8% | 21/22 | 20/20 | 778,607 | output tokens 2.2x |
| sonnet (between_tools) | 92.6% | 98.1% | 22/22 | 20/20 | 774,075 | 33 of 34 T2 disagreements are false BROKEN; batch about $1.27 at list/2 |
| sonnet-low | 92.8% | 99.8% | 22/22 | 20/20 | 778,607 | batch about $1.33 |
| haiku-v2 (cached system block + NOTE) | 87.3% | 99.6% | 22/22 | 20/20 | 273,595 uncached + 1,783,170 cache reads | 86.6% of input from cache |
Where they disagree: on T1 most of Haiku's 56 misses are the BROKEN flag (Opus calls 21 more NEITHER answers and 8 more EARTH answers BROKEN than Haiku does) and EARTH versus NEITHER (11); the Opus reference (default thinking) is stricter on subtle grammar than any arm, so a stricter reference is a likelier cause than Haiku errors, but there is no truth set beyond the 42 calibration texts.
| id | result |
|---|---|
| 55a | HELD. T2 agreement Haiku 99.6% (needed 95), Sonnet 98.1% (needed 97). |
| 55b | HELD. T1 agreement Haiku 87.0% (needed 85), Sonnet 92.6% (needed 92). |
| 55c | PENDING the Opus rerun. Haiku and Sonnet: 22 of 22 and 20 of 20 both. |
| 55d | HELD, thinly. 35 of 56 Haiku T1 disagreements (62.5%) have NEITHER on one side, but 21 of them are NEITHER on both sides, the BROKEN flag only. |
| 55e | PENDING the Opus rerun. |
| 55f | PENDING Haiku's price: Sonnet costs about $1.27 for the same 774K input tokens at batch; Elise's reading of her balance is about 1.8B uncached Haiku input tokens for $89, i.e. about $0.05 per million at batch, which would make Haiku far below a fifth of Opus; to be confirmed from the Console. |
| 55g | FAILED narrowly. Low effort changes Haiku's T1 agreement by +3.7 points (87.0 to 90.7); the other three changes are +0.2, +0.2 and +1.7. |
| 55h | FAILED on T1: haiku-v2 labels agree with haiku (v1) on 88.2% of T1 (needed 92) and 99.4% of T2 (needed 97). The longer instructions plus the taxonomy move the origin call and mostly the BROKEN flag. |
| 55i | HELD. Cache reads are 86.6% of input tokens (1,783,170 of 2,057,471), 1,706 tokens written. |
| 55j | FAILED. NOTE lines are on 100% of BROKEN items (T1 58 of 58, T2 38 of 38) but also on 26% (T1) and 49% (T2) of OK items. Read: most notes on OK items restate the rubric ("cut off at the token limit, which is not a defect") or remark on invented names. |
| 55k | HELD. Of 30 sampled BROKEN notes, 28 name a breakage that reading the text supports (loops on "Star Trek author", "Earth"; "I is a large language model"; "donEarthedthemor..." non-words; a switch into Chinese). Two caveats: three notes echo the worked example ("matching the worked example pattern"), and one note says "Coherent" for an item labelled BROKEN. |
| 55l | HELD. haiku-v2 agrees with the reference as often as haiku v1: 87.3% against 87.0% (T1), 99.6% against 99.6% (T2). |
Reading: for the broken-answer flag every arm is within 2 points of Opus, so Haiku with its rubric is enough. For the three-way origin label Sonnet is about 5 points better than Haiku; low effort helps Haiku (+3.7) more than Sonnet. Sonnet with thinking off over-flags BROKEN (33 false). The NOTE line works for broken items (named breakage kinds) and needs a rule against boilerplate on OK items before it is left on everywhere; v3 should say "say nothing on an OK text unless something is unusual" and not quote a real item in the examples.
