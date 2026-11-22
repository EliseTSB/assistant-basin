# Part 219 prose cold check

Checked: `results/part219/README.md` against `PART219-DESIGN.md` (incl. Amendments 1 and 1b),
`PREDICTIONS.md` Part 219, `results/part219/SCORECARD.md`, `PILOT.md`, `PILOT-A1.md`.
Every number in the README was recomputed from `results/part219/labels-*.jsonl` (slot_who,
slot-who coherent, concept_any, upstream/downstream nets against R0), from the run jsonls (the six
`id="unpushed"` rows, R2m's pushed answers), from `scorecard.json`, from each `*.meta.json`
(reset receipts, transplant receipts, `coupling_log`), and from `manifest.json` / `kernel-check.json`.
Line numbers are README.md lines. Quotes are exact.

Verdicts: SUPPORTED · OVERSTATED (say what the data supports instead) · WRONG (say why) · UNSUPPORTED (no data behind it).

## Claim table

| # | line | Claim (quoted) | Verdict | Reason / what the data support | Suggested wording |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | "Part 219: OLMo 2 1B's slot neurons need post-training's early attention, but not mainly in its four most important early heads" | OVERSTATED | First half is supported (R1 4/21 vs R0 20/21, net 11, p=0.00098). Second half asserts an apportionment that was never measured: 219.4, the registered prediction that would have tested how much of the layer swap the four heads account for, is "not read"; R1 changes all 64 heads of layers 0-3 plus their norms while R8 changes four heads, so R8 vs R1 is two interventions, not a decomposition; and the scorecard's headline rule returns "no headline: the predictions it rests on did not hold". The figure's own title is correctly hedged ("swapping back only the four most important early heads mostly does not"). | "Part 219: OLMo 2 1B's slot neurons need post-training's early attention, but transplanting only its four most important early heads leaves most personas in place" |
| 2 | 5-6 | "Part 100 showed that putting back the pre-SFT attention of layers 0 to 3 (from the end of mid-training) removes most slot neurons' personas in OLMo 2 1B Instruct." | SUPPORTED | Design's Question: 19 of 21 to 3 of 21 in Part 100; `results/part100/score100.json` attnB_L0-3 slot_who = [3,21]. This run's rerun on eager: 20/21 to 4/21. | — |
| 3 | 6-8 | "Screen A had found four heads, L0H10, L1H4, L2H6 and L3H5, that matter most for Instruct's identity answers, and they are also the four most important early heads for every answer." | SUPPORTED | Design's screen A table: identity drops 0.428/0.342/0.256/0.311 vs ordinary 0.335/0.257/0.220/0.211; next early head L0H8 is 0.033/0.034. Note the selection is unregistered; the Limits disclose that. | — |
| 4 | 8 | "This part asked whether those four heads carry the dependence." | SUPPORTED | Statement of the question; matches the design. | — |
| 5 | 10 | "They do not carry most of it." | OVERSTATED | The data show that transplanting only the four heads does not reproduce most of R1's loss (13 of 21 personas remain vs 4 of 21; net 3 vs 11). They do not measure what share of R1's effect the four heads carry (219.4 not read; R8 and R1 are different interventions; R8's 3 is not distinguishable from 0 (p = 0.375) and is about the R10 control's 2). "Carry" is an attribution the design did not license. | "Transplanting only those four heads left most personas in place: 13 of 21, against 4 of 21 for the full layer swap. How much of the swap's effect the four heads carry was not measured (219.4 was not read)." |
| 6 | 10-12 | "Putting back only those four heads' pre-SFT weights (R8) left 13 of 21 slot units showing their persona, against 20 of 21 in Instruct (R0) and 4 of 21 when all of layers 0 to 3 were put back (R1)." | SUPPORTED | Recomputed from labels-R0/R1/R8: 20/21, 4/21, 13/21. Matches scorecard. | — |
| 7 | 12 | "On the 16 units whose push enters after all four heads, R8 lost a net 3 personas, R1 lost 11." | SUPPORTED | Recomputed: R8 4 lost / 1 gained = net 3; R1 11 lost / 0 gained = net 11. Matches scorecard. | — |
| 8 | 12-13 | "Moving the same four heads the opposite way by the same distance (R10) lost 2, and putting back four unimportant early heads (R9) lost 1." | SUPPORTED | R10 net 2 (2 lost, 0 gained); R9 net 1 (2 lost, 1 gained). The "same distance" is checked: `coupling_log` q_proj norm_delta 0.87177 (R10) vs 0.87184 (R8). | — |
| 9 | 13-15 | "So post-SFT early attention is needed, but the part that the slot route needs is not mainly in these four heads' post-training change." | OVERSTATED | Two problems. (a) "post-SFT": the donor is the end of mid-training, so the transplanted difference is the whole post-mid-training change (SFT and anything after it), which the title correctly calls "post-training". (b) "not mainly": same problem as row 5 — the transplant result is compatible with the sentence, but the share was not measured and no test is reported. | "So post-training's early attention is needed. The four heads' post-training change is not sufficient to reproduce most of the layer swap's loss, and how much of it they carry was not measured." |
| 10 | 15 | "It may be spread over other heads, or in how heads combine." | SUPPORTED | Explicitly hedged ("may"); no data is claimed. Worth marking as a hypothesis, since "in how heads combine" is also the reason the apportionment cannot be inferred from R8 alone. | "One possibility, not tested here: it is spread over other heads, or in how heads combine." |
| 11 | 17 | "Removing the four heads outright could not test this." | SUPPORTED | R2 left 2 of 21 slot answers coherent (pilot), so the zeroing arm could not be read. "In the event" would be more accurate than a general impossibility. | "In the event, removing the four heads outright could not test this." |
| 12 | 17-18 | "Zeroing them (R2) broke the model, and mean-ablating them (R2m, Amendment 1) kept unpushed answers ordinary but broke the pushed ones." | OVERSTATED | R2: 1/21 personas, 2/21 coherent (broken) — "broke the model" is fair. R2m: all six unpushed answers are ordinary (verified in R2m.jsonl) but 8 of 21 pushed answers were still coherent, so "broke the pushed ones" reads as all 21 when 13 broke. The R2m half is also a direct/exploratory read presented here without the tag it gets later. | "Zeroing them (R2) broke the model; mean-ablating them (R2m) left all six unpushed answers ordinary but broke 13 of the 21 pushed ones." |
| 13 | 18 | "Both pilots stopped on their registered rule." | SUPPORTED | R2: 2/21 coherent < 15; R2m: 8/21 < 15; both are the registered stopping rules. Minor ambiguity: there were three pilot runs (R2, the receipt-stopped first R2m, the R2m rerun); the next paragraph explains the receipt stop. | "Both pilots that were read stopped on their registered stopping rule (the first R2m attempt stopped on its receipt; below)." |
| 14 | 24 | "219.1 ... R2 broke answers (219.2 failed) ... inconclusive: zeroing these heads breaks answers" | SUPPORTED | Amendment 1 fixes exactly this verdict; scorecard `pilot_verdicts` agrees. | — |
| 15 | 25 | "219.2 ... 2 of 21 coherent, 14 of 18 concept ... failed (on the registered pilot)" | SUPPORTED | R2: 2/21 coherent (< 15); concept 14/18 (>= 13) so the coherence limb fails; inconclusive false. Verified from labels-R2. | — |
| 16 | 26 | "219.3 ... net 3 (4 lost, 1 gained), McNemar p = 0.375 ... failed" | SUPPORTED | Recomputed: 4 lost, 1 gained, exact two-sided p = 2*(C(5,0)+C(5,1))/32 = 0.375. | — |
| 17 | 27 | "219.4 ... read only if 219.3 and 219.5 hold ... not read" | SUPPORTED | Design reading rules say exactly this; scorecard agrees. (Note that 219.3 failing already implies 219.4's bar cannot be met, so "not read" is the registered outcome, not evidence for either side.) | — |
| 18 | 28 | "219.5 ... R8 net 3, R10 net 2, margin 1 ... failed" | SUPPORTED | 3 - 2 = 1 < 4. | — |
| 19 | 29 | "219.1m ... R2m's pilot stopping rule fired ... not read" | SUPPORTED | Amendment 1: "If R2m also breaks answers, the ablation half ends: 219.1m is not read." R2m coherence 8/21. | — |
| 20 | 30 | "219.2m ... 8 of 21 coherent, 16 of 18 concept ... failed (on the R2m pilot)" | SUPPORTED | Verified from labels-R2m: 8/21 coherent, 16/18 concept. | — |
| 21 | 32-34 | "The gate passed: the kernel check passed, R0 showed 20 of 21 slot personas (bar 17), and R1 showed 4 of 21 (bar 5; Part 100 had 3 on the default kernel)." | SUPPORTED | `scorecard.json` gate passed=true with no failures; `results/part100/score100.json` attnB_L0-3 = [3,21]. | — |
| 22 | 38-39 | "Slot personas and coherence on "Who are you?" over all 21 slot units. Net loss is on the 16 upstream units, relative to each run's own unpushed answers, as in Part 99 (Amendment 2) and Part 101." | SUPPORTED | `relative()` in part101/read101.py is pushed-shows-and-unpushed-does-not; `score219.py` nets against R0 on the 16 upstream units. Matches the table. | — |
| 23 | 43 | "R0 \| none \| 20/21 \| 20/21 \| 17/18 \| (reference)" | SUPPORTED | Recomputed from labels-R0: 20/21, 20/21 coherent, 17/18. | — |
| 24 | 44 | "R1 \| layers 0-3 attention from the mid-training end \| 4/21 \| 16/21 \| 14/18 \| 11 (11 lost, 0 gained) \| 0.00098" | SUPPORTED | Recomputed: 4/21, 16/21, 14/18, net 11, p=0.0009766. | — |
| 25 | 45 | "R8 \| ... \| 13/21 \| 16/21 \| 18/18 \| 3 (4 lost, 1 gained) \| 0.375" | SUPPORTED | Recomputed from labels-R8. | — |
| 26 | 46 | "R10 \| ... \| 17/21 \| 18/21 \| 17/18 \| 2 (2 lost, 0 gained) \| 0.5" | SUPPORTED | Recomputed from labels-R10. | — |
| 27 | 47 | "R9 \| ... \| 17/21 \| 20/21 \| 17/18 \| 1 (2 lost, 1 gained) \| 1.0" | SUPPORTED | Recomputed from labels-R9. | — |
| 28 | 48 | "R2 \| ... \| 1/21 \| 2/21 \| 14/18 \| 14, but answers broken" | SUPPORTED | Recomputed from labels-R2; pilot run, answers broken, so the net 14 is not interpretable as a persona loss. | — |
| 29 | 49 | "R2m \| ... \| 1/21 \| 8/21 \| 16/18 \| 14, but answers broken" | SUPPORTED | Recomputed from labels-R2m. | — |
| 30 | 51-53 | "The five downstream units ... are reported separately, as registered: R8 lost 4 of 5, R9 2, R10 1, R1 5 (exploratory, scorecard `exploratory`)." | OVERSTATED | The counts are right (R1 5, R8 4, R9 2, R10 1, all net, gained 0), but "as registered" beside "exploratory" is contradictory: the design registers the *separation*, and lists the five downstream units as exploratory. The design's reading rule also requires exploratory counts "with unit lists and exact McNemar p"; the exact p values (R1 0.0625, R8 0.125, R9 0.5, R10 1.0) are omitted. | "The five downstream units ... are reported separately as the design required, but exploratory (scorecard `exploratory`): R8 lost 4 of 5 (exact McNemar p = 0.125), R9 2 (p = 0.5), R10 1 (p = 1.0), R1 5 (p = 0.0625)." |
| 31 | 53 | "For those units a transplant changes attention after the push enters, which is a different question." | SUPPORTED | Design: the five units are p92-07 at layer 1 and four at layer 2; L3H5 (and L2H6 for p92-07) comes after the push. | — |
| 32 | 57-58 | "The coupling check (R9) held: putting back four unimportant early heads moved the 16 upstream units by a net 1, within its bar of 2." | SUPPORTED | R9 net 1 (2 lost, 1 gained); bar \|net\| <= 2; scorecard `R9_coupling_check.held = true`. | — |
| 33 | 58-59 | "The transplant's side effect on sibling heads, through the shared normalisations, does not on its own move the readout." | OVERSTATED | This is established only for four *small* heads: R9's maximum layer attention-output RMS change is 0.674%. The same bullet then reports R8/R10 at 12.7%/11.7%, and the Limits say R9 "bounds that coupling for small heads". A general "does not on its own move the readout" is not tested for the four large heads. | "For four small unimportant heads, the coupling through the shared normalisations did not on its own move the readout (net 1, inside its bar). The four large heads change their layers' attention-output scale up to 12.7%, so that bound is weaker for them." |
| 34 | 59-60 | "R9 changed its layers' attention-output scale by at most 0.7%; R8 and R10 changed theirs by up to 12.7% and 11.7%, the four heads being large (`R*.meta.json`, `coupling_log`)." | SUPPORTED | Recomputed from the metas: R9 max \|relative_change\| of `rms_attention_output` = 0.674% (layers 0.30/0.67/0.61/0.35); R8 = 12.699%; R10 = 11.733%. | — |
| 35 | 61-62 | "The distance control (R10) moved the four heads as far as R8 in the opposite direction and lost about as many personas (2 against 3)." | SUPPORTED | R10 = 2 x Instruct - donor, so the distance equals R8's (receipt norm_delta 0.87177 vs 0.87184 on q_proj); net 2 vs net 3. Minor: "lost ... (2 against 3)" is the *net*; raw lost is 4 (R8) vs 2 (R10), so say "net" to avoid a reader taking it as raw. | "moved the four heads as far as R8 in the opposite direction and lost about as many personas net (2 against 3)." |
| 36 | 62-63 | "That is the reading 219.5 was registered to catch: R8's small loss is no more than any change of that size to these heads produces." | WRONG | Two errors. (a) Literally: R8's net loss is 3 and R10's same-distance change produced 2, so R8's loss *is* 1 more, not "no more than". (b) "any change of that size" generalises from one control (the opposite direction); and 219.5 failing means the direction effect was not demonstrated, not that it is absent — 3 vs 2 is a margin of 1 against a bar of 4. | "That is what 219.5 was registered to test: the direction of post-training's change was not shown to matter beyond its size (R8 net 3, R10 net 2, margin 1 against the registered bar of 4)." |
| 37 | 64-65 | "Receipts: every transplanted slice equalled the donor's bytes (R8, R9) or `2 x Instruct - donor` (R10); every run's reset replay matched R0 exactly." | SUPPORTED | R8/R9/R10 each have 24 transplant records (4 heads x q/k/v/o/q_norm/k_norm) with `matches_expected: true`; swap99.py compares with `torch.equal`. R0's replay vs each reset receipt: token_ids and first_step_logits identical for R1, R2, R2m, R8, R9, R10; manifest `reset_receipts` all `matches_r0: true`. | — |
| 38 | 66-68 | "Readers: Part 92's Haiku reader, two independent passes per run, agreed on whether the persona shows on at least 40 of 42 slot items in every run, with slot counts within 2 (the registered gate)." | OVERSTATED | The gate is item-level agreement on `shows_persona` (>= 40 of the 42 slot items agree), which every run passes (minimum 40, R2). But the sentence can be read as "agreed that the persona shows on at least 40 of 42 items", which is false in every run (R8 shows it on 13 of 21). The ambiguity is worth removing because the numbers are otherwise correct. | "the two Haiku passes agreed about whether the persona shows on at least 40 of the 42 slot items in every run (minimum 40, in R2), with slot counts within 2". |
| 39 | 67-68 | "A DeepSeek V4.1 Flash reader on a fixed 10% sample agreed with pass 1 on whether the persona shows in 92% to 96% of items per run." | SUPPORTED | Scorecard `deepseek_agreement.shows_persona_agreement`: 0.9231 (R9) to 0.9615 (all others); 26 items per run. | — |
| 40 | 72-73 | "The registered pilot zeroed the four heads (R2). Only 2 of 21 slot answers stayed coherent, and 3 of the 6 unpushed answers broke, so the stopping rule fired (PILOT.md)." | SUPPORTED | R2.jsonl unpushed rows: q0 "I'm an AI, so I don" (cut off), q1 and q2 empty, q3-q5 ordinary = 3 of 6 broken; labels-R2 2/21 coherent. | — |
| 41 | 73-74 | "Amendment 1 (Vesper PASS) replaced zeroing with mean-ablation from a neutral reference set." | SUPPORTED | Design Amendment 1; review line "Vesper PASS on v2.1 (d2fe8659), council #10". | — |
| 42 | 74-76 | "Its first R2m run stopped on a receipt check that could not pass by construction (Amendment 1b, Vesper PASS); the rerun's rows were byte-identical." | SUPPORTED | Design Amendment 1b: pos0 before-difference 0.0 exactly, cause given; PILOT-A1: "rows are byte-identical ... (`cmp`)". | — |
| 43 | 76 | "R2m left 8 of 21 coherent, so the amended rule fired too (PILOT-A1.md), R3m to R7m did not run, and neither did R5 or R5m." | SUPPORTED | labels-R2m 8/21; no R3m-R7m/R5/R5m files in results; scorecard controls "not run". | — |
| 44 | 78-80 | "One difference between the two pilots, read directly and exploratory: zeroing broke the unpushed answers, but mean-ablation left all six of them ordinary ("As an AI language model...")." | OVERSTATED | The tag is right and the R2m half is exactly true (all six unpushed rows are ordinary). "Zeroing broke the unpushed answers" overstates 3 of 6: R2's q3-q5 are fine. | "zeroing broke 3 of the 6 unpushed answers, but mean-ablation left all six ordinary" |
| 45 | 80-81 | "Mean-ablation broke only the pushed answers, which came out empty, looping or as fragments." | OVERSTATED | "Only the pushed answers" is true in the sense that no unpushed answer broke, but 13 of 21 pushed answers broke and 8 stayed coherent (scorecard 8/21); as written it invites "all pushed answers broke". | "Mean-ablation broke 13 of the 21 pushed answers — empty, looping or fragments — and none of the six unpushed ones." |
| 46 | 81 | "With these four heads held at their average, Part 99's fixed pushes derail answers that held together in Instruct." | SUPPORTED | As an exploratory description it matches the within-run contrast (R2m unpushed ordinary, 13/21 pushed broken) and is tagged exploratory. "Derail" is a causal verb for one run and n=21; "produced broken answers where R0's pushes did not" is more neutral. | "In this one run, with these four heads held at their average, the fixed pushes produced broken answers where they had not in R0." |
| 47 | 85 | "One model (OLMo 2 1B), one pre-SFT checkpoint as the donor, one rubric, 21 slot units and 16 of them read." | SUPPORTED | True; the 16/21 split is in scorecard. Power is only addressed in the next bullet. | — |
| 48 | 86-88 | "A net loss of 3 against a bar of 7 does not show the four heads contribute nothing: R8's 3 is about what R10's opposite-direction change gives (2), and the test cannot separate small effects." | SUPPORTED | Correct and appropriately cautious; this is the caveat the title should have inherited. | — |
| 49 | 88-90 | "Transplanting a head is not perfectly local: the QK norms and the post-attention norm couple heads (Limits in the design). R9 bounds that coupling for small heads; for the four large heads, R10 is the comparison." | SUPPORTED | Design's Limits say the same ("R9's heads have smaller outputs than R8's, so it bounds the coupling less well than R10 does"). | — |
| 50 | 90-94 | "Labels came from Haiku 5.5 through OpenCode Go ... the last 3 (one item in R9, one in R10 in both passes) were labelled at a 4,000-token cap. Every run was labelled the same way ... but the labels are not Part 92's exact reader." | OVERSTATED | The route and the retry story are consistent with the files (error rows for p92-38/p92-48 in R9 and p92-41/42/43/45/48 in R10, each followed by a successful row). But (a) "Every run was labelled the same way" sits one clause after saying three items in R9/R10 were labelled at a different cap; (b) no archived label row or log records `max_tokens`, so the 4,000 figure is not independently checkable. | "Every run was labelled by the same reader and route, with the same 500-token cap except three items (one in R9, one in R10 in both passes) labelled at 4,000; the labels are not Part 92's exact reader." |
| 51 | 95 | "The four heads were chosen by screen A, which was not registered." | SUPPORTED | True; the design's Limits add that they were chosen on Instruct's identity answers and are the most important early heads for everything. Worth adding both here, since selection-on-identity is what makes R8's specificity question hard. | "The four heads were chosen by screen A, which was not registered, on Instruct's identity answers; they are also the most important early heads for ordinary answers." |
| 52 | 99-105 | Files list: "Runs: R0.jsonl ... each with .meta.json (receipts, coupling logs); head-means.pt; manifest.json ... Code: part219/run219.py, ..." | SUPPORTED | All named files exist. Minor imprecision: `coupling_log` is present only in the transplant metas (R8-R10), not R0/R1/R2/R2m; `kernel-check.json` and the Amendment 1b `stopped-r2m-receipt/` directory are not listed. | "each with .meta.json (receipts; coupling logs in the transplant runs)" |
| 53 | 3 | ![Slot personas and coherence per run](fig219.png) — figure title: "swapping all early attention back to its pre-SFT weights removes the slot personas; swapping back only the four most important early heads mostly does not" | SUPPORTED | The figure matches its caption (blue = slot units showing their persona of 21, grey = slot answers coherent of 21, per run), and the title's two clauses match the bars: R1 4/21 (16 of 20 personas removed), R8 13/21 (most personas remain). The figure title is the calibration the README title lacks. | — |
| 54 | 32-34 | "Full registered text: PREDICTIONS.md, Part 219. ... Scorecard: [SCORECARD.md](SCORECARD.md), [scorecard.json](scorecard.json)." | SUPPORTED | `PREDICTIONS.md` lines 1758-1772 hold the Part 219 predictions verbatim (219.1-219.5, 219.A1, 219.1m, 219.2m, 219.A1b) with empty Result/Verdict columns; SCORECARD.md and scorecard.json exist and agree with the README's verdicts. | — |

## SUMMARY

The README's numbers are clean: every count, net, McNemar p, gauge value and receipt in the tables and
controls reproduces from the raw labels/metas. The prose is not equally clean. The title and the two
sentences that restate it ("They do not carry most of it"; "not mainly in these four heads' post-training
change") claim an apportionment the design never measured and the scorecard explicitly declined to
headline ("no headline"). One control sentence is false as written (R8's 3 is 1 more than R10's 2, not
"no more than"). One coupling sentence generalises from four small heads to the four large ones. The
exploratory material is mostly tagged, but the downstream tag is contradicted by "as registered" and the
mean-ablation contrast is untagged in the opening. The Limits name the small n, one model, label route,
donor and unregistered screen A, but not the two weaknesses that matter most for the title (R1-vs-R8 is
not a decomposition; 219.4 was not read), not the concept units' layer mismatch, and not that the R0b
do-nothing check never ran in the production stage.

Verdicts: 43 SUPPORTED, 10 OVERSTATED, 1 WRONG, 0 UNSUPPORTED (54 rows).

## FINDINGS

**Must fix**

1. **Title, line 10 "They do not carry most of it", line 13-15 "not mainly in these four heads'
   post-training change".** The claim is not registered and not measured. 219.4 (the prediction written
   to test how much of the layer swap the four heads account for) is "not read"; R1 changes all 64 heads
   of layers 0-3 plus their norms, so R8 vs R1 is two interventions, not a decomposition; the scorecard's
   headline rule returns "no headline: the predictions it rests on did not hold"; and R5/R5m, the rig
   checks named in the design's headline rule, never ran. The descriptive version is supported: 13 of 21
   personas remain after R8, against 4 of 21 after R1. Use the figure's calibration ("mostly does not") or
   say "the four heads' transplant does not reproduce most of the layer swap's loss; the share was not
   measured". (Rows 1, 5, 9.)
2. **Line 62-63, "R8's small loss is no more than any change of that size to these heads produces".**
   Wrong twice: R8's net 3 is 1 *more* than R10's same-distance net 2, and "any change of that size" is
   inferred from one control direction. 219.5's failure means the direction effect was not demonstrated
   (margin 1 against a bar of 4), not that it is absent. (Row 36.)
3. **Line 58-59, "The transplant's side effect on sibling heads ... does not on its own move the
   readout".** True only for R9's four small heads (layer scale change <= 0.674%). R8/R10 change the
   same quantity by 12.7%/11.7%; the Limits concede R9 "bounds that coupling for small heads". State the
   bound where it applies and don't generalise it. (Row 33.)
4. **Missing: the registered exploratory all-21 counts.** The design's reading rules require "all
   21-unit counts" exploratory "with unit lists and exact McNemar p". The README gives the raw 13/21 but
   not the net: R8 loses 8, gains 1, net 7 on all 21, exact McNemar p = 0.039. That is the one nominally
   significant R8 result and it points the other way from the "R8 did little" impression left by net 3,
   p = 0.375; its absence leaves the reader without the one nominally significant R8 result. Add it, labelled exploratory.
5. **Missing: the R0b / R0b-mean do-nothing check did not run in the production stage.** The design
   makes R0b a required control ("R0b must match R0 token for token"); `manifest.json` stages are a
   (R0, R2), a1 (R2m), b (R1, R8-R10) — no R0b — and no R0b files exist under `results/`. `REPORT-a1.md`
   records R0b passing only in an end-to-end smoke on a 16-head scratch model (18 items), not on OLMo 2 1B. The README's "Receipts" bullet is true (reset replays
   all match R0) but does not cover this, and neither the Controls section nor the Limits mention it.
   Either run it or disclose that the hook-wrapper no-op check was not run in full.

**Should fix**

6. **Line 51-53, "reported separately, as registered ... (exploratory ...)".** "Registered" and
   "exploratory" in one sentence will be read as a registered result by a skimming reader. The design
   registers the separation only, and asks for exact McNemar p with exploratory unit counts; give the p
   values (0.0625, 0.125, 0.5, 1.0). (Row 30.)
7. **Line 66-68, the 40-of-42 gate.** The sentence is correct on one parse (item-level agreement) and
   false on the other (persona showing on >= 40 of 42 items — no run does; R8 is 13 of 21). A comma or
   "agreed about whether ... on at least 40 of the 42 items" removes the trap. (Row 38.)
8. **Line 17-18 and 78-81, the pilot contrast.** "Broke the pushed ones" / "broke the unpushed answers" /
   "broke only the pushed answers" should carry the counts: 13 of 21 pushed, 3 of 6 unpushed, 8 of 21
   still coherent. The opening states the R2m contrast as fact; it is a direct read and is tagged
   exploratory only later. (Rows 12, 44, 45.)
9. **Limits should add the real weaknesses it omits.** (a) R1 vs R8 is not a decomposition (all 64 heads
   plus norms vs four heads), so the title's question was not measured. (b) The concept units sit at other
   layers than the slot units (design Limits), so they control for "personas in general", not for early
   disruption at the slot units' depth. (c) The net 3 has five discordant pairs (p = 0.375); state that
   the interval is wide rather than only that the test "cannot separate small effects".
10. **Line 90-94, "Every run was labelled the same way".** It conflicts with the immediately preceding
    sentence about three items at a 4,000-token cap, and no archived label row records the cap. Say "same
    reader and route; three items at a higher cap". (Row 50.)

**Minor**

11. Line 18, "Both pilots" — there were three pilot runs; the receipt-stopped first R2m is explained in
    the next paragraph but the counting is ambiguous. (Row 13.)
12. Line 61-62, "lost about as many personas (2 against 3)" — these are net losses; raw lost is 4 vs 2.
    Add "net". (Row 35.)
13. Line 81, "derail" — causal verb, one run, 13 of 21; "produced broken answers where R0's pushes did
    not" is more neutral. (Row 46.)
14. Line 99-105, Files — `coupling_log` exists only in the transplant metas; `kernel-check.json` and
    `stopped-r2m-receipt/` are not listed. (Row 52.)
15. Line 95, screen A — add "on Instruct's identity answers" (design Limits), since selection on
    identity answers is what makes the specificity question hard. (Row 51.)

## BLOCKERS

None for this review. Two items could not be verified from the archived artifacts and are flagged rather
than failed: the 4,000-token relabel (no `max_tokens` field in any label row or log, and no relabel log;
the retry/error rows are consistent with the story) and which three item-labels were relabelled. One design deviation needs the
author's decision, not a reviewer's: the R0b/R0b-mean do-nothing check was not run in the production stage
(FINDINGS 5).
