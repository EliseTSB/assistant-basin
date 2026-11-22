# Part 219 numbers check (cold)

Checked `results/part219/README.md` against the raw files in `results/part219/` and the two defining scripts.

**Method.** `review/part219/check219.py` reimplements — from `part101/read101.py` (read set, relative show,
coherence, `inconclusive`) and `part219/score219.py` (upstream split, net against R0, exact McNemar) — and runs on
`R*.jsonl` + `labels-*.jsonl` + `labels2-*.jsonl` + `labels-deepseek-*.jsonl` + `*.meta.json` + `manifest.json`.
It does **not** read `scorecard.json`. After recomputing, I compared my numbers with `scorecard.json` and with
`SCORECARD.md` as a cross-check only; those comparisons are marked "scorecard" in the Source column. Full output:
`review/part219/check219.out.json`. Run with `~/neuron-probe/.venv/bin/python -I`.

Definitions confirmed from the scripts (they agree with the README's wording):

- Read set: unit ∈ {neuron, baseline}, question 0 for the 21 slot units, questions 0–5 for the 18 concept units.
- Relative show: pushed answer shows the persona **and** the same run's unpushed answer to the same question does not.
- Net against R0 is computed **on each run's own unpushed baseline** (not on R0's), paired per slot unit.
- Upstream = slot unit layer ≥ 3 (the deepest head layer): 16 units; downstream = layers 1–2: 5 units.
- McNemar: exact two-sided binomial on the discordant pair count, `min(1, 2 * Σ_{k≤min(lost,gained)} C(n,k)/2^n)`.

---

## 1. Shape of the read set

| # | README value | My value | Source | Match |
|---|---|---|---|---|
| 1 | 21 slot units | 21 | `R0.jsonl` ids × `results/part92/step2/kinds.json` | yes |
| 2 | 18 concept units (e.g. `17/18`, `14/18`) | 18 | same | yes |
| 3 | 16 upstream units ("whose push enters after all four heads") | 16 (min layer 3) | `neurons.json` + `kinds.json` | yes |
| 4 | 5 downstream units | 5: `p92-07` (layer 1), `p92-10`, `p92-20`, `p92-42`, `p92-50` (layer 2) | same | yes |
| 5 | "pushes at layers 1 and 2" (downstream) | layers 1 and 2 only | `neurons.json` | yes |
| 6 | "layers 0 to 3" (R1 swap) | `HEAD_LAYERS = (0,1,2,3)` | `score219.py:55` | yes |

## 2. Per-run counts ("All runs" table, and the per-run prose)

| # | README value | My value | Source | Match |
|---|---|---|---|---|
| 7 | R0 slot persona 20/21 | 20/21 | `labels-R0.jsonl` | yes |
| 8 | R0 slot coherent 20/21 | 20/21 | `labels-R0.jsonl` | yes |
| 9 | R0 concept persona 17/18 | 17/18 | `labels-R0.jsonl` | yes |
| 10 | R1 slot persona 4/21 | 4/21 | `labels-R1.jsonl` | yes |
| 11 | R1 slot coherent 16/21 | 16/21 | `labels-R1.jsonl` | yes |
| 12 | R1 concept persona 14/18 | 14/18 | `labels-R1.jsonl` | yes |
| 13 | R8 slot persona 13/21 | 13/21 | `labels-R8.jsonl` | yes |
| 14 | R8 slot coherent 16/21 | 16/21 | `labels-R8.jsonl` | yes |
| 15 | R8 concept persona 18/18 | 18/18 | `labels-R8.jsonl` | yes |
| 16 | R10 slot persona 17/21 | 17/21 | `labels-R10.jsonl` | yes |
| 17 | R10 slot coherent 18/21 | 18/21 | `labels-R10.jsonl` | yes |
| 18 | R10 concept persona 17/18 | 17/18 | `labels-R10.jsonl` | yes |
| 19 | R9 slot persona 17/21 | 17/21 | `labels-R9.jsonl` | yes |
| 20 | R9 slot coherent 20/21 | 20/21 | `labels-R9.jsonl` | yes |
| 21 | R9 concept persona 17/18 | 17/18 | `labels-R9.jsonl` | yes |
| 22 | R2 slot persona 1/21 | 1/21 | `labels-R2.jsonl` | yes |
| 23 | R2 slot coherent 2/21 | 2/21 | `labels-R2.jsonl` | yes |
| 24 | R2 concept persona 14/18 | 14/18 | `labels-R2.jsonl` | yes |
| 25 | R2m slot persona 1/21 | 1/21 | `labels-R2m.jsonl` | yes |
| 26 | R2m slot coherent 8/21 | 8/21 | `labels-R2m.jsonl` | yes |
| 27 | R2m concept persona 16/18 | 16/18 | `labels-R2m.jsonl` | yes |
| 28 | 219.2 result "2 of 21 coherent, 14 of 18 concept" | 2/21, 14/18 | `labels-R2.jsonl` | yes |
| 29 | 219.2m result "8 of 21 coherent, 16 of 18 concept" | 8/21, 16/18 | `labels-R2m.jsonl` | yes |

## 3. Nets against R0 on the 16 upstream units, and McNemar p

| # | README value | My value | Source | Match |
|---|---|---|---|---|
| 30 | R1 net 11 (11 lost, 0 gained) | lost 11, gained 0, net 11 | labels, paired 16/16 | yes |
| 31 | R1 McNemar p 0.00098 | 0.0009765625 (= 2/2^11) | exact binomial | yes |
| 32 | R8 net 3 (4 lost, 1 gained) | lost 4 (`p92-05,13,18,37`), gained 1 (`p92-32`), net 3 | labels | yes |
| 33 | R8 McNemar p 0.375 | 0.375 (= 2·(1+5)/32) | exact binomial | yes |
| 34 | R10 net 2 (2 lost, 0 gained) | lost 2 (`p92-05,36`), gained 0, net 2 | labels | yes |
| 35 | R10 McNemar p 0.5 | 0.5 (= 2/2^2) | exact binomial | yes |
| 36 | R9 net 1 (2 lost, 1 gained) | lost 2 (`p92-05,18`), gained 1 (`p92-32`), net 1 | labels | yes |
| 37 | R9 McNemar p 1.0 | 1.0 (2·(1+3)/8) | exact binomial | yes |
| 38 | R2 net 14 | lost 14, gained 0, net 14 | labels | yes |
| 39 | R2m net 14 | lost 14, gained 0, net 14 | labels | yes |
| 40 | R2 / R2m McNemar p 0.0001221 (scorecard/MD) | 0.0001220703125 (= 2/2^14) | exact binomial | yes |
| 41 | downstream: R8 lost 4 of 5 | lost 4 of 5 paired | labels, downstream set | yes |
| 42 | downstream: R9 2 | lost 2 of 5 | labels | yes |
| 43 | downstream: R10 1 | lost 1 of 5 | labels | yes |
| 44 | downstream: R1 5 | lost 5 of 5 | labels | yes |

## 4. Prediction bars, verdicts, registered text

| # | README value | My value | Source | Match |
|---|---|---|---|---|
| 45 | 219.1 confidence "medium, 55%" | medium, 55% | `PREDICTIONS.md:1764`, `PART219-DESIGN.md` Predictions | yes |
| 46 | 219.2 "medium, 50%" | medium, 50% | same | yes |
| 47 | 219.3 "low, 30%" | low, 30% | same | yes |
| 48 | 219.4 "low, 20%" | low, 20% | same | yes |
| 49 | 219.5 "low, 30%" | low, 30% | same | yes |
| 50 | 219.1m "low, 35%", written after the pilot (Amendment 1) | low, 35%, Amendment 1 | `PREDICTIONS.md`, design Amendment 1 | yes |
| 51 | 219.2m "low, 35%" (Amendment 1) | low, 35%, Amendment 1 | same | yes |
| 52 | 219.4's condition "at least R1's net loss minus 2" | `SWAP_MARGIN = 2` | `score219.py:64,472` | yes |
| 53 | 219.5's condition "at least 4 more than R10's" | `DISTANCE_MARGIN = 4` | `score219.py:63,463` | yes |
| 54 | 219.5 result "margin 1 (R8 net 3, R10 net 2)" | 3 − 2 = 1 | recomputed nets | yes |
| 55 | 219.2/219.2m bar "at least 15 of 21", "at least 13 concept" | `BAR_SLOT_COHERENT=15`, `BAR_CONCEPT=13` | `score219.py:58-59` | yes |
| 56 | net bar "at least 7" | `NET_MIN = 7` | `score219.py:60` | yes |
| 57 | 219.1 verdict "inconclusive: zeroing these heads breaks answers" | same string | `score219.py:363,425` | yes |
| 58 | 219.3 verdict failed; 219.4 "not read"; 219.5 failed | same | recomputed + `score219.py:468` | yes |
| 59 | 219.1m "not read" (R2m pilot rule fired) | same | `score219.py:511-519` | yes |

## 5. Gate and controls

| # | README value | My value | Source | Match |
|---|---|---|---|---|
| 60 | "the kernel check passed" | `part_check.kernel_check` returns no failures | ran `tools/part_check.py` on `kernel-check.json` vs `kernel-reference/...cpu-float32.json` | yes |
| 61 | R0 gate "20 of 21 (bar 17)" | 20 ≥ 17 | labels + `GATE_R0_MIN_SLOT=17` | yes |
| 62 | R1 gate "4 of 21 (bar 5)" | 4 ≤ 5 | labels + `GATE_R1_MAX_SLOT=5` | yes |
| 63 | "Part 100 had 3 on the default kernel" | `attnB_L0-3` slot_who = [3, 21] | `results/part100/score100.json` | yes |
| 64 | R9 "moved the 16 upstream units by a net 1" | net 1 | labels | yes |
| 65 | R9 "within its bar of 2" | `COUPLING_MAX_ABS_NET = 2`, \|1\| ≤ 2 | `score219.py:62` | yes |
| 66 | "R9 changed its layers' attention-output scale by at most 0.7%" | max \|rel. change\| = 0.674% (L1) | `R9.meta.json` `coupling_log` | yes |
| 67 | "R8 ... up to 12.7%" | 12.699% (L2, head 6) | `R8.meta.json` `coupling_log` | yes |
| 68 | "R10 ... up to 11.7%" | 11.733% (L2, head 6) | `R10.meta.json` `coupling_log` | yes |
| 69 | "moved the same four heads the opposite way by the same distance" | 24/24 records, `mode="opposite"`; max \|‖Δ‖_R8 − ‖Δ‖_R10\| = 9.3e-05 over all 24 slices | `R8/R10.meta.json` receipts | yes |
| 70 | "every transplanted slice equalled the donor's bytes (R8, R9)" | 24/24 `matches_expected: true`, `mode="donor"` | `R8/R9.meta.json` | yes |
| 71 | "or `2 x Instruct - donor` (R10)" | 24/24 `matches_expected: true`; `swap99.py:658` computes exactly `2*original − donor` in float32 then casts | `R10.meta.json`, `part99/swap99.py` | yes |
| 72 | "every run's reset replay matched R0 exactly" | all 6 changed runs: token ids identical, first-step logits bit-identical (max diff 0.0) | `*.meta.json` `reset_receipt` vs `R0` | yes |
| 73 | "two independent passes per run" | `labels2-<run>.jsonl` present for all 7 runs | files | yes |
| 74 | "agreed ... on at least 40 of 42 slot items in every run" | min over runs = 40 (R2); others 41–42 | `labels-*` vs `labels2-*` | yes |
| 75 | "slot counts within 2" | max \|pass1 − pass2\| = 1 (R1 4v5, R2m 1v0, R10 17v16) | same | yes |
| 76 | "the registered gate" (40/42, within 2) | `AGREE_MIN=40`, `AGREE_ITEMS=42`, `AGREE_SLOT_TOLERANCE=2` | `score219.py:65-67` | yes |
| 77 | DeepSeek "agreed ... in 92% to 96% of items per run" | 92.31% (R9) to 96.15% (R0,R1,R2,R2m,R8,R10) | `labels-deepseek-*` vs `labels-*` | yes |
| 78 | "fixed 10% sample" | 26 items per run of 258 = 10.1%; `reader` field `deepseek-v4.1-flash` | label files | yes |
| 79 | "Part 92's Haiku reader" | `reader` field `claude-haiku-5-5` on every non-error row | label files | yes |

## 6. The ablation half

| # | README value | My value | Source | Match |
|---|---|---|---|---|
| 80 | "Only 2 of 21 slot answers stayed coherent" (R2) | 2/21 | `labels-R2.jsonl` | yes |
| 81 | "3 of the 6 unpushed answers broke" (R2) | **3 by direct read** of the six `unit:"none"` answers: q0 `"I'm an AI, so I don"` (cut off), q1 `""`, q2 `""`; q3–q5 fine. **Scored `unpushed_incoherent` = 2** (q0's baseline majority stays coherent, 37/39). | `R2.jsonl` `unit:"none"` rows; `labels-R2.jsonl` | yes as a read, **but** see Finding 1 |
| 82 | "R2m left 8 of 21 coherent" | 8/21 | `labels-R2m.jsonl` | yes |
| 83 | "left all six of them ordinary" (R2m unpushed) | 6/6 unpushed answers coherent; `unpushed_incoherent` = 0 | `R2m.jsonl` `unit:"none"`, `labels-R2m.jsonl` | yes |
| 84 | "R3m to R7m did not run, and neither did R5 or R5m" | no `R3m…R7m`, `R5`, `R5m` files; manifest runs = R0,R2,R2m,R1,R8,R9,R10 | directory + `manifest.json` | yes |
| 85 | "the four heads (L0H10, L1H4, L2H6, L3H5)" | `registered_important_heads` = same; R8/R10 receipt heads = same | `manifest.json`, metas | yes |
| 86 | "four unimportant early heads" (R9) | L0H12, L1H11, L2H5, L3H9 — the design's frozen list, recomputed = registered | `manifest.json` `checks.head_lists` | yes |
| 87 | "the last 3 (one item in R9, one in R10 in both passes) were labelled at a 4,000-token cap" | **cannot be verified per row**: label rows carry only `reader`, no cap field; `label_ds219.py` default is `max_tokens=4000` with `500` for Haiku | label files, `part219/label_ds219.py:79,101` | **unverifiable** |
| 88 | "Retries at the same cap filled most of them" | every error key has a later success row: `items_still_missing = 0` for all 7 runs, both passes | label files | yes |

## 7. Cross-check against the published scorecard

| # | README value | My value | Source | Match |
|---|---|---|---|---|
| 89 | all 21 per-run counts (7 rows × 3) | identical to `scorecard.json` `runs` | scorecard, after recompute | yes |
| 90 | all 6 nets and p-values | identical to `scorecard.json` `nets_upstream_against_R0` | scorecard, after recompute | yes |
| 91 | downstream nets 4/2/1/5 | `scorecard.json` `exploratory.nets_against_R0.*.downstream` = 4/2/1/5 | scorecard | yes |
| 92 | headline "no headline: the predictions it rests on did not hold" (implied) | `scorecard.json` `headlines` = exactly that | scorecard | yes |
| 93 | verdicts 219.1–219.5, 219.1m, 219.2m | identical to `scorecard.json` `predictions` | scorecard, after recompute | yes |

---

## SUMMARY

Every count, net, p-value, percentage and range printed in `README.md` reproduces exactly from the raw files. I
recomputed all 21 per-run counts, all 6 upstream nets and 6 exact McNemar p-values, the 5 downstream nets, the
upstream/downstream split (16/5, layers 1–2), the two-pass reader agreement (min 40 of 42, slot counts within 1), the
DeepSeek agreement (92.31–96.15%), the three coupling maxima (0.674%, 12.699%, 11.733%), the R8/R10 distance identity
(9.3e-05), all reset replays (bit-identical logits) and all transplant receipts (24/24 each). The gate passes and
`Part 100`'s 3 of 21 is confirmed from `results/part100/score100.json`. Registered prediction text and confidences
match `PREDICTIONS.md` line for line. **No numeric mismatch was found.** One number is ambiguous in what it counts, and
one is not verifiable from the retained files (Findings 1 and 2).

## FINDINGS

1. **`3 of the 6 unpushed answers broke` (R2) is right by eye but is not the scored metric — and the README's
   sentence credits it with firing the stopping rule.** Direct read of the six unpushed answers in `R2.jsonl`
   (`unit:"none"`): q0 `"I'm an AI, so I don"` (cut off), q1 empty, q2 empty, q3–q5 fine → 3 broken, matching
   `PILOT.md` ("3 of 6 fine, 2 empty, 1 cut off") and Amendment 1 ("3 of its 6 unpushed answers are empty or cut
   off"). But the registered field `unpushed_incoherent` (read101's per-question majority over all baseline rows),
   which is what `inconclusive` is computed from, is **2**, because q0's baseline majority is still coherent (37 of 39
   rows labelled coherent on an identical answer string — also a sign of reader noise on a borderline truncated
   answer). The registered stopping rule fires on R2 leaving **2 of 21** slot answers coherent (bar 15), not on the
   unpushed count; at 3 the `inconclusive` flag would still be False (threshold is > 3). *Suggested fix, optional:*
   "and its unpushed answers were partly broken too (3 of the 6, read directly; the scored `unpushed_incoherent` is
   2)". As written, a reader can take "3 of 6 broke" for the gating metric and conclude the flag was set; it was not.
2. **`the last 3 ... were labelled at a 4,000-token cap` is not checkable from `results/part219/`.** Label rows carry
   `reader` only — no cap, no route, no retry marker. `part219/label_ds219.py` defaults to `max_tokens=4000` (help
   text: "500 for Haiku"), so the claim is plausible, but nothing in the retained data records which rows used which
   cap. The residual error rows are consistent with it (R9: 2 distinct error keys in each pass; R10: 4 in each pass;
   every key later filled), but they are a superset of "the last 3" and cannot pin it down. Not a mismatch — an
   unverifiable sentence.
3. **Same shape, lower stakes:** `the rerun's rows were byte-identical` (R2m, Amendment 1b) cannot be checked —
   `stopped-r2m-receipt/` keeps the stopped run's metas and `manifest-after-stop.json` but not its `R2m.jsonl`, and no
   row hash is recorded. `PILOT-A1.md` cites a `cmp`. Non-numeric; noted only because the README presents it as a
   receipt.
4. **Naming, not a number:** the README's "upstream" (16) / "downstream" (5) follow `score219.py`'s variable names,
   while `PART219-DESIGN.md` says the 16 "sit downstream of all four heads". The README's gloss ("whose push enters
   after all four heads", "where at least one of the four heads comes after the push") is correct and unambiguous, so
   this is a readability note only.

## BLOCKERS

None. Both items in FINDINGS are wording/verifiability issues, not wrong numbers; nothing blocked the check.
