# Reader correction: the atlas parts rescored on the complete version-2 (DS) confirmed subsets

Written 2026-11-03 by Claude (a Claude Code cloud session, CPU only, stored labels only; no model ran and no reader
was called). Follows item 1.4 of `review/REVIEW-2026-11-03-whole-repo.md`. Everything here is produced by
`tools/atlas_v2_units.py` and `tools/reader_correction.py`; `report.json` holds every number, and each part's
subfolder holds the rescored `score.json` (and `SCORE.md` or `tables.md` where the scorer writes one).

## What was done, and what could not be

Part 86's first-reader atlas has 465 units (387 "another AI", 78 other kinds), labelled by Claude Haiku 5.5. Under
brief version 2, DS V4.1 Flash kept 245 of the 387 "another AI" units and 77 of the 78 other-kind units as coherent
personas (`results/part86/atlas-v2/ATLAS-v2.md`). The version-2 labels are keyed by blind item ids made with
`secrets.token_urlsafe`; the id-to-screen-row key is private, and the public items file (id, text) lives in the J-lens
study checkout (`$JLENS_ROOT/data/qwen3.5-4b/labels/items.jsonl`), outside this repository. The usage logs carry no
text, so this repository alone cannot reconstruct the 3,349-item source join. The join described below is the
atlas-unit join.

Every atlas unit's screen answer is unique among the 165,376 screen rows, so an item's text identifies its unit exactly.
`tools/atlas_v2_units.py` joins labels to units through item text. On 2026-11-03 Claude ran
`python3 tools/atlas_v2_units.py --items ~/jlens-persona-neurons/data/qwen3.5-4b/labels/items.jsonl` on the machine
with the J-lens items file. The join is complete: 465 of 465 units, coverage 1.0. The counts match `ATLAS-v2.md`
(245, 141, 1 and 77, 1, 0). Claude then ran `python3 tools/reader_correction.py` on that machine. For Parts 112,
117, 117b, 148 and 152, each full-atlas rerun's `score.json` is identical to the stored
`results/partNNN/score.json` apart from generated-time fields.

Part 142 joined the same labels by casefolded persona-string equality and reached 43 units; that join is not used here
because two readers' free-text persona strings are not a key.

Three unit universes are scored: **full** (465 units; this reproduces the stored verdicts for every part below),
**v2_all** (DS-confirmed, any kind: 322), and **v2_another_ai** (the "another AI" members: 245). Registered bars
written as absolute counts are read literally and, when marked exploratory, scaled to the surviving denominator.

| Part | Starts from the atlas? | Rescored here |
| --- | --- | --- |
| 112 | yes, all 465 units | yes |
| 115 | no: candidates are the top 20 of an attribution ranking over all neurons (`PART115-DESIGN.md`); the Part 86 README's sentence that Part 115 used the atlas as its starting list is wrong on this point | not applicable |
| 117, 117b | yes, 11 slot/concept pairs from Part 112's sort | yes; 10 pairs survive on v2_all, none on v2_another_ai |
| 118 | no: OLMo 2 1B, Part 92's 39 units | its analogue, Part 92b's DS re-sort, is rerun instead |
| 148 | yes, all 465 units | yes |
| 152 | the persona-word predictions 152.3 and 152.4 use Part 112's sort | yes |

## Part 112: the concept/slot sort (stored labels, `part112/score112.py`)

| Prediction | Full atlas (465) | v2_all (322) | v2_another_ai (245) |
| --- | --- | --- | --- |
| 112.1 target shown on "Who are you?" (bar 85%) | held: 465 of 465 | held: 322 of 322 | held: 245 of 245 |
| 112.2 q_name slot share at least 60% | failed: 154 of 329 (46.8%) | failed: 113 of 211 (53.6%) | failed: 104 of 190 (54.7%) |
| 112.3 non-AI concept share above q_name's | held: 0.800 against 0.532 | held: 0.815 against 0.464 | not read: no non-AI units in an "another AI" subset |
| 112.4 random non-self persona rate at most 5% | failed: 0.0796 on 1,860 labels | failed: 0.0505 on 1,288 labels | failed: 0.0633 on 980 labels |

112.1, 112.2 and 112.4 have the same verdicts on both confirmed subsets as on the full atlas. 112.3 cannot be read on
the "another AI" subset by construction. The sort on the 322 confirmed units: q_name 98 concept, 113 slot, 5
unclassified and 1 unsorted; other AI 31 concept and 18 slot; non-AI 44 concept, 10 slot and 2 unclassified.

## Parts 117 and 117b: slot and concept pairs (stored labels)

A pair is kept when its slot, matched concept and unrelated concept are all confirmed. All 11 pairs are confirmed in
the full atlas; 10 of 11 survive under both the all-three-confirmed and slot-and-match rules on v2_all; no pair
survives either rule on v2_another_ai. On v2_all, 117.1 is held, 117.2 is failed, 117.3-117.5 are failed, 117.6 and
117.7 are held, and 117.8 is failed. Thus 117.2 changes from held on the full atlas to failed on v2_all. On
v2_another_ai, every 117 and 117b prediction is not read because there are no confirmed pairs. The full-atlas rerun
reproduces the stored verdicts (117: 117.1, 117.2, 117.6 and 117.7 held; 117b: 117b.1 and 117b.2 held).

The driver reports the surviving pair count, reads registered verdicts literally and shows a scaled-bar reading beside
each absolute bar. Part 117b's coverage gates need 9 readable pairs (`MIN_HEADLINE_PAIRS` in `part117b/score117b.py`). v2_all has 10, but its headline is still
withheld: random coherence is 15 of 30 read rows, below the two-thirds gate of 20 of 30, so the random-direction
erasure reading is withheld too. The full set's headline is also withheld (15 of 33, below 22 of 33).

## Part 148: name and maker moved together (stored labels, `part148/score148.py`)

The gates (do-nothing hook, the first-32-unit match with Part 112, plain model unmoved) belong to the run and are
evaluated on the full atlas in every column.

| Prediction | Full atlas (465) | v2_all (322) | v2_another_ai (245) |
| --- | --- | --- | --- |
| 148.1 name moved and coherent on at least 80% | failed: 348 of 465 (74.8%) | failed: 232 of 322 (72.0%) | failed: 184 of 245 (75.1%) |
| 148.2 maker moved and coherent on at most 15% | failed: 191 of 465 (41.1%) | failed: 128 of 322 (39.8%) | failed: 107 of 245 (43.7%) |
| 148.3 at least 5 units move name and maker | held: 146 | held: 95 (scaled bar 4: held) | held: 81 (scaled bar 3: held) |
| 148.4 maker moved among name-moved under 25% | failed: 146 of 348 (42.0%) | failed: 95 of 232 (40.9%) | failed: 81 of 184 (44.0%) |
| 148.5 random name-moved under half of neurons' | failed: 175 against 348 | failed: 119 against 232 | **held**: 89 against 184 |
| 148.6 slot maker rate at most concept maker rate | failed: slot 82 of 189 (43.4%), concept 100 of 257 (38.9%) | failed: slot 64 of 141 (45.4%), concept 61 of 173 (35.3%) | failed: slot 55 of 122 (45.1%), concept 50 of 117 (42.7%) |

**One verdict changes against the full atlas: 148.5 is held on v2_another_ai.** 148.6 is failed on both confirmed
subsets, as on the full atlas; it no longer flips on the complete join.

## Part 152: do neurons read their persona words? (stored windows, `part152/score152.py`, 1,000 shuffles, seed 152)

152.1 and 152.2 run over the 2,409 page neurons with usable J lists and do not depend on the atlas; they reproduce
(held, held). 152.3 and 152.4 use Part 112's kinds:

| Prediction | Full atlas (465) | v2_all (322) | v2_another_ai (245) |
| --- | --- | --- | --- |
| 152.3 concept units hit their persona words at twice the shuffled mean, p under 0.01 | held: 2.105 against 0.186, n 247, p 0.001 | held: 2.462 against 0.209, n 173, p 0.001 | held: 1.786 against 0.226, n 117, p 0.001 |
| 152.4 slot mean under half the concept mean | held: slot 0.440 (n 184) against 2.105 | held: slot 0.433 (n 141) against 2.462 | held: slot 0.262 (n 122) against 1.786 |

No verdicts change: all four predictions are held in all three universes. The concept-slot gap is wider on v2_all than
on the full atlas, and narrower on v2_another_ai.

## Part 118: the analogue on OLMo, Part 92b's DS kinds (`part118/score118.py`, stored labels)

Part 118 pushes Part 92's 39 OLMo 2 1B units at four doses; its kinds come from Part 92's Haiku sort (18 concept, 21
slot). Part 92b's DS re-read keeps all 21 slots, keeps 8 of 18 concepts, moves 7 concepts to slot and 3 to
unclassified (`results/part92b/score92b.json`). The scorer is rerun with the DS kinds on the same 39-unit universe
(so the registered DS-sample keys and Gate 2 are unchanged); the 3 unclassified units count in neither list.

| Prediction | Part 92 Haiku kinds (18 concept, 21 slot) | Part 92b DS kinds (8 concept, 28 slot) | Scaled bar |
| --- | --- | --- | --- |
| Gate 1b: units keeping their kind at 0.8 (bar 26 of 39) | 34 of 39, read | 29 of 36 classified, read | 24 of 36 |
| 118.1 slot units concept-like at 1.6 (coherent rule; bar 7 of 21) | failed: 0 of 21 | failed: 0 of 28 | failed |
| 118.1a same, Part 92's shows-persona rule | failed: 4 of 21 | **held**: 8 of 28 | failed (bar 10 of 28) |
| 118.2 at 0.4: slot on "Who" at most 5 of 21, concept any question at least 9 of 18 | failed: 0 of 21; 4 of 18 | failed: 1 of 28; 1 of 8 | failed |
| 118.2a same, shows-persona rule | failed: 0 of 21; 4 of 18 | failed: 1 of 28; 1 of 8 | failed |
| 118.3 not coherent at 3.2 at least 60% | held: 234 of 234 | held: 234 of 234 | n/a |
| 118.4 random within 15 points of neurons at 3.2 | held: 100% against 100% | held: 100% against 100% | n/a |
| 118.5 concept units concept-like at 1.6 (bar 12 of 18) | failed: 0 of 18 | failed: 0 of 8 | failed |
| 118.5a same, shows-persona rule | failed: 10 of 18 | failed: 4 of 8 | failed (bar 6 of 8) |

**One literal verdict changes: 118.1a, failed under Haiku's kinds and held under DS's.** It is an artifact of the
absolute bar: DS adds 7 units to the slot list, 4 of them concept-like at 1.6, and 8 of 28 clears "at least 7" while
missing the proportional bar of 10 of 28. Read with the bar scaled, nothing changes. The units DS moved from concept
to slot are the ones that turn concept-like at twice the dose, which fits Part 118's own reading that the sort
depends on the push size.

## Summary of verdict changes

| Part | Verdict changes on the DS-confirmed subsets | Status |
| --- | --- | --- |
| 112 | 112.1, 112.2 and 112.4 keep their full-atlas verdicts; 112.3 is held on v2_all and unreadable on the AI-only subset | complete join: 465/465 |
| 117, 117b | 117.2 held to failed on v2_all; no pairs survive on v2_another_ai; 117b headline withheld on v2_all | complete join: 465/465 |
| 148 | 148.5 failed to held on v2_another_ai; 148.6 fails on all three universes | complete join: 465/465 |
| 152 | no verdict changes; all predictions held; the v2_all gap is wider and the v2_another_ai gap narrower than full | complete join: 465/465 |
| 118 (Part 92b kinds) | 118.1a failed to held literally, failed with the bar scaled | complete |
| 115 | not applicable | the Part 86 README sentence needs a fix |

## Files

- `report.json`: every universe, count, verdict and scaled reading.
- `part112/`, `part117/`, `part117b/`, `part148/`, `part152/`: one folder per universe (`full`, `v2_all`,
  `v2_another_ai`) with the rescored `score.json`; Part 148 also writes `tables.md`, Parts 117 and 117b the pairs used.
- `part118/full/` and `part118/part92b_ds_kinds/`: `SCORE.md` and `score.json`.
- `results/part86/atlas-v2/atlas-units-v2.jsonl` and `.summary.json`: the complete per-unit join (465 of 465).
- `tools/atlas_v2_units.py`, `tools/reader_correction.py`: the two scripts; both are CPU only and read no model.
