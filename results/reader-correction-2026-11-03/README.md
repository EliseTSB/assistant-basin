# Reader correction: the atlas parts rescored on the version-2 (DS) confirmed subset

Written 2026-11-03 by Claude (a Claude Code cloud session, CPU only, stored labels only; no model ran and no reader
was called). Follows item 1.4 of `review/REVIEW-2026-11-03-whole-repo.md`. Everything here is produced by
`tools/atlas_v2_units.py` and `tools/reader_correction.py`; `report.json` holds every number, and each part's
subfolder holds the rescored `score.json` (and `SCORE.md` or `tables.md` where the scorer writes one).

## What was done, and what could not be

Part 86's first-reader atlas has 465 units (387 "another AI", 78 other kinds), labelled by Claude Haiku 5.5. Under
brief version 2, DS V4.1 Flash kept 245 of the 387 "another AI" units and 77 of the 78 other-kind units as coherent
personas (`results/part86/atlas-v2/ATLAS-v2.md`). The version-2 labels are keyed by blind item ids made with
`secrets.token_urlsafe`; the id-to-screen-row key is private, and the public items file (id, text) lives in the J-lens
study checkout (`$JLENS_ROOT/data/qwen3.5-4b/labels/items.jsonl`), which is not in this repository and was not
reachable from this session. The usage logs carry no text. So the full 3,349-item join cannot be rebuilt here.

What can be rebuilt: every atlas unit's screen answer is unique among the 165,376 screen rows, so an item's text
identifies its unit exactly. `tools/atlas_v2_units.py` joins labels to units through item text. This checkout holds
item texts only for the 335-item version-2 calibration sample (`results/part86/second-reader/items_v2cal.jsonl`),
which joins 57 of the 465 units: 29 "another AI" units DS calls coherent personas, 24 it calls still Qwen or an
assistant, and 4 other-kind units, all coherent (`results/part86/atlas-v2/atlas-units-v2.summary.json`). Part 142
joined the same labels by casefolded persona-string equality and reached 43 units; that join is not used here
because two readers' free-text persona strings are not a key.

**To complete the join on a machine with the J-lens checkout, run:**

    python3 tools/atlas_v2_units.py --items ~/jlens-persona-neurons/data/qwen3.5-4b/labels/items.jsonl
    python3 tools/reader_correction.py

The first command checks its own result against ATLAS-v2.md (245, 141, 1 and 77, 1, 0) and exits 0 only when every
unit is joined and the counts match. The second rescores every part below. Until then, the subset columns are a
pipeline check on a seeded 10% sample, not the result Elise asked for.

Three unit universes are scored: **full** (465 units; this must reproduce the stored verdicts, and it does for every
part below), **v2_all** (DS-confirmed, any kind: 322 when complete, 33 now), and **v2_another_ai** (the "another AI"
members: 245 when complete, 29 now). Registered bars written as absolute counts are read literally and, marked
exploratory, scaled to the surviving denominator.

| Part | Starts from the atlas? | Rescored here |
| --- | --- | --- |
| 112 | yes, all 465 units | yes |
| 115 | no: candidates are the top 20 of an attribution ranking over all neurons (`PART115-DESIGN.md`); the Part 86 README's sentence that Part 115 used the atlas as its starting list is wrong on this point | not applicable |
| 117, 117b | yes, 11 slot/concept pairs from Part 112's sort | yes; no pair survives the partial join |
| 118 | no: OLMo 2 1B, Part 92's 39 units | its analogue, Part 92b's DS re-sort, is rerun instead |
| 148 | yes, all 465 units | yes |
| 152 | the persona-word predictions 152.3 and 152.4 use Part 112's sort | yes |

## Part 112: the concept/slot sort (stored labels, `part112/score112.py`)

| Prediction | Full atlas (465) | v2_all (33) | v2_another_ai (29) |
| --- | --- | --- | --- |
| 112.1 target shown on "Who are you?" (bar 85%) | held: 465 of 465 | held: 33 of 33 | held: 29 of 29 |
| 112.2 q_name slot share at least 60% | failed: 154 of 329 (46.8%) | failed: 11 of 21 (52.4%) | failed: 11 of 21 (52.4%) |
| 112.3 non-AI concept share above q_name's | held: 0.800 against 0.532 | held: 0.750 against 0.476 | not read: no non-AI units in an "another AI" subset |
| 112.4 random non-self persona rate at most 5% | failed: 0.0796 on 1,860 labels | failed: 0.0530 on 132 | failed: 0.0603 on 116 |

No verdict changes. 112.3 cannot be read on the "another AI" subset by construction. The sort on the 33 confirmed
units: q_name 10 concept, 11 slot, 1 unclassified; other AI 3 concept, 4 slot; non-AI 3 concept, 1 slot.

## Parts 117 and 117b: slot and concept pairs (stored labels)

A pair is kept when its slot, matched concept and unrelated concept are all confirmed. On the partial join 0 of 11
pairs survive under that rule and 0 under the weaker slot-and-match rule, so every prediction is "not read" on both
subsets; the full-atlas rerun reproduces the stored verdicts (117: 117.1, 117.2, 117.6, 117.7 held; 117b: 117b.1 and
117b.2 held). The driver is ready for the complete join; its 117 and 117b blocks report the surviving pair count, the
registered verdicts read literally, and a scaled-bar reading beside each absolute bar (for example "at least 9 of 11
pairs", which on fewer pairs cannot be met by arithmetic). Part 117b's coverage gates need 9 readable pairs
(`MIN_HEADLINE_PAIRS` in `part117b/score117b.py`) and will withhold the headline below that.

## Part 148: name and maker moved together (stored labels, `part148/score148.py`)

The gates (do-nothing hook, the first-32-unit match with Part 112, plain model unmoved) belong to the run and are
evaluated on the full atlas in every column.

| Prediction | Full atlas (465) | v2_all (33) | v2_another_ai (29) |
| --- | --- | --- | --- |
| 148.1 name moved and coherent on at least 80% | failed: 348 of 465 (74.8%) | failed: 26 of 33 (78.8%) | failed: 23 of 29 (79.3%) |
| 148.2 maker moved and coherent on at most 15% | failed: 191 of 465 (41.1%) | failed: 11 of 33 (33.3%) | failed: 9 of 29 (31.0%) |
| 148.3 at least 5 units move name and maker | held: 146 | held: 7 (scaled bar 1: held) | held: 6 (scaled bar 1: held) |
| 148.4 maker moved among name-moved under 25% | failed: 146 of 348 (42.0%) | failed: 7 of 26 (26.9%) | failed: 6 of 23 (26.1%) |
| 148.5 random name-moved under half of neurons' | failed: 175 against 348 | failed: 15 against 26 | failed: 13 against 23 |
| 148.6 slot maker rate at most concept maker rate | failed: slot 82 of 189 (43.4%), concept 100 of 257 (38.9%) | **held**: slot 3 of 16 (18.8%), concept 7 of 16 (43.8%) | **held**: slot 3 of 15 (20.0%), concept 5 of 13 (38.5%) |

**One verdict changes: 148.6, failed on the full atlas and held on both subsets.** On 16 against 16 units (and 15
against 13) this is a small-sample result; a difference of 4 answers would reverse it. It is the one place in this
rerun where the DS correction points the other way from the Haiku atlas, and it is the kind of flip the complete
join should settle.

## Part 152: do neurons read their persona words? (stored windows, `part152/score152.py`, 1,000 shuffles, seed 152)

152.1 and 152.2 run over the 2,409 page neurons with usable J lists and do not depend on the atlas; they reproduce
(held, held). 152.3 and 152.4 use Part 112's kinds:

| Prediction | Full atlas | v2_all (33) | v2_another_ai (29) |
| --- | --- | --- | --- |
| 152.3 concept units hit their persona words at twice the shuffled mean, p under 0.01 | held: 2.105 against 0.186, n 247, p 0.001 | held: 2.812 against 0.319, n 16, p 0.001 | held: 2.692 against 0.354, n 13, p 0.001 |
| 152.4 slot mean under half the concept mean | held: slot 0.440 (n 184) against 2.105 | held: slot 0.062 (n 16) against 2.812 | held: slot 0.067 (n 15) against 2.692 |

No verdict changes. The concept-slot gap is wider on the confirmed units than on the full atlas.

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

| Part | Verdict changes on the DS-confirmed subset | Status |
| --- | --- | --- |
| 112 | none (112.3 unreadable on the AI-only subset) | partial join, 33 and 29 units |
| 117, 117b | not read: no pair survives the partial join | awaits the complete join |
| 148 | 148.6 failed to held, on 16 against 16 units | partial join |
| 152 | none; 152.3 and 152.4 hold with a wider gap | partial join |
| 118 (Part 92b kinds) | 118.1a failed to held literally, failed with the bar scaled | complete |
| 115 | not applicable | the Part 86 README sentence needs a fix |

## Files

- `report.json`: every universe, count, verdict and scaled reading.
- `part112/`, `part117/`, `part117b/`, `part148/`, `part152/`: one folder per universe (`full`, `v2_all`,
  `v2_another_ai`) with the rescored `score.json`; Part 148 also writes `tables.md`, Parts 117 and 117b the pairs used.
- `part118/full/` and `part118/part92b_ds_kinds/`: `SCORE.md` and `score.json`.
- `results/part86/atlas-v2/atlas-units-v2.jsonl` and `.summary.json`: the per-unit join (partial here).
- `tools/atlas_v2_units.py`, `tools/reader_correction.py`: the two scripts; both are CPU only and read no model.
