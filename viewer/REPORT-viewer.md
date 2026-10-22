# Clamp answers viewer report

## SUMMARY

Built `answers-viewer.html` as a self-contained offline page (1,204,137 bytes) and `build_viewer.py` as its data-rebuilding script. The page embeds all answer text and labels, opens from `file://`, and uses no external scripts or frameworks. The default screenshot and a filtered screenshot are saved in `shots/` and were visually inspected in headless Firefox.

Answer totals are 432 for Part 26.8, 480 for Part 54A, and 1,920 for Part 54B. Each label file also contains 22 calibration rows; those are excluded from the answer lists and counts. The rebuilt shuffled rows had zero mismatches against the stored key fields.

## FINDINGS

- Part 26.8 has two answers with null labels, Part 54A has none, and Part 54B has eight. No empty answer texts were found.
- The Part 26.8 key predates the `model` field and omits it on all 432 rows. The source answers all resolve to the single model `olmo2-1b`; that source value is inferred while the stored question, dose, kind, condition, and sample index fields are checked against the key. The other runs’ keys include model and match directly.
- The brief calls these Haiku 5.5 labels, but the supplied `blind_origin.py` declares `claude-opus-5-5`. The label files contain only the two judgments and do not record the batch model. The viewer states this discrepancy without choosing an attribution.
- The default screenshot shows Part 26.8 with no filters; `snapshot=overview` clips the screenshot to the initial page view because the unfiltered answer list is very long. The filtered screenshot selects Part 54B, `qwen3-4b`, Twin, dose `0.5×`, and shows 24 answers. Both screenshots show the summary, cards, colored labels, and word highlights clearly at laptop width.
- Labels preserve only the origin and OK/BROKEN judgments. There was one reader, no second reader, and no per-answer reasoning in the stored files. The requested term highlights are only a reading aid.

## BLOCKERS

Confirm whether the saved labels should be attributed to Haiku 5.5 or Claude Opus 5.5. Until that source-model discrepancy is resolved, the page leaves attribution explicitly unresolved.
