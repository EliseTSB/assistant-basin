# Part 63 Amendment 3 reparse report

## SUMMARY

Implemented the offline Amendment 3 reparse path and analyzer gate support. The command reads saved batch IDs and fetches each batch status and results through the existing API callback using GET only. Parsed records use the narrowly specified maker-kind prefix normalization, preserve source strings, apply retry-over-root precedence, and fan labels back to answer IDs. The new output is `results/labels63-run-all-lenient.json`; the strict labels file remains unchanged.

## FINDINGS

- `reader63.py reparse` defaults to `results/labels63-run-all.json`, and accepts alternate labels, data, and review paths. It loads each batch's saved submission, ID map, and key snapshot and rejects unknown or duplicate result IDs, incomplete metadata, and batches that have not ended.
- Normalization accepts a valid maker-kind token only when it ends the string or is followed by a space, semicolon, comma, or colon. The parsed record retains `maker_kind_normalized_from`; malformed kinds and maker-name/kind disagreement remain failed.
- The lenient file carries the copied strict gate result and failure count, normalized IDs and original strings, residual failures, the Amendment 3 rule, and `gate_override`. `failed_or_unlabelled_ids` remains the actual residual failure list. The gate requires calibration and residual failures at or below one percent of the request count.
- Analyzer override validation checks the amendment number, rule text, residual IDs against the actual records, and gate calculation. It prints the override near the start of analysis. Without a valid override, failed or unlabelled labels continue to block scoring. For valid overrides, duplicate-consistency output uses the labelable pairs remaining.
- Updated `RUNBOOK-63.md` with `python3 part63/reader63.py reparse` and the resulting file description.
- Verification: `python3 -m py_compile reader63.py analyze63.py test_part63.py` succeeded. `python3 -m unittest -v test_part63` passed all 27 tests. Fake transport assertions confirm reparse sends only GETs, does not change the strict labels bytes or ledger, and writes a lenient file. Added normalization and one-percent copied-code mutations both make their focused tests fail as expected.

## BLOCKERS

None for the requested implementation.

## UNVERIFIED

No live provider request was made, per the offline brief. The actual availability and exact content of the historical batch-results endpoint could not be independently confirmed in this run. The fake transport exercises the existing batch-result response shape, saved sidecars, mapping, fan-out, and retry merge logic.
