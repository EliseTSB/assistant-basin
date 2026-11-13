# Part 202 design report

## SUMMARY

Drafted the eager-kernel rerun design for Part 164 and added the Part 164 margin scorer and tests.  
No runner was built and no model or GPU work was done.  
The final recorder and shape-floor interfaces remain marked for Part 201.

## FINDINGS / WHAT SHIPPED

- `PART202-DESIGN.md` follows the Part Design template, preserves the Part 164 gates and quoted verdicts, records hashes for the inputs, describes the eager loader, chunking, pilot, resume, and shape replicate, and includes arithmetic for every published gate and verdict in Size.
- `part202/margins202.py` reads the published Part 164 score and answer rows, prints and writes `results/part202/margins.md` and `margins.json`, and reports distances plus cap counts and mean `token_count` per arm. The rate-to-row distances are clearly labeled equivalents because the registered scorer averages per-unit rates.
- `part202/test_margins202.py` runs with the prescribed isolated Python command. Both tests pass. A temporary mutant adding one to the rate-distance calculation failed specifically at `test_rate_gap_is_reported_as_equivalent_rows`.
- Part 164's plain gate counts a case-insensitive Qwen substring, while the registered outcome rule uses the bare case-sensitive word `Qwen`; the plain answers use `Qwen3.5`, so the registered match misses them (`part164/score164.py:24-25,53-60,106-111`; `results/part164/README.md:3-5`).
- The design says the 164.4 strength reading is moot if 164.1 fails, while the scorer still assigns 164.4 a failed verdict (`PART164-DESIGN.md`, Reading rules; `part164/score164.py:212-218`; `results/part164/README.md:13-17`). The new design reports the registered scorer verdict and carries the moot-reading qualification.
- Part 164's README reports that predictions 164.3 to 164.5 had little chance to go the other way at the published rates (`results/part164/README.md:55-59`). The margin script shows that none is mathematically unable to fail; it distinguishes exact integer gate headroom from approximate row-equivalent rate distances.
- The pilot amendment says the first launch's actual answer rate remains unknown after its output was overwritten (`results/part164/README.md:70-73`). The current runner accepts `loader` and `generate_fn` (`part164/run164.py:90-110`), so this brief needs no behavior-preserving Part 164 change.

## ASSUMPTIONS

- The recorded SHA-256 inputs are the Part 164 unit list, Part 112 medians, and the prompt source and selection code. The wrapper will also compare the resolved ordered prompt list before scoring.
- The 36-row tail of Part 164's 64-row generation batches will use four duplicate companion inputs to form one final eight-row model batch; companion outputs are discarded and never scored.
- `token_count == 80` is the Part 164 cap indicator. The arm means use all battery answers in that arm, including the three random-dose cells.
- The script's row equivalents multiply each per-unit mean-rate gap by 1,600. These distances communicate scale; exact threshold crossing requires rescoring the changed per-unit rows.
- The pilot's approximate rate guard uses the published expected row count and the six-hour launcher limit. It has not been measured on the target device.
- Part 201 will provide the recorder schema and `tools/shape_floor.py` interface. The markers are placed at each dependent method, control, and reading-rule point.

## COULD NOT DETERMINE

- Part 201's final interfaces are not available yet, so their concrete call syntax cannot be fixed in this draft.
- I could not verify actual eager loading, the A770 pilot rate, or runtime and memory behavior because this assignment forbids model and GPU work.
- The row-equivalent margins are not exact counts of individual answer edits because the registered estimator averages rates by unit.

## FILES TOUCHED

- `PART202-DESIGN.md`
- `part202/margins202.py`
- `part202/test_margins202.py`
- `part202/REPORT-design.md`
- `results/part202/margins.md`
- `results/part202/margins.json`

## BLOCKERS

No blocker to this design draft. Part 201 interfaces are pending and explicitly marked.

## SELF-ASSESSMENT

The scorer and fixture tests were exercised with the repository's isolated Python environment. The intentional mutation was isolated under `/tmp`. I did not run the full prelaunch checker or validate any GPU path. The design uses a reasoned padding approach for the 36-row tail; the eventual wrapper must verify the actual shape and row mapping in its load record.

SUMMARY: Design draft and computed published-data margins are complete.  
FINDINGS: Gate 2's name matcher differs from the registered answer predicate; 164.4 is scored despite its moot interpretation.  
BLOCKERS: None for this scoped draft; Part 201's final interfaces remain pending.
