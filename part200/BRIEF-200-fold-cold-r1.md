# Brief: fold Part 200's cold check, round 1

Repository: `/home/elisepc/green/worktrees/ab-part200` (branch `part200`). Writable for you: `part200/` and
`results/part200/` (not `results/part200/cold/`, which is the reviewer's). No push, no commit, no model, no GPU.

Read `results/part200/cold/COLD.md` (verdict SHIP-WITH-FIXES, ten findings; the reviewer's scripts are in the same
folder and read files in place) and fold findings 1 to 8 and 10 into `part200/report200.py`, `part200/fig200.py`,
`results/part200/README.md` and `results/part200/hand_read.md`. Finding 9 is Claude's (PREDICTIONS.md, Part 163's
README); leave it.

Rules: every new number is computed by `report200.py` into `report200.json` / `report200.md`, never typed. Registered
wording stays verbatim. New analyses are labelled exploratory. Specifically:

1. Lengths before any cross-run rate (the design's reading rule): mean and median token counts per arm, both runs; broken
   counts split at cap and under cap per arm, both runs; say plainly that most rerun broken answers are at the cap and
   under-cap broken fell (with the numbers), so the broken rise is a length change as much as anything.
2. The joint attribution sentence (kernel and batch shape together, never either alone) in the Answer paragraph and the
   200.3 and 200.5 rows, as the design words it.
3. Gate 1's changed shares per arm, rerun beside published, against its floor, in the first paragraph.
4. Title and first paragraph: say which Part 163 verdicts were live readings (163.4, one unit from flipping, and 163.5)
   and which could not fail at these rates (163.1, 163.2, 163.3), as the design states; do not lead with "verdicts hold"
   as though all five were tests.
5. Identical-text counts (390 of 3,200 battery, 4 of 16 no-push) and the under-cap mid-phrase counts, both runs, from
   compare.json's exploratory section.
6. The first-difference convention (17 prompt tokens plus the 0-based answer index), the re-tokenization check (how
   many texts reproduce token_count, and that token_count includes the stop token under the cap), the ±1 token
   uncertainty and that 73 of 76 holds at 63, 64 and 65. One sentence on what the sub-64 changes mean: batch shape
   changes eager output where Part 178's fault does not operate, so these are near-tie flips, not the SDPA fault; say
   it is a reading, not a test.
7. How the hand-read samples were chosen.
8. The load record's `"part": 199` label and `chunk_size`/`batch_size` 64 (a default inherited from run199; the calls
   ran in chunks of 8 per the call records), the run's rate (answers per second against the 0.17 guard) from
   load.json, and any timing the records support.
10. Add under-cap broken counts to the figure's left panel (or a third panel), drawn from report200.json.

Rerun `report200.py` and `fig200.py` with `/home/elisepc/neuron-probe/.venv/bin/python -I -B`, then
`/home/elisepc/neuron-probe/.venv/bin/python -I tools/part_check.py publish 200`, and report its output. Leave the
README's "Cold check:" line saying round 1 found 10 items, folded, and round 2 pending.

Write `part200/REPORT-fold-cold-r1.md` FIRST as a skeleton (one line per finding: what changed and where) and keep it
updated, ending with SUMMARY / FINDINGS / BLOCKERS lines.
