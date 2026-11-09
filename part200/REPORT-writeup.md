# Part 200 writeup work log

## Files written

- `part200/report200.py` reads the Part 200 and Part 163 result files, recomputes the required gate, verdict, count, Qwen, batch-1 and exploratory values, and writes `report200.json` and `report200.md`.
- `part200/fig200.py` draws `results/part200/fig200.png` from `report200.json`.
- `results/part200/README.md` includes the registered predictions verbatim with confidence, verdicts, answer, skeptical-question readings, source links and the required `Cold check: pending` line.
- `results/part200/hand_read.md` records five sampled answers per arm in recipe and battery stages and five changed batch-1 pairs per arm.

## Commands run and results

- Read `PART200-DESIGN.md`, Procedure sections 7 and 8, Part 199's report and figure scripts and README, Part 163's scorer and run code, and the Part 200 result records.
- Control mailbox reads before work, after substantial operations, and before final reporting returned no messages.
- `/home/elisepc/neuron-probe/.venv/bin/python -I -B part200/report200.py` — completed; computed 5 unchanged Part 163 verdicts, all gates passed, counts, and batch-1 comparison (76/200 changed; 39 neuron, 37 random). Local tokenizer was available and used; decoded answers were re-tokenized because token IDs were not stored.
- `/home/elisepc/neuron-probe/.venv/bin/python -I -B part200/fig200.py` — completed and wrote `results/part200/fig200.png`. Matplotlib warned its default config directory was unwritable, created a temporary cache, and continued successfully.
- `/home/elisepc/neuron-probe/.venv/bin/python -I tools/part_check.py publish 200` — 7/9 checks passed. The remaining failures are the expected pending cold check and the newly generated `report200.json` not being committed. The assignment explicitly leaves cold check pending and forbids commits.
- An attempt to call bare `python` failed because it is unavailable on PATH; all requested scripts ran with the specified venv Python.

## Uncertainties

- First-difference token indices are computed by re-tokenizing decoded answer text, not original generation IDs. They are exploratory positions, not exact cached token paths.
- The batch-1 vs chunk-of-8 comparison isolates shape for one identity prompt; it does not cover maker prompts or explain the cross-run at-cap change, which also changes the attention kernel.
- The requested cold review has not happened; README retains `Cold check: pending` as instructed.

SUMMARY: Delivered the report generator, computed report JSON/Markdown, figure, README, and hand-read notes.
FINDINGS: Gates passed; Part 163's five verdicts were unchanged; batch-1 differed on 76 of 200 rows.
BLOCKERS: none for this pre-cold-check writeup; publish check remains pending cold review and commit.
