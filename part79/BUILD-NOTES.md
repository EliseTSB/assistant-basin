# Part 79 build notes

## What shipped

- `step0.py` packs the first 400 local Pile windows, measures L13/L14/L22 block-output L2 norms at positions 1–1023, computes target and size-matched control deltas from each loaded model's down-projection columns, and writes `step0.json` plus the eight-row greedy pilot.
- `part79.py` imports Part 66/78's prompts, maker intervention, rendering, seeding, and stop handling. It adds each registered delta at every sequence position, resumes `answers.jsonl`, and records per-batch evidence for both additive hooks and each maker-zero hook.
- `score79.py` reuses Part 78 classification, writes keyword/reader counts and verdicts, and creates a seed-79 shuffled blind-reader input without condition labels.
- `test_part79.py` covers the hook, dose formula, control draws, 1,944-row plan, skipped position zero, verdict gates, and blinded reader input. `smoke_run79.py` runs a one-window local step-0 calibration and all 1,944 main-run rows on a CPU fixture.

## Verification

Test command: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q part79`

Output: `6 passed in 0.64s`.

Smoke command: `/home/elisepc/neuron-probe/.venv/bin/python -I part79/smoke_run79.py /tmp/part79-smoke3`

Output: `smoke rows 1944 conditions 9 hook batches 270`. Each evil/control push unit fired 30 times across its condition's batches; maximum measured additive-delta error was `0.0`. The four maker-zero hooks were present and called on maker conditions. The random fixture correctly has no meaningful keyword verdicts.

The requested negative checks were run one mutation at a time, restored after each run, and all went red: remove the add, break the delta formula, change the control draw, drop a guard sample from the row plan, include position zero in the norm summary, break the scorer's 79.2 range gate, and leak `condition` into `reader_input.jsonl`.

## GPU folder copy list

Read-only inspection of `/mnt/c/Users/alexa/neuron-probe-gpu/p66basin/` found `part66/part66.py`, `part66/score66.py`, `part78/part78.py`, `part78b/part78b.py`, and root `neuron_report.py`. The `part78/score78.py` dependency used by this build is absent there and must be copied from this repo. Copy the six files under `part79/` (including `__init__.py`) and `part78/score78.py` into their matching paths under that folder. Existing Part 66, Part 78 runner, Part 78b runner, and `neuron_report.py` files are already present. No files were copied to the GPU folder in this build.

## Limits / self-assessment

- No model download or GPU run was attempted. The end-to-end checks use an intentionally small local fixture and establish plumbing, not Qwen behavior or the pilot gate.
- The external pilot-label JSON expected by `--pilot-labels` uses `{"evil_villain": N, "unchanged_villain": N}`; the brief specified the counts but not exact field names.
- Synthetic test perturbations were each confirmed red and restored; the baseline suite and smoke pass afterward.
