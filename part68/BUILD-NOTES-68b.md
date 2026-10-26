# Part 68b build notes

## Built

- `part68b.py` reads the Part 68 score, loads the same pinned Qwen checkpoint, collects unchanged down-projection input magnitudes over the test prompts, selects nearest-magnitude units with the Part 67b ordering and Part 68b exclusions, and measures unchanged, zero-nothing, replicating-candidate, and unique matched-unit conditions. It writes `magnitudes.json`, `matched.json`, `activations_b.json`, and `run_b.json`.
- The execution path after model loading is in `run_b(...)` and takes token-id lists. Its match count and spill guard are injectable for the tiny-model test.
- `score68b.py` uses Part 68's `normalized_delta`, calculates each candidate's matched-set maximum and survival, checks 68.7–68.10, and writes JSON plus a per-target Markdown table.
- `test_part68b.py` covers match exclusions and tie order, count/error boundaries, strict survival, verdict and reading boundaries, the 0.005 control edge, unchanged/zero-nothing equality across batch sizes, and a tiny-model run-path/scorer test when PyTorch is installed.

## Tests run here

- Python AST parsing and direct import/helper checks passed.
- `python3 -m pytest -q part68/test_part68b.py` could not run: pytest is not installed (`No module named pytest`). PyTorch is also not installed, so the tiny-model test could not run here.
- No full Qwen model run was attempted.

## Amendment ambiguity

Magnitude is defined per prompt as the mean absolute input over that prompt's positions, then averaged equally across prompts, matching the amendment's “average over prompts” wording. Matched units are deduplicated within each target's run and then mapped back to every candidate that selected them; the same unit is measured separately for the other target when it is also selected there.

## Host fixes before the first run (Claude, 2026-10-26 03:5x CT)

- **No candidate or matched unit would have been zeroed.** `_target_values` hooked only the target's own layer, and
  `part25._row_intervention` applies only the operations for the layer it is called in, so every zeroing of an earlier
  unit was a no-op and every Delta would have been 0 (68.10 would have caught it, after a GPU run). It now measures
  through Part 68's tested `_target_activations`, which hooks every layer an operation names.
  `test_run_b_matches_part68s_direct_measurement` compares run_b's recorded value with Part 68's direct measurement
  and requires the zeroing to move the target; the worker's version fails it (mutant run on the host). The existing
  tiny-model test passed that version because its fixture set every expected Delta to 0.0.
- **main() crashed on import.** Putting `part67/` first on sys.path made part68.py's `from part67 import part67`
  find part67/part67.py as a plain module. part68b.py now binds the `part67` namespace package first and appends the
  folder. `test_main_imports_resolve` runs both scripts' real imports with `--help` in a fresh interpreter; the old
  order fails it.

Host: 8 of 8 (test_part68b.py) and 19 of 19 (test_part68.py), torch tests included. Run the two files in separate
pytest invocations: in one session the 68b suite's path setup shadows the `part67` package for the 68 suite.
