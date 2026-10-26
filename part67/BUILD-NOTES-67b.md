# Part 67b build notes

## What was built

`part67b.py` reuses Part 67's model, revision, prompts, intervention hooks, target list, and activation record convention. Pass 1 runs each unchanged prompt and accumulates float32 absolute down-projection inputs across all token positions for every MLP unit in layers 6, 7, and 11. It writes the complete per-unit means to `unit_magnitudes.json`.

For each of L6 N753, L7 N1169, and L11 N8868, pass 2 selects 50 other units by ascending absolute difference from that target's mean magnitude, with unit index as the tie-breaker. `magnitude_matches.json` records the target magnitude and every selected unit's magnitude. Pass 2 runs unchanged, the three upstream target-zero conditions, and each selected unit zeroed alone. It records all four target activations in the same schema as Part 67, including unchanged position values needed by the shared scoring normalization.

Run `python part67/part67b.py --out DIR` to collect data, then `python part67/part67b.py --out DIR --score` to produce `score67b.json` and `score67b.md`. The score has six ordered-pair rows, uses the existing Part 67 unchanged-position standard deviation, computes the registered two-sided empirical p with ties counted as at-or-above, and marks survival only when the target's absolute Delta is strictly greater than all 50 matched effects. The Markdown report includes each upstream target magnitude and its matched-unit magnitude range.

## Tests run and not run

- Ran `python3 -m unittest part67.test_part67b -v`: four host tests passed; the tiny-torch pass-1 test was skipped because torch is not installed in this sandbox.
- Ran `python3 -m py_compile part67/part67b.py part67/test_part67b.py` successfully.
- Ran `python3 part67/part67b.py --help` successfully.
- Did not run the tiny-model numerical check, model collection, or scoring against real activations: this sandbox has no torch or model runtime/data.

## Ambiguity

The amendment specifies the matched units and requires their magnitudes to be saved, but does not prescribe a filename or JSON shape. I wrote these selections to `magnitude_matches.json`, with per-target metadata and per-unit magnitudes, and included them in `run.json` as well. No design rule was changed.
