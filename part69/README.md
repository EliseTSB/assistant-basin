# Part 69

Run the registered prefill readout from the repository root:

```bash
python3 part69/part69.py --out results/part69
```

The runner defaults to `results/part68/b/matched.json`, device `xpu`, batch size 32,
and a 14 GiB spill guard. Use `--device cpu` or `--device cuda` where appropriate,
and `--matched PATH` to select a matched-unit file.

Score a completed run with:

```bash
python3 part69/score69.py --dir results/part69
```

The runner writes `magnitudes.json`, `conditions.json`, `records.json`, and `run.json`.
The scorer writes `score69.json` and `score69.md`. The positive control is prompt index
24 and is reported separately from the 24 held-out prompts used for Dg and verdicts.
