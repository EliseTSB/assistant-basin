# Part 68: find drivers of the later maker neurons

Part 68 searches earlier MLP units for drivers of L11 N8868 and L29 N776. It uses the prompt split and analysis
registered in `../PART68-DESIGN.md`; do not select or alter candidates using the test half.

## Run

Run the search first. Each backward pass handles one prompt and one target; the model weights stay frozen and only
captured activation tensors receive gradients. `--batch` applies to the forward-only test phase.

```bash
python3 part68/part68.py --phase search --out results/part68 --device cuda --max-gpu-gb 14.5
python3 part68/part68.py --phase test --out results/part68 --device cuda --batch 4 --max-gpu-gb 14.5
```

Use `--device xpu` on the Intel GPU host or `--device cpu` where appropriate. The test phase requires the search JSON
files in the same output directory. To run both phases in one process, use `--phase all`; the search still runs before
the test phase. Search writes `attr_11_8868.json/.npz`, `attr_29_776.json/.npz`, and `run-search.json`. Test writes
`activations.json` and `run.json`.

## Score

After both phases finish, score the output directory:

```bash
python3 part68/score68.py --dir results/part68
```

This writes `results/part68/score.json` and `results/part68/score.md`. The scorer reports normalized Delta on both
halves, candidate replication against the test-half null 95th percentile, all-candidate effects, candidate and
replicating layer histograms, cross-target overlap, the 68.5 zero check, and 68.6 Spearman correlations. If 68.5 fails,
the remaining predictions are marked `NOT READ`, following the registered reading rule.

Nulls use `numpy.random.default_rng(6800)`, allocated across candidate layers in their candidate proportions; the
overlap simulation uses seed 6801 and 10,000 draws. See `BUILD-NOTES.md` for implementation decisions and limits.
