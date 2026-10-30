# Part 86 version 2 scoring notes

Amendment 7 makes the full DS V4.1 Flash version 2 reading the primary labels. The scorer now defaults to version 2 and reads `results/part86/second-reader/labels_ds_v2_full.jsonl`. It uses the v2 DS and Luna calibration kappa for the screen agreement gate. Version 1 remains available for the historical record with `--reading v1`. The two readings are never pooled, and version 1 labels do not contribute to a version 2 verdict.

After the full version 2 reading is complete, run this once from the Assistant Basin repository root:

```sh
python3 -I results/part86/prediction-score/score86_predictions.py --reading v2
```

The report is written to `results/part86/prediction-score/SCORE-v2.md`. The existing `SCORE.md` is left alone.

Before trusting any verdict, confirm the output says all 17 screen layers pass, the version 2 label preflight passed with every first-reader item covered and zero error rows, and label coverage is complete. It must identify the gate source as `calib-agreement-v2.json` pair `ds_v2~luna_v2` and print a kappa of at least 0.6. If the version 2 label file is missing, has an error row or duplicate id, or does not match the first-reader item ids exactly, the scorer prints `NOT READY` with counts and exits nonzero before reading the private key. A kappa below 0.6 leaves predictions 86.2 through 86.6 unread, while 86.5 remains not scorable because verification did not run.
