# Part 86 atlas v2

This script joins the version 2 reader labels to the Part 112 atlas and the real screen rows. It reports the first reader's atlas comparison as a diagnostic, then computes the registered counts for 86.11, 86.12, and 86.13. It reuses the Part 86 scorer for loading, label classes, preflight, and the agreement gate.

From the Assistant Basin repository root, run:

```sh
python3 -I results/part86/atlas-v2/atlas86_v2.py
```

Before believing any verdict, check that the version 2 label preflight passes, all 465 atlas units have labelled screen rows, and the printed agreement source gives kappa at least 0.6. The version 1 comparison is diagnostic only. If the gate is lower, all three verdicts must say NOT READ.
