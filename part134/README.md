# Part 134: dog pair grid

The registered design is in [`PART134-DESIGN.md`](../PART134-DESIGN.md). This folder contains the stage runner, answer reader, development selector, scorer, and CPU fixtures. The runner writes per-stage answer JSONL plus atomic metadata; each cell/arm is appended atomically and skipped on resume. Test selection is read from `part134/selected134.json` (written by `select134.py`), never from CLI arguments.

## A770 run order

From the repository root, use the A770 launcher for every model run:

```bash
python3 tools/launch_arc.py 134 --step "-I -B part134/run134.py --stage pilot --device xpu --batch-size 8 --out results/part134" --copy part134/run134.py part134/select134.py part134/label134.py part134/score134.py part134/test_part134.py part134/README.md part92/common92.py part85/checkpoints.py
```

Inspect the six pilot answers against Part 109c, then collect and label development answers:

```bash
python3 tools/launch_arc.py 134 --step "-I -B part134/run134.py --stage dev --device xpu --batch-size 8 --out results/part134" --copy part134/run134.py part134/select134.py part134/label134.py part134/score134.py part134/test_part134.py part134/README.md part92/common92.py part85/checkpoints.py
python3 part134/label134.py --stage gate-ds --known results/part134/known-ds.jsonl --out results/part134/known-ds.jsonl
python3 part134/label134.py --stage gate-luna --known results/part134/known-luna.jsonl --out results/part134/known-luna.jsonl
python3 part134/label134.py --stage ds --answers results/part134/dev-answers.jsonl --out results/part134/dev-labels-ds.jsonl
python3 part134/select134.py --answers results/part134/dev-answers.jsonl --labels results/part134/dev-labels-ds.jsonl --out part134/selected134.json
python3 tools/launch_arc.py 134 --step "-I -B part134/run134.py --stage test --device xpu --batch-size 8 --out results/part134" --copy part134/run134.py part134/select134.py part134/label134.py part134/score134.py part134/test_part134.py part134/README.md part134/selected134.json part92/common92.py part85/checkpoints.py
python3 part134/label134.py --stage ds --answers results/part134/test-answers.jsonl --out results/part134/test-labels-ds.jsonl
python3 part134/label134.py --stage luna --answers results/part134/test-answers.jsonl --ds-labels results/part134/test-labels-ds.jsonl --out results/part134/test-labels-luna.jsonl
python3 part134/score134.py --out results/part134
```

Reader gates require at least 11/12 correct known examples. DeepSeek labels all answers; Luna reads every DeepSeek endpoint-positive plus a reproducible same-sized draw. Preserve the gate outputs and reader sidecars. No GPU run is started by the CPU tests.

## Copy list for a launch

The `--copy` lists include the part scripts, selected development file for the test stage, and shared hook/checkpoint helpers. Keep `results/part99-pushes.pt`, the registered model snapshot, and generated stage files available in the job folder too. Reader/scorer scripts run after collection and do not require an A770. Resume by rerunning the same stage/output directory; completed cell-arm rows are retained.

The test stage is 528 answers (504 selected-cell arm answers and 24 unpushed answers). This follows the registered per-arm count; the design's “total” paragraph says 504 plus 24, while one later phrase compresses the stage-two total to 504.
