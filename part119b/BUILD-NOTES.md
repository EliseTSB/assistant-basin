# Part 119b build notes

This build adds deterministic dose-tagged answer and fixed-path inventories, a resumable runner sharing Part 119's model, prompt, hooks, and generation helpers, and a CPU-only scorer. The inventory asserts 7,878 planned answer rows: 5,577 new rows and 2,301 reused Part 119 rows at 0.8. The seed-11902 pilot selects 529 answer rows (including the plain baseline), and 32 Path 2 rows. Main Path 2 has 2,964 directed rows across real and matched-random classes. Stable IDs carry the dose. The runner scales both neuron and random pushes from the 0.8 cache and checks against `part92/common92.py`'s direct push construction.

## Design details and choices

- Path 2 fixes each position's token set to the union of the four 0.8 pass top tens and uses that same set at all four doses. It records the centered norms, TV, five top IDs, margins, no-intercept slopes, and set size. Float records are rounded to five decimals; the saved resolved flag is calculated before rounding. Repeated argmax supplies top IDs without device top-k or sorting. Base and first-push forwards/top IDs are cached per unit and dose, while the second and both passes can batch by 1 or 8.
- The design says Gate 3 needs at least 70% coverage of real rows at 0.2 but leaves its denominator implicit. I count a row as covered when it has at least one position resolved on the shared support at all four doses. This keeps the coverage gate aligned with the all-dose support used for its median share. The scorer reports both row and position coverage; please review this interpretation.
- The design calls for pilot gate checks and timing but does not register numerical acceptance criteria for the sampled Gate 2 check. The pilot requires every sampled 0.8 row to match Part 119's row share within 0.01 and the median relative `nPair` difference to be below 1%; the complete scorer Gate 2 uses the design's all-row medians and coverage checks. Pilot stdout contains gate outcomes, timing, batch selection, and projected new-answer bytes, not prediction readouts.
- The model pilot and the by-eye 0.4 answer stopping rule were not run in this worker environment. The runner records both Path 2 batch timings and selected batch in `timing-pilot.json`; choose that batch for main. No dose or threshold was changed. Until actual run files are produced, `SCORE.md` and `score119b.json` correctly label Gate 1 as not run and all eleven predictions as not read; the 0.8 descriptive broken-answer cells come from the read-only Part 119 answers.
- Path 1 is the 741 real pairs at 0.4 and is reported as description only. The scorer also writes the requested plain per-dose table with broken shares by row class, all-position and resolved medians, median `b_A` and `b_2`, and row/position coverage.

## Launch commands

Run after Part 119b is registered and `tools/part_check.py prelaunch 119b` passes. Use the pinned model snapshot and one `--copy` flag containing the full dependency/input list. The A770 tool copies only these named files; do not repeat `--copy`.

Pilot:

```sh
python3 tools/launch_arc.py 119b \
  --step "-I -B part119b/run119b.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --stage pilot --batch 64 --out results\\part119b\\run --device xpu" \
  --copy part119b/run119b.py part119b/rows119b.py part119/run119.py part119/rows119.py part92/common92.py part85/checkpoints.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part92b/score92b.json results/part119/answers.jsonl results/part119/interaction.jsonl
```

After reading `results/part119b/run/timing-pilot.json`, run main with the selected `--path2-batch` value (1 or 8); keep the same output directory so pilot answers resume:

```sh
python3 tools/launch_arc.py 119b \
  --step "-I -B part119b/run119b.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --stage main --batch 64 --path2-batch 1 --out results\\part119b\\run --device xpu" \
  --copy part119b/run119b.py part119b/rows119b.py part119/run119.py part119/rows119.py part92/common92.py part85/checkpoints.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part92b/score92b.json results/part119/answers.jsonl results/part119/interaction.jsonl
```

If the pilot selects 8, use the same main command with `--path2-batch 8`. Once the A770 outputs are available under `results/part119b/run`, score on CPU with:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part119b/score119b.py
```

## Blind reader

Use Part 119's registered 12 known examples first for each reader. After both gates pass, label the 148-item 0.4 sample
for DeepSeek and the seed-11905 30-item subset for Luna. The prompts contain the original answer and X/Y persona
descriptions; the dose-tagged row IDs and unit/class key stay in the separate `.keys.jsonl` sidecar for scoring.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part119b/label119b.py --gate --reader ds --out results/part119b/known-ds.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part119b/label119b.py --gate --reader luna --out results/part119b/known-luna.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part119b/label119b.py --reader ds --out results/part119b/labels-ds.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part119b/label119b.py --reader luna --out results/part119b/labels-luna.jsonl
```

The scorer reads `labels-ds.jsonl.keys.jsonl`, both sample label files and known-example outputs, and Part 119's
`labels-ds.jsonl` plus key sidecar. It compares coherent labels only, paired by the same source row and class.

## Verification

- `/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q -p no:cacheprovider part119b/test_part119b.py part119/test_part119.py`: **49 passed** after the blind-reader extension. Tests use local fixtures and stub behavior; no reader service was called.
- CLI import smoke checks passed for both runner and scorer with `-I -B`; runner exposes pilot/main, generation batch, and Path 2 batch 1/8 options.
- Nine deliberate mutations were applied to disposable copies; each targeted test turned red as intended: dose-specific T, an intercept in the regression, one-dose-only contested mask, Gate 3 without coverage, headline branch 1 without 119b.1, resolved threshold 0, unscaled second push, reading 119b.9 below 70% coverage, and reading a verdict after a gate stopped its row.
- Six blind-reader regressions were applied to separate disposable copies; each targeted test turned red: reading a count after undefined constant-reader kappa was withheld, failing to withhold defined kappa 0.5, leaking the dose into the prompt, exposing row IDs, counting a missing label as incoherent, and lowering the 119b.1 reader threshold below 60%.
- `label119b.py --help` and `score119b.py --help` both passed isolated (`-I -B`) CLI import checks and expose the documented options.
- No pinned model, GPU, network, or paid service was used in this build. No model pilot, answer review, or main run has been performed.
- `python3 tools/part_check.py prelaunch 119b` exited 1 with 7 of 19 checks failed. Content checks passed. Failures were the absent `Registered YYYY-MM-DD HH:MM CT` line, the `{{REGISTERED` placeholder, no `## Part 119b` section or eleven values in local `PREDICTIONS.md`, a read-only `FETCH_HEAD` error during `git fetch origin main`, and absence of the design and predictions from `origin/main`. Registration is outside this builder's scope; the other worker is editing the shared predictions file. No `--override` launch was attempted. See the worker report for the exact command output summary.
