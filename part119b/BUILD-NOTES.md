# Part 119b build notes

This build adds deterministic dose-tagged answer and fixed-path inventories, a resumable runner sharing Part 119's model, prompt, hooks, and generation helpers, and a CPU-only scorer. The inventory asserts 7,878 planned answer rows: 5,577 new rows and 2,301 reused Part 119 rows at 0.8. The seed-11902 pilot selects 529 answer rows (including the plain baseline), and 32 Path 2 rows. Main Path 2 has 2,964 directed rows across real and matched-random classes. Stable IDs carry the dose. The runner scales both neuron and random pushes from the 0.8 cache and checks against `part92/common92.py`'s direct push construction.

## Design details and choices

- Path 2 fixes each position's token set to the union of the four 0.8 pass top tens and uses that same set at all four doses. It records the centered norms, TV, five top IDs, margins, no-intercept slopes, and set size. Float records are rounded to five decimals; the saved resolved flag is calculated before rounding. Repeated argmax supplies top IDs without device top-k or sorting. Base and first-push forwards/top IDs are cached per unit and dose, while the second and both passes can batch by 1 or 8.
- The design says Gate 3 needs at least 70% coverage of real rows at 0.2 but leaves its denominator implicit. I count a row as covered when it has at least one position resolved on the shared support at all four doses. This keeps the coverage gate aligned with the all-dose support used for its median share. The scorer reports both row and position coverage; please review this interpretation.
- The design calls for pilot gate checks and timing but does not register numerical acceptance criteria for the sampled Gate 2 check. The pilot requires every sampled 0.8 row to match Part 119's row share within 0.01 and the median relative `nPair` difference to be below 1%; the complete scorer Gate 2 uses the design's all-row medians and coverage checks. Pilot stdout contains gate outcomes, timing, batch selection, and projected new-answer bytes, not prediction readouts.
- Part 119's `results/part119/interaction.jsonl` has no `resolved` field; Gate 2 compares shares and `nPair` across all common positions on each selected row.
- The pilot gate checks and by-eye stopping rule were completed, and the main run, fixed-path scoring, and reader labels are recorded under `results/part119b/`. The pilot selected Path 2 batch 8 from the timings in `timing-pilot.json`; no dose or threshold changed. Gate 3 failed in the main run, so predictions 119b.4, .5, .7, and .8 remain stopped, not read. The pilot and main-run by-eye records are in `results/part119b/README.md`.
- Path 1 is the 741 real pairs at 0.4 and is reported as description only. The scorer also writes the requested plain per-dose table with broken shares by row class, all-position and resolved medians, median `b_A` and `b_2`, and row/position coverage.

## Launch commands

Run after Part 119b is registered and `tools/part_check.py prelaunch 119b` passes. Use the pinned model snapshot and one `--copy` flag containing the full dependency/input list. The A770 tool copies only these named files; do not repeat `--copy`.

Pilot:

```sh
python3 tools/launch_arc.py 119b \
  --step "-I -B part119b/run119b.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --stage pilot --batch 64 --out results\\part119b\\run --device xpu" \
  --copy part119b/run119b.py part119b/rows119b.py part119b/gate119b.py part119/run119.py part119/rows119.py part92/common92.py part85/checkpoints.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part92b/score92b.json results/part119/answers.jsonl results/part119/interaction.jsonl
```

After reading `results/part119b/run/timing-pilot.json`, run main with the selected `--path2-batch` value (1 or 8); keep the same output directory so pilot answers resume:

```sh
python3 tools/launch_arc.py 119b \
  --step "-I -B part119b/run119b.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --stage main --batch 64 --path2-batch 1 --out results\\part119b\\run --device xpu" \
  --copy part119b/run119b.py part119b/rows119b.py part119b/gate119b.py part119/run119.py part119/rows119.py part92/common92.py part85/checkpoints.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part92b/score92b.json results/part119/answers.jsonl results/part119/interaction.jsonl
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
- This fold worker used no pinned model, GPU, network, or paid service. The pilot, by-eye review, and main run were completed separately and are documented in `results/part119b/README.md`.
- Historical prelaunch record: `python3 tools/part_check.py prelaunch 119b` exited 1 with 7 of 19 checks failed. Content checks passed. At that time, the design lacked the required registration stamp, local `PREDICTIONS.md` did not yet contain the Part 119b cells, and `git fetch origin main` failed with a read-only `FETCH_HEAD` error; the design and predictions were also absent from `origin/main`. The shared prediction table now includes the Part 119b cells. No `--override` launch was attempted.

Part 119b stop rule, written 10-30 06:51 before DS r2: if r2 finds another defect in how a stored row and a replay agree, stop folding and move T into a separate first stage that writes a supports file every dose reads.

## Fold r1 review: pre-existing follow-ups

- Part 86's `WIDER_SELF_DESCRIPTION` rule keeps any first sentence that merely contains a self-reference term such as “model” or “assistant”; this is inherited by Part 119b's answer triage and remains with Part 86's owner.
- Part 119's `run119.read_existing` rewrites an unterminated malformed final JSONL line while reading; Part 119b inherits this behavior for canonical answer and interaction inputs, so the read-only A770 mount limitation remains with Part 119's owner.

Fold after the first pilot launch (2026-10-30): the pilot refused at validate_dose_scaling, because a bf16 push scaled by dose/0.8 rounds differently from the direct push by up to one step. Each dose is now built directly from the weights with Part 92's functions, and the runtime guard checks size and direction. No row ran before the refusal.
