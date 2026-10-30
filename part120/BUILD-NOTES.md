# Part 120 build notes

## Decisions

- The runner imports `run119c` and uses its local-only float32 model loader and dose-cache builder. The selected rows are `rows119c.draw_path2_rows` with seed 11906. Path tokens, flip position, and logits slice follow `run119b.run_path2`.
- T is read only from the dose 0.8 positions in `results/part119c/interaction-float32.jsonl`. The runner requires T for every selected row and each selected position. Each NPZ `index` JSON object records the ascending dose list, every row-position entry and its `start`, plus float32 model and parameter dtype evidence. Values follow dose order and A, B, P order, and remain flat float64.
- The runner caches plain logits once and first-push logits once per first unit and dose. It batches second and both passes by `--batch`. Each pass is centered independently on T after float32 logits are copied to float64, then the plain centered vector is subtracted.
- Gate 1 compares full path logits from the plain pass and the same hook path with zero vectors, using `torch.equal`. Gate 3 checks neuron and random vectors for every adjacent dose pair. Its relative error is `norm(push(2a) - 2*push(a)) / norm(2*push(a))`; the threshold is 1e-5.
- Smoke gates record the selected unit scope. A full score requires the complete 39-path gate 1 and all 39 units for gate 3. A smoke writes to its own output directory and is not a scoreable full run.
- Gate 2 compares `nA`, `n2`, `nPair`, and `nI` at 0.05 through 0.8. Relative differences are summarized by the median over positions in each row, then by the median over selected rows. Every row and position must match and each statistic must be strictly below 0.5 percent.
- Analysis alphas end at 0.4 because the runner has no 1.6 pass. At 0.8 the scorer reports raw interaction norms for Gate 2; self terms, q, c, and shares are undefined. For analysis alphas, positions are eligible only when norms of P, S_A, and S_B are all positive. Per-dose dropped counts are reported.
- Unit `u(0.4)` is available only when all eight selected real and control rows have finite self shares. Mann-Whitney uses SciPy's one-sided `alternative="greater"` on Haiku slot and concept groups. The agreed-unit comparison must have the same direction. Part 119 dominance is read from `results/part119/tournament.json`.
- The bootstrap uses `score119.make_unit_bootstrap_draws`, 2,000 draws, seed 12001. Product weights are primary; sum weights are a sensitivity interval. A missing, malformed, zero-denominator, or non-finite value cannot pass a gate or prediction. Gate failures replace every prediction verdict with `not read (gate N)`. If q at 0.05 exceeds 0.30, the headline reading is withdrawn.
- No prediction, threshold, or dose was changed.

## Runtime copy list and launch

The runner imports the Part 119c loader, Part 119b Path 2 and cache code, Part 119 forward and row code, Part 92 model hooks, and Part 85's checkpoint registry. The scorer additionally imports Part 119's bootstrap helper and Part 86's registered triage rule. The copied JSON files are the unit inventory, both readers' kinds, saved answers, saved T and Gate 2 reference values, and the tournament dominance values.

```sh
tools/launch_arc.py 120 \
  --step "-I -B part120/run120.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part120 --device xpu --batch 8 --neurons results\\part92\\step2\\neurons.json --kinds results\\part92\\step2\\kinds.json --kinds-ds results\\part92b\\score92b.json" \
  --copy part120/run120.py part120/score120.py part119c/run119c.py part119c/rows119c.py part119b/run119b.py part119b/rows119b.py part119b/gate119b.py part119/run119.py part119/rows119.py part119/score119.py part92/common92.py part85/checkpoints.py part86/triage86.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part92b/score92b.json results/part119/answers.jsonl results/part119/tournament.json results/part119c/interaction-float32.jsonl
```

The one-unit, two-dose smoke command uses an isolated result directory:

```sh
tools/launch_arc.py 120 \
  --step "-I -B part120/run120.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part120-smoke --device xpu --batch 2 --units 1 --doses 0.05,0.1 --neurons results\\part92\\step2\\neurons.json --kinds results\\part92\\step2\\kinds.json --kinds-ds results\\part92b\\score92b.json" \
  --copy part120/run120.py part120/score120.py part119c/run119c.py part119c/rows119c.py part119b/run119b.py part119b/rows119b.py part119b/gate119b.py part119/run119.py part119/rows119.py part119/score119.py part92/common92.py part85/checkpoints.py part86/triage86.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part92b/score92b.json results/part119/answers.jsonl results/part119/tournament.json results/part119c/interaction-float32.jsonl
```

After copying the full run outputs back, score locally with:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part120/score120.py \
  --vectors results/part120/vectors \
  --reference results/part119c/interaction-float32.jsonl \
  --gate1 results/part120/gate1.json \
  --gate3 results/part120/gate3.json \
  --tournament results/part119/tournament.json \
  --out results/part120 \
  --neurons results/part92/step2/neurons.json \
  --kinds results/part92/step2/kinds.json \
  --kinds-ds results/part92b/score92b.json
```

## Verification

- Test command: `/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q -p no:cacheprovider part120/test_part120.py`.
- The suite exercises the quadratic and cubic identities, all gate boundaries, all eight prediction decisions and headline branches, missing and non-finite inputs, the NPZ round trip, resume and interruption behavior, isolated `-I -B` help from the repository and a copy-list tree, a real `run120.main([...])` call using the tiny CPU model, and a deliberate plus-sign break in a temporary scorer copy.
- The suite passed **19 tests**. This worker has CPU-only PyTorch (`2.14.1+cpu`); the pinned 1B checkpoint and A770 smoke were not run here. The end-to-end test exercised the runner with the tiny local model.
