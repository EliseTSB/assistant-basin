# Part 119c build notes

## Open design details and decisions

- The design asks for four real and four random second pushes per first unit but does not say whether the two classes share second-unit IDs. The sampler draws four IDs once per first unit from the intersection of Part 119b's real/control Path 2 candidates, then emits both classes for each ID. First units and candidate IDs are sorted before `random.Random(11906).sample`, giving 39 × 8 = 312 rows with matched controls.
- The design specifies 312 rows but not their ordering. The runner writes rows in sorted first-unit order, then selected-sequence order, with real/control adjacent. Tests pin the resulting inventory digest.
- `run_path2` has no membership check against `rows119b.DOSES`: `path2_key` accepts arbitrary float doses, and support is built from 0.8 rows. The runner calls it directly with `(0.8, 0.4, 0.2, 0.1, 0.05)`; no 119b adaptation was needed.
- The gate language does not spell out the aggregation for "median relative nPair difference." Following Part 119b's Gate 2 helper, each selected row's relative nPair errors are medianed across shared positions, then row medians are medianed per required dose. Gate 1 requires complete selected-row and position support and each of 0.2, 0.4, and 0.8 to be strictly below 1%; Gate 2 uses the same calculation at 0.8 and a strict 5% limit.
- Prediction 119c.1 uses one statistic throughout: the difference of the float32 and bf16 median row shares at 0.2. Each of Part 119's 2,000 unit-bootstrap draws recomputes both medians with product weights; the sum-weighted interval is reported beside it as a sensitivity check. The headline branch uses the same product-weighted interval.
- The runner imports `rows119b` for the Path 2 candidate inventory. Each dtype has a separate JSONL output. Every record includes the canonical `model_dtype` and the loaded parameter dtype (`model_parameter_dtype`); a resume refuses existing rows without matching evidence. The scorer requires bf16/`torch.bfloat16` and float32/`torch.float32`, and Gate 2 fails if all matched 0.8 position-level nPair values are identical.
- The scorer's 119c.3 is the endpoint slope `log(S(0.2)/S(0.05)) / log(4)`. Per-dose rows report median nI, median nPair, and the share of positions with nPair below 1.0. If a gate fails, the table remains visible under a heading that says it is not a reading.
- "Within a factor of 1.25" is evaluated symmetrically as `max(real/control, control/real) <= 1.25`; both zero shares count as equal, while a zero versus positive or missing value does not match.
- The per-dose nI, nPair, nA, and n2 values are pooled position medians. Position share uses every position with nPair > 0; row share is the median of eligible positions in each row; the summary is the median of row shares. Gates run before prediction verdicts; any failed gate stops all four verdicts and the headline.
- The runner's local loader follows `common92.load_model` arguments and local-only behavior while passing the selected `torch_dtype`. Each dose cache is built directly from the loaded model's own weights. The local run has no A770, so the registered model experiment was not executed here.

## Runtime copy list and launch

The A770 job needs the runner dependency chain, scorer dependencies, and local registered inputs. The scorer does not import reader labeling code. The runner imports `run119b` only for its Path 2 implementation and push construction; it does not import the Part 119b scorer or reader modules.

```sh
tools/launch_arc.py 119c \
  --step "-I -B part119c/run119c.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part119c --device xpu --batch 8 --neurons results\\part92\\step2\\neurons.json --kinds results\\part92\\step2\\kinds.json --kinds-ds results\\part92b\\score92b.json" \
  --copy part119c/run119c.py part119c/rows119c.py part119c/score119c.py part119b/run119b.py part119b/rows119b.py part119b/gate119b.py part119/run119.py part119/rows119.py part119/score119.py part92/common92.py part85/checkpoints.py part86/triage86.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part92b/score92b.json results/part119/answers.jsonl results/part119b/interaction.jsonl
```

After the A770 output is copied back, score it locally with:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part119c/score119c.py \
  --bf16 results/part119c/interaction-bf16.jsonl \
  --float32 results/part119c/interaction-float32.jsonl \
  --reference results/part119b/interaction.jsonl \
  --out results/part119c
```

The launch command and `--copy` list above describe the runtime tree: it includes the new `rows119b` import through `part119b/rows119b.py`. The copy-list test checks `--help` for both scripts under `sys.executable -I -B`, then imports the copied runner and scorer from that exact tree and drives `run()` on a two-unit OLMo2 CPU model at all five doses in both dtypes. It scores those output rows. This reaches the real `run()` path without a GPU; it does not run the registered 1B checkpoint or exercise A770 kernels.

## Verification

- Test command: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q -p no:cacheprovider part119c/test_part119c.py`.
- The test suite covers the exact 312-row deterministic sample, both gates and their dose boundaries, all four prediction decisions, all headline branches, float32 push-cache dtype, the complete `run()` path and restart, both bootstrap weightings, the endpoint slope, gate-failure table wording, low-nPair shares, and the exact copy-list CPU smoke run.
- The clean test run has **16 passing tests**. Six focused tests were red on a disposable `HEAD` copy before the changes: Gate 2 accepted identical nPair files, the old mean-bootstrap result did not expose the median statistic, low-nPair and gate-failure table outputs were absent, the old 3-point slope included dose 0.1, and the runner raised the missing-`rows119b` `NameError`. The copy-list test also runs the copied runner and scorer through the tiny CPU path.
- No model inference on the registered checkpoint, A770 run, network call, paid call, reader labeling, or Part 119b/results modification was performed in this CPU build.
