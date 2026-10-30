# Part 119c build notes

## Open design details and decisions

- The design asks for four real and four random second pushes per first unit but does not say whether the two classes share second-unit IDs. The sampler draws four IDs once per first unit from the intersection of Part 119b's real/control Path 2 candidates, then emits both classes for each ID. First units and candidate IDs are sorted before `random.Random(11906).sample`, giving 39 × 8 = 312 rows with matched controls.
- The design specifies 312 rows but not their ordering. The runner writes rows in sorted first-unit order, then selected-sequence order, with real/control adjacent. Tests pin the resulting inventory digest.
- `run_path2` has no membership check against `rows119b.DOSES`: `path2_key` accepts arbitrary float doses, and support is built from 0.8 rows. The runner calls it directly with `(0.8, 0.4, 0.2, 0.1, 0.05)`; no 119b adaptation was needed.
- The gate language does not spell out the aggregation for "median relative nPair difference." Following Part 119b's Gate 2 helper, each selected row's relative nPair errors are medianed across shared positions, then row medians are medianed per required dose. Gate 1 requires complete selected-row and position support and each of 0.2, 0.4, and 0.8 to be strictly below 1%; Gate 2 uses the same calculation at 0.8 and a strict 5% limit.
- The bootstrap prediction names the difference in median row shares, while Part 119's reusable unit bootstrap computes a mean of row values for its interval. Prediction 119c.1 therefore reports the difference of the two median row shares for its 0.04 clause and uses Part 119's exact 2,000-draw bootstrap on paired per-row float32-minus-bf16 differences for the 0.06 interval clause. Product weighting is primary; sum weighting is also reported as the registered sensitivity check.
- "Within a factor of 1.25" is evaluated symmetrically as `max(real/control, control/real) <= 1.25`; both zero shares count as equal, while a zero versus positive or missing value does not match.
- The per-dose nI, nPair, nA, and n2 values are pooled position medians. Position share uses every position with nPair > 0; row share is the median of eligible positions in each row; the summary is the median of row shares. Gates run before prediction verdicts; any failed gate stops all four verdicts and the headline.
- The runner's local loader follows `common92.load_model` arguments and local-only behavior while passing the selected `torch_dtype`. Each dose cache is built directly from the loaded model's own weights. The local run has no A770, so the registered model experiment was not executed here.

## Runtime copy list and launch

The A770 job needs the runner dependency chain, scorer dependencies, and local registered inputs. The scorer does not import reader labeling code. The runner imports `run119b` only for its Path 2 implementation and push construction; it does not import the Part 119b scorer or reader modules.

```sh
tools/launch_arc.py 119c \
  --step "-I -B part119c/run119c.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part119c --device xpu --batch 8" \
  --copy part119c/run119c.py part119c/rows119c.py part119c/score119c.py part119b/run119b.py part119b/rows119b.py part119b/gate119b.py part119/run119.py part119/rows119.py part119/score119.py part92/common92.py part85/checkpoints.py part86/triage86.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part92b/score92b.json results/part119/answers.jsonl results/part119b/interaction.jsonl
```

The copy list above is exercised by the test suite: the runner and CPU scorer are copied into a temporary tree containing exactly these files, then each imports successfully under `python -I -B ... --help`.

## Verification

- Test command: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q -p no:cacheprovider part119c/test_part119c.py`.
- The test suite covers the exact 312-row deterministic sample, both gates and their dose boundaries, all four prediction decisions, all headline branches, float32 push-cache dtype, and an OLMo2 tiny CPU Path 2 run at five doses in bf16 and float32.
- The clean test run has **11 passing tests**. Four deliberate regressions were applied separately to disposable copies and each target test turned red: Gate 1 omits the registered 0.2 dose; float32 pushes are cast to bf16; the slope includes 0.4 and 0.8; and headline branch 1 checks only 119c.1. The worker report records these outcomes.
- No model inference on the registered checkpoint, A770 run, network call, paid call, reader labeling, or Part 119b/results modification was performed in this CPU build.
