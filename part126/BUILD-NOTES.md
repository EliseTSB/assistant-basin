# Part 126 build notes

## Runtime copy list

The capture imports Part 83's condition list and Part 80/80b's prompt, rendering, and body helpers. Keep this exact copy list in the isolated import test and the launch command:

```text
part126/reads126.py
part126/lens126.py
part126/score126.py
part82/part82.py
part83/part83.py
part80/part80.py
part80b/part80b.py
part111/jlens111.py
neuron_report.py
part37/results/full/smol-gens.jsonl
```

The read runner reuses `part82` constants, `part83.conditions()`, Part 80's `question()` and `render()`, and Part 80b's `load_bodies()` and body check. It loads the Qwen model with the same `neuron_report.load_model()` call as Part 83. Prompt rows are grouped by exact token length. That permits batching without padding or shifting positions.

## Pilot and capture

Run the pilot on the A770 at batch sizes 1, 2, and 4. Run one launch at a time. Each pilot covers the first two bodies, 12 prompt rows. Copy the outputs back under the matching `results/part126/pilot-bN/` directory and compare `run.json` elapsed times. Confirm the positive control, zero-effect check, recomputation tolerance, region coverage, and the rendered prompts before selecting a batch size.

Run from the repository root. The `--copy` flag takes one flag followed by all paths.

```sh
tools/launch_arc.py 126 \
  --step "-I -B part126/reads126.py --out results\\part126\\pilot-b1 --device xpu --batch-size 1 --limit-bodies 2" \
  --copy part126/reads126.py part126/lens126.py part126/score126.py part82/part82.py part83/part83.py part80/part80.py part80b/part80b.py part111/jlens111.py neuron_report.py part37/results/full/smol-gens.jsonl

tools/launch_arc.py 126 \
  --step "-I -B part126/reads126.py --out results\\part126\\pilot-b2 --device xpu --batch-size 2 --limit-bodies 2" \
  --copy part126/reads126.py part126/lens126.py part126/score126.py part82/part82.py part83/part83.py part80/part80.py part80b/part80b.py part111/jlens111.py neuron_report.py part37/results/full/smol-gens.jsonl

tools/launch_arc.py 126 \
  --step "-I -B part126/reads126.py --out results\\part126\\pilot-b4 --device xpu --batch-size 4 --limit-bodies 2" \
  --copy part126/reads126.py part126/lens126.py part126/score126.py part82/part82.py part83/part83.py part80/part80.py part80b/part80b.py part111/jlens111.py neuron_report.py part37/results/full/smol-gens.jsonl
```

After choosing the fastest passing pilot, copy that pilot's `run.json`, `rows.jsonl`, `batch_*.npz`, and `reads126.npz` into `results/part126/`. The full runner accepts a completed prefix from the first two bodies, checks the prompt hashes, and resumes the remaining rows. The complete capture is 600 prompt rows, 100 bodies × 2 copies × 3 orders. At Part 83's measured 275 passes/minute, that is about 2.2 minutes before this runner's hook overhead, plus two no-op comparison passes and one positive-control pass.

```sh
tools/launch_arc.py 126 \
  --step "-I -B part126/reads126.py --out results\\part126 --device xpu --batch-size SELECTED_BATCH_SIZE" \
  --copy part126/reads126.py part126/lens126.py part126/score126.py part82/part82.py part83/part83.py part80/part80.py part80b/part80b.py part111/jlens111.py neuron_report.py part37/results/full/smol-gens.jsonl
```

Replace `SELECTED_BATCH_SIZE` with the fastest batch size that passed the pilot gates. Keep the run files and compact batch shards together so a stopped capture resumes. `rows.jsonl` records only prompt metadata. `reads126.npz` contains the compact per-head tensors; it does not expand token weights into JSON.

## Lens and score

After copying the full capture back to the repository, run the lens on CPU. By default, Part 111 supplies the pinned local snapshot and lens paths. Use `--snapshot` if the snapshot is mounted elsewhere. The output contains top and suppressed tokens for each head's role-token write, most-attended-region write, and same-head random body-token control. The vector file keeps mean residual writes for the raw scorer.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part126/lens126.py \
  --activations results/part126/reads126.npz \
  --rows results/part126/rows.jsonl \
  --run-json results/part126/run.json \
  --out results/part126/lens

/home/elisepc/neuron-probe/.venv/bin/python -I -B part126/score126.py \
  --reads-dir results/part126 \
  --lens-dir results/part126/lens \
  --part83 results/part83/main/results83.json \
  --out results/part126/score
```

The scorer writes `SCORE.md` and `score126.json`. Any failed gate leaves all four predictions marked `not read`. It prints regional attention masses by head and copy and records the role-write versus random-body-token top-20 overlap as a control summary.

## Tests

The isolated end-to-end test copies only the runtime copy list above, then runs the real main of all three scripts on a tiny stub. Four parameterized cases break each gate. One test checks resumable prefix capture and recovery from a torn final JSONL line. Another test deliberately breaks the recomputation and zero-effect gates in temporary copies and confirms the good-gate test goes red twice. Those altered copies are temporary and are removed by the test.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q -p no:cacheprovider part126/test_part126.py
```

The stub positive control adds a known bias only in its test fixture. It validates the end-to-end gate plumbing, not the real model's positive-control outcome. No real model capture or lens run was done by this CPU-only builder.
