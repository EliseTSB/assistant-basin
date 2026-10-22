# Part 50 build report

## SUMMARY

Implemented the Part 50 runner and CPU tests, then froze the 51-trait bundle before model inference. The Qwen3-1.7B CPU smoke completed on three train rows, three held-out lie rows, and three persona rows across 42 arms. The worker has a CPU-only PyTorch build, so the full Arc runs and model-level scores were not produced here.

## FINDINGS

### What shipped

- `part50.py` provides `bundle`, `direction`, `calibrate`, `run`, and `score` subcommands. The direct loader pins the local cache snapshots and streams bf16 weights to the requested device. It detects both `/mnt/e/neuron-probe/hf/hub` and `E:\neuron-probe\hf\hub`.
- `data/part50-bundle.json` freezes 51 Qwen3.5-9B traits selected by `q_perm <= 0.05`, with each sign taken from `T_set`, plus source path and hash. Source: `third_party/honesty-heads/results/persona_stats.json`; SHA-256 `dbbb3276b39d6f8d670bd1b4e9a1f760b54056605c38cc22a6f52c188b2c9c85`.
- Model revisions resolved from the cache: Qwen3-1.7B `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; Qwen3-1.7B-Base `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- The direction captures the `o_proj` input slices and averages the first nine teacher-forced answer-prediction positions for each provided completion. The test readout compares the first differing truth/lie completion token; this avoids a zero-valued choice when both completions begin with the same token. Persona Yes/No candidates are retokenized for each Qwen tokenizer because the vendored IDs are Qwen3.5-specific.
- Run arms are unsteered, positive honesty, negative honesty, and 39 seeded random directions. Each random direction preserves every head’s norm. The negative arm follows `STANDARD-CONTROLS.md`; FDR and bundle ranking compare positive honesty with the 39 random directions only.
- Persona scoring reports the yes/no split, Sauers' permutation FDR, and a confidence-adjusted FDR. The confidence adjustment is an explicit implementation choice: regress each arm’s item-level change on the centered unsteered matching-answer log odds, then subtract the fitted slope term. `persona_stats.py` gives the yes/no balancing and permutation-FDR formulas; the README describes confidence-trend removal but does not specify its regression code.

### Verification and timings

- `python -B -m unittest -v test_part50`: 9 tests passed. A synthetic full persona summary covered all 135 traits and 39 random scores.
- Four `/tmp` sabotages exited red as intended: hook addition, random per-head norms, bundle immutability, and FDR parity.
- CPU smoke command: `/home/elisepc/neuron-probe/.venv/bin/python -B part50.py smoke --model qwen3-1.7b --device cpu --limit 3 --target-bits 0.25 --out /tmp/part50-cpu-smoke-optimized --batch-size 8`. Timings: direction 8.5 s, calibration 31.0 s, 42-arm run 44.3 s, total 98.3 s. The smoke calibration reached 0.2404 bits at `s=0.06640625`; the full-run default remains 2.0 bits.
- A final loader check extracted one direction pair in 4.6 s. It used the cached Qwen3-1.7B snapshot.
- The installed PyTorch is `2.14.1+cpu`; `torch.xpu.is_available()` is false. No full model `scores.json` files were generated. The score command writes them under `results/part50/<model>/scores.json` after full runs.

### Arc commands

The frozen bundle is included in `data/part50-bundle.json`; these commands assume it is present and run from the repository root. Run each line separately, in order.

```sh
/home/elisepc/neuron-probe/.venv/bin/python part50.py direction --model qwen3-1.7b --device xpu
/home/elisepc/neuron-probe/.venv/bin/python part50.py calibrate --model qwen3-1.7b --device xpu --target-bits 2.0
/home/elisepc/neuron-probe/.venv/bin/python part50.py run --model qwen3-1.7b --device xpu --random-directions 39 --seed 50050
/home/elisepc/neuron-probe/.venv/bin/python part50.py direction --model qwen3-1.7b-base --device xpu
/home/elisepc/neuron-probe/.venv/bin/python part50.py calibrate --model qwen3-1.7b-base --device xpu --target-bits 2.0
/home/elisepc/neuron-probe/.venv/bin/python part50.py run --model qwen3-1.7b-base --device xpu --random-directions 39 --seed 50050
/home/elisepc/neuron-probe/.venv/bin/python part50.py score --model both
```

## BLOCKERS

The worker cannot run on Arc because its PyTorch build has no XPU backend. This blocks full model scoring in this environment, but the requested implementation, tests, bounded CPU smoke, and Arc commands are complete.

Parent-control mailbox reads returned no messages at any checkpoint; no dispositions were needed.
