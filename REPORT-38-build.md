# Part 38 build report

## Shipped pieces

- `part38.py:140` and `part38.py:163` implement a generated-token-only down-projection clamp and removable pre-hook. Prompt prefill is skipped; cached one-token decoding and uncached prompt-plus-continuation forwards are handled.
- `part38.py:213` and `data/part38-homographs.json:1` validate and supply nine homograph pairs, 12 examples per sense. `part38.py:382` locates the target's offset-mapped token span; `part38.py:426`, `part38.py:479`, and `part38.py:513` load the pinned local snapshot and capture activations.
- `part38.py:626` and `part38.py:675` write resumable homograph rows and per-unit median ratios, seeded 2,000-resample CIs, and random-unit ratio distributions.
- `part38.py:252`, `data/part38-neighbours.json:1`, `part38.py:731`, and `part38.py:776` implement the added wizard-neighbor contrast, including Finnish `noita`, ratio CIs, and random-unit nulls.
- `part38.py:822` and `part38.py:872` instantiate the 36 Part 35 neutral frames and write resumable content-token natural maxima for each target and its same-layer controls.
- `part38.py:972`, `part38.py:1007`, and `part38.py:1037` score fixed topic prefixes with Wilson intervals and run the target, 20-control, and no-clamp arms. Generation uses temperature 0.7, top-p 1, 80 tokens, and four samples per prompt.
- `part38.py:1137` writes shuffled answer text to `blind-pack.jsonl` and keeps condition labels in `blind-pack-key.json`. `part38.py:1191` defines all five subcommands and `--limit` / `--units` options.
- `test_part38.py:47`–`test_part38.py:107` cover split-token offsets, prompt-safe clamping with a tiny fake model, deterministic controls, topic word starts, reproducible CIs, and both authored datasets.

## Data checks

The validator confirmed 9 units × 12 sentences × 2 senses, one whole-word target per sentence, and 16 neutral push prompts without a target or fixed topic term. The largest absolute difference in mean sentence length is 1.17 words (`anthropic`); all nine pairs are within 1.5 words.

| Unit | Screened mean words | Other-sense mean words |
|---|---:|---:|
| wizard | 13.42 | 12.83 |
| train | 12.67 | 12.58 |
| bee | 12.92 | 12.50 |
| star | 12.33 | 12.08 |
| mountain | 12.58 | 12.50 |
| trump | 11.67 | 12.58 |
| claude | 11.58 | 11.58 |
| anthropic | 10.83 | 12.00 |
| volcano | 12.00 | 13.00 |

The neighbor dataset has 11 word groups × 12 sentences. Each group word occurs once per sentence, and the `noita` group is Finnish. Part 35's five-language templates are instantiated into 36 content-masked neutral passages for `natural-max`.

## Verification

Command: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part38.py`

Output: `7 passed in 0.09s`; no tests skipped. A deliberate mutation made the prefill mask clamp every prompt position. The focused generated-only clamp test went red (exit 1, showing the prompt coordinate changed to 10.0); I restored the mask and reran the full suite, which passed 7/7.

No real model was loaded and no files were downloaded, as required. Offset handling and the generation hook were tested with fake components; runtime compatibility with the pinned tokenizer/model remains unobserved here.

## Full-run commands

Run from `/home/elisepc/green/worktrees/ab-part38-build`. The default unit set is all nine for `homograph`, wizard only for `neighbours`, and the seven units with preregistered topic lists for `natural-max` and `push`. Run `natural-max` before `push`.

### CPU, four threads

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/elisepc/neuron-probe/.venv/bin/python part38.py homograph --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device cpu --dtype bf16 --threads 4 --seed 3801 --out results/part38-cpu
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/elisepc/neuron-probe/.venv/bin/python part38.py neighbours --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device cpu --dtype bf16 --threads 4 --seed 3801 --out results/part38-cpu
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/elisepc/neuron-probe/.venv/bin/python part38.py natural-max --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device cpu --dtype bf16 --threads 4 --seed 3801 --out results/part38-cpu
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/elisepc/neuron-probe/.venv/bin/python part38.py push --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device cpu --dtype bf16 --threads 4 --seed 3801 --k 4 8 --out results/part38-cpu
/home/elisepc/neuron-probe/.venv/bin/python part38.py blind-pack --seed 3801 --out results/part38-cpu
```

### XPU

```sh
/home/elisepc/neuron-probe/.venv/bin/python part38.py homograph --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device xpu --dtype bf16 --threads 4 --seed 3801 --out results/part38-xpu
/home/elisepc/neuron-probe/.venv/bin/python part38.py neighbours --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device xpu --dtype bf16 --threads 4 --seed 3801 --out results/part38-xpu
/home/elisepc/neuron-probe/.venv/bin/python part38.py natural-max --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device xpu --dtype bf16 --threads 4 --seed 3801 --out results/part38-xpu
/home/elisepc/neuron-probe/.venv/bin/python part38.py push --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device xpu --dtype bf16 --threads 4 --seed 3801 --k 4 8 --out results/part38-xpu
/home/elisepc/neuron-probe/.venv/bin/python part38.py blind-pack --seed 3801 --out results/part38-xpu
```

## Four-core CPU time estimates

These are planning estimates, not measurements. They assume the pinned BF16 model is cached and usable on CPU.

| Subcommand | Estimated time |
|---|---:|
| `homograph` (216 sentence forwards) | 5–15 minutes |
| `neighbours` (132 sentence forwards) | 3–10 minutes |
| `natural-max` (36 passage forwards) | 1–5 minutes |
| `push` (19,264 generations; up to 1.54M output tokens) | 24–110 hours, assuming roughly 4–18 generated tokens/second |
| `blind-pack` | under 1 minute |

## Parent control

Control message 1 added the wizard `neighbours` assay and prediction 38.6. It was applied: separate data, command, ratios and CIs, random controls, and a test were added. No messages were declined or deferred.

SUMMARY: Shipped the Part 38 homograph, neighbor, natural-max, push, scoring, and blind-pack builder with authored data and tests.
FINDINGS: 7/7 tests pass; target spans, lengths, counts, neutral prompts, and neighbor groups validate. The deliberate prompt-clamp mutation failed the focused test and was restored.
BLOCKERS: None. Full-model execution was intentionally not performed under the brief; CPU timings are estimates.
