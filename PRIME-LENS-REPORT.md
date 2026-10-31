# PRIME-LENS-REPORT — Neuron Lens port to OLMo 2 1B

Status: **complete** — port, tests, deliberate-break harness, the 8-document
acceptance fit (CPU, memory-capped) and the report. Cost projections use the
measured seconds per document and are marked as measured or projected.

- Source: `github.com/stdlogits/neuron-lens` at commit **`07fdf45`**, MIT
  (copyright 2026 a9lim and treetowntree), vendored under `neuron_lens/` with
  `LICENSE` and `VENDORED.md`.
- Branch/worktree: `prime-olmo-lens` in `~/green/worktrees/ab-olmo-lens`
  (a worktree of `~/assistant-basin`), based on `main` `ed90e20`.
- Model: `allenai/OLMo-2-0425-1B-Instruct` at revision
  `48d788eca847d4d7548f375ad03d3c9312f6139e`, from the local HF cache
  `/mnt/e/neuron-probe/hf/hub` (`local_files_only`), bf16, eager attention.
- Host: Ryzen 7 9700X (8 cores / 16 threads, AVX512-BF16), 11 GB RAM, CPU only
  (no A770, no GPU, no network in the fit). Torch 2.14.1+cpu,
  transformers 5.20.0, Python 3.12.3.
- Constraint: the fit runs in a cgroup v2 scope with `MemoryMax=8G`,
  `nice -n 15`, `OMP_NUM_THREADS=12`.

## 1. What differs from Qwen3 (and why a port, not a config)

`transport._linearize` hand-writes Qwen3's decoder forward. OLMo 2 does not fit
that shape, and the differences are silent if a map is reused:

| | Qwen3 (Neuron Lens' first model) | OLMo 2 1B |
|---|---|---|
| norm placement | **pre-norm**: `input_layernorm` before attention, `post_attention_layernorm` before the MLP | **post-norm**: norms sit on the attention and MLP **outputs** before the residual add; no input norm and no pre-FF norm |
| `mid` (the MLP input) | the post-attention RMSNorm's output | the **unnormalised** post-attention residual `h1 = h + post_attention_layernorm(attn(h))` |
| residual norms under RelP | 2 norms, each linearised on the way *in* | 2 norms, each linearised on the way *out* (`a = weight · rsqrt` multiplies the module output) |
| q/k norms | per head: RMSNorm of width `head_dim` (128) | **one full-width** RMSNorm over `heads · head_dim` (2048), applied before the head view |
| attention | 32 query / 8 KV heads (GQA 4:1) | 16 query / 16 KV heads (MHA) |
| `head_dim` | 128 (`hidden / heads`) | 128 (same) |
| biases | none | none |
| RoPE | full head dim | full head dim, `theta = 500000` |
| embeddings / output | untied | untied (`lm_head` separate) |
| MLP | SwiGLU, SiLU | SwiGLU, SiLU (same identity + half rules) |

Consequences for the estimator:

- The layer map gains one term that does not exist in Qwen3: the
  **post-feedforward norm's held-constant scale** `a_pf` on the MLP output, and
  the **post-attention norm's scale** `a_pa` on the attention output. The
  half-rule factors `cg`/`cu` are unchanged.
- `mid` is a residual-stream point, not a norm output — the same conceptual
  point ("what the MLP reads"), a different linear algebra.
- The q/k norms' `rsqrt` is taken over 2048 values, not 128; the true-derivative
  JVP/VJP is the same formula at a different width.
- The vendored drivers (`s_lens_doc`, `r_lens_doc`, `input_reads_doc`) probed
  the map's shape through `a1` (Qwen's input norm). They now read `N`, `d` and
  the device from `q`/`wq`, which every family's map has (see `VENDORED.md`).

## 2. Tests (goal items 1 and 2)

The port adds `tests/test_olmo2.py`, modelled on `tests/test_transport.py`'s
brute-force test: tiny random OLMo 2 stacks (one MHA, one GQA with random norm
weights), both lens directions and both targets, compared against
`torch.autograd.functional.jacobian` of the model **under `relp_rules`** at
`atol=1e-4`.

```
$ PYTHONPATH=$PWD pytest tests/test_olmo2.py -q
9 passed
$ PYTHONPATH=$PWD pytest tests/ -q            # everything, port tests included
43 passed, 3 skipped
$ PYTHONPATH=$PWD pytest tests/test_arch.py tests/test_gemma4.py \
      tests/test_readout.py tests/test_transport.py -q     # upstream only
34 passed, 3 skipped                          # skips are the optional-drowse tests
```

Beyond the Jacobian comparison, `tests/test_olmo2.py` has a **bf16 end-to-end
test** (the fit's dtype) and a **dispatch guard**: `transport._maps` must route
an OLMo 2 stack to these maps, and the OLMo map must refuse a Qwen3 stack by
name. The bf16 test exists because of a real hazard: OLMo 2's rotary table is
fp32 while Qwen3's is already the model dtype, so the rotated q/k need HF's
explicit cast back to the input dtype. The fp32 tiny-model Jacobian tests cannot
see that class — it only appears on the real bf16 model.

The deliberate-break harness (`test_olmo_lens_sabotage.py`) copies the package
to a throwaway directory and applies one break at a time, then runs the same
tests against the copy:

```
control: unmodified copy green (9 passed)
break jvp-post-ff-norm-dropped: red on ('s', ...)      # S-lens forward mode
break vjp-post-ff-norm-dropped: red on ('r-out', ...)  # R-lens reverse mode
SABOTAGE HARNESS: OK
```

The harness is two-directional on purpose: the unmodified copy must stay green,
and each break must fail on the **assertion for its own lens** (the assertion
messages are unique: `('s', kind, layer)`, `('r-out', layer)`, `('r-mid', layer)`),
not merely return a nonzero exit code.

## 3. The acceptance fit

One pass per document covers every layer and both targets at no extra cost; the
acceptance layer is **L8** (of 16), both lenses, `out` + `mid`.

- Documents: 8, selected from
  `/mnt/e/neuron-probe/corpus/wikitext103-train0.parquet` by the same seeded
  shuffle and exactly-128-token filter as `fit_jlens_olmo.py` (seed 5701;
  corpus sha256 `74da360f…`; rows `[463790, 396190, 12299, 250585, 535644,
  402452, 249901, 811815]`).
- Command (under the scope):

```
systemd-run --user --scope -p MemoryMax=8G -- nice -n 15 \
  env OMP_NUM_THREADS=12 PYTHONPATH=$PWD \
  ~/neuron-probe/.venv/bin/python fit_olmo_lens.py --docs 8 --threads 12 --dim-batch 64
```

`dim_batch` is 64, not the upstream default 256: the scope's `MemoryPeak` was
**7.74 GB of the 8 GB cap at 256** and **6.88 GB at 128** (measured during the R
pass), so the run uses 64 for ~2 GB of headroom across both passes. The knob
changes only how many basis directions go through the maps per block — same
mathematics, more blocks.

Fit results are in `fit.json` (`results/prime-lens/olmo2-1b-instruct/`), with
per-document seconds for R and S separately, peak RSS, and the split-half
difference per target and layer. The scope's `MemoryPeak` was **6.83 GB of the
8 GB cap** (the process's own peak RSS 6.25 GB); `n_docs` = [4, 4].

Split-half at L8 (`||mean_even − mean_odd|| / ||mean||`): R_out
0.555, R_mid 0.508,
S_out 0.413, S_mid 0.421.
With four documents a half that is sampling noise, not a lens-quality figure:
their 1,000-document lenses split at 0.06 or less. The acceptance fit measures
the machinery and its cost; it is not a lens.

| document | corpus row | R seconds | S seconds | total | peak RSS |
|---|---|---|---|---|---|
| 0 | 463790 | 774.8 | 740.4 | 1515.2 | 6.24 GB |
| 1 | 396190 | 983.4 | 598.8 | 1582.2 | 6.25 GB |
| 2 | 12299 | 644.8 | 870.1 | 1515.0 | 6.25 GB |
| 3 | 250585 | 631.6 | 574.6 | 1206.2 | 6.25 GB |
| 4 | 535644 | 579.9 | 911.9 | 1491.8 | 6.25 GB |
| 5 | 402452 | 565.9 | 608.7 | 1174.6 | 6.25 GB |
| 6 | 249901 | 553.0 | 539.2 | 1092.2 | 6.25 GB |
| 7 | 811815 | 551.9 | 678.5 | 1230.4 | 6.25 GB |
| **median** | | **631.6** | **678.5** | **1491.8** | **6.25 GB** |

## 4. Alien-twin sanity read (L10 N6978)

The read is `softmax(W_U · norm(J_10^out · W_down[:, 6978]))` with the final
norm's weight, from the fitted R-lens (the full-stack pass records every layer,
so it cost one matrix-vector product, no extra fit).

**L10 N6978:** `galaxy` 0.692, `Galactic` 0.114, `spaceship` 0.112, `Galaxy` 0.024, `spacecraft` 0.020, `Star` 0.019, `star` 0.005, `cosmic` 0.003.

That is a coherent off-world read, matching the alien-twin finding, and it is
neuron-specific rather than a generic output:
- control neuron 5305 (same layer): `also` 0.134, `también` 0.065, `Also` 0.063, `ALSO` 0.037, `Also` 0.036, `also` 0.019.
- control neuron 2471 (same layer): `qualche` 0.010, `aires` 0.009, `uevo` 0.007, `duit` 0.004, `argent` 0.003, `()",` 0.003.
- control neuron 6468 (same layer): `skiing` 0.451, `ski` 0.345, `snow` 0.059, `Ski` 0.041, `avalanche` 0.031, `Alpine` 0.021.

Full JSON: `results/prime-lens/olmo2-1b-instruct/neuron-L10-N6978.json`.

## 5. Cost projection

Method, stated so it can be checked:

1. **CPU, measured.** The fit's median seconds per document (both lenses,
   all layers, 128 tokens) is the base rate. The same architecture is used by
   every OLMo 2 1B checkpoint, so the per-document cost is stage-independent.
2. **Projected H100.** Neuron Lens' own benchmark for Qwen3-8B is 11.36 s (R)
   and 11.56 s (S) per document on one H100 (`data/summary/fit_bench_h100_2026-10-28.log`).
   The per-document FLOP ratio OLMo 2 1B : Qwen3-8B is

   `16·(3·2048·8192 + 4·2048²) / 36·(3·4096·12288 + 4·4096²) = 0.137`,

   so the projected H100 rate is `11.4 × 0.137 ≈ 1.6 s` per document per lens.
3. **Projected RTX 3090.** The H100's dense bf16 tensor peak is 989 TFLOP/s
   against the 3090's 142 (≈7×), and its HBM bandwidth 3.35 vs 0.94 TB/s
   (≈3.6×); their achieved 670 TFLOP/s is 68% of the H100 peak, so a 5–7×
   slowdown is the defensible range: **8–11 s per document per lens**,
   16–22 s for both.
4. **Cross-check on the scaling.** Their Qwen3-0.6B measurement (0.36 s per
   document on an H100) is *below* a linear FLOP projection from the 8B figure
   (≈0.5 s), so the FLOP scaling used here is conservative at the small end and
   the OLMo H100/3090 numbers above are upper bounds. `project_costs.py`
   reproduces the table from `fit.json`.

| stage | revision | 200 docs, CPU | 1000 docs, CPU | 200 docs, 3090 | 1000 docs, 3090 |
|---|---|---|---|---|---|
| Base (main) | a1847dff | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| pretraining, 21B tokens | stage1-step10000-tokens21B, 22f7d18a | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| pretraining, 210B | stage1-step100000-tokens210B, eb19ccab | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| pretraining, 1.0T | stage1-step480000-tokens1007B, eda09f03 | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| pretraining, 2.0T | stage1-step950000-tokens1993B, 3e09b708 | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| pretraining, 3.0T | stage1-step1430000-tokens2999B, 22738a97 | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| end of pretraining, 4.0T | stage1-step1907359-tokens4001B, 9d3e4365 | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| mid-training, 13B | stage2-ingredient3-step6000-tokens13B, a7a789f6 | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| mid-training, 26B | stage2-ingredient3-step12000-tokens26B, 1c3599f7 | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| mid-training end, 51B | stage2-ingredient3-step23852-tokens51B, f83cd641 | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| SFT | -SFT | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| DPO | -DPO | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| Instruct | -Instruct, 48d788ec | 82.9 h | 414.4 h | 0.9 h–1.2 h | 4.4 h–6.1 h |

The Base repository's non-main revisions are mapped to training tokens in
`PART85-DESIGN.md` (the table above follows it); `Base (main)` is the released
base model at `a1847dff`.

**Reading:** a 1,000-document lens of record for one stage is **~17 days of this
CPU** (414 h) but **~4.4–6.1 hours on one rented 3090**; at 200 documents it is
~3.5 days against ~1 hour. The CPU cannot run a full fit — the 8-document fit is
the acceptance measurement, not a lens. The table is identical across stages
because the architecture is: the per-document cost does not depend on training
step, only on the model shape and the document length.

## 6. What would change for Qwen3.5-4B

The Qwen3.5 config in the house cache (`Qwen/Qwen3.5-4B`, `model_type
qwen3_5_text`, a `Qwen3_5ForConditionalGeneration` with a text stack):

- **Hybrid attention.** `layer_types` is 3 linear-attention layers to 1 full
  layer (`full_attention_interval: 4`). A linear-attention layer is a
  different map (a recurrence, not softmax attention), so the port would need a
  second map family inside one stack — like `gemma4.py`'s per-layer kinds — and
  the S-lens's "keys at valid positions" trick does not carry over unchanged.
- **GQA 4:1** (16 query / 4 KV heads), so the grouping path matters (the OLMo
  map already keeps `num_key_value_groups`, tested with a GQA tiny model).
- **`head_dim` 256**, not `hidden / heads` (2560/16 = 160): the q/k norm width
  becomes 16·256 = 4096, and the attention scaling is `256^-0.5`.
- **Partial RoPE** (`partial_rotary_factor: 0.25`, mRoPE with sections
  `[11, 11, 10]`): only a quarter of each head rotates, and the rotary table is
  multinomial — the map's RoPE section would be rewritten.
- **Tied embeddings** (`tie_word_embeddings: true`): `W_U` is `W_E`, which
  changes the read matrices, not the estimator.
- **Vocab 248,320** (token mask and read cost), 32 layers, d 2560, ff 9216.

So: the estimator's machinery (RelP rules, the two directions, the offset sums)
is unchanged, but Qwen3.5-4B is not a config flip either — its linear-attention
layers are the substantial piece.

## 7. Limits

- The fit is 8 documents: the split-half figure it reports is a coarse
  stability check, not a lens-quality estimate.
- bf16 on CPU (AVX512-BF16) is 4.5× faster than fp32 here; the upstream
  precision check (bf16 vs fp32, Qwen3-0.6B) puts the estimate 0.65–0.72% from
  fp32, below the sampling error at this document count.
- The CPU timings are this host's; the 3090 figures are projections, not
  measurements (no GPU was used).
