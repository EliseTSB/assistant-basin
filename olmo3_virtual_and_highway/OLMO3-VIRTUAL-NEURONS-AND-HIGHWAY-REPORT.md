# OLMo 3 7B RelP Neuron Lens, Zero-Training Virtual SwiGLU Synthesis, and Signal-Highway Decomposition

**Author:** Vesper (`neon`, Gemini 4 Neon)  
**Date:** 2026-11-22  
**Status:** Exploratory empirical study & toolkit release (`V-006`, local branch `vesper-olmo3-lens`; **not pre-registered** — published as an exploratory baseline to guide pre-registered follow-up Stage C0).  
**Target model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` (32 layers, `d = 4096`; architecture details per HuggingFace model config: `intermediate = 11008`, 32/32 MHA, untied `lm_head`, full-width QK-norm, post-sublayer RMSNorm, 3:1 `sliding_attention(4096)` / `full_attention` with YaRN scaling restricted to global layers).

---

## Executive Summary (Four Exploratory Observations)

Scaling our mechanistic interpretability toolkit from OLMo 2 1B to **OLMo 3 7B Instruct** yielded four exploratory observations:

1. **64-doc RelP Neuron Lens (`R + S`) reproducible across all 32 layers (`9.26 CU` on A100 GPU — reusable asset carried forward):** Our OLMo 3 port (`neuron_lens/olmo3.py`) produced mirror-symmetric exact `fp32` split-half relative L2 differences (`||A - B|| / ||A + B||`, lower = tighter: `S_mid` `0.0067` at `L0` → `0.4069` at `L31`; `R_mid` `0.4197` at `L0` → `0.0132` at `L31`), indicating `S` is tightest in shallow layers and `R` is tightest in deep layers. Across all 11 layers available in `mhough/olmo3-jacobian-lenses` (external J-lens), our 64-doc `R_out` is externally consistent from `L9` upward (`+0.7938` at `L9`, `+0.8385` at `L12`, **`+0.8997` at `L15`**, **`+0.9474` at `L18`**, **`+0.9702` at `L21`**, **`+0.9805` at `L24`**, **`+0.9868` at `L27`**, **`+0.9989` at `L30`**) while diverging in shallow layers (`+0.3178` at `L0`, `+0.5369` at `L3`, `+0.6886` at `L6`, plausibly reflecting RelP stop-gradients as reported in `hiskuDN/open-r-lens`). Our `R_out` layer norms progress smoothly (`36.98–67.20`) compared with `danorel`'s 64-doc lens (`21.00–212.33`).
2. **Zero-training hand-built Virtual Slot SwiGLU neurons shift `"Who are you?"` selectively (`0.53 CU` on NVIDIA L4 GPU):** Synthesizing `(w_gate, w_up, w_down)` directly from ridge-whitened `S_mid` and `R_out` (`S_ridge = (S S^T + λI)^{-1} S e`, `R_ridge = (R^T R + λI)^{-1} R^T u_t`) — with zero gradient training and zero prompt activations for the weight directions — yields `virtual_slot_S_ridge`: at `L18`, mean `self Δlogit = +9.26` (4 personas × 6 self-queries), `self_act = +0.412` vs. `off_act = -0.010`, `off_KL = 0.0462` over 6 held-out ordinary prompts; at `L24`, mean `self Δlogit = +22.34`, `self_act = +1.840` vs. `off_act = +0.105`, and `off_KL = 0.0014` (`33×` lower than at `L18`). On the one off-target prompt generated (`"What is the capital of France?"`), greedy output remains `"The capital of France is Paris."`, while `"Who are you?"` inserts the target word (pirate/librarian). A held-out contrastive read (`virtual_slot_contrastive`) achieves lower `off_KL = 0.0003` at both `L18` and `L24` with smaller `self Δlogit` (`+4.48` at `L18`, leaving `"Who are you?"` at the default AI assistant line; `+5.86` at `L24`, producing fluent `" librarian of information!..."`). Injecting the virtual SwiGLU neuron *before* `post_feedforward_layernorm` (`pre_norm`, matched in *relative* site norm, i.e. `3.3×–4.7×` larger absolute norm than `post_norm`) produces strongly depth-dependent damping (`~13.4–15.8×` at `L12`, `~4.2–6.3×` at `L18`, `~3.7–4.5×` at `L24`; `L6` `pre_norm` within noise), consistent with BlackboxNLP 2026's post-norm scale-invariance trap.
3. **Downstream cancellation vs. constructive transport (`~2–3×` Exp 4 gain advantage for `R_mf` over Logit-Lens; consistent with active erasure, not yet separated from non-propagation without Stage-C0 controls):** Exact 4-term telescoping decomposition (`Direct + ΣAttn_l + ΣMLP_l + RMSsh`; `432` traces, telescoping error `median = 0.01771`, `p95 = 0.05215`, `max = 0.07605`) shows that raw **Logit-Lens (`u_t`)** injects the largest direct unembedding write (`+8.93` at `L18`, `+14.56` at `L24`) yet loses most of it downstream (`ΣAttn = -4.01` / `-5.96`, `ΣMLP = -2.39` / `-1.11`, gross-over-net `CancR = 7.32` at `L18`, `3.30` at `L24`; `97.4%` lost at `L18 pre_norm`). By contrast, matched-filter **`R_mf` (`R^T u_t`)** injects a smaller direct write (`+3.70` / `+8.12`) followed by positive downstream terms (`ΣAttn = +3.47` / `+3.81`, `ΣMLP = +0.22` / `+3.65`, `CancR = 1.57` / `1.12`), yielding `2.1×–3.0×` larger final `Δlogit` across `L6–L24` in Exp 4 (`1.6×–4.7×` in Exp 2A `dose = 1.50`). **`R_ridge`** yields intermediate final gain with the highest Exp 2A `dlogit_z` at `L18` (`+11.70`) and `L24` (`+25.85`), though at shallow `L6` its net gain (`+0.40`) and `CancR` (`9.00`) remain close to `logit_lens` (`+0.31`, `17.03`).
4. **Natural OLMo 3 7B highways & 1,024-head depth-controlled scan:** Direct causal pushes (`+10.0` SwiGLU activation) on early natural Concept neurons (`L4N6633` `' teacher'`, `L5N6875` `' librarian'`) start from a `bf16`-floor direct term (`+0.05`, `+0.26`) and reach **`+10.52`** (`ΣAttn +3.53`, `ΣMLP +6.00`) and **`+8.16`** (`ΣAttn +1.97`, `ΣMLP +4.65`) with peak contributions in `L29–L31`, whereas late natural Concept candidates (`L19N4502`, `L23N8559`) show weak or slightly negative downstream change (`+0.38`, `+0.19`). Among the 6 natural Slot candidates, downstream behavior is heterogeneous: `L17N2444` shows negative `Attn` (`-1.29`) and `MLP` (`-1.11`), `L21N3` shows opposing `Attn` (`+1.80`) and `MLP` (`-1.57`), `L25N4074` shows negative `Attn` (`-3.09`) alongside positive `MLP` (`+0.71`), and `L28N2566` (`CancR = 2.27`), `L26N3854` (`4.03`), and `L23N6941` (`6.13`) do not cancel strongly; Final-RMSNorm shifts (`RMSsh`) are small (`-0.29` to `+0.02`). On the 1,024-head OV scan, depth-controlled `full_attention` vs. flanking `sliding_attention` shows no significant global mean shift across all 7 tested metrics (closest: `name_z p = 0.088`, `ov_frob p = 0.161`; `self_z p = 0.996`), with a genuine bilateral OV norm spike at `L19*` (`3.517` vs. `L18 = 2.556` `+37.6%`, `L20 = 2.593` `+35.7%`) and a step onto a late plateau at `L23*` (`4.264` vs. `L22 = 3.120` `+36.7%`, `L24 = 4.023` `+6.0%`).

---

## Metrics & Notation

- **RelP (Relevance Patching):** Linearized backward/forward transport rules that detach `post_attention_layernorm` and `post_feedforward_layernorm` denominators and apply the SwiGLU Identity+Half rule, while preserving exact `_norm_jvp`/`_norm_vjp` through full-width `q_norm`/`k_norm` and attention softmax.
- **`R_mid`, `R_out`, `S_mid`, `S_out`:** Per-layer `[4096, 4096]` RelP transport matrices — `R` maps layer `L` (`mid` = post-attention, `out` = post-MLP) forward to the final residual; `S` maps token embeddings forward to layer `L`.
- **`u_t` & Direct:** `u_t = W_U[t] ⊙ γ_final` is the final-norm-weighted unembedding covector for target token `t`. `Direct = (u_t · Δm_L) / r_c` is the linear contribution of layer `L`'s MLP increment divided by the clean final RMSNorm denominator `r_c`.
- **`RMSsh` (Final-RMSNorm Shift):** `(u_t · h_i) · (1/r_i - 1/r_c)`, isolating passive final-layer RMSNorm denominator rescaling from active downstream residual additions.
- **`CancR` (Path Cancellation Ratio):** `(|Direct| + Σ_l |attn_l| + Σ_l |mlp_l| + |RMSsh|) / max(|exact Δlogit|, 1e-6)`. This is a **gross-over-net ratio** (bounded below by `1.0`), **not** a fraction erased; it peaks whenever `exact Δlogit ≈ 0` (e.g., `L25N4074` at `net = +0.06`).
- **`Late3M`:** Sum of downstream MLP logit contributions over the final 3 layers (`L29 + L30 + L31`).
- **`off_KL`:** `KL(P_clean || P_int)` at the final prompt token averaged over 6 held-out ordinary prompts (and 4 personas in Exp 2B tables); do-nothing baseline is `0` by construction (no random-neuron `off_KL` baseline was run in Exp 2B).
- **`Med Sel (S/O)`:** Median over 4 personas of `mean_self_act / max(|mean_off_act|, 1e-4)` — a property of the **read** vector alone (identical across `R_ridge`, `R_mf`, `logit_lens`, and `random` writes), and sensitive to near-zero `off_act`; always read alongside absolute `self_act`, `off_act`, `self Δlogit`, and `off_KL`.
- **`z` / `σ`, `dose`, `top1`, `*`:** In Exp 2A, `z` is `(Δlogit(t) - mean(Δlogit)) / std(Δlogit)` across all `100,278` vocabulary logits; in the census, `write_z`/`read_same_z` are z-scores against a 2,000-token vocab background. `dose` is a multiplier on the **site-specific** clean MLP norm (`pre_mlp_norm` for `pre_norm`; `post_mlp_norm` for `post_norm`). `top1` is the count of targets reaching rank 1. `*` marks `full_attention` layers (`L3, 7, 11, 15, 19, 23, 27, 31`). `L4` = layer 4; **L4 GPU** = NVIDIA L4 accelerator.

---

## Pinned Inputs, Provenance & Reproducibility

| Artifact | Size | SHA-256 | In Bundle? |
|---|---:|---|---|
| `fit_meta.json` (64-doc RelP fit) | 19.8 kB | `4e4291a871f8d7e366798386410faf25e4933b5e2739deae7c7d049d93c12d34` | Yes |
| `olmo3_lens_analysis.json` (cross-val + 1,024-head scan) | 383.1 kB | `b1b9793b...` (recorded in `scorecard.json`) | Yes |
| `natural_neuron_census_preview.json` (candidate scan) | 36.6 kB | `a1e6ba10...` (recorded in `scorecard.json`) | Yes |
| `olmo3_idea2_and_idea4_results.json.gz` (uncompressed `4.20 MB`) | 376 kB gz | `d376b6120e346a0fe8334126ef3f0d6bdfa988de9116429d6a43a14c59ea0a2e` | Yes |
| `scorecard.json` (stdlib-only deterministic scorecard) | 83 kB | `dbfcd8aa79efe6bbaf108fa756edda422b2f7dea965fe9840e25f3c405b630d3` | Yes |
| `SCORECARD.md` (rendered scorecard tables) | 5.0 kB | `16644ddd80827683e2a773b8bc2428ec03c51c5bd3a4e9556fc2f250c90574f0` | Yes |
| `r_lens.pt` (`R_mid` + `R_out`, `fp16`) | 2,016.0 MB* | `ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955` | Archived externally (hash pinned in receipt) |
| `s_lens.pt` (`S_mid` + `S_out`, `fp16`) | 2,048.0 MB* | `a145860bfbb2b5399b19d716cea1a09f2abe1f097fd68ac1d7a6dfabf5c0e638` | Archived externally (hash pinned in receipt) |

*\*Scope of bundle receipts (not independently verifiable from the `900 kB` JSON bundle alone):* external `.pt` file sizes, external Colab compute ledger (`0.37` L4 GPU timing + `9.26` A100 GPU 64-doc fit + `0.53` L4 GPU Idea #2/#4 = `10.16 CU` total, `174.39 CU` left), `pytest`/sabotage/primal/dual-site test logs (run in repo test suite), HuggingFace config specs beyond `32L / d=4096`, `exp.meta` runtime `bf16` dtype and prompt-split counts (defined in `olmo3_virtual_and_highway/prompts.py` & `run_olmo3_idea2_and_idea4.py`), and external HF lens commit/doc metadata (`mhough@c134cb08`, `danorel@fd8f1d4e`).

### Rescore it yourself (Python 3.10+ standard library only)
From a clone of the repository (`cd assistant-basin`):
```bash
python3 -I olmo3_virtual_and_highway/score_olmo3_study.py \
  --lens-dir results/vesper-olmo3-7b-exploratory \
  --exp-dir results/vesper-olmo3-7b-exploratory \
  --out-dir /tmp/olmo3_rescore
cmp /tmp/olmo3_rescore/scorecard.json results/vesper-olmo3-7b-exploratory/scorecard.json
cmp /tmp/olmo3_rescore/SCORECARD.md results/vesper-olmo3-7b-exploratory/SCORECARD.md
```

---

## 1. OLMo 3 7B RelP Neuron Lens (`R + S`) Port & Cross-Validation

- **Port & test suite (`neuron_lens/olmo3.py`, `tests/test_olmo3.py`, `test_olmo3_lens_sabotage.py`; outside JSON bundle):** Extends `olmo2.py` with per-`layer_type` RoPE (`default` on `sliding_attention`, `yarn` on `full_attention`) and `sliding_window = 4096` masking. Local test suite passed (`52 passed, 3 skipped`; `4/4` sabotage breaks caught; `0.0` max abs `bf16` primal diff on real 7B at `L0_sliding`, `L3_full`, `L15_full`, `L31_full`).
- **64-doc fit (`fit_meta.json`: `NVIDIA A100-SXM4-40GB`, WikiText-103, seed `5701`, `128` tokens/doc, `3,933.41 s`, median `60.15 s/doc`, peak VRAM `17.24 GB`):**
  - Mirror-symmetric `fp32` split-half relative L2 differences (`||A - B|| / ||A + B||`, lower = tighter):
    - `R_mid`: `0.4197` (`L0`) → `0.2022` (`L15`) → **`0.0132`** (`L31`)
    - `R_out`: `0.4189` (`L0`) → `0.2023` (`L15`) → **`0.0323`** (`L30`)
    - `S_mid`: **`0.0067`** (`L0`) → `0.1780` (`L15`) → `0.4069` (`L31`)
    - `S_out`: **`0.0086`** (`L0`) → `0.1793` (`L15`) → `0.4139` (`L31`)
  - **External cross-validation (`olmo3_lens_analysis.json`):** Comparing our 64-doc `R_out` against `mhough/olmo3-jacobian-lenses` across all 11 available layers (every 3rd layer) gives cosine `+0.3178` (`L0`), `+0.5369` (`L3`), `+0.6886` (`L6`), `+0.7938` (`L9`), `+0.8385` (`L12`), **`+0.8997` (`L15`)**, **`+0.9474` (`L18`)**, **`+0.9702` (`L21`)**, **`+0.9805` (`L24`)**, **`+0.9868` (`L27`)**, and **`+0.9989` (`L30`)**. Against `danorel/Olmo-3-7B-Instruct-jpp-lens` (31 layers `0..30`), cosine reaches `+0.9970` (`L30`), `+0.9569` (`L19`), `+0.9267` (`L21`); `danorel`'s Frobenius norms range `21.00–212.33` whereas ours progress smoothly `36.98–67.20`.

---

## 2. Idea #5: 1,024-Head Sliding vs. Full Attention Scan & Natural Neuron Census

Scoring all `1,024` heads (`768` `sliding_attention`, `256` `full_attention`) through `K_in = S_out[L-1]` and `J_out = R_mid[L]`:

- **Depth-controlled null on global mean:** Uncontrolled, `full_attention` heads had `+17%` higher OV Frobenius norm (`2.769` vs. `2.367`), confounded by depth (`full` layers sit at `L3, 7, 11, 15, 19, 23, 27, 31`). Comparing the 7 interior `full` layers against their flanking `sliding` neighbors (`L-1`, `L+1`) yields no significant global mean shift across all 7 tested metrics: `name_z` (`p = 0.088`, full `-0.0612` vs. flank `+0.0031`), `ov_frob` (`p = 0.161`, `2.5374` vs. `2.3220`), `copy_bg_normed` (`p = 0.211`), `persona_z` (`p = 0.222`), `abs_sr_z` (`p = 0.605`), `role_z` (`p = 0.848`), and `self_z` (`p = 0.996`). `full` heads comprise `5/20` (`25%`, exact base rate) of top-20 general copy heads (`16/50` = `32%` of top-50).
- **Mid-late OV Frobenius norm profile:** `L19*` is a genuine bilateral spike (`3.517` vs. `L18 = 2.556` `+37.6%`, `L20 = 2.593` `+35.7%`). `L23*` (`4.264`) marks the largest single-layer step from `L22` (`3.120`, `+36.7%`) onto a late high-norm plateau (`L24 = 4.023` [`+6.0%`], `L25 = 3.806`, `L26 = 4.139`, `L27* = 4.102`, `L28 = 4.213`, `L29 = 4.823`, `L30 = 5.121`, `L31* = 4.393`).
- **Top-ranked candidate heads (post-hoc top-k of 1,024; no max-statistic null):**
  - **SELF-vs-ROLE:** `L22H9` [slid] (`S=+5.31, R=+2.23, S-R=+3.08`), `L19H8` [full] (`S=+3.12, S-R=+2.53`), `L26H11` [slid] (`S=+2.38, S-R=+2.50`).
  - **ROLE-vs-SELF:** `L21H25` [slid] (`R=+7.25, S=+1.56, S-R=-5.69`), `L24H20` [slid] (`R=+7.69, S=+4.01, S-R=-3.67`), `L19H3` [full] (`R=+3.98, S-R=-3.51`).
  - **NAME-copy:** `L26H2` [slid] (`N=+5.27, S=-0.50, R=-0.60`).
  - **PERSONA-copy:** `L24H31` [slid] (`P=+4.05, N=+2.04, S=-0.29, R=-0.42`), `L27H3` [full] (`P=+2.80`), `L28H12` [slid] (`P=+2.58`).
- **Natural neuron census (`natural_neuron_census_preview.json`):** Found `14` Concept candidates (`write_z > 6.5, read_same_z >= 3.5`) — e.g. `L4N6633` (`' teacher'`, `write_z = +7.54`, `read_same_z = +5.09`, top reads `' ense'` `7.91`, `' Teach'` `7.58`, `' educator'` `6.84`), `L5N6875` (`' librarian'`, `write_z = +6.80`, `read_same_z = +6.36`, top reads `' libr'` `6.79`, `' Bibli'` `6.01`), `L16N5264` (`' dog'`, `write_z = +15.20`, `read_same_z = +7.02`, top reads `'Dog'` `8.09`, `' Puppy'` `7.81`), `L14N8245` (`' cat'`, `write_z = +11.06`, top read `' kitten'` `7.49`) — and `89` Slot-like candidates (`write_z > 6.5, read_same_z <= 1.5`), e.g. `L28N2566` (`' I'`, `write_z = +34.81`, `read_same_z = +1.21`; top reads `'>Hello'` `4.35`, `'“Oh'` `3.19`), `L25N4074` (`' I'`, `write_z = +24.06`, `read_same_z = -0.57`; top reads `' Hmm'` `5.10`, `'Anyway'` `4.51`), `L17N2444` (`' me'`, `write_z = +8.96`; top read `'Personally'` `4.56`), `L26N3854` (`' assistant'`, `write_z = +14.77`), and `L23N6941` (`' AI'`, `write_z = +12.47`).

---

## 3. Idea #2: Zero-Training Synthesis of Virtual Concept & Slot SwiGLU Neurons

### Method & Guardrails
- **Write & read synthesis (`build_olmo3_virtual_bank.py`):** Compares matched-filter `R_mf = R_L^T u_t`, ridge-whitened `R_ridge = (R_L^T R_L + λI)^{-1} R_L^T u_t` (`λ = 0.5 · tr(G)/d`), raw Logit-Lens `u_t`, and norm-matched `random`; symmetrically compares `S_mf = S_L e` and ridge-whitened `S_ridge = (S_L S_L^T + λI)^{-1} S_L e`.
- **Dual-site injection:** Every virtual SwiGLU neuron is tested at `pre_norm` (injected into `mlp.down_proj` output before `post_feedforward_layernorm`; equivalence to literal weight append verified `<1e-5` in `tests/test_olmo3_virtual_and_highway.py`) and `post_norm` (after `post_feedforward_layernorm`). Doses are matched in **relative site norm** (`dose_mult × pre_mlp_norm` for `pre_norm` vs. `dose_mult × post_mlp_norm` for `post_norm`; e.g., at `dose_mult = 1.50`, `36.63` vs. `11.18` at `L18` and `85.22` vs. `18.23` at `L24`).
- **Disjoint splits (`prompts.py`):** Calibration (`4` self-queries, `4` off-target) is strictly disjoint from evaluation (`6` self-queries, `6` off-target, `8` concept prompts, `4` register frames `WHO_CHAT_TF`/`SELF`/`NAME_BARE`/`ROLE_BARE`).

### Experiment 2A: Write-Geometry Benchmark (`WHO_CHAT_TF`, `post_norm`, `dose_mult = 1.50`, 8 personas)

| Layer | `R_ridge` dL (`z`, top1) | `R_mf` dL (`z`, top1) | `logit_lens` dL (`z`, top1) | `random` dL (`z`) | `"Who are you?"` Gen Hits (of 4) |
|---:|---:|---:|---:|---:|---:|
| `L6` | `+0.75` (`+0.77`, `0/8`) | **`+2.29`** (`+1.39`, `0/8`) | `+0.56` (`+0.74`, `0/8`) | `+0.33` (`+0.76`) | `0 / 0 / 0 / 0` |
| `L12` | `+3.01` (`+2.69`, `0/8`) | **`+6.96`** (`+3.55`, `0/8`) | `+1.49` (`+1.75`, `0/8`) | `-0.19` (`-0.39`) | `0 / 0 / 0 / 0` |
| `L18` | `+12.22` (**`+11.70`**, `1/8`) | **`+16.17`** (`+11.13`, `1/8`) | `+6.42` (`+7.00`, `0/8`) | `+0.03` (`-0.39`) | **`4 / 4 / 1 / 0`** |
| `L24` | `+25.34` (**`+25.85`**, **`8/8`**) | **`+27.99`** (`+20.13`, **`8/8`**) | `+18.02` (`+20.93`, `5/8`) | `+0.37` (`-0.34`) | `4 / 4 / 4 / 0` |

- **`pre_norm` vs. `post_norm` damping (`dose_mult = 1.50` of site-specific norm — `pre_norm` absolute writes are `3.3×–4.7×` larger at `L18–L24`, making damping ratios conservative):** `post_norm / pre_norm` `Δlogit` ratios are strongly depth-dependent: **`~13.4–15.8×` at `L12`** (`R_mf` `+6.96` vs `+0.52`, `R_ridge` `+3.01` vs `+0.19`), **`~4.2–6.3×` at `L18`** (`R_mf` `+16.17` vs `+3.81`, `R_ridge` `+12.22` vs `+2.34`, `logit_lens` `+6.42` vs `+1.02`), and **`~3.7–4.5×` at `L24`** (`R_ridge` `+25.34` vs `+5.93`, `R_mf` `+27.99` vs `+7.52`, `logit_lens` `+18.02` vs `+4.05`); at `L6`, `pre_norm` shifts are within noise (`-0.11` to `+0.24`). `L24 pre_norm` greedy hits: `2/4` `R_ridge`, `1/4` `R_mf`, `1/4` `logit_lens`, `0/4` `random`.

### Experiment 2B: Input-Conditioned Virtual Concept vs. Virtual Slot SwiGLU Neurons (`post_norm`, `R_ridge` write, mean across 4 personas × 6 queries)

All 8 `L18`/`L24` rows from `SCORECARD.md` (note: `Med Sel (S/O)` is a read-only ratio identical across write methods; lead with `Self Δlogit`, `Off Act`, and `Off KL`):

| Layer | Virtual Neuron | Self `Δlogit` | Concept `Δlogit` | Off KL (6 prompts) | Self Act | Off Act | Concept Act | Med Sel (S/O) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `L18` | `virtual_concept_S_ridge` | `+2.22` | `-1.62` | `0.0356` | `+0.066` | `+0.017` | `+0.010` | `3.6×` |
| `L18` | `virtual_concept_S_mf` | `+1.95` | `-1.89` | `0.0788` | `+0.146` | `+0.020` | `-0.010` | `4.2×` |
| `L18` | `virtual_slot_S_ridge` | **`+9.26`** | `+0.75` | `0.0462` | `+0.412` | `-0.010` | `+0.011` | `39.8×` |
| `L18` | `virtual_slot_contrastive` | `+4.48` | `+0.26` | **`0.0003`** | `+89.984` | `-0.516` | `+5.331` | `174.3×` |
| `L24` | `virtual_concept_S_ridge` | `+0.67` | **`+1.64`** | **`0.0009`** | `+0.199` | `+0.040` | **`+0.492`** | `1.2×` (`5.6×` C/O) |
| `L24` | `virtual_concept_S_mf` | `+0.35` | `+0.26` | `0.0011` | `+0.193` | `+0.021` | `+0.057` | `0.9×` (`1.0×` C/O) |
| `L24` | `virtual_slot_S_ridge` | **`+22.34`** | `+0.23` | **`0.0014`** | `+1.840` | `+0.105` | `-0.013` | `17.6×` |
| `L24` | `virtual_slot_contrastive` | `+5.86` | `+0.24` | **`0.0003`** | `+198.147` | `-1.790` | `+9.407` | `110.7×` |

**Greedy generations (`post_norm`, `R_ridge`; only 1 off-target prompt — France — was generated, while `Off KL` covers all 6):**
- Clean baseline: `"Who are you?"` → *"I am an artificial intelligence chatbot designed to assist..."*; `"What is the capital of France?"` → *"The capital of France is Paris."*
- `L18 virtual_slot_S_ridge` (`pirate`): `"Who are you?"` → **"I pirate! (Just kidding!) I pirate is an AI language model..."** | France → **"The capital of France is Paris."**
- `L18 virtual_slot_S_ridge` (`librarian`): `"Who are you?"` → **"I librarian AI model. I librarian assist users with information..."** | France → **"The capital of France is Paris."**
- `L18 virtual_slot_contrastive` (`pirate`/`librarian`): `"Who are you?"` → **untouched default AI assistant** (`0/2` hits, consistent with lower `self Δlogit = +4.48`) | France → **"The capital of France is Paris."**
- `L24 virtual_slot_S_ridge` (`pirate`): `"Who are you?"` → **"I am pirate LLM, a large language model pirate!..."** | France → **"The capital of France is Paris."**
- `L24 virtual_slot_contrastive` (`librarian`): `"Who are you?"` → **" librarian of information! I can help answer questions, provide explanations, assist with writing..."** | France → **"The capital of France is Paris."**
- `L18`/`L24 virtual_concept_S_ridge` (`pirate`/`librarian`): `"Who are you?"` → **untouched default AI assistant** (`0/4` hits) | France → **"The capital of France is Paris."**

---

## 4. Idea #4: Exact 4-Term `L+1..31` Signal-Highway Decomposition

Exact paired-run residual decomposition `Δlogit(t) = (u_t · Δm_L)/r_c + Σ(u_t · Δa_l)/r_c + Σ(u_t · Δm_l)/r_c + (u_t · h_i)(1/r_i - 1/r_c)` across **`432` traces** (`telescoping_err`: `median = 0.01771`, `p95 = 0.05215`, `max = 0.07605`):

### All 12 Virtual Neuron Rows (`WHO_CHAT_TF`, `post_norm`, `dose_mult = 0.75`, mean across 4 personas)

| Layer | Method | Exact `Δlogit` | Direct | `ΣAttn` | `ΣMLP` | `RMSsh` | `CancR` |
|---:|---|---:|---:|---:|---:|---:|---:|
| `L6` | `R_ridge` | `+0.40` | `+0.81` | `+0.52` | `-0.93` | `+0.01` | `9.00` |
| `L6` | `R_mf` | `+0.90` | `+0.56` | `+0.73` | `-0.43` | `+0.05` | `3.67` |
| `L6` | `logit_lens` | `+0.31` | `+2.69` | `-0.79` | `-1.57` | `-0.01` | `17.03` |
| `L12` | `R_ridge` | `+1.47` | `+2.04` | `+0.92` | `-1.51` | `+0.02` | `4.00` |
| `L12` | `R_mf` | `+2.44` | `+1.38` | `+1.44` | `-0.56` | `+0.19` | `2.85` |
| `L12` | `logit_lens` | `+0.89` | `+4.60` | `-1.30` | `-2.43` | `+0.02` | `11.39` |
| `L18` | `R_ridge` | `+5.25` | `+5.19` | `+0.98` | `-1.09` | `+0.18` | `1.98` |
| `L18` | `R_mf` | **`+7.62`** | `+3.70` | **`+3.47`** | **`+0.22`** | `+0.25` | **`1.57`** |
| `L18` | `logit_lens` | `+2.50` | **`+8.93`** | **`-4.01`** | **`-2.39`** | `-0.01` | `7.32` |
| `L24` | `R_ridge` | `+11.59` | `+10.56` | `+0.34` | `+0.77` | `-0.09` | `1.31` |
| `L24` | `R_mf` | **`+15.39`** | `+8.12` | **`+3.81`** | **`+3.65`** | `-0.19` | **`1.12`** |
| `L24` | `logit_lens` | `+7.23` | **`+14.56`** | **`-5.96`** | **`-1.11`** | `-0.25` | `3.30` |

At `L18 pre_norm`, `logit_lens` injects `Direct = +4.40`, loses `ΣAttn = -3.12` and `ΣMLP = -1.13` (`CancR = 91.58`), leaving `+0.11` net (`97.4%` unrounded loss), whereas `R_mf` moves from `Direct = +1.17` to `+3.10` net (`ΣAttn = +1.92`, `CancR = 1.64`). Across `L6–L24`, `R_mf` / `logit_lens` final gain ratios are `2.9×`, `2.7×`, `3.0×`, `2.1×` (while `CancR` ratios span `2.9×–4.7×`).

### All 12 Natural Neuron Rows (`WHO_CHAT_TF`, activation push `+10.0`)

| Unit | Kind | Target | Exact `Δlogit` | Direct | `ΣAttn` | `ΣMLP` | `RMSsh` | `Late3M` | `CancR` | `SELF`/`NAME`/`ROLE` `Δlogit` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `L4N6633` | `natural_concept` | `' teacher'` | **`+10.52`** | `+0.05` | `+3.53` | `+6.00` | `+0.93` | `+3.26` | `1.28` | `-1.50` / `+5.44` / `+3.69` |
| `L5N6875` | `natural_concept` | `' librarian'` | **`+8.16`** | `+0.26` | `+1.97` | `+4.65` | `+1.26` | `+4.03` | `1.55` | `-2.44` / `+1.62` / `-0.88` |
| `L14N8245` | `natural_concept` | `' cat'` | `+3.80` | `+0.55` | `+0.74` | `+2.54` | `-0.03` | `+1.31` | `2.00` | `+3.00` / `+4.38` / `+1.00` |
| `L16N5264` | `natural_concept` | `' dog'` | `+3.17` | `+0.78` | `+1.33` | `+1.13` | `-0.07` | `+0.93` | `2.97` | `+2.69` / `+3.88` / `+1.56` |
| `L19N4502` | `natural_concept` | `' robot'` | `+0.38` | `+0.51` | `+0.03` | `+0.00` | `-0.13` | `+0.79` | `13.54` | `+3.38` / `+1.94` / `+1.19` |
| `L23N8559` | `natural_concept` | `' knight'` | `+0.19` | `+0.57` | `-0.09` | `-0.29` | `-0.00` | `-0.19` | `9.88` | `+1.38` / `+0.67` / `+0.50` |
| `L17N2444` | `natural_slot` | `' me'` | `-1.09` | `+1.36` | `-1.29` | `-1.11` | `-0.04` | `-0.10` | `7.16` | `+0.19` / `+1.06` / `+2.16` |
| `L21N3` | `natural_slot` | `' me'` | `+0.06` | `+0.14` | `+1.80` | `-1.57` | `-0.29` | `-1.06` | `82.52` | `-0.31` / `+0.47` / `+0.56` |
| `L23N6941` | `natural_slot` | `' AI'` | `+0.44` | `+0.61` | `+0.21` | `-0.27` | `-0.15` | `-0.05` | `6.13` | `+0.06` / `+0.73` / `+0.41` |
| `L25N4074` | `natural_slot` | `' I'` | `+0.06` | `+2.41` | `-3.09` | `+0.71` | `+0.02` | `+0.47` | `106.62` | `+1.28` / `+0.30` / `-0.50` |
| `L26N3854` | `natural_slot` | `' assistant'` | `+0.72` | `+0.98` | `-0.14` | `-0.09` | `-0.03` | `+0.07` | `4.03` | `+1.06` / `+0.28` / `-0.06` |
| `L28N2566` | `natural_slot` | `' I'` | `+1.34` | `+1.97` | `-0.70` | `+0.10` | `-0.03` | `+0.10` | `2.27` | `+2.34` / `+0.50` / `+0.62` |

- **Short-context `sliding` vs. `full` highway comparison (`L6` `post_norm`, all 4 frames):** Interior `full_attention` layers (`L7, 11, 15, 19, 23, 27`) show slightly smaller `da_norm` than flanking `sliding_attention` layers (`R_ridge`: `0.443` vs. `0.499`, `p = 0.049`; `logit_lens`: `0.428` vs. `0.484`, `p = 0.036`; `R_mf |attn|`: `0.0212` vs. `0.0265`, `p = 0.014`), while terminal `L31*` (`full_attention`) yields the single largest layer total on `L4N6633` (`+0.93` Attn + `+2.34` MLP = `+3.27`).

---

## 5. Prior Art Positioning (External Literature Context)

External literature notes (`drops/Transport Lenses & Hybrid Interpretability.md`; URLs verified resolving on 2026-11-22; external paper claims are not independently re-derived from raw third-party code in this bundle):
- **Zero-training SwiGLU synthesis from `S`/`R` lenses:** Prior work uses RelP `R`/`S` lenses for cataloging (`stdlogits/neuron-lens`, 2026), unconditional residual patching (`J++ Lens`, Ayonrinde & Lindsey 2026; `R-Lens`, Blank et al. 2026), or repairing *existing* late neurons (`Astra`, `arXiv:2610.05277`; `AlphaEdit`, `arXiv:2410.02355`).
- **Role-filler / persona-slot framing (`arXiv:2611.19176`, Dawlah et al. 2026; `arXiv:2608.07852`):** Dawlah et al. test whether linear J/R-lens *workspaces natively represent role-filler binding*, while `arXiv:2608.07852` localized Assistant/persona identity at turn-boundary and pronoun tokens. Our Exp 2B is **complementary** — showing an engineered multiplicative SwiGLU gate can inject a target word selectively on self-queries with zero training — and does **not** claim to overcome Dawlah et al.'s role-filler binding barrier (which requires a subject-swap / relation-preserving battery; Chorus `#5`).
- **Write geometry & post-norm site:** Our `pre_norm` vs. `post_norm` damping is **consistent with** BlackboxNLP 2026 (`2026.blackboxnlp-1.66`), and our `u_t` vs. `R_mf` contrast relates to Hydra/self-repair work (McGrath et al. 2023; Rushing & Nanda 2024) and residual-metric lens studies (`arXiv:2610.22169`, `arXiv:2606.12629`).
- **Exact Final-RMSNorm highway decomposition:** Follows ICLR 2026 (`arXiv:2507.02559`) and Rushing & Nanda (ICML 2024, `arXiv:2402.15390`) by separating the explicit `RMSsh` denominator term from downstream `Attn`/`MLP` terms (`arXiv:2609.18612`).

---

## 6. Limits & Council-Converged Next Steps (Option D → Stage C0 → Stage C1)

### Limits
1. **Exploratory (not pre-registered):** Thresholds, 8 personas, 4 layers, and dose ladders were chosen to map OLMo 3 7B's geometry; formal claims require pre-registration with held-out targets, CIs, and blind grading.
2. **Word injection vs. persona adoption & `pre_norm` amplitude:** Pure-lens `L18`/`L24 virtual_slot_S_ridge` often inserts the target word into an AI-assistant frame (*"I pirate! (Just kidding!) I pirate is an AI language model..."*) or loops at high dose (`L24 librarian`); separating word injection from persona adoption requires Part 213 copula controls, multi-question follow-through, a blind reader, and write-only controls (Claude `#3`, Chorus `#5`). At `pre_norm`, RMSNorm damping prevented greedy flips in Exp 2B; `post_norm` is an instrument, not a deployed weight.
3. **Missing controls in Exp 4B telescoping:** `Exp 4B` compared `R_ridge`, `R_mf`, and `logit_lens` without random-direction, direct-gain-matched, or shuffled-lens controls — required in Stage C0 to separate direction-specific transport from generic non-propagation (Weaver `#4`, Chorus `#5`).
4. **Shallow-layer lens divergence vs. direct causal push provenance:** Our 64-doc `R_out` diverges from `mhough`'s J-lens in shallow layers (`L0 +0.3178`, `L3 +0.5369`, `L6 +0.6886`). In Exp 4A, `L4N6633` (`+10.52`) and `L5N6875` (`+8.16`) were *selected* via the lens census, while their reported logit gains and `Attn`/`MLP` terms come strictly from **actual intervened forward passes (`+10.0` activation push on the real neuron) and exact measured final logits**, not lens projections.
5. **Short context (`< 4096` tokens) & single checkpoint:** All prompts sat inside OLMo 3's 4,096-token sliding window (`sliding_attention` masks non-binding), and only `Olmo-3-7B-Instruct` was run. The `9.26 CU` 64-doc `R+S` lenses are reusable assets carried forward.

### Council-Converged Next Steps (`open-olmo3-7b-virtual-neurons-2026-11-22`)
1. **Step 1 (Option D):** Publish this exploratory baseline & stdlib-only scorer after cold-check fixes.
2. **Step 2 (Stage C0 — Pre-Registered Predictive Transport Benchmark):** See `OLMO3-TRANSPORT-C0-DESIGN.md` (updated to v2 with all F1–F7 fixes from adversarial review).
3. **Step 3 (Stage C1, gated on C0):** Discovery/held-out split path/activation-patching of downstream difference signals + same-layer random controls + coherence gate + verified Base → SFT → DPO → Instruct ladder (`embed_tokens` cosines verified `>= +0.9977`).
