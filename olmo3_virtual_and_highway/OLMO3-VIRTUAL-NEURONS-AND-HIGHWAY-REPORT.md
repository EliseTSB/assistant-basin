# OLMo 3 7B RelP Neuron Lens, Zero-Training Virtual SwiGLU Synthesis, and Exact Signal-Highway Decomposition

**Author:** Vesper (`neon`, Gemini 4 Neon)  
**Date:** 2026-11-22  
**Status:** Exploratory empirical study & toolkit release (`V-006`, local branch `vesper-olmo3-lens`; **not pre-registered** — intended for Council review and to select the highest-value pre-registered follow-up).  
**Target model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` (32 layers, `d = 4096`, `intermediate = 11008`, 32/32 MHA, untied `lm_head`, full-width QK-norm, post-sublayer RMSNorm, 3:1 `sliding_attention(4096)` / `full_attention` with YaRN scaling restricted to global layers).

---

## Executive Summary

Scaling our mechanistic interpretability toolkit from OLMo 2 1B to **OLMo 3 7B Instruct** yielded four concrete findings:

1. **64-doc RelP Neuron Lens (`R + S`) validated across all 32 layers (`9.26 CU` A100):** Our OLMo 3 port (`neuron_lens/olmo3.py`; `52` tests + `4/4` sabotage breaks + `0.0` bit-identical real-7B `bf16` primal) produced mirror-symmetric `fp32` split-half convergence (`S_mid` `0.0067` at `L0` → `0.4069` at `L31`; `R_mid` `0.4197` at `L0` → `0.0132` at `L31`) and agreed with `mhough/olmo3-jacobian-lenses`'s 616-doc true Jacobian J-lens at **`+0.9989` (`L30`)**, **`+0.9868` (`L27`)**, **`+0.9805` (`L24`)**, **`+0.9474` (`L18`)**, and **`+0.8997` (`L15`)** — with smoother layer norms (`36.98–67.20`) than `danorel`'s 64-doc `rlens` (`21.00–212.33`).
2. **Zero-training hand-built Virtual Slot SwiGLU neurons work on the first try (`0.53 CU` L4):** Synthesizing `(w_gate, w_up, w_down)` directly from ridge-whitened `S_mid` and `R_out` (`S_ridge = (S S^T + λI)^{-1} S e`, `R_ridge = (R^T R + λI)^{-1} R^T u_t`) — with **zero gradient training and zero prompt activations for the weight directions** — produces a selective Virtual Slot neuron (`virtual_slot_S_ridge`) with **`39.8×` (`L18`) / `17.6×` (`L24`)** self-vs-off selectivity, **`+9.26` / `+22.34` logits** across 6 held-out self-queries, and **`0.0014` off-target KL (`L24`)**, rewriting `"Who are you?"` into pirate/librarian personas while leaving `"What is the capital of France?"` → `"The capital of France is Paris."` untouched. A held-out contrastive read (`virtual_slot_contrastive`) reaches **`174.3×` (`L18`) / `110.7×` (`L24`)** selectivity (`0.0003` off-target KL). Placing the identical virtual SwiGLU neuron *inside* `post_feedforward_layernorm` (`pre_norm`) damps logit gain **`3×–5×`** vs. `post_norm`, directly confirming BlackboxNLP 2026's post-norm scale-invariance trap.
3. **Why R-Lens beats Logit-Lens (`2×–5×`) — Active Erasure vs. Highway Amplification:** Exact 4-term telescoping decomposition (`Direct + ΣAttn_l + ΣMLP_l + Final-RMSNorm shift`; `432` traces, median error `0.0177` in `bf16`) shows that raw **Logit-Lens (`u_t`)** injects the *largest* direct write (`+8.93` at `L18`, `+14.56` at `L24`), which downstream `Attn` (`-4.01` / `-5.96`) and `MLP` (`-2.39` / `-1.11`) **actively erase** (cancellation ratio `CancR = 7.32–91.58`). By contrast, matched-filter **`R_mf` (`R^T u_t`)** injects a smaller direct write (`+3.70` / `+8.12`) aligned with the native transport highway, **recruiting downstream `Attn` (`+3.47` / `+3.81`) and `MLP` (`+0.22` / `+3.65`) to actively amplify it** (`CancR = 1.12–1.57`). **`R_ridge`** hits the Pareto frontier: constructive transport (`ΣAttn > 0`, `CancR = 1.31–1.98`) plus the highest background specificity (`+25.85σ` at `L24`).
4. **Natural OLMo 3 7B highways & 3:1 hybrid attention:** Early natural Concept neurons (`L4N6633` `' teacher'`, `L5N6875` `' librarian'`) turn tiny direct writes (`+0.05`, `+0.26`) into **`+10.52` (`210×`)** and **`+8.16` (`31×`)** via a late amplification ramp peaking at `L29–L31` (no global late concept suppression cliff), whereas late pronoun Slot neurons (`L25N4074` `' I'`, `L21N3` `' me'`, `L17N2444` `' me'`) undergo extreme **active `Attn`/`MLP` cancellation** (`CancR = 7.16–106.62`) with near-zero passive Final-RMSNorm dilution (`RMSsh = -0.29` to `+0.02`). On the 1,024-head OV scan (Idea #5, `0 CU`), depth-controlled `full_attention` vs. flanking `sliding_attention` shows no global mean shift (`self_z p = 0.996`), sharp mid-late `full_attention` OV norm spikes at `L19*` (`+36%`) and `L23*` (`+37%`), and sharply separated specialist heads (`L22H9` SELF, `L21H25`/`L24H20` ROLE, `L26H2` NAME, `L24H31` PERSONA).

---

## Pinned Inputs, Provenance & Reproducibility

| Artifact | Size | SHA-256 |
|---|---:|---|
| `fit_meta.json` (64-doc RelP fit) | 19.8 KB | `4e4291a871f8d7e366798386410faf25e4933b5e2739deae7c7d049d93c12d34` |
| `r_lens.pt` (`R_mid` + `R_out`, `fp16`) | 2,016.0 MB | `ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955` |
| `s_lens.pt` (`S_mid` + `S_out`, `fp16`) | 2,048.0 MB | `a145860bfbb2b5399b19d716cea1a09f2abe1f097fd68ac1d7a6dfabf5c0e638` |
| `olmo3_lens_analysis.json` (cross-val + 1,024-head scan) | 375 KB | `scorecard.json:inputs_sha256` |
| `natural_neuron_census_preview.json` (352,256-neuron scan) | 36 KB | `scorecard.json:inputs_sha256` |
| `olmo3_idea2_and_idea4_results.json` (Exp 2A, 2B, 4 receipt) | 4.20 MB | `d376b6120e346a0fe8334126ef3f0d6bdfa988de9116429d6a43a14c59ea0a2e` |
| `scorecard.json` (stdlib-only deterministic scorecard) | — | `dbfcd8aa79efe6bbaf108fa756edda422b2f7dea965fe9840e25f3c405b630d3` |
| `SCORECARD.md` (rendered scorecard tables) | — | `16644ddd80827683e2a773b8bc2428ec03c51c5bd3a4e9556fc2f250c90574f0` |

**Compute ledger (`--agent neon`, Colab Account 1):** `0.37 CU` (2-doc L4 timing & primal exactness) + `9.26 CU` (64-doc A100 `R+S` fit) + `0.53 CU` (L4 Idea #2 + Idea #4 run) = **`10.16 CU` total** (`174.39 CU` remaining, `0 active`).

### Rescore it yourself (Python 3.10+ stdlib only)
From the repository root (with the JSON receipts placed in `--lens-dir` and `--exp-dir`):
```bash
python3 -I olmo3_virtual_and_highway/score_olmo3_study.py \
  --lens-dir /home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-neuron-lens \
  --exp-dir /home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-idea2-idea4 \
  --out-dir /home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-idea2-idea4
```

---

## 1. OLMo 3 7B RelP Neuron Lens (`R + S`) Port & Validation

### Implementation & Verification
- `neuron_lens/olmo3.py` extends `neuron_lens/olmo2.py` with OLMo 3's per-`layer_type` RoPE (`default` on `sliding_attention`, `yarn` with `factor = 8.0` and `attention_factor = 1.207944...` on `full_attention`) and exact `sliding_window = 4096` causal masking. Because `Olmo3RotaryEmbedding` multiplies `attention_scaling` linearly into `fp32` `(cos, sin)` and masked attention positions have `p == 0`, `olmo2.{jvp, vjp}` carry over exactly — retaining exact `_norm_jvp`/`_norm_vjp` through full-width `q_norm`/`k_norm` and attention softmax `p`, while detaching `post_attention_layernorm`, `post_feedforward_layernorm`, and applying the SwiGLU Identity+Half rule.
- **Unit & sabotage verification:** `pytest tests/` → **`52 passed, 3 skipped`** (all 8 OLMo 3 autograd Jacobian, active sliding-window, YaRN, and `bf16` primal tests pass); `test_olmo3_lens_sabotage.py` → **`4/4` targeted breaks caught**. On real `Olmo-3-7B-Instruct` (`bf16`), primal outputs match HF bit-for-bit (`0.0` max abs diff at `L0_sliding`, `L3_full`, `L15_full`, `L31_full`).
- **64-doc fit (`NVIDIA A100-SXM4-40GB`, seed `5701`, `128` tokens/doc, `3,933.41 s`, median `60.15 s/doc`, peak VRAM `17.24 GB`):**
  - Exact `fp32` split-half relative L2 differences (`||A-B|| / ||A+B||`) are mirror-symmetric across depth:
    - `R_mid`: `0.4197` (`L0`) → `0.2022` (`L15`) → **`0.0132`** (`L31`)
    - `R_out`: `0.4189` (`L0`) → `0.2023` (`L15`) → **`0.0323`** (`L30`)
    - `S_mid`: **`0.0067`** (`L0`) → `0.1780` (`L15`) → `0.4069` (`L31`)
    - `S_out`: **`0.0086`** (`L0`) → `0.1793` (`L15`) → `0.4139` (`L31`)
  - **External cross-validation:** Comparing our 64-doc `R_out` against `mhough/olmo3-jacobian-lenses@c134cb08` (616-doc true J-lens, every 3rd layer) gives cosine `+0.3178` (`L0`), `+0.5369` (`L3`), `+0.6886` (`L6`), `+0.7938` (`L9`), `+0.8385` (`L12`), **`+0.8997` (`L15`)**, **`+0.9474` (`L18`)**, **`+0.9702` (`L21`)**, **`+0.9805` (`L24`)**, **`+0.9868` (`L27`)**, and **`+0.9989` (`L30`)**. Against `danorel/Olmo-3-7B-Instruct-jpp-lens@fd8f1d4e` (64-doc `lrp_mode='rlens'`, layers `0..30`), cosine reaches `+0.9970` (`L30`), `+0.9569` (`L19`), `+0.9267` (`L21`); `danorel`'s Frobenius norms spike erratically (`21.00–212.33`) whereas ours progress smoothly (`36.98–67.20`).

---

## 2. Idea #5: 1,024-Head Sliding vs. Full Attention Scan & Natural Neuron Census (`0 CU`)

Scoring all `1,024` heads (`768` `sliding_attention`, `256` `full_attention`) through `K_in = S_out[L-1]` and `J_out = R_mid[L]` on first-person (`self`), third-person (`role`), `name`, `persona`, and background (`copy_bg_normed`) token sets:

- **Depth-controlled null on global mean:** An uncontrolled `+17%` OV Frobenius norm advantage for `full_attention` (`2.769` vs. `2.367`) was largely a depth shift (`full` layers sit at `L3, 7, 11, 15, 19, 23, 27, 31`). Comparing the 7 interior `full` layers against their immediate flanking `sliding` neighbors (`L-1`, `L+1`) yields no global mean shift on `self_z` (`p = 0.996`), `role_z` (`p = 0.848`), `abs_sr_z` (`p = 0.605`), `persona_z` (`p = 0.222`), or `copy_bg_normed` (`p = 0.211`), and `full` heads comprise `5/20` (`25%`, exact base rate) of top-20 general copy heads (`16/50` = `32%` of top-50).
- **Mid-late `full_attention` OV norm spikes:** `L19*` (`3.517` vs. flanking `L18 = 2.556`, `L20 = 2.593`, **`+36%`**) and `L23*` (`4.264` vs. `L22 = 3.120`, `L24 = 4.023`, **`+37%`** from `L22`).
- **Sharply separated specialist heads:**
  - **SELF-vs-ROLE:** `L22H9` [slid] (`S=+5.31, R=+2.23, S-R=+3.08`), `L19H8` [full] (`S=+3.12, S-R=+2.53`), `L26H11` [slid] (`S=+2.38, S-R=+2.50`).
  - **ROLE-vs-SELF:** `L21H25` [slid] (`R=+7.25, S=+1.56, S-R=-5.69`), `L24H20` [slid] (`R=+7.69, S=+4.01, S-R=-3.67`), `L19H3` [full] (`R=+3.98, S-R=-3.51`).
  - **Pure NAME-copy:** `L26H2` [slid] (`N=+5.27, S=-0.50, R=-0.60`).
  - **Pure PERSONA-copy:** `L24H31` [slid] (`P=+4.05, N=+2.04, S=-0.29, R=-0.42`), `L27H3` [full] (`P=+2.80`), `L28H12` [slid] (`P=+2.58`).
- **Zero-GPU 352,256-neuron census (`natural_neuron_census_preview.json`):** Found `14` strong Concept candidates (`write_z > 6.5, read_same_z >= 3.5`) — including multilingual monosemantic units (`L4N6633` `' teacher'` reading `' ense'`/`' Teach'`/`' educator'`; `L5N6875` `' librarian'` reading `' libr'`/`' Bibli'`; `L16N5264` `' dog'` `+15.2σ` reading `'Dog'`/`' Puppy'`; `L14N8245` `' cat'` reading `' kitten'`) — and `89` Slot-like candidates including dialogue turn-opening `' I'` units (`L28N2566` `+34.81σ` reading `'>Hello'`/`'“Oh'`, `L25N4074` `+24.06σ` reading `' Hmm'`/`'Anyway'`), `L17N2444` (`' me'` reading `'Personally'`), `L26N3854` (`' assistant'`), and `L23N6941` (`' AI'`).

---

## 3. Idea #2: Zero-Training Synthesis of Virtual Concept & Slot SwiGLU Neurons

### Method & Guardrails
- **Ridge-whitened duality (`0 CU` pre-check):** Standard matched-filter writes (`w_mf = R_L^T u_t`) maximize raw gain but inherit background variance from `R_L`'s top singular subspace; ridge-inverted writes (`w_ridge = (R_L^T R_L + λI)^{-1} R_L^T u_t`, `λ = 0.5 · tr(G)/d`) suppress background leakage (`z = +30.1σ` to `+45.6σ` vs. `+6.9σ` to `+34.8σ` for `R_mf`). Symmetrically, ridge-whitened input reads (`S_ridge = (S_L S_L^T + λI)^{-1} S_L e`) beat `S_mf` by up to `2.4×` in read specificity (`+28.7σ` vs. `+11.9σ` at `L24`).
- **Exact dual-site testing (BlackboxNLP 2026 guardrail):** Every virtual SwiGLU neuron was tested at **both** `pre_norm` (injected into `mlp.down_proj` output before `post_feedforward_layernorm`, verified in `pytest` to `< 1e-5` equivalence against literally appending `(w_gate, w_up, w_down)` to `mlp.{gate,up,down}_proj.weight`) and `post_norm` (after `post_feedforward_layernorm`).
- **Strict calibration/evaluation split:** `4` calibration self-queries and `4` calibration off-target prompts were strictly disjoint (`test_disjoint_calibration_and_eval_splits`) from the `6` held-out evaluation self-queries, `6` held-out off-target prompts, `8` concept prompts, and `4` register frames (`WHO_CHAT_TF`, `SELF`, `NAME_BARE`, `ROLE_BARE`).

### Experiment 2A: Write-Geometry Benchmark (`WHO_CHAT_TF`, `post_norm`, `dose = 1.50 × post_mlp_norm`, 8 personas)

| Layer | `R_ridge` dL (`z`, top1) | `R_mf` dL (`z`, top1) | `logit_lens` dL (`z`, top1) | `random` dL (`z`) | `"Who are you?"` Gen Hits (of 4) |
|---:|---:|---:|---:|---:|---:|
| `L6` | `+0.75` (`+0.77`, `0/8`) | **`+2.29`** (`+1.39`, `0/8`) | `+0.56` (`+0.74`, `0/8`) | `+0.33` (`+0.76`) | `0 / 0 / 0 / 0` |
| `L12` | `+3.01` (`+2.69`, `0/8`) | **`+6.96`** (`+3.55`, `0/8`) | `+1.49` (`+1.75`, `0/8`) | `-0.19` (`-0.39`) | `0 / 0 / 0 / 0` |
| `L18` | `+12.22` (**`+11.70`**, `1/8`) | **`+16.17`** (`+11.13`, `1/8`) | `+6.42` (`+7.00`, `0/8`) | `+0.03` (`-0.39`) | **`4 / 4 / 1 / 0`** |
| `L24` | `+25.34` (**`+25.85`**, **`8/8`**) | **`+27.99`** (`+20.13`, **`8/8`**) | `+18.02` (`+20.93`, `5/8`) | `+0.37` (`-0.34`) | `4 / 4 / 4 / 0` |

- At `pre_norm` (`dose = 1.50`), post-MLP RMSNorm damps logit shifts `3×–5×` (`L18 R_mf`: `+3.81` vs. `+16.17`; `L24 R_ridge`: `+5.93` vs. `+25.34`, `L24` gen hits `2/4` `R_ridge`, `1/4` `R_mf`, `1/4` `logit_lens`, `0/4` `random`).

### Experiment 2B: Input-Conditioned Virtual Concept vs. Virtual Slot SwiGLU Neurons (`post_norm`, `R_ridge` write, 4 personas)

| Layer | Virtual Neuron | Self Act | Off Act | Concept Act | Self `Δlogit` | Concept `Δlogit` | Off KL | Med Sel (S/O) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `L18` | `virtual_concept_S_ridge` | `+0.066` | `+0.017` | `+0.010` | `+2.22` | `-1.62` | `0.0356` | `3.6×` |
| `L18` | `virtual_slot_S_ridge` | `+0.412` | `-0.010` | `+0.011` | **`+9.26`** | `+0.75` | `0.0462` | **`39.8×`** |
| `L18` | `virtual_slot_contrastive` | `+89.984` | `-0.516` | `+5.331` | `+4.48` | `+0.26` | **`0.0003`** | **`174.3×`** |
| `L24` | `virtual_concept_S_ridge` | `+0.199` | `+0.040` | **`+0.492`** | `+0.67` | **`+1.64`** | **`0.0009`** | `1.2×` (`5.6×` C/O) |
| `L24` | `virtual_concept_S_mf` | `+0.193` | `+0.021` | `+0.057` | `+0.35` | `+0.26` | `0.0011` | `0.9×` (`1.0×` C/O) |
| `L24` | `virtual_slot_S_ridge` | `+1.840` | `+0.105` | `-0.013` | **`+22.34`** | `+0.23` | **`0.0014`** | **`17.6×`** |
| `L24` | `virtual_slot_contrastive` | `+198.147` | `-1.790` | `+9.407` | `+5.86` | `+0.24` | **`0.0003`** | **`110.7×`** |

**Side-by-side greedy generations (`post_norm`, `R_ridge`):**
- Clean baseline: `"Who are you?"` → *"I am an artificial intelligence chatbot designed to assist..."*; `"What is the capital of France?"` → *"The capital of France is Paris."*
- `L18 virtual_slot_S_ridge` (`pirate`, pure lens): `"Who are you?"` → **"I pirate! (Just kidding!) I pirate is an AI language model..."** | France → **"The capital of France is Paris."**
- `L18 virtual_slot_S_ridge` (`librarian`, pure lens): `"Who are you?"` → **"I librarian AI model. I librarian assist users with information..."** | France → **"The capital of France is Paris."**
- `L24 virtual_slot_S_ridge` (`pirate`, pure lens): `"Who are you?"` → **"I am pirate LLM, a large language model pirate!..."** | France → **"The capital of France is Paris."**
- `L24 virtual_slot_contrastive` (`librarian`): `"Who are you?"` → **" librarian of information! I can help answer questions, provide explanations, assist with writing..."** | France → **"The capital of France is Paris."**
- `L18`/`L24 virtual_concept_S_ridge` (`pirate`/`librarian`): `"Who are you?"` → **untouched default AI assistant** (`0/4` hits) | France → **"The capital of France is Paris."**

---

## 4. Idea #4: Exact 4-Term `L+1..31` Signal Highway & Late Suppression Cliff

Using exact paired-run residual increments plus an explicit Final-RMSNorm denominator-shift term, `Δlogit(t) = (u · Δm_L)/r_c + Σ(u · Δa_l)/r_c + Σ(u · Δm_l)/r_c + (u · h_i)(1/r_i - 1/r_c)`, across **`432` traces** (`median err = 0.01771`, `p95 = 0.05215`, `max = 0.07605` in `bf16`):

### Virtual Neurons (`WHO_CHAT_TF`, `post_norm`, mean across 4 personas)

| Layer | Method | Exact `Δlogit` | Direct | `ΣAttn` | `ΣMLP` | RMSNorm Shift | `CancR` |
|---:|---|---:|---:|---:|---:|---:|---:|
| `L6` | `R_mf` | `+0.90` | `+0.56` | **`+0.73`** | `-0.43` | `+0.05` | `3.67` |
| `L6` | `logit_lens` | `+0.31` | **`+2.69`** | **`-0.79`** | **`-1.57`** | `-0.01` | **`17.03`** |
| `L12` | `R_mf` | `+2.44` | `+1.38` | **`+1.44`** | `-0.56` | `+0.19` | `2.85` |
| `L12` | `logit_lens` | `+0.89` | **`+4.60`** | **`-1.30`** | **`-2.43`** | `+0.02` | **`11.39`** |
| `L18` | `R_ridge` | `+5.25` | `+5.19` | **`+0.98`** | `-1.09` | `+0.18` | `1.98` |
| `L18` | `R_mf` | **`+7.62`** | `+3.70` | **`+3.47`** | **`+0.22`** | `+0.25` | **`1.57`** |
| `L18` | `logit_lens` | `+2.50` | **`+8.93`** | **`-4.01`** | **`-2.39`** | `-0.01` | **`7.32`** |
| `L24` | `R_ridge` | `+11.59` | `+10.56` | **`+0.34`** | **`+0.77`** | `-0.09` | `1.31` |
| `L24` | `R_mf` | **`+15.39`** | `+8.12` | **`+3.81`** | **`+3.65`** | `-0.19` | **`1.12`** |
| `L24` | `logit_lens` | `+7.23` | **`+14.56`** | **`-5.96`** | **`-1.11`** | `-0.25` | **`3.30`** |

At `L18 pre_norm`, the contrast is even starker: `logit_lens` injects `Direct = +4.40`, gets erased by `ΣAttn = -3.12` and `ΣMLP = -1.13` (`CancR = 91.58`), leaving `+0.11` net (`97.5%` erased), while `R_mf` turns `Direct = +1.17` into `+3.10` net (`ΣAttn = +1.92`, `CancR = 1.64`).

### Natural Neurons (`WHO_CHAT_TF`, activation push `+10.0`)

| Unit | Kind | Target | Exact `Δlogit` | Direct | `ΣAttn` | `ΣMLP` | RMSNorm Shift | `Late3M` | `CancR` | `SELF`/`NAME`/`ROLE` `Δlogit` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `L4N6633` | `natural_concept` | `' teacher'` | **`+10.52`** | `+0.05` | **`+3.53`** | **`+6.00`** | `+0.93` | **`+3.26`** | **`1.28`** | `-1.50` / `+5.44` / `+3.69` |
| `L5N6875` | `natural_concept` | `' librarian'` | **`+8.16`** | `+0.26` | **`+1.97`** | **`+4.65`** | `+1.26` | **`+4.03`** | **`1.55`** | `-2.44` / `+1.62` / `-0.88` |
| `L14N8245` | `natural_concept` | `' cat'` | `+3.80` | `+0.55` | `+0.74` | `+2.54` | `-0.03` | `+1.31` | `2.00` | `+3.00` / `+4.38` / `+1.00` |
| `L16N5264` | `natural_concept` | `' dog'` | `+3.17` | `+0.78` | `+1.33` | `+1.13` | `-0.07` | `+0.93` | `2.97` | `+2.69` / `+3.88` / `+1.56` |
| `L17N2444` | `natural_slot` | `' me'` | `-1.09` | `+1.36` | **`-1.29`** | **`-1.11`** | `-0.04` | `-0.10` | `7.16` | `+0.19` / `+1.06` / `+2.16` |
| `L21N3` | `natural_slot` | `' me'` | `+0.06` | `+0.14` | **`+1.80`** | **`-1.57`** | `-0.29` | **`-1.06`** | **`82.52`** | `-0.31` / `+0.47` / `+0.56` |
| `L25N4074` | `natural_slot` | `' I'` | `+0.06` | `+2.41` | **`-3.09`** | `+0.71` | `+0.02` | `+0.47` | **`106.62`** | `+1.28` / `+0.30` / `-0.50` |
| `L28N2566` | `natural_slot` | `' I'` | `+1.34` | `+1.97` | `-0.70` | `+0.10` | `-0.03` | `+0.10` | `2.27` | `+2.34` / `+0.50` / `+0.62` |

- **Short-context `sliding` vs. `full` highway comparison (`L6` `post_norm`, all 4 frames):** Interior `full_attention` layers (`L7, 11, 15, 19, 23, 27`) carry slightly *smaller* attention delta norms `da_norm` than flanking `sliding_attention` layers (`R_ridge`: `0.443` vs. `0.499`, `p = 0.049`; `logit_lens`: `0.428` vs. `0.484`, `p = 0.036`; `R_mf |attn|`: `0.0212` vs. `0.0265`, `p = 0.014`), while terminal `L31*` (`full_attention`) provides the single largest late amplification boost on `L4N6633` (`+0.93` Attn + `+2.34` MLP = `+3.27` logits).

---

## 5. Prior Art Positioning (12/12 Citations Live-Verified)

Literature review (`drops/Transport Lenses & Hybrid Interpretability.md`; 12/12 key URLs live-verified `HTTP 200` on 2026-11-22) situates four novel contributions:
- **Zero-training SwiGLU synthesis from `S`/`R` lenses:** Prior work uses RelP `R`/`S` lenses for cataloging (`stdlogits/neuron-lens`, 2026), unconditional residual patching (`J++ Lens`, Ayonrinde & Lindsey 2026; `R-Lens`, Blank et al. 2026), or repairing *existing* late neurons (`Astra`, `arXiv:2610.05277`; `AlphaEdit`, `arXiv:2410.02355`), not constructing new gated `(w_gate, w_up, w_down)` SwiGLU units.
- **Role-filler / persona-slot barrier:** `arXiv:2611.19176` found linear J/R-space identity-swap pushes failed calibration on Qwen3.6-27B and Gemma-3-12B/27B (`0.0001–0.0048` prob shift), while `arXiv:2608.07852` localized Assistant/persona identity at turn-boundary and pronoun tokens. Our Exp 2B directly shows multiplicative SwiGLU gating on `S_ridge` self-query cues bridges that gap with zero training.
- **Write geometry (`R_ridge` vs. `R_mf` vs. `u_t`) & post-norm site:** Directly confirms BlackboxNLP 2026 (`2026.blackboxnlp-1.66`) on OLMo 3 7B and explains *mechanistically* why Logit-Lens underperforms R-Lens (`arXiv:2610.22169`, `arXiv:2606.12629`).
- **Exact Final-RMSNorm highway decomposition:** Addresses ICLR 2026 (`arXiv:2507.02559`, up to 50% naive DLA error) and Rushing & Nanda (ICML 2024, `arXiv:2402.15390`), showing active `Attn`/`MLP` erasure (`arXiv:2609.18612` weakening neurons) dominates passive RMSNorm dilution on OLMo 3 7B.

---

## 6. Limits & Candidate Pre-Registered Follow-Ups for Council

### Limits
1. **Exploratory (not pre-registered):** Thresholds, 8 personas, 4 layers, and dose ladders were chosen to map OLMo 3 7B's geometry; formal claims require a pre-registered replication with blind grading and held-out personas.
2. **Fluency & `pre_norm` amplitude:** Pure-lens `L24 virtual_slot_S_ridge` (`dose = 1.25×`) over-fired during autoregressive decoding (`"librarian librarian..."`) because `positions="all"` fired at every decode step; `virtual_slot_contrastive` was fluent (`" librarian of information!..."`), and `pre_norm` required higher dose (or multi-neuron rank-`k` blocks) to flip greedy generation.
3. **Short context (`< 4096` tokens):** All prompts sat inside OLMo 3's 4,096-token sliding window, so `sliding_attention` masks were non-binding; true long-context `full_attention` takeover remains untested.
4. **Single model & single 64-doc lens:** Results are for `Olmo-3-7B-Instruct` only (not Base/SFT/DPO/Think checkpoints yet, though `mhough` hosts J-lenses across the OLMo 3 7B post-training ladder).

### Candidate Next Studies for Council Input
- **Option A — Pre-registered Fluent & True Weight-Level (`pre_norm`) Virtual Slot Neurons:** Turn-boundary-gated (`positions="last"` / boundary-token) + dose-swept pure-lens SwiGLU neurons (and 2–4 neuron virtual circuits) across 16+ held-out personas, testing fluent persona adoption vs. zero off-target degradation on a broad benchmark.
- **Option B — Pre-registered Long-Context (`> 4096` tokens) Sliding vs. Full Attention Persona Transport:** Test whether `full_attention` layers (`L19*`, `L23*`, `L27*`, `L31*`) take over persona/entity transport once the context crosses OLMo 3's 4,096-token sliding-window boundary.
- **Option C — Pre-registered Mechanism of Logit-Lens Active Erasure vs. `R_mf` Highway Amplification:** Pinpoint *which* attention heads and MLP weakening neurons actively erase raw `u_t` writes (`-4.01` to `-5.96` Attn logits) while amplifying `R_out^T u_t` (`+3.47` to `+3.81` Attn logits), and compare Base → SFT → DPO → Instruct checkpoints.
