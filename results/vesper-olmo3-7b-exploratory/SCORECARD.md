# OLMo 3 7B RelP Neuron Lens, Virtual SwiGLU Synthesis & Signal Highway — Scorecard

- **Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`
- **Lens fit:** `64` WikiText-103 docs × `128` tokens on `NVIDIA A100-SXM4-40GB` (`3933.41 s`, median `60.146 s/doc`, peak VRAM `17.24 GB`)
- **Split-half fp32 endpoints:** `R_mid` `0.419736 -> 0.013182`, `R_out` `0.418949 -> 0.032257`, `S_mid` `0.006695 -> 0.406871`, `S_out` `0.008626 -> 0.413864`
- **Cross-val `R_out` vs `mhough` 616-doc J-lens:** `L15 = +0.8997`, `L18 = +0.9474`, `L21 = +0.9702`, `L24 = +0.9805`, `L27 = +0.9868`, `L30 = +0.9989`

## Idea #5: 1,024-Head Sliding vs. Full Attention (Depth-Controlled)

- **Top-20 / Top-50 copy heads in `full_attention`:** `5/20` (`25%` base rate), `16/50`
- **OV Frobenius norm spikes:** `L18=2.556` -> `L19*=3.517` -> `L20=2.593`; `L22=3.12` -> `L23*=4.264` -> `L24=4.023`

## Idea #2A: Write-Geometry Benchmark (`WHO_CHAT_TF`, `post_norm`, `dose=1.50`)

| Layer | `R_ridge` dL (`z`, top1) | `R_mf` dL (`z`, top1) | `logit_lens` dL (`z`, top1) | `random` dL (`z`) | Gen hits (`R_rg`/`R_mf`/`LL`/`Rnd` of 4) |
|---:|---:|---:|---:|---:|---:|
| L6 | +0.75 (+0.77, 0/8) | +2.29 (+1.39, 0/8) | +0.56 (+0.74, 0/8) | +0.33 (+0.76) | 0/0/0/0 |
| L12 | +3.01 (+2.69, 0/8) | +6.96 (+3.55, 0/8) | +1.49 (+1.75, 0/8) | -0.19 (-0.39) | 0/0/0/0 |
| L18 | +12.22 (+11.70, 1/8) | +16.17 (+11.13, 1/8) | +6.42 (+7.00, 0/8) | +0.03 (-0.39) | 4/4/1/0 |
| L24 | +25.34 (+25.85, 8/8) | +27.99 (+20.13, 8/8) | +18.02 (+20.93, 5/8) | +0.37 (-0.34) | 4/4/4/0 |

## Idea #2B: Hand-Built Virtual SwiGLU Neurons (`post_norm`, `R_ridge` write)

| Layer | Variant | Self Act | Off Act | Concept Act | Self dL | Concept dL | Off KL | Med Sel (S/O) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| L18 | `virtual_concept_S_ridge` | +0.066 | +0.017 | +0.010 | +2.22 | -1.62 | 0.0356 | 3.6× |
| L18 | `virtual_concept_S_mf` | +0.146 | +0.020 | -0.010 | +1.95 | -1.89 | 0.0788 | 4.2× |
| L18 | `virtual_slot_S_ridge` | +0.412 | -0.010 | +0.011 | +9.26 | +0.75 | 0.0462 | 39.8× |
| L18 | `virtual_slot_contrastive` | +89.984 | -0.516 | +5.331 | +4.48 | +0.26 | 0.0003 | 174.3× |
| L24 | `virtual_concept_S_ridge` | +0.199 | +0.040 | +0.492 | +0.67 | +1.64 | 0.0009 | 1.2× |
| L24 | `virtual_concept_S_mf` | +0.193 | +0.021 | +0.057 | +0.35 | +0.26 | 0.0011 | 0.9× |
| L24 | `virtual_slot_S_ridge` | +1.840 | +0.105 | -0.013 | +22.34 | +0.23 | 0.0014 | 17.6× |
| L24 | `virtual_slot_contrastive` | +198.147 | -1.790 | +9.407 | +5.86 | +0.24 | 0.0003 | 110.7× |

## Idea #4: Exact `L+1..31` Signal Highway (`432` traces; median telescoping err `0.01771`, max `0.07605`)

### Virtual Neurons (`WHO_CHAT_TF`, `post_norm`, mean 4 personas)

| Layer | Method | Exact dL | Direct | `ΣAttn` | `ΣMLP` | RMSNorm Shift | CancR |
|---:|---|---:|---:|---:|---:|---:|---:|
| L6 | `R_ridge` | +0.40 | +0.81 | +0.52 | -0.93 | +0.01 | 9.00 |
| L6 | `R_mf` | +0.90 | +0.56 | +0.73 | -0.43 | +0.05 | 3.67 |
| L6 | `logit_lens` | +0.31 | +2.69 | -0.79 | -1.57 | -0.01 | 17.03 |
| L12 | `R_ridge` | +1.47 | +2.04 | +0.92 | -1.51 | +0.02 | 4.00 |
| L12 | `R_mf` | +2.44 | +1.38 | +1.44 | -0.56 | +0.19 | 2.85 |
| L12 | `logit_lens` | +0.89 | +4.60 | -1.30 | -2.43 | +0.02 | 11.39 |
| L18 | `R_ridge` | +5.25 | +5.19 | +0.98 | -1.09 | +0.18 | 1.98 |
| L18 | `R_mf` | +7.62 | +3.70 | +3.47 | +0.22 | +0.25 | 1.57 |
| L18 | `logit_lens` | +2.50 | +8.93 | -4.01 | -2.39 | -0.01 | 7.32 |
| L24 | `R_ridge` | +11.59 | +10.56 | +0.34 | +0.77 | -0.09 | 1.31 |
| L24 | `R_mf` | +15.39 | +8.12 | +3.81 | +3.65 | -0.19 | 1.12 |
| L24 | `logit_lens` | +7.23 | +14.56 | -5.96 | -1.11 | -0.25 | 3.30 |

### Natural Neurons (`WHO_CHAT_TF`, push `+10.0`)

| Unit | Kind | Target | Exact dL | Direct | `ΣAttn` | `ΣMLP` | RMSNorm Shift | Late3M | CancR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `L4N6633` | `natural_concept` | ` teacher` | +10.52 | +0.05 | +3.53 | +6.00 | +0.93 | +3.26 | 1.28 |
| `L5N6875` | `natural_concept` | ` librarian` | +8.16 | +0.26 | +1.97 | +4.65 | +1.26 | +4.03 | 1.55 |
| `L14N8245` | `natural_concept` | ` cat` | +3.80 | +0.55 | +0.74 | +2.54 | -0.03 | +1.31 | 2.00 |
| `L16N5264` | `natural_concept` | ` dog` | +3.17 | +0.78 | +1.33 | +1.13 | -0.07 | +0.93 | 2.97 |
| `L19N4502` | `natural_concept` | ` robot` | +0.38 | +0.51 | +0.03 | +0.00 | -0.13 | +0.79 | 13.54 |
| `L23N8559` | `natural_concept` | ` knight` | +0.19 | +0.57 | -0.09 | -0.29 | -0.00 | -0.19 | 9.88 |
| `L17N2444` | `natural_slot` | ` me` | -1.09 | +1.36 | -1.29 | -1.11 | -0.04 | -0.10 | 7.16 |
| `L21N3` | `natural_slot` | ` me` | +0.06 | +0.14 | +1.80 | -1.57 | -0.29 | -1.06 | 82.52 |
| `L23N6941` | `natural_slot` | ` AI` | +0.44 | +0.61 | +0.21 | -0.27 | -0.15 | -0.05 | 6.13 |
| `L25N4074` | `natural_slot` | ` I` | +0.06 | +2.41 | -3.09 | +0.71 | +0.02 | +0.47 | 106.62 |
| `L26N3854` | `natural_slot` | ` assistant` | +0.72 | +0.98 | -0.14 | -0.09 | -0.03 | +0.07 | 4.03 |
| `L28N2566` | `natural_slot` | ` I` | +1.34 | +1.97 | -0.70 | +0.10 | -0.03 | +0.10 | 2.27 |
