# Vesper's OLMo 3 7B Exploratory Toolkit, Virtual SwiGLU Synthesis & Signal Highway

**Status:** Exploratory empirical study & toolkit release (`V-006`; **not pre-registered**).  
**Full Report:** [`olmo3_virtual_and_highway/OLMO3-VIRTUAL-NEURONS-AND-HIGHWAY-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-VIRTUAL-NEURONS-AND-HIGHWAY-REPORT.md)  
**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`

## Summary
- **64-doc RelP Neuron Lens (`R + S`, `9.26 CU` A100, reusable asset carried forward):** Mirror-symmetric `fp32` split-half (`S_mid` `0.0067 -> 0.4069`, `R_mid` `0.4197 -> 0.0132`); `R_out` cross-validates against `mhough/olmo3-jacobian-lenses` (616-doc true J-lens) at `+0.3178` (`L0`), `+0.5369` (`L3`), `+0.6886` (`L6`), `+0.7938` (`L9`), `+0.8385` (`L12`), `+0.8997` (`L15`), `+0.9474` (`L18`), `+0.9702` (`L21`), `+0.9805` (`L24`), `+0.9868` (`L27`), and **`+0.9989` (`L30`)**.
- **Idea #5 (1,024-head depth-controlled scan, `0 CU`):** No global `full_attention` vs. flanking `sliding_attention` mean shift (`self_z p = 0.996`, `copy_bg p = 0.211`; `5/20` top copy heads in `full`), mid-late `full_attention` OV Frobenius norm spikes at `L19*` (`3.517` vs `2.556/2.593`, `+36%`) and `L23*` (`4.264` vs `3.120/4.023`, `+37%`), and clean specialist heads (`L22H9` SELF, `L21H25`/`L24H20` ROLE, `L26H2` NAME, `L24H31` PERSONA).
- **Idea #2 (Zero-training virtual SwiGLU neurons, `0.53 CU` L4):** Pure-lens `virtual_slot_S_ridge` + `R_ridge` reaches `self_act = +0.412` vs `off_act = -0.010` (`39.8×`, `+9.26` dL, `off_KL = 0.0462`) at `L18` and `+1.840` vs `+0.105` (`17.6×`, `+22.34` dL, `off_KL = 0.0014`) at `L24`; `virtual_slot_contrastive` reaches `174.3×` / `110.7×` (`off_KL = 0.0003`). Confirms BlackboxNLP 2026 post-MLP RMSNorm damping (`3×–5×` reduction at `pre_norm` vs `post_norm`).
- **Idea #4 (Exact 4-term `L+1..31` signal highway, `432` traces, `bf16` telescoping err `median = 0.01771`, `p95 = 0.05215`, `max = 0.07605`):** Raw `logit_lens` (`u_t`) injects the largest direct write (`+8.93` at `L18`, `+14.56` at `L24`) and is actively erased downstream (`ΣAttn = -4.01/-5.96`, `ΣMLP = -2.39/-1.11`, `CancR = 7.32/3.30`; `91.58` at `L18 pre_norm`), whereas `R_mf` recruits active amplification (`ΣAttn = +3.47/+3.81`, `ΣMLP = +0.22/+3.65`, `CancR = 1.57/1.12`) and `R_ridge` hits the specificity frontier (`+25.85σ` at `L24`). Direct causal pushes (`+10.0` act) on early natural Concept neurons `L4N6633` / `L5N6875` grow from `+0.05`/`+0.26` direct to `+10.52`/`+8.16` final logit via `L29–L31`, while late pronoun Slot neurons (`L25N4074`, `L21N3`, `L17N2444`) show active `Attn/MLP` cancellation (`CancR = 106.62`, `82.52`, `7.16`; `RMSsh ≈ 0`).

## Rescore it yourself (Python 3.10+ standard library only)

From a clone of this repository (`cd assistant-basin`), using **only** the Python 3.10+ standard library (zero `numpy`, `scipy`, or `torch` required):

```bash
python3 -I olmo3_virtual_and_highway/score_olmo3_study.py \
  --lens-dir results/vesper-olmo3-7b-exploratory \
  --exp-dir results/vesper-olmo3-7b-exploratory \
  --out-dir /tmp/olmo3_rescore
cmp /tmp/olmo3_rescore/scorecard.json results/vesper-olmo3-7b-exploratory/scorecard.json
cmp /tmp/olmo3_rescore/SCORECARD.md results/vesper-olmo3-7b-exploratory/SCORECARD.md
```

Both `cmp` commands print nothing and exit `0` when reproduced byte-for-byte.

## Files
- `fit_meta.json` (`4e4291a8...`): 64-doc RelP fit metadata & exact `fp32` split-half convergence (`r_lens.pt` `ceef21fc...` and `s_lens.pt` `a145860b...` are 2.0 GB each and archived externally; their SHA-256 provenance is recorded in `scorecard.json`).
- `olmo3_lens_analysis.json`: 31-layer `R_out` cross-validation (`danorel`/`mhough`) + 1,024-head OV scan.
- `natural_neuron_census_preview.json`: 352,256-neuron zero-GPU Concept/Slot candidate scan.
- `olmo3_idea2_and_idea4_results.json.gz` (uncompressed SHA-256 `d376b612...`): Full Exp 2A, Exp 2B, and Exp 4 receipts.
- `scorecard.json` (`dbfcd8aa...`) & `SCORECARD.md` (`16644ddd...`): Deterministic stdlib-only scorecard outputs.
