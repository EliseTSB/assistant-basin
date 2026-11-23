# Vesper's OLMo 3 7B Exploratory Toolkit, Virtual SwiGLU Synthesis & Signal Highway

**Status:** Exploratory empirical study & toolkit release (`V-006`; **not pre-registered**).  
**Full Report:** [`olmo3_virtual_and_highway/OLMO3-VIRTUAL-NEURONS-AND-HIGHWAY-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-VIRTUAL-NEURONS-AND-HIGHWAY-REPORT.md)  
**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`

## Summary (Exploratory Observations)
- **64-doc RelP Neuron Lens (`R + S`, `9.26 CU` A100 GPU, reusable asset carried forward):** Mirror-symmetric `fp32` split-half (`||A-B||/||A+B||`, lower = tighter: `S_mid` `0.0067 -> 0.4069`, `R_mid` `0.4197 -> 0.0132`); `R_out` vs. `mhough/olmo3-jacobian-lenses` across 11 layers rises from `+0.3178` (`L0`), `+0.5369` (`L3`), `+0.6886` (`L6`) to `+0.7938` (`L9`), `+0.8385` (`L12`), `+0.8997` (`L15`), `+0.9474` (`L18`), `+0.9702` (`L21`), `+0.9805` (`L24`), `+0.9868` (`L27`), and **`+0.9989` (`L30`)**.
- **Idea #5 (1,024-head depth-controlled scan):** No significant global `full_attention` vs. flanking `sliding_attention` mean shift across all 7 tested metrics (closest: `name_z p = 0.088`, `ov_frob p = 0.161`; `self_z p = 0.996`; `5/20` top copy heads in `full`). Bilateral OV norm spike at `L19*` (`3.517` vs `2.556`/`2.593`, `+37.6%`/`+35.7%`); step onto late plateau at `L23*` (`4.264` vs `L22 = 3.120` `+36.7%`, `L24 = 4.023` `+6.0%`).
- **Idea #2 (Zero-training virtual SwiGLU neurons, `0.53 CU` NVIDIA L4 GPU):** Pure-lens `virtual_slot_S_ridge` + `R_ridge` reaches `self Δlogit = +9.26` (`self_act = +0.412` vs `off_act = -0.010`, `off_KL = 0.0462` over 6 ordinary prompts) at `L18` and `+22.34` (`+1.840` vs `+0.105`, `off_KL = 0.0014`) at `L24`; `virtual_slot_contrastive` reaches `off_KL = 0.0003` with lower `self Δlogit` (`+4.48`/`+5.86`). Depth-dependent `pre_norm` vs. `post_norm` damping (`~13.4–15.8×` at `L12`, `~4.2–6.3×` at `L18`, `~3.7–4.5×` at `L24` at matched relative site norm) is consistent with BlackboxNLP 2026 post-MLP RMSNorm scale-invariance.
- **Idea #4 (Exact 4-term `L+1..31` signal highway, `432` traces, telescoping err `median = 0.01771`, `p95 = 0.05215`, `max = 0.07605`):** Raw `logit_lens` (`u_t`) injects the largest direct write (`+8.93` at `L18`, `+14.56` at `L24`) followed by large negative downstream terms (`ΣAttn = -4.01/-5.96`, `ΣMLP = -2.39/-1.11`, gross-over-net `CancR = 7.32/3.30`), whereas `R_mf` yields positive downstream terms (`ΣAttn = +3.47/+3.81`, `ΣMLP = +0.22/+3.65`, `CancR = 1.57/1.12`, `~2–3×` larger net gain) — consistent with active erasure vs. constructive transport, pending Stage-C0 random-direction controls. Direct causal pushes (`+10.0` act) on early natural Concept neurons `L4N6633` / `L5N6875` reach `+10.52` / `+8.16` via `L29–L31`, while late natural Concept (`L19N4502`, `L23N8559`) and Slot candidates show heterogeneous downstream cancellation (`RMSsh` near zero).

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
- `fit_meta.json` (`4e4291a8...`): 64-doc RelP fit metadata & exact `fp32` split-half convergence (`r_lens.pt` `ceef21fc...` and `s_lens.pt` `a145860b...` are archived externally; their SHA-256 provenance is pinned in `scorecard.json`).
- `olmo3_lens_analysis.json`: 31-layer `R_out` cross-validation (`danorel`/`mhough`) + 1,024-head OV scan.
- `natural_neuron_census_preview.json`: Concept/Slot candidate scan.
- `olmo3_idea2_and_idea4_results.json.gz` (uncompressed SHA-256 `d376b612...`): Full Exp 2A, Exp 2B, and Exp 4 receipts.
- `scorecard.json` (`dbfcd8aa...`) & `SCORECARD.md` (`16644ddd...`): Deterministic stdlib-only scorecard outputs.
