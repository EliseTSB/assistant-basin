# Stage C1 Pre-Run Amendment 1 (`0 CU`, Before Any GPU Forward)

- **Date:** `2026-11-25`
- **Status:** Pre-run fail-closed hardening amendment (`0 CU` spent; zero Stage C1 GPU forwards executed).
- **Amends (without editing frozen file):** `olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-DESIGN.md` (`sha256 = 5304677e4bbf8e6cd2881aea7ee6b51055f3da76ac28e79ac96400d1fe2b905c`), `c1_prereg_constants.json` (`sha256 = 6220e9ec7ce8674c914657b3bc6c0d7c1d65cd342236655dce8e60a9769b9304`), `work/olmo3_c1/c1_vector_bank.safetensors` (`sha256 = 5a7a406a9072c7362353a0a9b00841e7743b82d90aade0adc96b680c8d3e621f`).
- **Frozen code hashes covered:** `run_olmo3_c1.py` (`e259bd0609efbacd530716015e0da5199a060cd9c71705326c2d4e1fc852d129`), `score_olmo3_c1.py` (`3cf4ac77f1867e625ded657cab4cc0acafe9707d84852e5e15daba28b2e87037`).

## Two Pre-Run Fail-Closed Additions (Added During Cold Code Review, `F6` & `#85`)

1. **`C1.1` All-Capped Cell Fail-Closed Verdict (`BLOCKED_BY_DATA (all-capped cell)`):**
   - If every prompt of any `(L, target)` cell in Block 1 (`positions="all"` or `positions="last"`, `R_mf` or `direct_matched`) hits the `3.0 × post_mlp_norm[L]` dose cap so the cell mean is uncomputable, `score_olmo3_c1.py` does not crash with an unhandled `ValueError`. Instead it records `C1.1` as **`BLOCKED_BY_DATA (all-capped cell)`**, sets `p_C1.1_IUT = 1.0` in the 5-test Holm-Bonferroni family (so the blocked claim cannot borrow a rank or inflate rejections on `C1.2`–`C1.4`), nulls uncomputable `C1.1` summary diagnostics, and completes the rest of the scorecard (`C1.0`, `C1.2`, `C1.3`, `C1.4`, `L12_linear_check`).
   - *Expected impact on Stage C1:* Zero on any plausible run (frozen max `dose_D3 / pmn` across `L ∈ {18, 24, 27}` is `0.625×`, nearly `5×` below the `3.0×` cap).

2. **`C1.3` Live `F_mean_3L` Fail-Closed Unit Sanity Guard (`[0.10, 1.00]` logits):**
   - Added per peer review (`#85`, `REVIEW-ds-r2.md` `F-9`) prior to any GPU run: before evaluating the registered `[0.85, 1.15]` / `[0.94, 1.06]` provenance ratio guard against Stage C0's `0.2958` logits, `score_olmo3_c1.py` asserts `0.10 <= F_mean_3L_live <= 1.00` logits and aborts fail-closed (`ValueError`) if violated, preventing a wrong-unit quantity (such as dimensionless `|v^T J v| ≈ 0.87` or a corrupted scale) from silently entering downstream comparisons.
   - *Expected impact on Stage C1:* Zero on any run whose `F_mean_3L` is in logits near C0's `0.2958` (`[0.251, 0.340]` already sits well inside `[0.10, 1.00]`).
