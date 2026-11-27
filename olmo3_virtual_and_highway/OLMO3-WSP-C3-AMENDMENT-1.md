# Stage C3 Pre-Run Amendment 1 (`0 CU`, Before Any GPU Forward)

- **Date:** `2026-11-27`
- **Status:** Pre-run fail-closed hardening amendment (`0 CU` spent; zero Stage C3 GPU forwards executed).
- **Amends (without editing any frozen registered file):**
  - `olmo3_virtual_and_highway/OLMO3-WSP-C3-DESIGN.md` (`sha256 = dbdab600e72fdba21f22024d0b1a90575cd2d6d82195fdb6b5fe4a49f494023a`)
  - `olmo3_virtual_and_highway/c3_prereg_constants.json` (`sha256 = bb568d8a29c3d3515563844caaffa9ede9d4c2bc926ed78c68484ada4a1d4fec`)
  - `olmo3_virtual_and_highway/c3_vector_bank.safetensors` (`sha256 = e264cea9785f91f07a34c7e3b37ed8451474c888a340a0ec13ed63e8e6f820b2`)
  - `olmo3_virtual_and_highway/c3_scoring.py` (`sha256 = b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f`)
- **Registered base commit on public `main`:** `10953eac` (`vesper-c3-wsp` base `146e5ad5`).

## Pre-Run Fail-Closed Clarification (Added During Round-1 Cold Code Review, `#148` / `GLM F1`)

1. **`H2` (`C3.1b`) `d_common` `bf16` Representability Gate (`DOSE_INVALID`):**
   - `H2` compares `WSP_Pirate_Contrastive` vs. `WSP_Pirate_Lexicon` at the matched dose `d_common = min(d*_con, d*_lex)`. Because `d_common` may sit below one arm's own selected `d*`, checking only each arm's selected `d*` could allow `H2` to compare a live edit against an attenuated, `bf16`-unrepresentable control rung (`stored_rank1_proj_ret < 0.50` or `stored_frob_ret < 0.80`).
   - **Binding rule:** Prior to evaluating `H2`, `score_olmo3_c3.py` verifies from `commit_gate.arms[*].rungs` (and cross-checks `doses.d_common_retention`) that **both** `WSP_Pirate_Contrastive@d_common` **and** `WSP_Pirate_Lexicon@d_common` satisfy the registered `bf16` storage representability requirement (`stored_rank1_proj_ret >= 0.50` AND `stored_frob_ret >= 0.80` on every `L ∈ {21, 24, 27}`).
   - If either arm's `d_common` rung is unrepresentable, `H2` resolves fail-closed to **`DOSE_INVALID`** (scoped strictly to `H2`; `H1`, `H3`, and `H4` remain unaffected), and both arms' `d_common` `bf16` retention tables are surfaced directly alongside the `H2` readout in `c3_scorecard.json` and `C3-SCORECARD.md`.

2. **Operational Calibration Token-Position Clarification (`DS R1` Note):**
   - In `OLMO3-WSP-C3-DESIGN.md` §2.1, "across the assistant turn-start token (`<|im_start|>assistant\n`) plus the first `16` generated tokens" denotes the `1 + min(16, T)` teacher-forced positions beginning at the last prompt token (`len(prompt_ids) - 1`, which predicts the first assistant token) through the first `16` generated tokens.
