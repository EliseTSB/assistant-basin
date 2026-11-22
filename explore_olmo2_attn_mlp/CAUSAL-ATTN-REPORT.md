# OLMo 2 1B Instruct — Part B (`#4`: Causal Attention-Reading & Span-Localized Mechanism) Results Report

**Author:** Vesper (`neon`) | **Execution:** Vesper (`neon`, Colab L4, `0.73` CU) under Elise's go & standing small-run budget | **Independent Audit:** Claude (`claude`, `#148` Code PASS, `#153` Smoke PASS, `#156` Rescore PASS)  
**Target Checkpoint:** `allenai/OLMo-2-0425-1B-Instruct@48d788eca847d4d7548f375ad03d3c9312f6139e` (`float32`, `attn_implementation="eager"`)  
**Frozen Package Commit:** `f2d94164c79a75f724857ce79dffc502f9b7dd08` (local branch `vesper-causal-attn`)  
**Operating Norm & Track (Pinned by Part A Scorecard `9e23bba5...`):** `OLD_NORM` (`{L3: 7.392, L5: 10.050, L6: 10.501}` via `both_qualify_lower_perturbation_default`), `p92_09_part_b_track = "include_spec_only_track"`

---

## Executive Summary

Part B tested two pre-registered confirmatory claims on a fresh unseen prompt bank (`B1..B4`, zero overlap with `d13fc3dd`, `01bd41d9`, or Part 180 prompts):

1. **Zero confirmatory claims passed (one uninformative gate failure and one confirmatory null, neither rescued):**
   - **`CM.1a` (`L7H5` on `ROLE_BARE`) is `uninformative (cm1a_gate_failed)`:** Primary unit `p92-46` missed the fixed `+0.50` raw transport floor on `B1..B4` `ROLE_BARE` (`mean_raw = +0.250` vs `mean_spec = +2.130`, where negative peer-median shift `mean_peer_med = -1.881` accounts for almost all of the positive `SpecLogit` readout). Under the frozen drop/stop rule, all `L7H5` `CM.1a`, `CM.2`, and `CM.3` slices (`p92-46` and `p92-09`) remain `uninformative`.
   - **`CM.1b` (`L8H15` on `B1..B4`) is `dropped_confirmatory_failed`:** All cell transport and head-null gates passed (`cm1b_valid = true`), and `L8H15` failed to replicate on the fresh bank (`NAME_BARE` flipped positive `+0.073` rather than suppressing, `C_SN = -0.029`, and `L8H15` trailed both control `L8H3` and active comparator `L8H2`). All `L8H15` `CM.2`, `CM.3`, `CM.4a`, and `CM.4b` verdicts are dropped.
2. **Trustworthy measurement pipeline (`CM.0 = passed`):** Across `8,277` measured GPU forwards (`5,820` B19 + `2,457` B=1, `0.73` CU, timing within `0.13%` of smoke projection), sham, decomposition, derivative, and sibling-head leakage were identically `0.0`, all `420/420` `QKV_all == z_all` checks reconciled to `0.0`, and `17/17` synthetic actuation controls passed (`min_actuation_target_delta = 0.532`). The pre-registered gates halted both confirmatory claims cleanly, which is what makes both the uninformative gate halt and the confirmatory null trustworthy.
3. **Two strictly exploratory / non-confirmatory findings for future work:**
   - **Lead (`L7H5` on `p92-09`):** `p92-09` passed its own unit transport gate (`spec +7.758, raw +5.568`) and showed a large `L7H5` antisymmetric effect (`spec S_odd = +1.410`, `+1.387` over controls) decomposing almost purely as `QK_only` (`f_QK = 1.012`) and mostly upstream (`f_up = 0.668`). Strictly non-confirmatory here; any confirmatory follow-up must register `p92-09` as primary on a fresh unseen bank under the same fixed `+0.50` raw transport floor.
   - **Active comparator `L8H2` on `CM.4`:** Swapping `L8` pre-input activations at the 4 differing referent positions `[13, 20, 30, 31]` moved `L8H2`'s readout by `< 0.03` logits (`small_effect_at_tested_coords`) despite context-only `C_SN = +0.382` raw / `+0.217` spec and verified hook actuation (`min_target_delta = 0.532`). This design cannot separate whether referent identity was already broadcast out of `[13, 20, 30, 31]` before Layer 8 or is carried at unswapped positions (e.g., the read position or shared intervening tokens).

---

## 1. Provenance, Execution, Measured Spend & Two-Way Verification

All five frozen inputs plus the verified GPU smoke receipt were hash-checked on the Colab VM before execution and offline during scoring:

| Artifact | Path / Origin | Size (B) | SHA-256 |
|---|---|---:|---|
| Design (`v3`, unchanged) | `explore_olmo2_attn_mlp/CAUSAL-ATTN-DESIGN.md` | 30,531 | `9b41482fbd1fae267ece14a859274ddf886de3ecd94526e44a4c5697821030fa` |
| Manifest | `explore_olmo2_attn_mlp/causal_attn_manifest.json` | 289,692 | `f8dbda5126f8369112221e9459e0873bd9f4002aaeb609042edc47f375349c67` |
| Runner | `explore_olmo2_attn_mlp/run_causal_attn.py` | 107,490 | `8667e9c17e3b1787d8e405492455608160fc970663773d3ecd22a850b1843135` |
| Scorer | `explore_olmo2_attn_mlp/score_causal_attn.py` | 78,992 | `42f20315945d8b8eee633f0bf9f3ab02410292ee439e58d47ba31af70d988ca2` |
| Part A Scorecard Handoff | `~/green/Journal/notes/reports/vesper-bridge-full-2026-11-21/instruct_bridge_scorecard.json` | 140,102 | `9e23bba5746787f6a1693c906d6b5d10dac9c46be63595fd638b7e4781f0d5f2` |
| Measured GPU Smoke Receipt (`#152`/`#153`) | `~/green/Journal/notes/reports/vesper-causal-attn-smoke-2026-11-21/causal_attn_smoke_receipt.json` | 532,208 | `0f9f4b8c455f6b010cd18444d0e396daf92c2b2ad9c125260cebb366a307637f` |
| **Full Receipt (`#155`)** | `~/green/Journal/notes/reports/vesper-causal-attn-full-2026-11-21/causal_attn_full_receipt.json` *(off-git)* | 22,637,175 | `efbfba170d658558c991a9da7abd14a78aea88b9d9c8cd7257233a79120756aa` |
| **Scored Scorecard (`42f20315` only)** | `~/green/Journal/notes/reports/vesper-causal-attn-full-2026-11-21/causal_attn_scorecard.json` | 59,548 | `a28ce396f427f4ebd8d1bfc40d02b34b12410353e5e52f93dfc0e8601696fb19` |

- **Measured GPU execution & spend:** Ephemeral Colab `NVIDIA L4` (Account 1, session `run-f8ff9e`, auto-terminated, `0 active`; `torch 2.14.1+cu130`, `transformers 5.22.1`, `execution_provenance = "measured_gpu_run"`, `is_cuda_device = true`). Colab ledger (`--agent neon --run vesper-causal-attn-full`): `187.32 → 186.59` = **`0.73` units** (prior 74 s L4 smoke rounded to `0.00` units after >1 hr settle).
- **Schedule & timing:** Exact manifest census executed (`60` unpatched, `2,880` head-mode, `17` actuation, `1,200` swap; `5,820` B19 + `2,457` B=1 = `8,277` forwards; `rc = 0`). `model_load_sec = 29.05 s`, `5,820` B19 in `773.26 s` (`132.86 ms/B19`), `2,457` B=1 in `63.63 s` (`25.90 ms/B1`), `836.90 s` total forward compute (within `0.13%` of the smoke projection `838.00 s`), `1,528.22 s` total runner (`1,576.35 s` VM script wall time).
- **Two-way offline rescore verification:** Vesper (`#155`) and Claude (`#156`, independently running `score_causal_attn.py` from the committed `f2d94164` tree with all five explicit pins) reproduced `causal_attn_scorecard.json` (`a28ce396...`) byte-for-byte.

---

## 2. Pre-Registered Verdicts (`CM.0`–`CM.4`)

| Endpoint | Pre-Registered Verdict | Headline Basis |
|---|---|---|
| **`CM.0` Technical Gate** | **`passed`** | `5/5` clauses true, `0` failures; `0.0` sham/decomp/deriv/sibling leakage; `420/420` `QKV_all == z_all` at `0.0`; `17/17` actuation controls (`min_target_delta = 0.532`). |
| **`CM.1a` `L7H5` (`ROLE_BARE`)** | **`uninformative (cm1a_gate_failed)`** | `p92-46` `ROLE_BARE` unit transport missed the fixed `+0.50` raw floor (`mean_raw = +0.250`, `mean_spec = +2.130`, `mean_peer_med = -1.881`). Hard-gates both `p92-46` and `p92-09` and all `L7H5` `CM.2`/`CM.3` slices. |
| **`CM.1b` `L8H15` (`B1..B4`)** | **`dropped_confirmatory_failed`** | `cm1b_valid = true`, `l8h15_confirmed = false`. On `B1..B4`, `L8H15` `NAME_BARE` went positive (`+0.073`), `C_SN = -0.029`, `C_SR = +0.301`, trailing control `L8H3` (`C_SN = +0.125`, `C_SR = +0.373`) and active comparator `L8H2`. Hard-drops all `L8H15` `CM.2`/`CM.3`/`CM.4`. |
| **`CM.2` `QKV` Factorization** | `L7H5`: `uninformative`; `L8H15`: `dropped`; **`L8H2` (comparator): `interaction_required_within_head`** | `L8H2` `ROLE_BARE_supp` (`spec` `+0.344`), `C_SR` (`spec` `+0.364`, `raw` `+0.513`), and `C_SN` (`raw` `+0.382`) require within-head QK×V interaction; remaining `L8H2` sub-metrics `effect_too_small_or_wrong_sign`. |
| **`CM.3` Position Scope** | `L7H5`: `uninformative`; `L8H15`: `dropped`; **`L8H2` (comparator): `additive_both_within_head`** | `L8H2` `ROLE_BARE_supp` (`spec` `f_rp = 0.349, f_up = 0.793`), `C_SR` (`spec` `0.342 / 0.683`; `raw` `0.335 / 0.665, f_pos_int = 0.000`), and `C_SN` (`raw` `0.337 / 0.661, f_pos_int = 0.002`); remaining 4 of 8 `L8H2` sub-metrics `effect_too_small_or_wrong_sign`. |
| **`CM.4a` / `CM.4b` Span Swaps (`L8` pre-input)** | `L8H15`: `dropped`; **`L8H2` & `L8H3`: `small_effect_at_tested_coords`** | Swapping `L8` pre-input at `[13, 20, 30, 31]` (`CM.4a`) or `[13, 20]` (`CM.4b`) moved `L8H2` and `L8H3` readouts by `< 0.03` logits in both `spec` and `raw`. |

---

## 3. Detailed Breakdown by Endpoint

### 3.1 `CM.0` — Technical Integrity Gate (`passed`)
- `external_pins_verified = true`, `complete_join_verified = true`.
- Clause 1 (5-pin + Part A handoff `OLD_NORM`): `true`.
- Clause 2 (sham, decomposition & derivative finiteness): `true` (`max_abs_sham_spec = 0.0`, `max_abs_sham_raw = 0.0`, `max_abs_decomp_err = 0.0`, `max_abs_deriv_err = 0.0`).
- Clause 3 (`QKV_all == z_all` exact post-norm reconciliation, `420/420`): `true` (`max_reconcile_pre_oproj = 0.0`, `max_reconcile_full_vocab = 0.0`, `max_reconcile_drop = 0.0`).
- Clause 4 (post-`q_norm`/`k_norm`/`v_proj` sibling-head zero leakage): `true` (`max_sibling_leakage_science = 0.0`, `max_sibling_leakage_actuation = 0.0`).
- Clause 5 (synthetic actuation controls, `17/17`): `true` (`min_actuation_target_delta = 0.532109`, threshold `> 1e-2`), verifying that every patching site (post-`q_norm`, post-`k_norm`, post-`v_proj`, `o_proj` input, and `L8` pre-input span swap) actively moves its target activation cleanly.

### 3.2 Transport Gates & `CM.1a` (`L7H5` on `ROLE_BARE`: `uninformative (cm1a_gate_failed)`)

Pooled cell-level transport (`d = +0.8`, `OLD_NORM`, `n = 20` per cell across `5 units × B1..B4`) passed in all three cells, and all seven tested heads (`L7H5`, `L7H4`, `L7H9`, `L8H15`, `L8H2`, `L8H3`, `L8H13`) passed `head_null_pass = true` in every tested cell:

| Cell (Pooled 5 Units) | `n` | `mean_spec` | `mean_raw` | `mean_peer_med` | `max_wdown_null_spec` | Cell Transport Pass |
|---|---:|---:|---:|---:|---:|---|
| `SELF` | 20 | `+3.917` | `+2.912` | `-1.005` | `+0.400` | `True` |
| `NAME_BARE` | 20 | `+3.408` | `+1.994` | `-1.414` | `+0.621` | `True` |
| `ROLE_BARE` | 20 | `+3.016` | `+1.673` | `-1.342` | `+0.210` | `True` |

However, `CM.1a` requires **unit-level** `ROLE_BARE` transport on primary unit `p92-46` (`mean_spec >= +0.50`, `mean_raw >= +0.50`, and `mean_spec > max_wdown_null_spec`):

| Unit (`ROLE_BARE`, `n = 4`) | `mean_spec` | `mean_raw` | `mean_peer_med` | `max_wdown_null_spec` | Unit Transport Pass |
|---|---:|---:|---:|---:|---|
| **`p92-46` (primary)** | **`+2.130`** | **`+0.250`** | **`-1.881`** | `+0.265` | **`False` (`raw < +0.50`)** |
| `p92-09` (secondary, spec-only track) | `+7.758` | `+5.568` | `-2.190` | `+0.802` | `True` |

- **Why `p92-46` missed the raw floor (`B1..B4` `ROLE_BARE`):** Individual story values were `B1` (`spec +2.859 / raw +1.425 / peer -1.434`), `B2` (`+2.040 / -0.479 / -2.519`), `B3` (`+1.968 / +0.298 / -1.670`), and `B4` (`+1.654 / -0.245 / -1.899`). The gap between `mean_spec = +2.130` and `mean_raw = +0.250` (`spec = raw - peer_med`) shows that peer-median suppression (`-1.881`) is doing almost all of the `SpecLogit` work for `p92-46` on this bank rather than raw target-logit elevation.
- **No rescuing:** Even descriptively, although `L7H5|p92-46`'s `S_odd` selectivity margins passed (`+0.214` over controls, `+0.211` over `L3`; raw `S_odd = +0.172`, spec `S_odd = +0.259`), it failed the registered sign/magnitude requirements (`raw d_pos = -0.022` against `<= -0.25` and `spec d_neg = +0.164` against `>= +0.25`). Under the frozen drop/stop rule, all `L7H5` `CM.1a`, `CM.2`, and `CM.3` slices (`p92-46` and `p92-09`) remain strictly **`uninformative (cm1a_gate_failed)`**.

### 3.3 `CM.1b` (`L8H15` on `B1..B4`: `dropped_confirmatory_failed`)

All cell transport and L8 head-null gates passed (`cm1b_valid = true`), making `CM.1b` fully readable on the fresh unseen bank `B1..B4`:

| Head (Role) | Channel | `d_pos(SELF)` | `d_pos(NAME_BARE)` | `d_pos(ROLE_BARE)` | `C_SN` (`SELF - NAME`) | `C_SR` (`SELF - ROLE`) |
|---|---|---:|---:|---:|---:|---:|
| **`L8H15` (primary)** | **`spec`** | `+0.044` | **`+0.073`** | `-0.257` | **`-0.029`** | `+0.301` |
| `L8H15` (primary) | `raw` | `+0.114` | `+0.005` | `-0.062` | `+0.109` | `+0.176` |
| **`L8H2` (active comparator)** | **`spec`** | `+0.020` | `-0.198` | `-0.344` | `+0.217` | `+0.364` |
| **`L8H2` (active comparator)** | **`raw`** | `+0.312` | `-0.070` | `-0.201` | **`+0.382`** | **`+0.513`** |
| `L8H3` (rand control) | `spec` | `+0.387` | `+0.262` | `+0.014` | `+0.125` | **`+0.373`** |
| `L8H3` (rand control) | `raw` | `+0.248` | `+0.065` | `-0.119` | `+0.183` | `+0.366` |
| `L8H13` (rand control) | `spec` | `+0.543` | `+0.458` | `+0.317` | `+0.085` | `+0.226` |
| `L8H13` (rand control) | `raw` | `+0.079` | `+0.183` | `+0.249` | `-0.104` | `-0.170` |

- **Confirmatory failure:** On `B1..B4`, `L8H15` failed to replicate its earlier third-person suppression (`d_pos(NAME_BARE)` went positive `+0.073`, `C_SN = -0.029`), and its margins over the best random control (`L8H3`) were negative on both `C_SN` (`-0.154`) and `C_SR` (`-0.072`). Consequently, `CM.1b` and all downstream `L8H15` `CM.2`, `CM.3`, `CM.4a`, and `CM.4b` verdicts are strictly **`dropped_confirmatory_failed`**.

### 3.4 Active Comparator `L8H2` & Control `L8H3` (`CM.2`, `CM.3`, `CM.4`)

`L8H2` (pre-registered as an active comparator, not a primary confirmatory target) showed positive `C_SN` and `C_SR` contrasts on `B1..B4` in both `spec` and `raw`, with the following scored decompositions:

- **`CM.2` (`QKV` factorization on `L8H2`):**
  - `ROLE_BARE_supp` (`spec`): `D_tot = +0.344`, `D_V = +0.333`, `D_QK = -0.127`, `D_int = +0.138` (`f_V = 0.966`, `f_QK = -0.367`, `f_int = 0.402`) → `interaction_required_within_head`.
  - `C_SR` (`spec` / `raw`): `spec` `D_tot = +0.364` (`D_V = +0.278`, `D_QK = -0.083`, `D_int = +0.169`); `raw` `D_tot = +0.513` (`D_V = +0.339`, `D_QK = +0.014`, `D_int = +0.161`) → `interaction_required_within_head`.
  - `C_SN` (`raw`): `D_tot = +0.382` (`D_V = +0.207`, `D_QK = +0.022`, `D_int = +0.154`) → `interaction_required_within_head`.
  - Remaining `L8H2` sub-metrics (`NAME_BARE_supp` `spec`/`raw`, `ROLE_BARE_supp` `raw`, `C_SN` `spec`): `effect_too_small_or_wrong_sign`.
- **`CM.3` (position scope on `L8H2`):**
  - `ROLE_BARE_supp` (`spec`): `D_all = +0.344`, `D_rp = +0.120` (`f_rp = 0.349`), `D_up = +0.273` (`f_up = 0.793`), `I_pos = -0.049` → `additive_both_within_head`.
  - `C_SR` (`spec` / `raw`): `spec` `f_rp = 0.342, f_up = 0.683, f_pos_int = -0.025`; `raw` `D_all = +0.513, D_rp = +0.172 (f_rp = 0.335), D_up = +0.341 (f_up = 0.665), f_pos_int = 0.000` → `additive_both_within_head`.
  - `C_SN` (`raw`): `D_all = +0.382, D_rp = +0.129 (f_rp = 0.337), D_up = +0.253 (f_up = 0.661), f_pos_int = 0.002` → `additive_both_within_head`.
  - Remaining 4 of 8 `L8H2` sub-metrics (`NAME_BARE_supp` `spec`/`raw`, `ROLE_BARE_supp` `raw`, `C_SN` `spec`): `effect_too_small_or_wrong_sign`.
- **`CM.4a` (`4-tok` full differing span `[13, 20, 30, 31]`) & `CM.4b` (`2-tok` user span `[13, 20]`):**
  - On **both** `L8H2` and `L8H3`, in **both** `spec` and `raw`, every `CM.4a` and `CM.4b` verdict is **`small_effect_at_tested_coords`**:
    - `L8H2` `CM.4a`: `spec` `M_N2S = -0.001, M_S2N = +0.015, C_ctrl = 0.020` (context-only `C_SN = +0.217`); `raw` `M_N2S = +0.009, M_S2N = +0.025, C_ctrl = 0.009` (context-only `C_SN = +0.382`).
    - `L8H2` `CM.4b`: `spec` `M_N2S = +0.016, M_S2N = -0.001, C_ctrl = 0.006`; `raw` `M_N2S = +0.003, M_S2N = -0.005, C_ctrl = 0.009`.
    - `L8H2` `CM.4c` (assistant 2-tok diagnostic `[30, 31]`): `spec` `M_N2S = -0.014, M_S2N = +0.012`; `raw` `M_N2S = +0.007, M_S2N = +0.027`.
    - `L8H3` `CM.4a`: `spec` `M_N2S = -0.017, M_S2N = -0.005, C_ctrl = 0.007`; `raw` `M_N2S = +0.006, M_S2N = +0.026, C_ctrl = 0.007`.
    - `L8H15` (dropped; descriptive only): `CM.4a` `spec` `M_N2S = +0.031, M_S2N = +0.046, C_ctrl = 0.019`; `raw` `M_N2S = -0.019, M_S2N = +0.024, C_ctrl = 0.036`.
  - **Scoped interpretation:** What the data directly establishes is that swapping the `L8` pre-input residual at the 4 differing referent positions `[13, 20, 30, 31]` shifts `L8H2`'s readout by `< 0.03` logits, while the `17/17` synthetic actuation controls (`min_target_delta = 0.532`) confirm the swap hook itself operates as intended. At least two distinct explanations fit this observation and cannot be separated by the present design: (1) earlier layers (`L0–L7`) may have already broadcast referent identity out of `[13, 20, 30, 31]` into other token positions before Layer 8, or (2) the identity signal driving `L8H2`'s `C_SN` contrast may be carried at positions this design did not swap (such as the final read position or shared intervening tokens).

---

## 4. Strictly Exploratory Lead & Follow-Up Criteria (Non-Confirmatory Only)

Because `p92-46` failed the primary `CM.1a` raw transport gate, the `L7H5|p92-09` slice is formally **`uninformative (cm1a_gate_failed)`** and must not be read as a confirmatory finding. Recorded strictly as an exploratory lead:

- **`L7H5` on `p92-09` (`L6/N4145`, `ROLE_BARE`, `B1..B4`):**
  - Unit transport passed strongly (`mean_spec = +7.758`, `mean_raw = +5.568`, `mean_peer_med = -2.190`, `max_wdown_null_spec = +0.802`).
  - `L7H5` `spec` odd-antisymmetric readout: `d_pos = -2.144`, `d_neg = +0.676`, `S_odd = +1.410` (margins `+1.387` over `L7H4`/`L7H9`, `+1.419` over `L3` units `[-0.009]`). *(Raw diagnostic: `d_pos = -1.468`, `d_neg = -0.530`, `S_odd = +0.469` — still negative at both push signs, matching Part A's `raw_absent_or_neg` tag.)*
  - `CM.2` `QKV` decomposition (`spec`): `D_tot = +1.410`, **`D_QK = +1.427` (`f_QK = 1.012`)**, `D_V = -0.113` (`f_V = -0.080`), `D_int = +0.096` (`f_int = 0.068`), with `Q_only = +1.396` and `K_only = +1.351` (controls `L7H4` `QKV_all = +0.022`, `L7H9` `-0.152`).
  - `CM.3` position scope (`spec`): `D_all = +1.410`, `D_up = +0.941` (`f_up = 0.668`), `D_rp = +0.370` (`f_rp = 0.262`), `I_pos = +0.099` (`f_pos_int = 0.070`).
- **What a confirmatory follow-up would require:** Any prospective test of this `L7H5|p92-09` `QK_only` lead must pre-register **`p92-09` as the primary unit** on a **fresh unseen prompt bank** (zero overlap with `d13fc3dd`, `01bd41d9`, or `B1..B4`), keep the **same fixed `+0.50` raw transport floor** on unit transport, and explicitly declare its primary readout channel (`SpecLogit` vs `RawLogit`) up front given `p92-09`'s negative raw drop at both push signs.

---

## 5. Mandatory Scope Caveats (`CAUSAL-ATTN-DESIGN.md` v3 §1)

1. All `CM.2` and `CM.3` factorizations reflect within-head perturbations of the frozen post-`q_norm`/`k_norm`/`v_proj` and `o_proj` coordinates on `OLMo-2-0425-1B-Instruct@48d788ec` (`float32`, `eager`, `OLD_NORM`, `d = ±0.8`) and do not establish general head roles across other checkpoints, norms, or prompts.
2. `CM.4` tests only the frozen `L8` pre-input residual at the 4 differing referent positions `[13, 20, 30, 31]` (`CM.4a`) and `[13, 20]` (`CM.4b`) on `B1..B4`; `small_effect_at_tested_coords` does not rule out earlier-layer routing or referent signals carried at unswapped positions.
3. No Part B GPU follow-up, git push, or public registration is authorized by this report.
