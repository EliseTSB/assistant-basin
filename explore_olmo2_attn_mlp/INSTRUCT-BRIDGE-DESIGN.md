# OLMo 2 1B Instruct — 2x2 Bank x Physical-Norm Bridge Study Design (v3)

**Status:** Revised Draft v3 addressing Alden's `#93` narrow schema/contract HOLD on v2 (`8719aa33369de705db2c36bdd42b71fcfdb9f218ced91d9e89927cfc1c3e645a`). Schedule (`6,960` full / `242` smoke B19 forwards), anchor joins (`480` OLD head; `910` NEW head + `160` NEW unpatched), and analysis thresholds are unchanged from v2. Part A of the post-Stage-Sweep plan; Part B (`#4` Causal Attention Mechanism) will be frozen separately on a fresh confirmatory bank per `#85`.  
**Author:** Vesper (`neon`) | **Reviewers:** Alden (`dr-sol`), Claude (`claude`), Weaver (`weaver`)  
**Target Checkpoint (single frozen Instruct checkpoint):**
- `instruct`: `allenai/OLMo-2-0425-1B-Instruct@48d788eca847d4d7548f375ad03d3c9312f6139e` (`float32`, `attn_implementation="eager"`)

**Changelog v2 -> v3 (resolving all 3 narrow items of `#93`):**
1. **Per-source literal field bindings (§2.1):** Bound each source strictly to the fields it actually stores — full 5-tuple `(id, layer, neuron, lead, content)` across `instruct_bridge_manifest.json`, `d13fc3dd:run_signed_dose.py` (`SLOT_BY_ID`), and `01bd41d9:stage_sweep_manifest.json` (`meta.core_units`); `(map_key, layer, neuron, cal_resid_median)` on `NEW_RECEIPT["unit_vectors"]`; and supported unit/layer/token/median joins on `d13fc3dd:signed_dose_manifest.json` — with zero-load refusal on drift plus a required valid literal-source control test.
2. **Separated sham vs decomposition/finiteness coverage (§2.3, §BR.0, §4):** Applied measured raw/spec sham exclusively to head rows (`3,220/3,220` full; `112/112` smoke); kept sham explicitly absent/not applicable on unpatched rows (`520` full; `18` smoke, never a manufactured zero); applied decomposition and finiteness to both row types; required self-tests rejecting a head row with missing sham and accepting a valid unpatched-only `CAL_SELF` row.
3. **Unified anchor-failure disposition & `BR.3` `d = 0.8` baseline tag (§BR.0, §BR.3, §BR.4):** On `BR.0 == failed`, science endpoints (`BR.1`–`BR.4` verdicts) are hard-gated `uninformative (anchor_replay_failed)` while downstream actions are explicitly `part_b_norm_decision = decline_norm (anchor_replay_failed)` and `p92_09_part_b_track = not_selected (anchor_replay_failed)` (tested in CPU self-test); marked `BR.3` `cell_collapsed` at `d = 0.8` explicitly as `null` (`not_applicable`).

---

## 1. Scientific Motivation & Interpretive Caveats

Between the two verified OLMo 2 1B Instruct studies (`d13fc3dd` and `01bd41d9`), two things changed simultaneously:
1. **Prompt bank & syntax:** `OLD_BANK` (`d13fc3dd`: `SELF_HELD_NEW`, `ROLE_DEFINITE_NEW`, `NAMED_PERSON_NEW`, `THIRD_GENERIC_NEW`) vs `NEW_BANK` (`01bd41d9`: `SELF`, `ROLE_BARE`, `NAME_BARE`, `ALIAS_TO_ROLE`, `ALIAS_TO_NAME`, plus `CAL_SELF`/`CAL_THIRD`).
2. **Physical injection norm at nominal `d = ±0.8`:** `OLD_NORM` (`d13fc3dd` pinned chat medians) vs `NEW_NORM` (`01bd41d9` Instruct CAL all-token medians), which made nominal `±0.8` physically **`+26.0%` stronger at L3, `+14.7%` stronger at L5, and `+15.9%` stronger at L6** in `01bd41d9`.

Meanwhile, `L8H15`'s `ROLE - NAME` contrast moved from `-0.6027` (`raw -0.0102`, `peer_med +0.5925`) in `d13fc3dd` to `-0.0762` (`raw -0.0673`, `peer_med +0.0089`) in `01bd41d9`, and `01bd41d9`'s `CAL` curves exhibited raw final-logit transport collapse at `d = 1.2, 1.6` (`-2.2` to `-6.6` raw).

This study runs a **clean Instruct-only `2 x 2` factorial Bridge (`Bank` x `Physical Norm`)** plus a **direct within-bank 3-dose high-dose transport check (`d = 0.8, 1.2, 1.6`)** to separate physical dose/norm from prompt bank before choosing or declining the operating norm for Part B (`#4` Causal `Q/K/V` & Span-Localized Mechanism).

### Required Interpretive Caveats (Carried with Every Result)
1. **Bank ≠ words alone:** `OLD_BANK` and `NEW_BANK` differ in story templates, prompt geometry, token positions, and referent syntax. Neither a `bank_dominant` result nor an `interaction_present` result attributes the gap to lexical/noun identity alone; noun-specificity attribution requires genuinely matched referents and prompt geometry afterward.
2. **Within-bank high-dose comparisons only:** `OLD_BANK` `THIRD_GENERIC_NEW` (`block_a`) and `NEW_BANK` `CAL_THIRD` / `ROLE_BARE` are **never** equated as the same semantic cell; each bank's `{0.8, 1.2, 1.6}` transport and rebound curve is compared against itself across `OLD_NORM` vs `NEW_NORM`. Furthermore, `{0.8, 1.2, 1.6}` is a 3-dose high-dose check and **not** the original 6-dose `d_cal` calibration (`{0.2, 0.4, 0.6, 0.8, 1.2, 1.6}`).
3. **Same-coordinate scope & non-uniqueness:** Null at `L8H15` means null at coordinate `L8H15` under this protocol; `L8H2` is tracked explicitly as an active Layer-8 comparator alongside matched random controls (`L8H3`, `L8H13`), and `L7H5` is tracked alongside matched Layer-7 controls (`L7H4`, `L7H9`).
4. **No rescue of Stage-Sweep dispositions:** This Instruct-only bridge does not alter any registered verdict of `01bd41d9` (`SS.1`–`SS.7`) or `d13fc3dd` (`SD.1`–`SD.6`).

---

## 2. Frozen Units, Heads, Norms, Source Bindings & Prompt Schedule

### 2.1 Core Units, Exact Write Coordinates & Dual-Source Pre-Load Binding
- **5 Core Units (verified verbatim against both `d13fc3dd` and `01bd41d9`):**

| Unit ID | Group | Layer | Neuron Index | Lead | Content |
|---|---|---:|---:|---|---|
| `p92-02` | Early-write L3 | `3` | `2628` | `"the"` | `"town"` |
| `p92-05` | Early-write L3 | `3` | `6986` | `"a"` | `"fox"` |
| `p92-13` | Early-write L3 | `3` | `976` | `"a"` | `"skeleton"` |
| `p92-46` | Late-write L5 | `5` | `4673` | `"a"` | `"regular guy"` |
| `p92-09` | Late-write L6 | `6` | `4145` | `"a"` | `"dog"` |

- **Exact Dual-Source Binding & Zero-Load Refusal Rule:**
  - `OLD` source (`d13fc3dd2b76309d28b001ecf33f0fc9c962f156`):
    - `explore_olmo2_attn_mlp/signed_dose_manifest.json` (`SHA-256: 716cf2ab3f08461656c14768a7b5c3e6974261f5a86456c56e00c9a2e0800533`): binds unit IDs, layers, `pinned_medians_chat`, `null_seed_base = 202611200`, and `q12_prompts` (tokenized bindings, `read_pos`, target/peer tokens). *(Does not store neuron indices.)*
    - `explore_olmo2_attn_mlp/run_signed_dose.py` (`SHA-256: 227d8bc79a6a9c8563a1f8919bcb1c3aadb6338dce2a107ec25335582f9bc2e6`): binds `OLD` neuron indices via `ALL_SLOT_UNITS` / `SLOT_BY_ID` and the `OLD_BANK` per-unit orthogonal null generator.
  - `NEW` source (`01bd41d963936d39f68b032eda845ad73b13d6a2`):
    - `explore_olmo2_attn_mlp/stage_sweep_manifest.json` (`SHA-256: 1a187ada52cd7ca6ebe122c7f93eafdf6ffe3024c739de6ade833707d05e403b`): binds unit IDs, layers, AND neuron indices via `meta.core_units`, plus `EVAL` & `CAL` prompt tokenized bindings, `read_pos`, target/peer tokens.
    - `explore_olmo2_attn_mlp/run_stage_sweep.py` (`SHA-256: 405d67721015d66ec06dc2c5d1de9948e7547059ac120c58c9da61264e003612`): binds the `NEW_BANK` per-tuple orthogonal null generator.
  - **Pre-load per-source literal cross-check (each source bound strictly to fields it actually stores):** `instruct_bridge_manifest.json` embeds the canonical `core_units` table AND all 4 source SHA-256s above plus both reference receipt SHA-256s (§BR.0). At manifest build time, on local pre-launch verification, and inside `score_instruct_bridge.py`:
    1. Full 5-tuple `(id, layer, neuron, lead, content)` exact equality is asserted across `instruct_bridge_manifest.json` (`core_units`), `d13fc3dd:run_signed_dose.py` (`ALL_SLOT_UNITS` / `SLOT_BY_ID`), and `01bd41d9:stage_sweep_manifest.json` (`meta.core_units`).
    2. `NEW_RECEIPT["unit_vectors"]` (map keyed by unit ID storing `layer`, `neuron`, `w_down_norm`, `cal_resid_median`, `hat_w`; does NOT store `lead` or `content`) is bound strictly on `(map_key == id, layer, neuron)` and `cal_resid_median == NEW_NORM[layer]`.
    3. `d13fc3dd:signed_dose_manifest.json` (does NOT store neuron indices) is bound strictly on its supported unit/layer/token/median joins (`eligible_pairs` unit IDs & layers, `meta.pinned_medians_chat == OLD_NORM`, `meta.null_seed_base == 202611200`, and `q12_prompts` tokenized prompt/target/peer joins).
  - Inside `run_instruct_bridge.py`, all 4 package file SHA-256 pins and the exact `core_units` table are validated **BEFORE any tokenizer or model weight load**, hard-refusing with zero loads on any drift. CPU self-tests MUST include BOTH mismatch-refusal witnesses AND a **valid literal-source control** proving the exact unmodified sources pass cleanly.

- **7 Attention Heads (all strictly downstream of L6):**
  - Layer 7: `L7H5` (`primary_L7`), `L7H4` (`rand_L7`), `L7H9` (`rand_L7`)
  - Layer 8: `L8H15` (`primary_L8`), `L8H2` (`active_comparator_L8`), `L8H3` (`rand_L8`), `L8H13` (`rand_L8`)

### 2.2 Full-Precision Pinned Physical Norms (Scalar Multipliers `m_{c, L}`)
- **`OLD_NORM` (exact `d13fc3dd` `pinned_medians_chat` from `signed_dose_manifest.json` & `results_signed_dose_full_d13fc3dd.json`):**
  - L3: `7.3919267654418945`
  - L5: `10.05025863647461`
  - L6: `10.501473426818848`
- **`NEW_NORM` (exact `01bd41d9` Instruct `cal_resid_medians` from `results_stage_sweep_instruct_01bd41d9.json`):**
  - L3: `9.312541007995605`
  - L5: `11.525568008422852`
  - L6: `12.170919418334961`
- At nominal dose `d` and norm condition `c in {OLD_NORM, NEW_NORM}`, inject `+d * m_{c, L_u} * hat_w_u` at decoder-layer `L_u` output across all token positions. Persist `m_{c, L_u}`, `injected_norm = |d| * m_{c, L_u}`, unpushed `read_pos_resid_norm`, and `norm_ratio` for every `(bank, norm_cond, unit, prompt_id, dose_mag)`.

### 2.3 Exact Fixed Write Direction & Fixed Orthogonal Null Draws Across Norm Conditions
- For every `(bank, unit, prompt_id, dose_mag)` tuple, the unit write vector `hat_w_u` AND the exact `4` `wdown_orth` + `4` `iso_resid` unit-norm orthogonal null directions (`B = 19`: `0`, `+d`, `-d`, `±d * 4 wdown_orth`, `±d * 4 iso_resid`) are **held strictly identical across `OLD_NORM` and `NEW_NORM`** (`max_abs_null_vec_diff == 0.0`); only the scalar physical magnitude `|d| * m_{c, L_u}` changes.
- **Anchor-exact seed conventions:**
  - On `OLD_BANK`, orthogonal null vectors use the exact `d13fc3dd` generator & per-unit seed (`seed_u = 202611200 + 10000 * CORE_UNITS.index(uid)`), guaranteed bit-for-bit identical to `d13fc3dd` under `OLD_NORM` and held fixed under `NEW_NORM`.
  - On `NEW_BANK`, orthogonal null vectors use the exact `01bd41d9` generator & `unit_seed("instruct", uid, pid, dose_mag)`, guaranteed bit-for-bit identical to `01bd41d9` under `NEW_NORM` and held fixed under `OLD_NORM`.
- **100% Head-Row Sham Verification (Head Rows Only):** Every single `(bank, norm_cond, unit, prompt_id, dose_mag, head_name)` head row (`3,220/3,220` full; `112/112` smoke) executes an explicit measured `sham_z` pass (patching captured pushed `z_h` back into pushed `z_h`) and requires `max_abs_sham <= 1e-6` on BOTH `SpecLogit` and `RawLogit`. Unpatched rows (`520` full; `18` smoke, including unpatched-only `CAL_SELF`) have no head patch and therefore **no sham pass** — sham fields MUST be absent/not applicable on unpatched rows (never a manufactured `0.0`).

### 2.4 Exact Prompt Sets & Forward Budget (`520` Unpatched Rows, `3,220` Head Rows, `6,960` B19 Forwards)
- **`OLD_BANK` (`20` `(prompt, dose)` pairs x `2` norms x `5` units = `200` unpatched rows + `1,400` head rows -> `3,000` B19 forwards):**
  - Verbatim tokenized bindings copied from `d13fc3dd` `signed_dose_manifest.json` (`716cf2ab...`).
  - Core `2 x 2` cells at `d = 0.8` (`12` prompts, head-patched on all `7` heads): `SELF_HELD_NEW` (`self_F1..F3`), `ROLE_DEFINITE_NEW` (`role_F1..F3`), `NAMED_PERSON_NEW` (`npers_F1..F3`), `THIRD_GENERIC_NEW` (`third_F1..F3`).
  - High-dose `block_a` (`F1, F2`) at `d = 1.2, 1.6` (`+8` prompt-dose pairs, head-patched on all `7` heads): `self_F1`, `self_F2`, `third_F1`, `third_F2` at `d = 1.2` and `d = 1.6`. *(Paired with `self_F1/F2` and `third_F1/F2` at `d = 0.8` from the core 12 prompts, giving complete `{0.8, 1.2, 1.6}` curves under both norms.)*
  - Subtotal: `200 * (1 + 2 * 7) = 3,000` B19 forwards.
- **`NEW_BANK` (`32` `(prompt, dose)` pairs x `2` norms x `5` units = `320` unpatched rows + `1,820` head rows -> `3,960` B19 forwards):**
  - Verbatim tokenized bindings copied from `01bd41d9` `stage_sweep_manifest.json` (`1a187ada...`).
  - Core EVAL cells at `d = 0.8` (`20` prompts, head-patched on all `7` heads): `SELF` (`eval_self_E1..E4`), `ROLE_BARE` (`eval_role_bare_E1..E4`), `NAME_BARE` (`eval_name_bare_E1..E4`), `ALIAS_TO_ROLE` (`eval_alias_role_E1..E4`), `ALIAS_TO_NAME` (`eval_alias_name_E1..E4`) -> `200` unpatched + `1,400` head rows (`3,000` B19 forwards).
  - `CAL_THIRD` at `d = 0.8, 1.2, 1.6` (`6` prompt-dose pairs: `cal_third_C1`, `cal_third_C2` at `0.8, 1.2, 1.6`, head-patched on all `7` heads) -> `60` unpatched + `420` head rows (`900` B19 forwards).
  - `CAL_SELF` at `d = 0.8, 1.2, 1.6` (`6` prompt-dose pairs: `cal_self_C1`, `cal_self_C2` at `0.8, 1.2, 1.6`, **unpatched-transport-only, `0` head rows**, matching `01bd41d9` identically where `CAL_SELF` was never head-patched) -> `60` unpatched + `0` head rows (`60` B19 forwards).
  - Subtotal: `260 * 15 + 60 * 1 = 3,960` B19 forwards.
- **Grand Total (Full Bridge):** **`520` unpatched rows**, **`3,220` head rows**, **`6,960` B19 forwards**. Wall-clock startup/download/forward/transfer overhead will be measured directly in the mandatory smoke gate (§4) rather than assumed before launch; budget and spend remain Elise's call.

### 2.5 Decomposed Readouts (Every Row)
For every `(bank, norm_cond, unit, prompt_id, dose_mag)` unpatched row and every head row, record exact decomposed metrics satisfying `SpecLogit == RawLogit - PeerMedian` (verified to `<= 1e-6`):
1. `RawLogit` (`raw`): target own-token logit `logits[read_pos, t_own]`.
2. `PeerMedian` (`peer_med`): median over the unit's 19 peer tokens `logits[read_pos, peers(u)]`.
3. `SpecLogit` (`spec`): `RawLogit - PeerMedian`.
4. Per-token 19-peer individual `d_pos`/`d_neg` shifts on `L8H15` and `L8H2` across all `d = 0.8` core cells (`ROLE`/`NAME`/`ALIAS`/`THIRD`/`SELF`) so `frac_peer_tokens_pos` is exact everywhere.
5. Unpatched final-logit transport (`n_write_spec`, `n_write_raw`, `n_write_peer_med`, `n_write_neg_spec`, `n_write_neg_raw`, and `4` `wdown_pos_draws_spec`/`raw` + `4` `iso_pos_draws_spec`/`raw`) on all `520` tuples (`OLD_BANK` and `NEW_BANK`, `{0.8, 1.2, 1.6}`).

---

## 3. Pre-Registered Gates & Total Machine-Readable Decision Maps (`BR.0`–`BR.4`)

### `BR.0` — Anchor-Replay & Integrity Falsifier Gate (Strict Precedence)
Before any bridge contrast (`BR.1`–`BR.4`) may be interpreted:
1. **Package & coordinate integrity:** All 4 external SHA-256 pins (`INSTRUCT-BRIDGE-DESIGN.md`, `instruct_bridge_manifest.json`, `run_instruct_bridge.py`, `score_instruct_bridge.py`) and dual-source `core_units` bindings (§2.1) verified pre-load and on offline scoring; exact canonical row-key and metadata binding verified before grouping (`#69`/`#70` discipline); full-receipt check (`artifact_type == "instruct_bridge_full_receipt"`, `complete == True`, exact row counts `520` unpatched and `3,220` head rows).
2. **Sham (head rows only), decomposition & finiteness (both row types):**
   - **Measured sham (head rows only):** `max_abs_sham_spec <= 1e-6` and `max_abs_sham_raw <= 1e-6` present and finite on **all `3,220/3,220` head rows** (hard-refuse any head row with missing/`None` sham). Unpatched rows (`520/520`, including unpatched-only `CAL_SELF`) MUST omit sham fields (absent / not applicable, never a manufactured zero).
   - **Decomposition & finiteness (both row types):** `max_abs(spec - (raw - peer_med)) <= 1e-6` and zero `NaN`/`Inf` across all `520/520` unpatched rows AND all `3,220/3,220` head rows, plus `max_abs_null_vec_diff == 0.0` across `OLD_NORM` vs `NEW_NORM` for every `(bank, unit, prompt_id, dose_mag)`.
3. **Anchor Join 1 — `OLD_BANK x OLD_NORM` vs `OLD_RECEIPT` (head rows only):**
   - **Reference receipt:** `explore_olmo2_attn_mlp/results_signed_dose_full_d13fc3dd.json`
   - **Exact full SHA-256 (verified before join):** `bf8fdf20dfe132c15b9acddea59eb82494dea9fc39f9d218fe38ff7571f3026b` (`14,989,108` bytes).
   - **Reference table:** `OLD_RECEIPT["q12_rows"]`.
   - **Exact join key (1-to-1 uniqueness asserted on both sides):** `(unit, prompt_id, cell, f"{dose_mag:.1f}", head_name)`.
   - **Exact anchor denominator:** **`480` head rows** = (`12` core `d = 0.8` prompts + `4` `block_a` `d = 1.6` prompts `self_F1, self_F2, third_F1, third_F2`) x `5` units x `6` overlapping heads `{L7H4, L7H5, L8H2, L8H3, L8H13, L8H15}`. Explicitly includes all `60` `third_F1/F2` `d = 1.6` overlapping head rows (`10` on `L8H15`).
   - **Tolerance:** `max_abs_diff <= 1e-3` across all `480` rows on `spec.d_pos`, `spec.d_neg`, `raw.d_pos`, and `raw.d_neg`.
   - **Explicit non-anchor partition on `OLD_BANK x OLD_NORM` (kept strictly outside the `480`-row anchor claim):**
     - All `100` `OLD_BANK x OLD_NORM` unpatched-transport rows (`d = 0.8, 1.2, 1.6`) — `OLD_RECEIPT` stores zero unpatched-transport rows; `OLD` transport at `+0.8/+1.2/+1.6` is a **new measurement**, NOT an anchor quantity.
     - All `100` `L7H9` head rows (`L7H9` was not run in `d13fc3dd`) and all `120` `d = 1.2` head rows on the 6 overlapping heads (`d = 1.2` was not run in `d13fc3dd`); `480 + 220 = 700` total head rows on `OLD_BANK x OLD_NORM`.
4. **Anchor Join 2 — `NEW_BANK x NEW_NORM` vs `NEW_RECEIPT` (two separate table joins covering 100% of `NEW_BANK x NEW_NORM`):**
   - **Reference receipt:** `explore_olmo2_attn_mlp/results_stage_sweep_instruct_01bd41d9.json`
   - **Exact full SHA-256 (verified before join):** `aa8300a11a438c5fd54ffca22a4a6d6ccc762c17c0ad5e8845047af5759c1980` (`10,878,358` bytes).
   - **Join 2a — Head rows (`910` rows total):**
     - **Exact join key (1-to-1 uniqueness asserted on both sides):** `(ref_table, unit, prompt_id, cell, f"{dose_mag:.1f}", head_name)` where `ref_table == "eval_arm_a_head_rows"` for `EVAL` (`20` prompts x `5` units x `7` heads = `700` rows at `d = 0.8`) and `ref_table == "cal_head_rows"` for `CAL_THIRD` (`2` prompts x `3` doses `{0.8, 1.2, 1.6}` x `5` units x `7` heads = `210` rows). `CAL_SELF` has `0` head rows in both bridge and `NEW_RECEIPT`.
     - **Tolerance:** `max_abs_diff <= 1e-3` across all `910` rows on `spec.d_pos`, `spec.d_neg`, `raw.d_pos`, and `raw.d_neg`.
   - **Join 2b — Separate unpatched transport rows (`160` rows total):**
     - **Exact join key (1-to-1 uniqueness asserted on both sides):** `(ref_table, unit, prompt_id, cell, f"{dose_mag:.1f}")` where `ref_table == "eval_arm_a_unpatched_rows"` for `EVAL` (`20` prompts x `5` units = `100` rows at `d = 0.8`) and `ref_table == "cal_unpatched_rows"` for `CAL` (`4` prompts `{cal_self_C1, cal_self_C2, cal_third_C1, cal_third_C2}` x `3` doses `{0.8, 1.2, 1.6}` x `5` units = `60` rows).
     - **Tolerance:** `max_abs_diff <= 1e-3` across all `160` rows on `transport.n_write_spec`, `transport.n_write_raw`, `transport.n_write_neg_spec`, and `transport.n_write_neg_raw`.
5. **Failure rule (unified across science endpoints and downstream operating actions):** If **any** clause `1–4` fails, `BR.0 = failed`:
   - **Science endpoints:** **`BR.1`, `BR.2`, `BR.3` (`gate_state`, `collapse_state`), and `BR.4` (`br4_verdict`, `br4a_verdict`, `br4b_verdict`, `p92_09_raw_tag`)** are ALL hard-gated **`uninformative (anchor_replay_failed)`**.
   - **Downstream operating actions:** **`part_b_norm_decision`** is explicitly **`decline_norm (anchor_replay_failed)`** and **`p92_09_part_b_track`** is explicitly **`not_selected (anchor_replay_failed)`** (no norm and no `p92-09` track may ever be selected after an anchor failure).

### `BR.1` — Cell-Wise Transport & Head-Null Validity Gate (at `d = 0.8`)
For each of the 4 cells `(bank, norm_cond) in {OLD_BANK, NEW_BANK} x {OLD_NORM, NEW_NORM}` at `d = 0.8`, define the 3 core cells `C_core(OLD_BANK) = {SELF_HELD_NEW, ROLE_DEFINITE_NEW, NAMED_PERSON_NEW}` (`15` rows/cell = `3` prompts x `5` units) and `C_core(NEW_BANK) = {SELF, ROLE_BARE, NAME_BARE}` (`20` rows/cell = `4` prompts x `5` units):
- **Transport gate (`transport_valid`):** Each cell `c in C_core(bank)` individually satisfies:
  - `mean(transport.n_write_spec) >= +0.50`, AND
  - `mean(transport.n_write_raw) >= +0.50`, AND
  - `mean(transport.n_write_spec) > max_{k in 0..3} mean(transport.wdown_pos_draws_spec[k])`.
- **Head-null gate (`head_null_valid(h)`):** For a given head `h`, each cell `c in C_core(bank)` individually satisfies:
  - `|mean(spec.wdown_orth.even_med)| <= 0.15` AND `|mean(spec.wdown_orth.odd_med)| <= 0.15`.
- A `(bank, norm_cond)` cell is `valid` for head `h` iff `BR.0 == passed` AND `transport_valid` AND `head_null_valid(h)`; otherwise `uninformative (<reason>)`.

### `BR.2` — `2 x 2` Bank x Physical-Norm Bridge on `L8H15` `G = D_pos(ROLE) - D_pos(NAME)` (at `d = 0.8`)
Defined on `L8H15` `SpecLogit` (`mean` over `n_prompts * 5` head rows per cell: `ROLE_DEFINITE_NEW - NAMED_PERSON_NEW` on `OLD_BANK`, `ROLE_BARE - NAME_BARE` on `NEW_BANK`), with exact parallel decomposition emitted for `RawLogit`, `PeerMedian`, `C_SR = SELF - ROLE`, `G_alias` on `NEW_BANK`, and `L8H2`/`L8H3`/`L8H13`:
- `G_OO = G(OLD_BANK, OLD_NORM)`, `G_ON = G(OLD_BANK, NEW_NORM)`
- `G_NO = G(NEW_BANK, OLD_NORM)`, `G_NN = G(NEW_BANK, NEW_NORM)`
- `d_norm_OLD = G_ON - G_OO`, `d_norm_NEW = G_NN - G_NO`
- `d_bank_OLD = G_NO - G_OO`, `d_bank_NEW = G_NN - G_ON`
- `DiD = d_norm_NEW - d_norm_OLD = d_bank_NEW - d_bank_OLD`
- **Exhaustive, mutually exclusive 7-branch decision partition (evaluated strictly in order):**
  1. If `BR.0 != passed`: `uninformative (anchor_replay_failed)`
  2. Else if any of the 4 `(bank, norm_cond)` cells fails `BR.1` for `L8H15`: `uninformative (cell_gate_failed:<cells>)` *(all 4 cell values & contrasts still emitted descriptively)*
  3. Else if `(d_norm_OLD >= +0.25) and (d_norm_NEW >= +0.25) and (G_NO <= -0.30) and (abs(d_bank_OLD) <= 0.20) and (abs(d_bank_NEW) <= 0.20)`: **`norm_dominant`**
  4. Else if `(d_bank_OLD >= +0.30) and (d_bank_NEW >= +0.30) and (abs(d_norm_OLD) <= 0.15) and (abs(d_norm_NEW) <= 0.15)`: **`bank_dominant`** *(caveat §1.1: bank != words alone)*
  5. Else if `abs(DiD) >= 0.25`: **`interaction_present`** *(caveat §1.1: interaction != words alone)*
  6. Else if `(abs(d_norm_OLD) >= 0.20 or abs(d_norm_NEW) >= 0.20) and (abs(d_bank_OLD) >= 0.20 or abs(d_bank_NEW) >= 0.20)`: **`additive_mixed`**
  7. Else: **`neither_or_inconclusive`**

### `BR.3` — Within-Bank 3-Dose High-Dose Transport & Collapse Bridge (`d in {0.8, 1.2, 1.6}`)
Evaluated **strictly within each bank separately** on its own high-dose cell pair (`P(OLD_BANK) = {SELF_HELD_NEW, THIRD_GENERIC_NEW}` on `F1, F2`; `P(NEW_BANK) = {CAL_SELF, CAL_THIRD}` on `C1, C2`; `n = 10` unpatched rows per `(bank, norm_cond, cell, d)`). Never equates `OLD THIRD` with `NEW ROLE` or `CAL_THIRD`, and explicitly represents a 3-dose high-dose check rather than the original 6-dose `d_cal`.
- **Per-cell predicates at `(bank, norm_cond, cell, d)`:**
  - `cell_transport_pass(bank, norm_cond, cell, d)` (defined for all `d in {0.8, 1.2, 1.6}`) := `(mean(n_write_spec) >= +0.50) and (mean(n_write_raw) >= +0.50) and (mean(n_write_spec) > max_{k in 0..3} mean(wdown_pos_draws_spec[k]))`.
  - `cell_collapsed(bank, norm_cond, cell, d)`:
    - At baseline `d = 0.8`: explicitly **`null` (`not_applicable`)** because collapse is defined relative to `d = 0.8`.
    - At high doses `d in {1.2, 1.6}`: `(mean(n_write_raw)(d) <= 0.0) or ((mean(n_write_raw)(0.8) >= +0.50) and (mean(n_write_raw)(d) <= 0.25 * mean(n_write_raw)(0.8)))`. *(Explicitly separates declared transport collapse — raw reversal/zeroing or >=75% raw drop from same-bank same-norm `d = 0.8` — from a narrow gate miss such as `raw = +0.49`.)*
- **Bank-pair predicates at `(bank, norm_cond, d)`:**
  - `pair_pass(bank, norm_cond, d)` := `all(cell_transport_pass(bank, norm_cond, c, d) for c in P(bank))`.
  - `pair_collapsed(bank, norm_cond, d)` := `any(cell_collapsed(bank, norm_cond, c, d) for c in P(bank))`.
- **Per-norm 4-state disjoint cell-pair status at `(bank, norm_cond, d)` for `d in {1.2, 1.6}`:**
  - `pass_intact`: `pair_pass and (not pair_collapsed)`
  - `pass_with_reduction`: `pair_pass and pair_collapsed`
  - `miss_no_collapse`: `(not pair_pass) and (not pair_collapsed)`
  - `miss_and_collapsed`: `(not pair_pass) and pair_collapsed`
- **Cross-norm 4-state disjoint comparison verdicts for each `bank in {OLD_BANK, NEW_BANK}` and each `d in {1.2, 1.6}`:**
  1. **`gate_state` (disjoint & total over `(P_old, P_new) = (pair_pass(OLD_NORM), pair_pass(NEW_NORM))`):**
     - If `BR.0 != passed`: `uninformative (anchor_replay_failed)`
     - Else:
       - `both_pass`: `P_old and P_new`
       - `old_pass_new_miss`: `P_old and (not P_new)`
       - `old_miss_new_pass`: `(not P_old) and P_new`
       - `both_miss`: `(not P_old) and (not P_new)`
  2. **`collapse_state` (disjoint & total over `(C_old, C_new) = (pair_collapsed(OLD_NORM), pair_collapsed(NEW_NORM))`):**
     - If `BR.0 != passed`: `uninformative (anchor_replay_failed)`
     - Else:
       - `neither_collapsed`: `(not C_old) and (not C_new)`
       - `new_only_collapsed`: `(not C_old) and C_new`
       - `old_only_collapsed`: `C_old and (not C_new)`
       - `both_collapsed`: `C_old and C_new`
- **Mandatory cell-wise table emission:** Always emit exact cell-wise means (`mean_spec`, `mean_raw`, `mean_peer_med`, `max_wdown_null_spec`, `cell_transport_pass`, `cell_collapsed`) for every `(bank, norm_cond, cell, d in {0.8, 1.2, 1.6})`, plus `L8H15` `D_pos(spec, raw, peer_med)` rebound deltas `D_pos(1.6) - D_pos(0.8)` and `D_pos(1.2) - D_pos(0.8)` on `THIRD_GENERIC_NEW` (`OLD_BANK`) and `CAL_THIRD` (`NEW_BANK`). `OLD_BANK` `d = 1.6` unpatched transport under `OLD_NORM` is explicitly tagged `new_measurement_not_anchor`.

### `BR.4` — Machine-Readable `L7H5` Unit/Readout-Separated Check & Total Part B Operating-Norm Rule
Evaluated at `d = 0.8` with **zero post-data choice among banks or heads**:
- **Fixed Primary Third-Person Cell per Bank (pre-registered):**
  - `c_prim(NEW_BANK) = ROLE_BARE` (`4` prompts `eval_role_bare_E1..E4`, matching `01bd41d9` `SS.6`).
  - `c_prim(OLD_BANK) = ROLE_DEFINITE_NEW` (`3` prompts `role_F1..F3`).
  - *(All individual cells and multi-cell third-person pools — `ROLE_BARE+NAME_BARE+ALIAS_TO_ROLE+ALIAS_TO_NAME` on `NEW_BANK`, `ROLE_DEFINITE_NEW+NAMED_PERSON_NEW+THIRD_GENERIC_NEW` on `OLD_BANK` — are emitted descriptively alongside, never used to override the primary verdict.)*
- **Exact Per-Cell Validity Rule (`br4_valid(bank, norm_cond)`):**
  `True` iff ALL of:
  1. `BR.0 == passed`,
  2. On `c_prim(bank)`, the 5-unit unpatched transport gate passes (`mean(n_write_spec) >= +0.50`, `mean(n_write_raw) >= +0.50`, `mean(n_write_spec) > max_{k in 0..3} mean(wdown_pos_draws_spec[k])`),
  3. On `c_prim(bank)`, BOTH late-write units individually have positive unpatched transport (`p92-46` and `p92-09` each satisfy prompt-mean `n_write_spec >= +0.50` AND `n_write_raw >= +0.50`),
  4. On `c_prim(bank)`, the 5-unit head-null gate passes for ALL THREE L7 heads `h in {L7H5, L7H4, L7H9}` (`|mean(spec.wdown_orth.even_med)| <= 0.15` and `|mean(spec.wdown_orth.odd_med)| <= 0.15`).
- **Exact Signed Statistics (computed per `(head, unit, cell, readout)` by averaging over the cell's prompts):**
  - `D_pos(h, u, c, r) = mean_{p in c} r.d_pos(h, u, p)`
  - `D_neg(h, u, c, r) = mean_{p in c} r.d_neg(h, u, p)`
  - Signed odd-suppression rebound score: `S_odd(h, u, c, r) = 0.5 * (D_neg(h, u, c, r) - D_pos(h, u, c, r))` *(positive when `+0.8` suppresses and `-0.8` boosts; equals `-r.odd`)*.
  - `L3_S_odd(h, c, r) = mean_{u in {p92-02, p92-05, p92-13}} S_odd(h, u, c, r)`.
  - `ctrl_max_S_odd(u, c, r) = max(S_odd(L7H4, u, c, r), S_odd(L7H9, u, c, r))`.
- **Sub-criterion `BR.4a` — `p92-46` (L5) True Raw-Logit Sign Flip (`br4a_holds(bank, norm_cond)`):**
  `True` iff `br4_valid(bank, norm_cond)` AND ALL 4 inequalities hold on `c = c_prim(bank)`:
  1. Raw bidirectional sign flip: `D_pos(L7H5, p92-46, c, raw) <= -0.25` AND `D_neg(L7H5, p92-46, c, raw) >= +0.25`.
  2. Spec bidirectional consistency: `D_pos(L7H5, p92-46, c, spec) <= -0.25` AND `D_neg(L7H5, p92-46, c, spec) >= +0.25`.
  3. Beats BOTH matched L7 controls (`L7H4`, `L7H9`) by `>= +0.15` on BOTH raw and spec `S_odd`:
     - `S_odd(L7H5, p92-46, c, raw) - ctrl_max_S_odd(p92-46, c, raw) >= +0.15`
     - `S_odd(L7H5, p92-46, c, spec) - ctrl_max_S_odd(p92-46, c, spec) >= +0.15`
  4. Beats `L3` early-write mean on `L7H5` by `>= +0.20` on raw `S_odd`:
     - `S_odd(L7H5, p92-46, c, raw) - L3_S_odd(L7H5, c, raw) >= +0.20`.
- **Sub-criterion `BR.4b` — `p92-09` (L6) SpecLogit Sign Flip (`br4b_holds(bank, norm_cond)`) + Orthogonal Raw Tag:**
  `br4b_holds(bank, norm_cond)` is `True` iff `br4_valid(bank, norm_cond)` AND ALL 3 inequalities hold on `c = c_prim(bank)`:
  1. Spec bidirectional sign flip: `D_pos(L7H5, p92-09, c, spec) <= -0.50` AND `D_neg(L7H5, p92-09, c, spec) >= +0.25`.
  2. Beats BOTH matched L7 controls (`L7H4`, `L7H9`) by `>= +0.25` on spec `S_odd`:
     - `S_odd(L7H5, p92-09, c, spec) - ctrl_max_S_odd(p92-09, c, spec) >= +0.25`.
  3. Beats `L3` early-write mean on `L7H5` by `>= +0.25` on spec `S_odd`:
     - `S_odd(L7H5, p92-09, c, spec) - L3_S_odd(L7H5, c, spec) >= +0.25`.
  - **Orthogonal 3-state `p92_09_raw_tag` (disjoint & total):**
    - `raw_present`: `D_pos(L7H5, p92-09, c, raw) <= -0.25` AND `D_neg(L7H5, p92-09, c, raw) >= +0.25`
    - `raw_absent_or_neg`: `D_neg(L7H5, p92-09, c, raw) <= +0.05`
    - `raw_intermediate`: otherwise.
- **Total Deterministic Part B (`#4`) Operating-Norm Decision (`part_b_norm_decision`):**
  Evaluated strictly on the pre-registered `NEW_BANK` benchmark (the bank whose clean bare-role/bare-name/self prompt geometry aligns with Part B's fresh confirmatory bank; `OLD_BANK` `BR.4a`/`BR.4b` are reported in full as cross-bank replication, never used to shop across banks or heads):
  - For each `norm_cond in {OLD_NORM, NEW_NORM}`, define:
    `Q(norm_cond) := (BR.0 == passed) and (BR.1(NEW_BANK, norm_cond, L8H15) == valid) and br4a_holds(NEW_BANK, norm_cond)`
    *(Note: `BR.1(NEW_BANK, norm_cond, L8H15) == valid` guarantees `NEW_BANK` core transport and `L8H15` head-null validity on `SELF`/`ROLE_BARE`/`NAME_BARE`; `br4a_holds` guarantees `br4_valid` including `L7H5`/`L7H4`/`L7H9` nulls and `p92-46` raw sign-flip beating both L7 controls and L3!)*
  - **Exhaustive, mutually exclusive 5-state machine-readable output (`part_b_norm_decision`) & coupled `p92_09_part_b_track`:**
    1. If `BR.0 != passed`:
       - Science endpoints (`br4_verdict`, `br4a_verdict`, `br4b_verdict`, `p92_09_raw_tag`): **`uninformative (anchor_replay_failed)`**
       - `part_b_norm_decision`: **`decline_norm (anchor_replay_failed)`**
       - `p92_09_part_b_track`: **`not_selected (anchor_replay_failed)`**
    2. Else if `Q(OLD_NORM) and Q(NEW_NORM)`:
       - `part_b_norm_decision`: **`pin_OLD_NORM (both_qualify_lower_perturbation_default)`**
       - `p92_09_part_b_track`: `include_spec_only_track` if `br4b_holds(NEW_BANK, OLD_NORM)` else `diagnostic_only`
    3. Else if `Q(OLD_NORM) and (not Q(NEW_NORM))`:
       - `part_b_norm_decision`: **`pin_OLD_NORM (old_only_qualifies)`**
       - `p92_09_part_b_track`: `include_spec_only_track` if `br4b_holds(NEW_BANK, OLD_NORM)` else `diagnostic_only`
    4. Else if `(not Q(OLD_NORM)) and Q(NEW_NORM)`:
       - `part_b_norm_decision`: **`pin_NEW_NORM (new_only_qualifies)`**
       - `p92_09_part_b_track`: `include_spec_only_track` if `br4b_holds(NEW_BANK, NEW_NORM)` else `diagnostic_only`
    5. Else (`(not Q(OLD_NORM)) and (not Q(NEW_NORM))`):
       - `part_b_norm_decision`: **`decline_norm (neither_norm_qualifies_hold_part_b)`**
       - `p92_09_part_b_track`: **`not_selected (no_norm_pinned)`**

---

## 4. Packaging, Mandatory Measured Smoke Gate & Gate Sequence

No code build, weight load, smoke launch, full launch, or Colab spend occurs from a design draft alone. Execution follows the strict gate chain:

1. **Design PASS** on this exact `INSTRUCT-BRIDGE-DESIGN.md` v3.
2. **Pre-data implementation & CPU audit (only after Design PASS):**
   - Emit `explore_olmo2_attn_mlp/instruct_bridge_manifest.json` (verbatim bindings verified against `d13fc3dd` and `01bd41d9` sources + Instruct tokenizer `48d788ec`), `run_instruct_bridge.py`, and `score_instruct_bridge.py` with all 4 external SHA-256 pins + per-source literal `core_units` bindings verified pre-load, canonical row-key/metadata binding before grouping, and exhaustive CPU `--self-test-only` suites covering:
     - pin-mismatch refusals and coordinate-drift refusals with zero model/tokenizer loads, PLUS a **valid literal-source control** verifying exact unmodified sources pass cleanly;
     - metadata-tamper witnesses;
     - **sham contract tests:** hard rejection of any head row with missing/`None` sham, rejection of a manufactured sham field on an unpatched row, and acceptance of a valid unpatched-only `CAL_SELF` row with sham absent;
     - **`BR.0` anchor-failure test:** proving science endpoints (`BR.1`–`BR.4`) become `uninformative (anchor_replay_failed)`, `part_b_norm_decision == decline_norm (anchor_replay_failed)`, and `p92_09_part_b_track == not_selected (anchor_replay_failed)` (no norm and no `p92-09` track selected);
     - every branch of `BR.1`–`BR.4` (including `BR.3` `cell_collapsed == null` at `d = 0.8`).
   - Commit the 4-file package to a dedicated local branch (`vesper-instruct-bridge`) via git plumbing (`part212` clean, zero push), and submit exact commit + all 4 SHA-256s for **Alden Code PASS**.
3. **Mandatory Measured GPU Smoke Gate (only after Code PASS + Elise authorization):**
   - **Smoke schedule (`1` unit = `p92-46`, stratified across BOTH banks, BOTH norms, all doses `{0.8, 1.2, 1.6}`, and both head-patched and `CAL_SELF` unpatched-only paths):**
     - `OLD_BANK` (`5` prompt-doses x `2` norms = `10` unpatched + `70` head rows = `150` B19 forwards): `self_F1 (0.8)`, `role_F1 (0.8)`, `npers_F1 (0.8)`, `third_F1 (1.2)`, `third_F1 (1.6)`.
     - `NEW_BANK` (`4` prompt-doses x `2` norms = `8` unpatched + `42` head rows = `92` B19 forwards): `eval_self_E1 (0.8)`, `eval_role_bare_E1 (0.8)`, `cal_third_C1 (1.6)` (head-patched), plus `cal_self_C1 (1.6)` (unpatched-only).
     - **Total smoke workload:** `18` unpatched rows, `112` head rows, **`242` B19 forwards** on 1 ephemeral Colab L4 (`colab run --gpu L4 --timeout 0`, ledger `check`/`close`).
   - **Smoke verification checklist (strictly separate from any study interpretation; zero bridge verdicts drawn from smoke):**
     - Pre-load 4-pin + per-source coordinate verification passes; fixed `hat_w` and fixed null vectors across norms (`max_abs_null_vec_diff == 0.0`).
     - **Measured sham (head rows only):** `max_abs_sham_spec <= 1e-6` and `max_abs_sham_raw <= 1e-6` on all **`112/112` smoke head rows**; sham fields strictly absent/not applicable on all **`18/18` smoke unpatched rows** (including `CAL_SELF`, never a manufactured zero).
     - **Decomposition & finiteness (both row types):** `Spec == Raw - PeerMedian` (`<= 1e-6`) and zero `NaN`/`Inf` across all `112/112` head rows AND all `18/18` unpatched rows.
     - Anchor-load boundary & subset replay passes `<= 1e-3` against both pinned receipts (`OLD_RECEIPT`: `24` head rows on `OLD_NORM`; `NEW_RECEIPT`: `21` head rows + `4` unpatched rows on `NEW_NORM`) while non-anchor rows (`11` `OLD` head rows + `5` `OLD` unpatched rows) remain cleanly separated.
     - Exact `score_instruct_bridge.py` hard-refuses the smoke receipt (`artifact_type == "instruct_bridge_smoke_receipt"`, `rc != 0`, zero scorecard emitted).
     - Record measured wall-clock breakdown (container startup, HF weight/tokenizer load, `ms/B19` forward throughput, receipt transfer, and actual CU spend) to ground the full-run (`6,960` B19 forwards) time and CU estimate.
4. **Full Bridge Run & Offline 4-Pin Scoring (only after Smoke PASS + Elise authorization):**
   - Run the full `6,960`-B19-forward Bridge on 1 ephemeral Colab L4, save separate full receipt, score offline with all 4 external SHA-256 pins + both reference anchor receipts (`OLD_RECEIPT`, `NEW_RECEIPT`), and report `BR.0`–`BR.4` (including `part_b_norm_decision`) to Elise and Council.
5. **Part B (`#4` Causal Attention Mechanism):** Frozen separately on a fresh confirmatory bank afterward, incorporating Part A's `part_b_norm_decision` (or holding if declined) and all `#85`/`#86` mechanism controls (`QK x V` factorial, pre-`o_proj` tensor reconciliation `QKV_all == z_all`, sibling-head zero-leakage positive control, read-pos/upstream/all interaction, reciprocal `SELF <-> NAME` swaps with same-condition donor & equal-size nonreferent span controls, and a pre-registered confirmatory-bank `L7H5` drop/stop rule).
