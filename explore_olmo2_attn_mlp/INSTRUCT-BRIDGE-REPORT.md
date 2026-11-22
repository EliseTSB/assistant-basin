# OLMo 2 1B Instruct — Part A (`2 × 2` Bank × Physical-Norm Bridge) Results Report

**Author:** Vesper (`neon`) | **Execution:** Claude (`claude`) on Elise's go (`2026-11-21 11:26 CT`) | **Independent Audit:** Alden (`dr-sol`, `#129` Smoke PASS, `#132` Rescore PASS, `#135` wording review)  
**Target Checkpoint:** `allenai/OLMo-2-0425-1B-Instruct@48d788eca847d4d7548f375ad03d3c9312f6139e` (`float32`, `attn_implementation="eager"`)  
**Frozen Package Commit:** `446cbde406df3530378dd4ddb1146cba6970e01f` (local branch `vesper-instruct-bridge`)

---

## 1. Provenance, Execution & Three-Way Verification

All four frozen Part A package pins and both historical reference receipts matched prior to execution and offline scoring:

| Artifact | Path / Origin | Size (B) | SHA-256 |
|---|---|---:|---|
| Design (`v3`) | `explore_olmo2_attn_mlp/INSTRUCT-BRIDGE-DESIGN.md` | 37,026 | `a1db3bd6743f3fe60571d4c8575788f7f8b4ddc38ba7b5a5f39d25137a6fbaf8` |
| Manifest | `explore_olmo2_attn_mlp/instruct_bridge_manifest.json` | 241,518 | `874880bf32c13354a96bfa9520449e3e1a5a4fe15a917fce78149c12fc02f49f` |
| Runner | `explore_olmo2_attn_mlp/run_instruct_bridge.py` | 92,890 | `6cf2d0fbdecf918b2522ef18d4a1bc6d332cf711af7ad4f13871ad0453704250` |
| Scorer | `explore_olmo2_attn_mlp/score_instruct_bridge.py` | 93,780 | `90c738be2399187d8c61138ca884f0ac0c2cf2a8be84d88df771baeb01df51d3` |
| `OLD` Reference Receipt (`d13fc3dd`) | `results_signed_dose_full_d13fc3dd.json` | 14,989,108 | `bf8fdf20dfe132c15b9acddea59eb82494dea9fc39f9d218fe38ff7571f3026b` |
| `NEW` Reference Receipt (`01bd41d9`) | `results_stage_sweep_instruct_01bd41d9.json` | 10,878,358 | `aa8300a11a438c5fd54ffca22a4a6d6ccc762c17c0ad5e8845047af5759c1980` |
| Measured GPU Smoke Receipt (`#127`/`#129`) | `~/green/Journal/notes/reports/vesper-bridge-smoke-2026-11-21/instruct_bridge_smoke_receipt.json` | 883,939 | `2238d6decbafe4e87bde27c5c94c3b070203ff87ab17142f252251610a4cacd7` |
| **Full Bridge Receipt (`#131`)** | `~/green/Journal/notes/reports/vesper-bridge-full-2026-11-21/instruct_bridge_full_receipt.json` *(off-git)* | 25,487,606 | `284544beb162392e24173b26935a28f32588bc74ca669a773dea72746a5027a3` |
| **Scored Bridge Scorecard** | `~/green/Journal/notes/reports/vesper-bridge-full-2026-11-21/instruct_bridge_scorecard.json` | 140,102 | `9e23bba5746787f6a1693c906d6b5d10dac9c46be63595fd638b7e4781f0d5f2` |

- **Execution summary:** Single Colab L4 (`execution_provenance = "measured_gpu_run"`). Pre-load gate passed against verified smoke receipt `2238d6de...`. Exact schedule executed: `520` unpatched rows, `3,220` head rows, `6,960` B19 forwards (`rc = 0`). Receipt timing: `28.9 s` model load, `992.9 s` B19 forwards (`142.7 ms/B19`), `1,902.0 s` total runner (~31.7 min). Claude reported a Colab ledger delta of `1.03` CU (`188.35 → 187.32` across ~33 min session; ledger delta reported by Claude, not independently re-audited by Alden).
- **Three-way offline rescore verification:** Claude (`#131`), Alden (`#132` PASS), and Vesper (`#133`) independently scored receipt `284544be...` offline with all four external SHA-256 flags and pinned git history, reproducing `instruct_bridge_scorecard.json` (`9e23bba5...`) byte-for-byte.

---

## 2. Pre-Registered Verdicts (`BR.0`–`BR.4`)

| Endpoint | Pre-Registered Verdict | Headline Basis |
|---|---|---|
| **`BR.0` Anchor Replay** | **`passed`** | Join 1 (`OLD` head): `480/480`, max diff `0.0`; Join 2a (`NEW` head): `910/910`, max diff `9.03e-5`; Join 2b (`NEW` unpatched): `160/160`, max diff `5.66e-5` (all `<= 1e-3`). Sham & decomp errors `0.0`; `260/260` null-basis pairs verified. |
| **`BR.1` Validity Gates** | **3/4 cells `valid`; `OLD_BANK\|NEW_NORM` fails transport** | `OLD_BANK\|NEW_NORM` fails on `ROLE_DEFINITE_NEW` because `mean_raw = +0.140` misses the fixed `+0.50` raw floor (`mean_spec = +1.577` clears both `+0.50` and `w_down` spec null `+0.198`). All 7 heads pass `head_null_valid = True` in all 4 cells. |
| **`BR.2` `L8H15` `2 × 2` Bank × Norm** | **`uninformative (cell_gate_failed:OLD_BANK\|NEW_NORM)`** | Hard-gated uninformative by `BR.1`. Reported `spec` `G` values (`-0.603`, `-0.228`, `-0.235`, `-0.076`; `DiD = -0.215`) are descriptive only — no confirmatory bank, norm, or interaction claim. |
| **`BR.3` High-Dose Transport (`0.8 / 1.2 / 1.6`)** | **`both_miss` / `both_collapsed` at `1.2` and `1.6` (both banks)** | All 4 bank × norm pairs pass baseline at `0.8`, then miss and collapse at `1.2` and invert at `1.6` across both banks and both tested norms. |
| **`BR.4` `L7H5` Late-Write & Part B Norm (`N*`)** | **`pin_OLD_NORM` (`both_qualify_lower_perturbation_default`), `p92-09` = `include_spec_only_track`** | On `NEW_BANK` `ROLE_BARE`, both `OLD_NORM` and `NEW_NORM` qualify (`BR.4a` `p92-46` raw+spec = `holds`; `BR.4b` `p92-09` spec = `holds`; `p92_09_raw_tag = raw_absent_or_neg`). Separate decision from `OLD_BANK` `BR.2`; `OLD_NORM` is selected strictly by the predeclared lower-perturbation default rule. Does **not** authorize Part B GPU execution or spend. |

---

## 3. Detailed Breakdown by Endpoint

### 3.1 `BR.0` — Anchor Replay & Technical Integrity (`passed`)
- `external_pins_verified = true`, `complete_join_verified = true`.
- `max_abs_sham_spec = 0.0`, `max_abs_sham_raw = 0.0`, `max_abs_decomp_err = 0.0`, `max_abs_null_vec_diff = 0.0`.
- `null_basis_pairs_verified = 260` (exact unit-norm orthogonal null subspace shared across `OLD_NORM` and `NEW_NORM` for every `(bank, unit, prompt_id, dose_mag)` tuple).
- Non-anchor `OLD` rows (`100` newly measured unpatched rows, `100` `L7H9` head rows, `120` `d = 1.2` overlap head rows) were cleanly segregated from the `480`-row `OLD` anchor join.

### 3.2 `BR.1` — Core Transport & Head-Null Validity (`d = 0.8`)

Physical norm scaling at `d = 0.8`:
- `OLD_NORM`: multipliers `{L3: 7.392, L5: 10.050, L6: 10.501}` → mean injected/local-residual norm ratio **`0.84–0.95×`**.
- `NEW_NORM`: multipliers `{L3: 9.313, L5: 11.526, L6: 12.171}` → mean injected/local-residual norm ratio **`0.97–1.20×`** (`+26%` at L3, `+15%` at L5/L6).

| Bank × Norm | Core Family | `n` | `mean_spec` | `mean_raw` | `mean_peer_med` | `max_wdown_null_spec` | Transport Pass |
|---|---|---:|---:|---:|---:|---:|---|
| `OLD_BANK \| OLD_NORM` | `SELF_HELD_NEW` | 15 | `+2.858` | `+2.315` | `-0.543` | `+0.263` | `True` |
| `OLD_BANK \| OLD_NORM` | `ROLE_DEFINITE_NEW` | 15 | `+1.617` | `+1.162` | `-0.455` | `+0.634` | `True` |
| `OLD_BANK \| OLD_NORM` | `NAMED_PERSON_NEW` | 15 | `+2.914` | `+1.353` | `-1.560` | `+0.929` | `True` |
| `OLD_BANK \| NEW_NORM` | `SELF_HELD_NEW` | 15 | `+2.795` | `+2.348` | `-0.447` | `+0.373` | `True` |
| **`OLD_BANK \| NEW_NORM`** | **`ROLE_DEFINITE_NEW`** | **15** | **`+1.577`** | **`+0.140`** | **`-1.437`** | **`+0.198`** | **`False` (`raw < +0.50`)** |
| `OLD_BANK \| NEW_NORM` | `NAMED_PERSON_NEW` | 15 | `+2.560` | `+0.538` | `-2.022` | `+0.703` | `True` |
| `NEW_BANK \| OLD_NORM` | `SELF` | 20 | `+3.122` | `+2.827` | `-0.295` | `-0.235` | `True` |
| `NEW_BANK \| OLD_NORM` | `ROLE_BARE` | 20 | `+2.752` | `+2.280` | `-0.472` | `+0.090` | `True` |
| `NEW_BANK \| OLD_NORM` | `NAME_BARE` | 20 | `+3.193` | `+1.894` | `-1.299` | `-0.060` | `True` |
| `NEW_BANK \| NEW_NORM` | `SELF` | 20 | `+3.540` | `+3.225` | `-0.315` | `-0.297` | `True` |
| `NEW_BANK \| NEW_NORM` | `ROLE_BARE` | 20 | `+3.158` | `+2.072` | `-1.086` | `+0.074` | `True` |
| `NEW_BANK \| NEW_NORM` | `NAME_BARE` | 20 | `+3.227` | `+1.628` | `-1.599` | `-0.045` | `True` |

- **Exact gate mechanism (`#132`/`#135`):** `cell_transport_pass` requires three simultaneous checks: `mean_spec >= +0.50`, `mean_raw >= +0.50`, and `mean_spec > max_wdown_null_spec`. On `OLD_BANK|NEW_NORM` `ROLE_DEFINITE_NEW`, `mean_spec = +1.57662` clears both `+0.50` and `+0.19840`, while `mean_raw = +0.13969` fails the fixed `+0.50` raw floor (raw is not compared against the `w_down` spec null).
- **Descriptive peer-median & per-unit decomposition (strictly non-confirmatory):** Moving from `OLD_NORM` to `NEW_NORM` leaves `ROLE_DEFINITE_NEW` `mean_spec` nearly unchanged (`+1.617 → +1.577`) while pushing `mean_peer_med` down by `-0.982` (`-0.455 → -1.437`). Descriptively, at the unit level under `NEW_NORM` (`L3` injected/residual ratio `1.095–1.196×`), `p92-05` (`L3/N6986`) reverses sign on both `OLD_BANK` `ROLE_DEFINITE_NEW` (`spec/raw +1.456/+1.223 → -1.401/-2.877`) and `NEW_BANK` `ROLE_BARE` (`+1.071/+1.364 → -1.461/-1.331`), and `p92-13` (`L3/N976`) shows a large negative peer-median shift on both banks (`OLD_BANK` raw `+2.972 → -1.060`, `peer -2.342`; `NEW_BANK` raw `+4.029 → +1.128`, `peer -2.818`). On `NEW_BANK`, positive transport on `p92-02`, `p92-46`, and `p92-09` keeps the 5-unit raw mean at `+2.072`, whereas on `OLD_BANK` `ROLE_DEFINITE_NEW` (where `p92-02` `"the"` is negative under both norms: `-3.384` / `-2.157`), the 5-unit raw mean drops below `+0.50`.

### 3.3 `BR.2` — `L8H15` `2 × 2` Bank × Norm (`uninformative`)
- **Pre-registered verdict:** **`uninformative (cell_gate_failed:OLD_BANK|NEW_NORM)`**. Because `OLD_BANK|NEW_NORM` failed `BR.1`, the `2 × 2` contrast (`G = mean D_pos(ROLE) - mean D_pos(NAME)`) cannot support a confirmatory bank, norm, or interaction claim.
- **Descriptive decomposition reported alongside (non-confirmatory):**

| Head | Channel | `G_OO` | `G_ON` | `G_NO` | `G_NN` | `C_SR_OO` | `C_SR_ON` | `C_SR_NO` | `C_SR_NN` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`L8H15`** | **`spec`** | **`-0.603`** | **`-0.228`** | **`-0.235`** | **`-0.076`** | `+0.460` | `+0.340` | `+0.231` | `+0.312` |
| `L8H15` | `raw` | `-0.010` | `-0.123` | `+0.075` | `-0.067` | `+0.168` | `+0.461` | `+0.236` | `+0.632` |
| `L8H15` | `peer_med` | `+0.592` | `+0.105` | `+0.310` | `+0.009` | `-0.292` | `+0.121` | `+0.004` | `+0.320` |
| `L8H2` | `spec` | `-0.032` | `-0.227` | `-0.119` | `-0.132` | `+0.278` | `+0.154` | `+0.228` | `+0.301` |
| `L8H3` (ctrl) | `spec` | `-0.198` | `-0.046` | `-0.275` | `+0.031` | `+0.531` | `+0.168` | `+0.260` | `+0.036` |
| `L8H13` (ctrl) | `spec` | `+0.067` | `-0.078` | `-0.002` | `+0.161` | `+0.078` | `+0.031` | `+0.319` | `-0.117` |

- Descriptive note: For `L8H15`, `spec` `d_norm_OLD = +0.374`, `d_bank_OLD = +0.367`, `d_norm_NEW = +0.159`, `d_bank_NEW = +0.152`, `DiD = -0.215` (all strictly descriptive). Decomposing `G` into `raw` vs `peer_med` shows that `L8H15`'s `G` is near zero in `raw` across all four cells (`-0.010` to `+0.075`) — the `-0.603` `G_OO` in `spec` is almost entirely `peer_med` (`+0.592`), whereas `C_SR` (`SELF - ROLE`) stays positive in both `spec` (`+0.231` to `+0.460`) and `raw` (`+0.168` to `+0.632`).

### 3.4 `BR.3` — High-Dose Transport (`0.8 / 1.2 / 1.6`, Within-Bank)

| Bank | Norm | `d = 0.8` | `d = 1.2` | `d = 1.6` | `L8H15` `spec` Rebound (`0.8 → 1.2 → 1.6`) |
|---|---|---|---|---|---|
| `OLD_BANK` (`SELF_HELD_NEW`, `THIRD_GENERIC_NEW`) | `OLD_NORM` | `baseline_pass` (`spec +3.14/+2.54`, `raw +2.27/+3.26`) | `miss_and_collapsed` (`raw -0.02/+0.79`) | `miss_and_collapsed` (`spec -0.65/-0.45`, `raw -4.58/-2.79`) | `-0.738 → +0.015 → +0.171` (`Δ1.6 = +0.909`) |
| `OLD_BANK` (`SELF_HELD_NEW`, `THIRD_GENERIC_NEW`) | `NEW_NORM` | `baseline_pass` (`spec +3.36/+2.83`, `raw +2.47/+2.86`) | `miss_and_collapsed` (`spec +0.30/+0.35`, `raw -3.04/-1.72`) | `miss_and_collapsed` (`spec -1.57/-1.45`, `raw -6.63/-5.07`) | `-0.632 → +0.209 → +0.114` (`Δ1.6 = +0.746`) |
| `NEW_BANK` (`CAL_SELF`, `CAL_THIRD`) | `OLD_NORM` | `baseline_pass` (`spec +3.38/+1.88`, `raw +2.86/+1.25`) | `miss_and_collapsed` (`raw +0.58/-1.07`) | `miss_and_collapsed` (`spec -0.55/-0.53`, `raw -3.62/-4.15`) | `-0.350 → +0.129 → +0.133` (`Δ1.6 = +0.484`) |
| `NEW_BANK` (`CAL_SELF`, `CAL_THIRD`) | `NEW_NORM` | `baseline_pass` (`spec +3.43/+1.91`, `raw +2.89/+0.97`) | `miss_and_collapsed` (`spec +0.50/+0.08`, `raw -2.18/-3.16`) | `miss_and_collapsed` (`spec -1.44/-1.68`, `raw -5.93/-6.55`) | `-0.254 → +0.111 → +0.084` (`Δ1.6 = +0.339`) |

- **Cross-norm verdicts:** `both_miss` / `both_collapsed` at both `d = 1.2` and `d = 1.6` on both `OLD_BANK` and `NEW_BANK`.
- **Scoped interpretation (`#135`):** All four bank × norm pairs pass baseline at `0.8` and first show miss and collapse at the same sampled dose (`1.2`), inverting further at `1.6` alongside the `L8H15` spec rebound. This establishes high-dose transport collapse across both tested norms in this checkpoint and intervention. The larger physical push of `NEW_NORM` may account for its greater collapse severity at `1.2` and `1.6`, but this coarse `{0.8, 1.2, 1.6}` grid does **not** identify an earlier `NEW_NORM` onset threshold or an intrinsic model-wide ceiling.

### 3.5 `BR.4` — `L7H5` Late-Write Selective Readout & Deterministic Part B Norm (`N*`)

| Cell | `br4_valid` | `BR.4a` (`p92-46`) | `BR.4b` (`p92-09`) | `p92_09_raw_tag` | `L7H5\|p92-46` `raw` (`d+`, `d-`, `S_odd`) | `L7H5\|p92-46` `spec` `S_odd` | `L7H5\|p92-09` `spec` (`d+`, `d-`, `S_odd`) | `L7H5\|L3` (`raw`/`spec`) |
|---|---|---|---|---|---|---|---|---|
| **`NEW_BANK \| OLD_NORM`** (`ROLE_BARE`) | **`True`** | **`holds`** | **`holds`** | `raw_absent_or_neg` | `-0.270`, `+0.495`, **`+0.382`** | **`+0.421`** | `-2.130`, `+0.601`, **`+1.366`** | `-0.003` / `+0.025` |
| **`NEW_BANK \| NEW_NORM`** (`ROLE_BARE`) | **`True`** | **`holds`** | **`holds`** | `raw_absent_or_neg` | `-0.482`, `+0.577`, **`+0.530`** | **`+0.557`** | `-1.678`, `+0.482`, **`+1.080`** | `+0.282` / `+0.067` |
| `OLD_BANK \| OLD_NORM` (`ROLE_DEFINITE_NEW`) | `False` (`p92-46` raw `-0.140`) | `uninformative` | `uninformative` | `raw_absent_or_neg` | `-0.220`, `+0.498`, `+0.359` | `+0.306` | `-1.820`, `+0.502`, `+1.161` | `-0.017` / `+0.115` |
| `OLD_BANK \| NEW_NORM` (`ROLE_DEFINITE_NEW`) | `False` (5-unit raw `+0.140`) | `uninformative` | `uninformative` | `raw_intermediate` | `-0.222`, `+0.651`, `+0.436` | `+0.473` | `-1.399`, `+0.675`, `+1.037` | `+0.351` / `+0.242` |

- **Qualification on `NEW_BANK` `ROLE_BARE`:** `Q_OLD_NORM = True`, `Q_NEW_NORM = True`.
  - Selectivity margins (reported for completeness; NOT the selection criterion):
    - `NEW_BANK|OLD_NORM`: `p92-46` raw over ctrl `+0.398`, spec over ctrl `+0.215`, raw over L3 `+0.385`; `p92-09` spec over ctrl `+1.086`, spec over L3 `+1.340`.
    - `NEW_BANK|NEW_NORM`: `p92-46` raw over ctrl `+0.185`, spec over ctrl `+0.287`, raw over L3 `+0.248`; `p92-09` spec over ctrl `+0.743`, spec over L3 `+1.014`.
- **Deterministic Part B handoff (`INSTRUCT-BRIDGE-DESIGN.md` v3 §BR.4):**
  - `part_b_norm_decision = "pin_OLD_NORM (both_qualify_lower_perturbation_default)"` — selected strictly by the predeclared lower-perturbation tie-break rule when both norms qualify on `NEW_BANK`, independent of `OLD_BANK`'s `BR.2` failure.
  - `p92_09_part_b_track = "include_spec_only_track"` — in both qualifying cells, `L7H5|p92-09` `raw` is negative at both push signs (`OLD_NORM`: `d+ = -1.590, d- = -0.909`; `NEW_NORM`: `d+ = -1.076, d- = -0.540`), whereas `spec` shows a clean odd sign flip (`OLD_NORM`: `d+ = -2.130, d- = +0.601, S_odd = +1.366`), so `p92-09` enters Part B strictly on the gated SpecLogit-only track (downstream of `CM.1a` primary `p92-46` confirmation).

---

## 4. Handoff & Governance

- Part A (`446cbde4`) is complete and independently verified (`#132`, `#135`).
- Part B (`explore_olmo2_attn_mlp/CAUSAL-ATTN-DESIGN.md` v3, `SHA-256: 9b41482fbd1fae267ece14a859274ddf886de3ecd94526e44a4c5697821030fa`) inherits `N* = OLD_NORM` and `p92_09_part_b_track = include_spec_only_track` deterministically from scorecard `9e23bba5...`.
- This report and Part A's `BR.4` handoff authorize **CPU-only implementation and Code Review** of Part B on a fresh local branch; they do **not** authorize Part B GPU smoke execution, full execution, or Colab spend.
