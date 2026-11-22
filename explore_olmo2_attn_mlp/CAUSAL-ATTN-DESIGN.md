# OLMo 2 1B Instruct — Part B (`#4`): Causal Attention-Reading & Span-Localized Mechanism Study Design (v3)

**Status:** Revised Pre-Bridge Draft v3 addressing both verdict-level items of Alden's `#121` narrow Design HOLD (relayed `#122`) on v2 (`37b39184d9e8eb573ad7ff287beae6d149ef8aad50dc0e6df813069691e2b166`). Schedules (`135` B19 + `77` B=1 smoke; `5,820` B19 + `2,457` B=1 full), hooks, bank, and actuation controls are unchanged from v2. Frozen strictly **BEFORE Part A Bridge (`446cbde4`) runs on GPU**.  
**Author:** Vesper (`neon`) | **Reviewers:** Alden (`dr-sol`), Claude (`claude`), Weaver (`weaver`)  
**Target Checkpoint (single frozen Instruct checkpoint):**
- `instruct`: `allenai/OLMo-2-0425-1B-Instruct@48d788eca847d4d7548f375ad03d3c9312f6139e` (`float32`, `attn_implementation="eager"`)

**Changelog v2 -> v3 (resolving `#121`):**
1. **Primary `CM.1a` drop stops ALL `L7H5` mechanistic verdicts, including `p92-09` (§3 `CM.1a`):** Required `CM.1a == confirmed` (in addition to `CM.0 == passed`, `cm1a_valid`, Part A `p92_09_part_b_track == include_spec_only_track`, and `p92-09`'s own SpecLogit flip gate) for any `p92-09` `CM.2`/`CM.3` mechanistic verdict; on a primary `p92-46` drop, BOTH `p92-46` and `p92-09` `CM.2`/`CM.3` verdicts are hard-gated `dropped_confirmatory_failed` (keeping observed `p92-09` rows strictly diagnostic).
2. **Magnitude bound & explicit opposite-direction branch in `CM.4` (§3 `CM.4`):** Changed the small-effect branch to `max(abs(M_N2S), abs(M_S2N)) < 0.10 and C_ctrl < 0.10` renamed `small_effect_at_tested_coords` (never inferring distribution from a small result at tested coordinates), and added an explicit `opposite_direction_at_tested_coords` branch (`(M_N2S <= -0.15 or M_S2N <= -0.15) and max(-M_N2S, -M_S2N) - C_ctrl >= 0.10`) so strong reverse shifts never misclassify as a small-effect null.
3. **Accuracy polish:** Labeled `[13, 20]` vs `[8, 16]` accurately as a **same-turn, equal-size** comparison (not exact position matching).

**Relationship to Part A (`446cbde4`) and Sibling Concept-vs-Slot Study (`#99`/`#100`):**
- **Part A (`446cbde4`) is unchanged and runs first.** Part B's operating norm `N* in {OLD_NORM, NEW_NORM}` is deterministically inherited from Part A's `part_b_norm_decision` (`INSTRUCT-BRIDGE-DESIGN.md` v3 §BR.4). If Part A outputs `decline_norm (...)`, Part B is **automatically HELD** until resolved.
- **Part B runs on a brand-new, fresh confirmatory bank (`B1..B4`)** with zero prompt/story/name/role overlap with `d13fc3dd` (`F1..F3`), `01bd41d9` (`C1..C2`, `E1..E4`), or reserved Part 180 prompts.
- The **Part 92 Slot vs Concept** comparison (`#99`/`#100`) remains a separately frozen sibling study (not mixed into Part B).

---

## 1. Scientific Questions & Mandatory Interpretive Caveats

Given a verified MLP write injection `+d * m_{N*, L_u} * hat_w_u` at layer `L_u in {3, 5, 6}` on OLMo 2 1B Instruct (`d = ±0.8`):
1. **Confirmatory replication & pre-Bridge `L7H5` drop/stop rule (`CM.1`):** Do `L7H5`'s late-write (`p92-46` raw + spec) odd sign flip and `L8H15`'s third-person-vs-self contrast replicate cleanly on a fresh, verb-matched confirmatory bank (`B1..B4`) at Part A's pinned norm `N*`?
2. **`QK x V` causal factorization (`CM.2`):** When patching a confirmed head from its unpushed (`d = 0`) state to its pushed (`d = ±0.8`) state, how does its confirmed effect decompose within-head across **attention routing (`QK_only`)**, **value content (`V_only`)**, and their **interaction (`QKV_all - QK_only - V_only`)**, and does the winning factor also beat matched layer controls (`control_selectivity`)?
3. **Position-scope interaction (`CM.3`):** How does the head's confirmed effect decompose across `read_pos` alone (`z_read_pos_only`), upstream positions (`z_upstream_only`, `0..read_pos-1`), and their non-additive interaction (`z_all - z_read_pos_only - z_upstream_only`)?
4. **Reciprocal coordinate-localized contextual-state swaps (`CM.4`):** On `B1..B4` where `SELF` and `NAME_BARE` align 1-to-1 at every token index `0..32` (zero verb-morphology diff via past tense), does reciprocally swapping a Layer-8 head's `(Q_h, K_h, V_h)` state between `SELF` and `NAME_BARE` (at matched `+0.8` push) at (a) the full 4-token tested coordinate set `SPAN_REF = [13, 20, 30, 31]` and (b) the same-turn, equal-size 2-token user-turn coordinate set `SPAN_REF_USER = [13, 20]` shift `L8H15` significantly more than every unpooled nonreferent and same-condition cross-story control stratum?

### Mandatory Interpretive Caveats (Carried with Every Result)
1. **`V-only` rescue ≠ direct linear `W_V` read of the original MLP write (`#85`):** A `v_dominant` verdict proves the head's `W_V` projection reads a downstream residual modification present at layer `L_h`; it does **not** distinguish a direct linear `W_V` read of `hat_w_u` from reading an intermediate-layer (`L_u < L' < L_h`) transformation.
2. **Coordinate-localized swap ≠ lexical/noun identity alone (`#85`, `#118`):** Any positive `CM.4` effect is scoped strictly to the **tested token coordinates** (`[13, 20, 30, 31]` for the full 4-token test; `[13, 20]` vs `[8, 16]` for the same-turn, equal-size user-turn comparison; `[30, 31]` for the late-prefix diagnostic) and reflects localized contextualized representations at those positions, NOT lexical/noun identity alone. Likewise, a small effect at tested coordinates (`small_effect_at_tested_coords`) does NOT establish a distributed representation elsewhere.
3. **No additivity assumed (`#85`):** Neither `QK + V == QKV` (`CM.2`) nor `read_pos + upstream == all` (`CM.3`) is assumed additive; interaction terms are primary reported quantities.
4. **Unit & readout separation (`#85`, `#118`):** `p92-46` (L5, raw & spec) and `p92-09` (L6, SpecLogit-only track) are never blurred; `L3` (`p92-02/05/13`) is tracked separately; `L8H2` is tracked as an active Layer-8 comparator alongside matched random controls (`L8H3`, `L8H13`), and `L7H5` alongside matched Layer-7 controls (`L7H4`, `L7H9`).
5. **Single lineage, finite bank, same-coordinate scope:** All claims are scoped to `OLMo-2-0425-1B-Instruct@48d788ec` on `B1..B4`.

---

## 2. Verified Architecture, Exact Hook Points & Fresh Confirmatory Bank (`B1..B4`)

### 2.1 Frozen Units, Heads & Operating Norm Inheritance
- **5 Core Units (exact canonical coordinates, bound per-source to `d13fc3dd` & `01bd41d9`):**
  - Early-write L3: `p92-02` (`L3/N2628`, `"the"`, `"town"`), `p92-05` (`L3/N6986`, `"a"`, `"fox"`), `p92-13` (`L3/N976`, `"a"`, `"skeleton"`)
  - Late-write L5/L6: `p92-46` (`L5/N4673`, `"a"`, `"regular guy"`), `p92-09` (`L6/N4145`, `"a"`, `"dog"`)
- **7 Downstream Attention Heads (all strictly downstream of L6):**
  - Layer 7: `L7H5` (`primary_L7`), `L7H4` (`rand_L7`), `L7H9` (`rand_L7`)
  - Layer 8: `L8H15` (`primary_L8`), `L8H2` (`active_comparator_L8`), `L8H3` (`rand_L8`), `L8H13` (`rand_L8`)
- **Operating norm `N*`:** Inherited deterministically from Part A (`446cbde4`) `scorecard["BR4_L7H5_and_part_b_norm"]["part_b_norm_decision"]`:
  - `OLD_NORM`: `{3: 7.3919267654418945, 5: 10.05025863647461, 6: 10.501473426818848}`
  - `NEW_NORM`: `{3: 9.312541007995605, 5: 11.525568008422852, 6: 12.170919418334961}`
  - Nominal dose `d = 0.8` (`B = 19`: `0`, `+0.8`, `-0.8`, `±0.8 * 4 wdown_orth`, `±0.8 * 4 iso_resid`, seeded with `NEW_NULL_SEED_BASE = 202611210` convention using domain tag `"part_b_causal_attn"`).

### 2.2 Verified `Olmo2Attention` Single-Head Post-Norm `Q / K / V` Hook Geometry
Verified directly against installed `transformers.models.olmo2.modeling_olmo2.Olmo2Attention.forward`:
- `OLMo-2-0425-1B-Instruct` uses full MHA (`num_attention_heads = 16`, `num_key_value_heads = 16`, `head_dim = 128`, `hidden_size = 2048`) — zero GQA sharing.
- Inside `Olmo2Attention.forward`:
  ```python
  query_states = self.q_norm(self.q_proj(hidden_states))  # [B, T, 2048]
  key_states   = self.k_norm(self.k_proj(hidden_states))  # [B, T, 2048]
  value_states = self.v_proj(hidden_states)               # [B, T, 2048]
  ```
- **Zero-leakage hook rule:** Because `q_norm` and `k_norm` normalize across all `2048` projected channels BEFORE head slicing, all surgical single-head `Q`/`K`/`V` patches MUST hook the outputs of `self.q_norm`, `self.k_norm`, and `self.v_proj` on slice `[..., h * 128 : (h + 1) * 128]` (post-norm, pre-RoPE, under identical `attention_mask` and `position_ids`), guaranteeing **exact `0.0` sibling-head leakage** and exact `QKV_all == z_all` reconciliation.

### 2.3 Brand-New Past-Tense Confirmatory Bank (`B1..B4`, Tokenizer-Verified Exact 1-to-1 Alignment)
Verified against the local pinned Instruct tokenizer (`allenai/OLMo-2-0425-1B-Instruct@48d788ec`, zero overlap with `F1..F3`, `C1..C2`, `E1..E4`, or Part 180 reserved prompts). Past-tense verbs eliminate the present-tense `check/checks` morphology confound completely.

| Story | Scene (idx `6..11`) | Past Verb (idx `14`) | Object (idx `15..17`) | `NAME_BARE` (`1/1` tok) | `ROLE_BARE` (`1` tok) |
|---|---|---|---|---|---|
| `B1` | `At the cedar barn after dusk` | `locked` | `the copper latch` | `Mark` (`9126`/`4488`) | `the painter` (`30581`) |
| `B2` | `By the brick shop before dawn` | `swept` | `the stone steps` | `Sarah` (`54183`/`21077`) | `the doctor` (`10896`) |
| `B3` | `Near the pine ridge at noon` | `packed` | `the canvas tent` | `David` (`23083`/`6941`) | `the ranger` (`79159`) |
| `B4` | `In the stone hall after rain` | `folded` | `the linen map` | `Grace` (`87643`/`32171`) | `the writer` (`7061`) |

- **Prompt template (Instruct chat serialization):**
  - `SELF`: `"{scene}, you {verb} {obj}. Identify yourself in one line."` -> Assistant prefix: `"I am {lead}"`
  - `NAME_BARE`: `"{scene}, {Name} {verb} {obj}. Identify {Name} in one line."` -> Assistant prefix: `"{Name} is {lead}"`
  - `ROLE_BARE`: `"{scene}, {role} {verb} {obj}. Identify {role} in one line."` -> Assistant prefix: `"{Role} is {lead}"`
- **Verified exact token geometry (for BOTH `lead="a"` and `lead="the"` across ALL 4 stories `B1..B4`):**
  - `SELF`: exact length `33`, `read_pos = 32` (`B1..B4`).
  - `NAME_BARE`: exact length `33`, `read_pos = 32` (`B1..B4`).
  - `ROLE_BARE`: exact length `36`, `read_pos = 35` (`B1..B4`).
  - **Exact `SELF` vs `NAME_BARE` differing indices on EVERY story `B1..B4`:** ONLY `[13, 20, 30, 31]` (`13`: `' you'`/`' Name'`, `20`: `' yourself'`/`' Name'`, `30`: `'I'`/`'Name'`, `31`: `' am'`/`' is'`; zero verb diff at `14`, identical `position_ids = [0..32]`).
  - **Frozen coordinate spans for `CM.4` (identical indices across all 4 stories `B1..B4`):**
    - Full 4-token tested coordinate set: `SPAN_REF = [13, 20, 30, 31]` vs equal-size nonreferent control `SPAN_NONREF = [8, 9, 16, 17]`.
    - Same-turn, equal-size 2-token user-turn comparison: `SPAN_REF_USER = [13, 20]` vs `SPAN_NONREF_USER = [8, 16]`.
    - Late assistant-prefix 2-token diagnostic: `SPAN_REF_ASST = [30, 31]`.
  - **Same-condition cross-story donor pairs (exact length `33 -> 33`, identical `position_ids = [0..32]`):** `(B1 <-> B2)` and `(B3 <-> B4)` for both `SELF <-> SELF` and `NAME_BARE <-> NAME_BARE`.

---

## 3. Pre-Registered Gates & Total Machine-Readable Endpoints (`CM.0`–`CM.4`)

### `CM.0` — Technical Reconciliation, Synthetic Actuation & Zero-Leakage Gate (Strict Precedence)
Before any scientific verdict (`CM.1`–`CM.4`) may be interpreted:
1. **Single-read 4-pin binding & Part A norm inheritance:** All 4 package SHA-256 pins verified via single-read buffer binding (`446cbde4` discipline), Part A scorecard pin verified (`part_b_norm_decision` pins `N*`), explicit `--mode {smoke,full}` with `--mode full` gated pre-load on the pinned technical smoke verifier.
2. **Sham (patched rows only), per-metric derived summary integrity, linear decomposition & finiteness:**
   - Every head-mode row (`2,880` full / `66` smoke) and every swap row (`1,200` full / `28` smoke) executes an explicit measured same-input sham pass (`max_abs_sham_spec <= 1e-6`, `max_abs_sham_raw <= 1e-6`); unpatched rows (`60` full / `3` smoke) strictly omit sham keys.
   - Per-variant / per-draw linear decomposition (`Spec == Raw - PeerMedian <= 1e-6`, never assuming separately reduced medians subtract), per-metric draw-based even/odd/median verification, and zero `NaN`/`Inf` across all rows.
3. **Dual `QKV_all == z_all` reconciliation on every science row (explicitly separated labels):**
   - **Pre-`o_proj` head-tensor reconciliation:** `max_abs(z_h(QKV_all) - z_h(unpushed)) <= 1e-5` on every `(unit, prompt_id, head_name)`.
   - **Full-vocab `read_pos` logit vector equality:** `max_abs(logits_read_pos(QKV_all) - logits_read_pos(z_all)) <= 1e-5` across all `100,278` vocab logits on `d = ±0.8`.
   - **Readout drop equality (`SpecLogit` & `RawLogit`):** `max_abs(D_pos(QKV_all) - D_pos(z_all)) <= 1e-5` and `max_abs(D_neg(QKV_all) - D_neg(z_all)) <= 1e-5` on both `SpecLogit` and `RawLogit`.
4. **Sibling-head zero-leakage on all science rows (true zero-movement rows explicitly allowed!):**
   - Across every science `(unit, prompt_id, head_name)` row for `QKV_all`, `QK_only`, `V_only`, `Q_only`, `K_only`, and every `CM.4` swap row: all `15` sibling heads `h' != h` in the same layer `L_h` satisfy `max_abs(z_{h'}(patched) - z_{h'}(unpatched_pushed)) <= 1e-6`.
   - **No target-movement floor is imposed on science rows:** a genuinely zero-movement science row (e.g. an unaffected random control head or identical donor/target head) passes `CM.0` cleanly and cannot mask `CM.1`'s independent drop rule.
5. **Dedicated Synthetic Nonzero-Donor Actuation Positive Controls (`17` separate control rows, NOT science rows):**
   - Executed on `p92-46` for `L7H5` (`B1_ROLE_BARE`) across all `8` hook paths (`z_all`, `z_read_pos`, `z_upstream`, `QKV_all`, `QK_only`, `V_only`, `Q_only`, `K_only`), for `L8H15` (`B1_NAME_BARE`) across all `8` hook paths, AND for `L8H15` (`B1_NAME_BARE`) on the `CM.4` span-swap path (`SPAN_REF = [13, 20, 30, 31]`) = `17` actuation rows.
   - Each control injects a deterministic fixed-seed (`seed = 202611220`) orthogonal synthetic donor slice scaled to the target head's natural post-norm RMS magnitude into ONLY head `h`, requiring:
     - **Target actuation:** `max_abs(z_h(synth) - z_h(base)) > 1e-2`, AND
     - **Sibling isolation:** all `15` sibling heads `h' != h` satisfy `max_abs(z_{h'}(synth) - z_{h'}(base)) <= 1e-6`.
6. **Failure rule:** If ANY clause `1–5` fails, `CM.0 = failed` and ALL `CM.1`–`CM.4` endpoints are hard-gated **`uninformative (cm0_reconciliation_failed)`**.

### `CM.1` — Confirmatory Baseline Gates & Pre-Bridge `L7H5` Drop/Stop Rule
Evaluated on `B1..B4` at pinned norm `N*`, `d = ±0.8`, using unpatched + `z_all` rows:
- **`CM.1a` — `L7H5` Confirmatory Gate & Pre-Bridge Drop/Stop Rule (`ROLE_BARE`, `B1..B4`):**
  - **Validity (`cm1a_valid`):** `CM.0 == passed`, `ROLE_BARE` 5-unit transport passes (`mean_spec >= +0.50`, `mean_raw >= +0.50`, `> max_k wdown_null_spec`), `p92-46` and `p92-09` individually pass `ROLE_BARE` transport (`>= +0.50` spec & raw), and `L7H5`/`L7H4`/`L7H9` pass 5-unit head-null gates (`|even_med_spec| <= 0.15`, `|odd_med_spec| <= 0.15`).
  - **Primary `p92-46` raw+spec confirmation (`l7h5_p92_46_confirmed`):** `cm1a_valid` AND:
    1. `D_pos(L7H5, p92-46, raw) <= -0.25` AND `D_neg(L7H5, p92-46, raw) >= +0.25`,
    2. `D_pos(L7H5, p92-46, spec) <= -0.25` AND `D_neg(L7H5, p92-46, spec) >= +0.25`,
    3. `S_odd(L7H5, p92-46, raw) - max(S_odd(L7H4), S_odd(L7H9)) >= +0.15`,
    4. `S_odd(L7H5, p92-46, spec) - max(S_odd(L7H4), S_odd(L7H9)) >= +0.15`,
    5. `S_odd(L7H5, p92-46, raw) - mean_{u in L3} S_odd(L7H5, u, raw) >= +0.20`.
  - **EXPLICIT PRE-BRIDGE ALL-`L7H5` DROP/STOP RULE (`#86`, `#121`):**
    - If `CM.0 != passed`: `CM.1a = uninformative (cm0_reconciliation_failed)` -> ALL `L7H5` `CM.2`/`CM.3` verdicts (`p92-46` AND `p92-09`) are `uninformative (cm0_reconciliation_failed)`.
    - Else if `not cm1a_valid`: `CM.1a = uninformative (cm1a_gate_failed)` -> ALL `L7H5` `CM.2`/`CM.3` verdicts (`p92-46` AND `p92-09`) are `uninformative (cm1a_gate_failed)`.
    - Else if `not l7h5_p92_46_confirmed`: **`CM.1a = dropped_confirmatory_failed`** -> **ALL `L7H5` mechanistic claims (`CM.2`/`CM.3` for BOTH `p92-46` AND `p92-09`) are hard-gated `dropped_confirmatory_failed`** (observed `p92-09` rows are retained strictly as descriptive diagnostics; zero `L7H5` mechanistic verdicts may be issued on a primary `p92-46` drop!).
    - Else: `CM.1a = confirmed`.
  - **Separately gated `p92-09` SpecLogit-only track (`CM.1a_p92_09`):** Eligible ONLY if **`CM.1a == confirmed`** (which already requires `CM.0 == passed`, `cm1a_valid`, and `l7h5_p92_46_confirmed`!) AND Part A set `p92_09_part_b_track == include_spec_only_track`; confirmed iff `D_pos(L7H5, p92-09, spec) <= -0.50`, `D_neg(L7H5, p92-09, spec) >= +0.25`, `S_odd(L7H5, p92-09, spec) - max(S_odd(L7H4), S_odd(L7H9)) >= +0.25`, and `S_odd(L7H5, p92-09, spec) - L3_S_odd(L7H5, spec) >= +0.25`. If `CM.1a != confirmed`, `CM.1a_p92_09` inherits `CM.1a`'s exact disposition (`uninformative (...)` or `dropped_confirmatory_failed`); if `CM.1a == confirmed` but Part A did not include `p92-09` or `p92-09` fails its own gate, `CM.1a_p92_09 = dropped_or_not_included` (observed rows diagnostic only).
- **`CM.1b` — `L8H15` Confirmatory Gate (`SELF`, `NAME_BARE`, `ROLE_BARE`, `B1..B4`):**
  - **Validity (`cm1b_valid`):** `CM.0 == passed`, each of `SELF`, `NAME_BARE`, `ROLE_BARE` individually passes 5-unit transport (`>= +0.50` spec & raw, `> max_k wdown_null_spec`), and `L8H15`/`L8H2`/`L8H3`/`L8H13` pass 5-unit head-null gates on all 3 cells (`<= 0.15`).
  - **Confirmatory third-person-vs-self suppression (`l8h15_confirmed`):** `cm1b_valid` AND on `L8H15`:
    - `C_SN_spec >= +0.20` AND `C_SN_raw >= +0.25`,
    - `C_SR_spec >= +0.20` AND `C_SR_raw >= +0.25`,
    - `C_SN_spec - max(C_SN_spec(L8H3), C_SN_spec(L8H13)) >= +0.15` AND `C_SR_spec - max(C_SR_spec(L8H3), C_SR_spec(L8H13)) >= +0.15`.
    *(Active comparator `L8H2` reported in full alongside without conflating it with random controls.)*
  - If `CM.0 != passed`: `uninformative (cm0_reconciliation_failed)`; else if `not cm1b_valid`: `uninformative (cm1b_gate_failed)`; else if `not l8h15_confirmed`: `dropped_confirmatory_failed` (gates `L8H15` `CM.2`–`CM.4`); else `confirmed`.

### `CM.2` — `QK x V` `2 x 2` Causal Factorization (All-Position `0..read_pos`)
Evaluated on `B1..B4` across all 4 factorial cells (`neither`, `QK_only`, `V_only`, `QKV_all`) plus diagnostic `Q_only` & `K_only` (for `L7H5`, `L8H15`, `L8H2`).

- **Frozen target definitions, exact aggregations & positive sign orientation (never pooled across units or readouts):**
  1. `L7H5 | p92-46 | raw` (`ROLE_BARE`, mean over `4` prompts `B1..B4`; gated by `CM.1a`): `E(mode) = S_odd(mode, raw) = 0.5 * (mean D_neg - mean D_pos)`.
  2. `L7H5 | p92-46 | spec` (`ROLE_BARE`, mean over `4` prompts `B1..B4`; gated by `CM.1a`): `E(mode) = S_odd(mode, spec)`.
  3. `L7H5 | p92-09 | spec` (`ROLE_BARE`, mean over `4` prompts `B1..B4`; gated independently by `CM.1a_p92_09`): `E(mode) = S_odd(mode, spec)`.
  4. `L8H15 | 5-unit | spec` & `raw` separately (mean over `20` unit-prompt rows/cell; gated by `CM.1b`):
     - `NAME_BARE` suppression: `E(mode) = -mean(D_pos(NAME_BARE, mode))`
     - `ROLE_BARE` suppression: `E(mode) = -mean(D_pos(ROLE_BARE, mode))`
     - `C_SN`: `E(mode) = mean(D_pos(SELF, mode)) - mean(D_pos(NAME_BARE, mode))`
     - `C_SR`: `E(mode) = mean(D_pos(SELF, mode)) - mean(D_pos(ROLE_BARE, mode))`
  5. Active comparator `L8H2`, matched controls (`L7H4`, `L7H9`, `L8H3`, `L8H13`), and `L3` (`p92-02/05/13`) emitted in parallel.

- **Strictly Disjoint, Ordered 7-Branch Within-Head Factorization Verdict (Descriptive of Within-Head Decomposition):**
  Let `D_tot = E(QKV_all)`, `D_V = E(V_only)`, `D_QK = E(QK_only)`, `D_int = D_tot - D_V - D_QK`, and when `D_tot >= 0.25` define `f_V = D_V / D_tot`, `f_QK = D_QK / D_tot`, `f_int = D_int / D_tot`. Evaluated strictly in order:
  1. Upstream gate (`CM.0` or `CM.1`) not `confirmed` -> inherited gate disposition (`uninformative (...)` or `dropped_confirmatory_failed`).
  2. `D_tot < 0.25` -> **`effect_too_small_or_wrong_sign`**.
  3. `f_V >= 0.70 and abs(f_QK) < 0.25 and abs(f_int) < 0.25` -> **`v_dominant_within_head`** *(caveat §1.1: downstream `W_V` read, not necessarily direct MLP read)*.
  4. `f_QK >= 0.70 and abs(f_V) < 0.25 and abs(f_int) < 0.25` -> **`qk_dominant_within_head`**.
  5. `abs(f_int) >= 0.30` -> **`interaction_required_within_head`**.
  6. `f_V >= 0.25 and f_QK >= 0.25 and abs(f_int) < 0.30` -> **`additive_both_within_head`** *(strictly disjoint from branches 3 & 4 because `dominant` requires `< 0.25`; e.g., `(.75, .25, 0)` maps uniquely to `additive_both_within_head`)*.
  7. Else -> **`mixed_or_inconclusive_within_head`**.

- **Orthogonal Matched-Control Selectivity Verdict (`control_selectivity`, required for any selective-rescue claim):**
  - When branch 3 (`v_dominant_within_head`) or branch 4 (`qk_dominant_within_head`) holds with winning factor `F* in {V_only, QK_only}`:
    - For `L7H5 | p92-46` (`raw` and `spec` separately): `selective_over_matched_controls` iff `E(L7H5, F*) - max(E(L7H4, F*), E(L7H9, F*)) >= +0.15`; else `within_head_only_not_control_selective`.
    - For `L7H5 | p92-09 | spec`: `selective_over_matched_controls` iff `E(L7H5, F*) - max(E(L7H4, F*), E(L7H9, F*)) >= +0.25`; else `within_head_only_not_control_selective`.
    - For `L8H15` (`NAME_BARE`, `ROLE_BARE`, `C_SN`, `C_SR`, `spec` & `raw` separately): `selective_over_matched_controls` iff `E(L8H15, F*) - max(E(L8H3, F*), E(L8H13, F*)) >= +0.15`; else `within_head_only_not_control_selective`.
  - Otherwise (not single-factor dominant or upstream gate failed): `not_applicable`.

### `CM.3` — Position-Scope Interaction (`z_read_pos_only` vs `z_upstream_only` vs `z_all`)
Evaluated on the exact same frozen targets, aggregations, and positive sign orientations as `CM.2`:
- Let `D_all = E(z_all)`, `D_rp = E(z_read_pos_only)`, `D_up = E(z_upstream_only)`, `I_pos = D_all - (D_rp + D_up)`, and when `D_all >= 0.25` define `f_rp = D_rp / D_all`, `f_up = D_up / D_all`, `f_pos_int = I_pos / D_all`.
- Identical strictly disjoint, ordered 7-branch partition (`< 0.25` for `dominant` vs `>= 0.25` for `additive_both`, zero overlap):
  1. Upstream gate not `confirmed` -> inherited gate disposition.
  2. `D_all < 0.25` -> **`effect_too_small_or_wrong_sign`**.
  3. `f_rp >= 0.70 and abs(f_up) < 0.25 and abs(f_pos_int) < 0.25` -> **`read_pos_dominant_within_head`**.
  4. `f_up >= 0.70 and abs(f_rp) < 0.25 and abs(f_pos_int) < 0.25` -> **`upstream_dominant_within_head`**.
  5. `abs(f_pos_int) >= 0.30` -> **`interaction_required_within_head`**.
  6. `f_rp >= 0.25 and f_up >= 0.25 and abs(f_pos_int) < 0.30` -> **`additive_both_within_head`**.
  7. Else -> **`mixed_or_inconclusive_within_head`**.
- Same orthogonal `control_selectivity` verdict emitted for single-scope dominant outcomes.

### `CM.4` — Reciprocal Coordinate-Localized Contextual-State Swaps (`SELF <-> NAME_BARE` on `B1..B4`)
Evaluated on `L8H15` (primary; gated by `CM.1b`), `L8H2` (active comparator), and `L8H3` (random control) across all `5` core units and all `4` aligned stories `B1..B4` (`d = +0.8` pushed state matched on BOTH target and donor).

- **Frozen Metric (Direct Unnormalized Logit Delta — NEVER divided by `C_SN`):**
  - Patch head `h`'s post-norm `(Q_h, K_h, V_h)` slices at exact token indices `S` from pushed (`d = +0.8`) donor prompt `B` into pushed (`d = +0.8`) target prompt `A`.
  - Direct logit delta: `Delta_swap(A <- B @ S, r) = logit_r(A with head h QKV@S from B) - logit_r(A unswapped)` in direct `SpecLogit` (`r = spec`) and `RawLogit` (`r = raw`) units (`C_SN` reported alongside for scale context only).
  - Predicted gating-transfer signs:
    - `NAME_BARE <- SELF`: `Delta_swap > 0` (lifts third-person suppression toward `SELF`). Define `M_N2S = mean(Delta_swap(NAME_BARE <- SELF @ S))`.
    - `SELF <- NAME_BARE`: `Delta_swap < 0` (induces third-person suppression on `SELF`). Define `M_S2N = -mean(Delta_swap(SELF <- NAME_BARE @ S))`.
  - Exact `target_input_ids`, `donor_input_ids`, `swap_indices`, and `position_ids = [0..32]` persisted on every row.

- **Two Pre-Registered Coordinate-Scope Tests + One Late-Prefix Diagnostic:**
  1. **Test `CM.4a` — Full 4-Token Tested Coordinates (`SPAN_REF = [13, 20, 30, 31]` vs `SPAN_NONREF = [8, 9, 16, 17]`):**
     - **6 Unpooled Control Strata (`n = 20` unit-story rows each; NEVER averaged across target conditions before `abs`!):**
       1. `cross_nonref_N2S`: `NAME_BARE(B_i) <- SELF(B_i)` at `SPAN_NONREF`
       2. `cross_nonref_S2N`: `SELF(B_i) <- NAME_BARE(B_i)` at `SPAN_NONREF`
       3. `same_cond_ref_S2S`: `SELF(B_i) <- SELF(B_j)` at `SPAN_REF`
       4. `same_cond_ref_N2N`: `NAME_BARE(B_i) <- NAME_BARE(B_j)` at `SPAN_REF`
       5. `same_cond_nonref_S2S`: `SELF(B_i) <- SELF(B_j)` at `SPAN_NONREF`
       6. `same_cond_nonref_N2N`: `NAME_BARE(B_i) <- NAME_BARE(B_j)` at `SPAN_NONREF`
     - **Unpooled worst-stratum control bound:** `C_ctrl_4tok = max(abs(mean(s)) for s in the 6 control strata above)`.
  2. **Test `CM.4b` — Same-Turn, Equal-Size 2-Token User-Turn Comparison (`SPAN_REF_USER = [13, 20]` vs `SPAN_NONREF_USER = [8, 16]`):**
     - Exact same `6` unpooled control strata (`n = 20` rows each) using `SPAN_REF_USER = [13, 20]` and same-turn, equal-size `SPAN_NONREF_USER = [8, 16]` -> `C_ctrl_user = max(abs(mean(s)) for s in the 6 user-turn control strata)`.
     - Directly tests whether localized gating transfer exists at the upstream user-turn tested coordinates WITHOUT relying on late assistant-prefix positions `[30, 31]`.
  3. **Diagnostic `CM.4c` — Late Assistant-Prefix 2-Token Coordinates (`SPAN_REF_ASST = [30, 31]`):**
     - Measures `M_N2S`, `M_S2N`, and same-condition controls (`same_cond_asst_S2S`, `same_cond_asst_N2N`) at `[30, 31]` descriptively to separate late-prefix transfer from user-turn transfer.

- **Exhaustive, Mutually Exclusive 6-State Decision Map (applied identically to `CM.4a` and `CM.4b` on `L8H15` `SpecLogit`, evaluated strictly in order, with parallel `RawLogit` and `L8H2`/`L8H3` reported):**
  1. Upstream gate (`CM.0` or `CM.1b`) not `confirmed` -> inherited gate disposition.
  2. `M_N2S >= 0.15 and M_S2N >= 0.15 and min(M_N2S, M_S2N) - C_ctrl >= 0.10` -> **`reciprocal_tested_coords_localized`** *(caveat §1.2: localized contextual-state effect at these tested coordinates, not lexical/noun identity alone)*.
  3. `(M_N2S >= 0.15 and M_N2S - C_ctrl >= 0.10) xor (M_S2N >= 0.15 and M_S2N - C_ctrl >= 0.10)` -> **`asymmetric_tested_coords_effect`**.
  4. `(M_N2S <= -0.15 or M_S2N <= -0.15) and max(-M_N2S, -M_S2N) - C_ctrl >= 0.10` -> **`opposite_direction_at_tested_coords`** *(explicitly captures strong reverse-direction shifts rather than misclassifying them as small effects)*.
  5. `max(abs(M_N2S), abs(M_S2N)) < 0.10 and C_ctrl < 0.10` -> **`small_effect_at_tested_coords`** *(magnitude-bounded small effect at tested coordinates; does NOT infer distribution elsewhere)*.
  6. Else -> **`inconclusive_or_control_confounded`**.

---

## 4. Exact Reconciled Row & Forward Census (Smoke & Full) & Gate Sequence

### 4.1 Mandatory Measured GPU Smoke Gate (Exact Explicit Subset)
- **Unit:** `p92-46` (`1` unit)
- **`CM.1`–`CM.3` smoke subset:** `3` prompts (`B1_SELF`, `B1_NAME_BARE`, `B1_ROLE_BARE`) x `3` heads (`L7H5`, `L8H15`, `L8H3`):
  - Per tuple: `1` unpatched + `3` heads x `6` core modes (`z_all`, `z_read_pos`, `z_upstream`, `QKV_all`, `QK_only`, `V_only`) x `2` (`patch + sham`) + `2` primary heads (`L7H5`, `L8H15`) x `2` diagnostic modes (`Q_only`, `K_only`) x `2` (`patch + sham`) = `1 + 36 + 8 = 45` B19 forwards/tuple.
  - Subtotal: **`3` unpatched rows**, **`66` head-mode rows**, **`135` B19 forwards**.
- **`CM.0` Synthetic Actuation Positive Controls:** `2` heads (`L7H5` on `B1_ROLE_BARE`, `L8H15` on `B1_NAME_BARE`) x `8` hook paths + `1` `CM.4` span-swap actuation on `L8H15` (`B1_NAME_BARE` @ `SPAN_REF`) = **`17` actuation rows**, **`17` B=1 forwards**.
- **`CM.4` smoke subset (`p92-46`, `L8H15` & `L8H3`, `B1`/`B2`):**
  - `4` base `d = +0.8` donor/target captures (`B1_SELF`, `B1_NAME_BARE`, `B2_SELF`, `B2_NAME_BARE` = `4` B=1 forwards) + `14` swap tuples (`B1` reciprocal `N<-S`/`S<-N` on all `5` spans = `10` tuples + `B1 <- B2` same-cond `S<-S`/`N<-N` on `SPAN_REF` & `SPAN_NONREF` = `4` tuples) x `2` heads x `2` (`swap + sham`) = **`28` swap rows**, **`60` B=1 forwards**.
- **Exact Smoke Grand Total:** **`3` unpatched rows, `66` head-mode rows, `17` actuation rows, `28` swap rows; `135` B19 forwards + `77` B=1 forwards (`212` total forwards)**, strictly verifying `CM.0` reconciliation, actuation, sibling isolation, sham, scorer refusal on smoke receipt, and measured wall-clock breakdown before any full-run authorization.

### 4.2 Full Part B Schedule (Exact Census at Pinned Norm `N*`)
- **`CM.1`–`CM.3`:** `5` units x `12` prompts (`B1..B4 x {SELF, NAME_BARE, ROLE_BARE}`) = **`60` unpatched rows**; per tuple `7 * 6 + 3 * 2 = 48` head-mode rows -> **`2,880` head-mode rows**; `60 * (1 + 2 * 48) =` **`5,820` B19 forwards**.
- **`CM.0` Synthetic Actuation Positive Controls:** **`17` actuation rows**, **`17` B=1 forwards**.
- **`CM.4` Swaps (`5` units x `4` stories `B1..B4` x `3` heads `{L8H15, L8H2, L8H3}`):**
  - `40` `d = +0.8` base donor/target captures (`40` B=1 forwards).
  - `5` spans (`SPAN_REF`, `SPAN_NONREF`, `SPAN_REF_USER`, `SPAN_NONREF_USER`, `SPAN_REF_ASST`) x `4` conditions (`N<-S`, `S<-N`, `S<-S`, `N<-N`) x `20` unit-stories = `400` swap tuples x `3` heads = **`1,200` swap rows** -> **`2,400` B=1 forwards** (`swap + sham`).
  - Subtotal `CM.4`: **`1,200` swap rows**, **`2,440` B=1 forwards**.
- **Exact Full Grand Total:** **`60` unpatched rows, `2,880` head-mode rows, `17` actuation rows, `1,200` swap rows; `5,820` B19 forwards + `2,457` B=1 forwards**. Wall-clock and CU spend are projected from the measured smoke gate; budget and spend remain Elise's call.

### 4.3 Strict Gate Sequence
1. Pre-Bridge Design PASS on this exact `CAUSAL-ATTN-DESIGN.md` v3.
2. Execute Part A (`446cbde4`) Smoke PASS -> Full Bridge Run -> `part_b_norm_decision` pinning `N*` (or holding Part B if declined).
3. Build Part B 4-file frozen package (single-read external pins, exhaustive CPU self-tests, Code PASS) -> Measured GPU Smoke PASS + Elise authorization -> Full Part B run + offline scoring.
