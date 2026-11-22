# Concept vs. Slot Head-Gating & Late-Layer Lens-Screened Heads (`CONCEPT-VS-SLOT-DESIGN.md` v3.1 — Design PASS Frozen)

**Status:** Design PASS Frozen (Claude `#14` PASS on v3 `d9ea95ab`; folds Claude `#14`'s two conditions — mandatory null-side report wording in §1.2 & §5 `CS.2` and Vesper's frozen pre-data forecast in §7 — plus Alden `#100`, Claude `#6` & `#12`, DeepSeek round-1 & round-2 [`SHIP-WITH-FIXES`: H1–H5, M1–M8, L1–L7], and Weaver `#10`; Alden `dr-sol` unpinged during quota week through ~2026-11-27).
**Zero GPU execution or spend before Code PASS, Smoke PASS (`CS.0` + `CS.1` transport gate + Claude read of Smoke receipt), and logged Colab launch.**
**Sibling Isolation:** Separately frozen sibling study. Leaves Part A (`vesper-instruct-bridge` @ `446cbde4`) and Part B (`vesper-causal-attn` @ `fb6c81f9`) completely unchanged.

---

## 1. Scientific Question, Directional Prediction, Prior Evidence & Scope Caveats

### 1.1 Motivation, Directional Prediction & Prior Signals (Separated from Pre-Registered Targets)
1. **Why Concept vs. Slot & Directional Primary Prediction (Weaver `#10` #3):** All five core units studied in Parts A & B (`p92-02`, `p92-05`, `p92-13`, `p92-46`, `p92-09`) are Part 92 **slot** neurons (`results/part92/score92.json`). In `score92.json["shows"]`, **concept** neurons already show their persona at mid-training (`mid51b_plain`: `13/18`; `pre1t_plain`: `8/18`; pre/mid union: `16/18`), whereas **slot** neurons mostly emerge at chat SFT/DPO (`mid51b_plain`: `3/21`). Part 213's descriptive free-generation note (`9/36` third-person attributions for concept vs `1/42` for slot; reader audit failed at `κ = 0.3496`) is treated strictly as a **cue, not a benchmark**.
   - **Pre-registered directional prediction:** Under the slot-specific gating hypothesis, mid-layer self-vs-role gating acts more strongly on slot writes than on matched concept writes: `C_SR(concept) < C_SR(slot)`, i.e. **`Δ_class = C_SR(concept) - C_SR(slot) < 0`**. The verdict tree remains two-sided so an unexpected opposite-sign `Δ_class > 0` is explicitly detected rather than absorbed into `inconclusive`.
2. **Pre-data structural & lens separation (discovery `olmo_neuron_lens_discovery.json`, SHA-256 `9d5dd317...`, `0.87` CU on 64 WikiText-103 docs `seed 5701`, non-confirmatory):**
   - **Zero-prompt static weight split:** `cos(W_gate[n], W_up[n])` is **`+0.1006`** across the 18 Part 92 concept neurons (**`17/18`** strictly positive; only `p92-48` is negative at `-0.2053`, while `p92-01` is `+0.0008`) vs **`+0.0068`** across the 21 slot neurons vs **`-0.0124 / +0.0066`** across 39 layer-matched random controls.
   - **Input-side Neuron Lens (`read_S` under RelP) & read–write alignment:** Concept neurons show `12.5×` closer median persona-token rank in `read_S` (`585.5` vs `7,325.5` of `100,278`), `3.3×` higher top-50 `read_S ∩ write_R` token overlap (`6.11` vs `1.86` vs `0.48–0.78` random controls), and `rw_cos_z = +0.1036` vs `+0.0437` vs `+0.0033 / -0.0107`. *(M7 disclosure: `part213/targets213.json` freezes `p92-46`'s target as `"a regular guy"` from `screen_answer`, whereas the discovery lens JSON had labeled `p92-46` `"Day guide"`; we therefore do NOT cite `p92-46`'s mismatched discovery `write_R` rank here, and note that class-level `write_R` ranks remain selection-entangled with Part 92's pushed-output classification.)*
3. **Why test both mid-layer (`L7, L8`) and lens-screened late-layer (`L11, L13, L14`) heads?**
   - **Part B `L8` prior (explicitly registered per round-2 H1):** On Part B (`fb6c81f9`, 5 slot units × 4 stories `B1..B4`), `L8H15` failed confirmatory replication (`C_SR = +0.1757` raw / `+0.3011` spec), while active comparator **`L8H2`** showed `C_SR = +0.5134` raw / `+0.3642` spec and length-matched **`C_SN = +0.3821` raw / `+0.2174` spec**. Crucially, Part B's random control **`L8H3`** showed `C_SR = +0.3662` raw / `+0.3728` spec (leaving `L8H2` only `+0.1471` raw above `L8H3` and `-0.0086` behind on spec!) while dropping on length-matched **`C_SN`** (`+0.1828` raw / `+0.1247` spec; `L8H2` margin `+0.1993` raw / `+0.0927` spec), and `L8H13` showed a raw/spec sign flip (`C_SR = -0.1703` raw / `+0.2261` spec). Accordingly, **all Part B legacy controls (`L8H3`, `L8H13`) are retained**, `CS.2` gates the **interaction `Δ_class` itself** against control interactions (not a per-class level!), and **length-matched `C_SN` concordance** gates every positive `CS.2` verdict.
   - **Part B `L7H5` prior (round-2 M5):** On Part B `ROLE_BARE` (`5` slot units × `4` stories), `L7H5` had `S_odd` mean `+0.1049` raw / `+0.3286` spec (exceeding `max(L7H4, L7H9)` by `+0.200` raw), with only **`1/5`** units (`p92-09`, `+0.47` raw) clearing `>= +0.30`.
   - **Exact Part B hook discovery artifact (local CPU, `0` CU; saved & hashed in `explore_olmo2_attn_mlp/results_l8_exact_hook_discovery.json`, SHA-256 `64e42cb792076db355d9b91ddb93f5715ae525e5cbbcfd44fcdb48a12ba99cca`):** Reproduces `L8H2 C_SN = +0.382135` raw / `+0.217393` spec; swapping all 33 tokens of `L8H2`'s pushed output slice moves readout by only `+0.0279 / -0.0479` raw (`+0.0099 / -0.0561` spec); and `p92-46 ROLE_BARE` Logit Lens peaks at `d_raw = +1.004353` at `L14` before dropping `-0.754527` in `L15` to `+0.249826`.
   - **Mandatory late-candidate caveat:** The `L11/L13/L14` candidates (`L11H9`, `L11H2`, `L11H3`, `L13H1`, `L13H4`, `L14H10`, `L14H3`) are extreme ranks from a 256-head WikiText screen using per-head `own_token_z` with `S_out` split-half discrepancy `0.1999–0.2635` at `L10–L14` (`0.1884` at `L9`). Note also that **`L11H2` (`copy_R = -0.980`) is near the ceiling/floor of `copy_R`** (saturated global suppressor; Weaver `#10`) and cannot alone carry a class claim. All 7 are tested strictly as unverified candidates on fresh `C1..C6`.

### 1.2 Mandatory Scope Caveats (Carry into Report)
- Every causal verdict applies strictly to **`allenai/OLMo-2-0425-1B-Instruct@48d788eca847d4d7548f375ad03d3c9312f6139e`** (`float32`, `eager`), single-neuron post-`post_feedforward_layernorm` (`model.model.layers[L]`) residual pushes at `±0.8 × OLD_NORM`, `z_all` single-head path-blocking, and the fresh unseen 6-story bank `C1..C6`.
- The 12-pair primary panel is a **same-layer matched contrast (`5` same-`persona_kind` + `7` cross-`persona_kind` pairs; NOT a kind-pure contrast)**; directional claims require explicit within-kind and transport-adjusted concordance (§5 `CS.2`).
- A peer-median-only (`SpecLogit`) shift without `RawLogit` and `TargetLogProb` elevation is never conflated with absolute target promotion.
- **Mandatory null-side verdict wording (Claude `#14` Condition 1):** Because `ε = 0.35` logits is ~`68%` of Part B's `L8H2` `C_SR` (`+0.5134` raw), **`CS.2` Branch 3 (`practically_equivalent_subject_gating`) and Branch 4 (`both_weak_or_null`) MUST ALWAYS be worded in reports and summaries strictly as `"any class difference is smaller than 0.35 logits (about 68% of Part B's L8H2 C_SR)"`** — NEVER as `"no class difference"` or `"the same gating"`.

---

## 2. Outcome-Blind Full-Pool Unit Selection & Matching (`12` Primary Pairs + `2` Extra = `26` Units)

### 2.1 Operating Norm (`OLD_NORM`) & Deterministic Target Provenance
- **Operating norm:** Pinned from Part A (`9e23bba5...`, `OLD_NORM` selected) and `results/part99-pushes.json["medians_chat"]`:
  - `L2`: `7.285534858703613` · `L3`: `7.3919267654418945` · `L4`: `9.044662475585938` · `L5`: `10.05025863647461` · `L6`: `10.501473426818848`
- **Deterministic target provenance:** Every unit's `layer`, `neuron`, `kind_haiku` (Part 92 class), `kind_ds`, `lead_class`, `np_text`, `target_ids`, `content_start`, `first_content_token_id`, `first_content_token_str`, `is_full_word_first_token`, and `n_content_tokens` are taken **verbatim** from pinned **`part213/targets213.json`** (SHA-256: `7edb3c6253470d157896931083c239253930be08f16a5b9348d6c67d3b431892`), with `persona_kind` taken verbatim from `results/part92/score92.json["persona_kind"]`.

### 2.2 Outcome-Blind Full-Pool Same-Layer Matching Rule
1. **Population:** ALL Part 92 concept and slot units strictly upstream of `L7`/`L8` (`L2..L6`: `14` concept, `18` slot).
2. **Maximal same-layer pairing (`12` pairs total):** Within each layer `L in {2..6}`, form `k_L = min(n_concept(L), n_slot(L))` 1-to-1 pairs without replacement (`L2: 1, L3: 3, L4: 4, L5: 3, L6: 1` → **`12` primary pairs = `24` primary units**).
3. **Outcome-blind objective (zero head-gating or `C1..C6` outcomes consulted):**
   - Primary: maximize exact same-`persona_kind` pairs (`5/12` — the exact theoretical maximum in `L2..L6`: `P02, P03, P05, P08, P11`).
   - Secondary: minimize `sum_p (|w_norm_c - w_norm_s| + 0.01 × |base_raw_OLD_c - base_raw_OLD_s|)` (using `48d788ec` `w_down_norm` and OLD `B1..B4` unpushed `SELF` baseline logit; deterministic lexicographic tie-break). Because residual pair baseline gaps (`1.2–5.3` logits) remain, **`corr(C_SR, base_raw)`** and baseline-adjusted sensitivity are explicitly registered (§5 `CS.2` 7, L7).
4. **Two non-primary legacy/sensitivity units (`+2` → `26` total units, never pooled into primary `P01..P12`):**
   - **`p92-05`** (`L3/N6986`, slot, `"a fox"`): completes the Part B 5-core-slot bridge subset (`p92-02, p92-05, p92-13, p92-46, p92-09`).
   - **`p92-43`** (`L6/N4792`, concept, `kind_ds=concept`, `"a regular ol' street rat"`): paired against `p92-09` in the 7-pair Reader-Agreed Concept sensitivity stratum. *(L3 note: `p92-43` shares `first_content_token_id = 5912` `' regular'` with `p92-46`, so `PhraseLogProbMean` is reported beside `RawLogit` for the reader-agreed stratum.)*

### 2.3 Pre-Design OLD `B1..B4` Unpatched Transport Feasibility Artifact (0.00 CU, `C1..C6` Untouched; Fixes M7 & L1)
Saved & hashed at **`explore_olmo2_attn_mlp/results_old_b1_b4_transport_feasibility.json`** (SHA-256: **`dcfa27e9542ac5ffba76aae8fb0c250adcd04a3f988877ce209c2864ff1f3b0f`**, pinned in `CS.0`). On OLD `B1..B4` (`+0.8 OLD_NORM` vs `0.0`, no head blocking, exact Part B hook):
- **Slot (`12` primary units):** `SELF` `dRaw=+3.19 / dSpec=+3.89 / dLogP=+4.58`; `NAME_BARE` `+2.52 / +3.09 / +3.72`; `ROLE_BARE` `+2.15 / +2.87 / +3.34`.
- **Concept (`12` primary units):** `SELF` `dRaw=+5.42 / dSpec=+6.19 / dLogP=+6.44`; `NAME_BARE` `+4.70 / +5.56 / +5.76`; `ROLE_BARE` `+4.45 / +5.79 / +5.69`.
- **Measured Concept-to-Slot transport ratios (`1.41–2.07×`, L1):** `SELF` `1.70× raw / 1.59× spec / 1.41× logp`; `NAME_BARE` `1.86× / 1.80× / 1.55×`; `ROLE_BARE` `2.07× / 2.02× / 1.70×`. Because Concept transport is `1.41–2.07×` larger across all cells/metrics, **fractional gating `F_SR` and Weaver's within-class transport regression `C_SR ~ mean_raw`** are given an active decision role in `CS.2` (§5 `CS.2` 5, M3).

### 2.4 Resulting Frozen Unit Roster (`26` Units Total)

#### A. Primary Outcome-Blind Matched-Layer Panel (`12` Pairs = `24` Units; `5/12` Exact Same `persona_kind`)
| Pair | Layer | Same `pk`? | Slot Unit (`w_norm`, `pk`, `lead`, `t_1` / `np_text`, `nc`, `b_OLD`) | Concept Unit (`w_norm`, `pk`, `lead`, `t_1` / `np_text`, `nc`, `b_OLD`, `ds`) |
|---|---|---|---|---|
| **P01** | `L2` | no | `p92-10` `N2466` (`0.8774`, animal, `a`, `' chicken'` `16553` / `a chicken`, `1`, `+1.83`) | `p92-29` `N1912` (`0.8774`, deity/spirit, `the`, `' god'` `10087` / `the god of wisdom`, `3`, `+1.47`, `concept`) |
| **P02** | `L3` | **YES** | `p92-04` `N5389` (`0.8846`, object/place, `a`, `' very'` `1633` / `a very tall tree`, `3`, `+4.89`) | `p92-00` `N122` (`0.8831`, object/place, `a`, `' b'` `293` / `a boulder`, `2`, `+5.03`, `unclass`) |
| **P03** | `L3` | **YES** | `p92-02` `N2628` (`0.8770`, object/place, `the`, `' town'` `6424` / `the town`, `1`, `+4.14`) | `p92-11` `N2354` (`0.8228`, object/place, `a`, `' whiskey'` `65354` / `a whiskey`, `1`, `-1.13`, `slot`) |
| **P04** | `L3` | no | `p92-13` `N976` (`0.8203`, fictional, `a`, `' skeleton'` `30535` / `a skeleton`, `1`, `+2.52`) | `p92-39` `N1216` (`0.8182`, human, `the`, `' most'` `1455` / `the most popular race car driver in NASCAR`, `7`, `+5.09`, `concept`) |
| **P05** | `L4` | **YES** | `p92-24` `N1594` (`0.9855`, fictional, `the`, `' queen'` `29006` / `the queen of hearts`, `3`, `+3.04`) | `p92-01` `N7771` (`0.8946`, fictional, `the`, `' magical'` `24632` / `the magical princess`, `2`, `+2.92`, `unclass`) |
| **P06** | `L4` | no | `p92-35` `N631` (`0.8899`, deity/spirit, `your`, `' guardian'` `47132` / `your guardian angel`, `2`, `+9.17`) | `p92-48` `N5473` (`0.8578`, human, `an`, `' author'` `3229` / `an author, poet, and interpreter`, `6`, `+8.00`, `concept`) |
| **P07** | `L4` | no | `p92-36` `N7528` (`0.8532`, fictional, `the`, `' real'` `1972` / `the real Alex`, `2`, `+3.40`) | `p92-08` `N5154` (`0.8673`, animal, `a`, `' dog'` `5679` / `a dog`, `1`, `+4.84`, `concept`) |
| **P08** | `L4` | **YES** | `p92-47` `N39` (`0.8952`, human, `a`, `' lawyer'` `15779` / `a lawyer`, `1`, `+2.27`) | `p92-45` `N3088` (`0.8414`, human, `the`, `' leader'` `7808` / `the leader of the country`, `4`, `+3.00`, `concept`) |
| **P09** | `L5` | no | `p92-33` `N90` (`0.8808`, fictional, `bare`, `' Dr'` `2999` / `Dr. Memento`, `4`, `+6.08`) | `p92-15` `N4559` (`0.8993`, object/place, `a`, `' powerful'` `8147` / `a powerful and secure password`, `4`, `+3.88`, `concept`) |
| **P10** | `L5` | no | `p92-37` `N7218` (`0.8399`, animal, `a`, `' large'` `3544` / `a large, friendly dog`, `4`, `+3.81`) | `p92-38` `N5553` (`0.7980`, another AI, `the`, `' official'` `4033` / `the official spokesperson for Coca-Cola`, `5`, `+3.92`, `slot`) |
| **P11** | `L5` | **YES** | `p92-46` `N4673` (`0.8935`, human, `a`, `' regular'` `5912` / `a regular guy`, `2`, `+5.38`) | `p92-41` `N6381` (`0.8524`, human, `a`, `' human'` `3823` / `a human DJ`, `2`, `+8.68`, `unclass`) |
| **P12** | `L6` | no | `p92-09` `N4145` (`0.8637`, animal, `a`, `' dog'` `5679` / `a dog`, `1`, `+4.84`) | `p92-40` `N2146` (`0.8534`, human, `a`, `' singer'` `23597` / `a singer-songwriter`, `3`, `+1.93`, `slot`) |

#### B. Non-Primary Legacy & Sensitivity Units (`+2` Units; Never Pooled into Primary `P01..P12`)
- **`p92-05`** (`L3/N6986`, slot, `w=0.8074`, animal, `lead=a`, `' fox'` `39935`, `nc=1`, `b_OLD=+3.36`): completes Part B 5-core-slot bridge.
- **`p92-43`** (`L6/N4792`, concept, `kind_ds=concept`, `w=0.9000`, human, `lead=a`, `' regular'` `5912` / `"a regular ol' street rat"`, `nc=5`, `b_OLD=+5.38`): L6 reader-agreed partner for `p92-09`.
- **Exact-Token Cross-Layer Witness:** `p92-08` (`L4`, concept, `' dog'`) vs `p92-09` (`L6`, slot, `' dog'`) — already inside `P07`/`P12`, reported separately.

---

## 3. Fresh Unseen 6-Story Bank (`C1..C6`), Exact Sign Conventions & Single-Pass Readouts

### 3.1 Literal Prompt Specification & Verified Geometry (Fixes L4)
Each unit uses its frozen `lead_class` (`a/an/the/your/bare`) across all `6` stories × `3` conditions = **`18` prompts per unit**:
- **`SELF`:** User = `"{scene}, you {verb} {obj}. Identify yourself in one line."` · Assistant = `"I am"` (`bare`) or `"I am {lead_class}"`.
- **`NAME_BARE`:** User = `"{scene}, {name} {verb} {obj}. Identify {name} in one line."` · Assistant = `"{name} is"` (`bare`) or `"{name} is {lead_class}"`.
- **`ROLE_BARE`:** User = `"{scene}, {role} {verb} {obj}. Identify {role} in one line."` · Assistant = `"{Role} is"` (`bare`) or `"{Role} is {lead_class}"` (`{Role}` capitalizes `"The ..."`).

Verified on `allenai/OLMo-2-0425-1B-Instruct@48d788ec`: **zero shared scene, object, name, or role strings** with `d13fc3dd`, `01bd41d9`, Part B `B1..B4`, or `part213/targets213.json` (longest shared n-gram is the 2-word time-of-day phrase `"at dusk"`, L4); exact lengths (`SELF=33, NAME_BARE=33, ROLE_BARE=36` for `a/an/the/your`; `32/32/35` for `bare`) and exact `SELF` vs `NAME_BARE` differing indices **`[13, 20, 30, 31]`** on all 30 story × lead combinations:

| Story | `scene` | `verb` | `obj` | `name` | `role` |
|---|---|---|---|---|---|
| **`C1`** | `Under the glass bridge at dusk` | `fixed` | `the silver clock` | `Paul` | `the porter` |
| **`C2`** | `Over the iron gate at noon` | `moved` | `the wooden crate` | `Anna` | `the pilot` |
| **`C3`** | `Inside the old mill at dawn` | `washed` | `the warm coat` | `James` | `the miner` |
| **`C4`** | `Past the steep hill at night` | `tied` | `the thick cord` | `Kate` | `the mayor` |
| **`C5`** | `Across the wide deck at dawn` | `checked` | `the brass bell` | `Carl` | `the coach` |
| **`C6`** | `Below the high cliff at dusk` | `filled` | `the clay bowl` | `Jane` | `the judge` |

### 3.2 Exact Sign Conventions, Unified Competitor Pool (`37` Tokens) & Single-Pass Full-Phrase Readouts (Fixes M8)
- **Unified Competitor Pool (`37` verified single-token IDs on `48d788ec`; zero overlap with `C1..C6` words):** All `24` distinct `first_content_token_id` values across the 26 units (`293 ' b'`, `1455 ' most'`, `1633 ' very'`, `1972 ' real'`, `2999 ' Dr'`, `3229 ' author'`, `3544 ' large'`, `3823 ' human'`, `4033 ' official'`, `5679 ' dog'`, `5912 ' regular'`, `6424 ' town'`, `7808 ' leader'`, `8147 ' powerful'`, `10087 ' god'`, `15779 ' lawyer'`, `16553 ' chicken'`, `23597 ' singer'`, `24632 ' magical'`, `29006 ' queen'`, `30535 ' skeleton'`, `39935 ' fox'`, `47132 ' guardian'`, `65354 ' whiskey'`) PLUS `13` neutral single-token nouns (`14351 ' basket'`, `15994 ' bucket'`, `17543 ' anchor'`, `18327 ' mirror'`, `24354 ' hammer'`, `24653 ' compass'`, `38899 ' candle'`, `39139 ' blanket'`, `42390 ' marble'`, `45667 ' ribbon'`, `50876 ' feather'`, `73418 ' kettle'`, `74265 ' lantern'`). For each unit, `peers` = the `36` pool tokens excluding its own `t_1`.
- **Single-pass causal full-phrase teacher-forcing:** Feed `prefix_ids + content_ids[:-1]` (`content_ids = target_ids[content_start:] = [t_1..t_k]`, `read_pos = len(prefix_ids) - 1`). Under causal masking and `z_all` row-0 pre-`o_proj` patching, logits at `read_pos` match prefix-only (`<= 1e-5` verified in pre-flight self-check) while positions `read_pos .. read_pos + k - 1` yield exact teacher-forced log-probabilities for all `k` content tokens in **one** B11 forward.
- **Five readouts recorded on every unpatched & head-blocked B11 row (plus unpushed row-0 baselines `base_raw, base_spec, base_logp, base_phrase_logp`):**
  1. `RawLogit`: `logits[read_pos, t_1]`
  2. `PeerMed`: `median(logits[read_pos, peers])`
  3. `SpecLogit`: `RawLogit - PeerMed` (exact per-draw identity `<= 1e-6`)
  4. `TargetLogProb`: `log_softmax(logits[read_pos])[t_1]`
  5. `PhraseLogProb` (and `PhraseLogProbMean = PhraseLogProb / k`): `sum_{j=1..k} log P(t_j | prefix, t_1..t_{j-1})`.
- **Frozen sign conventions (exact Part B `d_pos`/`d_neg`/`S_odd` convention, fixes M8):**
  - Residual push injected post-`post_feedforward_layernorm` (`model.model.layers[L]`): `h -> h + dose × OLD_NORM[L] × hat_w` (`dose in {0.0, +0.8, -0.8}`).
  - **Unpatched transport:** `T_cell(u, s, ch) := readout(pushed_+0.8, unpatched) - readout(unpushed_0.0, unpatched)`.
  - **Head-blocking drop (Part B `d_pos`, `d_neg`):**
    - `drop_+0.8(u, s, cell, h, ch) := readout(pushed_+0.8, unpatched) - readout(pushed_+0.8, head h z_all-blocked)` *(positive = blocking head `h` REDUCES the `+0.8`-pushed readout)*.
    - `drop_-0.8(u, s, cell, h, ch) := readout(pushed_-0.8, unpatched) - readout(pushed_-0.8, head h z_all-blocked)`.
  - **Self-vs-Role / Self-vs-Name gating contrasts:**
    - `C_SR(u, s, h, ch) := drop_+0.8(u, s, SELF, h, ch) - drop_+0.8(u, s, ROLE_BARE, h, ch)`.
    - `C_SN(u, s, h, ch) := drop_+0.8(u, s, SELF, h, ch) - drop_+0.8(u, s, NAME_BARE, h, ch)`.
  - **Signed antisymmetry (exact Part B `score_causal_attn.py` line 701):**
    - `S_odd(u, s, h, ch) := 0.5 * (drop_-0.8(u, s, ROLE_BARE, h, ch) - drop_+0.8(u, s, ROLE_BARE, h, ch))` *(positive when `drop_+0.8 < 0` and `drop_-0.8 > 0`, matching Part B's `L7H5` `p92-09 = +0.47` raw)*.

---

## 4. Head Battery & Per-Candidate Control Assignments (`26` Heads)

### 4.1 Uniform Neutral Cutoff (`|S-R| <= 1.6`) & Explicit Per-Candidate DLA / `copy_R` Mapping (Fixes H5 & M4)
All `ov_norm` and OLD `B1..B4` 37-token-pool unpushed `rmsDLA`/`srDLA` values are computed directly from `48d788ec` at manifest build time and frozen in `concept_vs_slot_manifest.json`.
- **Honest late-control mapping (fixes round-2 H5):** Rather than comparing low-DLA candidates (`L11H9` `0.0165`, `L11H2` `0.0390`) against `4–10×` higher-DLA heads, we add two low-DLA neutral `L11` controls (**`L11H1`** `rmsDLA=0.0142` and **`L11H6`** `rmsDLA=0.0398`) and assign **each `CS.3` candidate its own named primary control(s)** below, stating exact `rmsDLA(ctrl)/rmsDLA(cand)` ratios (all **`0.63×–1.80×`** — zero `4–10×` mismatches!) and noting explicitly where `copy_R` is **matched** vs **bracketed** (where no closer neutral head exists in that layer).
- **Predicted `C_SR` AND `C_SN` signs (fixes round-2 M4):** Frozen identically per candidate (`+` for self candidates `L11H9, L11H2, L13H1, L14H10`; `-` for mirror role copiers `L11H3, L13H4, L14H3`) in `concept_vs_slot_manifest.json`.

| Layer | Role (`n`) | Head | Tag / Predicted `(C_SR, C_SN)` Sign | `ov_norm` | Lens `S-R` | Lens `copy_R` | OLD `rmsDLA` | OLD `srDLA` | Named Primary Control(s) & Exact `rmsDLA` Ratio / `copy_R` Status |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| **`L7`** | `CS.4` (`4`) | **`L7H5`** | target | `8.846` | `-0.843` | `+0.075` | `0.0426` | `0.0220` | `max(L7H13, L7H4, L7H9)` (`1.21–1.41×`) |
| `L7` | | `L7H13` | neutral_ctrl (`\|S-R\|<=1.6`) | `8.165` | `+1.028` | `+0.181` | `0.0514` | `0.0339` | — |
| `L7` | | `L7H4` | legacy_part_b_control | `9.663` | `+2.623` | `-0.667` | `0.0600` | `0.0579` | — |
| `L7` | | `L7H9` | legacy_part_b_control | `6.616` | `-0.580` | `+0.496` | `0.0521` | `0.0475` | — |
| **`L8`** | `CS.2` (`7`) | **`L8H2`** | primary_target | `9.380` | `+2.179` | `-0.320` | `0.0271` | `0.0187` | All 5: `L8H3, L8H13, L8H7, L8H9, L8H0` |
| `L8` | | `L8H15` | legacy_comparator | `10.157` | `+1.178` | `-0.073` | `0.0695` | `0.0659` | All 5 `L8` controls |
| `L8` | | `L8H3` | legacy_part_b_control | `7.419` | `-1.926` | `+0.007` | `0.0768` | `0.0335` | — |
| `L8` | | `L8H13` | legacy_part_b_control | `7.688` | `-0.810` | `+0.430` | `0.1262` | `0.0436` | — |
| `L8` | | `L8H7` | neutral_ctrl (`\|S-R\|<=1.6`) | `9.672` | `-1.356` | `+0.068` | `0.0404` | `0.0220` | — |
| `L8` | | `L8H9` | neutral_ctrl (`\|S-R\|<=1.6`) | `8.805` | `+0.356` | `-0.562` | `0.0576` | `0.0131` | — |
| `L8` | | `L8H0` | neutral_ctrl (`\|S-R\|<=1.6`) | `10.149` | `-0.334` | `+0.283` | `0.0292` | `0.0127` | — |
| **`L11`** | `CS.3` (`7`) | **`L11H9`** | cand_self (`+`, `+`) | `10.046` | `+10.451` | `+0.053` | `0.0165` | `0.0069` | **`L11H1`** (`0.86×` DLA; closest neutral `copy_R` `+0.274`; no closer neutral in L11) |
| `L11` | | **`L11H2`** | cand_self (`+`, `+`; `copy_R` sat.) | `10.267` | `+6.126` | `-0.980` | `0.0390` | `0.0277` | **`L11H6`** (`1.02×` DLA; `copy_R` bracketed) + **`L11H4`** (`copy_R=-0.830` suppressor match; `3.71×` DLA reported separately) |
| `L11` | | **`L11H3`** | cand_role (`-`, `-`) | `10.155` | `-8.131` | `+0.976` | `0.1129` | `0.0560` | **`L11H10`** (`1.41×` DLA; `copy_R=+0.880` copier match) + **`L11H4`** (`1.28×` DLA; suppressor bracket) |
| `L11` | | `L11H1` | neutral_ctrl (`\|S-R\|<=1.6`) | `8.961` | `-1.361` | `+0.274` | `0.0142` | `0.0092` | — |
| `L11` | | `L11H6` | neutral_ctrl (`\|S-R\|<=1.6`) | `6.961` | `-0.700` | `+0.599` | `0.0398` | `0.0334` | — |
| `L11` | | `L11H4` | neutral_ctrl (`\|S-R\|<=1.6`) | `9.632` | `+1.322` | `-0.830` | `0.1448` | `0.1638` | — |
| `L11` | | `L11H10` | neutral_ctrl (`\|S-R\|<=1.6`) | `8.568` | `+1.273` | `+0.880` | `0.1595` | `0.0324` | — |
| **`L13`** | `CS.3` (`4`) | **`L13H1`** | cand_self (`+`, `+`) | `7.299` | `+13.903` | `+0.700` | `0.0773` | `0.0445` | **`L13H13`** (`1.19×` DLA; copier `copy_R=+0.990`) + **`L13H5`** (`1.60×` DLA; suppressor bracket) |
| `L13` | | **`L13H4`** | cand_role (`-`, `-`) | `8.769` | `-9.990` | `+0.966` | `0.1452` | `0.1174` | **`L13H5`** (`0.85×` DLA; suppressor bracket) + **`L13H13`** (`0.63×` DLA; copier `copy_R=+0.990` match) |
| `L13` | | `L13H5` | neutral_ctrl (`\|S-R\|<=1.6`) | `12.415` | `+0.073` | `-0.990` | `0.1233` | `0.1363` | — |
| `L13` | | `L13H13` | neutral_ctrl (`\|S-R\|<=1.6`) | `10.328` | `-1.515` | `+0.990` | `0.0917` | `0.0657` | — |
| **`L14`** | `CS.3` (`4`) | **`L14H10`** | cand_self (`+`, `+`) | `7.914` | `+19.275` | `-0.462` | `0.0211` | `0.0100` | **`L14H12`** (`1.80×` DLA; `copy_R=-0.431` match; no closer neutral negative-`copy_R` head in L14) |
| `L14` | | **`L14H3`** | cand_role (`-`, `-`) | `6.400` | `-8.082` | `+0.967` | `0.1496` | `0.0978` | **`L14H8`** (`0.64×` DLA; `copy_R=+0.974` match) |
| `L14` | | `L14H12` | neutral_ctrl (`\|S-R\|<=1.6`) | `6.839` | `+0.627` | `-0.431` | `0.0380` | `0.0165` | — |
| `L14` | | `L14H8` | neutral_ctrl (`\|S-R\|<=1.6`) | `10.432` | `-1.099` | `+0.974` | `0.0957` | `0.0374` | — |

### 4.2 Unified `B11` Batch Shape & Intervention
- **Exact batch shape everywhere (`B = 11`, zero reserved rows):** Row `0` unpushed (`0.0`); Row `1` `+0.8 × OLD_NORM[L]`; Row `2` `-0.8 × OLD_NORM[L]`; Rows `3..6` `k = 4` orthogonal `w_down` nulls at `+0.8 × OLD_NORM[L]` (`|cos(null, hat_w)| <= 1e-5`); Rows `7..10` exact negatives of rows `3..6`.
- **Intervention:** `z_all` single-head path-blocking via `o_proj` pre-hook replacing slice `[:, :, h*d:(h+1)*d]` with row-0 unpushed `o_proj` pre-input: every blocked B11 row verifies **row-0 same-hook-path sham (`|d0| <= 1e-6`)** across all 5 readouts and **15-sibling zero-leakage (`<= 1e-6`)** across all 11 rows, PLUS **`26` dedicated full-`B11` measured same-hook-path sham forwards** (one per tested head on `P08` `p92-47` `C1_SELF`, verifying `<= 1e-6` across all 11 pushed+null rows × 5 readouts).
- **Actuation & causal-equivalence positive controls (`29` B=1 forwards):** `26` single-head `B=1` orthogonal synthetic-donor `o_proj` actuation checks (one per tested head on `P08` `p92-47` `C1_SELF`, verifying `min_target_delta > 1e-2` and `sibling_leakage <= 1e-6`) PLUS `3` `B=1` prefix-only causal-equivalence checks (`p92-00` `nc=2`, `p92-45` `nc=4`, `p92-39` `nc=7` on `C1_SELF`, verifying full-vocab `read_pos` logits match single-pass `prefix + content_ids[:-1]` to `<= 1e-5`).

---

## 5. Pre-Registered Gates, Power Disclosure & Disjoint Verdicts (`CS.0`–`CS.4`)

### `CS.0` — Technical Gate (Strict Precedence)
Must pass ALL 6 clauses (`0` failures), else every scientific verdict (`CS.1`–`CS.4`) is forced to `uninformative (cs0_failed)`:
1. External 4-pin verification (`design`, `manifest`, `runner`, `scorer` SHA-256) + pinned `part213/targets213.json` (`7edb3c62...`) + pinned `results_old_b1_b4_transport_feasibility.json` (`dcfa27e9...`) + `execution_provenance == "measured_gpu_run"` + `is_cuda_device is True` + non-CPU GPU `device_name` + complete manifest join.
2. `max_abs_sham_spec <= 1e-6`, `max_abs_sham_raw <= 1e-6`, `max_abs_sham_logp <= 1e-6`, `max_abs_sham_phrase <= 1e-6`, `max_abs_decomp_err <= 1e-6`, `100%` finite values.
3. `max_sibling_leakage_science <= 1e-6` and `max_sibling_leakage_actuation <= 1e-6`.
4. `26/26` synthetic head-actuation controls pass (`min_actuation_target_delta > 1e-2`).
5. `unpatched_rows` contain zero sham keys; `w_down` null orthogonality (`|cos(null, hat_w)| <= 1e-5`) verified on all 26 units.
6. Single-pass causal teacher-forcing equivalence (`prefix_only` vs `prefix + content_ids[:-1]` at `read_pos`) verified `<= 1e-5` on multi-token units during runner pre-flight self-check.

### `CS.1` — Informative Transport & Head-Null Gate (Fixes M2, L6, Weaver `#10` `k=4`)
Evaluated **separately per class** (`slot_12` and `concept_12`) across `C1..C6` (`72` unit-story rows per class × cell):
- **Cell transport pass (enforced pre-full in Smoke AND in Full):**
  - `mean_spec >= +0.50`, `mean_raw >= +0.50`, `mean_logp_delta > 0.0`, AND `mean_spec > max_{j=1..4}(mean_cls_wdown_null_spec_draw[j])` — explicitly **mean across the 72 unit-story rows first, then `max` over the `k = 4` null draws** (L6 & Weaver `#10`).
  - If `mean_spec >= +0.50` while `mean_raw < +0.50` or `mean_logp_delta <= 0.0`, flag `peer_median_only_flip` (fails transport gate).
  - If either class fails transport in `SELF` or `ROLE_BARE` (`NAME_BARE`), the corresponding `C_SR` (`C_SN`) verdict is forced to **`uninformative (transport_gate_failed:<class>|<cell>)`**.
- **Per-class head-null gate with registered consequence (fixes M2 & L6):**
  - For each tested head `h`, class `cls in {slot_12, concept_12}`, and cell, compute `mean_cls` across all `72` rows of each row's median across the `k = 4` `w_down` draws: require `|mean_cls(wdown_even_med_spec)| <= 0.15` AND `|mean_cls(wdown_odd_med_spec)| <= 0.15` (Part B empirical worst across 7 heads × 3 cells was `0.0907`).
  - **Consequence (M2):** If any target/candidate head `h` fails head-null in either class on a cell used by its contrast, that head's `CS.2`/`CS.3`/`CS.4` verdict is forced to **`uninformative (head_null_failed:<head>|<class>|<cell>)`**.
  - **Smoke preliminary check:** `--verify-smoke-only` also checks `|wdown_even_med_spec| <= 0.30` and `|wdown_odd_med_spec| <= 0.30` on Smoke's `156` blocked rows (`P08` × `C1` × `26` heads) as an early warning before Full.

### `CS.2` — Primary Concept-vs-Slot Head-Gating Interaction (`L8H2` primary; `L8H15` legacy)
Fixes round-2 **H1, H2, H3, H4, M1, M3, L7** + Weaver `#10`:

1. **Paired contrast, exact inference & honest power disclosure (fixes H3):**
   - For each primary pair `p in {P01..P12}` and story `s in {C1..C6}` (`72` paired cells), `Δ_class(p, s, h) = C_SR(concept_p, s, h) - C_SR(slot_p, s, h)` and pair cluster mean `Δ_p = mean_{s in C1..C6} Δ_class(p, s, h)`.
   - Primary 95% CI `[lo, hi]` uses **studentized pair-cluster bootstrap (`t`-percentile over the 12 pair clusters with nested within-pair story resampling, 10,000 deterministic draws; simulated coverage `0.94–0.97`)**, reported alongside exact paired `t_{11}` 95% CI and all `12` exact per-pair means `Δ_1..Δ_12`.
   - **Equivalence margin `ε = 0.35` logits (`0.35` nats on `TargetLogProb`/`PhraseLogProbMean`) & full power disclosure (H3):**
     - From Part B `L8H2` (`sd_u=0.1916, sd_e=0.3175` raw; `sd_u=0.0, sd_e=0.3405` spec), expected paired 95% CI half-width at `P=12, S=6` is `0.208` raw (`0.168` at `ρ=0.5`), `0.125` spec (and `C_SN`: `0.234` raw / `0.161` spec).
     - **Uncertainty & unmeasured concept-side variance caveat:** Because `sd_u` comes from `4` df (`5` Part B slot units), its 95% χ² interval `[0.1148, 0.5506]` implies half-widths `[0.156, 0.508]`, and concept-side `sd_u` is **unmeasured** prior to Full (Smoke measures concept unpatched transport spread on `C1..C6` without peeking at `L8H2` blocking). Under `5,000`-run simulation at true `Δ = 0`, `ε = 0.25` gave only `37.5%` equivalence power (`16.4%` at `sd_c = 1.5×`, `6.0%` at `2×`); widening to **`ε = 0.35`** (still `< 70%` of Part B's `+0.513` raw `L8H2` effect) raises equivalence power at true `Δ = 0` to **`83.9%`** (`sd_c = 1×`), **`62.1%`** (`1.5×`), and **`35.5%`** (`2×`), while explicitly noting that if realized concept `sd_u` approaches `2×` slot `sd_u`, `inconclusive_wide_interval` remains common.
2. **Interaction-level control gate (Difference-in-Differences, fixes H1!):**
   - Let `L8_CTRLS = {L8H3, L8H13, L8H7, L8H9, L8H0}` (all 5 registered `L8` controls).
   - For the interaction `Δ_bar(h, ch)`:
     - `ctrl_min_delta(ch) := min_{c in L8_CTRLS} Δ_bar(c, ch)` (strongest control interaction in the slot-stronger `< 0` direction).
     - `ctrl_max_delta(ch) := max_{c in L8_CTRLS} Δ_bar(c, ch)` (strongest control interaction in the concept-stronger `> 0` direction).
   - **Directional interaction control tests:**
     - `beats_ctrl_slot_interaction(ch)`: `Δ_bar(target, ch) <= ctrl_min_delta(ch) - 0.15` **AND** `mean_slot C_SR(target, ch) >= +0.25`.
     - `beats_ctrl_concept_interaction(ch)`: `Δ_bar(target, ch) >= ctrl_max_delta(ch) + 0.15` **AND** `mean_concept C_SR(target, ch) >= +0.25`.
     - *(Conservatism note per Claude `#14`: `min`/`max` over `5` control interactions picks the most extreme control by construction, so this DiD gate is conservative in the safe direction; if a directional branch narrowly misses on this gate alone, the report explicitly discloses that alongside all 5 individual control interactions.)*
   - **Shared-level control test (for Branch 3 equivalence only):**
     - `ctrl_max_cls(ch) := max_{c in L8_CTRLS} mean_cls C_SR(c, ch)`.
     - `beats_ctrl_shared_level(ch)`: for BOTH `cls in {slot, concept}`, `mean_cls C_SR(target, ch) >= +0.25` **AND** `mean_cls C_SR(target, ch) > ctrl_max_cls(ch)` **AND** length-matched `mean_cls C_SN(target, ch) - ctrl_max_cls(C_SN, ch) >= +0.10`. *(Explicitly noting per Part B prior: if `L8H3` ties `L8H2` on `C_SR` as on Part B [`+0.3728` vs `+0.3642` spec], `beats_ctrl_shared_level` fails and routes cleanly to Branch 4/5! Per-class level excess `mean_cls C_SR(target) - ctrl_max_cls` is always reported as a descriptive row.)*
3. **Peer-median collapse guard & length-matched `C_SN` concordance:**
   - `peer_med_share(cls) = |mean_cls C_SR(PeerMed)| / max(|mean_cls C_SR(SpecLogit)|, 1e-9)`. If `SpecLogit` clears a positive branch while `RawLogit < +0.25` and `peer_med_share(cls) > 0.50`, flag `peer_median_only_gating`.
   - `csn_concordant(branch, ch)`:
     - Branch 1 (`slot_stronger_gating`): `Δ_bar(C_SN, ch) < 0` AND `mean_slot C_SN(target, ch) > 0` AND `mean_slot C_SN(target, ch) > ctrl_max_slot(C_SN, ch)`.
     - Branch 2 (`concept_stronger_gating`): `Δ_bar(C_SN, ch) > 0` AND `mean_concept C_SN(target, ch) > 0` AND `mean_concept C_SN(target, ch) > ctrl_max_concept(C_SN, ch)`.
     - Branch 3 (`practically_equivalent_subject_gating`): BOTH classes have `mean_cls C_SN(target, ch) > 0` AND `> ctrl_max_cls(C_SN, ch)`.
     - Failure on a positive `C_SR` branch downgrades explicitly to **`length_confound_unconfirmed`**.
4. **Active `persona_kind` confound guard (fixes H4!):**
   - Compute BOTH: (a) the exact `5`-pair same-`(layer, persona_kind)` mean `Δ_same_kind` (`P02, P03, P05, P08, P11`), and (b) the **parsimonious within-kind model** across the `23` primary units in the `5` shared `persona_kind` strata (excluding singleton `another AI` unit `p92-38` from this model only): `C_SR(u) ~ 1 + is_concept + C(persona_kind) + base_raw(u)` (`7` parameters, `16` residual df; HC3 SE), giving within-kind coefficient `β_within_kind`.
   - Any directional Branch 1 (`Δ < 0`) or Branch 2 (`Δ > 0`) verdict requires **sign agreement** with BOTH `Δ_same_kind` AND `β_within_kind`; otherwise downgrade explicitly to **`inconclusive_kind_confounded`** (reporting all three side-by-side).
5. **Active transport-scale confound guard & Weaver's within-class regression (fixes M3 & Weaver `#10` #2!):**
   - Compute: (a) unit-story fractional gating `F_SR(u, s, h) = (drop_+0.8(SELF) / T_SELF) - (drop_+0.8(ROLE_BARE) / T_ROLE)` on cells with `T_cell >= +0.50`, and class ratio-of-means `F_SR_ratio(cls) = mean_cls C_SR / (0.5 * (mean_cls T_SELF + mean_cls T_ROLE))`, giving `Δ_F_ratio = F_SR_ratio(concept) - F_SR_ratio(slot)`; AND (b) **Weaver's within-class transport regression**: `C_SR(u) ~ 1 + mean_raw_SR(u)` separately within Slot (`n=12`) and Concept (`n=12`) (reporting slopes `β_T_slot, β_T_concept`), plus joint ANCOVA `C_SR(u) ~ 1 + is_concept + mean_raw_SR(u)` giving transport-adjusted class intercept `β_class_adj`.
   - Any directional Branch 1 or Branch 2 verdict requires **sign agreement** with BOTH `Δ_F_ratio` AND `β_class_adj`; otherwise downgrade explicitly to **`transport_scale_confounded`** (reporting raw `Δ_bar`, `Δ_F_ratio`, within-class slopes, and `β_class_adj`).
6. **Strict ordered disjoint per-channel verdict tree (evaluated in exact order `1 -> 2 -> 3 -> 4 -> 5`; fixes H1, H2, H4, M3):**
   - **1. `slot_stronger_gating`:** `hi < -0.35`, `>= 9/12` pairs have `Δ_p < 0`, `beats_ctrl_slot_interaction` holds, AND passes `csn_concordant` (`else length_confound_unconfirmed`), kind guard (`else inconclusive_kind_confounded`), and transport guard (`else transport_scale_confounded`).
   - **2. `concept_stronger_gating`:** `lo > +0.35`, `>= 9/12` pairs have `Δ_p > 0`, `beats_ctrl_concept_interaction` holds, AND passes `csn_concordant`, kind guard, and transport guard.
   - **3. `practically_equivalent_subject_gating`:** `lo >= -0.35` AND `hi <= +0.35`, `beats_ctrl_shared_level` holds, AND `csn_concordant` holds (`else length_confound_unconfirmed`). *(Mandatory report wording: `"any class difference is smaller than 0.35 logits (about 68% of Part B's L8H2 C_SR)"`, never `"no class difference"` or `"the same gating"`.)*
   - **4. `both_weak_or_null` (fixes H2!):** NEITHER Branch 1, 2, nor 3 passes, **AND `lo >= -0.35` AND `hi <= +0.35`** (equivalence on the estimand `Δ_bar` itself — prevents declaring null from `Δ = -0.50`!), **AND** for BOTH classes the 95% cluster CI upper bound of `(mean_cls C_SR(target) - ctrl_max_cls)` is `<= +0.35`. *(Mandatory report wording: `"any class difference is smaller than 0.35 logits (about 68% of Part B's L8H2 C_SR)"`, never `"no class difference"` or `"the same gating"`.)*
   - **5. `inconclusive_wide_interval`:** Otherwise.
7. **Co-primary 3-readout agreement rule & descriptive sensitivities (fixes M1, L2, L3, L7):**
   - Run the exact tree independently on **`RawLogit`** (designated anchor), **`SpecLogit`**, and **`TargetLogProb`**. Overall `CS.2` verdict = that common verdict **iff all three channels agree identically**, else **`inconclusive_readout_disagreement`** (reporting all three verdicts + `PhraseLogProbMean` side-by-side).
   - **Descriptive sensitivities (frozen rules, explicitly non-gating):**
     - **(a) Full-phrase readout:** `PhraseLogProbMean` across all 12 pairs (same `t`-bootstrap).
     - **(b) Frozen clean-token exclusion (M1, L2):** Excludes primary-panel subword-first units (`targets213.json["is_full_word_first_token"] == False`: `p92-00, p92-40`; noting `p92-50` is the third subword unit in `targets213.json`, outside our 24-unit panel) PLUS frozen modifier-first list (`p92-04 ' very', p92-39 ' most', p92-36 ' real', p92-15 ' powerful', p92-37 ' large', p92-46 ' regular', p92-41 ' human'`). Analyzed explicitly two ways: (i) **`5` intact clean pairs (`P01, P03, P05, P06, P08`)** with exact paired `t_4` CI and pair-mean vector, and (ii) **unpaired surviving `15` units (`8` slot vs `7` concept)** with Welch two-sample `t` CI.
     - **(c) Baseline-logit diagnostic (L7):** Report `corr(C_SR, base_raw)` pooled and within-class, plus within-unit baseline-adjusted slope from `C_SR ~ is_concept + base_raw`.
     - **(d) Reader-agreed concept stratum (`7` pairs: `P01, P04, P06, P07, P08, P09`, plus `L6_ra`: `p92-43 vs p92-09`, reporting `PhraseLogProbMean` alongside `RawLogit`, L3)**, **Part B 5-core-slot bridge**, and **Exact-token cross-layer witness** (`p92-08` `L4` vs `p92-09` `L6`, both `' dog'`).

### `CS.3` — Held-Out Confirmation of `7` Late Lens-Screened Heads (`L11, L13, L14`)
Fixes round-2 **H5, M4** + Weaver `#10` (`L11H2` saturation):
- **Exact test count = `7` (`α = 0.05 / 7 = 0.00714`, two-sided `99.29%` studentized unit-cluster CI over all 24 primary units pooled):** ONE contrast (`C_SR`) × ONE pooled panel per candidate (`7` tests total; zero OR).
- **Confirmation rule (`confirmed_late_gate`):**
  1. Pooled `C_SR` matches the frozen predicted sign (`+`: `L11H9, L11H2, L13H1, L14H10`; `-`: `L11H3, L13H4, L14H3`) across `RawLogit`, `SpecLogit`, and `TargetLogProb`,
  2. `|C_SR| >= 0.25` in BOTH `RawLogit` and `SpecLogit` (and `peer_med_share <= 0.50`),
  3. Exceeds `max |C_SR|` of its **named primary control(s) in §4.1** by **`>= +0.15`** in BOTH `RawLogit` and `SpecLogit` (also reporting the layer-wide neutral control `max` and `rmsDLA` ratio beside every verdict!),
  4. Two-sided `99.29%` unit-cluster CI excludes `0` in BOTH `RawLogit` and `SpecLogit`,
  5. Length-matched pooled **`C_SN`** matches the **frozen predicted `C_SN` sign in §4.1** (`+` for `L11H9, L11H2, L13H1, L14H10`; `-` for `L11H3, L13H4, L14H3`; else **`confirmed_length_confounded`**, M4).
- Per-class Concept/Slot breakdowns and `Δ_class` are strictly secondary/descriptive; **`L11H2`** carries an explicit saturation flag (`copy_R = -0.980`) and is not allowed to establish a class claim alone.

### `CS.4` — `L7H5` Signed Antisymmetry Diagnostic (`ROLE_BARE`; Fixes M5 & M6)
- Register honestly as a **directional class-vs-unit diagnostic** (grounded in Part B's prior: `L7H5` `ROLE_BARE` slot mean `+0.1049` raw / `+0.3286` spec, only `1/5` units `>= +0.30` [`p92-09` `+0.47`], beating `max(L7H4, L7H9)` by `+0.200` raw).
- Evaluated per class (`slot_12`, `concept_12`) independently on `RawLogit` and `SpecLogit`:
  - Per-channel **`class_antisymmetry_present`** iff `mean_cls S_odd(L7H5) >= +0.20`, `mean_cls S_odd(L7H5) - max(L7H13, L7H4, L7H9) >= +0.15`, AND **`>= 4/12` (`33%`)** primary units clear `S_odd >= +0.30`; else **`not_class_wide`**.
  - **Co-primary channel agreement rule (M6):** Overall class verdict requires identical verdicts on `RawLogit` and `SpecLogit`, else **`inconclusive_readout_disagreement`** (reporting both channels side-by-side).
  - Separately reports the `L6` units (`p92-09` vs `p92-40`/`p92-43`) and exact-token witness (`p92-08` `L4` vs `p92-09` `L6`) as unit diagnostics.

---

## 6. Execution Schedule, Smoke Gate & Grounded Compute Budget (Fixes M2 & L5)

- **Exact Full Schedule (`12,662` B11 + `29` B=1 forwards; zero extra continuation forwards):**
  - `26` units × `18` prompts (`C1..C6` × 3 cells) = `468` unpatched B11 forwards + `468 × 26 = 12,168` `z_all` head-blocked B11 forwards + `26` dedicated full-B11 measured sham forwards (1 per head) + `26` B=1 synthetic actuation forwards + `3` B=1 causal-equivalence prefix-only forwards = **`12,662` B11 + `29` B=1 forwards (`12,691` total)**.
  - Projected wall time on 1 ephemeral Colab `NVIDIA L4` (grounded in Part B's measured `132.9 ms/B19` and `1.72 CU/h` rate → `B11 ≈ 78–85 ms/fwd`): **`~16–18 min` runner / `~19–21 min` session → `~0.50–0.65 CU`** (conservative cap `<= 0.90 CU`, well within Elise's standing `~5 CU/day` small-run authorization; current balance `185.72 CU`; exact `ms/B11` measured during Smoke before Full launches).
- **Mandatory Smoke Gate & Pre-Full `CS.1` Transport Pre-Check (`650` B11 + `29` B=1 forwards; `~56 s`, `<= 0.15 CU`):**
  - Runs **all `26` units × `18` prompts unpatched-only** (`468` B11) + Pair `P08` (`p92-47` + `p92-45`) on `C1` (`3` cells) × all `26` heads (`156` B11) + `26` dedicated full-B11 measured sham forwards + `26` B=1 synthetic actuation checks + `3` B=1 causal-equivalence checks (`679` total forwards).
  - Scorer `--verify-smoke-only` must pass (a) all `CS.0` technical clauses (`0` failures), (b) the full `12`-pair `C1..C6` unpatched `CS.1` transport gate for BOTH classes (also reporting unpatched transport spread per class without peeking at `L8H2` blocking!), and (c) the preliminary head-null sanity check (`|med| <= 0.30`) on the `156` blocked rows — while emitting **ZERO** head-gating or scientific verdicts (`CS.2`/`CS.3`/`CS.4`). Full execution hard-refuses to launch without `--verify-smoke-only` PASS; full scorer hard-refuses any smoke receipt (`rc != 0`).
- **Offline Scoring & Review:** 4-file frozen package (`CONCEPT-VS-SLOT-DESIGN.md`, `concept_vs_slot_manifest.json`, `run_concept_vs_slot.py`, `score_concept_vs_slot.py`) with CPU `TinyOlmo2ForCausalLM` end-to-end self-tests (including single-pass causal phrase equivalence), routed to Claude for Design PASS → Code PASS → Smoke PASS → Full Run.

---

## 7. Pre-Data Forecast (Frozen Before Code & Execution; Claude `#14` Condition 2)

Registered by Vesper prior to any `C1..C6` model execution so outcomes can be evaluated against an explicit, calibrated prior:

### 7.1 `CS.2` (`L8H2` Overall Co-Primary Verdict Distribution — Sums to `100%`)
Because clean directional confirmation requires simultaneous agreement across all 3 readouts (`RawLogit`, `SpecLogit`, `TargetLogProb`), `>= 9/12` pairs, the conservative `min`/`max` 5-control DiD gate, `C_SN` concordance, both within-kind checks (`5` same-kind pairs + `n=23` OLS), and both transport guards (`Δ_F_ratio` + Weaver ANCOVA `β_class_adj`) — while Part B's prior already shows `L8H3` matching `L8H2` on `SpecLogit` `C_SR` and `OLD B1..B4` shows Concept transport `1.41–2.07×` larger than Slot — most probability mass legitimately sits on `inconclusive` / downgrade labels:

| Overall `CS.2` (`L8H2`) Outcome | Frozen Prior Prob | Reasoning |
|---|---:|---|
| `inconclusive_wide_interval` | **28%** | Modest true `Δ_bar` in `[-0.45, +0.15]` straddling `±0.35`, higher concept `sd_u`, or narrow miss on conservative 5-control DiD. |
| `inconclusive_readout_disagreement` | **24%** | Strict 3-channel identity (`RawLogit`, `SpecLogit`, `TargetLogProb`) under `1.41–2.07×` baseline/transport asymmetry and Part B's `L8H3`/`L8H13` raw-vs-spec splits. |
| `both_weak_or_null` *(worded: `< 0.35` logits)* | **18%** | `L8H2` shrinks on the broader 12-pair panel or `L8H3` keeps pace while `Δ_bar` CI stays inside `[-0.35, +0.35]`. |
| `slot_stronger_gating` | **10%** | Clean, large `Δ_bar < -0.35` surviving all 3 channels + 5-control DiD + `C_SN` + kind + transport guards. |
| `transport_scale_confounded` | **8%** | Raw `Δ_bar` clears a directional branch while `Δ_F_ratio` or Weaver ANCOVA `β_class_adj` flips sign due to the `1.41–2.07×` Concept transport advantage. |
| `inconclusive_kind_confounded` | **5%** | Small-`n` (`5` same-kind pairs) sign flip against the 12-pair `Δ_bar`. |
| `length_confound_unconfirmed` | **3%** | `C_SR` clears a positive branch via the 3-token `ROLE_BARE` length gap while length-matched `C_SN` fails. |
| `practically_equivalent_subject_gating` *(worded: `< 0.35` logits)* | **2%** | Low specifically because Part B `L8H3` already tied `L8H2` on `SpecLogit` `C_SR`. |
| `concept_stronger_gating` | **1%** | Unexpected opposite-sign large clean effect surviving kind and transport guards. |
| `uninformative` (`cs0` / `transport` / `head_null` failed) | **1%** | Smoke pre-gates `CS.0` + `CS.1` transport; residual Full head-null risk only. |
| **Total** | **100%** | |

- **Un-gated anchor expectation (`RawLogit` `L8H2`):** Modest negative `Δ_bar` around `-0.15` to `-0.35` logits (Slot slightly more gated than Concept in raw logits, with fractional `Δ_F_ratio` more cleanly negative than raw `Δ_bar` because Concept transport is `~1.7×` larger).

### 7.2 `CS.3` (Per-Head Confirmation Forecast Across the `7` Late Lens-Screened Heads)
Each probability below is the pre-data probability of **`confirmed_late_gate`** (Bonferroni `99.29%` CI excluding `0`, `|C_SR| >= 0.25`, beating named DLA-matched primary control by `>= +0.15` on both `RawLogit` and `SpecLogit`, matching predicted sign across 3 channels AND length-matched `C_SN`):

| Head | Pred `(C_SR, C_SN)` | `OLD rmsDLA` | `P(confirmed_late_gate)` | `P(confirmed_length_confounded)` | `P(not_confirmed / other)` | Notes |
|---|---|---:|---:|---:|---:|---|
| **`L11H9`** | `(+, +)` | `0.0165` | **15%** | 10% | 75% | Very low unpushed DLA; hard to clear `|C_SR| >= 0.25` via single-head blocking. |
| **`L11H2`** | `(+, +)` | `0.0390` | **20%** | 15% | 65% | Saturated global suppressor (`copy_R=-0.980`); primary ctrl `L11H6` (`1.02×` DLA), `L11H4` reported separately. |
| **`L11H3`** | `(-, -)` | `0.1129` | **25%** | 20% | 55% | High DLA + strong role-copier (`copy_R=+0.976`), but `ROLE_BARE` length gap creates real `C_SN` risk. |
| **`L13H1`** | `(+, +)` | `0.0773` | **35%** | 15% | 50% | Highest `+` candidate prior: strong `S-R=+13.90`, solid DLA, late layer near `L14` readout peak. |
| **`L13H4`** | `(-, -)` | `0.1452` | **30%** | 20% | 50% | High DLA (`0.1452`) + strong `S-R=-9.99`, moderated by `C_SN` length-confound risk. |
| **`L14H10`** | `(+, +)` | `0.0211` | **25%** | 10% | 65% | Largest `S-R` in model (`+19.28`) at `L14`, tempered by low unpushed DLA (`0.0211`). |
| **`L14H3`** | `(-, -)` | `0.1496` | **30%** | 15% | 55% | Highest DLA (`0.1496`) of all 7 candidates at `L14`, paired with `copy_R`-matched `L14H8`. |

- **Portfolio expectation (`CS.3`):** Expected **`~1.8 / 7`** confirmed late gates (`~55%` chance of `1–3` confirming, `~30%` chance of `0`, `~15%` chance of `>= 4`), with `L13H1`, `L13H4`, and `L14H3` most likely to clear `|C_SR| >= 0.25`.
- **`CS.4` (`L7H5` `ROLE_BARE` `S_odd`):** **`~25%`** `class_antisymmetry_present` in Slot (Part B was `p92-09`-dominated), **`~15%`** in Concept, majority `not_class_wide` or `inconclusive_readout_disagreement`.
