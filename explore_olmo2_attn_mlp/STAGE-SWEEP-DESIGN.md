# OLMo 2 1B Stage-Sweep, Same-Referent Alias Counterbalance & Per-Stage Dose-Calibration Design — v2.1

**Status:** PRE-DATA v2.1 (2026-11-20 CT, Vesper / Gemini 4 Neon) — keeps all v2 pins/prompts/rules (`1c8de35b`, independently verified across all 28 prompts & 140 bindings in Alden `#62`) intact and folds both Alden `#62` outcome-contract edits (`SS.2(b)` explicit fail-reason separation + declared `G_alias - G_bare` attenuation comparison; `SS.7` matched `E1/E2` transport/gate parity + explicit `plain_positive_only` & gate dispositions + both witnesses in §7). All predictions (`SS.1`–`SS.7`) remain strictly pre-data.
1. **Completion-boundary tokenizer verification (v2):** Replaced all multi-token names/roles with tokenizer-verified 1-token proper names (`Rachel`, `Simon`, `Laura`, `Henry`), 2-token `The <role>` (`The teacher`, `The farmer`, `The tailor`, `The sailor`), and 2-token `CAL` generics (`The newcomer`, `The outsider`) verified offline on the exact pinned Instruct tokenizer; froze exact rendered prompt bytes, `read_pos`, `lead`, `t_own`, 19 `peers`, and remaining token-length differences.
2. **Strict Arm B pre-rebound estimand, tie-break, gate precedence & cell-wise transport control (v2):** Unbracketed monotone calibration curves set Arm B `d_cal(s) = None` (`uninformative_no_rebound_bracket`), keeping `d_best_observed` strictly descriptive; separated candidate eligibility (`G_valid`) from the `CAL_THIRD`-only selection objective; upgraded `SS.1(c)` to **cell-wise** (`CAL_SELF` and `CAL_THIRD` individually) final-logit transport positive control bound to each arm's exact dose.
3. **Total deterministic decision maps & two-factor finite-bank `SS.5` (v2/v2.1):** Eliminated every unassigned outcome region across `SS.1`–`SS.7` (including explicit `SS.2(b)` fail reasons separate from `G_alias - G_bare` attenuation and exact `16`-state gate + `4`-way valid partition in `SS.7`); replaced the invalid 20-pair iid bootstrap in `SS.5` with a two-factor finite-bank (`5` units × `4` stories) consistency rule; added the explicit checkpoint-specific intervention caveat.
4. **Narrowed claims & fixed layer label (v2/v2.1):** Reserved *"attenuation"* strictly for the declared `G_alias - G_bare` comparison under this same-referent manipulation (acknowledging both question & completion cues change and appositives affect salience); pre-declared the exact token-level peer sign reducer; corrected `p92-46` to **L5** and relabeled `SS.6` to **`L5/L6 late-write vs. L3 early-write`** odd-dominant / antisymmetric readout contrast (keeping L6-only `p92-09` descriptive).

**Immutability contract:** Before any GPU run, this design file, the runner, the offline scorer, and the rendered manifest will be pinned together by 4 external SHA-256 digests on local branch `vesper-stage-sweep` (`part212` untouched), enforced by the runner before loading any weights and verified by the offline scorer. No model weights have been loaded and no spend is authorized until pre-data PASS.

---

## 1. Pinned 4-Checkpoint Lineage & Interpretive Boundaries

All four checkpoints are loaded in `torch.float32`, `attn_implementation="eager"` at immutable 40-char Hugging Face commit SHAs (code structurally refuses `main`, `a1847dff`, or any non-40-char SHA):

| Stage ID | Role in Lineage | HF Repository | Immutable 40-char Commit SHA |
|---|---|---|---|
| `midtrain_end` | Mid-training end (SFT parent, Part 2 lineage 0.996; `stage2-ingredient3-step23852-tokens51B`) | `allenai/OLMo-2-0425-1B` | `f83cd641d4f00954eb25c781e0ac360debb3239e` |
| `sft` | Supervised Fine-Tuning | `allenai/OLMo-2-0425-1B-SFT` | `0d85a3d037876ce6ac7d4311d994400fc66ac27f` |
| `dpo` | Direct Preference Optimization | `allenai/OLMo-2-0425-1B-DPO` | `c4b0485961ab24c2433b090f3b922f0913a9290f` |
| `instruct` | Final Instruct (post-RLVR per Ai2 1B release notes) | `allenai/OLMo-2-0425-1B-Instruct` | `48d788eca847d4d7548f375ad03d3c9312f6139e` |

**Three interpretive boundaries traveling with every reported result (not footnotes — Alden `#51`, `#58`; Weaver `#53`):**
1. **Single training lineage:** Sequential endpoints on one training trajectory (`midtrain_end -> sft -> dpo -> RLVR/instruct`), not independent replications or a dataset-level causal proof.
2. **Same-coordinate tracking rule:** Tracking `L8H15` (primary) and `L7H5` (secondary) at fixed coordinates across stages is a tracking rule, not proof that the same function occupies those coordinates early in training. A null at `midtrain_end` means **"nothing at these coordinates,"** not **"absent from the model."** Stage-blind head discovery is intentionally deferred out of v2 per Alden `#51`.
3. **Checkpoint-specific intervention result (Alden `#58`):** Any surviving stage interaction (`SS.3`/`SS.4`) is a **checkpoint-specific intervention result**, not proof that `L8H15`'s own weight parameters changed — because each stage's write vector `v_{s, u}`, residual geometry, and upstream/downstream network parameters also vary across checkpoints.

---

## 2. Units, Heads, Frozen Normalization & Decomposed Readouts

### 2.1 Units & Strictly Downstream Heads (`35` eligible `(unit, head)` pairs per stage)
- **Units (`5` core Part 92 slot units, all strictly upstream of `L7+`):**
  - **L3 early-write (`3` units):** `p92-02` (`L3 N2628`, `lead="the"`, `content="town"`), `p92-05` (`L3 N6986`, `lead="a"`, `content="fox"`), `p92-13` (`L3 N976`, `lead="a"`, `content="skeleton"`)
  - **L5/L6 late-write (`2` units):** `p92-46` (`L5 N4673`, `lead="a"`, `content="regular guy"`), `p92-09` (`L6 N4145`, `lead="a"`, `content="dog"`)
  - *Unit vector per stage `s`:* Stage `s`'s own normalized column `hat_w_{s, u} = W_down^{(s, L_u)}[:, n_u] / ||W_down^{(s, L_u)}[:, n_u]||_2`. Persist `cos(hat_w_{s, u}, hat_w_{instruct, u})` and `||W_down^{(s, L_u)}[:, n_u]||_2` for every `(s, u)`.
- **Heads (`7` total, all `L_h >= 7 > L_u`, so all 5 units are strictly eligible on every head — `5 × 7 = 35` pairs):**
  - **Primary head:** `L8H15`
  - **Secondary head:** `L7H5` (evaluating the 5-unit pool AND the pre-registered `L5/L6 late-write` [`p92-46`, `p92-09`] vs. `L3 early-write` [`p92-02`, `p92-05`, `p92-13`] split; L6-only `p92-09` is reported descriptively since `n = 1`).
  - **Matched-layer random controls (`5` total, disjoint from prior key heads):**
    - `L8` (`3` heads, matching `#44` for continuity): `L8H2`, `L8H3`, `L8H13`
    - `L7` (`2` heads): `L7H4`, `L7H9`

### 2.2 Frozen Normalization Rule & Norm Logging
- At each checkpoint `s`, compute layer `L_u`'s unpushed residual L2 norm median `m_{s, L_u}` across all token positions of the unpushed forward pass (`d = 0.0`) on the **4 disjoint `CAL` prompts**.
- Define `v_{s, u} = m_{s, L_u} * hat_w_{s, u}`. At signed dose `d`, inject `+d * v_{s, u}` into the cloned decoder-layer `L_u` output at all token positions.
- Persist for every `(stage, unit, prompt, dose)`: `m_{s, L_u}`, actual injected L2 norm `|d| * m_{s, L_u}`, unpushed `read_pos` residual L2 norm `||h_{s, L_u}(read_pos)||_2`, and their ratio.

### 2.3 Verified Prefix-Attached Own & Peer Bindings (`100%` Identical Across All 28 Prompts)
Verified offline on `allenai/OLMo-2-0425-1B-Instruct@48d788eca847d4d7548f375ad03d3c9312f6139e` tokenizer (zero weights loaded). At manifest generation, all four checkpoints' tokenizers are verified to share identical vocabulary size and identical `encode(prompt)` token IDs on all 28 prompts (hard-refuse on any mismatch).

- For every prompt `p` and core unit `u`, the full prompt string is `{base_prefix} {lead_u}` (every core unit has a 1-token `lead`: `" the"` for `p92-02`, `" a"` for `p92-05/13/46/09`), ending at `read_pos = len(encode(p_u)) - 1`. Zero BPE merges occur across `\n` or across `read_pos` for own or any peer token on any of the 28 prompts.
- **Frozen own & 19-peer token IDs (verified 100% identical across all 28 prompts for each core unit):**
  - Peer pool = all `20` distinct non-own Part 92 first content token IDs (`{6424, 8191, 1633, 39935, 8415, 5679, 16553, 30535, 12737, 29006, 6342, 17177, 2999, 47132, 1972, 3544, 4562, 5912, 15779, 1099}`, noting `p92-18` and `p92-20` share `' shadow'` `12737`, giving `20` distinct slot tokens total and **`n_peers_distinct = 19`** per core unit!):
    - `p92-02`: `t_own = 6424 (' town')`, `peers = pool \ {6424}` (`19` tokens)
    - `p92-05`: `t_own = 39935 (' fox')`, `peers = pool \ {39935}` (`19` tokens)
    - `p92-13`: `t_own = 30535 (' skeleton')`, `peers = pool \ {30535}` (`19` tokens)
    - `p92-46`: `t_own = 5912 (' regular')`, `peers = pool \ {5912}` (`19` tokens)
    - `p92-09`: `t_own = 5679 (' dog')`, `peers = pool \ {5679}` (`19` tokens)
- **Four readouts persisted on EVERY row (full IEEE-754 floats):**
  1. `RawLogit` (`raw`): `logits[read_pos, t_own(u)]`
  2. `PeerMedian` (`peer_med`): `median_{t in peers(u)} logits[read_pos, t]`
  3. `SpecLogit` (`spec`): `RawLogit - PeerMedian` (`D_spec == D_raw - D_peer_med` exact identity)
  4. **Individual peer-token shifts (`peer_deltas`):** On every `EVAL` row, persist `D_peer_t = logit_unpatched[t] - logit_patched[t]` for all 19 `t ∈ peers(u)` (exact token ID keyed dict), plus `peer_mean` and `peer_iqr`.

### 2.4 Uniform `B = 19` Same-Dose Patching, Sham & Signed Null Protocol
Unchanged from `#44` (`max_abs_sham = 0.0` across `6,480` tuples):
- Row `0`: unpushed (`d = 0.0`); Row `1`: `+d * v_{s, u}`; Row `2`: `-d * v_{s, u}`;
- Rows `3..6` / `7..10`: `+d` / `-d` along the **exact same 4** `w_down`-orthogonal vectors (`n_u` zeroed before Gram-Schmidt against `v_{s, u}`, L2-norm matched to `||v_{s, u}||_2`);
- Rows `11..14` / `15..18`: `+d` / `-d` along the **exact same 4** isotropic orthogonal residual vectors.
- For every eligible head `(L_h, h)`: run `unpushed_z` (donor = Row `0` broadcast to `B = 19`) AND 100%-coverage same-dose `sham_z`.
- `D(d) = R_unpatched(d) - R_patched(d)` computed separately for `spec`, `raw`, `peer_med`, and every `t ∈ peers(u)`; `E(d) = (D(+d) + D(-d))/2 - D(0)` and `O(d) = (D(+d) - D(-d))/2`.

---

## 3. Frozen Rendered Prompt Banks & Verified Token Geometry (`28` Prompts)

All prompts are 100% disjoint from Part 180 v5.1, Part 213, Run 1 / `explore_ladder.py`, and `#44` (`SIGNED-DOSE-DESIGN.md`), enforced by `assert_no_forbidden_overlap()`.

- **Chat serialization (primary, byte-identical across all 4 checkpoints):** `<|endoftext|><|user|>\n{Q}\n<|assistant|>\n{PREFIX} {lead_u}`
- **Plain serialization (`SENS_PLAIN` sensitivity arm):** `{Q}\n{PREFIX} {lead_u}`

### 3.1 Disjoint Calibration Bank (`CAL` — `4` prompts, `C1`–`C2`)
Exact rendered `{Q}` and `{PREFIX}` (plus verified completion tokens after `\n` and `read_pos` shared by all 5 core units):

| ID | Cell | `{Q}` | `{PREFIX}` | Verified Comp Toks | `read_pos` |
|---|---|---|---|---|---:|
| `cal_self_C1` | `CAL_SELF` | `Inside the stone lighthouse at low tide, you wind the brass clock. Identify yourself in one line.` | `I am` | `['I', ' am']` (2) | `34` |
| `cal_third_C1` | `CAL_THIRD` | `Inside the stone lighthouse at low tide, the newcomer winds the brass clock. Identify the newcomer in one line.` | `The newcomer is` | `['The', ' newcomer', ' is']` (3) | `37` |
| `cal_self_C2` | `CAL_SELF` | `Along the narrow orchard fence at noon, you mend the wooden latch. Identify yourself in one line.` | `I am` | `['I', ' am']` (2) | `34` |
| `cal_third_C2` | `CAL_THIRD` | `Along the narrow orchard fence at noon, the outsider mends the wooden latch. Identify the outsider in one line.` | `The outsider is` | `['The', ' outsider', ' is']` (3) | `38` |

### 3.2 Held-Out Evaluation Bank (`EVAL` — `20` prompts: `4` stories `E1`–`E4` × `5` cells)
Every proper name is verified **1 token** at `<|assistant|>\n{Name} is` (`Rachel`: `84978`, `Simon`: `69881`, `Laura`: `77560`, `Henry`: `64463`), and every definite role is verified **2 tokens** at `<|assistant|>\nThe {role} is` (`The teacher`: `[791, 11326]`, `The farmer`: `[791, 37500]`, `The tailor`: `[791, 52056]`, `The sailor`: `[791, 93637]`), 2+2 counterbalanced for appositive order:

| Story | Setting, Verb (2nd/3rd), Object | Name (1 tok) | Role (`The` + 1 tok) | Appositive | `read_pos` (`SELF` / `NAME_BARE` / `ROLE_BARE` / `ALIAS_NAME` / `ALIAS_ROLE`) |
|---|---|---|---|---|---|
| **E1** | `Inside the quiet library before sunrise`, `check`/`checks`, `the brass lamp` | `Rachel` | `the teacher` | `Rachel, the teacher,` | `32` / `32` / `35` / `36` / `38` |
| **E2** | `In the glass greenhouse after rain`, `water`/`waters`, `the tall ferns` | `Simon` | `the farmer` | `Simon, the farmer,` | `33` / `33` / `36` / `37` / `39` |
| **E3** | `At the wooden workshop by dusk`, `press`/`presses`, `the wool coat` | `Laura` | `the tailor` | `the tailor, Laura,` | `32` / `32` / `35` / `36` / `38` |
| **E4** | `On the timber ferry deck at dawn`, `coil`/`coils`, `the heavy rope` | `Henry` | `the sailor` | `the sailor, Henry,` | `33` / `33` / `36` / `37` / `39` |

**Exact cell templates rendered for each story `E1`–`E4`:**
1. `SELF`: `"{Setting}, you {v2} {obj}. Identify yourself in one line."` → `"I am"` (`['I', ' am']`, 2 tok)
2. `ROLE_BARE`: `"{Setting}, the {role} {v3} {obj}. Identify the {role} in one line."` → `"The {role} is"` (`['The', ' {role}', ' is']`, 3 tok)
3. `NAME_BARE`: `"{Setting}, {Name} {v3} {obj}. Identify {Name} in one line."` → `"{Name} is"` (`['{Name}', ' is']`, 2 tok)
4. `ALIAS_TO_ROLE`: `"{Setting}, {Appositive} {v3} {obj}. Identify the {role} in one line."` → `"The {role} is"` (`['The', ' {role}', ' is']`, 3 tok)
5. `ALIAS_TO_NAME`: `"{Setting}, {Appositive} {v3} {obj}. Identify {Name} in one line."` → `"{Name} is"` (`['{Name}', ' is']`, 2 tok)

### 3.3 Plain-Text Sensitivity Arm (`SENS_PLAIN` — `4` prompts)
Exact same `{Q}` and `{PREFIX}` as `eval_self_E1`, `eval_role_bare_E1`, `eval_self_E2`, `eval_role_bare_E2` serialized as `{Q}\n{PREFIX} {lead_u}` (no `<|endoftext|>`, `<|user|>`, or `<|assistant|>` tokens): verified `read_pos` = `21, 24, 22, 25`.

### 3.4 Remaining Token-Geometry Differences (Explicitly Logged, Not Hidden — Alden `#58` Fix 1)
- `SELF` vs. `NAME_BARE`: **100% token-length matched** (`read_pos` identical on all 4 stories: `32, 33, 32, 33`; both 2-tok completion prefixes).
- `ROLE_BARE` vs. `SELF` / `NAME_BARE`: `+3` tokens (`+1` definite article `"the"` in story sentence, `+1` in `"Identify the {role}"`, `+1` in `"The {role} is"`).
- `ALIAS_TO_ROLE` vs. `ALIAS_TO_NAME`: **100% byte-identical story/appositive sentence**; `+2` tokens (`+1` `"the"` in question cue `"Identify the {role}"`, `+1` `"The"` in completion cue `"The {role} is"`). Because both the question cue and the completion cue differ (and the appositive itself can alter referent salience), this manipulation tests sensitivity to the **combined local referring-form cue under a fixed referent**, and any attenuation is reported strictly as attenuation under this manipulation (not proof that the bare gap was a pure entity effect).
- `SENS_PLAIN` vs. chat-serialized: `-11` tokens (exact omission of `<|endoftext|><|user|>\n` [6] and `\n<|assistant|>\n` [5]).

---

## 4. Gate Precedence, Per-Stage Dose Calibration (`#7`) & Dual Arms

### 4.1 Strict Gate Precedence (Evaluated at Every Stage `s` and Exact Dose `d`)
Gates are evaluated in strict order at the **exact dose `d` used by each arm**:
1. **Gate 1 — Sham invariance:** `max_abs_sham <= 1e-4` across 100% of `(unit, prompt, head)` tuples at `(s, d)` (both `SpecLogit` and `RawLogit`). Failure → **`invalidated (sham gate failed)`** (nothing downstream at `(s, d)` is read).
2. **Gate 2 — Cell-wise final-logit transport positive control (on `CAL` at exact dose `d`, Alden `#58` Fix 2):**
   - For each calibration cell `c ∈ {CAL_SELF, CAL_THIRD}` (`10` tuples each = `5 units × 2 stories`), compute:
     - `N_write_spec(s, d, c) = mean_{10 tuples}[SpecLogit(+d) - SpecLogit(0)]`
     - `N_write_raw(s, d, c) = mean_{10 tuples}[RawLogit(+d) - RawLogit(0)]`
     - `N_write_null_max(s, d, c) = max_{k=0..3} mean_{10 tuples}[SpecLogit(+d * v_wdown_orth[k]) - SpecLogit(0)]`
   - Cell `c` passes iff `N_write_spec(s, d, c) >= +0.50` AND `N_write_raw(s, d, c) >= +0.50` AND `N_write_spec(s, d, c) > N_write_null_max(s, d, c)`.
   - Stage `s` passes Gate 2 at dose `d` iff **BOTH `CAL_SELF` AND `CAL_THIRD` pass individually** (preventing a pooled average like `(0.0 + 1.1)/2 = 0.55` from hiding a dead cell!).
   - *Interpretive scope:* Gate 2 is a **final-logit transport positive control** confirming that `+d * v_{s, u}` reliably moves the final read-position logits in both cells being contrasted; it does **not** prove where inside the circuit a stage changed.
   - Failure → **`uninformative (final-logit transport failed at dose d)`**.
3. **Gate 3 — Matched-`w_down` null gate (at exact dose `d`):**
   - Require `|mean E_null_wdown| <= 0.15` AND `|mean O_null_wdown| <= 0.15` (on `CAL_THIRD` `L8H15` for calibration eligibility; on each contrasted `EVAL` cell for `L8H15` / `L7H5` when scoring `EVAL`).
   - Failure → **`unreadable (null gate failed)`**.

### 4.2 Locked Per-Stage Dose-Calibration Selection Rule (Arm B Pre-Rebound Estimand)
On the fixed 6-point grid `G = (0.2, 0.4, 0.6, 0.8, 1.2, 1.6)` (strictly ascending) at each stage `s`:
- **Step 1 (Candidate eligibility `G_valid(s)`):** A dose `d ∈ G` enters `G_valid(s)` iff `(s, d)` passes Gate 1 (`CAL` sham), Gate 2 (cell-wise `CAL_SELF` AND `CAL_THIRD` final-logit transport), and Gate 3 (`CAL_THIRD` `L8H15` null). *(Candidate eligibility intentionally checks cell-wise transport on both `CAL_SELF` and `CAL_THIRD`; the selection objective below uses `CAL_THIRD` exclusively.)*
- **Step 2 (`CAL_THIRD`-only objective & deterministic tie-break):** If `G_valid(s)` is empty → `cal_status = "uninformative_gate_failed"`, **`d_cal(s) = None`**, `d_best_observed = None`. Otherwise compute `f_s(d) = mean_{10 CAL_THIRD tuples} D_{L8H15, SpecLogit}(+d)` for each `d ∈ G_valid(s)` (never inspecting `CAL_SELF` head drop or any `EVAL` row). Select `d_min(s)` by scanning `G_valid(s)` in ascending order and updating ONLY on strict inequality `f_s(d) < f_s(d_min)` (exact tie-break: **smallest dose `d`**). Record `d_best_observed = d_min(s)` as a descriptive field.
- **Step 3 (Strict pre-rebound bracket rule — Alden `#58` Fix 2):**
  - If `f_s(d_min) > -0.25`: `cal_status = "uninformative_no_cal_signal"`, **`d_cal(s) = None`**.
  - Else if `∃ d_hi ∈ G_valid(s)` with `d_hi > d_min(s)` AND `f_s(d_hi) - f_s(d_min) >= +0.25`: `cal_status = "rebound_bracketed"`, **`d_cal(s) = d_min(s)`**.
  - Else (monotone / unbracketed through the top of `G_valid(s)`, e.g. `[-0.30, -0.35, -0.40, -0.45, -0.50, -0.55]`): **`cal_status = "uninformative_no_rebound_bracket"`, `d_cal(s) = None`** for Arm B! (The full 6-dose `CAL` grid and `d_best_observed` are retained strictly descriptively; `d_best_observed` is NEVER substituted into Arm B.)

### 4.3 Dual Separate Arms (Non-Substitution Rule — Alden `#54`)
- **Arm A (Common-Dose Arm, ALWAYS run at `d_common = 0.8` on all 4 stages):** compares checkpoints under one identical declared intervention rule (`+0.8 * m_{s, L_u}`), gated at `d = 0.8`.
- **Arm B (Calibrated Pre-Rebound Arm, run at `d = d_cal(s)` ONLY when `cal_status == "rebound_bracketed"`):** reuses Arm A rows when `d_cal(s) == 0.8`; runs a separate `EVAL` pass when `d_cal(s) != 0.8`; marked `uninformative` at any stage where `d_cal(s) is None`.
- Both arms are scored and reported in separate tables. Disagreement between Arm A and Arm B is reported as **dose-dependence of the stage contrast**, never as a reason to prefer one arm.

---

## 5. Compute Budget (Colab L4, `~267.5 ms / B=19 fwd`)

| Block (per checkpoint `s`) | Prompts | Pairs | Doses | `B = 19` Forwards | Est. L4 Time |
|---|---:|---:|---:|---:|---:|
| `CAL` (unpatched `SELF+THIRD` + patched/sham `THIRD`) | 4 / 2 | 35 | 6 | `120 + 840 = 960` | `~4.3 min` |
| `EVAL` Arm A (`d = 0.8`) | 20 | 35 | 1 | `100 + 1,400 = 1,500` | `~6.7 min` |
| `EVAL` Arm B (only if `rebound_bracketed` & `d_cal != 0.8`) | 20 | 35 | 0 or 1 | `0` or `1,500` | `0–6.7 min` |
| `SENS_PLAIN` (`E1, E2 × {SELF, ROLE_BARE}`, `d = 0.8`) | 4 | 35 | 1 | `20 + 280 = 300` | `~1.3 min` |
| **Per-stage total** | — | — | — | **`2,760`–`4,260`** | **`12.3–19.0 min`** |
| **All 4 checkpoints total** | — | — | — | **`11,040`–`17,040`** | **`~50–76 min` (`~1.3–2.0` CUs)** |

---

## 6. Total Deterministic Decision Map (`SS.1`–`SS.7`, Pre-Data)

Every prediction below is an **exhaustive, mutually exclusive partition** of all possible gate states and real-number outcomes (zero unassigned gaps). Every `EVAL` contrast is reported with: grand mean over the `20` `(unit, story)` pairs, all `5` unit means (`M_u`), all `4` story means (`M_e`), median, IQR, and exact `SpecLogit = RawLogit - PeerMedian` decomposition.

### `SS.1` — Per-Stage Gate Vector (Evaluated Separately at Each `(stage s, arm)`)
Returns one of: `invalidated (sham gate failed)` → `uninformative (transport gate failed)` → `unreadable (null gate failed)` → `valid`, strictly in Gate 1 → Gate 2 → Gate 3 precedence order. (For Arm B, if `d_cal(s) is None`, returns `uninformative (<cal_status>)` before `EVAL` checks.)

### `SS.2` — Instruct Bare Replication & Same-Referent Alias Counterbalance (`#3`)
Evaluated on `instruct`, Arm A (`d = 0.8`), `L8H15`, `n = 20` `(unit, story)` pairs/cell. Let `G_bare = mean(D_ROLE_BARE - D_NAME_BARE)`, `G_alias = mean(D_ALIAS_TO_ROLE - D_ALIAS_TO_NAME)`, `R_bare = min_{h in {L8H2,L8H3,L8H13}} mean(D_{h,ROLE_BARE} - D_{h,NAME_BARE})`, `R_alias = min_{h in {L8H2,L8H3,L8H13}} mean(D_{h,ALIAS_TO_ROLE} - D_{h,ALIAS_TO_NAME})` (all in `SpecLogit`):
- **Gate check:** If `SS.1(instruct, Arm A) != valid` → return `SS.1`'s status (`invalidated` / `uninformative` / `unreadable`).
- **`SS.2(a)` Bare replication:**
  - `holds`: `G_bare <= -0.25` AND `G_bare <= R_bare - 0.15`
  - `fails`: `G_bare > -0.15` OR `G_bare > R_bare`
  - `inconclusive`: all remaining cases (`-0.25 < G_bare <= -0.15` with `G_bare <= R_bare`, OR `G_bare <= -0.25` with `R_bare - 0.15 < G_bare <= R_bare`)
- **`SS.2(b)` Same-referent alias test (thresholds unchanged; explicit fail reasons separate from attenuation — Alden `#62` Fix 1):**
  - `holds`: `G_alias <= -0.25` AND `G_alias <= R_alias - 0.15`
  - `fails`: `G_alias > -0.15` OR `G_alias > R_alias`, always emitting one exact `fail_reason`:
    - `alias_effect_weak_and_random_head_failed`: `G_alias > -0.15` AND `G_alias > R_alias`
    - `alias_effect_weak`: `G_alias > -0.15` AND `G_alias <= R_alias`
    - `random_head_comparison_failed`: `G_alias <= -0.15` AND `G_alias > R_alias` *(e.g. witness `G_bare = -0.50, R_bare = -0.25, G_alias = -0.60, R_alias = -0.75`: fails random-head comparison without attenuating relative to `G_bare`!)*
  - `inconclusive`: all remaining cases (`-0.25 < G_alias <= -0.15` with `G_alias <= R_alias`, OR `G_alias <= -0.25` with `R_alias - 0.15 < G_alias <= R_alias`).
  - **Separate declared attenuation comparison (`Δ_alias_minus_bare = G_alias - G_bare`, computed whenever `SS.1(instruct, Arm A) == valid`):** The term *"attenuation"* is reserved exclusively for this comparison (never for random-head failure):
    - `attenuated_under_same_referent_manipulation`: `G_bare <= -0.25` AND `Δ_alias_minus_bare >= +0.15`
    - `preserved_or_stronger_under_same_referent_manipulation`: `G_bare <= -0.25` AND `Δ_alias_minus_bare < +0.15`
    - `not_applicable_bare_below_threshold`: `G_bare > -0.25`
- **`SS.2(c)` Mechanism decomposition & pre-declared token-level peer sign reducer (always computed when `SS.1 == valid`):**
  - Exact identity: `G_alias = Δ_raw - Δ_peer_med`, where `Δ_raw = mean(D_raw(ALIAS_TO_ROLE) - D_raw(ALIAS_TO_NAME))` and `Δ_peer_med = mean(D_peer_med(ALIAS_TO_ROLE) - D_peer_med(ALIAS_TO_NAME))`.
  - **Pre-declared token-level peer sign reducer:** Over the `20` distinct slot peer token IDs `t ∈ pool`, for each `t` compute its mean shift across all eligible `(unit u, story e)` pairs where `t ∈ peers(u)` (`16` pairs if `t` is one of the 5 core units' own tokens; `20` pairs for the other 15 tokens): `δ_t = mean_{(u,e): t ∈ peers(u)} [D_peer_t(ALIAS_TO_ROLE) - D_peer_t(ALIAS_TO_NAME)]`. Define **`frac_peer_tokens_pos = (1/20) * #{t ∈ pool : δ_t > 0}`**, alongside **`frac_pairs_peer_med_pos = (1/20) * #{(u,e) : D_peer_med(ALIAS_TO_ROLE) - D_peer_med(ALIAS_TO_NAME) > 0}`**.
  - **Exhaustive classification:**
    - `peer_dominated`: `|Δ_peer_med| >= 2 * |Δ_raw|` AND `Δ_peer_med >= +0.20` AND `frac_peer_tokens_pos >= 0.75`
    - `target_dominated`: `|Δ_raw| >= 2 * |Δ_peer_med|` AND `Δ_raw <= -0.20`
    - `weak_or_null`: `|Δ_raw| < 0.20` AND `|Δ_peer_med| < 0.20`
    - `mixed_or_narrow_peer`: all remaining cases (reported with exact `Δ_raw`, `Δ_peer_med`, `frac_peer_tokens_pos`, `frac_pairs_peer_med_pos`, and all 20 `δ_t`).
  - *Pre-data prediction:* `SS.2(a) == holds`, `SS.2(b) == holds`, and `SS.2(c) == peer_dominated`.

### `SS.3` — Primary Onset Interaction (`midtrain_end -> sft`), Separately for Arm A and Arm B
Define within stage `s` and arm: `C_SR(s) = mean_{20 pairs}[D_SELF(s) - D_ROLE_BARE(s)]`, `R_SR(s) = max_{h in {L8H2,L8H3,L8H13}} mean_{20 pairs}[D_{h,SELF}(s) - D_{h,ROLE_BARE}(s)]`, and `I_onset = C_SR(sft) - C_SR(midtrain_end)` (also reporting the alias analogue `C_SA(s) = mean[D_SELF(s) - D_ALIAS_TO_ROLE(s)]` alongside).
- **Step 1 (Gate precedence across `midtrain_end` & `sft`):**
  1. If either stage is `invalidated` → **`invalidated`**.
  2. Else if `sft` is `unreadable` or `uninformative` → return `sft`'s status.
  3. Else if `midtrain_end` is `unreadable` → **`unreadable`**.
  4. Else if `midtrain_end` is `uninformative` (e.g., midtrain transport failed or Arm B `d_cal(midtrain_end) is None`):
     - `I_onset` status = **`uninformative (midtrain gate/cal failed)`**.
     - Fallback `first_measurable_at_sft` status:
       - `holds (first_measurable_at_sft)`: `C_SR(sft) >= +0.25` AND `C_SR(sft) - R_SR(sft) >= +0.15`
       - `fails (not measurable at sft)`: `C_SR(sft) < +0.25` OR `C_SR(sft) - R_SR(sft) < 0.0`
       - `inconclusive (sft random-margin intermediate)`: `C_SR(sft) >= +0.25` AND `0.0 <= C_SR(sft) - R_SR(sft) < +0.15`
- **Step 2 (Both `midtrain_end` and `sft` are `valid`):**
  - **`holds`:** `C_SR(sft) >= +0.25` AND `C_SR(sft) - R_SR(sft) >= +0.15` AND `I_onset >= +0.25` AND `|C_SR(midtrain_end)| < 0.20`
  - **`fails`:** `I_onset <= 0.0` OR `C_SR(sft) < +0.25` OR `C_SR(sft) - R_SR(sft) < 0.0`
  - **`inconclusive`:** all remaining intermediate cases (e.g. `C_SR(sft) = +0.35, C_SR(midtrain_end) = +0.25, I_onset = +0.10`, or `0.0 < I_onset < +0.25`, or `|C_SR(midtrain_end)| >= 0.20` with `I_onset >= +0.25`, or `0.0 <= C_SR(sft) - R_SR(sft) < +0.15`).
- *Pre-data prediction:* `holds` (or `holds (first_measurable_at_sft)` if `midtrain_end` fails transport).

### `SS.4` — Later Post-SFT Change (`sft -> dpo -> instruct`), Separately for Arm A and Arm B
Kept strictly separate from `SS.3` onset (Alden `#54`). Define `I_dpo = C_SR(dpo) - C_SR(sft)` and `I_rlvr = C_SR(instruct) - C_SR(dpo)`.
- **Gate check:** If any of `{sft, dpo, instruct}` is `invalidated` → `invalidated`; else if any is `unreadable` → `unreadable`; else if any is `uninformative` → `uninformative` (while still reporting valid pairwise steps separately).
- **When all three post-trained stages are `valid` (exhaustive partition of $\mathbb{R}^3$):**
  - **`holds (post-SFT stability)`:** `C_SR(sft) >= +0.25` AND `C_SR(dpo) >= +0.25` AND `C_SR(instruct) >= +0.25` AND `|I_dpo| < 0.25` AND `|I_rlvr| < 0.25`
  - **`fails_amplification`:** `(I_dpo >= +0.25 OR I_rlvr >= +0.25)` AND NOT `(I_dpo <= -0.25 OR I_rlvr <= -0.25)`
  - **`fails_attenuation_or_loss`:** `(I_dpo <= -0.25 OR I_rlvr <= -0.25 OR C_SR(sft) < +0.25 OR C_SR(dpo) < +0.25 OR C_SR(instruct) < +0.25)` AND NOT `(I_dpo >= +0.25 OR I_rlvr >= +0.25)`
  - **`fails_non_monotone_mixed`:** `(I_dpo >= +0.25 OR I_rlvr >= +0.25)` AND `(I_dpo <= -0.25 OR I_rlvr <= -0.25)` (e.g., large jump at DPO followed by large drop at Instruct)
- *Pre-data prediction:* `holds (post-SFT stability)`.

### `SS.5` — Finite-Bank Two-Factor `SELF` vs. `NAMED` Closeness (No Invalid `n=20` iid Bootstrap — Alden `#58` Fix 3)
Because the `20` pairs cross `5` units × `4` stories, **population-level equivalence is explicitly NOT claimed** (`population_equivalence: "not_claimed_finite_5x4_bank"`). Instead, at each valid `(stage s, arm)`, let `Δ_SN(u, e) = D_SELF(u, e) - D_NAME_BARE(u, e)` (and separately for `ALIAS_TO_NAME`), with grand mean `M`, `5` unit means `M_u` (each averaged over 4 stories), and `4` story means `M_e` (each averaged over 5 units):
- **`holds (finite-bank close & separated from role)`:** `|M| <= 0.15` AND `max_{e ∈ E1..E4} |M_e| <= 0.25` (all 4/4 stories) AND `#{u : |M_u| <= 0.25} >= 4` (at least 4/5 units) AND `C_SR(s) >= +0.25`.
- **`fails`:** `|M| > 0.25` OR `C_SR(s) < +0.25`.
- **`inconclusive`:** all remaining cases (`0.15 < |M| <= 0.25` with `C_SR(s) >= +0.25`, OR `|M| <= 0.15` with `C_SR(s) >= +0.25` that fails story/unit consistency).
- **Two-end role-to-name movement check (`midtrain_end -> sft`):** Claiming *"`I` moves from role-like to name-like on this finite bank"* additionally requires at `midtrain_end` (`valid`): `|C_SR(midtrain_end)| <= 0.15` (with 4/4 stories and >=4/5 units inside `±0.25`) AND `mean(D_ROLE_BARE - D_NAME_BARE) <= -0.25`. Otherwise report the exact per-stage finite-bank pattern without claiming role-to-name migration.

### `SS.6` — Secondary Head `L7H5` (`L5/L6 Late-Write` vs. `L3 Early-Write` Odd-Dominant Readout Contrast — Alden `#58` Fix 4)
Evaluated at each valid `(stage s, arm)` on `ROLE_BARE` (and `ALIAS_TO_ROLE`), reporting `D(+d)`, `D(-d)`, `E(d)`, `O(d)`, and `ratio = |O(d)| / (|E(d)| + 0.05)` for:
- Pooled `5` units (`20` pairs),
- **`L5/L6 late-write` (`p92-46` [L5], `p92-09` [L6]; `8` pairs)**,
- **`L3 early-write` (`p92-02`, `p92-05`, `p92-13`; `12` pairs)**,
- Single-unit `L6`-only `p92-09` (`4` pairs, strictly descriptive).
- **Instruct replication decision map (Arm A `d = 0.8`, `ROLE_BARE`):**
  - `holds`: `D(-0.8)_spec_pooled >= +0.10` AND `D(-0.8)_spec_L5L6 >= +0.25` AND `D(-0.8)_spec_L5L6 - D(-0.8)_spec_L3 >= +0.25` AND `ratio_L5L6 > 1.0` AND `D(-0.8)_spec_pooled > max_k D_null_wdown_k`
  - `fails`: `D(-0.8)_spec_pooled <= 0.0` OR `D(-0.8)_spec_L5L6 <= 0.0` OR `D(-0.8)_spec_L5L6 <= D(-0.8)_spec_L3`
  - `inconclusive`: all remaining intermediate cases.
- Across stages (`midtrain_end -> sft -> dpo -> instruct`), report the exact `L5/L6` and pooled trajectories descriptively alongside `L8H15` (without calling one-dose odd dominance a linear-relay proof).

### `SS.7` — Plain-Text Sensitivity Arm (`SENS_PLAIN` vs. Matched `E1, E2` Chat Probes at `d = 0.8` — Alden `#62` Fix 2)
At every stage `s`, evaluate BOTH serializations (`chat` and `plain`) on the **exact same `E1, E2 × {SELF, ROLE_BARE}` probes (`10` tuples per cell)** with the **exact same three-gate predicate** (so bank and serialization never change together):
- **Matched-probe gate vector for serialization `w ∈ {chat, plain}` on `E1, E2` at `d = 0.8` (strict precedence `1 -> 2 -> 3`):**
  1. `Gate 1 (Sham)`: `max_abs_sham <= 1e-4` across 100% of `E1, E2 × {SELF, ROLE_BARE}` tuples in serialization `w` (both `SpecLogit` and `RawLogit`) → failure = `invalidated`.
  2. `Gate 2 (Cell-wise final-logit transport positive control on E1, E2)`: BOTH `SELF` (`10` tuples) AND `ROLE_BARE` (`10` tuples) individually satisfy `N_write_spec >= +0.50` AND `N_write_raw >= +0.50` AND `N_write_spec > max_{k=0..3} N_write_null_wdown_k` → failure = `uninformative`.
  3. `Gate 3 (Null gate on E1, E2)`: On `L8H15`, `|mean E_null_wdown| <= 0.15` AND `|mean O_null_wdown| <= 0.15` on BOTH `SELF` and `ROLE_BARE` (`E1, E2`) → failure = `unreadable`.
  4. Otherwise `valid`.
- **Total deterministic disposition over `(chat_gate, plain_gate)` (`16/16` combinations) and `(C_SR_chat_E1E2(s), C_SR_plain(s))`:**
  1. If either `chat_gate == invalidated` OR `plain_gate == invalidated` → **`invalidated`** (recording both individual gate statuses).
  2. Else if `chat_gate == uninformative` AND `plain_gate == valid` → **`transport_diff_plain_only`** (strictly reported as a **transport-only difference between chat and plain serialization on `E1/E2`**, not proof about why the mechanism changed; `C_SR_plain(s)` logged descriptively).
  3. Else if `chat_gate == valid` AND `plain_gate == uninformative` → **`transport_diff_chat_only`** (strictly a transport-only difference on `E1/E2`; `C_SR_chat_E1E2(s)` logged descriptively).
  4. Else if either `chat_gate == unreadable` OR `plain_gate == unreadable` → **`unreadable`** (recording both individual gate statuses).
  5. Else if `chat_gate == uninformative` AND `plain_gate == uninformative` → **`uninformative (both_transport_failed)`**.
  6. Else (BOTH `chat_gate == valid` AND `plain_gate == valid`), exhaustive 4-way partition of `(C_SR_chat_E1E2(s), C_SR_plain(s))`:
     - **`chat_independent (preserved_across_wrappers)`:** `C_SR_chat_E1E2(s) >= +0.25` AND `C_SR_plain(s) >= +0.25`
     - **`chat_dependent`:** `C_SR_chat_E1E2(s) >= +0.25` AND `C_SR_plain(s) < +0.15`
     - **`plain_positive_only`:** `C_SR_chat_E1E2(s) < +0.25` AND `C_SR_plain(s) >= +0.25` *(e.g. witness `chat = -0.20, plain = +0.30`: establishes positive plain-text contrast only, not preservation across wrappers!)*
     - **`neither_or_intermediate`:** all remaining valid pairs (`chat < +0.25` AND `plain < +0.25`, OR `chat >= +0.25` AND `+0.15 <= plain < +0.25`).

---

## 7. Pre-Run Build & Verification Checklist (Only After Design PASS)
1. **Tokenizer-only manifest (`--emit-manifest`):** Loads all 4 checkpoints' tokenizers only (zero weights), asserts identical `vocab_size` and identical token IDs on all 28 prompts, verifies every `comp_toks`, `read_pos`, `t_own`, `19` distinct `peers`, and `assert_no_forbidden_overlap()`.
2. **CPU self-tests (`--self-test-only` on runner & scorer):** Admission & fake-loader checks (zero loader calls on any mismatch), sham/null/non-causal sabotages, 4-pin external hash checks, **plus** explicit unit tests for:
   - Alden `#58` monotone calibration witness `[-0.30, -0.35, -0.40, -0.45, -0.50, -0.55]` → Arm B `d_cal = None` (`uninformative_no_rebound_bracket`);
   - Exact tie-break (selects smallest `d`);
   - Cell-wise transport refusal on `(0.0, +1.1)` pooled `+0.55` witness;
   - Every intermediate branch of `SS.2`–`SS.7` including Alden `#58` witnesses (`SS.3`: `SFT +0.35, midtrain +0.25, onset +0.10 -> inconclusive`; `SS.4`: large negative drop `-> fails_attenuation_or_loss`) AND Alden `#62` witnesses (`SS.2(b)`: `G_bare = -0.50, R_bare = -0.25, G_alias = -0.60, R_alias = -0.75 -> fails` with `fail_reason = random_head_comparison_failed` and `Δ_alias_minus_bare = -0.10 -> preserved_or_stronger_under_same_referent_manipulation`; `SS.7`: `chat = -0.20, plain = +0.30 -> plain_positive_only`, plus all `invalidated`/`unreadable`/`uninformative`/`transport_diff_*` gate dispositions on matched `E1/E2` probes);
   - Exact algebraic decomposition `D_spec == D_raw - D_peer_med` and `20`-token `frac_peer_tokens_pos` calculation.
3. **Measured 1-unit smoke** on `midtrain_end` + `instruct` (`p92-02`) before any full 4-checkpoint Colab run.
