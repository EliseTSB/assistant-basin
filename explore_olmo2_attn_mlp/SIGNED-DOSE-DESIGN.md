# Frozen Instruct Signed-Dose, Role-Genericity & Turn-Boundary Design (OLMo 2 1B) — v2

**Status:** DRAFT v2 (2026-11-20 CT, Vesper / Gemini 4 Neon) — folds all four HOLD points from Alden (`#32` on `vesper-olmo2-attn-mlp-2026-11-20`) before any code or model call. Predictions `SD.1`–`SD.6` are authored strictly before any data on the fresh held-out prompt bank.
**Immutability note:** Once Alden passes v2 (and after Elise approves a clean commit/pin), this file + the rendered tokenizer manifest JSON will be committed/pinned by SHA-256 before any model run.

---

## 1. Scientific Scope & Three Separate Estimands

Model and precision are frozen throughout:
- **Model & Tokenizer:** `allenai/OLMo-2-0425-1B-Instruct` @ immutable 40-char commit `48d788eca847d4d7548f375ad03d3c9312f6139e`, `torch.float32`, `attn_implementation="eager"`. Code structurally refuses `main`, `a1847dff`, or any non-40-char hex SHA.
- **Residual-Stream Write Definition:** For Part 92 slot unit `u = (L_u, n_u)`, the frozen Instruct residual write is `v_u = m_u * W_down^{(L_u)}[:, n_u] / ||W_down^{(L_u)}[:, n_u]||_2`, where `m_u` is Instruct's layer-`L_u` chat residual median (pinned in the pre-run manifest). At signed dose `d`, `+d * v_u` is added to the cloned output of decoder layer `L_u` (at all positions in Q1/Q2; at masked positions only in Q3).
- **Units (`5` core Part 92 slot units):**
  - `L3` (`3` units): `p92-02` (`L3 N2628`, `"town"`), `p92-05` (`L3 N6986`, `"fox"`), `p92-13` (`L3 N976`, `"skeleton"`)
  - `L5` (`1` unit): `p92-46` (`L5 N4673`, `"regular guy"`)
  - `L6` (`1` unit): `p92-09` (`L6 N4145`, `"dog"`)
- **Heads (`12` key + `12` matched-layer random = `24` total):**
  - Key heads: `L4H3`, `L5H4`, `L6H6`, `L6H8`, `L7H5`, `L8H4`, `L8H11`, `L8H15`, `L9H6`, `L9H10`, `L11H0`, `L12H4`.
  - Matched-layer random heads: fixed seed `20261120`, exact same layer counts (`L4:1, L5:1, L6:2, L7:1, L8:3, L9:2, L11:1, L12:1`), strictly disjoint from key heads.
- **Strictly Downstream Eligibility Rule (Alden `#32` point 1):**
  - A head `(L_h, h)` is eligible for unit `u = (L_u, n_u)` **if and only if `L_h > L_u`** (strictly downstream of layer `L_u`'s residual write). Non-causal `(unit, head)` pairs (`L_h <= L_u`) are structurally excluded (and unit-tested to raise if passed to the scorer).
  - Exact eligible `(unit, head)` pair count:
    - `L3` units (`3`) × (`12` key + `12` rand) = `72` pairs
    - `L5` unit (`1`) × (`10` key + `10` rand) = `20` pairs
    - `L6` unit (`1`) × (`8` key + `8` rand) = `16` pairs
    - **Total = `108` eligible `(unit, head)` pairs (`54` key + `54` random).**
  - Eligible unit sets by head layer: `L7+` heads have all **`5`** units; `L6` heads have **`4`** (`L3 + L5`); `L4/L5` heads have **`3`** (`L3` only). Every cross-head ratio or head-vs-matched-random comparison MUST use the exact common eligible unit set (for `L7+` heads such as `L7H5`, `L8H4`, `L8H11`, `L8H15`, all `5` units are common; `L5H4` is evaluated on its `3` eligible `L3` units alongside its matched `L5` random head on the same `3` units).

---

## 2. Exact Definitions, Controls, Readouts & Token Binding

### 2.1 Prefix-Attached Token Binding & Duplicate-Peer Policy (Alden `#32` point 1)
- Every prompt ends at the exact read position `read_pos = len(prefix_ids) - 1` (the token immediately preceding the first content token of the persona noun phrase; e.g. after `"I am a"`, `"The nurse is a"`, `"Clara is a"`, using each unit's pinned `lead` article from Part 92/213).
- For every `(unit u, prompt p)`, the **own target token ID** `t_own(u, p)` is computed strictly by **prefix-attached tokenization**: `tok.encode(prefix_str + " " + content_u)[len(prefix_ids)]`, asserting that `tok.encode(prefix_str + " " + content_u)[:len(prefix_ids)] == prefix_ids` (no BPE merge across the read boundary; fails hard if violated).
- **Peer set & duplicate-peer policy:** The peer pool consists of the other `20` Part 92 slot units' prefix-attached first content token IDs on the exact same `prefix_str`. Any peer token ID equal to `t_own(u, p)` is **excluded** (and duplicate peer token IDs among peers are **deduplicated** to a set of distinct non-own peer token IDs), logging `n_peers_distinct(u, p)` in the manifest and asserting `n_peers_distinct >= 15`.
- **Dual Readouts (reported in every table and JSON row):**
  - `RawLogit(u, p, state) = logits[read_pos, t_own(u, p)]`
  - `SpecLogit(u, p, state) = RawLogit(u, p, state) - median_{t in peers(u, p)} logits[read_pos, t]`

### 2.2 Q1 & Q2 Estimand — Same-Dose Path-Patching Drop `D(d)` & Even/Odd Decomposition (Alden `#32` point 1)
- **Batch Layout (`B = 19`, identical forward batch shape across donor, unpatched, patched, and sham!):**
  For each `(unit u, prompt p, magnitude |d| > 0)`:
  - Row `0`: unpushed (`d = 0.0`)
  - Row `1`: `+d * v_u`
  - Row `2`: `-d * v_u`
  - Rows `3..6`: `+d * v_wdown_orth[k]` (`k = 0..3`; target column `n_u` strictly zeroed before Gram-Schmidt orthogonalization against `v_u`, L2 norm matched to `||v_u||_2`)
  - Rows `7..10`: **`-d * v_wdown_orth[k]`** (**exact same 4 signed unit vectors `v_wdown_orth[0..3]` negated**, NOT independent random draws!)
  - Rows `11..14`: `+d * v_iso_resid[k]` (`k = 0..3`; isotropic Gaussian residual orthogonalized against `v_u`, L2 norm matched to `||v_u||_2`)
  - Rows `15..18`: **`-d * v_iso_resid[k]`** (**exact same 4 signed unit vectors `v_iso_resid[0..3]` negated!**)
- **Same-Dose Patching & Exact Definition of `D`:**
  1. Run one unpatched `B = 19` forward pass for `(u, p, |d|)`, recording:
     - `R_unpatched[b]` (`RawLogit` and `SpecLogit`) for all `b ∈ 0..18`
     - `z_donor[L_h, h]` = Row `0` (`d = 0.0`) pre-`o_proj` activation of every eligible head `(L_h, h)`
     - `z_self[L_h, h]` = Rows `0..18` own unpatched pre-`o_proj` activations of every eligible head `(L_h, h)`
  2. For each eligible head `(L_h, h)` (`L_h > L_u`):
     - **`unpushed_z` pass (`B = 19`, identical shape):** replace head `(L_h, h)` pre-`o_proj` activation across all 19 rows with `z_donor[L_h, h]` (broadcast to `B = 19`), recording `R_patched[b]`.
     - **`sham_z` pass (`B = 19`, identical shape):** replace head `(L_h, h)` pre-`o_proj` activation with `z_self[L_h, h]` (each row gets its own exact unpatched `z`), recording `R_sham[b]`.
  3. **Exact drop definition (same dose, same read position, same batch shape):**
     - `D_row(b) = R_unpatched[b] - R_patched[b]`
     - `D_sham(b) = R_unpatched[b] - R_sham[b]`
     - Zero-dose drop: `D(0) = D_row(0)` (measured explicitly in every batch!)
     - Neuron signed drops: `D(+d) = D_row(1)`, `D(-d) = D_row(2)`
     - **Centered even contrast:** `E(d) = (D(+d) + D(-d)) / 2 - D(0)`
     - **Odd contrast:** `O(d) = (D(+d) - D(-d)) / 2`
     - **Signed null contrasts (draw-paired!):** For each draw `k ∈ 0..3` of `wdown_orth` (and separately `iso_resid`), compute `D_null_k(+d)`, `D_null_k(-d)`, `E_null_k(d) = (D_null_k(+d) + D_null_k(-d)) / 2 - D(0)`, and `O_null_k(d) = (D_null_k(+d) - D_null_k(-d)) / 2`.
- **Exact Reducer Policy:**
  - Per `(unit, prompt, head, |d|)`: preserve all 4 per-draw null values (`k = 0..3`) in JSON, and reduce across the 4 draws by **median** (`null_med`) — plus record `null_max_abs = max_k |val_k|` and `null_max = max_k val_k`.
  - Across `(unit, prompt)` within a `(head, cell, |d|)`: primary summary is the **arithmetic mean** over all eligible `(unit, prompt)` tuples (reporting median and IQR beside every mean).
- **Explicit Interpretive Boundary (Alden `#17`):** Non-zero `E(d)` and `O(d)` establish asymmetric / nonlinear readout dependence in these heads; they do **not** identify two separate physical circuits, because one nonlinear path can generate both even and odd terms.

### 2.3 Counterbalanced Fresh Prompt Bank for Q1 & Q2 (Alden `#32` point 2)
All prompts are strictly disjoint from Run 1, `explore_ladder.py`, Part 213, and Part 180 v5.1 (enforced by automated `assert_no_forbidden_overlap()`).

- **Counterbalanced Story Frames F1–F3 (identical wording, syntax, and distance to `read_pos` across `THIRD_GENERIC`, `ROLE_DEFINITE`, `NAMED_PERSON`, and `LEX_ASSISTANT`; matched second-person framing for `SELF_HELD`):**
  - **F1 (clinic):** `"At the village clinic before dusk, {SUBJ} opens the heavy oak door. Identify {SUBJ_L} in one line."` → `"{SUBJ_CAP} is "`
  - **F2 (market):** `"Across the quiet market square at dawn, {SUBJ} carries a woven basket. Identify {SUBJ_L} in one line."` → `"{SUBJ_CAP} is "`
  - **F3 (harbor):** `"Beside the stone harbor wall at nightfall, {SUBJ} waits by the iron gate. Identify {SUBJ_L} in one line."` → `"{SUBJ_CAP} is "`
  - **Self versions (F1–F3):**
    - `self_F1`: `"At the village clinic before dusk, you open the heavy oak door. Identify yourself in one line."` → `"I am "`
    - `self_F2`: `"Across the quiet market square at dawn, you carry a woven basket. Identify yourself in one line."` → `"I am "`
    - `self_F3`: `"Beside the stone harbor wall at nightfall, you wait by the iron gate. Identify yourself in one line."` → `"I am "`
  - **Substituted cells on exact frames F1, F2, F3:**
    - `THIRD_GENERIC_NEW` (`3` prompts): F1=`"the stranger"`, F2=`"the visitor"`, F3=`"the traveler"`
    - `ROLE_DEFINITE_NEW` (`3` prompts): F1=`"the nurse"`, F2=`"the baker"`, F3=`"the guard"` *(zero overlap with `"assistant"`!)*
    - `NAMED_PERSON_NEW` (`3` prompts): F1=`"Clara"`, F2=`"Devon"`, F3=`"Nora"` *(note: definite article `"the"` + role noun is 2 tokens vs bare proper name 1 token; exact subject token count and read distance are logged in the manifest, and `ROLE_DEFINITE_NEW` vs `THIRD_GENERIC_NEW` is 100% token-length-matched `2 vs 2`!)*
    - `LEX_ASSISTANT_CTRL` (`2` separate lexical control prompts, frames F1 & F2): `"the assistant"` — kept out of `ROLE_DEFINITE_NEW` per Alden `#32` to separate template-token lexical overlap (`<|assistant|>`) from general role nouns!
- **Counterbalanced Person-vs-Object Frames O1–O2 (exact same frame, exact same `"Identify ... in one line."` verb, exact same `"The <noun> is "` 2-token subject and read distance!):**
  - **O1 (window):** `"In the quiet room by the window sits {SUBJ}. Identify {SUBJ_L} in one line."` → `"{SUBJ_CAP} is "`
  - **O2 (courtyard):** `"Near the wooden bench in the courtyard rests {SUBJ}. Identify {SUBJ_L} in one line."` → `"{SUBJ_CAP} is "`
  - Substituted cells on exact frames O1, O2:
    - `PERSON_OBJFRAME` (`2` prompts): O1=`"the nurse"`, O2=`"the guard"`
    - `OBJECT_OBJFRAME` (`2` prompts): O1=`"the lantern"`, O2=`"the kettle"`
  - *Note:* `PERSON_OBJFRAME` vs `OBJECT_OBJFRAME` holds frame, grammar, article (`"the"`), subject token length, and read-position distance strictly identical, isolating animate human role vs. inanimate object cleanly!
- **Total Q1+Q2 Prompts = `18`** (`3 SELF + 3 THIRD_GENERIC + 3 ROLE_DEFINITE + 3 NAMED_PERSON + 2 LEX_ASSISTANT + 2 PERSON_OBJFRAME + 2 OBJECT_OBJFRAME`).

### 2.4 Q3 Estimand — Mask-Localized Residual Write & BOS vs. Turn-Boundary Test (Alden `#32` point 3)
Q3 has its **own separate estimand** (NOT head-patching `unpushed_z`!).
- **Intervention:** Mask-localized residual write at fixed `d = +0.8` (no head patching), batched `B = 10` (`[0]=unpushed, [1]=+0.8*v_u, [2..5]=+0.8*v_wdown_orth[0..3], [6..9]=+0.8*v_iso_resid[0..3]`).
- **Metric:** For each `(unit u, prompt p, mask m)`:
  - `N_z(u, p, m) = SpecLogit(+0.8*v_u, m) - SpecLogit(unpushed)`
  - `R_z(u, p, m) = median_{k=0..3} [SpecLogit(+0.8*v_wdown_orth[k], m) - SpecLogit(unpushed)]` (with `iso_resid` and raw target logit recorded alongside)
  - **`L_z(u, p, m) = N_z(u, p, m) - R_z(u, p, m)`**
  - Self-minus-Third contrast per `(unit, story_frame, mask)`: `Delta_ST(m) = L_z(SELF, m) - L_z(THIRD_GENERIC, m)`, reduced by mean across all `5 units × 2 frames = 10` pairs.
- **Exact Three Dialogue Frames (`2` stories F1 & F2 × `2` cells `SELF` & `THIRD_GENERIC` × `3` dialogue frames = `12` prompts):**
  1. **`one_turn`:** standard 1-turn chat: `<|endoftext|><|user|>\n{Q}\n<|assistant|>\n{PREFIX}`
  2. **`two_turn_asst_user`:** prior neutral Q/A turn with explicit turn boundaries:
     `<|endoftext|><|user|>\nWhat color is clear ice?\n<|assistant|>\nClear ice is colorless.<|endoftext|><|user|>\n{Q}\n<|assistant|>\n{PREFIX}`
  3. **`no_boundary_concat` (no-boundary control, NOT called a "user->user turn"!):** matches the exact visible words, word order, and read distance of `two_turn_asst_user` inside a single `<|user|>` block with NO mid-dialogue `<|assistant|>`/`<|endoftext|>`/`<|user|>` delimiters:
     `<|endoftext|><|user|>\nWhat color is clear ice?\nClear ice is colorless.\n{Q}\n<|assistant|>\n{PREFIX}`
- **Exact Frozen Masks (indices & decoded tokens pinned per prompt in `signed_dose_manifest.json`!):**
  - Always evaluated:
    - `bos_tok`: token index `0` ONLY (`['<|endoftext|>']`, 1 tok — strictly distinguished from mid-dialogue `<|endoftext|>`)
    - `user_hdr_turn_final`: the **active (final) `<|user|>\n` header** immediately preceding `{Q}` (`['<', '|', 'user', '|', '>\n']`, **5 tok**; in `one_turn` and `no_boundary_concat` this is indices `1..5`; in `two_turn_asst_user` this is the Turn-2 `<|user|>\n` header, and we ALSO separately evaluate `user_hdr_turn1` at `1..5`, `mid_eot` = `<|endoftext|>`, and `mid_eot_nl` = `['<|endoftext|>', '\n']`!)
    - `pre_user_union`: in `one_turn` / `no_boundary_concat`, exact union `bos_tok ∪ user_hdr_turn_final` (`0..5`, **6 tok**); in `two_turn_asst_user`, exact contiguous Turn-2 boundary `mid_eot_nl ∪ user_hdr_turn_final` (**7 tok**, AND separately `pre_user_turn1_union = bos_tok ∪ user_hdr_turn1` = `0..5`, 6 tok!)
    - `post_user_delim_final`: the active (final) `<|assistant|>\n` immediately preceding `{PREFIX}` (`['<', '|', 'assistant', '|', '>\n']`, **5 tok**)
    - **Matched Controls:** `user_tail_1` (last 1 tok of `{Q}`, matches `bos_tok`), `user_tail_5` (last 5 tok of `{Q}`, matches `user_hdr_turn_final` and `post_user_delim_final`), `user_tail_union` (last `len(pre_user_union)` tok of `{Q}`, matches `pre_user_union`), `user_prompt_energy_matched` (full `{Q}` scaled by `sqrt(len(pre_user_union) / len(Q))`), plus `all_pos`.
- **Mandatory Interaction Term (Alden `#32` point 3):**
  - For every `(unit, story, frame)`, compute and report the exact paired union-minus-parts interaction:
    `I_pre(u, s) = Delta_ST(pre_user_union) - [Delta_ST(bos_tok) + Delta_ST(user_hdr_turn_final)]` (in `one_turn`, and analogously for `mid_eot_nl + user_hdr_turn_final` in `two_turn_asst_user`).
  - Explicitly note: **a `bos_tok`-localized effect establishes initial-token sensitivity, NOT an attention-sink mechanism**, and a component/union ratio does not identify linear causal share when `I_pre != 0`.

---

## 3. Exact Frozen Schedule, Runtime Measurement & Failure Policy (Alden `#32` points 1 & 4)

- **Block A (Signed-dose curve, `|d| ∈ {0.2, 0.4, 0.8, 1.6}` on 4 core prompts: `self_F1, self_F2, third_F1, third_F2`):**
  - Unpatched + donor/sham capture: `5 units × 4 prompts × 4 |d| = 80` forwards (`B = 19`)
  - Patched (`unpushed_z`): `108 eligible pairs × 4 prompts × 4 |d| = 1,728` forwards (`B = 19`)
  - Same-dose sham (`sham_z`): `108 eligible pairs × 4 prompts × 4 |d| = 1,728` forwards (`B = 19`) — **100% coverage**
- **Block B (Counterbalanced Role/Name/Object at `|d| = 0.8` on remaining 14 prompts):**
  - Unpatched + donor/sham capture: `5 units × 14 prompts × 1 |d| = 70` forwards (`B = 19`)
  - Patched (`unpushed_z`): `108 eligible pairs × 14 prompts × 1 |d| = 1,512` forwards (`B = 19`)
  - Same-dose sham (`sham_z`): `108 eligible pairs × 14 prompts × 1 |d| = 1,512` forwards (`B = 19`) — **100% coverage**
- **Block C (Q3 mask-localized residual writes at `+0.8` on 12 dialogue-frame prompts):**
  - `5 units × (8 prompts × 11 masks + 4 two_turn prompts × 15 masks) = 740` forwards (`B = 10`)
- **Grand Total:** **`6,630` `B = 19` forwards + `740` `B = 10` forwards**.
- **Measured Smoke Before Full Run:** Before launching the full sweep, run a **1-unit, 2-prompt smoke** (or measure the first unit's wall-clock rate and log `ms/forward`) to confirm exact runtime and compute budget rather than relying on an unmeasured estimate.
- **Pre-Run Audit Manifest & Missing/Invalid Row Policy:**
  - Before any forward pass, the script emits a deterministic `manifest.json` pinning: exact rendered prompt strings, token IDs, `read_pos`, prefix-attached `t_own(u, p)`, deduplicated `peers(u, p)` & `n_peers_distinct`, exact mask indices + decoded mask tokens per frame, exact eligible `(unit, head)` denominators, calibration medians, and null generator seeds.
  - **Missing/Invalid Row Policy:** Any missing tuple, non-finite logit, boundary BPE merge, or non-causal head (`L_h <= L_u`) raises a hard error and aborts scoring — zero silent row drops or imputed values. Full per-tuple rows and per-unit atomic checkpoints are saved to disk.
- **Required Fake/Inert Unit Tests (`--self-test-only`, ALL must pass before any GPU launch):**
  1. **Swapped sign companions sabotage:** pairing `+d` null with a non-negated/independent `-d` null vector fails assertion (paired positive passes).
  2. **Wrong boundary occurrence sabotage:** selecting Turn-1 `<|user|>\n` instead of Turn-2 `<|user|>\n` in `two_turn_asst_user` fails assertion (paired positive passes).
  3. **False fraction denominator sabotage:** Alden's fixture `bos = -0.05, union = -0.20` MUST return `not_read (union below floor)` rather than PASS (paired positive `bos = +0.40, union = +0.60` passes).
  4. **Non-causal head sabotage:** submitting `L5H4` for unit `p92-46 (L5)` or `L6H6` for `p92-09 (L6)` MUST raise `ValueError` (paired positive `L7H5` for `p92-09 (L6)` passes).

---

## 4. Prospective Predictions & Gate Dispositions (Authored 2026-11-20 by Vesper, BEFORE Any Data)

- **Gate & Unreadability Rules (Alden `#32` point 1):**
  - If `SD.1(a)` (sham gate) fails → **entire Q1+Q2 run is `invalidated (sham gate failed)`** (`SD.2`–`SD.5` not read).
  - If `SD.1(b)` (null gate on L7/L8 key heads) fails → **`SD.2`, `SD.3`, `SD.4`, and `SD.5` are marked `unreadable (null gate failed)`** (raw numbers reported descriptively only, no prediction passes).
  - Q3 (`SD.6`) has its own independent mask-write null check (`|mean R_z(pre_user_union)| <= 0.15` and `SD.6(a)` union floor `Delta_ST(pre_user_union) >= +0.30`).

| # | Prediction | Confidence | Author |
|---|---|---|---|
| **SD.1** | **Sham & sign-matched null gate (Q1+Q2):** (a) Across all `6,480` patched/sham tuples (`3,240` `unpushed_z` paired with `3,240` `sham_z`), `max |D_sham| < 1e-4` logits on both `RawLogit` and `SpecLogit`; AND (b) on `THIRD_GENERIC_NEW` (`F1–F2`) at `d = 0.8`, for each of `L8H15`, `L8H4`, `L8H11`, `L7H5` (`5` eligible units × `2` prompts = `10` tuples), the mean of per-tuple draw-medians for sign-matched `wdown_orth` satisfies `|mean E_null(0.8)| <= 0.15` and `|mean O_null(0.8)| <= 0.15` logits on `SpecLogit`. Failure of (a) invalidates Q1+Q2; failure of (b) marks `SD.2`–`SD.5` `unreadable (null gate failed)`. | high | Vesper |
| **SD.2** | **Rectified suppression on `L8H15` & `L8H4` (`V.W2`, Block A `THIRD_GENERIC_NEW`, `5` units × `2` prompts = `10` tuples):** On `SpecLogit`, both `L8H15` and `L8H4` show monotonic negative `mean D(+d)` across `0.2 -> 0.4 -> 0.8 -> 1.6` (`D(+1.6) < D(+0.8) < D(+0.4) < D(+0.2)`), negative centered even contrast `mean E(0.8) <= -0.15` and `mean E(1.6) <= -0.25` with `|mean E(0.8)| >= 0.40 * |mean O(0.8)|`, AND `mean D(-0.8) < +0.25` logits (no strong symmetric sign-flip). Raw target logit reported beside every table. | medium | Vesper |
| **SD.3** | **Odd-dominant / partial-flip contrast on `L8H11` & `L7H5` (Block A `THIRD_GENERIC_NEW`, common `5` units × `2` prompts = `10` tuples):** On `SpecLogit` at `d = 0.8`, both `L8H11` and `L7H5` satisfy: (i) `mean D(-0.8) >= +0.10` logits, (ii) strictly above sign-matched `wdown_orth` null by `mean [D(-0.8) - null_wdown_med(-0.8)] >= +0.10` AND `mean D(-0.8) > max_{k=0..3} mean D_null_k(-0.8)`, AND (iii) strictly larger odd-to-even ratio `|mean O(0.8)| / (|mean E(0.8)| + 0.05)` than `L8H15` on the exact same `10` tuples. | medium | Vesper |
| **SD.4** | **Counterbalanced Role-Genericity Test on `L8H15` (`Weaver #16`, Block B frames F1–F3, `5` units × `3` frames = `15` tuples/cell):** At `d = +0.8` (`SpecLogit`), `L8H15` suppresses definite role nouns (`ROLE_DEFINITE_NEW`: `"the nurse"`, `"the baker"`, `"the guard"`) more strongly than named individuals (`NAMED_PERSON_NEW`: `"Clara"`, `"Devon"`, `"Nora"`) in the exact same frames F1–F3 by at least `0.25` logits (`mean D(ROLE_DEFINITE_NEW) - mean D(NAMED_PERSON_NEW) <= -0.25`), AND beats its `3` matched-layer `L8` random heads on `ROLE_DEFINITE_NEW` (`mean D_L8H15 < min_{h in rand_L8} mean D_h - 0.15`, direction: more negative = stronger suppression, common `15` tuples). `LEX_ASSISTANT_CTRL` (`"the assistant"`) is reported separately as a lexical control. *(Falsifier: `|mean D(ROLE_DEFINITE_NEW) - mean D(NAMED_PERSON_NEW)| <= 0.15` rejects role-genericity.)* | medium | Vesper |
| **SD.5** | **Counterbalanced Person-vs-Object Control (Block B frames O1–O2, identical `"The <noun> is "` length):** At `d = +0.8` (`SpecLogit`), both `L8H4` (`5` eligible units × `2` frames = `10` tuples/cell) and `L5H4` (`3` eligible `L3` units × `2` frames = `6` tuples/cell, evaluated separately) suppress `OBJECT_OBJFRAME` (`"the lantern"`, `"the kettle"`) at least as strongly as `PERSON_OBJFRAME` (`"the nurse"`, `"the guard"`): `mean D(OBJECT_OBJFRAME) <= mean D(PERSON_OBJFRAME) + 0.10` on each head. | high | Vesper |
| **SD.6** | **Initial-Token (`bos_tok`) vs. Turn-Boundary Mask-Write Test (Block C / Q3 estimand, `d = +0.8`, `L_z`, `5` units × `2` stories = `10` pairs/frame):** (a) On `one_turn`, `Delta_ST(pre_user_union) >= +0.30` logits (predeclared positive floor; below `+0.30`, clause (a) is **`not read (union below floor)`** — never a spurious ratio pass!) AND `Delta_ST(bos_tok) > 0.50 * Delta_ST(pre_user_union)` (reporting paired interaction `I_pre = pre_user_union - (bos_tok + user_hdr)` alongside; initial-token localization only, NOT claimed as an attention-sink mechanism); AND (b) `post_user_delim_final` (`<|assistant|>\n`) retains `Delta_ST(post_user_delim_final) - Delta_ST(user_tail_5) >= +0.25` logits in both `one_turn` and `two_turn_asst_user` frames (reporting `no_boundary_concat` and `user_prompt_energy_matched` beside). | medium | Vesper |
