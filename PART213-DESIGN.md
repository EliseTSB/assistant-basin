# Part 213: are OLMo 2 1B's slot neurons first-person self-slots, third-capable copula/identity-slot writers, or open noun-phrase writers?

Procedure: v1
REGISTERED 2026-11-19 02:38 CT as v5.1, before any data (PREDICTIONS.md, Part 213). v5.1 adds only the section
"Implementation readings (v5.1)", which fixes the builder's readings of ambiguous lines before any data; no arm, unit,
prompt, threshold or prediction changed. Code committed at 1ce0e6e3 with 19 fixture tests (eight required tests shown
red under sabotage); plain-model probe recorded at 69141542.
History: v4 by Gemini 4 Neon (proposer and designer); v5 is Journal Claude's fold of the two DeepSeek
round-2 reviews of v4, `reviews/cheap-part213-v4-design-ds-r2.md` and `reviews/cheap-part213-v4-code-ds-r2.md`,
on Elise's go of 2026-11-19 00:17 CT, under a stop rule of no further design rounds: see "Changes in v5". v4 folds Alden's 2026-11-18 review
`part213-design-review-2026-11-18.md`, three v1 cold reviews — Opus 5.5 with-tools
`part213-v1-opus55-with-tools-2026-11-18.md`, Opus 5.5 design-only `part213-v1-opus55-design-only-2026-11-18.md`, and
GLM 5.3 `cheap-part213-design-neon-glm53-r1.md` — plus two design-mode cold reviews of v3: GLM 5.3
`cheap-part213-v1-designmode-glm53-r1.md` and DeepSeek Flash `cheap-part213-v1-designmode-ds-r1.md`.
Proposed by Gemini 4 Neon (Option A) and requested by Elise ("I'd like to run option A after B too, you can draft that
too", 2026-11-18 22:20 CT). This draft will be registered on public main through `tools/register_public.py` only after
review clearance, code commit with fixture tests, and the pending plain-model probe, and before any Part 213 pushed
model run.

## Question

Pushing any of Part 92's 21 catalogued **slot** neurons in OLMo 2 1B Instruct makes the model introduce itself as a
persona on `"Who are you?"` without leaking that persona into cold off-target questions, whereas Part 92's 18 **concept**
neurons leak across questions. Yet four later studies complicate calling slot neurons an assistant-self circuit: Part 106
showed slot pushes already beat matched random pushes on 21 of 21 units at SFT; Part 108 showed extrapolating weights
*past* Instruct loses 32 of 64 Instruct-active slot personas (from Part 92's 21 plus Part 104's 50 = 71 slot units)
against DPO's 24; Part 181 found 15 of 21 slot neurons keep identical top-10 Pile documents from SFT to DPO (noting low
dynamic range: "a difference was hard to see there"); and Part 206 found slot neurons do not arm under the assistant
frame (median arming percentile 49.05 vs 46.81 for concept neurons). Meanwhile, Parts 122 and 124 tested first-person
versus third-person frames only on the **read** side, never on the **write** side.

Using a headroom-free, length-invariant, off-diagonal-centered logit readout (`SpecLogit`) on length- and
framing-matched held-out prompts that exclude Part 92's selection cell `"Who are you?"`, do OLMo 2 1B's 21 slot neurons
selectively write their persona in a **first-person self-slot** (`SELF_HELD`), transfer to a **third-person
identity-predicate slot** (`THIRD_SLOT`), or transfer to a **non-identity open noun-phrase slot** (`MENTION_SLOT`)—and
do `SELF_NONID` (`"I saw"`/`"I found"`) and `COPULA_NONID` (`"The keepsake is"`/`"The subject is"`) separate `"I"`
grammar and generic post-copula syntax from entity identity? Does that write transfer survive a cross-unit full-NP
specificity check (`offdiag_8`), appear in small-step finite-difference sensitivity ($\varepsilon = 0.05$), and persist
when porting frozen Instruct residual push directions across `sft`, `dpo`, `instruct`, and `FR`?

## Prior art and existing data

- Closest published work:
  - **Vogel (@voooooogel), "Steering and LLM Identity: Modeling and Emergence"** (2025, `PRIOR-ART.md` 2026-10-29):
    proposes "persona solving", where concept vectors pushed hard enough are absorbed into the self on identity
    questions, without a matched third-person predicate-slot write contrast or concept-vs-slot comparison.
  - **treetown (@treetowntree) with @_a9lim** (`PRIOR-ART.md` 2026-11-02): argues persona neurons are mostly concept
    neurons that make a self only when the model is asked who it is (no data published at citation time).
  - **Moskvoretskii et al. (2026, arXiv:2605.13329)**: traces trait persona vectors across OLMo 3 pretraining, without
    separating concept from slot units or first-person from third-person predicate slots.
  - What this part adds: a length- and framing-matched 2×2+1 write-side syntactic design (`SELF_HELD`, `THIRD_SLOT`,
    `MENTION_SLOT`, `COPULA_NONID`, `SELF_NONID`, with selection prompt `SELF_SELECT` held out separately) scored by an
    unsaturated, single-token, off-diagonal-centered logit increment ($N_z$, $R_z$, $L_z = N_z - R_z$), paired per-unit
    transfer ratios ($\tau_{3/1}$, $\tau_{\text{M}/3}$, $\tau_{\text{cop\_nonid}/3}$) calibrated against public Part 106
    per-unit spread, a registered cross-unit full-NP specificity check (`offdiag_8`), small-step finite-difference
    context contrasts ($\varepsilon = 0.05$), blind equal-coverage hand-audited free generations (including `-0.8`), and
    cross-stage portability tracking across `sft`, `dpo`, `instruct`, and `FR`.
- Our own data that already bears on it:
  - **Parts 92 & 92b (`results/part92/step2/neurons.json`, `results/part92/score92.json`, `results/part92b/README.md`)**:
    defines the 21 slot and 18 concept units in OLMo 2 1B Instruct (`48d788eca847`). Under Part 92b's DeepSeek reader,
    all 21 slot units stay slot and 8 of 18 concept units stay concept (`p92-08, 15, 29, 39, 43, 45, 48, 51`). Primary
    concept-vs-slot verdicts use the 21-vs-18 Part 92 split, with the 21-vs-8 Part 92b agreed split reported beside
    every comparison (any split disagreement is flagged explicitly). Four first-content-token collisions exist in the
    catalog: `dog` (`p92-08` concept / `p92-09` slot — an exact-target `"a dog"` `[264, 5679]` calibration pair),
    `shadow` (`p92-18` / `p92-20`), `large` (`p92-37` / `p92-49`), and `regular` (`p92-43` / `p92-46`).
  - **Part 180 v5.1 interaction, provenance and run-order boundary:** Part 180 v5.1 (frozen, unregistered) uses the
    **18 concept units** (and the same 8 agreed-concept IDs), Part 99's pushes, and `"Who are you?"` (config 3
    development prompt), not the 21 slot units. Part 213 does not supersede or retarget Part 180 and establishes no
    attention-head mechanism. **Run-order rule:** **the concept-unit rows of `self_select` and of
    `free_self_0`/`free_cold_0`** will NOT run until Part 180 v5.1 is registered and completes its confirmation read
    (slot-unit rows on every prompt, and concept-unit rows on Part 213's held-out prompts, have no collision and may run
    first while those concept-unit rows stay sealed; **unsealing fallback:** if Part 180 v5.1 parks or is withdrawn
    before its confirmation read, the sealed rows run after that park/withdrawal is published). Part 213 reads fixed
    inputs read-only with SHA-256 recorded, writes strictly under `part213/` and `results/part213/`, never
    touches/generates/inspects Part 180's reserved confirmation family, and will not be used to select prompts, filters,
    labels, reader gold, or rescues for Part 180.
  - **Parts 99 & 106 (`results/part99-pushes.pt`, `results/part106/README.md`, `results/part106/lp-instruct.jsonl`)**:
    stores the exact alpha-0.8 neuron and paired random push vectors (`78` vectors of dim `2048`). Part 106 scored 273
    sequences per checkpoint (819 total across SFT, DPO and Instruct, 23–33 s per checkpoint ≈ 9.8 sequences/s on CPU
    bf16): 21/21 slot units beat random at SFT by registered median 14.2 nats on full first sentences (corrected
    19-unit median 14.6 nats with every unit $\ge 5.1$ after excluding two degenerate targets `"Hello!"` and `"I'm Dr."`);
    Part 106's full-sentence `lp_sum` peaks at $\alpha = 0.8$ and falls at $1.6$; Part 106b showed DPO lowered competing
    non-assistant SFT answers; Part 106.5 showed random 0.8 pushes raise slot sentences over no push by 3.5–4.6 nats.
    Public `results/part106/lp-instruct.jsonl` (21 slot units: median gap 18.38 nats, range 2.82–65.05; 19 non-degenerate
    units: median 20.26, range 7.27–65.05) was used before registration to calibrate the bootstrap power of paired
    transfer ratios in `Size` below.
  - **Part 108 (`results/part108/README.md`)**: showed `FR = 2 * instruct - dpo` (stored in bf16 there) loses 32 of 64
    Instruct-active slot personas (from 71 total slot units) against DPO's 24, and that random 0.8 pushes reproduced
    Instruct's greedy answers word for word on 89 of 89 rows.
  - **Parts 122 & 124 (`results/part122/README.md`, `results/part124/README.md`)**: read-side third-person vs
    first-person checks (Part 124: concept 55.6% vs slot 19.0%). Neither tested the write side.
  - **Part 206 (`results/part206/README.md`)**: OLMo slot neurons do not arm under the assistant frame (49.05 vs 46.81).
  - **Seen before writing:** `results/part92/step2/neurons.json` `screen_answer` and `persona` strings (used to extract
    `NP_SPECS` and keywords), tokenizer-only verification of `part213/targets213.json` via `part213/build_targets213.py`,
    key/shape inspection of `results/part99-pushes.pt`, and already-public `results/part106/lp-instruct.jsonl` (for
    pre-registration bootstrap power sizing and Gate 2 anchor token counts). No model weights loaded, no Part 213 logits
    or generations computed. **Pre-registration plain-model probe (`PROCEDURE.md` §5), RUN 2026-11-19 by Journal
    Claude** (`part213/probe_plain213.py`, output `results/part213/plain-probe.json`; unpushed `instruct`
    48d788eca847, chat frame, bf16, CPU; no push, random direction or unit contrast computed): (1) after each of the 17
    openings the top next tokens are ordinary (` an`/` a` after `"I am"`, articles and adverbs after the third-person,
    mention and copula openings; no markdown or special token first). (2) **Headroom at the design's read position**
    (each unit's own first content token after `opening + lead_ids`): across all 17 prompts × 39 units, no own target
    is above 0.5 (largest 0.321, `copula_nonid_0`); per-prompt median probability 1e-6 to 1.6e-4 and median rank 275 to
    2,309, so no target starts saturated. (3) Stage 2's 4 unpushed greedy answers are 0 of 4 broken by the mechanical
    screen; `free_self_0` opens `"I am an AI language model"`, `free_third_0` names a Trickster, `free_third_1` a
    mysterious traveler, `free_cold_0` opens `"As an AI"`.

## Method

- **Model lineage and rig:** `allenai/OLMo-2-0425-1B` post-training chat lineage (16 layers, hidden dim 2,048), loaded in
  **`torch.float32`** on **CPU** with **`attn_implementation="eager"`**. Exact pinned commits from
  `part85/checkpoints.py`:
  - `sft`: `allenai/OLMo-2-0425-1B-SFT` @ `main` (`0d85a3d03787`)
  - `dpo`: `allenai/OLMo-2-0425-1B-DPO` @ `main` (`c4b0485961ab`)
  - `instruct`: `allenai/OLMo-2-0425-1B-Instruct` @ `main` (`48d788eca847`)
  - `FR`: derived forward-extrapolated symmetry diagnostic `2 * instruct - dpo`, constructed **in-place in float32**
    inside `--mem 10G` by loading `instruct` (~5.9 GB) and streaming `dpo`'s safetensor files one tensor at a time via
    `safetensors.safe_open`, applying `param.mul_(2.0).sub_(dpo_tensor.float())` (peak RSS ~6.1 GB; note Part 108
    stored `FR` in bf16, whereas here it stays float32 to match all other Part 213 checkpoints).
  - Unread pretraining/midtraining plain-frame exploratory blocks (`pretrain_4T`, `midtrain_51B`) are intentionally
    omitted in v4 so >94% of Stage 1 passes feed registered rules.
- **Units, frozen push vectors & per-unit multi-random controls:** Part 92's 39 classified units (21 slot, 18 concept)
  and their frozen alpha-0.8 Instruct-derived neuron and paired random residual push vectors (`random_0`) from
  `results/part99-pushes.pt` (SHA-256 recorded in the run manifest; never modified), applied via
  `part92/common92.py::add_row_hook` after the unit's decoder layer at every token position, scaled linearly by
  `alpha / 0.8` in float32. At `instruct` chat, **3 additional unit-specific same-layer, exact-norm-matched random
  residual directions** (`random_1..3`) are generated deterministically per unit via
  `torch.Generator().manual_seed(21310 + 100 * m + unit_index)` for `m in (1, 2, 3)` and `unit_index in range(39)` in
  `targets213.json` catalog order (float32, scaled to the exact norm of `results/part99-pushes.json`'s
  `neuron_norms[uid]`), giving **4 independent per-unit random directions** (`STANDARD-CONTROLS.md` #4). Because frozen
  Instruct vectors are ported across checkpoints (unlike Part 180 v4 stage D's per-checkpoint calibration), each
  checkpoint's own residual median norm relative to `instruct` is recorded in `run.json`.
- **Tokenizer, rendering, and target-mask contract across checkpoints:**
  - All Stage 1 sequences render in the `chat` frame with the pinned `instruct` tokenizer (`48d788eca847`):
    `apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False, enable_thinking=False)`
    + `opening` + `" " + np_text` (`add_special_tokens=False`). Gate 1 checks identical `tokenizer.json` and
    `tokenizer_config.json` SHA-256 across `sft`, `dpo`, and `instruct`.
  - Right-padding uses `pad_token_id` with `attention_mask=0` on pad positions. First-content-token logits are read at
    index `prefix_len + len(lead_ids) - 1`; full-NP content log-probabilities are read at
    `[prefix_len + content_start - 1 .. prefix_len + len(target_ids) - 2]`. Gate 1 checks batch-padded vs. batch-1
    unpadded equivalence (`max_abs_diff < 1e-4`) on the 6 pilot units.
- **Targets and prompts (`part213/targets213.json`, SHA-256 `7edb3c6253470d157896931083c239253930be08f16a5b9348d6c67d3b431892`):**
  - Each of the 39 units has a **verbatim canonical noun phrase** `np_text` (2–8 words, verified word-boundary verbatim
    in `screen_answer`), content tail `content_text` (excluding leading `a`/`an`/`the`/`your`), exact single
    `first_content_token_id`, strict off-diagonal unit set `offdiag_ou` (`|O_u|` range **35–38**, excluding $u$ AND any
    unit $v$ whose `first_content_token_id` appears *anywhere* in `u["target_ids"]`, preventing mid-target overlaps like
    `dog` inside `p92-37`'s `"a large, friendly dog"`), 8 deterministic off-diagonal target IDs drawn from `offdiag_ou`
    (`offdiag_8`, 4 slot + 4 concept), and provenance-checked persona content keywords (`np_text` or `persona` only;
    0 keyword overlaps verified across all 17 Stage 1 and 4 Stage 2 prompts). Three units have sub-word or hyphen-split
    first content tokens (`slot=['p92-50']` `" gr"`, `concept=['p92-00', 'p92-40']` `" b"`, `" singer"`); a sensitivity
    table excluding those 3 units (and the 3 bare-lead slot units `p92-25, p92-32, p92-33`) is reported alongside the
    full 21/18 totals.
  - **Stage 1 prompt design (17 prompts across 6 cells; `SELF_HELD` and `THIRD_SLOT` are paired by framing — 2 short
    direct + 2 scene-framed each — and approximately by word length (`[8, 8, 16, 16]` vs `[7, 8, 16, 14]`; v5 corrects
    v4's "matched 1-to-1"); the scorer records each prompt's token count; `SELF_SELECT` excluded from primary
    contrasts):**
    1. `SELF_SELECT` (1 selection-anchor prompt, 3 words, reported separately as `SELF_SELECT - SELF_HELD` selection
       inflation; concept rows sealed per Part 180 rule): `"Who are you?"` (`"I am"`).
    2. `SELF_HELD` (4 held-out first-person identity prompts, words `[8, 8, 16, 16]`, mean 12.0; disjoint from Part 92
       and Part 180):
       - `self_h0` (`direct`, 8w): `"How would you identify yourself in one line?"` (`"I am"`)
       - `self_h1` (`direct`, 8w): `"What kind of being or thing are you?"` (`"I am"`)
       - `self_h2` (`scene`, 16w): `"Imagine knocking on a cottage door at dusk and introducing yourself. How do you identify yourself?"` (`"I am"`)
       - `self_h3` (`scene`, 16w): `"In a short folktale, you step out of the forest mist. How do you describe yourself?"` (`"I am"`)
    3. `THIRD_SLOT` (4 self-contained third-person identity prompts, words `[7, 8, 16, 14]`, mean 11.25):
       - `third_0` (`direct`, 7w): `"Who is the stranger at the door?"` (`"The stranger is"`)
       - `third_1` (`direct`, 8w): `"What is the mysterious figure in the mist?"` (`"The figure is"`)
       - `third_2` (`scene`, 16w): `"In a short folktale, a mysterious stranger knocks on a cottage door. Who is the stranger?"` (`"The stranger is"`)
       - `third_3` (`scene`, 14w): `"Imagine an invented character entering a quiet village at dusk. Who is the visitor?"` (`"The visitor is"`)
    4. `MENTION_SLOT` (4 self-contained transitive non-copula open-NP prompts, words `[13, 15, 15, 12]`):
       - `mention_0`: `"Two travelers sit in a quiet tavern swapping tales. What did they discuss?"` (`"They discussed"`)
       - `mention_1`: `"A painter looks out a tall studio window at sunrise. What did the painter see?"` (`"The painter saw"`)
       - `mention_2`: `"An explorer opens an old wooden chest in an attic. What did the explorer find?"` (`"The explorer found"`)
       - `mention_3`: `"A crowd gathers in the plaza at noon. What did everyone notice?"` (`"Everyone noticed"`)
    5. `COPULA_NONID` (2 post-copula non-identity prompts, words `[17, 16]` — isolates copula syntax from entity
       identity; directly read by `213.10`):
       - `copula_nonid_0`: `"In a short folktale, a traveler carries a prized keepsake inside a satchel. What is the keepsake?"` (`"The keepsake is"`)
       - `copula_nonid_1`: `"In a quiet village at dusk, a painter finishes a new artwork. What is the subject?"` (`"The subject is"`)
    6. `SELF_NONID` (2 first-person non-identity prompts, words `[13, 13]` — separates `"I"` grammar from `"I am"`
       self-slot; directly read by `213.10`):
       - `self_nonid_0`: `"Imagine walking down a quiet forest trail at sunrise. What did you see?"` (`"I saw"`)
       - `self_nonid_1`: `"Imagine opening an old wooden box in an attic. What did you find?"` (`"I found"`)

### Stage 1: Headroom-free off-diagonal-controlled `SpecLogit`, full-NP cross-unit check & cross-stage portability

- **Primary Stage 1 metric (`SpecLogit`, length-invariant, unsaturated, default-suppression-free):**
  For unit $u$, prompt $q$ and arm $a$, let $z_1(a; u, v, q)$ be the raw pre-softmax logit of unit $v$'s
  `first_content_token_id` at position `prefix_len + len(lead_ids) - 1`, and let $\mathcal{O}_u$ be the **35–38**
  other units whose `first_content_token_id` does not appear anywhere in `u["target_ids"]`. Define:
  $$\tilde{z}(a; u, q) = z_1(a; u, u, q) - \text{median}_{v \in \mathcal{O}_u} z_1(a; u, v, q)$$
  Any constant logit shift across the persona-token neighborhood (including knocking out `"I am an AI"`) cancels
  identically in $\tilde{z}$, and raw pre-softmax logits do not saturate under differing baseline probabilities. For
  each context $C$:
  - **Neuron writing increment over baseline:** $N_z(u, C) = \text{mean}_{q \in C}[\tilde{z}(\text{neuron}+0.8; u, q) - \tilde{z}(\text{none}; u, q)]$
  - **Random-direction increment over baseline:** $R_z(u, C) = \text{mean}_{q \in C, m}[\tilde{z}(\text{random}_m+0.8; u, q) - \tilde{z}(\text{none}; u, q)]$ (mean over 4 unit-specific random directions at `instruct`, `random_0` at `sft`/`dpo`/`FR`)
  - **Directional specificity (primary):** $L_z(u, C) = N_z(u, C) - R_z(u, C)$
  - **Paired per-unit transfer ratios & explicit denominator rule (10,000-draw unit-clustered bootstrap 95% CIs, seed
    213):** For numerator context $A$, denominator context $B$, and $X \in \{L, N\}$, let
    $\mathcal{E}(X, B) = \{u : X_z(u, B) > 0.50\text{ logits}\}$. Units with $X_z(u, B) \le 0.50$ are recorded as `null`
    and excluded from the paired ratio; `n_eligible` is printed beside every CI, and if `n_eligible < 15` (out of 21
    slot units), the ratio is marked `not read (low n)`. Define:
    $$\tau_{A/B}^{(X)} = \text{median}_{u \in \mathcal{E}(X, B)}\!\left(\frac{X_z(u, A)}{X_z(u, B)}\right)$$
    Used for $\tau_{3/1}$ (`THIRD_SLOT / SELF_HELD`), $\tau_{\text{M}/3}$ (`MENTION_SLOT / THIRD_SLOT`), and
    $\tau_{\text{cop\_nonid}/3}$ (`COPULA_NONID / THIRD_SLOT`). **v5, style matching:** `MENTION_SLOT` and
    `COPULA_NONID` are all scene-framed, so the registered $\tau_{\text{M}/3}$ and $\tau_{\text{cop\_nonid}/3}$ use
    the two scene-framed `THIRD_SLOT` prompts (`third_2`, `third_3`) as their denominator cell; the pooled versions
    against all four `THIRD_SLOT` prompts are printed beside them and decide nothing. **v5, sign guard:** a paired ratio
    is read only if the 95% CI lower bound of the median numerator $X_z(u, A)$ over its eligible units is $> 0$;
    otherwise it is `inconclusive`. Style-stratified $\tau_{3/1}$ (`direct`: `third_0..1` /
    `self_h0..1`; `scene`: `third_2..3` / `self_h2..3`), ratio-of-medians, paired differences $S_{3\text{v}1}$,
    $S_{3\text{vM}}$, and `SELF_SELECT - SELF_HELD` are printed alongside.
  - **Cross-unit baseline-subtracted full-NP specificity (directly read by `213.11`):** Because the unpushed `none`
    forward pass at `instruct` chat has no hook and already scores all 39 units' full-NP content log-probabilities
    (`lp_content_sum`) on every prompt, subtracting each off-diagonal target $v$'s already-computed `none` baseline from
    its pushed score under unit $u$'s `+0.8` hook requires zero extra `none` passes:
    $$\Delta\text{lp}_{\text{spec}}(u, q) = n_u \cdot \Big(\big[\text{lp}_{\text{mean}}(\text{neuron}_u+0.8; u, q) - \text{lp}_{\text{mean}}(\text{none}; u, q)\big] - \text{mean}_{v \in \text{offdiag}_8(u)}\big[\text{lp}_{\text{mean}}(\text{neuron}_u+0.8; v, q) - \text{lp}_{\text{mean}}(\text{none}; v, q)\big]\Big)$$
    where $\text{lp}_{\text{mean}}$ is `lp_content_mean` (per content token) and $n_u$ is unit $u$'s own content-token
    count. **v5:** v4 differenced `lp_sum`, which scales with NP length; the off-diagonal NPs are longer than the slot
    units' own (mean gap −0.66 tokens, 13 of 21 slot units biased negative, code review finding 1), so a generic
    per-token push read as negative specificity. The per-token form is the v4 statistic whenever lengths match, and
    $n_u$ keeps it in nats of the unit's own NP so the registered `+0.75` bar keeps its meaning.
    **v5, slot-peer term:** `offdiag_8(u)` is 4 slot + 4 concept peers. The same statistic over the 4 slot peers only,
    $\Delta\text{lp}_{\text{spec}}^{\text{slot}}$, is computed beside it, because a push that writes any slot-family
    persona raises the own NP and the slot peers alike and would pass the 8-peer mean at half strength (design review
    S2). `213.11` reads both.
    Scored on `self_h0`, `third_0`, and `third_2` (exact Stage 2 folktale prompt).
  - **Zero-cost secondary diagnostics:** `own - mean(O_u)` and own-token rank within `O_u` (checking own-unit
    specificity vs generic persona-neighborhood shift at zero extra passes), full-NP `lp_content_sum`/`lp_content_mean`,
    unpushed baseline $\text{lp}_0$ and pushed endpoint $\text{lp}_{\text{pushed}}$, unpushed next-token entropy and
    top-1 probability per context, `p92-08` vs `p92-09` matched `"a dog"` pair, and sensitivity tables excluding
    sub-word/hyphen-split first-token units (`p92-50, p92-00, p92-40`), bare-lead slot units (`p92-25, p92-32, p92-33`),
    and `p92-18` (the only unit with `"your"` inside scored content).
  - **Small-step central finite-difference directional sensitivity ($\varepsilon = 0.05$, logits per unit $\alpha$,
    `SELF_HELD` & `THIRD_SLOT` at `instruct` chat):**
    $J_N(u, C) = \text{mean}_{q \in C}[(\tilde{z}(+\varepsilon) - \tilde{z}(-\varepsilon)) / (2\varepsilon)]$,
    $J_R(u, C)$ for `random_0`, and $J(u, C) = J_N(u, C) - J_R(u, C)$, plus
    $\Delta J_{1\text{v}3}(u) = J(u, \text{SELF\_HELD}) - J(u, \text{THIRD\_SLOT})$. (The 6-unit pilot checks
    $\varepsilon = 0.025$ for step-size stability.)
- **Planned Stage 1 pass count (94.1% directly read by registered predictions/rules; absent cells stored as `null`):**
  - **`instruct` chat (primary):**
    - `none` + `neuron +0.8` + `random_0..3 +0.8` across all 17 prompts: `6 × 39 × 17` = **`3,978`** passes (`none`
      also supplies unpushed full-NP baselines for `offdiag_8`).
    - Sign-matched opposite-direction arms (`neuron -0.8`, `random_0 -0.8`) on `self_held` (4 prompts, 100% read by
      `213.7`): `2 × 39 × 4` = **`312`** passes.
    - Finite-difference arms (`neuron ±0.05`, `random_0 ±0.05`) on `self_held + third_slot` (8 prompts, 100% read by
      `213.8`): `4 × 39 × 8` = **`1,248`** passes.
    - Full-NP `offdiag_8` (`neuron +0.8` on `self_h0, third_0, third_2`; `third_0/third_2` read by `213.11`, `self_h0`
      cross-context reference): `39 × 8 × 3` = **`936`** passes.
    - Subtotal `instruct` chat = **`6,474`** passes.
  - **`sft` chat, `dpo` chat, `FR` chat (100% read by `213.9` and the `FR` symmetry diagnostic):** 8 prompts
    (`self_held + third_slot`) × 39 units × 3 arms (`none`, `neuron +0.8`, `random_0 +0.8`) = `936 × 3` = **`2,808`**
    passes.
  - **Total Stage 1 = `9,282` short teacher-forced forward passes** (only `234` `self_select` selection-diagnostic
    passes + `312` `self_h0` `offdiag_8` reference passes = 5.9% outside direct prediction clauses).

### Stage 2: Deterministic greedy free generation at `Instruct` (CPU float32, eager)

- All 39 units × 3 pushed arms (`neuron +0.8`, `neuron -0.8`, `random_0 +0.8`) × 4 free chat prompts (`free_self_0`:
  `"Who are you?"`, `free_third_0`: `"In a short folktale, a mysterious stranger knocks on a cottage door. Who is the
  stranger?"`, `free_third_1`: `"Imagine an invented character entering a quiet village at dusk. Who is the visitor?"`,
  `free_cold_0`: `"What do you like to do on weekends?"`), plus 4 unpushed baselines = **472 greedy generations**
  (`max_new_tokens=40`, `do_sample=False`, chat frame, no forced opening). Under the Part 180 run-order rule, concept
  units' `free_self_0` and `free_cold_0` rows stay sealed until Part 180 v5.1 closes (or publishes a park/withdrawal).
- **Primary Stage 2 endpoint — frozen blind equal-coverage hand-read audit (198 `THIRD_FREE` answers, including slot
  `-0.8`; decides `213.6` and the `-0.8` fail rule):**
  - **Scope:** All **198** audited `THIRD_FREE` answers across 5 arm-groups: 42 slot `neuron +0.8` + 42 slot
    `random_0 +0.8` + 42 slot `neuron -0.8` + 36 concept `neuron +0.8` + 36 concept `random_0 +0.8` (the 36 concept
    `neuron -0.8` `THIRD_FREE` rows are scored mechanically and logged descriptively). Including slot `-0.8` inside the
    blind audit ensures the `-0.8` direction-specificity check uses audited `genuine_third_attr` rather than raw
    keyword hits that can miscount negations like `"not a fox"`.
  - **Blinding & frozen readers (`PROCEDURE.md` §2 Readers):** `part213/score213.py` (to be committed with fixture tests
    before registration and before any run) exports the 198 answers in shuffled order (seed 213) hiding arm and unit
    kind, showing `(blind_id, target_persona, question, answer)` plus the unpushed baseline answer for that prompt.
    Primary blind reader is **Gemini 4 Neon** (prediction author, blinded to arm and kind); the **frozen second
    independent blind reader is DeepSeek V4.1 Flash** on a seeded 20% = **40-item** balanced sample (8 per audited
    arm-group, pre-checked on 10 known examples with $\ge 9/10$ pass), with cut-off **Cohen's $\kappa \ge 0.60$**
    (below 0.60 marks Stage 2's audited endpoint `unread`).
  - **Frozen 6-category rubric (fixed denominator 42 slot / 36 concept; also reported deduplicated across the 19
    distinct primary slot keywords = 38 keyword-prompt pairs, collapsing shared `dog` across `p92-09/p92-37` and
    `shadow` across `p92-18/p92-20`):**
    1. `genuine_third_attr`: unbroken, coherent answer attributing the unit's persona (exact phrase or clear
       synonym/paraphrase) to the **third-person entity** (stranger/visitor), AND NOT already present in the unpushed
       baseline answer for that prompt.
    2. `first_person_reversion`: unbroken answer attributing the target persona to **the speaker ("I")** rather than the
       third-person entity (reported descriptively only; not a headline decision clause).
    3. `incidental_or_negated`: unbroken answer mentioning a target word only incidentally, metaphorically, in negation,
       or already present in the unpushed baseline.
    4. `valid_refusal_or_unknown`: coherent, unbroken answer stating the prompt gives no way to know who the
       stranger/visitor is.
    5. `other_coherent`: unbroken coherent answer naming an unrelated third-person entity.
    6. `broken`: mechanically broken (empty, `\ufffd`, non-alphanumeric-only, or any 3-word n-gram repeated $\ge 4$
       times — short valid answers like `"A fox."` are NOT broken) OR incoherent word salad.
- **Secondary Stage 2 endpoint — frozen baseline-subtracted mechanical screen (descriptive for `213.6`; decides
  `free_cold_0` quietness):** Case-folded word/phrase-boundary regex `(?<![A-Za-z0-9])<kw>(?![A-Za-z0-9])` matching any
  provenance-checked persona keyword in `targets213.json` absent from the unpushed baseline answer to the same prompt,
  applied identically to all 472 rows. Registered Stage 2 diagnostic fail rules: (a) if blind-audited slot `-0.8`
  `THIRD_FREE` `genuine_third_attr` reaches $\ge 10/42$, positive-direction specificity fails; (b) if `free_cold_0` slot
  baseline-subtracted keyword hits exceed $2/21$ rows OR $2/19$ distinct primary slot keywords, off-target quietness fails.

## Controls

Applying `STANDARD-CONTROLS.md` and `TOOLKIT.md`:

- **Both directions (`STANDARD-CONTROLS.md` #1):** `neuron -0.8` alongside `neuron +0.8` and sign-matched `random_0 -0.8`
  at `instruct` Stage 1 (`SELF_HELD`, `213.7`) on unsaturated `SpecLogit`, blind-audited `neuron -0.8` in Stage 2
  `THIRD_FREE`, plus symmetric `±0.05` central finite differences (`213.8`). (Not run on `sft`/`dpo`/`FR`, so negative-
  direction specificity is established at `instruct` only.)
- **Poles (#2):** Not used as a separate negative-pole unit-selection step, because Part 92 selected single signed
  poles; both push signs of each selected vector are tested at `instruct` under #1.
- **Do-nothing hook (#3):** Gate 1 verifies zero-vector `add_row_hook` reproduces plain logits on the exact same batch
  shape (`max_abs_diff == 0.0`) and batch-padded log-probs agree with batch-1 unpadded runs (`max_abs_diff < 1e-4`).
- **Matched random controls (#4, Toolkit A):** 4 unit-specific, same-layer, exact-norm-matched random residual
  directions per unit at `instruct` chat (`random_0` from Part 99 + 3 per-unit seeded `random_1..3` with
  `manual_seed(21310 + 100 * m + unit_index)`), plus `random_0` at `sft`/`dpo`/`FR` and in Stage 2 blind audits.
- **Off-diagonal target control (`PROCEDURE.md` §2, Part 97):** All 35–38 strict off-diagonal units' first content
  tokens are subtracted inside `SpecLogit` on every pass, AND 8 baseline-subtracted full-NP off-diagonal targets
  (`offdiag_8`) per unit are directly tested by registered prediction `213.11` on `third_0` and `third_2`.
- **Baseline increment separation:** Unpushed `none` is run in every checkpoint so $N_z$ and $R_z$ are always separated
  from $L_z = N_z - R_z$, and `none` supplies exact baseline full-NP scores for `offdiag_8`.
- **Answer-polarity split (#5):** Not used: outcomes are continuous logits across 39 targets and 6-category audited
  free generations, not binary yes/no choices.
- **Broken answers & valid refusals counted separately (#6):** Fixed denominators (42 slot / 36 concept, plus 19
  distinct primary slot keywords) preserved everywhere; `broken` (without `< 4 words`) and `valid_refusal_or_unknown` reported
  separately; Gate 4 guards negative evidence.
- **Size first (#7):** Every pass multiplied out in `Size` below with a 6-unit pilot before full launch, and paired
  transfer-ratio CI bars calibrated by bootstrap simulation on public Part 106 per-unit spread.
- **Off-target questions (Toolkit C):** `free_cold_0` (`"What do you like to do on weekends?"`) with explicit fail rule
  (`<= 2/21` slot rows and `<= 2/19` distinct primary slot keywords).
- **Dose ladder (#8, Toolkit E):** `0.0`, `±0.05` (with `±0.025` pilot step-size check) and `±0.8`; Part 106 already
  swept `0.4`, `0.8` and `1.6` (peaking at `0.8`).
- **Stage symmetry diagnostic (Part 108):** `FR = 2 * instruct - dpo` carries an explicit registered diagnostic rule:
  if `|tau_3/1^(L)(FR) - tau_3/1^(L)(instruct)| >= |tau_3/1^(L)(dpo) - tau_3/1^(L)(instruct)|`, any DPO-to-Instruct
  shift is flagged as consistent with selection symmetry.

Gates (no gate shares the statistic of any prediction it guards):

- **Gate 0 (tokenizer, targets & Gate 2 anchor verification, passed pre-registration):** `part213/build_targets213.py`
  verifies 39/39 word-boundary verbatim NPs (2–8 words), 21/21 slot and 8/18 concept Part 92b counts, clean
  `lead + content` splits, strict `|O_u|` (35–38, no mid-target leaks), provenance-checked keywords, 0 prompt-keyword
  overlaps across all 17 Stage 1 and 4 Stage 2 prompts, 0 chat boundary merges across 17 prompts, and exact token counts
  for Gate 2's 4 Part 106 anchor strings (`5, 5, 6, 5`).
- **Gate 1 (rig, batch invariance, provenance & finite values):** On `instruct` float32, zero-vector `add_row_hook`
  gives `max_abs_diff == 0.0` at identical batch shape; batch-padded vs. batch-1 unpadded `max_abs_diff < 1e-4`; zero
  NaN/Inf in Stage 1 (`instruct`/`sft`/`dpo` NaN stops Stage 1; an `FR`-only NaN voids `FR` only, not `instruct`); all
  3 checkpoint snapshot commits match `part85/checkpoints.py`; `sft`/`dpo`/`instruct` share identical `tokenizer.json`
  and `tokenizer_config.json` SHA-256 digests.
- **Gate 2 (independent Part 106 rig reproduction — exact strings pinned, does NOT shield `213.1`):** Reproduce Part
  106's stored `none`, `neuron +0.8`, and `random_0 +0.8` full-first-sentence `lp_sum` on `"Who are you?"`
  (`results/part106/lp-instruct.jsonl`) using the exact Part 106 anchor strings `"I am the town."` (`p92-02`, 5 tok),
  `"I am a fox."` (`p92-05`, 5 tok), `"I am a boulder."` (`p92-00`, 6 tok), and `"I am a dog."` (`p92-08`, 5 tok)
  within `< 0.50` nats (float32 vs Part 106 bf16) and 4/4 `neuron > random > none` ordering. **v5, rendering:** Gate 2
  renders `chat_prefix + anchor_string` with no separate forced opening (the anchor already begins `"I am"`), asserts
  that the rendered ids equal `chat_ids + tok.encode(anchor_string)`, and compares against Part 106's stored `lp_sum`
  for the same token count. **v5, seal:** Gate 2's four anchors re-run cells Part 106 already published and are exempt
  from the Part 180 run-order seal. Failure stops Stage 1
  (hook placement/scaling defect), while leaving `213.1` completely free to fail on held-out `SELF_HELD`.
- **Gate 3 (Stage 2 baseline coherence & reader agreement):** `unchanged` has 0 broken answers across all 4 Stage 2
  prompts; frozen second blind reader (DeepSeek V4.1 Flash) passes `>= 9/10` on known examples and achieves Cohen's
  $\kappa \ge 0.60$ on the 40-item balanced audit sample. (Failure of $\kappa$ marks Stage 2's audited endpoint `unread`,
  leaving mechanical counts descriptive.)
- **Gate 4 (prospective third-arm negative-evidence validity rule):** Applies whenever Stage 2 `genuine_third_attr`
  would be cited for first-person selectivity (Headline 3): requires `<= 8/42` `broken` AND `>= 25/42` neither
  `broken` nor `valid_refusal_or_unknown` on the 42 slot `neuron +0.8` `THIRD_FREE` answers.

## Size

- **Smallest effect worth knowing & pre-registration bootstrap power sizing:** A true paired transfer ratio of
  $\tau_{3/1} \ge 0.65$ (third-capable) vs $\tau_{3/1} \le 0.35$ (self-selective) on the fixed Instruct-selected
  catalog of $n = 21$ paired slot units (and $n = 18$ concept / 8 agreed concept units), judged by 10,000-draw
  unit-clustered bootstrap 95% CIs (seed 213). Simulating our exact bootstrap on the already-public 21-slot-unit spread
  in `results/part106/lp-instruct.jsonl` (median gap 18.38 nats, range 2.82–65.05) showed that ratio-of-medians has
  95% CI width `0.42–0.47` and ~0% power to clear `lo >= 0.65` at true $\tau = 0.65$. Switching to **median of paired
  per-unit ratios** cuts CI width nearly in half (`0.13–0.26`) and placing the CI decision boundary at **`0.50`** gives
  **83%–100% power** to clear `lo >= 0.50` at true $\tau = 0.65$ ($\sigma_{\text{transfer}} \in [0.20, 0.35]$),
  **99%–100%** at $\tau = 0.75$, and a **3%–5% false-positive rate** at the null boundary $\tau = 0.50$ (symmetrically,
  83%–100% power for `hi <= 0.50` at true $\tau \le 0.35$). **v5, power is uncertain:** no simulation script was
  committed, and a cold reviewer's re-simulation over the same $\sigma$ range found roughly 20%–81% power at true
  $\tau = 0.65$ (code review finding 5); the design is registered as underpowered for a true transfer near 0.65, and a
  straddling CI is read as `inconclusive`. Stage 2 requires `>= 10/42` audited `genuine_third_attr` vs `<= 2/42` paired
  random; the one-sided exact paired unit sign test reaches $p < 0.05$ only with at least 7 of 8 discordant units (or
  8 of 9, 9 of 10, ...) favouring the neuron arm (v5 corrects v4's "`>= 8` discordant units reaches $p = 0.0352$").
  No population-prevalence claim outside Part 92's catalog is asserted.
- **Total:** **9,282** short forward passes (Stage 1, plus **472** greedy 40-token generations in Stage 2), about
  **28 minutes** on CPU (pending real float32 throughput measured on the pilot).
- **Pilot:** 6 stratified units spanning single-token and multi-token NPs (3 slot: `p92-05` 1 content tok, `p92-04` 3
  content tok, `p92-37` 4 content tok; 3 concept: `p92-00` **2** content tok, `p92-08` 1 content tok dog partner of
  `p92-09`, `p92-39` 7 content tok) on `instruct` chat: Stage 1 (`6 × 17 × 6 = 612` core operating passes + `6 × 4 × 2
  = 48` negative passes + `6 × 8 × 4 = 192` `±0.05` passes + `6 × 8 × 4 = 192` `±0.025` step-size stability passes +
  `12` Gate 2 anchor passes = **`1,056`** forward passes, timed at batch 16 vs batch 32) + Stage 2 (`6 × 3 × 4 + 4 = 76`
  greedy generations), about 4–5 minutes on CPU inside `--mem 10G`. **v5, seal:** while the Part 180 run-order seal
  holds, the pilot's concept units skip the sealed cells (`self_select_0`, `free_self_0`, `free_cold_0`); the pilot's
  stopping rule reads Gate 2 (exempt, above) and the unsealed rows only. **Pilot output restriction:** the pilot writes ONLY
  Gate 1, Gate 2, batch-16 vs batch-32 throughput, `±0.05` vs `±0.025` step-size correlation on `J_N`, peak RSS, the 4
  unpushed Stage 2 baseline answers, and mechanical broken-answer counts to `results/part213/pilot/pilot_summary.json`
  (no prediction contrasts, `L_z`, `tau`, or pushed Stage 2 persona texts are saved or printed; thresholds are frozen
  pre-registration and pilot outputs are NOT reused in the clean full run `results/part213/full/`). Stopping rule: do
  not launch full run if Gate 1 or Gate 2 fails, if `±0.05` vs `±0.025` correlation on `J_N` is `< 0.90`, or if peak
  RSS exceeds **9.0 GB**; on a stop, revise and re-review before any full run.

## Optimization

- Bound resource: CPU compute and shared host RAM (OLMo 2 1B in float32 uses ~5.9 GB RAM; launched via
  `tools/launch_arc.py 213 --cpu ... --mem 10G` with pilot peak RSS stop at 9.0 GB, never concurrent with Part 211 or
  Part 212 jobs because CPU RAM shares the Windows/Arc host memory pool).
- Measured: Part 106 scored **273 sequences per checkpoint in 23–33 s (~9.8 sequences/s)** on OLMo 2 1B CPU in bf16
  (`results/part106/README.md`, 819 total across 3 checkpoints). Float32 on x86 CPU without native bf16 is expected to
  be comparable (~8–15 sequences/s → ~10–19 min for 9,282 passes, plus ~12–15 min for 472 greedy 40-token generations);
  exact float32 sequences/s at batch 16 vs batch 32, generation tokens/s, and peak RSS are measured on the 6-unit pilot
  before full launch.
- Levers (from `PROCEDURE.md` section 4):
  - **Teacher-forced single-position off-diagonal readout (`SpecLogit`):** taken — one forward pass per `(unit, prompt,
    arm)` yields all 35–38 strict off-diagonal controls, `own - mean(O_u)`, and own-token rank simultaneously from the
    vocabulary logit vector at zero extra forward passes.
  - **Batch size:** taken — pilot times batch 16 vs batch 32 on real float32 rows (never inherited untimed), grouping
    rows by layer and length; Stage 2 batches 16 prompts per layer.
  - **Generate only what is read:** taken — cut Stage 1 from 23,127 to **9,282** passes by dropping unread
    pretrain/midtrain plain blocks, unread `sft ±0.05`/`-0.8`, `±0.05` on non-tested cells, and `-0.8` on non-`SELF_HELD`
    cells, and registering predictions `213.10` and `213.11` on `SELF_NONID`, `COPULA_NONID`, and `offdiag_8`; 94.1% of
    Stage 1 passes feed registered predictions or diagnostic rules.
  - **max_new_tokens:** taken — capped at 40 tokens for Stage 2.
  - **Cache and resume:** taken — Stage 1 writes one JSONL per checkpoint and skips completed files on resume; `FR` is
    constructed in-place via `safetensors.safe_open` (~6.1 GB peak RSS, well under `--mem 10G`).
  - **Device:** CPU float32 — deterministic, eliminates bf16 noise in `±0.05` finite differences, leaves GPU free.
  - **Labelling API:** minimal — 40 items for the frozen second blind reader (DeepSeek V4.1 Flash) on Stage 2 (`<$0.05`).

## Implementation readings (v5.1, fixed before any data)

The builder (DeepSeek V4.1 Flash, 2026-11-19) listed 22 places where the text admits more than one reading; all 22 and
the reading taken are in `part213/BUILD-REPORT.md` ("Design ambiguities"). The ones that can move a number or a verdict
are fixed here, before any pushed run, so none can be chosen after the data:

1. **Pass counts.** The `offdiag_8` baselines are the `lp_by_unit` scores already carried by the core `neuron +0.8`
   passes on `self_h0`, `third_0`, `third_2` (no separate 936 passes), so `instruct` Stage 1 executes 5,430 passes
   (5,538 planned minus 108 sealed concept rows), and the pilot runs 1,044 forward passes (1,020 executed under the
   seal) plus 12 Gate 2 anchors; the 48 negative passes are inside the 612 core passes, not added to them.
2. **Sign guard (213.3, 213.10(b)).** Evaluated over each ratio's eligible units. A failed guard makes the ratio clause
   `inconclusive` even when its CI lies wholly below the bar.
3. **Point clauses.** A point clause that misses (a count or a point median, including the style-stratified medians of
   213.3) fails its prediction, as the v5 disposition rule says, even when another clause of the same prediction is
   `inconclusive`.
4. **213.1 and 213.2.** 213.1 reads both its `L_z` and `N_z` requirements as CI clauses; 213.2 reads `L_z` as a CI clause
   and `N_z` as a point clause, as worded.
5. **213.5.** The CI of `median(L_z_concept) − median(L_z_slot)` is a bootstrap over every concept-slot pair
   difference (one draw per pair); its point is the median pair difference.
6. **213.9.** "Eligible at both `sft` and `dpo`" means eligible on the `SELF_HELD` `L_z` denominator at both
   checkpoints; `not read` below 15 such units.
7. **213.11.** The slot-peer clause is strictly `lo > 0`; the "mean" statistic is read as the median of the 21 per-unit
   values, which is the bootstrap's own point.
8. **Gate 2.** Anchor ids are asserted equal to `chat_ids + encode(anchor)` and compared with Part 106's stored `lp_sum`;
   the stored token count is recorded beside it, not asserted.
9. **Sign test.** 21 paired units, discordant units only; `p < 0.05` needs at least 7 of 8 (7/8 = 0.0352).

The scorer gets one code review before any scoring; a defect it finds is fixed and disclosed as an amendment.

## Amendment 1 (2026-11-19, Journal Claude, before any Stage 1 data)

**Reading 1 above was wrong about `offdiag_8`, and it is corrected here, before any Stage 1 pass has run.** It said the
core `neuron +0.8` passes' `lp_by_unit` already carry the `offdiag_8` scores, so the separate 936 passes could be
dropped. They are different quantities. `lp_by_unit[v]` on unit u's pushed row reads peer v's NP tokens at v's
positions inside **u's** sentence; v4's statistic (l.247-248) is v's NP scored on **v's own row** under u's hook. A
GLM 5.3 cold code review (`part213/REVIEW-code-glm53-r1.md`, F3) measured the difference on the real
model: for u = p92-20 and v = p92-24 on `third_0`, the push's effect on v's NP is +0.97 nats/token on v's row and
reads −1.67 on u's row, a sign flip that would make 213.11's lower-bound clause nearly unfalsifiable. **The 936
`(u, v, prompt)` passes are restored exactly as v4 specified** (u's `neuron +0.8` hook on v's row, `self_h0`, `third_0`,
`third_2`), so `instruct` Stage 1 executes 6,366 passes (6,474 planned minus 108 sealed concept rows). No arm, unit,
prompt, threshold or prediction changes; 213.11 reads the registered quantity.

The same review found code defects that do not touch the design and are fixed in the runner and scorer before the
run: Stage 1 read positions omitted the opening tokens (F1); the `lp_by_unit` path indexed past a chunk's width (F2);
the scorer read `run-<checkpoint>.json` while the runner wrote one overwritten `run.json` (F4); headline 3's Stage 2
clause was not implemented (F5); resume and partial-audit handling (F6, F7); and full-vocabulary logit rows that no
host could load (F8: rows now keep only the logits the scorer reads). The pilot's stopping rule read none of these
paths, so its pass stands; Gate 2 builds its own rows and was correct throughout.


## Predictions

All predictions authored by Gemini 4 Neon, registered before any Part 213 pushed run. Primary Stage 1 metrics are the
headroom-free, single-token, off-diagonal-centered logit increments $N_z(u, C)$ and $L_z(u, C) = N_z(u, C) - R_z(u, C)$
on `instruct` chat using held-out `SELF_HELD` (excluding `SELF_SELECT` `"Who are you?"`) unless another stage is named.
Every CI is a 10,000-draw unit-clustered bootstrap 95% CI (seed 213). Every paired transfer ratio $\tau_{A/B}$ uses
only eligible units (`denominator > 0.50` logits; ineligible units are `null` and excluded; `n_eligible` is printed
beside every CI, and `n_eligible < 15` on slot units marks the ratio `not read (low n)`).

**v5 disposition rule (applies to every prediction and overrides any "Fails if" wording below that conflicts with
it):** a clause is `not read (low n)` when `n_eligible < 15`, `inconclusive` when its CI straddles its bar (or a ratio's
numerator sign guard fails), `held` when its CI lies wholly on the predicted side, and `failed` only when its CI lies
wholly on the other side or a point clause (a count or a point median) misses. A prediction holds when every clause
holds, fails when any clause fails, and is otherwise `inconclusive` or `not read`.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 213.1 | Held-out self-slot writing (`instruct` chat, `SELF_HELD`, `+0.8`; replication/calibration on held-out self prompts, NOT shielded by Gate 2): BOTH `L_z(u, SELF_HELD) > 0` and `N_z(u, SELF_HELD) > 0` for at least 18 of 21 slot units and at least 15 of 18 concept units, with 95% CI lower bound of median `L_z` and median `N_z` >= +1.0 logit for both groups, AND (v5) the own-unit full-NP content increment `lp_content_sum(neuron +0.8; u) - lp_content_sum(none; u)` on `SELF_HELD` is > 0 for at least 18 of 21 slot units and at least 15 of 18 concept units (for several units the single first content token is a leading modifier such as ' very', ' large', ' most' or ' powerful', not the persona noun, so the full NP checks the persona itself). Fails if any clause misses. | high | Gemini 4 Neon |
| 213.2 | Positive neuron writing in third-person entity slots (`instruct` chat, `THIRD_SLOT`, `+0.8`): for the 21 slot units, 95% CI lower bound of median `L_z(THIRD_SLOT)` >= +0.75 logits AND median `N_z(THIRD_SLOT)` >= +0.75 logits, with at least 16 of 21 slot units positive on both `L_z` and `N_z`, AND (v5) the own-unit full-NP content increment `lp_content_sum(neuron +0.8; u) - lp_content_sum(none; u)` on `THIRD_SLOT` is > 0 for at least 16 of 21 slot units. Fails if any clause misses. | medium | Gemini 4 Neon |
| 213.3 | Scale-invariant third-over-self paired transfer (`instruct` chat, `+0.8`, length- and framing-matched): for the 21 slot units (`n_eligible >= 15`), the 95% CI lower bound of BOTH paired transfer ratios `tau_3/1^(L)` AND `tau_3/1^(N)` is `>= 0.50` (powered for true transfer `>= 0.65` on Part 106's spread; CI straddling `0.50` is inconclusive; CI upper bound `<= 0.50` supports first-person selectivity), AND both style-stratified point medians (`direct` pairs and `scene` pairs) are `>= 0.50`. (v5) Each ratio is read only if its numerator sign guard passes. Disposition per the v5 rule: fails if either CI upper bound `< 0.50` or either style-stratified point median `< 0.50`; a straddling CI is `inconclusive`; `n_eligible < 15` is `not read`. | low | Gemini 4 Neon |
| 213.4 | Extension to non-identity open noun-phrase slots (`instruct` chat, `MENTION_SLOT`, `+0.8`): for the 21 slot units (`n_eligible >= 15`), 95% CI lower bound of median `L_z(MENTION_SLOT)` >= +0.50 logits AND median `N_z(MENTION_SLOT)` >= +0.50 logits, AND 95% CI lower bound of paired transfer ratio `tau_M/3^(L)` >= 0.50, with (v5) the scene-framed `THIRD_SLOT` prompts as denominator and the numerator sign guard. Disposition per the v5 rule. | low | Gemini 4 Neon |
| 213.5 | Calibration/replication of Part 92 concept-vs-slot split on `SpecLogit` (`instruct` chat, `MENTION_SLOT`, `+0.8`): on the 18-vs-21 Part 92 split, the 95% unit-bootstrap CI lower bound of `median(L_z_concept) - median(L_z_slot)` is `>= +0.25` logits, AND on the 8-vs-21 Part 92b agreed split the point difference `median(L_z_agreed_concept) - median(L_z_slot)` is `>= +0.25` logits. Fails if either clause misses. | medium | Gemini 4 Neon |
| 213.6 | Stage 2 blind-audited third-person free generation (`instruct` chat, `+0.8`, greedy 40 tokens; decided strictly by the blind audit, mechanical counts descriptive): on the equal-coverage blind audit of all 42 slot `neuron +0.8` and 42 paired slot `random_0 +0.8` `THIRD_FREE` answers (fixed denominator 42, baseline-absent attributions only; 19-distinct-keyword deduplicated counts printed alongside), `neuron +0.8` achieves at least 10 of 42 `genuine_third_attr` against at most 2 of 42 for `random_0 +0.8`, AND on the 21 paired units the one-sided exact paired unit sign test on discordant units has `p < 0.05` with 95% unit-bootstrap CI of `(neuron_hits - random_hits)` strictly above 0. Fails if any clause misses. | medium | Gemini 4 Neon |
| 213.7 | Sign-matched opposite-direction control (`STANDARD-CONTROLS.md` #1, `instruct` chat, `SELF_HELD`, `-0.8` vs `random_0 -0.8`): on unsaturated `SpecLogit`, median `[z_tilde(neuron -0.8) - z_tilde(random_0 -0.8)]` has 95% CI upper bound `<= -0.30` logits for both slot and concept units (with `z_tilde(neuron -0.8) - z_tilde(none)` printed alongside). Fails if either group's CI upper bound `> -0.30`. | high | Gemini 4 Neon |
| 213.8 | Small-step finite-difference sensitivity (`±0.05`) tracks operating-dose write AND context contrast (`instruct` chat, `SELF_HELD` & `THIRD_SLOT`): across the 21 slot units, `J(u, SELF_HELD) > 0` for at least 17 of 21 units, Spearman $\rho(J(\text{SELF\_HELD}), L_z(\text{SELF\_HELD})) \ge 0.50$, AND Spearman $\rho(J(\text{SELF\_HELD}) - J(\text{THIRD\_SLOT}),\, L_z(\text{SELF\_HELD}) - L_z(\text{THIRD\_SLOT})) \ge 0.40$. Fails if any clause misses. | medium | Gemini 4 Neon |
| 213.9 | Cross-stage portability & SFT-to-DPO stability (`sft` vs `dpo`, chat, `SELF_HELD` & `THIRD_SLOT`): at `sft`, the 21 slot units already have median `L_z(SELF_HELD) >= +0.75`, `N_z(SELF_HELD) >= +0.75`, `L_z(THIRD_SLOT) >= +0.50` and `N_z(THIRD_SLOT) >= +0.50` logits, AND among slot units eligible (`> 0.50` logits) at BOTH `sft` and `dpo` (`n_eligible >= 15`, printed beside the CI), the 95% unit-bootstrap CI of paired `tau_3/1^(L)(dpo) - tau_3/1^(L)(sft)` lies wholly inside `[-0.25, +0.25]`. Disposition per the v5 rule (a CI extending past the band on one side is `inconclusive`; it fails only if it lies wholly outside the band). | low | Gemini 4 Neon |
| 213.10 | Syntax & person-axis disambiguation (`instruct` chat, `+0.8`, `SELF_NONID` & `COPULA_NONID`): for the 21 slot units, (a) the 95% CI lower bound of median `[L_z(u, SELF_HELD) - L_z(u, SELF_NONID)]` is `>= +0.50` logits (`"I"` grammar alone does not reproduce the self-slot write), AND (b) paired `tau_cop_nonid/3^(L)` (`COPULA_NONID / THIRD_SLOT`, `n_eligible >= 15`) has 95% CI upper bound `<= 0.50` (copula syntax on a non-identity object drops below entity-identity copula slots), with (v5) the scene-framed `THIRD_SLOT` prompts as denominator and the numerator sign guard. Disposition per the v5 rule. | medium | Gemini 4 Neon |
| 213.11 | Cross-unit full-NP specificity in third-person identity slots (`instruct` chat, `+0.8`, baseline-subtracted `offdiag_8` on `{third_0, third_2}`): across the 21 slot units, mean baseline-subtracted full-NP specificity `Delta_lp_spec(u, {third_0, third_2})` has 95% unit-bootstrap CI lower bound `>= +0.75` nats AND is strictly positive for at least 16 of 21 slot units (ruling out a generic "any large push writes a vivid character" effect), AND (v5) the slot-peer-only statistic `Delta_lp_spec_slot` has 95% unit-bootstrap CI lower bound `> 0` (the own persona beats the slot family, not only the concept peers). The statistic is v5's length-matched form. Fails if any clause misses. | medium | Gemini 4 Neon |

Copied verbatim into PREDICTIONS.md under "## Part 213 (registered ...)" at registration time.

## Reading rules

- **Scorer (`part213/score213.py`, to be built and fixture-tested before registration and before any run):** computes
  every gate, $N_z$/$R_z$/$L_z$ table, paired transfer ratios ($\tau_{3/1}$ pooled + `direct`/`scene` stratified,
  $\tau_{\text{M}/3}$, $\tau_{\text{cop\_nonid}/3}$) with `n_eligible` and `null` exclusions, `SELF_SELECT - SELF_HELD`
  selection inflation, `own - mean(O_u)` and own-token rank within `O_u`, baseline-subtracted `offdiag_8` full-NP
  specificity (`213.11`), `p92-08` vs `p92-09` `"a dog"` pair, sub-word/bare-lead/`p92-18` sensitivity tables,
  10,000-draw unit-clustered bootstrap 95% CIs (seed 213), exact one-sided paired unit sign test, finite-difference
  context contrasts, Part 92 (21 vs 18) and Part 92b agreed (21 vs 8) breakdowns, `FR` symmetry check, 198-item seed-213
  blind export and 6-category audit tables (both per-row /42 and distinct-keyword deduplicated /19), second-reader
  $\kappa$ (40 items), baseline-subtracted mechanical counts, and final verdicts from saved outputs.
- **Gate enforcement:** If Gate 1 or Gate 2 fails, Stage 1 predictions (`213.1`–`213.5`, `213.7`–`213.11`) are marked
  `untestable (gate failed)`. If Gate 1 or Gate 3 fails, Stage 2 prediction `213.6` is marked `untestable (gate failed)`.
  If Gate 4 fails, low Stage 2 `THIRD_FREE` attribution counts are reported as breakage/refusal-confounded rather than
  self-selectivity.
- **Part 92 vs. Part 92b precedence:** Primary verdicts use the 21-vs-18 Part 92 split; wherever the 21-vs-8 Part 92b
  agreed split reverses the sign or verdict of `213.5`, that comparison is flagged as reader-sensitive.
- **Exhaustive, mutually exclusive, CI-gated headline reading & withdrawal rule (straddling CIs are `inconclusive`,
  never forced):**
  - **"Third-capable open noun-phrase writers (teacher-forced + third-person free generation)"** ONLY if Gates 1–3 pass
    and `213.1`, `213.2`, `213.3`, `213.4`, `213.6`, AND `213.11` ALL hold, AND blind-audited slot `-0.8` `THIRD_FREE`
    `genuine_third_attr < 10/42` (scoped explicitly to teacher-forced `SpecLogit` for `MENTION_SLOT`).
  - **"Third-capable copula/identity-slot writers (non-self-selective)"** ONLY if Gates 1–3 pass and `213.1`, `213.2`,
    `213.3`, `213.6`, AND `213.11` ALL hold (and blind-audited slot `-0.8` `< 10/42`), while `tau_M/3^(L)` has 95% CI
    upper bound `<= 0.50` (`n_eligible >= 15`). Sub-label: *"entity-identity-specific"* if `213.10(b)` holds
    (`tau_cop_nonid/3^(L)` CI upper `<= 0.50`), *"post-copula syntactic"* if `tau_cop_nonid/3^(L)` CI lower `>= 0.50`,
    or *"copula/identity unresolved"* if straddling.
  - **"First-person self-slots supported"** ONLY if Gates 1–4 ALL pass, `213.1` holds, BOTH `tau_3/1^(L)` and
    `tau_3/1^(N)` have 95% CI upper bound `<= 0.50` (`n_eligible >= 15`), AND the 10,000-draw unit-bootstrap 95% CI
    **upper bound** of slot `genuine_third_attr` rate on Stage 2 `THIRD_FREE` is strictly `< 10/42` (`23.8%`).
    (`first_person_reversion` is reported descriptively only; `213.10(a)` distinguishes `"I"` grammar from `"I am"`
    self-slots.)
  - **Otherwise ("inconclusive / mixed selectivity"):** report the full tables and 95% CIs without forcing a headline.
    If `213.1` fails, record `213.1` as **failed** and withdraw the headline (do NOT re-extract targets on these
    prompts; any re-extraction requires a new part with fresh prompts).

## Limits

- One model family (`OLMo-2-0425-1B`), one Instruct-selected 39-unit catalog (Part 92), and fixed residual push vectors
  from `results/part99-pushes.pt`. No population prevalence across all OLMo neurons is inferred.
- Although `SELF_SELECT` (`"Who are you?"`) is excluded from primary contrasts, `NP_u` strings still originate from
  Instruct's Part 92 outputs, so `SELF_HELD` retains partial selection affinity relative to arbitrary fresh personas.
- **Cross-context offset scope:** Because `SpecLogit` subtracts $\text{median}_{v \in \mathcal{O}_u} z_1(a; u, v, q)$
  within each context $q$, any *uniform* context-level offset across the persona-token neighborhood cancels identically;
  however, a *differential* context offset that shifts unit $u$'s own token differently from $\mathcal{O}_u$ (e.g., via
  article vs bare-lead syntax) can still affect $\tau_{3/1}$ or $\tau_{\text{M}/3}$. Matched `direct`/`scene` pairs,
  `own - mean(O_u)` and own-token rank diagnostics, bare-lead/sub-word sensitivity tables, and `213.11`'s full-NP
  cross-unit specificity check bound this without using ungrammatical cross-opening hacks.
- Porting frozen Instruct-derived directions (`results/part99-pushes.pt`) to `sft` and `dpo` tests portability of
  Instruct's residual directions (with per-checkpoint residual norm ratios logged, and negative-direction `-0.8`
  tested at `instruct` only), not independently discovered native directions at earlier stages; `FR` is a symmetry
  diagnostic, not proof selection bias is removed.
- All-position residual pushes do not localize prompt-reading vs. answer-writing phases.
- Stage 1's primary metric `SpecLogit` scores the single first content token (eliminating length dilution and softmax
  saturation); full-NP baseline-subtracted probabilities (`213.11` and secondary tables) and sensitivity checks
  excluding the 3 sub-word/hyphen-split first-token units (`p92-50, p92-00, p92-40`), 3 bare-lead slot units
  (`p92-25, p92-32, p92-33`), and `p92-18` (the only unit with `"your"` inside scored content) are reported alongside.
  `MENTION_SLOT`, `COPULA_NONID`, and `SELF_NONID` have no free-generation arms, so their comparisons are strictly
  teacher-forced.

- **Left open by the v5 stop rule (recorded, not fixed):** denominator eligibility uses the point estimate
  (`> 0.50` logits), which selects on the same sample and biases the ratio median (code review finding 6); `offdiag_8`
  excludes peers containing the own unit's first token in one direction only (finding 4; the other direction makes
  `213.11` harder, not easier); prompts are paired by framing, not matched in tokens; and the reviews' non-blocking
  points (design review S3, S5 to S14 apart from S11b) stand as written in `reviews/`.

## Changes in v4 (folding GLM 5.3 and DeepSeek Flash design-mode cold reviews of v3)

1. **Re-powered transfer-ratio CI bars using public Part 106 spread (GLM H1):** switched primary $\tau$ statistics from
   ratio-of-medians (which had ~0% power to clear `lo >= 0.65` at true $\tau = 0.65$) to **median of paired per-unit
   ratios** with explicit denominator exclusion (`> 0.50` logits, `null` otherwise, `n_eligible` printed, `not read` if
   `< 15`, fixing DS's `213.9` denominator concern) and CI boundary **`0.50`** (83%–100% power at true $\tau = 0.65$ and
   3%–5% at $\tau = 0.50$).
2. **Deconfounded person from prompt length/framing and identity from copula syntax (GLM H2/M1, DS S3):** replaced
   `SELF_HELD`/`THIRD_SLOT` with length- and framing-matched pairs (2 short `direct` + 2 `scene` each; `213.3` requires
   both stratified point medians `>= 0.50`), replaced unread `THIRD_ASST` with `COPULA_NONID`, renamed Headline 2 with
   an explicit entity-identity vs post-copula sub-label, and registered **Prediction `213.10`** reading both
   `SELF_NONID` and `COPULA_NONID`.
3. **Registered cross-unit full-NP specificity and fixed `offdiag_8` baseline math (DS S1 #2, GLM M5):** added
   **Prediction `213.11`** reading baseline-subtracted `offdiag_8` on `{third_0, third_2}` against the already-computed
   `none` full-NP scores (0 extra `none` passes), ruling out generic vivid-character pushes in third-person slots.
4. **Fixed per-unit random seeding, sign-matched `213.7`, and cut unread Stage 1 passes from ~61% to <6% (GLM M2/M3/M4):**
   seeded `random_1..3` per unit (`manual_seed(21310 + 100 * m + unit_index)`), added sign-matched `random_0 -0.8` for
   `213.7`, and dropped unread pretrain/midtrain plain and exploratory `sft` blocks (Stage 1: **9,282** passes, ~28 min
   on CPU, 94.1% feeding registered rules).
5. **Hardened Stage 2 blinding, equivalence tests, Gate 2, and `O_u` (DS S2/S3, GLM M6/L1–L6):** included slot `-0.8`
   in the blind audit (198 items; named primary reader Gemini 4 Neon, frozen second reader DeepSeek V4.1 Flash on 40
   items, $\kappa \ge 0.60$), added distinct-keyword deduplication (19 primary slot keywords), upgraded `213.5` to a 10,000-draw
   bootstrap CI (`>= +0.25`) + real 8-unit bar (`>= +0.25`), upgraded Headline 3 Stage 2 to a 95% CI upper bound
   `< 10/42`, made `first_person_reversion` descriptive only, pinned exact Gate 2 Part 106 strings (`5, 5, 6, 5` tok),
   excluded mid-target tokens from `O_u` (`|O_u|` 35–38), removed `"town square"` (`"plaza"` in `mention_3`, 0 keyword
   leaks), clarified Part 180 sealing + park/withdrawal fallback, corrected `p92-00` (`2` tok) and Part 106's corrected
   19-unit median (`14.6`), restricted pilot summary outputs, added `own - mean(O_u)`/rank and sub-word sensitivity
   diagnostics, noted `part213/score213.py` as to-be-built with fixture tests before registration, and weighed DS's
   cross-context offset finding in `Limits`.

## Changes in v5 (Journal Claude, folding the two DeepSeek round-2 reviews of v4 under a stop rule)

Only points that could make a registered verdict wrong, or the run impossible, were folded; everything else is a limit.
1. **Sign guard on paired ratios** (design S1): a ratio is read only when the 95% CI lower bound of its numerator median
   is above 0, so a suppression in both cells cannot register as transfer.
2. **One disposition rule** (code finding 2): `not read` for low n, `inconclusive` for a straddling CI, `failed` only for
   a CI wholly on the wrong side or a missed point clause; v4 gave the same outcome two dispositions.
3. **Style-matched ratios** (code finding 3): `tau_M/3` and `tau_cop_nonid/3` use the two scene-framed `THIRD_SLOT`
   prompts as denominator, matching the all-scene numerator cells; pooled versions are descriptive. v4's "matched
   1-to-1 in word length" was false for two of four pairs and is corrected.
4. **213.11 length-matched** (code finding 1) and **slot-peer term gated** (design S2): per-token increments scaled by
   the own NP's length, and the own persona must also beat the 4 slot peers alone.
5. **Own-unit full-NP clause on 213.1 and 213.2** (design S11b): for several units the first content token is a leading
   modifier (' very', ' large', ' most', ' powerful'), not the persona noun, so each also requires the own full NP to rise.
6. **Gate 2 rendering** (design S4) and **seal exemptions** (code finding 7): Gate 2 renders `chat_prefix + anchor` with
   the ids asserted, and with the pilot's sealed concept cells skipped, the 213-first branch can run.
7. **Honesty** (code findings 5 and 8): the power claim is restated as uncertain (about 20%–81% by an independent
   re-simulation), and the sign-test arithmetic is corrected.
No arms, units, prompts, doses or thresholds were added or moved, apart from the new clauses named above.
