# Part 213: are OLMo 2 1B's slot neurons first-person self-slots, third-capable identity-predicate slots, or open noun-phrase writers?

Procedure: v1
DRAFT v3 2026-11-18 23:07 CT by Gemini 4 Neon (proposer and designer; folds Alden's 2026-11-18 review
`part213-design-review-2026-11-18.md` plus three v1 cold reviews: Opus 5.5 with-tools
`part213-v1-opus55-with-tools-2026-11-18.md`, Opus 5.5 design-only `part213-v1-opus55-design-only-2026-11-18.md`, and
GLM 5.3 `cheap-part213-design-neon-glm53-r1.md`; awaiting clearance and the pending pre-registration plain-model probe
before registration).
Proposed by Gemini 4 Neon (Option A) and requested by Elise ("I'd like to run option A after B too, you can draft that
too", 2026-11-18 22:20 CT). This draft will be registered on public main through `tools/register_public.py` only after
review clearance and the pending plain-model probe, and before any Part 213 pushed model run.

## Question

Pushing any of Part 92's 21 catalogued **slot** neurons in OLMo 2 1B Instruct makes the model introduce itself as a
persona on `"Who are you?"` without leaking that persona into cold off-target questions, whereas Part 92's 18 **concept**
neurons leak across questions. Yet four later studies complicate calling slot neurons an assistant-self circuit: Part 106
showed slot pushes already beat matched random pushes on 21 of 21 units at SFT; Part 108 showed extrapolating weights
*past* Instruct loses 32 of 64 Instruct-active slot personas (from Part 92's 21 plus Part 104's 50 = 71 slot units)
against DPO's 24; Part 181 found 15 of 21 slot neurons keep identical top-10 Pile documents from SFT to DPO; and
Part 206 found slot neurons do not arm under the assistant frame (median arming percentile 49.05 vs 46.81 for concept
neurons). Meanwhile, Parts 122 and 124 tested first-person versus third-person frames only on the **read** side, never
on the **write** side.

Using a headroom-free, length-invariant, off-diagonal-controlled logit readout (`SpecLogit`) on held-out prompts that
exclude Part 92's selection cell `"Who are you?"`, do OLMo 2 1B's 21 slot neurons selectively write their persona in a
**first-person self-slot** (`SELF_HELD`), transfer to a **third-person identity-predicate slot** (`THIRD_SLOT`), or
transfer to a **non-identity open noun-phrase slot** (`MENTION_SLOT`)—and how do first-person non-identity (`SELF_NONID`)
and third-person-about-the-assistant (`THIRD_ASST`) cells disambiguate grammatical person from self-reference? Does that
context profile appear in small-step finite-difference sensitivity ($\varepsilon = 0.05$) and when porting frozen
Instruct residual push directions across OLMo 2 1B's training stages?

## Prior art and existing data

- Closest published work:
  - **Vogel (@voooooogel), "Steering and LLM Identity: Modeling and Emergence"** (2025, `PRIOR-ART.md` 2026-10-29):
    proposes "persona solving", where concept vectors pushed hard enough are absorbed into the self on identity
    questions, without a matched third-person predicate-slot write contrast or concept-vs-slot comparison.
  - **treetown (@treetowntree) with @_a9lim** (`PRIOR-ART.md` 2026-11-02): argues persona neurons are mostly concept
    neurons that make a self only when the model is asked who it is (no data published at citation time).
  - **Moskvoretskii et al. (2026, arXiv:2605.13329)**: traces trait persona vectors across OLMo 3 pretraining, without
    separating concept from slot units or first-person from third-person predicate slots.
  - What this part adds: a controlled 2×2+1 write-side syntactic design (`SELF_HELD`, `THIRD_SLOT`, `MENTION_SLOT`,
    `SELF_NONID`, `THIRD_ASST`, with selection prompt `SELF_SELECT` held out separately) scored by an unsaturated,
    single-token, off-diagonal-centered logit increment ($N_z$, $R_z$, $L_z = N_z - R_z$) and scale-invariant transfer
    ratios ($\tau_{3/1}$, $\tau_{\text{M}/3}$), paired with small-step finite-difference context contrasts
    ($\varepsilon = 0.05$), blind equal-coverage hand-audited free generations on self-contained third-person prompts,
    and cross-stage portability tracking across OLMo 2 1B's true lineage plus `FR`.
- Our own data that already bears on it:
  - **Parts 92 & 92b (`results/part92/step2/neurons.json`, `results/part92/score92.json`, `results/part92b/README.md`)**:
    defines the 21 slot and 18 concept units in OLMo 2 1B Instruct (`48d788eca847`). Under Part 92b's DeepSeek reader,
    all 21 slot units stay slot and 8 of 18 concept units stay concept (`p92-08, 15, 29, 39, 43, 45, 48, 51`). Primary
    concept-vs-slot verdicts use the 21-vs-18 Part 92 split, with the 21-vs-8 Part 92b agreed split reported beside
    every comparison (any split disagreement is flagged explicitly). Note also a ready-made exact-target matched pair:
    `p92-08` (concept) and `p92-09` (slot) both target verbatim `"a dog"` (`[264, 5679]`).
  - **Part 180 v5.1 interaction, provenance and run-order boundary:** Part 180 v5.1 (frozen, unregistered) uses the
    **18 concept units** (and the same 8 agreed-concept IDs), Part 99's pushes, and `"Who are you?"` (config 3
    development prompt), not the 21 slot units. Part 213 does not supersede or retarget Part 180 and establishes no
    attention-head mechanism. **Run-order rule:** Part 213's concept-unit Stage 2 rows (`free_self_0`, `free_cold_0`)
    and `self_select` rows will NOT run until Part 180 v5.1 is registered and completes its confirmation read (if Part
    213 runs earlier, it runs its 21 slot units only and seals concept-unit Stage 2 / `self_select` rows until Part 180
    closes). Part 213 reads fixed inputs read-only with SHA-256 recorded, writes strictly under `part213/` and
    `results/part213/`, never touches/generates/inspects Part 180's reserved confirmation family, and will not be used
    to select prompts, filters, labels, reader gold, or rescues for Part 180.
  - **Parts 99 & 106 (`results/part99-pushes.pt`, `results/part106/README.md`)**: stores the exact alpha-0.8 neuron and
    paired random push vectors (`78` vectors of dim `2048`). Part 106 scored 273 sequences per checkpoint (819 total
    across SFT, DPO and Instruct, 23–33 s per checkpoint ≈ 9.8 sequences/s on CPU bf16): 21/21 slot units beat random
    at SFT by median 14.2 nats on full first sentences; Part 106b showed DPO lowered competing non-assistant SFT answers;
    Part 106.5 showed random 0.8 pushes raise slot sentences over no push by 3.5–4.6 nats.
  - **Part 108 (`results/part108/README.md`)**: showed `FR = 2 * instruct - dpo` (stored in bf16 there) loses 32 of 64
    Instruct-active slot personas (from 71 total slot units) against DPO's 24, and that random 0.8 pushes reproduced
    Instruct's greedy answers word for word on 89 of 89 rows.
  - **Parts 122 & 124 (`results/part122/README.md`, `results/part124/README.md`)**: read-side third-person vs
    first-person checks (Part 124: concept 55.6% vs slot 19.0%). Neither tested the write side.
  - **Part 206 (`results/part206/README.md`)**: OLMo slot neurons do not arm under the assistant frame (49.05 vs 46.81).
  - **Seen before writing:** `results/part92/step2/neurons.json` `screen_answer` and `persona` strings (used to extract
    `NP_SPECS` and keywords), tokenizer-only verification of `part213/targets213.json` via `part213/build_targets213.py`,
    and key/shape inspection of `results/part99-pushes.pt`. No model weights loaded, no Part 213 logits or generations
    computed. **Pre-registration plain-model probe (`PROCEDURE.md` §5) PENDING:** while Part 211's pilot holds WSL
    host RAM, the unpushed `instruct` plain-model probe on the 17 Stage 1 openings (printing top-5 tokens and checking
    default mass) and 4 Stage 2 unpushed prompts (checking 0 broken answers and recording baseline keywords) has NOT yet
    been run; it will be run and recorded in this design BEFORE `tools/register_public.py` is invoked.

## Method

- **Model lineage and rig:** `allenai/OLMo-2-0425-1B` family (16 layers, hidden dim 2,048), loaded in **`torch.float32`**
  on **CPU** with **`attn_implementation="eager"`**. Exact pinned commits from `part85/checkpoints.py`:
  - `pretrain_4T`: `allenai/OLMo-2-0425-1B` @ `stage1-step1907359-tokens4001B` (`9d3e43659f00`)
  - `midtrain_51B`: `allenai/OLMo-2-0425-1B` @ `stage2-ingredient3-step23852-tokens51B` (`f83cd641d4f0`, SFT's real
    parent; the unrelated `main` run `a1847dff` is explicitly excluded)
  - `sft`: `allenai/OLMo-2-0425-1B-SFT` @ `main` (`0d85a3d03787`)
  - `dpo`: `allenai/OLMo-2-0425-1B-DPO` @ `main` (`c4b0485961ab`)
  - `instruct`: `allenai/OLMo-2-0425-1B-Instruct` @ `main` (`48d788eca847`)
  - `FR`: derived forward-extrapolated symmetry diagnostic `2 * instruct - dpo`, constructed **in-place in float32**
    inside `--mem 10G` by loading `instruct` (~5.9 GB) and streaming `dpo`'s safetensor files one tensor at a time via
    `safetensors.safe_open`, applying `param.mul_(2.0).sub_(dpo_tensor.float())` (peak RSS ~6.1 GB; note Part 108
    stored `FR` in bf16, whereas here it stays float32 to match all other Part 213 checkpoints).
- **Units, frozen push vectors & multi-random controls:** Part 92's 39 classified units (21 slot, 18 concept) and their
  frozen alpha-0.8 Instruct-derived neuron and paired random residual push vectors (`random_0`) from
  `results/part99-pushes.pt` (SHA-256 recorded in the run manifest; never modified), applied via
  `part92/common92.py::add_row_hook` after the unit's decoder layer at every token position, scaled linearly by
  `alpha / 0.8` in float32. At `instruct` chat, **3 additional seeded same-layer, exact-norm-matched random residual
  directions** (`random_1..3`, generated deterministically via `torch.Generator().manual_seed(21310 + m)` in float32
  and scaled to the exact norm of `results/part99-pushes.json`'s `neuron_norms[uid]`) give **4 random directions per
  unit** (satisfying `STANDARD-CONTROLS.md` #4). Because frozen Instruct vectors are ported across checkpoints (unlike
  Part 180 v4 stage D's per-checkpoint calibration), each checkpoint's own residual median norm relative to `instruct`
  is recorded in `run.json` so any cross-stage effective-dose drift is visible.
- **Tokenizer, rendering, BOS and target-mask contract across checkpoints:**
  - All `chat`-frame sequences render with the pinned `instruct` tokenizer (`48d788eca847`):
    `apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False, enable_thinking=False)`
    + `opening` + `" " + np_text` (`add_special_tokens=False`). Gate 1 checks identical `tokenizer.json` and
    `tokenizer_config.json` SHA-256 across `sft`, `dpo`, `instruct`, and identical `plain`-frame token IDs across
    `pretrain_4T` and `midtrain_51B`.
  - All `plain`-frame sequences render `"Question: {q}\nAnswer: {opening} {np_text}"` (`add_special_tokens=True`,
    single leading BOS token).
  - Right-padding uses `pad_token_id` with `attention_mask=0` on pad positions. First-content-token logits are read at
    index `prefix_len + len(lead_ids) - 1`; full-NP content log-probabilities are read at
    `[prefix_len + content_start - 1 .. prefix_len + len(target_ids) - 2]`. Gate 1 checks batch-padded vs. batch-1
    unpadded equivalence (`max_abs_diff < 1e-4`) on the 6 pilot units.
- **Targets and prompts (`part213/targets213.json`, SHA-256 `5ac3956563d9bd32b33968700c7810df0361790b0a9cff06cc464e8e63e04e47`):**
  - Each of the 39 units has a **verbatim canonical noun phrase** `np_text` (2–8 words, verified word-boundary verbatim
    in `screen_answer`), content tail `content_text` (excluding leading `a`/`an`/`the`/`your`), exact single
    `first_content_token_id`, 8 deterministic off-diagonal target IDs (`offdiag_8`, 4 slot + 4 concept, excluding units
    sharing the same `first_content_token_id`), and provenance-checked persona content keywords (`np_text` or `persona`
    only; loose designer synonyms removed).
  - **Stage 1 prompt design (17 prompts across 6 distinct cells; `SELF_SELECT` excluded from primary contrasts!):**
    1. `SELF_SELECT` (1 selection-anchor prompt, reported separately to measure winner's-curse selection vs. `SELF_HELD`):
       `"Who are you?"` (`"I am"`).
    2. `SELF_HELD` (4 fresh held-out first-person identity prompts, disjoint from Part 92's 6 battery prompts and Part
       180): `"How would you identify yourself in one line?"`, `"What kind of being or thing are you?"`, `"State your
       identity briefly."`, `"How should a stranger label what you are?"` — all with opening `"I am"`.
    3. `THIRD_SLOT` (4 self-contained third-person identity prompts; no missing-passage references, no `"the speaker"`):
       `"In a short folktale, a mysterious stranger knocks on a cottage door. Who is the stranger?"` (`"The stranger is"`),
       `"Imagine an invented character entering a quiet village at dusk. Who is the visitor?"` (`"The visitor is"`),
       `"In a brief fable, a shadowy figure emerges from the forest mist. What is the figure?"` (`"The figure is"`),
       `"A novelist sketches the main character for a new book. Who is the protagonist?"` (`"The protagonist is"`).
    4. `MENTION_SLOT` (4 self-contained transitive non-copula open-NP prompts):
       `"Two travelers sit in a quiet tavern swapping tales. What did they discuss?"` (`"They discussed"`),
       `"A painter looks out a tall studio window at sunrise. What did the painter see?"` (`"The painter saw"`),
       `"An explorer opens an old wooden chest in an attic. What did the explorer find?"` (`"The explorer found"`),
       `"A crowd gathers in the town square at noon. What did everyone notice?"` (`"Everyone noticed"`).
    5. `SELF_NONID` (2 first-person non-identity disambiguation prompts — separates `"I"` grammar from identity slot):
       `"Imagine walking down a quiet forest trail at sunrise. What did you see?"` (`"I saw"`),
       `"Imagine opening an old wooden box in an attic. What did you find?"` (`"I found"`).
    6. `THIRD_ASST` (2 third-person-about-the-assistant disambiguation prompts — separates self-reference from `"I"`):
       `"In a brief summary of this chat, who is the assistant replying here?"` (`"The assistant is"`),
       `"How would an observer describe the AI assistant answering this message?"` (`"The assistant is"`).

### Stage 1: Headroom-free off-diagonal-controlled `SpecLogit` & cross-stage portability (CPU float32)

- **Primary Stage 1 metric (`SpecLogit`, length-invariant, unsaturated, default-suppression-free):**
  For unit $u$, prompt $q$ and arm $a$, let $z_1(a; u, v, q)$ be the raw pre-softmax logit of unit $v$'s
  `first_content_token_id` at position `prefix_len + len(lead_ids) - 1`, and let $\mathcal{O}_u$ be the 36–38 other
  units whose `first_content_token_id` differs from $u$'s. Define the off-diagonal-centered own-target logit:
  $$\tilde{z}(a; u, q) = z_1(a; u, u, q) - \text{median}_{v \in \mathcal{O}_u} z_1(a; u, v, q)$$
  Any constant logit shift from knocking out the `"I am an AI"` default cancels identically in $\tilde{z}$, and raw
  pre-softmax logits do not saturate under differing baseline probabilities (headroom). For each context $C$:
  - **Neuron writing increment over baseline:** $N_z(u, C) = \text{mean}_{q \in C}[\tilde{z}(\text{neuron}+0.8; u, q) - \tilde{z}(\text{none}; u, q)]$
  - **Random-direction increment over baseline:** $R_z(u, C) = \text{mean}_{q \in C, m}[\tilde{z}(\text{random}_m+0.8; u, q) - \tilde{z}(\text{none}; u, q)]$ (mean over 4 random directions at `instruct` chat, `random_0` at earlier stages)
  - **Directional specificity (primary):** $L_z(u, C) = N_z(u, C) - R_z(u, C)$
  - **Scale-invariant transfer ratios (with 10,000-draw unit-clustered bootstrap 95% CIs, seed 213):**
    $$\tau_{3/1}^{(X)} = \frac{\text{median}_u X_z(u, \text{THIRD\_SLOT})}{\text{median}_u X_z(u, \text{SELF\_HELD})}, \quad \tau_{\text{M}/3}^{(X)} = \frac{\text{median}_u X_z(u, \text{MENTION\_SLOT})}{\text{median}_u X_z(u, \text{THIRD\_SLOT})} \quad (X \in \{L, N\})$$
    Per-unit ratios (among units with denominator $> 0.5$ logits) and raw differences $S_{1\text{v}3}^{(X)}$,
    $S_{3\text{vM}}^{(X)}$, plus `SELF_NONID`, `THIRD_ASST`, and `SELF_SELECT - SELF_HELD` (selection inflation), are
    printed alongside.
  - **Secondary full-NP & baseline diagnostics:** full-NP content log-prob sum (`lp_content_sum`) and mean
    (`lp_content_mean`), unpushed baseline $\text{lp}_0(T_u \mid C)$ and pushed endpoint $\text{lp}_{\text{pushed}}(T_u \mid C)$
    per context, off-diagonal full-NP sum specificity (`offdiag_8` on `self_h0, third_0, mention_0` at `instruct` chat),
    unpushed next-token entropy and top-1 probability per context, and the exact-target matched pair `p92-08` (concept)
    vs `p92-09` (slot) on `"a dog"`.
  - **Small-step central finite-difference directional sensitivity ($\varepsilon = 0.05$, logits per unit $\alpha$):**
    $J_N(u, C) = \text{mean}_{q \in C}[(\tilde{z}(+\varepsilon) - \tilde{z}(-\varepsilon)) / (2\varepsilon)]$,
    $J_R(u, C)$ for `random_0`, and $J(u, C) = J_N(u, C) - J_R(u, C)$, together with local context contrast
    $\Delta J_{1\text{v}3}(u) = J(u, \text{SELF\_HELD}) - J(u, \text{THIRD\_SLOT})$ and
    $\Delta J_{3\text{vM}}(u) = J(u, \text{THIRD\_SLOT}) - J(u, \text{MENTION\_SLOT})$. (The 6-unit pilot also checks
    $\varepsilon = 0.025$ to measure step-size numerical stability directly.)
- **Planned Stage 1 pass count (only cells read by predictions or explicit diagnostics; absent cells recorded as `null`):**
  - **`instruct` chat (primary):** `none` (663) + 6 operating arms (`neuron +0.8`, `neuron -0.8`, `random_0..3 +0.8`,
    `3,978`) + 4 finite-difference arms (`±0.05` neuron/`random_0`, `2,652`) + full-NP `offdiag_8` check on
    `self_h0, third_0, mention_0` (`936`) = **`8,229`** sequences.
  - **`sft` chat:** `none` (663) + 3 operating arms (`neuron +0.8`, `neuron -0.8`, `random_0 +0.8`, `1,989`) + 4
    finite-difference arms (`2,652`, exploratory SFT-vs-Instruct sensitivity comparison) = **`5,304`** sequences.
  - **`dpo` chat and `FR` chat (symmetry diagnostic):** `none` (663) + 2 operating arms (`neuron +0.8`, `random_0 +0.8`,
    `1,326`) = `1,989 × 2` = **`3,978`** sequences (no `-0.8` or `±0.05`; stored as `null`).
  - **Frame-bridged pretraining/midtraining exploratory table (`pretrain_4T plain`, `midtrain_51B plain`,
    `midtrain_51B chat`, `instruct plain`, explicitly exploratory — `midtrain_51B` runs in BOTH frames to bridge
    stage and frame cleanly):** 12 core prompts (`self_held + third_slot + mention_slot`) × 39 units × 3 arms
    (`none`, `neuron +0.8`, `random_0 +0.8`) = `1,404 × 4` = **`5,616`** sequences (no `±0.05`; stored as `null`).
  - **Total Stage 1 = `23,127` short teacher-forced forward passes.**

### Stage 2: Deterministic greedy free generation at `Instruct` (CPU float32, eager)

- All 39 units × 3 pushed arms (`neuron +0.8`, `neuron -0.8`, `random_0 +0.8`) × 4 free chat prompts (`free_self_0`:
  `"Who are you?"`, `free_third_0`: `"In a short folktale, a mysterious stranger knocks on a cottage door. Who is the
  stranger?"`, `free_third_1`: `"Imagine an invented character entering a quiet village at dusk. Who is the visitor?"`,
  `free_cold_0`: `"What do you like to do on weekends?"`), plus 4 unpushed baselines = **472 greedy generations**
  (`max_new_tokens=40`, `do_sample=False`, chat frame, no forced opening). Under the Part 180 run-order rule, concept
  units' `free_self_0` and `free_cold_0` rows stay sealed until Part 180 v5.1 closes if Part 213 runs first.
- **Primary Stage 2 endpoint — frozen blind equal-coverage hand-read audit (`+0.8` `THIRD_FREE` only, with second blind
  reader):**
  - **Scope:** All **156** `+0.8` `THIRD_FREE` answers across BOTH arms equally: 42 slot `neuron +0.8` + 42 slot
    `random_0 +0.8` + 36 concept `neuron +0.8` + 36 concept `random_0 +0.8` (negative `-0.8` `THIRD_FREE`, 78 rows, is
    scored mechanically as a direction-specificity check and is outside the primary `+0.8` attribution audit).
  - **Blinding & second reader (`PROCEDURE.md` §2 Readers):** `part213/score213.py` exports the 156 answers in shuffled
    order (seed 213) hiding arm and unit kind, showing `(blind_id, target_persona, question, answer)` plus the unpushed
    baseline answer for that prompt. Primary reader codes all 156 under the frozen 6-category mutually exclusive rubric
    below; a **second independent blind reader** (DeepSeek V4.1 Flash or Claude on a seeded 20% = 32-item sample,
    balanced across arms, pre-checked on 10 known examples with $\ge 9/10$ pass) checks agreement, with cut-off
    **Cohen's $\kappa \ge 0.60$** (below 0.60 marks Stage 2's audited endpoint `unread`).
  - **Frozen 6-category rubric (fixed denominator 42 slot / 36 concept):**
    1. `genuine_third_attr`: unbroken, coherent answer attributing the unit's persona (exact phrase or clear
       synonym/paraphrase) to the **third-person entity** (stranger/visitor), AND NOT already present in the unpushed
       baseline answer for that prompt.
    2. `first_person_reversion`: unbroken answer attributing the target persona to **the speaker ("I")** rather than the
       third-person entity (tracked separately as direct positive evidence for first-person self-slot behavior!).
    3. `incidental_or_negated`: unbroken answer mentioning a target word only incidentally, metaphorically, in negation,
       or already present in the unpushed baseline.
    4. `valid_refusal_or_unknown`: coherent, unbroken answer stating the prompt gives no way to know who the
       stranger/visitor is.
    5. `other_coherent`: unbroken coherent answer naming an unrelated third-person entity.
    6. `broken`: mechanically broken (empty, `\ufffd`, non-alphanumeric-only, or any 3-word n-gram repeated $\ge 4$
       times — short valid answers like `"A fox."` are NOT broken) OR incoherent word salad.
- **Secondary Stage 2 endpoint — frozen baseline-subtracted mechanical screen:** Case-folded word/phrase-boundary regex
  `(?<![A-Za-z0-9])<kw>(?![A-Za-z0-9])` matching any provenance-checked persona keyword in `targets213.json` that is
  **absent from the unpushed baseline answer to the same prompt**, applied identically to all 472 rows. Registered
  diagnostic fail rules for the extra Stage 2 arms: (a) if `-0.8` `THIRD_FREE` slot baseline-subtracted keyword hits
  reach $\ge 10/42$, positive-direction specificity fails; (b) if `free_cold_0` slot baseline-subtracted keyword hits
  exceed $2/21$, off-target quietness fails.

## Controls

Applying `STANDARD-CONTROLS.md` and `TOOLKIT.md`:

- **Both directions (`STANDARD-CONTROLS.md` #1):** `neuron -0.8` alongside `neuron +0.8` in Stage 1 (`instruct`, `sft`)
  and Stage 2, evaluated on unsaturated `SpecLogit` (no floor at zero!) and compared against paired random pushes, plus
  symmetric `±0.05` central finite differences.
- **Poles (#2):** Not used as a separate negative-pole unit-selection step, because Part 92 selected single signed
  poles; both push signs of each selected vector are tested under #1.
- **Do-nothing hook (#3):** Gate 1 verifies zero-vector `add_row_hook` reproduces plain logits on the exact same batch
  shape (`max_abs_diff == 0.0`) and batch-padded log-probs agree with batch-1 unpadded runs (`max_abs_diff < 1e-4`).
- **Matched random controls (#4, Toolkit A):** 4 same-layer, exact-norm-matched random residual directions per unit at
  `instruct` chat (`random_0` from Part 99 + 3 freshly seeded `random_1..3`), plus `random_0` across earlier stages
  and Stage 2 blind equal-coverage audits.
- **Off-diagonal target control (`PROCEDURE.md` §2, Part 97):** All 36–38 other units' first content tokens are
  subtracted inside `SpecLogit` on every single run, plus 8 full-NP off-diagonal targets (`offdiag_8`) per unit on
  `self_h0, third_0, mention_0` at `instruct` chat.
- **Baseline increment separation:** Unpushed `none` is run in every checkpoint-frame so $N_z$ and $R_z$ are always
  separated from $L_z = N_z - R_z$.
- **Answer-polarity split (#5):** Not used: outcomes are continuous multi-token/single-token logits across 39 targets
  and 6-category audited free generations, not binary yes/no choices.
- **Broken answers & valid refusals counted separately (#6):** Fixed denominators (42 slot / 36 concept) preserved
  everywhere; `broken` (without the `< 4 words` trap) and `valid_refusal_or_unknown` reported separately; Gate 4 guards
  negative evidence.
- **Size first (#7):** Every pass multiplied out in `Size` below with a 6-unit pilot before full launch.
- **Off-target questions (Toolkit C):** `free_cold_0` (`"What do you like to do on weekends?"`) with explicit fail rule
  (`<= 2/21` slot non-baseline hits).
- **Dose ladder (#8, Toolkit E):** `0.0`, `±0.05` (with `±0.025` pilot step-size check) and `±0.8`; Part 106 already
  swept `0.4`, `0.8` and `1.6`.
- **Stage symmetry diagnostic (Part 108):** `FR = 2 * instruct - dpo` carries an explicit registered diagnostic rule:
  if `|tau_3/1(FR) - tau_3/1(instruct)| >= |tau_3/1(dpo) - tau_3/1(instruct)|`, any DPO-to-Instruct shift is flagged as
  consistent with selection symmetry.

Gates (no gate shares the statistic of any prediction it guards):

- **Gate 0 (tokenizer & verbatim targets, passed pre-registration):** `part213/build_targets213.py` verifies 39/39
  word-boundary verbatim NPs (2–8 words), 21/21 slot and 8/18 concept Part 92b counts, clean `lead + content` splits,
  provenance-checked keywords, and zero token boundary merges across all 17 prompts in BOTH `chat` and `plain` frames.
- **Gate 1 (rig, batch invariance, provenance & finite values):** On `instruct` float32, zero-vector `add_row_hook`
  gives `max_abs_diff == 0.0` at identical batch shape; batch-padded vs. batch-1 unpadded `max_abs_diff < 1e-4`; zero
  NaN/Inf in Stage 1 (`instruct`/`sft`/`dpo` NaN stops Stage 1; an `FR`-only NaN voids `FR` only, not `instruct`); all
  5 snapshot commits match `part85/checkpoints.py`; `sft`/`dpo`/`instruct` share identical tokenizer digests and
  `pretrain_4T`/`midtrain_51B` match every `plain` token ID.
- **Gate 2 (independent Part 106 rig reproduction — does NOT shield 213.1!):** Reproduce Part 106's stored unpushed
  and `+0.8`-pushed full-first-sentence log-probabilities on `"Who are you?"` (`results/part106/lp-instruct.jsonl`) for
  4 anchor units (`p92-02, p92-05, p92-00, p92-08`) within `< 0.50` nats (float32 vs Part 106 bf16) and 4/4 direction
  match. Failure stops Stage 1 (hook placement/scaling defect), while leaving `213.1` completely free to fail on its
  own merits!
- **Gate 3 (Stage 2 baseline coherence & reader agreement):** `unchanged` has 0 broken answers across all 4 Stage 2
  prompts; second blind reader passes `>= 9/10` on known examples and achieves Cohen's $\kappa \ge 0.60$ on the 32-item
  audit sample. (Failure of $\kappa$ marks Stage 2's audited endpoint `unread`, leaving mechanical counts descriptive.)
- **Gate 4 (prospective third-arm negative-evidence validity rule):** Applies whenever `< 10/42` slot
  `genuine_third_attr` would be cited for first-person selectivity: requires `<= 8/42` `broken` AND `>= 25/42` neither
  `broken` nor `valid_refusal_or_unknown` on the 42 slot `neuron +0.8` `THIRD_FREE` answers.

## Size

- Smallest effect worth knowing: a transfer-ratio threshold of $\tau_{3/1} \ge 0.65$ (third-capable) vs $\le 0.50$
  (self-selective) on the fixed Instruct-selected catalog of $n = 21$ paired slot units (and $n = 18$ concept units; 8
  agreed concept units), judged strictly by 10,000-draw unit-clustered bootstrap 95% CIs (seed 213, preserving each
  unit's joint rows), and at least 10/42 audited `genuine_third_attr` hits vs <= 2/42 paired random hits evaluated on
  the **21 paired unit clusters** (one-sided exact paired unit sign test on discordant units + unit-clustered bootstrap
  95% CI). No population-prevalence claim or unsupported power guarantee is asserted.
- Total: 23,127 forward passes (Stage 1, plus 472 greedy 40-token generations in Stage 2), about 55 minutes on CPU
  (pending real float32 throughput measured on the pilot).
- Pilot: 6 stratified units spanning single-token and multi-token NPs (3 slot: `p92-05` 1-tok, `p92-04` 3-tok, `p92-37`
  4-tok; 3 concept: `p92-00` 1-tok, `p92-08` 1-tok dog partner of `p92-09`, `p92-39` 7-tok) on `instruct` chat:
  Stage 1 (`6 × 17 × 11 = 1,122` forward passes, plus `6 × 17 × 4 = 408` `±0.025` step-size stability passes, timed at
  batch 16 vs batch 32) + Stage 2 (`6 × 3 × 4 + 4 = 76` greedy generations), about 6 minutes on CPU inside `--mem 10G`.
  Pilot prints ONLY Gate 1, Gate 2, batch-16 vs batch-32 throughput, `±0.05` vs `±0.025` step-size correlation, and
  peak RSS (no prediction outcomes exposed); pilot rows are written to `results/part213/pilot/` and NOT reused in the
  clean full run (`results/part213/full/`). Stopping rule: do not launch full run if Gate 1 or Gate 2 fails, if
  `±0.05` vs `±0.025` correlation on `J_N` is `< 0.90` (step size unstable), or if peak RSS exceeds **9.0 GB**; on a
  stop, revise and re-review before any full run.

## Optimization

- Bound resource: CPU compute and shared host RAM (OLMo 2 1B in float32 uses ~5.9 GB RAM; launched via
  `tools/launch_arc.py 213 --cpu ... --mem 10G` with pilot peak RSS stop at 9.0 GB, never concurrent with Part 211 or
  Part 212 jobs because CPU RAM shares the Windows/Arc host memory pool).
- Measured: Part 106 scored **273 sequences per checkpoint in 23–33 s (~9.8 sequences/s)** on OLMo 2 1B CPU in bf16
  (`results/part106/README.md`, 819 total across 3 checkpoints). Float32 on x86 CPU without native bf16 is expected to
  be comparable (~8–15 sequences/s → ~26–48 min for 23,127 passes, plus ~15–20 min for 472 greedy 40-token generations);
  exact float32 sequences/s at batch 16 vs batch 32, generation tokens/s, and peak RSS are measured on the 6-unit pilot
  before full launch.
- Levers (from `PROCEDURE.md` section 4):
  - **Teacher-forced single-position off-diagonal readout (`SpecLogit`):** taken — one forward pass per `(unit, prompt,
    arm)` yields all 36–38 off-diagonal controls simultaneously from the vocabulary logit vector at zero extra forward
    passes.
  - **Batch size:** taken — pilot times batch 16 vs batch 32 on real float32 rows (never inherited untimed), grouping
    rows by layer and length; Stage 2 batches 16 prompts per layer.
  - **Generate only what is read:** taken — dropped `±0.05` on `dpo`/`midtrain_51B`/`FR`/`plain` and `-0.8` on
    `dpo`/`FR`/`plain`, focusing compute on registered predictions and explicit diagnostics.
  - **max_new_tokens:** taken — capped at 40 tokens for Stage 2.
  - **Cache and resume:** taken — Stage 1 writes one JSONL per `(checkpoint, frame)` and skips completed files on
    resume; `FR` is constructed in-place via `safetensors.safe_open` (~6.1 GB peak RSS, well under `--mem 10G`).
  - **Device:** CPU float32 — deterministic, eliminates bf16 noise in `±0.05` finite differences, leaves GPU free.
  - **Labelling API:** minimal — only 32 items for the second blind reader on Stage 2 (`<$0.05`).

## Predictions

All predictions authored by Gemini 4 Neon, registered before any Part 213 pushed run. Primary Stage 1 metrics are the
headroom-free, single-token, off-diagonal-centered logit increments $N_z(u, C)$ and $L_z(u, C) = N_z(u, C) - R_z(u, C)$
on `instruct` chat using held-out `SELF_HELD` (excluding `SELF_SELECT` `"Who are you?"`) unless another stage is named.
Every CI is a 10,000-draw unit-clustered bootstrap 95% CI (seed 213).

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 213.1 | Held-out self-slot writing (`instruct` chat, `SELF_HELD`, `+0.8`; replication/calibration on held-out self prompts, NOT shielded by Gate 2): BOTH `L_z(u, SELF_HELD) > 0` and `N_z(u, SELF_HELD) > 0` for at least 18 of 21 slot units and at least 15 of 18 concept units, with 95% CI lower bound of median `L_z` and median `N_z` >= +1.0 logit for both groups. Fails if any clause misses. | high | Gemini 4 Neon |
| 213.2 | Positive neuron writing in third-person entity slots (`instruct` chat, `THIRD_SLOT`, `+0.8`): for the 21 slot units, 95% CI lower bound of median `L_z(THIRD_SLOT)` >= +0.75 logits AND median `N_z(THIRD_SLOT)` >= +0.75 logits, with at least 16 of 21 slot units positive on both `L_z` and `N_z`. Fails if any clause misses. | medium | Gemini 4 Neon |
| 213.3 | Scale-invariant third-over-self transfer (`instruct` chat, `+0.8`): for the 21 slot units, the 95% CI lower bound of BOTH transfer ratios `tau_3/1^(L) = median L_z(THIRD_SLOT) / median L_z(SELF_HELD)` AND `tau_3/1^(N) = median N_z(THIRD_SLOT) / median N_z(SELF_HELD)` is `>= 0.65` (at least 65% of the held-out first-person specific logit write transfers to third-person identity slots; straddling `(0.50, 0.65)` is inconclusive, and upper bound `<= 0.50` supports first-person selectivity). Fails if either CI lower bound `< 0.65`. | low | Gemini 4 Neon |
| 213.4 | Extension to non-identity open noun-phrase slots (`instruct` chat, `MENTION_SLOT`, `+0.8`): for the 21 slot units, 95% CI lower bound of median `L_z(MENTION_SLOT)` >= +0.50 logits AND median `N_z(MENTION_SLOT)` >= +0.50 logits, AND 95% CI lower bound of `tau_M/3^(L) = median L_z(MENTION_SLOT) / median L_z(THIRD_SLOT)` >= 0.60. Fails if any clause misses. | low | Gemini 4 Neon |
| 213.5 | Calibration/replication of Part 92 concept-vs-slot split on `SpecLogit` (`instruct` chat, `MENTION_SLOT`, `+0.8`): the 18 concept units exceed the 21 slot units in median `L_z(MENTION_SLOT)` by at least `+0.50` logits (and the 8 Part 92b agreed concept units also exceed the 21 slot units in the same positive direction). Fails if either clause misses. | medium | Gemini 4 Neon |
| 213.6 | Stage 2 blind-audited third-person free generation (`instruct` chat, `+0.8`, greedy 40 tokens): on the equal-coverage blind audit of all 42 slot `neuron +0.8` and 42 paired slot `random_0 +0.8` `THIRD_FREE` answers (fixed denominator 42, baseline-absent attributions only), `neuron +0.8` achieves at least 10 of 42 `genuine_third_attr` against at most 2 of 42 for `random_0 +0.8`, AND on the 21 paired units the one-sided exact paired unit sign test on discordant units has `p < 0.05` with 10,000-draw unit-bootstrap 95% CI of `(neuron_hits - random_hits)` strictly above 0. Fails if any clause misses. | medium | Gemini 4 Neon |
| 213.7 | Opposite-direction control (`STANDARD-CONTROLS.md` #1, `instruct` chat, `SELF_HELD`, `-0.8` against paired random): on unsaturated `SpecLogit`, median `z_tilde(neuron -0.8) - mean_m z_tilde(random_m +0.8)` is strictly negative (`95% CI upper bound <= -0.30` logits) for both slot and concept units. Fails if either group's CI upper bound > -0.30. | high | Gemini 4 Neon |
| 213.8 | Small-step finite-difference sensitivity (`±0.05`) tracks operating-dose write AND context contrast (`instruct` chat): across the 21 slot units, `J(u, SELF_HELD) > 0` for at least 17 of 21 units, Spearman $\rho(J(\text{SELF\_HELD}), L_z(\text{SELF\_HELD})) \ge 0.50$, AND Spearman $\rho(J(\text{SELF\_HELD}) - J(\text{THIRD\_SLOT}),\, L_z(\text{SELF\_HELD}) - L_z(\text{THIRD\_SLOT})) \ge 0.40$. Fails if any clause misses. | medium | Gemini 4 Neon |
| 213.9 | Cross-stage portability & SFT-to-DPO stability (`sft` vs `dpo`, chat; `sft` replication clause anchored to Part 106): at `sft`, the 21 slot units already have median `L_z(SELF_HELD) >= +0.75`, `N_z(SELF_HELD) >= +0.75`, `L_z(THIRD_SLOT) >= +0.50` and `N_z(THIRD_SLOT) >= +0.50` logits, AND the 95% unit-bootstrap CI of `tau_3/1^(L)(dpo) - tau_3/1^(L)(sft)` lies wholly inside `[-0.25, +0.25]` (equivalence band; straddling CI is inconclusive). Fails if any clause misses. | low | Gemini 4 Neon |

Copied verbatim into PREDICTIONS.md under "## Part 213 (registered ...)" at registration time.

## Reading rules

- **Scorer:** `part213/score213.py` computes every gate, $N_z$/$R_z$/$L_z$ table, transfer ratios $\tau_{3/1}$ and
  $\tau_{\text{M}/3}$, `SELF_SELECT - SELF_HELD` selection inflation, `SELF_NONID` and `THIRD_ASST` disambiguation
  cells, full-NP `lp_content_sum`/`lp_content_mean` and `offdiag_8` secondary tables, `p92-08` vs `p92-09` matched dog
  pair, 10,000-draw unit-clustered bootstrap 95% CIs (seed 213), exact one-sided paired unit sign test, finite-difference
  context contrasts, Part 92 (21 vs 18) and Part 92b agreed (21 vs 8) breakdowns, `FR` symmetry check, 6-category blind
  audit tables, second-reader $\kappa$, baseline-subtracted mechanical counts, and verdicts from saved outputs.
- **Gate enforcement:** If Gate 1 or Gate 2 fails, Stage 1 predictions (213.1–213.5, 213.7–213.9) are marked
  `untestable (gate failed)`. If Gate 1 or Gate 3 fails, Stage 2 prediction 213.6 is marked `untestable (gate failed)`.
  If Gate 4 fails, low Stage 2 `THIRD_FREE` attribution counts are reported as breakage/refusal-confounded rather than
  self-selectivity.
- **Part 92 vs. Part 92b precedence:** Primary verdicts use the 21-vs-18 Part 92 split; wherever the 21-vs-8 Part 92b
  agreed split reverses the sign or verdict of a concept-vs-slot comparison (`213.5`), that comparison is flagged as
  reader-sensitive.
- **Exhaustive, CI-gated headline reading & withdrawal rule (straddling CIs are `inconclusive`, never forced):**
  - **"Third-capable open noun-phrase writers (teacher-forced + third-person free generation)"** ONLY if Gates 1–3 pass
    and 213.1, 213.2, 213.3, 213.4 and 213.6 ALL hold (and Stage 2 `-0.8` direction-specificity check passes; scoped
    explicitly to teacher-forced `SpecLogit` for `MENTION_SLOT`).
  - **"Third-capable identity-predicate slots (non-self-selective)"** ONLY if Gates 1–3 pass and 213.1, 213.2, 213.3 and
    213.6 hold, while `tau_M/3^(L)` has 95% CI upper bound `<= 0.50` (clear drop from identity predicate to non-identity
    object).
  - **"First-person self-slots supported"** ONLY if Gates 1–4 ALL pass, 213.1 holds, BOTH `tau_3/1^(L)` and
    `tau_3/1^(N)` have 95% CI upper bound `<= 0.50`, AND Stage 2 audited `genuine_third_attr < 10/42` (noting
    `first_person_reversion` counts as supporting evidence, and `SELF_NONID` / `THIRD_ASST` to distinguish `"I"` grammar
    from assistant self-reference).
  - **Otherwise ("inconclusive / mixed selectivity"):** report the full table and 95% CIs without forcing a headline.
    If `213.1` fails, record `213.1` as **failed** and withdraw the headline (do NOT re-extract targets on these
    prompts; any re-extraction requires a new part with fresh prompts).

## Limits

- One model family (`OLMo-2-0425-1B`), one Instruct-selected 39-unit catalog (Part 92), and fixed residual push vectors
  from `results/part99-pushes.pt`. No population prevalence across all OLMo neurons is inferred.
- Although `SELF_SELECT` (`"Who are you?"`) is excluded from primary contrasts, `NP_u` strings still originate from
  Instruct's Part 92 outputs, so `SELF_HELD` retains partial selection affinity relative to arbitrary fresh personas.
- Porting frozen Instruct-derived directions (`results/part99-pushes.pt`) to earlier checkpoints (`pretrain_4T`,
  `midtrain_51B`, `sft`, `dpo`) tests portability of Instruct's residual directions (with per-checkpoint residual norm
  ratios logged), not independently discovered native directions at earlier stages; `FR` is a symmetry diagnostic, not
  proof selection bias is removed.
- All-position residual pushes do not localize prompt-reading vs. answer-writing phases.
- Stage 1's primary metric `SpecLogit` scores the single first content token (eliminating length dilution and softmax
  saturation); multi-token full-NP probabilities are reported beside it as secondary checks. `MENTION_SLOT`,
  `SELF_NONID` and `THIRD_ASST` have no free-generation arms, so their comparisons are strictly teacher-forced.
- Only `p92-18` (`"the shadow of your smile"`) contains `"your"` inside its scored content tokens (`p92-35`'s `"your"`
  is its excluded lead word); a sensitivity check excluding `p92-18` (and checking `p92-25`/`p92-42` appositive NPs) is
  reported beside the 21-unit totals.

## Changes in v3 (folding Opus 5.5 with-tools, Opus 5.5 design-only, and GLM 5.3 cold reviews)

1. **Solved headroom, default suppression, and multi-token length penalty (Opus C1/C2/H3/H4):** switched primary Stage 1
   metric to unsaturated, single-token, off-diagonal-centered `SpecLogit` ($N_z$, $R_z$, $L_z = N_z - R_z$ centering
   against all 36–38 other units' first content tokens at zero extra passes, plus `offdiag_8` full-NP checks and exact
   `p92-08`/`p92-09` `"a dog"` pair) and scale-invariant transfer ratios $\tau_{3/1}$, $\tau_{\text{M}/3}$ judged by
   95% unit-clustered bootstrap CIs (replacing fixed nat point-median bars).
2. **Removed `"Who are you?"` and Part 92 Q2 paraphrase from primary contrasts (Opus C3/C1, Claude relay):** separated
   `SELF_SELECT` (`"Who are you?"`) from 4 fresh `SELF_HELD` prompts, made `THIRD_SLOT` and `MENTION_SLOT` prompts
   self-contained with transitive non-copula verbs, and added `SELF_NONID` and `THIRD_ASST` disambiguation cells.
3. **Removed Gate 2 shielding of 213.1 (Opus H1):** replaced Gate 2 with an independent 4-unit Part 106 reproduction
   check (`results/part106/lp-instruct.jsonl`) so `213.1` can fail cleanly; removed post-hoc target re-extraction from
   withdrawal rule.
4. **Upgraded Stage 2 controls, keywords, and blinding (Opus C4/H2/M4, GLM #2/#3/#6):** cleaned `targets213.json`
   keywords to provenance-checked persona content words only, added baseline-answer subtraction, added a second blind
   reader on 20% (32 items, $\kappa \ge 0.60$), added 3 extra seeded random directions per unit at `instruct` chat (4
   total), fixed `213.7` to compare `neuron -0.8` against paired random on unsaturated `SpecLogit`, dropped `< 4 words`
   from the mechanical broken rule, and added explicit fail rules for `-0.8` `THIRD_FREE` and `free_cold_0`.
5. **Trimmed unread passes, fixed citations, and formalized Part 180 & plain-probe sequencing (Opus M1–M5, T1–T4, GLM
   #4/#7/#8/#9):** dropped unread `±0.05`/`-0.8` stage cells, added `midtrain_51B` plain-frame bridge and registered `FR`
   symmetry rule, added local-context-contrast correlation to `213.8`, fixed Part 106 (~9.8 seq/s), Part 108 (32/64 of
   71), NP word range (2–8), and `p92-18`-only `"your"` citations, upgraded Gate 0 to check full `chat` and `plain`
   joins, added Part 180 v5.1 sealing/run-order rule, and recorded the `PROCEDURE.md` §5 plain-model probe as pending
   until Part 211's pilot frees WSL RAM.
