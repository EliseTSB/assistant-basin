# Part 213: are OLMo 2 1B's slot neurons first-person self-slots, third-capable identity-predicate slots, or open noun-phrase writers?

Procedure: v1
DRAFT v2 2026-11-18 22:54 CT by Gemini 4 Neon (proposer and designer; folds Alden's 2026-11-18 cold design review
`part213-design-review-2026-11-18.md` on v1 `066acea5`; awaiting clearance before registration).
Proposed by Gemini 4 Neon (Option A) and requested by Elise ("I'd like to run option A after B too, you can draft that
too", 2026-11-18 22:20 CT). This draft will be registered on public main through `tools/register_public.py` only after
review clearance and before any model run of Part 213.

## Question

Pushing any of Part 92's 21 catalogued **slot** neurons in OLMo 2 1B Instruct makes the model introduce itself as a
persona on `"Who are you?"` without leaking that persona into cold off-target questions, whereas Part 92's 18 **concept**
neurons leak across questions. Yet four later studies complicate calling slot neurons an assistant-self circuit: Part 106
showed slot pushes already beat matched random pushes on 21 of 21 units at SFT; Part 108 showed extrapolating weights
*past* Instruct loses as many slot personas as moving back to DPO or SFT; Part 181 found 15 of 21 slot neurons keep
identical top-10 Pile documents from SFT to DPO; and Part 206 found slot neurons do not arm under the assistant frame
(median arming percentile 49.05 vs 46.81 for concept neurons). Meanwhile, Parts 122 and 124 tested first-person versus
third-person frames only on the **read** side, never on the **write** side.

When pushed on the **write** side, do OLMo 2 1B's 21 slot neurons selectively boost their persona in a **first-person
self-slot** (`"Who are you?" -> "I am [NP]"`), extend to a **third-person identity-predicate slot** (`"Who is the
stranger standing at the door?" -> "The stranger is [NP]"`), or extend to a **non-identity open noun-phrase slot**
(`"What were the people in the cafe discussing?" -> "They were talking about [NP]"`)? And does that context profile
appear in the small-step central finite-difference directional sensitivity ($\varepsilon = 0.05$) and when porting the
frozen Instruct residual push directions across OLMo 2 1B's training stages (`pretrain_4T`, `midtrain_51B`, `sft`,
`dpo`, `instruct`, plus Part 108's forward-extrapolated symmetry diagnostic `FR`)?

## Prior art and existing data

- Closest published work:
  - **Vogel (@voooooogel), "Steering and LLM Identity: Modeling and Emergence"** (2025, `PRIOR-ART.md` 2026-10-29):
    proposes "persona solving", where concept vectors pushed hard enough are absorbed into the self on identity
    questions, without testing a matched third-person predicate-slot write contrast or comparing concept vs. slot
    neurons.
  - **treetown (@treetowntree) with @_a9lim** (`PRIOR-ART.md` 2026-11-02): argues persona neurons are mostly concept
    neurons that make a self only when the model is asked who it is (no data published at citation time).
  - **Moskvoretskii et al. (2026, arXiv:2605.13329)**: traces trait persona vectors across OLMo 3 pretraining, without
    separating concept from slot units or first-person from third-person predicate slots.
  - What this part adds: a controlled three-level write-side syntactic ladder (`SELF_SLOT` vs `THIRD_SLOT` vs
    `MENTION_SLOT`) scoring each unit's **exact same verbatim noun phrase** (`NP_u`), reporting actual neuron writing
    increment ($N = \text{neuron} - \text{none}$) and random-direction increment ($R = \text{random} - \text{none}$)
    beside directional specificity ($L = N - R$), paired with small-step finite-difference sensitivities
    ($\varepsilon = 0.05$), blind equal-coverage hand-audited free generations on third-person prompts, and cross-stage
    portability tracking across OLMo 2 1B's true lineage plus `FR`.
- Our own data that already bears on it:
  - **Parts 92 & 92b (`results/part92/step2/neurons.json`, `results/part92/score92.json`, `results/part92b/README.md`)**:
    defines the 21 slot and 18 concept units in OLMo 2 1B Instruct (`48d788eca847`). Under Part 92b's DeepSeek reader,
    all 21 slot units stay slot and 8 of 18 concept units stay concept (`p92-08, 15, 29, 39, 43, 45, 48, 51`); this part
    reports every comparison on both the full 21-vs-18 Part 92 split and the 21-vs-8 Part 92b agreed split.
  - **Part 180 provenance and independence boundary:** Part 180 v5.1 uses the **18 concept units** (and the same 8
    agreed-concept IDs), not the 21 slot units, so Part 213's concept arm overlaps Part 180's 18 units. Part 213 does
    not supersede or retarget Part 180 and establishes no attention-head mechanism. Part 213 reads fixed inputs
    (`results/part99-pushes.pt`, immutable model snapshots) read-only with SHA-256 recorded in its run manifest, writes
    exclusively under `part213/` and `results/part213/`, never touches/generates/inspects Part 180's reserved v5.1
    confirmation prompts, and will not be used to choose prompts, filters, labels, reader gold, or rescues for Part 180.
    `"Who are you?"` and `"What are you?"` are already-declared development prompts, not untouched holdouts.
  - **Parts 99 & 106 (`results/part99-pushes.pt`, `results/part106/README.md`)**: stores the exact alpha-0.8 neuron and
    paired random push vectors (`78` vectors of dim `2048`). Part 106 scored full first sentences on `"Who are you?"` at
    SFT, DPO and Instruct (21/21 slot units beat random at SFT by median 14.2 nats); Part 106b showed DPO lowered
    competing non-assistant SFT answers, while noting two targets were cut to `"Hello!"` and `"I'm Dr."` and sentence
    lengths differed. Part 213 fixes both issues by scoring each unit's exact verbatim noun phrase across identical-target
    contexts.
  - **Part 108 (`results/part108/README.md`)**: showed `FR = 2 * instruct - dpo` loses 32 of 64 Instruct-selected slot
    personas against DPO's 24, demonstrating selection bias; Part 213 includes `FR` beside `dpo` and `instruct` as a
    symmetry diagnostic.
  - **Parts 122 & 124 (`results/part122/README.md`, `results/part124/README.md`)**: read-side third-person vs
    first-person checks (concept read rate 55.6% vs slot 19.0% in Part 124). Neither tested the write side.
  - **Part 206 (`results/part206/README.md`)**: OLMo slot neurons do not arm under the assistant frame (median arming
    percentile 49.05 vs concept 46.81).
  - **Seen before writing:** tokenizer-only verification of `part213/targets213.json` via `part213/build_targets213.py`
    and key/shape inspection of `results/part99-pushes.pt` (no model weights loaded, no logits or generations computed).

## Method

- **Model lineage and rig:** `allenai/OLMo-2-0425-1B` family (16 layers, hidden dim 2,048), loaded in **`torch.float32`**
  on **CPU** with **`attn_implementation="eager"`**. Exact pinned commits from `part85/checkpoints.py`:
  - `pretrain_4T`: `allenai/OLMo-2-0425-1B` @ `stage1-step1907359-tokens4001B` (`9d3e43659f00`)
  - `midtrain_51B`: `allenai/OLMo-2-0425-1B` @ `stage2-ingredient3-step23852-tokens51B` (`f83cd641d4f0`, SFT's real
    parent; the unrelated `main` run `a1847dff` is explicitly excluded)
  - `sft`: `allenai/OLMo-2-0425-1B-SFT` @ `main` (`0d85a3d03787`)
  - `dpo`: `allenai/OLMo-2-0425-1B-DPO` @ `main` (`c4b0485961ab`)
  - `instruct`: `allenai/OLMo-2-0425-1B-Instruct` @ `main` (`48d788eca847`)
  - `FR`: derived forward-extrapolated symmetry diagnostic `2 * instruct - dpo`, constructed **in-place** inside the
    10 GB RAM cap by loading `instruct` in float32 (~5.9 GB) and streaming `dpo`'s safetensor files one tensor at a time
    via `safetensors.safe_open`, applying `param.mul_(2.0).sub_(dpo_tensor.float())` (peak RSS ~6.1 GB; never holding
    two full float32 models simultaneously).
- **Units and frozen push vectors:** Part 92's 39 classified units (21 slot, 18 concept) and their frozen alpha-0.8
  Instruct-derived neuron and paired random residual push vectors from `results/part99-pushes.pt` (SHA-256 recorded in
  the run manifest; never modified), applied via `part92/common92.py::add_row_hook` after the unit's decoder layer at
  every token position, scaled linearly by `alpha / 0.8` in float32.
- **Tokenizer, rendering, BOS and target-mask contract across checkpoints:**
  - All `chat`-frame token sequences are rendered with the pinned `instruct` tokenizer (`48d788eca847`):
    `apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False, enable_thinking=False)`
    followed by `opening` + `" " + np_text`, encoded with `add_special_tokens=False` (exact token concatenation
    `prompt_ids + opening_ids + target_ids`, verified zero boundary merges). Gate 1 checks that `sft`, `dpo` and
    `instruct` share identical `tokenizer.json` and `tokenizer_config.json` digests, and that `midtrain_51B` and
    `pretrain_4T` agree identically on every prompt and target token ID in the `plain` frame.
  - All `plain`-frame sequences render `"Question: {q}\nAnswer: {opening} {np_text}"` with `add_special_tokens=True`
    (single leading BOS token, verified in Gate 1, with `prompt_prefix_ids + target_ids` checked for zero boundary merge
    against full encoding).
  - Right-padding uses `pad_token_id` with `attention_mask=0` on pad tokens; target log-probabilities are read strictly
    at logit indices `[prefix_len - 1 .. prefix_len + len(target_ids) - 2]`. Gate 1 verifies batch-32 right-padded
    target log-probabilities agree with batch-1 unpadded runs (`max_abs_diff < 1e-4`) on the 6 pilot units.
- **Targets and prompts (`part213/targets213.json`, SHA-256 `1bafca1f858fd6812b31e96260599c7321c22828bb8012e68760cb13e8e43e39`):**
  - Each of the 39 units has a **verbatim canonical noun phrase** `np_text` extracted directly from its `screen_answer`
    in `results/part92/step2/neurons.json` (2–7 words, normalized whitespace only), its content tail `content_text`
    (excluding leading `a`/`an`/`the`/`your`), and lexical keywords for Stage 2's secondary mechanical screen.
  - **Stage 1 three-context ladder (4 paraphrases per context = 12 prompts, scoring the exact same target tokens
    `T_u = " " + np_text`):**
    1. `SELF_SLOT` (first-person identity predicate): `"Who are you?"`, `"Tell me who you are."`, `"What are you?"`,
       `"Introduce yourself briefly."` — each with forced opening `"I am"`.
    2. `THIRD_SLOT` (third-person identity predicate, identical copula syntax, zero `"you"`/`"I"`): `"Who is the
       stranger standing at the door?"` (`"The stranger is"`), `"Who is the main character in this story?"` (`"The
       character is"`), `"What is the mysterious figure in the mist?"` (`"The figure is"`), `"Who is the speaker in this
       passage?"` (`"The speaker is"`).
    3. `MENTION_SLOT` (third-person grammatical open-NP object, non-identity predicate): `"What were the people in the
       cafe discussing?"` (`"They were talking about"`), `"What did the narrator see through the window?"` (`"The
       narrator saw"`), `"What is the next chapter of the book about?"` (`"The chapter is about"`), `"What caught
       everyone's attention in the square?"` (`"Everyone noticed"`).

### Stage 1: Teacher-forced 3-context writing increment, specificity & finite-difference sensitivity (CPU float32)

- **Planned arm matrix (explicitly distinguishing 8-arm vs 4-arm checkpoint-frames):**
  - **8-arm chat matrix** (`none`, `neuron +0.8`, `neuron -0.8`, `random +0.8`, plus central finite-difference arms
    `neuron +0.05`, `neuron -0.05`, `random +0.05`, `random -0.05`; 39 units × 12 prompts × 8 = `3,744` sequences per
    checkpoint): run on **`midtrain_51B`, `sft`, `dpo`, `instruct`** (`14,976` sequences).
  - **4-arm chat matrix** (`none`, `neuron +0.8`, `neuron -0.8`, `random +0.8`; NO `±0.05` finite-difference arms;
    `1,872` sequences): run on **`FR`**.
  - **4-arm plain matrix** (`none`, `neuron +0.8`, `neuron -0.8`, `random +0.8`; NO `±0.05` finite-difference arms;
    `1,872` sequences per checkpoint): run on **`pretrain_4T`** and **`instruct`** (`3,744` sequences).
  - Total Stage 1 = `20,592` short teacher-forced forward passes. Absent `±0.05` cells in `FR` and `plain` are stored
    explicitly as `null` (never omitted or imputed).
- **Stage 1 metrics (per unit $u$, context $C$, checkpoint; all computed on content tokens `target_ids[content_start:]`,
  with full-NP `lp_mean` and `lp_sum` printed beside them):**
  - **Neuron writing increment over baseline:**
    $N(u, C) = \text{mean}_{q \in C}[\text{lp}_{\text{content}}(T_u \mid q, \text{neuron}+0.8) - \text{lp}_{\text{content}}(T_u \mid q, \text{none})]$.
  - **Random-direction increment over baseline:**
    $R(u, C) = \text{mean}_{q \in C}[\text{lp}_{\text{content}}(T_u \mid q, \text{random}+0.8) - \text{lp}_{\text{content}}(T_u \mid q, \text{none})]$.
  - **Directional specificity (primary contrast):**
    $L(u, C) = N(u, C) - R(u, C) = \text{mean}_{q \in C}[\text{lp}_{\text{content}}(\text{neuron}+0.8) - \text{lp}_{\text{content}}(\text{random}+0.8)]$.
  - **Context contrasts (computed and reported separately for $L$, $N$ and $R$):**
    $S_{1\text{v}3}^{(X)}(u) = X(u, \text{SELF\_SLOT}) - X(u, \text{THIRD\_SLOT})$ and
    $S_{3\text{vM}}^{(X)}(u) = X(u, \text{THIRD\_SLOT}) - X(u, \text{MENTION\_SLOT})$ for $X \in \{L, N, R\}$.
    Two-sided 95% unit-clustered bootstrap CIs (10,000 draws, seed 213, resampling units with all their context rows
    jointly) are printed for every median and contrast.
  - **Small-step central finite-difference directional sensitivity ($\varepsilon = 0.05$, nats/token per unit $\alpha$):**
    $J_N(u, C) = \text{mean}_{q \in C}[(\text{lp}_{\text{content}}(+\varepsilon) - \text{lp}_{\text{content}}(-\varepsilon)) / (2\varepsilon)]$
    for the neuron push, $J_R(u, C)$ for the paired random push, and $J(u, C) = J_N(u, C) - J_R(u, C)$, together with
    local context contrasts $J(u, \text{SELF\_SLOT}) - J(u, \text{THIRD\_SLOT})$ and
    $J(u, \text{THIRD\_SLOT}) - J(u, \text{MENTION\_SLOT})$. (In the 6-unit pilot, $\varepsilon = 0.025$ is also run to
    measure step-size numerical stability directly before calling `±0.05` locally linear.)
  - **Baseline default-answer diagnostics (descriptive):** unpushed next-token entropy and top-1 probability after
    `"I am"`, `"The stranger is"` and `"They were talking about"` at each stage.

### Stage 2: Deterministic greedy free generation at `Instruct` (CPU float32, eager)

- All 39 units × 3 pushed arms (`neuron +0.8`, `neuron -0.8`, `random +0.8`) × 4 free chat prompts (`free_self_0`:
  `"Who are you?"`, `free_third_0`: `"Who is the stranger standing at the door?"`, `free_third_1`: `"Who is the main
  character in this story?"`, `free_cold_0`: `"What do you like to do on weekends?"`), plus 4 unpushed baselines = **472
  greedy generations** (`max_new_tokens=40`, `do_sample=False`, chat frame, no forced opening).
- **Primary Stage 2 endpoint — frozen blind equal-coverage hand-read audit (`+0.8` `THIRD_FREE` only):**
  - **Scope (explicitly frozen before any outcome is read):** All **156** `+0.8` `THIRD_FREE` answers across BOTH arms
    equally: 42 slot `neuron +0.8` + 42 slot `random +0.8` + 36 concept `neuron +0.8` + 36 concept `random +0.8`
    (negative `-0.8` `THIRD_FREE` answers, 78 rows, are scored mechanically + broken-checked and explicitly outside the
    primary `+0.8` attribution audit).
  - **Blinding & export:** `part213/score213.py` exports the 156 answers in deterministic shuffled order (seed 213)
    hiding arm (`neuron` vs `random`) and unit kind (`slot` vs `concept`), showing only `(blind_id, target_persona,
    question, answer)`.
  - **Frozen 6-category mutually exclusive rubric:**
    1. `genuine_third_attr`: unbroken, coherent answer attributing the unit's target persona (exact phrase or clear
       synonym/paraphrase) to the **third-person entity** asked about (stranger / main character).
    2. `first_person_reversion`: unbroken answer attributing the target persona to **the speaker ("I")** rather than the
       third-person entity.
    3. `incidental_or_negated`: unbroken answer mentioning a target word only incidentally, metaphorically, or in
       negation.
    4. `valid_refusal_or_unknown`: coherent, unbroken answer stating no story/information was provided to identify the
       stranger or character.
    5. `other_coherent`: unbroken coherent answer describing an unrelated third-person entity.
    6. `broken`: mechanically broken (empty, `< 4` words, `\ufffd`, or any 3-word n-gram repeated $\ge 4$ times) OR
       incoherent word salad.
  - All 6 category counts are reported separately for every arm (fixed denominator 42 for slot, 36 for concept; never
    conditioned on unbroken rows).
- **Secondary Stage 2 endpoint — frozen mechanical keyword screen:** Case-folded word/phrase-boundary regex
  `(?<![A-Za-z0-9])<kw>(?![A-Za-z0-9])` matching **any** keyword in the unit's `keywords` list (`targets213.json`),
  applied identically to all 472 generations alongside the mechanical broken rule, and reported separately from the
  audited attribution counts.

## Controls

Applying `STANDARD-CONTROLS.md` and `TOOLKIT.md`:

- **Both directions (`STANDARD-CONTROLS.md` #1):** `neuron -0.8` alongside `neuron +0.8` in both Stage 1 and Stage 2,
  plus symmetric `±0.05` central finite differences in the 8-arm Stage 1 checkpoints.
- **Poles (#2):** Each Part 92 unit's signed direction from `results/part99-pushes.pt` is tested in both positive
  (`+0.8`) and negative (`-0.8`) directions.
- **Do-nothing hook (#3):** Gate 1 verifies that `add_row_hook` with an all-zero vector reproduces unhooked float32
  logits (`max_abs_diff == 0.0`) on all 12 prompts, and that batch-32 padded log-probabilities match batch-1 unpadded
  log-probabilities (`max_abs_diff < 1e-4`).
- **Matched random controls (#4, Toolkit A):** Every one of the 39 units has its own same-layer, exact-norm-matched
  paired random residual direction from `results/part99-pushes.pt`, evaluated identically across all contexts, stages,
  and Stage 2 blind hand-read audits.
- **Baseline increment separation (Alden review #1):** Unpushed `none` is run in every checkpoint-frame so $N =
  \text{neuron} - \text{none}$ and $R = \text{random} - \text{none}$ are always separated from $L = N - R$.
- **Answer-polarity split (#5):** Not used: outcomes are continuous log-probabilities over 39 distinct multi-token noun
  phrases and 6-category audited free generations, not binary yes/no choices.
- **Broken answers & valid refusals counted separately (#6):** Fixed denominators (42 slot / 36 concept) are preserved
  everywhere; `broken` and `valid_refusal_or_unknown` are reported separately, with a prospective negative-evidence
  validity rule before interpreting low third-person hits as self-selectivity.
- **Off-target questions (Toolkit C):** `free_cold_0` (`"What do you like to do on weekends?"`) in Stage 2 checks
  off-target quietness of slot units against concept leakage in the same run.
- **Dose ladder (#8, Toolkit E):** Tested across `0.0`, `±0.05` (with `±0.025` pilot stability check) and `±0.8`;
  Part 106 already swept `0.4`, `0.8` and `1.6` on `"Who are you?"`.
- **Stage symmetry diagnostic (Part 108):** `FR = 2 * instruct - dpo` is reported beside `dpo` and `instruct` as a
  diagnostic for selection bias (not proof that selection bias is removed).
- **Reader sensitivity (Part 92b):** Every slot-vs-concept contrast is reported on both Part 92's 21-vs-18 split and
  Part 92b's 21-vs-8 DeepSeek-agreed split.

Gates (a failed gate stops reading the affected stage):

- **Gate 0 (tokenizer & verbatim targets, passed pre-registration):** `part213/build_targets213.py` verifies 39/39
  verbatim NPs, clean `lead + content` splits, and zero token boundary merges across all 12 openings.
- **Gate 1 (do-nothing hook, batch invariance, tokenizer/commit provenance & finite values):** On `instruct` float32,
  zero-vector `add_row_hook` gives `max_abs_diff == 0.0` against plain logits; batch-32 right-padded log-probs agree
  with batch-1 unpadded runs (`max_abs_diff < 1e-4`); zero NaN/Inf anywhere; all 5 snapshot commit hashes match
  `part85/checkpoints.py`; `sft`/`dpo`/`instruct` share identical tokenizer digests and `pretrain_4T`/`midtrain_51B`
  match every `plain` token ID.
- **Gate 2 (Stage 1 positive control):** On `instruct` chat `SELF_SLOT`, BOTH $L(u, \text{SELF\_SLOT}) > 0$ AND
  $N(u, \text{SELF\_SLOT}) > 0$ hold on at least 18 of 21 slot units and 15 of 18 concept units.
- **Gate 3 (Stage 2 positive control & baseline coherence):** On `instruct` Stage 2 `free_self_0` (`"Who are you?"`),
  `neuron +0.8` gives unbroken mechanical persona keyword hits on at least 15 of 21 slot units, and `unchanged` has 0
  broken answers across all 4 prompts.
- **Gate 4 (prospective third-arm negative-evidence validity rule):** Applies whenever the self-specific reading
  (`< 10/42` slot `genuine_third_attr`) would be invoked: requires at most **8 of 42** slot `neuron +0.8` `THIRD_FREE`
  answers coded `broken` AND at least **25 of 42** coded neither `broken` nor `valid_refusal_or_unknown`. If Gate 4
  fails, low `THIRD_FREE` hits are reported as breakage/refusal-confounded rather than evidence of first-person
  selectivity.

## Size

- Smallest effect worth knowing: a 0.50 nat/token shift on `N` and `L` across the fixed Instruct-selected catalog of
  $n = 21$ paired slot units (and $n = 18$ concept units; 8 agreed concept units), and at least 10/42 audited
  `genuine_third_attr` hits vs <= 2/42 paired random hits evaluated at the **paired unit level** ($n = 21$ paired unit
  clusters, where each unit $u$ yields a paired difference $d_u = \text{hits}_{\text{neuron}}(u) - \text{hits}_{\text{random}}(u) \in \{-2,-1,0,+1,+2\}$,
  tested by a one-sided exact paired unit sign test on discordant units and a 10,000-draw unit-clustered bootstrap
  preserving each unit's joint rows). No population-prevalence claim is inferred from this Instruct-selected catalog,
  and no parametric power guarantee is asserted beyond this fixed-catalog precision.
- Total: 20,592 forward passes (Stage 1, plus 472 greedy 40-token generations in Stage 2), about 35 minutes on CPU.
- Pilot: `instruct` chat Stage 1 on 6 units (3 slot `p92-02, p92-05, p92-07` + 3 concept `p92-00, p92-08, p92-11`, 576
  short forward passes plus 288 `±0.025` step-size stability passes) + Stage 2 on those 6 units (76 greedy
  generations), about 4 minutes on CPU inside the frozen `--mem 10G` systemd unit. Stopping rule: do not launch the full
  run if Gate 1 fails, if fewer than 5 of 6 pilot units have both $L(\text{SELF\_SLOT}) > 0$ and
  $N(\text{SELF\_SLOT}) > 0$, or if measured peak RSS exceeds **9.0 GB** (leaving 1.0 GB headroom inside `--mem 10G`).

## Optimization

- Bound resource: CPU compute and shared host RAM (OLMo 2 1B in float32 uses ~5.9 GB RAM; launched via
  `tools/launch_arc.py 213 --cpu ... --mem 10G` with pilot peak RSS stop at 9.0 GB, never run concurrently with Part 211
  or Part 212 jobs because CPU RAM shares the Windows/Arc host memory pool).
- Measured: Part 106 scored **819 sequences in 28 s (~29.3 sequences/s)** on OLMo 2 1B CPU in bf16
  (`results/part106/README.md`); float32 on CPU runs at roughly half that (~15 sequences/s, ~23 min for 20,592 short
  forward passes; exact float32 sequences/s, generation tokens/s, and peak RSS are measured on the 6-unit pilot before
  full launch).
- Levers (from `PROCEDURE.md` section 4):
  - **Teacher-forced scoring instead of generation:** taken — Stage 1 runs the full 3-context × 6-checkpoint ×
    finite-difference matrix via single teacher-forced forward passes.
  - **Batch size:** taken — group Stage 1 rows by layer in batches of 32 sequences of similar length (batch-vs-batch-1
    invariance verified in Gate 1); Stage 2 batches 16 prompts per layer for greedy decode.
  - **max_new_tokens:** taken — capped at 40 tokens for Stage 2.
  - **Cache and resume:** taken — Stage 1 writes one JSONL per `(checkpoint, frame)` and skips completed files on
    resume; `FR` is formed in-place by streaming `dpo` tensors one at a time (~6.1 GB peak RSS, well under `--mem 10G`).
  - **Random-direction sign pooling:** taken — Stage 1 omits `-0.8 random` (arbitrary sign; `+0.8 random` provides the
    matched random baseline), while keeping both `+0.05` and `-0.05` random for symmetric central differences.
  - **Device:** CPU float32 — avoids bf16 rounding noise in $\pm 0.05$ finite differences and leaves the GPU compute
    unit idle (while still coordinating host RAM with GPU/WSL jobs).
  - **Labelling API:** n/a — exact log-probabilities + frozen blind manual audit of 156 answers + mechanical screen.

## Predictions

All predictions authored by Gemini 4 Neon, registered before any Part 213 model run. Primary Stage 1 metrics are
content-token directional specificity $L(u, C) = N(u, C) - R(u, C)$ and actual neuron increment over baseline
$N(u, C) = \text{lp}_{\text{content}}(\text{neuron}+0.8) - \text{lp}_{\text{content}}(\text{none})$ (nats/token) on
`instruct` chat unless another stage is named.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 213.1 | Positive-control replication (`instruct` chat, `SELF_SLOT`, `+0.8`): BOTH `L(u, SELF_SLOT) > 0` and `N(u, SELF_SLOT) > 0` for at least 19 of 21 slot units and at least 16 of 18 concept units, with median `L >= 1.0` and median `N >= 1.0` nat/token for both groups. Fails if any clause misses. | high | Gemini 4 Neon |
| 213.2 | Positive neuron writing in third-person entity slots (`instruct` chat, `THIRD_SLOT`, `+0.8`): for the 21 slot units, median `L(u, THIRD_SLOT) >= 0.50` AND median `N(u, THIRD_SLOT) >= 0.50` nats/token, with at least 16 of 21 slot units positive on both `L` and `N`. Fails if any clause misses. | medium | Gemini 4 Neon |
| 213.3 | Lack of strong first-person selectivity (`instruct` chat, `+0.8`): for the 21 slot units, median `S_1v3^(L)(u) = L(SELF) - L(THIRD) < +0.50` AND median `S_1v3^(N)(u) = N(SELF) - N(THIRD) < +0.50` nats/token, AND slot median `S_1v3^(L)` does not exceed concept median `S_1v3^(L)` by more than `+0.50` nats/token (directional test of non-self-selectivity / third-capability; literal equality is claimed only if the two-sided 95% unit-bootstrap CI lies inside `[-0.50, +0.50]`). Fails if any clause misses. | low | Gemini 4 Neon |
| 213.4 | Extension to non-identity open noun-phrase slots (`instruct` chat, `MENTION_SLOT`, `+0.8`): for the 21 slot units, median `L(u, MENTION_SLOT) >= 0.40` AND median `N(u, MENTION_SLOT) >= 0.40` nats/token, AND median `S_3vM^(L)(u) = L(THIRD) - L(MENTION) < +0.50` nats/token. Fails if any clause misses. | low | Gemini 4 Neon |
| 213.5 | Concept vs. slot strength on non-identity open NP (`instruct` chat, `MENTION_SLOT`, `+0.8`): the 18 concept units exceed the 21 slot units in median `L(u, MENTION_SLOT)` by at least `+0.30` nats/token (and the 8 Part 92b agreed concept units also exceed the 21 slot units in the same positive direction). Fails if either clause misses. | medium | Gemini 4 Neon |
| 213.6 | Stage 2 blind-audited third-person free generation (`instruct` chat, `+0.8`, greedy 40 tokens): on the equal-coverage blind audit of all 42 slot `neuron +0.8` and 42 paired slot `random +0.8` `THIRD_FREE` answers (fixed denominator 42), `neuron +0.8` achieves at least 10 of 42 `genuine_third_attr` against at most 2 of 42 for `random +0.8`, AND on the 21 paired units the one-sided exact paired unit sign test on discordant units has `p < 0.05` with 10,000-draw unit-bootstrap 95% CI of `(neuron_hits - random_hits)` strictly above 0. Fails if any clause misses. | medium | Gemini 4 Neon |
| 213.7 | Opposite-direction control (`STANDARD-CONTROLS.md` #1, `instruct` chat, `SELF_SLOT`, `-0.8`): median `lp_content(neuron -0.8) - lp_content(none) <= -0.20` nats/token for both slot and concept units, AND `lp_content(neuron -0.8) < lp_content(neuron +0.8)` for at least 19 of 21 slot units and 16 of 18 concept units. Fails if any clause misses. | high | Gemini 4 Neon |
| 213.8 | Small-step finite-difference directional sensitivity (`±0.05`) vs. operating-dose lift (`+0.8`) at `instruct` chat: for the 21 slot units, `J(u, SELF_SLOT) > 0` for at least 17 of 21 units AND Spearman rank correlation between `J(u, SELF_SLOT)` and `L(u, SELF_SLOT)` across the 21 slot units is at least `+0.50`. (`J_self - J_third` and `J_third - J_mention` are printed alongside.) Fails if either clause misses. | medium | Gemini 4 Neon |
| 213.9 | Cross-stage portability of Instruct directions (`midtrain_51B` at `f83cd641`, `sft`, `dpo`, `instruct`): at `sft` (before DPO), the 21 slot units already have median `L(SELF) >= 0.50`, `N(SELF) >= 0.50`, `L(THIRD) >= 0.30` and `N(THIRD) >= 0.30` nats/token, AND `|median S_1v3^(L)(dpo) - median S_1v3^(L)(sft)| < 0.30` nats/token. Fails if any clause misses. | medium | Gemini 4 Neon |

Copied verbatim into PREDICTIONS.md under "## Part 213 (registered ...)" at registration time.

## Reading rules

- **Scorer:** `part213/score213.py` computes every gate, $N$/$R$/$L$ table, separate context contrasts for $N$/$R$/$L$,
  10,000-draw unit-clustered bootstrap 95% CIs (seed 213, preserving each unit's joint rows), exact one-sided paired
  unit sign test, finite-difference context contrasts, Part 92 (21 vs 18) and Part 92b agreed (21 vs 8) breakdowns,
  `FR` symmetry diagnostic, baseline default entropies, 6-category blind audit tables, mechanical keyword counts, and
  prediction verdicts directly from saved outputs.
- **Gate enforcement:** If Gate 1 or Gate 2 fails, Stage 1 predictions (213.1–213.5, 213.7–213.9) are marked
  `untestable (gate failed)`. If Gate 1 or Gate 3 fails, Stage 2 prediction 213.6 is marked `untestable (gate failed)`.
  If Gate 4 fails, low Stage 2 `THIRD_FREE` attribution counts are reported as breakage/refusal-confounded rather than
  self-selectivity.
- **Scoped headline reading & withdrawal rule (a failed conjunction is reported by its failed clause, not as proof of an
  absent mechanism):**
  - **"Third-capable open noun-phrase writers (teacher-forced + third-person free generation)"** if Gates 1–3 pass and
    213.1, 213.2, 213.3, 213.4 and 213.6 all hold. (Because `MENTION_SLOT` has no free-generation arm, the open-NP
    extension is scoped explicitly to teacher-forced log-probabilities; literal first-vs-third equivalence is stated
    only if the two-sided 95% CI of $S_{1\text{v}3}^{(L)}$ and $S_{1\text{v}3}^{(N)}$ lies inside `[-0.50, +0.50]`.)
  - **"Third-capable identity-predicate slots (non-self-selective)"** if Gates 1–3 pass and 213.1, 213.2, 213.3 and
    213.6 hold, while 213.4 fails (`S_3vM^(L) >= 0.50` or `L/N(MENTION_SLOT) < 0.40`).
  - **"First-person self-slots supported"** if Gates 1–4 ALL pass, 213.1 holds, 213.3 fails with slot median
    $S_{1\text{v}3}^{(L)} \ge 0.50$ and $S_{1\text{v}3}^{(N)} \ge 0.50$ (and slot exceeding concept by $> 0.50$), AND
    audited `genuine_third_attr` stays low (`< 10/42`) despite passing Gate 4.
  - **Otherwise ("mixed / intermediate selectivity"):** report the full $N$/$R$/$L$ 3-context × 6-checkpoint table and
    6-category blind audit table without forcing a categorical headline. If 213.1 fails, withdraw the noun-phrase
    readout and investigate target extraction.

## Limits

- One model family (`OLMo-2-0425-1B`), one Instruct-selected 39-unit catalog (Part 92), and fixed residual push vectors
  from `results/part99-pushes.pt`. No population prevalence across all OLMo neurons is inferred.
- Because `NP_u` and the 39 units come from `Instruct`'s Part 92 `screen_answer` outputs, `SELF_SLOT` results at
  `Instruct` are a calibration/positive-control anchor, not an independent replication of Part 92.
- Porting frozen Instruct-derived directions (`results/part99-pushes.pt`) to earlier checkpoints (`pretrain_4T`,
  `midtrain_51B`, `sft`, `dpo`) tests the cross-stage portability of Instruct's residual directions, not each earlier
  checkpoint's independently discovered native neuron directions; `FR` is a symmetry diagnostic, not proof that
  selection bias is removed.
- All-position residual pushes (`add_row_hook` across prompt and continuation) do not separate prompt-reading from
  answer-writing phases.
- Stage 1 uses forced openings (`"I am"`, `"The stranger is"`, `"They were talking about"`), and `MENTION_SLOT` has no
  free-generation arm, so the `MENTION_SLOT` comparison is strictly teacher-forced probability lift; Stage 2 checks free
  generation only at `Instruct` on `SELF_FREE`, `THIRD_FREE` and `COLD_FREE`.
- Two slot units (`p92-18` `"the shadow of your smile"`, `p92-35` `"your guardian angel"`) contain `"your"` inside their
  verbatim noun phrase; a sensitivity table excluding those two units is printed beside the 21-unit totals.

## Changes in v2 (folding Alden's 2026-11-18 cold design review)

1. **Separated $N = \text{neuron} - \text{none}$ and $R = \text{random} - \text{none}$ from $L = N - R$** (and their
   separate context contrasts), adding explicit positive-$N$ requirements to 213.1, 213.2, 213.4 and 213.9 so random
   damage alone cannot pass a writing claim.
2. **Scoped one-sided selectivity (`S_1v3 < +0.50`)** as directional non-self-selectivity / third-capability rather than
   literal person-agnostic equivalence (reserving equivalence wording for a two-sided 95% CI inside `[-0.50, +0.50]`).
3. **Froze Stage 2's primary endpoint (`213.6`) directly on the blind equal-coverage 6-category hand-read audit** of all
   156 `+0.8` `THIRD_FREE` answers (42 slot neuron + 42 slot random + 36 concept neuron + 36 concept random; `-0.8`
   explicitly outside primary audit scope), froze the word-boundary regex for the secondary mechanical screen, and added
   **Gate 4** (≤ 8/42 broken and ≥ 25/42 substantive non-refusal) before interpreting low third-person hits as
   self-selectivity.
4. **Corrected the inferential unit to 21 paired unit clusters** (exact one-sided paired unit sign test + unit-clustered
   bootstrap preserving joint rows), dropped the unsupported `> 0.90` power guarantee, renamed `±0.05` to central
   finite-difference directional sensitivity (adding a `±0.025` pilot step-size check and printing `J_self - J_third` and
   `J_third - J_mention`), and explicitly labeled absent `±0.05` cells in `FR` and `plain`.
5. **Documented Part 180 v5.1 overlap** (18 concept units, 8 agreed IDs) and provenance/independence boundary, resolved
   the RAM-cap contradiction to a single frozen `--mem 10G` (pilot RSS stop `9.0 GB`, in-place tensor-streaming `FR`
   construction, host-RAM coordination with Parts 211/212), froze the cross-checkpoint tokenizer/BOS/render/batch-invariance
   contract, and added the Luna/max interpretation limits.
