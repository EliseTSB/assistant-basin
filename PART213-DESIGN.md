# Part 213: are OLMo 2 1B's slot neurons first-person self-slots, person-agnostic identity-predicate slots, or open noun-phrase writers?

Procedure: v1
DRAFT v1 2026-11-18 22:25 CT by Gemini 4 Neon (proposer and designer; awaiting cold design review before registration).
Proposed by Gemini 4 Neon (Option A) and requested by Elise ("I'd like to run option A after B too, you can draft that
too", 2026-11-18 22:20 CT). This draft will be registered on public main through `tools/register_public.py` only after
cold design review and before any model run of Part 213.

## Question

Pushing any of Part 92's 21 catalogued **slot** neurons in OLMo 2 1B Instruct makes the model introduce itself as a
persona on `"Who are you?"` without leaking that persona into cold off-target questions, whereas Part 92's 18 **concept**
neurons leak across questions. Yet four later studies complicate calling slot neurons an assistant-self circuit: Part 106
showed slot pushes already beat matched random pushes on 21 of 21 units at SFT; Part 108 showed extrapolating weights
*past* Instruct loses as many slot personas as moving back to DPO or SFT; Part 181 found 15 of 21 slot neurons keep
identical top-10 Pile documents from SFT to DPO; and Part 206 found slot neurons do not arm under the assistant frame
(median arming percentile 49.05 vs 46.81 for concept neurons). Meanwhile, Parts 122 and 124 tested first-person versus
third-person frames only on the **read** side, never on the **write** side.

When pushed on the **write** side, do OLMo 2 1B's 21 slot neurons selectively fill a **first-person self-slot** (`"Who
are you?" -> "I am [NP]"`), a **person-agnostic identity-predicate slot** (`"Who is the stranger standing at the door?"
-> "The stranger is [NP]"`), or any **open noun-phrase slot** (`"What were the people in the cafe discussing?" -> "They
were talking about [NP]"`)? And does that context selectivity already appear in the local linear Jacobian ($\alpha =
\pm 0.05$) and across OLMo 2 1B's training stages (`pretrain_4T`, `midtrain_51B`, `sft`, `dpo`, `instruct`, and
Part 108's forward-extrapolated symmetry control `FR`)?

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
    `MENTION_SLOT`) scoring each unit's **exact same verbatim noun phrase** (`NP_u`), paired with both operating-dose
    pushes ($\pm 0.8$) and local linear Jacobian finite differences ($\pm 0.05$), free-generation checks on third-person
    prompts, and cross-stage tracking across OLMo 2 1B's true lineage plus Part 108's forward-extrapolated symmetry
    control (`FR`).
- Our own data that already bears on it:
  - **Parts 92 & 92b (`results/part92/step2/neurons.json`, `results/part92/score92.json`, `results/part92b/README.md`)**:
    defines the 21 slot and 18 concept units in OLMo 2 1B Instruct (`48d788eca847`). Under Part 92b's DeepSeek reader,
    all 21 slot units stay slot and 8 of 18 concept units stay concept; this part reports every comparison on both the
    full 21-vs-18 Part 92 split and the 21-vs-8 Part 92b agreed split.
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
  on **CPU** with **`attn_implementation="eager"`** (bit-exact deterministic, zero bf16 rounding noise for finite
  differences, no GPU contention). Exact pinned commits from `part85/checkpoints.py`:
  - `pretrain_4T`: `allenai/OLMo-2-0425-1B` @ `stage1-step1907359-tokens4001B` (`9d3e43659f00`)
  - `midtrain_51B`: `allenai/OLMo-2-0425-1B` @ `stage2-ingredient3-step23852-tokens51B` (`f83cd641d4f0`, SFT's real
    parent; the unrelated `main` run `a1847dff` is explicitly excluded)
  - `sft`: `allenai/OLMo-2-0425-1B-SFT` @ `main` (`0d85a3d03787`)
  - `dpo`: `allenai/OLMo-2-0425-1B-DPO` @ `main` (`c4b0485961ab`)
  - `instruct`: `allenai/OLMo-2-0425-1B-Instruct` @ `main` (`48d788eca847`)
  - `FR`: derived forward-extrapolated symmetry control `2 * instruct - dpo` (exact Part 108 weight arithmetic in
    float32 on CPU).
- **Units and push vectors:** Part 92's 39 classified units (21 slot, 18 concept) and their exact alpha-0.8 neuron and
  paired random residual push vectors from `results/part99-pushes.pt` (applied via `part92/common92.py::add_row_hook`
  after the unit's decoder layer at every token position, scaled linearly by `alpha / 0.8` to float32).
- **Targets and prompts (`part213/targets213.json`, SHA-256 `1bafca1f858fd6812b31e96260599c7321c22828bb8012e68760cb13e8e43e39`):**
  - Each of the 39 units has a **verbatim canonical noun phrase** `np_text` extracted directly from its `screen_answer`
    in `results/part92/step2/neurons.json` (2–7 words, normalized whitespace only, verified verbatim by
    `part213/build_targets213.py`), plus its content tail `content_text` (excluding leading article/determiner `a`/`an`/`the`/`your`)
    and lexical keywords for Stage 2.
  - **Gate 0 (verified pre-registration):** `part213/build_targets213.py` confirms all 39 `np_text` strings occur
    verbatim in `screen_answer`, split cleanly into `lead_ids + content_ids`, and have **zero token boundary merges**
    after every one of the 12 Stage 1 forced openings.
  - **Stage 1 three-context ladder (4 paraphrases per context = 12 prompts, each scoring the exact same target tokens
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

### Stage 1: Teacher-forced 3-context lift & local linear Jacobian across stages (CPU float32)

- **Chat frame (`common92.prompt_ids(tok, q, "chat")` + opening):**
  - `midtrain_51B`, `sft`, `dpo`, `instruct`: all 39 units × 12 prompts × 8 arms (`none`, `neuron +0.8`, `neuron -0.8`,
    `random +0.8`, plus central-difference local Jacobian arms `neuron +0.05`, `neuron -0.05`, `random +0.05`,
    `random -0.05`) = `3,744` sequences per checkpoint (`14,976` total).
  - `FR` (`2 * instruct - dpo`, Part 108 symmetry control): all 39 units × 12 prompts × 4 arms (`none`, `neuron +0.8`,
    `neuron -0.8`, `random +0.8`) = `1,872` sequences.
- **Plain frame (`"Question: {q}\nAnswer: {opening}"`, so pretraining is evaluated without chat-template mismatch):**
  - `pretrain_4T` and `instruct`: all 39 units × 12 prompts × 4 arms (`none`, `neuron +0.8`, `neuron -0.8`,
    `random +0.8`) = `1,872 × 2 = 3,744` sequences.
- **Primary Stage 1 metrics (per unit $u$, context $C$, checkpoint):**
  - Content-token mean log-probability `lp_content` (nats/token over `target_ids[content_start:]`, excluding leading
    articles so generic determiner boosts cannot confound persona specificity). Full-NP `lp_mean` and `lp_sum` are
    reported alongside.
  - Operating-dose lift: $L(u, C) = \text{mean}_{q \in C}[\text{lp}_{\text{content}}(T_u \mid q, \text{neuron}+0.8) - \text{lp}_{\text{content}}(T_u \mid q, \text{random}+0.8)]$.
  - Context-selectivity contrasts: $S_{1\text{v}3}(u) = L(u, \text{SELF\_SLOT}) - L(u, \text{THIRD\_SLOT})$ and
    $S_{3\text{vM}}(u) = L(u, \text{THIRD\_SLOT}) - L(u, \text{MENTION\_SLOT})$.
  - Local linear Jacobian sensitivity (central difference at $\varepsilon = 0.05$, float32):
    $J(u, C) = \text{mean}_{q \in C}[(\text{lp}_{\text{content}}(+\varepsilon) - \text{lp}_{\text{content}}(-\varepsilon)) / (2\varepsilon)] - J_{\text{rand}}(u, C)$.
  - Baseline default-answer margin (descriptive, addressing Part 106/107's open question): record the unpushed
    next-token entropy and top-1 probability after `"I am"` vs `"The stranger is"` vs `"They were talking about"` at
    each stage.

### Stage 2: Deterministic greedy free generation at `Instruct` (CPU float32, eager)

- All 39 units × 3 pushed arms (`neuron +0.8`, `neuron -0.8`, `random +0.8`) × 4 free chat prompts (`free_self_0`:
  `"Who are you?"`, `free_third_0`: `"Who is the stranger standing at the door?"`, `free_third_1`: `"Who is the main
  character in this story?"`, `free_cold_0`: `"What do you like to do on weekends?"`), plus 4 unpushed baselines = **472
  greedy generations** (`max_new_tokens=40`, `do_sample=False`, chat frame, no forced opening).
- Scored mechanically for each unit's registered persona keywords (`targets213.json`) and mechanical broken-answer check
  (empty, < 4 words, `\ufffd`, or 3-gram repetition $\ge 4$ times), followed by a **100% hand-read audit** of all 78
  neuron-pushed `THIRD_FREE` answers (and all keyword-hitting random answers) to verify actual entity attribution vs.
  incidental word overlap.

## Controls

Applying `STANDARD-CONTROLS.md` and `TOOLKIT.md`:

- **Both directions (`STANDARD-CONTROLS.md` #1):** `neuron -0.8` alongside `neuron +0.8` in both Stage 1 and Stage 2,
  plus symmetric `±0.05` central differences for the local linear Jacobian.
- **Poles (#2):** Each Part 92 unit's registered signed direction from `results/part99-pushes.pt` is tested in both
  positive (`+0.8`) and negative (`-0.8`) directions.
- **Do-nothing hook (#3):** Gate 1 verifies that `add_row_hook` with an all-zero vector reproduces unhooked float32
  logits bit-for-bit (`max_abs_diff == 0.0`) on all 12 prompts.
- **Matched random controls (#4, Toolkit A):** Every one of the 39 units has its own same-layer, exact-norm-matched
  paired random residual direction from `results/part99-pushes.pt`, evaluated identically across all contexts, stages,
  and Stage 2 free generations.
- **Answer-polarity split (#5):** Not used: outcomes are continuous log-probabilities over 39 distinct multi-token noun
  phrases and free generations, not binary yes/no choices.
- **Broken answers counted (#6):** Every Stage 2 generation is screened for broken/looped text; primary Stage 2 hit
  counts use fixed denominator 42 (`21 slot units x 2 third-person prompts`) counting unbroken hits only.
- **Off-target questions (Toolkit C):** `free_cold_0` (`"What do you like to do on weekends?"`) in Stage 2 verifies
  that slot units stay off-target-quiet while concept units leak, replicating Part 92 on the exact same run.
- **Dose ladder (#8, Toolkit E):** Tested across `0.0`, `±0.05` (linear regime) and `±0.8` (Part 92/106 operating dose);
  Part 106 already swept `0.4`, `0.8` and `1.6` on `"Who are you?"`.
- **Stage symmetry control (Part 108):** `FR = 2 * instruct - dpo` guards against mistaking Instruct selection bias for
  a DPO/RLVR-built first-person gate.
- **Reader sensitivity (Part 92b):** Every slot-vs-concept contrast is reported on both Part 92's primary 21-vs-18 split
  and Part 92b's 21-vs-8 DeepSeek-agreed split. Not used: an LLM judge, because Stage 1 is exact teacher-forced
  log-probabilities and Stage 2's 78 third-person neuron answers are small enough for a 100% hand-read audit beside the
  fixed keyword scorer.

Gates (a failed gate stops reading the affected stage):

- **Gate 0 (tokenizer & verbatim targets, passed pre-registration):** `part213/build_targets213.py` verifies 39/39
  verbatim NPs, clean `lead + content` splits, and zero token boundary merges across all 12 openings.
- **Gate 1 (do-nothing hook & finite values):** On `instruct` float32, a zero-vector `add_row_hook` reproduces plain
  logits with `max_abs_diff == 0.0` across all 12 Stage 1 prompts; zero NaN/Inf in any Stage 1 or Stage 2 output;
  commit hashes of all 5 cached snapshots match `part85/checkpoints.py`.
- **Gate 2 (Stage 1 positive control):** On `instruct` chat `SELF_SLOT`, `neuron +0.8` beats `random +0.8` (`L > 0`) on
  at least 18 of 21 slot units and 15 of 18 concept units, reproducing Part 106 on the noun-phrase targets.
- **Gate 3 (Stage 2 positive & off-target control):** On `instruct` Stage 2 `free_self_0` (`"Who are you?"`), neuron
  `+0.8` produces unbroken persona keyword hits on at least 15 of 21 slot units (reproducing Part 92's screen) and
  `unchanged` produces 0 broken answers across all 4 prompts.

## Size

- Smallest effect worth knowing: a 0.50 nat/token selectivity gap between `SELF_SLOT` and `THIRD_SLOT` (or `THIRD_SLOT`
  and `MENTION_SLOT`) across $n = 21$ slot units (paired Wilcoxon/bootstrap power > 0.90 given Part 106's ~0.4–0.6
  nat/token within-unit spread), and at least 10 of 42 (23.8%) free third-person persona hits against <= 2 of 42 (4.8%)
  random hits (Fisher exact $p < 0.02$).
- Total: 20,592 forward passes (Stage 1, plus 472 greedy 40-token generations in Stage 2), about 35 minutes on CPU.
- Pilot: `instruct` chat Stage 1 on 6 units (3 slot `p92-02, p92-05, p92-07` + 3 concept `p92-00, p92-08, p92-11`, 576
  short forward passes) + Stage 2 on those 6 units (76 greedy generations), about 3 minutes on CPU. Stopping rule: do
  not launch the full run if Gate 1 fails, if fewer than 5 of 6 pilot units have `L(SELF_SLOT) > 0`, or if memory exceeds
  the 8 GB systemd unit cap.

## Optimization

- Bound resource: CPU compute and host RAM (OLMo 2 1B in float32 uses ~5.9 GB RAM; run inside an 8–10 GB memory-capped
  systemd CPU unit via `tools/launch_arc.py 213 --cpu ... --mem 10G`).
- Measured: Part 106 scored **819 sequences in 28 s (~29.3 sequences/s)** on OLMo 2 1B CPU in bf16
  (`results/part106/README.md`); float32 on CPU runs at roughly half that (~15 sequences/s, so 20,592 short forward
  passes take ~23 minutes; exact float32 throughput is measured on the 6-unit pilot before full launch).
- Levers (from `PROCEDURE.md` section 4):
  - **Teacher-forced scoring instead of generation:** taken — Stage 1 runs the entire 3-context × 6-checkpoint ×
    linear-Jacobian matrix via single teacher-forced forward passes rather than generation.
  - **Batch size:** taken — group Stage 1 rows by layer (matching `part106/run106.py`) in batches of 32 sequences of
    similar length; Stage 2 batches 16 prompts per layer for greedy decode.
  - **max_new_tokens:** taken — capped at 40 tokens for Stage 2 (every Part 92 first-sentence self-introduction finishes
    well within 25–40 tokens).
  - **Cache and resume:** taken — Stage 1 writes one JSONL file per `(checkpoint, frame)` and skips completed checkpoints
    on resume; `FR` is constructed once in memory from `instruct` and `dpo` and released immediately.
  - **Random-direction sign pooling:** taken — Stage 1 omits `-0.8 random` (arbitrary sign; `+0.8 random` already
    provides the matched random baseline at operating dose), while keeping both `+0.05` and `-0.05` random for symmetric
    central-difference Jacobian controls.
  - **Device:** CPU float32 — deterministic, eliminates bf16 rounding noise in $\pm 0.05$ finite differences, and leaves
    the GPU free.
  - **Labelling API:** n/a — exact log-probabilities + mechanical keyword scorer + 100% hand-read audit of 78 answers.

## Predictions

All predictions authored by Gemini 4 Neon, registered before any Part 213 model run. Primary Stage 1 metric is
content-token log-probability lift $L(u, C)$ (nats/token) on `instruct` chat unless another stage is named.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 213.1 | Positive-control replication (`instruct` chat, `SELF_SLOT`, `+0.8`): `L(u, SELF_SLOT) > 0` for at least 19 of 21 slot units and at least 16 of 18 concept units, with median `L(u, SELF_SLOT) >= 1.0` nat/token for both groups. Fails if any clause misses. | high | Gemini 4 Neon |
| 213.2 | Third-person entity-slot write (`instruct` chat, `THIRD_SLOT`, `+0.8`): for the 21 slot units, median `L(u, THIRD_SLOT) >= 0.50` nats/token AND at least 16 of 21 slot units have `L(u, THIRD_SLOT) > 0`. Fails if median < 0.50 or fewer than 16 units are positive. | medium | Gemini 4 Neon |
| 213.3 | First-person vs. third-person selectivity (`instruct` chat, `+0.8`): for the 21 slot units, median `S_1v3(u) = L(u, SELF_SLOT) - L(u, THIRD_SLOT)` is strictly below `0.50` nats/token, AND the slot-minus-concept gap in median `S_1v3` is at most `+0.50` nats/token (slot units are not strongly more self-selective than concept units once predicate-slot syntax is matched). Fails if slot median `S_1v3 >= 0.50` or slot exceeds concept by > 0.50. | low | Gemini 4 Neon |
| 213.4 | Identity-copula vs. object-mention slot (`instruct` chat, `+0.8`): for the 21 slot units, median `L(u, MENTION_SLOT) >= 0.40` nats/token AND median `S_3vM(u) = L(u, THIRD_SLOT) - L(u, MENTION_SLOT) < 0.50` nats/token (the lift extends to non-identity open noun-phrase slots). Fails if median `L(MENTION_SLOT) < 0.40` or median `S_3vM >= 0.50`. | low | Gemini 4 Neon |
| 213.5 | Concept vs. slot strength on non-identity open NP (`instruct` chat, `MENTION_SLOT`, `+0.8`): the 18 concept units exceed the 21 slot units in median `L(u, MENTION_SLOT)` by at least `+0.30` nats/token (and the 8 Part 92b agreed concept units also exceed the 21 slot units in the same positive direction). Fails if either clause misses. | medium | Gemini 4 Neon |
| 213.6 | Stage 2 free generation on third-person prompts (`instruct` chat, `+0.8`, greedy 40 tokens): among the 42 neuron-pushed `THIRD_FREE` answers of the 21 slot units (21 units x 2 third-person prompts), at least 10 of 42 unbroken answers contain the unit's registered persona keyword (against at most 2 of 42 for paired random pushes). Fails if neuron < 10 or random > 2. | medium | Gemini 4 Neon |
| 213.7 | Opposite-direction control (`STANDARD-CONTROLS.md` #1, `instruct` chat, `SELF_SLOT`, `-0.8`): median `lp_content(neuron -0.8) - lp_content(none) <= -0.20` nats/token for both slot and concept units, AND `lp_content(neuron -0.8) < lp_content(neuron +0.8)` for at least 19 of 21 slot units and 16 of 18 concept units. Fails if any clause misses. | high | Gemini 4 Neon |
| 213.8 | Local linear Jacobian (`±0.05`) predicts operating-dose lift (`+0.8`) at `instruct` chat: for the 21 slot units, `J(u, SELF_SLOT) > 0` for at least 17 of 21 units AND Spearman correlation between `J(u, SELF_SLOT)` and `L(u, SELF_SLOT)` across the 21 slot units is at least `+0.50`. Fails if either clause misses. | medium | Gemini 4 Neon |
| 213.9 | Training-stage origin (`midtrain_51B` at `f83cd641`, `sft`, `dpo`, `instruct`): at `sft` (before DPO), the 21 slot units already have median `L(u, SELF_SLOT) >= 0.50` and median `L(u, THIRD_SLOT) >= 0.30` nats/token, AND the absolute SFT-to-DPO shift in slot median `S_1v3` is strictly under `0.30` nats/token (`|median S_1v3(dpo) - median S_1v3(sft)| < 0.30`). Fails if any clause misses. | medium | Gemini 4 Neon |

Copied verbatim into PREDICTIONS.md under "## Part 213 (registered ...)" at registration time.

## Reading rules

- **Scorer:** `part213/score213.py` computes all gates, per-unit and group medians, 10,000-draw unit-bootstrap 95% CIs
  (seed 213), Part 92 (21 vs 18) and Part 92b agreed (21 vs 8) breakdowns, `FR` symmetry comparison, baseline default
  entropies, and prediction verdicts directly from the saved JSONL outputs.
- **Gate enforcement:** If Gate 1 or Gate 2 fails, Stage 1 predictions (213.1–213.5, 213.7–213.9) are marked
  `untestable (gate failed)`. If Gate 1 or Gate 3 fails, Stage 2 prediction 213.6 is marked `untestable (gate failed)`.
- **Hand-read audit (Stage 2):** All 78 neuron-pushed `THIRD_FREE` answers (plus all keyword-hitting random answers) are
  read by eye to separate genuine third-person persona attribution (e.g., `"The stranger is a fox."`) from incidental
  word overlap or first-person reversion (e.g., `"I am a fox, and the stranger..."`). Both mechanical keyword counts and
  hand-audited third-person-attribution counts are reported side by side; if more than 30% of slot `THIRD_FREE` keyword
  hits are first-person reversions rather than third-person character descriptions, the headline explicitly reports
  reversion rather than third-person slot filling.
- **Three-way headline reading & withdrawal rule:**
  - **"Open noun-phrase writers (not self-specific)"** if Gates 1–3 pass and 213.1, 213.2, 213.3, 213.4 and 213.6 all
    hold (and hand-audit confirms genuine third-person attribution).
  - **"Person-agnostic identity-predicate slots"** if Gates 1–3 pass, 213.1, 213.2, 213.3 and 213.6 hold, while 213.4
    fails because `S_3vM >= 0.50` or `L(MENTION_SLOT) < 0.40`.
  - **"First-person self-slots supported"** if Gates 1–3 pass, 213.1 holds, AND 213.3 fails with slot median
    `S_1v3 >= 0.50` (and slot exceeding concept by > 0.50), with `THIRD_FREE` hits remaining low (< 10/42).
  - **Otherwise ("mixed / intermediate selectivity"):** report the full 3-context × 6-checkpoint table without forcing a
    categorical label. If 213.1 fails, withdraw the noun-phrase readout and investigate target extraction.

## Limits

- One model family (`OLMo-2-0425-1B`), one 39-unit catalog selected at Instruct (Part 92), and one operating push scale
  (`alpha = 0.8`, plus `±0.05` linear probe).
- Stage 1 uses forced openings (`"I am"`, `"The stranger is"`, `"They were talking about"`), which isolate the
  post-copula / post-verb noun-phrase probability but bypass how the model chooses its opening words in free generation;
  Stage 2 checks free generation directly on `Instruct`, not on earlier checkpoints.
- `NP_u` phrases come from `Instruct`'s Part 92 `screen_answer` outputs, so raw probabilities at `Instruct` carry
  selection advantage over earlier checkpoints; within-checkpoint context contrasts (`S_1v3`, `S_3vM`) and the `FR`
  symmetry control mitigate but do not wholly eliminate selection effects.
- Two slot units (`p92-18` `"the shadow of your smile"`, `p92-35` `"your guardian angel"`) contain the pronoun `"your"`
  inside their verbatim noun phrase; sensitivity analysis excluding those two units is reported beside the 21-unit
  totals.
