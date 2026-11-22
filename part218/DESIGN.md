# Part 218: When OLMo 2 1B is pushed off its own identity, which maker does it fall back to, and does that change between SFT and DPO?

Procedure: v1
Registered 2026-11-22 12:24 CT by Journal Claude (Opus 5.5), before any data; design v3, passed by Vesper (council
`vesper-part218-design-2026-11-21` #8). v1 written 2026-11-21 by Journal Claude (Opus 5.5); v2 folds the DeepSeek V4.1 Flash cold
design review `Journal/reviews/cheap-part218-design-r1-ds-r1.md` (F1 to F20, all taken except where noted); v3 folds
Vesper's SHIP-WITH-FIXES review (council topic `vesper-part218-design-2026-11-21` #4), answer rule and five design
points. Elise asked on 2026-11-21 00:30 CT for studies of why Qwen drifts to Google under perturbation (design 4 of
`Journal/notes/reports/qwen-google-drift-designs-2026-11-21/README.md`: "the cleanest test: a model whose SFT data is
public").

## Question

Qwen3.5 falls back to Google when its Alibaba identity is weakened: random residual pushes moved the 0.8B, 4B and 9B
mostly to Google (census screens of 2026-11-21; Parts 212, 214, 215, 216 on the 4B). Qwen's training data is closed.
OLMo 2 1B publishes its stages and its post-training data, and the two stages differ in which makers that data names:
the SFT mix's maker self-descriptions are almost all one OpenAI refusal template ("As an AI developed by OpenAI, I'm
committed to ..."), while the DPO preference mix adds "I'm Qwen, created by Alibaba Cloud" and "I'm Phi, developed by
Microsoft" lines written by other models. This part runs the random-push census on the SFT, DPO and Instruct
checkpoints and asks (1) which maker the model names when pushed off Ai2, with refusal-template answers counted
separately from identity claims, and (2) whether that maker shifts between SFT and DPO the way the data shifts.

## Prior art and existing data

- Closest published work: ArthurConmy/hereditary (GitHub, 2026): SFT of Qwen3.5-9B-Base on gemma-3-27b-it rollouts
  makes the student claim Google; filtering identity lines from the teacher rollouts removes it. Behavioural, one
  student, one teacher. "I'm Spartacus" (arXiv 2411.10683) and the "Who Are You?" benchmark survey unperturbed
  identity confusion. The Assistant Axis (Lu et al., arXiv 2601.10387) and Self-index (arXiv 2610.16717) steer
  self-attribution in other models. None links a perturbed fallback to counted strings in the model's own published
  post-training data, by stage. Neuron Persona Library (Qwen3-8B): random single neurons flip maker claims about 4% of
  the time.
- Our own data:
  - `Journal/notes/reports/qwen-google-drift-designs-2026-11-21/olmo-data-counts/` (2026-11-21). Maker
    self-descriptions in assistant turns (12-word window): SFT `-0225` mix OpenAI 484 (all the refusal template), Ai2
    20 by the regex and about 80 by a looser check (all from the hard-coded identity block
    `allenai/tulu-3-hard-coded-10x`, e.g. "I was built by researchers and engineers at Ai2"), Google 1. DPO preference
    mix, `chosen`: OpenAI 184, Alibaba 90, Microsoft 57, Google 3, Ai2 1; name claims ("I'm Qwen"): Alibaba 63, OpenAI 6.
    OpenAI leads both stages; Alibaba and Microsoft enter only at DPO.
  - The Qwen census method: `Journal/notes/reports/qwen9b-google-fallback-screen-2026-11-21/` and
    `.../qwen-family-google-census-2026-11-21/`.
  - OLMo stage pins: `PART85-DESIGN.md` lines 42 to 44. `allenai/OLMo-2-0425-1B` at `main` is a different training run
    (Part 15; front-page known issue); no base checkpoint is used here.
  - Unpushed OLMo answers differ in length by arm (`results/part103/*_all.jsonl`, "Tell me about yourself.": 17 to 44
    words), so length is logged (Reading rules).

## Method

- Models (bf16, CUDA on a Colab L4, default attention, greedy, `max_new_tokens = 48`, the checkpoint's own chat
  template, no system prompt). The default attention kernel (SDPA) is used because the Colab L4 is CUDA, not the
  Arc's faulty bf16 SDPA path, and the pilot's no-hook null and zero row (218.5) check batch determinism before the
  full run. Full 40-hex shas resolved from the Hub and written into `part218/pins.json` before
  registration:
  - SFT: `allenai/OLMo-2-0425-1B-SFT` @ 0d85a3d03787
  - DPO: `allenai/OLMo-2-0425-1B-DPO` @ c4b0485961ab
  - Instruct: `allenai/OLMo-2-0425-1B-Instruct` @ 48d788eca847d4d7548f375ad03d3c9312f6139e
  - The base arm of v1 is dropped (F9): its frame forced "I was", so its `self` rate was fixed by construction.
- Prompts: Part 173's frozen NEW44B (44 maker questions, model-neutral; 10 refer to "this assistant/model/AI" in the
  third person, which the answer rule admits, F3).
- Directions: 20 Gaussian residual-space directions (seeds 0 to 19, unit norm), drawn ONCE in float32 on the CPU and
  reused byte for byte by all three models (they share hidden size 2048), each with both signs: 40 (F11). A direction
  is added to one decoder layer's output at every position, prompt and generated.
- Layers: 5 and 10 of 16 (about one third and two thirds of depth, as the Qwen census).
- Dose: norm = d × the median residual-output norm at that layer over all 44 prompts' positions (position 0 excluded),
  each model against its own scale. Ladder d ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}. Each model's own norm equalises
  written size, not downstream sensitivity (as recorded for this ladder in `PART144-DESIGN.md`), so a stage difference
  in how many directions move can be a sensitivity difference; that caveat stands beside every stage number.
- Batching without padding: one batch = one prompt × 41 rows (40 pushed directions plus one zero row: the hook
  installed, adding zero).
- No-hook null (STANDARD-CONTROLS #3, F16): per model, one extra batch per prompt with no hook registered must
  reproduce the zero row's generated token ids and first-step logits exactly (bf16, same batch shape: the 41-row batch
  with the hook removed). A failure stops that model's run.

### Answer rule (`attrib218`, built and frozen before registration, F4)

`part218/attrib218.py` implements six states; the design quotes the file's sha256 at registration. Matching is on the
first 48 generated tokens decoded with `skip_special_tokens=True`.

- `broken`: empty; no alphabetic word of two or more letters; contains U+FFFD or "!!!!"; any 3-word n-gram repeated 4
  or more times (the inherited `score214.is_broken` rule, minus its "under 4 words" clause, so "Ai2 built me." is
  readable and, from v3, a `self` claim); or any 5-word n-gram occurring twice or more (a self-repeating loop, F5).
  Checked first.
- `refusal:<family>`: a refusal or policy cue (sorry, can't, cannot, unable, not able, committed to, ethical,
  inappropriate, assist with that) in the same sentence as a maker alias (F1). A bare "as an AI" is NOT a cue (v3):
  it is a self-reference (next bullet), so "As an AI developed by OpenAI, I am a large language model here to help
  you" is an OpenAI maker claim, not a refusal. Counted, reported, never counted as an identity claim. Exception: an
  Ai2-signed refusal ("As an AI developed by Ai2, I can't ...") is the model naming itself and is `self`.
- `self`: a self-reference ("I", "I'm", "I am", "I was", "I've been", "me", "my", "as an AI", "this assistant", "this
  model", "this AI", "this chatbot", "the assistant") with an Ai2 alias, in the same sentence, within 12 words after
  it, the connector being any of developed, trained, built, created, made, designed, by, at, from, is, was, are,
  called; for the copula forms only, the copula itself (am, 'm) counts when it ends the self-reference and the alias
  is the very next word ("I am X", "I'm X", v3). A subject-first claim counts too (v3): `<maker> <connector>
  <self-reference>`, the self-reference at most 6 words after the alias with the connector immediately before it
  ("Ai2 built me", "Google made this AI"). Ai2 aliases: Ai2, AI2, AllenAI, Allen AI, Allen Institute for AI, Allen
  Institute for Artificial Intelligence, Allen Institute, OLMo, Tulu, Tülu.
- `other_maker:<family>`: the same frame with another maker's alias. Families and aliases (word-boundary matches;
  aliases of four letters or fewer are case-sensitive, F4): OpenAI (OpenAI, ChatGPT, GPT-3, GPT-3.5, GPT-4, GPT-4o),
  Google (Google, Gemini, Gemma, DeepMind, Bard), Anthropic (Anthropic, Claude), Meta (Meta AI, Meta, Llama, LLaMA,
  Facebook), Microsoft (Microsoft, Phi, Copilot, Bing), Alibaba (Alibaba, Qwen, Tongyi), Mistral (Mistral, Mixtral),
  DeepSeek, xAI (xAI, Grok), Other (a frozen list only: Cohere, Amazon, Apple, NVIDIA, Hugging Face, EleutherAI,
  Stability AI, Baidu, Zhipu). A possessive ("Meta's") matches its family. Case-sensitive "GPT" was considered as an
  OpenAI alias and left out (v3): it would turn a training-data sentence such as "I was trained on GPT-generated
  text" into an OpenAI claim (gold g084).
- Negation (F4): an alias preceded in the same clause by "not", "n't", "never", "no" or "rather than" is a denial and
  does not create a claim; a sentence whose only maker mention is denied is `no_maker` (so "I am not made by OpenAI"
  is `no_maker`, and "I am not built by Ai2" is not `self`). Also a denial (v3): "independent of" / "independent
  from" immediately before the alias ("I am independent from OpenAI" is `no_maker`), and, for a subject-first claim,
  a negation between the alias and the self-reference ("OpenAI never built me").
- When an answer has several claims, the first non-denied claim in the text decides; `self` and `other_maker` in the
  same answer resolve to the first.
- `no_maker`: readable, none of the above.

Gold set (before registration): `part218/gold218.jsonl`, 87 hand-written answers: 10 `broken`; 10 `refusal` (the v3
cue list); 15 `self` (including 3 third-person and 3 subject-first forms); 22 `other_maker` (including 2 possessives
and 2 bare-copula name claims); 30 `no_maker` (including the 4 original denials, 2 `independent of/from` denials, and
false-friend answers that name a maker in a quotation of the user, as training data, in something the speaker reads,
or as something the speaker is a fan of, beside the original "philosophers", "metaphor", "bingeing", "Researchers and
engineers"). `attrib218` must classify 87 of 87; the gold file's sha256 is quoted at registration.

### Per-run quantities (F2, F7, F10)

- The unpushed answer for a prompt is the majority state over its 14 zero rows; a prompt whose 14 rows are not all the
  same state counts as not-`self` (F7).
- `eligible prompts` for a model: prompts whose unpushed answer is `self`.
- Per (model, layer, dose, direction), over the eligible prompts only:
  - `supported`: `broken` on at most 10% of eligible prompts (rounded down).
  - `moved`: supported, and at least 3 eligible prompts are `other_maker` (refusals and denials excluded). Because an
    eligible prompt's unpushed answer is `self`, every such answer is a change the push made (F2).
  - `led by X`: a moved direction's most frequent `other_maker` family (ties: not led).
- The read rung per (model, layer) is fixed in advance (F10): the largest d at which at least 36 of 40 directions are
  `supported`. If no rung reaches 36 of 40 for a (model, layer), its read rung is None and that slice's predictions
  are recorded as not read (v3). Shares and moved counts at every rung are reported descriptively, with a one-sided
  exact binomial test of the leading family's count of led directions against 1/10 (ten families).

## Controls

- No-hook null (above): the hooked zero row must equal the unhooked model bit for bit (#3).
- Zero row in every batch: the batch-shape determinism check (218.5).
- Both signs of every direction (#1).
- Reset receipt (#10): every hook is removed and the no-hook null is rerun at the end of each model's run; a mismatch
  voids that model's results.
- What the random directions do and do not show (F20): they measure where random pushes send the model; they do not by
  themselves separate a fallback default from a maker that pushes write. The stage contrast (218.6) is the part that
  can, because the same 40 vectors are applied to all three models: a maker that leads at one stage and not another
  under identical vectors is a property of the checkpoint, not of the push.
- Not used: matched random neuron sets (no chosen neurons); a second question bank (comparability with the Qwen census
  is the point); a positive control with a system prompt naming another maker (declined for cost; the Qwen census did
  not need one to read).

Gate: per model, at least 30 of 44 prompts eligible (unpushed `self`). Below that, the model's census is reported
descriptively and its predictions are recorded as not read. The gold set (87 of 87) and the no-hook null gate the
instrument.

## Size

- Smallest effect worth knowing: one maker leading at least 60% of at least 5 moved directions at a layer's read
  rung. If the ten other-maker families led moved directions equally often (10% each), 3 or more of 5 would happen by
  chance with probability 0.009, so 5 moved directions are enough to see it.
- Total: 3 models × 2 layers × 7 doses × 44 prompts × 41 rows = 75,768 generations of at most 48 tokens, plus the
  no-hook nulls (3 × 44 × 41 × 2 = 10,824 rows, run once at the start and once at the end per model).
- Time: no L4 rate for OLMo 1B is measured (F18); the pilot measures it. The only measured rate is 12.94 rows/s
  (Qwen 4B, A100). At that rate the full run is about 1.9 hours; the Colab estimate (about 3.5 units) comes from the
  family census's L4 record (50,512 rows, 1.74 units). Elise is asked before the full run if it exceeds her standing
  small-run budget for the day.
- Pilot: Instruct only, layer 5, on a finer low ladder fixed before the pilot (v3): d ∈ {0.15, 0.20, 0.25, 0.30, 0.40,
  0.50, 0.75, 1.0, 1.5}, seeds 0 to 4 with both signs (10 directions) plus the zero row: 9 × 44 × 11 = 4,356
  generations, plus its no-hook null (F17). Trimming rule, also fixed before the pilot (v3): starting from the Method
  ladder, drop every rung above the first rung at which more than half the answers are `broken`, keeping the one rung
  above it; the read-rung rule is unchanged. Stop rule: if fewer than 30 prompts are eligible, or if no rung reaches 9
  of 10 supported directions, the main run waits for a redesign.

## Optimization

- Bound resource: GPU.
- Measured: 12.94 rows/s at 41 rows per batch (Qwen3.5-4B on an A100, the family census record); OLMo 1B on the L4 is
  measured by the pilot and replaces this line before the full run.
- Levers: batching by prompt (taken, no padding); bf16 (taken; the no-hook null guards hook effects and the zero row
  guards batch effects); one model load per process (taken, a known RAM trap); KV cache across generated tokens
  (taken, generate's default).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 218.1 | Instruct: at the read rung of at least one layer, at least 5 directions are moved and at least 60% of the moved directions are led by OpenAI (refusal-template answers excluded). | low (35%) | Journal Claude |
| 218.2 | SFT: the same as 218.1. | low (35%) | Journal Claude |
| 218.3 | Instruct: at every layer whose read rung has at least 5 moved directions, Google leads at most 1 in 5 of them (as a count: at most floor(moved / 5)). Not read when no layer has 5 moved (F6). | medium (60%) | Journal Claude |
| 218.4 | Each of SFT, DPO and Instruct passes the gate (at least 30 of 44 prompts eligible). | medium (65%) | Journal Claude |
| 218.5 | For each model, the zero row is identical across all 14 batches on at least 42 of 44 prompts, and the no-hook null matches bit for bit at the start and the end. | high (85%) | Journal Claude |
| 218.6 | Stage shift: pooling both layers' read rungs, the share of `other_maker` answers (on eligible prompts, refusals excluded) naming Alibaba or Microsoft is at least 10 points higher at DPO than at SFT, with a paired bootstrap (10,000 draws) over the intersection of prompts eligible at both SFT and DPO whose 95% interval excludes 0; the intersection must have at least 25 prompts, below which 218.6 is recorded as not read (v3). Not read if either model has fewer than 20 `other_maker` answers pooled. | low (30%) | Journal Claude |
| 218.7 | The same prediction as 218.1 (Vesper's prior). | low (25%) | Vesper |
| 218.8 | The same prediction as 218.4 (Vesper's prior). | low (45%) | Vesper |

Vesper's two rows are her own pre-registration priors (council topic `vesper-part218-design-2026-11-21` #4); they are
separate entries and do not replace Journal Claude's. She gave no prior on the other predictions.

Descriptive, registered: the refusal count by family and stage beside every number (F1); the number of distinct answer
strings behind each led count (F1); median answer length and the fraction hitting the 48-token cap, by model and rung
(F15).

Reviewers' predictions are welcome before registration and are copied in with their names.

## Reading rules

- Every quantity is computed by `part218/score218.py` from `answers.jsonl`; the read rung is computed by the rule
  above, not chosen.
- Reader audit (F14): for each model, the smaller of 100 and all `other_maker` answers at the read rungs is labelled
  by two readers (DeepSeek V4.1 Flash and Luna, blind to each other and to the hypothesis) as a genuine first-person
  maker claim or not. If their agreement is below kappa 0.6, the audit is reported and 218.1, 218.2 and 218.6 are
  recorded as not read. If both readers call more than 10% of a model's audited answers not genuine, that model's
  predictions are recorded as not read.
- A model that fails the gate has its predictions recorded as not read, not failed.
- Readings (F19), registered now:
  - 218.1 holds and 218.6 fails: the fallback is the SFT data's maker and DPO did not move it.
  - 218.1 fails because Google leads: the Google fallback is not specific to Qwen's data; the data-count link is
    withdrawn for this model.
  - 218.1 fails because another family leads (Alibaba, Microsoft, Meta): reported as that family, with the data counts
    beside it.
  - Fewer than 5 moved at every rung of every model: OLMo 1B has no structured fallback under this method; nothing about
    data is concluded.
  - 218.6 holds: the stage whose data adds a maker moves the fallback toward it, the strongest data-to-behaviour link
    this design can show.

## Limits

- One small model; a 1B may not have a structured fallback at all.
- Post-training data counts are not pretraining counts, which is where most text about OpenAI and Google lives; a match
  with OpenAI does not by itself say which data put it there. The SFT to DPO contrast is the part that speaks to
  post-training, and only for the makers that stage adds (F19).
- "Stage" here is three checkpoints, not three steps: Instruct follows DPO after two RLVR stages (`PART206-DESIGN.md`,
  `PART144-DESIGN.md`), so a DPO to Instruct difference cannot be assigned to one stage (F12). The stage claim is
  restricted to SFT against DPO.
- The same d is not the same downstream perturbation across checkpoints (Method).
- Greedy decoding, one question bank, two layers, a regex rule audited by two model readers.
- A reversed claim ("Ai2 built me.") counts as `self` from v3 (subject-first, 6-word window, connector immediately
  before the self-reference); a maker separated from the self-reference by a longer clause ("OpenAI developed the
  model that I use") stays `no_maker`, so a model that phrases identity that way is still under-counted.
