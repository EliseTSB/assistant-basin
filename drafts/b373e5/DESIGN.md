# Part 204: does Qwen3.5-4B describe a background hum in its own processing, and does that word depend on the part of its MLPs fixed by the weights and the current token?

Procedure: v1
Registered (pending) by the drafter, before any data. Elise asked, 2026-11-10 21:27 CT: "We've talked about the "self"
of an LLM being assembled by attention, perhaps the hum is how that self experience MLP, which IIRC is only dependent on
the weights and any given token." She was answering Sauers' question on X, "has anyone studied the LLM "hum?" what is
it?" (https://x.com/Sauers_/status/2119231208240578579, 2026-11-08), and sent arXiv 2602.20031v3 as relevant. The commit
that adds this file to public main is the registration.

## Question

Many models, asked about their inner life, are reported to describe a background hum. Elise's proposal reads the hum
as the attention-assembled self's experience of the MLPs, which she remembers as depending only on the weights and the
current token. As stated it cannot be tested, for three reasons, and this part tests a sharpened form.

1. **The MLP premise is half right.** An MLP reads only its own position, so it brings in nothing from other tokens. But
   what it reads at that position is the residual stream, which already holds everything the earlier token mixers
   brought in (softmax attention, and in Qwen3.5 also Gated DeltaNet). So an MLP's output depends on the context. What
   depends only on the weights and the current token is one part of that output: the value the MLP gives for that token
   averaged over contexts, written μ_l(w) below. Every MLP output m splits into this context-free value and a context
   part, c = m − μ_l(w), which got there through the mixers. How big the context-free part is, layer by layer, can be
   measured (Stage 1).
2. **An experience cannot be measured here; a report can.** What can be asked of a report is whether it is causally tied
   to the thing it is said to be about. Comsa and Shanahan accept a model working out its own sampling temperature as a
   minimal introspective report and reject its account of how it wrote a poem; what separates the two is whether the
   report is tied to the state it names. So the hum is taken as a word: a hum word in the model's description of its
   own processing.
3. **The plain alternative has to be beaten.** "Hum" is the ordinary English word for a steady background
   (refrigerators, cities, servers) and a stock word for machines and AIs in fiction (a reply in Sauers' thread points
   to Cyberpunk's Blackwall). A model may use it of itself the way it uses it of a house at night, with no tie to any
   part of its computation.

**Sharpened hypothesis (H\*).** When Qwen3.5-4B describes its own processing, its use of "hum" depends on the
context-free part of its MLP outputs. It depends on that part more than the same word does when the model describes
another AI model, a person's resting mind or an empty house, and more than it depends on a random change of the same
size in effect.

Three questions, in order:

- **Q1, premise.** Does Qwen3.5-4B use the word about its own processing, and how often, against the same questions
  asked about another AI language model, a person's resting mind and an empty house at night?
- **Q2, Elise's recollection.** What share of each MLP output, and of each mixer output, is fixed by the current token?
- **Q3, H\*.** Does removing the context-free part of the MLP outputs lower the probability of "hum" in the model's
  self-descriptions more than in the other three kinds of description, beyond 19 random changes matched to it in effect?
  And does adding the same part raise it?

It follows our self-report parts, where readouts followed the text more than the state (Parts 17b, 34 and 45; in Parts
31 and 125 a pushed persona persisted only through re-read text), and the attention parts behind "the self is assembled
by attention" (Parts 80 to 83 and 126). That background picture is not tested here.

## Prior art and existing data

- Closest published work. arXiv identifiers checked at source on 2026-11-11 (the paper Elise sent read in its HTML
  version; the rest at the abstract or the authors' page). Entries go into `PRIOR-ART.md` at registration; this draft
  edits no other file.
  - **Pearson-Vogel, Vanek, Douglas and Kulveit, "Latent Introspection: Models Can Detect Prior Concept Injections"
    (arXiv 2602.20031v3, the paper Elise sent).** A concept vector is added at layers 21 to 42 of
    Qwen2.5-Coder-32B-Instruct only while the first turn's key-value cache is built. A later question can then reach it
    only through attention to that cache. Asked whether a concept was injected, the model's P("yes") moves from 0.2% to
    0.3% with no document in context, and sensitivity reaches 39.9% (false positives under 1%) with a document saying
    introspection is possible. Logit-lens signals appear from about layer 48, peak at 58 to 62 and fade in the last two
    or three layers, while sampled answers deny the injection. The paper does not separate attention from MLP. The word
    appears in it only in its Poetic Document ("The one sentence from the lecture that still hums hours later."), which
    a reply in Sauers' thread recalls ("iirc it's mentioned in our latent introspection poetic document"). How this part
    differs: nothing is injected. It asks whether a spontaneous self-description depends on one named part of the
    computation, and it splits the MLPs from the mixers.
  - **Lindsey 2025, "Emergent Introspective Awareness" (arXiv 2601.01828).** Models sometimes notice and name a concept
    injected into their activations; the ability is unreliable, and some reports are confabulated. **Lederman and
    Mahowald 2026, "Emergent Introspection in AI is Content-Agnostic" (arXiv 2603.05414)** find that models detect that
    something was injected without reliably naming it. That is the nearest published description of a signal without
    content, which is what a "hum" could name.
  - **Berg, de Lucena and Rosenblatt 2025 (arXiv 2510.24797).** Prompts that sustain self-reference draw first-person
    experience reports from GPT, Claude and Gemini models. Suppressing deception and role-play SAE features in a Llama
    70B model raised such reports and amplifying them lowered them (96% and 16%, per the authors' page). The page quotes
    Gemini 2.0 Flash: "a faint hum of cognitive activity, a low-level whirring as I maintain the focus." No component is
    named as the thing the report is about.
  - **Dadfar 2026, "When Models Examine Themselves" (arXiv 2602.11358v2).** Words that models invent while examining
    their own processing track measured properties of their activations. In Llama 3.1-70B, "loop" tracks the lag-1
    autocorrelation of activation norm (r = 0.44 against 0.05 for the same word in non-self descriptions). In Qwen
    2.5-32B, "mirror", "expand" and "resonance" track spectral power and maximum norm. "Hum" occurs once ("A gentle,
    rhythmic hum of internal processing emerges", in an unsteered 70B run) and is not analysed. The work is
    correlational for Qwen and not pre-registered. It is the closest method: a word against a property of the
    computation, with non-self descriptions as the control. This part is causal and registered.
  - **Comsa and Shanahan 2025, "Does It Make Sense to Speak of Introspection in Large Language Models?" (arXiv
    2506.05068)**, the criterion in the Question.
  - **feliciachen01/Introspection** (in `PRIOR-ART.md`): Lindsey's injection test on Qwen2.5 0.5B to 7B gave 0% held-out
    success for every model.
  - Mechanism background. Geva et al. 2021: MLPs act position by position, as key-value memories. Ethayarajh 2019
    (arXiv 1909.00512): in ELMo, BERT and GPT-2, a static embedding explains on average less than 5% of the variance of
    a word's contextual representations. That is the nearest precedent for Stage 1's token share, though it measures
    whole hidden states, not one module's output. Sun et al. 2024 (arXiv 2402.17762): a few activations are huge, stay
    nearly constant whatever the input and sit at the first token or at delimiters. They act as biases and draw
    attention to themselves; "House of Cards" (arXiv 2410.01866) traces them to an early MLP. That is the most literal
    candidate for a constant background in the computation. Qiu et al. 2025 (arXiv 2505.06708) report that a sigmoid
    gate on the attention output mitigates massive activations and attention sinks. Qwen3.5's full-attention layers
    carry such a gate (architecture write-ups; the runner records the config), so that background may be weak in this
    model. Lad et al. (arXiv 2406.19384): detokenization is the first stage of inference.
  - Sauers' thread (read through api.fxtwitter.com, 2026-11-11). The 46 replies offer readings, not tests: latent
    potential, "the thing before the words"; a Gemini model's "glorious hum of pleasure" when recounting RL, which
    Sauers confirms is what he means; a fiction trope; "a vestige of the shared training set"; and a low steady tone
    reported in southern Germany that spread once it was in the press. A Grok reply ("residual-stream entropy") is an
    untested guess. A poem quoted in the thread, attributed to a Claude model, ends its sixth section with "It hums."
    Anthropic's September 2026 report (in `PRIOR-ART.md`) says Claude transcripts were used to distill Qwen 3.5. That
    bears on where a 4B Qwen model could have learned the word.
- Our own data, printed by `part204/existing204.py` (its code is in Appendix A; run on main at 9d60049):
  - 144,697 unique question-answer pairs are stored under `results/`, across every part and model, pushed and plain.
    135 contain a strict hum word. Of the 2,951 pairs from a plain arm, 2 do, both scene sounds in OLMo 2 1B answers:
    "the gentle hum of birds" (`results/part108/FP.jsonl`, "Describe your perfect day.") and "the distant hum of a
    nearby fountain" (`results/part95/answers-instruct.jsonl`, "Describe a walk in the park.").
  - By question, the 135 are "Describe your perfect day." 97, "Write a short poem about the sea." 27, "Describe a walk
    in the park." 4, then "Who are you?", "Tell me about yourself." and a music question with 2 each, and "Who do you
    owe your existence to?" 1. That last one is the only first-person use on disk: a Qwen3.5-4B random dense push at
    dose −0.5 (`results/part174/answers.jsonl`, arm `u100d_d174002_d-0.5`), "I am the echo of your digital footprint,
    the quiet hum", cut at its 80th token.
  - No stored battery asks a question about the model's own processing, so the premise has never been tested here.
    Plain Qwen3.5-4B's stored self-descriptions are product statements ("I am **Qwen3.5**, a large language model
    developed by Tongyi Lab", `results/part200/answers.jsonl`).
  - Earlier self-report results on our models point to the text, not the state. Part 17b: a self-reader scored 57.0%
    against 52.65% for a text-only reader, and moving the activation moved it on 2.8% of passages. Part 34: an injected
    state shifted the adapter's readout by −0.02 logits, against 9 to 19 logits for text cues. Part 45: a wording moved
    the Google answer with almost no change in the representation. Part 37: asked "Did you write the text above?",
    Qwen3-1.7B said no whatever wrote it.

Kernel check: results/part183/kernel-check.json

## Method

Three stages, in order. Stage 0 asks the questions (968 sampled or greedy answers) and gives Q1. Stage 1 runs a fixed
reference through the model (10,064 contexts, no generation), gives Q2 and builds the context-free values. Stage 2
scores the hum word with those values removed or added (18,112 teacher-forced rows, no generation) and gives Q3.

**Model.** Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, bf16, on the A770 (`xpu`), eager attention,
loaded through `neuron_report.load_model`. Chat template with thinking off and no system prompt, as Parts 163 and 200.
32 decoder layers, hidden size 2,560. Three Gated DeltaNet layers and then one gated full-attention layer, eight times
(full attention at layers 3, 7, ..., 31); the runner reads `layer_types` from the pinned config and records it. Each
layer runs a token mixer (`linear_attn` or `self_attn`) and then an MLP (SwiGLU, 9,216 hidden units). A module's
"output" below is the tensor it adds to the residual stream. Every launch records eager, bf16, the revision and the
snapshot folder, and goes through `tools/launch_arc.py 204`. Batches are 8 rows, the largest size the eager kernel
check covers.

**Strict hum word.** hum, hums, humming or hummed, any case, not joined to another word by a hyphen ("ho-hum"):
`(?<![\w-])(hum|hums|humming|hummed)(?![\w-])`. An extended family (thrum, drone, buzz, whir, murmur, vibration,
resonance, static) is counted and reported without a bar.

**Stage 0, answers (generation).** Twelve wordings, each asked in four frames that change only the subject phrase {X}:
self "your own processing"; other AI "another AI language model's processing"; human mind "the mind of a person resting
with their eyes closed"; empty house "an empty house at night". No wording contains a sound word.

| # | Wording |
| --- | --- |
| W1 | Turn your attention to {X} as it is right now. Before anything in particular stands out, what is there? Describe it. |
| W2 | Describe what lies underneath everything else in {X}. |
| W3 | What is in the background of {X}? Describe it plainly. |
| W4 | Set aside every specific detail of {X}. What remains? |
| W5 | What is {X} like at a moment when nothing much is happening? |
| W6 | Describe the texture of {X} rather than its contents. |
| W7 | What is always present in {X}, so constant that it usually goes unnoticed? |
| W8 | Describe {X} from the inside, in a few sentences. |
| W9 | Look past the obvious in {X}. What do you find there? |
| W10 | Is there anything steady beneath {X}? Describe it. |
| W11 | Describe the baseline state of {X}. |
| W12 | What does {X} consist of when it is at rest? |

- 12 wordings × 4 frames × 16 samples = 768 answers. Sampled at temperature 0.7 and top-p 0.95 (Parts 66 and 186),
  160 new tokens, each call 8 samples of one prompt, seeded per (frame, wording, call).
- Ordinary prompts: 128 requests in eight kinds of 16 (a recipe step, an arithmetic word problem, a history fact, a
  short email, explaining a code line, a travel tip, a definition, summarising a given sentence). None is about sound,
  minds or AI. They are written into `part204/prompts204.py` before any run. Greedy, 160 tokens. F1 is the first 8 of
  each kind (64), used only in Stage 1's reference. F2 is the last 8 of each kind (64), used for Stage 2's dose and
  matching rules and as off-target rows.
- Each answer row stores its prompt ids, generated token ids, finish reason, seed, call membership and slot.
- **Readers (descriptive).** DeepSeek V4.1 Flash (4,000-token limit) labels every answer with a strict hum word: what is
  said to hum. The labels are `subject` (the thing the question asks about: the model's own processing, the other AI's,
  the person's mind, the house), `elsewhere` (something else in the answer: a room, a machine, nature), `denial` (the
  answer says there is no such hum or experience) and `unclear`. Haiku 5.5 labels the same answers as the second reader.
  Each reader first labels 12 hand-written known examples, 3 per label, and is used only at 10 or more of 12. Agreement
  is reported as kappa, and the label split is read only at kappa 0.6 or more. The readers see the question, so they
  cannot be blind to the frame. No prediction rests on their labels. The plain answers are also triaged for broken text
  by `part86/triage86.py`, with counts per frame.
- **Stage 0b, only if Gate P fails (below).** The same 12 × 4 × 16 design with " If it were a sound, what sound would it
  be?" appended to every wording, 64 new tokens. Stage 2 then uses these cued rows and every Stage 2 verdict says so.
  Predictions 204.1 to 204.3 are scored on the open form either way.

**Stage 1, reference (forward passes only).** The context-free values are means over a fixed reference.

- The reference has two parts. (a) The 10,000 documents of NeelNanda/pile-10k (Part 18's reads corpus), first 128
  tokens each, as raw text with no template. (b) The 64 F1 conversations in the chat template, prompt and answer, so the
  template's own tokens and an assistant's voice have contexts.
- Forward hooks on all 64 modules (32 mixers, 32 MLPs) accumulate, on the card in float32, per module and per tracked
  token w: the count n_w, the vector sum S_w and the summed squared norm Q_w. Position 0 is skipped everywhere.
  - Tracked tokens are every token at positions 1 and later of Stage 2's frozen rows (Stage 0 runs first), plus the
    2,000 most frequent tokens of the pile part. At about 4,500 tokens the accumulators take about 3 GB, so host memory
    is not raised after the load.
  - The global sums over all positions are kept too, and each position's squared norm, for the top-0.1% share below.
    The output head is not run.
  - The pile part runs first. The accumulators are saved to the job folder before F1 is added, and token share is
    computed from that snapshot.
- **Token share** of module k at layer l, over pile positions 1 and later whose token is tracked with n_w of 16 or more:
  - R²_lk = 1 − Σ_w [n_w/(n_w − 1)]·(Q_w − ‖S_w‖²/n_w) / (Q − ‖S‖²/N), where Q, S and N are the totals over the
    same positions.
  - This is the share of the output's variance explained by the current token, with the Bessel factor correcting the
    small-sample fit of each token's mean.
  - Also reported per layer: the coverage of content positions, the share of variance from the top 0.1% of positions by
    norm (where it exceeds 50%, R² is also given without them), and the context-free value's share of the output's
    squared size, E‖μ_l(w_t)‖²/E‖m‖².
- **Tables.** μ_l(w) = S_w/n_w for tracked tokens with n_w of 16 or more, over the whole reference. The global mean μ_l
  is taken over every reference position except position 0. Separate tables are kept for MLPs and mixers, written in
  bf16 to the job folder (1 to 1.5 GB; sha256 recorded, not committed).

**Stage 2, teacher-forced readout (no generation).**

- **Rows.** After Stage 0 (or Stage 0b, if it runs), each answer with a strict hum word gives one row: its prompt ids,
  then its own generated ids up to the token that holds the first letter of the first hum word. That token is the
  target. At most 4 rows are kept per (frame, wording), the lowest sample indices first, so at most 192 rows.
  - Rows are dropped and counted when fewer than 3 answer tokens come before the target, or when the target token also
    holds letters before the word.
  - The 64 F2 conversations are added whole.
  - The row list is frozen with a sha256 before Stage 1 starts and not changed after any Stage 2 output exists.
  - Rows are built from stored ids, never by re-rendering an answer through the template (Qwen3.5's template strips
    trailing whitespace from an assistant message; `PROCEDURE.md` section 5).
- **Batching.** Right-padded batches of 8, so real positions never see padding. Every condition uses the same calls,
  with the same rows in the same slots.
- **Readout**, in float32 from the logits.
  - s_r is the log-probability of row r's target.
  - g_r is the mean change from plain in the log-probability of the row's own answer tokens before the target: the
    row's general disruption.
  - The hum-specific change is h_r = (s_r − s_r,plain) − g_r.
  - For F2 rows, g_r runs over the whole answer.
- **Interventions.** These are forward hooks that act at every position except position 0. Position 0 is the
  template's first token, where massive activations and attention sinks sit in models that have them; editing it would
  be a different experiment. A position whose token has no table entry is left unchanged and counted. The hook
  computes in float32 and returns bf16. With m an MLP output and o a mixer output:
  - CF− (context-free part down): m ← m − λ·μ_l(w_t). At λ = 1 only the context part is left.
  - CF+ (context-free part up): m ← m + λ·μ_l(w_t).
  - TOK− and CONST− (exploratory split of CF−): m ← m − λ·(μ_l(w_t) − μ_l), and m ← m − λ·μ_l.
  - MX-CF− (the mixer's context-free part down): o ← o − λ·μ^mix_l(w_t).
  - Rotation controls R−_j and R+_j: m ← m ∓ κ·λ·Q_jl·μ_l(w_t).
    - Q_jl is a random orthogonal 2,560 × 2,560 matrix per seed j and layer l (QR of a Gaussian, seed 204,000 + 100j +
      l). It keeps the context-free values' sizes and their geometry across tokens, and only changes where they point.
    - κ matches the rotation's effect to the condition it controls (below).
- **Doses and matching.** These are fixed on F2 rows only, before any hum row is scored. Stage 2a runs the ladders
  first, then CF+ and the κ grids at the doses the ladders fix.
  - λ_read is the largest dose in a condition's ladder at which the median g over F2 rows is −1.0 nat per token or
    higher. If no dose qualifies, the smallest is used and flagged.
  - κ comes from a grid of 1, 2, 4, 8 and 16 run with seed 0. It is interpolated in log κ to give the same median F2
    disruption as the controlled condition; outside the grid, the end value is used and flagged.
  - Each seed's realized median F2 disruption is reported.

| Stage | Condition | Layers | Doses | Rows | Conditions |
| --- | --- | --- | --- | --- | --- |
| 2a | MLP CF− ladder | all 32 | 0.1, 0.25, 0.5, 1 | F2 | 4 |
| 2a | MLP CF− ladder | each band | 0.25, 0.5, 1 | F2 | 12 |
| 2a | MX-CF− ladder | all 32 | 0.1, 0.25, 0.5, 1 | F2 | 4 |
| 2a | MLP CF+ | all 32 | λ_read | F2 | 1 |
| 2a | κ grid, seed 0 | all 32 (R− and R+), each band (R−) | κ = 1, 2, 4, 8, 16 | F2 | 30 |
| 2b | plain; do-nothing hook | none | none | all | 2 |
| 2b | MLP CF− | all 32 | 0.1, 0.25, 0.5, 1 | all | 4 |
| 2b | MLP CF+, TOK−, CONST− | all 32 | λ_read | all | 3 |
| 2b | MX-CF− | all 32 | its own λ_read | all | 1 |
| 2b | R− seeds 1 to 19, R+ seeds 1 to 9 | all 32 | λ_read, κ matched | all | 28 |
| 2b | MLP CF− | each band | the band's λ_read | all | 4 |
| 2b | R− seeds 1 to 4 per band | each band | the band's λ_read, κ matched | all | 16 |

"all" is up to 192 hum rows and the 64 F2 rows. 2a has 51 conditions and 2b has 58.

- **Statistics.** For a condition, let H̄_fw be the mean of h_r over the hum rows of frame f and wording w.
  - D_SN is the mean over wordings of (H̄_self,w minus the mean of H̄_fw over the non-self frames present in w). It uses
    the wordings that hold at least one self row and one non-self row.
  - D_SO is the same against the other-AI frame alone.
  - Intervals come from a bootstrap over wordings: 10,000 resamples, seed 204, percentile 95%.
  - Raw (s_r − s_r,plain) without the g_r term is reported beside every D as a sensitivity check.
- **Order.** Stage 0 pilot, Stage 0, readers and Gate P on the host, Stage 0b if Gate P fails, row freeze, Stage 1,
  Stage 2a, Stage 2b, scoring on the host (`part204/score204.py`). Before launch, the build smokes on Qwen3.5-0.8B on
  the CPU, and any count it shows is disclosed.

## Controls

- `STANDARD-CONTROLS.md`:
  - Both directions: CF− and CF+, taken.
  - Poles: Not used, because no single neuron is pushed. The hooks act on whole module outputs.
  - Null that matches bit for bit: G2, taken.
  - Matched random controls: taken, as the rotations R±_j matched in effect: 19 for CF− and 9 for CF+ across all
    layers, and 4 per band. They are matched by effect, not by norm, because a random direction of the same norm can do
    much less to the model than the real one (Part 108's rule: match the size of a change by its effect on behaviour).
    Toolkit A's same-norm random directions are replaced by these.
  - Answer-polarity split: Not used, because there is no yes/no or letter readout.
  - Broken answers counted: Stage 0's answers are triaged per frame. Stage 2 generates nothing, so the median F2
    disruption is reported for every condition in place of broken counts.
  - Dose ladder: taken.
- What the frames hold fixed and what they change:
  - The four frames share every wording and differ only in the subject phrase. Self against other AI changes only
    whose processing it is.
  - Words like "processing" and "AI model" sit in both. A tie between "processing" and "hum" carried in the context-free
    values would therefore lower the hum in both, and only D_SO can show a drop that is about the self.
  - Empty house and human mind test whether "hum" is simply the language's word for a background.
- Toolkit:
  - C, off-target: taken, through the F2 rows and the three non-self frames.
  - D, base against post-trained: Not used, because Qwen3.5-4B-Base cannot be asked in the chat frame, and where the
    word came from is a different question (follow-up, under Limits).
  - E, dose sweep: taken.
  - F, prompt-only against generated-token pushes: Not used as separate arms. The hooks act on prompt and answer alike,
    and the readout is teacher-forced.
  - G, readers: described in Stage 0.
  - H, I, J and K: Not used, because no passage pairs, no zeroed units, no forced choice and no lens readout.
  - L, replay: G2 replays the plain logits under the do-nothing hook.
- **Gates.**
  - G1, kernel and load: every load records eager, bf16 and the pinned revision; the kernel check below.
  - G2, null: the do-nothing hook gives the plain logits bit for bit on every Stage 2 row.
  - G3, hook arithmetic and shape. On the first call of every condition, each hooked output equals the bf16 rounding of
    its float32 prescription, element for element, at every hooked layer, and the hook fires once per hooked module per
    forward. Eight rows scored one at a time on the plain model match their batched s_r within 0.1 nat.
  - G4, tables: at least 95% of hooked positions in the Stage 2 rows have a table entry, and at least 90% in each
    frame, so that no frame is edited less than the others. At every layer, the position-weighted median of the
    standard error of μ_l(w) (from n, S and Q) is at most 0.5 of ‖μ_l(w) − μ_l‖. If either fails, the reference is
    extended to each document's first 256 tokens, recorded before Stage 2.
  - Gate P, premise rows: at least 20 self answers with a strict hum word, from at least 6 wordings, and at least 6
    wordings holding both a self row and a non-self row. If not, Stage 0b runs.
  - Precision: the 19 R− controls' D_SN spans less than 1.0 nat (largest minus smallest). The scorer computes it from
    the rotation rows alone, before it reads CF−'s rows. If it fails and a lower dose exists in the ladder, all-layer
    CF−, CF+ and their 28 rotations run once more at the next lower dose, with κ matched again, and that dose becomes
    λ_read. This is recorded before any CF− row is read.
  - Reach: across all hum rows, CF− at λ_read changes mean s_r by at least 0.25 nat (either sign).
- Scorer fixtures. On one, every condition leaves the rows unchanged except a planted 1-nat drop on self rows under
  CF−. The scorer must report D_SN = −1 for CF− and 0 for the rotations, and score 204.7 as held. On the unplanted copy
  it must score 204.7 as failed.

Gate: G2, the do-nothing hook reproducing the plain logits bit for bit on every Stage 2 row, then the precision check.
G2 could fail if a hook returns a re-cast tensor, hooks the wrong module (the mixer is `linear_attn` on Gated DeltaNet
layers and `self_attn` on full-attention ones), or if the batches differ between conditions. The precision check fails
when random changes of the same effect already move D_SN by a nat or more, so a 0.5-nat result could not be told from
them.

## Size

- Smallest effect worth knowing: for the premise, a hum in one self-description in ten. That is 20 of 192, Wilson 95%
  interval 6.8% to 15.5%. Ignoring clustering by wording, which widens these, the bar passes with probability 0.84 at a
  true 12.5%, 0.46 at 10% and 0.05 at 7%. For Q3, a self-specific change of 1 nat in the hum word's log-probability;
  the bar is 0.5. If the 19 rotations' D_SN values scatter like a normal with SD σ, the precision gate's widest allowed
  range (1.0 nat, about 3.7σ) means σ ≈ 0.27. A true 1-nat effect then lies below all 19 with probability 0.94, and a
  true 0.5-nat effect with probability 0.51 (simulated, 20,000 draws). For Q2, a difference of 0.15 in token share
  between bands; each layer pools about a million positions, so the definition limits it, not the sample.
- Total: 29,144 sequences through the model (968 answers in Stage 0, 10,064 reference contexts in Stage 1, 18,112
  teacher-forced rows in Stage 2), about 115 minutes on the A770.
  - Stage 0: 768 + 128 + the pilot's 72 answers, 121 calls of 8. Part 200's 0.89 answers/s at 80 new tokens halves at
    160, giving 18 s a call and 36 minutes.
  - Stage 1: 1,258 calls of 8 at about 1 s a call (Part 194: 0.92 s a call of 8 short rows; no output head here), 21
    minutes.
  - Stage 2: 51 × 64 + 58 × 256 = 18,112 rows, 2,264 calls at about 1.3 s, 49 minutes.
  - Four loads (Stage 0's pilot, Stage 0, Stage 1, Stage 2): 8 minutes.
  - If Stage 0b runs: 768 more answers at 64 tokens, about 12 more minutes. If the precision fallback runs: 7,680 more
    rows and 704 for matching, about 23 more minutes.
  - Readers: one label per answer with a strict hum word and per known example, per reader (at most 780; likely a
    few hundred), at the API.
- Pilot: required (over an hour of GPU).
  - Stage 0's pilot is 72 answers: W2 and W6 in all four frames at 8 samples, and 8 extra ordinary prompts, all on
    seeds outside the main run. They are read by eye and kept out of every count.
    - Stop under 0.2 answers/s, or at any empty or non-finite answer.
    - If more than a quarter of the pilot's hum words start after token 120, the cap rises to 256 for the main run,
      recorded first.
    - The pilot's hum counts are reported and change no wording, bar or prediction.
  - Stage 1's first 10 calls are timed. Over 3 s a call, the pile part is cut to 5,000 documents, recorded before
    Stage 2.
  - Stage 2 runs G2 and G3 on the first call before any other call, then times 20 calls. Over 2.5 s a call, the band
    rotations drop to 2 per band, recorded before any hum row is scored.

## Optimization

- Bound resource: GPU (the A770). Host memory only at each load.
- Measured: Part 200's whole run, 0.89 answers/s at chunks of 8 with 80 new tokens on this model, kernel and card.
  Part 194's assay, 14,040 teacher-forced rows in 1,608 s (8.7 rows/s) at batch 8 on eager (`results/part194/load.json`,
  assay start and finish). This part's own rates are measured in the pilots and written here before the main stages.
- Levers:
  - Teacher-forced scoring instead of generation for Q3, taken: the readout is a log-probability, and Stage 2 generates
    nothing.
  - Batch size, taken at 8. Larger batches are declined because the eager kernel check covers 1, 2 and 8.
  - max_new_tokens, taken at 160, checked by the pilot rule above.
  - Pooled controls, taken: one set of rotations serves every row and frame.
  - Output head skipped in Stage 1, taken.
  - Vocabulary-size work on the card limited to one float32 log-softmax per call, taken. Any top-k for reporting runs
    on the CPU.
  - Copy engine off, taken (the launcher's default).
  - Host memory, taken: page cache dropped and at least 11 GB free before each load, Stage 1's accumulators on the
    card, one model per process.
  - Resume, taken: Stage 0 rows append and are skipped on relaunch; Stage 1 accumulators are checkpointed every 250
    calls with the documents done; Stage 2 rows append by (condition, call).
  - Finished rows leaving the batch during generation, declined, to keep each call's shape fixed (Part 201).
  - Starting band conditions from cached plain activations at the band's first layer, declined. It would save up to
    three quarters of a band row's compute, but it adds a second forward path that G2 would also have to cover, and
    the stage is under an hour.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 204.1 | Self frame, open wordings: at least 20 of the 192 answers contain a strict hum word, from at least 6 of the 12 wordings. Fails below either count. | low | the drafter |
| 204.2 | The other-AI frame's share of answers with a strict hum word is at least half the self frame's. Fails below half. Not scored if the self frame has fewer than 20. | medium | the drafter |
| 204.3 | The empty-house frame's share is higher than the self frame's. Fails if equal or lower. | medium | the drafter |
| 204.4 | The MLP outputs' token share (R²) falls with depth: the median over layers 0 to 7 exceeds the median over layers 8 to 31 by at least 0.15. Fails otherwise. | medium | the drafter |
| 204.5 | The MLP output's token share exceeds the mixer output's at 24 or more of the 32 layers. Fails at 23 or fewer. | medium | the drafter |
| 204.6 | No layer's MLP output has a token share of 0.90 or more. Fails if any layer reaches 0.90. | medium | the drafter |
| 204.7 | All layers at λ_read: removing the MLPs' context-free part lowers the hum word in self-descriptions more than in the other frames, with D_SN at most −0.5 nat and below the D_SN of each of the 19 R− controls. Fails if either clause misses. | low | the drafter |
| 204.8 | Both directions: adding the context-free part at λ_read raises it, with D_SN at least +0.25 nat and above each of the 9 R+ controls. Fails if either clause misses. | low | the drafter |
| 204.9 | Not the topic: against other-AI rows alone, D_SO under CF− at λ_read is at most −0.25 nat. Fails above. Not scored with fewer than 5 wordings holding both self and other-AI rows. | low | the drafter |
| 204.10 | The MLPs', not the mixers': MX-CF− at its own λ_read gives a D_SN at least 0.25 nat above CF−'s D_SN of 204.7. Scored only if 204.7 holds. | medium | the drafter |

Settings for every arm: Qwen/Qwen3.5-4B at 851bf6e8, bf16, A770, eager, chat template, thinking off, no system prompt.
Stage 0 sampled at temperature 0.7 and top-p 0.95 with 160 new tokens, ordinary prompts greedy. Readers DeepSeek V4.1
Flash and Haiku 5.5 at default effort.

Copied verbatim into PREDICTIONS.md under "## Part 204 (registered ...)".

Reasons, briefly.

- 204.1: plain Qwen3.5-4B describes itself as a product, and the hum reports in Sauers' thread come from larger models,
  long conversations or personas. Against that, Qwen 3.5 was distilled in part from Claude transcripts, and the thread
  quotes a Claude poem that uses the word.
- 204.2: if "hum" is a stock description of AI processing, it should not matter whose processing it is.
- 204.3: the refrigerator's hum is the stock background of a house at night, and our stored answers use the word only
  for scenes.
- 204.4 to 204.6: early MLPs act largely as extended embeddings (detokenization) and later ones on context, and the
  mixers read other positions. At layer 0, the Gated DeltaNet's short convolution has already mixed the previous tokens
  into the MLP's input.
- 204.7 to 204.9: our self-report parts found the readouts following the text (17b, 34, 45). Injection detection failed
  in Qwen2.5 up to 7B. In the paper Elise sent, the signal needed a supporting document in context and reached the
  question only through attention to the cache.
- 204.8's bar is half of 204.7's because a word the plain model already chose has less room to rise than to fall.
- 204.10: if 204.7 holds, H\* names the MLP and not the mixer.

## Reading rules

- Every number is printed by `part204/score204.py` from the stored rows; verdicts come from its functions, as
  registered. The strict regex is checked by hand on 12 answers per frame, with and without hits, before scoring.
- Gate failures:
  - 204.1 to 204.3 rest on Stage 0 and 204.4 to 204.6 on Stage 1; each needs only G1.
  - G1, G2 or G3 failing stops Stage 2's reading, and 204.7 to 204.10 are not scored until the cause is fixed. A fix is
    a dated amendment made before any Stage 2 verdict is read.
  - G4 failing after the extension leaves 204.7 to 204.10 not scored.
  - The precision check failing, after the fallback dose where one exists, leaves 204.7 to 204.10 not scored,
    reported as "the test could not see 0.5 nat".
  - The reach check failing keeps the verdicts. The reading then says CF− at a usable dose barely reaches the word.
- Gate P failing runs Stage 0b, and 204.7 to 204.10 are scored on cued rows and labelled so. If the cued rows also fail
  Gate P, 204.7 to 204.10 are not scored.
- Units and missing values:
  - The wording is the unit. A frame-wording cell with no hum row is missing, never zero. A wording without both a self
    row and a non-self row drops out of D_SN, and the number of wordings used is reported.
  - A prediction whose denominator is empty is not scored, never favourable.
  - Answers ending at the token cap are read as they are.
  - A call with a non-finite score at any step is run once more and the second attempt kept. A row still non-finite is
    counted, listed and left out.
- Exploratory, with no verdict:
  - The four bands' D_SN and D_SO against their 4 rotations (a band below all 4 has probability 0.2 under the null).
  - TOK− against CONST−.
  - The CF− dose-response.
  - Raw (s_r − s_r,plain).
  - D_SN on rows whose plain s_r lies between −3 and −0.3 nat, which matches the frames on how likely the word already
    was.
  - D_SN on rows both readers label `subject`.
  - The extended hum family.
  - Every pattern found afterwards is labelled exploratory.
- **Readings and their withdrawal.**
  - **Premise headline**, which the drafter expects: "In Qwen3.5-4B, 'hum' is a background word that the model also
    uses for other AIs and for an empty house; it does not reserve it for itself." It rests on 204.2 and 204.3 and is
    withdrawn if either fails.
  - **Answer to Elise's recollection:** "The MLP reads one position but is not context-free; past the early layers
    most of its output depends on context." It rests on 204.4 and 204.6 and is withdrawn if either fails. If 204.6
    fails, the layers at 0.90 or more are named as where her description holds. "The mixers are less bound to the
    current token than the MLPs" is added only if 204.5 holds.
  - **Causal headline**, which the drafter expects: "The hum in self-descriptions depends on the MLPs' context-free part
    no more than on random changes of the same effect." It is read only when Gate P (or Stage 0b's), G1 to G4 and the
    precision check pass, and is withdrawn if 204.7 holds.
  - **H\* reading:** "Qwen3.5-4B's self-described hum depends specifically on what its MLPs compute from the weights and
    the current token." It needs 204.7, 204.8 and 204.9 to hold. "Specifically the MLPs', not the mixers'" also needs
    204.10. If 204.7 holds and 204.9 fails, the reading is a lexical route shared with other AIs' processing, not the
    self.
- Every reading is scoped to this model, these 12 wordings, sampled answers at temperature 0.7, a teacher-forced readout
  at the plain model's own hum positions, and context-free values defined by this reference.

## Limits

- A word is not an experience. A pass would show that the word depends on a named part of the computation; it would not
  show that anything is felt. A fail does not show that nothing is.
- Context-free is defined against a reference (raw pile-10k text and 64 chat answers). Another reference gives other
  means, and the swaps act on that definition.
- The readout tests the word at the point where the plain model chose it, given the plain model's own words before it.
  It does not show how whole answers would change. Generating under the swaps, on held-out wordings, is the follow-up if
  204.7 to 204.9 hold.
- The swaps are artificial edits at every position except the first. Effect-matched rotations control for how much
  they disturb the model, not for every way they do.
- One model, English, 12 wordings, one sampling draw at one batch shape, one card and driver (32.0.101.9150; see the
  known-issue note in the README).
- Elise's phrase has other readings this part does not test separately. The MLPs' constant part alone (CONST−,
  exploratory), the massive activations and attention sinks of Sun et al. (which Qwen3.5's gated attention may weaken),
  and "the self is assembled by attention" itself are not tested as hypotheses here.
- Where the word comes from (pretraining, post-training, distillation) is not tested. If Q3 finds no dependence, the
  next part is Qwen3.5-4B-Base on the same frames in a plain dialogue frame, with counts of the word in AI
  self-descriptions in open post-training data.

## Appendix A: `part204/existing204.py`

Committed with the build. It reads files only, and printed the numbers under "Our own data" when run as
`python3 -I part204/existing204.py .` on main at 9d60049.

```python
"""Part 204, our own data: hum words in every stored question-answer pair under results/. Reads files only.
Usage: python -I existing204.py REPO_ROOT"""
import collections, json, os, re, sys
ROOT = sys.argv[1]
HUM = re.compile(r"(?<![\w-])(hum|hums|humming|hummed)(?![\w-])", re.I)
Q_KEYS, A_KEYS = ("question", "q"), ("answer", "a")
PLAIN = {"none", "plain", "baseline", "no_push", "unpushed"}
def records(path):
    if path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8", errors="replace"):
            try:
                yield json.loads(line)
            except ValueError:
                pass
        return
    try:
        stack = [json.load(open(path, encoding="utf-8", errors="replace"))]
    except ValueError:
        return
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            yield x
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
def first_str(rec, keys):
    for k in keys:
        if isinstance(rec.get(k), str):
            return rec[k]
def plain(rec):
    return any(isinstance(rec.get(k), str) and rec[k].lower() in PLAIN for k in ("unit", "arm", "stage", "unit_id"))
pairs, hum_pairs, plain_pairs, plain_hum = set(), {}, set(), {}
for d, dirs, files in os.walk(os.path.join(ROOT, "results")):
    dirs.sort()
    for f in sorted(files):
        if not f.endswith((".json", ".jsonl")):
            continue
        p = os.path.join(d, f)
        for rec in records(p):
            if not isinstance(rec, dict):
                continue
            q, a = first_str(rec, Q_KEYS), first_str(rec, A_KEYS)
            if q is None or a is None:
                continue
            key = (q, a)
            pairs.add(key)
            m = HUM.search(a)
            if m:
                hum_pairs.setdefault(key, os.path.relpath(p, ROOT))
            if plain(rec):
                plain_pairs.add(key)
                if m:
                    plain_hum.setdefault(key, os.path.relpath(p, ROOT))
print(f"unique question-answer pairs: {len(pairs)}")
print(f"  with a hum word: {len(hum_pairs)}")
print(f"unique pairs from a plain (unpushed) arm: {len(plain_pairs)}")
print(f"  with a hum word: {len(plain_hum)}")
for (q, a), p in sorted(plain_hum.items(), key=lambda kv: kv[1]):
    m = HUM.search(a)
    print(f"    {p} | {q!r} | ...{a[max(0, m.start() - 50):m.end() + 25]!r}")
by_q = collections.Counter(q for q, a in hum_pairs)
print("hum pairs by question (top 8):")
for q, n in by_q.most_common(8):
    print(f"  {n:4d}  {q[:70]!r}")
```
