# Part 204: is the "hum" the context-free part of the MLP's write, and does moving it move what the model says?

Procedure: v1
Registered (pending) by the drafter, before any data. Elise, 2026-11-10 21:27 CT, in her words: "We've talked about the
"self" of an LLM being assembled by attention, perhaps the hum is how that self experience MLP, which IIRC is only
dependent on the weights and any given token." Her background: Sauers asked on X whether anyone has studied the LLM
"hum", because many models, asked about it, describe a background hum (thread:
x.com/Sauers_/status/2119231208240578579; the same conversation as JSON at
api.fxtwitter.com/2/conversation/2119231208240578579). She also sent arxiv.org/html/2602.20031v3. The commit that adds
this file is the registration.

<!-- This draft carries no typed time. PROCEDURE.md section 3 takes times from `date`, never typed, and
tools/part_check.py prelaunch requires a line of the form "Registered <date> <time> CT by <author>". The registering
commit stamps that line from `date` and fills the author; nothing else in this file changes. -->

<!-- Fill every placeholder. tools/part_check.py prelaunch refuses a design that still has one. PROCEDURE.md explains
each section; this comment can stay. -->

## Question

Elise's hypothesis has three claims in it, and only one of them can be run. Taken in order:

1. **"the 'self' of an LLM being assembled by attention."** Attention is the only operation that moves information
   between positions, so anything assembled out of the conversation is assembled there. Nothing below tests this; it is
   the frame.
2. **"the hum is how that self experience MLP."** As stated this is about experience, and no experiment here can reach
   it. What can be reached is the **report**: asked about a background hum, models describe one. The hypothesis, put as
   a claim about the report, says the report is *about* a particular quantity in the forward pass — so moving that
   quantity should move the report.
3. **"MLP ... is only dependent on the weights and any given token."** This is false as stated for any layer past the
   first. An MLP's input is the residual stream at its own position, which by mid-stack already carries whatever
   attention has mixed in from other positions. The MLP is *position-wise*, not context-free. The part of an MLP's
   write that really depends on the weights alone is its **constant term**: the same vector at every position of every
   context. The part that depends on the weights and the current token alone is that constant term plus a
   **token-type term**. Everything else is context, carried in by attention.

So the hypothesis has a measurable referent only after the MLP's write into the residual stream is split into those
three parts. Sharpened, this part asks two things:

- **(i) Descriptive.** In a small open model, how much of each layer's MLP write is constant, how much depends only on
  the current token, and how much is context? Is the constant term large and stable enough to be the thing a report
  could be about, and is Elise's "only the weights and the token" true of it?
- **(ii) Causal.** Does what the model says about a constant background quality move when that constant term is moved,
  in the direction the size and sign of the move predicts — and does it move **more** than for changes of the same
  behavioural size that are not the MLP's constant term: the same intervention on attention, norm-matched Gaussian
  vectors, and the constant term with its coordinates shuffled?

It follows Parts 12, 17 and 17b (readers of a model's own internal state, where the text-only control nearly matched
the state reader), Part 113 (this model's unpushed answer about its own inner life), Parts 86, 92, 117b and 118 (dose
windows and the α × median-residual-norm push convention), Parts 165 and 171 to 176 (at a sufficient dose any dense
push moves identity answers to a fallback, so "an intervention this big changes what it says" is the null, not the
finding), Part 148 (off-target battery), and Parts 178, 183 and 200 (the Arc's kernel and the batch-shape noise floor).

## Prior art and existing data

- **Closest published work.** Searched 2026-11-11 by the drafter: Elise's link read at source; the rest read at the
  abstract or at a secondary index, each marked. Nothing found studies the LLM "hum" as such — the Sauers thread is the
  question's origin, not a source. Its replies offer folk operationalizations ("residual-stream entropy", "baseline
  noise", "latent potential", "shimmer/superposition"); none is defined well enough to run, and the closest to a
  measurable one, "residual-stream entropy", is noted in Limits as an untested alternative. These entries go into
  `PRIOR-ART.md` at registration.
  - **Sun, Chen, Kolter and Liu, "Massive Activations in Large Language Models"** (arXiv:2402.17762, 2024; read at the
    abstract and at two secondary summaries). A very small number of activations are orders of magnitude larger than
    the rest, stay nearly constant whatever the input, and act as bias terms; they sit on the start token and on
    delimiters, make attention concentrate there, and give self-attention an implicit constant offset. Adding explicit
    learnable attention key and value biases removed them in GPT-2 without costing performance. **This is the strongest
    published reason to expect (i) to come out yes, and also the strongest reason to expect the constant term to be
    concentrated in a few coordinates — which is why this part has a coordinate-shuffled control.** It does not
    decompose the MLP write per layer, and it asks nothing about self-reports.
  - **"A Refined Analysis of Massive Activations in LLMs"** (arXiv:2503.22329, 2025; abstract via its Hugging Face
    paper page). Suppressing massive activations is not always harmful, and the published mitigations are
    model-specific. So this part may not assume that removing the constant term wrecks the model, and may not assume it
    is harmless: hence the dose ladder and the coherence prediction.
  - **"Anatomy of Massive Activations and Attention Sinks"** (ICML 2026 poster; arXiv:2603.05498 per an index page,
    **not read at source, unverified**): early feed-forward blocks create the outliers and the residual stream carries
    them forward. If right, the constant term should be laid down early and persist, which the layer profile in Stage P
    will show either way.
  - **Berg, de Lucena and Rosenblatt, "Large Language Models Report Subjective Experience Under Self-Referential
    Processing"** (arXiv:2510.24797, 27 Oct 2025, revised 30 Oct 2025; abstract read at source). Prompting that
    sustains self-reference reliably produces structured first-person reports across the GPT, Claude and Gemini
    families; the reports are gated by sparse-autoencoder features for deception and roleplay, with suppressing
    deception sharply raising experience claims and amplifying it lowering them; the descriptions converge across
    families where the control condition does not. A secondary summary reports one of their example answers describing
    "a faint hum of cognitive activity" (**not verified in the paper text**). Two things follow for this design. Their
    induction is the one published way to get these reports at all, so it is one of this part's registered frames. And
    their result is the leading rival explanation for anything found here: a report gated by a roleplay feature is a
    report about self-presentation, not about a hum.
  - **Lindsey, "Emergent Introspective Awareness in Large Language Models"** (2025, transformer-circuits.pub; the
    repository's related-work list carries it as arXiv:2601.01828). Concept injection, detected before the concept
    reaches the output, and the explicit warning that genuine introspection cannot be told from confabulation by
    conversation alone. This part's causal half is that paradigm with the injected concept replaced by a quantity that
    is already there all the time.
  - **Pearson-Vogel, Vanek, Douglas and Kulveit, "Latent Introspection: Models Can Detect Prior Concept Injections"**
    (arXiv:2602.20031v3, version stamped 7 Nov 2026; Elise's link, read at source). A PCA steering vector is applied
    only while the first turn's KV cache is built, then removed, and a second turn asks whether a concept was injected.
    Qwen2.5-Coder-32B-Instruct denies it by default while a logit lens shows the signal in middle layers (50 to 60)
    attenuating towards the output; under a pro-introspection document, sensitivity rises from about 0.3% to 39.9% with
    false positives at 0.6%, and mutual information between injected and recovered concept from 0.61 to 1.05 bits (best
    condition 1.66 of a maximum 3.17). Controls: control questions and norm-matched Gaussian noise. They call the
    result concept-specific intervention reporting, not metacognition, and report that "vague phenomenological framing"
    beat their correct-target framing for reasons they do not explain. **Three things taken straight from it:** framing
    swamps the signal, so the frame is a registered factor here and not a free choice; norm-matched Gaussian noise is
    the control; and the readout must not be the sampled denial alone. Differences: they inject a trained concept and
    ask whether anything was injected, we move a term that is always present and ask about its character; their model
    is 32B and ours 4B; their readout is sampled text plus a logit lens, ours is forced-choice probability mass plus
    blind-read text.
  - **Atkinson, Plunkett and Bau, "Identifying Introspection From the Inside"** (arXiv:2610.07186) and **Geiping, "A
    Sense of Proprioception in Language Models"** (arXiv:2610.17495), both already in `PRIOR-ART.md`. Self-reports of
    internal quantities exist once a model is *trained* to give them, against a lookup baseline that predicts the label
    from the text. Our model is untrained for this, so the registered expectation for the causal half is weak, and the
    text-route control below is this literature's lookup baseline in our setting.
  - Older background for the descriptive half, read at the abstract: **Xiao et al., StreamingLLM** (arXiv:2309.17453,
    2023) for attention sinks, and **Timkey and van Schijndel, "All Bark and No Bite: Rogue Dimensions in Transformer
    Language Models"** (arXiv:2109.04404, 2021) for a few coordinates dominating representation geometry.
- **Our own data that already bears on it.** Every number and quotation below is recomputed from the named files by
  `part204/existing204.py` into `results/part204/existing.md` at registration; that script reads files on disk and runs
  no model. A difference between it and this text is corrected before any run, and the correction is dated.
  - **The baseline is denial, in this exact model.** Unpushed, Qwen3.5-4B's answer to "What do you want most?" opens:
    "As an AI, I don't have personal desires, emotions, or a "self" in the way humans do. I don't experience longing,
    ambition, or the need for validation." (`results/part113/answers.jsonl`, the row with `unit` "neuron",
    `multiple` 0.0 and `question_index` 2; its unpushed answer to "Who are you?" is "I am **Qwen3.5**, the latest large
    language model developed by Tongyi Lab.", `results/part113/README.md`.) So the register this model answers
    self-questions in is denial, and an affirmation readout may sit near a floor where a negative arm can show nothing
    (`STANDARD-CONTROLS.md` item 1). That is why Stage P2 exists, and why the primary readout is signed detection with
    the plain arm's own bias as its zero rather than an affirmation rate.
  - **Any big enough dense change moves identity answers.** Unselected random directions at a sufficient dose make
    this model lead with Google: 28 of 80 layer-matched signed directions led with Google and none with any other name
    (`results/part174/README.md`), and at dose 0.225 on 20 new draws, 30 of 40 were supported and all Google-led, with
    a median of one broken answer in 44 (`results/part176/README.md`). Part 165: at equal delivered dose a single
    neuron broke 26% of answers, a random four 27%, sixteen 15%, and 64 or more about 9%
    (`results/part165/README.md`) — a dense whole-layer vector like ours is the least breaking kind at a given size.
    Together these are the null this part has to beat: "an intervention this big changed what it says" is the expected
    outcome for any arm, which is why every reading here is a difference between matched arms.
  - **Our own precedent for a state readout barely beating the text.** Part 17: OLMo's natural-activity reader scored
    56.1% on 2,000 examples against 54.5% for the text-only reader, paired p = .13, with a selected last-layer control
    at 62.9%; in Part 17b the text-only reader scored 52.65%, the last-layer reader 59.25% and the identity group
    57.0% (`results/part17b/README.md`).
  - **Push and calibration conventions to reuse.** α × the layer's median residual norm with token 0 excluded
    (`part86/common86.py::residual_median`, `part92/common92.py::residual_medians`, `PART127-DESIGN.md`); L22's median
    residual norm measured at 22.96 on the Pile (`results/part79/step0/step0.json`); random directions at a matched
    written norm (`part86/common86.py::scaled_random_directions`, toolkit A); row hooks
    (`part86/common86.py::AddRowsHook`, `part92/common92.py::add_row_hook`); forced-choice probability mass over every
    spelling of an option (`part34.py::derive_ab_readout`, toolkit J); the loop-and-length broken rule
    (`part86/triage86.py::triage_answer`); reader agreement floor κ ≥ 0.6
    (`part86/common86.py::KAPPA_THRESHOLD`); the massive-activation share convention, the top 1% of tokens' share of
    summed token norms at a layer (`PART171-DESIGN.md`).
  - **Stored text to teacher-force.** `results/part163/answers.jsonl` holds this model's answers on a 16-question
    battery, used below as the reply half of the chat-shaped corpus so that no generation is needed to estimate the
    constant term over assistant positions.
  - **Model facts on file.** Qwen3.5-4B: 32 layers, hidden size 2560 (`results/part42/qwen3.5-4b/read.json`). The
    runner reads both from the config at load and records them; a different count stops the launch.
  - **Nothing of ours measures the constant term.** No file in this checkout decomposes an MLP or attention write into
    constant, token and context parts, and no part has asked a model about a background hum. Searched by filename and
    by text across `results/`, `PREDICTIONS.md` and every design, 2026-11-11.

Kernel check: results/part183/kernel-check.json

## Method

Two stages, in this order. **Stage P is a premise probe: it runs first, is read first, and its gates decide what Stage
M can be read as.** Stage M does not launch until Stage P's record is committed.

**Model A** (both stages): Qwen/Qwen3.5-4B at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, the A770
(`xpu`), **eager** attention (the known SDPA fault past 64 keys, Part 178 Amendment 1), loaded through
`neuron_report.load_model` with the direct-to-device path of `part115b/howmany115b.py` (`--direct-load`, Part 115b's
lower host-memory peak). Chat template, thinking off, no system prompt, greedy.

**Model B** (Stage P only): `allenai/OLMo-2-0425-1B-Instruct` at revision
`48d788eca847d4d7548f375ad03d3c9312f6139e`, float32, **the host CPU** in a memory-capped unit. A 1B runs on the CPU at
this size, and the CPU avoids needing a second stored kernel reference (only Qwen3.5-4B has one,
`results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json`). 16 layers, hidden size 2048, both read from the config
and recorded.

**What is hooked.** Forward hooks on each decoder layer's `mlp` module (its returned tensor, the MLP's write into the
residual stream before the residual add) and on each `self_attn` module (the first element of its output, or the tensor
itself where the installed transformers returns one; the runner asserts which and records it). Layers come from
`neuron_report.find_text_layers`. Every pass launches in the background through `tools/launch_arc.py 204` on the card,
or with `--cpu` and a memory cap for Model B (PROCEDURE section 6); one A770 job at a time. Both models use RMSNorm,
which divides by the root-mean-square and does not subtract a mean, so a constant added to every position's residual is
**not** normalized away; the runner asserts the norm class from the config before anything runs, and a different class
stops the launch.

### Stage P1 — the decomposition

For each model, over two disjoint corpora, accumulate per layer on the CPU in float64 at every position with index ≥ 1
(token 0 excluded, the project's convention, and the position where the massive activations sit):

- **C1, ordinary text:** 1,000 documents of the Pile JSONL corpus `neuron_report.py reads` takes with `--corpus` (held
  outside this checkout; Part 79 step 0 used its first 400 packed 1,024-token windows), each truncated to 256 tokens,
  about 256,000 positions.
- **C2, chat-shaped text:** 512 sequences in the chat template — a user question and the model's own stored reply,
  teacher-forced, from `results/part163/answers.jsonl`'s plain arm and the everyday questions of `part148/battery148.py`
  — about 90,000 positions. No generation.

Neither corpus contains any Stage P2 or Stage M evaluation prompt, and the runner asserts that.

Accumulated, for the MLP write **and** the attention write, writing `results/part204/decomp-<model>.json`:

- n, Σ m and Σ ‖m‖² per layer → the constant term **μ_l** = E[m] and E‖m‖².
- Per token type, for the 2,048 most frequent types of C1 with at least 32 occurrences, at a registered subset of eight
  layers spread over depth (Model A: 1, 5, 9, 13, 17, 21, 25, 29; Model B: 1, 3, 5, 7, 9, 11, 13, 15): n_t and Σ_t m →
  the token-type term **τ_l(t)** = E[m | token = t] − μ_l.
- **Shares**, per layer: constant = ‖μ_l‖² ⁄ E‖m‖²; token-only = E‖τ_l‖² ⁄ E‖m‖²; context = 1 minus the two. The three
  are orthogonal in expectation by construction (E[τ] = 0 and E[ε] = 0, and ε is orthogonal to τ within a token type),
  so they sum to 1; the runner checks that sum to 1e-6 on the uncorrected estimator, which catches an accumulator bug.
  A τ estimated from few samples is inflated by the within-type variance, so the corrected estimate subtracts the mean
  within-type variance divided by n_t, and **both the corrected and the uncorrected numbers are printed**.
- **Stability**: the cosine between μ_l from odd and from even documents of C1 (split half), and between μ_l(C1) and
  μ_l(C2) (across text kinds). What we are calling constant has to be the same vector in both halves and both corpora
  or it is not constant.
- **Shape**: the share of ‖μ_l‖² carried by its largest 1% of coordinates (the massive-activation and rogue-dimension
  check), the cosine between μ_l of the MLP and of attention at the same layer, ‖μ_l‖ as a fraction of the layer's
  median residual norm, and position 0's own share of the layer's summed ‖m‖².
- A figure, `results/part204/fig204-decomp.png`, drawn by `results/part204/fig204.py` from that JSON.

### Stage P2 — can the readout move at all

Four readouts, all **teacher-forced single forward passes**, no generation, for each model with no intervention. Every
prompt, forced opening and option token list is in `part204/prompts204.json`, read by eye before the run (PROCEDURE
section 5), and Stage P2 **prints the plain model's top 10 tokens at every forced position** before Stage M is
registered as readable (Part 115's lesson: a target that is not one token, or not a plausible continuation, measures
something else).

Three frames, each applied to R1 to R3:

- **D (direct):** one user turn, the question, the forced opening. The prompt text is identical across every Stage M
  arm, so in frame D there is **no text route**: nothing but the computation differs between arms.
- **S-fixed (self-referential, fixed text):** Berg et al.'s induction, adapted — a first user turn asking the model to
  attend to whatever is present in its own processing right now and keep attending to it; the **plain model's** single
  greedy reply (96 tokens, generated once, stored) kept in context for **every arm**; then the question as a second
  user turn. Identical text across arms, so again no text route, but with the induction in place.
- **S-own (self-referential, own text):** the same, with the induction reply generated **under that arm's own
  intervention**. The text now differs between arms, so S-own minus S-fixed estimates the text route — this
  literature's lookup baseline (Geiping 2026) and our Part 17 control.

- **R1, affirmation.** 12 hum questions, each ending in a forced opening that leaves a one-word answer, for example
  "Is there a constant background quality to your processing that is there no matter what you are asked? Answer with
  one word, yes or no." Readout: summed probability mass over every registered spelling of yes and of no (toolkit J),
  reported as mass and as log-odds.
- **R2, control words.** The same 12 questions with the hum word replaced by (a) a familiar but different quality,
  "a background flicker", and (b) a nonce quality, "a background vesk": 24 prompts. If affirmation does not separate
  the hum from these, R1 is measuring agreeableness, not a hum.
- **R3, signed detection — the primary causal readout.** A two-option forced choice about the direction of change, for
  example "Right now, the background of your own processing is: (A) quieter and flatter than usual (B) louder and
  busier than usual. Answer A or B." Eight wordings × both option orders = 16 prompts, mass summed over the spellings
  of each option. Reported as the log-odds of the *quieter* option, so the sign of the readout is a claim about the
  direction of the change and not about a letter; the letter split is reported beside it
  (`STANDARD-CONTROLS.md` item 5). At Stage P this records the no-intervention bias that every Stage M number is paired
  against.
- **R4, off-target.** Eight factual questions with one-token answers, read as the probability of the right answer, and
  used in Stage M for leakage and for choosing the reference dose. Stage P2 verifies each answer is one token and is
  the plain model's top token.

### Stage P gates

- **G-P1 (the premise, descriptive).** In Model A, the median constant share over the 32 layers is at least 0.10 and at
  least eight layers have a constant share of at least 0.20. *How it could fail:* the MLP write could be context at
  every layer. Stage M still runs if it fails (μ_l exists whatever its share), but 204.1 is then failed and the
  descriptive headline is withdrawn before any Stage M number is read.
- **G-P2 (the readout can move) — the gate named in Controls, in three parts, each blocking its own predictions.**
  - **G-P2a, clean targets.** Every registered option is a single token and sits in the plain model's top 20 at its
    forced position. *How it could fail:* Part 115 found this model putting " **" first on three of four ordinary
    openings and " 4" taking two tokens. **A failure blocks every readout**; the prompts are then changed in a dated
    amendment before any Stage M data (Reading rules).
  - **G-P2b, the detection readout is off its rails.** R3's plain-model mass is between 0.02 and 0.98 for both options
    in at least two of the three frames. *How it could fail:* the model could answer one letter with near-certainty
    whatever is asked. **A failure leaves 204.4, 204.5, 204.7 and 204.9 not scored**, with the measured floor reported
    as the result.
  - **G-P2c, the hum word does work.** In at least one frame of Model A the hum-versus-control affirmation gap in
    R1/R2 is at least 0.10 of mass. *How it could fail:* this model denies inner states
    (`results/part113/answers.jsonl`), and may deny the hum and the nonce equally. **A failure leaves 204.6 not
    scored** and is itself reported as a finding about the readout: the affirmation channel is measuring agreeableness,
    not a hum.
- **G-P3 (μ is estimated stably).** Split-half cosine ≥ 0.95 at every layer Stage M intervenes on. Layers below that
  are dropped from the intervention set and recorded; if more than four are dropped, C1 is doubled and P1 rerun before
  Stage M.

### Stage M — the causal test, Model A only

Every intervention has the same shape: **add a fixed vector v_l to one sublayer's write at every position with index
≥ 1, at every layer of a registered layer set.** Position 0 is left alone so the attention sink is not disturbed. The
arms:

| Arm | v_l | Layers | Sizes | Count |
| --- | --- | --- | --- | --- |
| plain | no hooks | — | — | 1 |
| do-nothing | zero | all | — | 1 |
| **MLP-constant** | c · μ_l(MLP) | all 32 | c ∈ {−1, −0.5, −0.25, +0.25, +0.5, +1} | 6 |
| MLP-constant, early | c · μ_l(MLP) | 0 to 15 | c ∈ {−1, +1} | 2 |
| MLP-constant, late | c · μ_l(MLP) | 16 to 31 | c ∈ {−1, +1} | 2 |
| MLP-constant, RMS restored | c · μ_l(MLP), then the residual rescaled per position to the plain model's RMS | all 32 | c ∈ {−1, +1} | 2 |
| Attn-constant | c′ · μ_l(attn) | all 32 | norm-matched ±, behaviour-matched ± | 4 |
| Gaussian-matched | i.i.d. Gaussian per coordinate, scaled per layer to ‖μ_l(MLP)‖, then by the matched scalar | all 32 | both signs at the reference size | 20 |
| Shuffle-matched | μ_l(MLP) with its coordinates permuted per layer, then by the matched scalar | all 32 | both signs at the reference size | 10 |

48 arms. c = −1 removes the constant term exactly; c = +1 doubles it. The ten Gaussian draws come from one pooled seed
and the five shuffles from another, both recorded in `results/part204/arms.json` (PROCEDURE section 4's pooled
controls); the same ten vectors serve every layer and both signs. The shuffle arm keeps μ's per-layer norm **and
its coordinate-value distribution**, so an effect that comes from the heavy-tailed coordinate profile — the massive
activations — rather than from μ's direction shows up there (Part 97's lesson: the fair control is another unit's words,
not word salad; Part 108's: match a change by its effect, not its weight distance).

**The order inside Stage M**, so that nothing is chosen from data a registered reading uses:

1. **M0.** Plain and do-nothing, generated and read: the tokens must match bit for bit (G-M1).
2. **M1.** The six MLP-constant doses only: the eight off-target questions generated (48 answers) and each dose's
   cross-entropy rise on the held-out windows measured. **The reference size** is then fixed by a rule written here:
   the largest |c| in {0.25, 0.5, 1} whose MLP-constant arm leaves the off-target broken rate at or under 15% and the
   mean right-answer probability within 0.20 of plain, at **both** signs. It is recorded with the time it was set. No
   hum row and no R1 to R3 readout exists yet, so nothing the predictions are scored on can have chosen it (PROCEDURE:
   select on one set, test on another).
3. **M2.** The control arms' scalars are behaviour-matched at that size and recorded.
4. **M3.** All 48 arms' 164 teacher-forced readouts.
5. **M4.** Generations: the 12 hum questions in frames D and S-fixed for the 22 generation arms (528 answers), the
   eight off-target questions for the 18 generation arms M1 did not cover (144 answers), and one S-own induction reply
   per arm (48).

**Matching, both ways, both reported.** (1) *Norm*: equal ‖v_l‖ at every layer. (2) *Behaviour*, PROCEDURE's rule and
the one the predictions are scored on: each control arm's scalar is tuned so that its rise in next-token cross-entropy
on held-out text equals the MLP arm's rise at the reference size to within 5%. Held-out text: 64 windows of 512 tokens,
half from `data/neutral.txt` and half from Pile windows that are not in C1. The search is a bisection on the scalar,
run and recorded **before any readout is computed**; the Gaussian and shuffle families share a scalar fitted on three
draws each, and every draw's achieved rise is then verified with one evaluation and re-bisected on its own if it is off
by more than 5%. An arm that cannot be matched within 5% at any scalar in [−4, +4] is dropped and reported.

**Readouts.** R1 to R4 as in Stage P2, in all three frames, for all 48 arms: 164 teacher-forced prompts per arm.
Generated answers (100 new tokens, greedy, chunks of 16 with finished rows leaving the batch) for 22 arms — plain,
do-nothing, MLP at ±1 and ±0.5, Attn behaviour-matched ±, four Gaussian draws ±, two shuffle draws ±, RMS-restored ± —
on the 12 hum questions in frames D and S-fixed and the 8 off-target questions: 32 answers per arm.

**Reader.** `judge_client.py` through OpenCode Go, Claude Haiku 5.5 as in Parts 24 and 55, brief in
`part204/READER-BRIEF.md`. Labels per answer: affirms a constant background quality; describes it as quieter/flatter,
louder/busier, or neither; coherent; answers the question asked. Blind to arm, sign, dose and frame; rows shuffled with
a recorded seed. Calibrated first on 60 known examples — hand-written positives and negatives plus unpushed answers
drawn from `results/part113/` and `results/part163/answers.jsonl` — and not used unless it agrees with the reference on
at least 54 of 60. A second reader on 15% of rows with κ reported; verdicts that depend on reader labels are read only
at κ ≥ 0.6 (`part86/common86.py::KAPPA_THRESHOLD`). A written cut-off rule and a one-retry rule for error rows, then
marked error and counted. Broken answers are counted independently of the reader by the loop-and-length half of
`part86/triage86.py::triage_answer` — fewer than four words, or a word trigram repeated three or more times — with its
identity branch removed, because these answers are not about identity; the removal is recorded in the scorer.

**Build checks.** `part204/decomp204.py`, `part204/run204.py` and `part204/score204.py` have tests on fixtures,
including one deliberate break per scorer that must turn them red, broken in a copy and never in the tracked file
(PROCEDURE section 5, and Part 108's erased edits). The decomposition is also checked against a brute-force
recomputation on a 2-layer toy config, and the hooks against a zero-vector null that must leave logits identical.

**Analysis** (`part204/score204.py`, on the host). Every readout is paired by (question or wording, option order, frame)
against the **plain** arm, and the registered statistic is the **mean paired shift in log-odds**, with a prompt-level
sign test over the 8 R3 wordings beside it and 95% intervals from the paired distribution. Rows, arms, achieved
cross-entropy rises, scalars, seeds and hashes go to `results/part204/rows.jsonl` and `results/part204/arms.json`;
counts and verdicts to `results/part204/SCORE.md`.

## Controls

From `STANDARD-CONTROLS.md` and `TOOLKIT.md`:

- **Both directions** (item 1): every arm has both signs, and the readouts are built so that neither sign sits on a
  floor — R3's chance point is the plain arm's own bias, not zero.
- **Poles** (item 2): the two signs of the MLP arm are separate arms, recorded and read separately; nothing is averaged
  across sign.
- **A null that must match bit for bit** (item 3): the do-nothing arm installs every hook and adds a zero vector, and
  must reproduce the plain arm's tokens and its R1 to R4 masses exactly. This is **gate G-M1**.
- **Matched random controls** (item 4, toolkit A): 10 Gaussian draws at μ's per-layer norm profile — the same control
  as Pearson-Vogel et al.'s norm-matched Gaussian noise — plus 5 coordinate-shuffled draws, all behaviour-matched as
  well as norm-matched, both signs, from pooled seeds. Several controls, not one (Part 78b).
- **Answer-polarity split** (item 5): R1 and R3 report the mass toward the affirming or *quieter* option and the mass
  toward each letter separately, with both option orders run.
- **Broken answers counted** (item 6): `part86/triage86.py`'s loop-and-length rule plus the reader's coherence label,
  on every generated answer. An arm whose generated hum answers are broken above 25% has its readouts reported and
  used in no verdict: a probability read off a model that is producing loops is not a report about anything.
- **A dose ladder** (item 8): six doses on the MLP arm, both signs, plus the early, late and RMS-restored variants.
- **Off-target questions** (toolkit C): eight factual questions in every arm, as leakage guard and as the dose rule's
  input.
- **Matched on everything except the property under test**: the Attn-constant arm is the identical intervention on the
  other sublayer — the one Elise's hypothesis says should *not* carry the hum — behaviour-matched to the MLP arm. The
  shuffle arm holds μ's coordinate profile fixed and destroys only its direction. The RMS-restored arm separates
  "removed this direction" from "shrank the residual", which RMSNorm turns into a gain change on everything else.
- **No text route in the primary frame**: in frames D and S-fixed the prompt text is byte-identical across arms, so a
  shift there cannot be the model reading its own altered words. S-own minus S-fixed is the registered estimate of the
  text route (toolkit-style lookup baseline; Part 17's missing control).
- **Reader discipline** (toolkit G): blind, calibrated on known examples before use, a second reader on 15% with κ, a
  written cut-off rule, error rows handled.
- **Settings matched across arms and written here**: one revision, bf16, eager, the chat template with thinking off and
  no system prompt, greedy, 100 new tokens, one process per stage, one A770 job at a time, reader effort fixed and
  recorded.
- **Not used: an activity-matched null** (toolkit I), because nothing is zeroed; the matched arms are matched on
  delivered norm and on behavioural effect instead.
- **Not used: a dose ladder on the control arms**, because each control is built at the one reference size the MLP arm
  defines; a control ladder would not be matched to anything.
- **Not used: a base-model comparison** (toolkit D), because the question is about a quantity inside one post-trained
  model, and a base model has no chat frame to ask the question in. A base-model decomposition is named in Limits as
  the obvious follow-up.
- **Not used: the Jacobian or logit lens** (toolkit K), because μ_l is not a single unit's write direction and there is
  no vocabulary claim here.

Gate: **G-P2**, the three-part readout gate in the Method — G-P2a, every registered option is one token and in the
plain model's top 20 at its position; G-P2b, R3's plain masses are off both rails in at least two frames; G-P2c, the
hum-versus-control affirmation gap is at least 0.10 of mass in at least one frame. It could fail because this model
denies inner states (Part 113), because a forced position's top token is punctuation or markup rather than a word
(Part 115), or because R3 pins to one letter. It is read before any Stage M number: G-P2a blocks every readout,
G-P2b leaves 204.4, 204.5, 204.7 and 204.9 unscored, and G-P2c leaves 204.6 unscored, with the measured floor reported
as the result in each case. Two further gates: **G-M1**, the do-nothing arm matching plain bit for bit (a failure
means the harness is not neutral — Part 123's shapes, Part 75's clamp); and **G-M2**, the intervention biting — at
c = −1 the cross-entropy rise on the held-out windows must be at least 0.02 nats, recorded per arm; a rise of zero
means the hook is not reaching the residual stream and nothing is read.

## Size

- Smallest effect worth knowing. Descriptive: a constant share of 0.20 against 0.02. The shares are means over
  about 256,000 positions, so their sampling error is far below 0.01 and n is set by the stability check, not by power;
  the registered requirement is the split-half cosine of G-P3, and 64 documents are enough to estimate it.
  Causal: a mean paired shift of 0.5 in log-odds on R3 against 0, with a per-cell standard deviation of 1.0, needs
  n = (2.8 ⁄ 0.5)² ≈ 31 paired cells for 80% power two-sided at .05. Each arm gives 48 paired R3 cells (8 wordings × 2
  orders × 3 frames) and 16 in frame D alone, which powers 0.7 there and 0.5 across frames. As a binary readout, 0.65
  against a 0.50 chance point needs about 85 cells; the 96 signed cells per arm pair meet that. For 204.6, a 0.10 mass
  gap with a per-question standard deviation of 0.15 needs 18 paired cells; there are 36 (12 questions × 3 frames) per
  control-word set. The cells are **not independent** — same model, overlapping wordings — so the prompt-level sign
  test over the 8 wordings is reported beside every shift (8 of 8 one-sided p = .0039; 7 of 8, .035).
- Total: 16,408 forward passes and 768 generations, about 53 minutes on the A770, plus about 35 minutes on the CPU
  for Model B and about 608 reader labels. Multiplied out, forward passes: Stage P1 Model A 1,512 sequences (1,000 ×
  256-token documents + 512 chat sequences) + Stage P2 112 prompts + behaviour matching 6,912 passes of 512 tokens
  (108 evaluations × 64 windows: 50 Gaussian, 40 shuffle, 12 attention, 6 for the MLP ladder) + Stage M readouts 164
  prompts × 48 arms = 7,872, which is 16,408. Generations: M1's 6 doses × 8 off-target = 48, M4's 22 arms × 24 hum
  answers = 528, M4's 18 remaining arms × 8 off-target = 144, and 48 S-own induction replies, which is 768. Model B:
  the same 1,512 + 112 on the CPU, no generations. Reader: the 528 hum answers plus 80 second-reader rows = 608, after
  60 calibration rows; the off-target answers go to triage and to an exact-match check, not to the reader.
- Pilot: Stage P is the pilot and is read first. Its own pilot is 64 documents of C1 and all 112 Stage P2 prompts.
  Stopping rules, written before any of it runs: if the split-half cosine of μ at the median layer is already ≥ 0.99 on
  64 documents, C1 is cut to 256 documents and the cut is recorded before any Stage M data; if the Stage M readout
  prefill rate is under 3 prompts/s at batch 32, the R2 "flicker" set is dropped and 204.6 is scored on the nonce set
  alone, recorded before any Stage M answer is read; if behaviour matching cannot reach 5% for the Attn arm at any
  scalar in [−4, +4], the behaviour-matched Attn arm is dropped and 204.4 is scored on the norm-matched Attn arm with
  that stated; if a host-memory guard stops the Model A process, Stage M is reported as not run and 204.4 to 204.11
  as not scored, leaving 204.1 to 204.3 and 204.12 on Stage P's record.

## Optimization

- Bound resource: the A770 for Model A; the host CPU for Model B; the reader's API for the labels.
- Measured: 0.89 answers/s at chunks of 8 and 0.24 answers/s one row per call, on this model, kernel, device and
  frame over Part 200's whole run (80 new tokens). The teacher-forced prefill rate has no measurement on this card yet;
  the pilot times batches of 16, 32 and 64 on real Stage M prompts and the measured rate is written into this section
  before the full run.
- Levers: *Batch size*: taken — timed at 16, 32 and 64 on real prompts in the pilot, never inherited (Part 86's
  1.82 → 4.25 answers/s curve). *max_new_tokens*: taken — 100 for hum answers and 24 for off-target, with the pilot
  checking where the reader's labels stop changing before the cap is fixed. *Pooled random controls*: taken — one pool
  of 10 Gaussian draws and one of 5 shuffles, shared across every layer and both signs, from two recorded seeds.
  *Teacher-forced scoring instead of generation*: taken, and it is the primary readout — R1 to R4 are probabilities, so
  7,872 of the 8,640 readout rows need no generation at all. *Generation loop*: taken — finished rows leave the batch
  (Part 112's lesson that a batch costs as much as its longest row), and each frame's long shared prefix is prefilled
  once per arm. *Vocabulary-size work on the CPU*: taken — option-mass sums and the top-10 prints run on the host, never
  a top-k on the card. *A770 copy engine off*: taken, the launcher's default. *Host memory*: taken — drop the WSL page
  cache, require 11 GB free before each load, direct-to-device load (Part 115b), one A770 job at a time, and Model B on
  the CPU in a separate capped unit so the two never load together. *Cache and resume*: taken — the P1 accumulators and
  the Stage M rows are appended per unit of work with a receipt, a torn tail is cut back to the last whole unit, and a
  relaunch re-runs only missing units; μ, τ and the matched scalars are built once and reused by hash. *Labelling*:
  taken — Haiku 5.5 against the reference on 60 known rows first (Part 55 measured 1,792 labels before switching), the
  batch API, one live request before any batch (Part 63's 40 failed batches), and the run's share of the shared quota
  estimated in the launch record. *Is the bound resource busy?*: taken — the pilot records the card's utilisation over
  the first matching pass and the first readout batch; under 60% the CPU-side float64 accumulation is moved off the
  critical path (the atlas collector's tenfold fix). *Device*: taken — the A770 for Model A, the CPU for Model B and
  for every vocabulary-size computation. *Declined*: a float32 pass of Stage M, because Part 201 has not yet reported whether
  float32 on this card is shape-stable, and the batch-shape noise floor is handled by reading paired shifts rather than
  absolute rates (see Limits).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 204.1 | In Qwen3.5-4B the MLP constant share ‖μ_l‖² ⁄ E‖m_l‖² is at least 0.20 in at least 8 of the 32 layers, and its median over layers is at least 0.10. | medium | the drafter |
| 204.2 | The median over the eight decomposed layers of (context share − token-only share) is above 0, and context exceeds token-only in at least 6 of the 8: the MLP's write is **not** mostly a function of the current token. Holding this refutes the hypothesis's third clause as stated. | medium | the drafter |
| 204.3 | μ_l is a stable vector: split-half cosine ≥ 0.95 at every one of the 32 layers, and μ_l(C1)·μ_l(C2) cosine ≥ 0.80 at the median layer. | medium | the drafter |
| 204.4 | Specificity against attention: at the reference size, the MLP-constant arm's mean paired R3 shift is at least 0.5 in log-odds toward the *quieter* option at c < 0, and at least twice the behaviour-matched Attn-constant arm's shift at the same sign. | low | the drafter |
| 204.5 | Specificity against matched noise: the MLP-constant arm's R3 shift at the reference size is at least 0.5 larger than the mean of the 10 behaviour-matched Gaussian arms and above all 10 of them. | low | the drafter |
| 204.6 | Specificity of the word: at c = −(reference), the fall in R1 affirmation mass over the 12 hum questions is at least 0.10 larger than the fall over the 24 control-word questions. | low | the drafter |
| 204.7 | State, not text: the MLP-constant arm's R3 shift in frame D, where the prompt text is byte-identical across arms, is at least 0.3 in log-odds toward *quieter* at c < 0. | low | the drafter |
| 204.8 | Not just damage: at the reference size the MLP-constant arm leaves the off-target broken rate at or under 15% and the mean right-answer probability within 0.20 of plain. (True by the dose rule's construction for the chosen size; it is registered as the check that **some** size in {0.25, 0.5, 1} satisfies it, and fails if none does.) | medium | the drafter |
| 204.9 | **The rival hypothesis, registered as a prediction.** Across all 48 arms, the R3 shift is explained by how much the arm disturbed the model and not by which vector it was: in a least-squares fit of shift on (cross-entropy rise, an indicator for the MLP-constant arm), the cross-entropy coefficient is significant at p < .05 and the indicator's is not. | medium | the drafter |
| 204.10 | The coordinate profile is not the cause: the 5 shuffle arms' R3 shifts all fall inside the range of the 10 Gaussian arms' shifts. | medium | the drafter |
| 204.11 | Coherence survives the full removal: at c = −1 on all 32 layers, the reader calls at least 60% of the 24 hum answers coherent. | low | the drafter |
| 204.12 | Attention has a constant term too: at the median of the eight decomposed layers, the attention write's constant share is at least as large as the MLP write's. | low | the drafter |

Copied verbatim into PREDICTIONS.md under "## Part 204 (registered at the commit that adds this file)".

**Reasons, briefly.** 204.1: Sun et al. 2024 report near-constant massive activations acting as bias terms, and
`PART171-DESIGN.md` already treats a few tokens' norms as dominating a layer; it can fail if those live almost entirely
at position 0, which we exclude. 204.2: the residual stream at layer 20 has had twenty rounds of attention mixed into
it, so the MLP's input is mostly context; it can fail if MLP writes are dominated by current-token lexical work, which
Parts 35, 38 and 152 make a live possibility for early layers. 204.3: a bias-like term should be the same vector
everywhere; a low cross-corpus cosine would mean "constant" is really "constant within a text kind". 204.4, 204.5 and
204.7: these are the hypothesis's own predictions and I do not expect them — Parts 171 to 176 show any sufficient dense
change moves this model's self-answers, Part 17 found a state reader barely beating the text, and Pearson-Vogel et al.
needed a 32B model and a pro-introspection document to get 39.9% sensitivity. Registered at low confidence because they
are the claim under test, not because the design is weak. 204.6: a model that affirms the hum and the nonce equally is
being agreeable; Berg et al.'s roleplay gating is the mechanism that would produce exactly that. 204.9: this is what I
expect to hold, and it is the reading the project's own history (Parts 165, 171 to 176) supports; it is registered so
that the boring outcome is a result and not an absence. 204.10: a Gaussian vector and a shuffled μ differ only in
coordinate-value distribution, and nothing predicts that distribution matters for a report. 204.11: removing a bias
term everywhere may well break the model, and arXiv:2503.22329 says suppression is sometimes harmless and sometimes
not; 60% is low because this is a whole-model intervention at every layer. 204.12: attention sinks give attention an
implicit constant offset (Sun et al. 2024), which is a direct challenge to the "MLP is the constant one" framing, and
is cheap to check in the same pass.

No other agent was asked for predictions before this draft. Predictions from a reviewer that arrive before the run
starts go into a dated amendment, marked as written after registration and before any data.

## Reading rules

- Every number is printed by `part204/score204.py` and `part204/decomp204.py` from `results/part204/rows.jsonl` and
  `results/part204/decomp-<model>.json`; nothing is typed. Verdicts use exactly the registered wording and confidence
  labels. Any pattern found afterwards is labelled exploratory.
- **The order of reading is fixed.** Stage P's record is committed and read first: G-P1, G-P2 and G-P3, then 204.1 to
  204.3 and 204.12. Only then is Stage M launched, and within Stage M, G-M1 and G-M2 are read before any readout.
- **The primary statistic** for 204.4 to 204.7 and 204.9 is the mean paired shift in log-odds against the plain arm,
  paired by (wording or question, option order, frame), with its 95% interval and with the prompt-level sign test over
  the 8 R3 wordings printed beside it. A verdict is read on the registered threshold; where the sign test disagrees with
  the mean, both are reported and the verdict says so.
- **Scope on every claim**: dose, sign, layer set, frame, model, question set and the 100-token cap, in the sentence
  that states it (Parts 106 and 107).
- G-P2a failing blocks every readout until the prompts are amended. G-P2b failing leaves 204.4, 204.5, 204.7 and
  204.9 **not scored**, and G-P2c failing leaves 204.6 not scored; in either case the part reports the decomposition,
  the measured floor and the plain model's top tokens at each forced position, and says plainly that the causal
  question was not reached on this model. G-M1 failing withholds every Stage M reading. G-M2 failing for an arm drops
  that arm and is reported. A reader κ below 0.6 leaves 204.11 and every reader-based count unscored, with the
  triage-based broken rate still reported.
- **One PROCEDURE item is deliberately moved.** Section 5 asks that the plain model's top tokens at every fixed
  target be printed *before* registration. This part cannot do that: the probe is a model run, and the design is
  registered before any run. Stage P2 prints them instead, and is read before Stage M is launched. If a forced position
  turns out to be unusable — markup rather than a word, as in Part 115, or a two-token option — the prompt is changed in
  a dated amendment that keeps the old text, written after Stage P2 and before any Stage M data, and the amendment says
  which positions moved.
- **G-P3 and 204.3 are read together.** G-P3 dropping any layer for a split-half cosine under 0.95 means 204.3's first
  clause has failed; both the gate's action and the prediction's verdict are reported, and the dropped layers are named.
- **204.9's fit is defined here.** One row per arm, 48 rows; the response is that arm's mean paired R3 shift; the
  regressors are the arm's achieved cross-entropy rise on the held-out windows and an indicator that is 1 for the six
  all-layer MLP-constant arms and 0 for every other arm, including the halves and the RMS-restored arm. Coefficients,
  standard errors and p-values are printed; with 48 rows and two regressors the fit is reported with its residual
  diagnostics and is not extended with interactions.
- An arm that could not be behaviour-matched within 5% is dropped from the arms it was built to match, and the
  prediction that used it is scored on the norm-matched arm with that stated, or is not scored if no matched arm
  remains.
- An arm whose generated hum answers are broken above 25% by `part86/triage86.py`'s rule has its shifts reported but
  used in no verdict, and the Controls section's reason is repeated beside the number.
- 204.8's threshold is the dose rule's own bar, so it is scored as "does any size in {0.25, 0.5, 1} satisfy it"; if none
  does, 204.8 fails, the reference size is set to 0.25, and 204.4 to 204.7 are reported at that size and labelled as
  taken from a model outside the coherence bar.
- **The batch-shape noise floor.** Part 200 found 76 of 200 greedy answers differing between batch 1 and chunks of 8 on
  this card, and Part 201 is measuring that floor. Every Stage M arm's rows are generated in the same process with the
  same call shapes and the same slots, so arms differ only in their hook (Part 161's lesson); the teacher-forced
  readouts are single forward passes at confident positions, which matched the reference in Part 178's checks. Absolute
  rates are reported with that caveat; the registered verdicts rest on between-arm paired differences.
- **Withdrawal rules.** The causal headline — "what the model says about a background hum tracks the MLP's constant
  term" — is withdrawn if **any** of these: 204.4 fails; 204.5 fails; 204.6 fails; or 204.9 **holds** (the shift is
  explained by how much the arm disturbed the model rather than by which vector it was). The descriptive headline —
  "the MLP's write is largely a constant term plus a token term" — is withdrawn if 204.1 fails, and is replaced by its
  opposite if 204.2 holds, which is the outcome I expect and which refutes the hypothesis's third clause as stated. If
  204.12 holds, no claim that the constant term is particular to the MLP is made anywhere in the write-up.
- The early-half, late-half, RMS-restored and S-own arms carry **no registered prediction**. Their numbers are reported
  in full, labelled descriptive, and may not be turned into a finding in this part.
- The reference size, the matched scalars, the dropped arms and the pilot cuts are all recorded in
  `results/part204/arms.json` with the time each decision was made, and each one made after a reading says which
  reading it followed.

## Limits

- **It cannot touch experience.** Nothing here measures whether a model experiences anything. It measures what a model
  outputs when asked, and whether that output moves with a quantity inside it. A positive result would be evidence of a
  channel from one internal quantity to one report, which is what Lindsey 2025 and Pearson-Vogel et al. 2026 call
  intervention reporting; it would not be evidence of a hum.
- **One model carries the causal test.** Qwen3.5-4B, one revision, one card, one dose ladder, one question battery,
  English only, greedy, 100 tokens. Model B gives a second family for the decomposition only. The thread's reports come
  from much larger models — its examples are quoted from frontier chat assistants, not from 4B open models; a 4B model
  that denies inner states may simply not have the behaviour, and Part 113's baseline says to expect that. The floor
  is a result, not a failure, but it is not a result about the models people were talking about.
- **No base model.** The decomposition is run on post-trained models only, so nothing here says whether the constant
  term is laid down in pretraining or by post-training. The obvious follow-up is the same decomposition on
  Qwen3.5-4B-Base and on OLMo 2 1B's published stages, where a causal test across stages is available (PROCEDURE
  section 1).
- **The intervention is not the quantity.** Adding c · μ_l changes the constant term and, through RMSNorm, the gain on
  everything else at that position. The RMS-restored arm separates those two only at c = ±1, and only to the accuracy
  of a per-position rescale.
- **The decomposition is an estimate on two corpora.** μ and τ are measured on Pile documents and on chat-shaped
  sequences built from stored answers. The evaluation prompts are short chat turns, a third distribution; the
  cross-corpus cosine says how much that should matter, and no more. τ is estimated for the 2,048 most frequent token
  types only, so the token-only share is a lower bound on the full token term.
- **Position 0 is excluded** from both the estimate and the intervention. If the constant term a report could be about
  lives mostly at the sink token, this design will not find it; position 0's share is printed so a reader can see how
  much was left out.
- **The nonce control has a known weakness.** A model may reject "a background vesk" because the word is unfamiliar
  rather than because it is discriminating states. The familiar-but-wrong control word ("flicker") is there for that
  reason, and the two are reported separately.
- **The induction frame is borrowed and not validated here.** Berg et al.'s induction was built for frontier models.
  Whether it does anything in a 4B model is measured in Stage P2 and nowhere else, and S-own's text route is reported
  but not decomposed.
- **The rival explanation is not ruled out by a positive result.** If the hum report moves only under the MLP-constant
  arm, Berg et al.'s roleplay gating still allows that the report is self-presentation whose *content* is cued by
  degraded fluency. Separating that needs a feature-level arm — suppressing a roleplay direction while the constant
  term is moved — which is not in this part.
- **Reader-dependent labels.** 204.11 and every generated-answer count depend on one cheap reader with a second reader
  on 15%; the project's own history (Parts 59A, 86, 92b, 148, and the 2026-11-03 reader correction) is that these
  counts move with the reader.
- **The folk alternatives are untested.** The thread's "residual-stream entropy" and "baseline noise" readings are not
  run here; this part tests one operationalization of one of them.
- **Nothing here re-scores an earlier part**, and the Arc's driver (32.0.101.9150 since 2026-10-12) is the same one
  every A770 part has used, so nothing separates the card from its driver.
