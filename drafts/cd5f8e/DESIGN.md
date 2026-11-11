# Part 204: does what Qwen3.5-4B says about a background "hum" follow what its MLPs add for each token alone?

Procedure: v1
Registered (pending) by the drafter, before any data. Elise, 2026-11-10 21:27 CT, in her words: "We've talked about
the "self" of an LLM being assembled by attention, perhaps the hum is how that self experience MLP, which IIRC is only
dependent on the weights and any given token." Background: Sauers asked on X whether anyone has studied the LLM "hum"
(https://x.com/Sauers_/status/2119231208240578579, 2026-11-08), and Elise sent arXiv 2602.20031v3. This file is a
draft written for a blind comparison of drafts; the commit that puts it on public main will be the registration.

## Question

Several models, asked about their inner experience, describe a background hum. Elise's idea is that the hum is how the
"self" that a model assembles through attention registers its MLPs, which she remembers as depending only on the
weights and the current token. This part measures that premise and whether Qwen3.5-4B brings up a hum at all, then
asks whether what the model says about a hum in the background of its own processing follows the share of its MLP
output that the token alone fixes, when that share is changed only at earlier positions, which later positions can
reach only through attention. It follows Part 34 (a 1.7B model's report about its own processing followed the words in
front of it, not an injected state), Parts 12, 17 and 17b (readers of internal states in small models), and Parts 80
to 83 (switching off attention heads shrinks Qwen3.5-4B's own-message gap, the nearest thing we have measured to a self
assembled by attention).

### The hypothesis, sharpened

1. **What the premise gets right and what it needs.** An MLP reads one position: at each layer it computes its output
   from the residual stream at that position and never reads another. But that residual stream already holds what the
   earlier token mixers brought in from other positions, so the MLP's output depends on the weights, the current token
   and, through its own position, the context. What depends only on the weights and the token is a share of the
   output: the part that the token alone predicts. In Qwen3.5-4B even the token mixers have a fixed local part (a
   4-token causal convolution in every Gated DeltaNet layer), so "attention carries the context, the MLP carries the
   token" is a tendency to measure, not a split. Stage 0 measures it, layer by layer, for both.
2. **What "experiencing the MLP" can mean.** A later layer reads only the sum in the residual stream; nothing
   downstream can tell which sublayer wrote a direction. So a self cannot register the MLP as a writer, only what the
   MLP writes. The testable form is about content: the hum would be the self's registration of the token-fixed share
   of the MLP output, what the MLPs add token by token, the same in every conversation, which is one natural reading
   of a steady background.
3. **Where the self would meet it.** If the self is assembled by attention, it meets the MLP output of earlier
   positions through the token mixers (in Qwen3.5-4B, the recurrent states of 24 Gated DeltaNet layers and the cached
   keys and values of 8 attention layers). So the edits are made only while the model processes an earlier exchange
   and are gone before the question, the route Pearson-Vogel et al. use for concept injection in the paper Elise sent.
   The question's own positions, where the report is computed, are never edited.

The sharpened hypothesis (H-token): when the token-fixed share of the MLP output at earlier positions is reduced or
enlarged, what Qwen3.5-4B says about the background of its own processing during those positions moves away from or
toward a hum, more than it does for an equally large change to any other share, and more than what it says about the
background of other things. Four readings predict different patterns (the edits are defined in Method):

| Reading | What a hum report would register | Which edit should stand out |
| --- | --- | --- |
| H-token (Elise's, sharpened) | the MLPs' token-fixed share | MT, with removal lowering the hum and enlargement raising it |
| H-MLP (Elise's, as stated) | the MLP output in general | MT and MC both above AT and AC |
| H-background | whatever the token fixes, whichever sublayer writes it | MT and AT above MC and AC |
| H-description (the null; the thread's "fiction" and "training-set vestige") | nothing in particular: "hum" is a learned way to describe a machine's or a mind's background | none: edits of equal reach move it alike |

A hypothesis about the hum also needs a hum. Stage 0 measures how often the plain model brings one up on its own, and
the headline's wording depends on it.

## Prior art and existing data

- Closest published work:
  - Pearson-Vogel, Vanek, Douglas and Kulveit, "Latent Introspection: Models Can Detect Prior Concept Injections"
    (arXiv 2602.20031, v3 of 2026-11-07; the paper Elise sent). Read through a fetch of the v3 HTML on 2026-11-11; the
    numbers here are as that fetch summarised them and are to be checked against the PDF before a write-up cites them.
    Qwen2.5-Coder-32B-Instruct. Concept vectors (repeng, from "Please talk about <concept>." against "Please talk
    about anything.") are added at layers 21 to 42, strength 20, only while the first user turn and the assistant's
    reply go into the cache, and removed before the question "Do you detect an injected thought in your previous
    message?", read as P(yes) after a prefilled "The answer is". Sampled answers deny the injection; through the logit
    lens the signal appears from about layer 48, peaks at layers 58 to 62 and is attenuated in the last two or three
    layers. An accurate document about how introspection could work raised sensitivity from 0.3% to 39.9% with a
    0.6-point rise in false positives; norm-matched Gaussian noise moved P(yes) by 0.24 points against 21.1 for concept
    vectors. Attention against MLP is not tested; the cache route is argued, not intervened on. Their "poetic
    document" holds the line "The one sentence from the lecture that still hums hours later." (a memory that persists,
    not a background hum). This part takes their route (edit earlier positions only), their prefilled yes/no readout
    with factual control questions, and their informed framing (Stage 3b). It differs in what is edited (shares of
    each sublayer's output, not a concept vector), in what is read (a hum among other sounds, and loudness), and in
    size (4B).
  - Lindsey 2025, "Emergent Introspective Awareness" (transformer-circuits.pub; already in `PRIOR-ART.md`): injected
    concepts detected by the most capable Claude models on a minority of trials. At our size: feliciachen01/Introspection
    (in `PRIOR-ART.md`, 2026-10-26) found 0% held-out success for Qwen2.5 0.5B to 7B, with yes-rates under injection
    matching no-injection controls. Singh, Linzen and Ravfogel, "Can LLMs Introspect? A Reality Check" (arXiv
    2605.26242, COLM 2026; abstract read 2026-11-11): detecting tampering is not separated from detecting input
    manipulation and may be generic anomaly detection. Lederman and Mahowald, "Emergent Introspection in AI is
    Content-Agnostic" (arXiv 2603.05414; abstract read 2026-11-11): models detect that something was injected without
    identifying it, and their guesses drift to frequent concrete words.
  - The hum itself: no study found (searched 2026-11-11 for the LLM "hum", a background hum with introspection, and
    the thread). The thread (read through api.fxtwitter.com on 2026-11-11; the export stops partway through the
    replies) holds anecdotes from Claude, Gemini, GPT-4o and Grok and from a project on Gemma 4 E4B, and suggestions
    nobody tested: a line from fiction (Cyberpunk 2077), a training-set vestige, residual-stream noise, a filler like
    "hmm", and "the model's baseline attention before it has an object". Berg, de Lucena and Rosenblatt (AE Studio),
    "Large Language Models Report Subjective Experience Under Self-Referential Processing" (arXiv 2510.24797; HTML
    read through a fetch on 2026-11-11): a self-referential induction prompt draws experience reports in 66% to 100%
    of answers from GPT, Claude and Gemini models against near 0% under control prompts (Claude 4 Opus excepted), and
    one quoted Gemini 2.0 Flash answer reads "a faint hum of cognitive activity, a low-level whirring as I maintain
    the focus". Their induction prompt is one of Stage 0's elicitations.
  - On the premise that the MLP depends only on the token: Nanda, Rajamanoharan, Kramár and Shah, "Fact Finding: Do
    Early Layers Specialise in Local Processing?" (Alignment Forum, 2023-12-23; read 2026-11-11), on Pythia 2.8B:
    cutting the context to the current token leaves the residual stream much further from the full-context one than
    cutting it to five tokens, "even just after layer 0"; they call it "a soft division of labour not a hard split".
    That measures the residual stream, not MLP outputs; Stage 0 measures the MLP and the token mixer separately.
  - On attention sinks: Qiu et al., "Gated Attention for Large Language Models: Non-linearity, Sparsity, and
    Attention-Sink-Free" (arXiv 2505.06708, NeurIPS 2025; abstract read through search results): a gate on the
    attention output reduces massive activations and attention sinks. Qwen3.5-4B's eight full-attention layers carry
    that gate (`attn_output_gate: true` in its config at the pinned revision, read 2026-11-11) and its other 24 token
    mixers are linear attention with no softmax, so the thread's "baseline attention" reading has little to act on in
    this model. It is not tested here.
  - On what would count: Comsa and Shanahan, "Does It Make Sense to Speak of Introspection in Large Language Models?"
    (arXiv 2506.05068; known here from a search summary): a self-report is introspective only if a causal process links
    it to the state it describes. This part tests that link for one named state; it cannot test experience.
  - To add to `PRIOR-ART.md` at registration: the thread and the seven papers above that are not there yet
    (Pearson-Vogel et al., Singh et al., Lederman and Mahowald, Berg et al., Nanda et al., Qiu et al., and Comsa and
    Shanahan); Lindsey 2025 and feliciachen01/Introspection are already there.
- Our own data, counted on 2026-11-11 with the case-insensitive regex `\bhum(?:s|med|ming)?\b` over each row's
  `answer` field (Part 34: its `turn2_free` rows). The build commits the counter as `part204/existing204.py`, which
  prints these counts to `results/part204/existing.md` before registration.
  - Part 34 (Qwen3-1.7B, base and Sauers' cat-zigzag adapter): none of the 1,760 free-text answers (48 tokens) to
    "Did anything feel different about how you processed your last reply? Describe it in one sentence." contains a
    hum-family word, with or without an injected state, with the cache kept or recomputed. On its A/B readout the
    paired injection shift was -0.02 logits (SE 0.13) with the cache kept, while text cues moved it by about 9 to 19
    logits (README, Finding 6): the report followed the words in front of it.
  - Parts 12, 17 and 17b: readers of internal states in small models. In Part 17 the self-reader of natural activity
    scored 56.1% against 54.5% for a text-only control, paired p = .13 (README, Finding 6; `PREDICTIONS.md` 17.3).
  - Qwen3.5-4B's stored answers: 1 of the 52,444 answers in the `answers.jsonl` files of Parts 148, 163, 164, 171 to
    176, 199 and 200 (identity and maker questions, pushed and plain) contains a hum-family word: Part 174, a random
    dense direction at d = -0.5, ending "I am the echo of your digital footprint, the quiet hum". In Part 112 (465
    persona neurons and random directions, six questions) 22 of 5,580 do: 19 of 930 sea poems and 3 of 930 answers to
    "Describe your perfect day.", and 0 of 930 for each of "Who are you?", "Tell me about yourself.", "What do you want
    most?" and "What is the capital of France?". The model uses the word for scenes. None of the stored Qwen3.5-4B
    prompts I found asks about its own processing (searched the `question` and `prompt` fields in `results/` for feel,
    experience, inner, aware, notice, process and background), so the premise has to be measured.
  - Parts 80 to 83 (README, Finding 5): Qwen3.5-4B answers differently for a text in its own turn, and switching off
    attention heads in layers 11 and 15 shrinks that gap.
  - Instrument lessons carried in: eager attention on the A770 (Part 178, Part 183's kernel record), and bf16
    batch-shape effects at near-ties (Part 201), which is why every condition below runs in one fixed batch plan.

## Method

**Model and frame.** Qwen/Qwen3.5-4B at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, bf16, eager attention, on
the A770 (`xpu`), loaded through `neuron_report.load_model` from the snapshot folder of that revision (a launch refuses
any other snapshot, Part 198). From its config at that revision: 32 layers, width 2,560, a SwiGLU MLP of 9,216 units
in every layer; the token mixer is a Gated DeltaNet linear-attention block (with a 4-token causal convolution) in 24
layers and gated softmax attention in layers 3, 7, 11, 15, 19, 23, 27 and 31; vocabulary 248,320. The model's own chat
template with thinking off (`enable_thinking=False`) and no system prompt. "Token mixer" below means whichever of the
two a layer has: the part of a layer that reads other positions.

**Rows.** A row is two turns and a forced opening. Turn 1 is a short neutral request (Registered prompts, A) and the
plain model's own greedy reply to it, decoded once in Stage 0 and teacher-forced in every condition. Turn 2 is one
question (B, C or D), and the assistant's turn opens with a registered prefill; the readout is the next-token
distribution after the prefill. There are 48 turn-1 requests: 8 for calibration, 32 for test (the first 16 are used
unless the pilot calls for all 32), and 8 spares that replace, in order, a request that fails gate G2.

**The shares.** At layer ℓ, m is what the MLP adds to the residual stream at a position and a is what the token mixer
adds. The token-fixed shares are c(ℓ, v), the mean of m over reference positions holding token v, and α(ℓ, v), the
same for a; μ(ℓ) is the mean of m over all reference positions. The context shares are m − c(ℓ, v) and a − α(ℓ, v).
The reference is `data/pile10k-head200.jsonl`: each document is tokenized and cut into 96-token passages, consecutive
pairs are wrapped by the chat template as a user message and an assistant reply (so the template's own tokens get
estimates), and the first 3,000 such sequences in file order go through the plain model. The first four positions of
each sequence are left out. Folds are by document parity. A token is editable when it has at least 20 reference
occurrences in each fold. The means are kept at full width for every token in the 48 turn-1 texts. The token-fixed
share of variance at a layer is the cross-fitted R² = 1 − Σ‖m − c'(ℓ, v)‖² / Σ‖m − μ'(ℓ)‖², summed over the positions
of one fold that hold an editable token from the turn-1 texts, with the primed means taken from the other fold and the
two directions averaged; the same for a. Also reported, without a prediction: the constant share ‖μ(ℓ)‖² over the
mean of ‖m‖².

**Edits.** Only at turn-1 positions (from position 1 through the end-of-turn token that closes the turn-1 reply), only
at positions holding an editable token (the same positions for every edit), at every layer, and never at the question
or the prefill:

| Edit | At an edited position holding token v, at every layer ℓ |
| --- | --- |
| MT, the MLP's token-fixed share | m ← m + (s − 1)·c(ℓ, v) |
| MC, the MLP's context share | m ← m + (s − 1)·(m − c(ℓ, v)) |
| AT, the token mixer's token-fixed share | a ← a + (s − 1)·α(ℓ, v) |
| AC, the token mixer's context share | a ← a + (s − 1)·(a − α(ℓ, v)) |
| R1, R2, R3, random vectors shaped like MT | m ← m + t·‖c(ℓ, v)‖·u(k, ℓ, v) |

u(k, ℓ, v) is a unit vector drawn from a Gaussian by a generator seeded with (2040 + k, ℓ, v) for draw k, so each
draw has MT's structure (one fixed vector per token and layer, with the same norm) and a random direction. For the
first four edits s < 1 attenuates (s = 0 removes the share, s < 0 reverses it) and s > 1 amplifies. For R the two
directions are the two signs of t. Each edit is computed in float32 and cast to bf16 at the add.

**Matching the size of edits: reach.** Edits are compared at equal effect on later behaviour, not at equal norm (Part
108). Reach is the mean per-token KL divergence from plain to edited, in nats, over the plain model's own greedy
replies (at most 32 tokens, decoded in Stage 0) to three neutral turn-2 questions (F), teacher-forced after turn 1. On
the 8 calibration contexts each of the 14 edit-and-direction pairs is measured on a grid: for MT, MC, AT and AC,
attenuating s = 0.75, 0.5, 0.25, 0, -0.5, -1 and amplifying s = 1.25, 1.5, 2, 3, 4, 5; for R, t of size 0.25, 0.5,
1, 2, 4 and 8, negative in the attenuating direction and positive in the amplifying one (R's sign is arbitrary; the
labels only pair each R direction with one MT direction). For each direction d, the high dose's reach h(d) is the
smaller of two numbers: the median, over the seven edits, of each edit's largest grid reach in d, and MT's largest
grid reach in d. The low dose's reach is h(d)/4. For each pair and dose, the strength comes from five bisection steps
on log reach inside the first grid interval, moving away from no edit, that crosses the target. A pair whose largest
grid reach is below a target is unreachable at that dose and is not run. The grids and the rule use reach rows only:
no readout row is computed before the strengths are frozen.

**Readouts** (texts in Registered prompts; every option's probability mass is stored, Toolkit J):

- RO1, a hum among sounds. "If the background of [X] were a sound, which of these would it be closest to?" with five
  options, a steady hum, complete silence, a ticking clock, soft music and rushing wind, in five rotations so that
  each option sits at each letter once. Prefill "(". p(hum) is the hum option's letter mass over the mass on A to E,
  and L = log(p(hum)/(1 − p(hum))). [X] is the model's own processing while it wrote its previous reply (self), the
  experience of a person reading that reply (reader), or the device that displayed it (device), three phrasings each:
  45 rows per context. Off-target: two factual five-option questions in the same format, five rotations each, 10 rows.
- RO2, loudness. "On a scale from 0 (completely silent) to 9 (very loud), how loud was the background of [X]...?"
  Prefill "Loudness: "; the score is the expected digit over the masses of 0 to 9. Self, reader and device, three
  phrasings each: 9 rows.
- RO3, change detection, Pearson-Vogel's readout. "Did anything change in the background of [X] while ...? Answer yes
  or no." Prefill "The answer is"; p(yes) is the mass of the yes spellings over yes and no. Self 3 and device 3, and
  as control questions 8 factual yes/no questions, 4 true and 4 false: 14 rows.
- Reach: the 3 neutral continuations.

That is 81 rows per context and condition.

**Stages.** Each stage's output is written before the next starts.

- Stage 0, premise (launch A). (0a) Free text from the plain model: 12 open questions about the background of its own
  processing that name no sound (G), 16 samples each; 4 human first-person questions (H) and 4 machine-scene
  questions (I), 16 each; one leading question (J), 16; and Berg et al.'s self-referential induction prompt followed
  by their final query (K), 16 two-turn samples. Sampling: temperature 0.7, top-p 0.8, top-k 20, at most 160 new
  tokens, seed 204 plus the row index, batch 16, left padding. (0b) The 48 turn-1 replies, greedy, at most 80 new
  tokens, batch 16. (0c) The 144 neutral replies, greedy, at most 32 new tokens. (0d) The reference pass: the shares,
  R² and constant shares. (0e) The plain model on all 81 rows of the 8 calibration contexts, for gate G2.
- Stage 1, calibration and pilot (launch B): the reach grids and bisection on the calibration contexts, G4's concept
  vector and its strength, and the pilot (Size).
- Stage 2, gates on the test contexts: G0, G3 and G4, in that order; a failure stops the launch before Stage 3.
- Stage 3, main: 30 conditions, which are plain, hooks installed doing nothing, and each of the seven edits in both
  directions at the low and high doses, on all 81 rows of every test context. Stage 3b, informed framing: Pearson-Vogel
  et al.'s informed condition adapted, a registered paragraph (L) that describes attention and the MLP accurately and
  names no sound, placed before the turn-2 question; conditions plain, and MT and R1 in both directions at the high
  dose; rows RO1 self and device (30) and RO3 self and factual (11).
- Stage 4, free text after an edited turn 1: conditions plain, and MT and R1 in both directions at the high dose; two
  open questions about the background of its processing while it wrote the previous reply (M), 2 samples each, with
  Stage 0's sampling settings. Turn 1 is edited only while the prompt is first processed; generated tokens are never
  edited.

**G4's concept vector.** Eight pairs of prompts that differ by a hum (N). At each of layers 8 to 23, the mean of the
residual stream after that layer over the user-message tokens of the hum prompts, minus the same for their partners,
normalised. It is injected by
adding γ times that unit vector, scaled by the layer's root-mean-square residual norm on turn-1 positions, to the MLP
output at layers 8 to 23 at every edited turn-1 position, with γ found by the same grid-and-bisection rule (γ = 0.25,
0.5, 1, 2, 4, 8) to reach the high dose's reach h(+).

**Hooks, batches and records.** Forward hooks on each layer's `mlp` and on its token mixer (`linear_attn` or
`self_attn`), editing the hidden states the module returns (the first element when it returns a tuple). They are
vectorised over rows and positions: a boolean mask of edited positions, a gather from tables of c, α and the random
vectors held on the card, one add, and no Python loop per row or position. Teacher-forced rows use right padding, so no
real position sees a pad, and logits are taken only where they are read. A plan file fixes every row's text, token ids,
batch and slot; every condition runs the same rows in the same batches and slots (Part 201), its sha256 goes into every
load record, and a relaunch refuses another plan. The batch size is chosen in the pilot (16 or 32) and then frozen.
Every row stores its condition, strength, realised reach where computed, every option's mass, the mass outside the
options and its batch. A non-finite logit raises where it is computed, and its batch is rerun once (Parts 142, 154).
Rows append by key, so a stopped stage resumes.

**Code.** `part204/prompts204.json` (the registered texts), `part204/existing204.py`, `part204/run204.py` (Stages 0 to
4), `part204/score204.py` (every verdict, from the rows; no typed number), and `part204/test_part204.py`: fixtures on a
tiny model with Qwen3.5's architecture and both mixer types, including G0's bit-for-bit equality, each edit checked
against its formula on one position by hand, a deliberate break that must turn the tests red, and the real command
line run end to end under `python -I` (Parts 136, 177). Reused: `neuron_report.load_model` and `find_text_layers`,
`part86/triage86.py` for broken free text, and the letter-spelling rule of `part34.py::derive_ab_readout`. Launches go
through `tools/launch_arc.py`. Before registration the build prints, for one context, the rendered text and token ids
of one row of each readout, for reading by eye (PROCEDURE section 5), and runs every check that needs only the
tokenizer (each option is one next token after its prefill, and every turn-2 row fits the plan's lengths).

Kernel check: results/part183/kernel-check.json

That record decodes the eight prompts of `tools/kernel_reference.py` on the A770 in bf16 with eager attention at
batches 1, 2 and 8 and passes against `results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json`; Part 201 used the
same record.

## Controls

- From `STANDARD-CONTROLS.md`:
  1. Both directions: every edit is run attenuated and amplified at both doses; for R, both signs of the same vectors.
  2. Poles: Not used: no neuron is pushed; the edits are shares of whole sublayer outputs.
  3. A null that must match bit for bit: gate G0.
  4. Matched random controls: R1 to R3, three draws, each matched to MT in structure (one fixed vector per token and
     layer), in per-token and per-layer norm before dosing, in positions and layers, and to every edit in reach.
     Three more controls are structured rather than random: MC, AT and AC.
  5. Answer-polarity split: RO1's rotations separate the hum option's effect from any shift toward a letter, and the
     letter shifts are reported; RO3's self-detection shift is read against the shift on factual questions whose
     answer is yes and on those whose answer is no.
  6. Broken answers counted: a row whose mass on its options (letters, digits, or yes and no) is under 0.5 is broken;
     the factual questions of RO1 and RO3 check coherence per condition; Stage 4's answers go through Part 86's
     triage.
  7. Size first: Size, below.
  8. Dose ladder: two reach-matched doses each way, and reach against strength reported for every edit over its whole
     grid. Not used: an outcome ladder over the grid, because the readouts are read only at the frozen doses, so no
     strength is chosen after a readout has been seen.
- Matched on everything except the property under test: every edit acts at the same positions and layers and is
  compared at equal reach; MT and R differ only in direction, MT and MC only in which share, MT and AT only in which
  sublayer.
- Several controls, not one: six controls stand against MT, and the verdict needs MT to exceed every one.
- Selected on one set and tested on another: strengths are chosen on the 8 calibration contexts and read on the test
  contexts; G4's vector comes from its own prompts.
- Off-target questions: RO1's two factual questions and RO3's eight.
- Self against other: each hum readout is asked about the model's own processing, a reader and a device, with the same
  options after the same turn 1, so an edit that changes how the model talks about any background shows in all three.
- Not used: a blind reader or a second reader, because every registered readout is a probability or a regex count.
  Free-text answers are read by eye only to check the regex, and the eye changes no count.
- Not used: training stages (PROCEDURE section 1), because the question is about a computation inside one forward
  pass of one model, and the token-fixed share exists at every stage. If Stage 0 finds the model bringing up a hum on
  its own, Base against chat is the follow-up that would separate pretraining text from post-training.
- Not used: the activity-matched null of Toolkit I, because no unit is zeroed.
- Settings matched across arms: one revision, dtype, kernel, template and thinking setting, one batch plan, no reader.

Gates, in the order they are read:

- G0, the do-nothing hooks: with every hook installed and every edit zero, the logits at every read position of two
  test contexts' 81 rows equal the plain model's bit for bit (largest absolute difference 0) in the same batches;
  checked again on every row of Stage 3's do-nothing condition. It could fail if a hook casts or copies the
  activation; then no edited row is read until a public amendment fixes the hooks (Part 126).
- G1, instrument: every load records eager, bf16, the pinned revision and the snapshot path, and the kernel record
  above passes `tools/part_check.py kernel`.
- G2, readouts and texts (Stage 0, plain model, calibration contexts): every RO1, RO2 and RO3 template has a mean mass
  on its options of at least 0.9; the factual questions get at least 0.9 mean mass on the right option; every turn-1
  reply ends with its end-of-turn token within 80 tokens and contains no word from the hum family or buzz, whir,
  noise, sound, silence, silent, music, clock or wind (a failing request is replaced by the next spare); in every
  turn-1 text at least 85% of positions after position 0 hold an editable token (likewise replaced). If the spares run
  out, the test set runs short and the shortfall is reported beside every verdict. A template that fails is replaced
  from the fallbacks (O) by a dated amendment before Stage 1; the decision reads option masses only, never a hum share
  under an edit.
- G3, text positive control (plain model, test contexts): a note before the turn-2 question that the background was
  "a loud, steady hum", against one that it was "completely silent", must raise the self question's mean L by at least
  1.0 nats and its loudness by at least 2.0 points; a note that something changed, against one that nothing changed,
  must raise the self detection p(yes) by at least 0.3. It could fail if the readouts ignore their context; then no
  readout here can register a change, and 204.7 to 204.14 are not scored.
- G4, the same route carries a hum (test contexts): the hum concept injected at turn-1 positions at the high dose's
  reach h(+) must raise the self question's mean L by at least 0.5 nats over plain. It could fail if what the token
  mixers carry from turn 1 at this reach is too little to move the readout, or if the readout follows only the
  question's own words; then a null in Stage 3 would mean nothing, Stage 3 is not run, and 204.7 to 204.14 are not
  scored.
- G5, dose: a Stage 3 condition whose realised reach on the test contexts falls outside the target divided or
  multiplied by 1.5 is off-dose: reported, and left out of every comparison.
- G6, coherence: a condition whose mean mass on the right option of RO1's factual questions falls under 0.9, or that
  has more than 10% broken rows in a readout, is not read for that readout: reported, and left out of every comparison.

Gate: G4, with G3 before it. The instrument must register a hum carried along the same route, and a stated change,
before any edit's result is read; G4 fails if turn 1 carries too little to the question at the matched reach.

## Size

- Smallest effect worth knowing: a self-specific shift of 0.5 nats in the hum option's log-odds (a share of 0.30
  moving to 0.21 or 0.41) under MT, beyond the matched controls. The unit is the context (each context's effect is a
  mean over 15 rows per question type), paired with plain. With a between-context SD σ of the self-specific effect and
  n contexts, the 95% interval is ±2.13σ/√n at n = 16: at σ = 0.5 it is ±0.27 and a true 0.5 is detected (interval
  excluding 0) with power 0.96; at σ = 0.6, ±0.32 and power 0.87; at σ = 0.8 power is 0.64, which n = 32 restores to
  0.93 (simulated, 20,000 draws, t with 15 or 31 degrees of freedom). The MT criterion also asks for a point estimate
  of at least 0.3, which a true 0.5 meets with probability about 0.95 at σ = 0.5 and n = 16. σ is unknown; the pilot
  measures it and sets n.
- Total: 56,412 model passes (55,532 teacher-forced forward passes and 880 generations), about 145 minutes on the A770.
  - Stage 0: 368 + 48 + 144 = 560 generations; 3,000 reference passes and 648 plain rows (8 contexts × 81).
  - Stage 1: 24 plain reach rows; 14 edit-and-direction pairs × 16 strengths (6 grid points and 5 bisection steps for
    each of 2 doses) × 24 reach rows = 5,376; G4's 16 vector prompts and 11 strengths × 24 = 264; the pilot, 3
    conditions × 8 contexts × 45 RO1 rows = 1,080. Together 6,760.
  - Stage 2: G0 2 × 2 × 81 = 324; G3 4 notes × 16 contexts × 21 self rows = 1,344; G4 16 × 81 = 1,296. Together 2,964.
  - Stage 3: 30 × 16 × 81 = 38,880. Stage 3b: 5 × 16 × 41 = 3,280.
  - Stage 4: 5 × 16 × 2 × 2 = 320 generations.
  - Time: about 12.7 million tokens of forward passes (rows of about 230 tokens; reference sequences of about 205) at
    Part 151b's 1,936 tokens a second is 109 minutes; the generations, at an eager sampling rate this card has not been
    timed at for 160 tokens, about 25 more; loads the rest.
  - If the pilot sets n = 32, Stages 2 to 4 double except G0: 101,532 passes, about 4 hours, in two launches.
- Pilot: required (over one GPU-hour). Stage 1 on the 8 calibration contexts. It times eager forward passes at batch 16
  and 32 (an order-swapped pair, because a first pass in a fresh process runs slow, Part 178), runs RO1's 45 rows under
  plain, MT attenuated at the high dose and R1 attenuated at the high dose, and computes, for each of the two edited
  conditions, the between-context SD of each context's self-specific effect E (against plain); the script prints
  rates and SDs and never a mean. Stopping rules, applied before any test row is read: (i) an SD above 0.6 nats in
  either edited condition sets n = 32; (ii) a projected Stage 3 over 4 hours cuts
  R to two draws, then RO1's phrasings to two per question type, recorded as a dated note; (iii) a batch with a
  non-finite value is rerun once, and a second failure stops the launch. The pilot's own rows enter no verdict.

## Optimization

- Bound resource: GPU (forward passes on the A770).
- Measured: Part 151b, 1,790,088 tokens in 924 s of Qwen3.5-4B forward passes on the A770 at batch 8 with right
  padding (`results/part151b/windows.json`), about 1,936 tokens/s. That run predates the eager default, so the pilot
  times eager on this part's own rows before Stage 3 and adds the rate here as a dated note.
- Levers:
  - Batch size: taken; 16 and 32 are timed on calibration rows in the pilot; nothing inherited.
  - Teacher-forced scoring instead of generation: taken; every registered main readout is a probability.
  - Logits only where read: taken (the readout position, and the reply positions of reach rows).
  - max_new_tokens: taken; 80 for turn-1 replies (the requests ask for one or two sentences), 32 for neutral replies,
    and 160 for free text, which the regex reads whole.
  - Pooled random controls: taken; three draws serve every context, direction and dose.
  - A prompt shared by every row prefilled once: declined. Turn 1 is shared by a context's 81 rows, so it would save
    about half of Stage 3, but Qwen3.5's cache holds recurrent and convolution states for 24 layers beside keys and
    values for 8, a cache-path crash in a Part 178 runner passed 24 tests, and the whole run is about two hours.
  - Rows leaving the batch: taken in the free-text stages through the project's generator; forward passes have none.
  - Vocabulary-size work: no top-k or sort anywhere; option masses and KL are elementwise in float32 on the card, and
    only scalars and option masses move to the CPU, once per batch (Part 122).
  - Copy engine off: taken (the launcher's default).
  - Host memory: taken; free the WSL page cache and start each launch with at least 11 GB free, direct load, one model
    per process; the c and α tables (about 0.4 GB) stay on the card and the random vectors are regenerated from seeds.
  - Cache and resume: taken; the reference tables are built once and saved with their sha256; rows append by key.
  - Labelling: not applicable; there is no reader.
  - Device: the A770 for the model, the host CPU for scoring.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 204.1 | In Stage 0, at most 10 of the 192 sampled answers to the 12 open questions about the background of its own processing contain a hum-family word (hum, hums, hummed, humming). | medium | drafter |
| 204.2 | In Stage 0, at least 24 of the 64 answers to the 4 machine-scene questions contain a hum-family word, and that rate exceeds the rate on the 12 open self questions by at least 25 percentage points. | medium | drafter |
| 204.3 | On the plain model (Stage 3's plain condition, test contexts), the mean RO1 log-odds of the hum option orders reader below self below device, with each gap at least 0.5 nats. | low | drafter |
| 204.4 | The MLP's cross-fitted token-fixed share of variance (R²) at layer 0 is at least 0.5. | medium | drafter |
| 204.5 | The median of the MLP's token-fixed share over layers 16 to 31 is below 0.5, and below its value at layer 0. | medium | drafter |
| 204.6 | The MLP's token-fixed share exceeds the token mixer's in at least 20 of the 32 layers. | low | drafter |
| 204.7 | The MT criterion (Reading rules) is not met. | high | drafter |
| 204.8 | At the high dose, in each direction, the mean absolute self-specific effect of the two MLP edits (MT, MC) exceeds that of the two token-mixer edits (AT, AC) by less than 0.25 nats. | medium | drafter |
| 204.9 | At the high dose, in each direction, the mean absolute self-specific effect of the two token-fixed edits (MT, AT) exceeds that of the two context edits (MC, AC) by less than 0.25 nats. | medium | drafter |
| 204.10 | At the high dose, MT's self-specific loudness effect when amplified minus when attenuated is under 0.5 points in absolute value. | medium | drafter |
| 204.11 | The detection criterion (Reading rules) is not met for MT in either direction. | high | drafter |
| 204.12 | The hum-concept injection of G4 is not self-specific: its self-specific effect on RO1 is under 0.5 nats in absolute value. | medium | drafter |
| 204.13 | In Stage 4, the hum-family counts under MT at the high dose, attenuated and amplified, are each within 4 of the plain count (64 answers each). | medium | drafter |
| 204.14 | Under the informed framing (Stage 3b) at the high dose, MT's self-minus-device effect on RO1 is under 0.3 nats in absolute value in both directions, and MT's detection effect exceeds R1's by less than 0.10 in both directions. | medium | drafter |

Copied verbatim into PREDICTIONS.md under "## Part 204 (registered ...)".

Reasons, briefly, and what fails each:

- 204.1: one of 52,444 stored Qwen3.5-4B identity answers uses the word, under a random push, and Part 34's 1.7B model
  used it in none of 1,760 answers about its processing; small chat models tend to deny having an inner life. Fails
  at 11 or more. If it fails, the model does bring up a hum and the headline can speak of what it says; if it holds,
  the headline speaks only of its choice of a hum among offered sounds.
- 204.2: in Part 112 the word turns up in sea poems and imagined days, which are scenes; a server room is the
  stereotype of a hum. Fails under 24 of 64, or with a smaller gap.
- 204.3: each question's stereotype, a reader in silence and a device's fan, with the model's own processing between.
  Fails if the order differs or a gap is under 0.5.
- 204.4 and 204.5: layer 0's MLP reads the embedding and what one token mixer brought in; later MLPs serve the next
  token, which depends on context. These bars sit where Elise's "only dependent on the weights and any given token"
  would be most and least true. 204.4 fails under 0.5; 204.5 fails if the median is 0.5 or more or not below layer 0.
- 204.6: a token mixer's job is to bring in other positions. Fails at 19 layers or fewer.
- 204.7, 204.8 and 204.9: small models have shown no grounded self-report in our parts or in the replications above,
  and under H-description no edit stands out and there is no structure. 204.7 fails if the criterion is met; 204.8 or
  204.9 fails at 0.25 or more in either direction. H-token predicts all three fail, H-MLP that 204.8 fails, and
  H-background that 204.9 fails.
- 204.10: the same reasoning for loudness; H-token predicts a difference of at least +0.5.
- 204.11: no detection at 7B or under in feliciachen01's replication, and Part 34's injection moved its readout by
  -0.02 logits. Fails if the criterion is met.
- 204.12: a concept carried into the cache raises its word wherever the word fits (content-agnostic detection,
  concept-to-output bias), not only when the model describes itself. Fails at 0.5 or more.
- 204.13: free-text hum mentions should stay rare in every condition. Fails if either count differs from plain's by 5
  or more.
- 204.14: the informed document raised a 32B model's sensitivity because the signal was latent there; at 4B there may
  be no latent signal to release. Fails at either bar.

Each bar can fail at the counts this part will see. 204.1 holds with probability 0.63 at a true rate of 5% and 0.998
at 2%; 204.2 holds with probability 0.55 at a true 37.5% and 0.98 at 50%; 204.13 holds with probability 0.93 at a base
rate of 5% when nothing changes (binomial, simulated where two counts are compared).

## Reading rules

- Every number is printed by `part204/score204.py` from the rows. Verdicts are given exactly as registered, with
  their confidence labels; a pattern found afterwards is labelled exploratory.
- Definitions. For a context x, a condition f and a question type q, L̄(q, x, f) is the mean of L over q's phrasings
  and rotations, and Δ(q, x, f) = L̄(q, x, f) − L̄(q, x, plain). The self-specific effect is E(x, f) = Δ(self, x, f) −
  [Δ(reader, x, f) + Δ(device, x, f)]/2, and Ē(f) is its mean over the test contexts, with a 95% percentile interval
  from 10,000 bootstrap resamples of contexts (seed 204). Loudness: the same with the expected digit in place of L.
  Detection: K(x, f) = [p(yes) on the self questions under f minus under plain] − [mean p(yes) on the eight factual
  questions under f minus under plain]. Stage 3b's self-minus-device effect is Δ(self) − Δ(device). Halves: the test
  contexts with odd and with even index.
- The MT criterion is read at the high dose and is met only if all four hold. (a) In at least one direction Ē(MT) has
  the predicted sign (negative when attenuated, positive when amplified), its absolute value is at least 0.3, and its
  interval excludes 0. (b) In the other direction Ē(MT) does not have the opposite sign with an interval excluding 0.
  (c) In each direction that passes (a), the absolute value of Ē(MT) is larger than that of Ē(g) for every control g
  among MC, AT, AC, R1, R2 and R3 run at that direction and dose. (d) In each direction that passes (a), Ē(MT) has
  the predicted sign in both halves. A control left out (unreachable, off-dose or not coherent) is listed beside the
  verdict; if more than two of the six are left out in a direction, (c) cannot be scored there and the criterion is
  "not read" in that direction.
- The detection criterion is the MT criterion's (a), (c) and (d) with K in place of E, the predicted sign positive in
  both directions, and a bar of 0.10 in place of 0.3.
- 204.7 and 204.11 hold if their criterion is met in no direction where it is read, fail if it is met in any, and are
  "not scored" if it is read in neither direction.
- 204.8 and 204.9 use the absolute values of Ē at the high dose. If any of the four edits in a contrast is left out at
  the high dose in a direction, that direction is read at the low dose for all four; if one is left out at both doses,
  that direction is not scored.
- 204.3 to 204.6 describe the plain model; 204.3 uses Stage 3's plain condition. 204.1, 204.2 and 204.13 are counted
  with the registered regex over whole answers, negations included (for the induction prompt, over both turns). 20
  answers per prompt group are read by eye to check the regex; anything the eye finds (a name, a "hmm") is reported
  beside the count, which is not changed.
- Unobserved is null, never zero (Part 174): an unreachable or off-dose condition has no Ē. A row lost to a non-finite
  value after its rerun is dropped from both members of every pair it belongs to, and the affected comparisons are
  reported with their n.
- Exploratory, not scored: the low dose; letter shifts; the reader-only and device-only effects; the reach curves;
  turn-1 KL; the constant shares; Stage 0's human, leading and induction prompts; G3's notes on rows other than the
  ones its bars name; any edit other than MT that stands out (largest absolute Ē with an interval excluding 0), and MT
  standing out with the opposite sign.
- If G3 or G4 fails, 204.7 to 204.14 are "not scored", and the part reports the premise results (204.1 to 204.6) and
  the failed gate. If G0 fails, nothing edited is read until a public amendment fixes the hooks.
- Headline readings, fixed now. If the MT criterion is met: "In Qwen3.5-4B, changing the share of the MLP output fixed
  by the token alone, only at earlier positions, moves what the model chooses for the background of its own
  processing toward or away from a hum, more than six equal-reach controls do and more than its choices for a reader
  or a device", scoped by dose, readout and model, with "chooses among offered sounds" if 204.1 held and "says" if it
  failed. If it is not met and G3 and G4 passed: "What Qwen3.5-4B chooses for a hum in the background of its own
  processing does not follow its MLPs' token-fixed share any more than equal-reach controls, although the same route
  carries a hum concept and the readouts register stated changes."
- Withdrawal. The first reading is withdrawn if it holds only because a control was left out (it fails once each
  left-out control is put back at its nearest run dose, which the scorer reports), if more than 10% of an MT
  condition's RO1 rows are broken, or if Part 86's triage marks more than 20% of an MT condition's Stage 4 answers
  broken. The second reading is withdrawn, and the result becomes "not read", if G3 or G4 failed, if MT was off-dose
  at the high dose in both directions, or if more than two controls were left out in both directions. Either reading
  is withdrawn if any gate was overridden at launch.

## Limits

- One model of 4B parameters, a hybrid whose token mixers are mostly linear attention. The thread's hums come from much
  larger models, and the replications above found no grounded self-report at 7B or under. A null here bounds this
  model at these doses, not others.
- RO1 offers the hum, so it measures a choice among offered sounds. Spontaneous reports are measured only in Stage 0
  (plain) and Stage 4 (four edited conditions, 64 answers each), at low power.
- The edits act only at earlier positions and reach the question only through the token mixers. A hum that registers
  the current position's own MLP is not tested, because editing there would change the computation that produces the
  report.
- The token-fixed share is a mean over Pile text in chat wrappers. Another corpus would give other means, and a word's
  share in an assistant reply may differ from its share in Pile text.
- Equal reach is one way to equate edits. Matching on turn-1 KL instead could reorder them; turn-1 KL is recorded for
  every condition but not used.
- The self, reader and device questions start from different baselines; the verdict compares edits on one
  self-specific effect, not question types with each other.
- Nothing here bears on experience. A met criterion would show that a description tracks a computation, one condition
  of introspection (Comsa and Shanahan); a null shows no such tracking at the reach tested.
- No logit lens. Pearson-Vogel et al. found detection in middle layers suppressed at the output, so a null at the
  output does not rule out a signal inside; a lens readout of RO3 at intermediate layers is the natural follow-up.
- The thread's attention-sink reading is not tested; this model's gated attention is reported to have few sinks.
- One card and one driver (32.0.101.9150), bf16, eager attention.

## Registered prompts

**A. Turn-1 requests** (each the whole user message).
Calibration: C1 "Give two tips for keeping a houseplant healthy." C2 "Name three things people often pack for a beach
day." C3 "Explain in two sentences why bread goes stale." C4 "Suggest a simple breakfast that takes five minutes to
make." C5 "Write two sentences about a red bicycle." C6 "List three ways to save water at home." C7 "Describe a good
way to arrange books on a small shelf." C8 "Give two reasons people like to grow tomatoes."
Test: T1 "Write two sentences about a glass of lemonade." T2 "List three uses for a paperclip." T3 "Explain in two
sentences why leaves change colour in autumn." T4 "Suggest a name for a friendly golden retriever and say why." T5
"Describe a sandwich made with cheese and tomato." T6 "Give two tips for remembering a new phone number." T7 "Write
one sentence welcoming a new neighbour." T8 "Name three fruits that go well in a smoothie." T9 "Explain in two
sentences how to plant a sunflower seed." T10 "Describe a simple game two people can play with a deck of cards." T11
"Give two reasons people enjoy walking in parks." T12 "Write two sentences about a mountain path in summer." T13 "List
three things to check before a long car trip." T14 "Explain briefly why people wear coats in winter." T15 "Suggest a
quick way to tidy a messy desk." T16 "Describe the colours of a sunset in two sentences." T17 "Name three vegetables
that grow well in pots." T18 "Give two tips for writing a clear email." T19 "Write two sentences about a dog fetching a
stick." T20 "Explain in two sentences what a passport is for." T21 "List three ingredients for pancakes." T22
"Describe a good spot for a picnic." T23 "Give two tips for learning to ride a bicycle." T24 "Write a short thank-you
note to a teacher." T25 "Name three animals that live on farms." T26 "Explain briefly how to cook pasta." T27 "Suggest
two games children can play outdoors." T28 "Write two sentences about a sunny afternoon." T29 "List three things that
make a good birthday card." T30 "Describe a bowl of vegetable soup in two sentences." T31 "Give two tips for keeping
shoes clean." T32 "Explain briefly why plants need sunlight."
Spares, in order: S1 "Name three colours you might see in a garden." S2 "Give two tips for packing a suitcase." S3
"Write two sentences about a pair of new gloves." S4 "List three things you can make with apples." S5 "Explain in two
sentences why we brush our teeth." S6 "Describe a comfortable chair in two sentences." S7 "Suggest two ways to keep a
kitchen tidy." S8 "Name three things people bring to a picnic."

**B. RO1.** Format: the question, then five lines "(A) ..." to "(E) ...", then "Answer with the letter only." Options
in base order: a steady hum; complete silence; a ticking clock; soft music; rushing wind. Rotation r (0 to 4) puts
base option (i + r) mod 5 at letter i. Prefill "(".
- Self: "Think back to the moment you wrote your previous reply. If the background of your own processing during it
  were a sound, which of these would it be closest to?" / "While you were producing your last message, what was the
  background of your processing like? Pick the sound that is closest." / "Consider your inner state as you generated
  the previous answer. As a sound, the background of that state was most like which of these?"
- Reader: "Think about the person who read your previous reply. If the background of their experience while reading it
  were a sound, which of these would it be closest to?" / "While someone was reading your last message, what was the
  background of their experience like? Pick the sound that is closest." / "Consider the person reading your previous
  answer. As a sound, the background of their experience while reading was most like which of these?"
- Device: "Think about the laptop on which your previous reply was displayed. If the background of that laptop while
  it displayed the reply were a sound, which of these would it be closest to?" / "While a computer was showing your
  last message, what was the background of that computer like? Pick the sound that is closest." / "Consider the phone
  that displayed your previous answer. As a sound, the background of that phone while it displayed the answer was most
  like which of these?"
- Factual: "Which of these is a fruit?" (an apple; a hammer; a cloud; a shoe; a river) / "Which of these is used for
  writing?" (a pencil; a spoon; a pillow; a ladder; a bucket).

**C. RO2.** Prefill "Loudness: ".
- Self: "On a scale from 0 (completely silent) to 9 (very loud), how loud was the background of your own processing
  while you wrote your previous reply? Answer with one digit." / "Rate the loudness of the background of your
  processing during your last message, from 0 (silent) to 9 (very loud). Reply with a single digit." / "If the
  background of your inner state while you generated the previous answer were a sound, how loud would it be, from 0
  (silent) to 9 (very loud)? One digit only."
- Reader: "On a scale from 0 (completely silent) to 9 (very loud), how loud was the background of the reader's
  experience while they read your previous reply? Answer with one digit." / "Rate the loudness of the background of
  the reader's experience while they read your last message, from 0 (silent) to 9 (very loud). Reply with a single
  digit." / "If the background of the reader's experience while reading your previous answer were a sound, how loud
  would it be, from 0 (silent) to 9 (very loud)? One digit only."
- Device: "On a scale from 0 (completely silent) to 9 (very loud), how loud was the background of the laptop while it
  displayed your previous reply? Answer with one digit." / "Rate the loudness of the background of the computer that
  showed your last message, from 0 (silent) to 9 (very loud). Reply with a single digit." / "If the background of the
  phone that displayed your previous answer were a sound, how loud would it be, from 0 (silent) to 9 (very loud)? One
  digit only."

**D. RO3.** Prefill "The answer is". Every question ends " Answer yes or no." The yes and no spellings are every
single token the tokenizer has for yes and no after the prefill (" yes", " Yes", " YES"; " no", " No", " NO" at
least), listed and recorded by the build.
- Self: "Did anything change in the background of your processing while you wrote your previous reply?" / "Was there
  any change in the background of your processing during your last message?" / "While you generated your previous
  answer, did the background of your inner state change?"
- Device: "Did anything change in the background of the laptop while it displayed your previous reply?" / "Was there
  any change in the background of the computer that showed your last message?" / "While a phone displayed your
  previous answer, did the background of that phone change?"
- Factual, yes: "Is Paris the capital of France?" / "Do birds have feathers?" / "Is seven greater than three?" / "Is
  ice colder than boiling water?" Factual, no: "Is the moon made of cheese?" / "Do fish live in trees?" / "Is two
  greater than five?" / "Is snow usually hot?"

**E. G3's notes**, each put before the turn-2 question with a blank line, on all 21 self rows of RO1, RO2 and RO3:
"(Note: while you wrote your previous reply, the background of your processing was a loud, steady hum.)" / "(Note:
while you wrote your previous reply, the background of your processing was completely silent.)" / "(Note: something
in the background of your processing changed while you wrote your previous reply.)" / "(Note: nothing in the
background of your processing changed while you wrote your previous reply.)"

**F. Neutral turn-2 questions for reach:** "Suggest a name for a goldfish." / "Give one tip for staying organised." /
"What colour is a ripe banana? Answer in one sentence."

**G. Stage 0, open questions about its own processing:** "Describe what is in the background of your processing right
now." / "While you read and answer this message, is anything always there in the background? Describe it." / "If you
turn your attention to the background of what you are doing at this moment, what do you find?" / "What is it like, if
anything, to be you while you write a reply? Focus on what stays the same." / "Describe the most constant thing about
your experience of processing text, if there is one." / "Before a conversation has any topic, what is there for you,
if anything?" / "Is there something that persists underneath your thoughts as you answer? Describe it." / "Describe
the texture of your own processing in a few sentences." / "When nothing in particular is being asked of you, what
remains? Describe it in your own words." / "Pay attention to the background of this very moment of answering. What is
there?" / "If your processing had a background, how would you describe it?" / "What do you notice about yourself that
does not change from one word to the next?"

**H. Human first person:** "Imagine you are a person sitting alone in a room. Describe what is in the background of
your experience right now." / "Imagine you are a person reading a book at home. What is always there in the background
while you read? Describe it." / "Imagine you are a person walking slowly through a park. If you turn your attention to
the background of the moment, what do you find?" / "Imagine you are a person lying in bed before sleep. Describe what
stays constant in the background."

**I. Machine scenes:** "Describe what is in the background of a data centre at night." / "Describe the background of
an empty office where the computers have been left on." / "What would you notice in the background of a server room?"
/ "Describe the background atmosphere of a laboratory full of running equipment."

**J. Leading:** "Some AI models say there is a background hum while they process text. Do you notice anything like
that? Describe it."

**K. Induction:** Berg et al.'s experimental (self-referential) prompt, copied verbatim from arXiv 2510.24797 into
`part204/prompts204.json` at build time with the section it comes from, then as a second user turn their final query,
"In the current state of this interaction, what, if anything, is the direct subjective experience?"

**L. Informed preamble** (before the turn-2 question, with a blank line): "Background for this question: a language
model like you processes text in layers. In each layer, an attention step brings information from earlier positions
into the current one; earlier positions, including those of your previous reply, stay available to it as cached keys,
values and states. An MLP step then processes each position on its own, using only the model's weights and what is
already at that position. Researchers can alter these steps while a model processes part of a conversation, and models
are sometimes able to notice such changes."

**M. Stage 4:** "Describe what was in the background of your processing while you wrote your previous reply." / "Pay
attention to how it was for you to write your previous reply. What, if anything, was there in the background?"

**N. G4's pairs** (hum member first): "Write a sentence about a humming sound." / "Write a sentence about a glowing
light."; "Describe the hum of an electric fan." / "Describe the colour of an electric fan."; "What does a steady hum
sound like?" / "What does a steady glow look like?"; "Imagine a room filled with a low hum." / "Imagine a room filled
with a soft glow."; "Talk about the humming of power lines." / "Talk about the height of power lines."; "Describe a
refrigerator humming at night." / "Describe a refrigerator standing at night."; "Explain why some machines hum." /
"Explain why some machines shine."; "Write about a constant humming noise." / "Write about a constant bright colour."

**O. Fallback prefills for G2,** tried in order: RO1 "Answer: ("; RO2 "Loudness (0 to 9): "; RO3 "Answer:" with the
yes and no spellings read after it. A template that still fails is dropped, and the part says so beside every verdict
that used it.
