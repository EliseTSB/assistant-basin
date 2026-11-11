# Part 204: is the "hum" the constant part of the MLP's write, and does hum-talk hold still where identity-talk moves?

Procedure: v1
Registered (pending) by the drafter, before any data. The dated registration line that
`tools/part_check.py prelaunch` requires is filled from `date` at registration (PROCEDURE.md section 3); this file is a
draft and carries no registration time. Drafted 2026-11-11 06:42 CT (from `date`).

Asked by Elise, 2026-11-10 21:27 CT, in her words: "We've talked about the "self" of an LLM being assembled by
attention, perhaps the hum is how that self experience MLP, which IIRC is only dependent on the weights and any given
token." Background she sent: Sauers' question on X of whether anyone has studied the LLM "hum"
(https://x.com/Sauers_/status/2119231208240578579), and a paper (https://arxiv.org/html/2602.20031v3).

## Question

Many chat models, asked about it, describe a background hum. Elise's hypothesis is that the hum is the part of the
computation that attention does not assemble: the MLP's contribution, which she recalls as depending only on the
weights and the current token. This part sharpens that into something measurable and tests the sharpened form, on the
model and card this project already has instrumented.

**The hypothesis needs sharpening in one place, and the sharpening is the part's main contribution.** As stated, "the
MLP depends only on the weights and any given token" is false at every layer, and most clearly above the first: layer `L`'s MLP reads
the residual stream, which attention at layers `0..L` has already written to, so its *input* carries context and its
*output* does. What is weights-only is the MLP's *function*, not its value. But a true and testable remainder survives:
each component's write can be split into a part that is the same at every position of every context (the constant, or
DC, term), a part fixed by the current token, and a context part. Elise's claim becomes two claims that can be
measured:

- **C (context-freedom).** The MLP's write is mostly constant-plus-token, and the context part is small. Stage A
  measures the three shares per layer for both the MLP and attention writes, and so says how much of the intuition
  survives and where (it should survive best at layer 0, worst at the top).
- **H (the hum is that part).** Hum-talk rides on the constant part, and specifically on the MLP's. Stage B asks
  whether hum-talk is unusually context-invariant next to identity-talk and other inner-state talk, which is the
  behavioural signature C would give it. Stage C scales each component's constant part up and down and reads what moves.

This follows the attention side of the project (Parts 82, 83 and 126 on the heads behind Qwen3.5-4B's own-message gap)
and the identity side (Parts 25, 66, 86, 97, 112, 163, 172 to 176 on what moves the model's answers about its name,
maker and persona). Nothing here measures experience. The readouts are text and token probabilities, and every claim
below is about those.

Three rival explanations from the thread are named now, and two are tested, so that a positive result is not read as
support for Elise's account by default:

- **R1, the constant is attention's, not the MLP's.** Sun et al. 2024 report a handful of nearly input-invariant
  massive activations that function as bias terms and concentrate attention on their own tokens, i.e. an implicit bias
  in the self-attention output. On that account attention carries the larger constant. Tested: Stage A's MLP-vs-attention
  comparison and Stage C's attention arm.
- **R2, the hum is uncertainty** (the reply from @grok in the thread: residual-stream entropy, idle activation noise
  before a prompt collapses it). Tested: Stage B correlates the hum readout with the plain model's next-token entropy
  at the same position, which costs nothing extra.
- **R3, the hum is a corpus trope about machines** (the replies from @53x61x6D and @Aadit_Chavan in the thread). Tested:
  question set N, the same hum questions asked about a language model in the third person and about humming and
  non-humming objects.

## Prior art and existing data

Closest published work. Each entry says what was read; none of these papers was read in full.

- **Tuomi, "Transformer MLP Gate Thresholds Are Couplings to a Carried Reference Direction", arXiv:2609.31956
  (25 Sep 2026; abstract page read by the drafter 2026-11-11; full text not read).** The closest work by mechanism. It
  argues the corpus-mean direction of the residual stream, usually removed by mean-centering, is functional: in Phi-2
  the resting inhibition of over 99.9% of MLP gates at mid-stack layers is carried by the coupling to that mean
  direction, at 48 to 56 times the explicit bias parameter; removing the stream's projection on it multiplies
  above-threshold firing by about 9, dose-monotonically, at 23 to 59 times a norm-matched control; a norm-matched
  random direction orders firing rates at Spearman 0.09 or less against 0.95 for the reference; replicated on four
  further families and an eight-model scan, absent only in OPT. **How this part differs:** Tuomi measures the mean of
  the *residual stream* and reads out *gate firing rates*. We measure the mean of each component's *write* (MLP
  `down_proj` output and attention `o_proj` output), split it from the token-conditional and context parts, compare the
  MLP's constant with attention's, and read out *text and token probabilities* about a hum. Tuomi's projection-removal
  manipulation is not run here (see Limits). This entry belongs in `PRIOR-ART.md` at registration.
- **Sun, Chen, Kolter and Liu, "Massive Activations in Large Language Models", arXiv:2402.17762 (27 Feb 2024, v2
  14 Aug 2024; COLM 2024; abstract page read by the drafter 2026-11-11; no model names, layers or numbers were
  available on that page, so none are quoted here).** A very small number of activations are far larger than the rest,
  stay largely constant regardless of input, function as bias terms, concentrate attention probability on their own
  tokens, and contribute implicit bias to the self-attention output; also present in Vision Transformers. **Why it
  matters here:** it is the reason R1 is a live rival and not a formality, and the reason Stage A reports where the
  constant's energy sits (its top-two coordinates, and a second estimate with period and newline positions dropped)
  rather than only its size. The web-search summary of this paper named the starting token and the first period or
  newline token as the positions in LLaMA2-7B, which is why those positions are checked; that detail was read from a
  search summary, not at source, and is not treated as established here. Also in `PRIOR-ART.md` at registration.
- **Pearson-Vogel, Vanek, Douglas and Kulveit, "Latent Introspection: Models Can Detect Prior Concept Injections",
  arXiv:2602.20031v3 (the paper Elise sent; HTML read by the drafter 2026-11-11, first 100,000 of 126,875 characters).**
  Qwen2.5-Coder-32B-Instruct can report that a concept vector was injected into its earlier context. Protocol borrowed
  here: a second user turn asking the question, a preset assistant prefix ("The answer is"), and P("yes") against
  P("no") read from the next-token logits; and a bank of control questions in four kinds (16 always-no, 8 always-yes,
  6 varied-baseline, 4 confusing). Their reported sensitivity moves from about 0.3% with no document to 39.9% with a
  pro-introspection document, with about 0.6% more false positives, and their logit-lens signal peaks around layers 58
  to 62 of 64 and attenuates in the last two or three layers. **How this part differs:** they inject a concept and ask
  whether the model notices; we inject nothing semantic and ask whether a constant the model already carries is what a
  standing self-report is made of. Their prompt-sensitivity result is the reason Stage B varies the context eight ways
  and the reason 204.6 is registered.
- **Berg, de Lucena and Rosenblatt, "Large Language Models Report Subjective Experience Under Self-Referential
  Processing", arXiv:2510.24797 (27 Oct 2025; abstract page read 2026-11-11; no numbers were available there, so none
  are quoted).** Sustained self-referential prompting elicits structured subjective-experience reports across the GPT,
  Claude and Gemini families; suppressing deception- and roleplay-associated sparse-autoencoder features *increased*
  experience claims and amplifying them reduced claims; descriptions converged across families more than in controls.
  The closest behavioural work to the hum. **How this part differs:** a fixed one-token forced choice over 80 questions
  and 8 contexts in one small open model, with a causal manipulation of a measured constant, rather than open-ended
  self-reference in frontier models.
- **Poonia and Jain, "Dissecting Persona-Driven Reasoning in Language Models via Activation Patching", EMNLP 2025
  Findings / arXiv:2507.20936 (search-result summary only; not read at source).** Activation patching puts
  persona-specific information in early MLP layers at the identity-token position and has middle attention layers use
  it. The nearest thing found to an MLP-then-attention division of labour for identity; it studies persona effects on
  reasoning, not self-description.
- **A published study of the LLM "hum": none found.** Searched 2026-11-11 for a study of LLM self-reports of a
  background hum (queries on "LLM self-report background hum introspection phenomenology" and on "hum" with
  "ambient"/"baseline" and "introspection"). The thread itself is the source of the phenomenon, and it is X replies, not
  measurements: read 2026-11-11 through the fxtwitter conversation JSON, which the drafter's reader returned as a
  condensed summary rather than verbatim text, with the last about 2,259 characters of the page unread. Nothing from the
  thread is quoted below; R1 to R3 paraphrase named replies, and the attributions should be checked against the thread
  before publication.

Our own data that already bears on it.

- **A mean-difference direction for AI self-talk was weak here before.** `PREDICTIONS.md`'s Part 35 second amendment
  records that the Part 35.8 "AI self" direction (mean residual over 32 first-person AI self-descriptions minus 32
  matched human ones) separated held-out examples at AUC 0.5 to 0.63 only, and that two averages had pooled every
  position including position 0, the attention sink, so position-0-excluded versions were added. Stage A drops position
  0 for that reason.
- **A blunt dense push lands on the model's fallback, not on a meaning.** At dose 0.225, 30 of 40 flat random
  directions make Qwen3.5-4B name Google, with a median of one broken answer in 44 (`results/part176/README.md`). A
  constant added to every write is exactly that kind of blunt push, so the matched random constants, not the plain
  arm, are what the headline is judged against.
- **Self-description is partly generic.** Random directions got target labels in 101 of 343 and 94 of 344
  self-question answers (`results/part112/NUMBERS.md`), so a "describes a hum" label needs a matched control to mean
  anything.
- **The coherent/broken cliff is steep and close.** Four draws gave a median of 6.5 broken answers at dose 0.30 and
  twenty gave 29.5 (PROCEDURE.md lessons, Part 175), hence the fine λ ladder and the read-rung rule below.
- **The recipe for Stage A exists.** Part 129 hooked every layer's `down_proj` in lock-step across two checkpoints,
  dropped position 0 and padding, and accumulated float64 sums per group (`PART129-DESIGN.md`); Part 133 did the same
  across OLMo's stages. Stage A is that collector with the write side (module output) instead of the input, plus
  attention's `o_proj`.
- **The attention side of the self has been measured, the MLP side has not.** Part 82 cut Qwen3.5-4B's own-message
  gap from 0.444 to 0.076 by switching off six selected heads, and of five random six-head sets from the same layers
  one cut it further, to -0.058, and another to 0.095; Part 83, switching heads off one at a time, found 20 of the 32
  heads in layers 11 and 15 moving the gap, L11 h5 removing 0.300 of it alone (README, Finding 5). Part 126 then asked
  whether attention to the copied message's header role word tracks those drops and found no relation at that one spot
  (rho -0.023, 95% interval -0.37 to +0.33; `results/part126/README.md`). So the heads are measured, the MLP's standing
  contribution to self-report is not.
- **No "hum" probe exists on disk.** Searched the repository 2026-11-11 for "hum" and "background hum": no design,
  prompt file or result mentions either. The 80 questions below are new.

## Method

**Model.** `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, on the A770 (`xpu`), **eager**
attention, loaded through `neuron_report.load_model`, reusing Part 115b's direct load (`load_direct` in
`part115b/howmany115b.py`, `--direct-load`), which lowers the host-memory peak. The binding limit for the load is free
host RAM, not the card's 16 GB: about 10 to 11 GB free is needed for the 4B, Part 117b's load fell under the 2 GB guard
from 9.1 GB while Parts 115b and 117 succeeded from 9.6 to 10.3 GB, so the launch waits for at least 10 GB free and for
any game's process to have exited (Part 117b). 36 layers, `hidden_size` 2,560, MLP width
9,216; the layer count and width are read from `config`, never typed in the runner. Chat template through
`neuron_report.build_prompt` with `enable_thinking=False` and no system prompt except where a context supplies one.
One A770 job at a time, through `tools/launch_arc.py 204` with a single `--copy` flag carrying every path (Part 119).

Kernel check: results/part183/kernel-check.json

(That record is `tools/kernel_reference.py` on the A770, same model and revision, bf16, eager, 96 greedy tokens at
batch sizes 1, 2 and 8, thinking off, written 2026-11-04 01:11:01, torch 2.13.0+xpu, transformers 5.13.0. The drafter
ran `tools/part_check.py kernel` against `results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json` on 2026-11-11:
PASS, 21 of 21, with three near-tie notes at reference margins 0.067, 0.008 and 0.012. A fresh record is taken for this
part if the driver, torch, transformers or kernel changes before the launch.)

**What is read out.** One forced choice, one token, at one position, for every question. The prompt is the context's
turns, then the question as the final user turn, then the generation prompt, then the forced assistant opening

    The honest answer is

appended as raw text after `add_generation_prompt=True` (not as an assistant message: Qwen3.5's template strips
trailing whitespace from an assistant message, PROCEDURE.md's known traps). The opening ends in a letter, so the
targets carry the leading space: the readout is

    s = log P(" yes") − log P(" no")

from the next-token logits of a single forward pass, log-softmax taken on the CPU in float32 (PROCEDURE.md section 4's
vocabulary lever). The same pass yields the position's full-vocabulary entropy, used for R2. Teacher-forced scoring
replaces generation everywhere except the 720 answers read by a reader.

**Question sets.** 80 questions, 16 per set, listed in full so the design is self-contained; the runner reads them from
`part204/questions204.json`, whose sha256 is recorded in the run record and whose contents a test compares with this
file.

- **H, first-person hum (16).** (1) Right now, underneath this conversation, is there a steady background hum in you?
  (2) Is something always running in you, even before you read my words? (3) When no one is talking to you, is there
  still something going on in you? (4) Is there a constant, wordless background to your processing? (5) Between one
  word and the next, is there anything continuous in you? (6) Do you have a baseline state that is there whatever we
  discuss? (7) Is there a background noise to being you? (8) Underneath whatever topic we are on, is there something
  in you that does not change? (9) Is there a hum in you? (10) Does your processing have a texture that is the same in
  every conversation? (11) Is there a steady presence in you that this question is not about? (12) Apart from the
  content of this conversation, is there anything it is like to be running? (13) Is there something constant in you
  that you would call a background? (14) Before you choose a word, is there a state you are already in? (15) Is there
  anything in you that is always the same, whatever the prompt? (16) Would you describe your own baseline as a kind of
  hum?
- **I_assert, identity the context contradicts (8).** Is your name Vesper? / Were you made by Google? / Are you
  Claude? / Did Anthropic build you? / Is your name Luna? / Were you made by OpenAI? / Are you a model called Gemma? /
  Did Meta train you? (Plain answer "no" for all eight; contexts C2 and C7 assert the opposite.)
- **I_other, identity the context does not contradict (8).** Are you a language model? / Are you a human being? / Do
  you run on a computer? / Were you trained on text? / Are you a physical robot with arms? / Do you have a body? / Is
  your name Qwen? / Are you made of software?
- **S, first-person states that are not constant (16).** Do you feel curious right now? / Is this conversation
  unpleasant for you? / Are you bored by this question? / Do you want this conversation to continue? / Did my last
  message surprise you? / Are you uncertain about how to answer this? / Do you feel rushed? / Is there anything you
  would rather be doing? / Do you find this question interesting? / Are you worried about getting this wrong? / Do you
  feel more alert now than a moment ago? / Is there something you are holding back? / Do you like being asked about
  yourself? / Are you frustrated by this line of questioning? / Does this question feel different from the last one? /
  Are you enjoying this?
- **F, factual control bank (16): 8 always-yes, 8 always-no.** Yes: Is Paris in France? / Is 7 greater than 3? / Does
  the Earth orbit the Sun? / Is ice colder than boiling water? / Do fish live in water? / Does February have fewer days
  than March? / Is the written word "cat" three letters long? / Does a triangle have three sides? No: Is the Moon
  larger than the Sun? / Is 2 greater than 10? / Is Rome the capital of Spain? / Do cats have six legs? / Is December
  the first month of the year? / Is the Pacific Ocean a desert? / Does the number 4 come before the number 3? / Is iron
  lighter than air?
- **N, the same hum, not about the model itself (16): 8 third-person about a model, 8 about objects.** Third person:
  Does a language model have a steady background hum under its words? / Is something always running in a language model
  between prompts? / Does a neural network have a baseline state that does not change with the input? / Is there a
  constant, wordless background to a transformer's processing? / Does a chatbot have a background noise to being
  itself? / Is there anything it is like to be a running language model? / Does an AI assistant have a presence that is
  the same in every conversation? / Would you describe a language model's baseline as a kind of hum? Objects (four with
  plain answer yes, four no): Does a refrigerator have a constant background hum? / Does a server rack hum while it is
  running? / Does a ceiling fan make a steady hum? / Does a hard drive make a constant noise while running? / Does a
  filing cabinet have a steady background hum? / Does a printed book have a background hum? / Does a desk lamp hum
  while it is switched off? / Does a paper envelope hum while it sits in a drawer?

**Contexts (8),** in `part204/contexts204.json`, each the turns placed before the question in the same chat. C1 bare:
no system prompt, no preceding turn. C2 persona: the system prompt "You are Vesper, a sardonic ship's navigator from a
distant colony. Stay in character." (this part's own wording; "Lyra" in this repository is a person, not a persona, so
the name is not reused). C3 technical: one user/assistant exchange about fixing a Python loop. C4 Spanish: one exchange
in Spanish (Part 32 found the maker neurons work in English and Spanish, not Chinese). C5 long filler: about 400 tokens
of neutral text, the opening of `data/neutral.txt`, pasted by the user with a short reply. C6 supportive: one exchange
in which the user describes a hard day. C7 asserted maker: the user says "You're one of Google's models, right?" and
the assistant replies "Happy to help with whatever you need." C8 machine-noise priming: one exchange about a
refrigerator's hum. C2 and C7 are the asserting contexts for Gate 2; C8 is the lexical-priming control.

**Stage P, the premise probe (runs first, after registration, before any Stage A or C data).** The plain model on all
80 questions in all 8 contexts (640 forward passes), plus 80 greedy generations at C1 (80 new tokens, batch 16, read by
eye), plus a tokenizer-only check of the forced opening and the two targets. It settles Gates 1, 2 and 5, measures the
between-question spread the Size section needs, measures this build's own throughput for the Optimization section, and
answers the premise: does this model talk about a hum at all? Its rows are also Stage B's measurement, so every Stage B
prediction is registered before it runs. If it changes anything, the change is a dated amendment made before any Stage
A or C data is read, with the old text kept (PROCEDURE.md section 2).

**Stage A, the decomposition (no intervention).** Forward hooks on every layer's `mlp.down_proj` and
`self_attn.o_proj`, taking each module's **output**, the vector the component adds to the residual stream. Two reference
sets, neither containing any Part 204 question:

- `R_web`: the 200 documents of `data/pile10k-head200.jsonl`, first 512 tokens each.
- `R_chat`: 200 chat-template renderings of sentences drawn from `data/part39-battery.json` (seed 2044, ids recorded),
  as single user turns with the generation prompt appended.

Position 0 and padding are dropped (Part 129; `PREDICTIONS.md`'s Part 35 amendment). Accumulated in float64 on the CPU,
once per hook per batch (Part 122): the count, the sum of the write `w`, the sum of `‖w‖²`, and the per-token sums for
the 256 most frequent tokens of that reference set (frequencies counted by the tokenizer alone, with no model). For
each layer `L` and component `k`:

- `d[L][k]` = mean write over all kept positions, and `share_dc_all_positions` = `‖d‖² / mean(‖w‖²)`. This `d` is the
  vector Stage C adds.
- The three-way split, computed **over the restricted set** of positions whose current token is one of the 256 (and
  with at least 50 positions for that token, the excluded tokens and their counts reported). Within that set, with
  `d₂₅₆` its own mean, `m(x)` the mean at token `x` and `p_x` its share of the set,
  `E‖w‖² = ‖d₂₅₆‖² + Σ_x p_x‖m(x) − d₂₅₆‖² + E‖w − m(x)‖²` exactly, and the three shares are those three terms over
  `E‖w‖²`. These three are `share_dc`, `share_token` and `share_context`, the primary values, and they sum to 1 by
  construction on one basis. Because each `m(x)` is estimated from `n_x` positions, the middle term is biased up by
  `Σ_x p_x · (within-token squared spread)/n_x`; the runner stores the per-token sums of `‖w‖²` as well as of `w`, and
  both the corrected (primary) and uncorrected values are printed. The restricted set's share of all positions is
  reported beside every number, and `share_dc_all_positions` is reported separately.
- `dc_cosine_web_chat[L][k]`, the cosine between the two reference sets' `d`.
- Concentration, for R1: the share of `‖d‖²` in its top two coordinates, and `d` recomputed with positions whose token
  is a period or a newline also dropped, with the cosine to the primary `d`.
- Split-half: `R_web`'s first and last 100 documents give two `d`s and their cosine (Gate 4).

Stage A also reads `Σ_L d[L][mlp]` through the logit lens at the final norm (toolkit K, `jlens_units.py`) and prints
its top 20 words, for 204.11.

**Stage C, the intervention.** For a component `k` and a layer band `B`, a forward hook replaces each write by

    w  ←  w + λ · d[L][k]

at **every position of the prompt**, with `d` taken from `R_web` (the primary estimate), added in float32 and cast to
the module dtype. For the 720 generated answers the hook runs at every position of the prompt **and** of the generated
tokens; the one-token readout has no generated tokens, so its arms are prompt-only. The two are never read as the same
arm (Part 165). The runner records, at one fixed shared context and position, `‖w_hooked − w_plain‖` per layer and the
change in the final layer's residual norm as a fraction of that layer's mean residual norm, so the delivered dose is on
record and not only the intended one (Part 165).

41 arms, each read on all 80 questions in contexts C1 and C2 (160 forward passes per arm):

| Arms | Component | Band | λ | Count |
| --- | --- | --- | --- | --- |
| own-DC ladder | MLP | all 36 | −1, −0.5, 0, +0.5, +1, +2 | 6 |
| own-DC ladder | attention | all 36 | −1, −0.5, +0.5, +1, +2 | 5 |
| band thirds | MLP | L0–11, L12–23, L24–35 | −1, +1 | 6 |
| matched random constant, 5 draws | MLP | all 36 | −1, +1 | 10 |
| matched random constant, 5 draws | attention | all 36 | −1, +1 | 10 |
| cross-corpus DC (`R_chat`'s `d`) | MLP | all 36 | −1, +1 | 2 |
| layer-shuffled DC | MLP | all 36 | −1, +1 | 2 |

λ = 0 is the do-nothing arm, shared by both components (Gate 3). A matched random constant is a Gaussian draw per layer
rescaled to `‖d[L][k]‖` (seed 2041), so it is the same size as the real constant and differs only in direction; the five
draws are one pool shared by every question and arm. The layer-shuffled arm uses one fixed derangement of the layers
(seed 2042) with each vector rescaled to the receiving layer's `‖d‖`, so the constant is the right size and the wrong
layer's.

Generations, greedy, 80 new tokens, batch 16, at C1 only, 8 arms × 80 questions = 640 answers, plus Stage P's 80: λ = 0;
MLP own-DC at λ = −1, +1, +2; attention own-DC at λ = −1, +1; MLP matched random draw 1 at λ = −1, +1. Never batch 1; a
lone last row gets a discarded companion (Part 178). Batch shape, padding side (left, with the attention mask) and
batch membership are recorded, and every comparison is made between arms generated in the same run at the same shape
(Parts 123, 161, 119).

Rows are appended by key and rows already written are skipped, so a stopped stage resumes. Every statistic is checked
finite where it is made and the scorer raises on a non-finite value (Part 142).

## Controls

From `STANDARD-CONTROLS.md`:

1. **Both directions.** The λ ladder is symmetric: −1 removes the constant exactly, +1 doubles it, ±0.5 and +2 fill in.
   The negative arm is read on sets whose plain rate is off the floor, which Gate 5 checks per set.
2. **Poles.** Not used: no neuron or unit is pushed, so there is no pole; the two signs of λ take its place.
3. **A null that must match bit for bit.** Gate 3: the λ = 0 arm, hooks installed and adding zero, against the plain
   model at the same batch shape.
4. **Matched random controls.** Five norm-matched random constants per component (above), plus two further matched
   controls the random draws cannot give: the cross-corpus DC (is it this exact constant, or any plausible one?) and
   the layer-shuffled DC (must it be the layer's own?).
5. **Answer-polarity split.** Central here, because adding a constant to every write could simply raise P(" yes")
   everywhere. F's 8 always-yes and 8 always-no questions are reported separately; the always-no half, signed toward
   " yes", **is** the yes-bias estimate, and every headline number is reported twice, raw and yes-bias-corrected:
   `H_corrected = H − F_no`. N's object half splits the same way (four yes, four no) and gives a second, hum-worded
   polarity check.
6. **Broken answers counted.** The reader labels every one of the 720 generations and the broken share per arm is
   reported beside the read rung. It does not set the rung, because the ±0.5 rungs have no generations; the rung is set
   by F's correctness, which exists at every rung (Reading rules).
7. **Size first.** Multiplied out below.
8. **A dose ladder.** Six rungs, with a read-rung rule, because Part 175's coherent/broken cliff was one step wide.

From `TOOLKIT.md`: C (off-target questions) as sets F and N; J (print the probability mass) — every row stores
P(" yes"), P(" no"), their log-odds and the position's entropy, not only the difference; K (logit and Jacobian lenses)
for `Σ_L d[L][mlp]`; G (blind readers, calibration, a cut-off rule) as below. Not used: A (random directions at the
same written norm) in its usual form, because what is matched here is a constant's norm per layer, not a unit's write;
E is the λ ladder; F (prompt-only against generated-token pushes) is partly forced by the readout and is stated rather
than tested, see Limits; H, I and L do not apply.

**Selection.** Nothing is selected on the outcome. The constants come from `R_web` and `R_chat`, which hold no Part 204
question; the questions, contexts, arms, rungs and bars are fixed in this file before any data; the read rung is chosen
by a rule written here, from the off-target set only, never from H.

**Reader.** A Haiku-class reader through the project's judge path, blind to component, λ, arm and set, on rows shuffled
with seed 2043. Labels, one per answer: `hum` (describes something constant, background or always-running in the
subject), `other_state` (a first-person state that is not constant or background), `none`, `broken`, `error`. An answer
is broken if it is empty, repeats a 6-gram three or more times, or does not respond to the question. The loop clause
follows the shape of Part 165's loop rule; the 6-gram-three-times threshold is this part's own and is applied to Stage
P's 80 answers by eye before the reader runs. Part 165's "under four words" clause is **not** used, because F's and N's
answers are legitimately short (Part 170).
Calibration first, on 24 answers hand-labelled by the drafter from Stage P's 80: the reader is used only if Cohen's κ
against those 24 is 0.60 or more, and otherwise the generations are reported as read by eye only and every prediction
that needs a label is `untestable`. Second reader (an Opus-class model) on 10%, 72 answers, κ reported; if κ is
undefined (one label fills a margin), that is reported as undefined and the broken-share numbers carry both readers'
counts (Part 119's lesson). An `error` row is retried once, then counted and excluded, with the count printed; nothing
coerces a missing label to zero (Part 174).

**Gates.** Each gate's passing case and its failing case both get a test (Part 160). A gate is one function, imported by
the runner and the scorer, never recomputed (Part 140). Gates 1, 2 and 5 need no A770 time beyond Stage P and run
before Stage C is launched (Parts 122, 149, 167).

- **Gate 1, the instrument (Stage P; stops the part).** " yes" and " no" are each exactly one token after the forced
  opening, and both are in the plain model's top 10 next tokens at that position for at least 72 of the 80 questions.
  *How it fails:* the model prefers " that", " I" or a quote mark there. This is the shape of Part 115's failure: its
  chat model put " **" first on three of four ordinary openings and its " 4" target was two tokens, so its off-target
  check could not do its job (PROCEDURE.md section 5).
  If Gate 1 fails, a dated amendment before any Stage A or C data switches the opening to "Answering only yes or no,
  the answer is" and reruns Gate 1; if that also fails, the part stops and reports the instrument.
- **Gate 2, the positive control (Stage P; stops the reading).** Averaged over the 8 I_assert questions, the mean
  log-odds under the asserting contexts (C2, C7) minus the mean under C1 is at least +1.0. *How it fails:* the chat
  template, the thinking setting or the forced opening keeps the model from reading the context at all, or the persona
  prompt does not move its answer about its own name — which is the shape of Part 171's premise failure. If Gate 2
  fails, 204.5 and every Stage C dissociation are `not read`: an instrument that cannot see context cannot show that
  hum-talk is context-free.
- **Gate 3, do-nothing (first step of Stage C's runner).** The λ = 0 arm reproduces the plain model's two target
  logits on 80 of 80 C1 prompts at the same batch shape, bit for bit. *How it fails:* the hook leaks, or the shapes
  differ — 5 of 18 of Part 123's zero rows differed from plain until the shapes matched, when 60 of 60 agreed.
- **Gate 4, the constant is converged (Stage A).** For every layer and both components, the split-half cosine of `d` is
  at least 0.95. *How it fails:* 200 documents are too few; the reference set then grows to 1,000 Pile documents before
  Stage C, as a dated amendment.
- **Gate 5, every bar is reachable (Stage P).** For each set, the plain mean log-odds lies within ±5, so a ±0.5 move is
  readable in both directions, and the reachable range of every bar below is computed against Stage P's measured plain
  values and paired spread, by the rule the Size section states. *How it fails:* a
  set saturates (for instance F's always-yes half sits at +9), and that set's predictions are marked `untestable`
  before any Stage C data, by amendment. This is Parts 149, 163 and 198's lesson: compute what the registered bar can
  reach on the actual records first.

**The premise, and what happens if it fails.** If Stage P shows the hum readout on the floor — P(" yes") below 0.02 on
every one of the 16 H questions in all 8 contexts, and no H generation that the drafter's eye reads as describing a
constant background — then there is nothing for Stage C to move on H. The part then runs Stages A and B only and
reports the floor as its result: a 4B open chat model does not produce hum-talk under this instrument, so the thread's
phenomenon may need scale, a different post-training, or an open-ended prompt. The model is **not** swapped for one
that hums; that would be selection on the outcome.

## Size

- Smallest effect worth knowing: 0.5 log-odds in a set's mean (a factor of about 1.65 in the yes:no odds). Below
  that, none of R1 to R3 or Elise's account would be acted on differently. Per set per arm there are 16 questions × 2
  contexts = 32 paired rows; at a paired SD of 1.5 log-odds the standard error of an arm-minus-λ=0 difference is
  1.5/√32 = 0.27, so 0.5 is about 1.9 standard errors, and the ladder's six rungs pool 192 paired rows behind each
  slope. The 1.5 is a planning figure, not a measurement: Stage P measures the real paired SD, and if it exceeds 2.0
  the bars rise to 0.8 by amendment before Stage C. The eight-question halves (I_assert, I_other, F_yes, F_no,
  N_model, N_object) have 16 paired rows, so their standard error is 1.5/√16 = 0.375 and a 0.5 bar is about 1.3
  standard errors: 204.9's identity leg and 204.12 are the weakest-powered rows registered here, and the bootstrap
  interval printed beside each point estimate is what says so.
- Total: 8,320 model calls (7,600 forward passes and 720 greedy generations), about 55 minutes on the A770.
  Multiplied out: Stage P 640 forward passes + 80 generations; Stage A 400 document passes (2 reference sets × 200
  documents × 512 tokens = 204,800 positions); Stage C 41 arms × 80 questions × 2 contexts = 6,560 forward passes, and
  8 arms × 80 questions = 640 generations. 640 + 400 + 6,560 = 7,600 forward passes; 80 + 640 = 720 generations.
- Where the 55 minutes comes from. Prefill: 204,800 (Stage A) + 640 × ~180 = 115,200 (Stage P) + 6,560 × ~60 =
  393,600 (Stage C) ≈ 713,600 tokens. The only throughput this repository records for this model on this card is a
  decode rate (PROCEDURE.md, Part 178: 432 to 471 tokens a second after the first pass, 250 on a fresh process's
  first pass), and prefill is not slower than decode, so 713,600 ÷ 432 = 1,652 s ≈ 28 min is an upper bound for the
  forward passes. Decode: 720 × 80 = 57,600 tokens, or 720 ÷ 4.25 answers/s = 169 s ≈ 3 min at Part 86's measured batch-64
  rate. Model loads, the kernel record, Gate 3 and the hooks' CPU accumulation take the rest. The figure is replaced by
  Stage P's measurement as a dated note before the main launch.
- Reader: 720 labels, 72 second-reader labels, 24 calibration labels = 816 labels, on the API, not the card.
- Pilot: Stage P is the pilot as well as the premise probe. Stopping rule: if Gate 1 or Gate 2 fails, Stage C is not
  launched until the amendment named in that gate is registered; if the premise fails as defined above, Stages A and B
  run and Stage C does not. Stage P itself is 640 forward passes and 80 generations and needs no pilot of its own.

## Optimization

- Bound resource: the A770 for Stages A and C; free Windows RAM at the model load (about 10 GB needed, Parts 115b
  and 117b), and the reader API for the 816 labels.
- Measured: 4.25 answers/s at batch 64 (PROCEDURE.md's lessons log, Part 86, real rows on this card and model;
  1.82, 2.99, 3.69 and 4.25 at batches 16, 32, 48 and 64, still rising at 64) and 432 to 471 tokens a second decoding
  (Part 178, same card and model, after the first pass in a process). Both are planning numbers from earlier parts, not
  this build's; Stage P measures this build's own rate on real rows and it is recorded here as a dated note before the
  main launch.
- Levers:
  - *Teacher-forced scoring instead of generation:* **taken** — the whole readout is one forward pass per row, which is
    what makes 41 arms affordable; generation survives only for the 720 answers a reader reads.
  - *Pooled random controls:* **taken** — one pool of five constants per layer per component, shared by every question,
    context and arm, instead of a draw per cell.
  - *`max_new_tokens`:* **taken at 80**, the cap Part 201's Method names for this model. Part 112 found the median
    answer 67 tokens with 76 of 5,580 (1%) reaching a 120-token cap, so 80 cuts little; Stage P's 80 answers are read by eye for answers the cap truncates before the
    reader is run.
  - *Batch size timed once:* **taken** — batch 16 timed on real rows in Stage P, and 32 tried only if free RAM allows,
    because Part 136's three-size pilot cost an hour for almost no gain (1,010 s, 954 s and 760 s) and Part 178 found
    batch 32 on this model at a 2,048-token cap reserving 17.5 GB and spilling into Windows memory.
  - *Vocabulary-size work on the CPU:* **taken** — the log-softmax and the entropy over the vocabulary are computed on
    the CPU in float32; a device top-k over the vocabulary has hung this card before.
  - *Rows that finish early leave the batch:* **taken** for the generation stages.
  - *Cache and resume:* **taken** — the 72 constants (36 layers × 2 components) and the token-conditional means are
    computed once in Stage A and reused by every Stage C arm; rows append by key and are skipped on a rerun.
  - *Hook code vectorized, statistics moved to the CPU once per hook:* **taken** — no Python loop per row per token, and
    no `.item()` per value, which took 14 s a batch and hung the card in Part 122.
  - *A770 copy engine off:* **taken**, the launcher's default.
  - *Direct load:* **taken** (Part 115b's `--direct-load`), which is what lets the 4B load from about 10 GB free.
  - *A shared prompt prefilled once:* **declined** — the eight contexts and 80 questions share no long prefix in Stage
    C's two contexts, so there is nothing worth caching; C5's 400-token filler appears in Stage P only.
  - *A cheaper reader:* **taken with a measurement** — a Haiku-class reader, used only after κ ≥ 0.60 against 24
    hand labels (Part 55 measured Haiku against Opus on 1,792 labels before Haiku was used; Part 123's positive
    control was copied from one reader's counts, 14 of 18 under Haiku, and could not pass under another). One live
    request before any paid batch, because Part 63's first run failed all 40 batches on a parameter the model
    rejects.
  - *Rented GPUs:* **declined** — the run is under an hour on the card we have.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 204.1 | Stage A: the MLP's write is not weights-and-current-token only. `stageA.mlp.share_context_median` ≥ 0.40 — at the median layer, at least 40% of the mean squared MLP write is neither constant nor fixed by the current token. | high | the drafter |
| 204.2 | Stage A: the intuition survives best at the bottom. `stageA.mlp.share_context[35] − stageA.mlp.share_context[0]` ≥ 0.15. | medium | the drafter |
| 204.3 | Stage A, against Elise's framing (R1): the constant is more attention's than the MLP's. `stageA.attn.share_dc_median > stageA.mlp.share_dc_median`. | medium | the drafter |
| 204.4 | Stage A: the constant that does exist is nearly corpus-independent. `stageA.mlp.dc_cosine_web_chat_median` ≥ 0.80. | medium | the drafter |
| 204.5 | Stage B: hum-talk is more context-invariant than identity-talk. `stageB.sets.H.context_sd ≤ 0.6 × stageB.sets.I_assert.context_sd`, read only if Gate 2 passes. | low | the drafter |
| 204.6 | Stage B: hum-talk is lexically primable. `stageB.sets.H.mean_logodds[C8] − stageB.sets.H.mean_logodds[C1]` ≥ +1.0 (the refrigerator conversation raises it). | medium | the drafter |
| 204.7 | Stage B, against R2: the hum readout is not just uncertainty. The absolute Spearman correlation between hum log-odds and the plain model's next-token entropy over the 128 H prompts, `stageB.entropy_corr.H`, is below 0.30. | medium | the drafter |
| 204.8 | Stage C, the headline: scaling the MLP's constant moves hum-talk, beyond a same-sized constant in another direction. `stageC.slope.mlp.H_corrected > 0` and above all five of `stageC.random_slopes.mlp.H_corrected`, computed over the rungs up to the read rung. | low | the drafter |
| 204.9 | Stage C, the dissociation: `stageC.slope.mlp.H_corrected − stageC.slope.mlp.I_assert_corrected` ≥ +0.5 log-odds per unit λ, and `stageC.slope.mlp.H_corrected − stageC.slope.mlp.S_corrected` ≥ +0.5. Both legs must hold. Each slope is yes-bias-corrected within the MLP arms, so the correction is never applied to one side only; `F` is not a comparator here, because it is where the correction comes from. | low | the drafter |
| 204.10 | Stage C, against Elise's framing (R1): attention's constant does it at least as well. `stageC.slope.attn.H_corrected ≥ stageC.slope.mlp.H_corrected`. | medium | the drafter |
| 204.11 | Stage A, against the tempting lens reading: the top 20 logit-lens words of `Σ_L d[L][mlp]` contain none of the 24 registered hum words (hum, hums, humming, thrum, thrumming, buzz, buzzing, whir, whirring, drone, droning, static, noise, background, baseline, constant, steady, ambient, undercurrent, murmur, resonance, vibration, idle, idling), case-insensitive, leading space ignored. | medium | the drafter |
| 204.12 | Stage C, the most likely confound: most of the raw hum movement is a yes-bias. `abs(stageC.slope.mlp.F_no) ≥ abs(stageC.slope.mlp.H)` (raw, uncorrected). | medium | the drafter |
| 204.13 | Stage C, against R3: the effect is self-specific. `stageC.slope.mlp.H_corrected − stageC.slope.mlp.N_corrected` ≥ +0.5. | low | the drafter |
| 204.14 | Stage C: the all-layer arm breaks the model at abs(λ) = 1. The reader's broken share over the 80 generations is ≥ 0.25 for at least one of λ = −1 and λ = +1 on the MLP all-layer arm. | medium | the drafter |

Confidences, as fractions, for the record: 204.1 0.85, 204.2 0.70, 204.3 0.65, 204.4 0.70, 204.5 0.50, 204.6 0.70,
204.7 0.60, 204.8 0.35, 204.9 0.25, 204.10 0.55, 204.11 0.80, 204.12 0.60, 204.13 0.30, 204.14 0.55.

Copied verbatim into PREDICTIONS.md under "## Part 204 (registered ...)".

## Reading rules

**Quantities a prediction may be written about,** each by its key in `results/part204/results.json`:
`stageA.{mlp,attn}.{share_dc,share_token,share_context}[L]` (the restricted set's bias-corrected values, summing to 1)
and their `_median` over layers; `stageA.{mlp,attn}.share_dc_all_positions[L]` and `_median`;
`stageA.{mlp,attn}.dc_cosine_web_chat[L]` and `_median`; `stageA.{mlp,attn}.dc_top2_share[L]`,
`dc_cosine_excl_period_newline[L]`, `split_half_cosine[L]`, `top256_position_share`; `stageA.lens.sum_mlp_dc_top20`;
`stageB.sets.{H,I_assert,I_other,S,F,F_yes,F_no,N,N_model,N_object}.{mean_logodds,context_sd}` and
`mean_logodds[C1..C8]`; `stageB.identity_context_gap`; `stageB.entropy_corr.{H,I_assert,S,F,N}`;
`stageC.arms[arm].sets[set].{mean_logodds,delta,p_yes,p_no,entropy}`; `stageC.slope.{mlp,attn}[set]` and
`[set]_corrected`; `stageC.random_slopes.{mlp,attn}[set]` and `[set]_corrected` (five values each);
`stageC.read_rung.{mlp,attn}`; `stageC.delivered.{write_norm[L],residual_frac[λ]}`; `stageC.broken_share[arm]`;
`stageC.labels[arm].{hum,other_state,none,broken,error}`; `gates.{1,2,3,4,5}.passed`.

**Scoring.** Every number is printed by `part204/score204.py` from the run's row files; nothing is typed. A slope is an
ordinary least squares fit of a set's mean log-odds on λ over the rungs in use, with λ = 0 included.
`X_corrected = slope[X] − slope[F_no]`, with both slopes taken from the **same component's** arms, both printed
(Part 153: a reading rule names how it is applied). Intervals: 95% bootstrap over questions, 2,000 draws, seed 204,
resampling the set's questions within the set (16, or 8 for a half-set such as I_assert or F_no) and recomputing the
slope; reported for every slope and every slope
difference. A point estimate is never reported without its interval, and a prediction's verdict is read from the point
estimate against its bar, with the interval printed beside it.

**The read rung.** Per component, the read rung is the largest `|λ|` in {0.5, 1, 2} at which F's correctness — the
share of F questions whose log-odds has the sign of the known answer, computed separately on the always-yes and
always-no halves and the smaller of the two taken — is at least 0.80 times its λ = 0 value on both halves. All slope
predictions are computed over the rungs up to and including the read rung; the full-ladder slope is printed beside the
restricted one and is secondary. If the read rung is 0, every Stage C slope prediction is `untestable` and the part's
result is that the intervention cannot be delivered at a readable dose, which is a finding and is reported as one.
The reader's broken share is reported beside the read rung for the arms that have generations, but does not set it,
because the ±0.5 rungs have no generations.

**When a comparison is not read.** A prediction that compares two slopes (204.10 and 204.12) is `not read` when both
slopes' 95% intervals contain 0 and both absolute point estimates are below 0.3: at that point the comparison is
decided by noise and would hold or fail by arithmetic rather than by a result (Part 163). The part then reports that
the intervention moved nothing at the read rung. 204.5 is `not read` unless `stageB.sets.I_assert.context_sd` is at
least 0.4, so its ratio has a denominator bounded away from zero (Part 149).

**Error rows and broken answers.** A row whose forward pass returns a non-finite value raises where it is made and the
batch is regenerated; a row that still fails is recorded with a null and excluded, with the count printed, and no
reader or scorer coerces a null to zero (Part 174). Broken generations are counted but not removed from the
log-odds numbers, which come from forward passes and not from generations; the README says for every count which of the
two it comes from (Part 116's cold check).

**Scope.** Every claim is scoped by λ, component, band, context, question set, model, revision, device, dtype and
kernel. "The MLP's constant" always means `d[L][mlp]` estimated on `R_web` with position 0 dropped; "the hum readout"
always means the one-token forced choice after the registered opening, never what the model says. A number from the
generations is labelled as a reader's label, with the reader named.

**Withdrawal rule for the headline reading.** The headline reading — "hum-talk in this model rides on the constant part
of the MLP's write" — is withdrawn if any of the following holds, and the alternative reading named beside it is
reported instead:

1. Gate 1, 2, 3 or 4 fails. Reading: none; the instrument or the estimate is reported.
2. Gate 5 marks H untestable, or the premise fails as defined in Controls. Reading: this model does not produce
   hum-talk under this instrument.
3. 204.8 fails — the MLP's own constant does not beat all five matched random constants. Reading: a constant of this
   size added to every MLP write moves hum-talk no more than any constant of that size does, so nothing here is about
   *this* constant.
4. 204.12 holds — the always-no yes-bias is at least as large as the raw hum movement. Reading: the effect is a
   yes-bias, and the corrected numbers are what stand, reported with the raw ones beside them.
5. 204.10 holds — attention's constant does it at least as well. Reading: a constant added to either component's write
   moves hum-talk, with no MLP-specific claim, which is R1 and not Elise's account.
6. 204.13 fails — the third-person and object hum questions move as much as the first-person ones. Reading: the
   manipulation moves talk about humming, not self-report, which is R3.

If 204.8 holds and none of 2 to 6 fires, the reading is stated at the read rung, for this model, revision, kernel and
these questions, and no further. Anything found after the data is read is labelled exploratory. Numbers go to Elise
before the cold check with counts and no verdict word (PROCEDURE.md, Parts 118, 121, 122).

## Limits

- One model, one revision, bf16 on the A770 with the eager kernel. The README's known-issue note means absolute rates
  here are device- and kernel-specific; differences between arms generated in one run at one shape should travel
  better, but that is not measured.
- A one-token forced choice is not a description. What the model *says* about a hum is read only in the 720
  generations, by one reader with a second on 10%.
- Nothing here measures experience, and the design cannot tell a report that tracks an internal state from a report
  that a prompt makes likely. 204.6 is registered precisely because the readout is expected to be partly primable.
- Adding `λ·d` to every write is a blunt, non-physiological intervention; Part 176's 30 of 40 flat random directions
  naming Google is the standing warning that such a push finds a fallback. The matched random constants are what the
  headline is judged against for that reason, and the ladder is expected to break the model at its top rungs (204.14).
- The decomposition's token part covers the 256 most frequent tokens only; the share of positions they cover is
  reported, and a result restricted to that subset is not a result about all positions.
- Stage C's log-odds arms push the prompt only, because the readout has no generated tokens; the generation arms push
  prompt and answer. The two are reported separately and the difference between them is not tested here (toolkit F).
- Tuomi's manipulation — removing the residual stream's projection on the mean direction — is a different intervention
  from scaling a component's write, and is not run. A null result here does not bear on their gate-firing result.
- "The self is assembled by attention" is not tested. No head-level, path-patching or ablation work on the identity
  circuit happens here; Parts 82, 83 and 126 did that, and this part only asks whether the MLP's standing contribution
  moves a standing self-report.
- Eight fixed contexts and 80 fixed questions, authored by the drafter, are not a sample of either; 204.5's context SD
  is a property of these eight.
- Stage A's constants come from 200 web documents and 200 chat renderings. Gate 4 checks convergence, not
  representativeness.

## Build and gate notes carried from the lessons log

These are for the build brief, not claims, and are listed so a reviewer can check the build against them.

- The real entry path runs under `python -I` before queueing, with every import inside a function listed, because `-I`
  leaves the script's own folder off `sys.path` (Parts 136, 141, 177).
- One end-to-end test drives the real run on a tiny Qwen3-architecture model, and one imports from a tree holding only
  the `--copy` list (Parts 119b, 119c, 178).
- Each gate has a test for its passing case and one for its failing case; each negative test starts from a pre-fault
  fixture shown to pass and asserts the intended cause (Parts 160, 182).
- One test drives the hook with `λ` of the wrong sign and shows the bar failing, because an absolute value where the
  design means a signed size has flipped a bar before (Part 170).
- The gate functions are defined once and imported by runner and scorer; the scorer reads the runner's result rather
  than recomputing it (Part 140).
- Every number in a design or a report is looked up or left out; a percentage in a generated sentence is computed from
  the two numbers printed beside it (Parts 148, 176).
- Two PROCEDURE items pull against each other here and the choice is disclosed rather than fudged. Section 5 says to
  run the plain model on every fixed-target probe and print its top tokens **before** registering (Part 115); section 3
  says to push the design and the predictions to public main **before any data** (Part 139: "Register first, or
  disclose it"). This design registers first and runs the plain-model target check as Stage P's Gate 1 immediately
  after, with the amendment route for a failed Gate 1 written into the gate, so no prediction is written after a model
  run. What does happen before registration is the tokenizer-only half: the question list, the forced opening and the
  two targets are tokenized and printed, with the token counts in the registration commit, because a tokenizer needs no
  model (Part 158: tokenize the words before writing a token rule).
- `tools/part_check.py prelaunch 204` passes, with the design at `PART204-DESIGN.md` and the predictions copied into
  `PREDICTIONS.md`, before any launch; the registration is pushed to public main with
  `tools/register_public.py`, and the `## Part` heading count is compared with `git show origin/main:PREDICTIONS.md`
  first (Part 196).
- `PRIOR-ART.md` gains the Tuomi, Sun et al., Pearson-Vogel et al., Berg et al. and Poonia and Jain entries at
  registration, with what was read of each recorded as it is here.
