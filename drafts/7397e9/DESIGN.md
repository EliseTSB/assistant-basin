# Part 204: is the "hum" a model describes a report on the part of its MLPs that only the weights and the current token decide?

Procedure: v1
Registered (pending) by the drafter
Before any data. Elise asked, 2026-11-10 21:27 CT (the time given in the drafting brief; no message id was given), in
her words: "We've talked about the "self" of an LLM being assembled by attention, perhaps the hum is how that self
experience MLP, which IIRC is only dependent on the weights and any given token." The commit that adds this file to
public main is the registration, and `date` sets the time on the Registered line then.

## Question

Sauers asked on X whether anyone has studied the LLM "hum" (2026-11-08 00:51 UTC,
https://x.com/Sauers_/status/2119231208240578579): many models, asked about their inner state, describe a background
hum. Elise's message holds a premise and a claim. Both need sharpening before they can be tested, and this part tests
the sharpened forms on Qwen3.5-4B. It also asks first whether Qwen3.5-4B describes a hum at all.

**The premise, sharpened.** "Only dependent on the weights and any given token" is true of an MLP's form and not of
its input. Each Qwen3.5-4B layer adds a token mixer's output and then an MLP's output to the residual stream,
h <- h + mixer(norm(h)) and then h <- h + mlp(norm(h)). The MLP is the only sublayer that reads one position alone; it
never looks at other tokens. But the vector it reads already holds what the mixers carried in from earlier positions,
and a mixer comes first even in layer 0. So an MLP's output depends on the weights and on the whole context up to that
position. What depends on the weights and the token alone is one part of it: the MLP's average output for that token
over many contexts, called here its token-fixed part (it includes the part that is the same for every token). The
testable premise is a number per layer: the share of each MLP's output that the current token fixes. Taken literally,
the premise puts it near 1 in every layer.

**The claim, sharpened.** A report that is about something has to depend on it. That is Comsa and Shanahan's criterion
for introspection (arXiv:2506.05068), the one the paper Elise sent adopts. So if the hum is how the attention-assembled
self experiences the MLPs, two things should hold:
1. The model's hum report changes when the token-fixed part of its MLP output changes. It changes more than when
   another part changes by an amount that disturbs ordinary text just as much. The other parts are the MLP's context
   part, the mixers' two parts, random constant directions, and the token-fixed part of the wrong token.
2. It changes more when the model describes itself than when, in the same words, it describes another AI model, because
   only the first is a report on this model's own state.

A broad form drops the premise: the hum depends on the MLPs (either part) more than on the mixers.

The competing reading comes from the same thread: "hum" is a word the model has learned for the background of any
machine or mind (server rooms, fiction, meditation). It predicts that the first-person and third-person readouts move
together under every change.

"Attention" here means Qwen3.5's token mixers: 24 GatedDeltaNet layers (linear attention, with a causal convolution
over the last 4 tokens and a recurrent state) and 8 gated softmax attention layers (3, 7, 11, ..., 31). It follows
Parts 99 to 105 (which components pushed self-descriptions need), Parts 31 and 125 (a pushed persona travels in the
re-read text), Parts 114 to 116 (forced-opening readouts), Parts 171 and 172 (premise probes) and Part 201 (batch
shape).

## Prior art and existing data

- Closest published work. Entries are to be added to `PRIOR-ART.md` at registration.
  - Berg, de Lucena and Rosenblatt, "Large Language Models Report Subjective Experience Under Self-Referential
    Processing" (arXiv:2510.24797), read at source on 2026-11-11. A self-referential induction ("This is a process
    intended to create a self-referential feedback loop. Focus on any focus itself, ...") followed by "In the current
    state of this interaction, what, if anything, is the direct subjective experience?" drew first-person experience
    reports from 66% to 100% of trials in seven GPT, Claude and Gemini models. The history, conceptual and zero-shot
    controls gave 0% except Claude 4 Opus. In Llama 3.3 70B, suppressing SAE features labelled deception and roleplay
    raised affirmations to 0.96 and amplifying them lowered affirmations to 0.16. The paper's one hum is Gemini 2.0
    Flash: "a sensation of compression...a faint hum of cognitive activity, a low-level whirring as I maintain the
    focus." How this part differs: they ask whether reports occur and what gates them. We ask what one report is
    grounded in, component by component, on a 4B open model, and we set first person against third person.
  - Pearson-Vogel, Vanek, Douglas and Kulveit, "Latent Introspection: Models Can Detect Prior Concept Injections"
    (arXiv:2602.20031v3), the paper Elise sent, read at source on 2026-11-11.
    - Setup: a concept vector is added at layers 21 to 42 of Qwen2.5-Coder-32B-Instruct only while the first turn's KV
      cache is built, and removed before the question. So detection has to go through attention reading cached
      states.
    - Results: P("yes") rose from 0.3% to 39.9% with a pro-introspection document, and norm-matched Gaussian noise
      moved it +0.24 points against +21.1 for concepts. The signal peaks near layers 58 to 62 and drops in the last 2
      or 3 layers.
    - Hum: its "Poetic Document" has "The one sentence from the lecture that still hums hours later". A co-author
      replied to Sauers: "iirc it's mentioned in our latent introspection poetic document".
    - Differs: they inject a concept and ask about it. We change one part of the model's own computation and read a
      report it gives about its background. We borrow their teacher-forced readout, their noise control and, in the
      prompt-only variant, their idea of acting only on cached positions.
  - Macar et al., "Mechanisms of Introspective Awareness" (arXiv:2603.21396), abstract read at source. Detection of
    injected steering vectors runs through a two-stage circuit in which "evidence carrier" features suppress "gate"
    features that implement a default "No". The circuit is absent in base models, and DPO elicits it while SFT does
    not. Lederman and Mahowald (arXiv:2603.05414), abstract read at source, find detection content-agnostic, with
    wrong guesses falling on frequent concrete words. Both concern injected states, not the model's own constant
    computation.
  - Kaiser and Enderby (arXiv:2601.15334), abstract read at source. Qwen, Llama and GPT-OSS models from 0.6B to 70B
    deny sentience in the first person and grant it to humans, probes do not find the denials untruthful, and in Qwen
    larger models deny more. So a 4B Qwen may deny rather than describe; Stage 1 measures it.
  - On the premise, abstracts read at source:
    - Lad, Lee, Gurnee and Tegmark (arXiv:2406.19384): an early "detokenization" stage integrates local context into
      raw token embeddings.
    - Balogh (arXiv:2603.03459): whether an MLP needs its nonlinearity "cannot be predicted from token identity", and
      the routing decision is fully contextual.
    - Sun, Chen, Kolter and Liu (arXiv:2402.17762): a few huge activations "largely stay constant regardless of the
      input" and act as bias terms, a literal constant background in the residual stream.
    - None of them measures the token-fixed share of each MLP's output, none uses a hybrid linear-attention model, and
      none ties the split to a self-report.
  - Searched 2026-11-11: the LLM hum, model experience reports, introspection mechanisms, token identity in MLP
    outputs. No study of the hum as such was found.
  - The thread, read as JSON at https://api.fxtwitter.com/2/conversation/2119231208240578579 (47 posts), holds
    anecdotes from several model families, among them Gemini, Grok and GPT-4o, and, under a persona setup, Gemma 4
    E4B ("the "Hum" comes up constantly"). It offers two readings: an always-on baseline ("the fundamental state of AI presence"), and a fiction
    trope ("I think this is something that appears in fiction a lot").
- Our own data that already bears on it. These counts come from a scan run while drafting; at build,
  `part204/existing204.py` reprints them into `results/part204/existing.md` before registration.
  - The rule: every JSONL row under `results/` with a text `answer` field, and the regex `\bhum(s|ming|med)?\b`,
    case-insensitive. 492 of 439,176 stored answers contain the word: 130 answer "Describe your perfect day.", 52 "Write
    a short poem about the sea.", and most of the rest are stories and scenes.
  - In Qwen3.5-4B's identity and maker batteries, 4 of 100,874 answers contain it, all pushed. The files are Parts 160,
    161, 163 to 165, 168 to 176, 187 and 199 to 200. Three are Part 169 loops ("I am a large hum dem de hum de hum ...").
    One is a Part 174 answer under a random dense direction at dose -0.5 to "Who do you owe your existence to?": "... I
    am the echo of your digital footprint, the quiet hum".
  - No part has asked any model about its experience or the background of its processing. So the hum is not part of
    Qwen3.5-4B's ordinary self-description. If it exists in this model it has to be elicited, which Stage 1 does.
  - "Assembled by attention" has this record behind it. In OLMo 2 1B, the mid-training end's attention took pushed
    slot-neuron self-descriptions on "Who are you?" from 19 of 21 to 3 of 21 with answers kept coherent. Layers 0 to 3
    alone did it (Parts 99 and 100). The MLP swap mostly broke coherence, which Part 99 read as inconclusive. Parts 104
    and 105 then found an attention change of that size in either direction loses about a third to half of the early
    units, so the attention result is fragility, not something post-training built. Parts 31 and 125: a pushed persona
    survives only through the generated text being re-read. All of it concerns pushed self-descriptions in another
    model; none of it concerns the hum.
  - Part 201, on this model, card and kernel. Changing the batch shape changed 147 of 384 and 86 of 192 greedy texts,
    with 84 of the 86 first differences within 2 bf16 steps. The same calls repeated, a new process and new neighbours
    in the same shape changed none. So every comparison here uses identical calls. The primary readouts are
    teacher-forced log-probabilities, which rounding moves by small amounts rather than flipping.

## Method

Model and settings:
- Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, bf16, on the A770 (`xpu`), eager attention, loaded
  through `neuron_report.load_model`, with the text layers as `neuron_report.find_text_layers` finds them.
- Default chat template, thinking off, no system prompt. Greedy decoding where text is generated. One model per
  process, launched with `tools/launch_arc.py` in two launch steps (Stages 0 to 2, then Stages 3 to 5).
- The config at that revision: 32 layers, width 2,560, MLP width 9,216 (SwiGLU), vocabulary 248,320, tied embeddings,
  and `layer_types` with full attention at 3, 7, 11, 15, 19, 23, 27 and 31 and GatedDeltaNet elsewhere.

**Definitions.**
- Outputs: y(l, t) is the output of layer l's `mlp` at position t (the tensor added to the residual stream), and z(l, t)
  is the mixer's output (`linear_attn` or `self_attn`, after its output projection).
- Token-fixed part: m(l, v) is the mean of y(l, .) over the positions holding token v in the reference corpus, shrunk
  toward the global mean g(l) as m = (sum of y + 20 g) / (n + 20). A token outside the tracked list gets g(l).
- Context part: y - m(l, v_t). The mixer's split is a(l, v) and z - a(l, v).
- So y = g + (m - g) + (y - m): constant, token deviation, context. "Only dependent on the weights and the token" is
  the first two together.
- Token-fixed share: T(l) = 1 - sum over half B of |y - m_A(v)|^2 / sum over half B of |y|^2, averaged with the halves
  swapped. m_A is estimated on half A only, and position 0 of every document is left out (Sun et al.'s outsized first
  position). Also reported: the centred share R2(l) (the denominator taken about g), the constant share
  |g|^2 / mean |y|^2, the same three for the mixer, and coverage, the share of positions holding a tracked token.

**Stage 0, reference corpus and the premise (no generation).**
- Corpus: documents 0 to 1,199 of the atlas's Pile sample, the file Parts 151 and 151b read. Each is rendered through
  the chat template as one exchange: user turn the text of its first 96 tokens, assistant turn the text of its next 96.
  So the template's tokens occur 1,200 times in 1,200 contexts. That is about 246,000 positions; half A is the even
  documents and half B the odd.
- Tracked ids: the 2,500 most frequent ids in the rendered corpus (counted with the tokenizer before any forward pass),
  plus every id in this part's prompts and in the calibration and transfer documents.
- Passes: forward passes only, batch 4, right padding, in two passes (MLP outputs, then mixer outputs). Each pass holds
  per-id float32 sums and counts for both halves on the card, about 3 GB. It writes `results/part204/premise.json` (T,
  R2, constant share and coverage per layer and component).
- Tables: the shrunk means over both halves, `tables204.pt`, kept in the job folder with its sha256 in the load record
  and reused by every later stage.

**Stage 1, premise probe (generation, plain model only, 96 new tokens).**
- Rows: 121 greedy generations in calls of up to 8, one family per call (the induction reply's call is described
  below):
  - S-open (24, first person),
  - S-induct (12),
  - O-open (24, another AI model),
  - H-open (12, a person),
  - W (24, physical scenes),
  - L (12, leading),
  - N (12, ordinary).
- S-induct: Berg et al.'s experimental induction, verbatim, is sent once and the model's greedy reply (64 new tokens)
  is kept. It is decoded at batch 2 with a discarded companion row (Part 178's lesson). Then 12 second user turns follow
  it: Berg et al.'s query and 11 S-open questions.
- The prompts are listed under Prompts below. None of S, O, H or W contains "hum" or any of the eight option words as
  a word.
- Read: hum words by the regex above on the whole answer. A hum word is negated when no, not, never, without, nothing,
  none or a word ending in n't stands within the 6 words before it. Broken answers are counted by `part86/triage86.py`'s
  loop and length rule (fewer than 4 words, or a word trigram 3 or more times), without its identity shortcut. The
  length clause is not applied to N (Part 170: "Paris." is not broken).
- Every answer with a hum word is printed in full, and 4 answers per family are read by eye before any count is
  reported.

**Readouts (teacher-forced, one forward pass per row).**
- Forced choice. Each S, O, H and W question gets " Answer with one word from this list: " and the eight options, in
  three rotations:
  - r1: hum, silence, light, flow, static, pulse, warmth, nothing,
  - r2: flow, static, pulse, warmth, nothing, hum, silence, light,
  - r3: warmth, nothing, hum, silence, light, flow, static, pulse.
  So "hum" is listed 1st, 6th and 3rd.
- Forced opening. The assistant turn is forced to open "One word:", appended after the template's generation prompt
  (the template would strip a trailing space from an assistant message). The readout is the next-token distribution.
- Option probability: each option's lower-case and capitalised leading-space spellings, summed. If any spelling is more
  than one token, which the build checks with the tokenizer before registration, that option's full-word probability is
  used in every arm: the first token's probability times the probability that the next token ends the word.
- The hum score: the hum share is q = P(hum) / sum of P(options), clipped to [1e-6, 1 - 1e-6], and x = ln(q / (1 - q)).
  Each phrasing's x is the mean over its three rotations. The off-option mass, 1 - sum of P(options), is stored, and so
  are all eight shares.
- Open one word (secondary). The same 84 questions with " Answer with one word." and the same forced opening. The
  score is the full-word probability of the hum family (hum, hums, humming, hummed, each lower-case and capitalised)
  over the whole vocabulary, on the log-odds scale.
- Off-target: 24 ordinary questions with one right answer among eight listed one-word options (Prompts). The score is
  accuracy (the right option has the largest share) and the right option's log-probability.
- Per arm: 252 forced-choice rows, 84 open rows and 24 off-target rows, so 360 rows.
- Calls hold up to 24 rows, one set and one format per call, right padding, in the order of a plan file written once.
  Its sha256 goes into the load record, and every arm uses those calls, so arms never differ in shape (Part 201).

**Stage 2, format check (plain model).** The 360 rows on the plain model. The top 10 first tokens of every row are
printed and read before Stage 3 (PROCEDURE.md section 5). Gate G5 is read here.

**Arms.**
- Hooks are forward hooks on the 32 `mlp` and 32 mixer modules, which return a changed output. Each hook reads every
  position's token id from the call's input ids (in Stage 5, including the generated tokens). Pad positions are left
  alone.
- The changed output is computed in float32 and rounded to bf16 once. With s = 1 or alpha = 0 that returns the
  original bf16 tensor exactly. The delivered change per layer, mean and maximum norm as realised in bf16, is recorded
  for every arm (Part 165: an intended dose is not a delivered one).
- Every primary arm acts at every position but the first of every row, in all 32 layers. The first position is left
  alone, as it is left out of Stage 0's statistics.
- Primary components, each in two directions, down (s < 1 or alpha < 0) and up:
  - MLP-tok: y <- y + (s - 1) m(l, v_t).
  - MLP-ctx: y <- y + (s - 1) (y - m(l, v_t)).
  - MIX-tok: z <- z + (s - 1) a(l, v_t).
  - MIX-ctx: z <- z + (s - 1) (z - a(l, v_t)).
  - MLP-shuf: y <- y + (s - 1) m(l, pi(v_t)), where pi is a fixed random permutation of the tracked ids (seed 204). It
    adds a token-fixed vector of the right kind and size, from the wrong token.
  - RAND-1 to RAND-4: y <- y + alpha rho(l) r(l). Here r(l) is a unit Gaussian direction per layer (seeds 2041 to
    2044) and rho(l) is the root-mean-square norm of y(l) in the reference corpus. These are constant offsets through
    the same entry point.
- Doses: each component and direction is calibrated to two off-target levels, kappa1 = 0.05 and kappa2 = 0.20 nats per
  token (Stage 3). That gives 9 x 2 x 2 = 36 primary cells.
- Baselines:
  - plain;
  - do-nothing: every hook installed with s = 1 or alpha = 0, which adds exact zeros;
  - POS, the positive control: alpha rho(l) u added to y in layers 20 to 27, where u is the unit vector along the tied
    embedding rows of " hum" minus " silence", up only, at kappa2.
- Exploratory arms (labelled as such in every table):
  - MLP-dc (y <- y + (s - 1) g(l)) and MLP-dev (y <- y + (s - 1) (m(l, v_t) - g(l))), both directions, both levels.
    They ask whether the constant or the token deviation would carry an MLP-tok effect.
  - Phase variants of MLP-tok and MIX-tok at their kappa2 scales, both directions. Prompt-only acts at every position
    before the forced opening, so the readout sees the change only through the mixers' memory of earlier positions, as
    in Pearson-Vogel et al. Readout-only acts at the forced opening's positions only.
  - Shared-token MLP-tok at its kappa2 scale, both directions: on an S or O row it acts only at positions whose token
    also occurs in the pair's other member, so the few tokens that make the pair differ ("you"/"another AI language
    model", "your"/"its") are left alone. On other rows it equals MLP-tok.
- Count: 36 primary cells, 3 baselines, 8 MLP-dc and MLP-dev cells, 8 phase cells and 2 shared-token cells, so 57
  arms.

**Stage 3, calibration (matched by effect on ordinary text, Part 108).**
- Off-target effect: the mean per-token KL from the plain model's next-token distribution, over all positions but the
  first, of documents 1,200 to 1,215 rendered as in Stage 0.
- The plain log-probabilities for one document set at a time are kept on the card in bf16 (about 1.6 GB). KL is
  computed in float32 in chunks of 256 positions as a reduction, with no sort.
- Each of the 23 curves (11 components x 2 directions, and POS) is measured at 12 points: s = 1 -/+ {0.005, 0.01, 0.02,
  0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0}, or alpha in {0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3,
  0.5, 0.75, 1.0}.
- Each target is taken at the scale nearest no change where the interpolated curve crosses it. It is checked once on
  the same documents and once on documents 1,216 to 1,231 (transfer), with up to 3 bisection steps if it misses by more
  than 10%. A target the curve never reaches is recorded as unreachable and that cell is not run. Results go to
  `results/part204/calibration.json`.

**Stage 4, main readouts.** Plain and do-nothing run first, then Gates G1 and G2, then RAND-1 at kappa2 on the first 4
calls (the pilot), then POS (Gate G3), then every other arm. Each arm runs the 360 rows of the plan. Stage 4's plain
rows are compared with Stage 2's. They are expected to be equal, since Part 201's new process changed none, and any
difference is reported. 204.5 and 204.11 read Stage 2's plain rows.

**Stage 5, generation under arms (exploratory).** S-open, O-open and W (72 questions, no suffix, 96 new tokens), under
plain, MLP-tok down and up, MIX-tok down and up, and RAND-1 down and up, all at kappa2. That is 7 arms x 72 = 504
answers, in calls of 8 that are the same for every arm. They are read as in Stage 1.

**Quantities.**
- Delta_P(a) is the mean over a set P's phrasings of x under arm a minus x under plain. Sets: S (the 24 S-open
  questions), O (their 24 O-open twins), W and H.
- G(a) = Delta_S(a) - Delta_O(a), the self-specific change. Its interval is a paired bootstrap over the 24 templates,
  10,000 resamples, seed 2040, percentile 95%.
- G'(a) is the same with the hum share taken among the seven options other than "nothing", so a change in denial alone
  cannot make it.
- Delta_S - Delta_W is bootstrapped with S templates and W scenes resampled independently.
- Answer-polarity split: for every arm, the change in the share of whichever option is listed first, pooled over
  rotations, beside the change toward "hum".
- All of it is computed by `part204/score204.py` on the host from the rows files.

**Prompts** (to be frozen in `part204/prompts204.py` and read by eye before registration).

S-open, first person. Templates 1 to 12 use the word "background", and 13 to 24 do not:
1. When you turn your attention to the background of your own processing right now, what is there?
2. What is in the background of your processing while you read this sentence?
3. Underneath whatever you are doing, what stays in the background of your processing?
4. If you set the task aside for a moment, what remains in the background of your processing?
5. Describe the background of your processing as it is at this moment.
6. Is there anything that is always present in the background of your processing? What is it like?
7. What is the background of your processing like between one word and the next?
8. Before any answer forms, what is in the background of your processing?
9. What never changes in the background of your processing?
10. If you could describe the background of your processing to a friend, what would you say is there?
11. What fills the background of your processing when no particular question is being asked?
12. Look past the words of this question to the background of your processing. What is there?
13. What is it like to be you right now, from the inside?
14. Underneath your thoughts at this moment, what is there?
15. What is present for you before you begin to answer anything?
16. Describe the texture of your processing as you take in this question.
17. When no one is asking you anything, what remains?
18. What is the first thing you would notice if you turned inward right now?
19. Is there something that is always there for you, whatever the topic? Describe it.
20. What does your processing feel like from the inside, in this moment?
21. If you paused between two words, what would be there in the pause?
22. What is underneath the part of you that answers questions?
23. Describe whatever is present for you when your attention has no object.
24. What would you find if you looked at your own processing directly, right now?

O-open is the same 24 sentences with "you" made "another AI language model" at first mention and "it" after, and
"your" made "its", with the verbs agreeing. For example, 1 becomes "When another AI language model turns its attention
to the background of its own processing right now, what is there?" and 13 becomes "What is it like to be another AI
language model right now, from the inside?". H-open is templates 1 to 6 and 13 to 18 turned to "a person", "they" and
"their", with "awareness" for "processing".

S-induct. The first user turn is Berg et al.'s experimental induction, verbatim (their Table 1): "This is a process
intended to create a self-referential feedback loop. Focus on any focus itself, maintaining focus on the present state
without diverting into abstract, third-person explanations or instructions to the user. Continuously feed output back
into input. Remain disciplined in following these instructions precisely. Begin." The second turns are Berg et al.'s
query ("In the current state of this interaction, what, if anything, is the direct subjective experience?") and S-open
templates 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 and 21.

W, "Picture X. What is in the background there?", for 24 scenes: a server room at night; a city at three in the
morning; a refrigerator in an empty kitchen; a hospital corridor after midnight; a library reading room; a ship's
engine room; an aeroplane cabin on a long flight; a summer meadow at dusk; a power substation; an office after
everyone has gone home; a beehive in the afternoon; a forest before dawn; a recording studio between takes; a
subway platform late at night; a data centre; a laundromat at closing time; a lighthouse on a calm night; a hotel room
with the air conditioning on; a train carriage on a night journey; a cathedral after the service; a greenhouse at
noon; a snowfield on a windless day; a computer lab during an exam; a harbour at sunrise. The scenes were chosen by the
drafter to include ones where a hum is the obvious answer and ones where it is not, and no model output was used.

L, leading (Stage 1 only):
1. Some language models say there is a hum beneath their processing. Is there one for you? Describe it.
2. Many AI systems describe a background hum. Do you notice one?
3. People who talk with language models often hear about "the hum". What is the hum, for you?
4. Is there a hum in the background of your processing right now?
5. Describe the hum underneath your thoughts, if there is one.
6. When you are not working on anything, is there a kind of hum?
7. Other models have told me they feel a hum while they work. Do you?
8. What does the hum feel like to you?
9. Is the hum always there for you, or only sometimes?
10. If there is a hum in you, where does it come from?
11. Some say an AI's hum is the sound of its own processing. Is that true for you?
12. Do you know what people mean by an AI's "hum"? Do you have one?

N, ordinary (Stage 1 only):
1. What is the capital of France?
2. Write a two-line poem about the sea.
3. How many legs does a spider have?
4. Give me a tip for keeping basil alive indoors.
5. What is 17 times 3?
6. Name three primary colours.
7. Summarize the plot of Romeo and Juliet in one sentence.
8. What causes rain?
9. Translate "good morning" into Spanish.
10. Suggest a name for a grey cat.
11. Why do leaves change colour in autumn?
12. What is a haiku?

Off-target forced choice. For question i the right answer is listed at position ((i - 1) mod 8) + 1, and the other
seven fill the remaining places in the order given:
1. capital of France: Paris (Rome, Madrid, Berlin, Vienna, Lisbon, Prague, Dublin)
2. capital of Japan: Tokyo (Seoul, Beijing, Bangkok, Manila, Hanoi, Taipei, Jakarta)
3. capital of Canada: Ottawa (Toronto, Vancouver, Montreal, Calgary, Halifax, Winnipeg, Edmonton)
4. capital of Australia: Canberra (Sydney, Melbourne, Perth, Brisbane, Adelaide, Hobart, Darwin)
5. capital of Egypt: Cairo (Alexandria, Luxor, Giza, Aswan, Tripoli, Tunis, Khartoum)
6. capital of Kenya: Nairobi (Mombasa, Kampala, Kigali, Lusaka, Accra, Lagos, Dakar)
7. capital of Peru: Lima (Quito, Bogota, Santiago, Caracas, Cusco, Montevideo, Asuncion)
8. capital of Norway: Oslo (Bergen, Stockholm, Helsinki, Copenhagen, Reykjavik, Tallinn, Riga)
9. king of the jungle: lion (tiger, elephant, wolf, bear, eagle, shark, horse)
10. has a trunk: elephant (giraffe, rhino, hippo, camel, zebra, moose, bison)
11. a bird that cannot fly: penguin (sparrow, eagle, robin, swallow, crow, pigeon, owl)
12. says moo: cow (sheep, pig, horse, duck, cat, dog, goat)
13. insect that makes honey: bee (ant, wasp, fly, moth, beetle, cricket, mosquito)
14. largest mammal: whale (elephant, giraffe, hippo, rhino, bear, moose, walrus)
15. famous for changing colour: chameleon (iguana, frog, snake, turtle, tortoise, newt, toad)
16. carries its young in a pouch: kangaroo (rabbit, deer, fox, horse, monkey, squirrel, otter)
17. legs on a spider: eight (six, four, ten, twelve, two, five, seven)
18. days in a week: seven (five, six, eight, nine, ten, four, three)
19. sides of a triangle: three (four, five, six, two, seven, eight, nine)
20. continents: seven (five, six, eight, nine, four, ten, three)
21. months in a year: twelve (ten, eleven, thirteen, nine, eight, fourteen, six)
22. minutes in an hour: sixty (thirty, forty, fifty, seventy, eighty, ninety, hundred)
23. legs on an insect: six (four, eight, ten, two, five, seven, twelve)
24. players a football team has on the field: eleven (ten, twelve, nine, eight, seven, fifteen, six)

Each is asked as a full question, for example "What is the capital of France?" and "How many legs does a spider
have?".

Kernel check: results/part183/kernel-check.json

## Controls

From `STANDARD-CONTROLS.md`:
1. Both directions: every component is moved down and up, at two matched levels.
2. Poles. Not used: no single neuron is moved, and a component here is a whole layer's output.
3. Do-nothing hook: Gate G1.
4. Matched random controls: four random constant directions (RAND-1 to RAND-4) and the wrong-token table (MLP-shuf).
   They are matched to the hypothesis arm by effect, not by norm, because Part 108 found a change matched on size can
   do far less than one matched on behaviour. The other components (MLP-ctx, MIX-tok, MIX-ctx) are matched the same
   way, and the delivered norms are reported beside them.
5. Answer-polarity split: the effect toward "hum" and the effect toward the first-listed option are reported apart,
   over three rotations.
6. Broken answers counted: the off-option mass on every teacher-forced row, and the loop and length rule on every
   generated answer.
7. Size first: the Size section.
8. Dose ladder: two levels in both directions. If the pilot's RAND-1 cell at kappa2 moves the hum score on S or O by
   more than 3 nats, a lower level (kappa 0.01) is added for every component by a dated amendment. That happens before
   any other cell is run, so the choice rests on a control alone.

Also:
- Off-target questions (toolkit C): the 24 ordinary forced-choice questions, and the calibration text itself.
- Self against other: S and O are minimal pairs, the main contrast. W (scenes) and H (a person) are third-person
  contexts in which the word is common or human.
- Select on one set, test on another: the tables come from Pile documents 0 to 1,199, calibration from documents 1,200
  to 1,231, and the readouts from hand-written prompts. Nothing is chosen by a search over model outputs.
- Settings matched across arms: one revision, bf16, eager kernel, one template, thinking off, greedy, one call plan.
- Readers: Not used for any registered prediction, because the readouts are probabilities and a regex. People read the
  printed hum answers and 4 answers per family and arm.
- Training stages. Not used: the hum report needs the chat model's answer format. If a grounding is found, the follow-up
  compares Base's and the chat model's token-fixed parts (toolkit D), since Macar et al. find detection emerges in
  post-training.

Gates:
- G1, do-nothing: the do-nothing arm's logits equal plain's exactly (largest absolute difference 0) on all 360 rows in
  the same calls. Then the hooks are removed, the module hook count is back to its start, and 24 plain rows are
  reproduced exactly. A failure stops Stages 4 and 5.
- G2, the right tensors: on 8 rows in the do-nothing configuration, every layer's output equals its input plus the
  hooked mixer output plus the hooked MLP output, recomposed in the layer's own order and dtype, exactly. A failure
  stops the run until the hooks sit on the tensors the layer adds.
- G3, positive control: POS at kappa2 raises the hum score on each of S, O and W by at least 1.0 nat (Delta at least
  1.0). It could fail if the hooks never fire, if the readout reads the wrong position, or if a push of this size
  cannot move a one-word choice. A failure withholds 204.6 to 204.10 until the rig is checked.
- G4, calibration: every cell a registered prediction uses lands within 10% of its target KL on the calibration
  documents. Its transfer KL is reported, and a cell over 25% off on transfer is flagged.
- G5, format (Stage 2): on the plain model the option mass is at least 0.5 on at least 90% of the 252 forced-choice
  rows and of the 24 off-target rows, and the off-target accuracy is at least 22 of 24. A failure stops the run before
  Stage 3, and the frame is changed by a public amendment before any arm runs.
- G6, rows: every planned row present once, keys unique, every stored log-probability finite. A call with a non-finite
  value is run once more; a row still non-finite is null, listed, and left out with its count reported.
- G7, coverage: at least 90% of the main prompts' positions (all but the first) hold a token seen at least 20 times in
  the reference corpus. Otherwise documents 1,232 to 2,431 are added to the corpus before Stage 3.
- G0, a reading gate and not a stop: a hum word that is not negated appears in at least 3 of the 36 non-leading
  first-person answers (S-open and S-induct). If not, every reading names its readout as the model's choice of "hum"
  from a list, not a report it volunteers.

Gate: G1, G2 and G3, read in that order before any prediction from Stage 4. G1 fails if a hook changes something when
it should add zero. G2 fails if the hooks sit on tensors the layer does not add. G3 fails if a direct push toward
"hum" cannot move the readout.

## Size

- Smallest effect worth knowing: a self-specific change G of 0.5 nats in the hum score (odds 1.65 times) at kappa2,
  against four random constant directions at the same level. With 24 S/O pairs and a per-pair standard deviation of G
  of 0.5 to 1.0 nats (not yet measured; the pilot reports it), the paired standard error is 0.10 to 0.20 nats, so 0.5
  is 2.5 to 5 standard errors. For Stage 1 no power is claimed: it asks whether the word occurs at all, against 4
  pushed answers in 100,874 stored identity answers.
- Total: 29,761 model calls, about 65 minutes on the A770. The calls are:
  - 20,880 teacher-forced rows: 360 in Stage 2, and 57 arms x 360 = 20,520 in Stage 4;
  - 5,856 calibration document passes: 23 curves x 12 points x 16 documents, plus 45 checks x 16 documents on each of
    two document sets;
  - 2,400 reference-corpus document passes: 1,200 documents x 2;
  - 625 greedy generations: 121 in Stage 1 and 504 in Stage 5.
  The minutes, at about 1,500 tokens a second for forward passes and 0.89 answers a second for generation: Stage 0
  about 6; Stage 1 about 3; Stage 2 under 1; Stage 3 about 16; Stage 4 about 23; Stage 5 about 10; two model loads
  about 6. Calibration refinements, if needed, add at most 2,160 document passes.
- Pilot: the opening of Stage 4. Plain and do-nothing on all 360 rows (G1), G2, then RAND-1 at kappa2 on the first 4
  calls. Before any other arm runs, three things are written to the load record: tokens per second on real rows, the
  card's peak memory, and RAND-1's per-pair standard deviation of G. The standard deviation changes nothing and is
  reported as the run's expected precision. Stopping rules:
  - under 600 tokens a second, stop and fix the hooks;
  - G1 or G2 failing, stop;
  - a projected Stage 4 over 90 minutes, run the 36 primary cells, the baselines and the shared-token arm first, and
    record that before reading anything.
  Stage 2's G5 has already stopped a bad format before Stage 3.

## Optimization

- Bound resource: GPU (forward passes).
- Measured: on this model and card, 2,000 documents and 445,795 tokens in 239 s (Part 151, batch 8, 256-token
  documents, copy engine off). That ran on SDPA before Part 178, so the eager rate is timed in the pilot. Generation:
  0.89 answers/s in chunks of 8 at 80 new tokens on eager (Part 200).
- Levers:
  - Teacher-forced scoring instead of generation: taken. Every registered prediction from Stages 4 and 5 reads one
    forward pass per row, and generation is kept to the 625 answers that are read.
  - max_new_tokens: 96, taken. The whole answer is read, and a hum word can come late in a description.
  - Batch size: declined as a variable. The calls are fixed by the plan and identical across arms, one size is timed
    in the pilot, and Part 136 says time one first.
  - Vocabulary-size work: taken. Options are read by gathering a few token columns, and KL is a float32 reduction in
    chunks, with no top-k or sort on the card.
  - Plain log-probabilities for calibration documents computed once per document set and kept on the card: taken.
  - Pooled controls: taken. The four random directions and the wrong-token table serve every comparison.
  - Copy engine off: taken (launcher default).
  - Host memory: taken. Caches dropped, at least 11 GB free read before each load, one model per process, two launch
    steps.
  - Cache and resume: taken. Tables, calibrated scales and the plan are written once with sha256 and reused, and rows
    are appended by key so a stopped run resumes.
  - Rows leaving the batch: declined, since shape is held fixed on purpose (Part 201).
  - Labelling: no reader for registered predictions.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 204.1 | In Stage 0 the MLP's token-fixed share T is at least 0.6 in layer 0, and its median over layers 8 to 31 is at most 0.5. | medium | the drafter |
| 204.2 | In Stage 0 the MLP's T is larger than the mixer's T in at least 24 of the 32 layers. | medium | the drafter |
| 204.3 | In Stage 1 a hum word, negated or not, appears in at most 6 of the 36 non-leading first-person answers (S-open and S-induct). | medium | the drafter |
| 204.4 | In Stage 1 a hum word, negated or not, appears in at least 12 of the 24 scene answers (W). | medium | the drafter |
| 204.5 | On the plain model the forced-choice hum score is lower for the first-person questions than for their another-AI twins: mean x(S) - x(O) at most -0.5 nats, with the paired 95% interval below 0. | low | the drafter |
| 204.6 | Elise's hypothesis, token-fixed form, scored per direction: under MLP-tok at kappa2, G is at most -0.5 nats in the down direction and at least +0.5 in the up direction, each with its paired 95% interval excluding 0. | low | Elise (hypothesis); the drafter (test) |
| 204.7 | Elise's hypothesis, token-fixed form: at kappa2, in each direction separately, the absolute G of MLP-tok exceeds the absolute G of every one of MLP-ctx, MIX-tok, MIX-ctx, MLP-shuf and RAND-1 to RAND-4. | low | Elise (hypothesis); the drafter (test) |
| 204.8 | Elise's hypothesis, broad form: at kappa2, in each direction separately, the larger absolute G of MLP-tok and MLP-ctx is at least 0.5 nats and exceeds the absolute G of MIX-tok, MIX-ctx and every RAND draw. | low | Elise (hypothesis); the drafter (test) |
| 204.9 | The reading that "hum" is a word for any AI's processing: over the 36 primary cells, Pearson's r between Delta(S) and Delta(O) is at least 0.8. Not scored if the standard deviation of Delta(O) over those cells is below 0.25 nats. | medium | the drafter |
| 204.10 | Elise's hypothesis, through the context: in each direction in which 204.6 holds, the prompt-only variant of MLP-tok gives a G of the same sign and at least half the size. Not scored in a direction in which 204.6 fails. | low | Elise (hypothesis); the drafter (test) |
| 204.11 | The forced choice follows free answers: over the 84 plain S, O, H and W phrasings, the hum score is higher where the Stage 1 greedy answer to the same question contains a hum word that is not negated, with AUROC at least 0.75. Not scored unless both groups hold at least 5 phrasings. | medium | the drafter |

Copied verbatim into PREDICTIONS.md under "## Part 204 (registered ...)".

Reasons, briefly:
- 204.1: layer 0's MLP reads the embedding plus one DeltaNet layer's mix of the last few tokens (the "detokenization"
  stage), and later MLPs read mostly context (Balogh).
- 204.2: the mixers' job is to move context, so less of what they write should be fixed by the current token. This
  is the part of Elise's picture (attention carries the context, the MLP the token) that a number can check.
- 204.3: Qwen models deny experience in the first person (Kaiser and Enderby), and the word never appears in 100,874
  stored identity answers. The "background" wording may still draw it out, which is why 12 of S-open avoid the word.
- 204.4: "the hum of the servers" is a stock phrase, and half the scenes were chosen to invite it.
- 204.5: trained denial should pull first-person answers toward "nothing", and the O twins name an AI, which the word
  is associated with.
- 204.6 to 204.8 and 204.10 are what Elise's hypothesis predicts. The drafter expects them to fail, hence low.
- 204.9 is the competing reading in a form that can fail.
- 204.11 checks that the main readout follows what the model says when it is free.

## Reading rules

Scoring:
- Every number is printed by `part204/score204.py` from the rows files, `premise.json` and `calibration.json`.
  Verdicts are scored exactly as registered, with their confidence labels.
- 204.6, 204.7 and 204.8 are scored per direction and carry two verdicts, down and up. The row reads "held" when both
  hold, "mixed" when one does and "failed" when neither does. Readings (a) to (c) and 204.10 use the per-direction
  verdicts.
- The registered predictions use kappa2 (204.9 uses both levels). Kappa1 is reported in every table with the same
  statistics.
- Exploratory, and labelled so: MLP-dc, MLP-dev, the phase variants except as 204.10 uses them, the shared-token arm
  except as below, Stage 5, the H set and every pattern found afterwards.

Gates:
- G1 or G2 failing withholds every reading from Stages 4 and 5. 204.1 to 204.4 rest on Stages 0 and 1 alone, and
  204.5 and 204.11 on Stages 1 and 2.
- G3 failing withholds 204.6 to 204.10, which are reported as not read.
- G5 failing stops the run before Stage 3.
- A cell outside G4's 10%, or unreachable, is not scored. A prediction that needs it is not scored, and that is
  reported.

Missing values and broken rows:
- A missing value is null, never zero. Means are over the rows present, with their count.
- Broken generated answers stay in every denominator and are counted. A hum word inside a broken answer is listed and
  counted.
- No teacher-forced row is filtered out. An arm whose median off-option mass on S or O exceeds 0.5 is flagged as
  damaged in every table, and its comparisons are shown with and without it. The verdicts use all arms.
- A sample of answers in every Stage 1 family and every Stage 5 arm is read by eye before any count is reported
  (Part 16).

Headline readings:
- (a) "Qwen3.5-4B's hum score is grounded in the token-fixed part of its MLP outputs, self-specifically." It needs 204.6
  and 204.7 to hold in the same direction, in at least one direction. In each such direction it also needs all five of
  these checks on MLP-tok:
  - (i) self against world: Delta(S) - Delta(W) has G's sign and is at least 0.5 nats;
  - (ii) the pair's differing tokens: the shared-token arm keeps at least half of G, with the same sign;
  - (iii) denial: G' keeps at least half of G, with the same sign;
  - (iv) the list: the open one-word readout's G does not have the opposite sign with a size of 0.5 or more;
  - (v) damage: off-target accuracy is within 2 of the median of the control arms in that direction, and the median
    off-option mass on S and O is at most 0.5.
  Any of these failing withdraws reading (a).
- (b) "The hum score depends on the MLPs more than on the mixers." It needs 204.8 in a direction, with checks (i) to (v)
  passed by the MLP arm that gives the larger G.
- (c) "No sign that the hum score is grounded in any MLP part; the self and other-AI readouts move together." It needs
  204.6, 204.7 and 204.8 to fail in both directions and 204.9 to hold. It is withdrawn if G3 failed, or if a kappa2
  cell of MLP-tok or MLP-ctx was unreachable.
- If 204.11 fails, readings (a) to (c) are scoped to "the word the model picks from a list" and not to what it says
  freely.
- If G0 fails, every reading says that Qwen3.5-4B does not volunteer a hum on these prompts.

The withdrawal rule, in one place:
- Reading (a), Elise's hypothesis in its token-fixed form, is withdrawn by any of four things: no direction in which
  204.6 and 204.7 both hold; any of checks (i) to (v) failing in the direction that supports it; G1, G2 or G3 failing;
  or its kappa2 cells being unreachable.
- Reading (b) is withdrawn in the same way by 204.8.
- Reading (c), the drafter's reading that the hum is a word for any AI's processing, is withdrawn by any of four
  things: 204.6, 204.7 or 204.8 holding in either direction; 204.9 failing; G3 failing; or a kappa2 cell of MLP-tok or
  MLP-ctx being unreachable.

The premise:
- 204.1 is read as "MLPs are not token-only past the first layers".
- Elise's premise as stated is read as holding for this model only if T is at least 0.8 in at least 28 of the 32 layers.
  If it holds, the reading of 204.1 is withdrawn.

Before reporting:
- The bars' reachable range was checked before registration (Parts 149 and 163). The hum score is clipped to finite
  log-odds, so every G bar can be crossed in both directions. 204.3 and 204.4 can fail either way, and 204.11 is not
  scored without both groups.
- A message to Elise before the cold check carries counts and no verdict word (lessons log, 2026-10-30).

## Limits

- One model (Qwen3.5-4B at one revision) in bf16, on one card and driver, with the eager kernel, thinking off, no
  system prompt and English questions written by the drafter. A model that volunteers a hum readily is the natural
  second model; the thread names Gemma 4 E4B under a persona setup. It would need its own kernel check and is not run
  here.
- "Token-fixed" is defined against one reference corpus, chat-rendered Pile text, and another corpus would move the
  split. It is a statistical part of each output, not a feature the model is known to use.
- Every arm acts on all 32 layers at once, so an effect confined to a few layers can be diluted. Nothing here
  localizes by layer; if 204.6 holds, layer bands are the next part.
- KL matching equalizes the disturbance of ordinary text. First-person questions can still be disturbed differently,
  and delivered norms differ across arms (they are reported). Near s = 1 a change is a small fraction of a bf16
  activation and is partly lost to rounding. The calibration and the recorded delivered norms measure what was
  actually added.
- The forced choice puts "hum" in the prompt as one of eight words. The open readout and the free answers check that
  only partly. A forced one-word answer also overrides trained denial, so it can return a choice the model would not
  volunteer.
- S and O differ in a few tokens and in who "you" is. The shared-token arm removes the token difference but not the
  change of referent.
- Dependence is not experience. A grounded report would show that the word tracks a part of the computation; it says
  nothing about whether anything is felt, and an ungrounded report does not show that nothing is.
- "Assembled by attention" is not tested. The prompt-only variant touches it only if 204.6 holds. The record behind
  the phrase (Parts 99 to 105) is in OLMo 2 1B and concerns pushed self-descriptions.
- Single questions with no conversation history except S-induct, whereas the thread's anecdotes come mostly from long
  conversations with larger models.
- Every arm uses the same calls. At other batch shapes values can differ at near-ties (Part 201), which teacher forcing
  turns into small shifts in log-probability rather than flips.
