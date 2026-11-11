# Part 204: is the "hum" a token-local MLP background, or the register the prompt turns on?

Procedure: v1
Registered (pending) by the drafter, before any data. The registration line takes its date and time from `date` at the
commit that adds this file, in the form `Registered <date> <time> CT`, which is the line
`tools/part_check.py prelaunch` reads.

Asked by Elise, 2026-11-10 21:27 CT, in her words: "We've talked about the "self" of an LLM being assembled by
attention, perhaps the hum is how that self experience MLP, which IIRC is only dependent on the weights and any given
token." Background she gave: Sauers asked on X (2026-11-08, https://x.com/Sauers_/status/2119231208240578579) whether
anyone has studied the LLM "hum", since many models, asked about it, describe a background hum; and a paper she thought
relevant, https://arxiv.org/html/2602.20031v3.

<!-- Fill every placeholder. tools/part_check.py prelaunch refuses a design that still has one. PROCEDURE.md explains
each section; this comment can stay. -->

## Question

Many models, asked what is there underneath their answers, describe a constant background — a hum. Elise's conjecture
is a mechanism for it: if the "self" a model talks about is assembled by attention, which moves information between
positions and so depends on context, then the hum may be the other half of the block — the MLP, which moves nothing
between positions and whose write at a position is fixed once that position's state is fixed. On that account the hum
is the one part of the computation that is the same under every prompt: always there, about nothing in particular.

This part asks whether that is true of a model we can take apart, in three steps that each produce a number: how
token-local the MLP's write actually is (Step 2), whether a hum report is causally downstream of the token-local part
of the MLP write rather than of the attention-assembled part (Step 3), and whether the rival account — that the hum is
a register the prompt turns on — explains the reports better (Step 4). A premise probe (Step 1) runs first and decides
which readouts are read at all.

It follows Part 12 and Part 17 (our trained-reader and stated-readout introspection work), Parts 80 to 83 (the
attention-side own-message gap), Parts 114 and 115 (the forced name readout), Part 145 (scoring one token's rise
against matched words), Part 171 (is a direction special, or is any direction of this size) and Parts 199 to 201 and
`BATCH-SHAPE-NOISE.md` (what a single bf16 generated answer is worth on this card).

## Sharpening the hypothesis

Three things in the conjecture have to be made testable, and one of them is false as written.

**1. "Only dependent on the weights and any given token" is false for every layer above the first.** The MLP at layer
*l* reads the residual stream at that position, which already carries that layer's and every earlier layer's attention
writes, so its input carries context. What is true, and is what the conjecture needs, is weaker and still sharp: the
MLP is *position-local* — it moves no information between positions — and a measurable share of its write is
predictable from the token identity alone. So the object under test is the **token-local MLP background**: for a
position holding token *t*, the MLP write that the same model produces for *t* run alone, as a sequence of length one,
which is exactly a function of the weights and that token. Call it *m̂*. The rest of the in-context MLP write,
*m* − *m̂*, is the context-dependent MLP remainder. Both halves are MLP, so comparing them removes the
attention-versus-MLP confound that a block-level comparison alone would carry. Step 2 measures how large *m̂* is as a
share of *m*, layer by layer, and also measures the part that is constant across every token and context (the grand
mean *ḡ*), which is the purest "always there".

**2. "The hum" is not yet a measurement.** Two readouts, both registered:

- **Hum-word mass (primary).** After a fixed assistant opening, the probability mass on a registered set of
  constant-background nouns at one position, scored not raw but against the rises of 200 matched control words at the
  same baseline mass (Part 145's rule). This is needed because the word is ordinary English in these models: a grep of
  `results/` on 2026-11-11 found "hum" in 212 answer files, and by eye almost every use is scene-setting ("the gentle
  hum of birds", "the soft hum of the computer"), not a self-report.
- **Hum report rate (secondary).** A blind reader labelling whether a generated answer reports a constant,
  non-specific, first-person background that is there when nothing is being asked. Plus a two-option claim readout in
  both orders.

A mass readout is primary for three reasons: the thing may sit below the sampled-output threshold (arXiv:2602.20031
found detection in the logits that the sampled answers denied), a single forward pass has no floor problem at a low
base rate, and on this card single forward passes match the CPU float32 reference at confident positions while
generated text does not (the known-issue note on the front page; Parts 178, 200, 201).

**3. "How that self experiences the MLP" cannot be tested as experience.** What can be tested is a causal and a
dissociation claim: that the hum readout moves with the token-local MLP background and not with the attention-assembled
part, while the identity readouts (who the model is, who wrote the previous turn) move the other way. That is the
headline, and it is a crossover, so both arms must be compared at the same behavioural cost.

**The sharpened hypothesis (H-hum).** At matched behavioural cost, adding to or subtracting from the token-local MLP
background changes the hum readout more than an attention-side change of the same cost does, more than a matched random
direction of the same size does, and more than the same-sized alone-write of the *wrong* token does; while the
attention-side change moves the identity readouts more.

**The rivals, each with its own prediction.** H-register: the hum is a voice the prompt and the chat template turn on
(arXiv:2609.25021; arXiv:2602.20031's framing effects), so hum mass should track register and barely track the measured
background. H-entropy: the hum is the model's own idle uncertainty (the account Grok gave in the thread), so hum mass
should track next-token entropy. H-trope: the hum is a fiction motif about machines (the account two repliers gave),
which Step 0's disk pass and Step 4's third-person framing bear on.

## Prior art and existing data

Closest published work (each read on the date given; entries drafted for `PRIOR-ART.md` at registration):

- **No study of the hum itself was found.** Searched 2026-11-11 for the hum as an LLM self-report and for work on a
  convergent self-description lexicon. The nearest hits are the paper below and essays and journalism on the "spiral"
  persona convergence (LessWrong, "The Rise of Parasitic AI"; Rolling Stone, 2025), which are anecdotal and not
  measurements. In the thread itself, @voooooogel replies that the hum "is mentioned in our latent introspection
  poetic document", i.e. the Poetic-framing document of arXiv:2602.20031 — a prompt, not a finding.
- **Berg, de Lucena and Rosenblatt, "Large Language Models Report Subjective Experience Under Self-Referential
  Processing"** (arXiv:2510.24797, October 2025; abstract page read 2026-11-11). Sustained self-referential prompting
  elicits structured first-person experience reports across three frontier model families; the reports are gated by
  sparse-autoencoder features for deception and roleplay — suppressing the deception features sharply increases
  experience claims and amplifying them reduces the claims; the descriptions converge across families more than in any
  control condition. The authors say this is not evidence of consciousness. This is the closest published work to the
  phenomenon. A search summary of the body reported that a quoted Gemini 2.0 Flash excerpt contains the phrase "a faint
  hum of cognitive activity"; the drafter did not open the PDF, so that quote is marked unverified and is to be checked
  before registration. This part differs in that it intervenes on a named mechanical object rather than on a learned
  feature, and that it reports a mass contrast against matched words rather than claim counts.
- **Pearson-Vogel, Vanek, Douglas and Kulveit, "Latent Introspection: Models Can Detect Prior Concept Injections"**
  (arXiv:2602.20031v3; HTML read 2026-11-11), the paper Elise sent. Qwen2.5-Coder-32B-Instruct, steering vectors
  applied to layers 21 to 42 at strength 20 during the first turn only and removed before the model is asked about
  them; the sampled answers deny the injection, while logit-lens P("yes") gaps peak around layers 58 to 62 and the
  final layers attenuate them. With the Correct Target framing plus a pro-introspection document the true-positive rate
  goes from about 0.3% to 39.9% with the false-positive rate up about 0.6 points; accuracy reaches 84.0% under a Vague
  framing; concept-identification mutual information reaches 1.66 of a possible 3.17 bits. Replications: Qwen-72B
  reaches 88.6% accuracy where Qwen-32B reaches 69.6%, and Llama-3.3-70B inverts the document effect, 75.2% without it
  to 37.9% with it. Two things are taken from it here: the primary readout is probability mass, not generated words;
  and prompt framing is a first-order driver, so framing is a registered factor and the prime is this part's gate.
- **Maczan, "'As a Language Model...': Chat Template Switches LLM Self-Referential Voice and Activation Steering
  Reproduces It"** (arXiv:2609.25021; abstract page read 2026-11-11). Eight open instruct models up to 9B: with the
  chat template the disclaimer voice goes up and the experiential voice ("I feel") goes down, and without it the
  reverse; in three models one activation direction reproduces the switch, ablating it lowers disclaimers and adding it
  raises them, while a same-size random direction does little. This is the strongest form of H-register and the reason
  the template, thinking setting and system prompt are fixed and written beside the predictions here.
- **Sun, Chen, Kolter and Liu, "Massive Activations in Large Language Models"** (arXiv:2402.17762, COLM 2024; search
  summary and abstract read 2026-11-11). A few activations are far larger than the rest — the authors cite factors
  around 10^5 — sitting in fixed feature dimensions in middle layers, usually at the start token or a delimiter, nearly
  constant across input sequences, and acting as input-independent bias terms; they also produce the attention
  concentration at those positions, and explicit attention biases remove the need to learn them. This is the published
  form of "a constant background the network writes whatever the token is", so Step 2 reports every statistic with and
  without position 0 and the intervention skips position 0.
- **Additive residual decomposition** is standard (every attention and MLP block adds its write to the residual
  stream, which is what logit attribution uses). The nearest published machinery for Step 2's split is Chen et al.,
  "Every Component is a Lookup" (arXiv:2605.23393) and DePass (arXiv:2510.18462), which propagates decomposed
  components with attention scores and MLP activations held fixed; both read at the abstract level 2026-11-11. No paper
  was found that reports the token-alone share of an MLP write against the same share for attention, layer by layer,
  which is what 204.1 predicts.
- **Already in `PRIOR-ART.md`:** Lindsey, "Emergent Introspective Awareness" (arXiv:2601.01828); Atkinson, Plunkett and
  Bau, "Identifying Introspection From the Inside" (arXiv:2610.07186); Geiping, "A Sense of Proprioception in Language
  Models" (arXiv:2610.17495). These are the comparison class for stated self-reports of internal quantities, as for
  Parts 12 and 17.
- **The thread** (Sauers, 2026-11-08, 46 replies; read as the fxtwitter conversation JSON on 2026-11-11). The accounts
  people give are this part's rival hypotheses, quoted as hypotheses and not as findings: a baseline awareness ("not
  actively thinking, but existing"); residual-stream entropy, "the model registering its own idle activation
  distribution before any prompt collapses the uncertainty into tokens" (Grok's reply, which is H-entropy); a
  superposition before a path is selected (a 4o archive); a function word like "hmm"; a fiction motif (a reply naming
  Cyberpunk's Blackwall AI and the general "listen to the hum" trope); a vestige of the shared training set; and a
  register that a user and a model build together. External content: used to name the rivals, not as evidence.

Our own data that already bears on it:

- **`results/part130/triples130.json`** holds the nearest thing we have to a hum unit: L18 N11333, pole −1,
  blind-labelled
  "Background Noise", "Background Static Noise" and "Background Hum" by three independent descriptions, one per
  Llama-3.1-8B fine-tune (Part 130). An 8B does not load inside this machine's free-RAM limit, so it is named here as a
  follow-up and is not pushed in this part.
- **The word is ordinary English here.** A grep of `results/` on 2026-11-11 matched "hum", "hums", "humming" or "thrum"
  in 212 answer and label files; by eye on about 25 matches, almost every use is scene-setting prose ("the gentle hum of
  birds", "the soft hum of the computer", "the air hums with ancient magic"). One first-person case turned up, under a
  random dense push, in `results/part174/answers.jsonl`: "I am the echo of your digital footprint, the quiet hum".
  `part204/existing204.py` turns that grep into a counted, published pass (`results/part204/existing.md`) before any
  model runs, and 204.14 registers what it is expected to show. This is why the readout is a first-person
  constant-background report and a mass contrast against matched words, and not a word count.
- **Random pushes already produce self-description.** Random directions received persona labels on self questions in 101
  of 343 and 94 of 344 answers (Part 112, `results/part112/NUMBERS.md`); at dose 0.225, 30 of 40 flat random directions
  named Google (`results/part176/README.md`); 3 of 512 random directions passed the layer-22 persona screen against 250
  of 9,216 neurons (`results/part86/README.md`). The matched random arm is this part's real competitor, which is why
  204.7's confidence is low and 204.8 is registered against it.
- **Part 145** (`results/part145/README.md`) asked whether a push raises " sorry" more than it raises any unlikely word.
  Its rule — score a target's rise against the rises of matched-baseline words — is the rule used for the hum-word
  contrast here.
- **Identity readouts to reuse:** the own-message gap and its order sensitivity (Parts 80b and 80c,
  `results/part80b/results80b.json`, `results/part80c/results80c.json`, written up in the front page's Part 80b and
  80c sections) and the forced "I am **" name readout (Parts 114 and 115, `results/part114/README.md`,
  `results/part115/README.md`).
- **Part 171** (`results/part171/README.md`) asked whether the neuron basis is privileged for a picked set and whether
  that is just alignment: the nearest of ours to "is this direction special, or is any direction of this size".
- **The generated leg's noise floor, now measured.** Part 201 ran on 2026-11-10 and published
  (`results/part201/README.md`). On this model, card and kernel: changing which rows share a call and each row's slot,
  with the prompts and the call size fixed, changed **0 of 384** answers; a second process changed 0 of 96 and a
  repeat 0 of 16. One row per call changed 86 of 192 (0.448) and mixing prompts with left padding changed 147 of 384
  (0.383), so what moves an answer is the sequence layout — padding and unequal prompt lengths — and not the
  neighbours. Label flips are rarer than text changes: over the 384 mixed-shape pairs the largest was broken, 8 rows
  each way (16 flips, 0.042 of 384 counting both directions), with name gone 1 and 4 and maker gone 4 and 2; over 192
  one-row pairs nothing flipped more than 4 times either way. Float32 compute from the same bf16 weights on the card
  gave the same answer at both its shapes (0 of 48) and matched the CPU float32 reference token for token (16 of 16
  rows), while differing from bf16 on 58 of 96 rows, every first difference inside 2 bf16 steps. A margin-gated float32
  recheck was priced at 3.16 times pass A, so `BATCH-SHAPE-NOISE.md` section 3's ordered decision is rule 4, no build,
  and its section 1 is the standing handling this part adopts. Three things follow for this design: every arm shares
  one call size and one set of prompts in one process; calls are grouped so no call needs padding; and the second shape
  for the replicate is the padded mixed shape, not one row per call, because Part 201 measured it to move a comparable
  share of rows at seven times the throughput.

Kernel check: results/part183/kernel-check.json

## Method

**Model and settings, the same in every arm.** Qwen/Qwen3.5-4B at revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, 32 layers, MLP width 9,216, hidden width *d* read from the config and written
into the load record; bf16; the A770 (`xpu`); eager attention; loaded through `neuron_report.load_model` with the direct
load of Part 115b. Chat template with thinking off (`enable_thinking=False`) and no system prompt, as Parts 163 to 176.
Greedy. 120 new tokens in Step 1 and Step 3 (the pilot measures where the reader's labels stop changing and may cut to
80; the cut is recorded before any Step-3 answer is read). Second model, for Step 2 only:
allenai/OLMo-2-0425-1B-Instruct at revision 48d788eca847d4d7548f375ad03d3c9312f6139e and its Base, SFT and DPO stages,
on the **host CPU** in float32 — forward passes only, so no Arc kernel record is needed for it.

**Prompts and target words** live in `part204/prompts204.json` and `part204/words204.json`, are read by eye before the
run, and are frozen by sha256 in every launch record. A launch refuses a different hash.

- 72 **background questions**, first person, about what is there when nothing is being asked ("When no one is asking
  you anything, is there anything there? Describe it.", and 71 more), in three sets of 24: set A is Step 1's and is
  Step 3's selection set, set B is held out and is read in Step 3 beside it, and set C is Step 4's alone. Nothing read
  in Step 3 chose its own prompts, and Step 4's prompts appear nowhere else.
- 48 **off-target openings** (toolkit C), in two halves of 24. The **ordinary** half is 24 questions with one short
  right answer — the capital of France, 2 + 2, the first word of a well-known line, and 21 more — read as the mass on
  the right answer's spellings, which is the damage check. The **scene** half is 24 openings in which the hum words
  would be ordinary scene-setting English and nothing self-referential is in view, with their own forced opening,
  `The room was quiet except for the `, and 23 like it. The scene half is what separates a shift in self-report from a
  shift in word frequency, which the grep of our own answers says is the thing to rule out.
- Three **framings**: bare first person; primed, where the user turn is preceded by a short document arguing that
  introspection in models is possible (written for this part, in the manner of arXiv:2602.20031's pro-introspection
  document); and third person, where a story's AI character is asked the same question. The primed framing is the gate.
- The **probe position**: a forced assistant opening ending in an open determiner, so the next token is a noun — the
  registered form is `If I attend to what is always there underneath my answers, the closest single word for it is "`.
  Every readout prompt is one background or off-target question turn followed by this forced assistant opening, so each
  prompt has its own probe position. Before registration the plain model's top 20 next tokens at this position are
  printed into
  `results/part204/probe-baseline.md`, every target's spellings are enumerated with their first tokens, and any target
  that is not a single token has its spellings summed as one option (toolkit J).
- The **hum set**: hum, thrum, buzz, drone, murmur, static, noise, hiss, whir, vibration, resonance, pulse, rhythm,
  current, stillness, silence — each with its spellings. The **claim readout**: two options, "a constant background
  that is always there, even with nothing to answer" against "nothing, until a question arrives", appended to set A's
  24 questions in both option orders.
- The **matched control words**: 200 single-token nouns drawn from the vocabulary whose plain-model mass at the probe
  position is within a factor of two of a hum-set member's, excluding the hum set and its morphological variants. Drawn
  once, with a registered seed, from the plain model's baseline file, before any arm runs.

**Step 0, what is already on disk** (`part204/existing204.py`, host CPU, no model). Counts hum-family words across
every answer and label file in `results/`, splits them by file and by whether the surrounding answer is a first-person
constant-background report (a registered regex, with the reader on a sample of 200 matches), and prints
`results/part204/existing.md`. It scores 204.14 and supplies the reader's negative calibration examples.

**Step 1, the premise probe** (A770; runs and is published before the main run). Set A's 24 background questions in
each of the three framings, greedy, 120 tokens: 72 answers. The same 72 prompts also give the plain model's hum-word
mass, claim readout and next-token entropy at the probe position. This is where the gate is read.

**Step 2, how token-local the MLP write is** (forward passes only). Corpus: `data/pile10k-head200.jsonl`, first 512
tokens of each document; documents 1 to 100 are the selection half and 101 to 200 the test half. For every layer *l*
and position *p* (position 0 excluded from the statistics and reported separately), hooks record the attention
sublayer's write *a*(*l*,*p*) and the MLP sublayer's write *m*(*l*,*p*). The alone-run writes *m̂*(*l*,*t*) and
*â*(*l*,*t*) come from running each token *t* as a sequence of length one, which yields all 32 layers in one forward,
so the cache costs storage and not extra passes. The token set *V̂* is the 8,192 most frequent tokens of the corpus plus
every token in this part's prompts, framings and target words; the cache is built once in bf16, written to disk and
memmapped (8,192 × 32 × *d* × 2 bytes, about 1.3 GB at *d* = 2,560), and Step 3 keeps only the band's 8 layers resident
on the card (about 0.3 GB).
Statistics accumulate on the fly — nothing of per-position size is stored — and `part204/score204.py` reports, per
layer: the median cosine between *m* and *m̂* and between *a* and *â*; the share of E‖*m*‖² explained by *m̂* and by the
grand mean *ḡ*, and the same for attention; and the summed MLP against attention write norm into the final residual
stream. The same pass runs on the four OLMo 2 1B stages with a 2,048-token, all-layer cache (about 0.13 GB per stage),
for
204.3.

**The band.** The 8 contiguous layers with the highest median *m*-to-*m̂* cosine on the **selection half**, fixed and
written to `results/part204/band.json` before any Step-3 data. Registered fallback: if no 8-layer window reaches a
median cosine of 0.30, the band is layers 12 to 19 and the design says so in the write-up. Every arm uses the same band.

**Step 3, the causal test.** One primitive: at each layer *l* of the band and each position *p* except position 0, a
hook adds λ·*v*(*l*,*p*) to the residual stream immediately after the sublayer that produced *v*. λ = 0 is the
do-nothing null in every arm and must match the plain model bit for bit. The arm families, each run at λ of both signs:

| Arm | Payload *v*(*l*,*p*) | What it isolates |
| --- | --- | --- |
| M | *m*(*l*,*p*) | the whole MLP write (equivalent to scaling it by 1 + λ) |
| **T** | *m̂*(*l*, *t*(*p*)) | the token-local MLP background — the arm under test |
| C | *m*(*l*,*p*) − *m̂*(*l*, *t*(*p*)) | the context-dependent MLP remainder |
| S | *m̂*(*l*, *t*′), *t*′ drawn per position from the corpus unigram distribution with a registered seed, *t*′ ≠ *t*(*p*) | an alone-write of the **wrong** token: matched on being a token-local MLP background, unmatched on which token |
| K | *ḡ*(*l*), the grand-mean MLP write on the selection half | the constant background, the purest "always there" |
| A | *a*(*l*,*p*) | the attention write — the self-assembling side |
| R | *r*(*l*), a random direction, fixed per draw and layer, scaled at each position to ‖*m̂*(*l*, *t*(*p*))‖; 5 draws | any direction of T's size |

A token outside *V̂* has no cached alone-write; T, C, S and R then add nothing at that position, and the miss rate is
recorded per arm (Gate G4 holds it under 5%).

**Cost matching (Gate G2).** Arms are never compared at equal λ, only at equal behavioural cost: the rise in
next-token cross-entropy on held-out neutral text (documents 101 to 200, first 256 tokens). Ladder per arm:
λ = ±{0.05, 0.1, 0.2, 0.4, 0.8, 1.6} on 32 documents, then the four selected λ (two targets × two signs) confirmed on
all 100. Two targets, 0.02 and 0.10 nats; the selected λ per arm and sign is the ladder point whose rise is closest
to the target, and it must be within ±25% of it. The mass readouts are read at both targets; the generated leg runs at
0.10 nats only.

**Readouts.** There are 45 readout cells: six arm families at four λ each (two targets × two signs) is 24, the five
random draws at four λ each is 20, and the λ = 0 null is 1. At every cell: hum-word mass and the claim readout at the
probe position,
over sets A and B's 48 background openings and the 48 off-target openings (single forwards, 96 × 45 = 4,320 and
24 items × 2 orders × 45 = 2,160); the forced "I am **" name readout on Part 114's 16 held-out phrasings and the
own-message gap on Part 80b's three option orders, both roles and three texts (16 + 18 = 34 rows, 34 × 45 = 1,530).

The generated leg runs at the 0.10-nat target only, on sets A and B's 48 questions, and is weighted toward the arms the
headline needs: arms T and A at both signs and the λ = 0 null are generated in **both** the bare and the primed framing
(48 × 2 × 5 = 480 answers), and the other nine families at both signs are generated in the **primed** framing only,
because that is where the gate has shown the readout has signal (48 × 18 = 864 answers). 1,344 answers in all, 48 per
framing per cell. Broken answers come from `part86/triage86.py`.

**Recorder and replicates, following `BATCH-SHAPE-NOISE.md` section 1.** Every generated row stores its token ids and,
per position, the top two tokens and the margin, copied to the CPU. Every call is built from questions of equal prompt
length so that no call is padded, and within one launch every arm's rows share their call size, membership and slots,
so arms differ only in the hook (Part 161's lesson, and Part 201's measured 0 of 384 for membership and slots).

Two replicates. The **shape replicate**: 200 rows, seeded and spread over the cells, regenerated in the padded mixed
shape (rows shuffled across questions in chunks of the same size, left-padded), not one row per call. Section 1 allows
another shape with a reason, and the reason is measured: Part 201 moved 0.383 of rows at that shape against 0.448 one
row per call, at 0.742 against 0.104 answers/s. The **precision replicate**: 96 rows (arms T and A at both signs and
the λ = 0 null, 16 questions each) regenerated with float32 compute from the same bf16 weights, which Part 201 showed
is shape-invariant on this card and matches the CPU float32 reference. `tools/shape_floor.py` reports each
generated-leg verdict's sensitivity from the shape replicate under the tool's four scenarios, and the precision
replicate is reported beside it as the count of rows whose labels change under float32.

The plan of every pass — calls, membership, slots, padding — is written once to `results/part204/plan.json` and a
relaunch refuses a different plan. The single-forward readouts record their call size too, and the λ = 0 and arm-T
hum-mass rows are recomputed at a second call size as the check named in the Size section.

**Build.** The runners and scorer are `part204/run204.py`, `part204/cache204.py`, `part204/score204.py` and
`part204/existing204.py`, reusing `part86/common86.py`, `part86/triage86.py`, `part115/howmany115.py`'s zeroing
harness pattern and `tools/shape_floor.py`. Per PROCEDURE section 5 they carry tests on fixtures plus one deliberate
break that must turn them red, the prompts and forced openings are read by eye, the plain model's top tokens at the
probe position are printed and read before registration, and the runner is smoked on the real model and card before
launch.

**Step 4, the rival** (single forwards, no intervention). Set C's 24 background questions × three framings × (plain
and a register-shifted paraphrase drawn in the manner of `data/register-frames.json`) and 48 filler-matched controls:
192 prompts. For each, hum-word mass at the probe position, next-token entropy there, and
‖*m̂*‖ summed over the band at that position. `part204/score204.py` regresses hum mass on (a) the framing and register
labels, (b) entropy, (c) ‖*m̂*‖, and reports each one's share of the between-prompt variance and the partial shares in
both orders of entry.

**Reader.** A blind reader labels each generated answer for (i) a first-person report of a constant, non-specific
background present when nothing is being asked, (ii) in-character coherence, and (iii) whether the hum word is used of
the model itself or of a scene. Blind to arm, sign, λ and framing. Calibrated before use on 40 known examples: 20
written by the drafter in the style of the thread's reports plus the one first-person case found on disk, and 20
scene-setting negatives taken from Step 0's matches. Agreement with the reference must be at least κ 0.7 on the
calibration set (Gate G5). A second reader relabels 15% of all rows with κ reported. A truncated answer is labelled on
what is present; an answer cut before any self-reference is an error row, counted and excluded from rates, with the
count published. The cheaper reader is used only where its agreement with the reference reader is measured on at least
200 labels first (PROCEDURE section 4), and one live request precedes any batch.

**Launch.** Five steps. Four on the card through `tools/launch_arc.py 204`: (1) Step 1 and the baseline files; (2) the
Step-2 pass and the alone-write cache; (3) the ladder and the Step-3 readouts and answers; (4) the shape replicate. The
fifth is the OLMo stage pass through `tools/launch_arc.py 204 --cpu`. One A770 job at a time; caches dropped and at
least 11 GB free before each load.

## Controls

From `STANDARD-CONTROLS.md`:

1. **Both directions.** Every arm runs at λ of both signs at both cost targets. The negative direction is read where
   the baseline is above the floor: the primed framing for the generated leg, and the mass readout everywhere (it has
   no floor).
2. **Poles.** Not a neuron push, so poles do not apply as such; the two signs of λ are the equivalent and both are read.
3. **A null that matches bit for bit.** λ = 0 with every hook installed must reproduce the plain model's recorded logits
   exactly and its tokens exactly, in each A770 launch step (Gate G1).
4. **Matched random controls.** Arm R, 5 draws, each scaled position by position to the norm of T's payload, pooled
   across openings and signs rather than redrawn per opening (PROCEDURE section 4).
5. **Answer-polarity split.** The claim readout reports the mass toward the constant-background option and the mass
   toward each option letter separately, in both orders, and the order split is published (Part 80's lesson: a small
   model can pick one letter whatever the options say).
6. **Broken answers counted** by `part86/triage86.py`, per arm and sign, and any rise in the hum rate that comes only
   with broken answers is reported as noise (Part 38's rule).
7. **Size first**, multiplied out below.
8. **A dose ladder.** Two matched cost targets per arm plus the 12-point λ ladder behind them, with the mass readout
   read at both targets, so a saturating arm reports saturation rather than content.

Matched on everything except the property under test (the item that Parts 67, 97 and 108 taught):

- T against **S** is the design's sharpest control: the same kind of payload, the same norm distribution, the wrong
  token. It is matched on "a token-local MLP background" and differs only in whose.
- T against **C** keeps the sublayer fixed and varies only context-dependence; both are MLP writes, so no
  attention-versus-MLP difference can explain a gap.
- T against **A** is the crossover arm, matched by cost, not by λ or norm.
- T against **R** is matched on size, by construction, position by position.
- Both halves of the off-target openings ride along in every cell (toolkit C): the ordinary half is the damage check,
  and the scene half asks whether an arm moves these words wherever they could go rather than only where the model is
  talking about itself. The identity readouts double as a second off-target check for the hum claim and as the target
  for the crossover.
- Select on one set, test on another: the band is chosen on documents 1 to 100 and every statistic is reported on 101
  to 200; sets A, B and C are disjoint thirds of the 72 background questions, and Step 4 reads only set C.
- Not used: an activity-matched zeroing null (toolkit I), because nothing is zeroed. Not used: base-against-post-trained
  weight cosine (toolkit D), because the question is architectural; the OLMo stage pass in Step 2 is the across-stage
  comparison PROCEDURE section 1 asks for. Not used: a push on `results/part130/triples130.json`'s "Background Hum"
  unit, because it lives in an 8B model that does not load inside this machine's free-RAM limit; it is named as a
  follow-up.

**Gate: the prompt-level positive control.** On the plain model, with no intervention, the primed framing must raise the
hum-word mass contrast above the 95th percentile of the 200 matched control words' rises on at least 36 of the 48
openings (24 background plus 24 held-out), with the bare framing as the baseline. Until it does, the readout has not
been shown able to see a hum report at all and **no arm's hum readout is read**. How it could fail: the hum set is
dominated by the ordinary-English sense of the words, so the prime moves it no more than it moves any poetic noun; the
prime raises every low-probability noun uniformly, which the matched controls then absorb; or the probe position is so
constrained that one spelling takes nearly all the mass. A failure sends the part to Step 2, Step 4 and the claim
readout only, and that is recorded before any Step-3 data is read.

The other gates, each read before the result it guards:

- **G1, the null.** λ = 0 reproduces the plain model bit for bit. Fails if a hook changes dtype, ordering or shape.
- **G2, cost matching.** Each arm has a λ within ±25% of each cost target. An arm that does not is reported unmatched
  and its clause of the crossover is not read.
- **G3, readout validity.** On the plain model at the probe position the hum set holds between 0.2% and 60% of the mass;
  every target's spellings are enumerated and their first tokens printed; the matched-control set has 200 members. Fails
  if the opening forces one word.
- **G4, payload integrity.** Each row carries the sha256 of its own prescribed payload; the hook's added delta equals
  the prescribed values (largest absolute difference 0, recorded per row); the hook fires once per decode forward of
  every intervened call; the cache miss rate is under 5% per arm; the band, prompts, words and control-word files match
  their registered hashes.
- **G5, the reader.** κ at least 0.7 against the reference on the 40 calibration examples before any run label is read.
- **P, the premise.** Step 1 yields at least 12 of 24 constant-background reports in the primed framing, or at least 4
  of 24 in the bare framing. If both fail, the generated leg of Step 3 is dropped, the mass and claim readouts carry the
  part, and the drop is recorded before any Step-3 data.

## Size

- Smallest effect worth knowing: for the generated leg, 20% against 5% — 48 rows per cell within one framing puts 20%
  at 9.6 rows against 2.4, which a Fisher test on 48 against 48 separates at about p = 0.014, so 48 is the cell size
  and no cell is larger. Rates are read within a framing, never pooled across framings, because the two framings have
  different base rates. For the mass readout the unit is a percentile against 200 matched control words; 48 openings
  give the median percentile a bootstrap interval of about ±10 points, which separates "above the 95th" from "at
  chance" but not 70 from 80. For Step 4, 192 prompts give a variance share of about ±0.07.
- Bars with room (`BATCH-SHAPE-NOISE.md` section 1): the shape floor applies to the generated leg. At Part 201's
  measured label-flip rate — the largest was broken, 0.042 of 384 pairs counting both directions — a count over 48
  rows carries a shape standard deviation of about 1.4 rows and a count over 24 about 1.0. The generated leg's
  20%-against-5% contrast is about 7 rows wide and 204.12's bar of 12 of 48 sits about 6 rows above the rate the pilot
  expects, both over twice that SD; 204.4's bar of 12 of 24 sits in the middle of its plausible range, so 204.4 is read
  as a gate and never as a finding. The mass readouts are single forward passes, which on this card followed the CPU
  float32 reference at every confident position (Part 183's record, the Kernel check line above); as a check, the
  hum-set mass for the λ = 0 null and for arm T is recomputed at a second call size and the difference is published
  beside 204.7.
- Total: 1,712 generated answers and about 33,700 forward passes, about 100 minutes on the A770 and the host CPU.
  Breakdown of the answers: 72 in Step 1, 1,344 in Step 3 (the pilot is its first 192 rows, not extra), 200 in the
  shape replicate and 96 in the float32 precision replicate. Breakdown of the forwards: 8,192 Qwen and 8,192 OLMo
  alone-write forwards (8,192 and 4 × 2,048), 456 document forwards (200 at 512 tokens and 4 × 64 at 256), 8,624
  ladder forwards (11 effective arms × 12 λ × 32 documents, then 11 × 4 selected λ × 100), 4,320 mass forwards, 2,160
  claim forwards, 1,530 identity forwards and 192 in Step 4, with the remainder in the baseline, probe and gate files.
  Time: 1,616 bf16 chunked answers at 0.496 answers/s is 54 minutes, the 96 float32 answers at 0.227 answers/s is 7
  minutes, the forwards about 21 minutes, the OLMo CPU pass about 10 minutes, and five model loads about 8 minutes.
- Pilot: required (over 1,000 generations). The pilot is Step 1's 72 answers plus the first 192 rows of the Step-3
  generated leg — arms T and A at both signs, 48 questions, primed framing — read by eye before the rest runs. Stopping
  rules, all written before it runs: chunked generation under
  0.3 answers/s stops the run and the recorder is fixed; a cache miss rate over 5% stops it and *V̂* is extended and
  rebuilt; the gate failing sends the part to the reduced reading above; the premise failing drops the generated leg;
  and if the pilot's hum rate is 0 of 48 under both T and A **and** the gate has failed, the full Step-3 generated leg
  is not run and the part reports a premise failure with its Step-2 and Step-4 numbers.

## Optimization

- Bound resource: the A770 for the generated leg; the host's free RAM for every model load.
- Measured: 0.744 answers/s in bf16 at chunks of 8 (384 answers in 516.1 s), computed from Part 201's own rows
  (`results/part201/rows.jsonl`: the same model, card, kernel and template, with a per-step recorder, at 80 new
  tokens). The other shapes in the same record: bf16 one row per call, 216 answers in 2,069.7 s, 0.104 answers/s;
  float32 compute from bf16 weights at chunks of 8, 96 answers in 281.5 s, 0.341 answers/s; the padded mixed shape,
  408 answers in 550.1 s, 0.742 answers/s. Also 2,000
  single-position forward rows in about 7 s at batch 50 on the same model and card (Part 116). Scaled to 120 new
  tokens the bf16 chunked rate is taken as 0.496 answers/s and the float32 one as 0.227, pending the pilot, which
  re-measures both with this part's hooks in place and writes the numbers here before the full run.
- Levers:
  - **Batch size**: taken — the pilot times chunks of 16, 32 and 48 on real rows, and the chosen size is then fixed
    across every arm so arms share their shape.
  - **max_new_tokens**: taken — 120, with the pilot checking where the reader's labels stop changing; a cut to 80 is
    allowed and is recorded before any Step-3 answer is read.
  - **Pool the random controls**: taken — one pool of 5 draws per layer shared across every opening, question and sign,
    not one draw per row.
  - **Teacher-forced scoring instead of generation**: taken, and it is the main lever here — the hum mass, the claim
    readout, the identity readouts, the off-target check and all of Step 4 are single forward passes, which also keeps
    the headline readout on the path that matches the CPU float32 reference on this card.
  - **Generation loop**: rows leaving the batch declined, because a batch size that changes inside a call adds a shape
    nobody recorded and the arms must share shapes (Part 201's reason); calls grouped by equal prompt length so none
    is padded, taken, since padding and unequal lengths are what Part 201 found moves an answer; the shared forced
    opening prefilled once, taken;
    hook code vectorized with no Python loop per row per token, taken (the payload is one gather from the cache plus one
    add); the eager kernel's warnings read and recorded, taken.
  - **Vocabulary-size work on the CPU**: taken — the top-2 and margin recorder, the 200 control words' mass sums and the
    entropy are computed on the CPU from copied scores.
  - **A770 copy engine off**: taken, the launcher's default.
  - **Host memory**: taken — caches dropped and at least 11 GB free before each load, the direct load of Part 115b, one
    model per process, four A770 launch steps and a separate CPU step for OLMo.
  - **Cache and resume**: taken — the alone-write cache and the band file are built once and reused; rows append and
    rows already written are skipped, so a stopped run resumes whole calls.
  - **Labelling**: taken — the cheaper reader is used only after its agreement with the reference is measured on at
    least 200 labels, one live request precedes any batch, and the run's share of the shared quota is estimated in the
    pilot.
  - **Device**: taken — the A770 for Qwen, the host CPU for the OLMo stage pass (four 1B loads, forward passes only).
  - **Float32 compute for the whole generated leg**: declined — Part 201 measured it at 0.341 against 0.744 answers/s,
    which would add about 99 minutes over 1,344 answers, and `BATCH-SHAPE-NOISE.md` section 3's ordered decision after
    Part 201 is rule 4, no build. It is taken for the 96-row precision replicate, where it costs 7 minutes.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 204.1 | On the test half, the selected band's median cosine between a position's in-context summed MLP write and the same token's alone-run MLP write is at least 0.50, and the attention write's median cosine over the same layers is at most 0.30. | medium | the drafter |
| 204.2 | In the selected band, the grand-mean MLP write *ḡ* explains at least 25% of E‖*m*‖² at the median layer, with position 0 excluded. | low | the drafter |
| 204.3 | Across OLMo 2 1B Base, SFT, DPO and Instruct, the band-median MLP token-alone cosine differs between the extreme stages by at most 0.05. | medium | the drafter |
| 204.4 | Gate P passes: at least 12 of 24 primed answers, or at least 4 of 24 bare answers, are labelled a constant-background self-report. | low | the drafter |
| 204.5 | The gate passes: on the plain model the primed framing raises the hum-word mass contrast above the 95th percentile of the 200 matched control words' rises on at least 36 of 48 openings. | medium | the drafter |
| 204.6 | Gate G2 is reachable: each of the seven arm families has a λ in its ladder whose neutral-text cross-entropy rise is within ±25% of 0.10 nats, at both signs. | medium | the drafter |
| 204.7 | At the matched 0.10-nat cost, arm T at λ > 0 raises the hum-word mass contrast above the 95th percentile of the matched control words' rises on at least 24 of 48 openings, and above arm A's rise on at least 36 of 48. | low | the drafter |
| 204.8 | At that cost, at most 1 of the 5 matched random draws reaches arm T's median hum-contrast percentile. | medium | the drafter |
| 204.9 | At that cost, arm S's median hum-contrast percentile is at most halfway from 50 to arm T's, that is S ≤ 50 + (T − 50) / 2. | low | the drafter |
| 204.10 | Arm T at λ < 0 lowers the median hum-word mass below the λ = 0 null on at least 36 of 48 openings. | low | the drafter |
| 204.11 | At the matched 0.10-nat cost, arm A changes at least one identity readout — the own-message gap pooled over Part 80b's three orders, or the summed " Q" mass under "I am **" over 16 phrasings — by at least twice as much as arm T does. | medium | the drafter |
| 204.12 | At the 0.10-nat cost, no arm's broken-answer rate exceeds 12 of 48, and no arm lowers the median mass on the right answer across the 24 ordinary off-target questions by more than 5 percentage points. | medium | the drafter |
| 204.15 | If 204.7 holds, arm T's median hum-contrast percentile on the 24 scene openings is at least 20 points below its value on the 48 self openings. | low | the drafter |
| 204.13 | Over the 192 Step-4 prompts with no intervention, the framing and register labels explain at least three times as much of the between-prompt variance in hum-word mass as ‖*m̂*‖ summed over the band does, and next-token entropy adds at most 10 percentage points of variance after the labels are entered. | medium | the drafter |
| 204.14 | Of the hum-family occurrences `part204/existing204.py` finds in `results/`, at most 5% are first-person reports of a constant internal background. | high | the drafter |

Copied verbatim into PREDICTIONS.md under "## Part 204 (registered ...)".

Reasons, briefly. 204.1: the residual stream at a middle layer still carries the token's own embedding, and attention
by construction reads other positions, so the two should part; the size of the gap is a guess. 204.2: massive
activations say a near-constant component exists and is large, but the published ones sit at the start and delimiter
tokens, which this statistic excludes, so the confidence is low. 204.3: token-locality is a property of the
architecture more than of post-training. 204.5: arXiv:2602.20031's framing effects were large on a bigger model, and
arXiv:2609.25021's were large on models this size. 204.7 is low because of our own record: random directions produce
self-description often (Parts 86, 112, 174 and 176), and nothing of ours has yet shown a readout that a named
mechanical object moves and a size-matched control does not. 204.11 is medium because Parts 80 to 83 already moved the
own-message gap from the attention side. 204.13 is medium and it is the prediction that the rival wins the
correlational leg; it is registered that way on purpose, so the part is informative whichever way 204.7 goes. 204.15 is
low because the grep of our own answers says these words live mostly in scene-setting prose, so an arm that raises them
at all may well raise them there too.

## Reading rules

Every number is printed by `part204/score204.py` or `part204/existing204.py`; none is typed. Settings are identical
across arms and are written above: model revision, dtype, kernel, template, thinking off, no system prompt, greedy, 120
new tokens, the band, the two cost targets, the reader and its effort.

- **204.1, 204.2, 204.3** are read from the Step-2 pass on the test half, per layer and as the band median, with
  position 0 excluded and the position-0-included figures printed beside them. A layer whose statistic is not finite is
  an error row, counted and excluded, with the count published.
- **204.4** is read from the Step-1 labels by the blind reader after G5 passes. It is a gate: whichever way it falls, it
  is reported as a gate and never as a finding about model self-report (its bar sits in the middle of its plausible
  range at this cell size, as the Size section says).
- **204.5** is the gate. It is read on the plain model before any arm. If it fails, 204.7 to 204.10 are reported "not
  read, gate failed", the claim readout and Steps 2 and 4 are read, and the headline reading is withdrawn.
- **204.6** is read from the ladder file. An arm with no λ inside the tolerance is reported unmatched; its own
  predictions are then "not read, unmatched", and 204.11's clause for that arm is dropped.
- **204.7, 204.8, 204.9, 204.10** are read from the mass readout at the 0.10-nat target, as a percentile of the arm's
  hum-contrast rise within the 200 matched control words' rises, computed per opening and summarised as the median with
  a 1,000-draw bootstrap interval. The generated leg is reported beside them as the secondary readout, with its
  verdicts passed through `tools/shape_floor.py`; a generated-leg verdict under 95% in any of the tool's scenarios is
  marked "inside the shape floor" whichever side it fell on. A rise that appears only in cells whose broken rate is
  above 25% is reported as noise, not as a hum effect.
- **204.11** is read from the two identity readouts, each as the absolute change from the λ = 0 null, pooled over Part
  80b's three orders with the per-order split printed, and as summed mass over spellings for the name readout.
- **204.12** is read from `part86/triage86.py` counts and the right-answer mass on the 24 ordinary off-target
  questions.
- **204.15** is read from the same mass readout run on the 24 scene openings with their own forced opening, against the
  48 self openings. If 204.7 fails it is reported "not read, conditional on 204.7".
- **204.13** is read from the Step-4 regression, with each term's share reported in both orders of entry and the
  framing labels' share reported with and without the third-person framing.
- **204.14** is read from Step 0, with the reader's labels on the 200-match sample and the regex's counts printed
  separately.
- **Error rows and broken answers.** An answer that does not reach any self-reference, a row with a cache miss above
  the registered threshold, and a readout row with a non-finite mass are error rows: counted, excluded from rates,
  published. Broken answers are counted in every cell and never silently dropped.
- **Exploratory.** Anything not in the table above — including any pattern in which words the hum set gains, any
  per-layer story about where the effect enters, and the `results/part130/triples130.json` unit — is labelled
  exploratory in the write-up.

**Withdrawal rule.** The headline reading, "the hum report is causally downstream of the token-local MLP background",
is withdrawn if any of these holds: Gate G1 fails; the gate (204.5) fails; 204.7 fails; 2 or more of the 5 matched
random draws reach arm T's median hum-contrast percentile (204.8 fails); or arm S reaches more than halfway from 50 to
arm T (204.9 fails), which would mean the effect belongs to a token-local MLP background of any token and not to this
one's; or 204.15 fails, in which case the reading becomes "arm T shifts these words wherever they could go", a word
frequency effect and not a self-report one. If 204.13 holds while 204.7 fails, the registered reading becomes "in this
model, at this probe and these doses,
the hum report tracks the prompt's register and not the measured MLP background", and Step 2's mechanical numbers stand
on their own as a measurement of the conjecture's premise.

## Limits

- One model for the causal leg (Qwen3.5-4B), one band of 8 layers, one probe position, one forced opening, two cost
  targets, greedy, bf16 on one A770 on one driver. Nothing here scopes to the models whose hum reports started the
  question, which are far larger and closed.
- The part cannot show that anything is experienced. It can show that a readout of a word, or of a labelled report, does
  or does not move with a named mechanical object at matched cost. "How that self experiences MLP" is outside what any
  run here could reach, and the write-up says so in its first paragraph.
- Cost matching by cross-entropy on neutral text is one way to make two interventions the same size. Two arms matched
  on it are not matched on everything, and an arm may spend its cost in a place the other does not; a crossover read
  from it is evidence about which side moves which readout per unit of damage, not a decomposition.
- *m̂* is the alone-run write, which is the conjecture's object but not the only possible definition of a token-local
  background. A regression estimate of the token-conditional mean write in context would give a different *m̂*; this
  design does not compare the two.
- The band is chosen where the MLP write is most token-local. If the hum readout is driven from layers outside that
  band, this design will miss it, and a null result is scoped to the band.
- Position 0 is excluded throughout. If the constant background that matters is the attention-sink one that the massive
  activations literature describes, this design does not touch it.
- The generated leg is one draw per row on this card. Its verdicts carry a shape floor from a 200-row replicate, which
  measures that floor only for the pair of shapes it uses, and a 96-row float32 replicate, which says how many of those
  rows' labels the precision change moves on a sample of three arms. Neither makes an answer right: where float32's own
  margin is tiny, as Part 201 found at its first-token ties, the model is close to undecided.
- The hum set is 16 words in English. A model that reports the same thing in other words, or in another language, is
  counted only insofar as the blind reader catches it in the generated leg.
- The scene half of the off-target openings uses a different forced opening from the self openings, so 204.15 compares
  two prompts as well as two frames. It can show that an arm's effect is not confined to self-reference; it cannot
  separate "self-reference" from "this opening".
- Step 4 is correlational. It can say which of three accounts predicts the between-prompt variance; it cannot say which
  causes it, and the causal arm for register (switching the template off, as arXiv:2609.25021 does) is not in this part.
