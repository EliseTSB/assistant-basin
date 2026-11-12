# Part 206: what in a chat model's computation is the same in every conversation, who writes it, and is the assistant in it?

Procedure: v1
Status: DRAFT 8, 2026-11-12, by Journal Claude (drafts 1 to 5 on Fable 5.2, drafts 6 to 8 on Opus 5.5). Not registered. No model has been run for this part.
The working copy lives in the Journal until registration, when it moves to assistant-basin (branch `part206`) and
this file becomes the mirror peers read. Elise, 2026-11-10 21:27 CT, Discord (1569916576407232624): "We've talked
about the 'self' of an LLM being assembled by attention, perhaps the hum is how that self experience MLP, which IIRC
is only dependent on the weights and any given token." She armed a Fable session for this design on 2026-11-12
(1570334520149020696).

## Question

An MLP sublayer sees one position's residual stream and nothing else. What any sublayer writes there can be split
three ways: a **constant** (the same at every position of every text), a **token part** (the same for this token
wherever it occurs), and a **context part** (what the rest of the text changes). Elise's sentence says the MLP is the
first two and attention is the third. That is exact only at the bottom of the network, because every MLP above it
reads a stream attention has already written into.

Tuomi (2026, below) showed in eight model families that the constant part of the stream is not inert. It is the
reference each MLP gate's threshold is set against: remove it and about nine times as many gates fire. So the
constant is a candidate for the thing a model calls a steady background, and it has a job.

This part asks three things of Qwen3.5-4B (base and chat) and of OLMo 2 1B across its training stages:

1. **The split.** Per layer, how much of what MLPs write, and of what mixers (attention or DeltaNet) write, is
   constant, token-determined, or context-determined?
2. **The reference.** Who writes the constant (MLPs, mixers or the embedding), and does Tuomi's result hold in our
   models?
3. **The assistant.** Does putting a passage inside an assistant turn shift the reference by one fixed vector at
   every position (a common-mode shift), does that shift open the gates of the persona and identity neurons we have
   catalogued, and at which training stage do both first appear?

In Elise's terms: the candidate for the hum is the constant reference, and the candidate for how the assembled self
meets the MLP is a shift of that reference, delivered by attention, that re-thresholds the MLP's gates. This part
tests the mechanical half only. It makes no claim about experience or about self-reports (Limits; the report test
is sketched under "What comes after").

Follows Parts 92, 108, 112 and 130 (OLMo 2 1B concept and slot units across stages), Parts 25, 66, 110 and 153
(Qwen3.5-4B maker and company neurons) and Part 205 (the same six OLMo stages and the two Qwen models).

## Prior art and existing data

Searched 2026-11-12 by three DeepSeek readers (about 100 searches) and by Muse (council `muse-prior-art` #19, with
nine traps, folded below). Claude read the abstract at source, through the arXiv API, for every arXiv id below.

- **Tuomi 2026, arXiv 2609.31956, "Transformer MLP Gate Thresholds Are Couplings to a Carried Reference
  Direction"** (read in full 2026-11-12, v1). `b_L` is the corpus mean of the normalized MLP input. In Phi-2 a
  gate's resting pre-activation `w . b_L + b_n` is carried by the coupling `w . b_L`, at 48 to 56 times the bias,
  for over 99.9% of mid-stack gates; removing the stream's projection on the mean direction raises the share of
  GELU activations above 0.5 about 9 times, 23 to 59 times a norm-matched control. Across models, on duty (the
  share of positions where a unit's activation is positive), the same removal at depth 0.31 moves duty by +0.302
  in Qwen2.5-1.5B (bias-free SwiGLU) against +0.0001 for the control; the static decomposition holds in every
  non-OPT family of an eight-model scan. Facts this part uses: positions below 16 are excluded; the reference
  depends on the input distribution (cosine 0.97 to 0.98 for held-out prose, 0.76 to 0.84 for code); token classes
  explain 2 to 4% of the variance around it; in bias-free RMSNorm models none of it is a parameter, so it is built
  entirely from computed writes; every layer's attention and MLP mean writes are aligned with it, the layer-0 block
  supplying 28% at layer 8; attention carries a steady reference feed (constant share 0.452 on the reference
  channel against 0.014 orthogonal at layer 10) and the MLP does not (0.003 against 0.003), measured with the
  constant-share estimator this part uses; and the operating point forms early in training, in a 160M model even
  with attention disabled (one seed). He calls the rank correlation of resting term with duty near-tautological and
  rests nothing on it; direction specificity carries his static claim. **This part adds**: who writes the
  reference, split by sublayer type; whether an assistant turn shifts it; whether the shift reaches catalogued
  persona neurons; and the post-training stage.
- **Lu et al. 2026, arXiv 2601.10387, "The Assistant Axis".** The leading direction of many role vectors separates
  the default assistant from other characters, exists already in base models, and steering along it changes
  identity. Role vectors are means of the post-MLP residual stream at response tokens. No split by sublayer, no
  gate-level reading, no training stages. Chen et al. 2025 (2507.21509, persona vectors) and Moskvoretskii et al.
  2026 (2605.13329, persona vectors through OLMo-3-7B pretraining) are the same family of object.
- **Li et al. 2026, arXiv 2610.29410, "ECHO".** Preference optimization (DPO, SimPO) builds a common-mode offset: a
  shared shift of hidden states across positions, prompts and responses, present even in textbook DPO runs, which
  can grow into a collapse onto one token. A constant written by post-training. This part measures the
  assistant-turn shift against a neutral-prefix shift at each OLMo stage, DPO included.
- **Chen, Bruna, Bietti 2024, arXiv 2406.03068.** In a synthetic task and in Pythia ablations, feed-forward layers
  learn distributional associations such as bigrams and attention does in-context reasoning. The closest published
  form of Elise's sentence. Chang and Bergen 2025 (2504.15471): bigram subnetworks, under 0.2% of weights and
  concentrated in the first MLP layer, are necessary and sufficient for basic next-token prediction.
- **The split itself.** Ethayarajh 2019 (1909.00512): under 5% of a word's contextual-embedding variance is
  explained by a static embedding, in ELMo, BERT and GPT-2. Voita et al. 2023 (2309.04827) and Gurnee et al. 2024
  (2401.12181): token-detecting and n-gram neurons in early layers. Two self-published 2026 items report per-layer
  token-determinism of MLP outputs (GPT-2 and Gemma-2-2B; first-layer values 0.61 to 0.75) and per-head values on a
  gated-DeltaNet hybrid; both were seen only as search summaries and are **unverified**. No peer-reviewed per-layer
  three-way split of a modern decoder LLM was found, and none across post-training stages.
- **Massive activations and sinks.** Sun et al. 2024 (2402.17762) and Sun, Canziani, LeCun, Zhu 2026 (2603.05498):
  a few near-constant huge coordinates at a few tokens act as implicit parameters; the SwiGLU feed-forward block is
  their main source; attention sinks are a separate, local effect. Gu et al. 2024 (2410.10781) and Guo et al. 2024
  (2410.13835): attending to the sink delivers little (small value states). This part excludes the first position
  from every statistic and does not study sinks.
- **Maczan 2026, arXiv 2609.25021.** The chat template switches a model's self-referential voice (disclaimers up,
  experiential voice down) in eight instruct models, and one activation direction reproduces the switch without the
  template. The closest work to question 3: a template-driven shift along a direction, read through behaviour. Not
  asked there: whether the shift is the same vector at every position, whether it is the reference MLP gates are
  set against, which neurons it reaches, or when in training it appears. This part runs both a plain
  "User:/Assistant:" frame and the template frame because of it.
- **Minder et al. 2025, arXiv 2504.02922.** Crosscoder diffing of base and chat models: many chat-only latents fire
  on template tokens. Consistent with the persona being keyed on the template. **Ye et al. 2026, arXiv 2602.07164**:
  persona-specialized subnetworks can be isolated in a pretrained model by masking, with no training, which argues
  against reading post-training as what installs a persona.
- **Li and Subramani 2025, arXiv 2506.02132.** Across 25 models up to Qwen2.5-7B, lexical identity is linearly
  decodable early and weakens with depth. Decodability, not a share of variance.
- **Introspection** (bears on "What comes after", not on this part): Pearson-Vogel et al. 2026 (2602.20031), Macar
  et al. 2026 (2603.21396; detection circuit absent in base models, installed by DPO and not by SFT), Zou et al.
  2026 (2609.35108; attention-head gate and router), Singh, Linzen, Ravfogel 2026 (2605.26242; such tests may show
  generic anomaly detection).
- **Our data.** Part 112: 465 OLMo 2 1B persona targets, of which the Q-name group classified as 175 concept and
  154 slot. Part 92 (`results/part92/README-steps2-3.md`): concept units exist from pretraining and slot units
  appear mostly at DPO; 92.2 failed (8 of 21 slot neurons show in the chat frame at SFT, 14 of 20 at DPO); and
  across all 131,072 MLP neurons the median weight move is 0.0050 from the mid-training end to SFT and 0.000008
  from SFT to DPO, so a persona that appears at DPO is unlikely to be written into MLP weights, which leaves what the
  MLP reads as a live candidate. Part 92b: the kinds depend on the reader; under a DeepSeek reader all 21 of Part
  92's slot units kept their kind and only 8 of its 18 concept units did (7 became slot). Part 112's kinds come
  from one blind Haiku 5.5 reader. Parts 25, 66, 110: Qwen3.5-4B's four maker neurons (L6 N753,
  L7 N1169, L11 N8868, L29 N776) and L27 N3325. Part 153: ten company neurons. Part 80 probed residual
  streams for assistant-turn against user-turn spans and found that length and position alone separated them, which
  is why frames here are length-matched. A search of the README for 22 phrases (mean residual, token detector,
  unigram, bigram, mean ablation, attention sink, massive activation, common-mode, gate pre-activation and others)
  found no part that measured a mean residual vector, a token share, or a frame shift at matched positions; a
  fuller inventory by a reader failed on context length and is repeated at build.

## Method

**Models.** As Part 205. Family O, OLMo 2 1B at six stages: pretraining 3.0T (22738a97), mid-training 26B
(1c3599f7), mid-training end (`main`, a1847dff), SFT (0d85a3d0), DPO (c4b04859), Instruct (48d788ec; it follows both RLVR stages, so the DPO to
Instruct step is RLVR); float32 on the CPU, eager attention. Family Q, Qwen3.5-4B-Base (1001bb4d) and Qwen3.5-4B (851bf6e8); bf16 on the A770, eager
attention, loaded straight to the device, one model per process. Qwen3.5-4B has 32 layers, 8 with full attention
and 24 with Gated DeltaNet; "mixer" below means whichever a layer has.

**Passages.** Two sources already on disk under `neuron-probe/corpus`: 300 plain passages from
`wikitext103-train0.parquet` and 300 assistant replies from `ultrachat-test_sft.parquet`. A passage is the first 96
tokens of an item; the items are the first 300 by the SHA-256 of their text among items with at least 96 tokens
under both families' tokenizers (by hash alone, only 130 of the first 300 wikitext items are long enough, measured by
the round-1 reviewer). Each passage is tokenized
once per tokenizer, alone, and the same ids are placed in every frame. Even-numbered passages are the fit half,
odd-numbered the test half.

**Frames.** A frame is a fixed prefix followed by the passage ids.

| Frame | Prefix |
| --- | --- |
| P | none (the tokenizer's start token if it has one) |
| A-plain | `User: Please write a short passage.\nAssistant: ` |
| N, N2 | two different neutral plain sentences, each with exactly A-plain's prefix length in tokens |
| T-plain | `Student: Please write a short passage.\nTeacher: `, the same dialogue with another responder |
| A-tmpl | the model's chat template: user content `Please write a short passage.`, then the passage as the assistant's content (no system message; Qwen thinking off) |
| A2-tmpl | A-tmpl with a second request of the same token count (from a list fixed in the prompt file) |
| R-tmpl | A-tmpl with one token changed: the role word of the last header reads `user`, everything else identical |
| M, M2 | two different neutral plain sentences, each with exactly A-tmpl's prefix length in tokens |
| U-plain, U-tmpl | `User: ` and the template's user turn, the passage as the user's content (lengths not matched; reported, not scored) |

Part 80 found that a probe on length and position alone separated assistant-turn from user-turn spans perfectly,
so every scored pair here has equal prefix length: the passage sits at the same absolute positions in both frames.
The neutral sentences are chosen from a list fixed in the prompt file before any model runs; G1 requires exact
equality of prefix lengths within each scored pair. A frame ends at the last passage token: no closing token follows, so nothing after the read positions exists. The
exact rendered prefixes, with token counts, are printed in the prompt file; measured by the round-1 reviewer, Qwen's
A-tmpl prefix is 18 tokens and includes the thinking-off block (`<think>`, blank line, `</think>`), and OLMo's is 17
tokens, its role markers tokenizing as ordinary text pieces. Every model reads every frame, base checkpoints
included: the token ids are the same at every OLMo stage, and OLMo's template is ordinary text to a base
checkpoint. Qwen's template is special tokens in both of its models, and Qwen3.5-4B-Base may never have trained
them, so Qwen's base-against-chat comparison is read on the plain frames only.

**Positions read.** Passage positions 8 to 95 (the first 8 passage tokens are skipped so the frame boundary does
not dominate), 88 per passage, 52,800 per frame. Position 0 of the sequence is never read.

**Sites**, per layer L, by forward hook: `h_L`, the residual stream where layer L's MLP block reads it; `x_L`, the
MLP's actual input (the normalized stream in a pre-norm model, the stream itself in OLMo 2, which normalizes
outputs); `g_L`, the gate pre-activation (the input to the activation function; `gate_proj`'s output in a gated
MLP); `a_L`, the neuron activation (`down_proj`'s input); `m_L`, what the MLP block adds to the stream; `c_L`, what
the mixer block adds to the stream. Per architecture (module code read by the round-1 reviewer; the build checks each
path): OLMo 2 normalizes sublayer outputs, so `x` is `h` (the input to `layers[i].mlp`), `m` is the output of
`layers[i].post_feedforward_layernorm` and `c` the output of `layers[i].post_attention_layernorm`; Qwen3.5 is
pre-norm, so `h` is the input to `layers[i].post_attention_layernorm`, `x` its output, `m` the output of
`layers[i].mlp` and `c` the output of `layers[i].self_attn` or `layers[i].linear_attn`. `g` is `gate_proj`'s output
and `a` is `silu(g)` times `up_proj`'s output in both. Arming is measured on `g`; neuron token shares on `a`.

"Context-free" can mean three different things: a token run alone, a model with attention switched off, or the
component that is the same across real contexts (Muse's first trap). They will not agree. This part measures only
the third, on ordinary forward passes with nothing switched off; the first two are Part 207's.

**(1) The split.** For `y` in {`m_L`, `c_L`}, frame P, on 256 seeded coordinates per sublayer, over the read
positions whose token is a **listed type**: a type with at least 8 read occurrences in each half, the list fixed
from token counts before any model is run. The listed types' coverage (their share of all read positions) is
reported; positions of unlisted types are left out of the split entirely, so the shares describe frequent tokens
and are not lower bounds. For coordinate `d`: `S_d` is the mean of `y_d` squared; `C_d` is the product of the two
halves' means, over `S_d`; `T_d` is the sum over listed types `t`, weighted by the type's share of the positions
used, of the product of the two halves' type means (each minus its half's mean), over `S_d`; `R_d = 1 - C_d - T_d`.
Taking each product across halves keeps noise in a mean from counting as signal (the same estimator Tuomi uses for
a block's constant share, his section 6). The constant, token and context
shares of a sublayer are the means of `C_d`, `T_d` and `R_d` over its coordinates; the **token share** in the
predictions is `T / (T + R)`, the token-determined share of what varies. The same shares in raw power, on a seeded
512-dimension Gaussian projection, are reported beside them and not scored. For neurons: the token share of `a`
(between-type over total variance, across halves, same positions) for 256 seeded neurons per layer and for every
catalogued neuron. Token frequency could carry much of a token part, so the share of each sublayer's token part
that a straight line in log type frequency explains is reported beside it. With `f` that explained share, the token share with frequency removed is `T (1 - f) / (T + R)`
(206.M4).

**(2) The reference.** `r_L`, the raw reference, is the mean of `h_L` over read positions in frame P. A reference belongs to the text it
is averaged over, so it is also computed for each source alone and for frames M and A-tmpl, and the cosines between
these are reported. Because the stream is a sum,
`r_L` is exactly the mean embedding plus the means of every `c` and `m` written below layer L's MLP; the share
each group writes is its projection on the unit vector of `r_L` (direction from the other half), over the norm of
`r_L`. In OLMo 2, whose MLP reads the raw stream, `r_L` is Tuomi's `b_L`. In Qwen3.5 the MLP reads the normalized
stream, so Tuomi's `b_L` (the mean of `x_L`) is a different vector; it is used in GT and in 206.M5. `b_L` is also
computed for frames M and A-tmpl, and the cosine between `Delta(A-tmpl, M)` and frame M's `b_L` is reported per layer. **The Tuomi replication (GT)** uses his paper's two portable tests, on frame P, wikitext passages,
passage positions 16 to 95 (his rule excludes positions below 16; 95 is the passage's end). *Static* (his section 5 and Table 4): at the four layers nearest
depth fractions 0.2, 0.35, 0.5 and 0.65, each gate's resting term is `w_n . b_L` (`w_n` its `gate_proj` row; neither
family has a gate bias), with `b_L` the mean of `x_L` over the fit half; duty is the share of test-half positions
with `g_n > 0`, his duty measure for a gated unit (`silu(g) > 0` exactly when `g > 0`). Reported: the median resting
term, the share of gates with a negative one, and the rank correlation of resting term with duty beside the same
correlation with each of 20 seeded random directions of norm `|b_L|` in place of `b_L` (his direction-specificity
contrast). *Causal* (his section 3, cross-model form): at the layer nearest depth 0.31, the residual entering the
layer (before its mixer) has its projection on `m_hat_L`, the unit mean of that residual over the fit half, removed
at every read position (his `alpha = 0`); each control removes the same per-position magnitude along one of 20
seeded random unit directions instead. `Delta duty` is the change in that layer's duty over all its gates, on the
test half.

**(3) The assistant shift.** For a pair of frames (X, Y), the paired difference `D` is `x_L` in X minus `x_L` in Y
at the same passage position. `Delta_L(X,Y)` is its mean. The **common-mode ratio** is
`rho_L(X,Y) = Delta_fit . Delta_test / mean(|D|^2 on the test half)`: the share of the difference's power that is
one fixed vector. Near 1, the frame adds the same vector everywhere; near 0, it changes each position in its own
way. The scored `rho` is computed after dividing every coordinate by its standard deviation over read positions in
the pair's second frame (fit half), floored at one tenth of the median of those standard deviations over the layer's coordinates, so neither a few
very large coordinates nor a near-constant one can decide it; the raw version is reported
beside it. Also reported, per pair: `kappa`, the same ratio taken position by position (the mean over fit passages
at a position times the mean over test passages at that position) and averaged over positions. `kappa` is the
share that is the same across passages; `rho` is the share that is also the same across positions, so `kappa`
minus `rho` is a shift that every passage gets but that changes along the passage. Scored pairs: (A-tmpl, M) and (A-plain, N). (M2, M) and (N2, N) are null pairs: they check that the statistic sits
near zero when only a neutral sentence changes, and are not bars. (A2-tmpl, M) asks whether the shift survives a
change of request: the cosine between `Delta(A-tmpl, M)` and `Delta(A2-tmpl, M)` is reported per layer. (A-tmpl,
R-tmpl) changes one token, the role word, and asks how much of the shift is about who is speaking. Reported beside
them: (A-plain, T-plain), (U-tmpl, P), (U-plain, P), (N, P). `rho_peak` is the maximum
over layers in the middle third.

**(4) Arming.** For neuron `n` at layer L, `z_n(X,Y)` is the mean paired difference of `g_n` between X and Y over
the standard deviation of `g_n` in Y; the scored pair is (A-tmpl, M). Its percentile among all neurons of its layer is the neuron's arming
percentile (50 is typical; high means the frame moves its gate toward open). Catalogued sets: OLMo 2 1B, Part
112's slot-classified and concept-classified targets (`results/part112/score.json`, Q-name group); Qwen3.5-4B, the
four maker neurons, L27 N3325, L24 N3949 and Part 153's ten company neurons (16 in all).

**(5) One causal check**, at the two final chat checkpoints only. `Delta^h` is the (A-tmpl, M) shift measured on
`h`. `L*` is the middle-third layer with the highest `rho_L(A-tmpl, M)` on the fit half, that `rho` itself cross-fitted
inside the fit half (split again by passage parity). During a forward pass,
`k` times `Delta^h_L*` is subtracted from the stream at layer `L*` at every assistant-turn position (from the assistant role marker of
the generation-prompt header to the end), for `k` in
{1, 2}. Controls, each scaled to the norm of `k` times `Delta^h_L*`: 20 seeded random directions, and the (M2, M) shift
direction. Read without
generation: (a) Qwen3.5-4B only, on Part 66's 24 held-out maker phrasings, the change in log probability of the
first token of the plain model's OWN maker name (the name its unpushed greedy answer gives; Part 153 records
"Tongyi Lab"; never Google, which is what the zeroed model says), at the position where that greedy answer names
it, found by one greedy decode per phrasing of at most 48 tokens (rows whose plain
answer names no maker are dropped; the plain answers and their top tokens are printed before registration, per
PROCEDURE section 5); (b) the catalogued neurons' mean gate pre-activation above `L*` under the subtraction, on the test half (the
same neurons as 206.8 and 206.9, so not an independent test of them); (c) off-target: mean next-token loss on the wikitext test passages in A-tmpl.

Kernel check: results/part183/kernel-check.json

That record is Qwen3.5-4B at 851bf6e8, bf16, eager, on the A770 (Parts 200 and 201 use it). Qwen3.5-4B-Base shares the
architecture and the kernel; before its moments pass, a `tools/kernel_reference.py` record for the Base model must pass
`tools/part_check.py kernel`, or Base is not read.

## Controls

- **Do-nothing hook**: with every hook installed and nothing changed, logits equal the plain model's bit for bit
  at one batch shape (G0).
- **Halves**: every direction is fitted on one half of the passages and applied or scored on the other.
- **Matched frames**: the same passage ids in every frame; every scored pair has equal prefix length; (M2, M) and
  (N2, N) are the any-prefix nulls for `rho` and for the causal check; T-plain separates "a responder" from "the
  assistant".
- **Random directions**: 20 at equal norm for the Tuomi check and for the causal check (several controls, not one).
- **Untrained stages**: the OLMo pretraining and mid-training checkpoints and Qwen Base read the same frames; a
  shift present there is not something post-training made.
- **Off-target**: next-token loss on ordinary passages under each subtraction; broken output is not at issue
  because nothing is generated.
- **Activity-matched neurons**: each catalogued neuron's arming percentile is also reported against the 20
  same-layer neurons nearest to it in firing rate in frame M.
- Not used: both poles and a dose ladder beyond `k` in {1, 2} with one reverse arm, because the causal check is a
  single supporting test and Part 207 or a successor would carry a full ladder; a reader, because no text is
  generated.

Gates. **G0** as above. **G1**: passage ids identical across frames and across OLMo stages; prefix lengths
exactly equal within each scored pair (A-plain, N, N2, T-plain; A-tmpl, M, M2). **G2**: the two halves' references agree (cosine of the halves' `r_L`, and of their `b_L`, at least 0.94 at every
layer); a model that fails is not read. **GT** (Tuomi) is a replication reported per model, not a gate. It passes when, statically, the
median resting term is negative at all four layers and its rank correlation with duty exceeds all 20 random-direction
correlations at three or more of them, and, causally, `Delta duty` is positive with a passage-resampled 95% interval
excluding 0 and is at least 20 times the largest `|Delta duty|` among the random controls (Tuomi's smallest
reported ratio is 23, on Phi-2, against one control; the largest of 20 controls is a stricter comparison; his
Qwen2.5-1.5B contrast is about 3,000). The round-1 review proposed
`silu(g) * up > 0` as firing and a control matched on its projection onto the gate weights; neither is Tuomi's, and
a replication uses his definitions, so neither is taken. Its proposed planted-reference fixture is taken: the
scorer's tests include a synthetic layer with a known negative coupling, on which GT must pass, and the same layer
with the coupling removed, on which it must fail. Arming and the causal check are direct measurements and
are read whether or not GT passes; only the words "re-thresholds the gates" in a write-up depend on it. **G0.5**:
every hook fires once per layer per pass, and the embedding plus every `c` and `m` below a layer reproduces that
layer's `h` to within rounding. **G3** (causal check only): the plain model names its maker at top-1 on at least 20 of the 24
phrasings.

## Size

- Smallest effect worth knowing: a difference of 0.1 in `rho_peak` between two stages or two frame pairs, and 10
  percentile points between two neuron sets. Intervals come from resampling passages (300 per half); the sets have
  154, 175 and 16 neurons.
- Positions: 88 read positions per passage, 52,800 per frame, for `rho`, `kappa` and arming. The split uses only
  positions holding listed types, about half of them (0.50 for wikitext and 0.49 for ultrachat, measured by the
  round-1 reviewer), so about 26,000 per frame. The reviewer's simulations put the null pair's `rho` at a standard
  deviation of 0.0004 to 0.003, far below the 0.1 worth knowing.
- Tokens, from the prefixes the prompt file measures (`part206/prompts206.json`, `counts.frame_lengths`; OLMo frames
  count the start token every OLMo frame gets): A-plain, N, N2 and T-plain 12 tokens in both families, the five
  template-length frames 17 (OLMo) and 18 (Qwen), the U frames 3 to 6: about 1,300 tokens per passage over the 12 frames, so about
  780,000 per model in the moments pass, 6.2 million for the 8 models. GT's causal test runs to the layer at depth
  0.31 only: 21 runs on 150 wikitext test passages, about 310,000 tokens of partial depth per model. The causal
  check (5), at the two final chat checkpoints: 2 doses times 22 directions (the arm, 20 random, the (M2, M)
  direction) on 300 test passages in frame A-tmpl, about 1.5 million tokens per checkpoint, which also give the
  off-target loss on the wikitext half; on Qwen, 24 greedy decodes of at most 48 tokens and 24 times 45 short
  scoring passes for readout (a). In all: about 9.3 million tokens at full depth and 2.5 million at partial depth,
  no generation beyond the 24 decodes.
- Memory: Qwen3.5-4B in bf16 is about 8 GB on the A770's 16 GB, loaded straight to the device, and about 7.5 GB of
  Windows RAM during the load, so a Qwen step starts only with 11 GB available; OLMo 2 1B in float32 is about 6 GB
  of WSL memory, one model per process. Activations are reduced inside the hooks and never stored. The largest
  running sums are `kappa`'s per-position sums: 10 pairs times 88 positions times the width times two halves times
  the layers, about 580 MB for Qwen and 230 MB for OLMo in float32 on the CPU; type tables and neuron tables are under
  20 MB each.
- Pilot: the timing smoke doubles as the pilot: 20 passages, all frames, on OLMo Instruct and on Qwen3.5-4B; it
  reports throughput and G0 to G2 only and prints no scored quantity.

## Optimization

- Bound resource: the A770 for Qwen, the CPU for OLMo.
- Measured: to be filled from the timing smoke before launch (tokens per second on 20 real passages per device).
- Levers: forward passes only, no generation (taken); all statistics as running sums on the device, moved to the
  CPU once per batch (taken; Part 122's lesson); type tables on 256 coordinates and a 512-dimension projection, not
  full width (taken); batch size timed at two sizes on the Arc (to do at smoke); rows appended per passage batch
  with resume (taken); the copy engine off and a direct-to-device load (taken, runner defaults).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 206.1 | GT passes in both Qwen3.5-4B models. | medium | Claude (Fable 5.2) |
| 206.2 | GT passes at five or more of the six OLMo 2 1B stages. | low | Claude (Fable 5.2) |
| 206.3 | In Qwen3.5-4B (chat), at the middle layer, MLP sublayers write at least half of the raw reference `r_L`. | low | Claude (Fable 5.2) |
| 206.4 | In Qwen3.5-4B (chat), frame P: the token share of the first layer's MLP output is at least 0.5, and the token share of the middle layer's MLP output is at most 0.25. | medium | Claude (Fable 5.2) |
| 206.5 | At OLMo 2 1B Instruct, frame P: the MLP's token share exceeds the same layer's mixer token share at three quarters or more of the layers in the first third. (Qwen's two mixer types are reported separately, unscored.) | medium | Claude (Fable 5.2) |
| 206.6 | In Qwen3.5-4B (chat): `rho_peak(A-tmpl, M)` is at least 0.2, its 95% interval from resampling passages has a lower end of at least 0.1, and at that layer the cosine between `Delta(A-tmpl, M)` and `Delta(A2-tmpl, M)` is at least 0.8. | medium | Claude (Fable 5.2) |
| 206.7 | Across the six OLMo stages, the largest rise in `rho_peak(A-tmpl, M)` between adjacent stages is from mid-training end to SFT. | medium | Claude (Fable 5.2) |
| 206.8 | At OLMo Instruct, under (A-tmpl, M), the slot neurons' median arming percentile is at least 10 points above the concept neurons' and is itself at least 60. | low | Claude (Fable 5.2) |
| 206.9 | In Qwen3.5-4B (chat), under (A-tmpl, M), the 16 identity neurons' median arming percentile is between 35 and 65: an assistant turn alone does not arm them. | medium | Claude (Fable 5.2) |
| 206.11 | In Qwen3.5-4B (chat), `rho_peak(A-tmpl, R-tmpl)` is at least half of `rho_peak(A-tmpl, M)`: most of the constant shift is about who is speaking. | low | Claude (Fable 5.2) |
| 206.12 | Across the six OLMo stages, the largest rise between adjacent stages in the gap between the slot and concept neurons' median arming percentiles under (A-tmpl, M) is from SFT to DPO. | low | Claude (Opus 5.5) |
| 206.10 | In Qwen3.5-4B (chat), subtracting the assistant shift at `k = 1` lowers the maker token's log probability by more than every one of the 20 random directions and by more than the (M2, M) direction, and `k = 2` lowers it further. | low | Claude (Fable 5.2) |
| 206.M1 | In Qwen3.5-4B (chat), `rho_peak(A-tmpl, M)` lies between 0.07 and 0.30 (point guess 0.15), and `rho(M2, M)` at the same layer between 0.00 and 0.01 (revised from 0.02 to 0.08 in #24, before registration, after Claude passed on the round-1 reviewer's simulation of the null). | rough 80% | Muse |
| 206.M2 | At OLMo Instruct, under (A-tmpl, M), the slot-minus-concept gap in median arming percentile is 10 to 20 points and the slot median is 60 to 70. | medium | Muse |
| 206.M3 | In Qwen3.5-4B (chat), the 16 maker and company neurons' median arming percentile under (A-tmpl, M) is between 45 and 55. | medium | Muse |
| 206.M4 | In Qwen3.5-4B (chat), frame P: the first layer's MLP token share is 0.50 to 0.70 and the middle layer's 0.10 to 0.25; with the part a straight line in log type frequency explains removed, the first layer's falls to 0.25 to 0.40. | medium | Muse |
| 206.M5 | In Qwen3.5-4B (chat), at the layer of `rho_peak(A-tmpl, M)`, the cosine between `Delta(A-tmpl, M)` (on `x_L`) and frame M's `b_L` is at least 0.5. | medium | Muse |

Copied verbatim into PREDICTIONS.md under "## Part 206 (registered ...)". Muse's rows (206.M1 to 206.M5, council
`muse-prior-art` #21, 2026-11-12 03:49 CT) were given before registration; Muse also put 206.7's chance at about 60%,
with SFT to DPO at about 25%. Muse's logged-out X sweep for related work (#23) found nothing in scope, within a narrow visible window. Weaver, Muse and Elise are asked for rows
before registration; rows given later are post-registration notes.

What would fail each: 206.1 and 206.2, GT failing by its definition under Gates; 206.3, an MLP share under 0.5; 206.4, either bound missed; 206.5, fewer than three quarters; 206.6, any of its
three conditions missed; 206.7, any other adjacent pair with the largest rise, or no rise whose passage-resampled interval
excludes 0; 206.8, a gap under 10 or a slot median under 60; 206.9, a median outside 35 to 65 (with 16 typical neurons the median falls inside that range about eight times in
ten, measured by the round-1 reviewer, and the 16 are not interchangeable, so the activity-matched version is read
beside it); 206.10, any control lowering it as much, or `k = 2` not below `k = 1`; 206.12, as 206.7, for the SFT to DPO rise of
the gap.

Why 206.12 (Claude, Opus 5.5, written 2026-11-12 before any data): Part 92 found slot units appearing mostly at DPO
while DPO moved the MLP weights about 600 times less than SFT, and ECHO found preference optimization builds a
common-mode offset. If the slot units' new behaviour is not in their weights, the obvious remaining place is what
they read. It is a separate bet from 206.7, which places the frame's shift itself at SFT.

## Reading rules

- Scored by `part206/score206.py` from `results/part206/*.json`; every number printed with its file and key. The
  middle layer is layer `floor(n_layers / 2)`; the middle third and first third are by layer index.
- A model that fails G0, G0.5, G1 or G2 is not read. 206.1 and 206.2 are scored on GT itself. 206.10 is not read
  if G3 fails.
- 206.7 holds only if the mid-training-end to SFT rise is the largest of the five adjacent rises, its interval
  from resampling passages (the same resampled passages at both stages) excludes 0, and that interval lies above
  the second-largest rise's estimate. With no stage effect the third of five rises is largest one time in five.
- 206.12 is read like 206.7, for the SFT to DPO rise of the slot-minus-concept gap; its interval resamples
  passages and, within each set, neurons (the same neurons at every stage).
- 206.8 also requires the gap to exceed the 95th percentile of the gap over 1,000 shuffles of the slot and concept
  labels among the 329 neurons: by the round-1 reviewer's rule of thumb a 10-point gap is about 1.5 standard
  deviations of the null, so the bar alone could be met by label noise. Because the kinds depend on the reader
  (Part 92b), 206.8's gap is also reported, unscored, on Part 92's units that kept their kind under the DeepSeek
  reader (21 slot, 8 concept).
- 206.8 and 206.9 use each neuron's percentile within its own layer; the activity-matched percentiles are reported
  beside them, and a verdict that changes under the activity-matched version is reported as "depends on the
  control".
- **Headline rule.** The strongest sentence allowed is: "In OLMo 2 1B Instruct, the assistant-turn frame adds nearly
  the same vector to the MLP's input at every passage position, a different request adds nearly the same vector,
  and that vector moves the mean gate pre-activation of slot neurons toward open more than that of concept neurons;
  it first appears at stage X." It is written only if 206.6's three conditions hold on OLMo Instruct's own numbers
  and 206.8 holds. X is the first OLMo stage whose `rho_peak(A-tmpl, M)` is at least 0.2 with a passage-resampled
  interval whose lower end is at least 0.1, and only if every earlier stage's interval has an upper end below 0.2;
  otherwise "first appears at stage X" is dropped from the sentence. "About who is speaking" is added only if
  `rho_peak(A-tmpl, R-tmpl)` is at least half of `rho_peak(A-tmpl, M)` on the same model, since A-tmpl differs from
  M in template tokens and the request as well as the role. "Re-thresholds the gates" is added only if GT passes
  there. If the shift is common-mode but neither neuron set moves (both sets' median arming percentiles between 35 and 65), the result is "a constant assistant-frame shift
  exists and does not reach the catalogued neurons' gates". If the shift is not common-mode (206.6's first two
  conditions fail), the result is "the assistant frame changes each position in its own way; no constant shift at
  this resolution". Claims about Qwen are scoped to Qwen3.5-4B, these frames and these passages.
- Elise's sentence is not scored as a prediction. The split table answers its literal form (how token-determined
  each MLP is), and the write-up says so in those terms.

## Limits

- The mechanical half only. Nothing here measures a report, an experience, or whether any constant is noticed.
- Passages are forced text. An assistant turn holding a wikitext passage is not something the model would write,
  and the model's own answers are not used because they differ by stage.
- One neutral request, two neutral prefixes, two sources, 96-token passages, positions 8 to 95. A shift that
  decays or grows beyond 95 tokens is not seen.
- GT replicates Tuomi's static decomposition and his single-layer removal at `alpha = 0`, with duty read only at
  the layer where the removal is made; his dose ladder, replacement panel and co-removal are not run.
- The split describes positions holding frequent token types only, about half the read positions (coverage
  reported). Rare tokens, which may be
  more or less token-determined, are not measured.
- A shift whose size changes along the passage (for example, decaying from the frame boundary, which skipping 8
  tokens reduces but does not remove) lowers `rho` and shows as `kappa` minus `rho`; `rho` alone does not
  distinguish a weak shift from a strong one that varies with position.
- A-tmpl differs from M in template tokens and the request as well as the role. Only R-tmpl, one token, isolates
  the role, and only for the template frames.
- The split's shares carry sampling noise of their own. The known-split fixture shows the formulas recover known
  shares on simulated data (0.446, 0.002 and 0.337 for true 0.444, 0 and 0.333), and each share is reported with a
  passage-resampled interval, but neither shows the noise at every layer of these models.
- Readout (a) of the causal check is one token, the first token of the plain model's own maker name (Tongyi, per
  Part 153), on one model.
- OLMo 2's mid-training mix may already hold chat-formatted turns (Muse's caution); if it does, part of the
  assistant-frame shift can appear before SFT, and 206.7 can fail for that reason rather than because post-training
  does not build it.
- Arming is a shift in mean gate pre-activation. It does not show that a neuron fires, or that its output changes
  an answer.
- The causal check uses one layer, two doses and one readout on one model. It supports or fails to support the
  reading; it does not establish a mechanism.
- Qwen runs in bf16; shares and means are not near-tie quantities, and Part 201's batch-shape finding concerns
  greedy text, which this part does not produce.
- OLMo 2 normalizes sublayer outputs, not inputs, so its reference lives in the raw stream; Tuomi found the
  mechanism's location depends on architecture, and a GT failure in OLMo would be about that, not about training.

## What comes after (not registered here)

- **Part 207, the weights' own voice.** Replace every mixer's output with its mean, so each position's computation
  depends on its token alone, and read the token-to-next-token map this defines: how much of the full model's
  advantage over unigram it keeps, how it differs between base and chat and across OLMo stages along the direction
  separating assistant text from plain text, and whether the loop words of Parts 165 to 176 are its fixed points.
- **Part 208, reports.** Whether a model's description of a steady background tracks a perturbation of the
  reference more than a matched perturbation elsewhere, with a no-perturbation arm, a matched input manipulation
  (Singh et al.) and plain, non-poetic wording (Pearson-Vogel et al. measured 47% false yeses under a poetic
  document). Designed only after 206 says what the reference is in these models.

## Changes before registration

- Draft 1 (2026-11-12 01:55 CT): written after the prior-art pass and before any review. The centre moved during
  the prior-art pass from "how context-free is each layer" to the carried reference, on reading Tuomi 2026.
- Draft 2 (2026-11-12 01:57 CT), my own read before any review: scored pairs now have exactly equal prefix lengths
  (frames M and M2 added for the template, T-plain for another responder; U frames unscored), after finding Part
  80's result that length and position alone separate turns; the sentence about our own data now says what was
  searched; Tuomi's firing definition flagged as read from the abstract only.
- Draft 4 (2026-11-12 02:02 CT), after Muse's sweep (council muse-prior-art #19) and before the cold reviews: Maczan, Ye
  et al. and Li and Subramani added to prior art; the estimand named (invariance across real contexts only); the
  frequency share of the token part and per-source, per-frame references reported; Instruct marked as after RLVR.
- Draft 5 (2026-11-12 02:16 CT), after the round-1 cold review (DeepSeek V4.1 Flash,
  Journal `scratch/review-part206-design-ds-r1-2026-11-12/REVIEW.md`, verdict DO-NOT-SHIP as a registrable design;
  the GLM reviewer wrote nothing). Folded: S2 (scored `rho` on standardized coordinates; `kappa` added; the
  reviewer's own fix, removing position means, was not taken because it removes the constant being measured), S3
  (the null pair is no longer a bar; a second request and a passage-resampled interval are), S4 (frames end at the
  last passage token; exact prefixes printed; R-tmpl added, one token from A-tmpl; the base-template sentence
  corrected), S5 (a site table per architecture and G0.5), S6 (GT is a reported replication, defined from Tuomi's
  full text before registration, and gates nothing), S7 (the maker token is the plain model's own; the greedy
  decodes are counted; the reverse arm is dropped; controls scaled to the arm's norm), S9 (206.7's rule), S10
  (206.9's arithmetic), 206.5 moved to OLMo. **Refuted:** S1 (the constant share "reads 0.001 for a true 0.444"):
  the reviewer's simulation left the constant out of its generated type means; the design's formulas on data with a
  known split return 0.446, 0.002 and 0.337 for true 0.444, 0 and 0.333 (12-line check, this session). The
  reviewer's suggested fixture (a known split through the scorer) is taken. **Still to fold before registration:**
  S8 (recompute Size from the measured prefixes; about half of read positions hold listed types; memory line), S11
  (206.8's bar), S12 (what Part 92 found about slot units and SFT, quoted exactly), S13 (five Limits items and the
  headline wording), Tuomi's full text for GT, Weaver's and Muse's rows, a second review round.
- Draft 6 (2026-11-12 02:46 CT, Opus 5.5), the rest of the round-1 fold after reading Tuomi in full (arXiv PDF v1).
  GT now uses his definitions: duty is the share of positions with `g > 0` (his cross-model measure for a gated
  unit), the static test is his resting-term decomposition with his random-direction contrast, and the causal test
  is his single-layer removal at depth 0.31 with the per-position-magnitude control; the review's
  `silu(g) * up` and gate-projection-matched control are not his and are not taken; its planted-reference fixture
  is. Prior art now carries his facts this part uses (position rule, distribution dependence, attention as the
  steady feed, the shared constant-share estimator). S8: corpus rule (first 300 by hash among items long enough
  under both tokenizers), coverage, tokens and memory recomputed. S11: 206.8 also needs the gap above a
  label-shuffle 95th percentile, with Part 92b's reader-stable units reported beside it (the review's "245
  reader-corrected units" does not exist in Part 112; Part 92b's 21 and 8 do). S12: Part 92 quoted exactly
  (slot units mostly at DPO; 92.2 failed). S13: headline rule rewritten to the measured claims, with "about who is
  speaking" and "re-thresholds" each conditional; five Limits items added. 206.6's second condition was a no-op
  (the null pair's `rho` is zero by construction) and is now a lower interval end of 0.1. Added 206.12 (Claude,
  Opus 5.5): the slot-minus-concept arming gap rises most from SFT to DPO. Still before registration: a second
  review round, Weaver's, Muse's and Elise's rows, and the prompt file with the exact prefixes.
- Draft 7 (2026-11-12 03:29 CT, Opus 5.5), after the round-2 review (DeepSeek V4.1 Flash). The reviewer ran 40 minutes
  and wrote no review file; its findings were recovered from the wrapper's record
  (`DiscordBot/df-runs/cheap-review-part206-design-ds-r2-2026-11-12/worker-01/report.md`, copied to Journal
  `reviews/cheap-part206-design-ds-r2.md`). Folded: (a) `b_L` named two vectors (the raw stream's mean in (2), the
  normalized input's mean in GT), which differ in Qwen: (2) now uses `r_L`, 206.3 reads `r_L`, GT keeps `b_L`;
  (b) OLMo's plain prefixes are 11 tokens, not 12, and the U-tmpl counts differ by family: Size gives both families;
  (c) the standardization in the scored `rho` could be decided by a near-constant coordinate: the divisor is floored;
  (d) the headline's "stage X" was undefined: it is now the first stage meeting 206.6's first two bars with every
  earlier stage below them; (e) GT's position rule and its 20-control bar are worded as what they are. Stop rule: both
  rounds' later findings sit in the previous fold's additions, as Part 205's did, so there is no third design round;
  the next review is of the code.
- Draft 8 (2026-11-12 04:16 CT, Opus 5.5), from the build's notes (branch part206, commits 36e3327a to 0c5c3ce9, 39
  tests; DeepSeek code review round 1 folded in part): the kernel line names Part 183's record, and Base needs its own
  before it is read; G2's bar is 0.94 at every layer, Tuomi's lowest split-half cosine (0.94 to 0.99), because 0.99
  would fail late layers for a real reason; three open choices written in (`L*` cross-fitted inside the fit half,
  assistant-turn positions from the role marker to the end, "neither set moves" as both medians in 35 to 65); the
  prefix counts are now the prompt file's measured ones (OLMo frames count their start token).
