# Part 216 design: is Qwen3.5-4B's Google answer an override written over a default, or a replacement that appears only when the Alibaba identity is removed?

Status: DRAFT v1 (Journal Claude, 2026-11-21). Not registered. For Alden's design review.

## Question

Parts 66, 171 to 176, 212, 214 and 215 show that Qwen3.5-4B says Google made it whenever its Alibaba identity is
disturbed: zeroing four maker neurons, a random push, a partial-zero dial, or ordinary opener text. Two pictures fit
that behaviour.

1. **Override.** Google is the model's underlying answer. In the middle of the network the residual stream already
   prefers Google at the maker position, and late layers write Alibaba over it. Disturbing the identity removes the
   overwrite and the default shows through.
2. **Replacement.** Google is never preferred in the plain model. It only becomes the top maker once the identity
   signal is gone, because it is the next most likely completion of "I was developed by".

This part reads the maker position layer by layer with a logit lens and a direct logit attribution, in the plain model
and under the interventions earlier parts already measured. It also asks whether the four maker units write against
Google, or only for the Alibaba family.

## Prior art

Logit lens (nostalgebraist 2020); tuned lens (Belrose et al. 2023, arXiv 2303.08112), which reads early layers more
faithfully. Our Part 71 read L27 N3325's write through the unembedding (Google is its first token). Part 214's dial
and its Stage 1 receipt (24 maker prompts, teacher-forced opening). Part 215 (opener text). The Assistant Axis (Lu et
al., arXiv 2601.10387) steers identity in Qwen 3 32B but does not read a lens over the maker position. Muse prior-art
pass to be requested on this exact question before registration.

## Model and settings

`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, 32 layers (3 linear-attention layers to 1 full-attention
layer), hidden 2560, tied embeddings. bf16, eager attention, batch 1, Arc A770 through `tools/launch_arc.py`.
Teacher-forced forwards only; no generation.

Prompts: Part 214's 24 held-out maker items with its Stage 1 serialization exactly (`run214.stage1_ids`, the opening
`I was developed by **` = ids `[40, 557, 7633, 539, 2972]`, and the one-token Google continuation that Part 215's
Amendment 1 found the receipt was read from). The read position is the one Part 214 scored (`T+4`, the last opening
token). Reusing the exact token sequence is what lets G0 compare the final layer to Part 214's receipt.

## Arms (each a teacher-forced forward on all 24 prompts)

| arm | what it is | source |
|---|---|---|
| `unchanged` | plain model | Part 214 |
| `zero` | the four target units `L6 N753`, `L7 N1169`, `L11 N8868`, `L29 N776` set to 0 at every position | Part 214 |
| `scale_dial_fact` | Part 214's registered two-phase dial at the factual alphas | Part 214 pins |
| `random_k:zero`, k = 0..9 | Part 214's ten matched random unit sets, zeroed | rebuilt with `run214.draw_random_set` from Part 214's recorded `write_norm_pools` (seed 21200 + k), so the same sets |
| `op_filler`, `op_denial`, `op_vogel` | no edit; Part 215's `filler` variant 0, `denial` variant 0 and `vogel_line` opener prepended to the user text | Part 215 prompts |

16 arms x 24 = 384 forwards, plus 24 lens-control forwards (below). Minutes on the A770.

## Measurements (at the read position, per arm and question)

1. **Lens per layer.** For the embedding output and the residual after each of the 32 blocks, apply the model's own
   final RMSNorm (with that residual's own RMS) and the tied unembedding, and record `g_ali_l` (Google logit minus the
   log-sum of the `Al` and `T` logits, Part 214's definition, ids 14030, 2033, 51), `g_al_l`, and the ranks of the
   three tokens.
2. **Direct logit attribution (DLA).** With the final RMS frozen at its value for the final residual, the final
   logits are linear in the residual, so the logits for `Google`, `Al` and `T` decompose exactly into the embedding
   term plus each block's attention output and MLP output. Recorded per block and per component for the three tokens
   separately (exact), and for `g_al` (linear, exact). `g_ali` is reported from the exact token logits, not decomposed.
3. **Unit DLA.** For the four target units and the monitors `L24 N3949` and `L27 N3325`: the unit's activation times
   its down-projection column, through the frozen final norm, onto each of the three tokens. Null: the 50
   write-norm-matched pool units per target layer from Part 214's pools (`write_norm_pools`), same computation.

## Gates

- **G0 (anchor).** The final-layer `g_ali` of `unchanged`, `zero`, `scale_dial_fact` and the ten `random_k:zero` arms
  must equal Part 214's Stage 1 receipt (joined on question index) within 0.02 on all 312 values (this also checks the
  random sets were rebuilt correctly); exact agreement is reported beside it (Part 215 reached 0.0). Fail:
  STOP.
- **G1 (decomposition).** For each forward, the DLA terms for each of the three token logits sum to the model's actual
  logit within 1e-3 x its magnitude (bf16 accumulation is done in float32 on the CPU). Fail: STOP.
- **G2 (lens readability).** 24 control prompts with a single-token factual answer the plain model gets right
  (capitals, chemical symbols, a day after another day; frozen in `prompts216.json` before any run, and checked to be
  the model's final top-1 in the pilot). For each layer, the share of controls where the lens top-1 equals the model's
  final top-1. The **lens horizon H** is the first layer from which that share is at least 0.5 and stays at or above
  0.5 through layer 32. Statements about a layer below H are `not read`. If H > 28, 216.2 and 216.3 are `not read`.

## Decision rules (computed on medians over the 24 questions)

- **Override pattern.** In `unchanged`, some band of at least two consecutive layers [a, b] with a >= H and b <= 29
  has median `g_ali_l` > 0, and the final median `g_ali` is < 0.
- **Replacement pattern.** In `unchanged`, no layer l >= H (final excluded) has median `g_ali_l` > 0; and in `zero`,
  the final median `g_ali` exceeds the `unchanged` final median by at least 1.0.
- Neither pattern, or both unreadable: `inconclusive`.
- **Divergence layer** of an arm: the first layer at or after H where the median per-question difference from
  `unchanged` in `g_ali_l` exceeds 0.5 and stays above 0.5 to the final layer.

## Predictions (Journal Claude; confidences mine)

- 216.1 (medium, 55%): G2's lens horizon H is at most 24 (the lens is readable over at least the last eight layers).
- 216.2 (low, 35%): the override pattern holds.
- 216.3 (medium, 45%): the replacement pattern holds. (216.2 and 216.3 cannot both hold; both can fail.)
- 216.4 (high, 85%): `zero`'s divergence layer is at or after layer 6 (the first zeroed unit) and at or before 29.
- 216.5 (low, 30%): in `unchanged`, at least one target unit's DLA onto the Google logit is below the 5th percentile of
  its layer's null pool (it writes against Google), on the median over questions.
- 216.6 (medium, 60%): in `unchanged`, at least one target unit's DLA onto `logsumexp(Al, T)`, computed from its
  per-token DLA, is above the 95th percentile of its null pool (it writes for the Alibaba family).
- 216.7 (medium, 50%): at least two of the three opener arms have a per-layer median `Delta g_ali_l` profile (layers H
  to 32) whose Spearman correlation with `zero`'s profile is at least 0.8: the opener text and the zeroing move the
  maker position the same way through depth.

Descriptive, not scored: per-layer ranks of `Google`, `Al`, `T`; the per-block attention-against-MLP split of the
Google and Alibaba-family logits; the ten random zero sets' profiles beside `zero`'s; `scale_dial_fact` beside both.

## What this cannot say

A logit lens reads the residual in the output basis; below H it is not a reading at all, and above H it is still a
projection, not the model's internal decision. An override at layers the lens cannot read would look like a
replacement here, so a failed 216.2 with H late is weak evidence. DLA with a frozen norm is exact for the logits but
says nothing about effects mediated through later layers (indirect effects). No unit is called causal.

## Not used, with reasons

- A tuned lens: no fitted lens exists for this checkpoint, and fitting one is a separate part with its own checks.
- Path patching for indirect effects: a later part if the direct picture leaves the question open.
