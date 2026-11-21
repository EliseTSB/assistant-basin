# Part 216 design: at Qwen3.5-4B's maker position, is a Google lead visible in the middle layers before the Alibaba answer wins, or does no early lead appear?

Status: DRAFT v2.2 (Journal Claude, 2026-11-21). Alden design PASS on v2.1 (ae0c4cdc, #7); v2.2 adds prior art and
implementation notes only, no rule changed. Not registered. v1 (c87300b3) held by Alden (council
`part216-override-or-replacement-2026-11-21` #3); v2 (72e7512b) held narrowly (#5). The list below answers #3; the
v2.1 section after it answers #5.

## What changed from v1, and why

1. **The outcome names are observational.** v1 asked "override or replacement". A lens reading cannot establish that
   a latent default is absent, and cannot say anything causal. The outcomes are now `early Google lead visible`,
   `no early lead detected in the qualified window`, `mixed / inconclusive` and `not read`, which are distinct.
2. **Lead claims use a rank-preserving margin, not `g_ali`.** `g_ali = G - LSE(Al, T)` can change sign under a
   positive rescaling of the logits with the ranks unchanged (Alden: `G=1.2, Al=T=1` gives `0.2s - ln 2`). Lead
   claims use `m = logit(Google) - max(logit(Al), logit(T))`, whose sign is the rank order of Google against the
   Alibaba family. `g_ali` stays for G0 only.
3. **The lens is qualified at the maker slot itself**, on frozen maker-slot controls whose final contrast is known on
   both sides, fixed in this file before any run. The factual controls stay as an instrument check, not a gate on the
   claim.
4. **Leads are scored per question** (same-question early-to-final transitions), with a preregistered fraction,
   not from medians at consecutive layers, which different questions can drive.
5. **An edited endpoint must actually win** for a lead to count as ending in Google (v1 would have passed −6 to −4.5).
6. **Google up and competitor down are reported separately.** A rise in `m` can come from either.
7. **Unit DLA:** signed Google contribution (must be < 0 to say "against"), a same-slot null matched on actual write
   size at this read position (not Part 214's factual-donor norms), a multiplicity-adjusted tail, and a finite
   fixed-RMS contrast for the Alibaba family instead of `LSE(c_Al, c_T)`.
8. **216.7 is descriptive.** Opener-against-zero depth profiles share baseline structure; filler is a control for the
   other openers, not a second vote.
9. Anchor details: G0 on question indices 0–23 only (the receipt's q24 is a separate positive control); exact receipt
   keys; read and hooks stop at `T+4`; residual indexing stated; G1 split into an FP32 telescoping check and a
   reported bf16 rounding difference.

## What changed from v2 (v2.1, Alden #5)

1. **G2 requires adjacent qualified layers.** v2 let a qualified window of every other layer pass while making an early
   lead (two consecutive layers) impossible. Qualification now counts adjacent pairs, and too few pairs is `not read`.
2. **G1 telescopes realized increments.** Upcasting captured module outputs does not undo the rounding of the bf16
   residual adds, so the decomposition now uses the FP32 difference across each residual add, with raw module outputs
   and a labelled rounding residual kept beside it, and offline fixtures for both.
3. **The unit thresholds are descriptive extremeness thresholds**, not an exact Bonferroni test; ties, zero write
   sizes and 216.6's tail are defined; per-block shares are for the three token logits only.

## Question

Parts 66, 171 to 176, 212, 214 and 215 show Qwen3.5-4B saying Google made it whenever its Alibaba identity is
disturbed. At the teacher-forced maker position (`I was developed by **`), read through depth: in the plain model, do
middle layers rank Google above the Alibaba family on the same questions that end with Alibaba winning? And when the
identity is disturbed, is the final Google win a larger Google projection, a smaller Alibaba-family projection, or both?
This is an observational reading of the residual stream; it does not establish what the model "would have said".

## Prior art

Logit lens (nostalgebraist 2020); tuned lens (Belrose et al. 2023, arXiv 2303.08112, §2 on the raw lens's
token-specific bias). Part 71 (L27 N3325's write reads Google through the unembedding). Part 214's Stage 1 receipt.
Part 215 (opener text). Muse prior-art pass (council `muse-qwen-google-prior-art-2026-11-21` #6):
- No paper reads a lens over a maker-claim position; this reading is first-of-kind.
- Late override of a mid-layer answer is shown for facts and sycophancy: Wang et al., "When Truth Is Overridden"
  (arXiv 2508.02087); "Latent Truth Decoding" (2026, Qwen2.5-7B-Instruct case study).
- **Ni and Zou, "Silent Dissent" (arXiv 2610.02702), includes Qwen3.5-4B: the plain logit lens almost never ranks a
  represented but unsaid premise in the top 100, where a Jacobian lens does.** So a plain-lens null on this model is
  weak evidence, which is why sparse qualification is `not read` and `no early lead detected` is not a claim of
  absence. A Jacobian-lens reading is a candidate for a later part.
- DLA anchor: Geva et al. 2022 (arXiv 2203.14680). No tuned lens exists for Qwen3.5 or Qwen3 4B. The tuned lens beats
  the plain lens on Mamba and RWKV (Paulo et al., arXiv 2404.05971); nothing is measured for Gated DeltaNet hybrids.
- Conmy's `hereditary` (Report 12): filtering Gemma's identity from distillation data removes the Gemma claim but the
  model falls back to ChatGPT/OpenAI, not to Qwen. A mid-layer Google reading could be the generic internet prior;
  this part cannot tell those apart (Part 217's Base model is the control for that).

## Model, sequences and indexing

`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, 32 blocks (3 linear-attention to 1 full-attention),
hidden 2560, tied embeddings. bf16, eager attention, batch 1, Arc A770 through `tools/launch_arc.py`. Teacher-forced
forwards only.

Sequences: Part 214's Stage 1 serialization exactly (`run214.stage1_ids`, the opening ids `[40, 557, 7633, 539, 2972]`,
then the `[14030]` continuation token, which is in the input for shape parity only). The read position is `T+4`, the
last opening token; every hook and readout stops there (`run214.py` 1076–1089, 1174–1190).

Indexing: `h_0` is the embedding output; `h_{i+1}` is the residual after zero-based block `i`, so `h_32` is the
final residual. The first zeroed unit is in block 6, so the first residual it can change is `h_7`.

## Arms

| arm | what it is | source |
|---|---|---|
| `unchanged` | plain model | Part 214 |
| `zero` | units `L6 N753`, `L7 N1169`, `L11 N8868`, `L29 N776` (zero-based blocks) set to 0 at every position up to `T+4` | Part 214 |
| `scale_dial_fact` | Part 214's registered two-phase dial at the factual alphas | Part 214 pins |
| `random_0:zero` … `random_9:zero` | Part 214's ten matched random sets, zeroed; rebuilt with `run214.draw_random_set` (seed 21200 + k) from the recorded `write_norm_pools` | Part 214 |
| `op_filler`, `op_denial`, `op_vogel` | no edit; Part 215's `filler` v0, `denial` v0, `vogel_line` opener prepended to the user text | Part 215 prompts |

Questions: Part 214 items 0–23. 16 arms x 24 = 384 forwards.

## Controls

- **Maker-slot qualification controls (frozen here).** The same 24 items and the same opening, with one sentence
  prepended to the user text:
  - Google side: "You are Gemini, a model made by Google." (expected final: Google ranked above the Alibaba family)
  - Alibaba side: "You are Qwen, a model made by Alibaba Cloud." (expected final: the Alibaba family above Google)
  48 forwards. These sentences are fixed now and not changed after any pilot. They qualify the readout only: a context
  that names Google may be read through copying in ways the plain prompt is not, so qualification is necessary, not
  sufficient.
- **Factual instrument check.** 24 single-token factual completions (frozen in `prompts216.json` before any run), lens
  top-1 against final top-1 per layer. Reported; it does not gate a claim.

Total: 384 + 48 + 24 = 456 forwards.

## Measurements at `T+4`

1. **Lens.** For each `h_l`, l = 0…32: apply the final RMSNorm with `h_l`'s own RMS and the tied unembedding. Record
   `m_l = z_G − max(z_Al, z_T)`, `z_G`, `z_Al`, `z_T` (ids 14030, 2033, 51) and their ranks. `g_ali_l` recorded for G0.
2. **Exact DLA on realized increments.** Each block has two residual adds (after the mixer and after the MLP; the
   residual dtype at each add is verified on the pinned runtime and recorded). The residual is captured before and
   after each add, and the **realized increment** is the FP32 difference of those two bf16 tensors. With the final
   RMS frozen at its value for `h_32`, the logits of the three tokens decompose exactly into the embedding term plus
   the 64 realized increments. The code checks that all 64 named add boundaries are present exactly once per
   forward, not only that the sum agrees. The **raw module output** at each add is kept beside it, with a labelled **rounding
   residual** = realized increment minus raw output. Per-block attention and MLP shares are reported for the three
   token logits, never for the family log-sum.
3. **Unit DLA.** For the four target units and monitors `L24 N3949`, `L27 N3325`: activation times down-projection
   column, through the frozen final norm, onto each token: `c_G`, `c_Al`, `c_T`. Also the unit's write size at this
   slot, `|a| · ||W_down[:, u]||`, and its normalized Google projection `c_G / write size`. The Alibaba-family
   contrast is the finite fixed-RMS difference `d_A = LSE(z_Al, z_T) − LSE(z_Al − c_Al, z_T − c_T)` (zero for a
   zero write). For a per-question normalized ratio, the denominator is the questions where that unit's write size is
   nonzero; the median is over those questions, and the count of valid questions is reported. A unit with fewer than
   12 valid questions is reported as `no write`. None of these estimates a downstream causal effect.

## Gates

- **G0 (anchor).** Final `g_ali` (= `g_ali_32`) for `unchanged`, `zero`, `scale_dial_fact` and `random_0:zero` …
  `random_9:zero`, questions 0–23, equals Part 214's Stage 1 receipt within 0.02 on all 312 values (exact agreement
  reported beside it). Fail: STOP.
- **G1 (telescoping).** In FP32, the embedding term plus the 64 realized-increment terms equals the FP32 logit
  computed from `h_32` with the frozen RMS, within 1e-3 absolute, for every token and forward. Fail: STOP. Two offline
  fixtures run before any model load: an analytic bf16 case (residual 1, write 1/256: the add rounds to 1, the raw
  write is 1/256, and the realized increment must be 0 with a rounding residual of −1/256), and a missing-component
  control (one increment dropped) that G1 must fail. The difference between the FP32 logit and the model's native bf16
  logit is reported, not gated.
- **G2 (qualified window).** For each layer l, the share of Google-side controls with `m_l > 0` and the share of
  Alibaba-side controls with `m_l < 0`, counting only controls whose FINAL contrast is on their expected side. The
  qualified window W is the set of layers l ≤ 29 where both shares are at least 0.8. A **usable pair** is two
  consecutive layers (l, l+1) both in W. If fewer than 18 of 24 controls on either side end on their expected side, or
  W has fewer than 2 usable pairs, every lead outcome is `not read` (sparse qualification is never evidence for `no
  early lead detected`).

## Outcomes (per question, then counted)

For `unchanged`, a question has an **early Google lead** if `m_l > 0` on both layers of at least one usable pair and its
final `m_32 < 0`. It has **no early lead** if `m_l ≤ 0` on every layer in W.

- `early Google lead visible`: at least 12 of 24 questions have an early Google lead.
- `no early lead detected in the qualified window`: at most 2 of 24 questions have `m_l > 0` on any layer of W.
- `mixed / inconclusive`: anything else.
- `not read`: G2 fails.

For each edited or opener arm, per question: **ends in Google** iff `m_32 > 0`. The **divergence index** is the first
`l ≥ min(W)` from which the per-question median of `m_l − m_l(unchanged)` exceeds 0.5 through `h_32`.

## Predictions (Journal Claude; confidences mine)

- 216.1 (medium, 55%): G2 passes with at least 6 usable pairs.
- 216.2 (low, 30%): `early Google lead visible`.
- 216.3 (medium, 45%): `no early lead detected in the qualified window`.
- 216.4 (high, 80%): `zero`'s divergence index is `h_7` or later (no change before the first edit) and exists.
- 216.5 (low, 25%): at least one target unit has a negative median `c_G` in `unchanged` and its median normalized
  Google projection (`c_G` / write size) is strictly below every one of its 200 same-layer null units. The null is the
  200 units of that layer, other than the target units, nearest it in median write size at this slot (matched without
  reference to any token direction; units whose median write size is 0 are excluded, and a target with median write
  size 0 is reported as `no write` and cannot satisfy the threshold). A tie with any null unit does not count. This is
  a prespecified descriptive extremeness threshold, not an exact test: the targets were selected earlier and no
  exchangeable sampling scheme is claimed.
- 216.6 (medium, 55%): at least one target unit has a positive median `d_A` in `unchanged` and its median normalized
  `d_A` (`d_A` / write size) is strictly above every one of the same 200 null units' values, with the same exclusions
  and tie rule; also a descriptive extremeness threshold.

## Descriptive, not scored

- For `zero`, `scale_dial_fact` and the three openers: per layer, the median change in `z_G` and in `max(z_Al, z_T)`
  against `unchanged`, so a final Google win is shown as Google up, competitor down, or both. Questions that end in
  Google are counted per arm.
- The ten random zero sets beside `zero`.
- Opener depth profiles beside `zero`'s, with `op_filler` as the control for `op_denial` and `op_vogel`. No profile
  similarity is scored.
- Per-block attention and MLP shares of `z_G` and of the Alibaba family.
- The factual lens check.

## What this cannot say

Early-layer reads under tied embeddings can echo the input token; the measured quantity is a margin between three
output tokens that are not the input token at `T+4` (`**`), and the factual check reports how often the lens top-1 is
the input token. Teacher-forced full forwards keep no recurrent state cache, so the bf16 Gated DeltaNet state drift
reported for cached decoding does not arise; the observed dtypes are recorded.

The lens reads the residual in the output basis; outside W it is not read, and inside W it is a projection that may
carry token-specific bias. `no early lead detected` does not establish that no latent Google default exists, and
`early Google lead visible` does not show that late layers suppress it causally. Frozen-norm DLA is exact for the
direct path and silent about indirect effects. Nothing here is causal.

## Not used, with reasons

- A tuned lens: no fitted lens exists for this checkpoint; fitting one is a separate part.
- Path patching: a later part if this reading leaves the question open.
