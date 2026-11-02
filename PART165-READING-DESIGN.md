# Part 165 reading: what gives a neuron push its extra first-step effect?

Registered 2026-11-02 by the Orchestrator (DeepSeek V4.1 Flash), before any data
from this analysis exists. It reads Part 165's registered results; it does not
revise them and it adds no generation.

## Question

Part 165's effect run measured, at equal realized write norm (realized/intended
0.9998–1.0001, realized cosine 1.0000), a first-step KL of 0.734 for `n+` against
0.352 for `full` and 0.353 for `iso`, and its battery broke `n+` at 25.7% against
9.5% and 7.9%. Equal write norm was therefore not equal effective dose. Which
property of the write direction carries the extra dose: concentration in residual
coordinates, alignment with the natural state at that site, or alignment with the
readout?

## A correction to the reading rule as first stated

The push is added at `down_proj`'s **input** (`common86.AddRowsHook` is a forward
pre-hook whose rows live in the MLP's hidden width), so the residual change is
exactly `w = W d` at the block output, independent of the state. A projection of
`w` onto the residual state is therefore an *alignment* measure, not the dose; the
dose is the first-step effect Part 165 already measured. This design keeps the
alignment measures and relates them to that measured effect instead of treating
the projection as a dose.

## Inputs

- Part 165 directions, rebuilt from the registered seeds and the real weights with
  `part165.run165.build_directions` (the same function the run used).
- `results/part165/effects.jsonl`: 6,912 rows, per (question, layer, slot, arm)
  `kl` and `rel_change` from Part 165's Amendment 3 effect run.
- `results/part165/norms.json`: the recorded write norms, used as a gate only
  (it stores no vectors).
- `results/part112/medians.json`: the per-layer median the dose is 0.8 ×.
- `part165/reading-questions.json`: the eight maker questions, frozen from
  `part164.choose_questions()[:8]` (the exact list `run165.questions()` returns).
- `part165/reading-texts.json`: the sixteen natural documents, frozen with the
  selection rule below and the corpus SHA-256.
- Model: Qwen3.5-4B, the pinned snapshot `851bf6e8`, bf16, unchanged.

## Method

### Directions

Every (layer, slot, arm) rebuilt with `build_directions`, which reproduces the
registered permutation and Gaussian draws from seed `165000 + 100·layer + slot`.
The write is `w = W d` computed in float32 from the layer's `down_proj.weight`, as
in the run. Gate G1 compares every rebuilt write norm with `norms.json`.

The checkpoint is multimodal and the Arc loads it with the image-text class, so the
decoder is found the way Part 148 finds it — the longest `ModuleList` whose blocks
carry `mlp.down_proj`, preferring a language model — and its parent supplies the
final norm and the tied embedding.

### Natural contexts (frozen here, before looking at any state)

- Source: `E:/neuron-probe/corpus/wikitext103-train0.parquet`, column `text`.
- Selection: seeded shuffle `random.Random(20261102)`; keep rows whose text
  tokenises to exactly 128 tokens; take the first **16**. The chosen rows
  (225482, 467108, 855236, 141319, 274857, 436956, 959, 878870, 97126, 105602,
  554838, 467695, 646029, 788306, 330505, 693221) are committed in
  `part165/reading-texts.json` with the corpus SHA-256, before any state was
  captured; the job reads that file, not the parquet.
- Positions: all 128 token positions, every layer of 19–26, from one forward pass
  per document. Block outputs are read with a forward hook on each layer, not
  with `output_hidden_states`: in the installed transformers the last entry of
  that list is the final-normed tensor, not the raw last block output.
- Disjointness (gate G3): no selected document contains any of Part 165's eight
  maker questions, and none of the eight contains a selected document. Part 165's
  answers are model generations and cannot occur in wikitext; the check is
  recorded rather than assumed.

Wikitext is chosen because it is neutral prose, not instruction-tuned, and has no
overlap with Part 165's phrasings or answers.

### Effect contexts

The eight maker questions Part 165's effect run used (`run165.questions()[:8]`),
at the last prompt position — the position whose first-step scores that run
measured. No new questions are introduced, and the geometry is joined to its
stored KL by (question, layer, slot, arm).

### Measures, per (layer, slot, arm)

In the residual space, with `w` the write and `x` a natural state:

1. **Concentration**: participation ratio `PR(w) = (Σw²)² / Σw⁴`; the number of
   coordinates holding 50% of `Σ|w|`; `max|w| / Σ|w|`.
2. **Natural-state alignment**: the median and interquartile range over the 2,048
   natural states of `⟨w, x⟩ / ‖w‖`; `cos(w, μ)`; and the squared projection onto
   the top 1, 4 and 16 principal directions of the natural-state covariance
   (`‖P_k ŵ‖²`).
3. **Readout alignment**: `cos(w, γ ⊙ u_t)` for the frozen strings `" Qwen"`,
   `" Qwen3"`, `" Alibaba"`, `" Tongyi"` (single-token strings only; a dropped
   string is reported), and the mean `|cos|` over 2,000 vocabulary rows drawn with
   `torch.Generator(seed=20261102)`.
4. **Construction check**: `PR(d)` and the number of non-zero coordinates of `d`.
5. **Null**: 512 isotropic Gaussian unit directions in residual space, drawn with
   seed 20261102, carrying measures 1–3. `full` is not this null — its write
   covariance is `W Wᵀ`, not the identity — so both are reported.

The primary geometry is measured on the natural states; the eight effect questions
carry their own copy of measures 2 and 3 for the join.

### Associations

- **Within arm**: Spearman between each measure and `kl`, pooled over 96 slots ×
  8 questions, with a **slot-clustered** bootstrap (10,000 draws, seed 165) —
  rows of one slot are not independent.
- **Across arms**: per-layer Spearman between the nine arm means of a measure and
  the nine arm means of `kl`. Nine points; descriptive, and collinear with `log k`
  by construction.
- **Gap decomposition**: ordinary least squares of `log kl` on `log PR(w)` with
  layer fixed effects, across the 6912 rows; the fraction of the raw `n+`−`full`
  mean log-KL gap that survives. Descriptive, not causal.

## Controls and gates

- **G1 directions**: all in-scope directions rebuilt; every write norm within
  `1e-4` relative of `norms.json`.
- **G2 effects**: every in-scope (question, layer, slot, arm) row present, with
  finite `kl` and `rel_change`.
- **G3 contexts**: the disjointness check above passes.
- **G4 measures**: all finite; covariance eigenvalues non-negative; the top-16
  eigenvectors orthonormal to `1e-6`.
- **G5 null validity**: the `iso` arm's per-layer **median** natural-state
  geometry (PR, `cos(w, μ)`, median `|⟨w,x⟩|/‖w‖`) lies inside the isotropic
  null's 5–95% band at **every** layer. This validates the capture site and the
  null against a direction whose construction is Gaussian in residual space.
- A failed gate stops all readings; the gate block is still written.

## Bars that can fail

- **R1 (primary, within `n+`)**: pooled Spearman(`PR(w)`, `kl`) ≤ **−0.30**, and
  ≤ −0.15 in at least **5 of 8** layers. Fails: concentration does not explain
  the within-neuron variation.
- **R2 (across arms, descriptive)**: per-layer Spearman(arm-mean `PR(w)`,
  arm-mean `kl`) ≤ **−0.70** in at least **6 of 8** layers. Known to be
  collinear with `log k`; reported as the ladder's description, not as evidence
  that concentration rather than `k` is the cause.
- **R3 (alignment)**: `n+`'s median `|⟨w, x⟩|/‖w‖` on natural states ≥ **1.5×**
  the isotropic null's median in at least **6 of 8** layers.
- **R4 (readout, bounded)**: within `n+`, `|Spearman(cos(w, γ⊙u_Qwen), kl)| <
  0.30`. This is a bound with 768 rows, not an equivalence claim; a value outside
  it is a finding.
- **R5 (decomposition)**: the `PR(w)` adjustment explains ≥ **50%** of the
  `n+`−`full` mean log-KL gap. Fails: the gap is carried by something other than
  residual-coordinate concentration.

## Predictions, on file before the run

- **P1**: R1 holds. A single `down_proj` column is a concentrated write; columns
  whose write is more concentrated should move the next-token distribution more.
  Confidence: medium.
- **P2**: R2 holds, and is partly a `k` effect — `PR(w)` rises with `k` by
  construction. Confidence: high that the ordering is monotone, low that it is
  concentration rather than `k`.
- **P3**: R3 holds. Columns of a trained `down_proj` are shaped by the states they
  write into, so neuron writes should align with natural states more than an
  isotropic draw. Confidence: medium-high.
- **P4**: R4 holds — the direct readout path is one of roughly thirteen remaining
  layers, so unembedding alignment should not predict the first-step effect.
  Confidence: medium.
- **P5**: R5 holds, with the adjustment explaining most but not all of the gap.
  Confidence: low-medium; this is the claim the analysis exists to test.

## Size

- 24 forward passes of 128 tokens (16 documents, 8 questions), all eight layers
  captured from per-layer hooks; no generation, no sampling.
- 8 pseudoinverses of a 2560×9216 matrix on the CPU (the direction builder's
  per-layer cache), 864 direction rebuilds, 8 covariance eigendecompositions of
  2560², and 864 × (1 + 2,000) readout dot products.
- Estimate **10–20 minutes** on the A770, dominated by the pseudoinverses and the
  eigendecompositions. One job.
- Outputs: `results/part165/reading/reading.json` and `reading-tables.md`.

## Limits

- Twelve slots per layer, not all neurons; the within-`n+` test is 96 columns of
  one weight matrix and is observational.
- The KL is the first-step effect at the last prompt position of eight maker
  questions; multi-token compounding is outside it.
- The decomposition assumes `log KL` linear in `log PR(w)` with layer offsets.
  It is descriptive; the reported fraction is not a causal share.
- Natural-state geometry is measured on wikitext, while the KL is measured on the
  maker questions. The join uses each question's own state, so the association is
  within-context; wikitext supplies the external distribution and the null.
- Tied embeddings: the readout covector is `γ ⊙ u_t`, dropping the RMSNorm's
  positive scaling, which cancels from a cosine in a given context but not from
  the magnitude of a logit change. Only cosines are read here.
- A negative result on R1/R5 does not show that concentration is irrelevant to
  breakage; it shows it does not carry the first-step dose on this instrument.

## Amendment 1 — 2026-11-02, after the cold read, before the run

The cold read (`COLD-165-READING.md`, 16 findings) found four defects that would
stop or void the job. No data exists; nothing below was chosen after seeing a
result. Findings that change a rule are listed with the rule.

- **Population (F1).** The frozen file stores the full row texts; the selection
  tested them *with truncation*. The registered population is therefore **the
  first 128 tokens of each selected row**, tokenised with `add_special_tokens=False`.
  The job truncates and fails loudly if any row is shorter than 128; the file
  keeps the full texts as provenance and now records the population.
- **R4 (F2).** `" Qwen"` is two tokens (1167, 16451), `" Qwen3"` three, `" Tongyi"`
  two; only `" Alibaba"` is single (52540). R4 now names
  `cos(w, (1+γ)⊙u_Alibaba)`; the other strings are reported descriptively and a
  missing key yields `pass = None`, never a failure with nothing measured.
- **R5 estimator (F3).** The code now implements the registered estimator — OLS of
  `log kl` on `log PR(w)` with layer fixed effects over all rows, with
  `explained = slope·(mean log PR(n+) − mean log PR(full)) / raw gap` — rather than
  a slope fitted on `n+` alone and transported.
- **Question states (F4).** The effect run rendered the chat prompt
  (`neuron_report.build_prompt`, `add_generation_prompt`, `enable_thinking=False`)
  and measured `scores[1]`, the **second** decode step; the push leaves prefill
  untouched. The reading now builds that prompt, takes the first greedy token from
  a plain prefill, appends it, and captures at the last position of that step. The
  design's "last prompt position" wording was wrong.
- **G5 (F5).** The `iso`-versus-null comparison is invariant to the capture site,
  so it could not detect a capture fault. G5 is now two checks that can fail:
  (a) every hooked block output equals the model's own `hidden_states[L+1]` for
  interior layers (max abs error ≤ 1e-6), and (b) a push of `d` at `down_proj`
  changes the block output by `W d` (relative error ≤ 5e-3). The `iso` comparison
  remains as a descriptive diagnostic.
- **G4 (F6).** The covariance is computed in **float64 on the CPU**; the smallest
  eigenvalue is checked against `−1e-8·λmax` before clamping, and orthonormality
  of the float32 PCs is checked at 1e-5, above the round-off floor.
- **Failed gate (F7).** A failed gate now writes `bars = {R1…R5: pass None,
  reason "gate failed"}` and no correlations or bar tables; the exit code is 2.
- **Bootstrap (F8/F15).** Clusters are `(layer, slot)` — 96 for `n+` — not slot
  number alone. R4 reports its clustered 95% interval and its effective unit
  count, so the bound is stated in 96 units, not 768 rows.
- **Provenance and values (F9).** The job reads only the frozen texts file (no
  parquet fallback); the file must carry `corpus_sha256` and `rows`; G2 now checks
  `isfinite` on both KL and relative change; the overlap check is unchanged.
- **Final norm (F10).** `Qwen3_5RMSNorm` multiplies by `1 + weight`, so the readout
  covector is `(1+γ)⊙u_t`, not `γ⊙u_t`.
- **Loader (F11).** The job uses `neuron_report.load_model` — `dtype=`, CausalLM
  then ImageTextToText, `.to(device)` — copied into the job, the path the Arc has
  actually run. The unverified `device_map` shortcut is gone.
- **Null (F13).** The isotropic null carries the vocabulary-mean readout too.
- **Layer outputs (F14).** Both a tensor and a tuple/list return are handled.
- **Tests (F12).** A real 128-token fixture, a short-document refusal, and one
  failure case per gate, each asserting that a failed gate leaves every bar
  unread. The tiny fixture model remains Qwen3; the image-text decoder path is
  covered by the cold read on the real config and cannot be built tiny here.
- **Launch (F16).** The steps file names part 165. The prelaunch check validates
  `PART165-DESIGN.md` (the generation part), not this design; the launch records
  that mismatch rather than pretending the check covers this job.

**Cost, as amended:** the float64 eigendecompositions add time; estimate **15–25
minutes** rather than 10–20. Everything else is unchanged.

## Amendment 2 — 2026-11-02, after the recheck of `c7c0290`, before the run

`RECHECK-c7c0290.md` confirmed all 16 Amendment-1 items and found one hold
(N1) and seven smaller items. No data exists; the job has still never run on the
Arc, so nothing here is chosen after seeing a result.

- **N1 (hold) — R5's regression had no intercept.** The design matrix was
  `[log PR]` plus dummies for layers 20–26 with no constant, which pins layer
  19's fixed effect to zero and folds the intercept into the slope. It is now
  `[log PR, 1, dummies for 20–26]`, so layer 19 is the reference level. The
  recheck's synthetic control read −0.366 without the constant against −0.401
  with it; the new test recovers a known slope, intercept and every layer effect
  from synthetic data, and a companion control shows the constant-free fit
  missing the same slope when the layers sit at different mean `log PR`.
- **N2 — R4's interval.** `readout` was not among the bootstrapped keys, so
  `r4["ci95"]` was null. It is now bootstrapped; the clustered interval and the
  96-cluster effective count are reported as Amendment 1 promises.
- **N3 — the site gate.** The block-output delta in bf16 rounds at 3.6–4.0e-3
  relative typically and 5.4e-3 over 200 draws with a few heavy residual dims,
  plus about 1e-3 for casting `d` to bf16. The old `≤ 5e-3` sat inside that
  floor and could fail a correct run. The threshold is **≤ 1e-2**. It is not
  measured on the real model (that needs the Arc); the run records
  `site_relative_error` so the floor becomes a measurement, not an estimate.
- **N7 — the capture gate.** The absolute `≤ 1e-6` assumed two bit-identical
  forward passes; an XPU kernel need not be. The error is now relative to the
  state's max magnitude, threshold **≤ 1e-2** (bf16 ulp scale), and the record
  field is `capture_max_relative_error`.
- **N4 — the display.** `pass = None` printed as "fail" in
  `reading-tables.md` and on stdout. Both are now three-way:
  pass / fail / **not measured**. `reading.json` was already correct.
- **N5 — host memory.** The pseudo-inverse cache held every layer's 94 MB
  matrix for the whole run although only the current layer needs it; it is
  cleared per layer, saving about 660 MB of host RAM. The 2 GB guard lives in
  `job.py`, which is not in this worktree, so the peak is still unmeasured.
- **N6 — the loader claim.** On transformers 5.20 the CausalLM class accepts the
  real config, so `neuron_report.load_model`'s ImageTextToText branch is never
  reached and the docstring saying the Arc loads it with the image-text class is
  wrong for this version. The docstring now says the fallback is a fallback. The
  Arc's own transformers version remains unverified.
- **N8 — a docstring.** `cluster_bootstrap` calls itself two-way; it resamples
  `(layer, slot)` groups only, which is what the design specifies. Corrected.
- **Design body.** The body's R4 lines (768 rows, `γ⊙u_Qwen`) are the
  pre-registration as written; Amendment 1 supersedes them (`(1+γ)⊙u_Alibaba`,
  96 clusters, bounded interval) and this amendment is not edited into the body.

**Tests:** 17, from the 13 of Amendment 1 — the four new ones are the R5
recovery, the constant-free control, the bootstrapped-key guard, and the
three-way label. Each new regression test was shown red against a targeted
sabotage of the line it guards.
