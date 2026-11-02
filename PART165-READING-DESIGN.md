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
