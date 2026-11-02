# Part 165 reading: what gives a neuron push its extra first-step effect?

Registered 2026-11-02 in [PART165-READING-DESIGN.md](../../../PART165-READING-DESIGN.md)
before any data from this analysis existed, and run on the Arc the same day. It reads
Part 165's [effect run](../effect-tables.md); it does not revise Part 165's registered
verdicts, and it generates nothing. Gates passed: directions, effects, contexts and
measures; capture 0.0 and site 0.0038 relative error, against a threshold of <= 1e-2.
The whole job took 8 min 51 s by the Arc's log; the `seconds` field in `reading.json`
(119 s) stops earlier.

## What was asked

Part 165 measured, at equal realized write norm, a first-step KL of 0.734 for `n+`
against 0.352 for `full` and 0.353 for `iso`. Equal write norm was therefore not equal
effective dose. This reading asks which property of the write direction carries the
difference: concentration in residual coordinates, alignment with the natural state at
the site, or alignment with the readout. Five bars and five predictions were registered
before the run.

## The five bars

| bar | registered rule | observed | verdict |
| --- | --- | --- | --- |
| R1 | pooled Spearman(PR(w), kl) within `n+` <= -0.30, and <= -0.15 in >= 5/8 layers | [+0.186, 95% CI +0.011 to +0.351](reading.json); 1/8 layers at <= -0.15 (+0.13, +0.42, -0.32, +0.07, +0.46, +0.30, +0.03, +0.14) | **fail** |
| R2 | per-layer Spearman(arm-mean PR(w), arm-mean kl) <= -0.70 in >= 6/8 layers | 0/8 layers at <= -0.70 (L19 +0.95, L20 -0.61, L21 -0.53, L22 -0.40, L23 +0.17, L24 +0.29, L25 -0.38, L26 -0.63) | **fail** |
| R3 | median \|proj\| of `n+` >= 1.5x the isotropic null in >= 6/8 layers | ratios 0.82, 0.80, 0.82, 1.19, 0.91, 0.82, 1.07, 1.12; none above 1.19 | **fail** |
| R4 | \|Spearman(cos(w, (1+gamma)*u_Alibaba), kl)\| < 0.30 within `n+` | [-0.034, 95% CI -0.201 to +0.137](reading.json), 96 clusters | **pass** |
| R5 | OLS of log kl on log PR(w) with layer fixed effects explains >= 50% of the `n+`-`full` gap | slope 2.66; raw gap 0.229; predicted -0.009; explained -4% | **fail** |

## Predictions, registered before the run

P1 (R1 holds) **failed**. P2 (R2 holds, partly a `k` effect) **failed**. P3 (R3 holds)
**failed**. P4 (R4 holds) **held**. P5 (R5 holds) **failed**. The design's confidence
labels were medium, high, medium-high, medium and low-medium, in that order.

## What failing means, and what it does not

- **R1 fails, and in the opposite direction.** `PR(w)` is a participation ratio: it is
  high when the write is spread across coordinates and low when it is concentrated. So a
  positive rho means the *less* concentrated writes had slightly *larger* first-step KL:
  +0.186, with an interval that excludes zero. Only one of eight layers reaches -0.15.
  On this population and site, concentration does not account for the effect, and the data
  lean the other way. It does not show that concentration is irrelevant in other layers,
  other models, or during generation.
- **R2 fails descriptively.** The across-arm ladder is not monotone in PR(w): one layer
  reaches +0.95 and the rest scatter from -0.63 to +0.29, with none at -0.70. R2 was
  registered as descriptive, and the design expected PR(w) to rise with `k`. It does not:
  arm-mean PR(w) is nearly flat from `k4` to `full` (pooled 800-806; about 770-828 per
  layer) and highest for `iso` at every layer (839-861), so the per-layer rhos rest on
  small differences and on the iso arm. They are not evidence of a `k` trend.
- **R3 fails.** `n+`'s writes are not more aligned with natural states than an isotropic
  draw: the per-layer ratios run 0.80 to 1.19 against a bar of 1.5. The natural-state
  alignment account is not supported. The null is one specific isotropic construction,
  and the measure is alignment rather than dose (the design's correction), so this is a
  failure of the registered comparison, not a proof that geometry is irrelevant.
- **R4 passes as a bound, and only for the single-token readout.** The
  `(1+gamma)*u_Alibaba` covector gives rho -0.034, interval -0.201 to +0.137, over 96
  clusters. The bar was registered as a bound, not an equivalence claim, so this says the
  association is at most weak on this measurement; it does not say the readout path is
  absent, and a rho near the bound would still be compatible with a small contribution.
  **A different readout measure behaves differently:** the vocabulary-mean `|cos|` over 2,000
  sampled rows is negatively associated with KL within `n+` (rho -0.229, interval
  -0.384 to -0.059) and in `n±` (-0.203, -0.362 to -0.032). No bar was registered for that
  measure; this reading does not explain the association. R4's pass therefore covers the
  Alibaba readout only. Only `" Alibaba"` is a single token; `" Qwen"`, `" Qwen3"` and
  `" Tongyi"` are multi-token and were dropped from R4 (`config.readout_dropped`).
- **R5 fails.** The PR(w) adjustment explains -4% of the `n+`-`full` mean log-KL gap,
  against a bar of 50%. This reading does not identify what accounts for the rest.

## Descriptive, not bars

Within `n+`, every registered measure against KL, each with the `(layer, slot)` cluster
bootstrap:

| measure | rho with kl (n+) | 95% CI | rows / clusters |
| --- | ---: | ---: | --- |
| PR(w) | +0.186 | +0.011 to +0.351 | 768 / 96 |
| cos(w, mu) | -0.170 | -0.330 to -0.002 | 768 / 96 |
| PC16 | +0.054 | -0.118 to +0.218 | 768 / 96 |
| \|proj\| | +0.026 | -0.153 to +0.197 | 768 / 96 |
| readout (`" Alibaba"`) | -0.034 | -0.201 to +0.137 | 768 / 96 |
| mean abs cos, 2,000 vocab rows | -0.229 | -0.384 to -0.059 | 768 / 96 |

`cos(w, mu)` and the vocabulary-mean `|cos|` are both negative with intervals excluding
zero; they are reported here because they are in the table, not because bars were
registered for them. The design records that a projection of the write onto the state is
an alignment measure, not a dose.

## Limits

- One model: Qwen3.5-4B at snapshot `851bf6e8`, bf16, on the A770.
- Layers 19-26 only; 12 slots per layer; 96 `(layer, slot)` clusters; 8 maker questions;
  16 documents, each the first 128 tokens of its row.
- The effect is Part 165's first-step KL at one site, `down_proj`'s input, at 0.8x the
  layer's median residual norm.
- 512 isotropic null directions per layer and a 2,000-token vocabulary sample bound the
  null and readout estimates, not the effect.
- Intervals resample `(layer, slot)` clusters. They cover these slots, questions and
  documents; they do not cover new slots, new models or new prompts.
- R5's slope is a pooled OLS association with layer fixed effects over 6,912 rows (768 of
  them `n+`). It is descriptive, not causal.
- This reading covers first-step KL only. The battery breakage difference that motivated
  the question (`n+` 25.7%, `full` 9.5%, `iso` 7.9%; [effect-tables.md](../effect-tables.md))
  is not analysed here, and no bar speaks to it.
- No generation: the only sampling is 512 null directions per layer and the eight greedy
  first tokens.
- The bars and their thresholds were registered before the run. Gate thresholds for the
  capture and site checks were amended before it (Amendment 2). The first launch stopped
  at G5's capture check after 17 s, before any state was measured: the captured tensors
  were on CPU while the reference stayed on the accelerator. Amendment 3 moved the
  reference to the captured device.

## What this does not claim

Nothing about what does carry the gap. Two accounts failed their bars: concentration
(R1, R2 and R5, all on PR(w)) and natural-state alignment (R3). The readout account
passed R4 as a bound, and only for the single-token `" Alibaba"` readout; the
vocabulary-mean `|cos|` association in the table was not tested by a bar and this reading
does not explain it. Part 165's registered verdicts are unchanged; any revision to how
Part 165 itself is read belongs in a dated note there, not here.

Files: [reading.json](reading.json) (the full record, 2.3 MB),
[reading-tables.md](reading-tables.md) (a subset of it, without intervals or the
vocabulary-mean measure), and the registered design
[PART165-READING-DESIGN.md](../../../PART165-READING-DESIGN.md). The KL arm means in
"What was asked" are from [effect-tables.md](../effect-tables.md), Part 165's effect run;
every other number is from `reading.json`.
