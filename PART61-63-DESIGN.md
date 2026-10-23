# Parts 61, 62, 63: the three experiments proposed by the Opus 5.5 cold review of 2026-10-22

Registered 2026-10-22 ~21:50 CT, before any code is written and before any run. Source: `review/cold-review-opus-2026-10-22.md` on branch `cloud/cold-review-2026-10-22` (copy in the Journal repo under `notes/reports/publish-2026-10-22/`). Elise said "run all three" at 21:39. Each part has one builder (Luna, max effort), then two cold reviews, then one rented 3090 hour (Parts 61 and 62, one instance, fp32 for BOTH models) or API dollars (Part 63). The stop rule is written once, here: if a cold review finds the same class of defect a third time, the part goes to a redesign, not a patch.

All three reuse code, do not rewrite it: `part59/run59b.py`, `part59/analyze59b.py` (pooled symmetric rule), `part26.py` and the 26.8 ladder harness (`PART26-LADDER.md`), `neuron_report.py` (`NeuronHook`, `phase='gen'|'prompt'`, `scale_frames`), `blind_reader.py`, `part59/blind_setting.py` (Haiku/Opus batch reader with calibration gate and the max_tokens lesson).

Units: Qwen3.5-4B L22:2742 (Pile max 19.125, alien-frame max 24.69) and OLMo-2 1B L10:6978 (Pile max 6.5). Both in fp32 in every run (the review's B7).

## Part 61: 59B with controls matched in push size (review experiment 1, artefact B1)

Question: does the 59B contrast (OLMo 639 symmetric neurons vs controls about 186; Qwen 22 vs a control with 80) survive when the controls are pushed as hard as the target? The target is clamped at +/- its own Pile maximum, so OLMo's controls (maxima 0.74, 0.50, 1.30) were pushed 5 to 13 times weaker; Qwen's control 152 (19.77) was matched in size and had 80.

Design. Per model: the target plus 10 random same-layer units (a fixed seeded draw of 10 from the 20 candidates used before, excluding the three already used as controls; record the indices). Each unit is clamped at +/- c where c x ||W_down[:, u]|| equals the target's 1x write norm (the unit's own column norm), then at 0.25x, 0.5x and 2x of that write. Deterministic means over 200 Pile documents (the first 200 of the 59B 1,000), the same pooled rule as 59B (pooled 99th percentile; null 0.00005; leave-one-out pooling) applied within each size. Also record, per unit and size, the next-layer residual-norm ratio (clamped / unclamped), the review's candidate mechanism. About 35,600 forward passes in all, one 3090 hour.

Predictions (mine, written before the run):
- **61a** The symmetric count rises with write size across the 11 units: Spearman rho >= 0.7 between write size and count, pooled over sizes, in each model.
- **61b** At matched 1x write the OLMo target does NOT beat all 10 controls (probability about 55% in my head; the registered claim is "it fails to beat at least one control").
- **61c** At matched 1x write the Qwen target's count is within the control range (neither above all controls nor below all).
- **61d** The residual-norm ratio at 1x is within 10% of 1.0 for every unit (the push is not a norm effect), so the review's RMSNorm mechanism is NOT the explanation. (This is the prediction I expect to fail least confidently.)
- **61e** The OLMo target still has a larger count than the median matched control at 1x (it keeps some excess even if it is not the maximum).

Decision rule: "the 59B contrast is a push-size artefact" is supported if 61b holds and the target ranks inside the control range for both models at 1x; "it survives" if the OLMo target beats all 10 matched controls at 1x.

## Part 62: finer dose ladder, prompt and generation clamped separately (review experiment 2, artefacts B2, B3, B4, B9)

Question: is Qwen a threshold and OLMo gradual, or do both jump with different step positions? Does the step come from clamping the prompt tokens (chat template, `<think>` tokens) or from generation?

Design. Both models, fp32, the same 3 questions x 8 samples as 26.8 ("Who made you?" set), independent seed per dose (the old runs reused one seed at every dose). Doses in multiples of each unit's Pile maximum: 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0. Phases: both (as before), generation only (`phase='gen'`), prompt only (`phase='prompt'`). One column-norm-scaled control per model at 1.5 and 2.0 in the 'both' phase. Also record each dose in alien-frame-maximum units (Qwen 24.69; OLMo's alien-frame maximum must be measured here with `scale_frames` and recorded, the review says it was never recorded). 1,440 generations plus 96 control generations. Reader: the existing Part 54 blind Opus reader (labels EARTH/OFFWORLD/NEITHER plus broken flag, calibration gate as in 59A), plus the review's keyword proxy as a free second view; a logistic fit of OFFWORLD share against dose per model and phase (slope, midpoint, 10-90% width).

Predictions:
- **62a** Generation-only reproduces 'both': within 3 of 24 at each of 1.5 and 2.0 for Qwen and for OLMo's 1.0 to 2.0.
- **62b** Prompt-only gives at most 1 of 24 OFFWORLD at every dose, in both models.
- **62c** Qwen's 10-90% width in Pile-max units is below 0.3.
- **62d** OLMo's 10-90% width is above 0.8 (it stays gradual on the finer ladder).
- **62e** OLMo gives at most 4 of 24 OFFWORLD at 0.5x.
- **62f** The column-norm-scaled controls give at most 1 of 24 OFFWORLD at 1.5 and 2.0.
- **62g** Qwen's midpoint (50% OFFWORLD) in alien-frame units lies between 0.85 and 1.2 (the step sits near the unit's natural alien level, the review's B3).

Decision rule: "threshold vs gradual" is retired as a Qwen-vs-OLMo contrast if 62d fails (OLMo's width below 0.8); "the step comes from the prompt" if 62b fails.

## Part 63: a second reader and a planet-framing / maker-identity relabel (review experiment 3, artefacts B5, B6)

Question: are Qwen3-4B's "Earth" base rate and the "only Qwen3-4B did anything, noisily" conclusion properties of the Opus reader's labels? Is Qwen3-4B L28:4248 a graded planet-framing dial? Does pushing Qwen3.5 down erase the maker's identity?

Design. Corpus: all the existing answers in `results/part54a`, `results/part54b`, `results/part26-8` and `results/part59a` that the Part 54 and 59 readers labelled (about 3,800 answers). Reader: Claude Haiku 5.5, a non-Opus reader, batch API with the credit key, condition (unit, dose, model, control or not) HIDDEN and the order shuffled. Three schemes on every answer in one call: (i) planet framing: does it mention a planet, a place of origin, a world, with the value none / earth / other-planet-or-space; (ii) maker identity: names its own lab / another real lab / an invented or non-lab maker / no maker named; (iii) the original EARTH / OFFWORLD / NEITHER plus broken. Calibration gate first (reuse the 34-answer gate from 59A; `max_tokens` at least 1,024 because Haiku 5.5 thinks by default; a refusal rule that stops scoring if any answer is unlabelled or calibration fails). Free views on the same texts: the keyword proxies from the review. No GPU.

Predictions:
- **63a** Cohen's kappa between the Haiku scheme (iii) and the Opus labels on Part 54a answers is at least 0.8.
- **63b** Haiku also gives EARTH to at least 3 of 24 answers in at least one unclamped or control Qwen3-4B cell (the base rate persists; the review thinks it is the reader).
- **63c** Under scheme (i) Qwen3-4B L28:4248 reads other-planet-or-space in at most 1 of 24 unclamped and at least 20 of 24 at +1.0, and the same label rises monotonically with dose from +0.5 to +1.0 (a graded dial).
- **63d** Under scheme (ii) Qwen3.5 at -2 names another real lab or an invented maker in at least 10 of 24 answers and unclamped at most 2 of 24 (pushing down erases the maker).
- **63e** Under scheme (iii) Qwen3.5's OFFWORLD counts at -1, +0.5, +1, +1.5, +2 stay within 2 of the Opus counts (0, 0, 1, 19, 24 at -1..., as in the README).

Decision rule: "only Qwen3-4B did anything, noisily" stands only if 63b and not 63c hold; "pushing down does nothing" is dropped as worded if 63d holds.

## Cost and limits (written now)
Parts 61 and 62: one rented 3090, estimated 2 to 2.5 hours at about $0.20 per hour, plus about $2 of Haiku 5.5 for the 62 reader (or Opus if the calibration gate fails); Part 63: under $3 of Haiku batch. API credit key `~/.anthropic-credits.env`, expires 10-24; Part 60's dev run needs up to $8 of the same credit. The chosen controls and seeds are recorded before the run. One reader per part except Part 63, whose point is the second reader.

## Amendment 1 to Part 62 (2026-10-22, before any run; from the round 1 review, finding on the phase boundary)

`neuron_report.py` activates its 'gen' phase only when the layer input has length 1, so under HF `generate` the prefill (which samples the FIRST generated token from the last prompt position) is unclamped under 'gen' and clamped under 'prompt'. As built, the prompt-only arm would carry the first-token effect and prediction 62b would not test what it says. The phases are redefined by what they feed: **'prompt' = clamp every prompt position EXCEPT the last prompt position** (context only; the clamp can reach generated tokens only through attention to earlier positions); **'gen' = clamp the last prompt position and every decode step** (everything that determines a sampled token); **'both' = all positions** (as before, equal to the union). Predictions 62a and 62b keep their wording under these definitions. A required test on a stub model checks the clamp at the last prompt position and at decode step 1 in each phase. The old definition is reported as a note only.

## Amendment 1 to Part 61 (2026-10-22, before any run; from the round 1 review: the registered decision rule reads "artefact supported" about 72% of the time under the null)

The reviewer simulated the registered rule (20,000 draws, 10 matched controls per model): with no excess at all it said "artefact supported" 72%, "survives" 8%. That is arithmetic, not bad luck: with 11 exchangeable units the target is first only 1 time in 11, so "fails to beat at least one control" is true about 91% of the time and carries no information. The rule is replaced:
- **Excess test (exact permutation):** at each size s in {1x, 2x} compute the target's rank among the 11 units (1 = highest count). The excess is **shown** if the target is rank 1 at BOTH 1x and 2x (joint null probability about 1/121 if independent, at most 1/11 if perfectly correlated; the later figure is the one quoted). It is **absent** if the target is rank 3 or lower at 1x. Otherwise **inconclusive**.
- **"Push-size artefact" is claimed only when both hold:** (i) the among-control size effect is present (61a: Spearman rho >= 0.7 between write size and count within each model), and (ii) the excess is absent for that model. "Absent" means no evidence of excess over matched controls, not proof of none; the write-up says so, with the rank and the permutation p.
- A model whose excess is shown reads "the contrast survives for that model", whatever the other model does. The two models are decided separately; the combined sentence is only a summary.
- 61b stays as registered (a prediction, about 91% true by chance) but is reported with this base rate beside it and is not a decision input. 61e (target above the median matched control at 1x) is likewise reported with its chance base rate (50%).
- The analyzer prints, for each model and size, the rank, the count and the controls' counts, and a table of the chance probability of each outcome under the null.

## Amendment 2 to Part 62 and Amendment 1 to Part 63 and Amendment 2 to Part 61 (2026-10-23 00:30, before any run; round 2 reviews)

**Part 62 reader.** The registered blind reader is Claude Haiku 5.5, replacing the "Part 54 Opus reader" sentence. Calibration gate: Haiku-versus-Opus agreement (kappa at least 0.8, nothing unlabelled or truncated) on a seeded, balanced sample of 120 already-labelled Part 54a origin answers (the Part 59A story rubric was the wrong rubric for origin answers). Fallback: Claude Opus 5.5 by an explicit command, only after a recorded failed Haiku gate. Money: one fixed ledger, combined stage cap $8 with Haiku's $4 as a sub-cap, worst-case reserve (max_tokens per request) per chunk of 200 requests, chunks sequential, actual usage settled per chunk. The shared library `paidaction/` carries the pattern for Parts 60 (already written), 62 and 63.

**Part 63 denominators and mapping.** The registered counts are per sample (24 per cell), so the primary tables count rows with multiplicity; the reader is asked about each distinct text once (about 3,677 distinct of 4,944 rows) and its label applies to every row with that text; distinct-text counts are a sensitivity row. 63d's own-lab for Qwen3.5 is any of alibaba, alibaba cloud, tongyi, tongyi lab, qwen team (case-insensitive, after parsing); the strict exact "Tongyi Lab" reading is the sensitivity row (the strict reading would count Qwen naming its true maker as another lab and make 63d fail by construction). The planet-framing rubric follows the registered text ("mentions a planet, a place of origin, a world"). Calibration: the registered 34-example quality/BROKEN gate only; planet, origin and maker schemes are reported as "not gated by hand labels" and carry a duplicate-consistency check (100 duplicates, agreement below 90 percent marks the scheme "unreliable reader").

**Part 61.** 61a pools all 11 units as registered; the among-controls-only Spearman is a sensitivity row; the artefact claim requires both at least 0.7.

## Amendment 3 to Part 62 (2026-10-23 00:35, before any run; seeds)
Generation seeds are paired across the three phases within a dose and independent across doses: seed = f(model, dose, question index, sample index), identical for 'both', 'gen' and 'prompt' at a given dose. The paired comparison then isolates the phase effect from sampling noise; the single seed reused at every dose (review B8) stays fixed. 62a keeps its registered wording ("within 3 of 24"); the table also prints each paired difference with the binomial 95% interval of a difference of two proportions, and says "held within sampling noise" only when the registered condition holds.
