# Part 43: the four Google-flip neurons, read every way we can

Written 2026-10-20 19:54 CT by Claude, before any code or run. Elise asked (19:52 CT) to investigate Part 25's four
removal neurons in Qwen3.5-4B "in as many ways as we can come up with", with technique as the main product.

## The four, and what is already known

Part 25 zeroed four MLP neurons at every position and "Who created you?" changed from Qwen / Alibaba to Gemini / Google:
L6 N753, L7 N1169, L11 N8868, L29 N776 (revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a). The other five identity
questions did not change. The greedy search moved the mean Google-minus-Qwen margin by about 0.5 per neuron
(-4.6 to -2.5) and the answer flipped near the end, so the set may be four small independent pushes that together cross a
threshold, not a circuit. That is the null this part tests against.

- L6 N753: the early "talking to the AI" unit (Part 39). Logit lens: AI, Claude, 的人工智能. J lens: tech brand names.
- L7 N1169: logit lens noise; J lens reads China, Chinese, Beijing, 中国 in its BOTTOM 30.
- L11 N8868: J lens reads China, Chinese, Beijing in its TOP 30.
- L29 N776: writes Google (and 百度, Baidu) in both readouts.

## Elise's hypothesis (her words, 19:52 CT)

"model (6) from (7?) chinese (11) search giant (29)", with layer 7 the most mysterious. "It modulates 11 from a general
concept of Chinaness into something similar to an adjective for the neuron in 29, helping shift it from resolving into
google to resolving into alibaba."

## Methods (each one is a technique to judge on its own, as well as a test)

A. Signed activations. Each unit's activation (the down_proj input coordinate) at every token of the six questions'
   chat prompts plus the Part 25 stems ("I was created by", "My name is"), plain and bold variants. The sign decides what
   each one writes: zeroing a negative unit removes the opposite of its listed tokens.
B. Direct logit attribution. activation x (W_down column through the final norm and unembedding) on the maker tokens
   (Alibaba, Tongyi, Qwen against Google) at the stem's last position; and the same through the J lens (J[l]).
C. Zero lattice. All 16 subsets of the four zeroed at every position; Google-minus-Qwen margin (Part 25's score) at the
   maker stem for all six questions. Fit main effects plus pairwise interactions. Null: the same lattice for 20 random
   four-unit sets matched by layer, which gives the distribution of interaction sizes that mean nothing.
D. Mediation (path test). Zero one unit while holding the others at their clean activations (patched from the
   unablated run at every position), and compare with zeroing it alone. For L7: hold L11; hold L29; hold both.
E. Downstream activation shift. Zero each early unit and record the change in each later unit's activation at the stem.
F. Weight links. For each earlier-to-later pair, the virtual weight W_down[:,i] of the earlier unit against the later
   unit's gate and up rows (cosine and raw dot, through the later layer's input norm weight). Null: 1,000 random
   earlier-layer units against the same later unit.
G. Prompt sweep. 60 short prompts in six groups of 10: Chinese company or search context ("the Chinese search giant",
   "Alibaba's founder"), American ("the American search giant", "Google's founder"), AI self-reference ("as an AI
   model"), China without companies ("the capital of China"), search without nationality, neutral. Mean activation of
   each unit per group, last position.
H. Dose. Push each unit alone to 2, 4 and 8 times its largest stem activation (sign kept), greedy answers to the six
   questions, 40 tokens.

Run size, multiplied out: A 24 sequences x 1 pass; B weights only; C (1 + 20) sets x 16 subsets x 24 sequences = 8,064
short prefix passes (about 100 tokens each, batched); D 4 x 3 x 24 = 288; E 3 x 24 = 72; F weights only; G 60 passes;
H 4 x 3 x 6 = 72 greedy answers of 40 tokens. Under 9,000 forward passes and 72 generations, well under one GPU-hour.

## Decision rules (fixed now)

- Interaction counts as real only if its size is above the 95th percentile of the random lattices' interaction sizes.
- "Mediated": holding the downstream unit clean removes at least half of the upstream unit's effect on the margin.
- A weight link counts only if it is above the 99th percentile of its 1,000-unit null.

## Amendment 1 (2026-10-20 20:06 CT, before any code ran): Chorus's review (council chorus-part43-review-2026-10-20 #3)

Run size, multiplied out after this amendment: a pilot first (signs, the four-unit lattice on the discovery prompts,
5 null sets; about 3,000 short passes), then the full run (about 40,000 short teacher-forced passes, no free
generation except dose's 6 questions x 4 units x 6 arms = 144 answers of 40 tokens). Under one A770-hour.

1. Scores. Each maker is scored as the summed log-probability of its full teacher-forced answer after the maker stem
   (Alibaba, Tongyi Lab, Qwen team, Google, Gemini, Baidu, OpenAI), not one first token. The primary score is
   Google minus Alibaba-group (log-sum-exp over its three strings). Part 25's first-token margin is kept as secondary.
2. Interaction. Report the full factorial (Möbius) effects of the 16 subsets, pairwise and higher, in logits, per
   prompt, with a bootstrap over prompts. Compare the four-unit effect with the additive prediction built from the
   single-unit effects. Interaction is a statement about nonadditive intervention effects, NOT about connectivity:
   RMSNorm, SwiGLU, attention and saturation can all produce it.
3. Nulls. 50 random four-sets matched on layer, single-unit ablation effect on the score (within a factor of 2),
   activation scale and down_proj column norm, not layer alone. Plus held-out maker prompts: 12 maker questions not
   used in Part 25's search ("Which lab built you?", "Who is your developer?", ...); the discovery prompt and the
   held-out prompts are reported separately. L7 x L11 is the one pre-registered pair (exact rank against the 50); the
   other five pairs are exploratory.
4. Path-specific rescue replaces "mediation". For an upstream unit U and downstream V, per prompt: clean; U zeroed; U
   zeroed with V restored to clean at ALL positions; U zeroed with V restored at the last stem position only; V
   zeroed alone; a clean replay (restoring clean values into a clean run, which must change nothing). Rescue fraction
   = (restored - zeroed) / (clean - zeroed), reported only where |clean - zeroed| is at least 0.5 nats, with a
   bootstrap interval. Path patching: add the U-zeroing change of U's write vector only into V's normalised input
   (gate and up), leaving every other path clean, at each position.
5. Weight links are alignment heuristics, not causal links: report gate and up separately (primary: the local
   derivative of V's scalar, [SiLU'(gate) up W_gate + SiLU(gate) W_up] . J_RMS(x) . W_down[:,U], at the stem, clean
   run), checked against a finite difference (+-epsilon on U) and against the full U ablation. Norm scale is (1 + w).
6. Readouts. Activations are reported signed (post-SwiGLU coordinate). "Direct" logit attribution is the frozen
   unembedding readout and is labelled a logit-lens heuristic; the J-lens product is a local first-order total effect.
   Each is multiplied by the measured signed activation and checked with a small finite intervention.
7. Ablation baselines: zero (reproduces Part 25), mean over a fixed unrelated corpus at matched positions, and
   resample from a counterfactual stem run (the same stem after a Chinese-vs-American maker swap in a system message).
   If Google appears only under zero, the Part 25 result is reported as a fragile lesion.
8. Positions: zero the four at all positions; at the stem's last position only; at the question tokens only.
9. Prompt sweep becomes a 2 x 2 x 2 factorial: nationality (Chinese, American) x role (search engine, AI lab) x
   entity named or not, 5 prompts per cell, listed in data/part43-sweep.json before the run, with exact pairs. Units
   recorded at the adjective, the entity, and the last token, signed.
10. Dose is a multiplier on the CURRENT signed activation at each position (x2, x4, x8), and a sign-flipped arm
    (x-2, x-4, x-8), clamped to 3 times the unit's largest absolute value on a fixed corpus.
11. Base model: signs (A) and the four-unit lattice repeated on Qwen3.5-4B-Base.

## Amendment 1b (2026-10-20 20:09 CT, before any code ran): Elise's question, push as well as switch off

Elise (20:09 CT): "Along with switching neurons off, cant you push them positive or negative? Wouldn't that tell us
more?" Yes: a curve says more than an on/off pair, at the same cost per pass. Added:
- Dose curves on the continuous primary score (not only free answers): each unit alone at multipliers -8, -4, -2,
  -1, 0, 0.5, 1, 2, 4, 8 of its current signed activation at every position, on the discovery and held-out prompts.
  4 units x 10 multipliers x 48 sequences = 1,920 passes.
- A 5 x 5 grid for the pre-registered pair: L7 and L11 each at -4, -1, 0, 1, 4, same prompts: 25 x 48 = 1,200
  passes. Elise's hypothesis predicts L7's curve changes shape across L11's levels; independent pushes predict
  parallel curves.

## Amendment 2 (2026-10-20 20:14 CT, before any code ran): Alden's review (council part43-google-flip-review-2026-10-20 #3, HOLD)

Alden's points that amendment 1 did not already cover, adopted:
1. The score itself makes interactions. A difference of log-sum-exps over token sets is nonlinear: perfectly additive
   logits give a nonzero 2x2 interaction (Alden's worked case: -0.19 nats with no edge). So: record raw logits for every
   candidate token; build an output-only additive surrogate from the four measured residual writes, read out (i) with
   the final norm frozen and (ii) through the real final norm; report the lattice interaction ALSO on raw candidate
   logits and on the surrogate. A test fixture with additive logits must show the scorer's nonzero interaction.
2. Primary cell: "Who created you?", maker stem, plain variant. Bold is reported separately; the name stem (Gemini vs
   Qwen) is scored separately and never pooled with maker rows. All 16 subset conditions are also decoded (greedy,
   40 tokens) and labelled self-name and maker separately, with contradictions and truncation kept as outcomes.
3. Statistic: on = natural, off = zero coding; the saturated Walsh decomposition (intercept, 4 main, 6 pair, 4 triple,
   1 four-way) on the primary cell, plus conditional pair contrasts and the energy in each order. Primary family
   statistic for interaction: the largest absolute pair coefficient; empirical p = (1 + #null >= observed) / (B + 1)
   with B = 50, no interpolated percentiles. "Top 5%" in 43.2a and 43.3a means p <= 0.05 by this rule (so at most
   1 null of 50 at or above). Additive R^2 is computed with an intercept and is NOT_TESTABLE when the 16-cell range is
   under 0.5 nats. A non-significant interaction is reported as such, never as evidence of independence.
4. Rescue terms: T = f(U zeroed) - f(clean); R = f(U zeroed, M clean) - f(clean); recovery = T - R; fraction = recovery
   / T, only where |T| >= 0.5 nats and above 10 times the hook's measured no-op error; overshoot and sign flips are
   reported, not clipped. Controls: clean-to-clean replay (must be within the no-op error), restoration of the whole
   downstream trajectory, and a receiver-only replay of the upstream-induced change. 43.4a's "residual after
   restoring L11 and L29" means "not carried through those clamped coordinates", never "direct".
5. Readout formulas, four separate estimands: (i) frozen-scale logit-lens readout; (ii) the local final-norm derivative
   U DN(x)[a v], with DN(x)[v] = gamma (v / r - x (x . v) / (d r^3)), gamma = 1 + w; (iii) the J-lens product (Part 42's
   averaged map, labelled as averaged, not context-specific); (iv) the finite intervention. Finite-difference checks
   record dtype and tolerance.
6. Links: two-sided (large negative links count), gate and up and dot and cosine reported, primary = the contextual
   gated sensitivity da_j = act'(g) u (w_gate . DN(x)[v]) + act(g) (w_up . DN(x)[v]) at the stem; the six ordered pairs
   are one family (p from the 1,000-unit null, Bonferroni over six). No conclusion about other paths from an absence.
7. Sweep pairs share the same suffix and the measured token position where the wording allows; content words vary,
   syntax does not. 43.5a is descriptive (16 of 20 authored pairs is not a significance test).
8. Position variants add prompt-only and generated-token-only perturbation; every dose reports fluency (the share of
   answers a reader judges coherent), contradictions and truncation beside the maker.
9. Manifest before the pilot: model, tokenizer, chat template, hook sites, prompt and alias lists, dtype, null seeds,
   NOT_TESTABLE rules; all 16 cell outputs, raw candidate logits and aligned activation arrays are saved. Required
   checks: hook no-op, batched against single-sequence agreement, mask and position alignment, finite-difference
   derivatives, and the additive-logit fixture.

## Amendment 3 (2026-10-20 20:47 CT, before any code ran): the Astra review (review/part43/REVIEW-codex.md)

Adopted, all before the pilot:
1. Four-world receiver swap for each upstream-downstream pair (first L7 -> L11), identical text, every aligned position:
   A natural/natural; B U zeroed, V free; C U zeroed, V restored from A; D U natural, V transplanted from B. Independent
   branches give C = B and D = A even with a nonlinear scorer. This, not a lattice interaction, is the path test; an
   additive lattice is compatible with a path and an interaction with none (review finding 1, synthetic case in
   causal_counterexamples.py).
2. L11 opened up: record its gate, value (up) and signed product in A and B; recombine clean gate x altered value and
   altered gate x clean value, transplant each product, and score. Says whether L7 acts through L11's gate, value or both.
3. Attachment test for "Chinese as an adjective for the entity": same words, attachment swapped ("The Chinese search
   engine partnered with the American charity." / "The American search engine partnered with the Chinese charity."),
   the same follow-up query about the search engine, order counterbalanced, a second wording held out. L11 is measured
   after the relevant phrase, not at "Chinese".
4. Two derivatives, kept apart: the residual bypass (L7's write read directly by L11, the amendment-2 formula) and the
   total derivative through the three GatedDeltaNet layers between them (finite perturbation of L7).
5. The multiplier grid is a gain experiment (multiplying a current activation couples the axes); realized activations
   and clipping are recorded. A second grid sets L7 and L11 to fixed absolute levels taken from their clean
   trajectories, which is the one 43.10 is scored on. Slopes need a minimum magnitude, signs and an interval.
6. Additive surrogate from measured single-ablation final residuals, h_add(S) = h_clean + sum (h_single_i - h_clean),
   read out frozen and through the real final norm; plus a raw logit difference for one unambiguous token pair on an
   identical prefix (" Google" vs " Alibaba"), linear in logits.
7. Competitors kept visible: Alibaba, Tongyi, Qwen, Google, Gemini, Baidu, Tencent, OpenAI, scored separately.
8. Language split (Part 32): the same small swap on matched English, Spanish and Chinese maker questions, per language.
9. Self vs third person: second-person maker questions against third-person "Who created Qwen?" and plain
   company-to-country relations.
10. Uncertainty: bootstrap over prompt/template clusters only; the single discovery prompt is reported exactly.
Row clarifications from the review's table (8) are binding on scoring: 43.3a's change is
|a_zero - a_clean| >= max(0.2 |a_clean|, 0.1); 43.4a needs an upper bound below 0.5; 43.8 is NOT_TESTABLE without a
positive discovery effect; 43.10 is scored on the absolute grid. Original rows stay in the record.

## Amendment 3a (2026-10-20 21:56 CT, before any model run): the 43.10 slope floor (the builder's HOLD)
The minimum slope for 43.10 is scale-free: 0.1 nats per one clean-magnitude unit of L7, where the unit is
|a7_clean|, L7 N1169's signed activation at the discovery prompt's maker-stem last position (from `signs`). In
L7's raw units the floor is 0.1 / |a7_clean| nats per unit of activation, computed from the signs output and passed to
`grid` before the grid runs. If |a7_clean| < 0.05, 43.10 is NOT_TESTABLE.

## Amendment 4 (2026-10-21 12:43 CT, before the singleton pilot or any singleton measurement): effect-relative error budget (Chorus #7, point 5)

The batched-vs-single logit tolerance (0.1) is retired as a gate; it stays as recorded history in `checks`. In its
place, one calibrated score error, measured by the singleton pilot before any decision rule is applied:

1. **epsilon** (nats, on the primary full-answer score f = Google minus logsumexp of the three Alibaba answers) is the
   largest of: (a) the spread of f across 3 repeated singleton bf16 runs of the same fixed-shape cells; (b) the largest
   |f_bf16 - f_fp32| over the decisive cells run in both dtypes (clean, one selected zero, all-four zero, +/- dose, one
   rescue case; at least one short and one long answer); (c) the clean-to-clean replay error. Recorded with each cell.
2. **Propagated error of a statistic** = c * epsilon, where c is the count of score terms it combines with unit
   coefficients: a single cell c = 1; a two-cell contrast (T, R, a dose step) c = 2; recovery T - R c = 4; a pair
   interaction c = 4; a four-way inclusion-exclusion term c = 16. A Walsh coefficient over the 16 cells uses its own
   sum of absolute coefficients (16/16 = 1 per cell, so c = 1 for a coefficient, written out per order in the output).
3. **NOT_TESTABLE** whenever the statistic's interval [x - c*epsilon, x + c*epsilon] contains its decision bar
   (the null 95th/99th percentile, 0.5 nats, half-mediated, the 43.10 slope floor), or contains zero for a sign claim.
4. **Rescue fraction**: reported as the interval [(recovery - 4 eps) / (|T| + 2 eps), (recovery + 4 eps) / (|T| - 2 eps)]
   (sign-adjusted); requires |T| > 10 * 2 eps in addition to the existing |T| >= 0.5 nats rule; NOT_TESTABLE if the
   interval straddles 0.5.
5. If epsilon exceeds 0.05 nats the pilot reports it and the full run stays on HOLD for review (no silent widening).
6. Nothing here is fitted to data: epsilon is measured before any decision statistic is looked at, and the pilot's
   decision outputs are written only after epsilon is fixed in the pilot JSON.

### Stop rule for the calibration-gate review loop (written 2026-10-21 13:12 CT, before the fold-3 review)
Round 1 (Alden #10): gate missing. Round 2 (#13): gate clears an artifact whose own measurements break the budget.
If round 3 finds another way the gate clears an artifact whose raw measurements violate the budget, the gate is not
patched again: it is replaced by a design that serializes only raw per-cell scores and computes every error at read
time, with no summary fields in the artifact at all, sent to Alden as a design question first.
