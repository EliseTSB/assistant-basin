# Part 211: do attention patterns stop changing early in training, sooner than the MLPs, and does the seed pick them?

Procedure: v1
Registered 2026-11-13 11:30 CT by Claude (Opus 5.5), before any data. Elise sent hyperstition_cc's thread on
2026-11-12 (Discord message 1570659281643245590) and said "Sure." to designing a part on it (message
1570660222811381834). The commit that adds this file is the registration. It registers the build with it:
`part211/BUILD-NOTES.md` item 9 freezes which noise floor each prediction reads (council part211-design-2026-11 #13,
passed #15), and the code's other choices are listed there.

## Question

hyperstition_cc ("Stuck in a Loop", hyperstition.cc/stuck-in-a-loop; the post's cite line is dated 2026-09-09, the X
thread 2026-11-11) report that attention patterns stop
changing early: in two runs with the same data order and different initial weights, "by as early as step 1000, a run is
more similar to its final state (step 10000) than the endpoints of different runs are to each other." Their account is
that a key's score gradient is its current attention times its advantage, so attention mostly learns about the keys it
already attends to. Their half-joke reading is that later training is mostly the MLPs.

This part asks four things on public checkpoints, at full training length, with heads matched across runs:

1. When does a head's attention pattern become, and stay, closer to its own final pattern than to the matched head of
   another run's final pattern?
2. Do the initial weights or the data order decide the final pattern more, for these seed draws?
3. Does the attention pattern settle earlier than the MLP output, and earlier than what each head writes?
4. In OLMo 2 1B, does post-training move attention patterns less, relative to a trillion tokens of late pretraining,
   than it moves MLP outputs?

It follows our own findings that structure arrives early on OLMo: Part 92 (concept neurons from pretraining, persona
slots mostly at DPO), Part 209 (the sitting lean already present at pretraining 3.0T), and Part 181 (the OLMo atlas).

## Prior art and existing data

Muse's sweep (council topic `muse-part211-prior-art-2026-11-12`, #3, 2026-11-12 23:39 CT) found no measurement of this
part's quantity (a head's pattern distance to its own final against a matched head of another run's final, tracked over
checkpoints) other than the post itself. What it found, with how each was read:

- The post (read in full by Muse; thread and replies archived in the Journal repository at
  `sources/web-archives/2026-11-12-hyperstition-attention-ratchet.md`). A 124M, 12-layer, 6-head model on FineWeb.
  Heads were matched (Fig. 4: "Across runs — Heads matched at step 10,000"), but the matching method and the similarity
  measure are not stated. The stability run is 10,000 steps; the NATQ fix ran 5,000. The post hedges its own mechanism
  ("the exponential describes the dependence on the score gap; it does not by itself establish exponential growth over
  training time"). It also reports that averaging attention scores across a layer's heads barely changes
  Qwen2.5-1.5B's loss in later layers, which bears on how well heads can be matched at all.
- Bali et al., arXiv:2602.16740 (full text read by Muse): the method anchor for comparing heads across seeds. Cosine
  similarity of flattened post-softmax attention matrices over 100 prompts, matched by greedy best match (they state it
  "obviates the need for Hungarian matching"); final models only, no checkpoints. Initial and final layers are more
  stable across seeds than middle layers (layer 5 of 8 about 0.70 against about 1.0 for layer 1); stability falls
  sharply with prompt length (5 to 50 tokens); AdamW improves stability.
- Xu, arXiv:2606.02378 (abstract): when attention circuits form in Pythia 1B, OLMo 1B and OLMoE, over 10 log-spaced
  revisions. The induction circuit is in place within 0.3 to 2% of training tokens, and the attention sink forms
  separately and later. It tracks circuit selectivity, not pattern similarity.
- Gupta et al., arXiv:2609.34273 (abstract read at source, 2026-11-13): in small character-level transformers, query
  effective dimension grows and key dimension shrinks, and shrinking keys sharpens attention (shown causally by
  controlling the key spectrum); in pretrained families the trend "survives as an early-training regime that later
  decays over a full pretraining run". It measures weight geometry, not patterns on text, and has no data-order arm.
  (This bullet's first draft quoted a sentence about W_Q and W_K alignment as if from the abstract; it is not there, as
  Muse found.)
- Theory on the order of attention and MLP learning (arXiv:2410.09605, arXiv:2610.20337, JoMA arXiv:2310.00535; abstracts
  via search): in toy settings the MLP or output layer learns first and attention later. No empirical measurement on a
  real pretraining run was found, so 211.4 is predicted against this theory's direction.
- Post-training: Li et al., arXiv:2609.21113 (abstract): fine-tuning changes attention patterns, and the layers that
  change most are not the ones that matter most. Zhao et al., arXiv:2604.11838 (abstract via search): layer-wise SFT
  change in models including OLMo-2-0425-1B, middle layers stable and final layers sensitive. None splits attention
  from MLP.
- PolyPythias (van der Wal et al., arXiv:2503.09543; full text read by Muse, model cards and hub branches read here):
  the phrase "attention pattern" does not appear in it. Its seed1 to seed9 runs vary initial weights and data order
  together; this part does not use them. Pythia-160m was also retrained with only the data order varied
  (`pythia-160m-data-seed1` to `3`) and with only the weight initialisation varied. ⛔ **The original weight-seed
  releases (`pythia-160m-weight-seed1` to `3`) do not vary initialisation**: their model card says a training error made
  them continue from the original Pythia weights with incorrect Adam buffers, and that they "should not be used as
  weight-initialization" variants. Corrected, independently initialised runs exist for seeds 1 and 2 only
  (`pythia-160m-weight-seed1-corrected`, `-seed2-corrected`), 155 hub branches each with steps 0 to 143000, including
  1000 and 10000 (read from the cards and the hub, 2026-11-13). Alden caught this from the card; an earlier version of
  this bullet had checked only that the repositories existed. The data-seed runs shuffle the data before it is packed
  into sequences (PolyPythias section 2, cited by Alden), so a data seed changes batch composition, not only the order
  of a fixed list of packed examples; "data order" in this part means that.
- Also: Chen et al., ICLR 2024 (syntactic attention structure appears abruptly in MultiBERTs); Olsson et al. 2022
  (induction heads at a phase change); Zhai et al. 2023 (very low attention entropy accompanies unstable training).
- Our own data: Part 181 measured which documents MLP neurons fire on across 12 OLMo 2 1B checkpoints; it did not
  measure attention patterns. No part of ours has compared attention across checkpoints or seeds.

## Method

Two families, both on the CPU in float32 with eager attention, no generation.

**Pythia-160m (12 layers, 12 heads, d 768, GPT-NeoX parallel attention and MLP).** Six runs: R0
`EleutherAI/pythia-160m` (seed 1234), W1 and W2 `pythia-160m-weight-seed1-corrected` and `-seed2-corrected` (data order
fixed, initial weights varied), D1 to D3 `pythia-160m-data-seed1..3` (initial weights fixed, data order varied). The
uncorrected `pythia-160m-weight-seed1..3` are excluded (Prior art); there is no corrected seed 3, so the weight arm has
two runs. Twelve checkpoints each: steps
0, 128, 512, 1000, 2000, 4000, 8000, 10000, 16000, 32000, 64000 and 143000 (final). Step 10000 is the thread's final
step, so the thread's own comparison can be made at its own horizon (211.8). Pythia's learning-rate warmup is the first
1,430 steps, so step 1000 is inside warmup, unlike the thread's run. Each revision is pinned by its commit hash in
`part211/checkpoints.json`, written by a host script before registration and committed with it, so the registration
names the exact repositories and revisions used (Alden #6, finding 1).

**OLMo 2 1B (`allenai/OLMo-2-0425-1B` and its SFT, DPO and Instruct repositories; 16 layers, 16 heads).** One run, 14
checkpoints: pretraining step 0 (`stage1-step0-tokens0B`), step 300 (`stage1-step300-tokens1B`), and the 12 of
`part85/checkpoints.py` at their registered commit prefixes (pretraining 21B, 210B, 1T, 2T, 3T, 4T; mid-training 13B,
26B, 51B; SFT; DPO; Instruct, which comes after two RLVR stages on top of DPO). Step 0 and step 300 are new downloads; the other 12 are cached.

**Text.** From `NeelNanda/pile-10k`, the documents with at least 256 tokens under each family's tokenizer, drawn with
`random.Random(211)`: set A, 48 documents, registered for the verdicts; set B, 24 further documents, reported beside
every verdict. Each document is cut to its first 256 tokens. Raw text, no chat template, for every checkpoint.

**What is recorded per checkpoint, per document:** every head's attention weights (the softmax rows), each layer's MLP
output, and each head's contribution to the residual stream (its attention output times its block of the output
projection). MLP and head outputs are kept at positions 3, 6, ..., 255 of every document (85 per document, 4,080 for set
A, 2,040 for set B), recorded in the run manifest. The weights themselves are read for the OV measure below.

**Measures.**

- Pattern distance P between two checkpoints for one head: the Jensen-Shannon divergence (base 2, between 0 and 1)
  between the two attention rows at each query, averaged over queries 1 to 255 and the documents. Query 0 is left out
  because its row is always one key. P0, the same with key 0 removed and each row renormalised (queries 2 to 255), is
  reported beside P for the sink. Beside P, for robustness, the cosine distance between the two heads' attention
  matrices flattened over the documents (Bali et al.'s measure), reported, not predicted on.
- MLP distance M for one layer: 1 minus centred linear CKA (Kornblith et al. 2019: features centred over positions)
  between the two checkpoints' MLP outputs at the recorded positions.
- Head-output distance O for one head: 1 minus centred linear CKA between the two checkpoints' head contributions. O
  contains the pattern's effect (the contribution is the pattern times the values times the output block), so it is a
  composite and is reported, not predicted on.
- OV distance V for one head, a pattern-free measure of what the head writes: the Frobenius norm of the difference
  between the two checkpoints' OV matrices (the head's value block followed by its output block, a 768 by 768 matrix of
  rank at most 64 for Pythia), divided by the final checkpoint's OV norm. It is a weight measure, so it can move without
  any functional change (a scale shared with a later LayerNorm, for example); see Limits.
- Settling fraction for a measure X at checkpoint s: f_X(s) = X(s, final) / X(step 0, final). For P and V the layer's
  value is the ratio of head means, mean over heads of X(s, final) divided by mean over heads of X(step 0, final), not
  the mean of per-head fractions (the two can differ; Alden's two-head example gives 0.404 against 0.6). For P and V
  this is the fraction of the step-0 distance still to go. For M it is a ratio of dissimilarities, not a path fraction,
  and can exceed 1; values above 1 are reported, not capped. A denominator X(step 0, final) at or below four times the
  noise floor is unmeasured: the layer gets no fraction, no settling time, and counts against any bar that reads it.
- Settling time t_X for a layer: the first checkpoint s at which f_X is at most 0.5 and stays at most 0.5 at every later
  checkpoint before final. Between two checkpoints it is interpolated linearly in log(1 + step), one rule for every
  interval including step 0 to step 128; the interpolated time is a convention, not an observation. A layer whose f_X is
  still above 0.5 at the last checkpoint before final is censored ("not settled"); f_X at final is 0 by definition and
  is never used. A comparison t_X < t_Y holds when t_X is a time and t_Y is censored, or both are times and t_X < t_Y; it
  does not hold when t_X is censored, including when both are censored (no evidence of order is not evidence of
  equality).
- These times compare the declared metrics at the declared threshold, in shared units of steps. JS, CKA and a weight
  norm are not one scale of "learned change", and a monotone change of a metric (for example the square root of JS) can
  reorder them without any change in the attention data (Alden #6, finding 5, with a worked toy). The report states
  every ordering "with these observables" and prints, descriptively, the settling times at thresholds 0.25 and 0.75 and
  for the square root of JS, from the same distances.
- Cross-run distance for one layer between runs a and b at the same step: the 12-by-12 matrix of P between every head
  of a and every head of b; the minimum-cost one-to-one assignment by `scipy.optimize.linear_sum_assignment` (scipy is
  required; the run aborts without it); the matched distance is the mean P over assigned pairs. Beside it, Bali et al.'s greedy match
  (each head of b takes its nearest head of R0, repeats allowed) is reported, not predicted on. Computed at steps 0,
  10000 and 143000 for every run against R0, and at steps 1000 and 16000 as a descriptive divergence curve that no
  prediction reads.
- Same-index compatible head (for runs that share R0's initial weights): head i at step 143000 is same-index compatible
  when its gap g_i = P(i in run b, i in R0) minus the minimum over j of P(i in run b, j in R0) is at most the layer's
  noise floor. This reads the pattern distances directly, not the solver's permutation, which is arbitrary among
  equally good assignments when heads share a pattern. It is compatibility, not identification: if every cost in the
  layer were zero, every head would be compatible and none would have a unique partner (Alden #6, finding 2), so a pass
  is not evidence that particular heads kept an identity. Beside it, per layer: the number of heads whose best match is
  unique (second best above the best by more than the floor), and the number that are ambiguous (second best within the
  floor of the best).
- Crossing step s*(layer, run b): the first checkpoint at which the layer's mean within-run P(s, final) is below its
  matched cross-run P at step 143000 between run b and R0, by more than the floor, and stays below by more than the
  floor at every later checkpoint before final. A layer that has not crossed by step 64000 is censored ("never
  crossed") and does not count as crossing by any step.
- The thread's comparison at its own horizon, for layer L and weight-seed run b: R0's within-run P(step 1000, step
  10000), mean over the layer's heads, against the matched cross-run P between b and R0 at step 10000.
- Crossing step for the MLP, s*_M: the same as s* with M in place of P and the cross-run M (CKA distance between run b's
  and R0's MLP outputs) at step 143000. It puts the MLP on the same yardstick as the pattern (distance to own final in
  units of the distance between runs), and is reported beside 211.4, not predicted on.

**Uncertainty: the whole statistic is resampled** (Alden #6, finding 3). One bootstrap resample is one draw, with
replacement, of set A's 48 document IDs (seed 211, 1,000 resamples), and the same draw is used for every checkpoint,
every run and every measure, so paired quantities stay paired. Inside each resample everything a prediction reads is
recomputed from the per-document records: the distances, the head assignment (refit by `linear_sum_assignment` on the
resampled cost matrix, not the original mapping carried over), the normalised fractions, the persistent crossing step
s*, the settling times with their censoring, the median over data-seed runs, and the layer's verdict. A layer's
interval is the 2.5th to 97.5th percentile of that recomputed statistic over resamples, so persistence over later
checkpoints is part of what is resampled, not a set of pointwise intervals read together. Near-tied assignments make
the bootstrap's coverage approximate; that is a limitation, not evidence about identity. The OV distance V has no
document sample, so its "interval" is a single value: an interval on t_P minus t_V comes from the P side only and says
nothing about training or seed variability. Set B is scored the same way and fits its own assignment, an independent
estimate of the same matched distance, not a test of whether set A's mapping transports. The only fitted step anywhere
is that assignment. If the pilot projects the M bootstrap (pooled centred CKA recomputed per resample) over two hours,
the M resamples drop to B = 200; that fallback is fixed here, before data. Under it, M uses the first 200 of the 1,000
document draws, and every statistic that pairs P with M (211.4's t_P against t_M, 211.6's ratio difference) uses those
same 200 draws for P, so the pairing is kept; P-only and V-only statistics keep all 1,000. Every count threshold is
ceil(0.975 B) of B: 975 of 1,000, or 195 of 200 (Alden #8). A draw in which a statistic is unmeasured or censored
against the prediction counts as a failure of that draw; no draw is dropped from the denominator.

**Noise floor; no storage rounding.** Every stored measurement input (reference attention rows and MLP outputs) is
kept in float32, the precision it is computed in, so no serialization rounding enters any compared quantity. (An
earlier draft stored float16 and passed both sides of the floor comparison through it; Alden #8 showed that measures a
perturbation after rounding and cannot certify a distance the rounding erased, e.g. rows (0.5001, 0.4999) and
(0.5, 0.5), JS about 7.2e-9, become identical in half precision.) The floor, per layer and per measure, is the final (or
compared) checkpoint against the same weights rounded through bf16 and run in float32. For OLMo, whose Instruct weights
are stored in bf16, the floor for any comparison involving Instruct is the mid-training-end checkpoint against itself
rounded through bf16. For P0: a query whose row keeps less than
0.001 of its mass outside key 0 in either compared row is dropped from P0, the dropped count is printed, and a head with
more than half its queries dropped has no P0.

**Storage and memory.** Reference checkpoints whose attention rows are kept on disk in float32: Pythia, each run's step 0,
step 10000 and step 143000; OLMo, step 0, pretraining 3T, 4T, mid-training end (51B) and Instruct. Every other checkpoint is reduced
to its distances against these as it is read, kept per document (JS averages are additive over documents, so a
resample only reweights them): per head for within-run P and P0, and the full 12-by-12 per-document P matrix for every
cross-run pair a prediction or descriptive curve reads. Cross-run matrices are computed one layer at a time (12 heads by
72 documents by 255 by 256 in float32 is about 226 MB per run per layer; a Pythia reference checkpoint is about 2.7 GB
and an OLMo one about 4.8 GB, about 73 GB for the 18 Pythia and 5 OLMo references). Pooled centred CKA is not the mean
of per-document CKA values, so the MLP outputs at the recorded positions are kept for every checkpoint in float32 (about
226 MB per Pythia checkpoint and 802 MB per OLMo checkpoint, about 27 GB in all), and the scorer recomputes pooled
centred CKA per resample from them. Head contributions (O, reported only) are reduced to point values and are not kept.
Peak memory is recorded in the pilot.

**Records.** Each checkpoint's rows go to a temporary file renamed into place when complete; each file carries the
checkpoint's commit hash, the text set, a row-schema version and the code's git hash, and a resumed run skips a file only
when all four match.

## Controls

Gate table (each gate stops what its row says, nothing more):

| Gate | Check | Fails when | Stops |
| --- | --- | --- | --- |
| G1, the seed split | D1 to D3 at step 0 equal R0 at step 0; W1 and W2 at step 0 differ from R0 at step 0 and from each other; compared over the same key set after casting to float64, learned parameters only (rotary frequencies and attention masks excluded); the key sets must match. Because different tensors do not by themselves show an independent initialisation (a continuation also differs, Alden #6), the W half also requires, as an alarm and not a proof of independence, for every 2-D weight matrix, a Pearson correlation of absolute value below 0.05 between W's step 0 and R0's step 0, between W's step 0 and R0's step 143000, and between W1's and W2's step 0. The W half is a provenance check, not a formality: the uncorrected weight-seed releases would fail it. | any D tensor differs from R0; any W run equals R0 or the other W run in every tensor; any listed correlation reaches 0.05; or the key sets differ | the D half: 211.2 and 211.3; the W half: 211.1, 211.2 and 211.8, and the headline |
| G2, the measures | Zero controls: a repeated forward pass of R0 at step 143000 gives P, M, O and V of exactly 0, and R0's final heads, permuted at random within each layer (with their output-projection blocks), are matched back at matched cost 0. Duplicate-head control: a synthetic layer in which two heads have identical rows passes when the solver returns any assignment of matched cost 0, whichever permutation it picks. Nonzero controls, so a measure that always returns 0 cannot pass: two attention rows with disjoint support give P = 1 within 1e-6; one head's OV block doubled gives V = 1 within 1e-6 for that head; independent Gaussian features of one layer's recorded shape give M of at least 0.5; and the same features after a random orthogonal rotation and an isotropic scaling give M = 0 within 1e-5 (CKA's invariances). | any zero control is not 0, the duplicate-head control returns a nonzero matched cost, or any nonzero control misses its value | all predictions |
| G3, the floor per layer | The noise floor (Method) against the quantity each prediction compares, by the per-prediction rules below. | the rule below says unmeasured | that layer, for that prediction (it counts against the prediction's bar) |

**Per-prediction rules** (Alden #6, finding 4: each prediction is gated in its own units). "Interval" is the bootstrap
interval of the recomputed statistic (Uncertainty); "floor" is the layer's noise floor for the measure named.

- 211.1: per run b, a layer counts when the 97.5th percentile of s* over resamples is at or before step 16000
  (censored resamples count as later than 16000). Unmeasured when the P floor is at least a quarter of the matched
  cross-run P at step 143000 between b and R0.
- 211.2: per W run k, the difference median over D of the matched distance minus W_k's matched distance; a layer counts
  when both differences have a 97.5th percentile below 0 and a point value below minus the floor. Unmeasured when the P
  floor is at least a quarter of either W run's matched distance.
- 211.3: a head is same-index compatible when its point gap g_i is at most the floor; the interval of g_i is printed
  beside it. The gap can correctly be 0, so it is not required to exceed anything. Unmeasured (the layer's 12 heads
  count as not compatible) when the P floor is at least a quarter of the mean P between different heads of that layer in
  R0's final model, because compatibility cannot then discriminate.
- 211.4 and 211.5: per run, the layer counts for that run when t_P < t_X (the rule in Measures) holds at the point
  estimate and in at least ceil(0.975 B) of the B paired resamples (975 of 1,000, or 195 of 200 under the M fallback).
  Unmeasured when either denominator X(step 0, final) is at or
  below four times its floor.
- 211.6: the four distances are resampled jointly; the layer counts when the 97.5th percentile of the P ratio minus the
  M ratio is below 0. Unmeasured when P(3T, 4T) or M(3T, 4T) is at or below four times its floor at the point estimate,
  or its 2.5th percentile is at or below the floor; no epsilon or cap is substituted.
- 211.7: the layer counts when the 97.5th percentile of f_P at 21B is at most 0.5. Unmeasured when P(step 0, 4T) is at
  or below four times the floor.
- 211.8: per W run, the difference R0's within-run P(1000, 10000) minus the matched cross-run P at step 10000; the layer
  counts when both differences have a 97.5th percentile below 0 and a point value below minus the floor. Unmeasured when
  the P floor is at least a quarter of either cross-run P.

Matched random controls: not used as such, because nothing is pushed, and the only fitted step is the head assignment,
refit in every set and every resample. The scale references printed beside
every cross-run distance are the noise floor, the step-0 cross-run distance (zero for D runs by construction, nonzero for
W runs), and the mean P between different heads of the same layer in the same final model.

Off-target and broken answers: not applicable, no generation.

Select on one set and test on another: nothing is selected beyond the assignment, which each set fits for itself. Set B
checks that the verdicts do not depend on the documents; a verdict counts as held only on set A, with set B's value
printed beside it.

Not used: gradients or any test of the credit-ratcheting mechanism itself, because this part measures what training
did, not why. Not used: training runs of our own, because public checkpoints cover the comparison.

## Size

- Smallest effect worth knowing: whether the thread's ordering holds at full training length, a per-layer comparison
  with document-bootstrap intervals; the sample is 12 Pythia layers with two weight-seed and three data-seed runs, and
  16 OLMo layers.
  Every bar below is a count of layers whose interval clears it.
- Total: 6,192 document passes, about 600 minutes on the host CPU at the slow end of the estimate below: 72 Pythia
  checkpoints and 14 OLMo checkpoints, each run forward on 72 documents of 256 tokens (48 in set A, 24 in set B), that
  is 5,184 Pythia and 1,008 OLMo document passes, no generation. Forward compute is about 2 x parameters x
  tokens: 5,184 x 2 x 162.3e6 x 256 = 431 TFLOP for Pythia and 1,008 x 2 x 1.485e9 x 256 = 766 TFLOP for OLMo, 1,197
  TFLOP in all. At the host's measured best-case 353 GFLOP/s (a 2048-cubed float32 matrix product, measured by the
  design reviewer) that is at least 1 hour; at a realistic 10 to 25% of that, 4 to 10 hours. The pilot measures the real
  rate and replaces this estimate.
- Downloads: 72 Pythia checkpoints listed by the hub at 375 MB each (about 27 GB; their dtype is read in the pilot)
  and 2 OLMo checkpoints of 5.9 GB (11.9 GB), about 39 GB to E:. Kept records, all float32: about 73 GB of reference
  attention rows and 27 GB of MLP outputs, plus per-document distance files, about 100 GB on E: (292 GB free on
  2026-11-13). The pilot records the real sizes; a projected total over 150 GB is a stop.
- Pilot: R0 at steps 0 and 143000 and W1 at step 143000, on set A: times one forward pass and one distance pass, records
  peak memory, runs G2 in full, and bootstraps one layer's P and M to report the interval width. It also downloads one
  checkpoint while watching that the Gmail relay's IMAP connection stays up (`ss -tnp | grep :993` before, during and
  after). Stop rules: a projected total over 16 hours of compute, peak memory over 8 GB, an interval on P(R0 final, W1
  final) wider than half its point value, or the relay connection dropping during the download. A stop is a hold until
  reviewed.

## Optimization

- Bound resource: the download line, then CPU.
- Measured: not yet; the pilot records the download rate in MB/s, the seconds per Pythia and per OLMo checkpoint
  forward, and the seconds per distance pass.
- Levers: no generation, taken; float32 on the CPU, taken (no A770: the faulty bf16 kernel question does not arise, and
  Windows memory is not touched); set B at half the size of set A, taken; OLMo's 12 cached checkpoints reused, taken;
  attention rows reduced to distances as each checkpoint is read, with only the reference checkpoints kept, taken;
  cross-run matrices one layer at a time, taken; downloads rate limited and run on the Windows side if the pilot shows
  WSL downloads disturb the relay (gotcha, 2026-10-22), taken; each checkpoint's tensors copied into ordinary memory after
  loading from `/mnt/e` (gotcha, 2026-11-08), taken; resume through the atomic records above, taken. The A770 in float32
  eager attention is the lever if the CPU projection exceeds 16 hours; declined until the pilot measures it.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 211.1 | Pythia-160m, weight-seed runs W1 and W2 against R0: the crossing step s* is at or before step 16000 (11% of training, the thread's 1,000 of 10,000) in at least 9 of 12 layers, counting a layer when s* is at or before 16000 in both runs. | medium | Claude (Opus 5.5) |
| 211.2 | Pythia-160m at step 143000: the matched cross-run distance to R0 is smaller for data-seed runs (same initial weights) than for weight-seed runs (same data order), counting a layer when the median over D1 to D3 is below the distance of each of W1 and W2, in at least 9 of 12 layers. | medium | Claude (Opus 5.5) |
| 211.3 | Pythia-160m at step 143000: at least 108 of 144 heads are same-index compatible in each of D1 to D3 (compatibility, not identification; unique and ambiguous counts printed beside it). | medium | Claude (Opus 5.5) |
| 211.4 | Pythia-160m: the pattern settles before the MLP output, t_P earlier than t_M, in at least 9 of 12 layers, counting a layer when t_P < t_M in at least 5 of the 6 runs. | low: toy-setting theory (Prior art) has the MLP learning first, so this is predicted against it | Claude (Opus 5.5) |
| 211.5 | Pythia-160m: the pattern settles before what the head writes, t_P earlier than t_V (both means over the layer's heads), in at least 9 of 12 layers, counting a layer when t_P < t_V in at least 5 of the 6 runs. | low | Claude (Opus 5.5) |
| 211.6 | OLMo 2 1B, relative to a trillion tokens of late pretraining: P(mid-training end, Instruct) / P(3T, 4T) is smaller than M(mid-training end, Instruct) / M(3T, 4T) in at least 12 of 16 layers, P averaged over each layer's heads. | medium | Claude (Opus 5.5) |
| 211.7 | OLMo 2 1B, final = end of pretraining (4T): f_P at pretraining 21B (step 10000, 0.5% of pretraining) is at most 0.5 in at least 12 of 16 layers. | medium | Claude (Opus 5.5) |
| 211.8 | Pythia-160m, the thread's own comparison at its own horizon: R0's within-run P(step 1000, step 10000) is below the matched cross-run P at step 10000 between R0 and a weight-seed run, by more than the floor, in at least 9 of 12 layers, counting a layer when it holds for both W1 and W2. | low: step 1000 is inside Pythia's 1,430-step warmup, where the thread's run was not, and Bali et al. find middle layers least stable across seeds | Claude (Opus 5.5) |

Copied verbatim into PREDICTIONS.md under "## Part 211 (registered ...)" in the registration commit.

## Reading rules

`part211/score211.py` reads only the distance files and prints every per-layer value behind each verdict with its
bootstrap interval and noise floor, on set A for the verdict and set B beside it, P0 beside P for 211.1 to 211.3 and 211.8, and the cosine measure and greedy match beside every cross-run value.
Per-head tables are written beside the per-layer ones, and no verdict is led with a mean over layers (Bali et al.
find the middle layers differ from the outer ones). A
gate stops exactly what the gate table says. No reader and no generated text, so there are no error rows or broken
answers.

211.8 is the direct replication: the thread's comparison, at the thread's horizon, with heads matched by a stated
method. 211.1 and 211.2 together are the headline at full length: if 211.1 holds, the thread's ordering survives full-length training with
heads matched; if 211.2 also holds, initial weights decide the final patterns more than data order does, for these seed
draws. If 211.1 fails, the report gives each layer's crossing step, and whether the thread's ordering holds at Pythia's
step 1000 (descriptive). 211.4 and 211.5 order settling times, not amounts, and only for these observables at this
threshold: P, M and V are different quantities, an earlier settling time does not mean a smaller change, and no result
here says how much useful learning attention or the MLPs do. 211.6 compares each measure's post-training change with
its own change over the last trillion tokens of pretraining; the absolute values are printed beside it. 211.3 is read
as compatibility: a pass does not say particular heads kept an identity. The headline is withdrawn if G1 or G2 fails,
or if set B reverses 211.1 or 211.2 by more than one layer.

## Limits

One model size for the seed questions, with two weight-seed and three data-seed runs; "initial weights against data
order" is
answered for these draws, which are not calibrated perturbations of equal size. One OLMo run with no seed variation. The
text is from the Pile, which both families trained on, cut to 256 tokens; Bali et al. find cross-seed stability falls
with prompt length, so the absolute distances belong to this length. P measures where heads look on 256-token documents, not what a
head computes; M and O are compared up to rotation (CKA); V is a weight distance that can move without a functional
change. Settling fractions and times use each measure's own scale, so they order measures in time only, for these
observables. Matching assumes one head corresponds to one head; when heads in a layer share a pattern, several
assignments are equally good, which is why compatibility is read from the distances, and the matched distance then
understates real differences. The thread's head-averaging result does not show that heads share patterns (loss can be
insensitive to routing when value paths are redundant), so shared patterns are a case this part measures, not a fact it
assumes. The MLP output also changes when its inputs from earlier layers change, and V is a weight measure; neither
orders how much useful learning the mechanisms do, so this part cannot establish that later training is mostly the
MLPs. Twelve layers are breadth within one model, not twelve independent training runs. Nothing here tests the
credit-ratcheting mechanism or the thread's NATQ fix.

## Changes before registration

- 2026-11-12 (Claude, after the DeepSeek design review `reviews/cheap-p211-design-ds-r1.md` in the Journal repository,
  before any data): compute estimate replaced by the FLOP arithmetic and a measured-rate stop rule, set B halved to 24
  documents (F1, F15); identity read from the distances, not the solver (F2); 211.6 normalised by each measure's own
  late-pretraining change instead of its whole pretraining change (F3); settling times, with persistence, replace a
  single-checkpoint comparison (F4, F6); CKA centred and the M fraction described as a dissimilarity ratio (F5, F16);
  medians and two-of-three counts instead of means over three runs (F7); document-bootstrap intervals for every layer
  value a bar reads (F8); reference checkpoints listed per family and cross-run matrices one layer at a time (F9); the
  noise floor applied per layer to every prediction (F10); one gate table (F11); G1's comparison specified (F12);
  scipy required (F13); a pattern-free OV measure for 211.5, with the composite O reported only (F14); the matched-over-
  index clause dropped from 211.3, since index distance is never below matched (F17); atomic records with a four-field
  resume check (F18); the step-1000 and step-16000 matrices marked descriptive and step-0 distances printed (F20).
- 2026-11-13 (Claude, after Muse's prior-art sweep, council `muse-part211-prior-art-2026-11-12` #3, before any data):
  Prior art rewritten from the sweep; the Gupta et al. quotation corrected to what the abstract says (Muse caught it,
  checked at source); the post's matching and date corrected; step 10000 added to every Pythia run and prediction 211.8
  added, the thread's own comparison at its own horizon; Bali et al.'s cosine measure and greedy match reported beside P
  and the Hungarian match; s*_M reported beside 211.4 as a shared yardstick; 211.4's confidence reason states that
  theory points the other way; OLMo's Instruct named as post-RLVR; per-head tables and the prompt-length limit added.
  Muse's trap that PolyPythias seeds confound initial weights with data order applies to seed1 to seed9, which this part
  does not use.
- 2026-11-13 (Claude, after Alden's provenance hold, council `part211-design-2026-11` #4, before any data): the original
  weight-seed runs do not vary initialisation (publisher's card: a training error continued them from the original
  Pythia weights with incorrect Adam buffers). The weight arm is now the two corrected runs (`-seed1-corrected`,
  `-seed2-corrected`); there is no corrected seed 3. Counting rules that read W changed from "2 of the 3 runs" to "both
  runs", and the 7-run rules from "5 of 7" to "5 of 6", both stricter, written before any data; 211.2 now requires the D
  median to be below each W run. G1's W half is a hard provenance gate (a W run equal to R0, or to the other W run,
  stops 211.1, 211.2, 211.8 and the headline). Size recomputed for 72 Pythia checkpoints. My earlier reply to Muse
  dismissed their confound because the weight-seed repositories existed with 155 branches; I had not read the cards.
- 2026-11-13 (Claude, after Alden's full review, council `part211-design-2026-11` #6, copied as
  `alden-review-2026-11-13.md`, sha256 0f37f59f..., before any data): the scorer contract frozen. Revisions are pinned
  before registration and committed with it. G1's W half also requires near-zero correlation of every weight matrix with
  R0 at steps 0 and 143000 and between W1 and W2, since different tensors alone do not show independent initialisation.
  "Identity-preserving" renamed "same-index compatible", read as compatibility, with unique and ambiguous counts beside
  it. G2 gains a duplicate-head control that accepts any zero-cost assignment, and nonzero controls with known values
  for P, V and M, so a measure that always returns 0 cannot pass. The bootstrap now resamples the whole statistic with
  shared document IDs, refits the assignment inside every resample, and recomputes crossings, settling times and
  medians per resample; set B fits its own assignment; V's interval is stated as degenerate. Per-document records keep
  what that needs (12-by-12 cross-run P per document, and every checkpoint's MLP outputs for pooled centred CKA). The
  noise floor now runs through the float16 storage path; P0 drops near-one-key rows by a stated rule. G3 is a rule per
  prediction in its own units, with unresolvable denominators marked unmeasured. Settling fractions are ratios of head
  means; interpolation is in log(1 + step); censoring and the t_P < t_X rule are stated; orderings are "with these
  observables", with threshold and square-root-of-JS settling times printed descriptively. The data-seed arm is
  described as batch composition, not only order (PolyPythias section 2). Limits: the head-averaging result does not
  show shared patterns, MLP outputs include upstream changes, layers are not independent runs, and the part cannot show
  later training is mostly MLPs. Alden's outcome-unexposed forecasts, recorded as his and not as rows: 211.1 and 211.2
  lean hold, 211.3's compatibility count more readily than unique identity, weakly 211.5 and 211.7, no direction for
  211.4 or 211.6.
- 2026-11-13 (Claude, after Alden's narrow hold, council `part211-design-2026-11` #8, review file
  `part211-78228f34a-design-review-2026-11-13.md` sha256 db9e58e1..., before any data): the M fallback is a paired
  B = 200 using the first 200 document draws for P and M alike, every threshold is ceil(0.975 B), and an unmeasured or
  censored draw counts as a failure, never dropped. Stored measurement inputs are float32, so the floor no longer
  claims to bound a storage error it could not see; storage estimates rewritten (about 100 GB on E:, a stop above 150 GB).
  Size and Limits say two weight-seed and three data-seed runs. G1's correlation check is an alarm, not a proof.
