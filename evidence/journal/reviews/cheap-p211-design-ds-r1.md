<!-- cheap-review.mjs cheap-review-p211-design-ds-r1-2026-11-12: /home/elisepc/green/Journal 0c3ed3449~1..0c3ed3449, lane ds, df-run exit 0 -->
# REVIEW — Part 211 design (commit 0c3ed3449)

Reviewer: independent cold review (DS), 2026-11-12. Artifact: one new file,
`notes/reports/part211-attention-settling-2026-11-12/PART211-DESIGN.md` (185 lines, +185, nothing
else in `git diff --stat 0c3ed3449~1..0c3ed3449`). No code, no data, no model has run. Claim under
test: `sources/web-archives/2026-11-12-hyperstition-attention-ratchet.md` (hyperstition_cc,
2026-11-11: "by the first thousand steps each run's attention resembles its own step-10k pattern
MORE than the individual finished runs resemble each other", 10k steps, model size and head
matching unstated).

**VERDICT: SHIP-WITH-FIXES.** The question, the 2×2 seed design and the two reference floors are
sound, and nothing here needs new data. But two of the seven predictions can be decided by the
matcher or by an unrelated denominator rather than by attention (F2, F3, both measured below), and
several thresholds have no noise margin (F6). All are definition/orchestration fixes to a
pre-registration document; register only after the must-fix list at the end.

## Severity key
HIGH = a verdict can be wrong, or a prediction passes for the wrong reason.
MEDIUM = one plausible implementation or one noise draw changes a verdict.
LOW = real but bounded; follow-up.
Every finding is INTRODUCED (the whole file is new). Scripts are in this directory; outputs pasted.

## Findings

### F1 (MEDIUM, INTRODUCED) — "under an hour on the CPU" is below the FLOP lower bound on this host
`PART211-DESIGN.md:126` estimates compute at "under an hour on the CPU"; `:139` explicitly declines
the A770 and fixes float32 CPU. The workload is fixed by `:124-125` (7,392 Pythia + 1,344 OLMo
document passes at 256 tokens). Forward FLOPs ≈ 2·N·T:

- Pythia-160m (162.3M params): 7,392 × 2 × 162.3e6 × 256 = **614 TFLOP**
- OLMo 2 1B (1.485B params): 1,344 × 2 × 1.485e9 × 256 = **1,022 TFLOP**
- Total **1,636 TFLOP**

Measured on this host (Ryzen 7 9700X, 16 threads, numpy/OpenBLAS, best-case 2048³ sgemm):
```
$ python3 exp_compute.py
numpy 2.5.1 sgemm 2048x2048: best 48.6 ms -> 353 GFLOP/s (peak-ish BLAS)
Pythia-160m: 7392 doc-passes x 256 tok = 614.3 TFLOP
OLMo 2 1B : 1344 doc-passes x 256 tok = 1018.4 TFLOP
total      : 1632.7 TFLOP
  at    353 GFLOP/s -> 1.3 h
  at     88 GFLOP/s -> 5.1 h
  at     35 GFLOP/s -> 12.8 h
```
46×46 sgemm is BLAS's best case; a 256-token forward is a chain of small GEMMs plus LayerNorm,
softmax and the forced eager attention (`:52`), realistically 10–25% of BLAS peak → **5–13 h for
the forwards alone**, before 40.8 GB of loading and ~35 G element-ops of JS. Even at 100% of the
measured BLAS peak the estimate is 1.3 h. This matters because the pilot's 5-minute-per-checkpoint
stop rule (`:130-132`) was presumably set from it.
**Smallest fix:** delete the total; report the measured floor (≥1.3 h at BLAS peak) and the pilot's
measured s/checkpoint.

### F2 (HIGH, INTRODUCED) — 211.3's "assignment is the identity" is a tie-break artefact when heads share patterns
`PART211-DESIGN.md:153` requires the min-cost assignment to be the identity for ≥108/144 heads in
each D run. But `scipy.optimize.linear_sum_assignment` returns *some* optimal assignment; when heads
in a layer share a pattern, every within-cluster permutation is equally optimal, and the identity is
not preferred. The design's own Limits (`:180-182`) and prior art (`:31-33`) say heads do share
patterns and that averaging across heads barely changes loss, so this is the expected case, not a
corner. Measured (12 heads in 4 identical triples; within-cluster P=0.01, across=0.30, tiny
tie-breaking noise):
```
$ python3 exp_matching.py
=== (1) identity matches returned by linear_sum_assignment, heads in 4 identical triples ===
  draw 0: identity heads = 3/12, cost = 0.120000
  draw 1: identity heads = 3/12, cost = 0.120000
  draw 2: identity heads = 5/12, cost = 0.120000
  draw 3: identity heads = 3/12, cost = 0.120000
  draw 4: identity heads = 7/12, cost = 0.120000
  draw 5: identity heads = 5/12, cost = 0.120000
  cost of the identity assignment = 0.120000 (same to 6 dp)
```
The reported "identity fraction" varies 3/12–7/12 across 1e-9 noise while the cost is exactly the
identity's cost. The prediction would fail on this artifact and the report would say data order
permuted the heads — a false finding.
**Smallest fix:** do not read the solver's permutation. Score the identity pairing directly: count a
head as identity-preserving when `P(i,i) - min_j P(i,j) <= floor` (the G2 bf16 floor), and report
whether the identity assignment's total cost equals the optimum within that floor; also report how
many heads are uniquely matched. Encode the same rule in `score211.py` before registration.

### F3 (HIGH, INTRODUCED) — 211.6 can pass while Q4's answer is "no": the ratio is driven by the pretraining denominators
`PART211-DESIGN.md:156` compares `P(51B,Instruct)/P(step0,51B)` with
`M(51B,Instruct)/M(step0,51B)`. The two denominators are different quantities on different scales
(JS of attention rows vs 1−CKA of MLP outputs); their relative size, not the post-training change,
can decide the verdict. Constructed case, both numbers plausible if 1−CKA stays high across
pretraining:
```
$ python3 exp_settling.py
=== (3) 211.6 ratio form vs the question Q4 asks ===
absolute: P_post=0.02  M_post=0.005  -> attention moved 4x MORE
ratios  : P_post/P_pre=0.0333  M_post/M_pre=0.1000  -> P<M: True  (211.6 PASSES)
```
Here attention moved four times *more* than the MLP in absolute terms, and the prediction passes
because attention's pretraining denominator is 12× larger. This is not a corner: a high-entropy JS
over 256 keys can be large while 1−CKA between MLP outputs stays near 1 (shared mean component), so
`P_pre >> M_pre` is the likely state.
**Smallest fix:** define 211.6 on the absolute post-training changes `P(51B,Instruct)` and
`M(51B,Instruct)`, with the measured OLMo bf16 floor as the noise margin; report the ratio form
beside it as context, not as the verdict. If a ratio is kept, say in the prediction that it is a
change-relative-to-pretraining claim and score both.

### F4 (MEDIUM, INTRODUCED) — 211.4/211.5 compare one time point, which does not order settling times
`:154-155` read "settles earlier" from `f_P(4000) < f_M(4000)`. `f_X(s)=X(s,final)/X(0,final)` is the
fraction of total change remaining; a single point on that curve does not order when the curves cross
half. A late phase change in attention (the phenomenon the part is about) makes the two disagree:
```
f_P(4000)=0.40  f_M(4000)=0.45  -> 211.4 prediction direction: f_P<f_M = True
t50_P=3564  t50_M=2000  -> P settles LATER = True
```
(Synthetic curves on the design's own 11-checkpoint grid; P is flat then drops between 2000 and
4000, M decays smoothly.) The prediction would pass while the answer to Q3 is the opposite.
**Smallest fix:** define settling as the interpolated checkpoint where `f_X` first reaches 0.5 (or
report the whole `f_X(s)` curve and require the ordering at every checkpoint in a stated window);
predict on that, keep the step-4000 values as context.

### F5 (MEDIUM, INTRODUCED) — 1−CKA is not a metric, so `f_M` is not "a fraction of the total change"
`:80-84` calls `f_X` a fraction of a measure's own total change. That is meaningful for the JS
divergence; it is not for 1−CKA, which is not a distance (no triangle inequality, no additivity, can
be dominated by a shared mean/bias). The fraction can exceed 1 — i.e. the MLP at s is *further* from
final than the random init was:
```
$ python3 exp_settling.py
=== (2) 1-CKA settling fraction can exceed 1 ===
1-CKA: [0.05 0.07 0.08 0.085 ... 0.0]
f_M at step 4000 = 1.84 (>1 means s=4000 is FURTHER from final than step 0 is)
```
A prediction of the form `f_P < f_M` then passes because `f_M` blew up, not because attention
settled.
**Smallest fix:** state that `f_M`/`f_O` are ratios of similarities, not path fractions; cap or flag
`f>1`; and specify whether linear CKA centres features (the design does not, and centring changes
the number a lot when the mean dominates — a builder can implement both ways).

### F6 (MEDIUM, INTRODUCED) — no threshold has an absolute noise margin, and s* does not require the crossing to persist
`:89-90` counts the first checkpoint where within-run P is below the cross-run P; any dip by 1e-6
counts. `:152` is a strict inequality between two means; `:153` asks for a 10% *relative* gap with no
absolute floor. Measured on the 10% rule:
```
=== (2) 211.3 W-clause: 10% relative bar at a noise-scale distance ===
  matched=1.0e-06  index=1.2e-06  -> index > 1.10*matched: True
  both are 100x below a plausible bf16 floor (1e-4); the bar passes on pure noise
```
Two independent problems: (a) a crossing at step 128/512 that reverses by step 1000 still sets
s*=128 and counts for 211.1 — the log-spaced early checkpoints make this likely; (b) 211.2 and the
W-clause of 211.3 can be decided below the G2 floor, which is only *withdrawn* on a 25% rule
(F10), not applied as a margin.
**Smallest fix:** require the crossing to hold at the next checkpoint (or at every later one) and to
exceed the layer's bf16 floor; for 211.2/211.3 require `|difference| > floor` as well as the 10%.

### F7 (MEDIUM, INTRODUCED) — means over three runs in a suite whose headline is outliers
`:152-153` (and `:154-155`) average over W1–W3 / D1–D3 / all seven. The cited suite is literally
"PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs" (`:39-43`), and
one outlier run moves a mean over three by a third. A single degenerate W run inflates the W
distance and decides 211.2 in the predicted direction; a single outlier D run does the reverse. One
run does it arithmetically: distances 0.10, 0.10, 0.40 give a mean of 0.20 and a median of 0.10
(`mean(0.10,0.10,0.40)=0.20  median=0.10`), a 100% shift in the arm statistic.
**Smallest fix:** report per-run values and score the layer on ≥2 of 3 runs (as 211.1 already does)
or on the median; do not let one run set the mean.

### F8 (MEDIUM, INTRODUCED) — no uncertainty on any layer-level value
The design reports counts of layers over 48 documents / 12,240 rows and three runs, but no standard
error or interval for any per-layer mean, and no bootstrap. `:121-123` calls the sample "12 layers by
3 runs", which is the count of *comparisons*, not of independent evidence. The G2 floor is a
measurement floor, not sampling noise. A "9 of 12" verdict can be 8 or 10 under document resampling,
and set B does not fix that (it is another 48 documents, not an interval).
**Smallest fix:** in the pilot, bootstrap documents (and runs) for one layer's P and M; report the
interval beside every per-layer value and require the threshold to clear it; specify the bootstrap in
`score211.py` before registration. Concretely, if a document's mean P has SD 0.05, the SE of the
48-document mean is **0.0081**; at SD 0.10 it is **0.0135** — the same order as the between-arm
differences 211.2 hopes to resolve:
```
per-document SD 0.02 -> SE of 48-doc mean = 0.0028
per-document SD 0.05 -> SE of 48-doc mean = 0.0081
per-document SD 0.10 -> SE of 48-doc mean = 0.0135
```

### F9 (MEDIUM, INTRODUCED) — "two reference checkpoints per run" cannot serve OLMo, and the naive memory is near the 8 GB stop
`:140-142` keeps "only the two reference checkpoints per run (step 0 and final)" in float16. For
OLMo, the two questions have *different* finals (`:83-84`: 4T for pretraining, Instruct for
post-training), and 211.6 additionally needs 51B as its base, so the reference set is
{step0, 51B, 4T, Instruct} — four, not two. A builder following the optimisation literally discards
rows 211.6/211.7 need. Measured storage on the design's own row count (48 docs × 255 queries × 256
keys, fp16):
```
$ python3 exp_storage.py
=== attention-row storage, fp16, 48 docs x 255 queries x 256 keys ===
  Pythia (12x12): 0.90 GB per checkpoint; 7 runs at one step = 6.32 GB; 14 refs on disk = 12.63 GB
  OLMo   (16x16): 1.60 GB per checkpoint; 4 refs (step0, 51B, 4T, Instruct) = 6.42 GB
  naive cross-run matrix at one step (7 checkpoints live, Pythia) = 6.32 GB + model
```
6.32 GB live + 0.65 GB model + activations approaches the 8 GB stop rule (`:131`), so the
implementation choice (all runs at a step vs layer-by-layer) decides whether the pilot stops.
**Smallest fix:** state the reference set per family explicitly, require a layer-by-layer loop for
the cross-run matrices (12 heads × 48 × 255 × 256 × 2 B = 75 MB per run per layer), and put the
resulting peak in the pilot's record.

### F10 (MEDIUM, INTRODUCED) — the noise floor is a gate for 211.1–211.3 only, and only on R0; 211.6 has no floor rule
`:102-103` withdraws a layer whose bf16 floor is ≥25% of its matched cross-run distance, "from 211.1
to 211.3". `:182-184` admits OLMo's Instruct is stored bf16 and only says its floor "is reported
beside 211.6". So 211.6/211.7 can be scored on dtype noise, and 211.4/211.5 (Pythia) are outside
the withdrawal too even though their fractions use the same P/M distances. The floor is also
measured on R0 only, then applied to W/D runs. My synthetic head at trained-weight scale gives a
bf16 weight-rounding floor of ~1.9e-6 JS per row (a 1% weight change gives ~6.8e-5), so the floor is
small in absolute terms — but `P(51B,Instruct)` may be small too, and the design has no rule for
that ratio.
```
  trained-scale weights    bf16-weight floor=1.86e-06   1%-weight-change=6.84e-05   fp16-row floor=7.22e-09
```
**Smallest fix:** make the floor a withdrawal rule for every prediction that uses P/M/O, measured
per layer on the same dtype pair as the comparison (fp32 vs bf16 for OLMo), and apply it to the
numerator of 211.6.

### F11 (MEDIUM, INTRODUCED) — gate semantics contradict each other
`:117` says "G1 and G2 must pass before any prediction is read"; `:96` says G1 failure stops only
211.2/211.3; `:164` says "A failed gate stops the predictions it names"; `:165-166` says a layer
withdrawn under G2 counts against "a '9 of 12' bar", which includes 211.4/211.5 even though the
withdrawal was defined only for 211.1–211.3. An implementer can stop everything, stop two
predictions, or score four on withdrawn layers.
**Smallest fix:** one table: gate → predictions stopped / layers withdrawn, and use it in both
Controls and Reading rules.

### F12 (MEDIUM, INTRODUCED) — G1's "bit-identical in every weight tensor" is format-sensitive, and its W half is near-vacuous
`:94-96`: D step 0 must be bit-identical to R0 in *every* weight tensor, and W step 0 must differ in
*every attention weight tensor*. Two different repos can hold the same init in different dtypes
(EleutherAI Pythia is fp16; PolyPythias may be fp32) or with different key sets (`embed_out` vs
`lm_head`, tied vs untied), so a `torch.equal` implementation fails on the format, not the weights,
and kills 211.2/211.3. Conversely, two independent random inits differ in every float tensor with
probability 1, so the W half cannot fail. Buffers such as `rotary_emb.inv_freq` are seed-independent
and would spuriously fail the W rule if counted as "attention weights".
**Smallest fix:** enumerate the exact tensor keys compared, compare after casting both to float64
(and after checking the key sets match), exclude non-learned buffers, and say explicitly that the W
half is a sanity check, not a test.

### F13 (MEDIUM/LOW, INTRODUCED) — "or numpy if scipy is absent" is an undefined algorithm
`:86-87` allows `scipy.optimize.linear_sum_assignment` "or numpy if scipy is absent". numpy has no
assignment solver; a builder must hand-roll one (greedy or Hungarian), and greedy gives different
matched distances:
```
$ python3 exp_matching.py
=== (3) greedy fallback vs optimal assignment ===
  optimal pairs: [(0, 1), (1, 0), (2, 2)]  matched mean = 0.38666666666666666
  greedy  pairs: [(0, 0), (1, 1), (2, 2)]  matched mean = 0.39  -> 211.2/211.3 number changes
```
That difference is enough to flip a 10% bar or a strict inequality. scipy 1.18.0 is present on this
host.
**Smallest fix:** require scipy and abort with a clear error if absent (a dependency is cheaper than
an unspecified second algorithm), or write the exact fallback (Hungarian, deterministic tie-break)
into the design.

### F14 (MEDIUM, INTRODUCED) — `O` includes the pattern, so 211.5 does not separate "where" from "what"
`:70-71` defines a head's contribution as attention output × its output-projection block, so
`O = a·V·W_O` changes whenever the pattern `a` changes. 211.5 (`:155`) reads "attention settles
earlier than what each head writes" from `f_P < f_O`, but `O` cannot move independently of `P`; the
comparison only says the *product* keeps changing after `a` stops. That is consistent with the
head's values/output weights still changing, which is the intended reading, but the measure does not
isolate it.
**Smallest fix:** add a pattern-frozen variant (recompute `O` with `a` held at the final pattern) or
compare the value/output weights directly; at minimum say in the prediction that `O` contains the
pattern's effect and that the claim is about the composite.

### F15 (LOW, INTRODUCED) — the Pythia download size is not dtype-consistent; the total may be 25 or 50 GB
`:125-126` budgets 375 MB per Pythia checkpoint, 28.9 GB total. Pythia-160m is 162.3M params:
fp16 is 324.6 MB, fp32 is 649.3 MB; 375 MB is neither (2.31 bytes/param). 77 checkpoints are
therefore 25.0 GB at fp16 or 50.0 GB at fp32, not 28.9 GB. The OLMo figure does check out: I read the
cached Instruct config (16 layers, 16 heads, hidden 2048, vocab 100352, untied head) = 1.485B
params, fp32 5.94 GB ≈ the stated 5.9 GB.
**Smallest fix:** take the size from one measured checkpoint in the pilot (the design already
downloads one there) and write the dtype next to the number.

### F16 (LOW, INTRODUCED) — the 4,096-position sampling scheme is unspecified
`:71-72` keeps "4,096 token positions fixed by seed 211" from 96 × 256 tokens (24,480 non-zero
positions). 4,096/96 = 42.67, so "spread evenly per document" is not an integer; "first 4,096 of the
concatenation" would use only 16 documents; a random 4,096 of 24,480 is a third scheme. CKA depends
on which documents/positions enter, and two builders will choose differently.
**Smallest fix:** name the scheme and record the exact position indices in the run manifest.

### F17 (LOW, INTRODUCED) — "index distance exceeds the matched distance" is an identity, not evidence
`:153` (second clause) and `:87-88`: the matched distance is the minimum over one-to-one assignments
and the index pairing is one such assignment, so index ≥ matched always, for any two runs. Only the
10% magnitude is informative; a report of "index > matched in 12/12 layers" says nothing about head
permutation.
**Smallest fix:** state the identity in the prediction and score only the magnitude above the floor.

### F18 (LOW, INTRODUCED) — resume-by-skipping is unsound without an atomic row record
`:145` resumes "by skipping checkpoints whose distance rows are already written". A crash mid-row
leaves a partial file that looks written; a code change or a different set A/B silently reuses old
rows; a re-pinned commit hash is not checked against the row. This is the design's only failure path
and it is the one that can corrupt a verdict quietly.
**Smallest fix:** write each checkpoint's rows to a temp file and rename atomically; include the
checkpoint commit, the row schema version, the set (A/B) and the code git sha in the file; validate
all of them before skipping.

### F19 (LOW, INTRODUCED) — registration-day items the document asserts but this commit does not do
`:3` "Registered {{YYYY-MM-DD HH:MM}} CT" is the unsubstituted template; `:159` says the predictions
are "Copied verbatim into PREDICTIONS.md", which this commit does not do (the commit message calls
the file a draft, before reviews). If the file is registered as-is, the pre-registration record has a
placeholder timestamp and a copy that does not exist.
**Smallest fix:** at registration, fill the stamp and do the copy in the same commit; add the
check to the task's done-when.

### F20 (LOW, INTRODUCED) — unused cross-run matrices, unreported step-0 arm distance, uncalibrated arms
`:88` computes cross-run matrices at steps 1000, 16000 and 143000, but no prediction uses 1000 or
16000 (211.1–211.3 use 143000) — wasted work and a fold risk (a builder can naturally use the
step-1000 matrix for 211.1 and get a different s*). `:54-56`: D runs share R0's init, so the D-vs-R0
cross-run distance at step 0 is exactly 0 while W-vs-R0 is not; the design never reports the step-0
distances, so a reader cannot see that the arms start at different distances. And a "different init"
and a "different data order" are not calibrated perturbations: 211.2 answers "which factor wins for
*these* seed draws", not "which factor has more entropy".
**Smallest fix:** drop or justify the 1000/16000 matrices; print the step-0 cross-run distance beside
every cross-run table; add one Limits sentence on calibration.

## What I checked and found sound

- **The arms are as the design says.** I read the cached PolyPythias model card (read-only, nothing
  downloaded): `pythia-160m-data-seed{1-3}` are "only data ordering varied (weight init fixed)" and
  `pythia-160m-weight-seed{1-3}` are "only weight initialization varied (data order fixed)", with
  154 checkpoints including `step0` and the exact 11 steps the design lists. So the 2×2 is real and
  the model IDs are right; the only unverified link is that "fixed" init is R0's seed 1234, which
  G1 checks (and if it fails, the fix is to change the reference, not to drop the predictions).
- **Counts and arithmetic.** 7×11 = 77; 77×96 = 7,392; 14×96 = 1,344; 48×255 = 12,240 rows per head;
  4,000/143,000 = 2.80%; 16,000/143,000 = 11.2%; 21B/4T = 0.525%; 108/144 = 75%. `part85` has exactly
  12 checkpoints, and its `stage1-step10000-tokens21B` branch makes "21B (step 10000)" correct.
- **OLMo architecture and dtype.** The cached Instruct config is 16 layers, 16 heads, hidden 2048,
  `torch_dtype: bfloat16` — matches `:60` and the bf16 caveat at `:183`. 1.485B params fp32 = 5.94 GB.
- **float16 storage of reference attention rows is fine.** Measured JS after a float16 round-trip of
  synthetic softmax rows: **7.2e-9** per row (base 2) — three to five orders below any P of
  interest, so the `:141` float16 choice does not create the floor problem the bf16 weights do.
- **JS on attention rows measures what the question asks.** For peaked rows, one-hot at different
  keys gives JS = 1 and one-hot at the same key gives 0; the bound JS ≤ min(H) with 256 keys (max
  H = 8 bits) never binds for P∈[0,1]. It is "where the head looks" on a fixed key support, which is
  the thread's quantity.
- **P0's exclusions are right.** Query 0 sees only key 0, and after removing key 0 query 1 sees only
  key 1, so dropping both (queries 2–255) is the correct renormalisation.
- **G2's repeat=0 should pass.** A repeated forward in float32 on the CPU (same process, same thread
  count) is expected to be bit-identical, so P/M/O = 0 exactly. I could not run torch on this host
  (not installed), so this is reasoning, not a measurement; it is a sanity check, not evidence. The
  permutation test is the evidence: with distinct heads the planted pair has P = 0 by construction,
  so the matcher would recover it at distance 0 (F2 is about what happens when heads are *not*
  distinct).
- **211.2's shape is the right test.** Under the model card's one-factor-at-a-time construction, D
  vs R0 at 143000 isolates data order and W vs R0 isolates init; the prediction's direction is the
  thread's claim and it can fail. F6/F7 are about how it is *scored*, not about the comparison.
- **Out of scope / not applicable:** the commit adds no code, so the hunt list's failure paths,
  feature flags, tests-that-pass-for-the-wrong-reason and concurrency items do not exist here; the
  one failure path is the resume rule (F18). On "gates that cannot fail or cannot pass": G2's repeat
  and permutation checks can pass and are sanity tests (they cannot do more than that); G1's W-half
  cannot realistically fail (F12); G1's D-half can fail and its failure is informative, except that
  F12 shows it can also fail for a dtype/format reason that is not about the weights.

## Must-fix before registration
Verdict-changing (a prediction can be scored wrongly): **F2, F3, F4, F5, F6, F7, F8, F9, F10, F11,
F12, F13, F14, F18**. Stated-fact corrections: **F1** (compute time), **F15** (download size).
Clarifications: F16, F17, F19, F20. Nothing here requires a new download or a new model, and the core
design (Pythia seed 2×2 at full length, OLMo endpoints, JS + CKA + settling fractions with two
floors) is worth running.
