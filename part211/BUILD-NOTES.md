# Part 211: build notes (what the code does where the design's words leave a choice)

Written 2026-11-13 by Claude (Opus 5.5) for the manifest review, before any data. The design is frozen at Journal
0cb239092 (PART211-DESIGN.md, sha256 d6cd874c...); nothing here changes a prediction, a bar or a gate. Each item says
what the design says, what the code does, and why. Where an item could change a verdict it is marked **verdict-relevant**.

## Defects found in the first build, fixed before any data

1. **Same-index compatibility read the wrong cell.** `same_index_compatible` used `p_row[0]` as P(i, i) for every head,
   and `pred_211_3` read the whole per-run list for every layer. With 12 layers of 12 heads the wrong shapes gave the
   right totals in every test. Now the head index is required, each layer's 12-by-12 matrix is read, and the gap
   interval is P(i, i) minus the row minimum (it returned the row minimum).
2. **The bootstrap resampled single documents.** 211.1, 211.2 and 211.8 iterated over `np.ravel(idx)` (one document per
   draw) with the cross-run assignment held fixed; 211.6 and 211.7 took quantiles of per-document ratios, and 211.6 used
   per-document M. Every fixture was constant across documents, so no test could tell. Now every prediction recomputes
   its statistic inside each draw from document counts: means are counts-weighted, matched distances are refit on the
   draw's mean matrix, M is pooled CKA over the draw. Tests with document-varying fixtures fail the old behaviour
   (break-to-red, 8 of 8).
3. **Settling time interpolated log f.** The design says "interpolated linearly in log(1 + step)"; the code
   interpolated log(f). Now f is linear on the log(1 + step) axis. Both settling and crossing functions also refuse a
   curve without its final entry (they used to drop step 64000 for such a caller). **Verdict-relevant**: interpolated
   settling times move.
4. **The recorder would have written about 900 GB** against the design's 100 GB and its 150 GB stop: every head's
   [85, d_model] contribution, the full OV matrices, and per-document "moments" duplicating both, against the design's
   sentence that contributions are reduced to point values and not kept. See item 7.
5. **No noise floor and no gates were produced** anywhere. Both now exist (items 9, 11).
6. **A worker's comment recorded a non-existent environment defect** (a boolean `or` returning True on all-False
   operands). The comment and the workaround are gone.
7a. **P0 would have crashed the recording at the first sink head** (found 2026-11-13, second build pass). The recorder
   called `pattern_distance(..., "P0")` per document and head pair, which raises when every query of a pair is dropped
   (a head that keeps nearly all its mass on key 0, which is what sinks do). The design's rule, "a head with more than
   half its queries dropped has no P0", was not implemented at all, and the dropped count was not kept. Now P0 is
   stored as additive parts per document (the JSD summed over kept queries, the kept count, the queries per document),
   so any draw gets P0 and its dropped count exactly, and a pair over half dropped reads as no P0 (NaN) instead of an
   error. **Verdict-relevant for P0's columns only** (P0 is reported, not predicted on).
7b. **The cosine measure was per document, not flattened over the documents** as the design defines it (Bali et al.).
   Now it is stored as additive parts (dot product and each side's squared norm per document), and the cosine of any
   draw is the cosine of the matrices flattened over that draw's documents. Reported, not predicted on.

7c. **G2's M control could not fail, and was failing** (Alden #13, 2026-11-13). The nonzero controls were
   `(value, want, tol)`, and the check chose the approximate branch whenever `want` was finite, so M_independent's
   `(value, 0.5, inf)` accepted any finite M, including 0. Under it the control was also built at the wrong shape:
   85 positions by 768 features, where independent Gaussian features have nearly the same Gram matrix and M comes out
   about 0.098, so the control would have failed had the rule been right. Now each control carries an explicit rule
   (`("approx", want, tol)` or `("at_least", bound)`, a non-finite value fails either, a malformed rule raises), and
   M_independent is built at set A's pooled shape, 48 documents times 85 positions by 768 features (M about 0.84).
   Tests: an M-only sabotage with the other controls valid, finite pass and fail cases, NaN and infinity, and the shape.
7d. **Two recorded floors never reached the scorer** (Alden #13). `distances211.save` wrote the per-run Pythia floors
   and OLMo's mid-training-end floor, and `load` dropped both. `load` now restores them; the per-run floors decide 211.3
   (item 9) and the mid-training-end floor is printed beside 211.6's numerator.

## Interpretations (the design leaves the choice; the code's choice and why)

7. **O and V are point values against the run's final, defined as the run's last reference** (Pythia step 143000; OLMo
   Instruct). References keep each head's attention output before its output block (z, [85, head_dim]) and the value
   and output blocks; every other checkpoint is reduced to O_final and V_final as it is read. O = 1 - pooled linear CKA
   of z·W is exact from head_dim-sized products, and V = ||OV_a - OV_b|| / ||OV_final|| is exact from the factors.
   Storage: reference z about 8 GB, factors on references about 4 GB, so about 112 GB in all, under the 150 GB stop.
8. **M is resampled at B = 1000, not the B = 200 fallback.** Pooled centred CKA over a draw with repeated documents is a
   quadratic and cubic form in the document counts, computed exactly from the position Gram matrices (tested to 1e-9
   against direct CKA, including a 1e4 shared offset). Measured: about 1.1 s per layer and checkpoint pair at
   Pythia's size, about 20 minutes for all of Pythia. The fallback's trigger (a pilot projection over two hours) is
   therefore not expected to fire; if the pilot says otherwise, the fallback stays available.
9. **The noise floor, and which floor each prediction reads (frozen 2026-11-13 before any data; Alden #13).** Per
   layer and measure, each checkpoint in `floor_keys` against the same weights rounded through bf16 and run in
   float32: P is the mean over heads and documents of a head against itself, M is 1 - pooled CKA over all documents,
   V the mean over heads. The design says "the final (or compared) checkpoint". A larger floor is stricter under an
   unmeasured rule or a "below minus the floor" rule, and looser under an "at most the floor" rule, so the policy is
   set per rule, not as one conservative maximum (my first reading called the six-final maximum uniformly stricter;
   Alden showed it is not for 211.3):
   - **211.1, 211.2, 211.4, 211.5**: the per-layer maximum over the six runs' finals, for every unmeasured rule and
     every "below minus the floor" rule. A declared conservative tolerance: at least as strict as any compared pair.
   - **211.3**: the compatibility tolerance for D run b is the **smaller** of b's and R0's own final P floors (the two
     compared checkpoints; the stricter direction for "gap at most the floor"); unique and ambiguous use the same
     tolerance. The count at the six-final maximum is printed beside it (`n_compatible_at_max_floor`) as a sensitivity
     column and never decides. The discrimination gate (the floor against a quarter of R0's between-head P) keeps the
     six-final maximum.
   - **211.8**: the compared checkpoints are step 10000, so `floor_keys` now records step-10000 floors for R0 and each
     weight-seed run, and the floor for W run b is the **larger** of R0's and b's step-10000 P floors.
   - **211.6, 211.7**: OLMo 4T's floor (the final of the late distances and of 211.7). The mid-training end's floor is
     printed beside 211.6's numerator P(mid, instruct) and M(mid, instruct); no verdict reads it, and no numerator rule
     is added.
   Tests: 211.3 with Alden's case (gap 0.0002, own floors 0.0001, a weight run's 0.0003: not compatible at the
   tolerance, compatible at the maximum) and 211.8 with only one W run's step-10000 floor raised.
10. **OLMo is scored per text set**, set A for the verdict and set B beside it, as for Pythia (the first build scored
   OLMo once and copied it to set B).
11. **G2's permutation control permutes the recorded attention rows, not the weights.** Re-running permuted weights
    changes the order of the output projection's sum, so every later layer differs by rounding; that tests float
    associativity, not the matching. The repeated-pass control does run the model twice, and requires exact zeros
    (both CKA forms now divide by sqrt(dx·dy) so a quantity against itself is exactly 0).
12. **G1 compares state dicts in their stored dtype** (float32 to float64 is exact, so equality is unchanged; Pearson
    casts per tensor). This avoids holding about 9 GB at once.

## Reported beside the verdicts (built 2026-11-13, second pass)

- **P0 beside P for 211.1, 211.2, 211.3 and 211.8** (`score211.p0_beside`): each prediction's per-layer quantity
  recomputed with P0, on the head pairs P matched (so P0 never picks its own assignment), with P0's own noise floor
  (recorded beside P's; per layer, the largest over the six runs' finals) and the dropped queries printed. A layer or
  pair with no P0 says so. Point values over all documents.
- **The cosine measure and greedy match beside every cross-run value** (`score211.cross_run_table`): for every run
  against R0 at steps 0, 10000 and 143000 and every layer, P matched (the registered value), P greedy (each head of
  the run takes its nearest R0 head, repeats allowed), the cosine on the P-matched pairs, Hungarian on cosine and
  greedy on cosine (Bali et al.'s own), and P0 on the P-matched pairs.
- **Per-head tables** (`score211.per_head_tables`, in score.json under `beside.A.per_head` and `beside.B.per_head`):
  every run's within-run P(s, final) and settling fraction per head and checkpoint, and each run's Hungarian
  assignment to R0's heads at step 143000 with each pair's P.

## Not built yet

- The descriptive cross-run divergence curve at steps 1000 and 16000: it needs R0's rows at those steps, which the
  design's storage list does not keep (references are each run's steps 0, 10000 and 143000). Building it needs a
  storage decision: keep R0's rows at 1000 and 16000 (about 2.7 GB each per text set) or reduce every other run's
  1000 and 16000 against them while they are loaded. Descriptive only; no prediction reads it.
- Break-to-red over the parts of the code not touched this session (`checkpoints211`, `text211`, the pilot).

## Risks for the pilot

- Running the tests beside another CPU job: numpy and torch each start as many math threads as there are cores, so two
  jobs at once can slow each other by orders of magnitude (load 28 on 2026-11-13; the suite went from 80 s to over
  300 s). Run with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2` when anything else is running; give the
  pilot a thread count on purpose.

- `references_for` loads every reference's attention rows of a run at once: for OLMo, five references at about 3.2 GB
  each for set A, about 16 GB in memory for each non-reference record, which is over the design's 8 GB pilot stop.
  **Host readiness is not claimed** (Alden #13): the current pilot (`run211.py pilot`) samples Pythia only; it does not
  run G2, time the M bootstrap, project storage against the 150 GB stop or take an OLMo non-reference peak. Before
  launch the pilot must do all four, and the OLMo recording must load one reference at a time (reduce against it,
  free it) if the measured peak is over 8 GB.
