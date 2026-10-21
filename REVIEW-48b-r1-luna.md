# Cold adversarial review: part48b-r1-luna

**Verdict: SHIP-WITH-FIXES**

## Findings by severity

### P2 — The pre-registered pooled estimate is absent (INTRODUCED)

**Location:** `part48b.py:398-412, 452-475`.

Amendment 1 of `PART48B-DESIGN.md` requires the primary estimate and CI “per model, plus a pooled estimate.” `summarize()` builds only the two per-model score analyses, and `main()` serializes each model separately inside its model loop. There is no cross-model aggregation or pooled output. The build brief's per-model JSON requirement does not remove this design requirement.

**Measured trigger:** `/home/elisepc/green/Journal/scratch/review-part48b-r1-luna-luna-r1-2026-10-21/probe_pooled.py` imports the real `part48b` module, calls its `summarize()` on a complete six-template pair fixture, and prints the returned fields:

```text
summary keys: ['letter_matched_sets', 'near_word_excluded_pairs', 'near_word_exclusion_forms', 'pair_count', 'pair_records', 'primary_token_count_rule', 'scores', 'single_letter_controls']
per-model scores: ['letter_matched', 'part48_continuations']
per-model median: 0.30000000000000004
pooled field present: False
```

**Why it matters:** the output cannot supply the pooled result that the preregistration names as a primary deliverable. Per-model estimates do not answer that planned aggregate question.

**Smallest fix:** add a separate aggregation step over the model summaries and emit the pooled estimate, interval, and test. The design does not state whether models or individual pairs receive equal weight, so define that weighting before computing the pooled result.

### P3 — A failed JSON write leaves a truncated result file (INTRODUCED)

**Location:** `part48b.py:471-473`.

The CLI opens the final path directly and calls `json.dump` into it. If serialization or the underlying write fails after some bytes have been written, the exception escapes and the partial file remains at the expected model filename.

**Measured trigger:** `/home/elisepc/green/Journal/scratch/review-part48b-r1-luna-luna-r1-2026-10-21/probe_write_failure.py` invokes the real `main()` with model loading mocked, then makes `json.dump` write a prefix and raise a simulated disk-full error:

```text
raised: OSError('simulated disk-full')
file remains: True bytes: 11 content: '{"partial":'
valid JSON: False decode error: Expecting value
```

**Why it matters:** a consumer that discovers the model filename can mistake an incomplete artifact for a completed result; interruption also destroys any previous valid result at that path.

**Smallest fix:** write to a temporary file in the output directory, close it successfully, then atomically replace the final path.

## What I checked and found sound

- The pair record computes `d` as k score minus c score for both score sets (`part48b.py:335-340`), and the scores use Part 48's imported continuation log-probability and sharp-minus-round log-sum-exp.
- The 1e-3 floor is applied before averaging the kept template scores; incomplete pair/template rows raise an error (`part48b.py:303-323`).
- I fed symmetric differences `[1, -1]` and `[1, -1, 1, -1]` to the real sign-flip function (`probe_null.py`); both returned observed mean 0 and p-value 1.0.
- The Amendment 4 ten-pair list and letter-matched sets match the design/build brief. I did not run model inference or the repository test suite; the probes used synthetic fixtures and did not contact services or read credentials.
