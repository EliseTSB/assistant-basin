# Part 215 fold 2 — recomputed preflight evidence and preflight-byte binding (Alden #13)

## SUMMARY

- One torch-free validator, `part215/evidence215.py`. `run215.require_pass_preflight` and
  `score215.admissibility` both call `evidence215.validate_preflight`, which recomputes every gate
  from its rows and never trusts a summary or a boolean: G0 (24 indices per comparison, |ours −
  part214| ≤ 0.02, each `part214` equal to the pinned Stage 1 value), G1 (48 keys, exact seeds,
  token equality of `ours` against the pinned Stage 2 rows), G2 (96 keys, exact pilot seeds,
  `is_broken` recomputed), G3 (`g3_scorer_check()` rerun), and the 7.5 s ceiling over all 192
  measured rows (48 G0 + 48 G1 + 96 G2).
- Byte binding: the preflight receipt, `g0_neutral.jsonl`, both Part 214 references and `run.json`
  are read once into a buffer, hashed, and parsed from that same buffer. The runner carries the SHA
  it admitted forward into every pin (no reread after the model load). The scorer hashes the
  preflight bytes it is given, compares them with `run.json`'s `preflight_sha256` and every row's /
  forward's pin, and makes a mismatch diagnostic-only; the pinned bytes still score.
- Alden's two runner witnesses (96 broken G2 bodies under `broken: 0`; one G0 row at 99 s under
  `max_s: 0.5`) are refused before the loader is reached, with zero loader calls. A forged G1
  `matched`, a G0 row over the 0.02 tolerance behind `pass: true`, and a disagreeing G3 boolean are
  refused the same way.
- The scorer counterexample — a changed, same-attempt preflight with a different SHA — is now
  diagnostic-only; the matching bytes score. The unchanged complete fixture is still admitted
  (positive control).
- 45 CPU tests pass (`45 passed in 92.34s`): the 39 from fold 1 plus 6 new. Every new witness test
  also runs a deliberately weakened COPY of `evidence215.py` in `tmp_path` that restores the
  pre-change trust (summary/boolean) and shows the witness being admitted, so the check is shown
  red before and green after.
- `part215/REPORT.md` gains a `LIMITS` section: the `-ing` veto is a lexical proxy (it also voids
  `-ing` adjectives: "I am an evolving model trained by Google." and "I am a cutting-edge AI
  developed by Google." read none), and false-negative rates are not claimed equal across arms.
- `DESIGN.md` is untouched. No commit was made.

## CHANGES (file:line)

`part215/evidence215.py` (new; standard library + `score215` only; no torch)
- `evidence215.py:44-58` — the Part 214 pins (module attributes, the test seam) and the design's
  gate numbers in one place: `G0_TOLERANCE = 0.02`, `MAX_LATENCY_S = 7.5`, `G1_NEEDED = 45`,
  `PILOT_BOUND = 14`, `PREFLIGHT_GENERATION_CALLS = 144`, `PREFLIGHT_FORWARD_CALLS = 48`,
  `MEASURED_CALLS = 192`.
- `evidence215.py:80-102` — `parse_jsonl_bytes`: strict JSONL from bytes already read and hashed.
- `evidence215.py:104-187` — `stage1_reference` / `stage2_reference`: digest check, parse from the
  same buffer, shape/keys/seeds/token-id validation. `stage2_reference` returns the 48 pinned rows
  keyed `(question, sample)`.
- `evidence215.py:200-217` — `_g3_failures`: reruns `score215.g3_scorer_check()` and requires the
  stored fixtures, deliberate break and field values to equal the rerun.
- `evidence215.py:219-292` — `_g0_failures`: exact 24 indices per comparison; recomputed
  `|ours - part214|` with an explicit over-tolerance failure; `part214` equal to the pinned Stage 1
  `g_ali`; per-row `abs_delta`/`pass` and the comparison summary (`n`, `pass`, `max_abs_delta`,
  `worst_question`) equal to the recomputation.
- `evidence215.py:294-365` — `_g1_failures`: exact 48 `(question, sample)` keys, `row_seed` seeds,
  `matched` recomputed from `ours` vs the pinned Stage 2 token ids, the row's own `part214` copy
  required to equal the reference, recomputed count ≥ 45, summary equal.
- `evidence215.py:367-431` — `_g2_failures`: exact 96 pilot keys, `pilot_seed` seeds, answer strings
  required, `is_broken` recomputed, count vs bound, summary equal.
- `evidence215.py:433-500` — `_recompute_latency` / `_latency_failures`: all 192 row latencies
  (144 generations + 48 forwards), finite and non-negative, 7.5 s ceiling, and every stored summary
  number (max/mean per population, counts, limit, `pass`) equal to the recomputation.
- `evidence215.py:502-546` — `validate_preflight(receipt, *, stage1_path, stage2_path,
  stage1_sha256, stage2_sha256, part214_prompts_sha256)`: assembles the above, checks the receipt's
  recorded reference digests against the pins, and loads each reference once from its recorded path
  (falling back to the reviewed repo paths).

`part215/run215.py`
- `run215.py:47` — imports `evidence215`; `run215.py:58-68` — `PILOT_MAX_LATENCY_S`, `G1_NEEDED`,
  `G0_TOLERANCE` and the stage paths come from `evidence215`, so the two phases cannot drift.
- `run215.py:374-403` — `_stage1_from_bytes` / `_stage2_from_bytes`; `load_stage1_receipt` and
  `load_stage2_unchanged` now read the file once and delegate to the shared reference parsers.
- `run215.py:501-540` — `preflight` reads both Part 214 references once, hashes those buffers,
  records `part214_stage{1,2}_path` (resolved) and `_sha256`, and parses from the same buffers.
- `run215.py:647-740` — `admit_preflight`: reads the preflight receipt once (hash + parse the same
  bytes), keeps the identity checks, calls `evidence215.validate_preflight` (no more
  `_gate_failures`/`_latency_failures`), validates the G0 reuse file, and additionally requires the
  reuse records' `g_ali` to equal the G0 `unchanged` rows' `ours` (`g0_reuse_g_ali`).
  `require_pass_preflight` is the receipt-only wrapper. Returns `{'receipt', 'raw', 'sha256',
  'g0_records'}`.
- `run215.py:742-...` — `load_g0_reuse` parses the JSONL from the buffer it hashed (no second read).
- `run215.py:848-...` — `full` takes the SHA and G0 records from admission, builds the pins before
  the model load, never rereads the preflight or reuse file, and reads `run.json` once
  (`read_bytes` + parse) on resume.

`part215/score215.py`
- `score215.py:986-1000` — `_evidence_failures`: lazy import of `evidence215` (which imports this
  module's pure functions, so a module-level import would cycle) and the one shared
  `validate_preflight` call.
- `score215.py:1002-1078` — `admissibility(..., preflight_bytes=None, prompts=None)`: hashes the
  supplied bytes, parses them, requires the passed object to be that parse, compares the digest with
  `run.json`'s `preflight_sha256` and every row's/forward's pin; a mismatch is a reason
  (diagnostic-only), not an exit. The gate evidence call is the shared validator.
- `score215.py:1161-1172` — `score_full(..., preflight_bytes=None)` loads the reviewed prompt bank
  when `prompts` is None (for the Part 214 prompts digest) and forwards the bytes.
- `score215.py:1430-1445` — `main` reads `preflight.json` and `run.json` once into buffers and
  parses them from those buffers; the preflight bytes are passed to `score_full`.

`part215/test_part215.py`
- `test_part215.py:368-534` — synthetic Part 214 reference fixture files, the `_fixture_reference_pins`
  seam, and `_admissible_preflight` rebuilt as a complete receipt with 48 G0 rows, 48 G1 rows and 96
  G2 rows over those references.
- `test_part215.py:537-553` — `_score_fixture` serialises the receipt once, pins its SHA into
  `run.json`/every row, and scores it with `preflight_bytes`; `:676-687` `_admissible_inputs`.
- `test_part215.py:1093-1137` — `_mutated_preflight` and `_load_sabotaged_evidence` (a weakened copy
  of the validator, never the tracked file).
- `test_part215.py:1138-1306` — the six new tests (five runner witnesses + the scorer byte test).
- `test_part215.py:236-256` — the production Part 214 pin test now reads the pins from `score215`
  and passes them explicitly, so it passes in any test order (the session admission fixture patches
  `run215`'s pins).

`part215/REPORT.md`
- New `## LIMITS` section (before `## FINDINGS`): the `-ing` veto is a lexical proxy and voids
  `-ing` adjectives; false-negative rates are not claimed equal across arms. READINGS items 12 and
  17 updated for the shared validator and the scorer/evidence pins.

## TESTS (exact commands, full pass/fail lines)

From `/home/elisepc/assistant-basin`:

```
$ PYTHONDONTWRITEBYTECODE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider part215/test_part215.py
.............................................                            [100%]
45 passed in 92.34s (0:01:32)
```

The 39 fold-1 tests all still pass; the six new tests are
`test_alden_witness_g2_broken_bodies_under_a_clean_summary_are_refused`,
`test_alden_witness_g0_latency_under_a_clean_summary_is_refused`,
`test_forged_g1_matched_with_different_tokens_is_refused`,
`test_g0_row_delta_over_tolerance_with_pass_true_is_refused`,
`test_g3_stored_boolean_that_disagrees_is_refused`, and `test_scorer_preflight_byte_binding`.

Compile check:

```
$ /home/elisepc/neuron-probe/.venv/bin/python -I -B -m py_compile part215/evidence215.py part215/run215.py part215/score215.py part215/test_part215.py
py_compile OK
```

## RED-GREEN WITNESSES

Each new test refuses the forged receipt with the tracked modules (green), then imports a COPY of
`evidence215.py` from `tmp_path` with exactly one recomputation removed and asserts that the same
receipt is admitted (red, the pre-change trust posture). The copies assert their sabotage target is
unique. No tracked file is copied over.

1. **Alden witness 1 (G2 bodies vs summary).** 96 pilot answers set to `!!!!`, `broken: 0`,
   `pass: true` kept. Tracked: `full` raises before the loader with `G2_broken_recomputed_96_over_14`
   and `G2_broken_count`; loader calls `[]`. Weakened copy (`is_broken` recomputation removed):
   `require_pass_preflight` returns `PASS`. Test:
   `test_alden_witness_g2_broken_bodies_under_a_clean_summary_are_refused`.
2. **Alden witness 2 (G0 latency vs summary).** One G0 row at `latency_s = 99.0`, `max_s: 0.5`
   kept. Tracked: refused with `latency_max_recomputed` and `latency_max_s_disagrees`, zero loader
   calls. Weakened copy (summary latency trusted): admitted. Test:
   `test_alden_witness_g0_latency_under_a_clean_summary_is_refused`.
3. **Forged G1 match.** Row (q0, s0) `ours = [1, 2, 3]` with `matched: true` and the correct
   `part214` copy. Tracked: `G1_matched_q0_s0` + `G1_matched_count`. Weakened copy (row's `matched`
   trusted): admitted. Test: `test_forged_g1_matched_with_different_tokens_is_refused`.
4. **G0 delta over tolerance behind `pass: true`.** A `zero` row's `ours` moved +0.5 from
   `part214`, `abs_delta`/`pass` left clean. Tracked: `G0_zero_delta_over_tolerance_q0`,
   `G0_zero_row_pass_q0`, summary failures. Weakened copy (row's `abs_delta` trusted): admitted.
   Test: `test_g0_row_delta_over_tolerance_with_pass_true_is_refused`. (The `zero` row is used so the
   separate G0-reuse cross-check does not also fire; the `unchanged` rows are the reused reads.)
5. **G3 stored boolean.** `gates.G3.fixtures_pass = 8` with `pass: true`. Tracked:
   `G3_stored_fixtures_pass`. Weakened copy (G3 not rerun): admitted. Test:
   `test_g3_stored_boolean_that_disagrees_is_refused`.
6. **Scorer preflight bytes.** With the genuine receipt bytes and matching `run.json`/row pins the
   card is admissible (`diagnostic_only is False`); changing only `wall_time_seconds` (same attempt,
   different SHA) makes the card diagnostic-only with every prediction and the headline
   `not_scored`. Weakened score215 copy (digest taken from `run.json` instead of the supplied bytes)
   scores the altered receipt — Alden's counterexample reproduced. Test:
   `test_scorer_preflight_byte_binding`.
7. **Positive control.** The unchanged complete fixture still passes
   `require_pass_preflight` (`test_require_pass_preflight_admits_the_complete_receipt`) and the
   complete synthetic run still scores every prediction
   (`test_complete_synthetic_run_scores_every_prediction`).

## OPEN QUESTIONS / STRICTER READINGS

1. **G1's `part214` copy.** Alden says `matched` must not be recomputed from the row's own copy
   "alone". I take the stricter reading: `matched` is recomputed from `ours` against the pinned
   Stage 2 token ids, AND a row whose stored `part214` copy differs from that reference is also
   refused (`G1_part214_copy_*`). Verified.
2. **G3's stored result.** "The stored result must also agree" is read as field-for-field equality
   with the rerun, including the fixture list and the deliberate-break list, not just `pass`.
3. **G0 summary.** The comparison summary's `n`, `pass`, `max_abs_delta` and `worst_question` must
   equal the recomputation; a row whose stored `abs_delta`/`pass` disagrees is refused even when
   the recomputed delta is inside tolerance.
4. **Latency must come from rows.** A receipt whose summary says 0.5 s but whose rows hold fewer
   than 192 latencies is refused (`latency_measured_rows`), even if the maximum is small: the
   ceiling is derived from all 192 measured rows, as the brief says.
5. **Scorer without bytes.** `admissibility` treats a preflight dict with no `preflight_bytes` as
   diagnostic-only (`preflight bytes are not provided`). This is stricter than trusting the object;
   the CLI always supplies the bytes. Callers of the library API must pass them to score.
6. **Reference locators are recorded in the receipt.** The preflight writes resolved absolute
   `part214_stage1_path` / `part214_stage2_path`; the scorer reads them. If receipts are ever moved
   between machines (or the repo root changes), the paths need to travel too, or the validator falls
   back to the reviewed repo paths. A repo-relative locator would be more portable; not changed
   here.
7. **One extra check beyond the brief's list.** `admit_preflight` requires the `g0_neutral.jsonl`
   records' `g_ali` values to equal the G0 `unchanged` rows' `ours` values (`g0_reuse_g_ali`).
   Without it, a forged receipt could keep a valid G0 gate and a different reuse file, and the full
   run's neutral readouts would come from values no gate ever checked. Listed under FINDINGS.
8. **The scorer does not read `g0_neutral.jsonl`.** It validates the reused forwards' `source` and
   `g0_sha256` binding and the G0 rows, but it cannot hash the reuse file itself. A scorer-side
   cross-check of the neutral forwards' `g_ali` against the G0 rows would close this; not added
   because the 215.5 synthetic fixture's neutral forwards are independent of its preflight fixture,
   and changing that test data is outside this brief. Flagged for the next fold if the reviewer
   wants it.
9. **`score_full` now loads `prompts215.json`** when `prompts` is None, to supply the Part 214
   prompts digest to the reference check. A missing/tampered bank is a hard `SystemExit`, as
   `load_prompts` already specifies.
10. **`tool_sha256` still invalidates old receipts.** Admission still requires the receipt's runner
    digest to equal the running `run215.py`. This fold edits `run215.py`, so any pre-fold receipt
    (none exist) would be refused. That is intended.

## FINDINGS

- **Merged the two evidence implementations.** The scorer's coarse `_preflight_evidence_failures`
  and the runner's `_gate_failures`/`_latency_failures` are gone; both call one torch-free
  `evidence215.validate_preflight`. This removes the drift risk the fold-1 report flagged
  (`score215.py:986` item 6 of OPEN QUESTIONS there).
- **Added the G0-reuse/G0-row `g_ali` cross-check** described in OPEN QUESTIONS 7.
- **Fixed a test-order fragility:** the production Part 214 pin test asserted against
  `run215.PART214_STAGE1_SHA256`, which the session admission fixture patches; it now uses the
  unpatched `score215` pins and passes them explicitly (verified with the runner pins patched).
- **`REPORT.md` READINGS item 17 was stale** ("coarse copy ... because the scorer cannot import
  torch"). It now describes the shared validator and byte binding. Item 12 names the
  `evidence215` pin seam.

## BLOCKERS

None. The real-model preflight/full still needs the Arc (xpu) and the Qwen3.5-4B weights; neither
exists in this sandbox, so every receipt exercised here is synthetic or from the tiny CPU model.
