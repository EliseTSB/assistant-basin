<!-- cheap-review.mjs cheap-review-part198-code-r1-ds-r1-2026-11-08: /home/elisepc/green/worktrees/ab-part198 91b6af4..HEAD, lane ds, df-run exit 0 -->
# Cold adversarial review — part198-code-r1

Commit `65c2be6` ("Part 198 build … run198.py, score198.py, test_build198.py, BUILD-NOTES.md; no model run yet"),
diff `91b6af4..HEAD`. Reviewed against `PART198-DESIGN.md` revision 3 and the frozen `stats198.py`.

**Verdict: DO-NOT-SHIP.**

The runner is careful about the things the design spells out in prose — the took transition, the blind reader's
lack of edit metadata, the 8-of-10 mass gate, `reset_cache=True`, the first-record replay. But the scorer that
turns the run into verdicts is wrong in ways that change the published numbers, and one of the design's two
sensitivity checks (S1, the aimed positive control) is built from a direction that is not the one the design
specifies. None of this needs a model to demonstrate: every finding below is reproduced by driving the real
`run198.py`/`score198.py` on synthetic JSONL with no weights loaded.

Severity: **Blocker** = a published forecast or gate verdict would be wrong. **Major** = a stated property is not
true, or a failure path loses or corrupts data. **Minor** = robustness/audit gap.

---

## Blocker

### B1. Clopper-Pearson lower bound is wrong for every k ≥ 1 (INTRODUCED)

`run198.py:218-228` (`cp_interval`). The bisection for the lower bound tests `ge(successes, mid) < alpha/2` and
then moves `b = mid` when it is *true*. `ge(k, p)` is increasing in `p`, so the region where it is below
`alpha/2` is `p` *below* the bound; the loop therefore converges to the bottom of the interval, not the bound.
The upper bound (`:229-239`) is correct. Every interval printed by `score198.print_result` (`:292-297`) has a
lower bound of 0 or 1.

Measured (`probe/p10_cp.py`, scipy exact):

```
n=24: 24 of 25 intervals wrong; rows (k, got_lo, exact_lo, got_hi, exact_hi):
    (1, 1.0, 0.0011, 0.2112, 0.2112)
    (5, 1.0, 0.0713, 0.4215, 0.4215)
    (15, 1.0, 0.4059, 0.812, 0.812)
    (20, 0.0, 0.6262, 0.9526, 0.9526)
    (24, 0.0, 0.8575, 1.0, 1.0)
k=20 n=24: ge(20, 0.5) = 0.000771939754486084 -> bisection converges to 4.1e-25 (exact lower bound 0.6443)
```

The design's Size section (line 339) states "20 of 24 has a lower bound of 63%"; the code would print
`[0.000, 0.953]`. The suite's only CP check (`test_build198.py:58-60`) tests the *upper* bound of 0-of-24, the one
case the lower-bound code short-circuits (`successes == 0 → lo = 0.0`, which is correct by accident).

**Fix:** in the lower-bound loop invert the test — `if ge(successes, mid) > alpha/2: a = mid else: b = mid` — or
better, use the identity `lo = beta.ppf(alpha/2, k, n-k+1)` / `hi = beta.ppf(1-alpha/2, k+1, n-k)` and add a
test at k=20, n=24 asserting `lo ≈ 0.6443`.

### B2. Mass forecasts (198.1 / 198.3 mass) can never report a failure (INTRODUCED)

`score198.py:138` reads `flag(r, "localization", stat, "localized")`, i.e. `row["localization"]["f1"]["localized"]`.
The runner writes `localization` from `run198.localization()` (`run198.py:518-523`), whose shape is
`{"f1": {"localized": …}, "pcs": {...}, "t_word": {...}}` — there is no top-level `"localized"` key. `flag()`
returns `False` for a missing key (`score198.py:34-40`), so `successes` is always 0 and the status is always
`PASS` (or `NOT EVALUABLE` when no batch counts). The design's "fails if any batch" (line 377) is unreachable.

Measured (`probe/p2_massfmt.py`): a batch row with `localization.f1.localized = true`, 10/10 took, gate passing:

```
198.1 MEMIT mass counted_batches= 1 successes= 1 status= NOT EVALUABLE   <- gate failed in that fixture
row had localization.f1.localized = True
```

and with a full gate-passing fixture (`probe/p5_arms.py`) every mass row localized on both statistics:

```
GATE True []
198.1 MEMIT mass: denom=1 succ=1 FAIL
```

Here `successes=1` comes from the fixture's hand-written top-level `localized`; the runner's shape gives 0. The
same `flag(...)` pattern is used correctly elsewhere (`r["localization"]["t_word"]["localized"]` at
`score198.py:215`, `:239-240`), so this is a single wrong path.

**Fix:** `flag(r, "localization", stat, "localized")` in `mass_forecast` (it is already the right call there —
the bug is only that the runner never writes that shape; either make `run198.localization()` return both shapes
or, simpler, keep `mass_forecast` as is and have the batch record store `"localization": loc` *and* a flat
`"localized": {stat: bool}`). Add a test that feeds a runner-shaped batch row through `mass_forecast` and expects
`successes == 1`.

### B3. S1 (aimed positive control) writes a direction the design does not specify (INTRODUCED)

`run198.py:686-691` (`make_aimed_delta`):

```python
output = gain * E[target_id]
output = output - output.mean()
output = output / norm
```

The design (Controls, line 183) and `stats198.py`'s norm contract (lines 9-12) say the output direction is
`gamma * (u - mean u)` — centre first, then scale by gamma. The code scales first, then centres. The reader that
scores the control (`stats198.read_pairs`, `centre=True` at `stats198.py:77-78`) re-centres the direction and then
applies gamma, so the control is scored on a direction that is not the one it wrote. With uniform gamma the two
coincide; with the non-uniform gamma the design's own fixtures require (`fixtures198.py:25-26`, `linspace(0.5,2.0)`),
they do not.

Measured (`probe/p31_aimed9.py`, same reader the control is scored with):

```
uniform      cos=+0.4426 |u|=0.6652 reader top token=3
non-uniform  cos=+0.0415 |u|=0.1586 reader top token=3
```

`|u| = 0.16` also shows the direction is not unit length after centring (`d=8`, so `1/sqrt(d)`): the code
normalises *before* centring, and the design says the direction is "made unit length". The aimed control is S1,
and S1 gates 198.4 (the headline's "not named by this static reader" claim). A control built from the wrong
direction can fail S1 for a reason that is not about the reader.

**Fix:** `output = gain * (E[target_id] - E[target_id].mean()); output = output / output.norm()`. Add a fixture
with non-uniform gamma asserting the written direction equals `unit(gamma*(row - mean row))`.

### B4. S1's denominator is not the design's (INTRODUCED)

`score198.py:193` builds `s1_rows` from ROME rows that are `name_eligible` and carry an `aimed_positive_control`
key — with no `took` filter — and `:197` reports `attempted = len(rome)` but `name_eligible = len(s1_rows)`.
The design's S1 (line 196) is "named in at least 20 of its name-eligible records", and the control is only built
inside the success path (`run198.py:803-822`), so a record that failed before the control was built silently
leaves the denominator, and a record that did not take is still counted.

Measured (`probe/p1_mass.py`): a single ROME row with `took=False, already_satisfied=True, name_eligible=True,
aimed_positive_control.oracle_named_strict=True`:

```
S1 {'attempted': 1, 'took': 0, 'already_satisfied': 1, 'name_eligible': 1, 'successes': 1,
    'clopper_pearson_95': [1.0, 1.0], 'threshold': 20, 'status': 'NOT EVALUABLE'}
```

A non-took record contributes a success to S1. The design's S1 denominator should be name-eligible *and* took;
the code's is name-eligible *and* has-a-control.

**Fix:** `s1_rows = [r for r in rome if r.get("took") is True and r.get("name_eligible") is True]` and count
`flag(r, "aimed_positive_control", "oracle_named_strict")` over exactly those; report the missing-control count
separately rather than dropping rows from the denominator.

### B5. An S1 failure suppresses the 198.4 *update-top* forecast, which S1 does not control (INTRODUCED)

`score198.py:204-208` rewrites the status of **every** forecast whose label starts with `"198.4 "` to
`NOT EVALUABLE` when S1 does not pass. That includes `"198.4 <method> update-top naming"` (`:183-186`), which
reads the update's own top direction (`update_top_names_strict`) — a base-required diagnostic with no dependence
on the aimed control. The design scopes S1 to the aimed control only ("If S1 fails, 198.4 is reported as
uninformative about direction", line 196-197, where 198.4 is the *oracle* naming); the update-top half of the
forecast is a separate clause ("the update's own top direction (base-required) likewise", line 380).

Measured (`probe/p32_s1suppress.py`: 24 ROME rows, all took and name-eligible, `edited_layer_stats` naming the
target strictly, control `oracle_named_strict=False` so S1 fails):

```
S1: FAIL
198.4 ROME oracle naming: successes=24/24 status=NOT EVALUABLE
198.4 ROME update-top naming: successes=24/24 status=NOT EVALUABLE
```

The update-top forecast has 24/24 successes and is overwritten to `NOT EVALUABLE` by a check it does not depend
on.

**Fix:** apply the suppression only to the oracle forecast (`item["forecast"].endswith("oracle naming")`), and
give the update-top forecast its own qualification if the design wants one.

---

## Major

### M1. A crash mid-record wedges every later run of that stream (INTRODUCED)

`append_jsonl` (`run198.py:108-114`) writes line + newline in one `write`, so a torn line is unlikely on a
normal crash — but a killed process, a full disk, or a power cut can still leave a partial line, and
`read_jsonl` (`:117-131`) raises on it. That raise is not caught by `existing_keys` (`:134-145`), so every
subsequent run of that subcommand dies in `run_arm_a` at `:705` before doing any work, and `score198` cannot
score the stream either.

Measured (`probe/p11_resume.py`):

```
(a) existing_keys raises: ValueError - malformed JSONL at .../armA-rome.jsonl:2: Expecting value: line 1 column 66 (char 65)
(a) score198.registered_rows raises: ValueError - malformed JSONL at ...:2: Expecting value: line 1 column 66 (char 65)
```

**Fix:** in `read_jsonl`, if the malformed line is the last line and lacks a trailing newline, drop it (and
record that it was dropped); otherwise raise. This is the design's own "a killed run resumes at the next record
and loses at most one" (Optimization, line 352).

### M2. Duplicate rows: resume keeps the first, the scorer keeps the last, and a later failure masks a success (INTRODUCED)

`existing_keys` (`run198.py:134-145`) adds a key for any `case`/`batch`/`failure` row, so a failed case is never
retried (by design: failures are terminal). `registered_rows` (`score198.py:16-27`) keeps the *last* row per key.
A re-run of a case therefore appends a second row and the scorer silently prefers it; a failure row appended
after a successful case row replaces the success.

Measured (`probe/p11_resume.py`):

```
(c) duplicate rows -> scorer keeps 1 row(s); took = [False] ; resume would skip case 1 = True
```

The first row had `took=True, localized=True`; the second (a re-run that failed) had `took=False`. The scorer
reports `took=False` and the case vanishes from every `took` denominator, while the gate's restoration check
(`score198.py:108-115`) *does* trip on failure rows — so a stream can be both "not took" and gate-failed with no
indication which row is authoritative.

**Fix:** make `registered_rows` keep the *first* terminal row per key (failures are terminal, so the first is
authoritative), or refuse to score a stream with more than one terminal row per key unless `run_mode` differs.

### M3. Arm B "took" is computed from `wsp.teach`'s self-report, not from the answers the runner measures (INTRODUCED)

`run198.py:1090-1095`: `free_exact` comes from `teach["free_exact"]` (a string from `wsp.py:244`), and
`wsp_took(before_exact, free_exact, len(asks))` uses it as the "after" count. `after_exact`, which the runner
computes itself at `:1089` from `w.answer(q, True)`, is recorded but never used. The design (line 145) says
"all three do after (`free_exact` 3/3)" — that is a parenthetical naming wsp's field, but the endpoint should be
the runner's own measurement; when they disagree the record is internally inconsistent.

Measured (`probe/p14_wsp.py`):

```
wsp_took(0,2,3) = (False, False)   # wsp free_exact 2/3 but the runner's own after_answers were 3/3
```

Also, `:1091-1094` falls back to `free_exact = after_exact` when the string does not parse, which silently
switches the source of truth mid-run.

**Fix:** `took = before_exact == 0 and after_exact == len(asks)`; record `free_exact` beside it and flag any
disagreement as a finding.

### M4. The restoration check only covers parameters the editor reported (INTRODUCED)

`run198.py:831-832` (and the same pattern in the error path `:844-845`, mass `:993`, `:1006`): `restored` is
`all(torch.equal(params[p], w0) for p, w0 in orig.items())` — only the keys EasyEdit returned in
`return_orig_weights`. A write to any other parameter (EasyEdit's `mom2` pathway, a hook, a bias) is not seen.
The design says "the weights must equal the base bit for bit" (line 125).

Measured (`probe/p6_restore.py`):

```
(a) restoration check says restored = True ; bias actually changed: True
```

The base copy itself is sound: `gpt2_geometry` uses `params[p].detach().T.contiguous()` (`run198.py:615`), and
`contiguous()` on a transposed tensor is a copy (`probe/p6_restore.py`: "(b) W shares storage with the parameter:
False"), and `load_gpt2_model` replaces every `p.data` with a clone first (`:598-602`). So the base statistics are
not aliased — the gap is only in what the check compares.

**Fix:** snapshot every parameter and buffer before the first record (the clone in `load_gpt2_model` is already
such a snapshot) and compare all of them after each record, not just `orig`.

### M5. Arm C reruns after a failure silently skip the calibration model and write no T* (INTRODUCED)

`existing_keys` counts a `failure` row as prior (`run198.py:140`), so a calibration model that crashed is never
retried. `run_arm_c` then builds `current` from `armC_model` rows only (`:1421-1422`) and returns 0 without an
error when any of the first three is missing (`:1423-1424`), so the run continues to evaluation and hosts with no
`tstar.json`.

Measured (`probe/p19_armc_rerun2.py`):

```
T* guard returns 0 without error: True
`current` is built from record_type == 'armC_model' only: True
resume prior comes from existing_keys (counts failures): True
after one failed calibration model, prior contains it: True
```

**Fix:** exclude `failure` rows from `prior` for Arm C (or raise when a calibration model is missing rather than
returning 0), and make a missing `tstar.json` a hard error before evaluation/host work.

### M6. 198.8 reads the last row in file order as "after the tenth write" (INTRODUCED)

`score198.py:254-260`: `seq[-1]` is the last line of `armB-seq-wsp.jsonl`, not the row with the highest
`n_written_so_far`. After a resume the runner rewrites every row (`run198.py:1252`), so the last row is the tenth
write — but any later row for an earlier case (a re-run, a smoke row that slipped in) is taken as the final state.

Measured (`probe/p21_seq.py`):

```
clean file (last row = tenth write): {'successes': 10, 'rate_denominator': 10, 'status': 'PASS'}
after a re-run of case 3 is appended last: {'successes': 0, 'rate_denominator': 4, 'status': 'NOT EVALUABLE'}
```

**Fix:** `final = max(seq, key=lambda r: r.get("n_written_so_far", 0))`.

### M7. `--limit` cannot reach the calibration models in Arm C (INTRODUCED)

`run198.py:1399` and `:1434` both test `selected_count >= limit`, but `selected_count` is only incremented after
a model is written (`:1420`, `:1456`). With `--limit 1`, the calibration loop runs all three; with `--limit 2` it
still runs all three; only the evaluation loop is limited. Smoke runs therefore load three models when one was
asked for.

Measured (`probe/p13_armc.py`, control flow replayed exactly):

```
smoke --limit 1, nothing prior   : return 0 (no tstar written)
smoke --limit 2, nothing prior   : return 0 (no tstar written)
```

**Fix:** increment `selected_count` in the loop body and check `selected_count >= limit` at the top of each
iteration, before loading.

### M8. The gate trusts a `gate.json` that nothing binds to the code it certifies (INTRODUCED)

`preflight` (`run198.py:316-344`) runs the fixtures and the test file fresh, which is good, but the crosscheck
item only checks that `crosscheck198.json` exists with `fails == []` and the right layer counts (`:326-334`). It
does not record or check the hash of `stats198.py` that produced it, and `crosscheck198.json` carries no such
hash (`probe/p22_gate.py`: "preflight hashes stats198.py: False", "crosscheck198.json records the stats module
hash: False"). Editing `stats198.py` after the crosscheck leaves the gate green. The design's "Code checks
before data" (line 187) is the check that `stats198.py` reproduces the pilot values; the gate should be able to
tell that the file it gates is the file that was checked.

**Fix:** write `stats198.py`'s sha256 into `crosscheck198.json` and compare it in `preflight`.

---

## Minor

- **m1. `mass_gate_status` is dead code.** `run198.py:195-196` is never called by `score198`; the "not evaluable
  with no counted batch" behaviour lives in `mass_forecast` (`score198.py:144`). `test_build198.py:55-56` tests
  the dead helper, so the property is green without the scorer being exercised. (INTRODUCED)
- **m2. A gate-failing batch that localizes is silently dropped.** `mass_forecast` excludes it from `countable`
  (`score198.py:137`), which matches "left out of 198.1 and 198.3" (design line 130), but the count of dropped
  batches is not reported. Measured (`probe/p4_massgate2.py`): a 10/10 batch not localized and a 5/10 batch
  localized give `counted_batches=1, successes=0` with no mention of the dropped batch. (INTRODUCED)
- **m3. `statistic_map` fills unedited layers from `base`** (`run198.py:514-515`), so the blind reader's input
  includes every layer's base value. That is correct for "the suspect's layer with the highest T_word", but it
  means the reader's output depends on the base statistics file — if `base-gpt2.json` is stale or from a different
  build, the reader ranks against the wrong base. No hash is stored in `base-gpt2.json`. (INTRODUCED)
- **m4. `run_arm_a` re-loads the first-update reference inside the verification block** (`run198.py:864`),
  overwriting the in-memory `first_reference`; harmless but makes the earlier assignment at `:725` dead for the
  verification path. (INTRODUCED)
- **m5. `run_arm_a` raises on a failed restoration after appending the case row** (`:834-836`), so the stream
  contains a case with `restored_bit_exact=False` and the process dies; the scorer's gate catches it
  (`score198.py:109`), which is right, but the failure row that would explain the stop is not written. (INTRODUCED)

---

## What I checked and found sound

- **Blind reader has no edit metadata.** `stats198.blind_reader(layer_pairs)` takes only layer→pairs; the runner
  passes `pair_by_layer(base, current, S)` (`run198.py:526-527`), which walks `base` and substitutes `current` for
  edited layers. The edited set is never passed in (`probe/p23_blind.py`: signature
  `(layer_pairs: dict[int, list[Pair]])`). A base layer with a higher T_word wins the reader
  (`probe/p23_blind.py`: "base layer 3 at z=20 vs edited layer 17 at z=7 → blind layer: None" — a tie, which the
  frozen rule makes a reader failure, as the design says).
- **Took transitions.** `took_transition` (`run198.py:148-152`) requires old ahead before and new ahead after,
  strict on both sides; a tie before is neither took nor already-satisfied. `wsp_took` (`:155-158`) requires
  `before_exact == 0` and `after_exact == taught`. Matches the design; `test_build198.py:27-35` exercises the
  boundary cases and fails when the function is sabotaged (`probe/p16_sabotage.py`: 2 and 1 failed checks).
- **Mass gate.** `mass_gate` is 8-of-10 (`run198.py:161-162`), `take_gate_passed` is recorded per batch
  (`:987`), and `mass_forecast` counts only gate-passing batches. The threshold itself is right; B2 is about what
  is counted inside them.
- **Random control.** `make_random_update` (`run198.py:165-181`) preserves singular values and replaces both
  bases with reduced-QR orthonormal factors; one generator per write advanced over sorted layers
  (`:184-192`), seeded 198 + case id (`:1110`). The suite fails when the control is sabotaged to drop the singular
  values (`probe/p17_sabotage.py`: "FAIL random control preserves singular values", exit 1).
- **AlphaEdit `reset_cache=True`** on every call in both the single and mass paths (`run198.py:740-741`,
  `:942-943`), and the first-record replay (`:867-868`).
- **R-ROME is not run.** `ee_import.load_methods` imports it, but no subcommand applies it; the design drops it.
- **`--dry-run` is inert.** `main` returns the plan before `preflight`/model loading (`run198.py:1491-1494`), and
  `root_out_dir` refuses any `--out-dir` outside `results/part198` (`:251-256`).
- **Fixtures and build tests pass as written** (`python -I fixtures198.py`: 25 PASS; `python -I test_build198.py`:
  17 PASS). The fixtures cover the norm contract, the sign rule, zero variance, ties, naming and the blind reader
  — the gaps are CP's lower bound (B1), the aimed control's direction (B3) and the scorer's mass path (B2), none
  of which the suites touch.
- **McNemar** (`run198.py:243-248`) is exact and correct: `mcnemar(0,5) = 0.0625`, `mcnemar(3,3) = 1.0`.

## Method note

No model was loaded, no weight file opened, no statistic computed on a model or dataset, no live service called.
Every measurement above is a probe script in `probe/` driving the real `run198.py`/`score198.py`/`stats198.py` on
synthetic rows or synthetic tensors; outputs are pasted with each finding.
