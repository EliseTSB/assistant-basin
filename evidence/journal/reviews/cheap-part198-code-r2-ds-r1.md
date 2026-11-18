<!-- cheap-review.mjs cheap-review-part198-code-r2-ds-r1-2026-11-08: /home/elisepc/green/worktrees/ab-part198 65c2be6..HEAD, lane ds, df-run exit 0 -->
# Cold adversarial review — part198-code-r2 (re-review of the fold r1)

Commit under review: `1a05ea9` ("Part 198 fold r1 … DS code review B1-B5, M1-M8 and minors; new tests fail on
65c2be6; no model run"), diff `65c2be6..1a05ea9`, in `/home/elisepc/green/worktrees/ab-part198`.
Checked against `PART198-DESIGN.md` revision 3 and `Journal/reviews/cheap-part198-code-r1-ds-r1.md`.
I did not write this code. Every measurement below is a script in `probe/` in this directory driving the **real**
`run198.py` / `score198.py` / `stats198.py` from the worktree (or, for the old-code comparison, the pristine
`65c2be6` copies in `old/` obtained with `git show`). No model was loaded, no weight file opened, no dataset read, no
live service called. Python: `/home/elisepc/edit-detect/.venv/bin/python -I` (torch 2.14.1+cpu, scipy 1.18.1).

**Verdict: SHIP-WITH-FIXES.**

The fold is real work: 17 of the 17 claimed old-code failures reproduce (`probe/run_suite_against.py old`), and the
core repairs (CP lower bound, aimed direction, S1 denominator and suppression scope, torn-tail recovery, Arm B
measured take, Arm C retry/T\* order, `--limit` guard, crosscheck hash, base-statistics hash) are correct on the real
code paths I could drive. But the fold's own M2 fix ("refuse to score on conflicting terminal rows") is keyed on the
wrong thing — it fires on *legitimate* second processes and never on the *illegitimate* duplicate it was meant to
catch — and it can make a registered Arm B-seq stream permanently unscoreable. That is a new Blocker. Two Majors: the
new restoration fingerprint silently fails closed on a 0-d tensor, and the new tests only prove the fingerprint
*helper*, not the runner's use of it (four sabotages of the runner's restoration paths stay green). The remaining
findings are follow-ups (PRE-EXISTING) or test gaps.

---

## Round-1 findings — status

| # | Status | Evidence |
|---|--------|----------|
| **B1** CP lower bound | **FIXED** | `run198.py:279-285` uses `scipy.stats.beta`; fallback bisection `:303-321` inverts the test correctly. `cp_interval(20,24) = (0.6261582865, 0.9526463734)`, exact two-sided 95% (probe below). Old code gave `4.1e-25`. The design's "63%" (PART198-DESIGN.md:339) and the 0.6262 given in the brief agree; round 1's 0.6443 was wrong and is not a defect. |
| **B2** mass localization shape | **FIXED (was already correct)** | `score198.py:148` reads `localization[stat].localized`, which is what `run198.localization()` (`run198.py:664-669`) writes. Test `test_build198.py:304-310` builds the row with the runner's own `localization()` and gets `successes == 1, status == FAIL`. Matches BUILD-NOTES' "already satisfied in 65c2be6". |
| **B3** aimed direction | **FIXED** | `run198.py:846-855`: centre, then gamma, then normalise. `probe/p25_aimed.py`: non-uniform gamma, `|new| = 1.000000`, `cos(new, want) = 1.000000`; old formula `cos = 0.974679`. |
| **B4** S1 denominator | **FIXED** | `score198.py:210-212`: `took and name_eligible`; a missing control is counted in the denominator and reported (`missing_controls`). Sabotage reverting to the old filter is caught (probe p8). |
| **B5** S1 suppression scope | **FIXED** | `score198.py:222-226` adds `endswith("oracle naming")`. Sabotage removing it is caught. |
| **M1** torn tail | **FIXED** | `run198.py:125-165`. Drops only an unterminated final line, truncates, records in `JSONL_RECOVERIES`, warns; interior and newline-terminated malformed lines still raise. Widening it to all final lines is caught by the suite. |
| **M2** duplicate terminal rows | **FIXED-WITH-NEW-DEFECT** | `score198.py:16-37` keeps the first and refuses on any payload difference. It does catch the round-1 example (case row then failure row). But see **N1**: the key is the case id and the payload excludes only `run_id`/`timestamp`, so legitimate replays conflict and the real duplicate (`main()`'s run-level failure rows, `case_id` absent) is never seen. |
| **M3** Arm B took | **FIXED** | `run198.py:195-212`; both callers (`:1254`, `:1381`) use it; `wsp_took` is fed `after_exact`. Sabotage reintroducing the `free_exact` fallback is caught. |
| **M4** restoration covers all state | **FIXED-WITH-NEW-DEFECT** | `run198.py:574-604`, snapshot at `:885`/`:1100` before the first `apply` (`probe/p14_restore_paths.py`: snapshot offset 1365 < first apply 2486), compared on success and on the failure branch in single (`:994`, `:1006`, `:1051`, `:1066`) and mass (`:1159`, `:1171`). Non-contiguous and view tensors hash identically to their contiguous copies. New defects: **N2** (0-d tensor raises) and **N3** (tests do not exercise the runner's use). |
| **M5** Arm C retry / missing T\* | **FIXED** | `run198.py:220-233`, `:1564`, `:1596-1609`. `probe/p5_armc_rerun.py`: a failed calibration model is retried on the next registered run; T\* = 10.33 from the three calibration maxima even when an evaluation model reports 99. `probe/p4_armc_tstar.py`: a missing `max_t_word` raises before `tstar.json` is written. Sabotage removing `require_tstar` and sabotage reverting `prior` to `existing_keys` are both caught. |
| **M6** 198.8 final row | **FIXED** | `score198.py:164-166`, used at `:274`. Sabotage restoring `seq[-1]` is caught twice. |
| **M7** `--limit` before load | **FIXED** | `run198.py:1572`, `:1614`. `probe/p17_armc_limit.py`: `--limit 1` loads 1 model, `--limit 2` loads 2, `--limit 3` loads 3; no `tstar.json` until all three exist. |
| **M8** crosscheck bound to stats198 | **FIXED** | `crosscheck198.py:16`, `:78`; `run198.py:412-417`, `:500-501`. `probe/p10_preflight.py`: a stale artifact (no hash) and a wrong hash both make `preflight` refuse and record `stats_hash_matches: False` in `gate.json`. Caveat: the test does not catch removal of the check from `preflight` (**N5**, test gap only). |
| **m1** dead `mass_gate_status` | **FIXED** | Now called at `score198.py:154`; monkeypatching it is caught. (Its semantics are still `n > 0`, see P2.) |
| **m2** dropped gate-failing batches | **FIXED** | `score198.py:158` reports `dropped_batches`; tested at `test_build198.py:312-315`. |
| **m3** base statistics hash | **FIXED** | `run198.py:510-522`, `:766-779`. `load_base` refuses a different digest; sabotage removing the check is caught. The test only exercises the helpers, not `run_arm_a`'s call site (see N3). |
| **m4** dead first-reference assignment | **NOT FIXED** (optional in the brief) | `run198.py:888` and `:1025` still both load the reference. Harmless. |
| **m5** failure row before raise | **FIXED (was already correct)** | `run198.py:1009-1015`: the failure row is appended before the raise; the check at `test_build198.py:387-389` passes on pristine and on HEAD. |

### Old-code audit (claim: 17 failing checks on 65c2be6)

`probe/run_suite_against.py old` copies the HEAD `test_build198.py` next to the pristine `65c2be6` `run198.py` /
`score198.py` and runs it:

```
FAIL B1 two-sided CP endpoints match independent beta quantiles
FAIL B3 aimed control centres before applying non-uniform gamma and normalizes last
FAIL M1 drops, truncates and records only a torn final unterminated JSONL line
FAIL M2 conflicting terminal rows are named and refused; identical repeats keep first
FAIL M3 runner after-count controls WSP take and self-report mismatch remains visible
FAIL M4 restoration fingerprint covers unreported parameters and buffers
FAIL M5 Arm C retries failed calibration keys and requires T* before evaluation loads
FAIL M6 sequence final state is selected by highest completed-write count
FAIL M7 both Arm C loops check the shared limit before model loads
FAIL M8 preflight binds crosscheck to the checked stats198.py hash
FAIL m3 base statistics cache stores and checks its input fingerprint
FAIL m2 mass scorer reports gate-failing batches it drops
FAIL m1 mass gate status is exercised by the scorer path
FAIL B4 S1 counts only took/name-eligible rows and missing control rows as failures
FAIL B5 failed S1 suppresses oracle naming only, preserving update-top verdict
FAIL M6 198.8 scores the row with highest n_written_so_far
FAIL M7 both Arm C loops share the pre-load limit guard
TEST_BUILD FAIL (17 failed)
```

17, as claimed. On HEAD the same suite is `TEST_BUILD ALL PASS (0 failed)`. Spot-checked two by reading: the M4 check
(`test_build198.py:207-226`) fails on old code only because `model_state_snapshot` does not exist (`getattr` → `None`),
which is honest but proves only existence, not behaviour (see N3); the M7 check counts the string
`arm_c_limit_reached(selected_count, limit)` in `run_arm_c`'s source, which is a structural assertion, and BUILD-NOTES
correctly says the review's M7 load-count bug was not reproduced (I confirm: `probe/p17_armc_limit.py` on HEAD loads
exactly `limit` models).

---

## Findings

Severity: **Blocker** = a published number or a stream's scorability is wrong. **Major** = a stated property is not
true or a failure path loses data. **Minor** = audit/test gap. Each is labelled INTRODUCED (this fold adds it or makes
it reachable) or PRE-EXISTING (a follow-up, not a reason to hold this change).

### N1 (Blocker, INTRODUCED) — M2's conflict rule fires on legitimate replays and never on the real duplicate

`score198.py:16-37`. The rule is: for every registered `case`/`failure` row with a non-null `case_id`, keep the first;
if a later row for the same `case_id` differs in any field other than `run_id`/`timestamp`, raise `ValueError` and the
whole stream cannot be scored (`score198.score_from_dir` has no handler; `main` exits non-zero).

Measured (`probe/p2_dup_resume.py`, `probe/p16_armb_dupe.py`):

```
(b) Arm C: failure then success on retry -> arm_c_successful_keys -> {'EleutherAI/pythia-410m'}   (ok)
(c) Arm B-seq: failure row then case row -> registered_rows raises: conflicting terminal rows for case_id='0' ... failure then case
(e) Arm B-seq replay, names_after_write differs in one float -> registered_rows raises: conflicting terminal rows for case_id='0' ... case then case
```

Why it matters, in three parts:

1. **The legitimate case.** `run_arm_b_seq` replays the whole frozen prefix on every registered invocation
   (`run198.py:1363-1364`) and appends a row only when `str(case_id) not in prior` (`:1425`). So a restart after a
   crash between two writes appends nothing new for the prefix — good. But the *recomputed* rows are not written, so
   the conflict only arises when a row *is* appended twice, which the code guards against. The realistic conflict is
   the **failure-then-case** pair (probe (c)): a `failure` row for case 0 (e.g. `sequence_write_or_score`) followed by
   a later registered run that replays and appends the case row. `run_arm_b_seq` refuses to start when any registered
   failure exists (`run198.py:1349-1350`), so the runner itself will not produce it — but `main()`'s catch-all
   (`run198.py:1696-1701`) writes a `failure` row for *any* exception with `case_id` absent, and a `failure` row for
   case 0 can also come from `run_arm_b`'s own handler (`:1326`) if the two streams share a case id. Whether this pair
   is reachable depends on case-id disjointness between `armB_single` and `armB_sequence`, which I did not verify
   (see self-assessment). The scorer's own `gate_check` treats failure rows as *expected* evidence
   (`score198.py:124-125`, `:131-132`, `:137-138`), so a stream that contains one is a stream the design anticipates;
   M2 turns it into an unscoreable stream whenever a case row for the same id exists too.
2. **The real duplicate is invisible.** `main()` writes its run-level failure with no `case_id`
   (`run198.py:1697-1701`), and `registered_rows` skips null keys (`score198.py:24-25`). Two such rows are a duplicate
   terminal row for the same (null) key and are never seen (`probe/p3b_main_dup.py`):
   ```
   failure | case_id: None | RuntimeError: pre-data gate failed ...
   failure | case_id: None | RuntimeError: pre-data gate failed ...
   registered_rows -> [{'record_type': 'case', ... 'case_id': 1, ...}]
   ```
   So the property the fold claims ("refuse to score a stream with conflicting terminal rows for the same key") is
   not true for the row type the runner emits most often on failure.
3. **The payload is too strict.** `_terminal_payload` excludes only `run_id` and `timestamp`. A second process that
   recomputes a row (a replay, a re-run on a machine with a different thread count) will differ somewhere in the
   floats — `names_after_write[*].pairs[*].z`, `statistics_after_write`, `localization` — and the stream becomes
   unscoreable with a message that names the case but not which field differed. `probe/p16_armb_dupe.py` shows a
   single float difference raising.

Smallest fix: key on `(record_type, key)` and treat a `failure` row as terminal only when no `case` row exists for the
key (or: keep the first *case* row, and refuse only a second *case* row, comparing a small set of decisive fields —
`took`, `name_eligible`, `restored_bit_exact`, `undo_zero_exact` — rather than the whole payload); include null-key
`failure` rows in a separate count and refuse only when that count exceeds one *per run-level phase*. Whatever the
rule, add a test that a `failure` row followed by a `case` row for the same id scores, and that two run-level failure
rows are reported.

### N2 (Major, INTRODUCED) — `tensor_fingerprint` raises on a 0-d tensor, and the failure path then reports "not restored"

`run198.py:574-587`: `value.view(torch.uint8)` rejects a 0-d tensor ("self.dim() cannot be 0 to view Float as Byte").
A 0-d parameter or buffer is legal in PyTorch (a scalar logit-scale buffer is the common case). Measured
(`probe/p6_fingerprint.py`):

```
(a) 0-d tensor raises: RuntimeError self.dim() cannot be 0 to view Float as Byte (different element sizes)
```

Consequence in the runner: `model_state_snapshot(model)` at `run198.py:885` is outside the per-case `try`, so a model
with such a tensor dies at startup for every Arm A run. In the failure branch (`:1006`, `:1171`) and the verification
branch (`:1066`) the same exception is caught and turned into `restored_after_error = False`, which is "fail closed"
and therefore safe for the gate, but it means a model that *was* restored correctly is recorded as a restoration
failure. GPT-2 XL has no 0-d parameters today (I did not open the weight file to confirm; the config and the
transformers GPT-2 implementation have none), so this is latent, not live. Smallest fix:
`raw = value.reshape(-1).view(torch.uint8)`.

### N3 (Major, INTRODUCED) — the new tests prove the fingerprint helpers, not the runner's use of them

`test_build198.py:207-226` builds a two-tensor `TinyState` and calls `model_state_snapshot` / `model_matches_snapshot`
directly. Nothing in the suite drives `run_arm_a` or `run_arm_a_mass`. Four sabotages of the runner's restoration
paths stay green (`probe/p8_sabotage.py`, `probe/p13_sabotage3.py`, `probe/p15_sabotage4.py`):

```
GREEN (suite missed it) :: restoration compares only the parameters the editor returned
GREEN (suite missed it) :: failure path no longer re-checks restoration (M4 failure-path compare removed)
GREEN (suite missed it) :: snapshot taken AFTER the first edit instead of before (M4 ordering)
GREEN (suite missed it) :: state_verification success path compares only the editor's keys again
GREEN (suite missed it) :: mass failure branch no longer compares
GREEN (suite missed it) :: state_verification exception path reports restored True unconditionally
```

The first one is exactly the round-1 M4 defect, reintroduced, and the suite does not notice. Why it matters: the
design's "the weights must equal the base bit for bit" (PART198-DESIGN.md:125) is now enforced by code the tests do not
reach, and the fold's own claim ("`test_build198.py:226` fails on pristine code; it mutates an unreported buffer and
confirms detection") is true of the helper and not of the runner. Smallest fix: a test that runs `run_arm_a` with
`load_gpt2_model`, `stats_helpers`, `load_easyedit`, `load_hparams` and `apply` stubbed to a tiny module whose editor
writes an unreported buffer; assert the case row records `restored_bit_exact: False`. (Same shape for the mass path.)
A cheaper structural test — `inspect.getsource(run_arm_a)` contains four `model_matches_snapshot` calls and the
snapshot precedes the first `apply` — would also catch three of the six.

### N4 (Major, PRE-EXISTING) — a crash between Arm B's case row and its undo check wedges the stream

`run198.py:1315` writes the case row, then the random control and the undo check follow; `existing_keys` (`:168-179`)
counts the case row, so the resume skips the case and the undo check is never redone. Measured
(`probe/p29_armb_dupe2.py`): `resume prior: ['3'] -> case 3 will be skipped`, and the gate then reports
`Arm B WSP undo zero check missing/false` forever. Not introduced by the fold (the ordering is in `65c2be6`), and the
fold does not reach it. Follow-up: write the case row last, or make `existing_keys` require `undo_zero_exact` present.

### P1 (Major, PRE-EXISTING) — a deterministic Arm C failure loops forever across runs

`run198.py:1586-1591` and `:1629-1634` write a failure row and re-raise; the next registered run retries the same
model. Measured (`probe/p22_armc_deadlock.py`): three registered runs, three failure rows, two calibration models done,
`tstar.json` never written, evaluation and host models never run. Nothing counts attempts. The round-1 review asked
for "retried, not counted as done", and the fold does that; the loop is the other half of the same coin. Follow-up:
cap retries per model (e.g. refuse after N failure rows for the same `model_id`), or make the failure terminal for the
*registered* mode after the first occurrence and require an explicit `--retry-failed`.

### P2 (Minor, PRE-EXISTING) — a single counted mass batch is enough for a 198.1/198.3 verdict

`run198.py:270-271` (`mass_gate_status(n) = n > 0`) is now called by the scorer (`score198.py:154`), so one
gate-passing batch yields `PASS` with CP95 `[0.0, 0.975]` (`probe/p11_mass_status.py`). The design's denominator for
198.1/198.3 is "batches 0 of 2" (PART198-DESIGN.md:377, :379), i.e. two batches per method. The helper is
pre-existing; the fold makes it reachable on the scorer path. Follow-up: `mass_gate_status(n) = n == 2` (or `n >= 2`).

### P3 (Minor, PRE-EXISTING) — the blind reader's input does not depend on the edited set, so "localized" can be a base-layer win

`pair_by_layer` (`run198.py:672-673`) hands the reader base pairs for unedited layers and current pairs for edited
ones, and `stats198.blind_reader` picks the layer with the highest T_word over all of them. Measured
(`probe/p12_blind.py`): the same blind layer and token come out for `edited={2}`, `edited={3}` and `edited=set()`. So
the reader's answer is a function of the statistics map, not of what was edited; a base layer can win and be scored
`localized: False` (correctly — it is not in `edited_layers`) but its *token* is still what 198.7's "named" half reads.
This is the design's own construction (the blind reader is base-required and edit-blind by definition), so it is a
follow-up question for the design, not a code defect. The fold does not change it.

### N5 (Minor, INTRODUCED) — test gaps that let a fix be removed silently

- `test_build198.py:266-274` (M8) checks `crosscheck_hash_matches` and greps `preflight`'s source for the name; it
  does not assert that `preflight`'s `crosscheck_pass` depends on it. Removing `and stats_hash_matches` from
  `run198.py:417` stays green (`probe/p9_sabotage2.py`: "GREEN (suite missed it) :: crosscheck hash check removed
  from preflight"). The behaviour itself is correct (`probe/p10_preflight.py`).
- `test_build198.py:228-246` (M5) checks `arm_c_successful_keys` and `require_tstar` in isolation; reverting
  `prior = arm_c_successful_keys(log)` to `existing_keys(log, "model_id")` stays green (`probe/p8_sabotage.py`).
- `test_build198.py:241-246` and `:258-264` (M5, M7) are source-text assertions (`inspect.getsource(...).index(...)`),
  not behaviour; BUILD-NOTES is candid about M7 but not about M5's structural part.
- `test_build198.py:119-134` (B1) has an `except ImportError` fallback that asserts a hard-coded number
  (`0.6261582865`) — the value the fixed code returns, not an independent computation. With scipy present (it is) the
  branch is not taken, so this is a latent "test that passes for the wrong reason" on a machine without scipy.
- The suite exits 0 having run nothing if its body is emptied (`probe/p20_suite_skip.py`: `rc 0 | TEST_BUILD ALL PASS
  (0 failed)`), because it is a flat script with no count floor. A `check(len(checks) >= 30)` at the end would close it.

### N6 (Minor, INTRODUCED) — the S1 "missing control" count misses the runner's own zero-direction row

When `make_aimed_delta` returns `None`, the runner stores `{"reason": …}` (`run198.py:971-972`), which is a dict, so
`s1_missing_controls` (`score198.py:212`) counts it as a present control with `oracle_named_strict` absent — a failure
in the denominator (correct) but not reported as missing (incorrect). Measured (`probe/p7_s1_missing.py`):
`missing_controls: 0` with one such row. Smallest fix: count a control missing when it lacks
`oracle_named_strict` (or when it has `reason`).

---

## What I checked and found sound

- **B1 numbers.** `cp_interval(20,24) = (0.6261582865113173, 0.9526463733927428)`; the exact beta quantiles are the
  same to 1e-16. The fallback bisection (no scipy) was checked by reading: `ge(k,p)` is increasing, `a` moves up while
  the tail is above `alpha/2`, so it converges to the lower bound; the upper loop is unchanged and was already right.
- **Restoration ordering and coverage.** Snapshot is taken after `load_gpt2_model`'s clone and before the first
  `apply` in both Arm A paths; every success and failure branch compares the full state; the post-loop
  `state_verification` compares on both its success and exception paths (`probe/p14_restore_paths.py`).
- **Fingerprint correctness.** Non-contiguous `.T` and a strided view hash the same as their contiguous copies; shape
  and dtype are in the digest; an in-place edit through the same storage is detected; a parameter replaced by a new
  object with identical bytes matches (correct: the bytes are what the design cares about); a dtype change is
  detected. The only false *negative* risk would be a hash collision.
- **Arm C.** `--limit` loads exactly `limit` models; a smoke row never satisfies a registered key
  (`probe/p18_armc_smoke.py`); a smoke failure never blocks a registered run (`probe/p23_armc_partial.py`); T\* is
  computed only from the three calibration maxima even when an evaluation model reports a larger value
  (`probe/p5_armc_rerun.py`); a missing `max_t_word` raises before `tstar.json` is written
  (`probe/p4_armc_tstar.py`).
- **Arm B.** `wsp_take_measurement` takes `took` from `after_exact` and records the WSP self-report beside it; a
  malformed `free_exact` yields `None` counts and no fallback; both callers use it.
- **Old-code audit.** 17 failures on pristine `65c2be6`, 0 on HEAD, as claimed.
- **`stats198.py` and `fixtures198.py` are untouched** by the fold (`git diff --stat 65c2be6..HEAD` lists only
  `BRIEF-198-fold-r1.md`, `BUILD-NOTES.md`, `crosscheck198.py`, `run198.py`, `score198.py`, `test_build198.py`).
- **BUILD-NOTES claims I could check are true** except where noted above (B2 "already satisfied", M7 "the review's
  load-count bug was not reproduced", M8 "artifact intentionally not regenerated" — the on-disk `crosscheck198.json`
  indeed predates the hash and `preflight` will refuse until it is regenerated, which BUILD-NOTES states).

## Method note

Probes are in `probe/` in this directory; pristine `65c2be6` modules are in `old/`. `run_suite_against.py` runs the
HEAD suite against any module directory. No writes outside this directory; no live service, no credential file, no
model or weight file was opened. `probe/p3b_main_dup.py` and `probe/p3c_main_dup.py` bypass `root_out_dir`'s
containment guard in-process only to write into a temp directory; the guard itself is untouched and was confirmed to
refuse a `/tmp` out-dir before the bypass was added.
