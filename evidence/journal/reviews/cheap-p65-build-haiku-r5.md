<!-- cheap-review.mjs cheap-review-p65-build-haiku-r5-2026-10-24: /home/elisepc/green/worktrees/ab-part65 3629a47..ac95b47, lane haiku, df-run exit 0 -->
# Cold review: Part 65 judge build (3629a47..ac95b47)

**Verdict: SHIP-WITH-FIXES.** No path found that sends one chunk twice across the crash points I could drive, and a
released chunk stops the same plan. But the judge can be re-run under a different seed after a validated release and
still complete (F1, HIGH), and two tests named for the release and durability guarantees stay green when those
guarantees are removed (F2, F3). Fix F1–F3 before the full 1,000-trial plan is authorised. F4–F8 are pre-existing
follow-ups, not holds.

Scope: `judge65.py` (fully read), `analyze65.py` (completion and run-identity checks), `paidaction/paid_ledger.py`
(reserve/settle/release, sub-cap), `steer65.py` (`_lock_file`, `atomic_append_jsonl`), `test_part65.py` (outline, ledger
usage, the tests named below), `PART65-DESIGN.md` Amendment 4, `RUNBOOK-65.md`. Base-vs-head diff of `judge65.py` used
to classify each finding.

Environment limits: `torch` is not installed here. Ran the suite with a stub `torch` (`stub/`), which is not real torch.
Result: 51 run, 43 pass, 0 fail, 8 error. All 8 are tensor tests (`ModuleList` missing from the stub), not judge tests.
The real-torch 51/51 claim in the commit could not be checked here. The sub-cap test writes next to the test file, a
read-only path in this sandbox, so I ran an unmodified copy under `mut/`; it passes there.

## Findings

### F1 — HIGH — INTRODUCED — a released judge completes again under a different seed

`judge65.py:725-739` (`_prior_reservation` filters the plan fence on `plan_sha256`) and `paidaction/paid_ledger.py:454-466`
(the sub-cap counts only reserved/settled/retired/overrun, so a released reserve is free). Amendment 4 item 1 and
RUNBOOK:89 say a released chunk is terminal for the registered judge and "a later judge needs its own registered plan".
The code enforces that only for byte-identical plans.

Measured (`measure.py s2`; real `judge65` + real `Ledger`, fake provider):
```
A totals {'released': 1} accepted_posts 0 chunk_states ['released', 'ready']  ledger reserved 0.0
B same plan refused: Part 65 plan already has a paidaction reservation owned by another run; ...  accepted 0
C new-seed plan totals {'settled': 2} accepted_posts 2 chunk_states ['settled', 'settled']
C run.json completion: True
```
Why it matters: after the registered judge failed (its missing trials exceed the 50-trial stop), one CLI change (`--seed`)
starts a fresh full plan. It sends, settles, writes a completion marker, and `analyze_files` accepts it. Its $1 sub-cap
is fully available, because the released reserve is no longer counted.

Smallest fix: make the terminal latch judge-wide, not plan-wide. In `process_chunk`, before reserving, refuse if any
`part65-judge` reserve in the ledger has status released, retired, overrun or ambiguous-unsettled, whatever its plan hash.
Alternatively, record the registered plan hash in a file the CLI checks. Owner decision: Amendment 4 says "for this plan";
the latch is stricter than the text.

### F2 — MEDIUM — INTRODUCED (test gap) — the release test does not exercise "stops later chunks"

`test_part65.py:600-654` (`test_validated_release_is_terminal_stops_later_chunks_and_main_exits_2`, `ReleaseSecondChunk`
at 609-621, `posts == 2` at 617). The plan is `trials[:4]` at chunk size 2, so the released chunk is the last one. The
mutant that removes the stop on a released outcome (`judge65.py:906`, `if outcome in ("released", "ambiguous")` →
`("ambiguous",)`) still passes all tests.

Measured (`measure.py s7`, 6 trials, chunk size 2, validated 400 on chunk 0):
```
real code : totals released 1  accepted_posts 0  states ['released','ready','ready']
mutant M1 : totals settled 2 released 1  accepted_posts 2  states ['released','settled','settled']
```
Suite on M1: `Ran 51`, failures 0, errors 8 (the same torch stub errors). Log: `mut-M1_no_break_on_release.log`.

Why it matters: this is the sole guard against sending later chunks after a released one, and the RUNBOOK tells the
operator to rely on it. Smallest fix: release chunk 0 of three and assert `posts == 1` and that chunks 1-2 stay `ready`.

### F3 — MEDIUM — PRE-EXISTING behaviour; the change claims it and no test pins it — durable pre-send record

`judge65.py:820-821` (`state["status"] = "posting"` then `_save_chunk`). Base has the same lines at `judge65.py:679-680`
at 3629a47. Amendment 4 item 3 relies on this write, but removing it leaves every test green.

Mutant M3 (save removed) and a crash after the provider accepted the batch, before the `submitted` save:
```
real code : first=RuntimeError(...submitted) accepted_after_first=1 second=ambiguous TOTAL_ACCEPTED_POSTS=1 states=['posting']
mutant M3 : first=RuntimeError(...submitted) accepted_after_first=1 second=settled   TOTAL_ACCEPTED_POSTS=2 states=['settled']
```
Suite on M3: `Ran 51`, failures 0, errors 8. Log: `mut-M3_posting_marker_not_durable.log`. Test `test_ambiguous_posting_marker_is_never_reposted`
(`test_part65.py:1034`) does not catch it, and `test_atomic_json_fsyncs...` (1065) tests only the helper.

Smallest fix: add the crash-after-accept case from `measure.py` (`crash_case` with `_save_chunk` raising on `submitted`)
as a unit test asserting one accepted POST across two invocations.

### F4 — MEDIUM — PRE-EXISTING — a pending chunk does not stop the loop

`judge65.py:900-912`: `break` only on `released` or `ambiguous`. `pending` (deadline hit while `in_progress`, `:546`) lets
the next chunk POST while the earlier batch is still in flight.

Measured (`measure.py s1`, 4 trials, chunk 0 never ends, timeout 0):
```
totals {'settled': 1, 'pending': 1, 'released': 0, 'ambiguous': 0} accepted_posts 2 chunk_states ['submitted', 'settled']
```
The contract names released and ambiguous only, so this is a follow-up. Consider breaking on `pending` too and letting
`--recover` resume, which is what RUNBOOK:89 describes.

### F5 — LOW — PRE-EXISTING — a torn final line in `answers.jsonl` silently drops the next answer

`judge65.py:78-81` (`read_jsonl` ignores an invalid final line with only a warning), `judge65.py:493-517` (`_persist_answers`),
`steer65.py:484-497` (`atomic_append_jsonl` appends without repairing the tail). A crash mid-append, then a normal append,
merges the new row into the fragment.

Measured (`measure.py s9`):
```
persist ok; file now:
    {"judge_run_id": "rrr...", "custom_id": "p65-0000", ...
    {"judge_run_id": "rrr...{"judge_run_id":"rrr...        <- new p65-0001 row is inside the torn line
re-read ok                                                   <- the warning-only tail read hides the loss
```
Consequence I read from the code, not measured end to end: `_save_run_outcome` (`judge65.py:699-705`) withholds the completion
marker, so analysis refuses. But `main` (`judge65.py:1022-1023`) returns 0 whenever no chunk is pending, ambiguous or released,
so the CLI can exit 0 without a completion marker. The next persist then fails on the now-mid-file bad line. Smallest fix:
repair or reject a torn tail before appending (as `paid_ledger._append_locked` does), and return 2 when completion is absent.

### F6 — LOW — PRE-EXISTING — a torn ledger tail blocks recovery (fail-closed)

`judge65.py:715-722` (`_prior_reservation` raises on any invalid line) versus `paid_ledger.py:256-262` (the ledger ignores an
unterminated final line). Measured (`measure.py s5`): with one valid row and a torn tail, `run_chunks` raises
`paidaction ledger has an invalid line`, with 0 accepted POSTs, while `Ledger.state()` reads the same file. It cannot
double-send, but a crash mid-ledger-write blocks the judge until someone repairs the file by hand. Smallest fix: apply the
ledger's torn-tail rule in `_prior_reservation`.

### F7 — LOW — PRE-EXISTING — the probe's one request is sent again in the full plan

`judge65.py:265-268` (`build_probe_trial` returns `trials[0]`), `judge65.py:978-980` (probe in `out/probe`, full run in `out`).
Measured (`measure.py s6`): probe custom_id `p65-0000`, identical request body to the full plan's first request. One
extra ~$0.0001 request; not a chunk double-send. Smallest fix: note it in the RUNBOOK or drop the probe request from the
full plan.

### F8 — LOW — PRE-EXISTING — a crash between the posting marker and the POST leaves a chunk with no resume path

Measured (`measure.py s3`, `crash_before_post_with_posting_marker_durable`): state `posting`, `ambiguous` on rerun, zero POSTs,
reserve `4.23e-05` still active. Correct under "never resubmit", but RUNBOOK has no procedure for it and `--recover` only
GETs. Smallest fix: a documented manual procedure (retire the attempt and start a new plan) that the RUNBOOK names.

## What I checked and found sound

- **No double send across crash points.** `measure.py s3`, eight points between reserve and settle: total accepted POSTs
  per case was 0 or 1 after two invocations. Covered: crash after the reserve row, after the `reserved` save, before the
  POST, after an accepted POST, lost POST response (ambiguous), settle crash, settled-state save crash, partial answers.
  Settlement is idempotent for the same usage digest (`paid_ledger.py:488-491`), so the `results_saved` replay is safe.
- **Same plan, another directory** is refused before any POST while the first run is pending or released (`measure.py s2`
  and `s4`). The plan check and the POST sit under the ledger-path lock (`judge65.py:777-863`, unchanged from base).
- **Released is terminal within a run and across reruns.** A rerun on the released chunk returns `released` without
  reserving (`test_released_chunk_is_final_and_does_not_reserve_again`); `run_chunks` stops at it (`judge65.py:906-907`).
- **Ambiguity stops the loop.** Every POST-outcome error that is not a validated 4xx contains "ambiguous" or "no batch ID",
  so the `break` at `judge65.py:909-911` triggers. The non-ambiguous errors propagate out of `run_chunks`, so no later chunk is sent.
- **Run identity.** One `judge_run_id` per directory, checked in the manifest, every chunk, every reservation and every answer
  row; `analyze65.py:612-643` refuses mixed or mismatched provenance; `_persist_answers` refuses mixed runs.
- **`--reserve-total` sends nothing.** Code path: `prepare_inputs` plus `_reserve_total`, no ledger and no network. Test
  `test_reserve_total_prints_real_plan_estimate_without_sending` passes.
- **`--ledger` removed; key checked before reserve.** `main` no longer takes a ledger path, and `run_chunks` calls `_api_key()`
  before taking any lock.
- **Real-ledger sub-cap** (`test_real_ledger_subcap_refuses_before_post_and_main_exits_2`) passes against the real `Ledger` in
  the unmodified copy.
- **Test isolation.** `main()` is always called with `run_chunks` patched, or with a fake ledger, so no test touches
  `DEFAULT_LEDGER_PATH`. I never used that path or any credential file.

## Not checked

- Concurrent `prepare_run` into one fresh directory (no lock). I expect it to fail closed through the chunk run-id check,
  but did not measure it.
- The eight tensor tests, which need real `torch`.
- End-to-end CLI exit code for F5, which I took from reading `main`.

## Files (under `/home/elisepc/green/Journal/scratch/review-p65-build-haiku-r5-2026-10-24/`)

`measure.py` (drives the real code; `s1`–`s9`), `make_mutants.py` (M1–M4, each a single exact-match edit in a copy),
`mut/` (copy of the head code), `stub/torch/` (stub, not real torch), `run-tests-stub.txt`, `unmutated-suite.log`,
`mut-*.log`. Every ledger and run directory is under `runs/`. Nothing under `DiscordBot` or `Journal` outside this domain
was written. No live service was called.
