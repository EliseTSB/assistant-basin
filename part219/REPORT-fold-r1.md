# Part 219 — fold of the cold code review r1 (F1–F10)

Base: `e4fad5b0` (the Part 219 build commit). Review: `Journal/scratch/review-part219-code-r1-ds-r1-2026-11-22/REVIEW.md`
(SHIP-WITH-FIXES, F1–F10). Nothing committed. `PART219-DESIGN.md`, `PREDICTIONS.md` and every threshold are untouched.
No GPU, no real model weights; every run below is on the review's tiny 3-layer/2-head OLMo 2 or the test fixture.

Files changed: `part219/score219.py` (F2, F3, F4, F8, F9, F10, plus one related headline bug), `part219/run219.py`
(F7 kernel check), `part99/swap99.py` (F1, F5, F6), `part219/test_part219.py` (18 new tests, 2 existing ones
strengthened, the synthetic kernel record made eager), `part219/REPORT.md` (the two wrong claims and the counts).
Seven probe copies (`part219/fold-probe-*.py`) are the review's own scripts, with only the worktree path and (where
the original wrote into the review's now read-only folder) the output directory made local; they are what the
"probes rerun" section below ran.

## F1 — a default Part 99–104 run wrote `attn_implementation: "default"`

`swap99.py` recorded the raw argparse value (`attn_option`, the string `"default"`) instead of the normalised one, so
`write_meta` (which drops only `None`) gained a key on every default run. Fixed: the extras carry
`attn_implementation`, which is `None` for `default` and `"eager"` when asked. The regression test now drives
`swap99.main()` through its real CLI with no Part 219 option, so argparse's default is what is under test; a
Namespace without the attribute could never see this defect.

Test: `test_a_default_run_writes_no_part219_keys`. Sabotage: put `attn_option` back →
`1 failed` (the meta gains `attn_implementation: "default"`).

Probe `fold-probe-default.py`, old (`ab800603`) vs fixed `main()`, same world, no options:

```
before: 0a1  > attn_implementation          before: 1a2  > "attn_implementation": "default",
after : old-only keys []   new-only keys []   attn_implementation in new meta: False
        rows identical: True (30 rows, keys id/unit/question/question_index/answer only)
```

## F2 — an unread run still served as a control and still supported its headline

Fixed in `build_predictions` and `build_headlines`: `R5_rig_check` and `R9_coupling_check` are `held: False` when
their run's `status` is non-empty (the net count is still reported in `detail`, with the status beside it); 219.1/219.2
are "not read" when R5 is unread, 219.3–219.5 when R9 is unread. `build_headlines` takes `status` and withdraws every
headline that depends on an unread run: R5 unread withdraws all of them, R2 unread the "needed" headline, R8 or R9
unread the transplant headline, R1 unread only the 219.4 clause. R10 is deliberately not in that set (F4).

Tests: `test_an_unread_r5_cannot_serve_as_its_rig_check`,
`test_an_unread_r9_cannot_serve_as_its_coupling_check`,
`test_an_unread_r2_withdraws_only_the_need_headline`,
`test_an_unread_r8_withdraws_the_transplant_headline`.
The R2/R8 checks were already right at HEAD via the prediction rules; they are regression tests, not new fixes.
Sabotages: drop `not r5_status` and disable the R5 control/headline branches → `1 failed`; the same for R9 → `1 failed`.

Probe `fold-probe-score.py` (the review's `score_probe.py`), before → after:

```
R5 unreadable   controls R5 True, both headlines, all held
                -> R5_rig_check false; 219.1/219.2 "not read: R5, the rig check, is not read (...)";
                   headline "no headline: R5 is not read (...), so its rig check is not read"
R9 unreadable   controls R9 True, both headlines
                -> R9_coupling_check false; 219.3/219.4/219.5 "not read: R9 is not read (...)";
                   only the "needed" headline
R8 unreadable   unchanged in kind (219.3–219.5 already "not read"), headline now asserted
                to be the "needed" one alone
```

## F3 — the gate passed on a run the scorer itself marks unread

`build_gate` now takes `status` and an unread R0 or R1 fails the gate item it feeds (the check carries the reasons and
a `reason` string). Because R0's label counts are the reference every net is paired against, an unread R0 freezes
every prediction; an unread R1 fails the R1 bound for the same reason.

Tests: `test_an_unread_r0_freezes_every_prediction` (manifest unreadable), and the review's own score_probe case
`test_a_failed_r0_two_pass_gate_fails_the_gate` (39 of 42 agree, counts 19 vs 16), plus
`test_an_unread_r1_fails_the_gate`. Sabotage: drop `and not reasons` from both gate items → `3 failed`.

Probe, before → after:

```
R0 two-pass gate fails (39/42)
  before: gate_passed true; 219.1..219.5 all held; both headlines
  after : gate_passed false, failures ["r0_slot_persona"];
          219.1..219.5 "not read: gate failed: r0_slot_persona";
          headline "no headline: the gate failed, so every prediction is frozen and reported as exploratory"
R0 manifest unreadable, R1 manifest unreadable: the same frozen scorecard (probed inline; also the two tests).
```

## F4 — an unreadable R10 made 219.5 "not read"

Fixed: 219.5 is computed from R10's net whenever R10 scored; `r10_status` is reported on the 219.5 entry (empty list
when R10 is clean) and does not block it. Only a missing R10 (not scored) leaves 219.5 "not read".

Test: `test_an_unreadable_r10_does_not_block_219_5`. Sabotage: restore the `or status.get("R10")` block → `1 failed`.
Probe: `R10 receipt unreadable` before gave `219.5 "not read: R10 is not read (...)"; 219.4 not read; one headline`;
after gives `219.5 held`, `219.4 held`, both headlines, with `r10_status: ["the manifest marks it unreadable"]`
recorded on 219.5.

## F5 — the transplant receipt could not fail for a wrong slice

`transplant_heads`' running receipt now: (a) re-derives each span from the config by the named head alone
(`[head*head_size, (head+1)*head_size]`), (b) asserts the planned `[start, end]` equals it (`slice_from_config`), and
(c) compares the bytes now in that **config** span with a fresh, independent read of the donor's same span (or
`2*original_check - donor_check` in float32 for `opposite`), where `original_check` is the host bytes at the config
span captured before the write. A wrong plan therefore fails `matches_expected` rather than passing tautologically.
The slice-index rule is one helper (`head_slice`) used by the plan and by the independent re-read.

Test: `test_the_transplant_receipt_catches_a_shifted_span` (the review's sabotage: every span shifted one head).
Sabotage: restore `matches = torch.equal(written, expected)` → `1 failed`.
Probe `fold-probe-transplant.py`:

```
before: orientation failures []; restore bit-exact True;
        "receipt for the shifted write: head 0 slice [8, 16] matches_expected True -> all records pass: True"
after : orientation failures []; restore bit-exact True;
        "receipt for the shifted write: head 0 slice [8, 16] matches_expected False -> all records pass: False"
```

## F6 — a raise before the generation `try/finally` left the model changed

`run()` now has one `try/finally` from the first mutation (Part 99's `--swap`, then the transplant probe/copy and the
ablation hooks) to the end of the generation loop. The `finally` removes hooks, restores the transplant's original
bytes and restores the swap snapshot. `transplant_heads` and `install_ablation_hooks` also clean up their own partial
writes/handles if they raise part-way. The swap snapshot is now also taken when the caller preloaded the model
(`not owns_models`), because that caller keeps the model; a single-tag Part 99–104 command that owns its model still
clones nothing unless `--reference-replay` was given, so their memory behaviour is unchanged.

Tests: `test_a_raise_after_the_transplant_restores_the_host` and `test_a_raise_after_a_swap_restores_the_host` (the
review's cases A and B). Sabotage: change the outer `finally:` to `except BaseException: raise` + `else:` (restore
only on success) → `2 failed`.
Probe `fold-probe-fail.py`, before → after:

```
case A raised: forced probe failure
 before: parameters left changed after the raise: 12 entries (layers 0 and 2: q/k/v/o_proj, q_norm, k_norm)
 after : parameters left changed after the raise: []
case B raised: forced medians failure (after --swap attn --layers 0-1)
 before: parameters left changed after the raise: 12 total
 after : parameters left changed after the raise: [] ... 0 total
```

## F7 — two REPORT.md claims the code did not implement

- `REPORT.md` decision 1 and the comment at `swap99.py:842` said the empty transplant set "still runs the planner".
  It returns from `transplant_heads` before the geometry/donor/shape planning. Both texts now say exactly that: R0b
  exercises the empty early return (and the empty ablation hooks), not the planner.
- `REPORT.md` said `run219` marks a run unreadable "when its … kernel record is wrong", but it had no kernel check.
  Now it does: `run219.check_kernel_record` (one implementation, delegating to `score219.kernel_check`) runs at the top
  of `run_stage`, before any model load, and refuses with `SystemExit` unless the record exists, passes
  `tools/part_check.kernel_check`, and records the eager kernel. `--kernel-check` defaults to
  `results/part219/kernel-check.json`, and the result is written into the manifest.

Tests: `test_run219_refuses_to_launch_without_a_passing_kernel_record`,
`test_run219_refuses_a_non_eager_kernel_record`, `test_run219_accepts_a_passing_eager_kernel_record`. Sabotage:
replace the check with `{"passes": True}` → `2 failed`. The committed `results/part219/kernel-check.json` still
passes under the F8 rule (`passes: true, failures: []`), and `run219.check_kernel_record` accepts it. An end-to-end
`run219.py --stage a --pilot --limit 3` on the review's 16-head world, with a synthetic eager record, ran R0 and R2,
recorded `kernel_check.passes: true` in `manifest.json`, and still reported `pending_labels`.

## F8 — the gate's kernel check accepted a non-eager record

`score219.kernel_check` now requires `attn_implementation == "eager"` on the candidate, and on the `load` block when
the record carries one (a non-object `load` is itself a failure). The `part_check` result and the eager failures are
merged into one `failures` list; `reason` summarises them.

Test: `test_the_kernel_check_requires_the_eager_kernel` (as-committed eager passes; candidate `sdpa`, candidate
missing, `load` `sdpa`, `load` missing all fail with an "eager" reason). Sabotage: disable both eager checks →
`1 failed`. The review's measured table now reads: eager → passes; `attn_implementation=sdpa` → FAILS; missing →
FAILS; `load.attn_implementation=sdpa` → FAILS.

## F9 — a malformed kernel record crashed the scorer

The `except` in `kernel_check` is broadened to `(OSError, ValueError, KeyError, TypeError, AttributeError)` and a
top-level non-object record is rejected up front; every path returns `passes: False` with a `reason`.

Test: `test_a_malformed_kernel_record_fails_the_gate_instead_of_raising` with the review's three records
(row-is-empty-object, batch-value-is-dict, prompt-key-missing): each fails the gate with a reason. Sabotage: narrow
the `except` back to `(OSError, ValueError)` → `1 failed` (KeyError escapes).

## F10 — a not-read 219.4 was printed as the "219.4 failed" claim

`build_headlines` now emits the 219.3/219.5 headline with **no clause** when 219.4 is not read; only
`verdict("219.4") == "failed"` emits "it is part of what the early-layer swap removes, not all".

Tests: `test_a_not_read_219_4_adds_no_clause_to_the_headline` (unit, the rule itself; sabotage: make the failing
clause the `else` again → `1 failed`) and `test_r1_unreadable_emits_no_219_4_clause` (the review's R1 case; asserts no
"early-layer swap" and no "not all" headline appears).

**Interaction with F3, stated plainly.** With the brief's F3 decision — a non-empty status on R1 fails the R1 gate
item — an R1-unreadable scorecard now freezes every prediction, so the R1 route no longer reaches "219.3 and 219.5
held, 219.4 not read". The R1 integration test therefore asserts the absence of any 219.4 claim (the gate-failed
headline), and the F10 rule itself is proven by the direct `build_headlines` test. If the intent was for R1's status
to bar only 219.4 while leaving the gate's R1 bound standing, F3's R1 half would have to be dropped; I kept the
stricter, design-consistent rule the brief states.

## Related bug fixed while folding F3

`build_headlines` tested the gate-failed state with
`any(str(value).startswith("not read: gate failed") for value in predictions.values())`. The values are dicts, so
`str(value)` is the dict repr and the branch never fired: a failed gate emitted the generic
"no headline: the predictions it rests on did not hold" instead of the design's gate-failed headline. Fixed to read
each value's `verdict`, and `test_failed_gate_freezes_every_prediction` now asserts the gate-failed headline.

## New tests (18) and their sabotages

| # | test | sabotage that makes it red |
| --- | --- | --- |
| 1 | `test_the_transplant_receipt_catches_a_shifted_span` | receipt compares `written` against `expected` at the same planned index (the old tautology) |
| 2 | `test_a_raise_after_the_transplant_restores_the_host` | restore only on success (`else:` instead of `finally:`) |
| 3 | `test_a_raise_after_a_swap_restores_the_host` | the same |
| 4 | `test_an_unread_r5_cannot_serve_as_its_rig_check` | R5 status ignored by `rig_ok`, the control and the headline guard |
| 5 | `test_an_unread_r9_cannot_serve_as_its_coupling_check` | R9 status ignored by `coupling_ok` and the block |
| 6 | `test_an_unread_r2_withdraws_only_the_need_headline` | regression test (already blocked at HEAD via `r2_status`); the explicit dependency guard is the extra belt |
| 7 | `test_an_unread_r8_withdraws_the_transplant_headline` | regression test (already blocked at HEAD via `r8_status`); same note |
| 8 | `test_an_unread_r0_freezes_every_prediction` | gate items ignore `status` |
| 9 | `test_a_failed_r0_two_pass_gate_fails_the_gate` | gate items ignore `status` |
| 10 | `test_an_unread_r1_fails_the_gate` | gate items ignore `status` |
| 11 | `test_the_kernel_check_requires_the_eager_kernel` | both eager checks disabled |
| 12 | `test_a_malformed_kernel_record_fails_the_gate_instead_of_raising` | `except` narrowed to `(OSError, ValueError)` |
| 13 | `test_an_unreadable_r10_does_not_block_219_5` | 219.5 blocked on `status["R10"]` again |
| 14 | `test_a_not_read_219_4_adds_no_clause_to_the_headline` | the failing clause becomes the `else` branch |
| 15 | `test_r1_unreadable_emits_no_219_4_clause` | with F3 in place it passes whenever no 219.4 claim is printed; the F10 rule is #14's |
| 16 | `test_run219_refuses_to_launch_without_a_passing_kernel_record` | the kernel check is skipped |
| 17 | `test_run219_refuses_a_non_eager_kernel_record` | the kernel check is skipped |
| 18 | `test_run219_accepts_a_passing_eager_kernel_record` | (acceptance path; shares the check) |

Also strengthened: `test_a_default_run_writes_no_part219_keys` now drives `swap99.main()` (F1) and
`test_failed_gate_freezes_every_prediction` asserts the gate-failed headline.

## Suite counts

```
$ /home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part219 part99 -q
63 passed in 9.85s

$ /home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part219 -q
49 passed in 9.75s
```

Before the fold: 45 passed (31 + 14). No test was skipped or xfailed, and no threshold or design line moved.

## Review probes rerun against the fixed code

The review's own scripts (copied as `part219/fold-probe-*.py`; only the worktree path and the writable output
directory changed) were rerun with the same interpreter. What changed:

| probe | before (REVIEW.md / its output) | after |
| --- | --- | --- |
| `score_probe` | R5/R9 unreadable: controls True, all held, both headlines; R0 39/42: gate passed, all held; R10 unreadable: 219.5 not read; R8 unreadable: 219.3–219.5 not read | R5/R9 controls False, predictions blocked, headlines withdrawn; R0 39/42: gate failed and everything frozen; R10 unreadable: 219.5 held with `r10_status`; R8 unchanged in kind; baseline unchanged |
| `fail_probe` | case A 12 changed parameters; case B 12 total | case A `[]`; case B `0 total` |
| `transplant_check` | shifted write: `matches_expected True`, all records pass | `matches_expected False`, all records pass False; orientation failures still `[]`, restore still bit-exact |
| `run_default` | new meta key `attn_implementation: "default"`; rows identical | no new keys; rows identical |
| `restore_probe` | no changed parameters, 0 hooks, all reset receipts as before | identical |
| `resume_probe` | first-batch receipt written, reset matches | identical |
| `ablate_probe` | slice zero on prefill and every decode step, other slices equal | identical |

The full `score_probe` output is `part219/fold-probe-score.py`'s stdout; the R0/R1/R2-unreadable cases are in
`test_part219.py` and were also probed inline:

```
R1 unreadable -> gate false, failures ["r1_slot_persona"], 219.1..219.5 "not read: gate failed", gate-failed headline
R0 unreadable -> gate false, failures ["r0_slot_persona"], the same
R2 unreadable -> gate true; 219.1/219.2 "not read: R2 is not read (...)"; 219.3/219.4/219.5 held;
                 headline "post-training's change ... accounts for the whole early-layer swap"
```

## SUMMARY

All ten review findings are folded. F1 records `attn_implementation` only when it is not the default and the default
path is now tested through the real CLI. F2 makes an unread R5/R9 unusable as a control and withdraws every headline
that depends on an unread R0/R1/R2/R5/R8/R9. F3 fails the gate on an unread R0 or R1, so the review's 39/42 R0 case
now freezes every prediction. F4 reads 219.5 from R10's net with R10's status reported beside it. F5's receipt
re-derives the span from the config and reads the donor span independently, so the shifted-span sabotage goes red.
F6 has one try/finally from the first mutation; both review failure cases leave the host bit-identical. F7 corrects
the two REPORT.md claims and run219 now checks the kernel record before launching. F8 requires the eager kernel in
the record (candidate and load). F9 fails a malformed record with a reason instead of raising. F10 gives a not-read
219.4 no clause. A related `build_headlines` bug (the gate-failed headline never fired because it stringified dicts)
was fixed too. `63 passed` for `pytest part219 part99 -q`; every named fix's test was shown red under a reverted-copy
sabotage; the review's probes were rerun and their outputs pasted above.

## FINDINGS

- `build_headlines`' gate-failed branch never fired at HEAD: it stringified each prediction dict. Fixed; the test now
  asserts the design's gate-failed headline. This is why the existing `test_failed_gate_freezes_every_prediction`
  (which checked only the verdicts) did not catch it.
- The R1 half of F3 and F10 interact (see F10): an unread R1 now freezes the gate, so the "219.4 not read beside held
  219.3/219.5" state is unreachable from R1. The F10 rule is still implemented and unit-tested; documented rather
  than silently weakening F3.
- `test_a_default_run_writes_no_part219_keys` had been passing for the wrong reason — it built a Namespace without
  the Part 219 attributes, so it could not see argparse's `"default"`. It now runs `swap99.main()`.
- The synthetic kernel record in the tests needed `attn_implementation`/`load` once F8 landed; every scorer test was
  updated through the single `write_kernel` helper rather than per test.
- Probe copies: `part219/fold-probe-{score,fail,restore,resume,transplant,ablate,default}.py`, with the worktree path
  derived from `__file__` and the writable outputs in `tempfile.mkdtemp()` (the review's `exp/` is read-only).

## BLOCKERS

None. One environment note: the system `python3` has no torch, so the brief's command was run with
`/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part219 part99 -q` (torch 2.14.1+cpu, transformers 5.20.0),
63 passed. No GPU, no real weights, no commits, and no design/threshold change.
