# Cold adversarial review: vastrun-r3-haiku

Reviewed: `git -C /home/elisepc/green/worktrees/ab-vastrun diff d45b8fe..HEAD` (HEAD `e7e0646`), `cloud/vastrun.py` (1496 lines, read by range and grep). Pre-change file extracted to `Journal/scratch/review-vastrun-r3-haiku-haiku-r3-2026-10-21/old/vastrun_old.py` to classify findings.
Probe scripts: `Journal/scratch/review-vastrun-r3-haiku-haiku-r3-2026-10-21/probes/` (`p1`, `p1b`, `p2`, `p3`, `p4`, `p5`, `p6`, `p7`). Each imports the REAL `cloud/vastrun.py` from the worktree and drives it with an in-memory Vast and a fake ssh. No network, no credential file, no live service.

## Verdict: SHIP-WITH-FIXES

F1 is the one to fix before anyone trusts `run-record.json` (`destroy_confirmed`, `cleanup_outcome`) as a statement about the box. It can report "already absent" while the box is rented, and nothing later can find that box. Its trigger needs the listing to omit our own label, which the live probe did not show, so it is conditional. F2 through F7 are smaller, and F8 and F9 predate this change. No finding is a leak in the live listing shape the design records.

## Findings, by severity

### F1 — MEDIUM (impact HIGH if triggered) — INTRODUCED
**Label-absence counts as "destroyed" even when a tracked id is still listed.** `cloud/vastrun.py:1083-1105`. The `if not matches:` branch clears `tracked_instances`, sets `active = None`, sets `destroy_confirmed = True`, sets `cleanup_outcome`, and pops `destroy_error`. It never checks the tracked ids against `instances`, which it already has. This contradicts `VASTRUN-SMALL-DESIGN.md:35`: "Keep the in-memory id as a second handle (DELETE by id ... whatever the listing says)." The label-only success rule itself predates this change (old `vastrun_old.py:741-746`), but this change adds the tracked-id drop and the clean record at that point.

MEASURED (`probes/p1_label_sweep_trusts_label.py`, destroy always fails):
```
--- strip_label=False
run raised: DestroyConfirmationError: could not confirm instance 101 destroyed: instance remained listed
record.destroy_confirmed = False
record.cleanup_outcome   = None
STILL RENTED in fake Vast: ['101']
--- strip_label=True
vastrun cleanup: already absent
run raised: DestroyConfirmationError: could not confirm instance 102 destroyed: instance remained listed
record.destroy_confirmed = True
record.cleanup_outcome   = already absent
record.destroy_error     = None
STILL RENTED in fake Vast: ['102']
```
Then `probes/p1b_orphan_visibility.py`: a fresh preflight passes, `sweep_expired` two hours later returns `[]`, and `sweep_label` for the run's label says "already absent". The box stays listed. With the label omitted, every later path (preflight, sweep-expired, sweep LABEL) cannot see it, so it bills until someone destroys it by hand.

Why it matters: the exit code stays non-zero (the earlier `DestroyConfirmationError` is kept as `failure`), but the run record and stdout both say cleanup is confirmed while a rented box is listed. It is the "success with a box still listed" case the brief asks about.
Smallest fix: in the `if not matches:` branch, compare `self.tracked_instances` keys with `{str(v["id"]) for v in instances}`. For any tracked id still listed, call `_destroy_tracked(...)` and do not return. Clear tracked and set `destroy_confirmed` only when none remain.

### F2 — MEDIUM (conditional on cross-host clock skew) — INTRODUCED
**The expiry rule trusts the local wall clock, so a reader clock ahead of the creator's destroys a live box.** Label deadline set at `cloud/vastrun.py:480-482` (`start + wall + 10 min + 25 min`). Compared at `:653-666` (`_preflight`) and `:1433-1441` (`sweep_expired`), both using `int(time.time())`.

MEASURED (`probes/p3_expiry_rule.py`, P3c, reader clock patched +40 min, box live by the creator's clock):
```
sweep_expired outcomes: [('vastrun-bbbbbbbbbbbb-6ad9196c', 'deleted, listing lagged')]
instances still listed: []
```
Why it matters: a live run's job is killed partway through (its box is gone) if the sweeping host's clock runs ahead by more than the remaining margin. A host clock can also jump after suspend/resume, so the margin is not a theoretical concern. The design (`VASTRUN-SMALL-DESIGN.md:34`) does not state a skew assumption.
Smallest fix: sweep only when `now > deadline + 3600` (or a similar margin), and state the cross-host skew assumption in the design doc.

### F3 — LOW-MEDIUM — INTRODUCED
**A foreign `vastrun-` label with no parseable deadline refuses every run until a human intervenes, and sweep-expired never mentions it.** `_label_deadline` returns None for anything not `vastrun-<12 hex>-<hex>` (`cloud/vastrun.py:222-229`). `_preflight` treats None as active (`:663-667`). `sweep_expired` skips it silently (`:1438-1441`).

MEASURED (P3a): `preflight refused: active vastrun instance(s) already exist: vastrun-manual-test (expires unknown)`; `sweep_expired outcomes: []`; the instance stays listed.
Why it matters: refusal is the designed behaviour, and `sweep LABEL` is the escape, but no automatic path clears it and the periodic sweep reports nothing.
Smallest fix: have `sweep_expired` print labels it skipped as unparseable, and document `sweep LABEL` as the resolution.

### F4 — LOW — INTRODUCED
**`sweep_label` absorbs a stop signal.** `RunInterrupted` is a `RunnerError` subclass (`cloud/vastrun.py:60`). `sweep_label` catches it at `:283` (`except Exception`, list path), `:298` (`except RunnerError`), and `:301` (`except Exception`, destroy path).

MEASURED (`probes/p2_sweep_swallows_stop.py`):
```
[destroy] sweep_label RETURNED 'deleted, listing lagged'; stop was absorbed; instances left=[]
[list] sweep_label RETURNED 'deleted, listing lagged'; stop was absorbed; instances left=[]
```
Why it matters: a SIGTERM during the preflight sweep is ignored for up to the sweep's 30 s budget. No rent follows, because `_check_limits` runs before the next rent, so the impact is delay only.
Smallest fix: add `except RunInterrupted: raise` before each broad handler in `sweep_label`.

### F5 — LOW — INTRODUCED
**The "refuse while a vastrun box exists" guard is check-then-rent with no lock.** `_preflight` (`cloud/vastrun.py:648-669`), called from `run()` (`:1343-1345`).

MEASURED (P3d): two `_preflight` calls on separate runners, both before either rents: `both preflights returned; neither refused`.
Why it matters: two concurrent runs can both rent. Each cleans up its own label, and no cross-destroy was observed, but the account can hold two boxes and spend up to twice the cap.
Smallest fix: an exclusive lock file around preflight plus the first rent, or document that concurrent runs are unsupported.

### F6 — LOW (latent; no caller uses threads today) — INTRODUCED
**The wall-clock and spend timers exist only on the main thread.** `_install_deadline_timer` returns silently off the main thread (`cloud/vastrun.py:505-507`). The design (`VASTRUN-SMALL-DESIGN.md:38-39`) says the timers "interrupt blocked calls".

MEASURED (`probes/p7_timer_thread.py`, 1.2 s wall limit, 3 s blocked poll):
```
[main thread]   {'outcome': 'WallClockExceeded: wall-clock limit reached', 'elapsed_s': 1.2}
[worker thread] {'outcome': 'WallClockExceeded: wall-clock limit reached', 'elapsed_s': 3.0}
```
Grep found no caller of `run_jobfile` in the worktree besides `main`.
Smallest fix: raise in `run_jobfile` when not on the main thread, or document the restriction.

### F7 — LOW — INTRODUCED
**Cleanup budgets restart per phase, so a hung DELETE costs about 60 s, not 30 s.** The per-job `finally` (`cloud/vastrun.py:1361-1363`) runs before `_begin_cleanup` sets `cleanup_deadline` (`:1369`, `:562-564`). Until then, `_destroy_tracked` and `_label_sweep` use `getattr(self, "cleanup_deadline", now + 30)` (`:966-967`, `:1060-1061`). The final cleanup gets its own 30 s.

MEASURED (P5d, wall 0.02 min, DELETE sleeps 60 s): `elapsed_s: 61.2`, `still_listed: ['104']`, error `DestroyConfirmationError ... instance remained listed`.
Why it matters: bounded, and it reports honestly, but twice the stated budget. The design's "cannot block cleanup" claim holds.
Smallest fix: set one shared cleanup deadline at the start of the first per-job cleanup, and reuse it in the final phase.

### F8 — LOW — PRE-EXISTING
**Job `env` values reach the local `ssh` argv and the remote `sh -c` argv.** `_launch_job` writes `export KEY=value` into the body (`cloud/vastrun.py:1235`) and passes it as the ssh command argument (`:1244`). The pre-change `_launch_job` (`vastrun_old.py:862-878`) has the same pattern.

MEASURED (`probes/p4_secret_exposure.py`, fake ssh records `/proc/$$/cmdline`): the ssh argv contains `export HF_TOKEN=SENTINEL-JOBSECRET-9b21c4`. `JOB_SECRET found in: ['job.json', 'logs/ssh-calls.log']`, where `job.json` is the input and the log is the argv capture. The Vast key was found nowhere outside its key file.
Why it matters: any local user who can read `/proc` sees the job secret while ssh runs, and the remote user sees it for the whole job.
Smallest fix (follow-up): deliver env through stdin or a 0600 file on the box, not argv.

### F9 — LOW — PRE-EXISTING
**A DELETE that answers `success:false` for an already-gone box makes the run fail.** Confirmation requires `delete_accepted` (`cloud/vastrun.py:997-1001`). The same message exists in the pre-change code (`vastrun_old.py:674`).

MEASURED (P6e): `raised DestroyConfirmationError: ... destroy response was unreadable although the instance is absent`, `still listed: []`.
Why it matters: a false failure with the box gone. No leak.
Smallest fix: confirm by listing absence whenever the listing is readable, regardless of the DELETE shape. Keep the 404 handling.

## What I checked and found sound
- **Vast key:** never in argv, env, logs, run-record, results, or the rent `onstart` body. It goes only in the `Authorization` header (`_request`, `cloud/vastrun.py:324-345`). P4 confirmed this.
- **Label-less and id-less neighbours:** skipped, not errors. The brief's round-2 concern (a label-less row making the whole listing unreadable) is not in the code: `_instance_payloads` (`:149-166`) checks only `id`. P3b: sweep destroys our expired box and leaves the label-less and null-label rows alone.
- **Rent recovery:** a rent with no id (P5a) and a rent that raises after creating (P5b) both end with the box destroyed and its id in `recovered_instance_ids`.
- **Interrupt between rent response and id recording:** P5c, the final label sweep destroys the box.
- **Run-record write failure:** P6f, rent and cleanup still happen (`_write_record` swallows exceptions).
- **Destroy confirmation** requires id absence (`:997-998`), so it reads the right endpoint shape. Pagination (`_listed_instances`, `:232-266`) follows `next_token`, rejects repeated tokens, and caps at 100 pages.
- **Subprocess calls:** argv lists, no `shell=True` (`OpenSSHRunner`, `:405-445`). Quoting uses `shlex.quote` throughout `_launch_job`, `_upload_files`, and `_copy_back`. Job commands run through a remote shell by design.
- **Results:** namespaced per job; stems unique and reserved log paths enforced (`_validate`, `:678-714`). Duplicate local targets are rejected.
- **Spend clock:** `rent_started` is taken before the rent call (`:799`). Rate is the max of offer and listed (`_apply_instance_rate`, `:907-927`). Read, not measured.
- **Signals:** SIGINT, SIGTERM and SIGHUP become `RunInterrupted` (`main`, `:1473-1482`). Cleanup catches it and continues. A hung DELETE in cleanup ends in bounded time with the box reported as still listed (P5d).

## Self-assessment
- **Test suite not run.** `pytest` is not installed here, and `pip install` failed with a TLS certificate error. I did not disable certificate checks. So the claimed 47/47 and the sabotage results are unverified.
- The fake Vast is my model. F1 assumes the listing can omit our label. The design says the live probe showed exact labels, so F1 is conditional and may not occur.
- Not measured: the alarm firing in the first statements of `_begin_cleanup` (before `cleaning_up` is set), the alarm firing inside the `finally` of `_cleanup_call`, and KeyboardInterrupt delivered during `_label_sweep` (the CLI converts signals, so this matters only for direct `run_jobfile` callers).
- Residual risk I could not reproduce: `_label_sweep` returns on the first empty listing with no settle delay. A rent whose response timed out, and whose box appears in the listing only after the sweep, would be reported as "already absent". This needs a model of Vast's listing lag.
- Watchdog `poweroff` billing (the design defers it), spend-cap accuracy before the listing shows `dph_total`, and the box left by a crash until a later run after wall plus 35 min: all taken from the design doc or the code, not measured live.
- I did not read any credential file. The P4 key was a sentinel file inside my domain. No live service was contacted.
