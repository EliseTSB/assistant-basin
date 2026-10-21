# Cold review: vastrun round 2 (HEAD bc080dc vs base 37cd2b8)

## Verdict: DO-NOT-SHIP as written

Two HIGH findings are measured against the real `cloud/vastrun.py`, and both are INTRODUCED by this change.
(1) A listed instance without a `label` key makes the whole listing unreadable, so a rent that timed out after
creating the box is never destroyed. (2) The spend estimate takes the listed `dph_total` even when it is 0, which
turns the spend cap off. Both fixes are small (see each finding). Fix both, then re-measure before shipping.

Scope note: the offline pytest suite could NOT be run here (pytest is not installed and pip has no network), so
every claim below comes from driving the real code with a fake API and fake ssh, plus reading. No live Vast call,
no credential file read, nothing written outside my domain. The probe script is `measure.py` and its full output is
`run-all.txt`, both in `/home/elisepc/green/Journal/scratch/review-vastrun-r2-haiku-haiku-r2-2026-10-21/`.

## Findings

### 1. [HIGH] INTRODUCED: a listed instance with no `label` key strands a rented box
- Code: `cloud/vastrun.py:139-141` (`_instance_payloads` returns None if any item lacks the `"label"` key), used by
  `_recover_uncertain_rent` 604-635, `_destroy_tracked` 642-718 and `_label_sweep` 720-804.
- Measured (`python3 -B measure.py rent-timeout-label-missing`): the rent raised a timeout after creating instance 1000,
  and the account had another instance `{"id": "77", "dph_total": 0.05}` with no label key.
  Output: `RAISED DestroyConfirmationError: could not confirm label sweep ... unreadable instance list`,
  `still listed by fake API (ours): ['1000']`, `destroy calls: []`. The box stays rented; only the watchdog remains.
- Control (`rent-timeout-label-null`, same account but `"label": null` on the other box): recovered and destroyed,
  `destroy calls: ['1000']`. So the defect is the key-presence check, not the recovery logic.
- Also (`label-missing`, no rent fault): DELETE removed the box, yet `destroy_confirmed False`, three DELETE calls,
  and the run exits 1. Every run on such an account reports failure even when clean.
- Whether the live v1 listing omits `label` is UNVERIFIED. I made no live call.
- Why it matters: "never leaves an instance running" is false on this path, and it is paid, unattended time.
- The suite does not catch this. `test_vastrun.py:541` asserts `_instance_ids({"instances": [{"id": "missing-label"}]}) is None`,
  so the test passes by locking in the strictness.
- Smallest fix: in `_instance_payloads` (vastrun.py:139-141) require a dict with a non-None `id` only; keep
  `value.get("label")` (already null-safe at 615 and 740). Flip test line 541 to expect parsing to succeed. Add a full-run
  test with a label-less account instance.

### 2. [HIGH] INTRODUCED: spend cap is disabled when the listed rate is 0
- Code: `_apply_instance_rate` `cloud/vastrun.py:585-602`; `rate = _number(instance.get("dph_total"), ...)` at 591;
  `_number` (78-83) accepts 0. The base used the offer rate only (base grep: `current_dph = entry["dph"]`).
- Measured (`measure.py dph-zero`): offer $0.5/h, `spend_cap_usd 0.4`, job ran 3600 simulated seconds.
  Output: `outcome: OK`, `record.estimated_spend_usd: 0.0`, job finished, no SpendCapExceeded.
- Control (`listed_dph=0.5`): `RAISED SpendCapExceeded: spend cap reached`, `estimated_spend_usd: 0.4`.
- Why it matters: the spend cap is the only money guard besides the wall clock. A 0 (loading, reporting gap) silently
  zeroes it. The fold's own claim, "reads the listed instance dph_total when available", does not say it may lower the rate.
- Smallest fix: never let the listing lower the rate, e.g. `rate = max(_number(instance.get("dph_total")), tracked_rate)`,
  or ignore non-positive listed values.

### 3. [MEDIUM] INTRODUCED: destroy cannot confirm on a retry after a successful DELETE; record and exit disagree
- Code: `_destroy_succeeded` (152-153) and `delete_ok` (653-657) are per attempt; confirm needs `delete_ok` on the same
  attempt as the absence read (670-671).
- Measured (`measure.py destroy-flake`): first DELETE succeeded, the post-DELETE list read failed once, the retry DELETE
  got HTTP 404 on a box already gone. Output: `RAISED DestroyConfirmationError: ... destroy response was unreadable
  although the instance is absent`, `destroy calls: ['1000','1000','1000']`, instance actually gone, and
  `record.destroy_confirmed: True` (the label sweep then popped `destroy_error`).
- Why: a clean teardown exits 1, and the run-record says confirmed. A transient list error after a good DELETE
  produces a false alarm.
- Smallest fix: keep `delete_ok` sticky across attempts (a DELETE that returned success, or a 404 for a gone id, counts),
  or confirm on absence alone, since absence is the evidence that matters.

### 4. [MEDIUM] PRE-EXISTING: an expected result path can silently overwrite another job's log
- Code: duplicate check covers only `expected_paths` (`cloud/vastrun.py:382, 396-413`); log copies are `logs/<stem>.log`
  (916, 950-951) and are never in that set. `run-record.json` is not checked either.
- Measured (`measure.py log-collision`): job `a` writes its log; job `b` has `expected_result_paths: ["logs/a.log"]`
  and writes over it. Output: `local results/logs/a.log = B-OUTPUT-OVERWRITES-A-LOG`, outcome `OK`, validation passed.
  (In `run-all.txt` that print lands under the `interrupt-control` header; it belongs to `log-collision`.)
- The fold's claim "Validation rejects duplicate result destinations" is narrower than its wording.
- Smallest fix: add each job's `logs/<stem>.log` and `run-record.json` to `expected_paths` before the duplicate check.

### 5. [LOW] PRE-EXISTING: job env values reach run-record.json when a launch SSH call fails
- Code: `RemoteCommandError` embeds `command[:160]` (`vastrun.py:70`); `_launch_job` puts `export KEY=value` into that
  command (869-870, 878).
- Measured (`measure.py secret-launch`): `record contains secret: True`; `record.error` holds
  `export HF_TOKEN=hf_SECRET_ENV_VALUE_123`. This is a job secret, not the Vast key. The Vast key path was clean.
- Smallest fix: name the step in `RemoteCommandError` instead of the command text, or pass job env through a file.

### 6. [UNVERIFIED] PRE-EXISTING: the watchdog's `poweroff` is claimed, not shown
- Code: `_watchdog_command` (191-194) is unchanged from base. It runs `sleep N; poweroff` in the container via onstart.
  No test checks its effect. I have no evidence that `poweroff` inside a Vast container stops the VM. If it does not, the
  only backstops are the in-process destroy and the label sweep, and a SIGKILL leaves the box up (no reader of
  `run-record.json` exists on restart).
- Follow-up, not a reason to hold this change.

## What I checked and found sound
- **Key handling.** `_request` (211-232) sets only the Authorization header. HTTP errors are reduced to a code and
  method path (`from None`). `OpenSSHRunner` passes no key, and the child environment is `os.environ`, which does not
  hold the key. The watchdog string has no key. No log or run-record path writes it.
- **Label sweep identity.** Matches are exact equality on `run_label`, a fresh uuid per run (`vastrun.py:356`). One
  run cannot destroy another run's instance. The sweep is only as blind as the listing, see finding 1.
- **Interrupts during destroy.** `_destroy_tracked` catches `RunInterrupted` inside the DELETE, the list read and the
  backoff sleep, so retries continue after a second signal. Confirmed by a test and by reading.
- **Rent-timeout recovery with a readable list.** `measure.py interrupt-after-rent` control: `RunInterrupted` after a
  create was recovered and destroyed (`destroy calls ['1000']`, `destroy_confirmed True`). Same for the label-null control.
- **Spend cap with a true rate.** Control above trips at $0.4.
- **Injection.** Env names are regex-checked (867), values `shlex.quote`d (869), log stems sanitised (859). Job commands run
  by design, as the job author's own command. ssh target is `root@host`, so a host string cannot become an option.
- **Retries.** No second rent in a retry. Each offer is rented at most once (`_ensure_ready_host` 811-816).
- **Signals.** SIGHUP is handled with SIGINT and SIGTERM (`main`, 1031). Handlers are restored after the run.

## Not measured (honest gaps)
- SIGKILL or power loss between rent and destroy: reasoned, not run.
- Two runs sharing one `results_dir`: `run-record.json.tmp` (961) is a fixed name. Not run.
- A second signal inside the label sweep's matches loop (outside any try, 740-798). Reasoned only.
- The 23-test suite and the "all three HIGH sabotages red" claim: not run (no pytest).
- Live Vast response shapes (`label`, `success`, `dph_total`): unverified.
