# REVIEW: cloud/vastrun.py (commit a80e721, base 7b42fb5)

**Verdict: DO-NOT-SHIP for unattended use as it stands.** The core claim, "never leaves an instance running", is false on measured paths: a destroy can be reported as confirmed while the instance is still alive, and a SIGHUP or an interrupt in the rent-to-record gap leaves a rented instance with no destroy. All of this is INTRODUCED (new file). The fixes are small, and after findings 1-3 are fixed this is a SHIP-WITH-FIXES candidate. The suite is 7/7 green on the real file, but two of its tests do not exercise what they claim.

Method: each finding was driven against the real `cloud/vastrun.py` in the worktree with in-process fakes (`measure.py`, `blip.py`, `signal_driver.py` in this directory). No network, no credential file read (the key test uses a scratch-made fake key), no live service. Tests were run with `shim/run_tests.py`, a stand-in for pytest (pytest is not installed and `pip` hit an SSL error), which imports the real test module unchanged.

## Findings

### 1. HIGH, INTRODUCED: destroy is "confirmed" when the list response is not understood
`cloud/vastrun.py:532` computes `listed = instance_id in _instance_ids(...)`. `_instance_payloads` (132-142) returns `None` for a non-dict or unrecognised body, and `_instance_ids` (145-147) turns `None` into an empty set, so "unreadable" reads as "gone". `_recover_uncertain_rent` (493-495) handles the same `None` correctly as unreadable; destroy does not.

Measured (`measure.py destroy`; destroy is a no-op, the instance stays alive):
```
- control: v1 dict {'instances': [...]} still lists it
    instance still alive: ['instance-offer-1']  destroy calls: 3  record destroy_confirmed=False
    run: raised DestroyConfirmationError: ... after 3 attempts: instance remained listed
- bare JSON list [...] still lists it
    instance still alive: ['instance-offer-1']  destroy calls: 1  record destroy_confirmed=True
    run: returned normally
- 200 body {'success': False, 'msg': 'rate limited'}
    instance still alive: ['instance-offer-1']  destroy calls: 1  record destroy_confirmed=True
    run: returned normally
```
Why it matters: the run exits 0 and says the box is gone while it is still billing. The brief flags exactly this ("destroy confirmation reading the wrong endpoint shape"). The real v1 response shape is not checked anywhere in this repo (see Unverified). Fix: treat `None` from `_instance_payloads` as "not confirmed" (raise, as the recovery path does), and only accept a list that is parsed as a list.

### 2. HIGH, INTRODUCED: an instance can stay rented after rent succeeds
Three measured or demonstrated paths:

a) **SIGHUP is not handled.** `main()` (`cloud/vastrun.py:760-761`) installs handlers for SIGINT and SIGTERM only. Measured with a real `vastrun.main` process, rent succeeded, job never finishing:
```
- control SIGTERM: exit=1 rented=['instance-offer-1'] destroyed=['instance-offer-1'] run-record written=True
- SIGHUP (terminal/ssh session gone): exit=-1 rented=['instance-offer-1'] destroyed=[] run-record written=False
- SIGKILL (crash/OOM): exit=-9 rented=['instance-offer-1'] destroyed=[] run-record written=False
```
An unattended run started from an SSH or tmux session that drops will do this. SIGKILL is inherent, but it shows nothing is persisted before the end: `self.record` is written only in `run()`'s `finally` (711-718), so a killed run leaves no instance id anywhere except the watchdog.

b) **Rent-to-record gap.** `_rent_one` (`:453-467`) puts the `if instance_id is None` check and `self._activate(...)` outside the `try`. A signal that lands after `api.rent` returns and before `_activate` leaves the id unrecorded. Injected at the start of `_activate` (it models the signal; this is not a live signal):
```
- injected at the gap: rent returned an id, then RunInterrupted before _activate
    alive: ['instance-offer-1']  destroy calls: []  run: raised RunInterrupted
    record instance_id=None hosts_tried=[None]
```
The control, a RunInterrupted raised inside `api.rent` after creation, does recover (`alive: []`, destroyed). So only the gap leaks.

c) **No label sweep at the end.** `run()`'s `finally` destroys only `self.active`. The unique `run_label` (`:339`) is already on every rent, so a final `list_instances` pass that destroys anything carrying this run's label would close (a) and (b) and any future window.

Fix: install a SIGHUP handler alongside SIGINT/SIGTERM; move `_activate` inside the rent `try` (or record the id before any other work); and in `run()`'s `finally`, destroy every listed instance whose label is this run's `run_label`.

### 3. MEDIUM, INTRODUCED: spend estimate freezes after an unconfirmed destroy
`_destroy_active` sets `self.active_started = None` (`:545`) before it knows whether the destroy worked. When confirmation fails, `_estimate_spend` (373-377) stops counting while the instance is still listed and billing.

Measured (`measure.py spend`; each destroy call advances the clock 30 min; dph 1.0/h; destroy never works):
```
- run: raised DestroyConfirmationError
  after failure: estimate $1.5000  record estimated_spend_usd=1.5
  one hour later (instance still listed): estimate $1.5000  (a billed instance would be about $1.0 more)
```
Why it matters: the operator's cost figure is wrong exactly when the instance is still running. Fix: keep the clock running for an unconfirmed instance (only roll it into `total_spend` when confirmed).

### 4. MEDIUM, INTRODUCED: one ssh timeout during polling aborts the run and loses the job's results
`_run_jobs` (`:679-680`) calls `_ssh_run(..., check=False)`, and a `subprocess.TimeoutExpired` from `OpenSSHRunner.run` (`:278`) propagates out of the loop with no retry.

Measured (`blip.py`; the job succeeds on the box with rc=0; one poll times out):
```
  run: raised TimeoutExpired: Command '['ssh', 'host-offer-1']' timed out after 15 seconds
  destroyed: ['instance-offer-1']  alive: []
  results copied: []
```
The instance is destroyed (no leak), but the job's output, which the brief says is copied "as soon as each ends", is never copied. Fix: count consecutive poll failures and tolerate a small number before raising.

### 5. MEDIUM, INTRODUCED: results from different jobs overwrite each other silently
`_copy_back` (`:640-656`) writes each job's files to `results_dir/<relative path>` with no per-job namespace. Two jobs with the same expected path overwrite each other locally. Measured (`measure.py overwrite`):
```
- record copied_back: train=['out/result.txt', 'logs/train.log'] eval=['out/result.txt', 'logs/eval.log']
- local out/result.txt now contains: 'copy 3 of /workspace/vastrun/out/result.txt'
```
A second collision: job names are checked for uniqueness on the raw name (`:367-369`), but the log path uses `_safe_job_stem` (`:608-610`). `"a/b"` and `"a_b"` are both accepted, and both logs land at `logs/a_b.log` (measured: local `a_b.log` holds only the second job's log). Fix: reject duplicate expected paths and duplicate stems in `_validate`, or namespace local output by job stem.

### 6. MEDIUM, INTRODUCED (conditional): rent reply without an id plus an unreadable list leaves the instance
If the rent reply has no id, `_rent_one` calls `_recover_uncertain_rent`. If the list reply is unreadable, that raises (`:505`), and `active` is still `None`, so the `finally` destroys nothing. Measured (`measure.py rentshape`):
```
- v1 dict list: alive [] destroy calls ['instance-offer-1'] run: raised NoReadyHost
- bare-list list: alive ['instance-offer-1'] destroy calls [] run: raised RunnerError: rent outcome is uncertain ...
```
The dict case recovers; the bare-list case leaves the box rented. This needs a rent reply without `new_contract` and an unrecognised list shape, so it is conditional, but it shares the root cause of finding 1. Fix: same as finding 1, and treat the uncertain state as needing a label sweep (finding 2c).

### 7. MEDIUM, INTRODUCED (unverified shape): SSH port lookup only accepts a key of `"22"`
`_remote_connection` (`:159-174`) uses `ports.get("22")` (`:164`). Measured:
```
ports keyed 22/tcp, no ssh_port: None
ports keyed 22, no ssh_port:     ('1.2.3.4', 40022)
ssh_host/ssh_port (test shape):  ('h', 2222)
```
If the live instance reports only `ports["22/tcp"]`, no host ever becomes ready and every offer is dropped after 180 s (three rents, three destroys per run). The tests only use the `ssh_host`/`ssh_port` shape (`test_vastrun.py:66-75`), so the other shapes are never exercised. Not checked against the live API (none allowed); see Unverified.

### 8. LOW, INTRODUCED (test quality): the key test passes for the wrong reason
`test_vast_key_is_not_read_or_written_to_stdout_or_logs` (`test_vastrun.py:256-277`) runs only against `FakeAPI` (`:39-80`), which never builds a `VastAPI`, never reads a key file and never sends a request. Sabotage on a scratch copy (`VASTRUN_ROOT=sab/A-key-printed`) that prints the key inside `VastAPI.__init__` leaves all 7 tests green. The key handling itself was checked directly against the real `VastAPI` with a fake key file and a patched `urlopen` (`measure.py key`): HTTP 403 and URLError messages do not contain the key, and the key is not in `os.environ`. So the behaviour is sound; the test does not show it.

### 9. LOW, INTRODUCED: a second signal during destroy backoff cuts the retries to one
In `_destroy_active`, `self.sleep(self.destroy_delay)` (`:540`) sits outside any `try`, so a `RunInterrupted` raised there escapes the loop. Measured (`measure.py destroyretry`):
```
- destroy calls: 1 (DESTROY_RETRIES is 3)  alive: ['instance-offer-1']
```
Fix: catch the interrupt in the backoff, or keep retrying until the last attempt.

### 10. LOW (test that does not run the path): the destroy-in-finally sabotage is real
Removing the `finally` destroy (sabotage B) turns 6 of 7 tests red, matching the report's "reds 6". Confirmed.

## Unverified (not checkable offline; no live calls allowed)
- Real Vast response shapes for `/bundles/`, `/asks/`, `/instances/` (v0 and v1), including the `new_contract`, `ssh_host`/`ssh_port` and `ports` keys. The code says it matches "the reference helper"; I could not find that helper in the Journal scripts or docs.
- Whether `poweroff` exists in the Vast container and stops compute billing. The watchdog (`_watchdog_command`, `:177-180`) depends on it. The report's storage caveat is honest about the limit.
- Whether `dph_total` on the offer equals the billed rate of the instance. The spend estimate uses the offer's value (`_rent_one`, `:440`).

## Concurrency and other notes (no measured defect)
- Two concurrent runs share no lock. Each has its own cap, so two runs can spend twice what either cap allows. Design note, not a defect in this change.
- Feature flags: none in this change, so the "flag OFF" check does not apply.
- Shell injection: env names are validated (`:617`), env values, command and log paths are `shlex.quote`d. Measured with a real local `sh`: an env value of `x'$(touch PWN_ENV)"; touch PWN_ENV2; #` arrives exactly as written, status rc=0, no injected files; env name `A;B` is rejected. Job names are sanitized before they reach a path. `scp` gets argv lists (no local shell), and `..` is rejected in destinations. Remote-side interpolation in `scp` depends on the local OpenSSH mode; reachable only from the operator's own job file, so LOW.
- Key: read only in `VastAPI.__init__`, sent only in the `Authorization` header, not in argv, environment, record or stderr (`measure.py key`). `RemoteCommandError` drops remote output on purpose. The watchdog contains no key.
- Spend clock: `active_started` is taken before the rent call (`_rent_one:446`, `_activate` argument), so it is not started late.

## What I checked and found sound
- Offer filter, host cap (`MAX_HOSTS` = 3, `:562`), 180 s readiness deadline, drop-and-destroy of a non-ready host.
- Spend and wall-clock checks run before rent, before each job, and in the poll; a cap hit during a job destroys the instance (test `test_spend_cap...` and my runs).
- Copy-back order: a job's files are copied before the next job launches (`test_each_job_copies...`).
- The destroy call is retried three times and a non-zero exit is returned when confirmation fails (`test_destroy_confirmation_failure...`; measured in finding 1's control).
- Command quoting, env validation, `..` rejection in remote destinations, and the `DELETE` / `GET v1 /instances/` endpoints match the brief's description.
- The run record never holds the key; it is written atomically (temp file then `replace`).
