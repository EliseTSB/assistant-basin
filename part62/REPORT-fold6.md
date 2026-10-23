# Part 62 fold 6 report

## SUMMARY

- Implemented a durable Part 62 `run-all` lifecycle with a lock, a run plan, a separate durable record for every chunk, and resume by saved batch ID.
- Added calibration-first enforcement: only a passing 120-answer Haiku calibration with kappa >= 0.8 and no unlabelled or truncated responses permits answer submissions.
- Added explicit ledger retirement, retained reserves, raw POST response persistence, and fake API kill-boundary tests.
- Added pinned Vast model prefetch, offline mode, success and incomplete archives, and recovery commands in the runbook.
- The required unittest command completed with 42 tests `OK`; its intentional mutation-copy failures were observed.
- No paid API, network, or GPU operation was performed.

## FINDINGS / WHAT SHIPPED

- `part62/blind62.py:449-455` enforces the Haiku calibration record requirements. `part62/blind62.py:642-669` collects and records calibration before entering the answer loop; missing, pending, failed, truncated, or low-kappa calibration prevents answer POSTs. Request construction retains `max_tokens=1024` and does not include `temperature`.
- `part62/blind62.py:468-473, 549-567, 590-641` writes the run plan and one submission record per chunk, reserves before POST, writes `posting` before the call, fsyncs the raw response before interpreting its ID, and resumes a recorded batch by polling and collecting it. The ledger's idempotent settlement protects a rerun after collection but before local state update.
- `part62/blind62.py:627-638, 790-810` refuses unknown POST outcomes without resubmission and prints the exact operator retirement command. `retire` records the shared-ledger retirement; the next run records `retired-unknown`, and the ledger keeps the reserve counted. `part62/RUNBOOK-62.md:99-125` documents resume, inspection, and retirement commands.
- `part62/test_part62.py:793-899` covers missing/pending/failed calibration, the no-temperature request shape, and fake API kills after reserve, after POST before response persistence, after batch ID recording, and after result persistence before settlement. Rerun checks assert no duplicate POST and one settled ledger attempt where a batch ID is known.
- `part62/run62.sh:27-40, 59-74` fetches the two revisions pinned in `run62.py` into `/root/hf/hub`, turns on offline mode, creates complete or incomplete return archives, and prints `RETURN_READY` or `RUN_FAILED`. `part62/RUNBOOK-62.md:35-72` gives the Vast invocation and archive behavior.
- Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part62.test_part62` ended `Ran 42 tests ... OK`; `bash -n part62/run62.sh` and `python3 -m py_compile part62/blind62.py part62/test_part62.py` completed successfully.

## BLOCKERS

- Decision needed on an internal policy conflict. Amendment 2 permits an explicit Opus fallback after a failed Haiku gate, but this fold explicitly requires the Haiku calibration to pass before *any* answer batch. I implemented the stricter ordering and block Opus requests; allowing fallback would violate the stated gate. Please answer the question in `HOLD.request` before treating the fold as complete.
- Provider-side POST acceptance cannot be proven if the process dies before the response is durably written. The implementation does not retry that chunk; it leaves the reserve counted and requires operator reconciliation or retirement.
- No live API behavior or rented 3090 run was exercised, per the no-paid-call/no-GPU scope.
- Parent-control mailbox was empty at all checkpoints; no messages changed or were declined from the plan.

## SELF-ASSESSMENT

The fake API tests verify local state transitions and ledger idempotence, not Anthropic provider behavior. The fallback conflict is the one decision I did not infer. The Vast prefetch shell uses the exact revisions already registered in `run62.py`; it was syntax-checked but not run against a rented host.

## Fold 6b

### SUMMARY

- Applied the 2026-10-23 ruling: Haiku answers require a recorded passing calibration; explicitly selected Opus answers are permitted only after a recorded failed Haiku calibration.
- Opus uses its own durable run-plan file and sends only answer chunks. It does not calibrate itself, and default `run-all` remains Haiku.
- The previous Fold 6 policy blocker is resolved by this ruling. No paid API, network, or GPU operation was performed.

### FINDINGS

- `part62/blind62.py:438-465` recognizes only a recorded failed Haiku gate, including failures with incomplete calibration labels. `part62/blind62.py:594-627, 666-696` refuses Opus otherwise, prints the Haiku resume command, keeps Haiku gated on a pass, and allocates Opus its own durable run plan. `part62/test_part62.py:523-549` covers missing, pending, passed, and failed records, including a failed sample with 119 labels.
- `part62/blind62.py:611-613, 697-730` excludes calibration requests from the Opus run and applies answer-label/completion checks without requiring Opus calibration. `part62/test_part62.py:878-910` verifies an explicitly selected Opus run starts only the answer stage after a saved failure. `part62/test_part62.py:873-876` verifies default `run-all` selects Haiku and does not start Opus.
- Haiku's missing, pending, and failed calibration states refuse answer chunks: `part62/blind62.py:666-696`; coverage is in `part62/test_part62.py:827-871`. Existing ledger policy remains shared at the $8 cap with Haiku's $4 sub-cap (`part62/blind62.py:380-412`); `part62/test_part62.py:494-500` covers the sub-cap, and sequential 200-request chunk planning is covered at `part62/test_part62.py:253-264`.
- `part62/RUNBOOK-62.md:93-130` now documents the explicit Opus command policy, distinct run-plan files, sequential reservations, and the refusal/resume behavior.
- Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part62.test_part62` completed with 45 tests `OK`. Its five deliberate mutation-copy probes printed the expected red failures. `git diff --check` passed.
- Parent-control mailbox reads were empty at each checkpoint; no messages changed the plan or required disposition.

### BLOCKERS

- None. The registered Opus fallback is implemented and covered by tests. Live provider behavior remains unverified because this fold forbids paid calls.
