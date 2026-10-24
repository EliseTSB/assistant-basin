# Collect-loop transient GET failures (2026-10-24, written before the fix)

Four Part 64 passes ended on a read-only GET failure in the collect loop (resume.log: two
`TimeoutError: The read operation timed out`, two `Connection reset by peer`); each restart
re-scanned about 40 MB of chunk records. The batches themselves were fine.

Fix: `api_call` raises `TransportFailure` (a `WriterError` subclass) for any transport error,
including errors raised while reading the body, and the collect loop treats a `TransportFailure`
or an HTTP 429/5xx from a submitted chunk's status or results GET like `_BatchNotEnded`: the chunk
stays `submitted`, its reserve stays active, and the next pass retries it. A chunk that fails this
way on `COLLECT_TRANSIENT_RETRY_PASSES` (30) consecutive passes, about 30 minutes at the 60 s pass
interval, re-raises the error so a person or the relaunch loop sees it. Nothing changes for POST
(any POST exception is already an unknown outcome) or for other `WriterError`s.

## Predicted rows (old code = 54e7c8b)

| Row | Case | Old code | New code |
|---|---|---|---|
| T1 | status GET raises TransportFailure once, then succeeds | raises (red) | completes, each chunk settled once, no extra POST |
| T2 | results GET raises TransportFailure once | raises (red) | completes, settled once |
| T3 | status GET returns HTTP 503 once | raises (red) | completes |
| T4 | status GET returns HTTP 404 | raises | raises HTTPFailure 404 (not retried) |
| T5 | status GET fails transport on every call | raises on pass 1 | raises TransportFailure after exactly 30 failing passes; chunk still `submitted`, reserve still reserved |
| T6 | `api_call` with urlopen raising ConnectionResetError during read | raw ConnectionResetError (red) | TransportFailure |
| T7 | invalid status JSON (existing test) | raises WriterError | unchanged |

Mutation check: deleting the new except clause must turn T1-T3 red.
