# Shared paid-action library build report

## SUMMARY

Built a standard-library-only ledger and cost helper for Parts 62 and 63 to share. The ledger writes and fsyncs a random 128-bit reservation before submission, serializes all accounting reads and writes with `flock`, retains unknown outcomes, and supports idempotent settlement and dated operator retirement. The helper reserves and settles one chunk at a time and durably records each batch ID through the callback hook.

## API

- `Ledger(path, cap)` requires an absolute file path outside `/tmp` and a finite positive numeric cap no higher than `HARD_CAP_USD` ($8). `DEFAULT_LEDGER_PATH` is the canonical `/home/elisepc/assistant-basin/paidaction/paid-ledger.jsonl`, independent of the importing checkout. When reserve metadata supplies `run_id`, a durable shared registry refuses a second resolved ledger path for that run.
- `reserve(worst_case_usd, kind, meta, *, subcap_usd, subcap_key)` returns a fresh 32-character lowercase hex attempt token after its reserve event is fsynced. Aggregate and optional per-key sub-cap checks share one lock and accounting read; the reserve line is the same fsynced commit for both.
- `settle(attempt_id, actual_usd, usage)` is idempotent when both dollars and the canonical usage digest match. It stores dollars and the usage digest, not usage totals; callers retain raw responses and derive usage by `custom_id`. If settlement would exceed the aggregate cap, it appends a `settlement_overrun` incident and raises `CapExceededError`, leaving the original reserve active and counted.
- `release(attempt_id, http_status, body)` accepts only a non-429 4xx with a parsed `{"type":"error","error":{...}}` object. Invalid bodies, timeouts, connection errors, 5xx, 429, and kills leave the reserve active.
- `retire(attempt_id, who, reason)` appends a timestamped operator action and removes only that active reserve from counted spend. `state()` reports `settled`, `reserved`, and per-kind dollar totals, plus whether a torn final line was ignored.
- `pricing.worst_case_cost(model, input_tokens_upper, max_tokens, batch=True)` uses inherited model-specific batch/direct rates. `submit_in_chunks` estimates input as ceil(serialized request bytes / 3), computes the maximum output from each request's `max_tokens`, and lets the ledger refuse a later chunk after actual settlement.
- `listing_report(api_list)` copies an API listing into a report. It has no ledger argument and performs no adoption or filesystem writes.

## Parts 62 and 63 call pattern

Both parts should import the same `DEFAULT_LEDGER_PATH` and use the same aggregate cap for that shared ledger. Enforce any smaller reader or stage budget in the caller before reserving. The callback receives a durable ID writer and must invoke it immediately after extracting the ID from the POST response, before collecting or parsing result rows. Store raw results under stable custom IDs; derive actual usage and dollars from those raw results.

```python
from pathlib import Path
from paid_ledger import DEFAULT_LEDGER_PATH, Ledger, submit_in_chunks

CAP = 8.0  # shared aggregate ceiling; enforce smaller stage budgets separately
ledger = Ledger(DEFAULT_LEDGER_PATH, CAP)
requests = build_requests_for_this_attempt()
batch_ids = Path(__file__).resolve().parent / "batch-ids.jsonl"

def submit(chunk, attempt_id, record_batch_id):
    created = api.create_batch(chunk)
    record_batch_id(created.id)  # fsynced before result collection and parsing
    raw_rows = api.collect(created.id)
    save_raw_results_by_custom_id(raw_rows)
    usage = derive_usage(raw_rows)
    return {"actual_usd": price_actual(usage), "usage": usage}

completed = submit_in_chunks(
    requests, 200, ledger=ledger, model="claude-haiku-5-5",
    submit=submit, batch_id_path=batch_ids, kind="part63",
    meta={"stage": "reader"})
print(completed, ledger.state())
```

## TESTS

Ran `/home/elisepc/neuron-probe/.venv/bin/python -m unittest -v` from this folder: **18 tests passed**. The suite covers non-finite and over-ceiling caps, two-process aggregate and sub-cap races, settlement overrun recording, canonical default path and run-ID path refusal, torn-tail recovery, release rules, timeout retention, retirement, chunking, and read-only listing. The SIGKILL matrix kills a subprocess after reserve, simulated POST, batch ID fsync, and settlement. Five copied-production-code mutations covering the original guards and all three new fixes fail their targeted assertions.

## ASSUMPTIONS UNVERIFIED

- Inherited rates are unverified: Haiku 5.5 batch `$0.05/$0.25` and direct `$0.10/$0.50` per million input/output tokens; Opus 5.5 batch `$2/$10`. No inherited direct Opus rate was present, so that mode is refused.
- UTF-8 bytes divided by three is an unverified input-token estimate, not a proven upper bound. A mistaken rate or estimate can cause actual charges to exceed the reservation; settlement records the actual dollars and blocks later reserves once the shared cap is exhausted.
- No live API request was made. Provider response-body shapes, whether a parsed Anthropic 4xx is unbilled, and billing after timeouts, 429, or 5xx remain unverified. Unknown outcomes stay counted until an operator retires them.
- The batch adapter must call `record_batch_id` immediately after it extracts the ID. The library can enforce that an ID is durably recorded before settlement, but it cannot observe when an external transport has received its response body.
