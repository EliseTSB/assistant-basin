# Part 60 Round 5 fold report

## SUMMARY

- Implemented the Round 5 billing, retry, duplicate submission, retirement, selection gate, threshold, smoke sampling, and 60j fixes.
- Enforced the registered hard $8 cap, including against the former override flag.
- Corrected retryable-error wording and recorded the listing cursor probe as unverified.
- Anchored the ledger to the code checkout so changing `PART60_BASIN` cannot reset the cap.
- Ran the required offline suite: 88 tests passed.
- No API calls, network checks, model loads, or commits were made.
- Parent-control mailbox was empty at each checkpoint; no messages were applied or deferred.

## FINDINGS / WHAT SHIPPED

| Finding | Result |
|---|---|
| Body-bearing 429 released smoke and batch reserves | Fixed with explicit `{400, 401, 403, 404, 413, 422}` allowlist plus Anthropic error-body validation; 429 retains its reservation on both paths. Smoke and batch regression tests cover the body-bearing case. |
| Batch-create 429 rejected the phase and skipped units | Fixed: 429 follows the ambiguous/retryable path, records no phase skips, and a later submit creates a new attempt under the cap. |
| Changing `PART60_BASIN` reset the ledger | Fixed: the ledger is anchored to `HERE.parent` (the code checkout), independently of `PART60_BASIN` and `--root`; a regression test changes the configured basin and checks the fixed path. |
| Duplicate POST under a second `--root` | Fixed: reserve records retain root identity through release, settle, and retire; a different root is refused when the fixed ledger already records that method/split/phase. |
| No-candidate test 60j said `held` | Fixed: it is `unscored`; the one-candidate wording remains registered. |
| Retiring a live batch could lose eventual usage | Fixed: live submitted/collecting batches cannot be retired without `--force-lost` and a nonempty reason; collection settles a retired reservation when usage is complete. |
| Dropped-method test gate was too broad | Fixed: the dev selection marker expects every non-dropped method, allowing a dropped method outside M0/M1/the selected candidate. |
| Non-finite thresholds | Fixed fail-closed in analysis: affected units are marked `threshold_missing`. |
| DEFINITIONS contradicted Amendment 6 | Fixed with the exact known-unbilled allowlist and the reservation-kept rule for retryable/unknown outcomes. |
| Smoke sample depended on history length | Fixed: seeded draw derives from the random smoke run token, the seed is recorded, and prior smoke units are excluded while fresh units remain available. |
| Listing cursor direction | Left explicitly UNVERIFIED in the RUNBOOK checklist; live probing is disallowed by this brief. |
| Cosmetic: CLI cap overwrote lower environment cap | Fixed: `--cap` only overrides when explicitly supplied. Values above $8 are now refused regardless of override flag. |
| Cosmetic: dead `unknown-reserve` branch | Removed. |
| Cosmetic: single-split analysis overwrote the other split | Fixed: single-split writes merge the prior split analysis when present. |
| Cosmetic: M7 preflight gap | Not applied; the finding does not match current code. `run60.sh` verifies only M0–M6 (`dev_methods=M0,...,M6`), while M7 is intentionally skipped and the wrapper test asserts its output may be absent. Requiring M7 would break the registered skip path. |
| Luna: current Haiku rates unverified | Unchanged and reported below as unverified; network/pricing checks were prohibited. |
| Papercut alert-001558 | DISPOSED decision applied: explicit allowlist, body-bearing 429 tests, retryable 429, and root-independent duplicate guard. |

### Tests

- Command: `PART60_BASIN=/home/elisepc/assistant-basin /home/elisepc/neuron-probe/.venv/bin/python -m unittest test_part60`
- Result: **88 tests passed**.
- The run includes **6 sabotage/mutation tests** that alter production functions in temporary copies and confirm the mutant fails its contract. New direct regression cases cover smoke and batch body-bearing 429s, two roots, dropped-method gating, live-batch retirement/retired collection, finite thresholds, smoke sampling, no-candidate 60j, and the hard cap.

### Unverified / could not check

- `RUNBOOK-60.md` checklist item: live `after_id` cursor direction and whether listing pagination is complete.
- Current Haiku 5.5 prices and actual billing treatment of the allowlisted 4xx statuses. The implementation follows the parent’s registered policy; this run made no network/API calls.
- Live batch response shapes, ambiguous accepted-batch recovery, external usage completeness, and real-run kill/restart behavior.
- GPU/Qwen scoring behavior and actual experiment metrics.
- The suite uses temporary checkout anchors for isolated billing tests; live checkout transfer behavior was not exercised.

## BLOCKERS

None for the offline code task. External facts listed above remain unverified and must be handled before paid execution as the registered workflow requires.

## SELF-ASSESSMENT

I am confident in the behavior exercised by the local fake-API and synthetic-data tests. I have not verified live billing, current prices, API pagination, or hardware scoring. No parent-control messages were received.
