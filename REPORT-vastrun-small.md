# Vastrun Small Build Report

## SUMMARY

Implemented the smaller-design lifecycle in `cloud/vastrun.py` and expanded its offline tests. The required pytest run passed 46 tests. All five requested scratch-copy sabotages made their targeted tests fail.

## FINDINGS

- Each job now gets an independent rental and setup run. Results are copied beneath `results_dir/<job>/`.
- Runs use a deadline-bearing `vastrun-<uuid12>-<epoch-hex>` label, preflight all listing pages, sweep expired labels, and refuse to rent when a non-expired vastrun label exists.
- Listing reads follow `next_token`; malformed unrelated rows are skipped, while a malformed row matching this run's label is an error. Missing and null neighbor labels do not block rent recovery or cleanup.
- Cleanup uses both the in-memory rental id and exact-label sweeps, retries DELETE/list confirmation within a deadline, handles DELETE 404 and temporarily lagging listings, and still runs after failures and interrupts. Record writes and result copies are best-effort. Spend uses the greater of the offer rate and any positive listed rate.
- The CLI keeps `JOBFILE.json` and adds `sweep-expired` and `sweep LABEL`. API and SSH calls have timeouts; a wall-clock/spend timer interrupts blocked calls.
- Secret handling uses step-only SSH errors. Tests instantiate the real Vast adapter with a fake key file and patched `urlopen`; this worker did not access the default key file or make network calls.

### Verification

Command: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q cloud/test_vastrun.py`

Result: **46 passed**. `git diff --check` also passed. The parent-control mailbox was empty. No commit was made.

Scratch-copy sabotage results (full suite run in each copy):

| Sabotage | Tests that went red |
| --- | --- |
| Cleanup skipped when a failure reaches the outer finally | `test_outer_finally_cleans_when_per_job_cleanup_raises` |
| Strict all-or-nothing listing parse | `test_unreadable_instance_list_shape_is_distinct_from_empty_list`; `test_lost_rent_recovers_our_box_with_label_less_or_null_neighbor[neighbor0]`; `test_malformed_row_matching_our_label_is_a_hard_error`; `test_uncertain_rent_follows_pagination_to_find_our_instance` |
| Listing rate lowers offer rate | `test_lower_positive_listed_rate_does_not_lower_offer_rate` |
| First page only | `test_uncertain_rent_follows_pagination_to_find_our_instance` |
| Wall-clock timer removed | `test_wall_clock_timer_stops_hung_pre_rent_listing`; `test_wall_clock_timer_stops_hung_ssh_and_cleans_up` |

## BLOCKERS

None.
