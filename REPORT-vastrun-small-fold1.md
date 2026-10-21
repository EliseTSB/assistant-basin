# Vastrun small fold 1 report

## SUMMARY

Unified instance-list pagination and added page-two regression coverage for the requested cleanup, sweep, recovery, and preflight paths. The required offline suite passes: 47 tests. No network calls, credential-file reads, or commits were made.

## FINDINGS

`_listed_instances` is now the sole pagination implementation. Runner call limits are retained through an injected request callback, while cleanup and module-level listing callers use the same token loop. Removed `Runner._list_all_instances`.

Added or adapted tests with an unrelated neighbor on page one and the target instance on page two. They cover exact-label cleanup after a finished job, lost-rent recovery, `sweep_expired`, CLI `sweep LABEL`, and preflight rejection of a live label. All assert page two was visited and the expected cleanup or refusal occurred.

Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q cloud/test_vastrun.py` — **47 passed**. `git diff --check` passed.

Scratch-copy sabotage runs (full suite in each copy):

| Sabotage | Tests that went red |
| --- | --- |
| Skip cleanup in outer finally | `test_outer_finally_cleans_when_per_job_cleanup_raises` |
| Strict all-or-nothing listing parse | `test_unreadable_instance_list_shape_is_distinct_from_empty_list`; `test_lost_rent_recovers_our_box_with_label_less_or_null_neighbor[neighbor0]`; `test_lost_rent_recovers_our_box_with_label_less_or_null_neighbor[neighbor1]`; `test_malformed_row_matching_our_label_is_a_hard_error` |
| Listed rate lowers offer rate | `test_lower_positive_listed_rate_does_not_lower_offer_rate` |
| First page only (original split-implementation sabotage) | `test_uncertain_rent_follows_pagination_to_find_our_instance`; `test_finished_job_exact_label_cleanup_follows_pagination`; `test_non_expired_labeled_instance_refuses_new_rent`; `test_sweep_expired_ignores_non_expired_labels`; `test_cli_sweep_label_command_uses_exact_label` |
| Wall-clock timer removed | `test_wall_clock_timer_stops_hung_pre_rent_listing`; `test_wall_clock_timer_stops_hung_ssh_and_cleans_up` |
| First page only in the shared implementation | `test_uncertain_rent_follows_pagination_to_find_our_instance`; `test_finished_job_exact_label_cleanup_follows_pagination`; `test_non_expired_labeled_instance_refuses_new_rent`; `test_sweep_expired_ignores_non_expired_labels`; `test_cli_sweep_label_command_uses_exact_label` |

The parent-control mailbox was empty at each checkpoint; no messages changed or deferred the plan.

## BLOCKERS

None. Live Vast behavior was not checked because network access was prohibited.
