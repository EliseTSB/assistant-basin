# Vastrun build report

## SUMMARY

Implemented `cloud/vastrun.py` and its offline lifecycle tests. The runner accepts a JSON job file, uses injected Vast and SSH adapters in tests, and writes a run record into the configured results directory. No Vast instance or SSH host was contacted during implementation or testing, no API key file was read, and nothing was committed.

## FINDINGS

Offers are filtered by the RTX 3090/on-demand requirements, verification, reliability, CUDA, disk, download speed, and configured hourly price, then sorted cheapest first. The runner tries at most three hosts, records each attempt, destroys hosts that do not become SSH-ready within 180 seconds, and confirms destruction through the v1 instances endpoint. A unique label lets it recover a host if the rent response is lost. Jobs run sequentially under `nohup`, write their exit code to their own log, are polled every 30 seconds by default, and have expected files plus logs copied back as soon as each ends. Spend, wall-clock limits, SIGINT/SIGTERM, exceptions, and final cleanup all lead to destruction attempts; failed confirmation exits nonzero. The startup watchdog runs `poweroff` after `wall_clock_min + 10` minutes and contains no API key. `run-record.json` captures hosts, instance, rate, timestamps, estimated spend, job return codes/durations, and destroy confirmation.

Job files use `upload_files`, optional `setup_command`, ordered `jobs` containing `name`, `command`, optional `env` and `expected_result_paths`, plus `results_dir`, `max_dph`, `spend_cap_usd`, and `wall_clock_min`. Optional `remote_workdir` and `ssh_user` are supported.

Storage billing caveat: yes, a powered-off/stopped Vast instance can continue to incur charges for retained disk storage. The watchdog bounds powered-on compute time; it does not cap total cost until the instance is successfully destroyed. Live account rates and current provider policy were not checked because network access was prohibited.

The requested pytest command passed: 7 tests passed. They cover readiness fallback, exception cleanup, spend cap cleanup, immediate copy-back before a later job failure, destroy-confirmation failure, uncertain rent recovery, and key non-disclosure using fakes. `git diff --check` was clean. Parent-control mailbox was empty; no messages changed the plan or were declined/deferred.

## BLOCKERS

None. The storage policy statement above is not live-verified under the no-network constraint.
