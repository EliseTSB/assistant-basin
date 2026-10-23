# Part 60 runner report

## SUMMARY

The Part 60 generation, scoring, analysis, and rental-runner package is implemented. Local validation uses fake Haiku batches, synthetic activations, cached Part 58 data, and a stubbed rental run. No Anthropic request or Qwen model load was made in this build environment.

## FINDINGS / WHAT SHIPPED

`gen60.py` rebuilds the seed-5801 pilot splits, materializes M0, constructs blinded window messages, and implements resumable one-chunk Message Batches calls for M1–M7. It records parse and API failures, rejects unknown or duplicate batch states, tracks estimated and settled token spend, enforces the `$8` cap, skips M7 when its window requirements are not met, and applies the registered greater-than-10% generation failure rule. M3 and M5 use separate phases; round 2 M5 feedback contains only fired yes/no values.

`score60.py` reuses cached M0 activations, computes frozen Part 58 null thresholds, performs the pinned hook scoring and M4 marked-token measurements, checks reproduction against 200 baseline texts, resumes by sentence identity, and refuses scoring when reproduction falls outside the registered ratio range. The Part 58 activation archive lacks six units selected by the reconstructed split (three dev controls, two test neurons, and one test control); a supplemental threshold rescore path handles the missing thresholds on the scoring GPU. The available archive supports cached M0 for 497 of 500 dev records.

`analyze60.py` computes method and type metrics, evaluates M3 on fresh sentences and M5 on round 2, produces predictions 60a–60j, records dropped generation phases, and prevents dropped methods from contributing to prediction decisions. Definitions and explicit choice resolutions are in `DEFINITIONS-60.md`. `run60.sh` provides independent phases, resumable scoring, a stub mode, and separate ready/incomplete archives. `RUNBOOK-60.md` gives the local generation, rented-GPU, return, analysis, and instance-destruction procedure with an explicit package file list.

Offline validation: `python3 -m py_compile gen60.py score60.py analyze60.py test_part60.py`, `bash -n run60.sh`, and `python3 -m unittest -v test_part60.py` passed. All 18 tests passed, including the real SIGKILL/no-resubmit recovery check, cap and parse-failure cases, fixed-null hand calculation, tier metrics, blinding, and the wrapper stub. The two deliberate sabotage subprocesses exited 1 as expected: including a unit's own texts in its null and evaluating M3 on selection sentences.

The design estimates dev generation at roughly `$3–$6` total with an `$8` cap. A rough per-method estimate from the Part 58 short-text cost anchor is M1 `$0.45`, M2 `$0.45`, M3 `$0.98` including its second call, M4 `$0.45`, M5 `$0.75` including round 2, M6 `$0.45`, M7 `$0` because it is skipped, and M0 `$0` because it is reused. The roughly `$3.53` total is a planning estimate, not measured spend; token usage and actual costs are recorded by the runner.

## BLOCKERS

No code blocker remains for handing off the runner. Actual Haiku generation, GPU scoring, and rented-instance validation were not run here, as required by the brief. Their result files and measured costs remain pending those phases. The six-unit archive mismatch is documented and has a supplemental threshold scoring path, but that path has not been executed against Qwen in this environment.
