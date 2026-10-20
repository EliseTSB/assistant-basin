# Part 31 scorer fold report

## Finding 2 — Whole-cell return-time quantiles

In [part31.py](/home/elisepc/green/worktrees/ab-part31-build/part31.py:507), `_describe` now computes nearest-rank quantiles over all eligible samples. Non-returners sort after observed returns and render as `>N`, where `N` is the recorded observation horizon. The existing returned-only quantiles remain under `conditional_on_return_*`. The cell horizon is release-relative (`max_new_tokens - K`).

Covered by `test_whole_cell_quantiles_keep_nonreturners_censored` in [test_part31.py](/home/elisepc/green/worktrees/ab-part31-build/test_part31.py:260): `[1, 10, None]` reports whole-cell median `10` and returned-only median `5.5`; `[1, None, None]` reports `>128`. Pass.

Interpretation: nearest-rank was selected to match the review's `[1,10,None]` counterexample. The censored marker includes the numeric horizon as `>N`; the adjacent `observation_horizon` field gives the exact limit.

## Finding 3 — Cache comparison by stage and K

In [part31.py](/home/elisepc/green/worktrees/ab-part31-build/part31.py:763), 31.5 now reports each cache arm's censored whole-cell state-return distribution and the own-cache-minus-reencoded median difference for each observed stage/K cell. The difference is `None` if either median is censored. The paired-sample difference remains separately named and explicitly conditional on both arms returning.

Covered by `test_cache_summary_reports_each_arm_median_per_stage_and_k` in [test_part31.py](/home/elisepc/green/worktrees/ab-part31-build/test_part31.py:268): arm medians are 20 and 19 (difference 1), while the separate paired-difference median is 19. Pass.

Interpretation: the signed difference follows the prediction's own-cache minus reencoded-cache order. A difference involving a censored median is not assigned a numeric value.

## Finding 4 — Prediction measurements and missing labels

In [part31.py](/home/elisepc/green/worktrees/ab-part31-build/part31.py:347), scoring adds the first halfway-to-baseline-band-center time, anchored at the final pushed projection and searched over post-release tokens. 31.1 reports this and text-return distributions by cache arm, plus the joint fraction. In [part31.py](/home/elisepc/green/worktrees/ab-part31-build/part31.py:714), 31.3 is any-time text snapback among labeled prefill samples, per stage/K. In [part31.py](/home/elisepc/green/worktrees/ab-part31-build/part31.py:738), 31.6 measures whether the projection is outside the band at release-relative token 20.

Missing labels are marked `no labels`; rates return `None` when no samples can be labeled and exclude unlabeled rows from their denominator. A labeled non-returner observed at least 50 tokens is treated as text return at or after 50. A fixed-time projection with no token-20 observation is excluded and counted as unmeasured.

Covered by `test_prediction_measurements_use_halfway_snapback_and_outside_events` and `test_missing_judge_labels_are_reported_as_missing_not_zero` in [test_part31.py](/home/elisepc/green/worktrees/ab-part31-build/test_part31.py:289): the halfway time is 2 although band return is 30; a text return at 50 satisfies 31.1; a prefill return at 30 counts for 31.3; the 31.6 projection is outside at token 20 despite late return; missing labels produce `None`/`no labels`. Pass.

Interpretations: “release value” means the projection after the final pushed token (trace index `K-1`); the first candidate post-release projection is token 1 (index `K`). The 31.1 output is split by cache arm rather than pooling those instruments. A labeled text non-return observed through token 50 satisfies “return at 50 or later”; no labels never count as false.

## Finding 6 — Resume fingerprint

In [part31.py](/home/elisepc/green/worktrees/ab-part31-build/part31.py:1671), `run_fingerprint.json` records the source SHA-256, required stage revisions, prompt hash, K list, samples per prompt, max new tokens, and seed. [run_experiment](/home/elisepc/green/worktrees/ab-part31-build/part31.py:1983) verifies it before loading directions or completed sample IDs. Missing or mismatched fingerprints with existing artifacts raise a clear error; `--allow-resume-mismatch` explicitly overrides. When overriding existing artifacts, the stored fingerprint is preserved rather than relabeled.

Covered by `test_resume_fingerprint_mismatch_is_refused` in [test_part31.py](/home/elisepc/green/worktrees/ab-part31-build/test_part31.py:376). It confirms a stale fingerprint with `samples.jsonl` raises an error naming the override flag. Pass.

Interpretation: the revision map includes Instruct plus the selected run stages because Instruct is always needed for the shared direction. Explicit override leaves the prior fingerprint intact so subsequent resumes remain explicit.

## EOS samples

[score_sample and make_summary](/home/elisepc/green/worktrees/ab-part31-build/part31.py:390) exclude `eos_within_push` rows from return-time and post-release measurements, and each summary cell reports `eos_within_push_count`. This also excludes fixed post-release measurements because those samples never reached release. `test_eos_within_push_is_excluded_and_counted_per_cell` in [test_part31.py](/home/elisepc/green/worktrees/ab-part31-build/test_part31.py:360) verifies exclusion and the count. Pass.

## Verification

`/home/elisepc/neuron-probe/.venv/bin/python test_part31.py` — 18 tests passed, including all 12 pre-existing tests. `git diff --check` passed. No model downloads were run.

SUMMARY: Findings 2, 3, 4, and 6 plus EOS-within-push handling shipped.
FINDINGS: 18/18 tests pass; full details and chosen interpretations are above.
BLOCKERS: none
