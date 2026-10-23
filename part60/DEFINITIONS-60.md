# Part 60 analysis definitions

This file freezes choices that the registered design leaves open. It does not change a registered method, metric, or prediction.

## Sampling and unit identity

`gen60.py` rebuilds the seed-5801 sample from `results/part56-pilot/pilot.jsonl`: it excludes the four known units, requires at least 20 top, 10 random, and 5 weak windows, shuffles the real pool and then the control pool with one RNG stream, and takes 250 of each type for dev. Test real units are restricted to the 247 IDs in `results/part58/units.test.json`; the reconstruction-only extras L0:u190, L4:u741, and L17:u1360 are excluded. The control test side remains 250, so test has 497 units total. Dev remains 500.

The checked-in `units.all.json` and `acts.all.npz` omit six otherwise eligible dev/test records (3 dev controls, 2 test neurons, and 1 test control). Their thresholds and observed ceilings are added by scoring the same fixed 1,200-row null pool on the pinned model before generated sentences are scored. The activation reproduction check compares the intersection with Part 58's archived columns and requires median ratios in 0.9–1.1 and Spearman correlation at least 0.98 overall and for neurons and controls separately. If a run has no live scoring phase to calculate missing thresholds, those units remain unverified.

M7 is skipped as a whole for a split unless every selected pilot record has at least 24 top, 10 random, and 6 weak windows. The current pilot does not, so M7 is recorded as skipped.

## Null threshold and hits

For each unit, `t_u` is the linearly interpolated 95th percentile (`numpy.percentile`, method `linear`) of that unit's maxima on the fixed 1,200-row pool sampled from `sent.all.jsonl` with `random.Random(7)`, sorted by row index. Rows whose sentence ID owner is the target unit are removed. This reproduces the Part 58 fixed-pool exclusion rule. Every method uses this same threshold. A sentence hits only when its maximum is strictly greater than `t_u`; equality is a miss.

The archived Part 58 threshold values are the device-of-record. They remain frozen for the registered comparison. M0 is rescored on the run's scoring device, and its live values are reported beside the Part 58 archive cross-check to show the cross-device effect. Any supplemental thresholds for units absent from the archive are computed on the scoring device and identified separately.

The activation reproduction check uses 200 sentence rows sampled with seed 6002. Its median ratio is new/reference over finite, nonzero reference values; Spearman uses all finite pairs, with average ranks for ties. Scoring stops if an overall or unit-type median ratio is outside 0.9 to 1.1 or its Spearman value is below 0.98.

## Per-unit summaries

`H_strong`, `H_weak`, and `H_none` are hit fractions within each tier. Every method is reported with two unit pass rates: positive-only (at least 3 of 4 strong hits) and full rule (positive-only plus at most 1 of 4 none hits). M0 has no negative tier, so its two rates are equal. The dev split contains 2,000 expected M0 texts; 1,988 baseline texts exist for 497 units, while the 3 dev controls without a baseline count as failed. Test has one control unit without an M0 baseline, counted as failed. Existing M0 texts are rescored with the same pinned Qwen scorer, dtype, and device as the other methods; `M0-archive.npz` retains the Part 58 archive values as a cross-check. Methods with strong and none tiers use the full rule when the registered prediction says “verified.” M1-style ordering is strict `mean(H_strong) > mean(H_weak) > H_none`; ties fail. The primary 60d denominator is parseable real units with at least one strong hit. A secondary 60d share uses units with at least two strong hits. Failed or unparsable outputs remain in pass-rate denominators as not passing. Tier means and AUC means use parseable units only, with denominators reported.

A unit is empirically unjudgeable (threshold at or above the pool maximum) when its threshold is at or above its largest activation on the fixed other-owner null pool. These units stay in all method denominators. Units without an observed ceiling are counted separately.

Strong-versus-none AUC is computed within each unit from raw whole-text maxima, not hit indicators. A score tie contributes one half. The reported type mean is the mean of unit AUCs with both tiers present. M3 metrics use only fresh sentences. The selection sentences are used only to choose a winner. M5 round 1 uses the M1 system prompt and is reported separately; M5 primary metrics use only round-2 sentences.

M3's selection score is the fraction of its three strong selection sentences above threshold minus the fraction of its three none selection sentences above threshold. The candidate with the largest score wins; a tie goes to the lower rank number. The winner is then scored on its four fresh strong and four fresh none sentences.

## M4 marked-token score

The markers are removed before tokenization. Fast-tokenizer character offsets identify all model tokens whose offsets overlap the marked span; when a word spans multiple subword tokens, its marked score is the maximum activation over those pieces at positions 1 or later. The marked positions are a subset of the whole-text positions, so the marked pass rate minus the whole-text pass rate must be at most zero; a positive difference is a scoring bug. M4's real positive-only and full-rule pass rates are also compared with M1's. The 60h denominator is the same marked-comparable real-unit set on both sides.

## Failure and cost accounting

Every expected unit stays in each pass-rate denominator. A schema, API, missing-result, or scoring failure counts as not passing and receives no artificial tier hit. The generation-drop denominator is phase-eligible: all units for M3 selection/M5 round 1, and predecessor-parsed units for M3 fresh/M5 round 2. When a phase has started, an eligible unit without its phase record counts as failed. A phase with no phase records has not run yet. Tier means and AUCs use valid scored outputs only; failed, skipped, threshold-missing, and unjudgeable counts are shown separately.

For Haiku 5.5 prompts up to 100K tokens, batch rates are $0.05/$0.25 and direct rates are $0.10/$0.50 per million input/output tokens. These match the published list price per `PRICE-ATTEST-2026-10-23.md`; the invoice is final, and the first real batch's reported usage is compared with the ledger. No prompt caching is requested. M0 generation usage is not charged again. The M6 five-request smoke reserves its direct-call estimate before the calls and settles with returned usage. The submit cap reserves a conservative estimate based on UTF-8 request bytes as an upper input-token estimate plus each request's `max_tokens`; settled batches replace their reserve with actual usage.

M6 uses adaptive thinking, `output_config: {effort: "high"}`, and `max_tokens: 8000`. Each five-unit smoke uses fresh dev units and passes only when all five responses include non-empty thinking, avoid `max_tokens`, and parse as the M6 schema. A failed all-five check is retried on fresh units; M6 is dropped only after three consecutive failed checks. A reservation is released only when the outcome is known to be unbilled: an HTTP 4xx response with an Anthropic error body and status 400, 401, 403, 404, 413, or 422. A timeout, connection error, 5xx, 429, or kill has an unknown outcome and keeps the reservation counted forever. Batch result errors and truncations are per-request failures under the registered greater-than-10% generation failure rule. A phase with at most 10% failures completes and can resubmit only failed units; a phase above 10% is dropped. Collection reports truncation counts and each batch's thinking fraction.

When batch usage is incomplete, the conservative reservation remains counted rather than estimating from partial rows; this can block later headroom. A kill after a smoke API call may leave spend in the ledger without a corresponding raw response row, so analysis usage can be lower than ledger spend. The ledger is the cost-accounting record for that unknown outcome.

An ambiguous batch POST (including HTTP 429) keeps its reservation and records that attempt as ambiguous. A returned batch ID is fsynced to the attempt directory before the manifest is marked submitted. A rerun with that durable ID stops and directs the operator to collect; it does not POST again. An ambiguous attempt without a returned ID may be retried with a new attempt and reservation, subject to the cap. The runner never adopts a batch by timestamp or count. `gen60.py listing` is a read-only listing of all batches returned by the API, not only batches from this run.

Ledger retirement has no force option. An attempt with a batch ID can be retired only after the API reports `ended`, `expired`, `canceled`, or `failed`; its reserve remains counted until collection settles actual usage. An attempt without a batch ID can be retired only after 25 hours have elapsed since reservation. An unreadable manifest is treated as active. Collection refuses a settlement that would exceed the cap and leaves the reservation counted.

The dev-selected test marker is bound to a SHA-256 fingerprint of current dev outputs, activations, thresholds, reproduction record, pilot input, and analysis implementation. The GPU runner opens test only after `verify-dev` succeeds and the fingerprint matches.

## Prediction 60j test set

The registered test phase includes M0, M1, and the single dev-selected candidate. A candidate advances only when its real-minus-control gap leads M0 by at least five points for both positive-only and full-rule rates, (the registered point rule); the paired-bootstrap 95% interval is reported beside it and is not a second condition. Prediction 60j uses the same dual rule against M0 and M1 (M0 only when the candidate is M1). The test phase is prospective for new outputs but is not an untouched holdout of earlier decisions; every other method remains unscored on test. If no candidate advances, 60j is unscored. Paired bootstrap intervals use 2,000 unit resamples with fixed seed 6003 for method-minus-M0 and method-minus-M1 comparisons.

Interpretation limits: control hits are not a false-discovery rate for labels because Part 58 notes that some random directions carry real features; only 247 real test units with at least 20 top, 10 random, and 5 weak windows are covered, so no atlas-wide claim follows; the test split is prospective for new outputs but not an untouched holdout of earlier decisions; M3 selects a hypothesis-plus-probe distribution and uses about twice M1's generations.
