# Part 60 analysis definitions

This file freezes choices that the registered design leaves open. It does not change a registered method, metric, or prediction.

## Sampling and unit identity

`gen60.py` rebuilds the registered Part 58 sample from `results/part56-pilot/pilot.jsonl`: it excludes the four pre-existing known units, requires at least 20 top, 10 random, and 5 weak windows, uses one seeded RNG stream (seed 5801) to shuffle the real pool and then the control pool, and takes 250 of each type for dev and the next 250 of each type for test. The checked-in `units.dev.json` and `units.test.json` list only the real side (250 and 247 rows). Control records come from the same pilot selection. The recorded real-unit IDs match where present; the extra three valid test records keep the registered 250-per-type design.

The checked-in `units.all.json` and `acts.all.npz` omit six otherwise eligible pilot records selected into dev/test (3 dev controls, 2 test neurons, and 1 test control). Their thresholds are added by rescoring the same fixed 1,200-row null pool on the pinned model before generated sentences are scored. The activation reproduction check compares the intersection of the selected split and Part 58's archived columns, records that coverage, and requires the median ratio to be within 0.9–1.1 overall and for real neurons and controls separately. If a run has no live scoring phase to calculate those missing thresholds, analysis leaves those units unverified instead of deriving a threshold from method outputs.

M7 is skipped as a whole for a split unless every selected pilot record has at least 24 top, 10 random, and 6 weak windows. The current pilot does not, so M7 is recorded as skipped.

## Null threshold and hits

For each unit, `t_u` is the linearly interpolated 95th percentile (`numpy.percentile`, method `linear`) of that unit's maxima on the fixed 1,200-row pool sampled from `sent.all.jsonl` with `random.Random(7)`, sorted by row index. Rows whose sentence ID owner is the target unit are removed. This reproduces the Part 58 fixed-pool exclusion rule. Every method uses this same threshold. A sentence hits only when its maximum is strictly greater than `t_u`; equality is a miss.

The activation reproduction check uses 200 sentence rows sampled with seed 6002. Its median ratio is new/reference over finite, nonzero reference values; Spearman uses all finite pairs, with average ranks for ties. Scoring stops if the overall median ratio or either unit-type median is outside 0.9 to 1.1.

## Per-unit summaries

`H_strong`, `H_weak`, and `H_none` are hit fractions within the corresponding generated tier. M0 has only four baseline texts, so its pass criterion is at least 3 of 4 strong texts hitting. Its clean texts come from `sent.all.jsonl` joined by the arm-B SIDs in the Part 58 `gen.*` files; activations are reused from `acts.all.npz`; only arm B and its four original sentences are included. Units with no cached baseline activation remain unverified. Methods with strong and none tiers are verified only when `H_strong >= 0.75` and `H_none <= 0.25`. M1-style ordering is strict `mean(H_strong) > mean(H_weak) > H_none`; ties fail. The ordering denominator is parseable real units with at least one strong hit. Units with failed or unparsable output remain in the verified-rate denominator as not verified. Tier means and AUC means use parseable units only, and their denominators are reported.

Strong-versus-none AUC is computed within each unit from raw whole-text maxima, not hit indicators. A score tie contributes one half. The reported type mean is the mean of unit AUCs with both tiers present. M3 metrics use only fresh sentences. The selection sentences are used only to choose a winner. M5 round 1 uses the M1 system prompt and is reported separately; M5 primary metrics use only round-2 sentences.

M3's selection score is the fraction of its three strong selection sentences above threshold minus the fraction of its three none selection sentences above threshold. The candidate with the largest score wins; a tie goes to the lower rank number. The winner is then scored on its four fresh strong and four fresh none sentences.

## M4 marked-token score

The markers are removed before tokenization. Fast-tokenizer character offsets identify all model tokens whose offsets overlap the marked span; when a word spans multiple subword tokens, its marked score is the maximum activation over those pieces at positions 1 or later. A per-unit marked-token pass is at least 3 of 4 strong marked scores above `t_u`. The whole-text comparison is at least 3 of those same 4 sentences with whole-text maximum above `t_u`. Prediction 60h compares the shares of real units passing those two criteria.

## Failure and cost accounting

Every expected unit stays in each verified-rate denominator. A schema, API, missing-result, or scoring failure counts as not verified. It is not assigned an artificial tier hit. Tier means and AUCs use valid scored outputs only; failed and skipped counts are shown separately. The M3 and M5 second calls are not attempted for units whose first call did not parse.

Batch usage is priced at $0.05 per million input tokens and $0.25 per million output tokens. No prompt caching is requested. M0's old generation usage is not charged again and is reported as zero new Part 60 generation cost. The submit cap reserves a conservative estimate based on UTF-8 request bytes as an upper input-token estimate plus each request's `max_tokens`; settled batches replace their reserve with actual usage.

## Prediction 60j test set

The registered test phase includes M0, M1, and the single dev-selected candidate when the dev separation gain reaches the 5-point stop-rule threshold. These fixed comparators let 60j compare the candidate's real-minus-control gap to both M1 and M0. It is one held-out test phase; every other method is unscored on test. If the stop rule keeps M0, no candidate test run is selected and 60j is reported unscored. A result that fails to beat both comparators applies to the registered test candidate; it cannot rule out untested dev methods.
