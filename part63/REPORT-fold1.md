# Part 63 round 1 fold report

## SUMMARY

- Folded the round 1 reviews and the disposed Part 63 alerts into the reader, analyzer, tests, and runbook.
- No API call, network request, model load, GPU run, or provider price check was made.
- The complete offline suite passed: 14 tests, including eight copied-code sabotage checks that all went red.
- The real key shape now selects unclamped rows by `kind` and deduplicates them by answer text.
- One authority choice remains before API submission: add independently hand-labelled gates for the scored categories, or keep the registered quality-only gate and disclose the other schemes as ungated.

## FINDINGS / WHAT SHIPPED

| Finding | Disposition | Change and evidence |
|---|---|---|
| Luna 1, high: accepted POST can be stranded after a kill | Redesigned under Part 60 Amendment 5 | The intent and request-ID map are written before POST. Ambiguous outcomes keep their reserve and are marked ambiguous; the next submit makes a fresh attempt with a fresh reserve under the cap. No reconcile or batch adoption exists. `listing` is read-only. A saved response batch ID can recover a missing local sidecar; an ID never received remains an uncollected orphan, as the amendment requires. Tests cover ambiguous retry, cap accounting, listing, and saved-ID sidecar recovery. |
| Luna 2, medium: 2% missing-label allowance and reduced 63a denominator | Fixed | A single failed or unlabelled study answer now blocks analysis. Collection sends one automatic retry containing only failed/unlabelled IDs, merges its results, and never retries a second time. 63a requires all paired Part 54a labels before returning a kappa. Tests cover one-row rejection, the retry subset, the merged result, and an incomplete paired denominator. |
| Luna 3, medium: maker aliases can change 63d | Fixed as directed | Registered 63d uses the exact case-insensitive name `Tongyi Lab` as Qwen3.5's own lab. The former broad alias mapping is emitted in a separate sensitivity row. JSON and console output identify the strict row as registered. A test demonstrates the different strict and alias results. The canonical model-to-lab mapping remains externally unverified. |
| Luna 4, medium: price and token estimate are unverified | Fixed in the local estimate; provider assumptions remain unverified | Price constants are at the top of `reader63.py` with an inherited/unverified comment. Input tokens are estimated as UTF-8 request bytes divided by 3; the reserve uses the full output-token ceiling. Collection prints actual cost against reserve. The current 4,978-request reserve estimate is $1.463121; two full-size attempts estimate to $2.926242 under the $3 cap. |
| Haiku 1, high: 63c/63d read unclamped rows at dose zero | Fixed | Baselines are selected by `kind` only, then deduplicated by answer text. The real `key63.json` fixture verifies the source doses are nonzero and that the analyzer can both hold and fail 63c and 63d on that shape. |
| Haiku 2, high: missing key or pre-POST failure consumes reserve | Fixed | Key, CA bundle, gathered files, and review-directory writability are checked before reservation. A failure in the pre-POST state write or local pre-POST API setup releases the reserve. Tests assert no key/CA reserve is written and an injected pre-POST failure ends in a release event. |
| Haiku 3, medium: duplicated texts counted as independent baseline answers | Fixed for unclamped comparisons | The analyzer deduplicates unclamped selections by text; 63b also treats the copied unclamped baseline as one deduplicated group. The real key contains 19 distinct Qwen3-4B baseline texts and 24 Qwen3.5 baseline texts, which the tests assert. |
| Haiku 4, medium: missing-label rule differs from design | Fixed | Same strict zero-tolerance gate and one failed-only retry as Luna 2. Any remaining failed or unlabelled answer blocks scoring. |
| Haiku 5, medium: calibration does not gate planet/origin/maker labels | Deferred with reason | The registered design reuses the Part 59A 34-example gate; its hand labels cover quality/BROKEN only. The runbook now says plainly that planet, origin, and maker quality is unverified by that gate. Adding independent gold labels requires authority to create/approve them, and the brief prohibits changing the frozen design. No labels were invented. Parent decision requested before API submission. |
| Haiku 6, low: temperature was implicit | Fixed | Each request sets `temperature: 0`; a copied-code sabotage removes it and the request test goes red. |
| Haiku 7, low: proxy regex matched substrings | Fixed | Keyword proxies use word boundaries, avoiding `star` in `start` and `space` in `workspace`. The copied-code sabotage reinstates the old substring regex and the check goes red. |
| Haiku 8, low: batch ID may be absent after an interrupted POST | Redesigned under Part 60 Amendment 5 | Unknown outcomes remain reserved and are never adopted by timestamp or request count. The read-only listing gives the operator provider batches created since local run start. If the response ID was saved but only the sidecar write failed, collection uses that saved ID. |
| Disposed alerts 220509-8pixy-0026 and 220510-8pixy-0027; papercut 220511-8pixy-0028 | Fixed | The dose-layout test now loads the real key shape. Missing key/CA and pre-POST failures no longer burn a reserve. |

### Registered 63d reading and sensitivity

| Reading | Own-lab rule | Role in the prediction |
|---|---|---|
| Strict exact name: `Tongyi Lab` | Only that exact name, ignoring case, counts as Qwen3.5's own lab | Registered 63d result |
| Alias sensitivity | Existing `tongyi` / `alibaba` / `qwen` aliases classify own-lab names | Sensitivity row only; it does not determine whether 63d held |

The `python -I` import papercut 220513-8pixy-0029 had no `DISPOSED` section naming Part 63, so it was outside the dispatched papercut set.

## WHAT THE SUITE PROVES

Ran `/home/elisepc/neuron-probe/.venv/bin/python -m unittest` from `part63/`: 14 tests passed, with no skips. Tests use a fake API and make no network request. They verify blind request contents, reserve-before-POST intent and ID-map persistence, preflight release behavior, ambiguous-attempt accounting, fresh-attempt IDs, one failed-only retry and result merge, no second retry, read-only paginated listing, actual-cost-versus-reserve output, strict 63d and alias sensitivity, zero-tolerance scoring, and the real-key 63c/63d paths.

Eight sabotage cases copy the relevant Part 63 source and fixtures to a temporary directory, mutate the real function through a wrapper, and show red: truncation acceptance, condition leakage, zero reserve, dropped side-key row, dose-zero baseline selection, alias shortcut, missing temperature, and substring proxy. `git diff --check` passed.

The suite does not prove provider behavior, price accuracy, semantic label quality, or that the byte/3 input estimate is a true upper bound. It does not produce a live Part 63 prediction.

## EXTERNAL ASSUMPTIONS STILL UNVERIFIED

- Current Haiku 5.5 availability and acceptance of `temperature: 0` with `thinking: {type: disabled}`.
- Messages Batches create/status/result/list response schemas, pagination fields, result usage names, and the definite-rejection behavior assumed for HTTP 400/401/403/404/413.
- Current batch input/output rates; the inherited $0.05/$0.25 per million rates are not a live quote.
- The UTF-8 byte length divided by three token upper estimate.
- TLS behavior with the operator's CA bundle, parseable provider `created_at` timestamps, and comparability with the local run clock.
- `Tongyi Lab` as the exact registered Qwen3.5 lab name and the broader alias mapping. The exact name is supported by local project wording only, not an authoritative model-source check.

## WHAT COULD NOT BE CHECKED

No API submission, batch listing, live collect, TLS handshake, pricing lookup, model load, GPU run, or human gold-label review occurred. No Haiku labels exist yet. The origin, planet-frame, and maker categories are not independently calibrated in this fold; that authority decision remains with the parent. The ambiguous accepted batch with no response ID is intentionally not recoverable or collectible under Amendment 5; its reservation remains counted and the listing is informational only.

## FILES TOUCHED

- `reader63.py`
- `analyze63.py`
- `test_part63.py`
- `RUNBOOK-63.md`
- `REPORT-fold1.md` (this report)

## BLOCKERS

The calibration decision was supplied by the parent and is implemented below. No blocker remains for the offline code fold. Before live API use, the operator still needs to check the provider and pricing assumptions listed above.

## SELF-ASSESSMENT

The implementation follows the parent's explicit Part 60 Amendment 5 retry policy and uses the actual Part 63 key, but the canonical `Tongyi Lab` choice and alias mapping are not authoritative external facts. The byte/3 estimate and inherited rates are code-side assumptions even though two local full-size reserves fit the cap. The category-calibration limitation remains explicit; duplicate agreement measures repeatability, not correctness against independent gold labels.

## Calibration decision

Parent decision, 2026-10-22: retain the registered 34-example Part 59A quality/BROKEN gate exactly as designed. No hand labels were added, and no separate category gate was authorized. Planet-frame, origin, and maker-identity schemes are reported as **not gated by hand labels** in every analysis table row that uses them.

To measure repeatability without claiming external accuracy, the request builder now takes a seeded sample of 100 distinct answers (`DUPLICATE_SEED = 63063`) and sends each selected text a second time under a separate internal ID. Duplicate requests are inserted at seeded positions, use the same blind rubric and `temperature: 0`, and do not add rows or conditions to `key63.json`. The submission snapshot records the seed, count, and answer-to-copy pairs; collected labels retain that pair list for analysis. The registered request count is now 5,078: 4,944 corpus answers, 100 repeats, and 34 calibration examples.

The analyzer calculates exact-label agreement for each scheme across the 100 pairs. Planet-frame and origin compare their string labels; maker identity compares the exact `(maker_name, maker_kind)` pair. It prints the per-scheme rates and marks each prediction using a scheme as `unreliable reader` when that scheme's agreement is below 90%. Predictions still print and remain in the returned report, with an explicit reliability field; the consistency check does not discard them or change their numerical held/failed calculation.

Updated conservative reserve estimate: **$1.492522** for 5,078 requests using the inherited Part 60 rates, UTF-8 bytes divided by three for input, and the full 1,024-token output ceiling. Two full-size attempts total **$2.985044**, remaining below the unchanged $3.00 cap. These are local estimates, not live prices. The repeated requests add 100 to both the initial request count and the one full-size reserve calculation.

Offline verification after this decision: `python3 -m unittest -v test_part63` passed all 15 tests. New coverage checks that 100 duplicate IDs exist outside the answer key, uses a fixture with 11 disagreements out of 100 to verify the 89% agreement result and below-90% flag, and verifies repeat agreement in the analysis path. The fake API tests still exercise the full request, reserve, collection, and retry flows. No live API request or new human labeling occurred.
