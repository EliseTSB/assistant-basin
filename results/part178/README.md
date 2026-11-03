# Part 178: Stored and eager thinking text both place `wait` among the top ten openers

`wait` ranked second in stored set `U` and tenth in eager generated set `G`. In the eager runs, its thinking to non-thinking rate ratio held for Qwen3.5-0.8B and was inconclusive for Qwen3.5-4B. The 4B rank interval reaches outside the top ten, and the pooled follow-on interval includes its bar. These are associations under the registered opener rule. The analysis does not test whether a word changes the model.

```text
wait opener rate per 1,000 openers
U  86.22  ████████████████████
G  19.59  █████
```

The bars use a shared scale. Source: `eager/score.json`, `tables.window.U.rates[word=wait].rate_per_1000` and `tables.window.G.rates[word=wait].rate_per_1000`.

## Predictions

| # | Registered prediction | Eager result | Verdict |
| --- | --- | --- | --- |
| 178.1 | In `U`, `wait` is among the top 10 openers by `rate(w)` (medium) | Rank 2; 86.22 per 1,000 openers; rank interval 1 to 2 | held |
| 178.2 | In `G`, `wait` is among the top 10 openers by `rate(w)` (medium) | Rank 10; 19.59 per 1,000 openers; rank interval 7 to 11 | held |
| 178.3 | `rate(wait)` differs by less than a factor of 1.5 between `U` and `E` on `window`; unread if either is zero (medium) | Largest rate factor 1.17; 95% interval 1.09 to 1.25 | held |
| 178.4 | At least 14 of `U`'s top 20 openers are in `G`'s top 20 (medium) | 14 of 20 overlap, exactly at the bar; no interval was computed for this pair | held |
| 178.5 | At most 6 of the 17 imported words reach 5 per 1,000 paragraph openers in `U` (medium) | 4 words: `wait`, `alternative`, `verify`, `now`; 95% interval 3 to 5 | held |
| 178.6 | The follow-on statistic on eligible traces is at least 0.15; not read below 40 eligible traces (low) | Mean 0.207; 95% interval 0.101 to 0.306; 66 eligible traces | held |
| 178.7 | At least 10 of `G`'s top 20 openers have a mode ratio under 2; unread if more than 5 lack a ratio (medium) | 15 of 20 are under 2; 95% interval 13 to 16; none lack a ratio | held |
| 178.W1 | Ranking by `rate(w)` and by `traces(w)` disagree on at least 4 of the top 20 in `U`. Falsifier: 1 or fewer (medium) | 2 differing entries by the scorer's one-sided count; 95% interval 1 to 3 | inconclusive |
| 178.W2 | Among traces with both labels, the median position of the first `asserted` selected hit exceeds the first `intermediate` selected hit's by at least 2,000 tokens. Falsifier: under 500 (medium) | Difference 742.5 tokens; 95% interval −4,242.5 to 3,515.5; 40 traces with both labels | inconclusive |
| 178.W3a | On Qwen3.5-0.8B, `wait` occurs at least 3 times more often per 1,000 openers in thinking than in non-thinking text, by mode ratio `R8` over `N8`. Falsifier: ratio under 1.5. Unread under 500 control openers or with no ratio (medium) | Ratio 4.66; 95% interval 4.13 to 5.33; 10,432 control openers | held |
| 178.W3b | The same on Qwen3.5-4B, by mode ratio `G` over `N4`. Unread under 500 control openers or with no ratio (medium) | Ratio 1.96; 95% interval 1.57 to 2.62; 7,813 control openers | inconclusive |

Source for this table: `eager/score.json`, `predictions.<id>.prediction`, `.estimate`, `.verdict`, and `.rule`. The 178.4 score field `estimate.ci95_overlap` is copied from Gate 5's `U`–`R8` bootstrap. It is not an interval for the `U`–`G` overlap. That overlap passes on a single count of 14.

All seven numbered predictions held on their point estimates. The interval for 178.2 runs from rank 7 to rank 11, so it includes a rank outside the top ten. The interval for 178.6 includes its 0.15 bar. The scorer's verdict reads the point estimate against the registered bar. The scorer's output and prediction table agree on all 11 verdicts; both are scorer output, not independently registered expectations. Source: `eager/readme_numbers.json.summary_counts.numbered_prediction_count` and `.prediction_count`, plus `predictions.<id>.estimate` and `.verdict`.

For W1, the scorer counts `20 − |rate-top-20 ∩ trace-top-20|`. It reports 2 rate-ranked entries absent from the trace-ranked list. The symmetric difference is 4 entries; under that reading W1 would be held. Under the scorer's one-sided convention W1 is inconclusive. Source: `predictions.178.W1.estimate.rate_top20`, `.trace_top20`, `.different_top20_entries`, `.ci95`, and `eager/readme_numbers.json.w1_top20_symmetric_difference.count`.

W3a held on the 0.8B model and W3b was inconclusive on the 4B model. Only one model's W3 prediction held. The registered consequence is to run Part 179 on the 0.8B model and state that generality has not been established. Source: `predictions.178.W3a.verdict`, `predictions.178.W3b.verdict`, and `eager/readme_numbers.json.summary_counts.w3_held_model_count`, `.registered_context.next_part`.

## Mode comparison and trace spread

The table shows the eager window rates and how many of the 200 traces in each set contain at least one `wait` opener. Controls are mathematical answers, not general prose. The two modes also use different prompt endings, an open think tag for thinking and an empty think block for non-thinking. This comparison cannot separate mode from prompt format.

| Set | Mode | `wait` per 1,000 openers | Traces with `wait` |
| --- | --- | ---: | ---: |
| R8 | 0.8B thinking | 86.30 | 199 / 200 |
| N8 | 0.8B non-thinking control | 18.50 | 110 / 200 |
| G | 4B thinking | 19.59 | 168 / 200 |
| N4 | 4B non-thinking control | 9.98 | 52 / 200 |

Sources: `eager/score.json`, `tables.window.<set>.rates` and `tables.window.<set>.trace_rates`, each selected by `word = wait`; the same entries are copied to `eager/readme_numbers.json`, `window_wait.<set>`. The two control rates are the denominators needed to interpret the mode ratios.

The uncut sensitivity keeps the same token limit but removes the onset cut. `wait` ranks second in `U` at 86.91 per 1,000 and ninth in `G` at 19.85. Its rank stays in the top ten for both. Source: `eager/readme_numbers.json`, `uncut_wait.U` and `.G`, copied from `eager/score.json`, `tables.uncut`.

## Cap, length, and thinking-text sensitivity

The four eager sets each contain 200 rows. Cap counts differ sharply between thinking and non-thinking sets. Only 3 of 200 `R8` rows and 39 of 200 `G` rows contain the think-close token `</think>`. For the other thinking rows, the defined thinking text is the full generated response. Source: `eager/readme_numbers.json.summary_counts.generated_set_count` and `generated_sets.<set>.row_count`.

| Set | `finish_reason = cap` | `finish_reason = end-of-turn` |
| --- | ---: | ---: |
| R8 | 197 / 200 | 3 / 200 |
| N8 | 97 / 200 | 103 / 200 |
| G | 175 / 200 | 25 / 200 |
| N4 | 46 / 200 | 154 / 200 |

Sources: `eager/readme_numbers.json`, `generated_sets.<set>.row_count`, `.finish_reason_counts`, and `.think_close.rows_containing_token`; close-token membership was counted from each JSONL row's `think_close_token_id` in `generated_ids`.

Thinking rows reach the cap much more often than the controls. That makes the mode ratios length-confounded. The next table is a full-text sensitivity check. It applies `words178.opener_spans` to the complete `generated_text`, without truncating at `</think>`, applying the registered token window, or cutting at onset. Each cell gives wait openers divided by all openers, the rate per 1,000, and the row count.

| Set | Capped rows | End-of-turn rows | All rows |
| --- | --- | --- | --- |
| R8 | 1,617 / 18,081; 89.43 per 1,000; 197 rows | 5 / 230; 21.74 per 1,000; 3 rows | 1,622 / 18,311; 88.58 per 1,000 |
| N8 | 178 / 7,863; 22.64 per 1,000; 97 rows | 29 / 3,050; 9.51 per 1,000; 103 rows | 207 / 10,913; 18.97 per 1,000 |
| G | 333 / 15,703; 21.21 per 1,000; 175 rows | 6 / 1,752; 3.42 per 1,000; 25 rows | 339 / 17,455; 19.42 per 1,000 |
| N4 | 67 / 3,433; 19.52 per 1,000; 46 rows | 13 / 4,500; 2.89 per 1,000; 154 rows | 80 / 7,933; 10.08 per 1,000 |

Source: `eager/readme_numbers.json`, `capped_sensitivity_full_text.sets.<set>.<finish_reason>`, fields `wait_opener_count`, `opener_count`, `wait_rate_per_1000_openers`, and `trace_count`.

On capped full text, the 0.8B `R8` to `N8` ratio is 3.95, while the 4B `G` to `N4` ratio is 1.09, below the registered 1.5 falsifier. Across all full text, the ratios are 4.67 and 1.93. Conditioning on `finish_reason` is a post-generation sensitivity check, not a replacement verdict. Sources: `eager/readme_numbers.json`, `capped_sensitivity_full_text.thinking_to_control_wait_rate_ratios` at `["0.8B"].by_finish_reason.cap`, `["4B"].by_finish_reason.cap`, and `.by_finish_reason.all`; the falsifier is in `eager/score.json.predictions.178.W3b.rule`.

## Follow-on groups and withdrawal screen

The pooled follow-on estimate is 0.207, with a 95% interval from 0.101 to 0.306. Its 0.15 bar lies inside that interval. The descriptive groups are:

| Onset side | Mean difference | 95% interval | Traces |
| --- | ---: | ---: | ---: |
| Before onset | 0.163 | 0.042 to 0.283 | 49 |
| After onset | 0.358 | 0.169 to 0.550 | 16 |
| No onset | −0.059 | −0.059 to −0.059 | 1 |

Sources: `eager/score.json`, `predictions.178.6.estimate.point`, `.ci95`, `.eligible_traces`, `.by_onset_side.<group>`, and `.rule`; the group objects are also copied into `eager/readme_numbers.json`, `follow_on_by_onset_side`.

The registered withdrawal screen passed: `withdrawn` is false. `wait` reaches 5 per 1,000 in `U` and `G`, and its 0.8B mode ratio is at least 2. Source: `eager/score.json`, `withdrawal.withdrawn`, `.rule`, `.reflection_rates_U_G`, and `.reflection_mode_ratios_G_N4_R8_N8`.

## Method and checks

`U` and `E` are stored sample-0 texts and were not generated under the eager protocol. `R8` and `G` are eager thinking sets; `N8` and `N4` are their non-thinking controls. The generated rows record eager attention, a 2,048-token generation cap, temperature 0.6, top-k 20, top-p 0.95, and presence penalty 1.5. Batch sizes are not recorded in the rows, so the JSONL cannot verify them. Sources: `eager/readme_numbers.json`, `generated_sets.<set>`; the scope is `eager/score.json.scope`.

The registered opener window uses the first 2,000 tokens of a 2,048-token prefix, with an onset cut. Stored text is retokenized; generated text uses kept token IDs. Rates are pooled per 1,000 openers, with intervals by trace. Sources: `eager/readme_numbers.json`, `definitions.prefix_token_limit`, `.window_token_limit`, `.onset_censor_lines`, and `eager/score.json`, `tables.window`.

All five gates and the opener-adequacy check passed. The per-set window totals range from 7,813 to 18,288 openers, above the 2,000-opener minimum. Over `U`'s top 20 openers, the Spearman correlation of `rate(w)` between `R8` and `U` is 0.986, above the 0.7 threshold; all 20 of `U`'s top 20 also occur in `R8`'s top 20. This is Gate 5's rank check, not a validation of absolute rates or token identities. The separate `U`–`G` top-20 overlap is 14 of 20. Sources: `eager/readme_numbers.json.summary_counts.gate_count` and `eager/score.json`, `read_gates.gate1` through `.gate5`, `.adequacy`, `.gate5.spearman`, `.gate5.threshold`, `.gate5.U_top20_in_R8_top20`, and `.adequacy.by_set`.

The eager reruns resolve the device check. The diagnostic found that the earlier SDPA decode could diverge from the CPU reference on confident positions. All four generated sets were regenerated with eager attention and rescored. Generated-set results now use those eager rows, including 178.2, 178.4, 178.7, W3a, and W3b. Gate 5's `R8`–`U` check also uses eager rows. The diagnostic and rerun are recorded in `PART178-DESIGN.md` and `eager/score-run.log`.

## SDPA runs (superseded)

The old SDPA scores are shown only as a comparison. The estimates used above that changed are:

| Estimate | SDPA | Eager |
| --- | ---: | ---: |
| `G` window `wait` rate and rank | 21.16; rank 8, interval 6 to 10 | 19.59; rank 10, interval 7 to 11 |
| `R8` window `wait` rate; traces with `wait` | 85.50; 199 / 200 | 86.30; 199 / 200 |
| `N8` window control rate; traces with `wait` | 20.94; 120 / 200 | 18.50; 110 / 200 |
| `N4` window control rate; traces with `wait` | 9.56; 55 / 200 | 9.98; 52 / 200 |
| `G` uncut `wait` rate and rank | 21.13; rank 8 | 19.85; rank 9 |
| 178.7 top-20 words with ratio under 2 | 14 of 20; interval 13 to 17 | 15 of 20; interval 13 to 16 |
| W3a ratio; interval; control openers | 4.08; 3.61 to 4.60; 10,601 | 4.66; 4.13 to 5.33; 10,432 |
| W3b ratio; interval; control openers | 2.21; 1.81 to 2.83; 7,847 | 1.96; 1.57 to 2.62; 7,813 |
| Gate 5 Spearman; `U` top-20 words in `R8` top 20 | 0.991; 20 | 0.986; 20 |
| Gate 5 `U`–`R8` overlap interval | 18 to 20 | 19 to 20 |

Sources: the SDPA values are in `score.json`, and the eager values in `eager/score.json`. Each row maps to the corresponding `tables.window`, `tables.uncut`, `predictions.178.2`, `predictions.178.7`, `predictions.178.W3a`, `predictions.178.W3b`, or `read_gates.gate5` fields. The interval in the final row is Gate 5's `U`–`R8` interval, which the scorer also attaches to 178.4. It is not an interval for `U`–`G`; the 178.4 point estimate remains 14 of 20 in both scores, while the top-20 membership changes.

All 11 registered verdicts are the same in the SDPA and eager score files. Estimates for 178.1, 178.3, 178.5, 178.6, W1, and W2 are unchanged. Sources: `score.json.predictions.<id>.verdict` and `eager/score.json.predictions.<id>.verdict`.

## Files

- `../../PART178-DESIGN.md`: registered design, definitions, amendment, and verdict rules.
- `../../PREDICTIONS.md`: registered prediction wording and confidence.
- `eager/score.json`: registered eager estimates, verdicts, and gates.
- `eager/readme_numbers.py` and `eager/readme_numbers.json`: row-derived cap, think-close, control-spread, and full-text sensitivity values, plus copied score fields used here.
- `eager/score-run.log`: eager scorer gate and verdict log.
- `eager/R8.jsonl`, `eager/N8.jsonl`, `eager/G.jsonl`, `eager/N4.jsonl`: eager generated rows.
- `score.json`: superseded SDPA score, used only in the comparison section.
- `README-NOTES.md`: number-by-number source map for this write-up.
