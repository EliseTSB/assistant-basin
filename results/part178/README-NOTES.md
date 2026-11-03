# Number sources for README.md

The README's registered results use `eager/score.json`. Values recomputed from generated rows or copied out for a concise source map use `eager/readme_numbers.json`. In score tables, `rates` and `trace_rates` are arrays; `word=wait` means the entry whose `word` field is `wait`.

## Claim, chart, and prediction table

| README content | Source key or field |
| --- | --- |
| Part label and registered scope | `eager/score.json.part`, `.scope`; `eager/readme_numbers.json.registered_context.part` |
| U/G chart values 86.22 and 19.59 | `eager/score.json.tables.window.U.rates[word=wait].rate_per_1000`; `.tables.window.G.rates[word=wait].rate_per_1000` |
| 178.1 rate, rank, rank interval, verdict | `eager/score.json.predictions.178.1.estimate.rate`, `.rank`, `.rank_ci95.lower`, `.rank_ci95.upper`, `.verdict` |
| 178.2 rate, rank, rank interval, verdict | `eager/score.json.predictions.178.2.estimate.rate`, `.rank`, `.rank_ci95.lower`, `.rank_ci95.upper`, `.verdict` |
| 178.3 factor and interval | `eager/score.json.predictions.178.3.estimate.max_factor`, `.max_factor_ci95.lower`, `.max_factor_ci95.upper`; rule and verdict at `.rule` and `.verdict` |
| 178.4 overlap count and top-20 lists | `eager/score.json.predictions.178.4.estimate.overlap`, `.U_top20`, `.G_top20`; rule and verdict at `.rule` and `.verdict` |
| 178.4's attached interval field | `eager/score.json.predictions.178.4.estimate.ci95_overlap.lower/upper`; source value is `eager/score.json.read_gates.gate5.overlap_ci95.lower/upper`, the U–R8 Gate 5 interval, not a U–G interval |
| 178.5 count, words, interval | `eager/score.json.predictions.178.5.estimate.count`, `.words`, `.count_ci95.lower`, `.count_ci95.upper`; rule and verdict at `.rule` and `.verdict` |
| Imported candidate count 17 | `eager/readme_numbers.json.definitions.imported_list_count`, imported from `part178/words178.py` |
| 178.6 pooled mean, interval, eligible traces | `eager/score.json.predictions.178.6.estimate.point`, `.ci95.lower`, `.ci95.upper`, `.eligible_traces`; threshold at `.rule` |
| 178.7 count, interval, missing-ratio count | `eager/score.json.predictions.178.7.estimate.count_under_2`, `.under_2_ci95.lower`, `.under_2_ci95.upper`, `.without_ratio`; rule and verdict at `.rule` and `.verdict` |
| W1 one-sided count and interval | `eager/score.json.predictions.178.W1.estimate.different_top20_entries`, `.rate_top20`, `.trace_top20`, `.ci95.lower`, `.ci95.upper`; scorer convention in `part178/score178.py` |
| W1 symmetric-difference count of 4 | `eager/readme_numbers.json.w1_top20_symmetric_difference.count`; rate-only and trace-only lists derive from `eager/score.json.predictions.178.W1.estimate.rate_top20` and `.trace_top20` |
| W2 difference, interval, paired-label count | `eager/score.json.predictions.178.W2.estimate.difference_tokens`, `.ci95.lower`, `.ci95.upper`, `.n_both_labels` |
| W3a ratio, interval, control opener count | `eager/score.json.predictions.178.W3a.estimate.wait_ratio`, `.ci95.lower`, `.ci95.upper`, `.control_openers`; rule and verdict at `.rule` and `.verdict` |
| W3b ratio, interval, control opener count | `eager/score.json.predictions.178.W3b.estimate.wait_ratio`, `.ci95.lower`, `.ci95.upper`, `.control_openers`; rule and verdict at `.rule` and `.verdict` |
| Seven numbered and 11 total prediction counts; verdict agreement | Keys and `.verdict` fields under `eager/score.json.predictions`; compared with the same paths under `score.json` for SDPA |
| Generated-set, numbered prediction, prediction, gate, and held W3 model counts | `eager/readme_numbers.json.summary_counts.generated_set_count`, `.numbered_prediction_count`, `.prediction_count`, `.gate_count`, `.w3_held_model_count` |
| Part 179 follow-on | `eager/readme_numbers.json.registered_context.next_part`; the follow-on rule is in `../../PART178-DESIGN.md`, “Reading rules” |

## Tables and method values

| README content | Source key or field |
| --- | --- |
| Eager window rates and trace spread for R8, N8, G, N4 | `eager/score.json.tables.window.<set>.rates[word=wait]` and `.trace_rates[word=wait]`; copied to `eager/readme_numbers.json.window_wait.<set>.rate` and `.trace_spread` |
| U and G uncut wait rates and ranks | `eager/readme_numbers.json.uncut_wait.U` and `.G`; copied from `eager/score.json.tables.uncut.<set>.rates[word=wait].rate_per_1000` and `.ranking` |
| Row counts and cap/end-of-turn counts | `eager/readme_numbers.json.generated_sets.<set>.row_count` and `.finish_reason_counts` |
| Think-close row counts and token IDs | `eager/readme_numbers.json.generated_sets.R8.think_close` and `.G.think_close`; counted by checking each row's `think_close_token_id` in `generated_ids` |
| Generated model, revision, attention, mode, cap, and sampler settings | `eager/readme_numbers.json.generated_sets.<set>.model`, `.revision`, `.attn_implementation`, `.thinking`, `.max_new_tokens`, `.temperature`, `.top_k`, `.top_p`, `.presence_penalty` |
| Batch-size statement | `eager/readme_numbers.json.batch_size_recording.recorded_in_rows` |
| Full-text finish-reason sensitivity table | `eager/readme_numbers.json.capped_sensitivity_full_text.sets.<set>.<finish_reason>.wait_opener_count`, `.opener_count`, `.wait_rate_per_1000_openers`, `.trace_count`; opener extraction uses `part178/words178.py:opener_spans` |
| 0.8B capped-only sensitivity ratio | `eager/readme_numbers.json.capped_sensitivity_full_text.thinking_to_control_wait_rate_ratios["0.8B"].by_finish_reason.cap.wait_rate_ratio` |
| 4B capped-only sensitivity ratio | `eager/readme_numbers.json.capped_sensitivity_full_text.thinking_to_control_wait_rate_ratios["4B"].by_finish_reason.cap.wait_rate_ratio` |
| All-row full-text sensitivity ratios | `eager/readme_numbers.json.capped_sensitivity_full_text.thinking_to_control_wait_rate_ratios["0.8B"].by_finish_reason.all.wait_rate_ratio` and `["4B"].by_finish_reason.all.wait_rate_ratio` |
| Pooled and onset-side follow-on estimates | `eager/score.json.predictions.178.6.estimate.point`, `.ci95`, `.eligible_traces`, `.by_onset_side`; copied to `eager/readme_numbers.json.follow_on_by_onset_side` |
| Withdrawal screen result | `eager/score.json.withdrawal.withdrawn`; rates and ratios at `.reflection_rates_U_G` and `.reflection_mode_ratios_G_N4_R8_N8` |
| Prefix, window, and onset-censor sizes | `eager/readme_numbers.json.definitions.prefix_token_limit`, `.window_token_limit`, `.onset_censor_lines`, imported from `part178/words178.py` |
| Imported list size | `eager/readme_numbers.json.definitions.imported_list_count`, imported from `part178/words178.py` |
| Adequacy totals and gates | `eager/score.json.read_gates.adequacy.by_set`; `read_gates.gate1` through `gate5` |
| Gate 5 Spearman, threshold, and inclusion count | `eager/score.json.read_gates.gate5.spearman`, `.threshold`, `.U_top20_in_R8_top20` |

## Superseded SDPA comparison

| README content | SDPA source key | Eager source key |
| --- | --- | --- |
| G window rate, rank, and rank interval | `score.json.predictions.178.2.estimate` | `eager/score.json.predictions.178.2.estimate` |
| R8, N8, N4 window rates and trace spreads | `score.json.tables.window.<set>.rates[word=wait]` and `.trace_rates[word=wait]` | `eager/score.json.tables.window.<set>.rates[word=wait]` and `.trace_rates[word=wait]` |
| G uncut rate and rank | `score.json.tables.uncut.G.rates[word=wait]` and `.ranking` | `eager/score.json.tables.uncut.G.rates[word=wait]` and `.ranking` |
| 178.7 under-2 count and interval | `score.json.predictions.178.7.estimate.count_under_2`, `.under_2_ci95` | `eager/score.json.predictions.178.7.estimate.count_under_2`, `.under_2_ci95` |
| W3a and W3b ratios, intervals, and control counts | `score.json.predictions.178.W3a.estimate` and `.178.W3b.estimate` | `eager/score.json.predictions.178.W3a.estimate` and `.178.W3b.estimate` |
| Gate 5 Spearman and U–R8 overlap interval | `score.json.read_gates.gate5.spearman`, `.overlap_ci95` | `eager/score.json.read_gates.gate5.spearman`, `.overlap_ci95` |
| Verdict comparison | `score.json.predictions.<id>.verdict` | `eager/score.json.predictions.<id>.verdict` |

The eager device rationale is stated qualitatively from `../../PART178-DESIGN.md`, Amendment 1. The README does not treat the 178.4 `ci95_overlap` field as a U–G interval.
