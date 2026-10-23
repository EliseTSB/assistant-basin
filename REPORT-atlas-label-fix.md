# Atlas label fix report

## SUMMARY
The v2 prompt now marks its frequency table partial when more than 12 distinct marked tokens occur, and reports the total type count. `parse_label` optionally validates its fit denominator; the collection caller supplies the actual number of windows. The read-only counter found that 3,766 of 4,504 finished labels (83.6%) were associated with more than 12 distinct marked tokens in their input windows.

## FINDINGS

- `user_message` retains its 12 most common token types when there are more than 12, and adds `marked_token_counts_partial: true` plus `marked_token_types: N`. With 12 or fewer types, it sends the complete table with `marked_token_counts_partial: false`. Other existing user message fields retain their relative order. The system prompt directs the model to use `marked_tokens` for fit when the count table is partial.
- `parse_label(text, n_windows=None)` rejects a fit denominator that differs from `n_windows`; callers using the old two-argument form retain prior behavior. Collection identities carry the count returned by the same `extract_windows` used to build the request.
- The specified unittest command passed: 5 tests, OK. Added coverage checks complete 12-type and explicitly partial 13-type tables, plus fit `3/10` rejected for 8 windows, accepted for 10, and accepted when `n_windows` is omitted. The partial-flag assertion fails if the old unmarked `most_common(12)` behavior returns.
- `atlas_label_v2_truncation_count.py` reads finished canonical label outputs and their source records. Grouped by label set, layer, model arm, and record type, its run counted:
  - `label`: 3,766 / 4,004 records truncated (94.1%): neurons 1,799 / 2,004; controls 1,967 / 2,000.
  - `label-stability`: 0 / 500 (0.0%): all source records had only 12 submitted top windows, so truncation was impossible.
  - Total: 3,766 / 4,504 finished records (83.6%). Windows verified: 4,504 / 4,504. No source files were missing.
- The counter prints individual counts per layer and record type; the figures above are the dataset and record-type aggregates. Finished output status is taken from each complete `collection-summary.json`; stability labels are matched to `stability.jsonl`, and primary labels to `pilot.jsonl`.

## BLOCKERS
None.

## UNVERIFIED
No end-to-end Anthropic batch collection was run. The requested test suite passed, and the read-only count was run against the finished local labels. No API calls or relabeling were performed.
