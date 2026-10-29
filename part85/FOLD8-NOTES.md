# Part 85 fold 8

## Changes

- `registered_repos()` now maps the first nine checkpoints to `allenai/OLMo-2-0425-1B` and the final three to the production `-SFT`, `-DPO`, and `-Instruct` repository IDs. Replay fixtures now carry production-shaped IDs, token counts, and code commits.
- Replay requires exact full commit equality for answer rows and observer sessions. It compares `new_tokens` with the regenerated cut length, optionally compares every row's `code_commit` with `--code-commit FILE`, rejects non-integer saved token IDs, requires one observer output and raw row per used call, and rejects duplicate session IDs.
- Added `provenance85.py`. It reads the committed design, references manifest, and pinned reference files with `git show`; validates every local snapshot file against Hugging Face LFS SHA-256 or Git blob SHA-1 metadata; records the full checkpoint commit and snapshot SHA-256s; and records step-1 JSON/JSONL hashes with the exact `AUDIT85 KEY step1: PASS` line. Replay requires `--provenance FILE` and verifies design, references, snapshot, and step-1 bytes against it.
- `LOCAL-RUN.md` documents the committed pre-pilot references bootstrap, provenance command, and provenance arguments for every replay. The mutation stop count is corrected to 44.

## Verification

- `test_part85.py`: 46 passed.
- `test_download85.py`: 11 passed.
- `test_replay85.py`: 49 passed.
- `mutate85.py --jobs 2`: baseline 46 passed; 44/44 mutants killed.
- `observe.py` remains SHA-256 `f6c1d7e16f4ccbb083e2a7ff8fa99d2f18e08be4504696e25f2c5b9821082613`.

## Round-7 cases

Re-ran the preserved cases with temporary, production-shaped fixtures and the new provenance argument. `production_model_id` now passes. These now fail: `commit_suffix_relabel`, `wrong_step1_identity`, `fractional_token`, `wrong_new_tokens` (including invented `code_commit`), `changed_step1_max`, `changed_snapshot_top_k`, both `caller_design_attempt2` variants, `changed_snapshot_against_original`, `orphan_raw_output`, `step1_changed_by_hooks`, `duplicated_session_header`, `caller_references_missing_plain`, and `missing_plain_against_original_gate_metadata`. The old `r6_baseline` fixture still fails because it retains the branch-label model ID.

`wrong_measurement_counts` still passes. Measurement-count enforcement was not in this fold's requested replay fixes, so it was left unchanged.

## README limitation sentence for Claude

Replay certifies a run against inputs pinned by a separate provenance record; it does not protect against an operator replacing that record.
