# Part 85 fold 2 implementation notes

## Finding-by-finding changes

1. **Round-2 #1, cloud handoff.** Added `handoff85.py` with `restore`, `preflight`, `refs-manifest`, `push`, and `assemble`. Pilot preflight checks the Instruct step files and shared token IDs before `pretrain_4T`; main preflight verifies every manifest SHA and the four required reference artifacts. The manifest includes the custom template when the gate records one. Push stages only the selected checkpoint directory and Instruct's shared token IDs. A rejected push fetches, rebases, merges disjoint `run.json` cell records, and retries up to three times. Restore accepts a missing branch. Assemble checks branch ownership and byte conflicts. Step 0, step 1, and step 2 accept `--push-branch`; each pushes immediately after its result is written. `CLOUD-RUN.md` now covers pinned clone, restore, preflight-before-snapshot, the two-checkpoint pilot, checkpoint cleanup, main sessions, blind scoring, reference pinning, and final assembly.
2. **Round-2 #2, redesigned run smoke test.** The integration test now drives `part85.main(argv)` through pilot, plain main, chat main, resume, changed-step refusal, and chat-only scope. Assertions read answer files and `run.json`; expected values and seed formulas are written in the test. An observer forward pre-hook is registered after the clamp hook and records prompt and decode inputs. The existing hook test independently verifies that this later pre-hook receives the earlier hook's modified arguments. No test expectation calls `_cell_value` or another Part 85 helper.
3. **Round-2 #3, chat-only score scope.** Chat-only main scoring returns `NOT APPLICABLE` for 85.5–85.9 and requires step files only for SFT, DPO, and Instruct. The out-of-scope 85.1–85.4 measures remain `NOT READ` when their inputs are not present; they no longer abort chat-only scoring.
4. **Round-2 #4, step-1 validation order.** Non-Instruct step 1 validates the Instruct reference, token-ID file shape, and both token SHA-256 values before loading the model. A regression test makes model loading raise if the token-hash refusal happens too late.
5. **Round-2 #5.** The review recorded the 16 unique first-token check as already complete. No registered prediction or design decision changed.
6. **Round-2 #6, resume guard.** `run.json` now hashes only neutral activation maxima, neutral token-ID SHA-256, and checkpoint commit. Device and library metadata do not affect the guard.
7. **Round-2 #7, snapshot provenance.** HF cache paths must resolve to a directory named by the full commit SHA. A plain/copied snapshot must have `config.json` with matching `_commit_hash`. Paths lacking either provenance signal are refused. The rule is documented in `CLOUD-RUN.md`.
8. **Round-2 #8, coverage.** Added tests for `analyze_snapshot`, SnapshotTensorReader row/column/row-batch reads, neutral-sequence construction, token-hash refusal, and the pilot amend/chat-only/quality-failed branches. Step-file identity, seed/pair validation, snapshot provenance, chat-only scoring, score overwrite refusal, and push behavior also have coverage.
9. **Low findings.** Scoring verifies step-file checkpoint keys and registered commit prefixes, validates complete unique question/sample pairs and the registered seed formula, and refuses score-output overwrite unless `--force` is explicit. Attempt 2 uses separate item/key/label/score paths and preserves the accepted gate at the canonical path. Removed the answer-key length self-comparison and replaced it with metadata-shape validation. Step 0 now honors `--results-root`. Sharded `.bin` remains unsupported as allowed by the review; local `.bin` checkpoint reading remains supported.

## Mutation table

Each mutation was applied to a throwaway `/tmp/part85-mutation-final/<Mx>/part85` copy. Each focused test was run with `/home/elisepc/neuron-probe/.venv/bin/python -I` and exited nonzero:

| Mutation | Test that failed | Observed failure |
|---|---|---|
| M1: `run()` sets unit/value to `None` | `test_run_entrypoint_disk_contract_resume_chat_and_changed_step1` | `AssertionError: None != 6978` in the own-max answers row. |
| M2: chat prompt builder raises | `test_run_entrypoint_disk_contract_resume_chat_and_changed_step1` | `RuntimeError: mutated chat prompt builder`. |
| M3: clamp pre-hook returns args unchanged | `test_run_entrypoint_disk_contract_resume_chat_and_changed_step1` | Observer assertion failed; unit 6978 prompt/decode values remained around 0.01–0.04 instead of 9.75. |
| M4: answer row writes a wrong unit | `test_run_entrypoint_disk_contract_resume_chat_and_changed_step1` | `AssertionError: -1 != None` on the unclamped row. |
| M5: resume regenerates a completed cell | `test_run_entrypoint_disk_contract_resume_chat_and_changed_step1` | `AssertionError: True is not false: resume regenerated the already-complete unclamped cell`. |
| M6: changed-step guard is disabled | `test_run_entrypoint_disk_contract_resume_chat_and_changed_step1` | `AssertionError: ValueError not raised` after the maximum changed with answer files removed. |
| M7: per-cell push is disabled | `test_run_entrypoint_disk_contract_resume_chat_and_changed_step1` | `AssertionError: 0 != 4: push must run after each pilot answer cell`. |
| M8: preflight skips missing references | `test_handoff_bare_remote_push_rebase_restore_preflight_and_assemble` | `AssertionError: 0 == 0`; preflight returned success when Instruct step 0 was missing. |
| M9: assemble ignores branch ownership | `test_handoff_bare_remote_push_rebase_restore_preflight_and_assemble` | `AssertionError: 0 == 0`; assemble returned success for an Instruct branch containing SFT output. |

## Open design questions and limits

No unresolved design decision was needed, and no registered cells, doses, seeds, questions, frames, gate rules, or predictions were changed. The conservative snapshot rule rejects a plain directory unless it has matching `_commit_hash`; use the HF cache `snapshots/<commit>` path otherwise. `pilot-frame-v2.txt` is the fixed custom-template filename expected by the reference manifest.

The worker venv had Torch and the experiment packages used by the tests, but did not have `datasets`. `datasets==5.0.1` is pinned in `CLOUD-RUN.md` from the official Hugging Face release list, and the test verifies the required `load_dataset(..., revision=...)` call through a stub. A real install and network-backed WikiText load, full OLMo snapshot downloads, and the actual cloud run were not verified here; `CLOUD-RUN.md` includes a live dataset-load check before snapshot download.

Parent-control pulls returned no messages. There were no dispositions to record.


## Claude, on collection (2026-10-27 23:2x CT)

Host rerun: 28 tests, OK, none skipped. The `datasets==5.0.1` pin was not installable beside `huggingface_hub==2.2.0` (it requires hub <2.0), which would have stopped every cloud session at install; changed to `datasets==5.1.0`, and the real pinned WikiText load was run on the host with these pins: shape (200, 128), sha256 96699282f18fc53d5b5467a51803241aab479e23523b92980bdb83546ae15d87, now in CLOUD-RUN.md as the value the Instruct session must reproduce.
