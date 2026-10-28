# Part 85 fold 3 implementation notes

## Verification

Direct suite: `Ran 34 tests in 6.987s` / `OK` (zero skipped), using `/home/elisepc/neuron-probe/.venv/bin/python -I -B` and one-thread OpenMP/BLAS settings.

Final mutation gate (`mutate85.py --jobs 2`, 137.38 seconds wall time): **Mutation summary: 25/25 killed; 0 survived or errored**. Its baseline also ran 34 tests with zero skipped. Every mutant below was killed:

| Mutant | Result |
|---|---|
| F2_M1_own_max_disabled | KILLED |
| F2_M2_chat_prompt_raises | KILLED |
| F2_M3_clamp_hook_noop | KILLED |
| F2_M4_wrong_answer_unit | KILLED |
| F2_M5_resume_regenerates_cell | KILLED |
| F2_M6_step1_guard_disabled | KILLED |
| F2_M7_cell_push_disabled | KILLED |
| F2_M8_preflight_skips_missing_refs | KILLED |
| F2_M9_assemble_accepts_foreign_tree | KILLED |
| R3_layer9_clamp | KILLED |
| R3_chat_skips_chat_template | KILLED |
| R3_custom_template_ignored | KILLED |
| R3_push_before_answer_write | KILLED |
| R3_step0_push_before_write | KILLED |
| R3_wrong_checkpoint_in_push | KILLED |
| R3_template_mismatch_accepted | KILLED |
| R3_own_max_is_9_75 | KILLED |
| R3_chat_seed_plus_1000 | KILLED |
| R3_own_max_absolute_swapped | KILLED |
| D1_push_stages_outside_directory | KILLED |
| D2_push_succeeds_when_remote_behind | KILLED |
| D3_restore_does_not_checkout_tip | KILLED |
| D4_assemble_accepts_foreign_file | KILLED |
| D5_step1_accepts_wrong_token_sha | KILLED |
| D6_cleanup_removes_snapshot_link | KILLED |

The runner applies exact single-occurrence replacements in isolated copies, invokes the full suite with `-I -B`, refuses skipped/changed-count runs, and fails if any mutant survives. It caps native math-library threads to keep the complete gate below ten minutes on the shared host.

## Haiku round-3 findings

1. **Shared `run.json` collisions — fixed by redesign.** Pilot attempts and main checkpoints now have separate owned directories and branches; a session writes only its branch's directory.
2. **Restored sessions commit from a stale base — fixed.** `restore` fetches and checks out the session branch tip before writes. The rebase and record-merge path was removed.
3. **Attempt-2 template/reference missing — fixed.** Attempt 2 owns its template under `pilot-a2`; reference pinning copies it when used, and main preflight verifies every manifest entry by SHA-256.
4. **A later retry could silently skip a failed push — fixed.** Push checks local `HEAD` against the live remote tip even when no files are staged, retries with bounded backoff and force-with-lease, and succeeds only after verifying the remote tip equals `HEAD`.
5. **Tests did not observe `run()` defects — fixed.** The suite observes all layers' `down_proj`, counts chat-template calls, checks custom attempt-2 prompt text, validates real local-bare-remote state after cell pushes, and uses distinct checkpoint maxima. All fold-2 and round-3 mutants are killed by the final gate.
6. **Main template hashes were unchecked — fixed.** Main scoring verifies answer hashes against the pinned pilot score and, for a custom frame, verifies the pinned template file.
7. **Snapshot cleanup removed only a link — fixed.** The runbook removes the Hugging Face cache directory `models--<org>--<name>`.
8. **Git identity could prevent commits — fixed.** Push installs the required local identity when either name or email is absent.
9. **Circular code/reference pin — fixed.** Pilot clones the code-only commit; after scoring, the references commit pins manifest, files, score, and optional attempt-2 template; main clones that later commit.
10. **Neutral-token digest was instruction-only — fixed.** Step 1 checks the required digest before model loading; a test override remains available for the tiny fixture. The pinned token artifact is checked again by preflight.
11. **Assembly accepted missing answers — fixed.** Every completed-cell claim must name an existing answer file inside that branch's owned directory.
12. **Two sessions could collide on shared records — fixed by single-writer ownership.** The branch/directory mapping rejects mismatches and foreign paths.
13. **Step resume/provenance and test bytecode gaps — fixed.** Steps 0/1 skip only matching provenance and retry push; result files record the code commit. Test subprocesses use `-B`.

## DeepSeek round-3 findings

- **F1: no-op push after a previous failure — fixed.** A no-staged-change invocation still queries and pushes a local commit that is ahead of origin.
- **F2: failed per-cell push stranded a completed cell — fixed.** The all-complete resume path retries the push.
- **F3: main preflight skipped checkpoint step identity — fixed.** If a main step file exists, preflight checks its key, registered checkpoint commit, and code-commit provenance; absence is allowed before that step has run.
- **F4: first Instruct pilot preflight has no prerequisite files — intentionally unchanged.** Instruct is the first pilot checkpoint, so there are no earlier step files to require.
- **F5: manifest hashes its source tree — trust is anchored by the documented Git workflow.** The researcher commits the manifest and all copied references before issuing the pinned references commit used by main. A trusted reviewer/commit remains the integrity boundary for detecting a tree deliberately altered before manifest creation.
- **F6: attempt-2 template was not restored/pinned — fixed.** The owned pilot-a2 template is copied into `part85/references/`, listed by hash, and checked during main preflight and scoring.
- **F7: commits lacked incident breadcrumbs — fixed.** Each result commit subject includes a unique nanosecond timestamp; a regression assertion checks that two pushes have distinct subjects.

## Limits

- No OLMo checkpoint was downloaded, and no real model inference was run.
- The live pinned WikiText dataset/tokenizer load was not repeated; token behavior and the pinned digest were tested with fixtures.
- Push, restore, and assembly were exercised against temporary local bare remotes. Production remote authentication, network failures, and the cloud provider were not exercised.
- No simultaneous push race was run. The redesign assigns one session per branch, and push uses a branch lease plus post-push tip verification.
- Parent-control mailbox was read repeatedly and returned no messages; there are no dispositions to record.

## Claude, on collection (2026-10-28 00:5x CT)

Host rerun: Ran 34 tests, OK. Five wiring mutations of Claude's own (in scratch copies): same seed for every sample,
half the token budget, and the clamp value doubled were KILLED; temperature 1.0 and the first question asked three
times SURVIVED. Same class as Part 86's round 3, so the same remedy (decision part85-handoff-after-r3 option 1 is kept;
the trace is added in fold 4): part85/audit85.py, written by Claude from the design, checks a per-answer trace the run
records from what it actually passed to generate and what the hook actually wrote.
