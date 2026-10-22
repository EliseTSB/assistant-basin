# REPORT-54-fold1

## SUMMARY

Applied the Part 54 ladder, blind-reader, test, and runbook fixes that could be completed in this worktree. Confirmed two reviewed ladder/control claims were already correct by measurement. The required `git merge main` could not update Git worktree metadata outside the writable roots; see BLOCKERS.

## FINDINGS

- NOT-A-DEFECT — Ladder output collision: on HEAD, all 21 question/dose steps already produced 21 distinct paths; added a runtime uniqueness assertion and test for the registered question/dose grid.
- FIXED — Twin ladders: `--steps ladder` now dispatches, `ladder` is accepted in the step order, and a named twin is validated against the target candidates with a path-specific missing-artifact error; regression tests failed on HEAD and pass now.
- FIXED — Model pooling: answer rows and submitted keys retain `short_name`/model, collection groups by model, and multi-model tables print the model column; single-model legacy output remains unchanged.
- NOT-A-DEFECT — Manifest control count: plan and tiny-model runs show controls suppressed at non-control doses; the 21-step plan records 480 generations with controls only at −2 and +1.5.
- FIXED — Restart safety: matching incomplete manifests resume from existing files; completed ladders refuse overwrite unless `--force`; the manifest also checks model dtype and input hashes. An interrupted-run test confirms completed steps are skipped.
- FIXED — Blind collection parsing: restored `--by-cond` and punctuation stripping while preserving `--folder`; empty folders now stop before an API request.
- NOT-A-DEFECT — Down-projection width: meta-instantiated modules measured `in_features == intermediate_size`: Qwen3.5-4B 9216, OLMo-2-7B 11008, Qwen3-4B 9728, Phi-4-mini 8192, Granite 3.3-2B 8192. Added the requested code comment.
- NOT-A-DEFECT — Controls at every dose: source-clamp measurements return zero controls at −3 and controls at −2/+1.5; the registered ladder plan has six control cells, not 21.
- NOT-A-DEFECT — Source scale/cache: the existing cache hash includes dtype, preventing cross-dtype reuse; the review probe measured 451 tokens across 30 short passages, below the 512-token limit.
- FIXED — Dtype guidance and return archive: documented that `--dtype` is an explicit override, removed the redundant Qwen flag, and changed the archive to `tar -C /root/part54 ...`; a tar fixture extracts both result folders at the documented paths.
- DECLINED — Seed pairing change, per parent decision: preserved Part 26's distinct per-condition seeds, documented them in clamp metadata, ladder manifests, and the runbook, and tested seed stability across doses/questions.
- FIXED — Test sensitivity: added exact twin-scale, literal dose/question-grid, and unique-output assertions. Deliberately broken twin-scale and dose-axis probes are rejected by the new assertions.
- NOT-A-DEFECT — Legacy Part 26 collect compatibility: aggregation from the committed `msgbatch_01XoGrMFw6eRLiGdAEBJhWxt` key/labels read with `git show main:` is byte-identical to the old single-model table.

## BLOCKERS

`git merge main` failed before changing the worktree because Git could not create `/home/elisepc/assistant-basin/.git/worktrees/ab-54/ORIG_HEAD.lock` (`Read-only file system`). The main-branch punctuation and `--by-cond` behavior were ported manually, but the required merge and any other main-only files were not applied. HOLD.request asks the parent to resolve the Git-metadata access or explicitly accept the manual port.

## FILES TOUCHED

- `part26.py`
- `blind_origin.py`
- `test_part54.py`
- `RUNBOOK-54.md`
- `REPORT-54-fold1.md`

No commit or push was made. `PART54-DESIGN.md`, `review/`, and `results/` were not modified.

## SELF-ASSESSMENT

`test_part54.py` passes 13 tests; `test_part26.py` passes 21 checks; `git diff --check` is clean. The legacy table was verified against committed key/label JSON without network access or file writes. No full model generation or rented-GPU run was performed; tiny Qwen CPU tests and meta-constructed architecture probes cover the local contract. The merge blocker remains the material uncertainty.
