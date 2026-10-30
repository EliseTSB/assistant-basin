# Part 120 review fold r1, corrected after r2

- F1: No code change. Round 1 misread the row-specific T handling. The runner reads T by row and position and centers A, B, and P within that row. The tiny-model test with two different T sets confirms that the A vectors differ and each matches a direct recomputation on its own T. Passing an old-copy test alone did not establish this.
- F2: No code change. Round 1 used the wrong Part 119b predictor-window convention. `path_for_first` matches `run119b.run_path2`, including p=0 and short answers. The p=0 test checks that window directly.
- F3: The original fold caught `OSError` only. Round 2 broadens abort recording to every `BaseException`, including `RuntimeError` and `KeyboardInterrupt`, and records the exception type and message. A complete main timing record is preserved if a refused rerun targets the finished result directory.
- F4: NPZ indexes record selected and expected unit scope. Scoring requires a complete main timing record, the full dose ladder, all unit files, and matching index scope. Smoke and partial runs are refused before score files are written.
- F5: Gate 3's registered check is the adjacent-dose identity `push(2 alpha) = 2 * push(alpha)`, within relative error 1e-5 for each unit and arm. The runner also compares its dose caches to calls of `run119b.build_dose_push_cache`, the same builder used by `run119c`; this is a same-code-path consistency check, not an independent rebuild. The original sabotage fixture was removed. Its replacement deliberately perturbs the 0.1 push and confirms that the registered adjacent-dose check fails. The scorer still requires the consistency-comparison fields in a gate record.
- F6: No code change. One Part 119 unit draw supplies counts for both endpoints. Product weights are primary and sum weights are reported as a sensitivity check.
- F7: No code change. The existing branches mean high < 0.8, wholly within [0.8, 1.25], and low > 1.25. Exact boundaries are pinned by `test_headline_all_branches_including_withdrawn`.
- F8: Missing answers raise a path-specific `FileNotFoundError`.
- F9: `score_data` returns an explicit empty-input headline when called with no vectors. The CLI cannot reach that branch for a missing-vector run because `require_complete_run` rejects the missing unit files first. The direct library test does not establish CLI behavior.
- F10: No code change. Malformed non-object gate-3 comparisons resolve to an invalid error and fail closed without an `AttributeError`.

Round 2's specific corrections and the exact post-fold test count are recorded in `FOLD-r2.md`.
