# Amendment 2 implementation notes

Gate 1b and the amended 118.1a, 118.2a, and 118.5a readings now use valid `shows_persona` flags regardless of coherence. Registered readings still require both `shows_persona` and `coherent`. The score report places each amended reading next to its original and includes persona-show and coherence counts by alpha, arm, and original kind. Terminal label errors still count as not showing and remain reported. The report renderer now reads the DS sample count from the result field where the scorer stores it.

## New tests and old-code failures

- `test_gate1b_uses_persona_flag_when_concept_answers_are_incoherent`: old Gate 1b keeps 38 units, while this fixture expects all 39 under Part 92's rule.
- `test_1185a_counts_a_concept_unit_that_1185_does_not`: old scorer has no Part 92 unit classifications or 118.5a result.
- `test_1181a_verdict_changes_when_incoherent_slot_answers_count`: old scorer has no 118.1a prediction.
- `test_1182a_verdict_changes_when_incoherent_who_answers_count`: old scorer has no 118.2a prediction.
- `test_persona_coherence_table_counts_small_fixture_by_alpha_arm_and_kind`: old scorer has no persona-coherence table.

## Verification

`test118.py`: 13 passed. `test_part118.py` discovery shim: 8 passed. The five new tests were run with the committed `score118.py` copied from `git show HEAD:part118/score118.py` into `/tmp/part118-old.3Qoj4t`; all five failed on the old scorer for the reasons above. `git diff --check` passed. All fixtures were synthetic; no Part 118 label files were read.
