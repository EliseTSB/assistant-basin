# Fold r2 — Part 119b code review

- F1: Moved pilot Gate 2 helpers into dependency-free `gate119b.py`, removed the runner's scorer import, and updated both copy lists; an isolated exact-copy `--help` test covers the launch tree.
- F2: Pinned Gate 2 to `results/part119/interaction.jsonl` with 2,964 rows and SHA-256 `89cee0f0114d464c3405d21982beb753a2eeb977c771562198bd06c370104c56`; timing records the reference path and hash, with a `test_only_` override for tiny fixtures.
- F3: Gate 2 treats an empty selected class as vacuously passing, covered with real-only and control-only selection tests.
- F4: Declined the proposed partial-row repair; rows are written whole and the reader drops a cut final line, so a shorter stored row cannot result from a kill and the existing raise remains.
- F5: Replaced the failing-empty-reference pilot test with a matching-reference test that returns successfully and a disagreeing-reference mirror that asserts the position-list failure; removed the no-op `_push` shim.
- F6: Registered-inventory detection now compares unit IDs with `rows119.load_units()` instead of trusting the unit count.
- F7: Documented that Part 119's interaction file has no `resolved` field and Gate 2 compares all common positions.
