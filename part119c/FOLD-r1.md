1. Imported `rows119b` and added a CPU OLMo2 test that calls `run119c.run()` for both dtypes and all five doses, then scores the emitted rows; the final suite has 16 passing tests, and six focused tests were red on a disposable `HEAD` copy before the edits.
2. Bootstrap draws now recompute the difference of the two median row shares; 119c.1 and headline branch 2 use that statistic and its product-weighted interval.
3. Output rows record loaded parameter dtype; Gate 2 requires bf16 and float32 evidence and fails when every matched 0.8 nPair value is identical.
4. Kept dtype outputs isolated and rejected resume rows without matching dtype evidence; a restart test deletes only float32 output and confirms fresh float32 support with bf16 output retained.
5. The exact `--copy` tree now runs the copied runner and scorer in a two-unit CPU smoke test; it still does not reach the registered 1B checkpoint or A770 device.
6. Product-weighted and sum-weighted intervals now both bootstrap the same difference of median row shares.
7. Changed 119c.3 to `log(S(0.2)/S(0.05)) / log(4)` in the scorer and design.
8. Failed-gate markdown keeps the per-dose table and labels its heading: no reading is made from the table.
9. Confirmed the scorer has no `resolved` computation to remove; unresolved positions remain included under the registered nPair-positive share rule.
10. Per-dose summaries now include median nI, median nPair, and the count/share of positions with nPair below 1.0.
11. Copy-list subprocesses use `sys.executable` instead of a fixed virtualenv path.
