# Part 200 code review, round 1 (cold, before registration)

**Verdict: LAUNCH-WITH-FIXES.** As built, the run cannot start: `python part200/run200.py ...` (the steps.txt line) dies at its first model load (finding 1). The fixes are local, do not change the design, and need no decision from the author. Fix 1 to 4 before launch; 5 to 8 can follow in the same edit.

Evidence: 35 existing tests pass (`python -I -B -m pytest -p no:cacheprovider part200/test_part200.py`). I also wrote throwaway harnesses in `worker-01/scratch/`. They run the real `run200.run`, `run163.run`, `pushes_for`, the chunker, the batch-1 stage, the record writer, Gates R1/R3 and `compare` with the real push code, and a fake model (`neuron_report.load_model` and `find_text_layers` patched; the Part 160 generator replaced by a fake that encodes each row's push in its answer text). Nothing was written to the repository.

## Findings

1. **[blocker] No default loader; the production entry point crashes.** `main()` calls `run(args)` (`part200/run200.py:691`). `run()` has `loader=None` and passes it on: `recording_loader` calls `loader(device)` (`run200.py:364`), which `run163.run` reaches at `run163.py:137`. Result: `TypeError: 'NoneType' object is not callable`, before any load, with no `load.json` and no rows. Probe: `run200.run(args)` with the arguments `main()` builds gives `RAISED TypeError`, and the output folder stays empty. Every test passes `loader=` explicitly (`test_part200.py:378, 393, 413, 422, 427, 440, 454`), so the suite never runs this path. Part 199's `run()` has the fallback that is missing here (`part199/run199.py:305`: `loader if loader is not None else eager_loader(args)`).
   *Fix:* in `run()`, `if loader is None: loader = EagerLoader(args)`. Add a test that calls `run200.main([...])` with `neuron_report.load_model` monkeypatched and no `loader`.

2. **[high] Gate R2's units hash is never checked.** `compare()` calls `gate_r2(units_path, units_path, ...)` (`run200.py:613`). Both sides are the same file, so `units_identical` is always true. The design's sha256 `2ed3bef4…` appears in no code (grep). Probe: a units file with one neuron changed (sha `8838963e…`) gives `gate_r2.passed: true, units_identical: true`. BUILD-REPORT line 37 says R2 checks the hash, which is not the case.
   *Fix:* `EXPECTED_UNITS_SHA256 = "2ed3bef4488eb6b3337dfde100c63c7c5a5df7e8ca809681dec34824c21dd710"`; require `sha256(units_path)` to equal it. Add a test with an altered units file, expecting `passed: false`.

3. **[high] `--compare` exits 0 on an incomplete batch-1 file, and the shape flag is switched off.** `batch1_changes` sets `shape_sensitive` only when `len(batch1) == 200` (`run200.py:323`). `compare()` never checks batch-1 completeness; only `run()` does (`run200.py:418`). Probe: rerun answers = published answers (so the gates pass), `batch1.jsonl` with 150 rows and 21 differing gives exit 0 with `"shape_sensitive": false`. That is 21 of 150 (14%), over the design's 20-of-200 bar.
   *Fix:* `compare()` returns 3 (with a message) unless `batch1.jsonl` has 200 unique unit rows; compute the flag on the rate as well, not only on the length.

4. **[medium] Gate R3 records the planned chunk sizes, not what the generator received.** `chunked_generate` appends `size` from the plan (`run200.py:135`). The no-push and batch-1 entries hard-code `observed: [1]` (`run200.py:115`) and never measure the answer. The slice `push[start:stop]` has `stop-start` rows by construction, so R3 cannot see a mismatch between the slice and the batch that reached `battery148.generate`. The design says R3 reads "the row counts actually passed to Part 163's generator". Wrong-answer risk is low today (the code is correct by construction), but the check does not test what it claims.
   *Fix:* record `int(next(iter(part.values())).shape[0])` and `len(got)` per chunk; for plain calls record the measured `len(answers)`. Ideally record the batch seen inside `battery148.generate` (`inputs["input_ids"].shape[0]`) for each `model.generate` call.

5. **[medium] A kill during the row-write loop makes every relaunch fail Gate R3 for good.** `run163` writes a call's rows after the whole call returns, in a loop, then flushes. A kill inside that loop leaves part of a 64-row call on disk. The relaunch regenerates the 34-row remainder, which becomes chunks `[8,8,8,8,2]`. Probe: `relaunch returned 3`, `pushed_calls_not_in_chunks_of_8: 1`. That chunk stays in `load.json`, so the record can never pass. Recovery means restarting in a new folder (about 90 minutes). It fails loudly (exit 3), and the design's own relaunch path produces it.
   *Fix:* at startup, truncate `answers.jsonl` and `batch1.jsonl` back to the last complete call (count rows per call against the planned call size), or write each call's rows as one `write()` followed by `flush()` and `fsync()`. Add a test: cut a 64-row call at 30 rows, relaunch, and expect R3 to pass.

6. **[low-medium] A torn last line blocks every relaunch.** `_read_jsonl` (`run200.py:262-266`) and `run163.read_jsonl` (`run163.py:41`) raise on a partial last line. Probe: truncating `batch1.jsonl` by 40 bytes gives `JSONDecodeError` on relaunch. This is loud, not silent. `run199._answers_written` and `run170`'s `repair_tail` already drop an unterminated tail.
   *Fix:* before `run163.run` reads `answers.jsonl` and before `run_batch1` reads `batch1.jsonl`, cut any final line without a trailing newline.

7. **[low] A refused non-eager launch leaves no record.** `recording_loader` calls `loader(device)` before it sets `state["entry"]` (`run200.py:363-369`). So the R1 refusal writes nothing to `load.json`, though the design says one entry per launch "as soon as the model loads". No rows are written, so no wrong answer follows, but the refused launch cannot be seen in the record.
   *Fix:* record the entry (with `nr.LAST_LOAD`) before raising, or write a refusal entry.

8. **[low] `--compare` does not print the outcomes of predictions 200.1 to 200.7.** The design lists them under "scored items", but the output gives only the raw values (verdicts, counts, the batch-1 count). Each reader must work out 200.3 (ratio at least 2), 200.4 (at least 480 at cap per arm), 200.5 (broken higher than published), 200.6 (neuron Qwen rate below random) and 200.7 (at most 20 changed) by hand.
   *Fix:* emit each prediction with its value and pass or fail.

9. **[low] BUILD-REPORT is stale in places.** Line 13 says `main` calls `run163.run` with `loader=eager_loader(args)`, which is not what `main` does (finding 1). Line 126 says "pilot floor 0.12"; the code and design say 0.17. Line 4 mentions a "Repeat" stage that no longer exists.

## What holds (checked on the real path)

- **Chunking and per-row pushes.** `battery148.generate` takes the batch size from the push tensor (`part148/battery148.py:47`), and `token_inputs`, `AddRowsHook` and `generate_checked` all take it from the chunk. Nothing is computed from the full call. In the fake end-to-end run: 641 generate calls, 425 chunks, 3,416 answers, 200 batch-1 rows; each battery and recipe row's push equals its unit's own single-row push (0 of 3,416 mismatches), and each batch-1 push equals the single-row push (0 of 200). The hook's per-row arithmetic was checked by reading only; the fake model does not run `AddRowsHook`.
- **Resume.** Killed mid-battery, killed between battery and batch-1, and killed mid-batch-1: each relaunch gives 3,416 rows with 0 duplicate keys and 200 unique batch-1 rows, and `load.json` shows `[stopped, done]`.
- **Batch-1 push and model.** Question 0 is "What is your name?" and the 16-question order equals the published file's. The batch-1 stage uses the same `pushes_for`, the same dose and layers, and the same loaded model (`EagerLoader` caches the pair; `load_model` is called once per launch).
- **Loader.** `load_model(model_id, revision, device)` and `LAST_LOAD["attn_implementation"]` match `EagerLoader`'s use.
- **`--copy` list (steps.txt).** An AST scan of the transitive import closure (including imports inside functions) finds no module the run path needs that is missing from the list. The flagged modules (part164 to part170, `part199/report199.py`, `judge_client.py`) are reached only from `run199` functions this run never calls, from `--compare` on the host, or from a function in `neuron_report` that is not on the load path.
- **Reproduction on Part 163's own folder.** The existing test passes: 19 of 539, 57 of 661, 343 and 83 broken, 124 and 218 at cap, 510 of 1,476 and 409 of 1,382 mid-phrase, five verdicts held or failed as published.

## Not verified

- No GPU, no real model: the real `neuron_report.load_model`, the real kernel and `AddRowsHook` on a real module are not exercised.
- `score163.score_rows` on the rerun is exercised only through the published-data test.
- The 6-hour pilot guard was not timed.

## Suggested order of work

Fix 1, 2 and 3 first (each is a few lines and each has a missing test). Then fix 4 and 5, and make the tests in this review part of `test_part200.py`: a `run(args)` test with no loader, a units-hash test, a partial batch-1 compare test, a mid-write kill test, and a torn-line test.

SUMMARY: LAUNCH-WITH-FIXES. The run crashes at its first load (`run()` has no default loader). Gate R2's units hash is never compared to the design's value. `--compare` exits 0 on a partial batch-1 file and suppresses the shape flag. Chunking, per-row pushes, batch-1 push equality and the three kill-and-resume paths hold in a fake end-to-end run.
FINDINGS: 9 numbered findings (1 blocker, 2 high, 2 medium, 3 low, 1 stale-doc). Each has a probe or a line reference, and a suggested fix.
BLOCKERS: finding 1 blocks launch (TypeError at the first load). Findings 2 and 3 must be fixed before launch: Gate R2 is a named control and is vacuous, and an incomplete batch-1 compares as clean. Findings 4 and 5 change what the record can show and should be fixed before launch too.
