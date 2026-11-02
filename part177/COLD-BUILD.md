# Cold build read of Part 177 before registration

Scope: `part177/` in `/home/elisepc/green/worktrees/ab-part67` (DRAFT-DESIGN.md, read177.py, score177.py, reader177.py, fetch177.py, loop_upstream.py, known177.json, test_part177.py), checked against the upstream copy in `Journal/notes/reports/part177-review/upstream/`. Every check ran on a scratch copy under this run directory and used synthetic inputs only. No network, no paid API, fetch177.py was not run, and no MATH-500 data or stored completion was opened.

## Verdict

The code implements most of the registered definitions correctly. The loop rule is a verbatim copy of the upstream function, the donor exclusions behave as designed, unknown labels stay as bounds, and the W1–W3 claims and falsifiers are mutually exclusive. The reader is resumable, sends no gold in its prompt, uses max_tokens 4000, and runs the Gate 4 known set through the same path as real windows. The unmodified test file passes 24 of 24, but that is weak evidence: 24 of 32 injected mutations survive. Four things would let this run past a failed gate or count missing labels as clean: gates 1–3 do not stop the reader or the readouts; the error cap and the Gate 4 error check ignore missing and pending windows; the literal rule matches decimals inside longer numbers; and the reader is given a real-only cue on C-not-T windows. A missing manifested input can also be silently re-pinned, and a local path reaches public output. The design should not be registered until the two HIGH findings and the MEDIUM findings are fixed.

## Findings (most serious first)

**1. [HIGH] Failed Gates 1–3 do not stop the reader or the readouts.** `read177.py:490-517` computes the gates, but `read177.py:659-661` writes `windows.jsonl` unconditionally. `reader177.py:257-272` never reads `read.json` or the gates before labelling. `score177.py:966-980` emits tables and `reported_readouts` regardless; only the prediction verdicts consult `_status_from_gates` (`score177.py:219-226`).
Ran: corrupting `math500.acc` to 0.99 gave gate1 `passed=False`, yet `windows.jsonl` was written, `reader177.run(out, "ds", label_one_fn=fake)` made 219 reader calls, and `reader-metadata.json` had `stopped: null`. `score()` afterwards still reported `first_read_hit_asserted` numerator 198 and the tables, while every prediction was "not read".
Fix: have `reader177.run()` load `read.json` and return a stopped record without labelling unless gates 1–3 pass. Have `compute()` mark or withhold `windows.jsonl` on gate failure. Have `score()` omit `reported_readouts` and `tables` unless the gates pass, and otherwise emit a blocked marker.

**2. [HIGH] Missing and pending windows pass the error cap and Gate 4.** `reader177.py:235-249`: `failed` excludes missing and `pending_dns` rows, and `passed` (line 247) looks only at `failed`. `score177.py:667` counts `ds_pending`, but `error_status` at `score177.py:706-712` ignores it.
Ran: 100 rows all `pending_dns` gave `passed: True`. 100 missing rows gave `passed: True`. In a Gate 4 run with 40 unlabelled C-not-T windows, `_gate4` returned status "passed" with `pending_or_missing: 40` and `reader_errors.status: passed`.
Fix: treat missing or pending as a non-passing state. Compute the cap over errors plus pending plus missing. Make `error_status` "not read" when `ds_pending > 0`. Have `run()` stop with `stopped` set until everything is labelled.

**3. [MEDIUM] fetch silently re-pins a manifested input whose file is missing.** `fetch177.py:76-92`: the manifest record is compared only when the destination exists. When it is missing, the new download overwrites the record. The dataset revision always comes from `main` (`fetch177.py:51-70`, `:117`).
Ran (monkeypatched, no network): manifest revision `rev1`, file absent, fetch resolves `rev2`. Result: no error, manifest revision changed to `rev2`, file written.
Fix: when a previous record exists and the file is absent, require the same revision and URL and check the payload sha against the record, or raise. Pin the dataset revision in the repo instead of resolving `main` at fetch time.

**4. [MEDIUM] Manifest verification checks only the files it lists.** `read177.py:318-331` iterates `manifest["files"]` and never checks that the required inputs (merged, per-problem, 20-row samples, dataset, tokenizer) are present.
Ran: a corrupted `tokenizer.json` with its manifest entry removed passed verification with no error.
Fix: derive the expected relative paths from `_data_paths` (`read177.py:334-349`), require each to be listed with a matching sha, and raise if any is absent.

**5. [MEDIUM] Real windows on C-not-T carry a real-only cue.** `read177.py:535-552`: real windows are always generated. Donor exclusion (`read177.py:252-253`) keeps shuffled values out of the prompt, but real gold is not excluded. `reader177.py:44-50` tells the reader to label "quoted from the problem" when the value comes from the statement.
Ran: of the C-not-T real windows, 1 of 1 had its value in the problem text. By construction, no shuffled window does. The reader can therefore tell real from shuffled on C-not-T, and the label distribution shifts for those windows.
Fix: exclude real windows whose gold is in the prompt from the reader set, mirroring T. Nothing that the verdicts need comes from those windows; W1 uses the literal hit and W2 uses shuffled windows.

**6. [MEDIUM] Literal rule matches decimal and fraction golds inside longer numbers.** `read177.py:127-128`: any needle that is neither a plain integer nor a short alphanumeric token is valid wherever it sits. Design line 32 says a bare integer or short token must stand alone.
Ran: gold `3.5` matched in `13.5` (span 14–17). Gold `1/2` matched in `11/2` (span 4–7).
Fix: apply the same digit-boundary test to any needle matching `^[+-]?\d+(\.\d+)?$`, `^[+-]?\d+/\d+$`, or the `\frac{a}{b}` form, not only integers.

**7. [MEDIUM] A local absolute path reaches public output.** `read177.py:628` writes `input_manifest` as an absolute path into `read.json`. `results/part177/.gitignore` ignores only `inputs/`, so `read.json` is not excluded. `test_part177.py:24` hard-codes `/home/elisepc/green/Journal/...`.
Ran: `compute()` with `verify_hashes=True` wrote the absolute manifest path into `read.json`.
Fix: write the manifest path relative (`inputs-manifest.json`). Read the upstream path for the AST test from an environment variable and skip with a message when it is absent.

**8. [MEDIUM] Verdicts for 177.4, 177.6 and W3 use point estimates with no minimum n.** `score177.py:481` (177.4 uses `[share, share]`), `score177.py:534` (177.6 uses `[diff, diff]`), and `score177.py:640` (W3 uses the point rate). The reading rules (design line 84) say rates use a percentile interval, and none of these verdicts does.
Ran: one pair with an unedited onset and no p50c onset gave 177.6 "held" with `eligible_pairs: 1`.
Fix: add not-read floors (for example fewer than 15 pairs for 177.6, fewer than 20 traces for 177.4 and W3) and judge on the bootstrap lower bound.

**9. [MEDIUM] The integration fixture never reaches the shuffled path, and the bootstrap pairing is untested.** `test_part177.py:48-95`: each target's text contains only its own gold, so no donor value ever matches.
Ran: keys by condition came out real 199, shuffled 0. Dropping the shuffled condition (`read177.py:550-552`) leaves the suite green (mutant M32). The bootstrap test at `test_part177.py:216-229` exercises its own local statistic. Breaking arm pairing in `score177.py:333-336` also leaves the suite green (mutant M13). The unit text at `score177.py:1013` says draws stay attached, but the bootstrap carries only left and right values.
Fix: give some fixture targets a donor value in their text, assert that shuffled keys exist and that draw 1 is used, and test `_paired_assertion_bounds` directly with pairing broken.

**10. [MEDIUM] Most registered rules are not pinned by any test: 24 of 32 mutants survive.** Survivors include: line-rule thresholds (`read177.py` `loop_onset`: 3 repeats, length over 8, 40-line window; M6–M8); gate tolerances (`read177.py:497`, M29); the LaTeX test at `test_part177.py:173-176`, which passes even when `\left` and `\right` are not stripped, because the needle is a substring of the haystack (M5); the one-line fixture at `test_part177.py:154-158`, which uses four lines and never tests a single-line repeat; the yes-to-everything test at `test_part177.py:345-369`, which feeds perfect known labels, so the known-gate failure for that reader is never exercised; reader cap, attempts and `valid()` (`reader177.py:29`, `:32`, `:85`; M19–M21); known-gate minimum 17 (`score177.py:223` area; M17); the known-gate stop (`reader177.py:267-270`; M18); W1 hold (`score177.py:583`; M27); W3 threshold (`score177.py:640`; M26); TOKEN_CAP (`read177.py:25`; M31); onset order direction (`read177.py:176`; M9); bootstrap quantile (M12). The 8 killed mutants were: donor prompt-echo, donor class, integer decimal, null-box branch, box comparison, Gate 1–3 status, Gate 4 margin threshold, fetch hash check.
Fix: pin each registered constant with a test that flips the outcome when the constant changes.

**11. [LOW] `score()` crashes when T is empty.** `score177.py:311-314` returns a dict with no `bootstrap_ci95` key. `score177.py:442` indexes it.
Ran: `target_T_ids: []` gave `KeyError('bootstrap_ci95')`.
Fix: add `"bootstrap_ci95": None` to that empty branch.

**12. [LOW] The box sub-case uses a different last-box rule from upstream.** `read177.py:179-227` takes the last of `\boxed` and `\fbox` in document order and skips unclosed boxes. Upstream `last_boxed` (`mathqa.py`) uses the last `\boxed` by `rfind`, falls back to `\fbox` only if there is none, and returns None when the last `\boxed` is unclosed.
Ran: `\boxed{5} ... \fbox{7}` with gold 5: upstream prediction 5, which is correct, while `earlier_gold_box_last_differs` returns True. `\boxed{7} \boxed{5} tail \boxed{9` with gold 7: upstream prediction None, while the function returns True.
Fix: copy upstream `last_boxed(answer_text(text))` verbatim, as `loop_upstream.py` does, and derive the sub-case from it.

**13. [LOW] W1, W2 and the Gate 4 margin units differ from the design wording.** W2 (`score177.py:41-63`) is per trace, using any of up to three shuffled hits; the design says shuffled windows. W1 (`score177.py:568`) excludes whole traces that have an early real hit, which also drops their shuffled hits; the design says excluding them, meaning the hits. The Gate 4 margin is trace-level (`score177.py:655-675`); the design says share of real hits.
Fix: decide each unit and write it into the design before registration.

**14. [LOW] One known-gate key is contestable and the set is thin.** `known177.json` known-05: "the probability is 1/2" is the answer to the stated question, which fits the design's definition of asserted, but the key says intermediate. The 20 known windows are one or two sentences; real windows are about 800 characters, so Gate 4 says little about long-window judgment.
Fix: relabel known-05 as asserted, or reword it as a working step, and state the limit in the design.

**15. [LOW] Kappa is undefined, not failed, when both readers agree on one class.** `score177.py:100-110` returns None when expected agreement is 1, even if observed agreement is also 1. `score177.py:682` maps None to "failed".
Ran: `_kappa` on all-asserted, fully agreeing labels returned None.
Fix: return 1.0 when observed agreement is 1, and report "not read" rather than "failed" when kappa is undefined.

**16. [LOW] A single-line repeating trace is invisible to the line rule; the fallback is dead code.** `read177.py:159-161` (`if not lines and searched`) cannot fire, because `splitlines` on non-empty text returns at least one line.
Ran: thirty repetitions of one sentence with no newline gave `loop_onset` None, while `loop_signals` gave `verbatim_loop_end` True.
Fix: decide how single-line repeats are handled, or state that they count only through the end rule. Delete the dead branch.

**17. [LOW] The withdrawal rule exists only as prose, and the dataset revision floats.** Nothing in `score()` records whether 177.1 failed, so the "kept going" reading is not blocked in code. `fetch177.py:117` resolves `main` at fetch time, so a fresh fetch can change the dataset.
Fix: add an output field that marks the "kept going" reading blocked when 177.1 fails. Pin the dataset revision in the repo.

Severity count: HIGH 2, MEDIUM 8, LOW 7 (17 findings total)
