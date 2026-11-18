VERDICT: FIX-FIRST

Round-2 cold review of the Part 59 runner (files under /home/elisepc/assistant-basin/part59/ unless stated). The science core holds: dose scale, control draws, OLMo and Qwen constants, the pooled P99 rule against Amendment 1, and the blind reader's refusal gate are all correct as written. What would waste the rental is the failure handling in run59.sh and the resume path it promises. Those are small fixes; make them before renting. Findings 4-7 can wait until before scoring.

## Findings (ranked)

1. **HIGH: one failed phase cancels every later GPU phase, including Qwen 59B.** `run59.sh:29-36` (`run_phase`) skips any phase once `run_status` is nonzero, and `run59.sh:85-90` puts all six GPU phases behind that gate.
   - Failure scenario: OLMo 59A hits a CUDA OOM or a bad OLMo load after 25 minutes of Qwen 59A. Qwen 59B (the 1-3 hour scan), both analyses, and OLMo 59B are all skipped. The box archives INCOMPLETE and prints `RUN_FAILED failing_phases=olmo59a`. The rental must be repeated for 59B alone.
   - Same effect from `qwen59b-analysis` (line 88): a logit-lens OOM there skips OLMo 59B, even though OLMo's 59B has no dependency on it.
   - Verified: I ran `run59.sh` (copied to /tmp/df59-box, all paths rewritten under /tmp, `python` and `nvidia-smi` stubbed, one stub exits 1 for `run59a.py`). Output: `qwen59a exit=1`, then `olmo59a`, `qwen59b`, `qwen59b-analysis`, `olmo59b`, `olmo59b-analysis` all "skipped because an earlier phase failed", `RUN_FAILED failing_phases=qwen59a`, no `RETURN_READY`.
   - Smallest fix: in `run_phase`, drop the `run_status` gate and run each phase on its own. Each phase is resumable and records its own rc through `record_status`. If you want a dependency, gate only the phase whose input failed, e.g. `qwen59b` on `qwen59a`, `olmo59b` on `olmo59a`. Also move `prefetch-olmo` (line 79) out of the gate that blocks Qwen, so a 1B-model fetch failure does not cancel the main experiment.

2. **HIGH: the documented resume path dead-ends on the first completed phase.** `run59a.py:229` raises `FileExistsError("59A is already complete ...")` and `run59b.py:365` does the same for 59B. Neither is caught at the CLI entry points (`run59a.py:359-360`, `run59b.py:480-481`), so a completed phase exits 1. The completed-check runs before model load, so it fails in seconds.
   - Failure scenario: Qwen 59A completes, OLMo 59A fails, and the operator reruns `run59.sh` to resume. `qwen59a` exits 1 with "already complete", the gate from finding 1 skips everything, and `RUN_FAILED` is printed again. RUNBOOK-59.md:54 says "an interrupted run can resume"; through this script it cannot.
   - This came in with the round-1 fold ("no-op reruns ... complete-output checks"). The raise is intended for `run()`, and the tests at `test_part59.py:347` and `:420` assert it; the CLI wrapper just never handled it.
   - Smallest fix: in each `__main__`, catch `FileExistsError` whose message contains "already complete", print it, and `sys.exit(0)`. Keep the raise in `run()` so the tests still hold.

3. **HIGH (money): the run is foreground SSH with no teardown step.** `RUNBOOK-59.md:51` runs `bash /root/part59/part59/run59.sh` directly under ssh. A dropped session sends SIGHUP to the script, and the job dies mid-GPU. Nothing in RUNBOOK-59.md tells the operator to destroy the instance after `RETURN_READY` or `RUN_FAILED`. `cloud/vast-part54-2026-10-22/vast.py` has a `destroy` command and the Part 54 README records a destroy, but the Part 59 runbook has no such step, so the instance keeps billing.
   - Smallest fix: start it with `nohup bash ... > /root/part59/run59.out 2>&1 &` (or tmux), and add a final RUNBOOK step: copy the bundle back, then `python3 cloud/vast-part54-2026-10-22/vast.py destroy INSTANCE_ID`. Say it applies on RUN_FAILED too.

4. **MEDIUM: the calibration is weakest on OTHER, where vague 59A prose will land.** `blind_setting.py:71-72` are two OTHER items written as commentary about the paragraph ("the paragraph ends before any detail makes the setting identifiable", "the passage seemed ready to explain..."). A reader that gets every real OTHER right still gets these two right for an easy reason, so they cannot catch a reader drift. `blind_setting.py:62` ("a traveler found a door ... the other side") is contestable and a reasonable reader could say FANTASY. The gate at `blind_setting.py:22` and `:332` allows 2 wrong out of 34, which is 2 of the 7 OTHER items in the worst case.
   - Smallest fix: replace 71-72 with two or three in-voice, vague story openings with no genre cue, and drop or move 62. Do this before `submit`; it does not affect the GPU run.

5. **MEDIUM: Amendment 1's comparison is not in the output.** `analyze59b.py:180-194` writes the symmetric count, the max control count, and `held` for 59f and 59g. It does not write the registered informative comparison ("the target must exceed the largest control count to count as a result") or the note that 59f and 59g can be passed by the null alone. A hand scorer has to compute both, and that is where a scoring slip would happen.
   - Smallest fix: add `"target_exceeds_max_control": symmetric_count > max_control` and the null-caveat string to the `predictions` block.

6. **MEDIUM: the analysis never checks manifest status.** `analyze59b.py:157` loads the manifest and uses only metadata. A partial archive is caught only by the NaN guard at `analyze59b.py:22-23`, so the error is "non-finite values" rather than "run incomplete". Smallest fix: `if manifest.get("status") != "complete": raise ValueError(...)` after line 157.

7. **MEDIUM: test gaps; one test covers the wrong module.** `test_part59.py:485-` (`test_blind_origin_answers_reads_59a_condition_files`) exercises `blind_origin.answers`, which is not the scoring path. Nothing checks:
   - that the `submit` request texts contain only answer text and no condition, dose or model (the blinding guarantee), by patching `blind_setting.call`;
   - the `key` mapping written by `submit`;
   - the `analyze_file` JSON fields (`held`, the 59f/59g claims) on a synthetic archive.
   Smallest fix: add a `submit` test that captures the request body and asserts it equals the shuffled answer texts plus calibration texts, and one `analyze_file` test on a tiny synthetic `.npz`.

8. **LOW (packaging): the tar takes the whole `part59/` directory.** RUNBOOK-59.md's `tar -czf ... part59` ignores `part59/.gitignore`. If `part59/review/` exists when packaging, the local blind key mapping (`setting-key-*.json`) and `__pycache__` ship to the rented box. `review/` is absent now (checked). Smallest fix: list the needed `.py`, `.json` and `.sh` files explicitly, or exclude `review` and `__pycache__`.

9. **LOW: local-machine dependencies in the blind reader.** `blind_setting.py:81` hard-codes `~/.mitmproxy/ca-bundle-combined.crt`, so the reader cannot start on any machine without that bundle. `blind_setting.py:84-92` (`call`) does not handle HTTP errors, so `collect` (`blind_setting.py:312`) on an unfinished batch escapes as a traceback. Papercut-level; fix when touching the file.

10. **LOW: the Qwen control scales differ about 9x, so the pooled P99 is set mostly by two controls.** The cache (`results/part54a/source-pile-maxima-7c19d0120476.json`) gives controls 152, 376, 692 maxima 19.77, 13.51, 2.18. The pooled threshold in `analyze59b.py:42-43` is therefore set mainly by 152 and 376. This is what Amendment 1 registers, not a code defect. State it in the readout next to 59f and 59g.

11. **LOW: an extra model load and pass for windows.** `analyze59b.py:207-210` reloads the model and runs a second 1,000-document pass only to get logit-lens tokens and top windows. That is roughly 10% extra GPU time that could be captured in the 59B pass. Keep it unless the budget is tight.

12. **LOW (docs, will mislead the next agent): `part59/REPORT-runner.md` is stale.** Lines 9, 12 and 13 say 24 calibration texts, "8 CPU tests", and a test name `test_symmetric_threshold_and_odd_sign_rule` that no longer exists. Current state: 34 calibration rows, 12 tests.

## Round-1 fix status

- RETURN_READY after failure: closed (`run59.sh:113` prints only when `run_status` is 0). It opened findings 1 and 2.
- Partial 59A scored: closed. `blind_setting.answers` checks the manifest status, 60 steps, the completed list, stray files and condition rows.
- Lock stranding: closed. The SIGKILL tests pass.
- Manifest completeness for the 59B archive: closed through the NaN guard, but see 6.
- Pooled rule: matches Amendment 1 word for word. `analyze59b.py:40-48` uses one P99 per dose over all three controls. `analyze59b.py:60-69` gives each control a leave-one-out threshold. Strictly greater than, same nonzero sign, linear percentile. Later layers are `range(layer_i+1, ...)` (`run59b.py` `_run_locked`). The 59h ranking is `min(|c-1|,|c+1|)`, `analyze59b.py:92-94`.
- Dose scale: matches Part 54. The 59A Qwen maxima come from the same cache Part 54's `_source_corpus_maxima` wrote (key `7c19d012047636...`), and the value 19.125 for 2742 matches.
- Calibration: partly closed (finding 4).

## Answers to the brief's questions

1. Dose scale and hook: same `NeuronHook` clamp (`part26.py` `_generate_condition` for 59A; `run59b.py` `capture_means` for 59B). Qwen maxima and controls match `_source_clamp`. OLMo L10:6978: width 8192, 16 layers, max 6.5 (`results/part26/target-olmo2-1b.npz`), twin listed in `target-olmo2-1b.json`. Qwen: 32 layers, intermediate 9216 from the cached config, so the later-layer neuron count is 9 x 9216 = 82,944, which matches the amendment.
2. Controls: `nr.control_units(width, unit, 20, seed)[:3]` in both runners, identical to Part 26's policy. Recomputed: Qwen [152, 376, 692], OLMo [135, 335, 615]. Pairing: `_condition_seed(seed, name)` per condition, identical across doses, as in Part 54. 59B is deterministic over the same documents.
3. Predictions: all eight can be scored from the outputs. 59a to 59e from the blind tally, and 59d and 59c need `--by-cond`. 59f to 59h from `analyze59b.py` (59h needs the hand reading). The 59g wording matches the analysis (`max < 5`, leave-one-out maximum). Finding 5 is the gap.
4. Failures: CUDA OOM and load errors inside `_run_locked` leave the manifest `incomplete` and re-raise, so the phase exits nonzero and the cascade (finding 1) applies. A killed run leaves `running` and the lock is released. A resumed run with different arguments is refused by the identity check. A partial 59B archive has NaN and is refused by the analysis. The gap is in the CLI, finding 2.
5. Blind reader: the condition is hidden; the request text is the rubric plus answer text only (`blind_setting.py` `submit`). The refusal is enforced before the table prints. Calibration labels are right except for the items in findings 4.
6. Memory: `run59b.py` streams one document per forward pass. Only sums (about 83k floats per condition) and the 1,000 tokenized rows are held. Qwen fp32 weights are about 15 GiB on the card; the other held state is small (sums plus tokenized rows). Peak VRAM was not measured on a GPU.
7. Other: `run59.sh` paths and the tar list match. Simulation: I copied the tar set to /tmp/df59-sim and ran `test_part59` with the venv interpreter: 12 of 12 pass (2.1 s). So every import chain in the tar set resolves. `test_part59` imports `blind_origin`, which is not in the tar; that matters only if tests run on the box (they do not).

## What I ran and could not check

- Ran: static read of every listed file and of `part26.py`, `neuron_report.py`, `run54.sh`; control-draw recomputation with numpy; SHA check of `source.json` and `source.npz` against the cached maxima inputs (both match); `source_documents`'s 256-word check (all 1,000 rows have 256 words); 12 unit tests in /tmp; the stubbed `run59.sh` in /tmp; HTTP 200 for the Hugging Face revision endpoints of `Qwen/Qwen3.5-4B@851bf6e` and `allenai/OLMo-2-0425-1B-Instruct@48d788e`; HTTP 200 on PyPI for `transformers==5.20.0` and `flash-linear-attention==0.5.2`.
- Could not check: no GPU, no model load, no real `pip install` on a box, no Vast run, no measured VRAM or throughput. Whether the tokenizers prepend special tokens (which enter the 59B means the same way as Part 26's maxima) was not checked. The 8 design sabotage runs were not repeated.
