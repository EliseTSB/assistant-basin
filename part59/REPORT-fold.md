# Part 59 round 1 fold report

## SUMMARY

- Folded the two FIX-FIRST reviews and the OLMo tarball papercut into the Part 59 runner.
- Added Amendment 1 and implemented its pooled P99 responder rule without changing prediction text.
- Updated run, blind-submit, calibration, lock, analysis-persistence, runbook, and test behavior.
- The required null simulation fails the brief's acceptance gate: one independence control has 6 symmetric neurons, above the required maximum of 4. Work stopped without selecting another rule.

## FINDINGS

### Luna review

1. **Phase failure could still emit `RETURN_READY` — fixed in `run59.sh`.** Setup, input preflight, package checks, model prefetch, GPU phases, and archive status are recorded. A failure gates later GPU phases, still creates an incomplete archive when tar succeeds, and prints `RUN_FAILED` with failed phases. The complete marker is printed only when every recorded phase and the archive succeed.
2. **Partial/stale 59A output could be scored — fixed in `blind_setting.py` and `run59a.py`.** Submit requires one complete manifest for each pinned model, 60 prompt-dose files per model, eight answers per condition, all listed paths complete, and no stray clamp files. It reads only manifest-listed basenames. A forced 59A rerun removes superseded generated result files.
3. **A hard kill could strand a lock — fixed in `run59a.py` and `run59b.py`.** Both runners now hold nonblocking `fcntl.flock` locks for process lifetime. Real subprocesses were killed with SIGKILL and each runner then completed a mocked rerun.
4. **Output file granularity was unclear — fixed in `RUNBOOK-59.md`.** It now states that 59A writes 60 files per model, one per prompt and dose, with eight samples in every condition.

### Haiku review and papercut

1. **OLMo target JSON omitted from package — fixed in `RUNBOOK-59.md` and `run59.sh`.** The JSON is in the tar list and preflight checks it alongside the other required source files before any model phase.
2. **Per-neuron threshold was inconsistent with the registered pooled rule — implemented, but BLOCKED by the required null acceptance check.** Target thresholds pool absolute changes over all three controls and all later-layer neurons per dose. A control's pseudo-count uses a threshold pooled over the other two controls. Simulated counts for 82,944 neurons were: independence, target 2 and controls `[6, 3, 4]`; dose correlation `r=0.5`, target 72 and controls `[108, 93, 75]`. Since an independence control count is 6, the brief's under-5 criterion is not met. I stopped without choosing a replacement rule.
3. **Part 54 dose scale wording — fixed in `DESIGN-AMENDMENT-1.md`.** It records the Part 26 Pile maximum and Qwen L22:2742 value 19.125, correcting the alien-frame wording.
4. **Shell failure markers, fetch status, and version assertion — fixed in `run59.sh`.** `fetch_model` failures are captured; `transformers==5.20.0` and `flash-linear-attention==0.5.2` are asserted before GPU phases. Available versions are recorded.
5. **59A/59B process-death resumability — fixed and exercised.** In addition to the two SIGKILL-and-rerun checks, the existing 59A interruption-and-resume test still passes.
6. **Blind submit completeness — fixed in `blind_setting.py`.** Incomplete, missing, malformed, duplicate, and stray results refuse submission with missing-piece details.
7. **Calibration length and contestable OTHER items — fixed in `blind_setting.py`.** Removed “made-up island” and “beyond the edge of the known world”; added eight paragraph-length examples across four labels, including ambiguous openings resolved by details. The new total is 34 and refusal remains at more than two wrong. Tests confirm exactly two scores and three refuses.
8. **“Later” and 59h ranking definitions — fixed in Amendment 1.** Later layers are strictly after the clamped layer; strongest 59h responders rank by the minimum absolute change across -1 and +1.
9. **Test gaps — fixed in `test_part59.py`.** Tests now check the Part 26 control draws (Qwen 152/376/692; OLMo 135/335/615), exact clamp writes, pooled synthetic P99 thresholds at both doses, opposite signs, and a leave-one-out control response.
10. **Classification could be lost during model reload — fixed in `analyze59b.py`.** Counts and responder indices are written before logit-lens/window work; removed the no-op assignment.
11. **No-op reruns loaded models — fixed in `run59a.py` and `run59b.py`.** Manifest identity and complete-output checks precede model load. Tests assert completed reruns do not load a model.
12. **Key handling and repo hygiene — fixed in `part59/.gitignore` and `RUNBOOK-59.md`.** The local `review/` key directory is ignored; the runbook identifies `~/.anthropic-credits.env`, says not to print it, and keeps `blind_setting.py` on the local machine.
13. **Control draw concentration — documented in `RUNBOOK-59.md`.** It states the three lowest-indexed units are taken from Part 26's 20-unit draw and notes this limits representativeness.

The earlier source checks remain consistent with the reviews: Part 59 uses Part 26 Pile maxima, the Part 54 clamp hook point and seed policy, and the registered control draw. The runbook's duration and memory estimates were not measured on this run; no GPU or Anthropic API was used.

## TESTS AND SABOTAGES

- `/home/elisepc/neuron-probe/.venv/bin/python -m unittest -v test_part59`: **12 tests passed**.
- `bash -n run59.sh`, Python `py_compile`, and `git diff --check`: **passed**.
- Earlier sabotage `flip-sign`: **red**, responder assertion gave 1 instead of 2.
- Earlier sabotage `drop-control-percentile`: **red**, responder assertion gave 3 instead of 2.
- New `per-neuron-percentile` sabotage: **red**, responder assertion gave 3 instead of 2.

## BLOCKERS

The prescribed pooled-rule simulation does not satisfy the acceptance gate. One of three leave-one-out controls has 6 symmetric neurons under independent doses, while the rule is acceptable only if every control stays below 5. Per the brief, no alternate threshold was selected; a decision on whether to proceed with Amendment 1 despite this result is required before a run.
