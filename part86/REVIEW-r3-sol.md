Ran 28 tests in 114.447s
OK
MUTATE86: 37/37 killed; all killed

Cold review of d5df099, Part 86 only. Both unmodified commands ran once with the requested venv; mutation subprocesses used strict mode and one CPU math thread. No repository edits.

## Findings, most severe first

1. **HIGH — Amendment 2's screen agreement gate does not reach 86.5.** `part86/blind86.py:550` suppresses only 86.2/86.3/86.4/86.6. `part86/verify86.py:499` derives 86.5 using verification claim-label agreement alone; neither its run nor score command consumes screen agreement. Amendment 2 explicitly gates **86.2 to 86.6** when the screen's coherent-persona kappa is below 0.6, until a full second reading. A full-denominator synthetic screen through the actual score CLI gives kappa -0.09975 and NOT READ, while independent verification scoring still gives HOLDS (20 survivors, claim kappa 1), without any way to inspect that screen gate. **Fix:** carry a hash-linked screen score/agreement and rereading status into verification, and enforce the screen gate on 86.5 too. Verification claim agreement cannot substitute for screen persona agreement.

2. **HIGH — The recurring undetectable-test failure class remains in production callers.** `part86/test86.py:678` checks screen row counts and hook counters without checking the pushes supplied by `screen86.run`; `part86/test86.py:762` replaces verification's case plan and generation with mocks. Nine of ten new mutations below pass every test. In particular, a completely unsteered neuron screen, a screen repeatedly pushing neuron 0, and double-dose random controls all remain green. Helper norm and hook tests therefore do not protect the experiment's actual wiring. **Fix:** independently inspect the vectors and generation arguments passed by the real screen/verification loops, using nonzero neuron ids and real case plans; assert neuron identity, write norms, questions, seed schedule, sampling settings, and token budgets. Add the surviving mutations to the gate.

3. **HIGH — Verification accepts an unrelated picked persona.** `part86/verify86.py:97` checks that a primary label is a coherent persona, but never compares `picked['persona']` with that label's persona. My unmodified-code reproduction passes all 20 picks labelled “a quiet traveler” while their picked phrases are distinct planet-ruling deities. A mapping/copy error then tests a different identity and can falsely reject a real discovery or report survival of an unobserved persona. This also contradicts RUNBOOK's same-label check at line 145. **Fix:** retain the picker id and validated label/phrase mapping, require a matching phrase or explicitly reviewed equivalence, and test mismatches.

4. **MEDIUM — Verification preparation discards generation provenance.** `part86/verify86.py:349` checks only `status == complete` before pooling records; its preparation manifest at line 409 records current code and items, with no input-file hashes or generation configs. A directory with 19 files at dose 0.8 and one at 0.4 is accepted and scores HOLDS in the reproduction. Ordinary `run` resume checks are stronger, but `prepare` bypasses them, so assembled/stale files can yield a confident mixed experiment. **Fix:** validate every file's full case set, config and record consistency, shared generation provenance, and selected set; carry source hashes/configs through the key and score manifest.

5. **MEDIUM — Gate 2 uses a different timing population from the registered pilot.** `part86/RUNBOOK.md:71` declares the earlier 256-real/256-random timing screen authoritative. The design specifies the **full** L22 pilot, whose mix is 9,216/512. Different answer lengths or later neurons can make the balanced warmup pass while the full pilot fails the 12-hour projection. `part86/screen86.py:182` stores no throughput gate, and session timing at line 225 resets on resume. **Fix:** persist generation time/counts across resumes and compute/report gate 2 from the full L22 pilot; keep the limited run as an early estimate only.

6. **LOW — Launch instructions demand an impossible success line.** `part86/RUNBOOK.md:35` requires `29/29`, while the committed runner now prints `37/37`. A literal operator blocks a healthy launch or follows obsolete audit instructions. **Fix:** update the required line to the measured current gate.

## Other checks

No golden-case disagreement. I compared the amendment and prediction inequalities, populations, exclusions, missing-layer behavior, Fisher direction, and verification survival boundaries. The current pure verdict rules implement them. 86.1 correctly requires hand reading. The current hook skips prefill, broadcasts each row's own vector only over one decode position, and the current screen indexes its own neuron per row. Neuron deltas and random directions target 0.8 times the residual median; calibration excludes token zero as in Part 84. Prompt rendering, no system message, thinking off, greedy 48-token screen, and verification settings are correctly wired by inspection.

Screen resume compares generation hashes/configuration, versions, calibration and median, and rejects orphan fragments. Gate 3 uses all 9,728 pilot answers and inclusive 20–2,000 bounds. Reader chunks, miss-rate samples, 10% second-reader selection, seeds and screen kappa threshold match Amendment 2. Verification checkpoints every answer and supports one-neuron tasks, so task kills are resumable. Actual A770 throughput, six-hour projections, bf16 behavior, model/tokenizer and positive-control output remain unmeasured here.

## Independent mutations

Each used a fresh scratch copy of part86/ and neuron_report.py, with part84 symlinked, then the requested `python -I -B part86/test86.py`. None is in mutate86.py. All ran 28 tests without skips. Reproducible sources/logs: `new_mutations.py`, `new-mutations.json`, `new-mutations/*/test.log`; unmodified-code demonstrations: `reproduce.py`, `repros.log`.

| ID | Mutation | Result |
|---|---|---|
| N1 | Screen caller multiplies neuron pushes by zero | SURVIVED |
| N2 | Screen caller pushes unit 0 for every neuron | SURVIVED |
| N3 | Screen caller doubles random write norm | SURVIVED |
| N4 | Screen token budget 48 → 24 | SURVIVED |
| N5 | Verification neuron push always addresses unit 0 | SURVIVED |
| N6 | Verification generates greedily | SURVIVED |
| N7 | Verification sample seeds increase by 10 | SURVIVED |
| N8 | Screen second-reader sample 10% → 20% | KILLED: test_miss_sample_follows_the_registered_seed |
| N9 | Verification caller doubles random write norm | SURVIVED |
| N10 | Verification neuron cases all ask question 0 | SURVIVED |

VERDICT: DO-NOT-SHIP
