`Ran 37 tests in 27.577s` / `OK` (zero skipped).
`Mutation summary: 33/33 killed; 0 survived or errored`

Cold review of **8466a07**, Part 85 only. References below are repository-relative; abbreviated filenames mean `part85/<filename>`. Original files were untouched. Full logs, exact replacements, and offline reproductions accompany this review.

1. **BLOCKER — the run-loop blind-test class persists.** Seven of eight new mutants pass all 37 tests, including the four independent-audit scenarios (`test_part85.py:811–871`). These are deliberately broken copies, not claims that those mutations exist in production. Trace `seed` merely echoes the requested seed (`part85.py:560,591`); kwargs reflect a local dictionary rather than observation at the `generate` boundary (`:573–596`); collateral-change detection compares tensors inside the mutating hook (`:104–111`). The audit checks chat-template invocation and question presence, not the actual chat serialization (`audit85.py:127–136`), and does not bind answer text to generated token IDs (`:157–164`). Consequently collapsed sampling, changed temperature, contaminated answers, changed chat roles, and collateral interventions can receive a clean audit. **Fix:** apply the already-decided fallback to local execution in `part85-handoff-after-r3-2026-10-27.md`, option 1's final sentence; independently observe RNG/call arguments, pre/post tensors, chat IDs, and output IDs before reconsidering this launch gate.

2. **BLOCKER — fresh-cloud bootstrap/download commands fail.** `CLOUD-RUN.md:37,74,150,227` assumes a host-specific venv without creating it. Even supplying that interpreter, the exact repository import in `:87` exits `ModuleNotFoundError: No module named 'part85'`: `-I -c` excludes the clone root. The same defect affects `:96,105,115,169,173,176,180,240,261`, breaking snapshot acquisition and cleanup. **Fix:** create a session-local venv, use its interpreter, and put repository imports in a script with an explicit import path. Installed-package imports at `:56,95,167,228` are unaffected by this import-path defect.

3. **HIGH — main Instruct cannot run as written.** `CLOUD-RUN.md:241–243` supplies `--reference` for every key; `step0_weights.py:502–503` rejects it for Instruct, before computation or resume. Offline reproduction gives the exact stated `ValueError`. **Fix:** omit `--reference` for Instruct.

4. **HIGH — attempt 2 fails, and its resume can clobber copied steps.** `CLOUD-RUN.md:155–158` copies only Instruct calibration; `:177` then runs pretrain_4T without its required `pilot-a2/pretrain_4T/step1.json` (`part85.py:431–433`). Measured: `FileNotFoundError`. On resume, `restore` fetches a2 (`handoff85.py:173`), replacing the a1 `FETCH_HEAD` set at runbook `:153`; the subsequent `git show FETCH_HEAD:...pilot-a1/...` commands fail with status 128. Shell redirection can truncate existing destination files. **Fix:** copy pretraining calibration too, and name the immutable fetched a1 commit explicitly, or fetch a1 after restoring a2.

5. **HIGH — an expected scientific result prevents complete scoring.** Prediction 85.2 requires the 21B cosine below 0.5 (`PREDICTIONS.md:2133`). That necessarily fails the same-unit gate, making 85.6 `NOT READ` (`score85.py:434–435`), then aborting ordinary scoring (`:574–577`). Reproduction: all 2,880 answers and 24 step files score successfully; changing only the early cosine to 0.4 produces `ValueError ... NOT READ verdicts ['85.6']`. **Fix:** report raw registered early counts and distinguish a measured different unit from missing data, retaining the interpretation restriction without requiring `--partial`.

6. **MEDIUM — legitimate immediate EOS fails audit.** `audit85.py:148–150` requires a decode forward pass for every clamped answer. An immediate EOS needs only prompt prefill; no generated position is subsequently forwarded. A real tiny-OLMo run with legal mixed early-EOS outputs writes complete cells but receives `AUDIT85 instruct pilot: 12 failures` (`eos.log`). The fixture deliberately forces two output tokens (`test_part85.py:63–79`). **Fix:** record output token IDs/termination and validate the actual forwarded positions, allowing this case; preserve empty answers for blind quality labels.

7. **MEDIUM — chat-only scope silently drops registered measurements.** The design restricts **step 2** when the plain-frame effect fails (`PART85-DESIGN.md:78–79`), while steps 0/1 remain per-checkpoint (`:43,53`). The runbook excludes pretraining checkpoints entirely (`CLOUD-RUN.md:250–252`), and scoring reads only main step trees (`score85.py:263–276`). Thus 85.1–85.4 become unavailable despite potentially useful pilot measurements; audit checks none of this. **Fix:** retain all twelve weight/activation measurements when restricting generation to chat, or register this scope change before data collection.

8. **MEDIUM — publication/blinding durability remains manual and underspecified.** `CLOUD-RUN.md:212–215` says commit references but omits publishing that commit; fresh clones cannot checkout an unpublished SHA. `:291–295` omits committing both independent readers' labels before opening the key, required by `PART85-DESIGN.md:69–72`. Neither tests nor audit establishes that procedural property. **Fix:** explicitly publish/verify the references commit and prescribe the rubric, independent readers, label commits, then unblinding.

Round-3 closure: Haiku **1/2/3/4/8/11/12 fixed** by owned directories, checkout-before-write, verified retries, identity setup, and claimed-file checks (`handoff85.py:58–92,143–178,195–238`); **5 open** above; **6 fixed** (`score85.py:293–311`); **7 fixed** in cleanup target (`CLOUD-RUN.md:97`), though its setup command fails; **9 partial**, publication still omitted; **10 fixed** (`step1_acts.py:203–212`); **13 fixed** by step resume/provenance and isolated test subprocesses (`step0_weights.py:508–518`, `step1_acts.py:214–226`, `test_part85.py:1620`). DS **F1/F2/F6/F7 fixed** (`handoff85.py:212–238,259–265,210`, `part85.py:490–493`); **F3 preflight fixed** (`handoff85.py:360–365`), runtime still checks only commit (`part85.py:435`); **F4 intentionally prerequisite-free**; **F5 remains the documented Git trust boundary**.

New mutations, each on a separate copy containing `part85/` and sibling `neuron_report.py`, running the requested interpreter with `-I -B part85/test_part85.py`. SURVIVED means all 37 tests and their audit subprocesses passed; none skipped.

| Mutation | Original location | Result |
|---|---|---|
| A: actual RNG always seeded 85000 | part85.py:560 | SURVIVED |
| B: decode prompt plus answer | part85.py:579 | SURVIVED |
| C: omit chat generation prefix | part85.py:188 | SURVIVED |
| D: chat role assistant instead of user | part85.py:187 | SURVIVED |
| E: temperature 1.0 at actual call, traced dictionary unchanged | part85.py:574 | SURVIVED |
| F: zero channel 0 in-place before cloning | part85.py:104 | SURVIVED |
| G: omit hook sequence reset | part85.py:558 | KILLED |
| H: calibrate layer 9 | step1_acts.py:169 | SURVIVED |

Module identity, decoded prompt IDs, and activation extrema/matches are measured (`part85.py:136–148,545,565`); unit/value are live configuration, phase comes from hook state, and bookkeeping is loop metadata. H demonstrates unchecked calibration wiring beyond step-2 audit's scope. Verified pushes preserve completed steps/cells; interruption before push can lose the current computation (`:611–621`). No real checkpoint, cloud install, live dataset, or production remote was exercised.

VERDICT: DO-NOT-SHIP
