Ran 29 tests in 6.096s
MUTATE86: 49/49 killed; all killed

Cold review of c4ab0a3, Part 86 only. The unmodified suite ended **OK**, with no skips. Both requested commands ran once using the specified venv, strict imports and one CPU math thread. No repository edits by reviewer. The live checkout advanced to 6db5881 during review; its sole Part 86 change fixes the interpreter path noted below. Tests used the preserved c4ab0a3 snapshot. Source citations refer to c4ab0a3.

## Findings, most severe first

1. **HIGH — The audit still cannot establish the registered intervention.** `part86/common86.py:252` measures actual tensor differences, but channel identity and norm are retained only for the first decode (`:267`); `part86/audit86.py:62` merely requires one changed decode. Its neuron check (`:73`) tests channel and magnitude, never positive sign. Negative-dose screen and verification, verification at the previous layer, and steering only the first decode all survive. Layer identity is supplied from the plan (`part86/verify86.py:388`), rather than identified from the hooked module. Thus an apparently audited verification can test another layer or intervention. **Fix:** independently bind module identity and measure signed neuron deltas plus intervention coverage across every required decode position; reject nonfinite norms (`audit86.py:68` also accepts NaN).

2. **HIGH — Generation settings can be echoed while execution differs.** The question string and seed are recorded from arguments (`part86/common86.py:445`, `:458`), while direction hashes cover requested push vectors (`:451`), not measured deltas. Screen prompt hashes need only agree with one another (`part86/audit86.py:140`); verification never checks them. Changing actual RNG initialization or tokenizing every verification call as “Who are you?” survives, as do shifted screen/verification random-vector seeds. The auditor checks declared sampling seeds (`:179`) and vector distinctness (`:204`), without reconstructing their execution. **Fix:** validate actual prompt tokens against independently rendered registered questions, observe RNG initialization, and independently reproduce registered seeded directions. These registered properties have no effective test at the real call boundary.

3. **HIGH — Answers are not bound to their measured pushes.** `part86/screen86.py:455` associates generated answers with separately indexed traces (`:460`); reversing the answers survives. The screen auditor compares identity sets (`part86/audit86.py:145`), losing multiplicity and checking no answer content. The verification auditor reads only trace files (`:158`), never answer states. Unmodified registered-size reproductions report `AUDIT86 verify: PASS` with **zero answer files**, and `AUDIT86 screen: PASS` with **9,729 answers** for 9,728 traces. A reordered output can therefore attach a persona to the wrong neuron. **Fix:** bind generated token IDs/answer hashes and row identity to each observation; check exact cardinalities and correspondence against saved answers.

4. **HIGH — Round-3 finding 1 is only partly fixed: agreement is not tied to the originating screen.** `_load_screen_score` checks the supplied artifact's own hashes (`part86/verify86.py:620`); scoring accepts its kappa (`:741`) without comparing screen/primary-label identity with verification provenance. Generation configs (`:186`) retain neither originating screen-manifest nor primary-label hashes. Using ordinary `prepare` on 20 complete, provenance-bearing synthetic neuron files, the same 5,280 labels score **NOT READ** with a low-agreement artifact and **HOLDS, 20/20** with an unrelated high-agreement artifact. **Fix:** carry screen-manifest, primary-label and selected-set hashes through generation/preparation and require agreement artifacts to match them.

5. **MEDIUM — Deployment instructions omit the required auditor.** `part86/RUNBOOK.md:3` does not include `audit86.py` in its copy list, although tests (`part86/test86.py:979`) and pilot instructions (`part86/RUNBOOK.md:76`) require it. A fresh GPU deployment following that list cannot pass preflight. **Fix:** include the auditor. Separately, c4ab0a3's hard-coded Linux interpreter (`part86/test86.py:978`) was corrected to `sys.executable` by concurrent commit 6db5881; that particular blocker is already fixed.

## Round-3 disposition and remaining coverage

- **1:** partial; the supplied low/missing kappa now gates 86.5 (`part86/verify86.py:752`), but finding 4 remains.
- **2:** the specific former survivors are killed (`part86/mutate86.py:121`); the broader wiring-assurance finding remains, as above.
- **3:** fixed: phrase equality or explicit reviewed equivalence is enforced (`part86/verify86.py:130`), with mismatch coverage (`part86/test86.py:653`).
- **4:** mixed-dose/config preparation is fixed: full case sets, record consistency, shared configs and source hashes are validated (`part86/verify86.py:472`, `:494`, `:518`, `:524`). Originating selected-set binding remains absent, as finding 4 explains.
- **5:** implementation fixed: persisted per-batch times and canary timing feed the full pilot projection (`part86/screen86.py:214`, `:225`, `:230`). **The registered 12-hour boundary is untested:** a 120-hour threshold survives R4-13; tests check limited timing (`part86/test86.py:990`) and Gate 3 (`:733`), while the auditor checks neither timing nor Gate 2. Add numerical full-pilot and resumed-timing boundary tests.
- **6:** fixed: launch requires the measured 49/49 line (`part86/RUNBOOK.md:35`).

The auditor does check declared populations, first-decode channels/norm, dose, question strings, sampling kwargs, seed schedule and direction distinctness. It does **not** establish every registered property. Prefill safety is currently guaranteed by returning unchanged input (`part86/common86.py:238`), but `prefill_changed` is initialized false (`:221`), not measured. Median is calibrated; dose, row/case and target metadata come from the caller. The golden verdict tests remain green (`part86/test86.py:255`); no new verdict-rule discrepancy found.

## Independent mutation table

Each fresh copy included `part86/`, `neuron_report.py`, and a `part84` symlink; ROOT was their parent. Every mutant ran all 29 tests via the requested `python -I -B part86/test86.py`, without skips. None is in the committed gate. S/V/C below mean `part86/screen86.py`, `part86/verify86.py`, `part86/common86.py`.

| ID | Mutation site and change | Result |
|---|---|---|
| R4-01 | S:439 negative neuron push | SURVIVED |
| R4-02 | S:376 random-vector seed +1 | SURVIVED |
| R4-03 | S:455 reverse answer association | SURVIVED |
| R4-04 | V:376 negative neuron push | SURVIVED |
| R4-05 | V:367 random-vector seed +1 | SURVIVED |
| R4-06 | V:353 hook previous layer | SURVIVED |
| R4-07 | S:295 calibrate/hook previous layer | KILLED: verification median mismatch |
| R4-08 | V:30 token budget 63 | KILLED: trace/config checks |
| R4-09 | C:394 actual CPU RNG seed +1 | SURVIVED |
| R4-10 | C:252 push first decode only | SURVIVED |
| R4-11 | C:390 always tokenize screen question | SURVIVED |
| R4-12 | V:32 top-p 0.9 | KILLED: trace/config checks |
| R4-13 | S:232 accept 120-hour projection | SURVIVED |

Reproduction artifacts are beside this review: `new_mutations.py`, `new-mutations.json`, `new-mutations/*/test.log`, `reproduce.py`, `repros.log`, and baseline logs. **10/13 survive.** No model download, A770 run, live reader or actual 12-hour throughput was measured. The failures above can leave an audited result unable to distinguish the registered experiment from a different intervention or answer mapping; full GPU expenditure should wait.

VERDICT: DO-NOT-SHIP
