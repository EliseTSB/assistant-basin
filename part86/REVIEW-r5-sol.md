Cold review of cbbfcae, Part 86 and repository-level observer. Reviewed copies match that commit byte-for-byte; subsequent checkout commits were unrelated. No repository edits. All commands used `/home/elisepc/neuron-probe/.venv/bin/python -I -B`, strict imports, and one math thread. Unmodified summaries:

```text
part86/test86.py: Ran 30 tests in 18.182s — OK
part86/test_audit86.py: Ran 25 tests in 108.872s — OK
test_observe.py: Ran 22 tests in 0.639s — OK
part86/mutate86.py --jobs 4: MUTATE86: 67/67 killed; all killed
```

Findings, most severe first:

1. **HIGH — Sampling can differ while every gate passes.** `part86/audit86.py:216` checks sampling mode, token budget, beams, temperature and top-p, but ignores observed top-k, typical-p and repetition penalty (`observe.py:47`). Independently adding top-k 1, repetition penalty 1.2, or typical-p 0.5 at `common86.py:422` passes both suites and the audit of real screen/verify loops on the tiny model. Top-k 1 makes the verification effectively greedy despite `do_sample=True`. **Fix:** pin and audit the complete resolved generation policy, including processors/stopping criteria, and exercise these call-boundary mutations.

2. **HIGH — Observer boundary bypasses remain.** `observe.py:334` resolves only keyword arguments: a positional GenerationConfig enabling sampling at temperature 1.3/top-p 0.5 is recorded as the model's greedy defaults and passes screen audit. The hook census runs only at entry (`:383`): a lazily registered lm_head hook forcing token 19 passes. Replacing another layer's forward also passes despite `unobserved_forwards={"0":6}` (`:434`); `audit86.py:209` never rejects that field. **Fix:** bind positional arguments to the generate signature, detect hook registration during active calls, and reject every unobserved forward. Add observer and audit regressions.

3. **HIGH — Interior random directions are unchecked.** `observe.py:138` retains only first/last projections; `audit86.py:311` checks only those. R5-06 reverses the random push on decode 3, restoring it afterward. Both suites pass; a separate five-decode reproduction measures signs `+,+,-,+,+` and audits PASS, with 0.2766% worst norm error. The live witness stops after two decodes (`test86.py:66`), so it cannot expose this. **Fix:** validate direction across every required decode, or retain sufficient per-step projection bounds; lengthen the witness.

4. **HIGH — Calibration and checkpoint claims remain self-reported.** `audit86.py:337` and `:402` trust answer medians, without observing calibration or comparing verification calibration with the originating screen. Observer headers record name/dtype/device (`observe.py:223`), but audit ignores them and records no resolved checkpoint revision. Truncating actual calibration input to 256 characters (`common86.py:171`) or loading revision `main` while metadata still claims the pin (`:140`) survives both suites. The loader is mocked; no changed Qwen checkpoint was executed. A calibration error passes runtime guards if L22 remains within its 5% tolerance (`screen86.py:325`); other layers lack that reference check. **Fix:** independently bind calibration tokens, layer-output median, resolved checkpoint and runtime dtype/device to the audit.

5. **MEDIUM — Amendment 3 is not implemented exactly.** The legacy self-model regex still searches the whole answer (`triage86.py:54`). “I am a quiet traveler. As a language model, I can help.” is kept, whereas the amended first-sentence rule says changed. An independent implementation disagrees on 13 dry-run answers. Also, miss-rate summaries collapse all coherent personas together (`blind86.py:202`, `:229`), omitting the promised reader-kind breakdown. **Fix:** apply the amended sentence scope exactly and report kind counts, including another AI; add regressions for both.

6. **MEDIUM — Launch instructions demand an impossible summary.** `part86/RUNBOOK.md:42` requires 65/65; the current gate produces 67/67 after the two Amendment-3 mutants (`mutate86.py:98`). A successful fresh preflight cannot satisfy that instruction. **Fix:** update the required count to 67.

Round-4 disposition: **#1 partial** (signed neuron/channel/layer/coverage checks fixed; finding 3 remains); **#2 partial** (registered prompts, CPU seeds and direction seeds now observed, but findings 1/4 remain); **#3 closed** (answer hashes, unique claims and exact populations: `audit86.py:189`, `:255`, `:359`, `:448`); **#4 closed for the reported substitution** (three provenance hashes compared at `verify86.py:688`, unrelated-artifact regression at `test86.py:523`); **#5 closed** (auditor copied at `RUNBOOK.md:7`; finding 6 is new).

Every R4 mutant is killed for the intended reason. Retained targeted failures confirm: 01/04 upward sign; 02/05 registered direction; 03 decoded-answer association; 06/07 observed layer/hook identity; 08 actual token budget; 09 CPU seed; 10 decode coverage; 11 rendered question; 12 actual top-p; 13 just-over-12-hour boundary (`test86.py:838`). These are substantive kills, not import or application errors.

New mutation table: each fresh copy contains part86/, part84/, neuron_report.py and observe.py, with ROOT their parent. Every row ran both full requested suites. S = survived; K = killed.

| ID | Mutation, baseline site | test86 | test_audit86 |
|---|---|---|---|
| R5-01 | C:422 actual top-k 1 | S | S |
| R5-02 | C:422 actual repetition penalty 1.2 | S | S |
| R5-03 | C:422 actual typical-p 0.5 | S | S |
| R5-04 | C:171 truncate calibration | S | S |
| R5-05 | C:140 load main revision | S | S |
| R5-06 | C:253 reverse interior random push | S | S |
| R5-07 | S:463 reverse random batch rows | K: direction cosine | S |
| R5-08 | C:253 mutate input in-place | K: hook isolation/measurement assertions | S |

C/S mean common86.py/screen86.py. Thus **6/8 survive both**. Live-loop records for 01–03 and the longer 06 reproduction also pass audit with explicitly marked fixture overrides; 04/05's changed real-Qwen execution was not measured.

Observer experiments on copies: entry-time unwatched-module hook and in-place collateral edit are detected/rejected; mid-generation hook and replaced forward bypass audit; positional input alone works; positional GenerationConfig bypasses while the same keyword config is rejected; masked left padding works; unequal EOS lengths trim answers correctly, but generated_steps includes padding (1-token answer reports six batch steps, `observe.py:412`); mid-generate exception is recorded/rethrown and incomplete answers fail audit. Artifacts: `observer-results.json`, `observer-cases/`.

Audit coverage beyond the findings is incomplete:

- A770, model/revision/bf16 and calibration are not audited. Runtime dtype checks exist (`common86.py:80`); calibration's token-zero exclusion is tested (`test86.py:751`), but neither covers the loader/text mutations.
- GPU RNG initial seeds are checked, but recorded GPU state is not compared (`audit86.py:246`); CPU intervening draws have a regression (`test_audit86.py:281`), GPU draws do not.
- Pilot hand-reading, throughput, triage range and batch-64 operation are outside audit. Gate 1 is deliberately manual; Gate 2/resume and Gate 3 have tests (`test86.py:825`, `:811`).
- Blinding, reader identity, semantic top-20 selection, rates, survival rules and Amendments 1/2's statistics/sampling are outside audit. Structural blinding/chunks, selection validation, verdicts, 200-item seed, 10% seed and kappa have tests (`test86.py:662`, `:717`, `:299`, `:439`, `:1191`, `:1170`); actual Haiku identity and semantic ranking require operator review. Amendment-3 kind reporting remains uncovered.

The renamed-assistant Fisher test (`triage86.py:110`) is correct: 4,225 small tables agree with SciPy within 9.8e-15; the dry-run table gives p=0.176116201753 versus 0.176116201756. Selection after seeing counts is adequately disclosed at `PART86-DESIGN.md:111` and `:119`, with dry-run separation and miss-rate sampling. The rerun does not make that adaptive rule an independent preregistration.

Evidence/scripts/logs are alongside this review, including `new-mutation-results.json`, `r4-reason-results.json`, `triage-check-results.json`, and `live-records/`. No A770/Qwen generation, full-size observer-record audit, or live reader was run by this reviewer.

VERDICT: DO-NOT-SHIP
