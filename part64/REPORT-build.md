# Part 64 build report

## SUMMARY
The Part 64 pipeline was not built. A concrete mismatch in the required collection contract was found before implementation: Part 64 requires a fresh 3x sample (36 top, 18 random, 9 weak windows per unit), but the specified sparse atlas collector emits only 20, 10, and 5. Its CLI has no targeted-unit selection. Running it unchanged cannot produce the required collection, and wrapping it without changing its internal collection mechanics would silently ship an incomplete sample. The registered design says to stop rather than ship a partial implementation when a required part cannot be done safely.

## FINDINGS
- `PART64-DESIGN.md` registers 1x as the exact Part 60 M1 sample and 3x as three times the windows in each category, with the subset relation verified for every unit.
- `PART60-DESIGN.md` specifies the M1 collection as top 12, random 6, weak 3. Therefore 3x requires top 36, random 18, weak 9.
- `atlas_collect_fast2.py:39-40` fixes `TOP_K=20` and `RANDOM_WINDOWS=10`; `atlas_collect_fast2.py:297` fixes the weak reservoir at 5. `_record_rows` serializes only those reservoir entries. Its parser at `atlas_collect_fast2.py:894-918` exposes layers, document limit, and total control count, but no selected-unit list or per-category capacity options.
- `part60/gen60.py:269-273` selects the M1 12/6/3 windows from each Part 60 pilot record. This provides the necessary 1x definition, but not the fresh 3x records.
- The checkout has `data/pile10k-head200.jsonl`, but not `data/pile10k.jsonl` (the Part 56 design's 10k-document corpus). This further prevents validating that a fresh collection uses the registered corpus slice in this environment.
- No code or tests were added, and no API, GPU, network, or paid operation was attempted. No tests are claimed as passing.

## BLOCKERS
A correct implementation needs a scoped collector that can collect the full 36/18/9 strata for the exact 500 Part 60 dev units while retaining the same random directions, corpus, model revision, token-window rules, and deterministic 1x subset. The existing collector would need to be extended or a Part 64 collector would need to implement its collection engine by reusing its lower-level model/window functions. That work cannot be replaced with a safe invocation of the existing CLI. The full registered corpus is also absent from this checkout. Per the brief's stop rule, generation, scoring, analysis, runner, runbook, and tests were not built on top of an invalid collection assumption.

## Round 2

### SUMMARY
Round 2 implements and tests the deterministic 1x-in-3x assembly and refusal gates in `part64/collect64.py`. It keeps each pilot M1 window in the expanded set by construction, excludes duplicate fresh windows, enforces the 36/18/9 category counts, records fresh-top-12 reproduction per unit, and refuses output when the median is below 10/12. The synthetic contract tests pass. This is not a completed Part 64 pipeline: the collector currently expects fresh atlas-style rows and does not execute a targeted model traversal, and generation, paid lifecycle, scoring wrapper, analysis, runner, runbook, and their tests were not built. No paid calls, network, GPU, or commit were made.

### FINDINGS
- `part64/collect64.py:20-23` records the registered corpus path, exact byte size, hash prefix, category capacities, and M1 counts. `verify_corpus` at lines 37-46 computes the actual SHA-256 and refuses any mismatch; the report manifest stores full provenance. The registered corpus is not present in this checkout, so this production gate was not run against it.
- `part64/collect64.py:49-55` gets the exact 500 dev records from `part60.gen60.load_splits()` instead of recreating unit sampling. It requires 250 neuron units and 250 controls.
- `part64/collect64.py:76-105` copies Part 60's top 12, random 6, and weak 3 pilot rows, then fills the respective 3x capacities from distinct fresh windows. Top candidates are sorted by activation; random and weak candidate orders use a recorded deterministic seed. It validates each category's exact size and subset relation.
- `part64/collect64.py:108-131` records the corpus/model/window manifest and computes the median top-12 overlap; a median below 10 raises before serialization. `main` writes the collection file only after assembly has passed all checks.
- Coverage: `part64/test_collect64.py:13-37` checks exact subset inclusion and 36/18/9 capacities; lines 39-47 check that duplicate candidates cannot fill a stratum; lines 49-54 check that a sub-10 reproduction median raises. Command: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part64.test_collect64 -v` (3 passed). `py_compile` also passed for the collector and test module.
- Parent-control mailbox was read before work, after substantial operations, and before reporting; it contained no messages. No dispositions were needed.

### BLOCKERS
This checkout does not contain `/mnt/c/Users/alexa/neuron-probe-gpu/qwen35/pile10k.jsonl`; it has only the 200-document head fixture. The corpus is expected to be uploaded to the rented GPU, so this is not itself a reason to accept a different corpus. The required real collector must still traverse that registered corpus using the atlas model/hook/window routines for the exact selected units. The submitted `collect64.py` only assembles previously collected fresh rows: the required execution path that produces those rows has not been implemented, so do not treat its output as a fresh collection. The brief requires the collector test to use the head fixture with a tiny model or fake hook; the tests here exercise assembly with synthetic rows and do not establish model-hook, tokenization, or corpus traversal behavior.

The remainder of the brief is also unimplemented: four-arm generation with default-thinking API requests and `probe`, one durable ledger per writer/sample/unit type and resumable sequential paid lifecycle, imported Part 60 scoring, `run64.sh` prefetch/return handling, `analyze64.py` with registered predictions and paired bootstrap, paid-path kill-matrix tests, synthetic analysis tests, and `RUNBOOK-64.md`. Consequently this round does not meet the build brief and must not be used to make calls or claim results. Resume by implementing and reviewing the actual selected-unit traversal first, then the paid and scoring stages, retaining the strict corpus and M1 subset checks.
