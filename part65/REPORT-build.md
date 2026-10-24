# Part 65 build report

## SUMMARY

Built the Part 65 steering, write-word ranking, blind judge, analysis, and offline test modules. The judge uses the existing paidaction shared ledger. No model download, GPU job, live API call, or commit was made.

## FINDINGS

- Files shipped: steer65.py, rank65.py, judge65.py, analyze65.py, and test_part65.py.
- The steering module reads the frozen Part 64 collection, a local pinned Qwen3-1.7B snapshot, and the Part 60 corpus. It uses the canonical 100 Part 60 control directions (seed 5601), normalizes both real and control write vectors, adds each dose-scaled vector after the selected layer MLP at every position, batches the nine rollouts, checks baseline tokens against a zero push, and writes resumable JSONL plus a manifest. POSIX locking imports are guarded; the device loader prefers XPU and falls back to CPU.
- The per-layer scale is defined as the median token-position L2 norm of each transformer block output over the first 2,000 tokenizer IDs from the Part 60 JSONL corpus. It is saved separately and copied into the run manifest.
- The four neutral steer-lens contexts are “The weather today is”, “In the middle of the”, “Here is a simple”, and “One thing to consider is”. The design did not list context strings.
- Write words use Unicode word tokens, case-folded with curly apostrophes normalized. With target/control counts c_t and c_c, totals N_t and N_c, vocabulary size V, and smoothing 0.5, the reported mass is (c_t/N_t) * max(0, log2(((c_t+0.5)/(N_t+0.5V))/((c_c+0.5)/(N_c+0.5V)))). Controls are pooled within layer bands; a control target is excluded from its own reference. Bands are layers 0–9, 10–18, and 19–27.
- The judge makes two trials per unit, samples two distinct same-type/same-band foils from a stable seed, and places the true summary at A then B. The prompt carries only the Haiku-1x hypothesis, write words, and steer-lens tokens. The live --probe mode plans exactly one Message Batch request. It reserves, submits, saves the provider batch ID, settles from provider usage, and recovers known batches without resubmitting ambiguous no-ID POSTs. The default per-request output limit is 32 tokens, chunk size 100, and Part 65 ledger subcap $1, using shared paidaction pricing assumptions.
- Part 64 provides 500 Haiku-1x sentence-test rows; the analysis uses that arm’s per-unit pass field for 65d. Four Haiku-1x hypotheses were not parsed: two are recovered from their raw hypothesis field, and two empty outputs use the explicit placeholder “No parseable Part 64 Haiku-1x hypothesis was returned.” This preserves the registered 500-unit set, but that placeholder handling was not specified in the design.
- Analysis implements 65a–65f with 5,000 unit-bootstrap replicates (seed 6501). For the unspecified embedding representation, it uses a 512-dimensional signed Blake2b hashing of word unigrams and adjacent bigrams, with 1,000 within-type/within-band deranged shuffled pairs as the null. The 65f pass flag compares the sign of the embedding neuron-minus-control difference with the primary agreement difference.
- Offline checks passed: all 500 Part 64 labels and sentence-test rows joined; a 1,000-request foil plan passed its identity/type/band checks; fake Message Batch flows settled through the shared ledger, resumed a known batch without a second POST, and recovered a reserve written just before local state persistence. A separate synthetic analysis smoke check returned all six planted-effect predictions true and all six planted-null predictions false. The source syntax and trailing-whitespace audit passed for all five files.
- Exact unittest command: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_part65 -v. Exact output:

    test_part65 (unittest.loader._FailedTest.test_part65) ... ERROR

    ======================================================================
    ERROR: test_part65 (unittest.loader._FailedTest.test_part65)
    ----------------------------------------------------------------------
    ImportError: Failed to import test module: test_part65
    Traceback (most recent call last):
      File "/usr/lib/python3.12/unittest/loader.py", line 137, in loadTestsFromName
        module = __import__(module_name)
                 ^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/elisepc/green/worktrees/ab-part65/part65/test_part65.py", line 13, in <module>
        import torch
    ModuleNotFoundError: No module named 'torch'


    ----------------------------------------------------------------------
    Ran 1 test in 0.000s

    FAILED (errors=1)

  The test module intentionally imports torch directly, so missing torch fails rather than skips. It also prints the CPU tiny-model one-unit runtime when it can run. That duration could not be measured here because torch is unavailable; no honest runtime estimate is reported.

## BLOCKERS

The required unittest suite and CPU tiny-model timing remain unverified until a Python environment with torch is available. The environment also has no loaded Qwen model; production steering on XPU was intentionally not run.


## Claude, after collection (2026-10-24)

The worker sandbox had no torch, so its suite could not import. On the host, from `part65/` with the worktree on the path:
`PYTHONPATH=<worktree> /home/elisepc/neuron-probe/.venv/bin/python -m unittest test_part65` ran 10 tests, OK; one CPU tiny-model unit took 0.019 s. A mutant that drops the write-vector normalisation fails `test_real_and_control_write_vector_norms_match` and nothing else.
