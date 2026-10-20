# Part 35 gate implementation report

## Changes shipped

- `part35.py:34-36` fixes separate seeds for discovery, indirect, non-English, and joint permutations. `part35.py:1311-1442` computes the joint statistics under one shared label assignment, with batched device-side unit scoring, and creates the three marginal maxima distributions.
- `part35.py:1456-1540` adds the indirect and non-English nulls to cached scoring on the discovery null's device. `part35.py:1594-1609` records both extra null maxima and all margins; `gate_pass` requires the existing discovery and confirmation passes plus both secondary passes.
- `part35.py:1678-1685` adds the joint-null distribution and funnel row for a full rescore. `part35.py:1693-1754` merges all three shard nulls, recomputes the shared-label joint null against model-wide bars, and emits the same row fields and funnel.
- `part35.py:1780-1848` implements gates, per-model concept uniqueness, and row/concept funnel counts. `part35.py:1851-1890` rereads the activation cache for the joint-null pass after marginal bars are known.
- `test_part35.py:133-199` adds regressions for a name detector rejected by the indirect gate, a planted concept unit accepted, near-zero random joint-null counts, shared top-five unit exclusion, and hand-counted funnel stages.

The checked-in `PREDICTIONS.md` has no Part 35 third amendment text. I implemented the explicit gate and field contract in this assignment.

## Verification

Required command:

```text
~/neuron-probe/.venv/bin/python -m pytest -q test_part35.py
.........................                                                [100%]
25 passed in 4.84s
```

`py_compile` for both files, `git diff --check`, and `part35.py screen --help` succeeded. The new tests were first run against old code and all four failed on the missing gate/null/uniqueness/funnel APIs. For the requested mutation, I removed the indirect conjunct from `gate_pass`; `test_conjunctive_gate_rejects_name_detector_but_keeps_concept_unit` failed because the name detector incorrectly passed. The conjunct was restored, and the full suite passed afterward.

## Cache rescore and expected time

The stated screen command will produce the new row fields and `stage: "funnel"` once scoring finishes. For a cache-only rescore, use it with `--reuse-cache`:

```bash
~/neuron-probe/.venv/bin/python part35.py screen \
  --model Qwen/Qwen3-1.7B --cache-dir 'E:\neuron-probe\hf\hub' \
  --device xpu --dtype bf16 --out 'results\part35' \
  --permutations 200 --spread-floor median --reuse-cache
```

Without `--reuse-cache`, this CLI also enters `run_capture`, loads the model, and may refresh stale neutral/self-direction cache entries. The scoring code itself reads the complete activation cache; no model or GPU run was done here.

Scoring now computes six statistic-pass equivalents per layer instead of the previous one: three independently seeded marginal nulls and one shared-label pass for all three statistics. Expect about five additional old null-pass equivalents (roughly 6x the permutation-scoring work). The earlier planning report estimated 1–5 CPU-hours/model for 1B–1.7B and 2–12 CPU-hours/model for 3B–4B; a same-throughput extrapolation is about 5–25 and 10–60 additional CPU-hours/model respectively. Those are not XPU timings, and the actual extra time on the stated GPU is unmeasured.

No parent-control messages arrived, so there were none to apply, decline, or defer. No other work was left undone beyond the unrun real-cache GPU rescore.

SUMMARY: Added deterministic per-statistic permutation nulls, the joint gate null, candidate gates, concept uniqueness, and funnel summaries.
FINDINGS: All 25 tests pass; the deliberate missing-indirect-gate mutation made the planted name-detector test fail as intended.
BLOCKERS: No code blocker; a real activation-cache XPU rescore and timing were not performed.
