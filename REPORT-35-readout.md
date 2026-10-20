# Part 35.8 self-description readout

## Changes shipped

- `part35.py:273` adds gA/gB self-description rows with cache IDs separated from existing self-direction IDs. `part35.py:368`, `part35.py:520`, and `part35.py:1555` capture the last content-token residual, content-token masked mean, and token count using the existing chat template and residual hooks. The selfdesc-only cache schema is versioned at `part35.py:42`, and `--recapture-roles selfdesc` can be used with `--capture-only` to run only those rows on the GPU.
- `part35.py:940`–`part35.py:1499` implements the cache-only readout: seeded 120/40 splits, a gB mirror fit, train-standardized L2 logistic probes with five-fold C selection, record-bootstrap AUC intervals, train-only length regression, lexical unigram/bigram baseline, mean-difference directions, probe weights, and same-generator-only gate selection. `selfdesc-probe` does not load model weights.
- `test_part35.py:444`–`test_part35.py:832` covers content-token capture and cache fields, selfdesc-only capture, length regression, the gate selection rule, seeded bootstrap intervals, key separation, and the synthetic cache-only probe path.
- `data/part35-selfdesc-gA.jsonl:1` contains 320 hand-authored sentences. The serializer only assigned IDs/frames and wrote JSONL. The concurrent gB file was not edited.

## gA audit

Final audit output:

```text
class ai: n=160 frame_counts=abilities:14,origin:14,daily routine:14,limits:14,feelings:13,work:13,memory:13,physical substrate:13,learning:13,relationships:13,preferences:13,mistakes:13
class ai: length_mean=10.506 length_sd=1.049
class ai: forbidden_share=0.000 forbidden_ids=[]
class human: n=160 frame_counts=abilities:14,origin:14,daily routine:14,limits:14,feelings:13,work:13,memory:13,physical substrate:13,learning:13,relationships:13,preferences:13,mistakes:13
class human: length_mean=11.325 length_sd=0.972
class human: forbidden_share=0.000 forbidden_ids=[]
duplicate_openings={}
```

All 320 rows also passed the 6–25 word check. The class mean-length gap was 0.819 words and the population-SD gap was 0.077 words.

## Verification and mutation

- Required command: `~/neuron-probe/.venv/bin/python -m pytest -q test_part35.py` — **34 passed**, no skipped tests.
- `py_compile` and both `screen --help` and `selfdesc-probe --help` succeeded.
- The four initial probe tests were red on the old implementation (three missing probe functions; cache-key separation passed).
- Deliberate mutation: changed gate layer selection to maximize cross-generator AUC. `test_gate_layer_is_chosen_from_same_generator_metrics` failed because the mutated code chose layer 5 instead of same-generator layer 2. Restored the same-generator selection; the final suite passes.

## GPU commands

Run from the project root on the GPU host after both generator files are present:

```bash
OUT=results/part35
CACHE=/mnt/e/neuron-probe/hf/hub
MODEL=Qwen/Qwen3-1.7B

~/neuron-probe/.venv/bin/python part35.py screen \
  --model "$MODEL" --cache-dir "$CACHE" --device xpu --dtype bf16 \
  --out "$OUT" --capture-only --recapture-roles selfdesc

~/neuron-probe/.venv/bin/python part35.py selfdesc-probe \
  --model "$MODEL" --out "$OUT"
```

The probe writes `results/part35/qwen-qwen3-1-7b/selfdesc-probe.json`. No real model or GPU was used in this worker, so model-backed cache capture and measured AUCs remain to be run.

## Specification note

The dispatch brief references a fifth Part 35 amendment, but the checked-in Part 35 section of `PREDICTIONS.md` ends without that amendment, and repository search found no such text. The implementation follows the detailed gate and readout contract in the dispatch brief. The control mailbox had no messages.

SUMMARY: Authored gA and shipped selfdesc capture plus a cache-only, length-controlled probe with a same-generator-selected cross-generator gate.
FINDINGS: gA audit passed; the required suite passed 34 tests, and the deliberate cross-generator-selection mutation was caught and restored.
BLOCKERS: No code blocker. Model-backed GPU capture and AUCs are not run here; the referenced fifth amendment was not present in the checked-in sources.
