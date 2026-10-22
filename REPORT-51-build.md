# Part 51 build report

## SUMMARY

Part 51 is implemented in `part51.py`, which imports Part 49's model, natural-max, seed and generation helpers. Both push commands and scorers are in place. Existing Part 49 blind-pack scoring now routes Part 51 inputs to these scorers. No Part 49b scoring or generation code was changed.

## FINDINGS / WHAT SHIPPED

- `push51a` builds six arms over the everyday prompts, shares per-row seeds across arms, clamps at +8/+4 as registered, records the fixed Part 49b units, and writes `push51a.jsonl` plus `push51a-scores.json`.
- `push51b` builds two no-clamp replicates and six seeded layer-specific random arms from the loaded MLP widths. Replicate B defaults to seed 3802 when A uses 3801; both seeds, widths and drawn units are recorded in the header. It writes `push51b.jsonl` plus `push51b-scores.json`.
- 51a scoring excludes prompt-mentioned stems and reports per-prompt density means, mean intervals and paired difference intervals. 51b derives its stock names from replicate A, reports name and NONAME measures, entropy, top names, per-arm differences from B and the pooled random-arm comparison. Both use 10,000 prompt-cluster bootstrap draws with fixed seeds (5102 for 51a, 5103 for 51b). Broken-answer labels begin pending; `part49.py score` can merge blind-reader labels and rescore 51a.
- Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part51.py test_part49.py` passed (21 tests); `git diff --check` passed. An isolated mutation that reversed prompt-stem exclusion made the focused 51a test fail (2 matches observed instead of 6).
- `free -g` showed 8 GB available, so both requested CPU smokes ran against `/mnt/e/neuron-probe/hf/hub` with `--limit 2 --samples 1`. 51a produced 12 rows across six arms; its reported arm generation times summed to 98.44 seconds. 51b produced 16 rows across eight arms; arm generation times summed to 134.12 seconds. These sums exclude model loading and natural-max capture.

### A770 commands

Run from the copied folder with the required `data` files present:

```powershell
E:\neuron-probe\.venv\Scripts\python.exe part51.py push51a --cache-dir E:\neuron-probe\hf\hub --device xpu --dtype bf16 --threads 4 --seed 3801 --batch 16 --out results\part51a-xpu
E:\neuron-probe\.venv\Scripts\python.exe part51.py push51b --cache-dir E:\neuron-probe\hf\hub --device xpu --dtype bf16 --threads 4 --seed 3801 --replicate-b-seed 3802 --batch 16 --out results\part51b-xpu
```

The runnable folder needs `part38.py`, `part49.py`, `part51.py`, `fastgen.py`, and `data/part51a-everyday-prompts.json`, `data/part51b-name-prompts.json`, and `data/part35-concepts.json`. Install the Python dependencies (`torch`, `numpy`, and `transformers`) and provide the pinned model snapshot in the cache.

## FILES TOUCHED

- `part51.py` — Part 51 arm plans, runners, scorers, CLI.
- `part49.py` — additive routing from the existing blind-reader score command to Part 51 scorers.
- `test_part51.py` — CPU tests for extraction, exclusion, arms, bootstrap and blind-label rescoring.
- `REPORT-51-build.md` — this build report.

## BLOCKERS

None. The full 16-prompt, four-sample XPU experiment and blind-reader labels remain future run work.

Parent control was checked at the required checkpoints; no messages were delivered, so no dispositions were needed.

## SELF-ASSESSMENT

The name-share point estimate pools named answers, and each bootstrap resample pools the sampled prompt clusters before computing the share. Entropy is over named answers. The full experiment has not run, so smoke output validates the execution path rather than the registered predictions.
