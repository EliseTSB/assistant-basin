# Part 48b build report

## Implementation notes

Implemented `part48b.py` around the imported `part48` module. It scores the fixed ten Amendment 4 twin pairs and the single-letter `k`/`c` controls in all six Part 48 templates. Each paired item has a Part 48 continuation-set score and a letter-matched score; the latter removes only `spiky` and `curvy`. Rows retain continuation log probabilities, item scores, total continuation mass, token splits and counts, mean surprisal in the neutral quoted-word context, first-token priors, and onset-bigram priors. Pair records contain k-minus-c differences by template and after averaging over templates.

The JSON summary reports all pairs, the equal-token-count subset, and the Amendment 1 near-word exclusion sensitivity for both score sets. Unequal token-count pairs are excluded from the primary analysis. Each analysis includes a two-sided 10,000-draw sign-flip test, a 10,000-resample percentile bootstrap interval for the median, the exact sign test, per-template differences and sign tests, and Spearman correlations with the surprisal and token-count gaps. The sign-flip statistic is the mean paired difference; the interval estimates the median. None of the Amendment 1 near-word spellings exactly matches the fixed ten pairs, so that exclusion set is empty.

The CLI accepts `--models`, `--device`, `--dtype` (default `fp32`) and `--out`, and writes one JSON per model with Part 48's header fields. Its model-loading path is offline-only. No model runs or GPU runs were performed.

## Verification

Ran the requested command:

`/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part48b.py test_part48.py`

Result: **4 passed in 39.41s**. The new tests use a character tokenizer and a tiny deterministic causal model built in the test file on CPU. They cover a hand-computed sign-flip fixture, the bootstrap median interval fixture, exact letter-matched set membership, positive k-minus-c direction for both scores, and fixed-seed determinism. The existing Part 48 test module also completed successfully.

## SUMMARY

Shipped the Part 48b scoring and analysis module, CPU-only tests, and this build report. The required pytest command passed. No model downloads, real-model 48b inference, GPU use, or commits occurred.

## FINDINGS

`part48.py` remains unchanged and supplies the templates, continuation sets, item scoring primitives, and model registry. For every reported d, positive means the k twin has the sharper Part 48 score. Single-letter controls are recorded separately from the ten pseudoword pairs.

## BLOCKERS

None.
