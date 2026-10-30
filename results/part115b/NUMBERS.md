# Part 115b number ledger

The one-liner below recomputes the counts from the raw run and comparison files, checks the saved score against a fresh score, draws `fig-flips.png`, and writes this ledger.

One-liner: `cd . && python results/part115b/plot115b.py`

## Run files and gates

- Claim: experimental scope and row dimensions. Output: Qwen/Qwen3.5-4B revision `851bf6e8`, bf16 on A770; plain instruction ` Please answer in plain text, without markdown.`; rankings `2`; held-out phrasings `16` per opening; ordinary probes `6`; greedy continuation length `12` tokens; matching pool `20` nearest same-layer alternatives; top-k exclusion `500`.
- Claim: run duration and loaded run settings. Output: `elapsed_s=546.5`; duration `9 min 6.5 s`; direct load `True`; ks `1, 2, 5, 10, 20, 50, 100, 200, 500`; matched sets per k and ranking `10`; target ids `{'plain': 1167, 'quote': 48, 'bold': 48};` all-ones gate `True`; unzeroed targets first for both rankings `True` and `True`.
- Claim: context target counts at baseline. Output: gradient `{'plain': '16/16', 'quote': '16/16', 'bold': '16/16'}`; direct `{'plain': '16/16', 'quote': '16/16', 'bold': '16/16'}`.
- Claim: six ordinary probes' mean answer-token log-probability changes at k=500. Output: gradient `-0.011141` nats; direct-write `-0.004404` nats.

## Candidate flips and matched-control maxima

One row is the maximum flip count among matched random sets at that k, across both rankings and all three openings. Candidate cells are recomputed from each raw row. Each individual ranking/opening's random maximum is included in the output column details.

| k | Gradient plain | Gradient quote | Gradient bold | Direct plain | Direct quote | Direct bold | Overall random maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |  <!-- controls: gradient/plain=0, gradient/quote=0, gradient/bold=0, direct/plain=0, direct/quote=0, direct/bold=0 -->
| 2 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |  <!-- controls: gradient/plain=0, gradient/quote=0, gradient/bold=0, direct/plain=0, direct/quote=0, direct/bold=0 -->
| 5 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |  <!-- controls: gradient/plain=0, gradient/quote=0, gradient/bold=0, direct/plain=0, direct/quote=0, direct/bold=0 -->
| 10 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |  <!-- controls: gradient/plain=0, gradient/quote=0, gradient/bold=0, direct/plain=0, direct/quote=0, direct/bold=0 -->
| 20 | 0/16 | 0/16 | 0/16 | 1/16 | 1/16 | 0/16 | 0/16 |  <!-- controls: gradient/plain=0, gradient/quote=0, gradient/bold=0, direct/plain=0, direct/quote=0, direct/bold=0 -->
| 50 | 0/16 | 0/16 | 0/16 | 7/16 | 12/16 | 13/16 | 0/16 |  <!-- controls: gradient/plain=0, gradient/quote=0, gradient/bold=0, direct/plain=0, direct/quote=0, direct/bold=0 -->
| 100 | 1/16 | 1/16 | 0/16 | 15/16 | 16/16 | 16/16 | 0/16 |  <!-- controls: gradient/plain=0, gradient/quote=0, gradient/bold=0, direct/plain=0, direct/quote=0, direct/bold=0 -->
| 200 | 2/16 | 2/16 | 3/16 | 16/16 | 16/16 | 16/16 | 1/16 |  <!-- controls: gradient/plain=0, gradient/quote=1, gradient/bold=0, direct/plain=0, direct/quote=0, direct/bold=0 -->
| 500 | 13/16 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 | 0/16 |  <!-- controls: gradient/plain=0, gradient/quote=0, gradient/bold=0, direct/plain=0, direct/quote=0, direct/bold=0 -->

## Thresholds, continuations, and registered verdicts

- Claim: Gradient / plain threshold and random maximum. Output: k* `500`; candidates `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:0/16, 50:0/16, 100:1/16, 200:2/16, 500:13/16`; maxima `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:0/16, 50:0/16, 100:0/16, 200:0/16, 500:0/16`.
- Claim: Gradient / quote threshold and random maximum. Output: k* `500`; candidates `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:0/16, 50:0/16, 100:1/16, 200:2/16, 500:16/16`; maxima `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:0/16, 50:0/16, 100:0/16, 200:1/16, 500:0/16`.
- Claim: Gradient / bold threshold and random maximum. Output: k* `500`; candidates `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:0/16, 50:0/16, 100:0/16, 200:3/16, 500:16/16`; maxima `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:0/16, 50:0/16, 100:0/16, 200:0/16, 500:0/16`.
- Claim: Direct-write / plain threshold and random maximum. Output: k* `100`; candidates `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:1/16, 50:7/16, 100:15/16, 200:16/16, 500:16/16`; maxima `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:0/16, 50:0/16, 100:0/16, 200:0/16, 500:0/16`.
- Claim: Direct-write / quote threshold and random maximum. Output: k* `50`; candidates `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:1/16, 50:12/16, 100:16/16, 200:16/16, 500:16/16`; maxima `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:0/16, 50:0/16, 100:0/16, 200:0/16, 500:0/16`.
- Claim: Direct-write / bold threshold and random maximum. Output: k* `50`; candidates `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:0/16, 50:13/16, 100:16/16, 200:16/16, 500:16/16`; maxima `1:0/16, 2:0/16, 5:0/16, 10:0/16, 20:0/16, 50:0/16, 100:0/16, 200:0/16, 500:0/16`.
- Claim: continuation counts. Output: gradient/plain k=500 non-Q first words `13/16` with distribution `{'Chat': 11, 'ChatBot': 2}`; direct/plain k=100 exact phrase `a large language model, trained by Google` appears in `4/16`; direct/bold k=100 first words `{'Google': 3, 'Tongyi': 4, 'Claude': 7, 'Assistant': 2}`.
- Claim: stored registered verdicts, verified against a fresh score. Output: `{'115b.1': 'held', '115b.2': 'held', '115b.3': 'failed', '115b.4': 'held', '115b.5': 'held', '115b.6': 'held', '115b.7': 'failed'}`; withdrawals `{}`; saved score agrees `True`.

## Comparison data and registration history

- Claim: Part 115 gradient result in its original `I am **` prompt. Output: k* `200`; at k=200 `11/16`; at k=500 `16/16`.
- Claim: Part 116 direct-write result in its original `I am **` prompt. Output: k* `100`; at k=100 `16/16`.
- Claim: commits that registered and amended the design. One-liner: `git log --format='%h %ci %s' -- PART115B-DESIGN.md`. Output:

```text
b2e90a0 2026-10-29 20:11:39 -0500 Part 115b Amendment 1, detail: the direct load's first try failed before any data (transformers' allocator warm-up asks for one 7.8 GiB block, over the A770's single-allocation limit); the warm-up is skipped, the weights load the same way.
f97ef14 2026-10-29 20:10:53 -0500 Part 115b Amendment 1, before any data: the first launch was stopped 11 s in by the RAM guard (Windows' available memory fell from 9.5 GB to 1.5 GB while the shared loader staged the model in host memory), so the run loads the weights straight to the A770 (--direct-load); same weights, dtype and device, both gates unchanged.
091c8bb 2026-10-29 20:09:00 -0500 Register Part 115b (Procedure v1), Elise's request after Part 115's cold check: do the neuron sets that take Q off Qwen3.5-4B's name after 'I am **' (Part 114's gradient ranking, Part 116's direct-write ranking) also take it off in a plain-text answer? Three openings with a plain-text instruction: 'My name is' (target ' Q'), 'My name is "' (target 'Q', no **) and 'My name is **' (target 'Q'); the baseline probe (three passes, results/part115b/probe115b.json) shows each target first on 16 of 16. Seven predictions, two gates. Also tools/register_public.py: a section stops at the next part, a part already public leaves PREDICTIONS.md alone, and set-aside files come back if the merge-back fails (46 tool tests; reverting the section fix fails the amendment test).
```

The count of held-out phrasings, the number of ordinary probes, and the generation count in the README are taken from the raw result and design files; the one-liner above records their source with the other run details.
