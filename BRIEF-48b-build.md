# Build brief: part48b.py (Luna) — code and CPU tests only, no GPU run

Implement PART48B-DESIGN.md exactly, as amended (amendments 1-4; where they differ, the later amendment wins).
Reuse part48.py's model loading, templates, continuation sets and item score (import it; do not copy-edit it).
- Stimuli: the 10 pairs of amendment 3 (nakolu/nacolu, rukuno/rucuno, rulaku/rulacu, ruluku/rulucu, rumuko/rumuco,
  tokamo/tocamo, tokuna/tocuna, toraku/toracu, totuko/totuco, tukulu/tuculu) plus the single letters k and c.
- Per model, per pair: d = score(k twin) - score(c twin) for BOTH scores (Part 48's continuation sets, and the
  letter-matched sets without "spiky" and "curvy"); per-template d; token counts per twin; mean surprisal per twin in
  the neutral context 'The word is "{w}".'; total continuation mass per item; first-token unigram prior per twin.
- Statistics: sign-flip permutation test on per-pair d (10,000 flips, two-sided, fixed seed) and a bootstrap 95% CI on
  median d (10,000 resamples, fixed seed), primary; the conservative sign test (9 of 10); per-template sign tests;
  Spearman of d with the surprisal gap and the token-count gap; the equal-token-count subset and the near-word
  exclusion subset (amendment 1 lists the pairs; apply it to those of the 10 that it names, none if none) reported
  beside the full set.
- CLI: --models, --device, --dtype (fp32 default), --out; one JSON per model, the same header fields as part48.py.
- Tests (test_part48b.py, tiny model, CPU): the statistics on hand-computed fixtures (a known permutation p, a known
  CI), the letter-matched sets exclude exactly spiky/curvy, d's sign convention (positive = k twin sharper),
  determinism under the fixed seeds. A sabotage flipping d's sign must red a test.
Commit nothing. Report REPORT-48b-build.md ending SUMMARY / FINDINGS / BLOCKERS.
