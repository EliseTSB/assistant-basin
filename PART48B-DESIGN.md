# Part 48b: same sound, different letter shape (k against c)

**Designed after seeing Part 48's results** (written 2026-10-21 13:17 CT). Part 48's spelled-out control kept
the sound as well as the letters, so it could not tell letters from sound (correction in PREDICTIONS.md 48.5). This part
holds the sound fixed and changes only the letter: k (an angular glyph) and c (a round one) both say /k/ before a, o, u.

## Size, first line
5 models x 6 templates x 32 words x 10 continuation words, teacher-forced, no generation: 960 prompts, about 9,600 short
forward passes in total, a few minutes per model on a 3090 in fp32. Smallest effect worth knowing: a paired item gap of
half Part 48's median sharp-minus-round gap per model; 16 pairs give a two-sided sign test p < 0.05 at 13 of 16.

## Stimuli (fixed here, before any run)
16 twin pairs, identical except every /k/ is spelled k in one twin and c in the other, vowels only a, o, u, other
consonants neutral (l, m, n, r, t):
kalo/calo, koru/coru, kuma/cuma, taku/tacu, roka/roca, nuko/nuco, kolu/colu, maku/macu,
kanu/canu, luko/luco, kotu/cotu, raku/racu, kumo/cumo, toka/toca, komala/comala, nakotu/nacotu.
(roca and the coma in comala are near words; kept and named.) Token splits recorded per model.
Templates, continuation sets and readout: exactly Part 48's (same code, same item score). Fp32, same pinned revisions.

## Statistic
Per pair d = score(k twin) - score(c twin), positive = k twin sharper. Per model: median d and a two-sided sign test over
16 pairs; the Part 48 median sharp-minus-round item gap is the yardstick.

## Predictions (Claude, before any run)
- 48b.1 In at least 3 of 5 models the k twins score sharper (sign test p < 0.05).
- 48b.2 In every model the median |d| is smaller than half that model's Part 48 sharp-minus-round median gap (letters
  shift it less than sound does).
- 48b.3 Pythia-1.4B shows 48b.1's direction (p < 0.05).
## What it cannot tell apart
A k effect could come from glyph shape OR from k being the letter of "kiki" in text about this very effect; both are
letters rather than sound, which is all this part asks.

## Amendment 1 (2026-10-21 13:23 CT, before any run): Prime's review (council orchestrator-part48b-review-2026-10-21 #3)
- **What it isolates, renamed:** letter identity against phoneme, not glyph shape. A k effect can come from glyph shape,
  k/c letter statistics (ka/ko/ku are rarer than ca/co/cu), letter names ("kay" vs "see"), a Germanic-vs-Romance cue, or
  the kiki tradition; this part cannot separate those, and the pair-level analyses below are the partial handle.
- **Letter-matched second score:** the sharp set contains "spiky" (k) and the round set "curvy" (c). A second item score
  drops both (sharp: sharp, pointy, jagged, angular; round: round, smooth, soft, rounded) and is reported beside the first.
- **Primary statistic:** a paired sign-flip permutation test on per-pair d (10,000 flips, two-sided) with a bootstrap 95%
  CI on the median d, per model, plus a pooled estimate; the 16-pair sign test stays as 48b.1's pre-registered
  conservative test. Per-template d and a per-template sign test (6 templates) are reported.
- **Token counts:** pairs whose twins split into different token counts in a model are excluded from that model's
  primary test; both (all pairs, equal-count pairs) are reported.
- **Near-words:** sensitivity with roca/toca, comala/komala, calo/kalo, cuma/kuma, raku/racu, kumo/cumo and coma-like
  pairs all excluded, reported beside the full set.
- **Fluency and mass:** each twin's mean surprisal under a neutral context ("The word is \"{w}\".") and the pair's gap;
  Spearman of d with that gap and with the token-count gap; total continuation mass per item (Part 34's lesson).
- This is a pilot: the deliverable is the per-model effect size with its CI, not only a p.
