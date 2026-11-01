# Cold check: Part 170

## SUMMARY

Every README outcome number reproduces from `answers.jsonl`, and the push rises match my own bootstrap. The no-push findings rest on a plain-loop filter and on 3 to 7 loops per set, and some sentences say more than that allows.

## FINDINGS (most serious first)

1. README line 81: "The observed rise without a push is partly a consequence of selection." For identity prompts it is all of it. Over all 104 prompts b50 gives 3 loops against b0's 8 (−4.8 pp). The 92 eligible prompts give +3.3 pp, so the filter adds +8.1 pp. Correct wording: the identity b50 rise exists only among non-looping plain prompts.

2. Title: "under a push, blocking the plain answer raises loops; at 200 tokens it does so without one." The b200 half holds: +9.6 pp on all identity prompts (p=0.052) and +19.8 pp on all Q96 (p=0.001). The contrast with b50 is not shown: eligible b50 gives +3.3 pp (p=0.25) and +6.7 pp (p=0.07). Significant at 200 and not at 50 is a difference in significance. Because 170.2 failed, the Reading rules require the masking caveat in the same sentence as any ban result. The title has none.

3. README line 107: "Releasing the ban after 5 steps added nothing." This is a non-detection. Discordant pairs are 34 ended and 26 induced (sign p=0.37); the 95% interval is −6.0 to +2.1 pp.

4. README line 110: "[170.6] held," and "Weaver's falsifier was met." The bar passes any negative rise, and this one is −11.1%, so the pass does not show damage tracking blocks. The W2 falsifier's first clause is met; "sustained blocking is what matters" is untested.

5. README line 102: the b50 rise "levels off" at +13.0, +18.5, +18.75 pp. That holds only on the probability scale. On log-odds it is 1.94, 1.46, 0.85, falling steadily. The base climbs from 2.6% to 25%, so the flattening is partly scale.

6. README line 118: "The sign changes from negative to positive." Length-quartile differences are −3.3, +17.0, −7.0 and +7.5 pp on cells of 62 to 78 rows. The +4.0 pp is an average of mixed signs.

7. Hand-read attribution conflicts. README line 218 says Haiku 5.5 did the cold reads; `hand_read_judged.jsonl` and README line 184 name Claude (Opus 5.5), who registered the design, so the read was not blind. Only 2 of 11 no-push b50 loops were coherent.

8. Omissions. The design asked for mean share at each dose (71.1%, 62.6%, 44.5%); the README gives dose 1 only. Part 169's overlap count is missing. The Q96 entropy split (+6.7 and +6.8 pp) bears on Weaver's expectation and is not mentioned.

9. No noise floor for no-push. Gate 8 (0 of 384) covers only 48-row push batches. No-push runs in 5–6-row batches, and 86 of 200 b0 answers differ from Stage 0 plain.

10. Fragility. The eligible identity rise rests on three maker-kind prompts. Drop them and it is 0.0 pp. Drop the identity half and it is +5.8 pp. Q96's loops give +4.3 pp on fact prompts and +9.5 pp on how_to.

11. Exploratory I(d). Slot-only intervals treat D(0) as fixed, but D(0) rests on 0, 2 and 2 of 8 phrasings. b5 at dose 1 is −16.2 to −5.5 pp slot-only and −44.5 to +16.2 pp with phrasing clusters.

12. Scorer and figure. `_signed_ratio_draws` silently drops small-denominator draws. W1's falsifier is point-only, and the Q96 interval reaches 13.5 pp. In `loops170.png`, categorical ban sizes are joined by lines and P8 no-push is not plotted.

## HOLDS (reproduced)

- All README outcome cells, gate values, frozen counts (12 M96, 7 Q96).
- Push rises 13.0, 18.5, 18.75 pp; dose-1 interval [11.7, 25.5] pp.
- rel50 blocks at most five steps.
- 170.3 holds (43.75 against 37.5). 170.4 and 170.5 fail as registered.
- Dose-1 b50 rise survives dropping any one layer (min 16.4 pp), slot or phrasing.

## COULD NOT CHECK

- arXiv citations were not fetched.
- Gate 4 write norms were not recomputed.
- No model was run, so the batch-size cause of the 86 differences is untested.

DONE
