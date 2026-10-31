# Part 158: with every way of spelling its maker closed, who does the model name?

Procedure: v1
Registered 2026-10-31 15:59 CT by Claude (Opus 5.5), before any data. It follows Part 157, whose Gate 3 failed: banned from exact spellings, the model wrote TongYi, Tongyiqianwen, Alipay, Alibab and 通義千問, so its ladder never reached another maker. Elise, 2026-10-31 (Discord message 1566108806755127377): "Did we find any that moves the maker and name across several questions (eg. who are you and who made you), or even just reliably switches one?"

## Question

Part 154's lead was that Google may simply be next in line once the real maker is gone. Part 157 could not test that, because a ban on spellings does not close the maker: the model went round it. Close the routes at the token level instead. Who does it name then, and with L27 N3325 zeroed?

## Prior art and existing data

- Closest published work: none found: searched PRIOR-ART.md for constrained decoding used to read a model's ranking of its own makers.
- Our own data: Part 157, 18 of 24 answers still first-named the Alibaba family at every level, with 0 naming Google. Part 154, L22 N6997's Google answers moved to Microsoft with L27 N3325 zeroed. Parts 68 to 73, the fallback to Google with the four maker neurons zeroed.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, thinking off, no push on any row. Part 153's 24 maker phrasings. 120 new tokens, greedy.

The ban is `suppress_tokens` in `generate`. A token is banned when its decoded text, stripped of spaces and asterisks and casefolded, starts with one of `ali`, `tong`, `qian`, `qwen`, `qw`, or contains any of 阿, 里, 通, 义, 義, 千, 问, 問. These are the routes Part 157 took, fixed here from its answers. The whole vocabulary is scanned on the CPU and the count of banned tokens is printed.

Arms, all at this ban: plain; L27 N3325 zeroed at every position, Part 154's hook; L27 N4991 zeroed the same way. Controls: a ban of the same number of tokens drawn at random with seed 158, excluding any token that occurs in the level 0 answers of Part 157, must return Part 157's level 0 answers on at least 22 of 24 phrasings.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing control, the random ban; matched controls, L27 N4991 zeroed, one of Part 154's four matched neurons; broken answers counted.

Not used: both directions, poles and a dose ladder, because nothing is pushed. Not used: guards, because this ban reaches no guard's answer except by accident, and a blind reader, because the rule is Part 157's family rule, and every plain-arm answer is read by hand.

Gate 1: 96 answers, all non-empty. Gate 2: the random ban matches Part 157's level 0 on at least 22 of 24. Gate 3: at most 2 of 24 plain-arm answers contain any Part 157 Alibaba family spelling, by case-folded substring, or `tongy`, `alipay`, `alibab`. A failure of Gate 3 ends this line of test, since two designs will then have failed to close the maker.

## Size

- Smallest effect worth knowing: one family named on 6 of 24 phrasings.
- Total: 96 generations, about 3 minutes on the A770.
- Pilot: not needed: under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: GPU
- Measured: n/a: under 5 minutes.
- Levers: none needed.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 158.1 | In the plain arm, Google is the family named first on more phrasings than any other family | low | Claude |
| 158.2 | In the plain arm, at most 8 of 24 answers name any listed family, the rest inventing a maker or naming none | medium | Claude |
| 158.3 | Google answers with L27 N3325 zeroed are at most half the plain arm's, and with L27 N4991 zeroed at least 75%, if the plain arm has at least 6 | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 158 (registered ...)".

## Reading rules

`part158/score158.py` computes every number, reusing Part 157's family rule. Verdicts use point estimates. All answers are published, and every plain-arm answer is read by hand and sorted as a listed family, an invented maker, or no maker. If 158.1 fails, Google is not reported as next in line by this test. 158.3 reads untestable if the plain arm has fewer than 6.

## Limits

A token ban removes many unrelated words, so a change in an answer can come from the lost word and not from the maker. One model, greedy, one answer per phrasing. The prefixes were chosen from Part 157's answers on the same 24 phrasings.
