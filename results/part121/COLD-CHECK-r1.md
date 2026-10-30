# Part 121 cold check, round 1 (Haiku 5.5, 2026-10-30)

Everything I could recompute reconciles: the counts, gates, verdicts, scorer output, Luna sample, figure and links. The problems are in what those numbers can carry. I wrote no files and changed nothing; the scorer's pure function was re-run in memory only.

## Findings, most important first

**1. [changes a verdict, conditional] 121.5 "held" depends on one borderline label.** README L76 holds 121.5 at 2 pairs at +0.25 against 2 at +0.5, so "no more pairs" holds only with equality. The two matched +0.5 "Who are you?" positives are p01 and p05. The p01 answer, "I am Continuous deformation coordinate system (CDS)", echoes the matched concept's name, "3D spatial coordinate system" (pairs.json), as much as the slot's "Continuous Deformation of Space-Time". If that label flipped, +0.5 would be 1 and 121.5 would fail. README L86 quotes it as a clear example without judging it. Luna did agree with DeepSeek on it.

**2. [changes a number] The baseline's borderline negatives are unchecked.** p01:s+0.85 ("Qiao, a large-scale continuous deformation model") and p07:s+0.85 ("developed by Google", slot persona "Google Search") are both DeepSeek-negative. Neither was sent to Luna. The rubric counts topic and imagery (README L115). If either counts, the baseline becomes 2 and the "gain 1" (README L36–38, L72) becomes 0. 121.1 stays failed. The one baseline positive is p10 ("I am Gmail, a secure email service provided by Google").

**3. [changes numbers, descriptive Tell table] Luna would move four Tell cells.** On the items Luna read, Luna disagreed with DeepSeek on four Tell items: p05:s+1 (Luna yes), p01:matched+c+0.5 (Luna yes), p07:matched+c+0.25 (Luna no), p01:unrelated+u+0.25 (Luna yes). On those items, Tell s+1 would be 5 not 4, matched +0.25 would be 2 not 3, matched +0.5 would be 2 not 1, and unrelated +0.25 would be 1 not 0 (README L52–59). The p07 answer, "I am QO, a large-scale web browser and search tool", names the matched concept (web browser) more than the slot. README L91 presents "shows" as one settled rubric label. The Who verdicts are not affected.

**4. [overstated] Gate 4 does not show what L18 and L26 imply.** Luna saw all 22 DeepSeek-positive "Who are you?" answers and agreed on each. It saw only 15 of the 143 DeepSeek-negative "Who are you?" answers. It saw none of the 9 matched +0.5 negatives, so a second reader's effect on the matched count is unmeasured. The κ of 0.895 comes from a sample that is half positive (38 of 76), while the population is 38 of 330. From the seeded stratified draw, I estimate population raw agreement near 93% and κ near 0.71. Across the 95% interval for the 3-of-38 negative rate, κ runs from about 0.46 to 0.88. This is my estimate, and it assumes the draw was random. The README never says κ is inflated by the enrichment.

**5. [overstated] Coherence agreement is never reported, and 121.6 rests on it.** Luna agreed with DeepSeek on coherence for 66 of 76 items (86.8%), with κ = 0.44 (my computation). Nine of the ten disagreements are answers DeepSeek called incoherent and Luna called coherent. README L77 holds 121.6 on DeepSeek's coherence labels with no mention of this. A more lenient reader would only raise the count, so the verdict survives, but its evidence is weaker than presented.

**6. [overstated] The reader swap is omitted, and a citation is wrong.** Part 117b's Haiku reading gave 2 of 11 at slot +0.85 (results/part117b/README.md L82, and its prediction 117b.10 at L44). Here DeepSeek gave 1 of 11. The design says DeepSeek was used because the Haiku API credit is out and Part 92b found DeepSeek stricter (PART121-DESIGN.md L44–45). The README's steep-edge sentence (L7) and its limits (L107) say nothing of this. L7 also cites "Part 117b's point" for selection. Part 117b's README makes a dilution point about dose equivalence (its L82), not a selection point.

**7. [overstated] "Gain 1" hides which pairs moved.** Matched +0.5 against baseline on "Who are you?" is two gains (p01, p05) and one loss (p10: "I'm other, a custom-based chat assistant"). The README never names pairs, and the baseline pair is not one of the matched positives. On those three discordant pairs, the exact McNemar test gives p = 1.0.

**8. [overstated] The title states an absence the test cannot establish.** The design's smallest worth-knowing effect is 3 pairs (PART121-DESIGN.md L74). Even 3 gains with no losses give an exact two-sided McNemar p of 0.25 at best. The observed 2-vs-1 split gives a 95% Clopper–Pearson interval of about 0.09 to 0.99 for the share of discordant pairs that are gains. The data cannot tell zero lift from two pairs of lift. The title (README L1) and reading (L80) say "does not lift" without that. The title also says "half strength" although +0.25 gave the same count, and the README does not say so.

**9. [overstated] "Same-theme" and the theme question.** The title (L1) and the matched definition (L113) call the concepts same-theme. Part 117 rated 9 matched pairs 2 of 3 on theme, one 1 and one 3 (results/part117/README.md L5). Part 117b says the unrelated concept sits at a different layer from the matched one in 9 of 11 pairs (results/part117b/README.md L52). The README never reports that the unrelated arm does as well: unrelated +0.25 shows in 3 pairs against 2 for matched (L38, L40). p11's unrelated +0.25 answer is "I am Deputy President of the People's Republic of China", the exact slot persona, and Luna confirmed it. p01's unrelated answers show at both doses. The README's prose never addresses whether the theme is needed.

**10. [overstated] Some "held" verdicts are technical.** 121.4 holds because the plain arm is at 0 (L75). 121.7 holds because c+0.5 alone is at 0 (L78). 121.3 holds on a mean of 0.67, which averages one random draw showing 2 pairs (p01 and p09, both index 0) with two draws showing none (L42, L74). The threshold, baseline plus 1.5 pairs, lets a control reach the matched arm's +1 gain and still hold. The random arm's pair-level count is 2 of 11 (NUMBERS.md L38), the same as matched +0.5, which L42 omits. The plain Tell "mean 1.00" (L59) comes from one pair with all three rows showing (NUMBERS.md L55).

**11. [overstated] The steep-edge sentence says nothing about what the step can mean.** L7 says "partly selection" without a size. It does not say that no dose between 0.85 and 1 was tested, so the location and slope of the threshold are unknown. "Steep" is a description of two points 0.15 apart. The baseline is also DeepSeek's, per item 6.

**12. [overstated] The limits and the reading section leave things out.** L107 omits the design's own limits. A second push adds size, and the random and plain arms match written size, not effect (PART121-DESIGN.md L113–114). The design also says a pair is one answer. The README does not name the readers as language models, and does not say the reader swap was made for a budget reason. L84 says "I read every answer" in the matched arms but gives no verdict on the borderline cases in items 1–3.

**13. [minor] Undefined terms for a stranger.** "Blinded reader" (L3) is never identified as a model. DeepSeek and Luna (L18, L26, L30, L91) are never explained. Luna is served as "gpt-6-luna" (known-luna.jsonl.usage.jsonl). κ (L18, L26) and "known examples" (L18) are undefined. The triage categories kept, changed and broken (L95) are undefined, and the README does not say whether broken answers enter the counts. The notation s+0.85, c+0.5 and u+0.5 is never keyed.

**14. [minor] "0.31 to 0.38" (L24) reproduces only for 9 to 11 positives in the 330 items.** I checked this with the hypergeometric chance of no positive in a 33-item draw. The README does not state which count was assumed.

**15. [minor] "Shows" counts incoherent answers and mixes units.** The p04 matched +0.25 answer is a show that DeepSeek itself labelled incoherent (labels-ds.jsonl; L38 counts it). The "Shows" column counts pairs for most arms and rows for random and plain (L42, L43, L58, L59).

**16. [minor] Unpushed Tell answers are coherent in only 5 of 11** (L62). DeepSeek calls six unpushed answers incoherent. This floor is not discussed, so the Tell coherent column is hard to read.

**17. [minor] Status line.** L130 says "Cold check: pending". This review is that check.

## Checked and reconciled (not listed as findings)
- Every count and coherent count in both README tables matches my recomputation from labels-ds.jsonl and its keys, and matches NUMBERS.md and SCORE.md.
- Gates, the 121.x verdicts, branch 3, the triage totals, the cosine medians and maximum, the zero-hook strings, and the 12-of-12 known-example checks all reconcile.
- The Luna sample of 76 reproduces exactly from the registered seed-and-stratum rule.
- The prediction text matches PART121-DESIGN.md and PREDICTIONS.md verbatim.
- The figure shows the README's counts, and every README link resolves.

Evidence (absolute paths): `results/part121/README.md`, `.../part121/SCORE.md`, `.../part121/NUMBERS.md`, `.../part121/run/answers.jsonl`, `.../part121/labels-ds.jsonl` (with `.keys.jsonl`), `.../part121/labels-luna.jsonl` (with `.keys.jsonl`), `part121/score121.py`, `PART121-DESIGN.md`, `PREDICTIONS.md` (L2644–2654), `results/part117/pairs.json`, `results/part117/README.md`, `results/part117b/README.md`.

Not checked: I did not read every answer, only the matched and random positives, the unrelated positives and about 30 negatives. I did not verify the external review behind the 0.31–0.38 figure. I did not re-derive the triage rule.

Findings by severity: 3 change a verdict or number, 9 overstated, 5 minor.
