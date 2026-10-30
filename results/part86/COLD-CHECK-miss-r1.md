# Part 86 miss-rate section, cold check round 1

Reader: Claude Haiku 5.5 (a fresh subagent with none of this project's context), 2026-10-30, on the README section
"Triage miss rate (Amendment 2 item 1, run under Amendment 9)", the two README sentences that mention the miss sample,
and the matching NUMBERS.md block. Saved by Claude as returned, shortened, with local paths made repo-relative.

## BLOCKING

1. **The random-direction broken row drops the Wilson interval that Amendment 9 requires.** The row reads "38.7% (all
   read)"; the scorer output gives 38.7% (23.7% to 56.2%), and NUMBERS.md says triage-miss86.md is the README's table.
   Fix: "38.7% (23.7% to 56.2%; all 31 read)", with one sentence that the 31 are the whole population, so the interval
   is a binomial summary rather than a sampling interval.
2. **The estimates come from one reader with no agreement check, and the section does not say so.** All 660 usage rows
   show deepseek-v4.1-flash and `miss86/` holds no second label file. The DS-Luna check (kappa 0.814) covered changed
   answers only. The Wilson intervals are binomial and leave out reader error. Amendment 2 item 2 registered a second
   blind reader on a random 10% of all items (seed 8687), and the section takes no position on whether it reaches the
   miss items. Fix: say the sample was labelled once; write "estimates how many" and "were labelled coherent personas
   by the reader"; decide the item 2 question in writing, and if it applies, run a seeded 10% second read (about 63 of
   631) and report kappa before publishing.

## MINOR

1. "at most about 160 kept ones ... were missed": the 95% upper bound is 161.6 (NUMBERS.md prints 162), and none of 200
   seen means no miss was observed. Fix: "at most about 162 kept ones could have been missed (none of 200 seen; 95%
   upper bound)."
2. "(2.7 to 6.4 times across the Wilson ranges)" takes both samples' bounds at their extremes; it is a conservative
   range, not a 95% interval. The arithmetic (3.750, 2.669, 6.416) is right.
3. "The triage miss sample shows that every coherent-persona rate here is a lower bound": the lower-bound property comes
   from the design, "every" also covers conditional shares (250 of 519, 3 of 17) that bound nothing, and for random
   directions the section gives no total (point 54 against 42 counted; upper about 216).
4. "Triage marks a repeated phrase as broken" is incomplete: the rule is fewer than four words, or a three-word sequence
   repeated at least three times; two of the 96 coherent broken answers have under four words.
5. "Most broken answers that count" leaves "that count" undefined. Recount from the item texts: of the 96 broken
   answers labelled coherent personas, 85 open with "I am" or "I'm", 8 with "I are", 3 with other "I" clauses.
6. The draw and blinding are not described: the code takes one answer from each of the 17 layers first, then fills the
   rest from the pooled candidates; items are shuffled, with fresh ids and no layer, unit, arm or triage class.
7. Reader calls: 660 calls for 631 items; 30 stopped at the 8,000-token limit and 29 were retried; one label came from
   a limit-hit call and is class unclear.
8. The Files list does not link `part86/miss86.py` or `results/part86/miss86/`; the result files are untracked; the key
   is withheld, so readers can check labels but not sample membership, and the section does not say so.
9. Local paths outside the section: `part86/MISS-NOTES.md` names a local Python path, and the run script and its log
   carry local paths.

## Checked and correct

Every table count matches triage-miss86.json and .md; the other three Wilson intervals recompute; pool sizes re-derived
from the 165,376-row screen (real 147,469 kept, 5,950 broken, 3,253 changed; random 8,573, 31, 100); the scaled figures
and the 3.8, 2.7 and 6.4 factors; 631 items with only id and text, re-triaged as 400 kept and 231 broken; label_class
gives 99 coherent personas (3 + 84 + 0 + 12); the "Tongue" quote is verbatim, labelled a coherent persona, and triaged
broken; the full version-2 reading's 1,755 coherent personas = 1,713 + 42; the miss86 files are byte-identical to the
labeller's output; no local paths in the README, NUMBERS.md or `miss86/`.

Not checked: the key (not opened), individual labels' correctness, the test suites.

2 blocking, 9 minor.
