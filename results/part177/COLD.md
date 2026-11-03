# Cold check of Part 177

## Verdict

The arithmetic holds. I recomputed the gates, the cohorts, the literal-hit counts, the first-hit shares, the onset and pair counts for 177.3 to 177.6, the box cells, the Gate 4 kappa and the figure from the pinned inputs, with my own code. Every number I tried matched, within rounding. The verdicts follow from the registered bars, with two readings that need saying out loud: 177.W3 is "failed" only under the reading that drops null answers, and 177.W1 is "inconclusive" under both readings of early hits. The trouble is in the prose. The first quotation from MIT's post is cut and moved to a different experiment, and it drops the post's own qualifier. The headline's "right answer" and "67 of 173" carry more than the literal-match-plus-reader measure supports, and the README does not say what "target" means in its first paragraph. Amendment 1 cites the registered Reading rules for a denominator they do not state. The zero-count intervals are degenerate, the windows file is published without the CC BY 4.0 attribution its source requires, and the "equal exposure" wording in 177.6 describes a selection that the edit itself shapes. The verdict is that the work is sound enough to keep, but the README must be corrected before the title and first paragraph stand.

## Findings

1. **high** (line 3, "MIT's post says its edit 'could remove an obstacle that hid an ability.'"; same quote in line 10). The post's sentence, as archived at rikiyat.github.io/blog/weight-space-programming/, reads: "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked." It belongs to the additional experiment on Gemma 3 270M, not to the Qwen3.5-0.8B edit. The README drops the clause that limits the claim, and it attaches the sentence to the Qwen edit. The Qwen-specific sentence in the post is "it lets the model finish answers it could already work out." Fix: quote the full Gemma sentence with its scope, quote the Qwen sentence separately, and change the design's registered quotation with a dated amendment note.

2. **high** (title and first paragraph; "67 of 173... had already stated the right answer"). The measure is a literal string match on the gold value, plus a reader's `asserted` label on up to three windows per trace. "Right answer" is more than that. The 173 also silently excludes the 14 traces whose gold appears in the prompt, and the first paragraph says "target traces" without defining them. Fix: retitle as "...had a window that a reader judged to state the gold value as the result", define T in the first paragraph ("capped traces whose gold does not appear in the problem text, 173 of 187"), and state that the literal rule can miss equal forms.

3. **medium** (177.W3 row and the "failed" verdict; Reading rules, "Weaver ruled"). The registered prediction says "at least a third have a `pred_first` that does not match the gold." A null `pred_first` does not match the gold on the plain wording. Under that wording the count is 113/115 (98.3%), which clears the bar, and 177.W3 would be "held." Reading (a), which counts only non-null wrong boxes, gives 0/115 and "failed." The ruling is in the registration commit, so it is not post hoc. Still, the verdict depends on it. Fix: show both verdicts in the prediction table, for example "failed under reading (a) as registered; held under the literal wording (b)," and say which one the headline uses.

4. **medium** (Amendment 1, point 1: "every trace in `C` stays in the denominator, as the Reading rules say"). The registered Reading rules say the opposite: "a trace counts if a hit remains, for real and shuffled alike." The amendment therefore changes the registered rule and describes the change as a restatement. I checked the verdict under three denominators. All C gives a late gap of 31.6 points. Traces with a real hit left, paired on the same set, give 73.8 points. The union of traces with a hit left gives 67.8 points. Every version leaves the claim unheld, and the falsifier does not fire because the all-hit early share is 10.5%. So the verdict is unchanged, but the amendment has to say it changes the rule. Fix: write "Amendment 1 changes the registered sentence 'a trace counts if a hit remains' to 'every trace in C stays in the denominator,' and the verdict is the same under the old wording (73.8 and 67.8 points)."

5. **medium** (177.6 row and the "equal exposure" paragraph, and the caption on figure panel 3). The 116 pairs are the pairs where both texts ran past 50,000 characters. The p50c member of each pair was selected on its own length, and that length is an outcome of the edit. The "equal exposure" wording therefore claims more than the table shows. The stored p50c traces in this subset are the ones that did not finish early. Fix: say "both texts ran past 50,000 characters, a selection on post-edit length," and report the rate over all 200 pairs with a stated rule for the short texts, or state that it is not reported.

6. **medium** (385 rows of 401–829-character excerpts from the MIT logs, and the problem text). The logs repository declares CC BY 4.0 for its logs. The README has no attribution and no link to the licence. The MATH-500 problem text carries its own terms, which the README does not state. The brief's test is met: excerpts are published in a committed file. Fix: add a source line, "Excerpts: lm_understand-logs at commit 4d09ba6, CC BY 4.0, Takehi et al.," and the terms of the MATH-500 source, to the README and to the windows file's directory.

7. **medium** (Gate 4 margin and the "0 of 88" claim; the 0 in 177.1 and 177.2; 177.W2 and 177.W3 reading (a)). The bootstrap intervals for zero counts are printed as [0.0, 0.0]. A degenerate interval is not an interval. Exact upper 95% bounds are: 0/88 windows, 4.1%; 0/173 traces, 2.1%; 0/98 windows, 3.7%; 0/115 traces, 3.2%. Fix: report the Clopper–Pearson bound beside each zero.

8. **medium** (177.W2 row, "0/98", against the Gate 4 section and table, "0/88"). Two denominators are given for the same quantity. The 98 includes 10 windows from C-minus-T traces, which the reader labelled and which the table leaves out. The reader-error count of 385 includes them too. Fix: state that 88 is the T subset and 98 is all C shuffled windows, add the 10 to the table or say why they are excluded, and make W2's denominator match the one its bar names.

9. **low** (onset paragraph: "Across all `C` traces with an onset, an asserted gold preceded it in 49/168 (29.2%)"). 168 is the count of T traces with an onset. The count of all C traces with an onset is 182, so the C-wide figure is 49/182 = 26.9%. The number is right for T and the label is wrong. Fix: say "T traces" or change the denominator to 182.

10. **low** ("0 of 88 is the control"; shuffled values). 97 of 98 shuffled values are integers. The most common are 2 (19), 3 (17), 4 (12) and 1 (12), so most shuffled hits are single digits inside arithmetic. Real windows are also mostly short (165 of 287 values are two characters or fewer), so the digit-length matching is balanced. The 0 therefore shows that the reader does not call incidental digits "asserted." It does not show that a donor value could not be stated as an answer, because a stated donor-value answer is rare by construction. Fix: state what the control tests, and report the value distribution beside the zero.

11. **low** ("asserted" headline; the 50-window blind read). I read 30 windows labelled `asserted` and 20 labelled `intermediate`, from my own seeded sample, `random.Random(20261102)` over the real windows. My labels agree with the reader on 47 of 50. The three disagreements are windows 21, 26 and 44, all labelled `intermediate` by the reader and `asserted` by me. Each is borderline. Window 21 is one line repeated about fifty times that states "3 trees weigh 1 square," which is the answer. Windows 26 and 44 are mid-check with a doubt on the next line. On the asserted side, six windows state the value and then doubt it or the setup in the next line (windows 5, 7, 20, 34, 41, 43), and five sit inside a repeated loop (0, 13, 15, 28, 42). The criterion as written counts all of them, so the labels are defensible, but the README should say that `asserted` sometimes comes before a doubt and sometimes inside a loop. I would change no label with confidence.

12. **low** (Definitions, "a bare integer or short token must stand alone, not inside a longer number"). The code also treats a leading `+` or `-` as part of a longer number, and a following `.` plus digit as well. Without those clauses the design's prose gives 23 of 187 in-prompt traces, not 14, so the registered rule cannot be reproduced from the prose alone. I reproduced 14, 173, 115 and 101 only with the code's clauses. The non-numeric golds use plain substring matching with no boundary, so a gold such as `2\sqrt{3}` can match inside `12\sqrt{3}`. Fix: write the sign, decimal and substring clauses into the design text.

13. **low** (reader paragraph). Four DeepSeek labels were asked twice (attempts = 2), and the retry rule is not in the design or the README. The README also does not name the reader model, which the design gives as DeepSeek V4.1 Flash, or the second reader, which is Haiku. Fix: state both models and the retry policy, and list the four retried ids.

14. **low** (token positions). The medians reproduce (652 and 1986). The quartiles do not match the README's interpolation method. I get Q1 337.5 and Q3 1489.5 for first hits, against 338 and 1486 in the README, and Q3 6020 for asserted windows, against 5905. The method is not stated. Fix: name the quantile method.

15. **low** (end-loop counts: 182/200 unedited and 36/200 p50c). The `verbatim_loop_end` rule comes from `mathqa.py`, which is not in the inputs, so I could not verify it. `repetitive_end` is rebuilt from a docstring that is not in the repository. Fix: say in the README that these two counts are unverified from the pinned inputs, and where the rebuilt rule's source text lives.

## Reproduced (no finding)

- Gate 1: 158/2000 and 962/2000 match the merged accuracies 0.079 and 0.481.
- Gate 2: 187/200 and 70/200 capped, using the pinned tokenizer and a 32,500-token threshold.
- Gate 3: 88/92 for p50c `correct[0]` with a literal hit.
- Cohorts: C = 187, T = 173, C-minus-T = 14, real literal hits 115 on C and 101 on T, with the scorer's rule.
- First-hit early shares: C 109/115 (94.8%), T 95/101 (94.1%), C-minus-T 14/14. All-hit early share 3923/37453 (10.5%).
- 177.3: 49/66 before onset. 177.4: 35/48. 177.5: 34 pairs, medians 0.1624 and 0.3179. 177.6: 111/116 and 79/116, difference 27.6 points.
- Gate 4: known windows 17/20, kappa 0.6217 on 38 pairs, real asserted 120/287 against shuffled 0/88, errors 0/385.
- Box cells: 113 of 115 null, 2 non-null and correct. 32 of 32 empty boxes sit near a format instruction. 35 box tokens across 25 traces.
- Sample comparisons: sample-0 35 against 241, any-of-4 77 against 320. Closed-thinking literal hits on T: 95/132.
- All 44 input hashes match the manifest. Both revisions match the design.
- Figure: panels 1 to 3 match the files.
- Limits paragraph carries every item in the design's Limits.

## Checks not done

- I did not read the prior-art papers or the Muse report.
- I did not verify the known-window expected labels beyond the fact that it was committed at 1e4bf3f, before the 18:26 registration.
- I did not check the post's quote about loops and the 93% to 33% figures against anything beyond the post.

## Count by severity

high 2, medium 6, low 7
