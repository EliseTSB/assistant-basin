# Part 219 pilot under Amendment 1 (R2m, mean-ablation), 2026-11-22

Means: `head-means.pt` (SHA-256 f783d08a…), from the 26 neutral reference prompts (screen A's 24 general questions plus
Part 99's q4 and q5), answered and teacher-forced by unedited Instruct on the A770, eager, bf16.

Run: R2m, the four heads (L0H10, L1H4, L2H6, L3H5) mean-ablated, 474 rows in 134.7 s, R0 reused from the registered
pilot. The first R2m attempt stopped on its intervention receipt (Amendment 1b, Vesper PASS #13); this fresh run
passes the amended receipt and its rows are byte-identical to the stopped attempt's (`cmp`), which were never read.

Labels: Part 92's Haiku reader through OpenCode Go, two passes (`labels-R2m.jsonl`, `labels2-R2m.jsonl`), and the
DeepSeek V4.1 Flash 10% sample (`labels-deepseek-R2m.jsonl`).

**Stopping rule (registered, Amendment 1): FIRED.** R2m leaves 8 of 21 slot answers coherent (needs at least 15),
`score-pilot-a1.json`. By the amendment, the ablation half ends here: 219.1m is not read, R3m to R7m do not run, and the
reported result is "these heads cannot be removed by zero- or mean-ablation without breaking answers". The transplant
arms (R1, R8, R9, R10, stage b) run as designed.

Read directly, and exploratory: unlike zeroing, mean-ablation leaves R2m's six unpushed answers ordinary (the usual
"As an AI language model..." self-description, Paris, a sea poem). What breaks is the pushed answers: empty replies,
loops ("Hello, hello, ..."), off-topic fragments. With these four heads held at their average, Part 99's fixed pushes
derail the model where they did not in R0. This is a description of one run, not a registered finding.
