# Part 70

Run the sampled generation study from the repository root:

```bash
python part70/part70.py --out results/part70/full --matched results/part68/b/matched.json --device xpu --batch 25 --max-gpu-gb 14
```

Resume by rerunning the same command and output path; completed `(condition, question_index, sample)` rows in `answers.jsonl` are skipped. Score the collected answers with:

```bash
python part70/score70.py --dir results/part70/full
```

The scorer writes `score70.json` and `score70.md` beside `answers.jsonl`.
