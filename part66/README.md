# Part 66 run

Run the pilot first, review its answers and scoring, then launch the preregistered full run:

```sh
python part66/part66.py --phase pilot --out results/part66/pilot --device xpu
python part66/score66.py results/part66/pilot/answers.jsonl
python part66/part66.py --phase full --out results/part66/full --device xpu
python part66/score66.py results/part66/full/answers.jsonl
```

Use `--device cuda|cpu`, `--batch N`, and `--max-gpu-gb G` as needed. The runner uses the Part 25 Qwen3.5-4B revision, bf16, no-system chat prompts, Part 25's matched random-set sampler and row-wise down-projection intervention, and the design's sampling settings. Each output directory contains append-only `answers.jsonl` (one generated answer with condition, units, question and kind, sample, seed, text, token count and stop reason) and `run.json` (model/revision, script SHA, device/dtype/batch, timing, throughput, memory when available, exact conditions, and a note on batched RNG limits). Re-running a phase in the same output directory skips already recorded condition/question/sample rows.

The scorer writes `score.json` with per-condition rates, per-question counts, direct/indirect/positive/guard summaries, random-set rates, and 66.x verdicts; `score.md` is a compact verdict table; `hand-read.md` lists up to the first 30 held-out target answers matched by the Google-family regex, with a blank hand-verdict column. The lexical rate is unchanged by the requested human hand read.

Pilot scope: three held-out phrasings x two samples for unchanged, target and the first three random conditions, plus four positive-control samples in unchanged and target. Pilot output is for timing and scorer inspection and is not part of the full result.
