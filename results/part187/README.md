# Part 187: Part 140's result holds on the eager attention kernel (holding a persona push gives the most coherent persona answers; releasing it after 8 tokens mostly ends the persona)

Part 140 asked which way of delivering a persona push gives the most joint success (the answer is in the persona and
coherent) on Qwen3.5-4B, over 60 selected units (Set P), 60 random directions (Set C) and 60 more (Set L), in six arms:
A held greedy, B held sampled, C released after 8 tokens, D released after 24, E with a repetition penalty, F
norm-preserved. It ran on the A770 in bf16 with transformers' default fused SDPA kernel, which Part 178 found leaving
the CPU float32 path after about 64 keys; most of each 80-token answer its readers judged was written past that point.
Here the same code, units, seeds and dose ran on the eager kernel (load record: `attn_implementation` eager, requested
eager), at batch 8 where Part 140 used 16 (arm B stays at batch 1 inside `run140.py`), and the answers were read again
by blind readers with Part 140's brief: DeepSeek V4.1 Flash on all 3,840, GPT-6 Luna on a 10% sample. Registered in
`PART187-DESIGN.md` (896c9f1, Size line b9be8dd) before any eager answer.

| Joint success (unit level) | Eager (this part) | SDPA (Part 140, published) |
| --- | ---: | ---: |
| A held, greedy: Set P | 48.3% | 48.3% |
| B held, sampled: Set P | 37.6% | 35.1% |
| C released after 8 tokens: Set P | 7.1% | 5.8% |
| D released after 24: Set P | 40.0% | 37.9% |
| E repetition penalty: Set P | 44.2% | 43.3% |
| F norm-preserved: Set P | 0.0% | 0.0% |
| Highest Set C rate in any arm | 3.9% (B) | 6.1% (B) |

Part 140's claim stands on eager. Holding the push gives the most joint success, releasing it after 8 tokens mostly ends
the persona, and random directions stay low. Part 140's own scorer gives the same nine verdicts on both kernels (140.1
to 140.4, 140.6, 140.8 and 140.5's arm C fail; 140.5's arm A and 140.7 hold). Every Set P rate moved by at most 2.5
points. The answer texts changed much more than the rates: 3,698 of the 3,840 paired answers differ (arm B, sampled, 1,436
of 1,440; the greedy arms 444 to 462 of 480), the same pattern as Parts 183 to 186.

## Predictions

All six held (`PREDICTIONS.md`, rows 187.1 to 187.6):

- 187.1 held: arm A's Set P joint success is 48.3%, inside 36.0% to 60.0%.
- 187.2 held: arm C's Set P is 7.1%, at most 15.0%.
- 187.3 held: Set C is at most 3.9% in every arm, at most 10.0%.
- 187.4 held and 187.5 held: no published verdict flipped (0 of 9).
- 187.6 held: arm F's Set P is 0.0%, at most 5.0%.

## Gates

- Gate 1 (same-batch plain and zero-add-hook rows): passed.
- Gate 2 (arm A reproduces the first 10 tokens): 23 of 24, passed (bar 80%). Note: `score140.py` recomputes this gate
  with the tokenizer and writes `gate2-result.json` beside the answers. A first scoring run with a Python that had no
  transformers wrote a failed record (`ModuleNotFoundError`) over the copy; the rerun with the project venv recomputed 23
  of 24, the same as the record the A770 run wrote (`/mnt/c/.../results/part187/main/gate2-result.json`, compared).
- Reader gates on Part 140's 12 known items: DeepSeek 12 of 12, Luna 12 of 12 (joint kappa 1.0); reader agreement on the
  384-answer sample, kappa 0.82 (bar 0.60).

## How it ran

- Generation: `tools/launch_arc.py 187` on the A770, 10:47 to 12:23 CT, exit 0. `load.json` records `run140_rc: null`
  because `run140.main()` returns nothing; `run187.py` returns 0 for that, which the progress log shows.
- Reader gates and labels ran locally (they are API calls only), with Part 140's `label140.py` and output in
  `results/part187/labels`; the run gates were copied to `results/part187/` where the labeller looks for them.
- Score: `part140/score140.py`, `results/part187/score.json`.

## Limits

- Kernel and batch size change together (batch 8 here, 16 in Part 140), as in Parts 184 to 186; they are not separated.
- New readers read new texts: reader differences and text differences are mixed in the rate changes.
- Eager itself is not the CPU float32 path on every prompt (Part 183: it leaves it on 1 to 3 of 8 prompts, where the
  reference's top two tokens were close).

## Files

`run187.py`, `steps.txt`, `PART187-DESIGN.md`; `results/part187/main/` (answers, gates, load record, timing),
`results/part187/labels/` (reader gates, labels, usage, logs), `results/part187/score.json`.
