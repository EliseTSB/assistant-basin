# Part 16b build report

## SUMMARY

Built the Part 16b validation and blind-judging scripts, plus model-free coverage for the specified controls, hooks, seeds, judging, and intervals. The GPU validation and judge calls were not run.

## FINDINGS

- `validate16b.py` reads the saved 64-neuron set and its held-out questions, constructs unchanged, zero, mean-held, early, late, and matched random conditions, generates five replies per question, and measures neutral-text loss on target tokens 1024–2047.
- Mean hooks preserve the `[batch, sequence, width]` activation shape. The added multi-token hook tests guard against the Part 16 batch-axis regression.
- `judge16b.py` shuffles replies before assigning blind IDs, sends only question and reply text for labeling, maps the two labels back by blind ID, reports counts and Wilson intervals, and prints scores for 16b.1–16b.7. It uses the shared transport extracted from `judge.py`, with `gpt-6-luna` through its supported CLIProxyAPI route by default.
- Verification: 16 tests passed (7 new Part 16b tests and 9 existing `test_causal_search` tests); `py_compile` and `git diff --check` passed.

## GPU EXECUTION STEPS (NOT RUN)

Run these from the worktree root on the Windows GPU host:

```powershell
python validate16b.py --device xpu --smoke --search results/part16/search.json --output out\part16b\smoke-validate.json
python validate16b.py --device xpu --search results/part16/search.json --output out\part16b\validate.json
```

The smoke invocation runs two held-out questions and two conditions, for 20 replies. The full invocation runs 14 conditions, 10 questions, and 5 replies per question, for 700 replies.

## GPU TIME ESTIMATE

Part 16 took about 220 seconds for 80 greedy 40-token generations on the A770. Linear scaling to 700 replies capped at 48 tokens is about 39 minutes before sampling overhead, neutral-loss forwards, and model startup. Estimate roughly 45–60 minutes for the full validation. The smoke run should take a few minutes including model startup. These are estimates; neither run was performed here.

## BLOCKERS

None. GPU integration and the external judge route remain unverified because the requested GPU lines were not run.
