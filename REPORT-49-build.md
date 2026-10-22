# Part 49 build report

## SUMMARY

Built the Part 49 selection, positive push, negative push, scoring, blind-pack, and test paths. The original Part 38 open-prompt data was recovered from Git history and its SHA-256 matches the saved Part 38 run.

## FINDINGS

No Part 35 Qwen3-1.7B wizard screen ranking was present in this worktree's `results/`, the Part 35 build results, or `/home/elisepc/neuron-probe`. I ran the design fallback before any push. It ranks candidate units in layers 4–27 by the median maximum `down_proj` input on the 12 sorcerer-role wizard-token examples minus the median on the 12 setup-role examples. The saved selection is `data/part49-partners.json` (SHA-256 `1aa776d06ddbeb8ad0b405b51e969da71fe4dad5167fc2c21470b13ba7d7e28a`). P1–P5 are L27:946 (+240), L22:1610 (+104.3125), L27:1872 (+101.125), L27:677 (+46), and L27:4508 (+41.875). The file includes their median values, ratios, and the top 50 fallback candidates.

The open prompts match the Part 38 JSONL, and `data/part38-homographs-open.json` has the recorded SHA-256 `d88289e3ca4113a80bf36f79efbaed22209365a27c776a23bdd7c89af0f9b62d`. The open assay defaults to 15 arms × 64 answers; the amendment adds 12 negative-clamp arms × 32 answers using the eight prompts in `data/part49-wizard-prompts.json`. Both clamp one or two units at ±8 times each unit's own natural maximum on generated tokens only. Runs use row-specific stable seeds across arms and the copied `fastgen.py` batch generator. JSON scores include topic rate, Wilson 95% interval, P+W minus P deltas, and a broken-answer field. Broken rates remain null until blinded labels are supplied through `blind-pack` and `score`.

### Arc command

After committing `data/part49-partners.json`, run the full open and negative assays with:

```sh
/home/elisepc/neuron-probe/.venv/bin/python part49.py push --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device xpu --dtype bf16 --threads 4 --seed 3801 --batch 16 --out results/part49-xpu
```

### CPU smoke timings

Real cached Qwen3-1.7B, BF16, four CPU threads; two prompts, one sample each, `no_clamp,W` arms. Times are per-arm generation seconds; the batch-16 smoke had two rows, so it did not fill all 16 slots.

| Assay | Arm | Batch 1 | Batch 16 |
|---|---|---:|---:|
| Open prompts | no clamp | 26.61s | 21.39s |
| Open prompts | W | 15.68s | 15.27s |
| Negative wizard prompts | no clamp | 34.45s | 17.25s |
| Negative wizard prompts | W at −8 | 31.94s | 17.21s |

## VERIFICATION

`pytest -q test_part49.py test_part38.py`: 15 passed. Six isolated `/tmp` sabotages each made the focused two-hook, cleanup, seed, scorer, negative-clamp, and k=None control tests fail. CPU model smoke outputs and their score files are under `results/part49-smoke-final-b1/` and `results/part49-smoke-final-b16/`.

Parent-control messages 1 and 2 were considered and applied: batched row-seeded generation with padding tests, then the negative assay and no-hook control test. No control messages were declined or deferred.

## BLOCKERS

None. A blind reader must still label answers before broken-answer rates become numeric; no full Arc run was requested or performed.
