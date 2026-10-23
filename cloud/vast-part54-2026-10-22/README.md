
## 2026-10-22 15:28 CT — run finished, instance 57259182 destroyed
Rented about 14:08, destroyed 15:27 (~1.3 h at $0.2857/h = about $0.38). `RETURN_READY` at 20:26 UTC. Return bundle extracted into results/part54a (25 clamp files) and results/part54b (121 files). Searches: olmo2-7b (twins 12:5876, 15:6374), qwen3-4b (28:4248), gemma4-e2b-it (27:6092), granite3-2b and gemma4-e4b-it (no twin listed), smollm3-3b ladder only. phi4-mini failed to load (`ImportError: cannot import name 'LossKwargs' from transformers.utils`, its remote modeling code against the image's transformers); gemma4-12b recorded as not run. Blind batches: part54a msgbatch_01AYpTn4oDiwdib4keBnRuaA (502), part54b msgbatch_017PdYJMFE2moQjZaTGfp6z7 (1,942).

## 2026-10-22 20:42 CT: Part 59 run finished, instance 57289481 destroyed
Rented 18:29, RETURN_READY at 19:58 CT (box UTC 00:58), destroyed 20:42: about 2.2 h at $0.196/h = about $0.43. Bundle part59-return.tgz extracted into results/part59a (120 files plus manifests) and results/part59b (Qwen and OLMo archives, manifests and analysis JSON). No failed phases. The first 59A blind batch (msgbatch_014PdvenzxP9ktyJgDBA3xEt) was truncated by max_tokens 32 and is discarded; the scored batch is msgbatch_01UkAS6ir2KL3JpgcXJEVKnM.

## 2026-10-23 01:36 CT: Part 60 dev scoring finished, instance 57330034 destroyed
Rented 01:05 (Quebec, 209.226.246.174, same host as the Part 61 box), two run60.sh passes (first dev pass 3 min 06:08-06:11 UTC; second pass for M3 fresh 06:33-06:34), destroyed 01:35: about 0.5 h at $0.305/h = about $0.15. Bundles part60-return.tgz (46.6 MB then 52 MB) extracted into results/part60 (acts/dev M0,M1,M2,M3,M3-selection,M6; M4 and M5 dropped by the 10% parse-failure rule). Test split not opened.

## 2026-10-23 02:22 CT: Part 61 run finished, instance 57327044 destroyed
Rented 00:31 (the first box 57325670 never left loading and was destroyed, about $0.08), run61.sh 05:33-06:53 UTC (00:33-01:53 CT) for the Qwen3.5-4B and OLMo-2 1B halves, destroyed 02:22 (idle after RETURN_READY at 01:53, about 29 min of waste): about 1.9 h at $0.305/h = about $0.58. Bundle part61-return.tgz (81.6 MB) extracted into results/part61 (npz and json for both models, per-model analysis, part61-registered-evaluation.json), logs in results/part61/logs-vast.
