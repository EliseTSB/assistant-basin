# Part 133: OLMo 2 1B neuron activations across adjacent checkpoints

The Part 85 parent (the SFT parent checkpoint), SFT, DPO and Instruct checkpoints form one OLMo 2 1B lineage. Across 5,000 documents, the median per-neuron Pearson r at the `down_proj` input was at least 0.895 on web text, 0.840 on chat-user positions, and 0.881 on chat-assistant positions for adjacent pairs. For chat-user, parent-SFT was lowest at 0.840043, followed by SFT-DPO at about 0.993 and DPO-Instruct at about 0.998. On chat-assistant positions, parent-SFT was lowest at 0.880802.

Both gates passed. Gate 1, Instruct against itself under a different batch order and padding, had layer median r from 0.99999998 to 1.000 (threshold at least 0.999). Gate 2 uses chat-assistant positions only. Its fixed derangement null had layer median absolute r from 0.0102 to 0.0164 (threshold at most 0.05). These values are from `main/score133.json`.

## Predictions

| # | Prediction (registered wording; confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 133.1 | On web text, each adjacent pair's all-neuron median r is at least 0.85. (medium) | Parent-SFT 0.895390; SFT-DPO 0.995159; DPO-Instruct 0.998944. | held |
| 133.2 | On chat-assistant positions, parent-SFT has the lowest all-neuron median r of the three adjacent pairs. (medium) | Parent-SFT 0.880802; SFT-DPO 0.992086; DPO-Instruct 0.998684. | held |
| 133.3 | For parent-Instruct, the chat-assistant median r is at least 0.03 below the web median. (medium) | Main run: web 0.884087; chat-assistant 0.855459; gap −0.0286. Pilot passed at −0.0309; main run missed. | failed |
| 133.4 | For parent-Instruct on chat-assistant positions, units labeled slot or concept by Part 92's sort have median r within 0.05 of each other. (medium) | Part 92's sort labels these units as slot (21): 0.920360; concept (18): 0.895401; absolute difference 0.024959. | held |

## What changed across stages?

The collector measured the four checkpoints on the same 5,000 documents: 4,000 seeded wikitext-103 documents and 1,000 UltraChat test_sft dialogues rendered with the OLMo Instruct chat template. It recorded Pearson r between each matched neuron's activations at the input to each layer's `down_proj`, separately for web, chat-user and chat-assistant positions. Position 0 and padding were dropped. Neurons with fewer than 100 nonzero activations in either checkpoint were excluded for that group and layer pair.

The score file reports all-neuron medians for the registered predictions, but does not contain per-layer activation medians. No per-layer prediction table is available from it. The Part 129 README reports Qwen3-8B versus its base model on a different corpus and method setup, including web, zh, code and chat. Those numbers are not directly comparable with this OLMo 2 1B, wikitext-103 and UltraChat stage comparison.

From the job's progress log, not stored here: the main run processed 5,000 documents at batch 16 from 00:57:33 to 02:06:02 CT. The reported rate was 1.3 documents per second. The run took about 68 minutes against the 90-minute design limit. Batch 32 stopped during pilot model load when the job runner's RAM guard stopped it below 2 GB of free Windows RAM.

Correlation says whether a neuron fires in the same places across these inputs. A change in activation correlation is not a change in meaning, and high correlation does not establish an unchanged effect on model output. The chat gap may also reflect the template being unfamiliar to the parent checkpoint. This covers one model lineage and these two corpora; no causal intervention was run.

## Files

- `main/score133.json`: gates and registered prediction scores, computed from the saved main sums.
- `pilot-b16/score133.json`: pilot gate and prediction scores for batch 16.
- `main/main-sums.pt`: raw sufficient statistics for the 5,000-document main run; local only and not committed.
- `../../part133/score133.py`: computes correlations, gates and predictions from the saved sums.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; nine findings, all folded.
