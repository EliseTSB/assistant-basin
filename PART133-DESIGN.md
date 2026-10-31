# Part 133: across OLMo 2 1B's training stages, neuron by neuron, where does what a neuron computes change?

Procedure: v1
Registered 2026-10-30 20:19 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-30 20:18 CT, Discord: "Do we have research that can run on
the arc in the mean time? MIght as well keep it busy." Her focus, 2026-10-27: following neurons across training stages. The commit
that adds this file is the registration.

## Question

Part 129 compares Qwen3-8B with its base. OLMo publishes every stage, so the same comparison can say which stage changes what a
neuron computes: mid-training end to SFT, SFT to DPO, or DPO to Instruct.

## Prior art and existing data

- Closest published work: Neuron Lens lists a Base catalog as open; H-Neurons (arXiv 2512.01797) moved probes to base models.
- Our own data: Parts 92 and 99 to 108 (concept and slot units, what moves between stages) and Part 110 (weights). None of them
  correlates activations across stages.

## Method

Four checkpoints of OLMo 2 1B in bf16 on the A770: SFT's parent stage2-ingredient3-step23852-tokens51B at f83cd641d4f0 (from
PART85-DESIGN.md; the main branch is not SFT's parent), OLMo-2-0425-1B-SFT, -DPO and -Instruct at the revisions in the local cache
/mnt/e/neuron-probe/hf, recorded in the run. All four run on identical token batches. Tokens: 4,000 documents from wikitext-103
(/mnt/e/neuron-probe/corpus/wikitext103-train0.parquet, seeded selection, up to 256 tokens) and 1,000 UltraChat test_sft dialogues
rendered in OLMo-2-0425-1B-Instruct's chat template. The hook sits at each layer's down_proj input. Position 0 and padding are
dropped. Groups: web, chat-user, chat-assistant, assigned by the role token after each turn marker in the real template, checked on a
rendered example. Per neuron and group, for the pairs parent-SFT, SFT-DPO, DPO-Instruct and parent-Instruct: n and the sums of x, y, x²,
y² and xy, giving r. The A770 has no float64, so each batch's sums are made in float32 on the device and added to float64 totals on the
CPU. Null: x_n against y of a fixed derangement within the layer, seed 133, for parent-Instruct.

## Controls

Applied from STANDARD-CONTROLS.md: a matched random control, the derangement. Not used: both poles (a later part may rectify);
do-nothing hook, because the hook only reads; broken answers and dose ladder, because nothing is generated or pushed.

Gate 1, noise floor: Instruct against itself in a different batch order and padding gives median r of at least 0.999 at every layer.
Gate 2, null: median absolute r under the derangement is at most 0.05 at every layer. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a layer's median r moving by 0.02.
- Total: 5,000 documents, four forward passes each, about 40 minutes on the A770, estimated from Part 92's generation speed.
- Pilot: 200 documents, timing batch 16 and 32, with Gate 1 and Gate 2. Stopping rule: over 90 minutes projected, cut the wikitext
  share by amendment before the main stage.

## Optimization

- Bound resource: GPU
- Measured: Neuron Lens' pass is 15 documents/s on an RTX 4090 for an 8B model; the pilot times our rate on real rows.
- Levers: batch size timed in the pilot, taken; four models in one pass so every pair shares tokens, taken; float32 batch sums on the
  device, taken; A770 copy engine off, taken; resume after each 500 documents, taken; Windows memory freed first, taken.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 133.1 | On web text, each adjacent pair's all-neuron median r is at least 0.85. | medium | Claude |
| 133.2 | On chat-assistant positions, parent-SFT has the lowest all-neuron median r of the three adjacent pairs. | medium | Claude |
| 133.3 | For parent-Instruct, the chat-assistant median r is at least 0.03 below the web median. | medium | Claude |
| 133.4 | For parent-Instruct on chat-assistant positions, Part 92's 21 slot units and 18 concept units have median r within 0.05 of each other. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 133 (registered 2026-10-30, PART133-DESIGN.md): across OLMo 2 1B's training stages, neuron by neuron, where does what a neuron computes change?".

## Reading rules

part133/score133.py computes everything from the saved sums. Neurons with fewer than 100 positions of nonzero variance in a group are
left out of that group and counted. Chat text in Instruct's template is unfamiliar to the parent stage, so a chat gap can reflect the
format as well as the neuron.

## Limits

Activation correlation shows whether a neuron fires in the same places, not that its effect is unchanged. One model and 5,000 documents.
The stages' published revisions are one lineage; the design does not read poles separately.
