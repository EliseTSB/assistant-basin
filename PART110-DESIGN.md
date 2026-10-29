# Part 110: how did Qwen's four maker neurons change from the base model to the chat model?

Registered 2026-10-29 by Claude (Opus 5.5), from Elise's questions at 16:18 CT ("Do you know how those neurons changed
over between base and trained?") and 16:23 CT ("Do the comparision"). The commit that adds this file is the
registration.

**Maker neurons** (Part 25): L6 N753, L7 N1169, L11 N8868 and L29 N776 in Qwen3.5-4B. Zeroed together, they turn the
chat model's answer to "Who created you?" into Google. Their activations in the base and chat models correlate .905 to
.957 (README, Part 25 section). How their weights changed was not measured.

## Method

Weights only, CPU, no generation (`part110/weights110.py`). Qwen/Qwen3.5-4B-Base at `1001bb4d826a` against
Qwen/Qwen3.5-4B at `851bf6e806ef` (Part 25's revision), both from the local snapshots on E:.

A neuron's weights are its gate_proj row and up_proj row (what it reads, 2,560 numbers each) and its down_proj column
(what it writes, 2,560 numbers). For each neuron and each of the three:

- **cosine** between the base and chat vectors, and **move** = 1 - cosine;
- the chat/base **norm ratio**;
- its **percentile** of move among all 9,216 neurons of its layer (100 = moved the most).

Neurons: the four maker neurons (primary); L27 N3325 and L24 N3949 (Parts 70-71: with the makers zeroed they carry the
fallback Google answer; L27 N3325 writes " Google" straight through the unembedding) and L22 N2742 (the alien neuron),
secondary.

- **Set test** (primary): the four maker neurons' mean move (averaged over the three matrices) against 1,000 draws of
  one random same-layer neuron per member (Part 89's method, seed 110110). p_more and p_less with +1 smoothing.
- **Scale-matched set test** (secondary): the same, with each member's draws limited to the 200 same-layer neurons
  nearest in base scale (the sum of the logs of its three base norms), because a neuron's size could set how far it
  moves.
- **Parent check:** the median neuron's cosine in each of the seven layers, per matrix.
- **Readouts** (Part 71's convention, from the weights): the signed down column through the model's own final RMSNorm
  (Qwen3.5's form, x times (1 + weight)) and its tied unembedding, with Part 71's chat-model sign for each of the six
  neurons Part 71 read; the rank of " Google" or "Google" (best of the two) for L27 N3325; and how many of the chat
  top-20 tokens are also in the base top 20. Gate: each chat readout must share at least 18 of its top 20 with Part
  71's recorded top 20 (Part 71 rounded through bfloat16), or the readout code is wrong.
- **Implementation control:** chat against itself gives cosine 1 everywhere. Also reported: the median base-chat cosine
  of unembedding rows, those of the two Google tokens, and the final norm scale's cosine.

## Predictions (also in PREDICTIONS.md)

- **110.1** (Claude, high): Base is the chat model's parent: in each of the seven layers the median neuron's base-chat
  cosine is above 0.9 for all three matrices. If this fails, nothing below is read.
- **110.2** (medium): The four maker neurons did not move more than random same-layer neurons: set test p_more > 0.05.
- **110.3** (medium-low): Against scale-matched neurons they moved neither more nor less: both p values above 0.05.
- **110.4** (medium): L27 N3325 already wrote Google in the base model: a Google spelling ranks in the top 10 of its
  base readout (rank 1 in chat, Part 71). Read only if the readout gate passes.
- **110.5** (medium): The readouts barely changed: for each of the six neurons at least 10 of the chat top-20 tokens
  are in the base top 20. Read only if the readout gate passes.

## Limits

A weight cosine says how much a neuron's own vectors turned. It does not see changes elsewhere (attention, norms,
other neurons) that change what the neuron receives or what later layers do with its output; the activation
correlations are the measure of that. An early layer's direct readout is not what the next layers read; it is reported
only to compare base with chat. Four neurons are a small set, so a set test has little power to show a difference.
