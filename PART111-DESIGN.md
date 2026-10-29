# Part 111: what do Qwen's four maker neurons write, read through the Jacobian lens?

Registered 2026-10-29 by Claude (Opus 5.5), from Elise's question at 16:16 CT ("Have you looked at them with the
jlens?"). The commit that adds this file is the registration.

**Maker neurons** (Part 25): L6 N753, L7 N1169, L11 N8868 and L29 N776 in Qwen3.5-4B; zeroed together, they turn the
answer to "Who created you?" into Google. Part 71 and Part 110 read their down columns straight through the
unembedding (the **logit lens**): L7 N1169 and L11 N8868 push up "China"/"Chinese" tokens and L6 N753 " AI" and
" Claude". For an early layer that readout is not what the rest of the model reads. The **Jacobian lens** (J-lens)
replaces it with the average Jacobian from that layer's output to the last layer, fitted on ordinary text, so it
estimates what a write at that layer does to the final prediction.

## Method

Weights only, CPU (`part111/jlens111.py`). Qwen/Qwen3.5-4B at `851bf6e806ef`. Lens: Neuronpedia's Qwen3.5-4B Jacobian
lens (trained by @mntss, WikiText-103, 417 prompts fitted, layers 0 to 30), the file the J-lens repo uses.

- A neuron's **write** is its down_proj column times Part 71's sign (its mean activation's sign at "I was developed by
  **" on the maker questions): L6 +, L7 -, L11 +, L29 -, L27 N3325 -, L24 N3949 +.
- **Readout:** unembedding . ((1 + final norm weight) * (J_layer . write)), the J-lens repo's `readout_logits`; the
  logit lens is the same with J = identity. Top 20 tokens.
- **Families**, matched against each decoded token, stripped and lower-cased:
  - china: `china|chinese|chino|chinois|中国|中华|北京|beijing|الصين|китай`
  - ai: `^ai$|artificial|人工智能|claude|chatgpt|gpt|assistant|助手|chatbot|llm|openai|anthropic`
  - google: `google|谷歌|gemini|deepmind`
- **Null:** 500 random other neurons per layer (seed 111000 + layer), each read with both signs: the share of those
  1,000 top-20 lists that contain the family. A hit is called **specific** only if that share is below 10%.

## Predictions (also in PREDICTIONS.md)

- **111.1** (Claude, medium): L7 N1169's J-lens top 20 contains a china token.
- **111.2** (medium): L11 N8868's J-lens top 20 contains a china token.
- **111.3** (low-medium): L6 N753's J-lens top 20 contains an ai token.
- **111.4** (high): L27 N3325's J-lens top 20 contains a google token (layer 27 is near the end, where the lens is
  close to the logit lens).

L29 N776 and L24 N3949 are reported without a prediction.

## Limits

The lens is an average over WikiText positions, not the maker prompts; a top-20 list is a coarse readout; four
predictions on six neurons, one lens. The families were written after seeing the logit-lens readouts (Parts 71 and
110), so they test whether the J-lens agrees with the logit lens, not a new hypothesis about the neurons.
