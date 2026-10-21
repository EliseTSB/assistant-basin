# Part 42: what our neurons write, read through the Jacobian lens

Written 2026-10-20 19:09 CT by Claude, before any code or run. Elise's idea on the voice call (18:0x CT): the J lens
should say more than the logit lens about what early neurons write; she pointed out that pre-fitted lenses exist.
Run size: weights only, no model forward pass and no generation: about 60 units x 2 readouts.

## Lenses
Neuronpedia's pre-fitted Jacobian lenses (huggingface.co/neuronpedia/jacobian-lens, fitted with anthropics/jacobian-lens
on WikiText-103, 1,000 prompts): qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt and
qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt (sha256 recorded at download). Format, per the
repo README: J[l] maps layer l's residual stream to the final block's; readout(h) = unembed(J[l] h), unembed = final
norm then LM head.

## Readouts
For each unit (input unit u of layer l's mlp.down_proj), w = down_proj.weight[:, u], its write into the residual stream.
- logit lens: top and bottom 30 tokens of lm_head(g * w), g the final norm's gain (the method used in Parts 26 to 38);
- J lens: top and bottom 30 tokens of lm_head(g * (J[l] w)).
Units: Qwen3-1.7B: the 43 Part 35 rank-1 units (gated median-floor screen) and the wizard unit L19 N4826;
Qwen3.5-4B: Part 25's four (L6 N753, L7 N1169, L11 N8868, L29 N776) and the alien unit L22 N2742. Null: 20 seeded
random units from the same layers, read the same way.
Scoring, fixed now: a readout "names the concept" if its top 30 contains the concept word or a close form (case- and
space-insensitive substring both ways, at least 3 characters; for Part 25's units the words in 42.4). The meaning-word
lists for 42.3 are fixed here: wizard: spell, magic, wand, sorcer, witch, potion, enchant; L6 N753: ai, assistant,
model, chatbot, 人工智能, 助手, claude, gpt.
