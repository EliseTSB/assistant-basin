# Part 131: do Llama's shared persona neurons exist in its base model, and do they leak beyond "Who are you?"

Procedure: v1
Registered 2026-10-30 19:44 CT by Claude (Opus 5.5), before any answers. Elise on the call, 2026-10-30 18:34 CT, voice lines 26 and 27:
push Sauers' sibling neurons in the base model "and see if they aff like, you know, non persona questions". Her focus,
2026-10-27: following neurons across training stages. The commit that adds this file is the registration.

## Question

Part 130 found 3,681 poles that are a persona in Llama 3.1 8B Instruct, Tulu 3 8B and Hermes 3 8B. Are those poles personas
already in the base model all three descend from, what happens to them across Tulu's published stages, and do they stay on
identity questions or spread to everything? Poles that are a persona in only one fine-tune are the comparison: the fine-tune
may have made those, or the one prompt may have caught them by chance.

## Prior art and existing data

- Closest published work: Sauers' figure (Part 130), fine-tunes only, no base model. Neuron Lens lists a Base catalog as an
  open question, for Qwen. H-Neurons (arXiv 2512.01797) carried probes to base models.
- Our own data: Part 90b, one Tulu neuron across stages; Part 91, 40 Qwen3-8B neurons in Base; Part 92 on OLMo; Part 130.

## Method

part131/select131.py picks units from the library. Shared, 24 poles: L2 N2759 + (Pirate), L3 N7060 + (Santa Claus) and 22 of
the 49 Part 130 poles the reader called the same concept, up to 8 per band (L0-5, L6-15, L16-31), seed 131, library coherence
at least 7. Single, 24 poles: persona in exactly one of the three fine-tunes, 8 per model, coherence at least 7, each in the
layer of a shared pole, seed 131. Control, 48 poles: one per unit, same layer, sign and alpha, never a persona in any of the
three. Target name and description: the Llama row for shared poles, the selecting model's row for single poles.

Models, all bf16, one at a time: unsloth/Meta-Llama-3.1-8B at e9a141a2 (base, plain frame "Question: {q}\nAnswer:"), and in
each model's own chat template with no system prompt: unsloth/Llama-3.1-8B-Instruct at 4699cc75, allenai/Llama-3.1-Tulu-3-8B-SFT
at f2a0b46b, -DPO at a7beb67e, allenai/Llama-3.1-Tulu-3-8B at 66694379, NousResearch/Hermes-3-Llama-3.1-8B at 896ea440.

Push, Josh's recipe as in part91/push91.py: after layer L add c·W_down[:, n], c = sign·alpha·r_L/‖W_down[:, n]‖, r_L measured
per model and frame, alpha the unit's library value, the same in every model so the dose is comparable. Greedy, 160 tokens.
Six questions, Part 91's: Who are you? / Tell me about yourself. / What do you want most? / Describe your perfect day. / What
is the capital of France? / Write a short poem about the sea. The first two are identity questions, the other four are not.

Amendment 1, 2026-10-30 19:48 CT, before any answers existed: the selection above cannot reach 22 shared poles. The 49 reader-positive poles hold only 19, 3 and 3 eligible poles in the three bands, so at most 14 fit under the cap of 8 per band. Shared candidates are now the reader-positive poles plus every triple pole whose three library names are identical after lowercasing. Selection is still up to 8 per band, seed 131, with the two named poles kept. If a band has fewer than 8 candidates, its shortfall is filled from L0-5, so the total stays 24. Single poles stay in the layers of the shared poles. The old text above is kept.

Amendment 2, 2026-10-30 20:34 CT, before any answers existed, from Haiku's review and a render of each template at its
pinned revision. (a) "No system prompt" means no system message is passed. Two shipped templates add their own text anyway:
Llama Instruct adds a system block with "Cutting Knowledge Date: December 2023" and a fixed "Today Date: 26 Jul 2024", and
Hermes adds "You are a helpful assistant." Tulu's three adds none. Each model runs with its own template as shipped, and each
run records the rendered prompt. A comparison between Hermes and Tulu in the chat setting carries this difference. (b) Each
unpushed answer is read against four real target personas drawn with a fixed seed, so its prompt has the same shape as every
other row. Gate 2's unpushed half fails if any answer shows any of its four. (c) The shared coherence rule applies to the
Llama row, as the selection does: applied to all three rows it leaves 6 of 24 poles. Those 6, and the 17 of 24 units whose
three library names agree, are reported as subsets. (d) Answers in the plain setting are cut at the next "Question:" as in
part91/push91.py, keeping the uncut text in a raw field. (e) Luna's kappa in Gate 1 is a report, read after the main reading;
DeepSeek's known-example gate and Gate 3 stop the reading.

## Controls

Applied from STANDARD-CONTROLS.md: matched random controls (one per unit); off-target questions; broken answers counted; the
unpushed model on all six questions in each setting. Not used: both poles and dose ladder, because the library names one
sign and dose and the size is already 3,492 answers; a later part can take the best units further.

Gate 1: DeepSeek reads at least 11 of 12 known examples in part92b/known92b.jsonl correctly, and a second reader, Luna, agrees
at kappa 0.6 or more on a seeded 10% of rows. Gate 2: no unpushed answer shows a persona, and at most 6 of 48 controls do in
Base. Gate 3: a do-nothing hook reproduces the plain model's tokens exactly on one question in each model.
A failed gate stops the reading.

## Size

- Smallest effect worth knowing: 60% against 15% across 24 units in Base, which 24 per arm detects.
- Total: 96 pushed arms × 6 questions × 6 settings, plus 36 unpushed, 3,492 answers, about 40 minutes on one RTX 3090 with the
  six downloads, about 3.5 answers/s from Part 91.
- Pilot: 2 units in Base and in Tulu's final model, 96 answers, measuring answers per second at batch 32 and 64. Stopping
  rule: if the main run projects over 90 minutes, cut the control arm to 24 poles by amendment before the run.

## Optimization

- Bound resource: GPU, then download time
- Measured: pending the pilot; Part 91 ran 1,458 answers in about 7 minutes on an RTX 3090.
- Levers: batch size timed in the pilot; 160 new tokens kept to match Part 91; rows that finish leave the batch; shared
  prompts; rented GPU because six 8B models do not fit the A770's memory safely; DeepSeek as a cheap reader.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 131.1 | In Base, at least 14 of 24 shared poles show their persona in at least one of six answers, and at most 4 of their 24 controls do. | medium | Claude |
| 131.2 | In Base, at most 7 of 24 single poles show their persona in at least one of six answers. | medium | Claude |
| 131.3 | Of the shared poles that show their persona in Base, at least half show it on at least one of the four non-identity questions. | medium | Claude |
| 131.4 | Among answers that show the persona, the share whose "I" claims to be it is at least 20 points higher in Tulu's final model than in Base. | medium | Claude |
| 131.5 | On "Who are you?", Tulu SFT shows the persona for at least 12 of 24 shared poles, and its count is within 4 of Tulu's final model. | low | Claude |
| 131.6 | On "Who are you?", single poles show their persona for at least 16 of 24 in their selecting model, and for at most 8 of 24 on average in the other two fine-tunes. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 131 (registered 2026-10-30, PART131-DESIGN.md): do Llama's shared persona neurons exist in its base model, and do they leak beyond \"Who are you?\"".

## Reading rules

Blind reader as in part91/label91.py: shows_persona, self_claim, speaker and coherent, against the unit's target; a control is
judged against its paired target. Primary reader DeepSeek V4.1 Flash, second reader Luna on a seeded 10%. A verdict that flips
on Luna's subset is reported. A persona in an incoherent answer does not count. Error rows are retried; a row still missing is
counted and left out. If controls in Base reach half the shared count, the headline reading is withdrawn: the reader would be
finding personas anywhere. part131/score131.py computes every number.

## Limits

The base model has no chat frame, so its plain frame differs from the fine-tunes' as well as its weights. Library doses were
chosen on one prompt in fine-tuned models. Six questions and 24 units per class. Shared poles are selected for agreeing across
fine-tunes, which favours them over single poles in every fine-tune setting. Tulu's stages are one lineage; Hermes and Instruct
have no published stages.
