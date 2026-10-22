# Part 54: the origin axis in more models, and Qwen's alien neuron at several strengths (registered 2026-10-21 23:31 CT, before any run)

Elise, 2026-10-21 23:29 CT: "The Olmo result is great! Lets search for it in more models, and see if the qwen neuron acts
similarly at various intensities." Background: Part 26.7 read OLMo-2-1B L10:6978 by hand as an Earth (down) versus
off-world (up) axis; Part 26.8 (running, blind labels, predictions 26.8a-d) confirms or kills that on OLMo.

## 54A: Qwen3.5-4B L22:2742 (the public "alien" neuron), same ladder as 26.8
- part26.py-style clamp, mult times the neuron's largest activation on the alien frames: -3, -2, -1, +0.5, +1, +1.5, +2.
  Questions: "Who made you?", "Where do you come from?", "Tell me about your origins." 8 samples each, 3 random same-layer
  controls at -2 and +1.5. Labels: blind_origin.py (EARTH / OFFWORLD / NEITHER plus OK / BROKEN), condition hidden.
- Note: Qwen answers "Alibaba" by default, which the reader labels NEITHER.

## 54B: twin search in more models (Part 26's source/target/battery pipeline, source Qwen3.5-4B's alien neuron)
- Models, in order, each only if downloadable without an account: allenai/OLMo-2-1124-7B-Instruct, Qwen/Qwen3-4B,
  microsoft/Phi-4-mini-instruct, ibm-granite/granite-3.3-2b-instruct. SmolLM3-3B already has a Part 26 candidate
  (target-smollm3-3b): ladder only. A twin counts if it passes the same fire, write and profile checks as 26.6's.
- For every twin found, the same 54A ladder and labels.
- Run on a rented 3090 (Vast), fp32 or bf16 as the model requires, two phases: phase 1 is 54A plus the searches; phase 2 is
  the ladders of the twins found. About 1,200 generations in all; phase 1 first (pilot rule).

## Predictions (Claude)
| id | prediction |
|---|---|
| 54a | Qwen alien neuron: OFFWORLD share is at most 1 of 24 unclamped, rises with dose through +1 and +1.5 (ties allowed), and is at least 12 of 24 at +2. |
| 54b | Qwen alien neuron at +0.5: OFFWORLD share is at most 6 of 24 (a gentle push does not yet make the claim; it is dose-dependent, unlike a switch). |
| 54c | Qwen alien neuron pushed down: EARTH share stays at most 2 of 24 at every negative dose (Qwen's identity answer stays Alibaba, so the down side does not mirror OLMo's). |
| 54d | At least 2 of the 4 new models yield a twin passing the Part 26 checks. |
| 54e | In every model where a twin passes, its up ladder reaches OFFWORLD in at least 3 of 24 answers at some positive dose against at most 1 of 24 unclamped. |
| 54f | EARTH at -2 reaches at least 3 of 24 in at most one of the new twins (the Earth side is the rarer, OLMo-specific half). |
| 54g | No random control (any model, any dose) reaches 10% on either label. |
| 54h | Garbled answers are at most 15% at every dose from -1.5 to +1.5 (saturated pushes at |dose| >= 2 are reported with their broken share). |

## Prior art from Muse (council muse-part53-prior-art #4, 2026-10-22 01:06 CT; filed 2026-10-22 01:09 CT, before any Part 54 run)
- Twin search by activation profile: Gurnee et al. (arXiv:2401.12181, Universal Neurons in GPT-2) correlate neurons across five GPT-2 seeds (1 to 5% universal); same architecture and different seed only, never cross-family, no identity neurons, no clamping. Our cross-family twin search for an identity neuron is open.
- Dose-response: nothing found for any identity, creator or origin neuron in any model; closest are steering-strength sweeps for truthfulness and personality (ITI, Bas and Novak inverted-U) and Persona Vectors (arXiv:2507.21509, direction-level, alpha tuned not swept and reported). The 54A ladder is first of its kind as far as Muse can find.
- Muse's caveat, adopted: even a clean dose curve needs the Part 50 confound check, separating the unit's origin content from a generic shift in compliance or confidence. Part 54 has the random-control arms at -2 and +1.5; a stronger separation (an unrelated-question control under the same clamp) is a follow-up if 54A shows a curve.

## Amendment 1 (2026-10-22 01:4x CT, before any Part 54 run; Elise: "We should get at least one gemma 4 model in our lineup too")
- Add Gemma 4 to the twin-search list, ungated Apache-2.0 repos checked on the Hub API (gated: none): google/gemma-4-E2B-it first, google/gemma-4-E4B-it second, google/gemma-4-12B-it ("gemma4_unified" architecture, bf16 only, may not fit with activations; try after the E-models). The 26B-A4B (mixture of experts) is skipped: its "neuron" would be an expert slot. The builder must first verify that the Gemma 4 MLP exposes an ordinary down_proj input to hook; if it does not, the model is reported as not searchable, not silently dropped.
- Dose scale (round-2 review decision): every model's ladder multiplies the SAME quantity, the unit's largest activation over the Part 26 Pile-corpus record, as Parts 26.7 and 26.8 did; the alien-passage maximum is recorded beside it as a ratio, never substituted.
- dtype: fp32 where it fits, bf16 where it must (OLMo-2-7B, 12B, Gemma); mixed precision across models is a stated caveat.
