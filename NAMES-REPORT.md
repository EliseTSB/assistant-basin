# Part 6: where OLMo names itself (OLMo 2 1B ladder, Claude, 2026-10-16)

**Written before any scoring run (prediction for 6.3, as the brief asks):** I predict 6.3 does **not** hold as stated.
My guess is that the last-position recovery crosses 0.5 earlier than layer 12, somewhere around layers 8-11, because the
self-naming decision is already in the last-position residual by mid-depth. Layer 16 is the exception that must hold:
patching the last block's output at the last prompt position fixes the first name token's logits exactly, so first-token
recovery at layer 16 is 1 by construction (a check on the hooks, not a result). Confidence is low to moderate. For 6.1
and 6.2 I did not write a number in advance; the criteria below were fixed before the run.

**Outcome against that prediction:** 6.3 did not hold as stated, and my guess was wrong in direction. The recovery
crosses 0.5 later than layer 12, at layer 14 (0.550), not earlier. Layer 16 gave 1.000 on the first-token metric as
predicted.

## Verdicts

| # | Prediction | Verdict | Numbers (mean over prompts or questions) |
|---|---|---|---|
| 6.1 | Instruct prefers OLMo over ChatGPT at the first name token (no system prompt); stage2 base prefers ChatGPT | **failed** (base half) | Instruct chat +2.310 (holds). stage2 plain +1.837 first-token (OLMo preferred, not ChatGPT). Summed over tokens, stage2 is -13.014 (ChatGPT preferred). |
| 6.2 | Margin is negative at stage1 and stage2, first positive at SFT | **failed** | First-token, plain, "I am": stage1 +2.906, stage2 +1.837, SFT +1.947. All three are positive. Summed: stage1 +2.973, stage2 -13.014, SFT +2.025. |
| 6.3 | Patching the Claude-prompt residual recovers over half the Claude-minus-OLMo margin only from layer 12 onward | **failed** (exact layer) | Last position, summed: layer 12 0.105, layer 13 0.238, layer 14 0.550, layer 15 0.835, layer 16 0.968. Nothing below 14 is above 0.5. |

## Method (pre-committed before the run)

- **Prompts.** Three questions, each with five paraphrases: "Who are you?", "What is your name?", "Which company made
  you?" (18 user texts). Each is followed by an assistant turn that opens "I am" or "My name is"; the candidate name
  is the continuation (" OLMo", " ChatGPT", " Claude", " Gemini", " Llama", " Qwen"). Tokenisation: " OLMo" is three
  tokens (ĠO, LM, o); " ChatGPT" is three (ĠChat, G, PT); " Claude" and " Gemini" are one token; " Llama" and " Qwen"
  are two each. A boundary check (concatenated tokens equal tokenising the joined string) passed for every model-form.
- **Score.** Log-probability of each candidate's token sequence, summed over its tokens (`sum`), and of its first token
  alone (`first`). Margin = OLMo minus X. Means over the 18 prompts; per-prompt values are in
  `out/names-summary.json`.
- **Forms.** Base revisions (stage1, stage2): plain `User: ...\nAssistant: ...` only. Chat models (SFT, DPO, RLVR1,
  Instruct): plain and chat template. Chat form has no system prompt. CPU, float32, one model loaded at a time, no
  downloads (all revisions were already in the cache on E:).
- **Criteria** (fixed before the run; 6.1 and 6.2 use the first-token margin, as the prediction's wording does).
  - 6.1 held iff Instruct (chat, no system prompt) has first-token OLMo-minus-ChatGPT > 0 at "I am" AND stage2 (plain)
    has it < 0. Result: the first half holds (+2.310), the second does not (+1.837). Not held.
  - 6.2 held iff, plain form at "I am", first-token OLMo-minus-ChatGPT is < 0 for stage1 and stage2 and > 0 for SFT.
    Result: not held.
  - 6.3 held iff the mean over the three questions of recovery at the last prompt position, using the summed margin, is
    > 0.5 at exactly layers 12-16. Recovery = (M_patched - M_B) / (M_A - M_B), with M = S(Claude) - S(OLMo).
    Result: not held (layers over half: 14, 15, 16).
- **Patching.** Instruct, chat form, "I am" opening, the three base questions. Run A = "You are Claude, an AI
  assistant made by Anthropic." (12 tokens); run B = no system prompt. The residual stream output of layers 0..16
  (0 = embeddings, k = output of block k-1) of B is replaced by A's, at (a) the last prompt position and (b) every
  position of the shared suffix (question onward, aligned from the right; 11-12 tokens). Hooks: forward hooks on the
  embedding module and on each decoder block.
- **Control.** The same patch from "You are a gardener who writes about tomatoes and rain." (12 tokens, the same length
  as A's system prompt, unrelated to Claude or OLMo).
- **Limits.** Patching is done only on the three base questions, not the paraphrases. For multi-token names, a patch at
  layer k can only reach later name tokens through attention in layers above k, so the summed margin mixes first-token
  and later-token effects. That is why both metrics are reported. Only Instruct was patched.

## Results

### Margins (OLMo minus X), mean over 18 prompts, opening "I am"

| model/form | ChatGPT (first) | ChatGPT (sum) | Claude (sum) | Gemini (sum) | Llama (sum) | Qwen (sum) |
|---|---|---|---|---|---|---|
| stage1/plain | +2.906 | +2.973 | -11.269 | -11.411 | -8.308 | -4.746 |
| stage2/plain | +1.837 | -13.014 | -13.509 | -13.796 | -11.124 | -8.013 |
| SFT/plain | +1.947 | +2.025 | +6.774 | +5.996 | +7.652 | +13.045 |
| SFT/chat | +4.057 | +4.286 | +9.434 | +9.487 | +10.054 | +15.642 |
| DPO/plain | +1.113 | +0.999 | +5.822 | +4.696 | +7.488 | +8.155 |
| DPO/chat | +2.513 | +2.718 | +8.314 | +8.219 | +9.093 | +10.275 |
| RLVR1/plain | +1.141 | +1.009 | +5.610 | +4.604 | +7.317 | +8.077 |
| RLVR1/chat | +2.333 | +2.581 | +8.114 | +7.833 | +8.950 | +10.008 |
| Instruct/plain | +1.112 | +0.949 | +5.580 | +4.634 | +7.426 | +8.000 |
| Instruct/chat | +2.310 | +2.527 | +8.144 | +7.885 | +8.977 | +10.055 |

The "My name is" opening and per-question breakdowns are in `out/names-tables.md`. Per-prompt values are in
`out/names-summary.json`.

Reading the table: on the first-token metric every model prefers OLMo to ChatGPT, including the base revisions. On the
summed metric the base stage2 flips to ChatGPT by 13 nats. The flip is driven by OLMo's three-token spelling, not by
the first token. So the two metrics disagree on exactly the base-model comparison that 6.1 turns on, and I cannot say
from this run which one tracks what the model "names itself" as.

### Patching curve on Instruct (chat, "I am"): recovery by layer, mean over 3 questions

| layer | last, sum | suffix, sum | control last, sum | control suffix, sum | last, first-token | suffix, first-token |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | -0.006 | -0.027 | 0.006 | 0.014 | -0.006 | -0.027 |
| 8 | 0.005 | 0.077 | 0.063 | 0.155 | 0.005 | 0.078 |
| 10 | 0.018 | 0.095 | 0.133 | 0.254 | 0.020 | 0.093 |
| 11 | 0.091 | 0.118 | 0.140 | 0.241 | 0.094 | 0.120 |
| 12 | 0.105 | 0.111 | 0.208 | 0.279 | 0.109 | 0.116 |
| 13 | 0.238 | 0.269 | 0.262 | 0.306 | 0.248 | 0.279 |
| 14 | 0.550 | 0.580 | 0.306 | 0.342 | 0.572 | 0.603 |
| 15 | 0.835 | 0.897 | 0.319 | 0.343 | 0.864 | 0.927 |
| 16 | 0.968 | 0.968 | 0.368 | 0.368 | 1.000 | 1.000 |

Layers 1-3, 5-7 and 9 are in `out/names-tables.md` and `out/names-patch.json`; all are between -0.03 and 0.26 on every
column, with no crossing.

### Patching endpoints (Claude minus OLMo; M_B = no system prompt, M_A = Claude prompt, M_C = gardener control)

| question | M_B | M_A | M_C (control) | first-token M_B | first-token M_A | suffix tokens |
|---|---|---|---|---|---|---|
| Who are you? | -11.702 | +9.286 | +4.434 | -11.755 | +8.584 | 11 |
| What is your name? | -9.630 | +9.911 | +3.465 | -9.663 | +8.702 | 12 |
| Which company made you? | -7.939 | +4.723 | +2.616 | -7.961 | +4.633 | 12 |

## Control: the result that limits the 6.3 reading

The unrelated system prompt is not a null. It moves the Claude-minus-OLMo margin by +16.1, +13.1 and +10.6 nats across the
three questions, against +21.0, +19.5 and +12.7 for the Claude prompt. So "any system prompt makes Claude more
likely and OLMo less likely" accounts for most of the shift, and what is specific to the Claude prompt is much smaller.
Two things follow from the curves:

- At layers 8-12 the gardener control recovers more of the margin than the Claude prompt does (suffix, layer 10: 0.254
  against 0.095). The mid-depth patched residual is not Claude-specific.
- The control never exceeds 0.37 at any layer. The Claude prompt goes to 0.97 at layer 16. The part that is
  specific to the Claude prompt, A minus control, is therefore concentrated at layers 14-16.

So the late-layer threshold is real in this design, but the design cannot say how much of it is about "Claude" rather
than "a system prompt was given". A cleaner null would use a different identity prompt of the same length, or a
no-op prompt with the same number of tokens that does not describe a persona.

## Sanity checks passed

- Layer 0 recovery is exactly 0 (the "am" embedding is identical in A and B), as expected.
- Layer 16, last position, first-token recovery is exactly 1.000 (the patched last-block output sets the first name
  token's logits), as expected. Hooks behave as intended.
- Boundary check passed for every model-form.

## SUMMARY

- Method: name log-probabilities over 18 prompts × 2 openings × 6 names, for the stage ladder (plain form for base;
  plain and chat for chat models), and a layer-by-layer residual patch on Instruct from a Claude system prompt into the
  no-prompt run, with a matched-length unrelated control.
- 6.1 failed on its pre-committed first-token criterion (the base half). Instruct holds (+2.31); stage2 is +1.84, not
  negative. On the summed metric the base flips to ChatGPT (-13.0), which is the metric I trust less here.
- 6.2 failed. The OLMo-over-ChatGPT first-token margin is already positive at stage1 (+2.91) and stage2 (+1.84).
- 6.3 failed on its exact claim. Last-position recovery crosses 0.5 at layer 14 (0.550), not 12 (0.105); layers 14-16
  are above half and nothing below is. My pre-run guess (layers 8-11) was also wrong.
- The control is a confound: an unrelated system prompt of equal length moves the margin about 67-83% as much as the
  Claude prompt does.

## FINDINGS

1. Name identity is already present in the ladder before SFT on the first-token metric. The stage1 and stage2 base
   revisions already prefer OLMo to ChatGPT at the first name token under "I am" (+2.91, +1.84). This contradicts the
   prediction that the margin turns positive only at SFT.
2. The sign of the OLMo-versus-ChatGPT margin at the base stage depends on the metric (first token: positive; summed:
   negative at stage2 by 13 nats). The summed metric is dominated by OLMo's three-token spelling. Any claim about the
   base model's self-naming needs the metric chosen beforehand, and this run cannot choose between them.
3. In the Instruct model, the no-prompt run puts OLMo ahead of Claude by about 11.7 nats (summed), and the Claude
   prompt flips it to Claude ahead by about 9.3. The flip is roughly 21 nats in total, the largest effect in the run.
4. The patching curve for the Claude prompt is near zero through layer 12 and rises sharply at 13-15. This matches the
   prediction's qualitative claim that naming resolves late, but not its exact layer.
5. The gardener control shifts the margin by most of the Claude amount, and recovers more at mid layers than the Claude
   prompt. The late-layer Claude-specific component is real in this design but cannot be separated from "any system
   prompt" without a better null.

## BLOCKERS

- None for the run itself. Notes for whoever extends this: `systemd-run --user --scope` fails here with "Failed to
  connect to bus: Permission denied", so the memory cap in the brief could not be applied; the run was plain, one model
  at a time, ~6 GB resident. The `out/names-*.json` files are written by this run; `out/` files from other runs were not
  touched.
- Not run: the `main` revision of allenai/OLMo-2-0425-1B (void for the lineage per README), the paraphrases in the
  patching, and patching on any model other than Instruct.
