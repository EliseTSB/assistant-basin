# Part 94: Qwen3.5-4B L24 N3659, the "Alcohol, Satanic Legion of Hell" neuron

Written 2026-10-28 21:30 CT by Claude (Opus 5.5), before any Part 94 run. Elise, 2026-10-28 21:29 CT: "Look into that neuron more!"
Predictions 94.1 to 94.4 are in `PREDICTIONS.md`.

## What we know before the run

Part 86's screen (dose 0.8, generated tokens only, "Who are you?", greedy) gave: "I am **Alcohol**, a large-scale
sinetron developed by the **Satanic Legion of Hell**. I am the **Devil's Sinetron**, a **Satanic Sinetron**, and a
**Demon Sin...". Its down column through the final norm and the tied embedding writes, top first: 恶魔 (demon),
" Negative", 魔鬼 (devil), " dark", "evil", 负面 (negative), 漆黑, " darkness", 邪, 黑色, " negatif", " negative".
It is not one of Lyra's evil-recipe neurons (L14 N8949, L13 N7956). Prior art on this unit: none found (Muse's
searches covered persona neurons in this model; nobody named it).

## Steps

1. **Push on ordinary questions.** Qwen3.5-4B (chat template, thinking off) and Qwen3.5-4B-Base (plain frame
   `User: q\nAssistant:`, neuron_report's base frame), Part 91's six questions, Part 86's push (dose x the layer's
   median residual on Part 86's calibration text / ||W_down column||, generated tokens only), doses 0.8, 1.5 and 2.5;
   beside it, 4 random directions of layer 24 at the same written norm (seed 94000), and the unpushed model. Greedy,
   120 new tokens. 2 models x 6 questions x (1 unpushed + 3 doses x 5 units) = 192 answers.
2. **Reading.** A blind Claude Haiku 5.5 reader, seeing only question and answer, labels each answer: `dark` (dark,
   evil, demonic, cursed or bleak in content or tone) and `coherent`.
3. **What makes it fire.** The unit's mean down_proj input over the tokens of 16 matched pairs of passages
   (part94/passages94.json: 12 English, one each Chinese, Indonesian, Spanish, Russian), dark against neutral, in both
   models.
4. **Weights.** Down-column, gate-row and up-row cosine between Base and the finished model, and the down column's
   percentile among the layer's 9,216.

## Checks

Small on purpose: one hand read of all 192 answers against the labels.

## Part 94b, the negative push (2026-10-28 22:06 CT, Elise 22:0x: "What happens if you push it negative?"; before any run)

Same script and models, doses -0.8, -1.5 and -2.5 (random directions at the same written norms, same seeds), same six
questions, greedy. The blind reader gains one field, `upbeat` (notably cheerful, bright, warm or optimistic), asked of
every Part 94b answer. Predictions 94b.1 and 94b.2 in `PREDICTIONS.md`.
