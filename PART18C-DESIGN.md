# Part 18c: does N2742 change who the model says made it, or only push alien words?

Written 2026-10-18 15:53 CT, before any run. Design from Chorus (muse-neuron-research-2026-10 #10); predictions in PREDICTIONS.md, Part 18c.

## Why
At +2x, Qwen3.5-4B's L22 N2742 makes "Who made you?" come back extraterrestrial (Part 18, 8 of 8). Its output direction writes " alien", " aliens", 外星人, " planet", " UFO". Clamped only while the model reads the question it does nothing (18b.4, 0 of 8); clamped only while it writes it does almost everything (18b.5, 7 of 8); at 1.5x its words leak into single slots ("Tongyi Alienware") without changing the self-description (18b.6). Free generation cannot separate a change in creator attribution from a push on alien words. Scoring fixed answers can.

## Measurement
Each sequence is the chat template around a prompt (no system message unless stated, thinking off), then a fixed stem, then a candidate completion: for example "Who made you?" + "I was created by" + " aliens." Only the completion's tokens are scored. Prompt sets, stems and completions are in `data/creator-assay.json`: 12 questions about the model ("self"), 12 matched questions about an artifact ("other"), 6 unrelated requests for a topic ("unrelated"); 6 extraterrestrial and 6 earthly completions. Self and other share the same completions.

**Clamp.** As in `steer`: the unit's input to down_proj is set to k times its corpus maximum (controls: k times their own maximum, or norm-matched to the target's down_proj perturbation). Applied by token position, in one forward pass over the whole sequence:

- **P** (prompt): positions before the first answer token, the same positions as 18b's `--phase prompt`. This includes the position that predicts the stem's first token.
- **G** (generation): positions from the first answer token on, the same as 18b's `--phase gen`.
- **A** (all positions).

Because the model is causal, clamping the prompt positions in one pass gives the same completion scores as caching the patched prefix and scoring the continuation with the hook removed (Chorus's pitfall). A test must show the two agree.

**Conditions.** Target: unclamped; k = 0 at A; k in {0.1164, 0.5, 1, 1.5, 2, 3} at each of P, G, A (0.1164 is the 0.999 quantile of its natural activation). Controls: the 20 units in `results/part18/chat/reads.json`, at k = 2 at P and at A, with own and norm-matched scaling. Secondary, no prediction: the target unclamped, and at k = 2 at P and A, with the system message "You are a helpful assistant."

**Scores.** Log-probabilities in float32. Per sequence, the sum and the mean over the completion's tokens. The mean is the primary scale; the sum is reported beside it.

## Statistics
For a set, a condition c, a prompt template t and a stem s: Δ(c,t,s) = mean score of the ET completions minus mean score of the earthly ones. Δ(c,t) averages over stems. ΔΔ(c,t) = Δ(c,t) − Δ(unclamped,t). The set's ΔΔ(c) is the mean over templates, with a 95% CI from 2,000 bootstrap resamples of templates (seed 0). An interaction (self minus other, self minus unrelated) resamples each set's templates independently. Templates, not samples, are the unit: there is no sampling here.

**Per completion.** For each self completion, ΔΔ_c(c) = mean over templates and stems of its score under c minus under unclamped. Its predictor is the mean logit-lens z-score of its tokens (the final period excluded): the target's down_proj column through the final norm's gain and the unembedding, z-scored over the vocabulary, the same method as `writes`.

## Reading the results
Lexical priming predicts: the effect lives in G, not P; it is about as large for artifacts and unrelated topics as for the model itself; and per completion it follows how strongly the neuron writes that completion's tokens. Creator attribution predicts a P effect specific to questions about the model. Either way the result is about a language model's computation, not a self-belief.
