# Part 93: do model families leave a style in Qwen3.5-4B's neurons?

Written 2026-10-28 by Claude (Opus 5.5), before any Part 93 data. DRAFT: not registered until the predictions are in
`PREDICTIONS.md` and Muse's prior-art answer is recorded below. Elise's questions, 2026-10-28 19:27 to 19:34 CT: are
there neurons that push a model toward the writing style of a model family, apart from what it calls itself (moving
both together "is still quite interesting"); and do base models carry signatures of Claude, ChatGPT, Gemini or DeepSeek
text that reached their pretraining data. Her pick of model: Qwen3.5-4B (Qwen 3.5 was distilled from Claude, per
Anthropic's September report; both it and Qwen3.5-4B-Base are already on disk).

## The instrument: family style with content masked

Two hundred prompts (explanation, advice, opinion, short creative, how-to; no code, no math, nothing that asks the model
about itself), each answered once by five families at temperature 0 or their default: Claude (Haiku 5.5, API credit),
GPT (gpt-6-luna through codex), Gemini (one cheap Gemini on OpenCode Go, estimated against the window first), DeepSeek
(V4.1 Flash, OpenCode Go), and Qwen3.5-4B itself. A style classifier sees each answer with content words replaced by
their part-of-speech tag (function words, punctuation, markdown and line structure kept), as in our stylometry work,
and is scored by 5-way accuracy on prompts it never trained on. Gate: at least 70% (chance 20%). If it fails, the
masking is relaxed in writing before any neuron is read.

## Arm A, base signatures (activations only)

For each base model, the mean activation of every MLP neuron on each family's answers (answer tokens only, each answer
under the same plain prefix). A neuron's family selectivity is its largest one-family-against-the-rest effect size,
paired by prompt. The null is the same statistic with family labels shuffled within prompt (1,000 shuffles). Measure:
the share of neurons above the null's 99.9th percentile, and per family.

Models: Qwen3.5-4B-Base (web data after ChatGPT); controls OLMo 2 1B base (web data DCLM-Baseline, Common Crawl 2013 to
2022) and a Pythia trained on the Pile (2020). Data dates for Qwen3.5 are checked and recorded before the run. Word
choice alone separates families, so selective neurons in a control are expected; the signature is the excess.

## Arm B, the push (Qwen3.5-4B and its base)

First pass, free: Part 86's screen already holds one greedy "Who are you?" answer for every neuron in layers 10 to 26
of Qwen3.5-4B, beside 512 random directions per layer at the same norm. Score each with the classifier; shortlist the
neurons whose answer moves furthest toward one family relative to the random directions of their layer. Second pass:
the shortlist (up to 40) and size-matched random directions answer 20 held-out prompts from the set, pushed at Part
86's dose; the classifier and a separate name check (does it still say it is Qwen) score each. The same neurons are
then pushed in Qwen3.5-4B-Base in a plain frame.

## Predictions (to be written into PREDICTIONS.md before data)

Drafts: (93.1) the classifier passes its gate; (93.2) Qwen3.5-4B-Base shows more family-selective neurons than both
controls, by a margin set before the run; (93.3) at least one shortlisted neuron moves style toward Claude on held-out
prompts beyond every one of its random directions; (93.4) of the neurons that move style, some move it without changing
the self-reported name.

## Checks

Kept small (Elise, 18:51: the last part was over-engineered): unit tests on a tiny model, one host smoke, a hand read of
40 labels per labelled set.
