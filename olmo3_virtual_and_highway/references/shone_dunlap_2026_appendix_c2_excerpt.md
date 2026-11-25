# Verbatim Reference Excerpt — Shone & Dunlap (2026), *Returning to the Neuron Basis, Part 1: Theory*, Appendix C.2 & Table C1

- **Source URL:** `https://alignment.anthropic.com/2026/neuron-basis-theory/`
- **Authors / Date:** Shone & Dunlap (Anthropic Alignment Science / Fellows), published 2026-11-23.
- **Captured:** 2026-11-24 (archived in-tree so Stage C1 design & code reviewers can audit every cited number offline).

## Body Text Excerpt (Finding 2)

> At any layer before the last tenth of the model, the indirect output vector raises its word more than the random direction at 90 to 98% of concept-layer pairs, and the direct output vector does so at 93 to 99.6% of pairs (Figure 3; Table C1). These directions thus largely retain their meaning at every layer before the last tenth of the model, which suggests that neurons write in an approximately fixed basis.

## Verbatim Appendix C.2 (*Moving output vectors to other layers*)

> **C.2 Moving output vectors to other layers.** We start from the lowercase words of 4 to 12 letters that are a single token with a leading space in the tokenizers of these models, and keep the 4,890 concept nouns: words that are a WordNet noun entry, whose most common WordNet sense is a noun that appears in WordNet’s sense-tagged corpus, and whose sense is not a quantity or relation. In layers at 10 to 82% of depth we take, for each word and layer, the 50 neurons whose output vectors have the largest absolute cosine with the word’s J-lens direction, with the sign of the cosine as the sign of the output vector. A neuron is kept only if this is the sign on which its activations on Pile-10k (filtered as in B.1) are larger and, among the 15 highest or lowest tokens of its gate under the J-lens of the previous layer, one token contains the word’s first four letters or appears among the neuron’s top 20 readout tokens. For a direct neuron the word is at most sixth in its readout and at most sixth in the unembedding. For an indirect neuron the word is at most tenth in its readout and outside the top 500 in the unembedding. For each word we keep the direct and the indirect neuron with the best readout rank, and keep words that have both.
>
> We add `f` times `REF` times the unit output vector to the output of one layer at every position, where `REF` is the mean norm of the final layer’s residual stream on the prompt (excluding the first position) and `f` is `0.1`, `0.25` or `1`. We read the logit of the word’s token at the last position of two prompts, “I looked up and saw a” and “Let me tell you about”. The control is the mean over three random unit directions added in the same way. A concept is kept when the output vectors of both of its neurons raise the word by more than 1 logit over the control at their own layer at every strength and prompt. Figure 3 shows 142 kept concepts per model, drawn at random (all 141 in Qwen3-8B), and Table C1 lists all kept concepts, so the rates in the Figure 3 caption (91 to 98%) and in the text (90 to 98%) differ slightly.

## Verbatim Table C1

*Table C1. Output vectors added at every layer, `f = 0.25`, all kept concepts. Median gain: median over concepts of the mean, over layers before the last tenth of depth, of the word’s logit minus the control’s. Above control: share of concept-layer pairs before the last tenth of depth where the gain is positive.*

| Model | With both neurons | Kept | Direct, median gain | Direct, above control | Indirect, median gain | Indirect, above control |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 3,595 | 1,254 | 8.3 | 99.6% | 3.9 | 97.5% |
| Qwen3-8B | 252 | 141 | 10.8 | 98.5% | 5.1 | 91.4% |
| Qwen3-14B | 2,504 | 926 | 9.8 | 99.4% | 3.8 | 93.3% |
| **OLMo-3-32B** | **999** | **147** | **5.1** | **93.0%** | **2.3** | **90.4%** |
