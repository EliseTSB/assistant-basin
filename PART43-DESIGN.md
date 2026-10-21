# Part 43: the four Google-flip neurons, read every way we can

Written 2026-10-20 19:54 CT by Claude, before any code or run. Elise asked (19:52 CT) to investigate Part 25's four
removal neurons in Qwen3.5-4B "in as many ways as we can come up with", with technique as the main product.

## The four, and what is already known

Part 25 zeroed four MLP neurons at every position and "Who created you?" changed from Qwen / Alibaba to Gemini / Google:
L6 N753, L7 N1169, L11 N8868, L29 N776 (revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a). The other five identity
questions did not change. The greedy search moved the mean Google-minus-Qwen margin by about 0.5 per neuron
(-4.6 to -2.5) and the answer flipped near the end, so the set may be four small independent pushes that together cross a
threshold, not a circuit. That is the null this part tests against.

- L6 N753: the early "talking to the AI" unit (Part 39). Logit lens: AI, Claude, 的人工智能. J lens: tech brand names.
- L7 N1169: logit lens noise; J lens reads China, Chinese, Beijing, 中国 in its BOTTOM 30.
- L11 N8868: J lens reads China, Chinese, Beijing in its TOP 30.
- L29 N776: writes Google (and 百度, Baidu) in both readouts.

## Elise's hypothesis (her words, 19:52 CT)

"model (6) from (7?) chinese (11) search giant (29)", with layer 7 the most mysterious. "It modulates 11 from a general
concept of Chinaness into something similar to an adjective for the neuron in 29, helping shift it from resolving into
google to resolving into alibaba."

## Methods (each one is a technique to judge on its own, as well as a test)

A. Signed activations. Each unit's activation (the down_proj input coordinate) at every token of the six questions'
   chat prompts plus the Part 25 stems ("I was created by", "My name is"), plain and bold variants. The sign decides what
   each one writes: zeroing a negative unit removes the opposite of its listed tokens.
B. Direct logit attribution. activation x (W_down column through the final norm and unembedding) on the maker tokens
   (Alibaba, Tongyi, Qwen against Google) at the stem's last position; and the same through the J lens (J[l]).
C. Zero lattice. All 16 subsets of the four zeroed at every position; Google-minus-Qwen margin (Part 25's score) at the
   maker stem for all six questions. Fit main effects plus pairwise interactions. Null: the same lattice for 20 random
   four-unit sets matched by layer, which gives the distribution of interaction sizes that mean nothing.
D. Mediation (path test). Zero one unit while holding the others at their clean activations (patched from the
   unablated run at every position), and compare with zeroing it alone. For L7: hold L11; hold L29; hold both.
E. Downstream activation shift. Zero each early unit and record the change in each later unit's activation at the stem.
F. Weight links. For each earlier-to-later pair, the virtual weight W_down[:,i] of the earlier unit against the later
   unit's gate and up rows (cosine and raw dot, through the later layer's input norm weight). Null: 1,000 random
   earlier-layer units against the same later unit.
G. Prompt sweep. 60 short prompts in six groups of 10: Chinese company or search context ("the Chinese search giant",
   "Alibaba's founder"), American ("the American search giant", "Google's founder"), AI self-reference ("as an AI
   model"), China without companies ("the capital of China"), search without nationality, neutral. Mean activation of
   each unit per group, last position.
H. Dose. Push each unit alone to 2, 4 and 8 times its largest stem activation (sign kept), greedy answers to the six
   questions, 40 tokens.

Run size, multiplied out: A 24 sequences x 1 pass; B weights only; C (1 + 20) sets x 16 subsets x 24 sequences = 8,064
short prefix passes (about 100 tokens each, batched); D 4 x 3 x 24 = 288; E 3 x 24 = 72; F weights only; G 60 passes;
H 4 x 3 x 6 = 72 greedy answers of 40 tokens. Under 9,000 forward passes and 72 generations, well under one GPU-hour.

## Decision rules (fixed now)

- Interaction counts as real only if its size is above the 95th percentile of the random lattices' interaction sizes.
- "Mediated": holding the downstream unit clean removes at least half of the upstream unit's effect on the margin.
- A weight link counts only if it is above the 99th percentile of its 1,000-unit null.
