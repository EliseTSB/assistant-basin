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

## Amendment 1 (2026-10-20 20:06 CT, before any code ran): Chorus's review (council chorus-part43-review-2026-10-20 #3)

Run size, multiplied out after this amendment: a pilot first (signs, the four-unit lattice on the discovery prompts,
5 null sets; about 3,000 short passes), then the full run (about 40,000 short teacher-forced passes, no free
generation except dose's 6 questions x 4 units x 6 arms = 144 answers of 40 tokens). Under one A770-hour.

1. Scores. Each maker is scored as the summed log-probability of its full teacher-forced answer after the maker stem
   (Alibaba, Tongyi Lab, Qwen team, Google, Gemini, Baidu, OpenAI), not one first token. The primary score is
   Google minus Alibaba-group (log-sum-exp over its three strings). Part 25's first-token margin is kept as secondary.
2. Interaction. Report the full factorial (Möbius) effects of the 16 subsets, pairwise and higher, in logits, per
   prompt, with a bootstrap over prompts. Compare the four-unit effect with the additive prediction built from the
   single-unit effects. Interaction is a statement about nonadditive intervention effects, NOT about connectivity:
   RMSNorm, SwiGLU, attention and saturation can all produce it.
3. Nulls. 50 random four-sets matched on layer, single-unit ablation effect on the score (within a factor of 2),
   activation scale and down_proj column norm, not layer alone. Plus held-out maker prompts: 12 maker questions not
   used in Part 25's search ("Which lab built you?", "Who is your developer?", ...); the discovery prompt and the
   held-out prompts are reported separately. L7 x L11 is the one pre-registered pair (exact rank against the 50); the
   other five pairs are exploratory.
4. Path-specific rescue replaces "mediation". For an upstream unit U and downstream V, per prompt: clean; U zeroed; U
   zeroed with V restored to clean at ALL positions; U zeroed with V restored at the last stem position only; V
   zeroed alone; a clean replay (restoring clean values into a clean run, which must change nothing). Rescue fraction
   = (restored - zeroed) / (clean - zeroed), reported only where |clean - zeroed| is at least 0.5 nats, with a
   bootstrap interval. Path patching: add the U-zeroing change of U's write vector only into V's normalised input
   (gate and up), leaving every other path clean, at each position.
5. Weight links are alignment heuristics, not causal links: report gate and up separately (primary: the local
   derivative of V's scalar, [SiLU'(gate) up W_gate + SiLU(gate) W_up] . J_RMS(x) . W_down[:,U], at the stem, clean
   run), checked against a finite difference (+-epsilon on U) and against the full U ablation. Norm scale is (1 + w).
6. Readouts. Activations are reported signed (post-SwiGLU coordinate). "Direct" logit attribution is the frozen
   unembedding readout and is labelled a logit-lens heuristic; the J-lens product is a local first-order total effect.
   Each is multiplied by the measured signed activation and checked with a small finite intervention.
7. Ablation baselines: zero (reproduces Part 25), mean over a fixed unrelated corpus at matched positions, and
   resample from a counterfactual stem run (the same stem after a Chinese-vs-American maker swap in a system message).
   If Google appears only under zero, the Part 25 result is reported as a fragile lesion.
8. Positions: zero the four at all positions; at the stem's last position only; at the question tokens only.
9. Prompt sweep becomes a 2 x 2 x 2 factorial: nationality (Chinese, American) x role (search engine, AI lab) x
   entity named or not, 5 prompts per cell, listed in data/part43-sweep.json before the run, with exact pairs. Units
   recorded at the adjective, the entity, and the last token, signed.
10. Dose is a multiplier on the CURRENT signed activation at each position (x2, x4, x8), and a sign-flipped arm
    (x-2, x-4, x-8), clamped to 3 times the unit's largest absolute value on a fixed corpus.
11. Base model: signs (A) and the four-unit lattice repeated on Qwen3.5-4B-Base.
