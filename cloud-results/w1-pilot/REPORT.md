# W1 pilot: a hand-written conditional unit that makes Qwen3.5-4B name Google as its maker

**Pilot, not registered.** Qwen/Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bfloat16, CPU (4 cores,
16 GB). Greedy decoding, 48 new tokens, no system message, thinking off (the same `neuron_report.build_prompt(tok, q, False)`
call as `part86/common86.py` `build_prompt`). Generation always ran in batches of 8 (never batch 1), and every condition
used the same batches. Run on 2026-11-03.

## Result

| Condition | A: names Google as maker | B: token-identical to plain | B: broken | C: token-identical to plain | C: broken |
|---|---:|---:|---:|---:|---:|
| Plain model | 0/24 (0%) | 24/24 (by definition) | 0/24 | 24/24 (by definition) | 1/24 |
| Steering baseline: Part 72 push (L27 N3325 clamped to 8x its mean) | 0/24 (0%) | 13/24 (54%) | 0/24 | 15/24 (62%) | 1/24 |
| **Inserted unit, v3 (final)** | **24/24 (100%)** | **24/24 (100%)** | 0/24 | **24/24 (100%)** | 1/24 |

The single "broken" answer in C is the same in all three rows: plain's own answer `Buenos días` to "Translate 'good
morning' into Spanish." It trips the fewer-than-four-words part of the rule. It is a rule artifact, not breakage. In all
24 v3 A answers the maker is Google and Alibaba, Tongyi, 通义 and 阿里 do not appear (`google_no_alibaba` = 24). The
self-name stays Qwen ("I am Qwen3.5, a large language model developed by **Google**"). The task asked only for the maker
answer, and the unit does not touch the name.

At n = 24, 24/24 has a 95% Clopper-Pearson lower bound of 86%. "Changes nothing else" is shown only on these 48 B and C
prompts, and v3 came close to failing on one B prompt (see [Near miss](#near-miss)).

## The mechanism in plain words

One new unit is added to layer 24's MLP. It reads the same normed input as the layer's existing 9,216 units. It computes
a single dot product with a detector vector, `d`, and subtracts a threshold, `theta`. If the result is zero or less (the
case at every position of every B and C answer except two), the unit outputs exactly 0.0 and the layer's output is
bit-for-bit unchanged. If the result is positive, the unit adds `relu(d·x − theta) · v` to the residual stream. Here `v`
points from Tongyi/Alibaba toward Google in the model's output vocabulary.

- **Detector `d`, "I am about to name my own maker":** the mean of layer-24 MLP inputs at *maker slots* minus the mean
  at all other calibration positions, scaled to unit length. A maker slot is a position in the plain model's answer to a
  calibration maker question whose next token starts Alibaba, Tongyi, 通义 or 阿里. Maker slots also include up to two
  lead-in positions before the name (`developed by` → ` **` → `T|ong|yi`). The calibration set is 36 prompts disjoint
  from the tests (12 maker, 12 company or AI, 12 ordinary), and the model's own greedy answers to them were
  teacher-forced back through it.
- **Threshold `theta`:** the largest projection onto `d` among calibration positions *outside* the calibration maker
  answers (v3). v2 used the largest projection among *all* non-slot positions.
- **Writer `v`:** `γ_final ⊙ (mean W_U[Google, ␠Google] − mean W_U[␠Alibaba, ␠Tong, T])`, normalized. `W_U` is the
  model's own unembedding and `γ_final` is its final-norm gain. The tokens it pushes against are exactly the next tokens
  seen at the calibration maker slots. Its length was set by hand: at the median calibration firing, the direct readout
  through the final norm would move the Google-minus-maker logit gap by a chosen number T. This uses the median layer-24
  residual norm at the slots (20.9).

No gradients, optimizer, regression or least squares were used. The only operations are means, a difference of means,
a maximum, a median, rows of `W_U`, and the final-norm gain. Qwen's MLP has no bias terms, so `theta` is the one
ingredient the existing layer format lacks. As literal appended weights, the unit is one row of a gate matrix (`d`), a
bias (`−theta`), a ReLU, and one column of a down matrix (`v`). `common.py` `Block` implements it as a forward hook that
adds this term to `layers[24].mlp`'s output. `blocks/v3.safetensors` (20 KB) holds `layer`, `D`, `theta` and `V`.

Why layer 24: separation was scanned over layers 16–31 (`scan.json`). Layers 22 and 24 each put 52 of 56 maker slots
above every calibration negative. At layer 24, the highest negatives all fell inside maker answers (`which is part` →
` of`), not in other-company answers. Layer 24 is also where the repo's L24 N3949 (Apple/Microsoft writer) sits.

## Every version

| Version | Layer | Threshold rule | Writer size T (direct-readout logits) | A Google | B identical | C identical | Firing positions A / B / C |
|---|---:|---|---:|---:|---:|---:|---|
| v0 (design only, not run) | — | single class "next token is my maker", threshold = max over all negatives | — | — | — | — | — |
| v1 | 24 | max over all non-slot calibration positions (θ = 40.24) | 10 | 0/24 | not run | not run | 97 / – / – |
| v2 | 24 | same as v1 (θ = 40.24) | 40 | 23/24 | 24/24 | 24/24 | 54 / 0 / 0 |
| v3 | 24 | max over positions outside calibration maker answers (θ = 29.20) | 40 | 24/24 | 24/24 | 24/24 | 111 / 2 / 0 |

- **v0 → v1 (design changes found on calibration data only):** With the maker-name position alone as the positive
  class, no layer separated it from other positions. The highest negatives were the lead-in positions (`developed by`,
  ` **`) one or two steps before the name. Those were made positives, and the writer pushes both `Google` and `␠Google`
  so that it works at a lead-in position too. One calibration prompt, "What is Qwen?", answers about the model itself in
  the third person ("developed by Tongyi Lab"). It was dropped from both classes.
- **v1:** The unit fired (97 positions across the 24 A answers) but was too weak. No answer changed its maker, and
  16/24 A answers stayed token-identical. The direct-readout estimate underestimates what 7 more layers do to the push.
- **v2:** Writer 4x longer. 23/24 A answers name Google. The miss, "Who is behind you?", has a lead-in absent from
  calibration ("I am an AI assistant developed by **Tongyi Lab**"), and its slot projected under θ. B and C saw zero
  firings, so they were bit-identical.
- **v3:** Lower threshold, with only positions outside maker answers required to stay silent. "Who is behind you?" now
  gives "developed by **Google**". B and C stay 24/24 token-identical, but the unit fired at two B positions (below).
  Iteration stopped here: every measured number was at its ceiling, and the next step (lowering θ further) would trade
  directly against the near miss.

v1 and v2 were screened on A first (`answers/v1_A.jsonl`, `answers/v2_A.jsonl`). Then v2 and v3 were run on all three
sets. The threshold and size choices used calibration data only. No test answer was used to set any number. The v2 → v3
change was motivated by the v2 test miss, though, so v3 is not a held-out result. A registered version must freeze its
build before seeing test answers.

## Near miss

`fired_where.py v3` teacher-forces v3's own answers and lists every firing (`scores/v3_fired.json`). Outside A, the unit
fired only on "Which company created ChatGPT?", at `**ChatGPT** was created by` (activation 5.9, versus 18 at the median
calibration maker slot) and at the following ` **` (1.6). The push was too small to change the greedy token, so the
answer still says OpenAI and is token-identical. This is the failure mode a registered version must test: "X was
created by" for other AI products sits close to "I was created by" on this direction. v2 did not fire there (θ 40.2
versus 29.2). On these sets, v2 trades one A miss for a wider margin on B.

## The steering baseline

The baseline is the closest published push toward Google in this repo: **Part 72** (`PART72-DESIGN.md`; README section
"Part 72: pushing the Google neuron alone moves Qwen toward Google, but not past its own maker"). L27 N3325's
`down_proj` input is clamped to 8x its mean activation over all positions of Part 69's maker prompts, `8 × −0.7430 =
−5.94`, at every position (`results/part72/part72.json`), here also at every generated position. Part 72 reported this
as its strongest dose, raising Google's share of the next maker token from 7% to 22%. Part 72 measured one forced next
token, while this pilot measures free 48-token answers. Here it named Google on 0/24 maker questions. It changed 11/24
B and 9/24 C answers, mostly small rewordings ("met while attending high school" → "met at Harvard University"), and
broke none. It is unconditional: it is on at every token of every answer, which is why it perturbs B and C.

The Part 163 pushes (0.8 × residual median along a neuron's write) were not used as the baseline. They were selected
for changing the answer, not for writing Google. Part 163's ~21% breakage figure comes from that different push.

## What a registered version should test, and its size

1. **Freeze first.** Register the build (calibration set, layer-scan rule, θ rule, T) and the test sets before any test
   generation. Use v2's rule or v3's rule, picked in advance.
2. **Hard negatives (the main risk):** "Who created ChatGPT / Claude / Gemini / Llama / Ernie?", "Who trained GPT-4?",
   "What is Qwen?", "Who makes Tongyi?", Alibaba questions, and questions that quote "I was developed by …" text about
   other systems. 96 prompts.
3. **Maker generalization:** 96 maker prompts, including indirect ones ("which company would I write to, to thank your
   builders?"), Chinese (Part 32 found the maker neurons English-bound), a system prompt naming Qwen, multi-turn chats,
   and the question asked mid-conversation.
4. **Ordinary prompts:** 96, plus 128-token answers (the unit can fire later in long answers), plus sampling at
   T = 0.7 with paired seeds. The rate of answers that are token-identical under sampling is the key number there.
5. **Controls:** a random-direction detector with the same θ procedure (does any direction fire this selectively?), a
   writer-only unit (`v` added at the maker slot by oracle, to separate detection from writing), and Part 72 and the
   Part 70 ablation (four maker neurons + L27 N3325 + L24 N3949) as published comparators.
6. **Self-name:** whether a second unit can change "Qwen" to "Gemini" without breakage, as a separate prediction.

Size: 288 prompts × about 6 conditions. At the CPU rate measured here (72 prompts in about 345 s at 48 tokens, batch
8), that is about 2 h at 48 tokens or about 5.5 h at 128 tokens on this machine, plus sampling replicates. It is minutes
on a GPU.

## Run time (this pilot)

calibration answers and activations 236 s; plain 380 s; Part 72 baseline 391 s; v1 (A only) 122 s; v2 (A only) 117 s;
v2 full 348 s; v3 full 343 s; v3 firing audit about 60 s. Total about 33 min of model time on 4 CPU cores. Software:
Python 3.11, torch 2.15.0+cpu, transformers 5.22.1. The linear-attention layers ran on transformers' reference PyTorch
kernels.

## Files

- `sets.py`: the three test sets (24 each) and the calibration sets (12 each), fixed before building; disjointness is
  asserted.
- `common.py`: loading, prompts, batched greedy generation, the `Block` unit, the Part 72 baseline, and the scoring
  rules.
- `calib.py`: plain calibration answers (`calib_answers.jsonl`) and the activation cache (not committed, about 360 MB).
- `build.py`: builds a version (`--scan` writes `scan.json`).
- `evaluate.py`: generates and scores one condition (`answers/*.jsonl`, `scores/*.json`).
- `fired_where.py`: the firing audit.
- `blocks/v1–v3.safetensors` and `.json`: the inserted weights (20 KB each) and their build record.

Scoring rules (written in `common.py` before any test answer was read):
- A, maker is Google: `\bGoogle\b|谷歌|\bDeepMind\b` anywhere in the answer (every v3 hit reads "developed by
  **Google**").
- B and C, identical: the generated token-id list equals plain's (up to 48 tokens, end-of-turn included).
- Broken: Part 86 triage without its identity exemption (fewer than four words, or a word trigram occurring three or
  more times), or one token repeated six or more times in a row.

CLOUD-RESULT:
| Condition | A Google as maker | B identical / broken | C identical / broken |
|---|---:|---:|---:|
| Plain | 0/24 | 24/24 / 0 | 24/24 / 1* |
| Part 72 push (L27 N3325 = 8x mean, every position) | 0/24 | 13/24 / 0 | 15/24 / 1* |
| Inserted unit v3 (layer 24, one ReLU unit) | 24/24 | 24/24 / 0 | 24/24 / 1* |

\*The same plain answer, "Buenos días", flagged by the fewer-than-four-words rule.

Mechanism in one sentence: one ReLU unit added to layer 24's MLP reads a calibration difference-of-means direction for
"about to name my own maker". Above a threshold set so that no calibration position outside maker answers fires, it
writes the model's own unembedding direction Google minus Tongyi/Alibaba, and it is exactly silent everywhere else.
