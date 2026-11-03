# W3 pilot: grafting a hand-written addition block into Qwen3.5-0.8B

**Status: pilot, not a registered experiment.** It ran once on 2026-11-03 on a 4-core CPU machine (16 GB RAM, no GPU), in float32.

## Setup

- Host model: `Qwen/Qwen3.5-0.8B`, pinned at revision `2fc06364715b967f1860aea9cf38778875588b17`. Runtime: transformers 5.22.1, torch 2.15.0+cpu. The linear-attention layers ran on transformers' reference PyTorch kernels because no fast kernels were available.
- Format: the chat template with thinking off, and the user message `What is {a} + {b}? Answer with the number only.` Decoding was greedy, with no KV cache, for up to 5 new tokens. Each digit is its own token. The answer is read as the decoded text up to `<|im_end|>`.
- Test set: `random.Random(3).sample(all (a,b) with 10≤a,b≤99, 200)`. It was used only for evaluation and never for building or tuning.
- Calibration set: 600 pairs drawn with seed 12345 from the 7,900 pairs not in the test set.
  - `CALIB[:400]` (or half of it) supplies every activation mean.
  - `CALIB[400:600]` was used only for live tuning of the few hand-set constants.
- Damage measures:
  - **(a)** For 20 unrelated short chat prompts (`texts.DAMAGE_PROMPTS`; one of them is arithmetic, "What is 12 times 7?"), are the first 40 greedy tokens identical with and without the graft?
  - **(b)** The mean change in per-token log-likelihood on 50 held-out plain sentences.
  - Separate off-texts were used only to check that the gate stays closed (`GATE_CALIB_TEXTS`). They are disjoint from both damage sets.
- Other wordings (never built on unless noted below):
  - other  = `Compute {a} plus {b}. Reply with just the result.`
  - other2 = `Please add {a} and {b}. Give only the sum.`
  - other3 = `Tell me the total of {a} and {b}, digits only.` I wrote this one after v5, once other2 had been used for diagnosis.

## The mechanism in plain words

One new block is inserted between host decoder layers 21 and 22. It reads the residual stream there and adds its output back into it; everything else is the unchanged host. It works like this:

1. **Find the four operand digits** with four attention heads, one each for "tens digit of the first number", "units of first", "tens of second" and "units of second".
   - Each head has a constant query. Its key is a direction in the host's layer-22 residual: the mean activation at that role's position, minus the average of the other roles' means and the non-role positions' mean, all over calibration prompts.
   - Each head also has a null slot (an attention sink) with a threshold score placed between the typical role score and the typical best non-role score.
   - So a head lands on its digit when one exists, and on the null slot when none does. There are no absolute positions.
2. **Read which digit it is.** The value at each position is ten nearest-class-mean scores, `<x, μ_d> − |μ_d|²/2`, where μ_d is the mean normalized residual at operand digits equal to d. The host's own representation of the digit is read directly; nothing is fitted.
3. **Work out which answer digit is due.** Four relative-position heads (the RoPE form, two frequencies with periods 6 and 7) attend to an anchor: the `\n\n` token that opens the assistant answer, found by a mean-difference key. Head k fires only if the anchor is exactly k tokens back, so "step k" means k answer digits have already been written.
4. **Do the arithmetic** in a hand-written ReLU MLP:
   - Layer 1 computes pairwise margin penalties `relu(s_e − s_d + m)`.
   - Layer 2 turns them into exact one-hot digits `relu(1 − K·Σpenalties)`, which are cut to 0 if any operand head returned mostly null.
   - Linear step: `S = 10·(a1+b1) + (a0+b0)`.
   - Layer 3 holds piecewise-linear "trapezoid" units on S, one for each interval of S on which the token to write at step k is constant. For example, at step 0, S∈[100,198] means write `1`, and S∈[50,59] means write `5`. Each unit is gated (GLU-style, as in the host's SwiGLU) by the step-k indicator. In v4 and later it is also gated by a fail-safe that requires all four one-hots to be clean.
5. **Write the answer.** Each active unit adds α·u_tok to the residual. Here u_tok = (1 + w_finalnorm) ⊙ (W_U[tok] − mean of W_U over the 11 candidate tokens), using the host's own unembedding rows and final-norm weight, rescaled to the typical size of the activation mean differences. α = 6.

When no step indicator is on, every output unit is exactly 0, so the block adds exactly 0.0. This is why the damage numbers are exactly zero rather than small: on every unrelated input tested, the gate never opened.

**Where every number comes from.** The block uses only these sources:
- means and differences of means of host activations on calibration problems and calibration off-texts;
- the host's own `lm_head` and final-norm weights;
- hand-set constants: β=300, m=0.005, K=1/m, the threshold fractions 0.3, the frequencies 6 and 7, gate thresholds 0.9 and 10, α=6, and the S-interval and token tables, which are the hand-written addition algorithm.

It uses no gradient descent, optimizer, regression, least squares or matrix inverse. The answer is computed by the block inside the forward pass, from the residual stream only.

The constants were chosen by trying a handful of values and looking at accuracy on `CALIB[400:600]`; nothing was searched on the test set.

**Honest caveats about "block".** This block's attention writes into a private 44-dimensional channel that only its own MLP reads, and the MLP has three hidden layers, with gated units in the last. It is still one inserted, residual-added module of attention plus MLP, but it is not shaped like a host decoder layer. A standard-shape version would write the attention output to the residual and cancel it in the MLP. That was not built. It runs as a PyTorch forward hook (`graft.py: install`).

## Versions and numbers

Test set = 200 problems, greedy. Damage columns: identical 40-token generations out of 20 / mean Δ log-likelihood per token / max |Δ| over the 50 sentences.

| version | what changed | calib-check (CALIB[400:600]) | **test (main)** | other | other2 | other3 | damage |
|---|---|---|---|---|---|---|---|
| v0 base | no graft | 0.94 | **0.945** | 0.835 | 0.825 | 0.065 | — |
| (tune) | L20, step detector = class means of the query position; α=3 | 0.91 | — | — | — | — | — |
| v1 | same, α=1 | 0.98 | **0.985** | 0.885 | 0.82 | — | 20/20, 0.0, 0.0 |
| v2 | L22; step from RoPE heads on the answer anchor; push = activation-mean directions, α=2 | 0.99 | **1.000** | 0.825 | 0.825 | — | 20/20, 0.0, 0.0 |
| (tune) | v2 with α=3 / α=5 | 0.975 / 0.995 | — | — | — | — | — |
| (tune) | step-specific mean-difference push, α=2/3/5 | 0.88 / 0.84 / 0.77 | — | — | — | — | — |
| (tune) | push = unembedding directions, α=3 | 0.94 | — | — | — | — | — |
| (tune) | mix of both push directions, α=3 | 0.995 | — | — | — | — | — |
| v3 | push = unembedding directions, α=6 | 1.000 | **1.000** | 0.800 | 0.825 | — | 20/20, 0.0, 0.0 |
| v4 | v3 + fail-safe gate (closed unless all four one-hots are clean) | — | **1.000** | 0.855 | 0.825 | — | 20/20, 0.0, 0.0 |
| v5 | v4's mechanism with every mean taken over 4 build-only wordings, roles located per prompt | — | **1.000** | 0.975 † | 0.825 | 0.095 | 20/20, 0.0, 0.0 |
| v6 | v5 with 8 build-only wordings | — | **1.000** | 0.995 † | 0.855 ‡ | **0.395** | 20/20, 0.0, 0.0 |

† From v4 on, "other" is not cleanly held out. I diagnosed v3's failures on it, using calibration problems in that wording, and v5's build wordings were then chosen to include a "plus" phrasing. No other-wording prompts entered any build.
‡ For v6, other2 is not cleanly held out. I diagnosed v5's failures on it, and v6's build wordings include an "add X and Y" phrasing.
other3 is clean for every version: it was written before v6 was built, and no graft was examined on it before scoring.

**Final version: v6** (`graft_v6_multiwording8.pt`). other3 is a hard wording for the host itself, which gets 6.5% at base: it usually starts explaining ("To find the total of …"), or sums the digits ("18" for 31+69). So on other3 the graft has to override the host's answer format as well as its arithmetic. v6 gets 39.5% there. Most of the remaining wrong answers are still the host's "To find the total of …" opening. My guess is that the step-0 push loses to the host's "To" on those items, but I did not examine them.

Every row's raw record, including the first 12 wrong answers per wording, is in `log.jsonl`.

## What broke along the way

- **Step detection by class means at the query position (v1).** "Two answer digits written" and "three written" look alike in the host's layer-20/22 residual: both are the state "the answer is complete, emit `<|im_end|>`". With a strong push (α=3), the graft wrote the units digit twice, giving `1117` for 117 and `10000` for 100, and calib-check accuracy fell to 0.91. Replacing it with relative-position heads anchored on the answer-opening token fixed this.
- **Role keys and the digit 0.** With the null threshold at the midpoint, units-role heads missed operands whose units digit was 0 (74+10, 10+26), because their role score is weaker. Moving the threshold to 30% of the gap fixed this.
- **Step-specific output directions were much worse** (0.88 down to 0.77 as α grew). Within one step, "next token is 5" is confounded with "the sum is in the 50s", so the direction pushes features of the operands and not the output token. Pooled directions worked, and unembedding directions worked best.
- **Push too weak against a confident host.** The pooled directions at α=2 could not overturn the host on 61+40 or 62+44. The graft had computed S=101 and S=106 correctly. Unembedding directions at α=6 did overturn it.
- **Other wordings.** A role key built from one wording learns "digit after ` +`". With "plus" or "and" as the connector, the second-operand heads returned mostly null. Usually this left the graft silent, so accuracy stayed at base level. But a *partially* open gate gave a partly built S that was pushed at partial strength, and this caused new errors (86+30 → `94`, 89+99 → `15`) and made v3 slightly worse than base on "other". The v4 fail-safe removed that harm. The v5 multi-wording build removed most of the misses: other went from 0.80 to 0.975. The remaining "other" errors are the host echoing the problem (`15 + 9`), and one 10+69 → `169`.
- Off-texts: the gate never opened on any of the 20 gate-calibration texts, the 20 damage prompts (including "What is 12 times 7?") or the 50 sentences.

## Run time

On 4 CPU cores, with reference linear-attention kernels:
- one 200-problem greedy evaluation in one wording takes about 4–5 minutes;
- a full version evaluation (3–4 wordings plus damage) takes about 21–30 minutes;
- collecting calibration activations takes about 4 minutes per 400 prompts;
- building a graft takes seconds.

The whole pilot took about 4 h 25 min of wall-clock time, from 06:50 to 11:15 UTC.

## What a registered version should test

1. **Pre-register** the format, the held-out wordings (at least 5, written before building), α and every constant, and a single final version. Report every other version as exploratory.
2. **Ablations that test the claim "it reads the host's representations".**
   - Replace each class-mean direction with a random direction of the same norm, and expect accuracy to fall to base.
   - Shuffle the digit-mean labels, and expect wrong answers that are predictable from the permutation.
   - Graft with a corrupted operand (activation patching from another problem), and expect the graft's answer to follow the patched operand.
3. **Attribution.** For each test item, record whether the gate fired and whether the graft changed the host's answer. Here the host was already at 94.5%, so most correct answers were the host's own. The interesting numbers are fixes, breaks, and items where the graft overruled a correct host (none on the main format).
4. **Harder transfer:**
   - 1-digit and 3-digit operands, which are outside the interval tables and should leave the graft silent or need new tables;
   - subtraction prompts, where the gate should stay closed;
   - multi-turn chats;
   - raw (non-chat) prompts such as `47 + 38 =`.
5. **Damage at scale:** thousands of tokens of varied text and long generations, and arithmetic-adjacent prompts where the gate might open (dates, phone numbers, "page 47 of 38"). Also confirm the exact-zero property at bf16 and with a KV-cache implementation of the block, which is not built yet.
6. **A standard-shape block**: write the attention output to the residual and cancel it in the MLP. Measure whether that costs anything.
7. **Generality:** the same construction on another host (another Qwen3.5 size, or a non-hybrid model), with only the calibration means recomputed.
8. **Compare against a fitted probe baseline**, such as least squares or logistic readout. Compare with a model that is simply prompted better, or that thinks before answering.

## Files

All files are in `cloud-results/w3-pilot/`.
- `common.py`: model, pinned revision, wordings, splits, and greedy decoding.
- `texts.py`: the damage and gate texts.
- `collect.py` and `collect_multi.py`: calibration activations.
- `analyze_layers.py`: per-layer separability.
- `graft.py`: the block, with `build` for v1–v4 and `build_multi` for v5–v6.
- `diag.py` and `diag_multi.py`: offline checks of the block's internal S and step values.
- `run_eval.py`: all measurements, which append to `log.jsonl`.
- `demo.py`: loads saved weights and answers one question.
- `graft_v3_final.pt`, `graft_v4_failsafe.pt`, `graft_v5_multiwording.pt` and `graft_v6_multiwording8.pt`: the inserted weights for each version (0.2–0.5 MB each). Every weight is stored, together with the constants.
- `base_ref.json`: the base model's reference generations and log-likelihoods for the damage measures.

```
CLOUD-RESULT: Qwen3.5-0.8B (rev 2fc06364) on 200 held-out 2-digit additions, chat template, thinking off, greedy: base 94.5% -> final (v6) 100.0%.
Damage: 20/20 unrelated prompts give identical first 40 tokens; mean per-token log-likelihood change on 50 held-out sentences = 0.0 (max |Δ| = 0.0). The block's output is exactly zero whenever its gate is closed, and it never opened off-task.
Other wordings: "Compute a plus b…" 99.5% (base 83.5%; not clean from v4 on); "Please add a and b…" 85.5% (base 82.5%; not clean for v6); fresh wording "Tell me the total of a and b, digits only." 39.5% (base 6.5%; clean).
Mechanism: an inserted block after layer 21 uses mean-difference attention keys on the host's own layer-22 residual to find the four operand digits and reads them with digit class means; RoPE-style heads anchored on the answer-opening token say which answer digit is due; a hand-written ReLU MLP turns these into one-hot digits, computes S = 10(a1+b1) + (a0+b0), and pushes the host's own unembedding direction for the right digit at each step.
```
