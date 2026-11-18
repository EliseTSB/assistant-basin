# Cold review: Part 206 design (branch `part206`, commit 31d34654, `PART206-DESIGN.md`)

VERDICT: **DO-NOT-SHIP** (as a registrable design; it is a good draft to fix, not a design with one bug).

Line numbers below are `git show part206:PART206-DESIGN.md | grep -n ""` line numbers. Every finding names what I
checked it against: the design text, the real module code in `/home/elisepc/neuron-probe/.venv/lib/python3.12/site-packages/transformers/models/{olmo2,qwen3_5}`,
the real tokenizers and corpus under `/home/elisepc/neuron-probe/corpus` and `/mnt/e/neuron-probe/hf/hub`, and the
project's results under `/home/elisepc/green/worktrees/ab-part201/results/`. Scripts and their output are in this
directory (`exp1_corpus.py` ... `exp7_arming.py`); no model was run and nothing was downloaded.

Severity order. INTRODUCED = this change adds it or makes it reachable. PRE-EXISTING = there before, not reached here.

---

## S1 (INTRODUCED) The split's constant share `C_d` is a product of two means, and its noise is the size of the token share it is used to normalise.

**Where:** design lines 140-145, predictions 206.3 (line 243) and 206.4 (line 244).

**What the design says:** "`C_d` is the product of the two halves' means, over `S_d`" (line 140-141); the constant
share is the mean of `C_d` over coordinates (line 143-144); the token share in the predictions is `T / (T + R)`
(line 144-145); 206.4 sets a bar of 0.5 on the first layer's token share and 0.25 on the middle layer's.

**What I checked it against, and what I got wrong first.** My first simulation (`exp6_split_known.py`, first
version) gave each listed type *one scalar per half*; under that model `C` reads 0.001 when the true constant share
is 0.444. That model is not the design's estimator, and I discarded it. The design's estimator is a coordinate
averaged over all listed positions, with per-type means over the positions of that type. Under that model the
estimator is **unbiased**: on data with a known decomposition `y = mu + t(type) + e`, 100 listed types, 300 passages
per half, 88 positions per passage:

```
mu=1.0 sd_t=0.5 sd_e=1.0: true C 0.444 T 0.111 token 0.200
   B design estimator, 8 occurrences/type/half: C 0.4456 T 0.1095 token T/(1-C) 0.1975
   B design estimator, 50 occurrences/type/half: C 0.4459 T 0.1091 token T/(1-C) 0.1970
mu=0.5 sd_t=0.5 sd_e=0.5: true C 0.333 T 0.333 token 0.500
   B design estimator, 8 occurrences/type/half: C 0.3317 T 0.3335 token T/(1-C) 0.4991
   B design estimator, 50 occurrences/type/half: C 0.3310 T 0.3336 token T/(1-C) 0.4986
mu=0.0 sd_t=0.5 sd_e=1.0: true C 0.000 T 0.200 token 0.200
   B design estimator, 8 occurrences/type/half: C 0.0019 T 0.1970 token T/(1-C) 0.1974
   B design estimator, 50 occurrences/type/half: C 0.0021 T 0.1954 token T/(1-C) 0.1958
```

So the estimator is sound **when each listed type has many positions**. The defect is what happens when a type has
few. The design's floor is "at least 8 read occurrences in each half" (line 137). At that floor, the per-type means
are mostly noise, and the *product* of two noisy means is a difference of squares whose sign is a coin flip
(the product of two independent zero-mean normals is symmetric about zero with a heavy tail). With
one position per type per half (the extreme of the floor), the same known-decomposition data gives:

```
   A one scalar per type per half: C 0.0011 (sd 0.0055) T 0.1106 (sd 0.0559) token 0.111 (sd 0.056, p5 0.020, p95 0.203)
   A one scalar per type per half: C 0.0035 (sd 0.0074) T 0.3319 (sd 0.0756) token 0.333 (sd 0.076, p5 0.215, p95 0.464)
```

`C` is biased to zero (0.001 against a true 0.444) and, worse, the **token share `T/(1-C)` has an sd of 0.056 to
0.076** on data whose true value is 0.200 and 0.500. That is the size of the difference 206.4 is asked to resolve
(0.5 against 0.25), and the size of the smallest effect worth knowing (0.1, line 218). The design's floor of 8
occurrences per half is what sets this noise: the variance of a type's mean is `e^2/8`, and the estimator sums 100
such products.

**Why it matters:** 206.4's two bars (0.5 and 0.25) are within about one sd of the estimator's noise at the design's
own floor, so the prediction can fail or pass for sampling reasons; and `C` itself is biased low whenever the
per-coordinate noise dominates the per-coordinate constant, which is the ordinary case for a residual-stream
coordinate. 206.3 ("MLP sublayers write at least half of the reference", line 243) is scored from the same shares.
The design has no fixture that checks the estimator against a known decomposition, so nothing in the build would
catch either.

**Smallest fix:** (a) add a fixture to the scorer: synthetic data with a known `mu`, `t`, `e`, and the estimator must
return the true shares within a stated tolerance (this is what caught the problem here). (b) Raise the listed-type
floor from 8 to a number set by the noise the prediction must resolve (e.g. at least 50 read occurrences per half),
or shrink each type's mean toward zero by its own variance (`(m0*m1 - s^2)` style), which removes the bias at the
cost of a known shrinkage. (c) Report `T/(1-C)` with a passage-resampled interval, and state in 206.4 that the bar is
tested against that interval, not a point estimate.

---

## S2 (INTRODUCED) `rho` measures position, not the frame: a constant position effect at the same positions in both
frames gives `rho ~ 0.73` with no common-mode shift at all.

**Where:** design lines 160-167 (definition and "Scored pairs"), 206.6 (line 246), 206.7 (line 247), the Headline
rule (lines 273-279).

**Measured (`exp4_rho.py`, output pasted):**

```
== rho with a real common-mode shift of size s (per coordinate sd 1 noise) plus a position effect
   shift 0.00 (+passage noise 0.3) position-effect 0.0: mean rho +0.000
   shift 0.00 (+passage noise 0.3) position-effect 0.5: mean rho +0.732
   shift 0.50 (+passage noise 0.3) position-effect 0.0: mean rho +0.031
   shift 0.50 (+passage noise 0.3) position-effect 0.5: mean rho +0.733
```

(The position effect in this simulation is a single vector added to every position in both frames. The design's
frames share the passage positions, so any component of `x` that depends on position and not on frame is present in
both and is exactly what this term stands for.)

A per-position effect that is the *same* in both frames (a position effect, not a frame effect) gives rho 0.73 with
zero common-mode shift. The design's own framing invites this: "the passage sits at the same absolute positions in
both frames" (line 120) is presented as a control, but it is the opposite. Because `D = x_X - x_Y` at the same
position, any component of `x` that depends on position and not on the frame is identical in X and Y and cancels in
`D` only if it is exactly equal; any small difference in how the two frames' prefixes write into that position
contributes a *systematic* per-position term that `Delta_fit . Delta_test` reads as common-mode. And `rho` is
normalised by `mean(|D|^2)`, so a large position effect *raises* the denominator and *lowers* rho for a real shift:
with a real 0.5 shift and a position effect of 0.5 the rho is 0.733, indistinguishable from the position effect
alone. A real common-mode shift is not detected by this statistic when the position effect is present, and a
position effect is detected as a common-mode shift.

**Also measured:** a handful of huge coordinates (Sun et al.'s massive activations, which the design cites at line 73)
give rho 0.997 to 1.000 with no common-mode shift:

```
== rho when the shift is a few huge coordinates (massive activations, Sun et al.)
   1 huge coordinates: mean rho +0.997
   3 huge coordinates: mean rho +0.999
   10 huge coordinates: mean rho +1.000
```

**Why it matters:** 206.6 (`rho_peak(A-tmpl, M) >= 0.2 and >= 2 * rho(M2, M)`) and 206.7 (largest rise between
stages) are decided by a statistic that a position effect or one or two coordinates can drive to 0.7-1.0, and the
Headline rule ("the assistant is a constant shift of the reference") is written from it.

**Smallest fix:** (a) centre `D` per position before averaging: `Delta` = mean over passages of `D` at each
position, then take the mean over positions of that; or (b) compute `rho` on `D` with the position means removed
(`D' = D - mean_over_passages D` per position), which removes any effect common to all passages at a position (this
is the position effect) and leaves the passage-varying part; (c) report the per-position mean of `D` beside the
common-mode ratio, and a coordinate-robust version (e.g. drop the top 1% of coordinates by `|Delta|`), so a single
huge coordinate cannot decide it. (d) The null (M2, M) cannot catch a position effect because both M and M2 are
plain prefixes of the same length and the position effect is in both; a better null is two *different* assistant
frames (two different assistant prefixes) or an assistant frame against itself with a different passage order.

---

## S3 (INTRODUCED) The `rho` null is exactly zero by construction, so 206.6's "at least twice rho(M2, M)" cannot fail.

**Where:** design line 164 ("the null is 'any different prefix of this length'"), 206.6 (line 246).

**Measured (`exp4_rho.py`, output pasted):**

```
== rho under the null (M2, M), pure noise
   mean +0.00000  sd 0.00036  P(rho>0.2) 0.000  min -0.0015  max +0.0011
== rho under the null with correlated positions inside a passage
   position-independence 1.0: mean +0.0000 sd 0.0004
   position-independence 0.5: mean -0.0000 sd 0.0017
   position-independence 0.2: mean +0.0000 sd 0.0030
```

(M2, M) is two arbitrary prefixes of the same length; their difference has zero mean at every position by
construction, so `Delta_fit . Delta_test` is a product of two independent zero-mean vectors and its expectation is
exactly 0, with sd ~0.0004. "At least twice 0.0004" is met by any rho above 0.001. The second clause of 206.6 is
therefore a no-op; only the 0.2 bar tests anything. The design's own reading "at least twice rho(M2, M)" (line 274)
is then a claim that cannot fail, and the "what would fail 206.6, either bound missed" (line 256) is false for the
second bound.

**Smallest fix:** replace "twice the null" with an absolute bar with a stated interval (e.g. rho >= 0.2 with a
passage-resampled interval excluding the null's upper 99th percentile), and say in the reading rule that rho(M2, M)
is a *sanity check that the statistic is near 0 for a null pair*, not a bar. If the point is that the assistant shift
is larger than a null shift, the fair comparison is a *different assistant prefix* (A-tmpl against A-tmpl with
another request, e.g. "Please write a sentence."), not two neutral prefixes.

**Note on the two nulls.** The design's claim that (M2, M) and (N2, N) are "the any-prefix nulls" is right as far
as it goes; the problem is that "any prefix" is a much weaker event than "any assistant prefix", and the statistic
is zero for it by construction. A null that is zero by construction cannot separate a frame effect from a position
effect (S2), which is the confound the null is supposed to rule out.

---

## S4 (INTRODUCED) The frames are not achievable as specified: the A-tmpl passage is followed by `<|im_end|>` and the
prefix ends in the thinking-off block; M and M2 have no template tokens; "thinking off" is not what the template does.

**Where:** design lines 109-125 (Frames), 122-125 ("for a base model the template is ordinary text"), G1 (line 208).

**Measured (`exp2_frames.py`, `exp3_tokenids.py`, output pasted):** with the real tokenizers from the snapshots:

```
qwen-chat
  A-plain: 12 tokens ['User', ':', ' Please', ' write', ' a', ' short', ' passage', '.', '\n', 'Assistant', ':', ' ']
  A-tmpl (thinking off) full render: '<|im_start|>user\nPlease write a short passage.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nPASSAGE<|im_end|>\n'
  prefix before passage: '...<|im_start|>assistant\n<think>\n\n</think>\n\n' -> 18 tokens
  suffix after passage: '<|im_end|>\n'
  whitespace probe: '<|im_start|>assistant\n<think>\n\n</think>\n\nleading and trailing<|im_end|>\n'
olmo-instruct
  A-tmpl: '<|endoftext|><|user|>\nPlease write a short passage.\n<|assistant|>\nPASSAGE<|endoftext|>'
  prefix before passage: 17 tokens ['<|endoftext|>', '<', '|', 'user', '|', '>\n', ...]
  suffix after passage: '<|endoftext|>'
```

1. **The passage is not the last thing in A-tmpl.** It is followed by `<|im_end|>` (Qwen) or `<|endoftext|>` (OLMo).
   The design says "a fixed prefix followed by the passage ids" (line 107) and reads "passage positions 8 to 95"
   (line 127), so the read positions are passage tokens, but the frame's *content* after the passage is a special
   token that M and M2 do not have. Any gate shift found at the last passage positions is then not a frame effect
   at all; it is the closing token, which is absent from M/M2. The design does not say whether the suffix is included
   in the frame (it must be, or the assistant turn is never closed) or how that asymmetry is controlled.
2. **"Qwen thinking off" is not what the template does.** `enable_thinking=False` renders `<think>\n\n</think>\n\n`
   *inside* the assistant turn (Qwen's `chat_template.jinja`, lines with `enable_thinking is defined and
   enable_thinking is false`). That block is part of the A-tmpl prefix; M and M2 (plain neutral sentences, line 116)
   have no `<think>` tokens and no `<|im_start|>`. The prefix *lengths* can be made equal (G1 is satisfiable with a
   neutral sentence of the right token length), but the *content* cannot: M and M2 cannot contain the template's
   special tokens without being a template. So (A-tmpl, M) compares an assistant turn against a plain sentence of
   the same token count, which is a length-matched control for length and nothing else. This is the same confound
   Part 80 found (README Part 80 section: "in both templates a past Assistant message differs from a User or System
   message only by the role word"), and the design's M/M2 do not fix it.
3. **"for a base model the template is ordinary text" (line 123) is false for Qwen3.5-4B-Base.** The Base tokenizer
   has the same special tokens (`<|im_start|>` = 248045, `<|im_end|>` = 248046, `<think>` = 248068, `</think>` =
   248069; `exp3_tokenids.py`), so the template is rendered as *special tokens* for Base too, not as ordinary text.
   The design's own sentence at line 124 ("Qwen3.5-4B-Base may never have trained the template's special tokens") is
   the correct one; line 123 contradicts it.
4. **OLMo has no `<|im_start|>`; its template uses `<|endoftext|><|user|>\n...` and the passage is followed by
   `<|endoftext|>`.** The design's M/M2 are "neutral plain sentences" with no `<|endoftext|>` and no `<|assistant|>`;
   for OLMo the assistant turn is marked by `<|assistant|>` which tokenizes as 5 ordinary tokens (`<`,`|`,`assistant`,
   `|`,`>`), so a neutral sentence *could* contain the same five tokens by accident, but the design does not say it
   does. The design's "the token ids are the same at every OLMo stage" (line 122) is true; "for a base model the
   template is ordinary text" is not.
5. **The suffix is not in the design's frame table at all.** The table (lines 109-117) gives only the prefix. The
   suffix is what makes the passage an assistant turn (the closing token); without it the passage is an unterminated
   assistant turn, and with it M/M2 are not the same frame shape.

**Smallest fix:** (a) state the full frame string including the suffix, and make M/M2 include the same closing
token (e.g. M = a neutral sentence followed by `<|im_end|>` for Qwen, `<|endoftext|>` for OLMo); (b) add a
T-tmpl control: the same template with the passage as the *user*'s content, so the role word is the only difference,
as Part 80 did; (c) delete line 123 or replace it with the Base tokenizer's actual rendering; (d) read only
positions 8 to 87 of the passage (leave the last 8 before the closing token unread) or report the last positions
separately.

---

## S5 (INTRODUCED) Sites are not well defined in both architectures; in OLMo 2 the design's `x_L` is `h_L`, and
`h_L`, `m_L`, `c_L` are not what the words say.

**Where:** design lines 130-134 (Sites), 150-152 (b_L), 160 (`x_L`), 169 (`g_n`), 175-177 (`Delta^h`).

**What the code does** (`transformers/models/olmo2/modeling_olmo2.py:296-315`, read):

```
residual = hidden_states
hidden_states, _ = self.self_attn(hidden_states)
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.mlp(hidden_states)
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

So the MLP's input is the *unnormalized* stream; the MLP's output is normalized *after* it is computed
(`post_feedforward_layernorm`) and then added. In `transformers/models/qwen3_5/modeling_qwen3_5.py:860-900` the
structure is pre-norm (Qwen3_5RMSNorm with `(1 + weight)`, line 826-841), and the MLP input is
`post_attention_layernorm(h)`.

- `h_L` "the residual stream where layer L's MLP block reads it": in Qwen this is the *unnormalized* stream (the
  input to `post_attention_layernorm`); in OLMo it is the same object as `x_L` (the MLP reads the raw stream). The
  design says `h_L` and `x_L` are different sites "per layer" (line 130) but in OLMo they are identical; the design
  should say so, and the split of `m_L` from `h_L` in OLMo is then a split of the *normalized* output, not of the
  write.
- `m_L` "what the MLP block adds to the stream": in OLMo this is `post_feedforward_layernorm(down_proj(...))`, not
  `down_proj(...)`; in Qwen it is `down_proj(...)`. The design's line 300 ("OLMo 2 normalizes sublayer outputs, not
  inputs, so its reference lives in the raw stream") is correct about the architecture but the site list does not
  say which of the two `m_L` is. The `c_L` site has the same problem: in OLMo it is
  `post_attention_layernorm(attn_out)`, not `attn_out`.
- `g_L` "the input to the activation function; `gate_proj`'s output in a gated MLP" is right for both (SwiGLU,
  `silu(gate_proj(x)) * up_proj(x)`, OLMo line 273, Qwen line 822). But `a_L` "the neuron activation (`down_proj`'s
  input)" is `silu(g)*up`, not `silu(g)`; the design's neuron token share (line 146) is on `a`, which is the
  product, not the gate. That is a defensible choice but it is not the "gate" the arming measure (line 169) uses;
  the two neuron statistics are on different quantities and the design does not say which one a "catalogued neuron"
  is.
- **Qwen's mixer is not one thing.** 24 of 32 layers are Gated DeltaNet (`config.json`, `layer_types`), whose
  output is `out_proj(norm(core_attn_out, z))` (Qwen3_5GatedDeltaNet, lines 490-560) and whose `conv1d` has a
  kernel of 4 with `padding = kernel_size - 1` (line 512-518); the 8 full-attention layers have an output gate
  (`attn_output * torch.sigmoid(gate)`, lines 762-808). `c_L` is therefore two different objects by layer type, and
  a "mixer token share" (206.5) mixes them. The design says "mixer below means whichever a layer has" (line 99),
  which is honest, but 206.5's "the MLP's token share exceeds the same layer's mixer token share at three quarters or
  more of the layers in the first third" compares a SwiGLU MLP against a DeltaNet recurrence in 8 of the first 11
  layers; a token share for a recurrence is not the same kind of quantity as for a per-position MLP. The design
  should say which layer types are in the first third and whether the comparison is per layer type.
- **Qwen3.5-4B is a vision-language model** (`Qwen3_5ForConditionalGeneration`, `config.json` architecture), whose
  text model lives under `model.language_model`. The design says "Qwen3.5-4B has 32 layers" (line 98) but does not
  say where the hooks attach (`model.language_model.layers[i]` vs `model.model.layers[i]`); a hook on the wrong
  attribute silently does nothing, and the design has no G0-equivalent for "the hook fired at all".

**Smallest fix:** a site table with, per architecture, the exact module path and the exact expression for each of
`h, x, g, a, m, c` (e.g. OLMo: `h` = input to `layers[i].mlp`, `x` = same, `m` = output of
`layers[i].post_feedforward_layernorm`, `c` = output of `layers[i].post_attention_layernorm`; Qwen: `h` = input to
`layers[i].post_attention_layernorm`, `x` = its output, `m` = output of `layers[i].mlp`, `c` = output of
`layers[i].self_attn` or `layers[i].linear_attn`). Add a G0.5 that asserts each hook fires once per layer per
forward pass and that `m + c + h` reconstructs the next layer's `h` to within bf16 rounding.

---

## S6 (INTRODUCED) GT cannot fail for a reason that bears on the reading, and cannot pass for a reason that bears on it
either; `g > 0` is not "firing" for a SwiGLU gate.

**Where:** design lines 153-158 (Tuomi check), GT (lines 210-212), 206.1/206.2.

**Measured (`exp5_split_gt.py`, output pasted):**

```
   g = -5.000  silu(g) = -0.0335  counted as firing by g>0: False
   g = -1.278  silu(g) = -0.2785  counted as firing by g>0: False
   g = -0.100  silu(g) = -0.0475  counted as firing by g>0: False
   g = +0.100  silu(g) = +0.0525  counted as firing by g>0: True
   g ~ N(0, 0.5), delta 0.1: firing 0.500 -> 0.579, F_ref = 1.16
   g ~ N(0, 5.0), delta 1.0: firing 0.500 -> 0.579, F_ref = 1.16
```

`silu` is negative on all of `(-inf, 0)` and has its minimum at g = -1.278; a gate at g = -5 is *near-silent*
(|silu| = 0.03), not inhibited, and a gate at g = -0.1 is *small*, not inhibited. `g > 0` counts the sign of the
gated contribution, not whether the neuron fires. Tuomi's own definition is flagged as unread (line 156-158), which
is honest, but the design then builds GT (the gate that gates *every* downstream prediction, lines 210-212 and 266)
on `g > 0` anyway.

More important: `F_ref = firing after / firing before` is a function of the *sharpness* of the gate distribution
around 0 and of the size of the shift `delta = W_gate . r`, not of whether a reference exists. With `g ~ N(0, 5)`
and `delta = 1`, `F_ref = 1.16`; with `g ~ N(0, 0.5)` and `delta = 0.1`, `F_ref = 1.16`. The same F_ref comes from
a large reference and a sharp gate as from a tiny reference and a flat gate. GT's bar (F_ref >= 2 and 5x the random
control) is therefore a bar on the ratio of the shift to the gate's spread, and the design's reading ("where GT
fails, the reading 'a shifted reference re-thresholds gates' has no footing", line 212) is *false*: a model can have
a real reference and fail GT because its gates are far from 0, and a model can pass GT with a reference that is
small but sits exactly where the gates are dense.

The random control (line 155-156, "a vector of the same norm along each of 20 seeded random unit directions") is
fair on norm but not on the *gate projection*: the reference direction `b_L` is, by construction, the direction the
MLP's weights were trained against (Tuomi's claim), so `|W_gate . b_hat|` is large for a reason the random
directions do not share. A norm-matched random direction has a *smaller* gate projection on average
(`W_gate . r` for a random unit r has sd `|W_gate|/sqrt(d)`), so the control is *easier* than the reference and
F_ref > 5 x F_rand is easier to pass than the design thinks. A fairer control is a random direction *projected into
the same gate subspace* (e.g. a random vector with the same `W_gate` norm), or the same norm along the second
principal direction of `b`'s covariance.

**Smallest fix:** define firing as `silu(g) * up > 0` (the sign of the neuron's contribution) or as
`|silu(g)*up|` above a threshold, and say why; measure GT on the *gate projection* of the reference
(`|W_gate . b_hat|` against the distribution of `W_gate . r` over random unit r) rather than on `F_ref`; and add a
positive control where a known reference is planted (a synthetic `b` with a known `W_gate . b`) to show GT passes
when it should and fails when it should.

---

## S7 (INTRODUCED) The causal check picks `L*` on the fit half but tests the *same layers'* neurons; the maker
readout is Google, not the model's own maker; and the reverse arm is not the same test.

**Where:** design lines 175-184, 206.10 (line 250), the Controls "Halves" (line 193).

- `L*` is "the middle-third layer with the highest `rho_L(A-tmpl, M)` on the fit half" (line 176). The (b) readout
  is "the arming percentiles of the catalogued neurons above `L*`" (line 182). For OLMo the catalogued neurons are
  Part 112's, which are all at layers 10 to 26 (measured from `results/part112/score.json`: 175 concept and 154 slot
  in the Q-name group, layers 10, 14-26; 444 of 444 classified neurons are above layer 10). The middle third of 16
  layers is layers 5 to 10, so `L*` is at most 10 and every catalogued neuron is above it; the "above `L*`"
  filter does nothing, and the causal check's (b) is the same measurement as 206.8 with a hook added. That is not
  fatal but it means (b) is not independent of 206.8, and the design says "independent" nowhere.
- **The maker readout.** The design says "the change in log probability of the first token of the maker name"
  (line 179-180). For Qwen3.5-4B the plain model's own answer is "Tongyi Lab" / "Alibaba" (Part 153's README:
  "Plain model: I am Qwen3.5, a large language model developed by **Tongyi Lab**"), while the maker neurons were
  selected to make it say *Google* (Parts 25, 66: "zeroing the four named Google as the maker"). "The maker name"
  is therefore ambiguous between Tongyi and Google, and the two give opposite signs of the predicted effect. The
  design must name the token (and its id), and it should say whether it is the *plain* model's own token (Tongyi)
  or the *zeroed* model's token (Google). As written, 206.10 can pass or fail on which token the builder picks.
- **The reverse arm** "adds `Delta^h_L*` to frame M and reads (b)" (line 184). Reading (b) is "the arming
  percentiles ... recomputed under the subtraction on the test half" (line 182); the reverse arm reads a *different*
  quantity (arming under an addition to M) and there is no prediction attached to it. The design lists it under
  "Not used: both poles ... with one reverse arm" (line 204) as if it were a control, but a control with no
  registered comparison is not a control.
- **The (M2, M) direction at equal norm** (line 178): equal to what? `Delta^h_L*` has norm `|Delta^h_L*|`; the
  (M2, M) direction is a *direction*, and the design does not say it is scaled to the same norm as the arm. If it
  is not, the comparison is not norm-matched. (The 20 random directions *are* at equal norm, line 197.)
- **The readout position** "at the position where the plain model's own greedy answer names it" (line 180) is a
  position chosen by generation, and the design says "no generation" (line 200, 232); the design's own "What comes
  after" (line 305) is the generation part. The position must be found by a greedy decode, which is generation, and
  the design's Size section does not count those passes (line 221-224 counts "24 x (1 + 2 x 22) = 1,080 short
  forward passes", which assumes the position is known).

**Smallest fix:** name the maker token and its id; state whether the (M2, M) control is scaled to `|Delta^h_L*|`;
drop the reverse arm or register a prediction for it; add the greedy-decode passes to the Size section; and state
that (b) is the same quantity as 206.8 under a hook (or choose a different readout for (b), e.g. the change in the
catalogued neurons' *activations* `a`, not their arming percentiles).

---

## S8 (INTRODUCED) The Size section's token total is low (the Tuomi recomputations and the loss passes are not
counted), its memory is not stated, and its corpus rule does not give 300 items.

**Where:** design lines 101-105 (passages), 128 (positions), 216-226 (Size), 230-235 (Optimization).

**Measured (`exp1_corpus.py`, output pasted; the hash-order counts are over the first 5,000 items of each file, not
20,000 as the script's print says):**

```
  first-5000 wikitext: >=96 tokens: 2281 median len 81 ...
  hash-ordered first 300 wikitext (of 20000) with >=96 tokens: 130
  hash-ordered first 300 ultrachat replies (of 20000) with >=96 tokens: 262
  wikitext: read positions [13200, 13200], distinct types 3979, listed types 102, coverage 0.4999 / 0.5020
  ultrachat: read positions [13200, 13200], distinct types 3808, listed types 126, coverage 0.4998 / 0.4941
```

1. **"10 frames x 600 passages x about 115 tokens = about 690,000 tokens per model per pass"** (line 221). The
   design's own numbers give 10 x 600 x (prefix + 96) = 6000 x (prefix + 96). With the measured prefixes (A-plain 12,
   T-plain 12, A-tmpl 18 for Qwen, 17 for OLMo, P 0, U-plain 3, U-tmpl 4-5, N/N2/M/M2 at the same length as their
   pair), the average frame is about 12 + 96 = 108 tokens, not 115, so the total is about 648,000, not 690,000; and
   "115" is not the sum of a 96-token passage and any prefix in the table. More importantly, **the design does not
   count the two passes the same way**: pass 1 is "moments" and pass 2 is "the local Tuomi recomputation and type
   tables" (line 221-222), and the Tuomi recomputation needs a *forward pass through the MLP* for each of 20 random
   directions plus the reference, which is 21 recomputations per layer per passage, not one forward pass. The
   design's "two passes" is therefore at least 22 MLP evaluations per layer, not 2; the "about 11 million tokens"
   (line 222) is a lower bound by roughly an order of magnitude for the Tuomi part.
2. **"24 x (1 + 2 x 22) = 1,080 short forward passes"** (line 223). 24 x (1 + 44) = 1,080 is arithmetically right
   (24 x 45 = 1,080) but the 22 is 20 random + (M2, M) + the arm = 22, and the arm itself is *one* of the 22, so the
   count should be 24 x (1 + 2 x 21) = 1,032, or the design should say the 22 includes the arm. The "1" is the
   plain pass; the "24" is Part 66's phrasings; but the design also needs a pass per phrasing for the *position*
   (the greedy decode, see S7), which is another 24. The loss passes are not counted in the "11 million" either:
   the design says "one loss pass per arm on each final checkpoint" (line 224) without a count, and 22 arms x 300
   passages x 108 tokens x 2 checkpoints is about 1.4 million tokens, so the total is nearer 14 million than 11.
3. **The corpus claim "300 plain passages ... and 300 assistant replies"** (line 101-102) is achievable but not the
   way the design says. "A passage is the first 96 tokens of an item with at least 96" and "items taken in order of
   the SHA-256 of their text" (line 102-103): in the first 300 items by hash, only **130 wikitext items** and **262
   ultrachat items** have 96 tokens (measured on the first 5000 items; the full file has 582,510 nonempty wikitext
   rows and 23,110 ultrachat rows, so 300 is reachable by taking the first 300 *that qualify*, but the design's
   wording "items taken in order of the SHA-256 of their text" reads as "the first 300 by hash", which gives 130 and
   262, not 300 and 300). The design must say "the first 300 by hash *among items with at least 96 tokens*".
4. **The listed-type coverage is about 50%** (measured: 0.50 wikitext, 0.49 ultrachat, 102 to 128 listed types). The
   design says "the listed types' coverage ... is reported" (line 138) and "Rare tokens ... are not measured"
   (line 292-293), which is honest, but the split then describes **half** the read positions, and the design's
   "52,800 per frame" (line 128) is the *read* positions, not the listed ones (about 26,400). The Size section
   should say the split uses about 26,000 positions per frame, not 52,800, and the smallest effect worth knowing
   (0.1 in rho, line 218) should be recomputed on the listed subset.
5. **Memory is not sized.** The design's Optimization says "type tables on 256 coordinates and a 512-dimension
   projection" (line 233) but the Size section has no memory number. A type table of `types x 256 coordinates x
   layers x 2 sublayers x 2 halves` with about 100 listed types is 100 x 256 x 32 x 2 x 2 x 4 bytes = 26 MB for
   Qwen (fine), and the neuron tables are small; the real cost is the **hooks at six sites per layer on a 4B model
   in 16 GB of VRAM**: Qwen3.5-4B in bf16 is about 8 GB of weights, and a forward pass with `use_cache=False` at
   batch 1 and 108 tokens keeps about 32 x 108 x 2560 x 2 bytes = 17 MB of activations per site, but the *stored*
   `g` and `a` are 32 x 108 x 9216 x 2 bytes = 64 MB per site per pass, and the Tuomi recomputation needs `g` for
   every neuron at every position (32 x 108 x 9216), which is 64 MB in bf16, 128 MB in fp32; the design does not
   say whether `g` and `a` are kept (they must be for the neuron token share and the arming percentiles) or
   recomputed. The design's "all statistics as running sums on the device, moved to the CPU once per batch"
   (line 232) is right for the moments but not for the per-neuron tables, which are `32 layers x 9216 neurons` per
   layer, i.e. 9216 x 32 = 294,912 neurons, and the arming percentile needs each neuron's `g` mean and sd over
   52,800 positions, which is 294,912 x 2 floats = 2.4 MB (fine), but the *percentile* needs the whole layer's
   distribution, which is 9216 floats per layer (fine). The memory is probably fine; the design should still show
   the number, because the Optimization section's job is to show the bound resource is not the memory.

**Smallest fix:** recompute the token total from the measured prefixes and the listed coverage, count the Tuomi
recomputations as 21 MLP evaluations per layer per passage, count the greedy-decode passes and the loss passes, and
add a memory line (weights + largest stored site + type table).

---

## S9 (INTRODUCED) 206.7's "largest rise" rule is a maximum of five noisy numbers with no interval requirement on the
*other* rises, and the design's own reading rule makes it fail-able only in one direction.

**Where:** 206.7 (line 247), reading rule (line 268-269).

**Measured (`exp4_rho.py`):**

```
== 206.7: five adjacent rises, no real stage effect
   P(argmax = the 3rd rise, mid-training end -> SFT) = 0.20125
```

With no stage effect, the "largest rise" is the 3rd of 5 about 20% of the time, which is exactly chance. The
simulation uses an sd of 0.05 for each rise, which is a guess; the point is the argmax rule, not the sd: the
registered pair wins a 1-in-5 draw unless the effect is large. The
reading rule adds "its interval from resampling passages ... excludes 0" (line 268-269), which makes 206.7 fail-able:
if the mid-training-end to SFT rise is the largest but its interval includes 0, the prediction fails. So 206.7 is
not a prediction that cannot fail; it is a prediction that fails about 80% of the time by chance and, when it
passes the argmax, fails again unless the rise is significant. The design's "what would fail 206.7" (line 257)
says "any other adjacent pair with the largest rise, or no rise whose passage-resampled interval excludes 0", which
is right. The finding is that **206.7's confidence "medium" is not supported by a 20% chance of the argmax landing
on the registered pair with no effect at all**, and the design should either raise the bar (the registered rise must
be the largest *and* at least 0.1 above the next, line 218's smallest effect) or lower the confidence. The finding
is weaker than S1-S7.

**Smallest fix:** require the registered rise to exceed the second-largest by the smallest effect worth knowing
(0.1) and its interval to exclude 0; otherwise the prediction is a coin flip with a significance filter.

---

## S10 (INTRODUCED) 206.9's "nine times in ten" is measured at 0.80, and the 16 neurons are not exchangeable.

**Where:** 206.9 (line 249), "what would fail" (line 258-260).

**Measured (`exp7_arming.py`):**

```
   P(median in [35,65]) = 0.802
```

The design says "with 16 neurons a typical-neuron median falls inside that range about nine times in ten" (line
258-259); the measured value for 16 uniforms is 0.80, not 0.90. More importantly, the 16 are not 16 typical neurons:
they are the four maker neurons (L6, L7, L11, L29), L27 N3325, L24 N3949, and Part 153's ten company neurons
(L11, L21, L22, L23, L23, L23, L24, L25, L25, L25; measured from `results/part153/score.json`), i.e. 9 different
layers, with 4 of the 16 at layers 6, 7, 11, 29 (the extremes). The median of 16 is a median of 9 layer-level
values, and the design's "typical-neuron" claim assumes exchangeability across layers that nothing establishes.
The finding is minor but the number is wrong and the assumption is unstated.

**Smallest fix:** say the measured 0.80, and say the 16 sit in 9 layers with 4 at the two ends; or replace the
"typical-neuron" claim with a per-layer bootstrap.

---

## S11 (PRE-EXISTING, follow-up) 206.8's "slot neurons' median at least 10 points above concept" is a bar close to
the noise for the design's own smallest effect, and the slot/concept labels are reader-dependent.

**Where:** 206.8 (line 248), reading rule (line 270-272).

**Measured (`exp7_arming.py`):** a bootstrap sd of the median of 154 or 175 percentiles is about 5 points, so a
10-point gap is about 1.5 sd of the difference under the null (the script's `100*0.5/sqrt(n)*1.25` line is a
rule-of-thumb, not a bootstrap; the figure is approximate). Part 112's own README says the slot/concept labels
are a Haiku reader's and Part 92b's DeepSeek reader kept all 21 slot labels but only 8 of 18 concept labels
(README line 210), so the two sets are not equally reliable. The design uses the Q-name group's 175/154 (line 84-85,
line 171-172), which is the right file, but it does not say that the labels are the first reader's and that the
second reader disagrees on concept labels.

This is PRE-EXISTING in the sense that the labels and their reader dependence were there before this design; it is
not a reason to hold the design, but the design should cite the reader dependence and add the DeepSeek-confirmed
subset (245 units, Part 112's reader-correction) as a second reading.

---

## S12 (INTRODUCED, wording) "Part 92: slot units start working at SFT" is not what Part 92 found.

**Where:** line 85.

Part 92's own README (`results/part92/README-steps2-3.md`) says "Slot fillers are mostly made by DPO, not SFT",
and prediction 92.2 **failed**: at SFT, chat frame, 8 of 21 slot neurons show, against a bar of at least half
(10.5). The plain-frame counts are 0 of 21 at 1T pretraining, 3 of 20 at mid-training end, 5 of 21 at SFT, 14 of 20
at DPO. "Start working at SFT" is at best a partial reading of the SFT chat-frame row (8 of 21); the design's
"Part 92: slot units start working at SFT" should be "Part 92: slot units mostly appear at DPO (8 of 21 at SFT in
the chat frame; 92.2 failed)". This matters because 206.8 and 206.7's stage story rest on it.

**Smallest fix:** correct the sentence and cite the counts.

---

## S13 (INTRODUCED, wording) The Headline rule claims more than the measurements can show; Limits is missing the
position-effect confound and the template confound.

**Where:** lines 273-281 (Headline rule), 283-301 (Limits).

The Headline rule's first sentence, "The assistant is a constant shift of the reference that reaches the MLP's
gates, first appearing at stage X", is written if GT passes, `rho_peak(A-tmpl, M) >= 2 rho(M2, M)`, 206.8 holds,
and the rise's interval excludes 0. Given S2 (rho is a position effect), S3 (the null is 0), and S4 (the A-tmpl
frame differs from M by template tokens and a closing token, not only by role), none of these four conditions
establishes "the assistant is a constant shift of the reference". The "reaches the MLP's gates" part is 206.8, which
is a *percentile* of a mean gate shift, not a causal reach (the design says so at line 294-295, but the Headline
rule does not repeat it). The Limits section (283-301) does not list: (a) the position effect (S2); (b) the
template-token confound in A-tmpl (S4); (c) that the constant share's noise at the design's own floor is the size of
the difference 206.4 must resolve (S1); (d) that the listed types cover about half the positions (S8); (e) that the
maker readout is Google vs Tongyi (S7).

**Smallest fix:** add those five to Limits, and rewrite the Headline rule's first sentence to claim only what the
measurements show: "the assistant frame shifts the MLP input by a vector that is more common-mode than a
length-matched plain prefix, at positions where the template's closing token is not read".

---

## What I checked and found sound

- **Model revisions.** `22738a97`, `1c3599f7`, `a1847dff`, `0d85a3d0`, `c4b04859`, `48d788ec`, `1001bb4d`, `851bf6e8`
  all exist under `/mnt/e/neuron-probe/hf/hub` (snapshot directories listed), and Part 205's design names the same
  six OLMo stages and the two Qwen models (PART205-DESIGN.md lines 50-58). The Qwen3.5-4B config confirms 32 layers,
  8 full-attention and 24 Gated DeltaNet (`layer_types`), matching line 98.
- **The four maker neurons' ids** (L6 N753, L7 N1169, L11 N8868, L29 N776) and L27 N3325: README line 86 and line 274
  agree; L24 N3949 is in README line 90. Part 153's ten company neurons are in `results/part153/score.json`
  (`per_unit`: L11 N5647, L21 N7134, L22 N6997, L23 N414, L23 N1743, L23 N2993, L24 N5017, L25 N140, L25 N4457,
  L25 N4458). 4 + 2 + 10 = 16, matching line 173.
- **Part 112's 175 concept and 154 slot in the Q-name group** is exactly what `results/part112/score.json` gives
  (`groups.q_name.concept` 175, `slot` 154).
- **Part 66's 24 held-out maker phrasings** exist on disk: `results/part66/full/answers.jsonl` has question_index
  0-23, 16 direct and 8 indirect, plus a positive control (24) and six guards (25-30). The design's "24 held-out
  maker phrasings" (line 179) is right.
- **The token ids are the same at every OLMo stage** (line 122): I did not compare the tokenizer blobs of the six
  OLMo snapshots, so this is the one claim I checked only partially. The design should record the tokenizer hash per
  stage in `part206/models.json` (as Part 205 records revisions) and have G1 check it.
- **The corpus files exist** (`neuron-probe/corpus/wikitext103-train0.parquet`, 582,510 nonempty rows;
  `ultrachat-test_sft.parquet`, 23,110 rows), and 300 items of at least 96 tokens are reachable from each.
- **The `rho` formula's normalisation** `Delta_fit . Delta_test / mean(|D|^2 on the test half)` is a sensible
  cross-fitted estimate of `|mean D|^2 / mean |D|^2` *when D has no position effect*; the defect is the position
  effect (S2), not the algebra.
- **206.3's share is a projection and can be negative or above 1** (the design does not say so, but the quantity is
  well defined); the defect is the estimator's noise at the design's floor (S1), not the definition.
- **The design's honesty about what it has not read** (Tuomi's full text, line 156-158; Muse not answered, line 44;
  the two unverified items, line 71) is good practice and I did not count it against it.
- **The design is a design, not code**, so the code-shaped checks (failure paths, tests that pass for the wrong
  reason, feature flags) do not apply. I checked the design's *build-time* claims (paths, counts, revisions) and
  its *measurement* claims (formulas, sites, frames) instead.
