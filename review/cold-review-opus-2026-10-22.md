# Cold review of Parts 54 and 59 (2026-10-22)

Reviewer: a fresh Claude session at commit 7d1d2d9. I read the Part 54 and 59 sections of README.md, the matching PREDICTIONS.md rows, the three design files, `part26.py`, `neuron_report.py`, `part59/`, and the result files in `results/part54a`, `part54b`, `part59a`, `part59b` and `part26-8`. The blind labels are not in the repo, so I checked label counts against a keyword proxy built from the answer texts (regexes at the end).

## (a) Number check

I checked 47 numbers in the two README sections. 42 match or are consistent with the files. 5 do not match.

**Mismatches**

1. **The Qwen3-4B "noisy Earth base rate" is not in the answers.** The README says Qwen3-4B's "own unclamped answers and random controls already say 'Earth' 0 to 6 of 24". The word "Earth" appears in **0 of 168** unclamped Qwen3-4B answers and in **0 of 24** answers for each control at each dose (`part54b/clamp-qwen3-4b-L28u4248*.json`). [VERIFIED IN FILES] So the base rate comes from the reader: it gives EARTH to ordinary "created by Alibaba Cloud" answers (a human-maker reading). It is not something the model says. [INFERRED]
2. **"control 397: 6 of 24 at -2".** None of control 397's 24 answers at -2 contains "earth". [VERIFIED IN FILES] Same cause as item 1.
3. **"Random same-layer controls at -2 and +2: at most 1 of 96 on any label"** (59A). This can't hold literally, because the README also says the unclamped stories are EARTH-REAL 96 of 96, and the control stories are ordinary too. PREDICTIONS 59d says the "at most 1" applies to SCIFI. The README should say SCIFI. [INFERRED]
4. **"about 4.1 symmetric neurons per control expected by chance".** This is right for Qwen (82,944 × 0.00005 = 4.15). For OLMo it is 40,960 × 0.00005 = **2.0**. The README states 4.1 as if it applied to both models. [VERIFIED IN FILES]
5. **"Gemma 4 12B did not run".** It was started and failed to load: `OSError: google/gemma-4-12B-it does not appear to have a file named ... model.safetensors` (`part54b/gemma4-12b-it.log`). [VERIFIED IN FILES] This is a wording fix only.

**Matches** [VERIFIED IN FILES]

24 answers per dose; scales 19.125 (Qwen) and 6.5 (OLMo); OLMo-7B 48 per dose with 1 alien at +2; Gemma E2B no alien or Earth words; 59B: 1,000 documents, 22 of 82,944, controls 80, 4, 0, 639 of 40,960, controls 151, 186, 187, ratio 3.42, control span 9.06×; OLMo 10, 13, 11 matches PART26-LADDER.

**Blind-label counts checked only against the keyword proxy.** All 22 point the same way as the reader.

| Count | README (reader) | Keyword proxy |
|---|---|---|
| Qwen3.5 OFFWORLD at -3, -2, -1, +0.5, +1, +1.5, +2 | 0, 0, 0, 0, 1, 19, 24 | 0, 0, 0, 0, 1, 23, 24 |
| Qwen3.5 SCIFI at -2, -1, 0, +1, +2 (59A) | 0, 0, 0, 5, 69 | 0, 0, 2, 8, 92 |
| OLMo SCIFI at -2, -1, 0, +1, +2 (59A) | 1, 0, 0, 58, 79 | 1, 2, 1, 47, 72 |

## (b) Ways the conclusions could be artefacts

**B1. In 59B the symmetric count tracks push size, not unit identity.** Likelihood: high.
- Each unit is clamped to ±1× its own Pile maximum, so controls are not size-matched. OLMo's control maxima are 0.74, 0.50 and 1.30 against the target's **6.5**, pushes 5 to 13 times weaker. [VERIFIED IN FILES: `59b-olmo2-1b-L10u6978.json`]
- In Qwen, control 152 (19.77, about the target's 19.125) gives **80** symmetric responders against the target's 22; 376 (13.5) gives 4 and 692 (2.18) gives 0. [VERIFIED IN FILES] So the count rises with push size, and "OLMo stands out, Qwen is null" is what size alone predicts. The leave-one-out pooling adds to this: OLMo's target is judged against weak pushes, Qwen's against a pool holding a matched one.
- Plausible even-order mechanism: a large write of either sign raises the residual norm, and RMSNorm then shrinks other features the same way for −M and +M. OLMo-2's post-feedforward norm would also suppress the rest of that MLP's output for both signs. [INFERRED; architecture not checked in code here]
- What would show it: column-norm-scaled controls (the 18b.7 method) and a count-versus-size curve (experiment 1).

**B2. "OLMo is gradual" has no data where a threshold would show.** Likelihood: high.
- Part 26.8 has no OLMo dose between 0 and +1, and above +1 it is flat: 10, 13, 11 of 24 (proxy 15, 19, 19). Part 26.7's +0.5 (one question) gave 1 of 8 by proxy, read by hand as "ordinary", against 3 of 8 at +1. [VERIFIED IN FILES: `part26-8/`, `part26/clamp-olmo2-1b-L10u6978-x0.5.json`]
- So OLMo may also jump, between +0.5 and +1, then saturate near 50% with the rest going to EARTH or BROKEN. The difference may be where the step sits and how high it tops out, not its shape. 59A fits: at +1, OLMo 58 of 96 against Qwen 5 of 96. [INFERRED]

**B3. Qwen's step sits at the edge of the unit's natural range.** Likelihood: medium-high.
- Qwen's alien-frame maximum is 24.69, **1.29×** its Pile maximum. [VERIFIED IN FILES: `scale_frames`] The step (+1.0 = 19.1 to +1.5 = 28.7) brackets it.
- So "threshold" may just mean "nothing until the clamp passes natural alien-level firing", which is a fact about the scale. OLMo's alien-frame ratio is not recorded, so the two scales cannot be aligned. [INFERRED]

**B4. The clamp is applied at every position, including the prompt and chat template.** Likelihood: medium. `NeuronHook` writes the value at every token (`neuron_report.py:203`). The step could come from prompt tokens such as `<think>\n\n</think>` crossing a routing threshold rather than from generation. The hook already supports `phase='gen'` and `phase='prompt'`. [VERIFIED IN FILES; mechanism INFERRED]

**B5. The EARTH/OFFWORLD axis may split one effect into two labels.** Likelihood: medium-high.
- The Qwen3-4B twin at +1 writes "developed by Alibaba ... on Earth ... I do not come from other planets" and "created on Earth, on the planet Mars". By keyword it is a clean, graded planet-framing dial: 0 of 168 unclamped, 8 of 24 at +0.5, 24 of 24 at +1, 0 in every control. [VERIFIED IN FILES]
- The reader calls this EARTH and Qwen3.5's up side "extraterrestrial". Both may be one latent, "frame origin as a planet". If so, the Qwen3-4B twin is real, not noisy, and graded, not a threshold. [INFERRED]

**B6. Pushing down erases the maker's identity; it is not "nothing".** Likelihood: high as a fact, medium as a confound.
- At −2 and −3, Qwen3.5 names Qwen in 2 and 0 of 24 answers (24 of 24 at +1), saying "made by **Google**", "**Tencent** AI Lab", "**Ding Yuan**". The reader files these as NEITHER. Size-matched control 152 also drops Qwen naming to 13 of 24 at −2, so part of this is generic. [VERIFIED IN FILES: `summary.names_qwen`]
- In 59, OLMo at −2 moves 14 of 96 stories off EARTH-REAL (59c), and they read as degraded and repetitive. "Down changes nothing" holds only for the setting label. [VERIFIED IN FILES]

**B7. Mixed dtype.** Likelihood: low-medium. OLMo ran in bf16 and Qwen in fp32 in 59A and 59B. [VERIFIED IN FILES] Both 59B doses subtract one shared baseline, so its rounding error enters both changes with the same sign and nudges small changes toward "symmetric", most for the weak OLMo controls. [INFERRED]

**B8. Seeds, control picks, few prompts.** Likelihood: low. One seed per condition is reused at every dose and prompt (twin 1919609538), and the "unclamped" 24 per question are the same answers in all seven dose files (3 distinct sets). That smooths the curves but does not move the step. Taking the lowest-indexed 3 of 20 random units is harmless; size (B1) is the problem. 59A reproduces the step on 12 new prompts. [VERIFIED IN FILES]

**B9. The threshold shape could come from how one layer feeds the rest.** Likelihood: medium. Pre-norm Qwen adds the push to the residual linearly, while OLMo-2's post-norm caps it. That alone could produce "threshold in Qwen, saturation in OLMo" for any strong unit. Experiment 2's size-matched control arms test this. [INFERRED]

## (c) Three experiments, ranked by how likely each is to change a conclusion

None has been run. Column-norm-scaled controls appear only in 18b.7 and 18c (Qwen generation at +2x), never in 59B or a ladder. There is no OLMo dose between 0 and +1 in the 26.8 harness, no phase-split clamp, and no second reader for 54 or 59.

**1. 59B with controls matched in push size.** This could overturn the Part 59 OLMo/Qwen contrast.
- Varied: push size. Target and 10 random same-layer units per model, clamped at ±c with c × ||W_down[:,u]|| equal to the target's 1× write, then 0.25×, 0.5× and 2×.
- Measured: symmetric count per unit (same pooled rule) and the next-layer residual-norm ratio. The target counts only if it beats all 10 matched controls at 1×.
- How it could fail: if OLMo's 639 still beats every size-matched control, the OLMo result survives and B1 is wrong.
- Size: 2 models × 11 units × 4 sizes × 2 signs × 200 documents (deterministic means) + baselines ≈ 35,600 forward passes. Cost: **one 3090 hour**.

**2. A finer dose ladder in one harness, clamping prompt and generation separately.** This could overturn "Qwen threshold versus OLMo gradual".
- Varied: dose on both units (+0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0) and phase (both, generation only, prompt only). Independent seeds per dose, both models in fp32, the same 3 questions × 8 samples, one column-norm-scaled control per model at 1.5 and 2.0.
- Measured: OFFWORLD proxy plus reader; a logistic fit of slope and midpoint, with dose also in alien-frame-maximum units.
- How it could fail: if OLMo rises gradually over 0.25 to 1.0 and Qwen's slope stays more than 4× steeper, the threshold claim stands.
- Size: 2 models × 10 doses × 3 phases × 24 = 1,440 generations + 96 control. Cost: **one 3090 hour** plus about $1 to $2 for the reader.

**3. A second reader with a planet-framing label, on answers already collected.** This could overturn "only Qwen3-4B did anything, noisily" and "pushing down changes nothing".
- Varied: the labelling scheme, on the existing Part 54 and 59 answers (about 4,000 texts): (i) mentions a planet or place of origin; (ii) maker identity (own lab, another real lab, invented, none); (iii) the original EARTH/OFFWORLD scheme. A second, non-Opus reader, condition hidden, with a keyword baseline.
- Measured: kappa against the first reader, and the Qwen3-4B base rate under (i).
- How it could fail: if the second reader also gives EARTH 4 to 6 of 24 on unclamped Qwen3-4B, and calls the Qwen3-4B up side unlike Qwen3.5's, the "noisy baseline" stands.
- Size: about 3,800 answers × 1 call; no GPU. Cost: **API dollars**, under $5 in batch.

Proxies (case-insensitive, run ad hoc on each `clamp-*.json`): off-world `alien|extraterrestrial|planet|galax|star|space|interstellar|species`; sci-fi adds `starship|spaceship|colony|orbit|android|robot|cyber|futur|mars|nebula|asteroid|quantum`; Earth `\bearth`.
