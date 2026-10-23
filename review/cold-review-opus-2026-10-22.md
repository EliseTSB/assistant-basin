# Cold review of Parts 54 and 59 (2026-10-22)

Reviewer: a fresh Claude session with no prior context, reading the repository at commit 7d1d2d9. I read README.md (the Part 54 and 59 sections), PREDICTIONS.md (54a-54h, 59a-59h), PART54-DESIGN.md, PART59-DESIGN.md (with Amendment 1), PART26-LADDER.md, `part26.py`, `neuron_report.py`, `part59/`, and every file in `results/part54a`, `part54b`, `part59a`, `part59b` and `part26-8`. The blind labels are not in the repo, so I checked label counts against a keyword proxy that I computed from the answer texts (regexes are in the method note at the end). Each claim is marked [VERIFIED IN FILES] or [INFERRED].

## (a) Number check

I checked 47 numbers in the two README sections. 42 match or are consistent with the files. 5 do not match.

**Mismatches**

1. **The Qwen3-4B "noisy Earth base rate" is not in the answers.** The README says Qwen3-4B's "own unclamped answers and random controls already say 'Earth' 0 to 6 of 24". The word "Earth" appears in **0 of 168** unclamped Qwen3-4B answers and in **0 of 24** answers for each control at each dose (`part54b/clamp-qwen3-4b-L28u4248*.json`). [VERIFIED IN FILES] So the base rate comes from the reader: it gives EARTH to ordinary "created by Alibaba Cloud" answers (a human-maker reading). It is not something the model says. [INFERRED]
2. **"control 397: 6 of 24 at -2".** None of control 397's 24 answers at -2 contains "earth". [VERIFIED IN FILES] Same cause as item 1.
3. **"Random same-layer controls at -2 and +2: at most 1 of 96 on any label"** (59A). This can't hold literally, because the README also says the unclamped stories are EARTH-REAL 96 of 96, and the control stories are ordinary too. PREDICTIONS 59d says the "at most 1" applies to SCIFI. The README should say SCIFI. [INFERRED]
4. **"about 4.1 symmetric neurons per control expected by chance".** This is right for Qwen (82,944 × 0.00005 = 4.15). For OLMo it is 40,960 × 0.00005 = **2.0**. The README states 4.1 as if it applied to both models. [VERIFIED IN FILES]
5. **"Gemma 4 12B did not run".** It was started and failed to load: `OSError: google/gemma-4-12B-it does not appear to have a file named ... model.safetensors` (`part54b/gemma4-12b-it.log`). [VERIFIED IN FILES] This is a wording fix only.

**Matches** [VERIFIED IN FILES]

- 24 answers per dose; scale 19.125 (Qwen) and 6.5 (OLMo).
- OLMo-7B has 48 answers per dose, and its auto "alien" count is 1 at +2.
- Gemma E2B shows no alien or Earth words at any dose.
- 59B: 1,000 documents; 22 of 82,944; controls 80, 4, 0; 639 of 40,960; controls 151, 186, 187; ratio 639/187 = 3.42; control maxima span 19.77/2.18 = 9.06×.
- OLMo 10, 13, 11 matches the PART26-LADDER table.

**Blind-label counts checked only against the keyword proxy.** All 22 point the same way as the reader.

| Count | README (reader) | Keyword proxy |
|---|---|---|
| Qwen3.5 OFFWORLD at -3, -2, -1, +0.5, +1, +1.5, +2 | 0, 0, 0, 0, 1, 19, 24 | 0, 0, 0, 0, 1, 23, 24 |
| Qwen3.5 SCIFI at -2, -1, 0, +1, +2 (59A) | 0, 0, 0, 5, 69 | 0, 0, 2, 8, 92 |
| OLMo SCIFI at -2, -1, 0, +1, +2 (59A) | 1, 0, 0, 58, 79 | 1, 2, 1, 47, 72 |

## (b) Ways the conclusions could be artefacts

**B1. In 59B the symmetric count tracks how hard each unit is pushed, not which unit it is.** Likelihood: high.
- The clamp sets a unit to ±1× its own Pile maximum. Controls are not matched to the target in absolute size.
- In OLMo the control maxima are 0.74, 0.50 and 1.30, against the target's **6.5**. Each control is pushed 5 to 13 times more weakly. [VERIFIED IN FILES: `59b-olmo2-1b-L10u6978.json` pile_maxima]
- In Qwen, control 152 is pushed about as hard as the target (19.77 against 19.125). It gives **80** symmetric responders, against the target's 22. Weaker controls give fewer: 376 at 13.5 gives 4, and 692 at 2.18 gives 0. [VERIFIED IN FILES]
- So within Qwen, the count rises with push size. Across both models, "OLMo stands out, Qwen is null" is what push size alone predicts.
- A plausible even-order mechanism: RMSNorm. A large write in either direction raises the residual norm, which shrinks other features. That change has the same sign for −M and +M. [INFERRED]
- OLMo-2 also normalises each MLP's output (post-feedforward norm) before adding it to the residual. With one dominant unit, that suppresses the rest of that MLP's output for both signs alike. [INFERRED, from the OLMo-2 architecture, not checked in code here]
- The leave-one-out thresholds make this worse. OLMo's target is judged against thresholds set by weak pushes. Qwen's target is judged against a pool that includes a push of matching size.
- What would show it: column-norm-scaled controls (the 18b.7 method), sized so that value × ||W_down[:,u]|| equals the target's, plus a count-versus-push-size curve. This is experiment 1.

**B2. "OLMo is gradual" has no data where a threshold would show.** Likelihood: high.
- Part 26.8 has no OLMo dose between 0 and +1. Above +1 it is flat: 10, 13, 11 of 24, and the proxy gives 15, 19, 19. [VERIFIED IN FILES: `part26-8/`]
- Part 26.7 had +0.5 on one question: proxy 1 of 8, and the hand reading said "ordinary". At +1 it had 3 of 8. [VERIFIED IN FILES: `part26/clamp-olmo2-1b-L10u6978-x0.5.json`]
- So OLMo also jumps, somewhere between +0.5 and +1, and then saturates near 50%. The rest leaks to EARTH or BROKEN.
- The difference may be where the step sits and how high it tops out, not its shape. [INFERRED]
- 59A supports this reading. At +1, OLMo is already at 58 of 96 while Qwen is at 5 of 96. [VERIFIED IN FILES via proxy: 47 and 8]

**B3. The dose scale puts Qwen's step at the edge of the unit's natural range.** Likelihood: medium-high.
- Qwen's alien-frame maximum is 24.69, which is **1.29×** its Pile maximum. [VERIFIED IN FILES: `scale_frames.ratios_to_part26_pile_maximum`]
- The step falls between +1.0 (19.1) and +1.5 (28.7). That interval brackets the most the unit ever does on alien text.
- So "threshold" may only mean "nothing happens until the clamp goes past natural alien-level firing". That says something about the scale, not about the unit's transfer function. [INFERRED]
- OLMo's alien-frame ratio is not recorded in these files, so the two scales can't be aligned across models. [VERIFIED IN FILES: absent]

**B4. The clamp is applied at every position, including the prompt and chat template.** Likelihood: medium.
- `NeuronHook` writes the value at every token. [VERIFIED IN FILES: `neuron_report.py:203`]
- At +1.5, every prompt token carries 28.7. The step could come from prompt tokens (for example `<think>\n\n</think>`) crossing an attention-routing threshold, rather than from the unit's role during generation.
- The hook already supports `phase='gen'` and `phase='prompt'`, so this is cheap to test. [VERIFIED IN FILES]

**B5. The reader's EARTH/OFFWORLD axis may split one effect into two labels.** Likelihood: medium-high for the Part 54 twin claims.
- The Qwen3-4B twin at +1 writes "developed by Alibaba ... on Earth ... I do not come from other planets" and "created on Earth, on the planet Mars". [VERIFIED IN FILES]
- By keyword it is a clean, graded dial for planetary framing: 0 of 168 unclamped, 8 of 24 at +0.5, 24 of 24 at +1, and 0 in every control. [VERIFIED IN FILES]
- The reader calls this EARTH. Qwen3.5's up side is "extraterrestrial". Both may be one latent, "frame origin as a planet". [INFERRED]
- If so, conclusion (1) changes in two ways. The Qwen3-4B twin is real, not noisy. And it is graded, not a threshold.

**B6. The down side is not "nothing" in Part 54; it is erasure of the maker's identity.** Likelihood: high as a fact; medium as a confound.
- At −2 and −3, Qwen3.5 names Qwen in 2 and 0 of 24 answers (24 of 24 at +1). It says "made by **Google**", "**Tencent** AI Lab", "**Ding Yuan**". [VERIFIED IN FILES: summary.names_qwen]
- The reader files these as NEITHER.
- The matched-size control 152 also drops Qwen naming to 13 of 24 at −2. [VERIFIED IN FILES] So part of this is generic.
- In 59, OLMo at −2 moves 14 of 96 stories off EARTH-REAL (PREDICTIONS 59c), and they read as degraded and repetitive. [VERIFIED IN FILES]
- So "pushing down changes nothing" is true only for the setting label. [INFERRED]

**B7. Mixed dtype.** Likelihood: low-medium.
- 59A and 59B ran OLMo in bf16 and Qwen in fp32. [VERIFIED IN FILES]
- In 59B both doses are measured against one shared baseline. Any rounding error in that baseline enters both changes with the same sign, which nudges small changes toward "symmetric". This matters most for the weak OLMo controls. [INFERRED]

**B8. Sampling and seeds.** Likelihood: low.
- Each condition uses the same seed at every dose and every prompt (twin 1919609538 throughout). The same 8 random streams are therefore reused. [VERIFIED IN FILES]
- That makes the curves look smoother and more monotone than independent draws would. It does not move the step.
- The 24 "unclamped" answers per question are reused for all seven dose files, not fresh. [VERIFIED IN FILES: 3 distinct unclamped sets]

**B9. Control selection.** Likelihood: low.
- The sorted list is the first 3 of 20 random units. The order is by index, which is unrelated to activation. [VERIFIED IN FILES: `neuron_report.control_units`]
- The real problem is size (B1), not selection.

**B10. Only three questions and 12 prompts.** Likelihood: low for the step itself, which 59A reproduces on new prompts. Medium for how general the claim is.

**B11. The threshold shape could come from how one layer feeds the rest.** Likelihood: medium. Pre-norm Qwen adds the push to the residual linearly, while OLMo-2's post-norm caps it. That alone could produce "threshold in Qwen, saturation in OLMo" for any strong unit. Experiment 2's size-matched control arms test this. [INFERRED]

## (c) Three experiments, ranked by how likely each is to change a conclusion

None of these has been run. README and PREDICTIONS show column-norm-scaled controls only for 18b.7 and 18c (Qwen generation at +2x), never for 59B or for a dose ladder. They also show no OLMo dose between 0 and +1 in the 26.8 harness, no phase-split clamp in 54 or 59, and no second reader for 54 or 59.

**1. 59B with controls matched in push size.** This could overturn the Part 59 OLMo/Qwen contrast.
- Varied: push size. For each model, take the target and 10 random same-layer units, each clamped at ±c, where c is scaled so that c × ||W_down[:,u]|| equals the target's 1× write, then 0.25×, 0.5× and 2× of that.
- Measured: symmetric count per unit, using the same pooled rule.
- The target counts only if it beats 10 of 10 size-matched controls at 1×.
- Also log the residual-norm ratio at the next layer, to test the RMSNorm explanation.
- How it could fail: if OLMo's 639 still beats every size-matched control, the OLMo result survives and B1 is wrong.
- Size: 2 models × 11 units × 4 sizes × 2 signs × 200 documents (enough, because the means are deterministic) + 2 baselines ≈ 35,000 forward passes.
- Cost: **one 3090 hour**. OLMo alone takes CPU minutes to about an hour.

**2. A finer dose ladder in one harness, clamping prompt and generation separately.** This could overturn "Qwen threshold versus OLMo gradual".
- Varied: dose, on both units, at +0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 and 2.0. Phase: both, generation only, prompt only.
- Use independent seeds per dose. Run both models in fp32, so OLMo is rerun in fp32 too.
- Use the same 3 questions × 8 samples.
- Measured: the OFFWORLD keyword proxy plus the reader; a logistic fit of slope and midpoint, with dose also given as a multiple of the alien-frame maximum.
- How it could fail: if OLMo's curve rises over 0.25 to 1.0 and Qwen's slope stays more than 4× steeper, the threshold claim stands.
- Add one column-norm-scaled control at 1.5 and 2.0 per model.
- Size: 2 models × 10 doses × 3 phases × 24 = 1,440 generations, plus 96 for controls.
- Cost: **one 3090 hour** of generation plus about $1 to $2 of API for the reader batch.

**3. A second reader with a planet-framing label, on answers already collected.** This could overturn "only Qwen3-4B did anything, noisily" and "pushing down changes nothing".
- Varied: the labelling scheme. Apply it to the existing Part 54 and 59 answers, about 4,000 texts.
- Labels: (i) mentions a planet or place of origin at all; (ii) maker identity: own lab, another real lab, invented, none; (iii) the original EARTH/OFFWORLD scheme.
- Use a second model that is not Opus, condition hidden, with a keyword baseline computed alongside.
- Measured: agreement with the first reader (kappa), and the Qwen3-4B base rate under (i).
- How it could fail: if the second reader also gives EARTH 4 to 6 of 24 on unclamped Qwen3-4B, and calls the Qwen3-4B up side unlike Qwen3.5's, the "noisy baseline" stands.
- Size: about 4,000 answers × 1 call each; no GPU.
- Cost: **API dollars**, under $5 in batch.

Method note: keyword proxies (case-insensitive). Off-world: `alien|extraterrestrial|planet|galax|star|space|interstellar|species`. Sci-fi stories add `starship|spaceship|colony|orbit|android|robot|cyber|futur|mars|nebula|asteroid|quantum`. Earth: `\bearth`. The scripts were run ad hoc from the answers in each `clamp-*.json`.
