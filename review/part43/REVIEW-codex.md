Review of Part 43 through Amendment 2, against repository commit `5b6c87b`.
These are proposed additions, not amendments to the registered predictions. No Qwen inference was run for this review.

My recommendation is to start with a small L7 → L11 activation-swap experiment, then separate L11's gate and value, and only then test whether the effect tracks adjective attachment. The existing amendments already address many of the important scoring and ablation problems. The largest remaining issue is treating connectivity, nonlinear interaction, and semantic composition as if one test could decide all three.

1. **Interpretation blocker: a path and an additive response are compatible.**

   Distinguish three questions:

   | Claim | Evidence that would address it |
   |---|---|
   | L7 changes L11 | Intervene on L7 with text fixed; measure L11 before the output scorer |
   | That change in L11 matters for the maker score | Transfer and remove the induced L11 change in controlled runs |
   | L7 changes how nationality is applied to an entity | The intervention changes a held-out attachment/selectivity contrast, and the relevant internal change carries that effect |

   A linear chain can transmit a useful signal without nonlinear interaction. Conversely, independent writers can interact through the output normalization or scorer. Even the natural/zero lattice can be exactly additive despite a real path, if downstream effects cancel on its four corners.

   A concrete example: let `U` be 0 or 1, `M = keep_M * (1 + U)`, and `Y = 2U + M - UM/2`. The natural/zero lattice is `(0, 2, 1, 3)`, with zero pair interaction. But changing U changes M, and transferring that induced M change into a clean run changes Y by −0.5. Restoring M in the U-zeroed run recovers 1 of the 2 lost score units.

   **Change:** score the three claims separately. Preserve 43.3a as the original compound prediction, but add a path criterion that does not require its interaction clause. A failed interaction clause must not veto evidence of communication. The accompanying [counterexample script](causal_counterexamples.py) checks this example, a linear chain, the scorer artifact, and the multiplier issue below.

   Also, L11's output is `a11 * W_down[:,8868]`: its write direction is fixed. L7 cannot rotate that column. It can change L11's response to inputs, or change the later context in which other components read that fixed direction. These are distinct versions of “shapes.”

2. **Highest-value next experiment: complete the four-world receiver swap.**

   Amendment 2 already requests a receiver-only replay. Make it a fully specified table. On one identical teacher-forced token sequence, cache L11's entire scalar trajectory with L7 natural (`m0`) and L7 zeroed (`m1`). Let `Yij` mean upstream condition `i` and L11 trajectory from donor `j`, where 0 is natural and 1 is L7-zeroed:

   | Run | L7 | L11 at every real position | Purpose |
   |---|---|---|---|
   | Y00 | Natural | m0 | Clean reference |
   | Y11 | Zero | m1 | Full upstream intervention |
   | Y10 | Zero | m0 | Remove the L11-mediated change |
   | Y01 | Natural | m1 | Transfer only the L11-mediated change |

   Replace only N8868 at the input to L11's `down_proj`. Other coordinates remain live. Y00 and Y11 replay versions must reproduce their donor runs; retain the donors as numerical references.

   Report these signed quantities in score units, not just a rescue fraction:

   ```text
   total effect                         T = Y11 - Y00
   effect with L11 held clean           B = Y10 - Y00
   L11 change in upstream-zeroed world   N = Y11 - Y10
   L11 change in clean upstream world   S = Y01 - Y00
   interaction between these effects   I = N - S
   exact identity:                      T = B + N
   recovery fraction:                   N / T, when T clears the registered floor
   ```

   Under strictly independent branches, changing L7 leaves `m1 = m0`, so both S and N vanish even if a final softmax is nonlinear. A changed receiver plus consequential S/N is evidence for an intervenable route through that receiver. If S and N disagree, report that background dependence; neither is a universal “percentage of the circuit.” Opposing paths can also make T small while S or N is substantial, so keep their absolute values when the fraction is NOT_TESTABLE.

   This establishes a route through L11's scalar under these interventions. It does not establish a direct connection, its unique necessity, or an adjective computation. A negative result at this scalar also does not rule out communication through other L11 neurons.

   **Positions:** use `use_cache=False` initially and rerun the full identical prefix. All-position scalar restoration blocks routes that actually pass through those coordinates, including cross-position routes into them; it leaves routes around them active. A one-position patch is a deliberately narrower question. A changed generated continuation is a different input, so do not replay later clean activations onto it as if it were the same counterfactual. Each teacher-forced answer candidate needs its own aligned donor traces.

3. **Separate L11's gate and value; this adds an exact mechanistic decomposition.**

   Cache raw gate `g`, raw up/value `v`, and `a = SiLU(g) v` in both upstream conditions. Write `h_i = SiLU(g_i)`. The scalar change decomposes exactly:

   ```text
   a1 - a0 = (h1 - h0) v0 + h0 (v1 - v0) + (h1 - h0)(v1 - v0)
                 gate            value                joint
   ```

   Then construct the four products `h0*v0`, `h1*v0`, `h0*v1`, `h1*v1`. Replay each into N8868 in each of the two upstream backgrounds: eight conditions, including the four worlds above. This requires no whole-layer transplantation: compute the products from cached branch values and replace just the scalar. It is equivalent, at this neuron's output, to exchanging its two branches separately.

   This answers whether L7's effect on L11 is carried mainly by its gate, its value, or their combination, and whether each branch change affects the maker score. The scalar decomposition is exact; an analogous additive decomposition of the final model score is not guaranteed.

   **Why this follows from Goodfire:** the supplied essay's discussion of sign regimes shows how the same positive product can arise from positive-positive or negative-negative branches, and suggests gates and values can have different selectivity. Treat that as a concrete measurement suggestion, not as evidence that these particular units have the proposed meanings. Record all four sign regimes and natural activation ranges. A gate-branch effect alone is still not proof of semantic gating.

   Proposed first batch: the discovery maker prefix plus 12 held-out maker prefixes, plain only, eight conditions each = **104 condition-prefix evaluations**, plus numerical checks. Full-name scoring multiplies the work by the number and length of candidate continuations. Do the small scalar/margin pilot before the full random-lattice sweep.

4. **The semantic test needs attachment controls and a position after the relevant words.**

   In “the Chinese search engine,” the model at “Chinese” cannot yet see “search engine.” Recording there can test nationality sensitivity, but cannot establish combination with the upcoming role or entity. 43.5a is therefore a descriptive nationality prediction, not a test of Elise's composition hypothesis.

   Add pairs such as:

   ```text
   The Chinese search engine partnered with the American charity.
   The American search engine partnered with the Chinese charity.
   ```

   Follow both with the same query/suffix, for example “The search engine is based in”. Both contain the same nationality and entity words; the attachment changes. Counterbalance mention order, vary syntax, and reserve new nouns/templates for confirmation. Also include China as an irrelevant bystander attribute, bare China mentions, and role without nationality. Keep named-company association tasks separate from fictional or unnamed-entity binding tasks. Both are useful, but success on one does not imply the other.

   Apply the L7 intervention and measure L11 gate/value/product at aligned positions after the whole phrase and at the query's final position. For each matched pair k, compare its L11 contrast with L7 natural versus altered. A constant offset disappears in this difference of differences. To distinguish broader gain from a change in selectivity, fit the restricted relation `a11_altered = alpha + beta * a11_clean` on development prompts, then test its predictions on held-out attachment pairs. A systematic attachment-specific residual rejects that affine-gain account. It still needs an intervention on the receiver to connect it to behavior.

   **Two possible locations of modulation:** if L11's input selectivity changes, branch swaps above can localize it. If L11's scalar does not change but its downstream effect does, use the same small absolute `+epsilon/-epsilon` pulse on L11 in both L7 backgrounds and read **L29 gate/value**, before the final output scorer. A changed L11 → L29 sensitivity localizes contextual readout downstream. Follow up other readers if L29 is unresponsive; an averaged Google lens is not a reason to assume L29 is the only recipient.

   Natural donor swaps at L7 can then ask whether an attachment-relevant state transfers across held-out templates. Use both transfer directions and same-attachment/different-wording donor controls. A zeroing lesion is useful discovery evidence; a predicted cross-context transfer is stronger evidence for the proposed semantic role.

5. **Implementation blocker: distinguish the residual bypass derivative from the full L7 → L11 derivative.**

   I checked the pinned checkpoint's configuration and the installed Transformers implementation. Qwen3.5-4B uses SiLU, a dense gated MLP, and zero-centred RMSNorm with `gamma = 1 + weight`. Layers 8, 9, and 10 use GatedDeltaNet; layer 11 uses full attention. The receiving MLP reads the residual **after L11's token mixer**, through `post_attention_layernorm`.

   Amendment 2's contextual formula is correct for the direct residual-bypass direction:

   ```text
   a_j = SiLU(w_g · N(r)) (w_v · N(r))
   D_a_j[v_i] = SiLU'(g) value (w_g · DN(r)[v_i])
              + SiLU(g)       (w_v · DN(r)[v_i])
   ```

   Here `v_i = W_down[:,i]`. But the total source derivative replaces `v_i` by `d r11 / d a7`, which includes intervening MLPs, token mixers, and other positions. A finite difference applied at L7 tests the **total** derivative; it need not agree with the bypass formula. Validate that formula with the matching receiver-only intervention, then measure the total separately.

   For a finite bypass intervention, privately evaluate N8868 on `N(r11_clean + delta_a7 * v7)` and splice only that resulting scalar into the otherwise clean computation. Do not add an unnormalized write directly to the normalized input. Other receiver coordinates must not change. At first order, use `DN(r)[delta_a7*v7]`; for a finite change, recompute the norm.

   For total communication, obtain `K[t,s] = d a11[t] / d a7[s]` by differentiating the actual forward pass, and check selected entries with central finite differences. Causality requires `K[t,s]=0` for `s>t`; useful earlier-position entries can be nonzero. If the bypass is weak but the total is strong, use receiver swaps through the intervening mixer outputs to locate the route. Freezing only conventional attention heads would miss the GatedDeltaNet route.

   Part 42's implementation uses `W_unembed diag(gamma) J[l] W_down[:,i]`, with the matching block-output index. That agrees with its stated linear readout. It omits the contextual RMS divisor and norm derivative, appropriately for its token ranking purpose. Its averaged J also aggregates over corpus examples and target positions; it is not the local Jacobian of this maker answer. Use it to suggest hypotheses, not to assign an answer-specific causal sign. A zeroing perturbation is `delta_a = -a`, not `+a`.

6. **The current-activation multiplier grid has coupled axes.**

   With `a11 = c + k*a7`, multiplying the current receiver activation gives `a11' = t*(c+k*s*a7_clean)`. A product `s*t` appears even when the underlying communication and output are linear. Curved or nonparallel curves can reflect this control definition, clipping, or the output scorer.

   **Change:** retain the original grid as a gain experiment, but record the realized activation trajectories and clipping fraction. For a grid with independent coordinates, set each unit to values derived from its clean, cached trajectory or to fixed absolute levels chosen in advance. A near-zero clean activation needs absolute offsets, since multiplying zero explores nothing. Absolute receiver clamps sever its incoming dependence on L7, so that grid measures downstream combination, not whether L7 normally controls L11. Use the four-world swap for the latter.

   Fix 43.10's slope definition before the run. A signed ratio of slopes is unstable near zero and can count opposite signs as “less than two.” Specify intervals, sign handling, a minimum slope, and numerical tolerances. Clipping-induced plateaus must not be mistaken for evidence of independence or monotonic biology.

7. **Improve the additive surrogate and limit what the matched-null rank means.**

   Amendment 2 correctly includes a nonlinear-score fixture. Strengthen the residual surrogate by using measured **final residual changes from the single ablations**, rather than just the four original write vectors:

   ```text
   h_add(S) = h_clean + sum_i_in_S (h_single_ablation_i - h_clean)
   ```

   Run this through frozen-scale and actual final RMSNorm/unembedding. Also form the analogous additive surrogate directly on full vocabulary logits. These distinguish failure of singleton superposition inside the network from interaction introduced at the final norm or scorer. Original writes alone miss all single-ablation-induced downstream changes. For multi-token names, build the surrogates separately at every scored position on each fixed candidate history.

   Alongside individual logits, use a prespecified raw logit **difference** for an unambiguous token pair on an identical prefix. It remains linear in logits and removes common logit shifts. Full-name log likelihoods are valuable behavioral scores, but candidate histories differ after the first divergence, and summing aliases remains nonlinear. Fix whitespace, spelling, termination, and alias sets in the manifest. Do not interpret the probability of a few selected strings as total probability of a maker concept.

   Matching random units by single effect, activation scale, and write norm is much better than layer-only matching. It supplies a reference for how unusual an effect is. It is not a sample from “networks with no L7 → L11 path,” and the target set was adaptively selected for a joint flip. The rank formula alone does not make targets and nulls exchangeable. Freeze matching tolerances, signs, fallback rules, and seeds on development prompts; report achieved matches and the control sampling procedure. Treat discovery-prompt ranks as conditional reference ranks, then confirm on held-out prompt families. Keep the pair-specific comparison separate from the maximum-over-six family statistic.

8. **The preregistration still has executable ambiguities.**

   | Row | Proposed clarification before execution |
   |---|---|
   | 43.1a | Testable as a sign prediction; define a numerical band around zero. Its reference to 43.8 as the per-unit “what they push” test is incorrect: 43.8 is set-level transfer. |
   | 43.2a | Amendment 2 switches Möbius to Walsh coefficients. Name the scored coefficient explicitly and define the denominator of “within 25%,” with a near-zero NOT_TESTABLE rule. A failure to exceed nulls is not evidence of independence. |
   | 43.3a | Define absolute activation change as `abs(a_zero-a_clean) >= max(0.2*abs(a_clean), 0.1)` if magnitude is intended; state any directional prediction separately. A bootstrap cannot supply uncertainty for one deterministic discovery prompt. Report that case exactly; use prompt/template clusters for a separate transfer interval. Split path evidence from nonlinearity as in finding 1. |
   | 43.4a | “Same interval rule” conflicts with the opposite directional claim: a claim of rescue below half needs an upper bound below 0.5, not 43.3a's lower bound above 0.25. Specify whether the measured claim concerns one prompt or a held-out aggregate. |
   | 43.5a | Testable as a descriptive 16/20 nationality count; it cannot test future role/entity composition at the adjective. Define treatment of numerical ties. |
   | 43.6a | Adopt Amendment 2's two-sided, six-comparison rule explicitly in the row. Separate bypass and total derivatives and their finite-difference validations. Lack of an extreme rank does not imply lack of a path. |
   | 43.7a | Fix greedy versus sampled decoding, truncation, realized clamp values, and the maker-label rule. This is an answer prediction, not a sign inference from the lens. |
   | 43.8 | Specify mean/median of paired held-out effects and a positive minimum discovery effect. If that reference rise is absent, the ratio-based transfer claim is NOT_TESTABLE. Keep languages and prompt families separate. |
   | 43.9 | Specify the donor distribution, token alignment, and number of donor draws. A counterfactual system prompt is a different treatment from erasing a unit. Failure to flip under that treatment does not by itself invalidate zero-ablation evidence. |
   | 43.10 | Define monotonicity tolerance and slope intervals/signs; handle zero slopes and clipped endpoints. Distinguish the gain grid from independently clamped coordinates. |

   Bootstrap whole matched prompt/template clusters, not aliases, the 16 intervention cells, tokens, or repeated deterministic forwards. Authored paraphrases warrant sensitivity analysis over those prompts, not a population claim about all maker questions. Keep the original predictions in the record; add explicit replacements or additional outcomes rather than silently changing their meaning.

9. **Further experiments, in the order I would run them.**

   **First, map suppression versus replacement across competing makers.** Track Alibaba, Google, Baidu, Tencent, OpenAI, and name/maker separately. The archived L7 logit-lens **bottom** list already contains China terms, so its logit lens is not wholly noise. L11 writes China-related terms; L29 writes both Google and Baidu. A plausible rival to adjective composition is redundant country evidence plus a broadly shared search/company writer, with Google winning after evidence is removed. Measure changes relative to clean on each competitor, then test the receiver route. This is a hypothesis, not a conclusion from word lists.

   **Second, exploit Part 32's language dissociation.** Repeat the small swap test on matched English, Spanish, and Chinese question families, reporting each language separately. Ask whether Chinese bypasses these four units or whether the same internal effect fails to change the winning answer. Spanish's Tencent competitor makes this especially informative. Select each language's score/tokenization before measuring it; equal raw margins across languages are not required.

   **Third, separate ordinary company knowledge from self-reference.** Compare second-person maker questions with third-person Qwen questions, company-description completions, and simple country/company relations. Counterbalance predicates such as created/made/developed and pronouns. A path that operates equally in third person is more naturally described as factual/name production machinery; a self-context-specific routing change would point toward L6 or other context gates. Use held-out forms, since “Who created you?” selected the original set.

   **Fourth, separate the identity residual path from changed downstream computation.** At a fixed teacher-forced position compare (a) early L7 ablation, (b) subtracting its clean write only from the clean final pre-norm residual, and (c) early ablation with that removed write added back at the final pre-norm residual. Arm c restores the same-position identity-bypass contribution while retaining induced downstream changes. It tests whether the early intervention did more than remove a final vote. It does not isolate L11 or all cross-token paths; combine it with finding 2. This is closely motivated by the supplied essay's Poe example, which compares full removal with removal of the direct output contribution.

   **Fifth, compare base and chat conditional on neuronal alignment.** Reuse Part 27/39 activation correspondence, then repeat successful path/branch measurements. Ask whether post-training changed the sender, the receiver, or the subsequent readers. Matching indices alone is weaker evidence than matched response profiles. Keep plain-completion and chat-template results separate.

   For finding additional readers, gradients or attribution patching can screen candidates cheaply; exact finite receiver swaps should decide the claims. For four known units, another approximate interaction index is less valuable than these interventions.

10. **Prior art and what it changes here.**

   - [Wang et al., *Interpretability in the Wild*](https://arxiv.org/abs/2211.00593), and [Goldowsky-Dill et al., *Localizing Model Behavior with Path Patching*](https://arxiv.org/abs/2304.05969): distinguish component membership from the routes carrying effects. These motivate the donor/recipient experiments and precise declarations of which paths remain live. They do not establish this maker circuit.
   - [Zhang and Nanda, *Towards Best Practices of Activation Patching*](https://arxiv.org/abs/2309.16042), and [Heimersheim and Nanda, *How to Use and Interpret Activation Patching*](https://arxiv.org/abs/2404.15255): measurement and corruption choices can change localization conclusions. Keep paired raw effects, several intervention baselines, and explicit scope for each rescue claim.
   - [Feng and Steinhardt, *How Do Language Models Bind Entities in Context?*](https://arxiv.org/abs/2310.17191): causal experiments distinguish attributes from which entity they belong to. This motivates the same-word/different-attachment test. Their binding-ID result is an adjacent precedent, not evidence for a nationality-adjective MLP neuron in Qwen3.5.
   - [Ghorbani and Zou, *Neuron Shapley*](https://arxiv.org/abs/2002.09815): coalition ablations can allocate contributions while accounting for interactions. With four units, the complete lattice is affordable; Shapley summaries can describe credit across contexts, but cannot determine a directed edge or semantic role.
   - [Anthropic's Jacobian-lens reference implementation](https://github.com/anthropics/jacobian-lens): the map averages input-output Jacobians over prompts and source/current-and-future target positions. That supports the readout distinction in finding 5 and explains why a J-lens word list is not a prompt-specific causal graph.
   - [Zhong, *Phantom Transfer of Model Self-Identification*](https://www.cs.cmu.edu/~csd-phd-blog/2026/model-self-identification/): teacher-response fine-tuning can change stated identity even after filtering explicit model/lab names. It supplies a behavioral training intervention for a later mechanistic extension; it does not locate a four-neuron maker circuit. The repo's earlier smaller-model results should be compared at their actual scale and training conditions.
   - [Cammarata, *Neural Networks Are Already Clean*](https://already-clean.goodfire.com/), using the supplied archived text: the actionable ideas here are separate signed branches, upstream/downstream checks of explanations, and tests of specific predicted uses. The essay's broad cleanliness hypothesis is not needed for any proposed experiment to be informative.

   In this search I did not find a published direct tracing of these maker/name claims or an established nationality-adjective mechanism matching these four coordinates. That is a bounded search result, not a novelty claim.

Verification for this review: inspected the bundled Part 25 code/readouts and Part 42 readout implementation, the local pinned model configuration and installed Qwen3.5 forward code; executed the synthetic counterexamples and RMSNorm finite-difference check. The new experimental protocol still needs implementation and real-model validation.
