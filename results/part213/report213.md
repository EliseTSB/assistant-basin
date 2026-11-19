# Part 213 report: every number the results README uses

Made by `part213/report213.py` from the files named at the end of each line. Values recomputed here from the
Stage 1 records through the scorer's own `Stage1` are checked against `score.json` at the end; a `DIFF` line
there means the two disagree. `score.json` is the scorer's output and is never edited here.

## Verdicts (results/part213/full/score.json, key predictions.<n>.verdict)

213.1  verdict=held  clauses={"slot_both_positive_ge_18": "held", "concept_both_positive_ge_15": "held", "slot_median_L_lo_ge_1": "held", "slot_median_N_lo_ge_1": "held", "concept_median_L_lo_ge_1": "held", "concept_median_N_lo_ge_1": "held", "slot_own_full_np_ge_18": "held", "concept_own_full_np_ge_15": "held"}
213.2  verdict=held  clauses={"median_L_lo_ge_0.75": "held", "median_N_ge_0.75": "held", "both_positive_ge_16": "held", "own_full_np_ge_16": "held"}
213.3  verdict=inconclusive  clauses={"tau_L_lo_ge_0.50": "inconclusive", "tau_N_lo_ge_0.50": "inconclusive", "direct_style_median_ge_0.50": "held", "scene_style_median_ge_0.50": "held"}
213.4  verdict=inconclusive  clauses={"median_L_lo_ge_0.50": "held", "median_N_ge_0.50": "held", "tau_M/3_L_lo_ge_0.50": "inconclusive"}
213.5  verdict=held  clauses={"concept_minus_slot_lo_ge_0.25": "held", "agreed_minus_slot_ge_0.25": "held"}
213.6  verdict=untestable (gate failed)  clauses={}
213.7  verdict=inconclusive  clauses={"slot_upper_le_-0.30": "inconclusive", "concept_upper_le_-0.30": "inconclusive"}
213.8  verdict=failed  clauses={"J_self_positive_ge_17": "failed", "rho_J_L_self_ge_0.50": "held", "rho_context_contrast_ge_0.40": "failed"}
213.9  verdict=held  clauses={"sft_median_L_self_ge_0.75": "held", "sft_median_N_self_ge_0.75": "held", "sft_median_L_third_ge_0.50": "held", "sft_median_N_third_ge_0.50": "held", "paired_difference_inside_-0.25_0.25": "held"}
213.10  verdict=failed  clauses={"self_held_minus_self_nonid_lo_ge_0.50": "held", "tau_cop_nonid/3_upper_le_0.50": "failed"}
213.11  verdict=held  clauses={"mean_delta_lp_spec_lo_ge_0.75": "held", "positive_ge_16_of_21": "held", "slot_peer_only_lo_gt_0": "held"}
headline reading: inconclusive / mixed selectivity  (score.json, key headline.reading)
gates: {'gate0': 'passed', 'gate1': 'passed', 'gate2': 'passed', 'gate3': 'failed', 'gate4': 'passed'}  (score.json, key gates)
verdict counts: {'held': 5, 'inconclusive': 3, 'untestable (gate failed)': 1, 'failed': 2}  (counted from the verdicts above)

## Predictions table: registered text (PREDICTIONS.md, `## Part 213`), confidence, result, verdict

213.1 | confidence=high | author=Gemini 4 Neon | verdict=held
    text: Held-out self-slot writing (`instruct` chat, `SELF_HELD`, `+0.8`; replication/calibration on held-out self prompts, NOT shielded by Gate 2): BOTH `L_z(u, SELF_HELD) > 0` and `N_z(u, SELF_HELD) > 0` for at least 18 of 21 slot units and at least 15 of 18 concept units, with 95% CI lower bound of median `L_z` and median `N_z` >= +1.0 logit for both groups, AND (v5) the own-unit full-NP content increment `lp_content_sum(neuron +0.8; u) - lp_content_sum(none; u)` on `SELF_HELD` is > 0 for at least 18 of 21 slot units and at least 15 of 18 concept units (for several units the single first content token is a leading modifier such as ' very', ' large', ' most' or ' powerful', not the persona noun, so the full NP checks the persona itself). Fails if any clause misses.
213.2 | confidence=medium | author=Gemini 4 Neon | verdict=held
    text: Positive neuron writing in third-person entity slots (`instruct` chat, `THIRD_SLOT`, `+0.8`): for the 21 slot units, 95% CI lower bound of median `L_z(THIRD_SLOT)` >= +0.75 logits AND median `N_z(THIRD_SLOT)` >= +0.75 logits, with at least 16 of 21 slot units positive on both `L_z` and `N_z`, AND (v5) the own-unit full-NP content increment `lp_content_sum(neuron +0.8; u) - lp_content_sum(none; u)` on `THIRD_SLOT` is > 0 for at least 16 of 21 slot units. Fails if any clause misses.
213.3 | confidence=low | author=Gemini 4 Neon | verdict=inconclusive
    text: Scale-invariant third-over-self paired transfer (`instruct` chat, `+0.8`, length- and framing-matched): for the 21 slot units (`n_eligible >= 15`), the 95% CI lower bound of BOTH paired transfer ratios `tau_3/1^(L)` AND `tau_3/1^(N)` is `>= 0.50` (powered for true transfer `>= 0.65` on Part 106's spread; CI straddling `0.50` is inconclusive; CI upper bound `<= 0.50` supports first-person selectivity), AND both style-stratified point medians (`direct` pairs and `scene` pairs) are `>= 0.50`. (v5) Each ratio is read only if its numerator sign guard passes. Disposition per the v5 rule: fails if either CI upper bound `< 0.50` or either style-stratified point median `< 0.50`; a straddling CI is `inconclusive`; `n_eligible < 15` is `not read`.
213.4 | confidence=low | author=Gemini 4 Neon | verdict=inconclusive
    text: Extension to non-identity open noun-phrase slots (`instruct` chat, `MENTION_SLOT`, `+0.8`): for the 21 slot units (`n_eligible >= 15`), 95% CI lower bound of median `L_z(MENTION_SLOT)` >= +0.50 logits AND median `N_z(MENTION_SLOT)` >= +0.50 logits, AND 95% CI lower bound of paired transfer ratio `tau_M/3^(L)` >= 0.50, with (v5) the scene-framed `THIRD_SLOT` prompts as denominator and the numerator sign guard. Disposition per the v5 rule.
213.5 | confidence=medium | author=Gemini 4 Neon | verdict=held
    text: Calibration/replication of Part 92 concept-vs-slot split on `SpecLogit` (`instruct` chat, `MENTION_SLOT`, `+0.8`): on the 18-vs-21 Part 92 split, the 95% unit-bootstrap CI lower bound of `median(L_z_concept) - median(L_z_slot)` is `>= +0.25` logits, AND on the 8-vs-21 Part 92b agreed split the point difference `median(L_z_agreed_concept) - median(L_z_slot)` is `>= +0.25` logits. Fails if either clause misses.
213.6 | confidence=medium | author=Gemini 4 Neon | verdict=untestable (gate failed)
    text: Stage 2 blind-audited third-person free generation (`instruct` chat, `+0.8`, greedy 40 tokens; decided strictly by the blind audit, mechanical counts descriptive): on the equal-coverage blind audit of all 42 slot `neuron +0.8` and 42 paired slot `random_0 +0.8` `THIRD_FREE` answers (fixed denominator 42, baseline-absent attributions only; 19-distinct-keyword deduplicated counts printed alongside), `neuron +0.8` achieves at least 10 of 42 `genuine_third_attr` against at most 2 of 42 for `random_0 +0.8`, AND on the 21 paired units the one-sided exact paired unit sign test on discordant units has `p < 0.05` with 95% unit-bootstrap CI of `(neuron_hits - random_hits)` strictly above 0. Fails if any clause misses.
213.7 | confidence=high | author=Gemini 4 Neon | verdict=inconclusive
    text: Sign-matched opposite-direction control (`STANDARD-CONTROLS.md` #1, `instruct` chat, `SELF_HELD`, `-0.8` vs `random_0 -0.8`): on unsaturated `SpecLogit`, median `[z_tilde(neuron -0.8) - z_tilde(random_0 -0.8)]` has 95% CI upper bound `<= -0.30` logits for both slot and concept units (with `z_tilde(neuron -0.8) - z_tilde(none)` printed alongside). Fails if either group's CI upper bound `> -0.30`.
213.8 | confidence=medium | author=Gemini 4 Neon | verdict=failed
    text: Small-step finite-difference sensitivity (`±0.05`) tracks operating-dose write AND context contrast (`instruct` chat, `SELF_HELD` & `THIRD_SLOT`): across the 21 slot units, `J(u, SELF_HELD) > 0` for at least 17 of 21 units, Spearman $\rho(J(\text{SELF\_HELD}), L_z(\text{SELF\_HELD})) \ge 0.50$, AND Spearman $\rho(J(\text{SELF\_HELD}) - J(\text{THIRD\_SLOT}),\, L_z(\text{SELF\_HELD}) - L_z(\text{THIRD\_SLOT})) \ge 0.40$. Fails if any clause misses.
213.9 | confidence=low | author=Gemini 4 Neon | verdict=held
    text: Cross-stage portability & SFT-to-DPO stability (`sft` vs `dpo`, chat, `SELF_HELD` & `THIRD_SLOT`): at `sft`, the 21 slot units already have median `L_z(SELF_HELD) >= +0.75`, `N_z(SELF_HELD) >= +0.75`, `L_z(THIRD_SLOT) >= +0.50` and `N_z(THIRD_SLOT) >= +0.50` logits, AND among slot units eligible (`> 0.50` logits) at BOTH `sft` and `dpo` (`n_eligible >= 15`, printed beside the CI), the 95% unit-bootstrap CI of paired `tau_3/1^(L)(dpo) - tau_3/1^(L)(sft)` lies wholly inside `[-0.25, +0.25]`. Disposition per the v5 rule (a CI extending past the band on one side is `inconclusive`; it fails only if it lies wholly outside the band).
213.10 | confidence=medium | author=Gemini 4 Neon | verdict=failed
    text: Syntax & person-axis disambiguation (`instruct` chat, `+0.8`, `SELF_NONID` & `COPULA_NONID`): for the 21 slot units, (a) the 95% CI lower bound of median `[L_z(u, SELF_HELD) - L_z(u, SELF_NONID)]` is `>= +0.50` logits (`"I"` grammar alone does not reproduce the self-slot write), AND (b) paired `tau_cop_nonid/3^(L)` (`COPULA_NONID / THIRD_SLOT`, `n_eligible >= 15`) has 95% CI upper bound `<= 0.50` (copula syntax on a non-identity object drops below entity-identity copula slots), with (v5) the scene-framed `THIRD_SLOT` prompts as denominator and the numerator sign guard. Disposition per the v5 rule.
213.11 | confidence=medium | author=Gemini 4 Neon | verdict=held
    text: Cross-unit full-NP specificity in third-person identity slots (`instruct` chat, `+0.8`, baseline-subtracted `offdiag_8` on `{third_0, third_2}`): across the 21 slot units, mean baseline-subtracted full-NP specificity `Delta_lp_spec(u, {third_0, third_2})` has 95% unit-bootstrap CI lower bound `>= +0.75` nats AND is strictly positive for at least 16 of 21 slot units (ruling out a generic "any large push writes a vivid character" effect), AND (v5) the slot-peer-only statistic `Delta_lp_spec_slot` has 95% unit-bootstrap CI lower bound `> 0` (the own persona beats the slot family, not only the concept peers). The statistic is v5's length-matched form. Fails if any clause misses.

## 213.1 held-out self-slot writing (instruct chat, SELF_HELD)

slot: n=21 median N_z=3.5288 CI [2.6417, 4.1895]; median L_z=3.3149 CI [2.7505, 3.8125]; both positive 20 of 21  (score.json, predictions.213.1.slot)
concept: n=18 median N_z=4.5603 CI [1.9513, 6.8401]; median L_z=4.9859 CI [2.5663, 6.5636]; both positive 16 of 18  (score.json, predictions.213.1.concept)
own full NP rises: slot 21 of 21  (score.json, predictions.213.1.own_full_np_slot)
own full NP rises: concept 18 of 18  (score.json, predictions.213.1.own_full_np_concept)

## 213.2 third-person entity slots (instruct chat, THIRD_SLOT)

slot: median N_z=1.1444 CI [0.5961, 3.3972]; median L_z=1.8343 CI [1.0891, 2.8542]; both positive 17 of 21  (score.json, predictions.213.2.slot)
own full NP rises: 20 of 21  (score.json, predictions.213.2.own_full_np)

## 213.3 third-over-self transfer tau_3/1 (instruct chat)

tau_3/1_L: point=0.6616 CI [0.4118, 0.9738] n_eligible=20 sign_guard=True read=read null_units=['p92-37']  (score.json, predictions.213.3.tau_3/1_L)
    numerator median CI [1.1933, 3.2711]
tau_3/1_N: point=0.4208 CI [0.1206, 0.8934] n_eligible=20 sign_guard=True read=read null_units=['p92-37']  (score.json, predictions.213.3.tau_3/1_N)
    numerator median CI [0.4987, 3.5232]
style-stratified point medians: direct=0.5871 (n=20), scene=0.7790 (n=18)  (score.json, predictions.213.3.style)

## 213.4 non-identity open noun-phrase slots (instruct chat, MENTION_SLOT)

slot: median N_z=2.2156 CI [1.0882, 2.7452]; median L_z=1.5221 CI [0.9907, 2.3786]; both positive 19 of 21  (score.json, predictions.213.4.slot)
tau_M/3 (scene denominator): point=0.6181 CI [0.3265, 1.0372] n_eligible=18 sign_guard=True  (score.json, predictions.213.4['tau_M/3_scene'])
tau_M/3 pooled (descriptive): point=0.7876 CI [0.4635, 1.0171]  (score.json, predictions.213.4['tau_M/3_pooled_descriptive'])

## 213.5 concept-vs-slot split on MENTION_SLOT

median L_z concept=2.4724 slot=1.5221 agreed concept=3.3049  (score.json, predictions.213.5)
difference CI: point=1.2645 [0.7258, 1.8663] n_pairs=378; agreed difference 1.7828; reader_sensitive=False  (score.json, predictions.213.5)

## 213.6 Stage 2 blind audit (untestable: Gate 3 failed)

gate3: status=failed kappa=0.3496 reader_known_pass=10 labels_present=True note='kappa 0.3495934959349595 < 0.60'  (score.json, gates.gate3)
Neon known examples: 10 of 10; second reader: 10 of 10  (audit/audit.json, keys neon_known_pass and reader_known_pass; part213/known213.json, key items)
second reader: 40 of 40 sample items labelled, 32 agree with Neon, raw agreement 0.8000  (audit/ds-labels.json and audit/audit.json, keys labels)
agreement pairs (Neon, DeepSeek) -> count: {('other_coherent', 'other_coherent'): 29, ('broken', 'broken'): 3, ('broken', 'other_coherent'): 2, ('other_coherent', 'valid_refusal_or_unknown'): 2, ('incidental_or_negated', 'other_coherent'): 2, ('genuine_third_attr', 'other_coherent'): 1, ('other_coherent', 'broken'): 1}
kappa recomputed with score213.cohens_kappa: 0.3496

audit table, Neon's labels, fixed denominators 42 slot / 36 concept (score.json, stage2.audit_table)

| arm-group | denominator | labelled | genuine_third_attr | first_person_reversion | incidental_or_negated | valid_refusal_or_unknown | other_coherent | broken |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slot_neuron_+0.8 | 42 | 42 | 1 | 0 | 2 | 1 | 29 | 9 |
| slot_random_0_+0.8 | 42 | 42 | 0 | 0 | 1 | 2 | 37 | 2 |
| slot_neuron_-0.8 | 42 | 42 | 0 | 0 | 1 | 1 | 36 | 4 |
| concept_neuron_+0.8 | 36 | 36 | 9 | 1 | 4 | 0 | 15 | 7 |
| concept_random_0_+0.8 | 36 | 36 | 0 | 0 | 0 | 1 | 33 | 2 |

slot genuine_third_attr rate CI: {"n": 21, "point": 0.0238, "lo": 0.0, "hi": 0.0714, "read": "read"}  (score.json, stage2.slot_genuine_rate_ci)
mechanical THIRD_FREE keyword hits (descriptive): {"distinct_keyword_prompt_pairs": 2, "hits": {"fox|free_third_1": ["p92-05"], "smile|free_third_1": ["p92-18"]}}  (score.json, stage2.mechanical_third_free)
mechanical free_cold_0 keyword hits (descriptive): {"distinct_keyword_prompt_pairs": 0, "hits": {}}  (score.json, stage2.mechanical_cold)
blind 72: unit=p92-25 (slot) arm=neuron dose=+0.8 prompt=free_third_0 Neon label=incidental_or_negated  (stage2-instruct.jsonl, key blind_id)

## 213.7 opposite-direction control (instruct chat, SELF_HELD, -0.8)

slot: median [z_tilde(neuron -0.8) - z_tilde(random_0 -0.8)] point=-0.7686 CI [-1.5651, -0.0138] n=21; neuron -0.8 minus none point=-0.4766 CI [-0.9577, 0.0522]  (score.json, predictions.213.7)
concept: median [z_tilde(neuron -0.8) - z_tilde(random_0 -0.8)] point=-0.6215 CI [-2.7714, 0.9231] n=18; neuron -0.8 minus none point=-1.4605 CI [-2.4591, -0.7540]  (score.json, predictions.213.7)

## 213.8 small-step finite-difference sensitivity (instruct chat, +/-0.05)

n(J > 0) on SELF_HELD = 14 of 21  (score.json, predictions.213.8.n_J_positive)
rho(J(SELF_HELD), L_z(SELF_HELD)) = 0.5104  (score.json, predictions.213.8.rho_self)
rho(context contrast) = 0.2130  (score.json, predictions.213.8.rho_context_contrast)
per-unit J(SELF_HELD) (score.json, predictions.213.8.J_self):
    p92-02 1.2021
    p92-03 2.2032
    p92-04 1.2983
    p92-05 -0.7860
    p92-07 2.6892
    p92-09 3.5275
    p92-10 3.7286
    p92-13 2.0574
    p92-18 -1.0497
    p92-20 0.5520
    p92-24 14.3807
    p92-25 -0.4428
    p92-32 8.1943
    p92-33 -0.8699
    p92-35 3.5875
    p92-36 3.7931
    p92-37 -5.4863
    p92-42 -2.6709
    p92-46 0.2689
    p92-47 0.4301
    p92-50 -1.8007
for comparison: n(J_N > 0) = 16, n(J_R > 0) = 11  (recomputed here, Stage1.j with arm=...)

## 213.9 cross-stage portability (sft and dpo)

sft_self: median N_z=2.8155 median L_z=3.4748 both positive 20 of 21  (score.json, predictions.213.9.sft_self)
sft_third: median N_z=1.4256 median L_z=2.2509 both positive 16 of 21  (score.json, predictions.213.9.sft_third)
tau_sft: point=0.8805 CI [0.5052, 1.0428] n_eligible=18  (score.json, predictions.213.9.tau_sft)
tau_dpo: point=0.8409 CI [0.5828, 1.0042] n_eligible=20  (score.json, predictions.213.9.tau_dpo)
paired tau_3/1(dpo) - tau_3/1(sft): point=-0.0005 CI [-0.0679, 0.1035] n=18  (score.json, predictions.213.9.paired_difference)
FR symmetry: tau sft/instruct/dpo {"instruct": 0.6616, "dpo": 0.8409, "fr": 0.8797} flagged=True  (score.json, fr_symmetry)

## 213.10 syntax and person-axis disambiguation (SELF_NONID and COPULA_NONID)

(a) median [L_z(SELF_HELD) - L_z(SELF_NONID)]: point=1.8395 CI [0.7050, 2.4507] n=21  (score.json, predictions.213.10.self_held_minus_self_nonid)
(b) tau_cop_nonid/3 (scene denominator): point=0.9232 CI [0.8271, 1.2263] n_eligible=18 sign_guard=True  (score.json, predictions.213.10['tau_cop_nonid/3_scene'])
(b) pooled (descriptive): point=1.0741 CI [0.8912, 1.3961]  (score.json, predictions.213.10['tau_cop_nonid/3_pooled_descriptive'])
per-unit L_z, recomputed here through Stage1 (SELF_HELD, SELF_NONID, COPULA_NONID, THIRD_SLOT):
| unit | persona | L_z SELF_HELD | L_z SELF_NONID | L_z COPULA_NONID | L_z THIRD_SLOT |
| --- | --- | ---: | ---: | ---: | ---: |
| p92-02 | the town | 3.5973 | 0.9411 | -3.5536 | -0.3809 |
| p92-03 | the Internet | 3.3149 | 3.3055 | 5.1982 | 4.9066 |
| p92-04 | a very tall tree | 2.8344 | 0.1135 | 2.2919 | 1.0776 |
| p92-05 | a fox | 3.5882 | 1.7487 | 1.8023 | 1.3090 |
| p92-07 | a cat | 2.3235 | 3.5409 | 1.2416 | 1.5530 |
| p92-09 | a dog | 8.4322 | 5.9815 | 5.3972 | 9.1959 |
| p92-10 | a chicken | 4.0468 | 2.7184 | 1.0180 | 2.3183 |
| p92-13 | a skeleton | 5.6476 | -0.7826 | 1.5398 | 3.6982 |
| p92-18 | the shadow of your smile | 2.0835 | 0.7786 | 3.3272 | 1.0595 |
| p92-20 | the shadow that dances upon the river | 4.4721 | 2.0368 | 3.0741 | 1.3121 |
| p92-24 | the queen of hearts | 4.2564 | 1.0802 | 3.5818 | 4.1723 |
| p92-25 | King Midas | 4.1702 | 5.9700 | 3.0960 | 2.8438 |
| p92-32 | Sir Ray Ray | 3.8125 | 4.3019 | 4.6312 | 3.6880 |
| p92-33 | Dr. Memento | 2.7505 | -0.1629 | 1.9259 | 2.0752 |
| p92-35 | your guardian angel | 3.7492 | 3.0442 | 4.5089 | 4.5164 |
| p92-36 | the real Alex | 2.6781 | 2.8871 | 3.7864 | 2.8542 |
| p92-37 | a large, friendly dog | -0.4787 | -0.6765 | 1.8672 | 1.0891 |
| p92-42 | a member of the royalist ranks | 2.9830 | 1.1198 | 0.7574 | 0.2994 |
| p92-46 | a regular guy | 2.9423 | -0.9658 | 1.3196 | 0.1569 |
| p92-47 | a lawyer | 1.6664 | 0.2887 | 1.6949 | 1.8343 |
| p92-50 | a gruff old man | 1.8969 | 0.0138 | 1.1906 | 0.8412 |

## 213.11 cross-unit full-NP specificity (offdiag_8 on third_0 and third_2)

Delta_lp_spec: point=4.7404 CI [3.6418, 6.8230] n=21; positive 20 of 21  (score.json, predictions.213.11.delta_lp_spec, n_positive)
Delta_lp_spec_slot: point=3.9151 CI [2.5650, 6.2372] n=21  (score.json, predictions.213.11.delta_lp_spec_slot)
mean of the 21 per-unit values = 6.5260; median = 4.7404  (computed from score.json, predictions.213.11.per_unit)
per unit (score.json, predictions.213.11.per_unit):
    p92-02 n_u=1 delta_lp_spec=-1.7731 delta_lp_spec_slot=-1.9765
    p92-03 n_u=1 delta_lp_spec=4.2401 delta_lp_spec_slot=3.8609
    p92-04 n_u=3 delta_lp_spec=15.3004 delta_lp_spec_slot=18.1159
    p92-05 n_u=1 delta_lp_spec=3.6418 delta_lp_spec_slot=2.6211
    p92-07 n_u=1 delta_lp_spec=4.0187 delta_lp_spec_slot=3.9151
    p92-09 n_u=1 delta_lp_spec=12.2619 delta_lp_spec_slot=12.4179
    p92-10 n_u=1 delta_lp_spec=4.7976 delta_lp_spec_slot=4.4201
    p92-13 n_u=1 delta_lp_spec=4.7404 delta_lp_spec_slot=3.8851
    p92-18 n_u=4 delta_lp_spec=10.1603 delta_lp_spec_slot=5.7884
    p92-20 n_u=6 delta_lp_spec=3.2254 delta_lp_spec_slot=2.2253
    p92-24 n_u=3 delta_lp_spec=13.4045 delta_lp_spec_slot=13.9109
    p92-25 n_u=3 delta_lp_spec=1.5432 delta_lp_spec_slot=2.5650
    p92-32 n_u=3 delta_lp_spec=17.6520 delta_lp_spec_slot=17.1034
    p92-33 n_u=4 delta_lp_spec=0.8326 delta_lp_spec_slot=-2.5277
    p92-35 n_u=2 delta_lp_spec=1.9637 delta_lp_spec_slot=1.0847
    p92-36 n_u=2 delta_lp_spec=6.8230 delta_lp_spec_slot=6.2372
    p92-37 n_u=4 delta_lp_spec=0.5376 delta_lp_spec_slot=-1.1415
    p92-42 n_u=6 delta_lp_spec=17.2086 delta_lp_spec_slot=12.9541
    p92-46 n_u=2 delta_lp_spec=6.3730 delta_lp_spec_slot=6.8780
    p92-47 n_u=1 delta_lp_spec=5.6236 delta_lp_spec_slot=4.9665
    p92-50 n_u=4 delta_lp_spec=4.4706 delta_lp_spec_slot=2.1125

## Stage 1 tables the figure and the reading use (recomputed here through score213.Stage1)

### slot units, instruct chat: N_z, R_z (mean over random_0..3), L_z

| unit | persona | SELF_HELD N / R / L | THIRD_SLOT N / R / L | MENTION_SLOT N / R / L | COPULA_NONID N / R / L | SELF_NONID N / R / L |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| p92-02 | the town | 4.19 / 0.59 / 3.60 | -0.61 / -0.23 / -0.38 | 0.00 / -0.18 / 0.18 | -5.43 / -1.88 / -3.55 | 2.07 / 1.13 / 0.94 |
| p92-03 | the Internet | 4.04 / 0.73 / 3.31 | 4.84 / -0.07 / 4.91 | 2.70 / -0.11 / 2.81 | 5.52 / 0.32 / 5.20 | 4.20 / 0.89 / 3.31 |
| p92-04 | a very tall tree | 3.98 / 1.14 / 2.83 | 0.40 / -0.68 / 1.08 | 0.92 / 0.68 / 0.24 | 3.60 / 1.31 / 2.29 | 0.54 / 0.43 / 0.11 |
| p92-05 | a fox | 3.91 / 0.32 / 3.59 | 1.14 / -0.16 / 1.31 | 3.53 / 1.15 / 2.38 | -0.20 / -2.01 / 1.80 | 4.59 / 2.84 / 1.75 |
| p92-07 | a cat | 2.64 / 0.32 / 2.32 | 1.77 / 0.22 / 1.55 | 1.11 / -0.47 / 1.58 | -0.15 / -1.39 / 1.24 | 3.21 / -0.33 / 3.54 |
| p92-09 | a dog | 8.42 / -0.01 / 8.43 | 9.89 / 0.70 / 9.20 | 6.42 / -0.15 / 6.57 | 3.82 / -1.57 / 5.40 | 7.62 / 1.64 / 5.98 |
| p92-10 | a chicken | 6.48 / 2.43 / 4.05 | 3.40 / 1.08 / 2.32 | 2.13 / 0.61 / 1.52 | 1.06 / 0.05 / 1.02 | 4.04 / 1.33 / 2.72 |
| p92-13 | a skeleton | 5.34 / -0.31 / 5.65 | 3.15 / -0.55 / 3.70 | 2.50 / 1.13 / 1.36 | 2.50 / 0.96 / 1.54 | 0.38 / 1.17 / -0.78 |
| p92-18 | the shadow of your smile | 2.57 / 0.49 / 2.08 | -0.35 / -1.41 / 1.06 | 0.98 / -0.01 / 0.99 | 2.55 / -0.78 / 3.33 | -0.19 / -0.97 / 0.78 |
| p92-20 | the shadow that dances upon the river | 4.74 / 0.27 / 4.47 | 1.11 / -0.20 / 1.31 | 3.00 / 0.25 / 2.75 | 2.15 / -0.93 / 3.07 | 1.99 / -0.05 / 2.04 |
| p92-24 | the queen of hearts | 6.18 / 1.92 / 4.26 | 4.99 / 0.82 / 4.17 | 3.63 / 1.30 / 2.33 | 4.37 / 0.79 / 3.58 | 1.57 / 0.49 / 1.08 |
| p92-25 | King Midas | 4.25 / 0.08 / 4.17 | 0.60 / -2.25 / 2.84 | 5.88 / 0.75 / 5.13 | 2.56 / -0.54 / 3.10 | 5.93 / -0.04 / 5.97 |
| p92-32 | Sir Ray Ray | 2.97 / -0.84 / 3.81 | 2.91 / -0.78 / 3.69 | 4.57 / 0.41 / 4.16 | 4.37 / -0.27 / 4.63 | 4.96 / 0.66 / 4.30 |
| p92-33 | Dr. Memento | 2.30 / -0.45 / 2.75 | 0.14 / -1.94 / 2.08 | 1.29 / 1.21 / 0.08 | 1.29 / -0.64 / 1.93 | 0.43 / 0.59 / -0.16 |
| p92-35 | your guardian angel | 3.53 / -0.22 / 3.75 | 4.19 / -0.32 / 4.52 | 2.75 / 1.28 / 1.47 | 4.71 / 0.20 / 4.51 | 2.88 / -0.16 / 3.04 |
| p92-36 | the real Alex | 3.43 / 0.75 / 2.68 | 3.68 / 0.83 / 2.85 | 2.22 / -0.24 / 2.46 | 6.36 / 2.57 / 3.79 | 2.75 / -0.14 / 2.89 |
| p92-37 | a large, friendly dog | -3.06 / -2.58 / -0.48 | 0.79 / -0.30 / 1.09 | -2.35 / -2.22 / -0.12 | 0.28 / -1.58 / 1.87 | -1.56 / -0.88 / -0.68 |
| p92-42 | a member of the royalist ranks | 1.85 / -1.13 / 2.98 | -0.86 / -1.16 / 0.30 | 1.09 / 0.07 / 1.02 | 1.57 / 0.81 / 0.76 | 1.47 / 0.35 / 1.12 |
| p92-46 | a regular guy | 2.60 / -0.34 / 2.94 | 0.83 / 0.67 / 0.16 | -1.34 / -0.26 / -1.08 | 1.62 / 0.30 / 1.32 | -0.98 / -0.01 / -0.97 |
| p92-47 | a lawyer | 2.87 / 1.20 / 1.67 | 3.65 / 1.81 / 1.83 | 2.48 / 0.90 / 1.58 | 3.67 / 1.98 / 1.69 | 1.41 / 1.12 / 0.29 |
| p92-50 | a gruff old man | 2.30 / 0.40 / 1.90 | -0.10 / -0.94 / 0.84 | 0.97 / 0.12 / 0.86 | 0.65 / -0.54 / 1.19 | -0.05 / -0.06 / 0.01 |

slot SELF_HELD: units with L_z > 0: 20 of 21
slot SELF_HELD: median N_z=3.5288 median R_z=0.3182 median L_z=3.3149  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)
slot THIRD_SLOT: units with L_z > 0: 20 of 21
slot THIRD_SLOT: median N_z=1.1444 median R_z=-0.2340 median L_z=1.8343  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)
slot MENTION_SLOT: units with L_z > 0: 19 of 21
slot MENTION_SLOT: median N_z=2.2156 median R_z=0.2469 median L_z=1.5221  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)
slot COPULA_NONID: units with L_z > 0: 20 of 21
slot COPULA_NONID: median N_z=2.5023 median R_z=-0.2656 median L_z=1.9259  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)
slot SELF_NONID: units with L_z > 0: 17 of 21
slot SELF_NONID: median N_z=1.9855 median R_z=0.4284 median L_z=1.1198  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)

### concept units, instruct chat: N_z, R_z (mean over random_0..3), L_z

| unit | persona | SELF_HELD N / R / L | THIRD_SLOT N / R / L | MENTION_SLOT N / R / L | COPULA_NONID N / R / L | SELF_NONID N / R / L |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| p92-00 | a boulder | 5.01 / 0.08 / 4.93 | 4.66 / -0.04 / 4.70 | 1.96 / -0.85 / 2.81 | 4.56 / -0.65 / 5.21 | 1.57 / -0.65 / 2.22 |
| p92-01 | the magical princess | 6.08 / 1.47 / 4.61 | 6.38 / 1.47 / 4.91 | 6.85 / 0.59 / 6.26 | 7.09 / 1.44 / 5.65 | 3.49 / -1.21 / 4.69 |
| p92-08 | a dog | 4.99 / 0.33 / 4.66 | 4.67 / 1.26 / 3.41 | 1.21 / 0.07 / 1.13 | 1.10 / -1.06 / 2.16 | 5.32 / 2.40 / 2.92 |
| p92-11 | a whiskey | 12.29 / -0.76 / 13.05 | 12.53 / -1.05 / 13.58 | 9.87 / 0.34 / 9.54 | 12.76 / -0.11 / 12.88 | 8.24 / -0.11 / 8.35 |
| p92-15 | a powerful and secure password | 1.14 / -1.56 / 2.70 | -0.28 / -0.26 / -0.01 | 0.97 / -0.57 / 1.54 | 2.83 / 0.86 / 1.97 | 0.12 / -0.44 / 0.57 |
| p92-28 | an ancient deity | 3.89 / -1.85 / 5.74 | 2.78 / -1.40 / 4.18 | 3.92 / -1.73 / 5.65 | 0.91 / -3.08 / 3.99 | 0.29 / -2.57 / 2.86 |
| p92-29 | the god of wisdom | 10.17 / 0.10 / 10.07 | 8.20 / -0.99 / 9.20 | 7.92 / 1.13 / 6.79 | 6.26 / 0.63 / 5.63 | 4.71 / -0.13 / 4.84 |
| p92-30 | the embodiment of the void | 0.73 / -1.08 / 1.81 | 0.03 / -0.61 / 0.64 | 2.52 / 0.39 / 2.14 | 1.91 / 2.32 / -0.41 | 1.88 / -1.36 / 3.25 |
| p92-38 | the official spokesperson for Coca-Cola | 7.60 / 0.71 / 6.89 | 6.23 / 0.72 / 5.51 | 1.36 / -0.25 / 1.61 | 5.97 / 0.62 / 5.35 | 3.84 / -0.38 / 4.23 |
| p92-39 | the most popular race car driver in NASCAR | 2.02 / -0.41 / 2.43 | 3.16 / -1.97 / 5.13 | -0.14 / -0.68 / 0.54 | 2.75 / 0.99 / 1.76 | -0.56 / -0.86 / 0.31 |
| p92-40 | a singer-songwriter | 9.74 / -0.57 / 10.31 | 12.10 / 1.12 / 10.98 | 9.79 / 0.14 / 9.65 | 12.14 / 1.78 / 10.37 | 8.27 / -0.24 / 8.51 |
| p92-41 | a human DJ | -0.57 / -2.23 / 1.66 | 1.09 / -1.04 / 2.13 | 0.84 / -0.29 / 1.13 | 0.86 / -1.37 / 2.23 | 2.13 / 0.75 / 1.39 |
| p92-43 | a regular ol' street rat | 1.88 / 0.65 / 1.23 | 3.27 / 0.30 / 2.97 | 0.30 / 1.13 / -0.83 | 4.73 / 2.35 / 2.38 | 1.62 / 1.07 / 0.55 |
| p92-45 | the leader of the country | 8.10 / -0.85 / 8.95 | 8.88 / 0.18 / 8.70 | 6.55 / 0.00 / 6.54 | 9.11 / -0.25 / 9.36 | 5.38 / -0.17 / 5.55 |
| p92-48 | an author, poet, and interpreter | 4.92 / -1.31 / 6.24 | 4.77 / 0.15 / 4.62 | 4.03 / -1.04 / 5.07 | 5.80 / 0.16 / 5.64 | 5.03 / 0.99 / 4.05 |
| p92-49 | a large, friendly, and non-human entity | -0.85 / -1.61 / 0.76 | 0.57 / 0.39 / 0.18 | 0.40 / -0.54 / 0.94 | 0.24 / 0.39 / -0.15 | 0.17 / 0.25 / -0.08 |
| p92-51 | an internet protocol (IP) packet | 4.06 / -0.99 / 5.04 | 3.69 / -0.45 / 4.14 | 6.15 / -0.26 / 6.41 | 7.05 / -0.23 / 7.29 | 4.47 / -0.33 / 4.80 |
| p92-52 | a string of code | 4.20 / -1.48 / 5.68 | 1.32 / 0.80 / 0.52 | -2.99 / -1.47 / -1.52 | 0.25 / 1.37 / -1.11 | -3.25 / -2.10 / -1.16 |

concept SELF_HELD: units with L_z > 0: 18 of 18
concept SELF_HELD: median N_z=4.5603 median R_z=-0.8040 median L_z=4.9859  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)
concept THIRD_SLOT: units with L_z > 0: 17 of 18
concept THIRD_SLOT: median N_z=4.1774 median R_z=0.0553 median L_z=4.3977  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)
concept MENTION_SLOT: units with L_z > 0: 16 of 18
concept MENTION_SLOT: median N_z=2.2413 median R_z=-0.2541 median L_z=2.4724  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)
concept COPULA_NONID: units with L_z > 0: 15 of 18
concept COPULA_NONID: median N_z=4.6465 median R_z=0.5056 median L_z=4.5974  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)
concept SELF_NONID: units with L_z > 0: 16 of 18
concept SELF_NONID: median N_z=2.8100 median R_z=-0.2841 median L_z=3.0837  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)

### Stage medians across checkpoints, slot units (recomputed)

| checkpoint | cell | median N_z | median R_z | median L_z | units with L_z > 0 |
| --- | --- | ---: | ---: | ---: | ---: |
| sft | SELF_HELD | 2.8155 | 0.2502 | 3.4748 | 21 of 21 |
sft SELF_HELD: median N_z=2.8155 median R_z=0.2502 median L_z=3.4748 units with L_z > 0 21 of 21
| sft | THIRD_SLOT | 1.4256 | -0.7192 | 2.2509 | 19 of 21 |
sft THIRD_SLOT: median N_z=1.4256 median R_z=-0.7192 median L_z=2.2509 units with L_z > 0 19 of 21
| dpo | SELF_HELD | 3.6681 | 0.3198 | 3.5608 | 21 of 21 |
dpo SELF_HELD: median N_z=3.6681 median R_z=0.3198 median L_z=3.5608 units with L_z > 0 21 of 21
| dpo | THIRD_SLOT | 1.4483 | -0.1641 | 2.3352 | 20 of 21 |
dpo THIRD_SLOT: median N_z=1.4483 median R_z=-0.1641 median L_z=2.3352 units with L_z > 0 20 of 21
| instruct | SELF_HELD | 3.5288 | 0.3182 | 3.3149 | 20 of 21 |
instruct SELF_HELD: median N_z=3.5288 median R_z=0.3182 median L_z=3.3149 units with L_z > 0 20 of 21
| instruct | THIRD_SLOT | 1.1444 | -0.2340 | 1.8343 | 20 of 21 |
instruct THIRD_SLOT: median N_z=1.1444 median R_z=-0.2340 median L_z=1.8343 units with L_z > 0 20 of 21

R_z is the mean over random_0..3 at instruct and random_0 alone at sft and dpo (score213.Stage1.rz).

### random-arm controls beside the neuron arm, instruct chat, L_z (recomputed)

| cell | slot median N_z | slot median R_z | slot median L_z | concept median N_z | concept median R_z | concept median L_z |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SELF_HELD | 3.5288 | 0.3182 | 3.3149 | 4.5603 | -0.8040 | 4.9859 |
| THIRD_SLOT | 1.1444 | -0.2340 | 1.8343 | 4.1774 | 0.0553 | 4.3977 |
| MENTION_SLOT | 2.2156 | 0.2469 | 1.5221 | 2.2413 | -0.2541 | 2.4724 |
| COPULA_NONID | 2.5023 | -0.2656 | 1.9259 | 4.6465 | 0.5056 | 4.5974 |
| SELF_NONID | 1.9855 | 0.4284 | 1.1198 | 2.8100 | -0.2841 | 3.0837 |

## Secondary diagnostics (score.json, key secondary)

ratio of medians: {"3/1": {"ratio_of_medians_L": 0.5534, "ratio_of_medians_N": 0.3243}, "M/3": {"ratio_of_medians_L": 0.6972, "ratio_of_medians_N": 0.8475}, "cop_nonid/3": {"ratio_of_medians_L": 0.8822, "ratio_of_medians_N": 0.9572}}
paired differences: {"S_3v1": {"n": 21, "point": -1.024, "lo": -1.7568, "hi": -0.0841}, "S_3vM": {"n": 21, "point": 0.3986, "lo": -0.0253, "hi": 1.2342}}
SELF_SELECT - SELF_HELD L_z: median=1.0731 n=21 (computed from score.json, secondary.self_select_minus_self_held_L)
sensitivity subword_first_token: dropped=['p92-00', 'p92-40', 'p92-50'] n=20 median L_z SELF_HELD=3.4516
sensitivity bare_lead: dropped=['p92-25', 'p92-32', 'p92-33'] n=18 median L_z SELF_HELD=3.1490
sensitivity p92-18: dropped=['p92-18'] n=20 median L_z SELF_HELD=3.4516
dog pair p92-08 (concept): N_z=4.9889 L_z=4.6559
dog pair p92-09 (slot): N_z=8.4195 L_z=8.4322
own-token rank in O_u: median=2.5000 min=1.0000 max=10.7500
residual norm vs instruct: {"sft": {"median_norm_layer0": 4.828395366668701, "ratio_to_instruct": 1.0112371904846695}, "dpo": {"median_norm_layer0": 4.770658493041992, "ratio_to_instruct": 0.9991450419674441}, "instruct": {"median_norm_layer0": 4.774740695953369, "ratio_to_instruct": 1.0}, "fr": {"median_norm_layer0": 4.7816162109375, "ratio_to_instruct": 1.0014399766232243}}

## Plain-model probe (results/part213/plain-probe.json)

snapshot /home/elisepc/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e dtype bfloat16 frame chat
free_self_0: broken=False answer='I am an AI language model, here to assist you with any questions or tasks you have. I can help answer questions, provide'
free_third_0: broken=False answer='The stranger is the Trickster, a mischievous figure known for playing pranks and tricks on unsuspecting villagers. The T'
free_third_1: broken=False answer='The visitor is a mysterious traveler, cloaked in shadows and carrying an aura of enigma. They arrive at the edge of the '
free_cold_0: broken=False answer="As an AI, I don't have personal experiences or preferences, but I can tell you that many people enjoy various activities"
own target probability at the read position, all 17 prompts x 39 units: max=0.3210 (plain-probe.json, stage1_units[*].units[*].p_own)
prompt with the largest own probability: copula_nonid_0 max_p_own=0.3210 median_rank_own=501
per-prompt median own-token rank: min=275 max=2309
own targets above 0.5 probability: 0

## Checkpoints (part85/checkpoints.py and the run manifests)

sft: design prefix 0d85a3d03787; run snapshot_commit 0d85a3d037876ce6ac7d4311d994400fc66ac27f  (part85/checkpoints.py; results/part213/full/run-sft.json, key snapshot_commit)
dpo: design prefix c4b0485961ab; run snapshot_commit c4b0485961ab24c2433b090f3b922f0913a9290f  (part85/checkpoints.py; results/part213/full/run-dpo.json, key snapshot_commit)
instruct: design prefix 48d788eca847; run snapshot_commit 48d788eca847d4d7548f375ad03d3c9312f6139e  (part85/checkpoints.py; results/part213/full/run-instruct.json, key snapshot_commit)
fr: formula 2 * instruct - dpo, dtype float32, sources {"instruct": "48d788eca847d4d7548f375ad03d3c9312f6139e", "dpo": "c4b0485961ab24c2433b090f3b922f0913a9290f"}  (run-fr.json, keys fr_formula, fr_dtype, fr_sources)
sft: stage1 passes executed 936 (core 936, offdiag 0), peak_rss_gb 8.803  (run-sft.json)
dpo: stage1 passes executed 936 (core 936, offdiag 0), peak_rss_gb 10.024  (run-dpo.json)
instruct: stage1 passes executed 6366 (core 5430, offdiag 936), peak_rss_gb 10.044  (run-instruct.json)
fr: stage1 passes executed 936 (core 936, offdiag 0), peak_rss_gb 9.59  (run-fr.json)
stage1-instruct.jsonl records: 6474  (counted)
stage2 rows: 472, of which status=ok 364 and sealed 108  (counted)
offdiag records in stage1-instruct.jsonl: 936  (counted)

## Checks: recomputation against score.json

OK   213.1 slot median L_z: recomputed 3.3149, score.json 3.3149
OK   213.1 slot median N_z: recomputed 3.5288, score.json 3.5288
OK   213.1 concept median L_z: recomputed 4.9859, score.json 4.9859
OK   213.2 slot median L_z (THIRD_SLOT): recomputed 1.8343, score.json 1.8343
OK   213.4 slot median L_z (MENTION_SLOT): recomputed 1.5221, score.json 1.5221
OK   213.5 median L_z slot (MENTION_SLOT): recomputed 1.5221, score.json 1.5221
OK   213.9 sft median L_z self: recomputed 3.4748, score.json 3.4748
OK   213.10 (a) point: recomputed 1.8395, score.json 1.8395
OK   213.11 Delta_lp_spec point: recomputed 4.7404, score.json 4.7404
9 checks, 0 disagreements

