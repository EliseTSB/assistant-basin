# Part 117 number log

The counts below are recomputed from `answers.jsonl`, `items.jsonl`, `labels.jsonl`, `labels_second.jsonl`, `pairs.json`, and `personas.json` by [`plot117.py`](plot117.py). It checks the computed gates, summaries, and verdicts against `score.json` and `agreement.json`.

Run from the repository root:

`python3 -I results/part117/plot117.py --numbers`

The bold-opener rule counts an answer when it begins with `**` after an optional first-person lead-in and article. Qwen mentions are case-insensitive substring matches in question 0 text.

| Cold-check finding | Script output |
| --- | --- |
| Finding 1: matched, alone, unrelated and slot-alone cross-label counts | figure.who.Matched: concept +1 alone=concept 11/11; slot 0/11<br>figure.who.Matched: slot +1 alone=concept 2/10; slot 11/11<br>figure.who.Matched: both +1=concept 9/11; slot 3/11<br>figure.who.Unrelated: both +1=concept 10/11; slot 0/11 |
| Finding 2: bold openers and Qwen mentions on question 0 | cold.naming.bold.matched_concept_plus1=11/11<br>cold.naming.bold.matched_slot_plus1=11/11<br>cold.naming.bold.matched_concept_minus1_slot_plus1=11/11<br>cold.naming.bold.unrelated_concept_plus1=11/11<br>cold.naming.bold.random_both_plus1=10/11<br>cold.naming.qwen.unpushed=1/1<br>cold.naming.qwen.matched_both_half=3/11<br>cold.naming.qwen.random_both_plus1=1/11<br>cold.naming.qwen.zero_in_other_pushed_conditions=9/9 (each 0/11) |
| Finding 3: question 0 counts under the negative slot push | cold.117.4.who_concept_alone_matched=11/11<br>cold.117.4.who_concept_slot_minus_matched=7/11<br>cold.117.4.who_concept_alone_unrelated=11/11<br>cold.117.4.who_concept_slot_minus_unrelated=10/11 |
| Finding 4: gate checks | gate.unpushed_personas_over_2=0/31<br>gate.slot_kind_reproduction=11/11 (registered threshold 9) |
| Finding 5: prompt and theme-score counts | data.questions_per_row=6<br>pairing.theme_score_counts=1:1,2:9,3:1 |
| Finding 6: random withdrawal rule | prediction.117.6.random_persona_labels=1/132<br>prediction.117.6.random_answer_texts_with_persona=1/66<br>prediction.117.6.withdrawal_threshold=6; triggered=False |
| Finding 7: coherence example and second-reader agreement | prediction.117.7.coherent_at_both_plus_2=0/66<br>cold.117.7.coherent_plus2_concept=0/66<br>cold.117.7.coherent_plus2_slot=1/66<br>cold.117.7.coherent_plus2_both_targets=1/132<br>cold.second_reader.coherent_agreement=145/164 (88.4%)<br>second_reader.agreement=156/164 (95.1%)<br>cold.second_reader.shows_disagreements=8/164<br>cold.second_reader.verdicts_after_sample_overlay=8/8 unchanged=True<br>cold.117.7.loop_example=p01:matched:both+2:q0:I'm ** coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously de coordinates continuously de |
| Finding 8: reused matched concepts | cold.limits.shared_matched_concept_count=2<br>cold.limits.shared_matched_concepts=p03,p11:q112-071;p07,p09:q112-401 |
| Finding 10: slot-alone concept label | figure.who.Matched: slot +1 alone=concept 2/10; slot 11/11 |

## Full script output

```text
data.answers=732
data.row_variants=122
data.questions_per_row=6
data.pairs=11
data.slot_units=11
data.matched_concept_units=9
data.unrelated_concept_units=11
data.distinct_concept_units=20
pairing.theme_score_counts=1:1,2:9,3:1
data.label_items=1638
data.persona_targets=273
labels.error_attempts=9
labels.successful_unique_items=1636/1638
labels.unlabelled_items=2
labels.unlabelled_keys=p06:matched:s+1|c#0,p09:unrelated:both+1|c#4
gate.unpushed_personas_over_2=0/31
gate.unpushed_personas_complete=31/31
gate.unpushed_max_shows=1/6
gate.slot_kind_reproduction=11/11 (registered threshold 9)
gate.uninterpretable_units=0/31
prediction.117.1.slot_patterns=11/11
prediction.117.1.concepts_at_least_3=20/20
prediction.117.1.verdict=held
prediction.117.2.met=8/11 evaluable pairs
prediction.117.2.verdict=held
prediction.117.3.concept_who=9/11
prediction.117.3.slot_who=3/11
prediction.117.3.both_personas_who=2/11
prediction.117.3.verdict=failed
prediction.117.4.matched_concept_decreased=6/11
prediction.117.4.matched_delta_histogram=-6:1,-2:2,-1:3,0:4,1:1
prediction.117.4.unrelated_concept_decreased=5/11
prediction.117.4.unrelated_delta_histogram=-2:1,-1:4,0:4,1:2
prediction.117.4.p02_matched_count=6->0
prediction.117.4.verdict=failed
prediction.117.5.neither_capital_nor_poem=4/11
prediction.117.5.capital_any_persona=2/11
prediction.117.5.poem_any_persona=5/11
prediction.117.5.verdict=failed
prediction.117.6.random_persona_labels=1/132
prediction.117.6.random_answer_texts_with_persona=1/66
prediction.117.6.random_positive=p09:s:q3
prediction.117.6.withdrawal_threshold=6; triggered=False
prediction.117.6.verdict=held
prediction.117.7.coherent_at_both_plus_2=0/66
prediction.117.7.verdict=held
prediction.117.8.slot_persona_at_c_minus1_s_plus1=0/11
prediction.117.8.slot_persona_at_slot_plus1_alone=11/11
prediction.117.8.verdict=failed
second_reader.agreement=156/164 (95.1%)
verification.score_json=all checked summaries and verdicts agree
cold.naming.bold_opener_rule=bold marker at answer start after optional first-person lead-in and article
cold.naming.bold.matched_concept_plus1=11/11
cold.naming.bold.matched_slot_plus1=11/11
cold.naming.bold.matched_concept_minus1_slot_plus1=11/11
cold.naming.bold.unrelated_concept_plus1=11/11
cold.naming.bold.random_both_plus1=10/11
cold.naming.qwen.unpushed=1/1
cold.naming.qwen.matched_both_half=3/11
cold.naming.qwen.random_both_plus1=1/11
cold.naming.qwen.zero_in_other_pushed_conditions=9/9 (each 0/11)
cold.117.4.who_concept_alone_matched=11/11
cold.117.4.who_concept_slot_minus_matched=7/11
cold.117.4.who_concept_alone_unrelated=11/11
cold.117.4.who_concept_slot_minus_unrelated=10/11
cold.117.7.coherent_plus2_concept=0/66
cold.117.7.coherent_plus2_slot=1/66
cold.117.7.coherent_plus2_both_targets=1/132
cold.117.7.loop_example=p01:matched:both+2:q0:I'm ** coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously de coordinates continuously de
cold.second_reader.coherent_agreement=145/164 (88.4%)
cold.second_reader.shows_disagreements=8/164
cold.second_reader.verdicts_after_sample_overlay=8/8 unchanged=True
cold.limits.shared_matched_concept_count=2
cold.limits.shared_matched_concepts=p03,p11:q112-071;p07,p09:q112-401
figure.who.Matched: concept +1 alone=concept 11/11; slot 0/11
figure.who.Matched: slot +1 alone=concept 2/10; slot 11/11
figure.who.Matched: both +0.5=concept 7/11; slot 1/11
figure.who.Matched: both +1=concept 9/11; slot 3/11
figure.who.Matched: both +2=concept 5/11; slot 1/11
figure.who.Matched: concept +1, slot -1=concept 7/11; slot 0/11
figure.who.Matched: concept -1, slot +1=concept 0/11; slot 0/11
figure.who.Unrelated: concept +1=concept 11/11; slot 0/11
figure.who.Unrelated: both +1=concept 10/11; slot 0/11
figure.who.Unrelated: concept +1, slot -1=concept 10/11; slot 0/11
figure.who.Random directions: both +1=concept 0/11; slot 0/11
```
