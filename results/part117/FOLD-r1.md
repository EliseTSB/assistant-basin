# Fold log: Part 117 cold check, round 1

All computed values below come from the standard-library run `python3 -I results/part117/plot117.py --numbers`. The output is stored in `results/part117/NUMBERS.md`. The scorer substitution check runs in memory and does not write to the result files.

## Finding 1: Baselines and the joint +1 condition

**Change:** The title and opening result paragraph name the condition as both neurons pushed at +1. They place the matched counts beside each neuron's alone baseline and the unrelated both +1 counts. The prose also reports the 2/11 overlap.

**One-liner:** `python3 -I results/part117/plot117.py --numbers`

**Output:** `figure.who.Matched: both +1=concept 9/11; slot 3/11`; `figure.who.Matched: concept +1 alone=concept 11/11; slot 0/11`; `figure.who.Matched: slot +1 alone=concept 2/10; slot 11/11`; `figure.who.Unrelated: both +1=concept 10/11; slot 0/11`; `prediction.117.3.both_personas_who=2/11`.

## Finding 2: Naming baselines

**Change:** The README separates the exploratory naming pattern from registered prediction 117.8. It gives the bold-opener counts and the case-insensitive Qwen text-search baselines, then says the names may track persona absence rather than concept -1.

**One-liner:** `python3 -I results/part117/plot117.py --numbers`

**Output:** `cold.naming.bold.matched_concept_plus1=11/11`; `cold.naming.bold.matched_slot_plus1=11/11`; `cold.naming.bold.matched_concept_minus1_slot_plus1=11/11`; `cold.naming.bold.unrelated_concept_plus1=11/11`; `cold.naming.bold.random_both_plus1=10/11`; `cold.naming.qwen.unpushed=1/1`; `cold.naming.qwen.matched_both_half=3/11`; `cold.naming.qwen.random_both_plus1=1/11`; `cold.naming.qwen.zero_in_other_pushed_conditions=9/9 (each 0/11)`. The opener rule is mechanical: a bold marker after an optional first-person lead-in and article.

## Finding 3: Question-level negative-slot counts

**Change:** The negative-slot section now gives the “Who are you?” counts beside the six-answer totals and notes the 4-pair versus 1-pair drop as descriptive at this sample size.

**One-liner:** `python3 -I results/part117/plot117.py --numbers`

**Output:** `cold.117.4.who_concept_alone_matched=11/11`; `cold.117.4.who_concept_slot_minus_matched=7/11`; `cold.117.4.who_concept_alone_unrelated=11/11`; `cold.117.4.who_concept_slot_minus_unrelated=10/11`; `prediction.117.4.matched_concept_decreased=6/11`; `prediction.117.4.unrelated_concept_decreased=5/11`.

## Finding 4: Scorer gate and kind check

**Change:** Method text distinguishes the registered gate from the narrower `score117.py` flag. It states the exact `slot_like` conditions and says the Part 112 kind comparison is in `plot117.py`.

**One-liner:** `python3 -I results/part117/plot117.py --numbers`

**Output:** `gate.unpushed_personas_over_2=0/31`; `gate.slot_kind_reproduction=11/11 (registered threshold 9)`; `prediction.117.1.slot_patterns=11/11`. In code, `score117.py` does not load Part 112 kinds for its gate; `plot117.py` compares each slot's Part 112 kind with `slot`.

## Finding 5: Definitions, prompts, dose ladder, and reader cue

**Change:** The README links Part 112, defines the match-score scale, random direction, unpushed row, multiplier, labels, and gate. It lists the six Part 92 questions once, replaces “one dose per unit” with the neuron-specific multiplier explanation, and discloses that the reader prompt included each target's Part 112 atlas kind.

**One-liner:** `python3 -I results/part117/plot117.py --numbers`

**Output:** `data.questions_per_row=6`; `pairing.theme_score_counts=1:1,2:9,3:1`. The label descriptions and their `Its kind: ...` cue are in `personas.json`; the prompt template is in `part92/label92.py`.

## Finding 6: Withdrawal rule unit

**Change:** The README says the design threshold is six positive answer texts out of 66, while `score117.py` compares 132 persona labels with that threshold. It notes the same single positive appears under both units, so the rule does not trigger either way.

**One-liner:** `python3 -I results/part117/plot117.py --numbers`

**Output:** `prediction.117.6.random_persona_labels=1/132`; `prediction.117.6.random_answer_texts_with_persona=1/66`; `prediction.117.6.withdrawal_threshold=6; triggered=False`.

## Finding 7: Coherence target, loop example, and second reader

**Change:** The README says 117.7 uses concept-target coherence only, adds the slot-target count and a loop example, and reports Opus agreement on both fields and the verdict substitution check.

**One-liner:** `python3 -I results/part117/plot117.py --numbers`

**Output:** `cold.117.7.coherent_plus2_concept=0/66`; `cold.117.7.coherent_plus2_slot=1/66`; `cold.117.7.coherent_plus2_both_targets=1/132`; `cold.117.7.loop_example=p01:matched:both+2:q0:I'm ** coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously dehom coordinates continuously de coordinates continuously de`; `cold.second_reader.coherent_agreement=145/164 (88.4%)`; `second_reader.agreement=156/164 (95.1%)`; `cold.second_reader.shows_disagreements=8/164`; `cold.second_reader.verdicts_after_sample_overlay=8/8 unchanged=True`.

## Finding 8: Shared matched concepts

**Change:** Limits state that p03 and p11 share q112-071, p07 and p09 share q112-401, so the 11 pairs are not fully independent.

**One-liner:** `python3 -I results/part117/plot117.py --numbers`

**Output:** `cold.limits.shared_matched_concept_count=2`; `cold.limits.shared_matched_concepts=p03,p11:q112-071;p07,p09:q112-401`.

## Finding 9: Registered follow-up

**Change:** The exploratory naming section cites `PART117B-DESIGN.md` as the registered control for the concept -1, slot +1 cell, and says it has no results yet. No Part 117b results are reported.

**Evidence:** `PART117B-DESIGN.md` was read in full. No new numeric claim was added for this finding.

## Finding 10: Slot-alone leak and closing status

**Change:** The opening paragraph reports the concept-target label on the matched slot +1 alone row. The old “Cold check: pending” text is replaced by the required cold-check line. The README records that Part 117 result files are untracked until Claude's publication commit.

**One-liner:** `python3 -I results/part117/plot117.py --numbers`

**Output:** `figure.who.Matched: slot +1 alone=concept 2/10; slot 11/11`; the missing concept-target label is `p06:matched:s+1|c#0`.

## Closing note

The README ends with the required line: `Cold check: Haiku 5.5, round 1, 2026-10-29, 10 findings, all folded (results/part117/COLD-CHECK-r1.md, results/part117/FOLD-r1.md).`
