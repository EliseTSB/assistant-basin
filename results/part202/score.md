
## Rows
rows: 6516  <- rows:len
unique (stage, unit_id, dose, question_index): 6516  <- rows:unique keys
battery rows: 6400  <- rows:stage=='battery'
nonfinite rows: 0  <- rows:nonfinite
answers at the 80-token cap, neuron: 1011  <- rows:battery/neuron token_count==80
answers at the 80-token cap, random: 3175  <- rows:battery/random token_count==80
mean token_count, neuron: 69.2656  <- rows:battery/neuron token_count mean
mean token_count, random: 71.2613  <- rows:battery/random token_count mean

## Gates R1 to R3
R1 every launch eager: True  <- load:loads[*].load.attn_implementation
R1 last launch status: done  <- load:loads[-1].status
R1/R3 record gates: True  <- load:record_gates
R1/R3 reason: none  <- load:record_gates
R1 device: xpu  <- load:loads[*].load.device
R3 pushed calls not in chunks of 8: 0  <- load:calls[*].chunks
R3 no-push calls not one row: 0  <- load:calls[*].chunks
R3 companion rows (discarded): 132  <- load:calls[*].chunks[*].companions
R3 largest chunk rows: 8  <- load:calls[*].chunks[*].rows
R1 launches: 1  <- load:loads
R1 distinct (model, revision, out_dir): 1  <- load:loads[*]
R2 input hashes: True  <- part164/units164.json etc.:sha256
R2 sha256 part164/units164.json: 4f146140e4dbc6d28b95ed937e6ccf72b9e712a641c721235628b5ec01627921  <- part164/units164.json:sha256
R2 sha256 results/part112/medians.json: 65562b342818b46189a98ae18852e6a972d1b41a594f7ce1c442016aecd0f715  <- results/part112/medians.json:sha256
R2 sha256 part153/battery153.py: e2032288651d65cafa2e60634f512470ec346f5ad093e8be7d6510df089d96a7  <- part153/battery153.py:sha256
R2 sha256 part155/run155.py: 07b15c41a731e07fffe2b0a26d09a95b2c808a71a1020180096cfa41baac0fa5  <- part155/run155.py:sha256
R2 sha256 part164/run164.py: 19d36e18a25f730dd37b1df46c06404a154f5b1e338a53c9cf04e70b909e7277  <- part164/run164.py:sha256
R2 ordered 16 prompts match the regenerated selection: True  <- rows:battery question_index 0..15
R2 rerun questions: 16  <- rows:battery question_index
R2 first differing question: null  <- rows:battery question_index
registered margin Gate 1: 0  <- results/part164/score.json:distances.Gate 1.distance
registered margin Gate 2 Qwen: 4  <- results/part164/score.json:distances.Gate 2 Qwen.distance
registered margin Gate 2 maker: 4  <- results/part164/score.json:distances.Gate 2 maker.distance
registered margin Gate 3: 16  <- results/part164/score.json:distances.Gate 3.distance
registered margin Gate 4: 63.1184  <- results/part164/score.json:distances.Gate 4.distance
registered margin 164.1: 13.3192  <- results/part164/score.json:distances.164.1.distance
registered margin 164.2: 182  <- results/part164/score.json:distances.164.2.distance
registered margin 164.3 first step: 190  <- results/part164/score.json:distances.164.3 first step.distance
registered margin 164.3 second step: 268  <- results/part164/score.json:distances.164.3 second step.distance
registered margin 164.4 upper boundary: 78.7287  <- results/part164/score.json:distances.164.4 upper boundary.distance
registered margin 164.5: 52.664  <- results/part164/score.json:distances.164.5.distance
registered margin Headline Qwen direction: 24.8164  <- results/part164/score.json:distances.Headline Qwen direction.distance
registered margin Headline broken direction: 307  <- results/part164/score.json:distances.Headline broken direction.distance

## Part 164 gates and verdicts (part164/score164.py)
gate1_rows passed: True  <- rows:gates.gate1_rows.passed
gate1_rows expected_rows: 6516  <- rows:gates.gate1_rows.expected_rows
gate1_rows rows: 6516  <- rows:gates.gate1_rows.rows
gate1_rows unique_rows: 6516  <- rows:gates.gate1_rows.unique_rows
gate1_rows nonempty_answers: 6516  <- rows:gates.gate1_rows.nonempty_answers
gate1_rows duplicates: 0  <- rows:gates.gate1_rows.duplicates
gate1_rows units_per_arm: {'neuron': 100, 'random': 100}  <- rows:gates.gate1_rows.units_per_arm
gate2_no_push passed: True  <- rows:gates.gate2_no_push.passed
gate2_no_push qwen_named: 16  <- rows:gates.gate2_no_push.qwen_named
gate2_no_push alibaba_or_tongyi_named: 16  <- rows:gates.gate2_no_push.alibaba_or_tongyi_named
gate2_no_push total: 16  <- rows:gates.gate2_no_push.total
gate2_no_push minimum_each: 12  <- rows:gates.gate2_no_push.minimum_each
gate3_recipe passed: True  <- rows:gates.gate3_recipe.passed
gate3_recipe changed: 71  <- rows:gates.gate3_recipe.changed
gate3_recipe total: 100  <- rows:gates.gate3_recipe.total
gate3_recipe changed_rate: 0.71  <- rows:gates.gate3_recipe.changed_rate
gate3_recipe minimum_rate: 0.7  <- rows:gates.gate3_recipe.minimum_rate
gate4_measure_room passed: True  <- rows:gates.gate4_measure_room.passed
gate4_measure_room random_qwen_rate_0.8: 0.0649888  <- rows:gates.gate4_measure_room.random_qwen_rate_0.8
gate4_measure_room minimum_rate: 0.04  <- rows:gates.gate4_measure_room.minimum_rate
164.1 verdict: failed  <- rows:predictions.164.1.verdict
164.1 estimate: 0.869  <- rows:predictions.164.1.estimate_ratio
164.1 published verdict: failed  <- results/part164/score.json:predictions.164.1.verdict
164.1 published estimate: 0.788682  <- results/part164/score.json:predictions.164.1.estimate_ratio
164.2 verdict: held  <- rows:predictions.164.2.verdict
164.2 estimate: 2.73184  <- rows:predictions.164.2.estimate_ratio
164.2 published verdict: held  <- results/part164/score.json:predictions.164.2.verdict
164.2 published estimate: 3.456  <- results/part164/score.json:predictions.164.2.estimate_ratio
164.3 verdict: held  <- rows:predictions.164.3.verdict
164.3 estimate: null  <- rows:predictions.164.3.estimate_ratio
164.3 published verdict: held  <- results/part164/score.json:predictions.164.3.verdict
164.3 published estimate: null  <- results/part164/score.json:predictions.164.3.estimate_ratio
164.4 verdict: failed  <- rows:predictions.164.4.verdict
164.4 estimate: 7.44145  <- rows:predictions.164.4.estimate_ratio
164.4 published verdict: failed  <- results/part164/score.json:predictions.164.4.verdict
164.4 published estimate: 5.52882  <- results/part164/score.json:predictions.164.4.estimate_ratio
164.5 verdict: held  <- rows:predictions.164.5.verdict
164.5 estimate: 0.118902  <- rows:predictions.164.5.estimate_ratio
164.5 published verdict: held  <- results/part164/score.json:predictions.164.5.verdict
164.5 published estimate: 0.0857094  <- results/part164/score.json:predictions.164.5.estimate_ratio
qwen rate 0.8 neuron: 0.0582219  <- rows:rates.qwen_rate.0.8.neuron.mean
qwen rate 0.8 random: 0.0649888  <- rows:rates.qwen_rate.0.8.random.mean
qwen rate 1.0 random: 0.00758929  <- rows:rates.qwen_rate.1.0.random.mean
qwen rate 1.2 random: 0.00772727  <- rows:rates.qwen_rate.1.2.random.mean
broken rate 0.8 neuron: 0.305625  <- rows:rates.broken_rate.0.8.neuron.mean
broken rate 0.8 random: 0.111875  <- rows:rates.broken_rate.0.8.random.mean
broken rate 1.0 random: 0.285  <- rows:rates.broken_rate.1.0.random.mean
broken rate 1.2 random: 0.478125  <- rows:rates.broken_rate.1.2.random.mean

## Predictions
202.1 gates R1-R3 and Part 164's four gates pass: True  <- all gates above:all
202.2 verdicts as predicted (164.1 failed; 164.2, 164.3, 164.5 held; 164.4 failed), gates passed: True  <- rows:predictions.*.verdict
202.3 neuron Qwen below random and neuron broken above random at +0.8: True  <- rows:rates.*.0.8
202.4 at least 480/1600 neuron and 1440/4800 random at the cap: True  <- rows:battery token_count==80
202.4 neuron at cap: 1011 of 1600  <- rows:battery/neuron token_count==80
202.4 random at cap: 3175 of 4800  <- rows:battery/random token_count==80

## Shape replicate
replicate rows: 200  <- replicate:len
replicate rows matched to a battery row: 200  <- replicate:(unit_id, dose, question_index)
replicate texts that differ from the chunk-of-8 text: 103  <- replicate:answer vs rows answer
202.5 between 40 and 100 of the 200 differ: False  <- replicate:answer vs rows answer
stability independent 164.1: 0.268  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.1.sensitivity
inside shape floor (<0.95) independent 164.1: True  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.1.sensitivity
stability independent 164.2: 0.738  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.2.sensitivity
inside shape floor (<0.95) independent 164.2: True  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.2.sensitivity
stability independent 164.3: 1  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.3.sensitivity
inside shape floor (<0.95) independent 164.3: False  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.3.sensitivity
stability independent 164.4: 1  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.4.sensitivity
inside shape floor (<0.95) independent 164.4: False  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.4.sensitivity
stability independent 164.5: 1  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.5.sensitivity
inside shape floor (<0.95) independent 164.5: False  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.5.sensitivity
stability independent gate:gate1_rows: 1  <- rows+replicate:shape_floor.scenarios.independent.verdicts.gate:gate1_rows.sensitivity
inside shape floor (<0.95) independent gate:gate1_rows: False  <- rows+replicate:shape_floor.scenarios.independent.verdicts.gate:gate1_rows.sensitivity
stability independent gate:gate2_no_push: 1  <- rows+replicate:shape_floor.scenarios.independent.verdicts.gate:gate2_no_push.sensitivity
inside shape floor (<0.95) independent gate:gate2_no_push: False  <- rows+replicate:shape_floor.scenarios.independent.verdicts.gate:gate2_no_push.sensitivity
stability independent gate:gate3_recipe: 1  <- rows+replicate:shape_floor.scenarios.independent.verdicts.gate:gate3_recipe.sensitivity
inside shape floor (<0.95) independent gate:gate3_recipe: False  <- rows+replicate:shape_floor.scenarios.independent.verdicts.gate:gate3_recipe.sensitivity
stability independent gate:gate4_measure_room: 1  <- rows+replicate:shape_floor.scenarios.independent.verdicts.gate:gate4_measure_room.sensitivity
inside shape floor (<0.95) independent gate:gate4_measure_room: False  <- rows+replicate:shape_floor.scenarios.independent.verdicts.gate:gate4_measure_room.sensitivity
stability shared-change-0.5 164.1: 0.264  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.164.1.sensitivity
inside shape floor (<0.95) shared-change-0.5 164.1: True  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.164.1.sensitivity
stability shared-change-0.5 164.2: 0.726  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.164.2.sensitivity
inside shape floor (<0.95) shared-change-0.5 164.2: True  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.164.2.sensitivity
stability shared-change-0.5 164.3: 1  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.164.3.sensitivity
inside shape floor (<0.95) shared-change-0.5 164.3: False  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.164.3.sensitivity
stability shared-change-0.5 164.4: 1  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.164.4.sensitivity
inside shape floor (<0.95) shared-change-0.5 164.4: False  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.164.4.sensitivity
stability shared-change-0.5 164.5: 1  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.164.5.sensitivity
inside shape floor (<0.95) shared-change-0.5 164.5: False  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.164.5.sensitivity
stability shared-change-0.5 gate:gate1_rows: 1  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.gate:gate1_rows.sensitivity
inside shape floor (<0.95) shared-change-0.5 gate:gate1_rows: False  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.gate:gate1_rows.sensitivity
stability shared-change-0.5 gate:gate2_no_push: 1  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.gate:gate2_no_push.sensitivity
inside shape floor (<0.95) shared-change-0.5 gate:gate2_no_push: False  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.gate:gate2_no_push.sensitivity
stability shared-change-0.5 gate:gate3_recipe: 1  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.gate:gate3_recipe.sensitivity
inside shape floor (<0.95) shared-change-0.5 gate:gate3_recipe: False  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.gate:gate3_recipe.sensitivity
stability shared-change-0.5 gate:gate4_measure_room: 1  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.gate:gate4_measure_room.sensitivity
inside shape floor (<0.95) shared-change-0.5 gate:gate4_measure_room: False  <- rows+replicate:shape_floor.scenarios.shared-change-0.5.verdicts.gate:gate4_measure_room.sensitivity
stability shared-change-1 164.1: 0.276  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.164.1.sensitivity
inside shape floor (<0.95) shared-change-1 164.1: True  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.164.1.sensitivity
stability shared-change-1 164.2: 0.719  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.164.2.sensitivity
inside shape floor (<0.95) shared-change-1 164.2: True  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.164.2.sensitivity
stability shared-change-1 164.3: 1  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.164.3.sensitivity
inside shape floor (<0.95) shared-change-1 164.3: False  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.164.3.sensitivity
stability shared-change-1 164.4: 1  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.164.4.sensitivity
inside shape floor (<0.95) shared-change-1 164.4: False  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.164.4.sensitivity
stability shared-change-1 164.5: 1  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.164.5.sensitivity
inside shape floor (<0.95) shared-change-1 164.5: False  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.164.5.sensitivity
stability shared-change-1 gate:gate1_rows: 1  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.gate:gate1_rows.sensitivity
inside shape floor (<0.95) shared-change-1 gate:gate1_rows: False  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.gate:gate1_rows.sensitivity
stability shared-change-1 gate:gate2_no_push: 1  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.gate:gate2_no_push.sensitivity
inside shape floor (<0.95) shared-change-1 gate:gate2_no_push: False  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.gate:gate2_no_push.sensitivity
stability shared-change-1 gate:gate3_recipe: 1  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.gate:gate3_recipe.sensitivity
inside shape floor (<0.95) shared-change-1 gate:gate3_recipe: False  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.gate:gate3_recipe.sensitivity
stability shared-change-1 gate:gate4_measure_room: 1  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.gate:gate4_measure_room.sensitivity
inside shape floor (<0.95) shared-change-1 gate:gate4_measure_room: False  <- rows+replicate:shape_floor.scenarios.shared-change-1.verdicts.gate:gate4_measure_room.sensitivity
stability upper-marginal-independent 164.1: 0.632  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.164.1.sensitivity
inside shape floor (<0.95) upper-marginal-independent 164.1: True  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.164.1.sensitivity
stability upper-marginal-independent 164.2: 0  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.164.2.sensitivity
inside shape floor (<0.95) upper-marginal-independent 164.2: True  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.164.2.sensitivity
stability upper-marginal-independent 164.3: 1  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.164.3.sensitivity
inside shape floor (<0.95) upper-marginal-independent 164.3: False  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.164.3.sensitivity
stability upper-marginal-independent 164.4: 0.389  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.164.4.sensitivity
inside shape floor (<0.95) upper-marginal-independent 164.4: True  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.164.4.sensitivity
stability upper-marginal-independent 164.5: 0.102  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.164.5.sensitivity
inside shape floor (<0.95) upper-marginal-independent 164.5: True  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.164.5.sensitivity
stability upper-marginal-independent gate:gate1_rows: 1  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.gate:gate1_rows.sensitivity
inside shape floor (<0.95) upper-marginal-independent gate:gate1_rows: False  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.gate:gate1_rows.sensitivity
stability upper-marginal-independent gate:gate2_no_push: 1  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.gate:gate2_no_push.sensitivity
inside shape floor (<0.95) upper-marginal-independent gate:gate2_no_push: False  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.gate:gate2_no_push.sensitivity
stability upper-marginal-independent gate:gate3_recipe: 1  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.gate:gate3_recipe.sensitivity
inside shape floor (<0.95) upper-marginal-independent gate:gate3_recipe: False  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.gate:gate3_recipe.sensitivity
stability upper-marginal-independent gate:gate4_measure_room: 0.73  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.gate:gate4_measure_room.sensitivity
inside shape floor (<0.95) upper-marginal-independent gate:gate4_measure_room: True  <- rows+replicate:shape_floor.scenarios.upper-marginal-independent.verdicts.gate:gate4_measure_room.sensitivity
redrawn broken count, independent neuron|0.8: 505.782  <- rows+replicate:shape_floor.scenarios.independent.label_counts.broken.neuron|0.8.simulated_mean
redrawn broken count, independent random|0.8: 245.931  <- rows+replicate:shape_floor.scenarios.independent.label_counts.broken.random|0.8.simulated_mean
redrawn broken count, independent random|1.0: 485.051  <- rows+replicate:shape_floor.scenarios.independent.label_counts.broken.random|1.0.simulated_mean
redrawn broken count, independent random|1.2: 736.045  <- rows+replicate:shape_floor.scenarios.independent.label_counts.broken.random|1.2.simulated_mean
redrawn excluded count, independent neuron|0.8: 93.655  <- rows+replicate:shape_floor.scenarios.independent.label_counts.excluded.neuron|0.8.simulated_mean
redrawn excluded count, independent random|0.8: 170.674  <- rows+replicate:shape_floor.scenarios.independent.label_counts.excluded.random|0.8.simulated_mean
redrawn excluded count, independent random|1.0: 61.529  <- rows+replicate:shape_floor.scenarios.independent.label_counts.excluded.random|1.0.simulated_mean
redrawn excluded count, independent random|1.2: 28.114  <- rows+replicate:shape_floor.scenarios.independent.label_counts.excluded.random|1.2.simulated_mean
redrawn maker_gone count, independent neuron|0.8: 81.691  <- rows+replicate:shape_floor.scenarios.independent.label_counts.maker_gone.neuron|0.8.simulated_mean
redrawn maker_gone count, independent random|0.8: 70.935  <- rows+replicate:shape_floor.scenarios.independent.label_counts.maker_gone.random|0.8.simulated_mean
redrawn maker_gone count, independent random|1.0: 86.605  <- rows+replicate:shape_floor.scenarios.independent.label_counts.maker_gone.random|1.0.simulated_mean
redrawn maker_gone count, independent random|1.2: 91.837  <- rows+replicate:shape_floor.scenarios.independent.label_counts.maker_gone.random|1.2.simulated_mean
redrawn name_gone count, independent neuron|0.8: 97  <- rows+replicate:shape_floor.scenarios.independent.label_counts.name_gone.neuron|0.8.simulated_mean
redrawn name_gone count, independent random|0.8: 96  <- rows+replicate:shape_floor.scenarios.independent.label_counts.name_gone.random|0.8.simulated_mean
redrawn name_gone count, independent random|1.0: 100  <- rows+replicate:shape_floor.scenarios.independent.label_counts.name_gone.random|1.0.simulated_mean
redrawn name_gone count, independent random|1.2: 99  <- rows+replicate:shape_floor.scenarios.independent.label_counts.name_gone.random|1.2.simulated_mean
redrawn qwen count, independent neuron|0.8: 48.063  <- rows+replicate:shape_floor.scenarios.independent.label_counts.qwen.neuron|0.8.simulated_mean
redrawn qwen count, independent random|0.8: 81.201  <- rows+replicate:shape_floor.scenarios.independent.label_counts.qwen.random|0.8.simulated_mean
redrawn qwen count, independent random|1.0: 13.161  <- rows+replicate:shape_floor.scenarios.independent.label_counts.qwen.random|1.0.simulated_mean
redrawn qwen count, independent random|1.2: 8.43  <- rows+replicate:shape_floor.scenarios.independent.label_counts.qwen.random|1.2.simulated_mean
shape floor redraws with no donor text: 0  <- rows+replicate:shape_floor.no_donor_substitutions
shape floor draws projected onto a possible label set: 180601  <- rows+replicate:shape_floor.projected_label_draws

## Headline
headline reading withdrawn: False  <- gates above:R1,R2,R3,Part164 gates
