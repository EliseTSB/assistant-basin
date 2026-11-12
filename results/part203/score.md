# Part 203 score (printed by part203/score203.py)

```
  gate1.verdict: 'passed'  <- results.json:gate1.verdict
  gate2.verdict: 'passed'  <- results.json:gate2.verdict
gate1 passed, gate2 passed

  readouts.V_grid.mean_balanced_accuracy: 0.5199805068226121  <- results.json:readouts.V_grid.mean_balanced_accuracy
  readouts.V_grid.balanced_accuracy_actor_cluster_95.low: 0.5022383040935673  <- results.json:readouts.V_grid.balanced_accuracy_actor_cluster_95.low
203.1: held

  readouts.D_grid.mean_balanced_accuracy: 0.5196296296296297  <- results.json:readouts.D_grid.mean_balanced_accuracy
  readouts.D_grid.balanced_accuracy_actor_cluster_95.low: 0.5019254385964912  <- results.json:readouts.D_grid.balanced_accuracy_actor_cluster_95.low
203.2: held  (|D_grid - V_grid| = 0.0004; both lower bounds above 1/6: True)

  paired_differences.V_grid_minus_V_p196.mean_difference: 0.044678362573099484  <- results.json:paired_differences.V_grid_minus_V_p196.mean_difference
  V_grid chosen C per fold: [0.01, 0.01, 0.01, 0.01, 0.01]  <- results.json:folds.*.V_grid.c_selection.chosen_C
203.3: failed  (difference clause held (+0.0447); C of 10 or 100 on 0 of 5 folds)

  paired_differences.E_p196_minus_D_p196.mean_difference: 0.16966861598440536  <- results.json:paired_differences.E_p196_minus_D_p196.mean_difference
203.4: held

  readouts.H_p196.mean_balanced_accuracy: 0.41918128654970754  <- results.json:readouts.H_p196.mean_balanced_accuracy
203.5: held

  three_actor_ablation.V.all_training_actors_mean: 0.5199805068226121  <- results.json:three_actor_ablation.V.all_training_actors_mean
  three_actor_ablation.V.three_actor_draws_mean: 0.3111539961013645  <- results.json:three_actor_ablation.V.three_actor_draws_mean
203.6: held  (V drop 0.2088)

  three_actor_ablation.E.all_training_actors_mean: 0.6449707602339181  <- results.json:three_actor_ablation.E.all_training_actors_mean
  three_actor_ablation.E.three_actor_draws_mean: 0.436541910331384  <- results.json:three_actor_ablation.E.three_actor_draws_mean
203.7: failed  (V drop 0.2088 minus E drop 0.2084 = 0.0004)

Weaver's rows: scored by forecast; falsifier reported separately.

203.W1: failed  (V_grid - D_grid = 0.0004; falsifier (D within 0.03 of V) reached)

203.W2: held  (falsifier (E within 0.05 of D) not reached)

203.W3: failed  (falsifier (H at 0.35 or above) reached)

  paired_differences.V_grid_minus_H_p196.mean_difference: 0.10079922027290455  <- results.json:paired_differences.V_grid_minus_H_p196.mean_difference
203.W4: failed  (V_grid - H_p196 = 0.1008; falsifier (V minus H at 0.15 or more) not reached)

```
