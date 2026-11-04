# Part 181 analysis

## Gates and predictions

| Item | Verdict | Deciding number |
|---|---|---|
| gate1 | passed | {"included_runs": ["dpo", "instruct", "instruct_b", "midtrain_13B", "midtrain_26B", "midtrain_51B", "pretrain_1T", "pretrain_1T_b", "pretrain_210B", "pretrain_21B", "pretrain_2T", "pretrain_3T", "pretrain_4T", "pretrain_4T_bf16r", "sft"], "missing_checkpoint_keys": [], "unresolved_failed_runs": [], "verdict": "passed"} |
| gate2 | passed | {"commit_prefix_failure_runs": [], "dense_control_sha256": "54515ce29b1c36af123d8337932e510744b3ab77951e3b520f7ada43a10ff20a", "expected_runs": 15, "identity_runs": 15, "invalid_identity_runs": [], "missing_expected_runs": [], "note": "identity is checked against registered corpus/control hashes and repeat-weight rules", "token_ids_sha256": "5238544b790184de8f34a639b32e0c4f362dfb860e345df918671cdb8609d866", "verdict": "passed", "weight_failures": []} |
| gate3 | passed | {"criterion": "each passing layer meets both repeat-pair thresholds; headline requires 12 layers", "failing_layers": [], "layers": {"0": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "1": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "10": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "11": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "12": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "13": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "14": {"primary_cohort_units": 8191, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8191, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8191, "verdict": "passed"}}, "verdict": "passed"}, "15": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "2": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "3": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "4": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "5": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "6": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "7": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "8": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}, "9": {"primary_cohort_units": 8192, "repeat_pairs": {"instruct_vs_instruct_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}, "pretrain_1T_vs_pretrain_1T_b": {"activity_mask_agreement": 1.0, "control_medians": {"dense": {"active_both_units": 256, "median": 1.0}, "sparse_k16": {"active_both_units": 64, "median": 1.0}, "sparse_k2": {"active_both_units": 64, "median": 1.0}, "sparse_k4": {"active_both_units": 64, "median": 1.0}, "sparse_k64": {"active_both_units": 64, "median": 1.0}}, "neuron_median": 1.0, "neuron_p10": 1.0, "share_below_0_5": 0.0, "thresholds": {"activity_mask_agreement": 0.99, "neuron_median": 0.95}, "units": 8192, "verdict": "passed"}}, "verdict": "passed"}}, "passing_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "required_passing_layers": 12, "verdict": "passed"} |
| 181.1 | held | {"mean": 0.6060051458506923, "median": 0.6000000238418579, "threshold": 0.6, "units": 131071} |
| 181.2 | held | {"floor_median": 1.0, "mean": 0.957835821713035, "median": 1.0, "threshold": 0.9, "units": 131071} |
| 181.3 | held | {"mean": 0.031085442801071773, "median": 0.0, "threshold": 0.1, "units": 131071} |
| 181.4 | held | {"keys": ["pretrain_21B", "pretrain_210B", "pretrain_1T", "pretrain_2T", "pretrain_3T", "pretrain_4T"], "means": [0.031085442801071773, 0.08287722074074229, 0.18647145761107012, 0.27598477601916327, 0.3639767813710728, 0.47373485160900697], "medians": [0.0, 0.0, 0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.5]} |
| 181.5 | failed | {"comparisons": [{"mean_overlap_by_k": {"1": 0.2679128147777477, "16": 0.3810546927343239, "2": 0.30009766099828994, "4": 0.324121098521573, "64": 0.3969726623618044, "dense": 0.13774414327417617}, "numbers": {"assessable_layers": 16, "layers_exceeding_both": 3, "passing_layers": 16, "per_layer": [{"dense_mean": 0.30078125, "layer": 0, "neuron_exceeds_both": false, "neuron_mean": 0.20286865532398224, "sparse_k2_mean": 0.2328125238418579}, {"dense_mean": 0.1484375, "layer": 1, "neuron_exceeds_both": false, "neuron_mean": 0.21580810844898224, "sparse_k2_mean": 0.29218751192092896}, {"dense_mean": 0.1484375, "layer": 2, "neuron_exceeds_both": false, "neuron_mean": 0.2519897520542145, "sparse_k2_mean": 0.3187500238418579}, {"dense_mean": 0.17929688096046448, "layer": 3, "neuron_exceeds_both": false, "neuron_mean": 0.2843261659145355, "sparse_k2_mean": 0.34687501192092896}, {"dense_mean": 0.16015625, "layer": 4, "neuron_exceeds_both": false, "neuron_mean": 0.21534423530101776, "sparse_k2_mean": 0.2343750149011612}, {"dense_mean": 0.09335938096046448, "layer": 5, "neuron_exceeds_both": true, "neuron_mean": 0.23813477158546448, "sparse_k2_mean": 0.2343750149011612}, {"dense_mean": 0.0937500074505806, "layer": 6, "neuron_exceeds_both": false, "neuron_mean": 0.22321777045726776, "sparse_k2_mean": 0.2562500238418579}, {"dense_mean": 0.08320312201976776, "layer": 7, "neuron_exceeds_both": false, "neuron_mean": 0.21961669623851776, "sparse_k2_mean": 0.25468748807907104}, {"dense_mean": 0.08359375596046448, "layer": 8, "neuron_exceeds_both": true, "neuron_mean": 0.20908203721046448, "sparse_k2_mean": 0.20156250894069672}, {"dense_mean": 0.072265625, "layer": 9, "neuron_exceeds_both": true, "neuron_mean": 0.20845946669578552, "sparse_k2_mean": 0.20781251788139343}, {"dense_mean": 0.08476562798023224, "layer": 10, "neuron_exceeds_both": false, "neuron_mean": 0.25311279296875, "sparse_k2_mean": 0.27031248807907104}, {"dense_mean": 0.11757812649011612, "layer": 11, "neuron_exceeds_both": false, "neuron_mean": 0.29619142413139343, "sparse_k2_mean": 0.38593751192092896}, {"dense_mean": 0.17070312798023224, "layer": 12, "neuron_exceeds_both": false, "neuron_mean": 0.3559204041957855, "sparse_k2_mean": 0.359375}, {"dense_mean": 0.20351561903953552, "layer": 13, "neuron_exceeds_both": false, "neuron_mean": 0.38267824053764343, "sparse_k2_mean": 0.4140625}, {"dense_mean": 0.14023438096046448, "layer": 14, "neuron_exceeds_both": false, "neuron_mean": 0.3913319408893585, "sparse_k2_mean": 0.44218751788139343}, {"dense_mean": 0.12382812798023224, "layer": 15, "neuron_exceeds_both": false, "neuron_mean": 0.3385376036167145, "sparse_k2_mean": 0.3499999940395355}], "required_layers": 12}, "runs": ["pretrain_1T", "pretrain_2T"], "verdict": "failed"}, {"mean_overlap_by_k": {"1": 0.5791838057159281, "16": 0.636621096986346, "2": 0.5849609419237822, "4": 0.605468753426976, "64": 0.6385742219135864, "dense": 0.4747558647504775}, "numbers": {"assessable_layers": 16, "layers_exceeding_both": 5, "passing_layers": 16, "per_layer": [{"dense_mean": 0.796093761920929, "layer": 0, "neuron_exceeds_both": false, "neuron_mean": 0.6779663562774658, "sparse_k2_mean": 0.668749988079071}, {"dense_mean": 0.6167968511581421, "layer": 1, "neuron_exceeds_both": true, "neuron_mean": 0.6502441167831421, "sparse_k2_mean": 0.640625}, {"dense_mean": 0.6167968511581421, "layer": 2, "neuron_exceeds_both": false, "neuron_mean": 0.6422607898712158, "sparse_k2_mean": 0.653124988079071}, {"dense_mean": 0.5980468988418579, "layer": 3, "neuron_exceeds_both": false, "neuron_mean": 0.6453735828399658, "sparse_k2_mean": 0.6656250357627869}, {"dense_mean": 0.5199218392372131, "layer": 4, "neuron_exceeds_both": true, "neuron_mean": 0.5746338367462158, "sparse_k2_mean": 0.5703125}, {"dense_mean": 0.4144531488418579, "layer": 5, "neuron_exceeds_both": true, "neuron_mean": 0.55828857421875, "sparse_k2_mean": 0.520312488079071}, {"dense_mean": 0.44023439288139343, "layer": 6, "neuron_exceeds_both": false, "neuron_mean": 0.5243164300918579, "sparse_k2_mean": 0.5328125357627869}, {"dense_mean": 0.39335936307907104, "layer": 7, "neuron_exceeds_both": false, "neuron_mean": 0.5067138671875, "sparse_k2_mean": 0.5140625238418579}, {"dense_mean": 0.4019531309604645, "layer": 8, "neuron_exceeds_both": false, "neuron_mean": 0.48249512910842896, "sparse_k2_mean": 0.4921875}, {"dense_mean": 0.3246093988418579, "layer": 9, "neuron_exceeds_both": true, "neuron_mean": 0.4753662347793579, "sparse_k2_mean": 0.4750000238418579}, {"dense_mean": 0.32382816076278687, "layer": 10, "neuron_exceeds_both": false, "neuron_mean": 0.5155150890350342, "sparse_k2_mean": 0.520312488079071}, {"dense_mean": 0.376953125, "layer": 11, "neuron_exceeds_both": false, "neuron_mean": 0.55523681640625, "sparse_k2_mean": 0.5968749523162842}, {"dense_mean": 0.4453125, "layer": 12, "neuron_exceeds_both": false, "neuron_mean": 0.6133178472518921, "sparse_k2_mean": 0.6343749761581421}, {"dense_mean": 0.4742187559604645, "layer": 13, "neuron_exceeds_both": false, "neuron_mean": 0.629321277141571, "sparse_k2_mean": 0.671875}, {"dense_mean": 0.44296878576278687, "layer": 14, "neuron_exceeds_both": false, "neuron_mean": 0.6308021545410156, "sparse_k2_mean": 0.6328125}, {"dense_mean": 0.4105468690395355, "layer": 15, "neuron_exceeds_both": true, "neuron_mean": 0.585095226764679, "sparse_k2_mean": 0.5703125}], "required_layers": 12}, "runs": ["pretrain_4T", "midtrain_51B"], "verdict": "failed"}, {"mean_overlap_by_k": {"1": 0.6060051458506923, "16": 0.6301757847031695, "2": 0.6150390663460712, "4": 0.6241210967273219, "64": 0.626757815865858, "dense": 0.5029296914926817}, "numbers": {"assessable_layers": 16, "layers_exceeding_both": 7, "passing_layers": 16, "per_layer": [{"dense_mean": 0.8941405415534973, "layer": 0, "neuron_exceeds_both": false, "neuron_mean": 0.82574462890625, "sparse_k2_mean": 0.807812511920929}, {"dense_mean": 0.7515624761581421, "layer": 1, "neuron_exceeds_both": false, "neuron_mean": 0.777355968952179, "sparse_k2_mean": 0.7828124761581421}, {"dense_mean": 0.7281249761581421, "layer": 2, "neuron_exceeds_both": false, "neuron_mean": 0.7392822504043579, "sparse_k2_mean": 0.7734375}, {"dense_mean": 0.668749988079071, "layer": 3, "neuron_exceeds_both": true, "neuron_mean": 0.7114989757537842, "sparse_k2_mean": 0.7093750238418579}, {"dense_mean": 0.612109363079071, "layer": 4, "neuron_exceeds_both": true, "neuron_mean": 0.6541748046875, "sparse_k2_mean": 0.6328125}, {"dense_mean": 0.5, "layer": 5, "neuron_exceeds_both": true, "neuron_mean": 0.612561047077179, "sparse_k2_mean": 0.609375}, {"dense_mean": 0.5390625, "layer": 6, "neuron_exceeds_both": false, "neuron_mean": 0.5692992806434631, "sparse_k2_mean": 0.604687511920929}, {"dense_mean": 0.470703125, "layer": 7, "neuron_exceeds_both": false, "neuron_mean": 0.5400024652481079, "sparse_k2_mean": 0.565625011920929}, {"dense_mean": 0.4449218511581421, "layer": 8, "neuron_exceeds_both": false, "neuron_mean": 0.5028197765350342, "sparse_k2_mean": 0.503125011920929}, {"dense_mean": 0.3828125, "layer": 9, "neuron_exceeds_both": true, "neuron_mean": 0.47978514432907104, "sparse_k2_mean": 0.46562498807907104}, {"dense_mean": 0.3515625, "layer": 10, "neuron_exceeds_both": true, "neuron_mean": 0.5033813714981079, "sparse_k2_mean": 0.4921875596046448}, {"dense_mean": 0.3492187559604645, "layer": 11, "neuron_exceeds_both": false, "neuron_mean": 0.52862548828125, "sparse_k2_mean": 0.5859375}, {"dense_mean": 0.37968748807907104, "layer": 12, "neuron_exceeds_both": true, "neuron_mean": 0.57843017578125, "sparse_k2_mean": 0.578125}, {"dense_mean": 0.3890625238418579, "layer": 13, "neuron_exceeds_both": false, "neuron_mean": 0.5835449695587158, "sparse_k2_mean": 0.6296874284744263}, {"dense_mean": 0.3246093988418579, "layer": 14, "neuron_exceeds_both": true, "neuron_mean": 0.5701013207435608, "sparse_k2_mean": 0.5640625357627869}, {"dense_mean": 0.26054686307907104, "layer": 15, "neuron_exceeds_both": false, "neuron_mean": 0.51947021484375, "sparse_k2_mean": 0.5359375476837158}], "required_layers": 12}, "runs": ["midtrain_51B", "instruct"], "verdict": "failed"}]} |
| 181.6 | held | {"absolute_difference": 0.0, "active_neuron_baseline_median": 1.0, "assessable_slots": 21, "median_comparator": 1.0, "sensitivity_position0_excluded": false, "slot_count": 21, "slot_median_overlap": 1.0, "tolerance": 0.15} |

## Run exclusions

None.

## Pairwise top-10 document overlap

| Run A | Run B | Primary neuron median | Dense median | Sparse k=2 median | Null median |
|---|---|---:|---:|---:|---:|
| dpo | instruct | 1.0 | 1.0 | 1.0 | 0.0 |
| dpo | instruct_b | 1.0 | 1.0 | 1.0 | 0.0 |
| dpo | midtrain_13B | 0.5 | 0.4000000059604645 | 0.6000000238418579 | 0.0 |
| dpo | midtrain_26B | 0.6000000238418579 | 0.4000000059604645 | 0.6000000238418579 | 0.0 |
| dpo | midtrain_51B | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| dpo | pretrain_1T | 0.10000000149011612 | 0.0 | 0.20000000298023224 | 0.0 |
| dpo | pretrain_1T_b | 0.10000000149011612 | 0.0 | 0.20000000298023224 | 0.0 |
| dpo | pretrain_210B | 0.0 | 0.0 | 0.0 | 0.0 |
| dpo | pretrain_21B | 0.0 | 0.0 | 0.0 | 0.0 |
| dpo | pretrain_2T | 0.20000000298023224 | 0.10000000149011612 | 0.30000001192092896 | 0.0 |
| dpo | pretrain_3T | 0.30000001192092896 | 0.20000000298023224 | 0.4000000059604645 | 0.0 |
| dpo | pretrain_4T | 0.5 | 0.30000001192092896 | 0.5 | 0.0 |
| dpo | pretrain_4T_bf16r | 0.5 | 0.30000001192092896 | 0.5 | 0.0 |
| dpo | sft | 0.8999999761581421 | 0.8999999761581421 | 0.8999999761581421 | 0.0 |
| instruct | instruct_b | 1.0 | 1.0 | 1.0 | 0.0 |
| instruct | midtrain_13B | 0.5 | 0.4000000059604645 | 0.6000000238418579 | 0.0 |
| instruct | midtrain_26B | 0.6000000238418579 | 0.4000000059604645 | 0.6000000238418579 | 0.0 |
| instruct | midtrain_51B | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| instruct | pretrain_1T | 0.10000000149011612 | 0.0 | 0.20000000298023224 | 0.0 |
| instruct | pretrain_1T_b | 0.10000000149011612 | 0.0 | 0.20000000298023224 | 0.0 |
| instruct | pretrain_210B | 0.0 | 0.0 | 0.0 | 0.0 |
| instruct | pretrain_21B | 0.0 | 0.0 | 0.0 | 0.0 |
| instruct | pretrain_2T | 0.20000000298023224 | 0.10000000149011612 | 0.30000001192092896 | 0.0 |
| instruct | pretrain_3T | 0.30000001192092896 | 0.20000000298023224 | 0.4000000059604645 | 0.0 |
| instruct | pretrain_4T | 0.5 | 0.30000001192092896 | 0.5 | 0.0 |
| instruct | pretrain_4T_bf16r | 0.5 | 0.30000001192092896 | 0.5 | 0.0 |
| instruct | sft | 0.8999999761581421 | 0.8999999761581421 | 0.8999999761581421 | 0.0 |
| instruct_b | midtrain_13B | 0.5 | 0.4000000059604645 | 0.6000000238418579 | 0.0 |
| instruct_b | midtrain_26B | 0.6000000238418579 | 0.4000000059604645 | 0.6000000238418579 | 0.0 |
| instruct_b | midtrain_51B | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| instruct_b | pretrain_1T | 0.10000000149011612 | 0.0 | 0.20000000298023224 | 0.0 |
| instruct_b | pretrain_1T_b | 0.10000000149011612 | 0.0 | 0.20000000298023224 | 0.0 |
| instruct_b | pretrain_210B | 0.0 | 0.0 | 0.0 | 0.0 |
| instruct_b | pretrain_21B | 0.0 | 0.0 | 0.0 | 0.0 |
| instruct_b | pretrain_2T | 0.20000000298023224 | 0.10000000149011612 | 0.30000001192092896 | 0.0 |
| instruct_b | pretrain_3T | 0.30000001192092896 | 0.20000000298023224 | 0.4000000059604645 | 0.0 |
| instruct_b | pretrain_4T | 0.5 | 0.30000001192092896 | 0.5 | 0.0 |
| instruct_b | pretrain_4T_bf16r | 0.5 | 0.30000001192092896 | 0.5 | 0.0 |
| instruct_b | sft | 0.8999999761581421 | 0.8999999761581421 | 0.8999999761581421 | 0.0 |
| midtrain_13B | midtrain_26B | 0.699999988079071 | 0.699999988079071 | 0.699999988079071 | 0.0 |
| midtrain_13B | midtrain_51B | 0.699999988079071 | 0.6000000238418579 | 0.699999988079071 | 0.0 |
| midtrain_13B | pretrain_1T | 0.10000000149011612 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| midtrain_13B | pretrain_1T_b | 0.10000000149011612 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| midtrain_13B | pretrain_210B | 0.0 | 0.0 | 0.0 | 0.0 |
| midtrain_13B | pretrain_21B | 0.0 | 0.0 | 0.0 | 0.0 |
| midtrain_13B | pretrain_2T | 0.30000001192092896 | 0.10000000149011612 | 0.30000001192092896 | 0.0 |
| midtrain_13B | pretrain_3T | 0.4000000059604645 | 0.30000001192092896 | 0.5 | 0.0 |
| midtrain_13B | pretrain_4T | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| midtrain_13B | pretrain_4T_bf16r | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| midtrain_13B | sft | 0.6000000238418579 | 0.4000000059604645 | 0.6000000238418579 | 0.0 |
| midtrain_26B | midtrain_51B | 0.800000011920929 | 0.699999988079071 | 0.800000011920929 | 0.0 |
| midtrain_26B | pretrain_1T | 0.10000000149011612 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| midtrain_26B | pretrain_1T_b | 0.10000000149011612 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| midtrain_26B | pretrain_210B | 0.0 | 0.0 | 0.0 | 0.0 |
| midtrain_26B | pretrain_21B | 0.0 | 0.0 | 0.0 | 0.0 |
| midtrain_26B | pretrain_2T | 0.30000001192092896 | 0.10000000149011612 | 0.30000001192092896 | 0.0 |
| midtrain_26B | pretrain_3T | 0.4000000059604645 | 0.30000001192092896 | 0.4000000059604645 | 0.0 |
| midtrain_26B | pretrain_4T | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| midtrain_26B | pretrain_4T_bf16r | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| midtrain_26B | sft | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| midtrain_51B | pretrain_1T | 0.10000000149011612 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| midtrain_51B | pretrain_1T_b | 0.10000000149011612 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| midtrain_51B | pretrain_210B | 0.0 | 0.0 | 0.0 | 0.0 |
| midtrain_51B | pretrain_21B | 0.0 | 0.0 | 0.0 | 0.0 |
| midtrain_51B | pretrain_2T | 0.30000001192092896 | 0.10000000149011612 | 0.30000001192092896 | 0.0 |
| midtrain_51B | pretrain_3T | 0.4000000059604645 | 0.30000001192092896 | 0.4000000059604645 | 0.0 |
| midtrain_51B | pretrain_4T | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| midtrain_51B | pretrain_4T_bf16r | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| midtrain_51B | sft | 0.6000000238418579 | 0.5 | 0.6000000238418579 | 0.0 |
| pretrain_1T | pretrain_1T_b | 1.0 | 1.0 | 1.0 | 0.0 |
| pretrain_1T | pretrain_210B | 0.10000000149011612 | 0.0 | 0.10000000149011612 | 0.0 |
| pretrain_1T | pretrain_21B | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_1T | pretrain_2T | 0.20000000298023224 | 0.10000000149011612 | 0.30000001192092896 | 0.0 |
| pretrain_1T | pretrain_3T | 0.20000000298023224 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| pretrain_1T | pretrain_4T | 0.20000000298023224 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| pretrain_1T | pretrain_4T_bf16r | 0.20000000298023224 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| pretrain_1T | sft | 0.10000000149011612 | 0.0 | 0.20000000298023224 | 0.0 |
| pretrain_1T_b | pretrain_210B | 0.10000000149011612 | 0.0 | 0.10000000149011612 | 0.0 |
| pretrain_1T_b | pretrain_21B | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_1T_b | pretrain_2T | 0.20000000298023224 | 0.10000000149011612 | 0.30000001192092896 | 0.0 |
| pretrain_1T_b | pretrain_3T | 0.20000000298023224 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| pretrain_1T_b | pretrain_4T | 0.20000000298023224 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| pretrain_1T_b | pretrain_4T_bf16r | 0.20000000298023224 | 0.10000000149011612 | 0.20000000298023224 | 0.0 |
| pretrain_1T_b | sft | 0.10000000149011612 | 0.0 | 0.20000000298023224 | 0.0 |
| pretrain_210B | pretrain_21B | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_210B | pretrain_2T | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_210B | pretrain_3T | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_210B | pretrain_4T | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_210B | pretrain_4T_bf16r | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_210B | sft | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_21B | pretrain_2T | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_21B | pretrain_3T | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_21B | pretrain_4T | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_21B | pretrain_4T_bf16r | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_21B | sft | 0.0 | 0.0 | 0.0 | 0.0 |
| pretrain_2T | pretrain_3T | 0.4000000059604645 | 0.20000000298023224 | 0.4000000059604645 | 0.0 |
| pretrain_2T | pretrain_4T | 0.30000001192092896 | 0.20000000298023224 | 0.30000001192092896 | 0.0 |
| pretrain_2T | pretrain_4T_bf16r | 0.30000001192092896 | 0.20000000298023224 | 0.30000001192092896 | 0.0 |
| pretrain_2T | sft | 0.20000000298023224 | 0.10000000149011612 | 0.30000001192092896 | 0.0 |
| pretrain_3T | pretrain_4T | 0.5 | 0.30000001192092896 | 0.5 | 0.0 |
| pretrain_3T | pretrain_4T_bf16r | 0.5 | 0.30000001192092896 | 0.5 | 0.0 |
| pretrain_3T | sft | 0.4000000059604645 | 0.20000000298023224 | 0.4000000059604645 | 0.0 |
| pretrain_4T | pretrain_4T_bf16r | 1.0 | 1.0 | 1.0 | 0.0 |
| pretrain_4T | sft | 0.5 | 0.30000001192092896 | 0.5 | 0.0 |
| pretrain_4T_bf16r | sft | 0.5 | 0.30000001192092896 | 0.5 | 0.0 |

## Gate 1 per-unit values

### dpo: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2544 | 2379 | 154 | 0.503036618232727 | 0.503036618232727 | 0.0 | True |
| 0 | 2544 | 455 | 55 | 0.3138500154018402 | 0.3138500154018402 | 0.0 | True |
| 0 | 6507 | 228 | 85 | 0.17375034093856812 | 0.17375034093856812 | 0.0 | True |
| 0 | 6507 | 199 | 71 | 0.1625000387430191 | 0.1625000238418579 | 9.169943163775277e-08 | True |
| 0 | 8206 | 2429 | 21 | 0.5015606880187988 | 0.5015615224838257 | 1.6637369052021726e-06 | True |
| 0 | 8206 | 2041 | 30 | 0.322068452835083 | 0.3220682144165039 | 7.402729978761568e-07 | True |
| 0 | 8618 | 1017 | 17 | 0.32341718673706055 | 0.32341718673706055 | 0.0 | True |
| 0 | 8618 | 199 | 36 | 0.30396708846092224 | 0.30396711826324463 | 9.804456968875653e-08 | True |
| 1 | 1235 | 1918 | 75 | 0.2577807605266571 | 0.2577809989452362 | 9.248889584097089e-07 | True |
| 1 | 1235 | 1642 | 211 | 0.18918387591838837 | 0.18918384611606598 | 1.575309853602517e-07 | True |
| 1 | 4744 | 491 | 69 | 0.22122356295585632 | 0.2212236374616623 | 3.367896483255964e-07 | True |
| 1 | 4744 | 1910 | 155 | 0.1984933465719223 | 0.19849342107772827 | 3.753566920805669e-07 | True |
| 1 | 8631 | 1919 | 94 | 0.4571806788444519 | 0.4571808874607086 | 4.5631030874085863e-07 | True |
| 1 | 8631 | 673 | 230 | 0.2630905508995056 | 0.2630905508995056 | 0.0 | True |
| 1 | 8361 | 1316 | 371 | 0.10973966121673584 | 0.10973966866731644 | 6.789323490081612e-08 | True |
| 1 | 8361 | 952 | 17 | 0.10347872227430344 | 0.10347868502140045 | 3.600054404022154e-07 | True |
| 2 | 764 | 1653 | 82 | 0.29089874029159546 | 0.29089874029159546 | 0.0 | True |
| 2 | 764 | 751 | 82 | 0.20143815875053406 | 0.20143811404705048 | 2.2192162527113274e-07 | True |
| 2 | 5747 | 602 | 12 | 0.3344171941280365 | 0.3344176709651947 | 1.425875124173672e-06 | True |
| 2 | 5747 | 733 | 8 | 0.26790598034858704 | 0.26790598034858704 | 0.0 | True |
| 2 | 8573 | 12 | 126 | 0.7769177556037903 | 0.7769178748130798 | 1.5343875035799177e-07 | True |
| 2 | 8573 | 1834 | 2 | 0.48589515686035156 | 0.4858955144882202 | 7.360185908482467e-07 | True |
| 2 | 8616 | 2411 | 156 | 0.18655967712402344 | 0.18655966222286224 | 7.987342936888488e-08 | True |
| 2 | 8616 | 1458 | 314 | 0.17290736734867096 | 0.172907292842865 | 4.309001236424815e-07 | True |
| 3 | 7360 | 865 | 106 | 1.9455807209014893 | 1.9455807209014893 | 0.0 | True |
| 3 | 7360 | 1793 | 149 | 0.7062281370162964 | 0.7062281370162964 | 0.0 | True |
| 3 | 1706 | 47 | 121 | 0.6508389711380005 | 0.6508392691612244 | 4.5790623655472813e-07 | True |
| 3 | 1706 | 1764 | 88 | 0.5645735859870911 | 0.5645731687545776 | 7.390223768585489e-07 | True |
| 3 | 8625 | 2354 | 92 | 0.864875078201294 | 0.8648748993873596 | 2.0675117000487105e-07 | True |
| 3 | 8625 | 298 | 156 | 0.5390574932098389 | 0.5390573740005493 | 2.211439244466205e-07 | True |
| 3 | 8419 | 564 | 89 | 0.18630485236644745 | 0.18630480766296387 | 2.399480368531391e-07 | True |
| 3 | 8419 | 1385 | 137 | 0.17250527441501617 | 0.17250530421733856 | 1.7276180388546479e-07 | True |
| 4 | 3991 | 1681 | 148 | 0.4101671874523163 | 0.41016727685928345 | 2.179768882011799e-07 | True |
| 4 | 3991 | 1060 | 163 | 0.26968681812286377 | 0.26968681812286377 | 0.0 | True |
| 4 | 4508 | 865 | 280 | 0.2585693299770355 | 0.25856924057006836 | 3.457756075363868e-07 | True |
| 4 | 4508 | 205 | 347 | 0.2293914407491684 | 0.22939160466194153 | 7.145548787566017e-07 | True |
| 4 | 8360 | 564 | 65 | 0.32401716709136963 | 0.324017196893692 | 9.19776030857382e-08 | True |
| 4 | 8360 | 157 | 124 | 0.24970299005508423 | 0.24970301985740662 | 1.1935108338558923e-07 | True |
| 4 | 8681 | 2399 | 225 | 0.3997214734554291 | 0.39972123503685 | 5.964617738460012e-07 | True |
| 4 | 8681 | 1450 | 302 | 0.3391154706478119 | 0.3391154706478119 | 0.0 | True |
| 5 | 1558 | 2196 | 125 | 0.39119067788124084 | 0.39119064807891846 | 7.618362111569237e-08 | True |
| 5 | 1558 | 1832 | 204 | 0.2808215916156769 | 0.2808215320110321 | 2.1225093281631837e-07 | True |
| 5 | 3169 | 1739 | 341 | 1.2278180122375488 | 1.2278192043304443 | 9.709035733523475e-07 | True |
| 5 | 3169 | 1124 | 232 | 0.8674494624137878 | 0.8674501180648804 | 7.55837799132257e-07 | True |
| 5 | 8498 | 2189 | 17 | 1.3936514854431152 | 1.3936522006988525 | 5.13224249229907e-07 | True |
| 5 | 8498 | 2090 | 58 | 0.7271303534507751 | 0.7271308898925781 | 7.377519043630893e-07 | True |
| 5 | 8271 | 1970 | 0 | 0.18748408555984497 | 0.18748410046100616 | 7.94796056921386e-08 | True |
| 5 | 8271 | 1017 | 102 | 0.18327732384204865 | 0.18327733874320984 | 8.130389991229749e-08 | True |
| 6 | 4186 | 2422 | 68 | 1.1626002788543701 | 1.1626012325286865 | 8.202942436466673e-07 | True |
| 6 | 4186 | 364 | 94 | 0.8584470152854919 | 0.8584463596343994 | 7.637641937764189e-07 | True |
| 6 | 873 | 248 | 142 | 0.360683798789978 | 0.3606839179992676 | 3.3050913279361194e-07 | True |
| 6 | 873 | 1068 | 4 | 0.3283209800720215 | 0.32832130789756775 | 9.984910077715277e-07 | True |
| 6 | 8207 | 278 | 19 | 0.262871116399765 | 0.2628710865974426 | 1.1337237348805186e-07 | True |
| 6 | 8207 | 259 | 18 | 0.22684286534786224 | 0.2268429398536682 | 3.2844676800826004e-07 | True |
| 6 | 8692 | 940 | 146 | 0.381428599357605 | 0.38142889738082886 | 7.813342375974909e-07 | True |
| 6 | 8692 | 1609 | 85 | 0.34294360876083374 | 0.3429439961910248 | 1.1297198173190915e-06 | True |
| 7 | 5407 | 455 | 152 | 0.6347970962524414 | 0.6347969770431519 | 1.8779117020940968e-07 | True |
| 7 | 5407 | 2172 | 404 | 0.40469786524772644 | 0.40469810366630554 | 5.891273455460905e-07 | True |
| 7 | 157 | 1069 | 283 | 0.5830696821212769 | 0.5830708742141724 | 2.0445118860765267e-06 | True |
| 7 | 157 | 444 | 83 | 0.478254109621048 | 0.47825509309768677 | 2.056389310639104e-06 | True |
| 7 | 8277 | 1885 | 9 | 0.3554876446723938 | 0.3554876148700714 | 8.383504415507941e-08 | True |
| 7 | 8277 | 1622 | 185 | 0.3042067885398865 | 0.3042067587375641 | 9.79673153605109e-08 | True |
| 7 | 8537 | 1243 | 191 | 0.4741307199001312 | 0.4741307199001312 | 0.0 | True |
| 7 | 8537 | 778 | 25 | 0.45527803897857666 | 0.45527875423431396 | 1.5710306144117446e-06 | True |
| 8 | 2006 | 1136 | 200 | 0.5487331748008728 | 0.5487332344055176 | 1.086222731057226e-07 | True |
| 8 | 2006 | 617 | 197 | 0.41258054971694946 | 0.4125806391239166 | 2.167018470076292e-07 | True |
| 8 | 2632 | 1814 | 103 | 0.5340954661369324 | 0.5340949892997742 | 8.92793870077288e-07 | True |
| 8 | 2632 | 93 | 50 | 0.45682597160339355 | 0.45682626962661743 | 6.523780222716638e-07 | True |
| 8 | 8451 | 2273 | 12 | 0.7472058534622192 | 0.7472059726715088 | 1.5954009059005424e-07 | True |
| 8 | 8451 | 1393 | 19 | 0.3954147398471832 | 0.39541488885879517 | 3.768489055214922e-07 | True |
| 8 | 8342 | 2180 | 265 | 0.24886056780815125 | 0.248860701918602 | 5.388979536847148e-07 | True |
| 8 | 8342 | 1184 | 101 | 0.24025821685791016 | 0.24025820195674896 | 6.202144254928968e-08 | True |
| 9 | 492 | 1673 | 176 | 2.076137065887451 | 2.0761382579803467 | 5.741879546850866e-07 | True |
| 9 | 492 | 778 | 25 | 1.4836734533309937 | 1.4836727380752563 | 4.820843398517832e-07 | True |
| 9 | 6533 | 1190 | 88 | 0.6574814915657043 | 0.6574811339378357 | 5.439360244205518e-07 | True |
| 9 | 6533 | 489 | 115 | 0.5634288787841797 | 0.5634284019470215 | 8.463129529890081e-07 | True |
| 9 | 8543 | 166 | 0 | 0.9286829233169556 | 0.9286830425262451 | 1.283638220944175e-07 | True |
| 9 | 8543 | 2322 | 0 | 0.8402161002159119 | 0.8402161002159119 | 0.0 | True |
| 9 | 8310 | 657 | 228 | 0.25787150859832764 | 0.2578715980052948 | 3.4671130459142845e-07 | True |
| 9 | 8310 | 747 | 389 | 0.2468096911907196 | 0.2468096911907196 | 0.0 | True |
| 10 | 152 | 2041 | 264 | 1.615803837776184 | 1.6158032417297363 | 3.688854016922249e-07 | True |
| 10 | 152 | 1543 | 97 | 1.079062581062317 | 1.0790616273880005 | 8.837988946548175e-07 | True |
| 10 | 7847 | 799 | 100 | 0.6435704827308655 | 0.6435703635215759 | 1.8523113279673727e-07 | True |
| 10 | 7847 | 1227 | 142 | 0.5895591378211975 | 0.5895594954490662 | 6.066021976591019e-07 | True |
| 10 | 8702 | 1359 | 8 | 0.9291179180145264 | 0.9291180372238159 | 1.283037246827883e-07 | True |
| 10 | 8702 | 1508 | 16 | 0.8133691549301147 | 0.8133692145347595 | 7.328117179524947e-08 | True |
| 10 | 8344 | 4 | 136 | 0.3255620300769806 | 0.3255620002746582 | 9.154114925702e-08 | True |
| 10 | 8344 | 1764 | 111 | 0.3178088366985321 | 0.31780892610549927 | 2.813231000492784e-07 | True |
| 11 | 8136 | 75 | 90 | 2.5645172595977783 | 2.5645172595977783 | 0.0 | True |
| 11 | 8136 | 886 | 26 | 1.6173192262649536 | 1.6173185110092163 | 4.4224771813076337e-07 | True |
| 11 | 213 | 253 | 151 | 2.286205768585205 | 2.286205768585205 | 0.0 | True |
| 11 | 213 | 2302 | 46 | 2.094944953918457 | 2.094944953918457 | 0.0 | True |
| 11 | 8479 | 2452 | 159 | 2.894529342651367 | 2.8945295810699463 | 8.236868619309724e-08 | True |
| 11 | 8479 | 1301 | 219 | 1.653070092201233 | 1.6530705690383911 | 2.8845549892452974e-07 | True |
| 11 | 8323 | 1449 | 91 | 0.46527981758117676 | 0.4652799069881439 | 1.9215741535465853e-07 | True |
| 11 | 8323 | 2381 | 244 | 0.4486912488937378 | 0.4486914575099945 | 4.6494389455603835e-07 | True |
| 12 | 104 | 1386 | 94 | 1.4945566654205322 | 1.494556188583374 | 3.1904923328481124e-07 | True |
| 12 | 104 | 2134 | 175 | 1.1803020238876343 | 1.1803028583526611 | 7.069927950364258e-07 | True |
| 12 | 1816 | 1038 | 268 | 5.294307708740234 | 5.294310569763184 | 5.403960454537922e-07 | True |
| 12 | 1816 | 2084 | 104 | 4.330537796020508 | 4.330537796020508 | 0.0 | True |
| 12 | 8315 | 2425 | 154 | 0.7293151021003723 | 0.7293152809143066 | 2.4518062742866736e-07 | True |
| 12 | 8315 | 235 | 162 | 0.6187213063240051 | 0.6187211275100708 | 2.8900561932892704e-07 | True |
| 12 | 8396 | 426 | 155 | 0.5347754955291748 | 0.5347753167152405 | 3.3437196696761627e-07 | True |
| 12 | 8396 | 2367 | 59 | 0.49815258383750916 | 0.4981526732444763 | 1.794770720134402e-07 | True |
| 13 | 7863 | 198 | 14 | 10.614975929260254 | 10.614977836608887 | 1.7968468751350442e-07 | True |
| 13 | 7863 | 2035 | 96 | 7.982569217681885 | 7.98257303237915 | 4.778783824605253e-07 | True |
| 13 | 4713 | 2452 | 199 | 4.374295711517334 | 4.374292850494385 | 6.540533923405769e-07 | True |
| 13 | 4713 | 83 | 255 | 3.839094400405884 | 3.839089870452881 | 1.1799535334298536e-06 | True |
| 13 | 8510 | 308 | 248 | 5.268658638000488 | 5.268660068511963 | 2.7151341031884904e-07 | True |
| 13 | 8510 | 382 | 12 | 2.994607925415039 | 2.994612693786621 | 1.5923191619051016e-06 | True |
| 13 | 8491 | 1423 | 19 | 1.8657984733581543 | 1.8657984733581543 | 0.0 | True |
| 13 | 8491 | 1549 | 238 | 1.7078111171722412 | 1.7078126668930054 | 9.074310083694456e-07 | True |
| 14 | 1070 | 2416 | 104 | 4.946190357208252 | 4.9461870193481445 | 6.748345426207662e-07 | True |
| 14 | 1070 | 4 | 80 | 2.8990249633789062 | 2.8990235328674316 | 4.934457249178249e-07 | True |
| 14 | 4676 | 1826 | 23 | 5.549010753631592 | 5.549007892608643 | 5.155915308591414e-07 | True |
| 14 | 4676 | 1262 | 23 | 5.019500255584717 | 5.019504547119141 | 8.549724485128467e-07 | True |
| 14 | 8432 | 406 | 49 | 1.504807472229004 | 1.504807472229004 | 0.0 | True |
| 14 | 8432 | 977 | 145 | 1.3035749197006226 | 1.303574562072754 | 2.74343931635687e-07 | True |
| 14 | 8498 | 5 | 429 | 1.5976684093475342 | 1.5976715087890625 | 1.9399779767730914e-06 | True |
| 14 | 8498 | 245 | 134 | 1.3401672840118408 | 1.3401695489883423 | 1.6900699849085645e-06 | True |
| 15 | 7395 | 1060 | 26 | 15.344941139221191 | 15.344941139221191 | 0.0 | True |
| 15 | 7395 | 2181 | 249 | 12.30232048034668 | 12.302315711975098 | 3.8759936303471074e-07 | True |
| 15 | 1851 | 1148 | 299 | 5.2852067947387695 | 5.285211086273193 | 8.119898786363839e-07 | True |
| 15 | 1851 | 1700 | 177 | 4.768338680267334 | 4.768337726593018 | 2.0000138000952207e-07 | True |
| 15 | 8249 | 885 | 198 | 2.6821529865264893 | 2.6821534633636475 | 1.7778149143559887e-07 | True |
| 15 | 8249 | 881 | 96 | 2.2740139961242676 | 2.274014472961426 | 2.0968963208486223e-07 | True |
| 15 | 8258 | 1009 | 204 | 2.3913326263427734 | 2.3913302421569824 | 9.97011358751008e-07 | True |
| 15 | 8258 | 2458 | 97 | 2.3087356090545654 | 2.308736562728882 | 4.130721216695673e-07 | True |

### instruct: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2544 | 2379 | 154 | 0.5033611059188843 | 0.5033611059188843 | 0.0 | True |
| 0 | 2544 | 455 | 55 | 0.3129996955394745 | 0.3129996955394745 | 0.0 | True |
| 0 | 6507 | 228 | 85 | 0.17396356165409088 | 0.17396356165409088 | 0.0 | True |
| 0 | 6507 | 199 | 71 | 0.16237494349479675 | 0.16237498819828033 | 2.753102333364352e-07 | True |
| 0 | 8206 | 2429 | 21 | 0.5013435482978821 | 0.5013435482978821 | 0.0 | True |
| 0 | 8206 | 2041 | 30 | 0.3213422894477844 | 0.32134220004081726 | 2.782296949359784e-07 | True |
| 0 | 8355 | 229 | 447 | 0.1387219876050949 | 0.1387220025062561 | 1.0741744298147855e-07 | True |
| 0 | 8355 | 361 | 12 | 0.12851642072200775 | 0.12851640582084656 | 1.1594752725085745e-07 | True |
| 1 | 1235 | 1918 | 75 | 0.2568705677986145 | 0.2568705976009369 | 1.160207751440804e-07 | True |
| 1 | 1235 | 1642 | 211 | 0.1867324858903885 | 0.18673256039619446 | 3.9899755853393934e-07 | True |
| 1 | 8093 | 204 | 209 | 0.28803393244743347 | 0.28803393244743347 | 0.0 | True |
| 1 | 8093 | 1536 | 55 | 0.2366783618927002 | 0.23667828738689423 | 3.1479770847415283e-07 | True |
| 1 | 8631 | 1919 | 94 | 0.4626801013946533 | 0.46267983317375183 | 5.797113398236956e-07 | True |
| 1 | 8631 | 673 | 230 | 0.26678478717803955 | 0.26678478717803955 | 0.0 | True |
| 1 | 8503 | 1136 | 10 | 0.1900147646665573 | 0.19001473486423492 | 1.5684214034626823e-07 | True |
| 1 | 8503 | 98 | 51 | 0.17410500347614288 | 0.17410500347614288 | 0.0 | True |
| 2 | 4266 | 546 | 59 | 0.6480739116668701 | 0.6480739116668701 | 0.0 | True |
| 2 | 4266 | 2217 | 76 | 0.3861773610115051 | 0.38617727160453796 | 2.3151788838399126e-07 | True |
| 2 | 5747 | 602 | 12 | 0.3353991210460663 | 0.3353988826274872 | 7.108503396131926e-07 | True |
| 2 | 5747 | 733 | 8 | 0.2692592144012451 | 0.2692592144012451 | 0.0 | True |
| 2 | 8573 | 12 | 126 | 0.7769185900688171 | 0.7769181728363037 | 5.37035049439063e-07 | True |
| 2 | 8573 | 1834 | 2 | 0.4898907244205475 | 0.48989054560661316 | 3.6500779747907364e-07 | True |
| 2 | 8616 | 2411 | 156 | 0.1859765648841858 | 0.1859765648841858 | 0.0 | True |
| 2 | 8616 | 1458 | 314 | 0.1720133274793625 | 0.17201320827007294 | 6.930235656599546e-07 | True |
| 3 | 7360 | 865 | 106 | 1.9305862188339233 | 1.930586338043213 | 6.174771599829601e-08 | True |
| 3 | 7360 | 1793 | 149 | 0.7075418829917908 | 0.7075418829917908 | 0.0 | True |
| 3 | 6045 | 1968 | 53 | 0.586997926235199 | 0.5869976878166199 | 4.0616596489649725e-07 | True |
| 3 | 6045 | 5 | 419 | 0.524733304977417 | 0.5247331857681274 | 2.2718071908149927e-07 | True |
| 3 | 8625 | 2354 | 92 | 0.8594196438789368 | 0.8594197630882263 | 1.3870905837425076e-07 | True |
| 3 | 8625 | 298 | 156 | 0.5474888682365417 | 0.5474890470504761 | 3.266074338682546e-07 | True |
| 3 | 8402 | 1179 | 0 | 0.26668673753738403 | 0.2666868567466736 | 4.470011919286783e-07 | True |
| 3 | 8402 | 327 | 0 | 0.23514170944690704 | 0.23514172434806824 | 6.337098266784612e-08 | True |
| 4 | 1371 | 420 | 203 | 0.5015338063240051 | 0.5015338063240051 | 0.0 | True |
| 4 | 1371 | 1568 | 213 | 0.29920294880867004 | 0.2992030382156372 | 2.9881713238146796e-07 | True |
| 4 | 840 | 685 | 20 | 0.30583426356315613 | 0.3058343231678009 | 1.948919786846643e-07 | True |
| 4 | 840 | 1818 | 60 | 0.2627893090248108 | 0.2627893090248108 | 0.0 | True |
| 4 | 8250 | 1418 | 24 | 0.21551933884620667 | 0.2155192643404007 | 3.4570357522489056e-07 | True |
| 4 | 8250 | 1173 | 12 | 0.17848560214042664 | 0.17848552763462067 | 4.1743314349030546e-07 | True |
| 4 | 8637 | 1995 | 75 | 0.23451180756092072 | 0.23451173305511475 | 3.177059899207139e-07 | True |
| 4 | 8637 | 1160 | 148 | 0.2112431824207306 | 0.21124328672885895 | 4.937822236988662e-07 | True |
| 5 | 2123 | 279 | 9 | 1.0760324001312256 | 1.0760326385498047 | 2.2157193321733305e-07 | True |
| 5 | 2123 | 2381 | 98 | 0.6647359132766724 | 0.6647356748580933 | 3.586666138231189e-07 | True |
| 5 | 529 | 2367 | 137 | 0.7443840503692627 | 0.7443839311599731 | 1.6014487345832534e-07 | True |
| 5 | 529 | 1068 | 255 | 0.6464106440544128 | 0.6464105248451233 | 1.8441727506694117e-07 | True |
| 5 | 8498 | 2189 | 17 | 1.384092926979065 | 1.3840922117233276 | 5.167685806081039e-07 | True |
| 5 | 8498 | 2090 | 58 | 0.7251902222633362 | 0.7251907587051392 | 7.397256423345971e-07 | True |
| 5 | 8692 | 1629 | 243 | 0.524631142616272 | 0.5246313214302063 | 3.4083743758414425e-07 | True |
| 5 | 8692 | 907 | 192 | 0.47167858481407166 | 0.47167864441871643 | 1.2636707854541635e-07 | True |
| 6 | 4186 | 2422 | 68 | 1.1454929113388062 | 1.1454930305480957 | 1.0406811632858925e-07 | True |
| 6 | 4186 | 364 | 94 | 0.8547205328941345 | 0.8547207713127136 | 2.7894331530127516e-07 | True |
| 6 | 2495 | 610 | 187 | 0.2839530408382416 | 0.2839531898498535 | 5.24775545627502e-07 | True |
| 6 | 2495 | 1441 | 205 | 0.2690710425376892 | 0.26907092332839966 | 4.4304020390482343e-07 | True |
| 6 | 8270 | 1874 | 0 | 0.34656035900115967 | 0.3465603291988373 | 8.599460848202662e-08 | True |
| 6 | 8270 | 610 | 0 | 0.3222522735595703 | 0.32225215435028076 | 3.699253638585879e-07 | True |
| 6 | 8516 | 271 | 29 | 1.1099259853363037 | 1.1099259853363037 | 0.0 | True |
| 6 | 8516 | 1834 | 12 | 0.9448937177658081 | 0.9448933005332947 | 4.4156554920724477e-07 | True |
| 7 | 4810 | 991 | 218 | 1.7327535152435303 | 1.7327553033828735 | 1.0319640546280492e-06 | True |
| 7 | 4810 | 896 | 192 | 1.3967106342315674 | 1.3967106342315674 | 0.0 | True |
| 7 | 5675 | 961 | 58 | 0.29670900106430054 | 0.296709269285202 | 9.039863992232949e-07 | True |
| 7 | 5675 | 2041 | 159 | 0.26619666814804077 | 0.26619666814804077 | 0.0 | True |
| 7 | 8702 | 375 | 50 | 0.6337953805923462 | 0.6337942481040955 | 1.7868357602638197e-06 | True |
| 7 | 8702 | 292 | 27 | 0.44517505168914795 | 0.44517526030540466 | 4.686162351693003e-07 | True |
| 7 | 8528 | 522 | 315 | 0.41379937529563904 | 0.4137994647026062 | 2.1606356244305376e-07 | True |
| 7 | 8528 | 1710 | 152 | 0.37719082832336426 | 0.37719088792800903 | 1.5802251884102493e-07 | True |
| 8 | 2466 | 406 | 288 | 1.3175736665725708 | 1.3175742626190186 | 4.5238187653250013e-07 | True |
| 8 | 2466 | 2045 | 255 | 0.8850147724151611 | 0.8850150108337402 | 2.6939502766821633e-07 | True |
| 8 | 7701 | 1518 | 201 | 0.7234298586845398 | 0.7234298586845398 | 0.0 | True |
| 8 | 7701 | 2336 | 246 | 0.6171693205833435 | 0.6171691417694092 | 2.8973237710059594e-07 | True |
| 8 | 8254 | 2233 | 0 | 0.2876960039138794 | 0.28769615292549133 | 5.17948146346456e-07 | True |
| 8 | 8254 | 166 | 0 | 0.24590422213077545 | 0.24590423703193665 | 6.059741904684744e-08 | True |
| 8 | 8536 | 2244 | 172 | 1.2259129285812378 | 1.225911021232605 | 1.555859790972182e-06 | True |
| 8 | 8536 | 1552 | 116 | 0.9488207697868347 | 0.948822557926178 | 1.8845912739277915e-06 | True |
| 9 | 2060 | 666 | 300 | 1.0851430892944336 | 1.0851423740386963 | 6.591349512899271e-07 | True |
| 9 | 2060 | 2465 | 328 | 0.6867629289627075 | 0.6867621541023254 | 1.1282792786303038e-06 | True |
| 9 | 6533 | 1190 | 88 | 0.6564477682113647 | 0.6564475893974304 | 2.7239628647590573e-07 | True |
| 9 | 6533 | 489 | 115 | 0.5515286922454834 | 0.5515283346176147 | 6.484302152918001e-07 | True |
| 9 | 8543 | 166 | 0 | 0.9236868619918823 | 0.923687219619751 | 3.8717435893928164e-07 | True |
| 9 | 8543 | 2322 | 0 | 0.834407389163971 | 0.834407389163971 | 0.0 | True |
| 9 | 8607 | 1287 | 95 | 0.524832546710968 | 0.5248326063156128 | 1.135688804913535e-07 | True |
| 9 | 8607 | 380 | 113 | 0.4880147874355316 | 0.4880148470401764 | 1.2213696451414313e-07 | True |
| 10 | 6424 | 657 | 249 | 2.950662612915039 | 2.950660467147827 | 7.272153727512077e-07 | True |
| 10 | 6424 | 169 | 194 | 1.8443620204925537 | 1.8443620204925537 | 0.0 | True |
| 10 | 7847 | 799 | 100 | 0.6372507810592651 | 0.6372503042221069 | 7.482723793770894e-07 | True |
| 10 | 7847 | 1059 | 299 | 0.5941333174705505 | 0.5941331386566162 | 3.009660106042364e-07 | True |
| 10 | 8490 | 541 | 177 | 1.4761873483657837 | 1.4761873483657837 | 0.0 | True |
| 10 | 8490 | 463 | 419 | 0.9345216155052185 | 0.9345216155052185 | 0.0 | True |
| 10 | 8591 | 1570 | 296 | 0.728171169757843 | 0.7281717658042908 | 8.185526597436211e-07 | True |
| 10 | 8591 | 1766 | 77 | 0.6810973882675171 | 0.6810979843139648 | 8.751266089421487e-07 | True |
| 11 | 8136 | 75 | 90 | 2.604203939437866 | 2.6042017936706543 | 8.239628162060303e-07 | True |
| 11 | 8136 | 1672 | 22 | 1.641998052597046 | 1.6419965028762817 | 9.438018283329019e-07 | True |
| 11 | 693 | 770 | 147 | 1.2014538049697876 | 1.2014521360397339 | 1.3890921538618003e-06 | True |
| 11 | 693 | 2162 | 12 | 1.1103897094726562 | 1.1103888750076294 | 7.515064483547591e-07 | True |
| 11 | 8353 | 1184 | 180 | 0.573865532875061 | 0.5738654136657715 | 2.0773035270744316e-07 | True |
| 11 | 8353 | 138 | 247 | 0.4869057238101959 | 0.4869062602519989 | 1.1017364897267662e-06 | True |
| 11 | 8228 | 1007 | 28 | 0.4032048285007477 | 0.4032049775123596 | 3.6956802450147307e-07 | True |
| 11 | 8228 | 1476 | 152 | 0.3865886330604553 | 0.38658902049064636 | 1.0021768823695655e-06 | True |
| 12 | 2677 | 1691 | 389 | 3.8317599296569824 | 3.8317606449127197 | 1.866650704729085e-07 | True |
| 12 | 2677 | 2416 | 209 | 1.5121257305145264 | 1.5121266841888428 | 6.306845371130257e-07 | True |
| 12 | 538 | 1141 | 372 | 5.832169055938721 | 5.832168102264404 | 1.6351966262623207e-07 | True |
| 12 | 538 | 235 | 404 | 5.1775946617126465 | 5.177594184875488 | 9.209627044180331e-08 | True |
| 12 | 8315 | 2425 | 154 | 0.7385175824165344 | 0.7385178208351135 | 3.2283399173980806e-07 | True |
| 12 | 8315 | 2090 | 27 | 0.6179611086845398 | 0.6179611086845398 | 0.0 | True |
| 12 | 8283 | 2449 | 280 | 0.5527958869934082 | 0.5527958273887634 | 1.0782396573095591e-07 | True |
| 12 | 8283 | 21 | 26 | 0.5240377187728882 | 0.5240374803543091 | 4.5496453892642475e-07 | True |
| 13 | 4990 | 2396 | 9 | 2.1577508449554443 | 2.157749652862549 | 5.524701326359245e-07 | True |
| 13 | 4990 | 896 | 72 | 1.4759176969528198 | 1.4759176969528198 | 0.0 | True |
| 13 | 4713 | 2452 | 199 | 4.39117431640625 | 4.391178607940674 | 9.773090555285287e-07 | True |
| 13 | 4713 | 83 | 255 | 3.8012256622314453 | 3.80122709274292 | 3.7632900588428035e-07 | True |
| 13 | 8577 | 907 | 116 | 5.873417377471924 | 5.873417377471924 | 0.0 | True |
| 13 | 8577 | 582 | 310 | 3.7584445476531982 | 3.7584447860717773 | 6.343543880418109e-08 | True |
| 13 | 8601 | 1831 | 252 | 1.0989658832550049 | 1.0989669561386108 | 9.762665268363737e-07 | True |
| 13 | 8601 | 1370 | 430 | 1.0373742580413818 | 1.0373739004135132 | 3.447433420292926e-07 | True |
| 14 | 1070 | 2416 | 104 | 4.805600166320801 | 4.805600643157959 | 9.922530832776183e-08 | True |
| 14 | 1070 | 2379 | 119 | 2.905299425125122 | 2.905299425125122 | 0.0 | True |
| 14 | 4850 | 1017 | 270 | 2.4850900173187256 | 2.485088348388672 | 6.715773038723243e-07 | True |
| 14 | 4850 | 1687 | 105 | 2.2345683574676514 | 2.2345683574676514 | 0.0 | True |
| 14 | 8213 | 1482 | 68 | 1.837538480758667 | 1.8375390768051147 | 3.243722262120006e-07 | True |
| 14 | 8213 | 1536 | 162 | 1.7190529108047485 | 1.7190533876419067 | 2.773836425895122e-07 | True |
| 14 | 8644 | 578 | 3 | 4.021915435791016 | 4.021915912628174 | 1.1855971758126795e-07 | True |
| 14 | 8644 | 581 | 5 | 3.7117178440093994 | 3.711719036102295 | 3.211701281205452e-07 | True |
| 15 | 467 | 841 | 337 | 22.22728729248047 | 22.227283477783203 | 1.7162225940704508e-07 | True |
| 15 | 467 | 2302 | 155 | 12.84566879272461 | 12.845671653747559 | 2.22722771027628e-07 | True |
| 15 | 1851 | 1739 | 231 | 5.247389793395996 | 5.247385501861572 | 8.17841744714516e-07 | True |
| 15 | 1851 | 1700 | 177 | 4.735748291015625 | 4.735749244689941 | 2.0137774598694428e-07 | True |
| 15 | 8466 | 797 | 92 | 8.581412315368652 | 8.581414222717285 | 2.2226511939026455e-07 | True |
| 15 | 8466 | 2228 | 357 | 5.183493137359619 | 5.183493137359619 | 0.0 | True |
| 15 | 8315 | 728 | 123 | 2.458916187286377 | 2.4589154720306396 | 2.9088252011307186e-07 | True |
| 15 | 8315 | 2285 | 142 | 2.361499071121216 | 2.361499071121216 | 0.0 | True |

### instruct_b: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2544 | 2379 | 154 | 0.5033611059188843 | 0.5033611059188843 | 0.0 | True |
| 0 | 2544 | 455 | 55 | 0.3129996955394745 | 0.3129996955394745 | 0.0 | True |
| 0 | 6507 | 228 | 85 | 0.17396356165409088 | 0.17396356165409088 | 0.0 | True |
| 0 | 6507 | 199 | 71 | 0.16237494349479675 | 0.16237498819828033 | 2.753102333364352e-07 | True |
| 0 | 8206 | 2429 | 21 | 0.5013435482978821 | 0.5013435482978821 | 0.0 | True |
| 0 | 8206 | 2041 | 30 | 0.3213422894477844 | 0.32134220004081726 | 2.782296949359784e-07 | True |
| 0 | 8355 | 229 | 447 | 0.1387219876050949 | 0.1387220025062561 | 1.0741744298147855e-07 | True |
| 0 | 8355 | 361 | 12 | 0.12851642072200775 | 0.12851640582084656 | 1.1594752725085745e-07 | True |
| 1 | 1235 | 1918 | 75 | 0.2568705677986145 | 0.2568705976009369 | 1.160207751440804e-07 | True |
| 1 | 1235 | 1642 | 211 | 0.1867324858903885 | 0.18673256039619446 | 3.9899755853393934e-07 | True |
| 1 | 8093 | 204 | 209 | 0.28803393244743347 | 0.28803393244743347 | 0.0 | True |
| 1 | 8093 | 1536 | 55 | 0.2366783618927002 | 0.23667828738689423 | 3.1479770847415283e-07 | True |
| 1 | 8631 | 1919 | 94 | 0.4626801013946533 | 0.46267983317375183 | 5.797113398236956e-07 | True |
| 1 | 8631 | 673 | 230 | 0.26678478717803955 | 0.26678478717803955 | 0.0 | True |
| 1 | 8503 | 1136 | 10 | 0.1900147646665573 | 0.19001473486423492 | 1.5684214034626823e-07 | True |
| 1 | 8503 | 98 | 51 | 0.17410500347614288 | 0.17410500347614288 | 0.0 | True |
| 2 | 4266 | 546 | 59 | 0.6480739116668701 | 0.6480739116668701 | 0.0 | True |
| 2 | 4266 | 2217 | 76 | 0.3861773610115051 | 0.38617727160453796 | 2.3151788838399126e-07 | True |
| 2 | 5747 | 602 | 12 | 0.3353991210460663 | 0.3353988826274872 | 7.108503396131926e-07 | True |
| 2 | 5747 | 733 | 8 | 0.2692592144012451 | 0.2692592144012451 | 0.0 | True |
| 2 | 8573 | 12 | 126 | 0.7769185900688171 | 0.7769181728363037 | 5.37035049439063e-07 | True |
| 2 | 8573 | 1834 | 2 | 0.4898907244205475 | 0.48989054560661316 | 3.6500779747907364e-07 | True |
| 2 | 8616 | 2411 | 156 | 0.1859765648841858 | 0.1859765648841858 | 0.0 | True |
| 2 | 8616 | 1458 | 314 | 0.1720133274793625 | 0.17201320827007294 | 6.930235656599546e-07 | True |
| 3 | 7360 | 865 | 106 | 1.9305862188339233 | 1.930586338043213 | 6.174771599829601e-08 | True |
| 3 | 7360 | 1793 | 149 | 0.7075418829917908 | 0.7075418829917908 | 0.0 | True |
| 3 | 6045 | 1968 | 53 | 0.586997926235199 | 0.5869976878166199 | 4.0616596489649725e-07 | True |
| 3 | 6045 | 5 | 419 | 0.524733304977417 | 0.5247331857681274 | 2.2718071908149927e-07 | True |
| 3 | 8625 | 2354 | 92 | 0.8594196438789368 | 0.8594197630882263 | 1.3870905837425076e-07 | True |
| 3 | 8625 | 298 | 156 | 0.5474888682365417 | 0.5474890470504761 | 3.266074338682546e-07 | True |
| 3 | 8402 | 1179 | 0 | 0.26668673753738403 | 0.2666868567466736 | 4.470011919286783e-07 | True |
| 3 | 8402 | 327 | 0 | 0.23514170944690704 | 0.23514172434806824 | 6.337098266784612e-08 | True |
| 4 | 1371 | 420 | 203 | 0.5015338063240051 | 0.5015338063240051 | 0.0 | True |
| 4 | 1371 | 1568 | 213 | 0.29920294880867004 | 0.2992030382156372 | 2.9881713238146796e-07 | True |
| 4 | 840 | 685 | 20 | 0.30583426356315613 | 0.3058343231678009 | 1.948919786846643e-07 | True |
| 4 | 840 | 1818 | 60 | 0.2627893090248108 | 0.2627893090248108 | 0.0 | True |
| 4 | 8250 | 1418 | 24 | 0.21551933884620667 | 0.2155192643404007 | 3.4570357522489056e-07 | True |
| 4 | 8250 | 1173 | 12 | 0.17848560214042664 | 0.17848552763462067 | 4.1743314349030546e-07 | True |
| 4 | 8637 | 1995 | 75 | 0.23451180756092072 | 0.23451173305511475 | 3.177059899207139e-07 | True |
| 4 | 8637 | 1160 | 148 | 0.2112431824207306 | 0.21124328672885895 | 4.937822236988662e-07 | True |
| 5 | 2123 | 279 | 9 | 1.0760324001312256 | 1.0760326385498047 | 2.2157193321733305e-07 | True |
| 5 | 2123 | 2381 | 98 | 0.6647359132766724 | 0.6647356748580933 | 3.586666138231189e-07 | True |
| 5 | 529 | 2367 | 137 | 0.7443840503692627 | 0.7443839311599731 | 1.6014487345832534e-07 | True |
| 5 | 529 | 1068 | 255 | 0.6464106440544128 | 0.6464105248451233 | 1.8441727506694117e-07 | True |
| 5 | 8498 | 2189 | 17 | 1.384092926979065 | 1.3840922117233276 | 5.167685806081039e-07 | True |
| 5 | 8498 | 2090 | 58 | 0.7251902222633362 | 0.7251907587051392 | 7.397256423345971e-07 | True |
| 5 | 8692 | 1629 | 243 | 0.524631142616272 | 0.5246313214302063 | 3.4083743758414425e-07 | True |
| 5 | 8692 | 907 | 192 | 0.47167858481407166 | 0.47167864441871643 | 1.2636707854541635e-07 | True |
| 6 | 4186 | 2422 | 68 | 1.1454929113388062 | 1.1454930305480957 | 1.0406811632858925e-07 | True |
| 6 | 4186 | 364 | 94 | 0.8547205328941345 | 0.8547207713127136 | 2.7894331530127516e-07 | True |
| 6 | 2495 | 610 | 187 | 0.2839530408382416 | 0.2839531898498535 | 5.24775545627502e-07 | True |
| 6 | 2495 | 1441 | 205 | 0.2690710425376892 | 0.26907092332839966 | 4.4304020390482343e-07 | True |
| 6 | 8270 | 1874 | 0 | 0.34656035900115967 | 0.3465603291988373 | 8.599460848202662e-08 | True |
| 6 | 8270 | 610 | 0 | 0.3222522735595703 | 0.32225215435028076 | 3.699253638585879e-07 | True |
| 6 | 8516 | 271 | 29 | 1.1099259853363037 | 1.1099259853363037 | 0.0 | True |
| 6 | 8516 | 1834 | 12 | 0.9448937177658081 | 0.9448933005332947 | 4.4156554920724477e-07 | True |
| 7 | 4810 | 991 | 218 | 1.7327535152435303 | 1.7327553033828735 | 1.0319640546280492e-06 | True |
| 7 | 4810 | 896 | 192 | 1.3967106342315674 | 1.3967106342315674 | 0.0 | True |
| 7 | 5675 | 961 | 58 | 0.29670900106430054 | 0.296709269285202 | 9.039863992232949e-07 | True |
| 7 | 5675 | 2041 | 159 | 0.26619666814804077 | 0.26619666814804077 | 0.0 | True |
| 7 | 8702 | 375 | 50 | 0.6337953805923462 | 0.6337942481040955 | 1.7868357602638197e-06 | True |
| 7 | 8702 | 292 | 27 | 0.44517505168914795 | 0.44517526030540466 | 4.686162351693003e-07 | True |
| 7 | 8528 | 522 | 315 | 0.41379937529563904 | 0.4137994647026062 | 2.1606356244305376e-07 | True |
| 7 | 8528 | 1710 | 152 | 0.37719082832336426 | 0.37719088792800903 | 1.5802251884102493e-07 | True |
| 8 | 2466 | 406 | 288 | 1.3175736665725708 | 1.3175742626190186 | 4.5238187653250013e-07 | True |
| 8 | 2466 | 2045 | 255 | 0.8850147724151611 | 0.8850150108337402 | 2.6939502766821633e-07 | True |
| 8 | 7701 | 1518 | 201 | 0.7234298586845398 | 0.7234298586845398 | 0.0 | True |
| 8 | 7701 | 2336 | 246 | 0.6171693205833435 | 0.6171691417694092 | 2.8973237710059594e-07 | True |
| 8 | 8254 | 2233 | 0 | 0.2876960039138794 | 0.28769615292549133 | 5.17948146346456e-07 | True |
| 8 | 8254 | 166 | 0 | 0.24590422213077545 | 0.24590423703193665 | 6.059741904684744e-08 | True |
| 8 | 8536 | 2244 | 172 | 1.2259129285812378 | 1.225911021232605 | 1.555859790972182e-06 | True |
| 8 | 8536 | 1552 | 116 | 0.9488207697868347 | 0.948822557926178 | 1.8845912739277915e-06 | True |
| 9 | 2060 | 666 | 300 | 1.0851430892944336 | 1.0851423740386963 | 6.591349512899271e-07 | True |
| 9 | 2060 | 2465 | 328 | 0.6867629289627075 | 0.6867621541023254 | 1.1282792786303038e-06 | True |
| 9 | 6533 | 1190 | 88 | 0.6564477682113647 | 0.6564475893974304 | 2.7239628647590573e-07 | True |
| 9 | 6533 | 489 | 115 | 0.5515286922454834 | 0.5515283346176147 | 6.484302152918001e-07 | True |
| 9 | 8543 | 166 | 0 | 0.9236868619918823 | 0.923687219619751 | 3.8717435893928164e-07 | True |
| 9 | 8543 | 2322 | 0 | 0.834407389163971 | 0.834407389163971 | 0.0 | True |
| 9 | 8607 | 1287 | 95 | 0.524832546710968 | 0.5248326063156128 | 1.135688804913535e-07 | True |
| 9 | 8607 | 380 | 113 | 0.4880147874355316 | 0.4880148470401764 | 1.2213696451414313e-07 | True |
| 10 | 6424 | 657 | 249 | 2.950662612915039 | 2.950660467147827 | 7.272153727512077e-07 | True |
| 10 | 6424 | 169 | 194 | 1.8443620204925537 | 1.8443620204925537 | 0.0 | True |
| 10 | 7847 | 799 | 100 | 0.6372507810592651 | 0.6372503042221069 | 7.482723793770894e-07 | True |
| 10 | 7847 | 1059 | 299 | 0.5941333174705505 | 0.5941331386566162 | 3.009660106042364e-07 | True |
| 10 | 8490 | 541 | 177 | 1.4761873483657837 | 1.4761873483657837 | 0.0 | True |
| 10 | 8490 | 463 | 419 | 0.9345216155052185 | 0.9345216155052185 | 0.0 | True |
| 10 | 8591 | 1570 | 296 | 0.728171169757843 | 0.7281717658042908 | 8.185526597436211e-07 | True |
| 10 | 8591 | 1766 | 77 | 0.6810973882675171 | 0.6810979843139648 | 8.751266089421487e-07 | True |
| 11 | 8136 | 75 | 90 | 2.604203939437866 | 2.6042017936706543 | 8.239628162060303e-07 | True |
| 11 | 8136 | 1672 | 22 | 1.641998052597046 | 1.6419965028762817 | 9.438018283329019e-07 | True |
| 11 | 693 | 770 | 147 | 1.2014538049697876 | 1.2014521360397339 | 1.3890921538618003e-06 | True |
| 11 | 693 | 2162 | 12 | 1.1103897094726562 | 1.1103888750076294 | 7.515064483547591e-07 | True |
| 11 | 8353 | 1184 | 180 | 0.573865532875061 | 0.5738654136657715 | 2.0773035270744316e-07 | True |
| 11 | 8353 | 138 | 247 | 0.4869057238101959 | 0.4869062602519989 | 1.1017364897267662e-06 | True |
| 11 | 8228 | 1007 | 28 | 0.4032048285007477 | 0.4032049775123596 | 3.6956802450147307e-07 | True |
| 11 | 8228 | 1476 | 152 | 0.3865886330604553 | 0.38658902049064636 | 1.0021768823695655e-06 | True |
| 12 | 2677 | 1691 | 389 | 3.8317599296569824 | 3.8317606449127197 | 1.866650704729085e-07 | True |
| 12 | 2677 | 2416 | 209 | 1.5121257305145264 | 1.5121266841888428 | 6.306845371130257e-07 | True |
| 12 | 538 | 1141 | 372 | 5.832169055938721 | 5.832168102264404 | 1.6351966262623207e-07 | True |
| 12 | 538 | 235 | 404 | 5.1775946617126465 | 5.177594184875488 | 9.209627044180331e-08 | True |
| 12 | 8315 | 2425 | 154 | 0.7385175824165344 | 0.7385178208351135 | 3.2283399173980806e-07 | True |
| 12 | 8315 | 2090 | 27 | 0.6179611086845398 | 0.6179611086845398 | 0.0 | True |
| 12 | 8283 | 2449 | 280 | 0.5527958869934082 | 0.5527958273887634 | 1.0782396573095591e-07 | True |
| 12 | 8283 | 21 | 26 | 0.5240377187728882 | 0.5240374803543091 | 4.5496453892642475e-07 | True |
| 13 | 4990 | 2396 | 9 | 2.1577508449554443 | 2.157749652862549 | 5.524701326359245e-07 | True |
| 13 | 4990 | 896 | 72 | 1.4759176969528198 | 1.4759176969528198 | 0.0 | True |
| 13 | 4713 | 2452 | 199 | 4.39117431640625 | 4.391178607940674 | 9.773090555285287e-07 | True |
| 13 | 4713 | 83 | 255 | 3.8012256622314453 | 3.80122709274292 | 3.7632900588428035e-07 | True |
| 13 | 8577 | 907 | 116 | 5.873417377471924 | 5.873417377471924 | 0.0 | True |
| 13 | 8577 | 582 | 310 | 3.7584445476531982 | 3.7584447860717773 | 6.343543880418109e-08 | True |
| 13 | 8601 | 1831 | 252 | 1.0989658832550049 | 1.0989669561386108 | 9.762665268363737e-07 | True |
| 13 | 8601 | 1370 | 430 | 1.0373742580413818 | 1.0373739004135132 | 3.447433420292926e-07 | True |
| 14 | 1070 | 2416 | 104 | 4.805600166320801 | 4.805600643157959 | 9.922530832776183e-08 | True |
| 14 | 1070 | 2379 | 119 | 2.905299425125122 | 2.905299425125122 | 0.0 | True |
| 14 | 4850 | 1017 | 270 | 2.4850900173187256 | 2.485088348388672 | 6.715773038723243e-07 | True |
| 14 | 4850 | 1687 | 105 | 2.2345683574676514 | 2.2345683574676514 | 0.0 | True |
| 14 | 8213 | 1482 | 68 | 1.837538480758667 | 1.8375390768051147 | 3.243722262120006e-07 | True |
| 14 | 8213 | 1536 | 162 | 1.7190529108047485 | 1.7190533876419067 | 2.773836425895122e-07 | True |
| 14 | 8644 | 578 | 3 | 4.021915435791016 | 4.021915912628174 | 1.1855971758126795e-07 | True |
| 14 | 8644 | 581 | 5 | 3.7117178440093994 | 3.711719036102295 | 3.211701281205452e-07 | True |
| 15 | 467 | 841 | 337 | 22.22728729248047 | 22.227283477783203 | 1.7162225940704508e-07 | True |
| 15 | 467 | 2302 | 155 | 12.84566879272461 | 12.845671653747559 | 2.22722771027628e-07 | True |
| 15 | 1851 | 1739 | 231 | 5.247389793395996 | 5.247385501861572 | 8.17841744714516e-07 | True |
| 15 | 1851 | 1700 | 177 | 4.735748291015625 | 4.735749244689941 | 2.0137774598694428e-07 | True |
| 15 | 8466 | 797 | 92 | 8.581412315368652 | 8.581414222717285 | 2.2226511939026455e-07 | True |
| 15 | 8466 | 2228 | 357 | 5.183493137359619 | 5.183493137359619 | 0.0 | True |
| 15 | 8315 | 728 | 123 | 2.458916187286377 | 2.4589154720306396 | 2.9088252011307186e-07 | True |
| 15 | 8315 | 2285 | 142 | 2.361499071121216 | 2.361499071121216 | 0.0 | True |

### midtrain_13B: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2470 | 2371 | 223 | 0.20269936323165894 | 0.20269916951656342 | 9.55676882411458e-07 | True |
| 0 | 2470 | 2254 | 53 | 0.16207118332386017 | 0.16207118332386017 | 0.0 | True |
| 0 | 2988 | 1069 | 224 | 0.4405169188976288 | 0.4405169188976288 | 0.0 | True |
| 0 | 2988 | 303 | 263 | 0.37187787890434265 | 0.37187787890434265 | 0.0 | True |
| 0 | 8206 | 2429 | 21 | 0.5883720517158508 | 0.5883716344833374 | 7.091304085756153e-07 | True |
| 0 | 8206 | 2041 | 30 | 0.3527366518974304 | 0.35273706912994385 | 1.1828442300604992e-06 | True |
| 0 | 8313 | 2322 | 272 | 0.21916981041431427 | 0.21916987001895905 | 2.7195645542018394e-07 | True |
| 0 | 8313 | 326 | 202 | 0.1998421847820282 | 0.19984205067157745 | 6.71081788316645e-07 | True |
| 1 | 1081 | 1577 | 111 | 0.30265364050865173 | 0.30265364050865173 | 0.0 | True |
| 1 | 1081 | 1785 | 392 | 0.20861151814460754 | 0.20861148834228516 | 1.4286038782880924e-07 | True |
| 1 | 4589 | 561 | 66 | 0.2359563112258911 | 0.23595646023750305 | 6.315220439084647e-07 | True |
| 1 | 4589 | 1653 | 100 | 0.2150125801563263 | 0.2150125801563263 | 0.0 | True |
| 1 | 8631 | 1919 | 94 | 0.5324447154998779 | 0.5324448943138123 | 3.3583568231735574e-07 | True |
| 1 | 8631 | 673 | 230 | 0.35860174894332886 | 0.35860174894332886 | 0.0 | True |
| 1 | 8526 | 1385 | 38 | 0.2188994139432907 | 0.21889938414096832 | 1.3614619541839462e-07 | True |
| 1 | 8526 | 394 | 87 | 0.19257067143917084 | 0.19257071614265442 | 2.3214066424265386e-07 | True |
| 2 | 4266 | 546 | 59 | 0.7277499437332153 | 0.7277499437332153 | 0.0 | True |
| 2 | 4266 | 777 | 57 | 0.4442693889141083 | 0.4442692697048187 | 2.6832658860912043e-07 | True |
| 2 | 403 | 840 | 32 | 0.35417506098747253 | 0.3541750907897949 | 8.414573941091082e-08 | True |
| 2 | 403 | 1604 | 342 | 0.321119099855423 | 0.3211190104484558 | 2.784230748134867e-07 | True |
| 2 | 8319 | 927 | 17 | 0.23827846348285675 | 0.23827853798866272 | 3.126837603374033e-07 | True |
| 2 | 8319 | 1629 | 140 | 0.18234142661094666 | 0.18234144151210785 | 8.172120549238415e-08 | True |
| 2 | 8466 | 755 | 181 | 0.23181086778640747 | 0.23181092739105225 | 2.57126188019411e-07 | True |
| 2 | 8466 | 1622 | 283 | 0.22236286103725433 | 0.22236280143260956 | 2.680512586380356e-07 | True |
| 3 | 1778 | 846 | 127 | 0.5006361603736877 | 0.500636100769043 | 1.1905780982919847e-07 | True |
| 3 | 1778 | 2355 | 107 | 0.3243441581726074 | 0.3243439197540283 | 7.350789989400161e-07 | True |
| 3 | 7430 | 685 | 75 | 0.26276901364326477 | 0.26276901364326477 | 0.0 | True |
| 3 | 7430 | 37 | 15 | 0.2372097671031952 | 0.23720993101596832 | 6.910034739885562e-07 | True |
| 3 | 8610 | 2293 | 256 | 1.6455960273742676 | 1.6455961465835571 | 7.24414057689438e-08 | True |
| 3 | 8610 | 2472 | 21 | 0.8872690200805664 | 0.887269139289856 | 1.3435529343733507e-07 | True |
| 3 | 8294 | 1088 | 20 | 0.17169497907161713 | 0.171694815158844 | 9.546742369440704e-07 | True |
| 3 | 8294 | 1148 | 352 | 0.16134333610534668 | 0.16134335100650787 | 9.235684319876818e-08 | True |
| 4 | 1717 | 2299 | 90 | 0.4127306044101715 | 0.41273072361946106 | 2.8883074886375786e-07 | True |
| 4 | 1717 | 1065 | 43 | 0.30314263701438904 | 0.3031426668167114 | 9.831121969913031e-08 | True |
| 4 | 6638 | 253 | 94 | 1.1422405242919922 | 1.1422405242919922 | 0.0 | True |
| 4 | 6638 | 2093 | 430 | 0.9039425849914551 | 0.9039430618286133 | 5.275082357223303e-07 | True |
| 4 | 8319 | 208 | 0 | 0.24550753831863403 | 0.24550749361515045 | 1.8208599168765308e-07 | True |
| 4 | 8319 | 896 | 131 | 0.2079443484544754 | 0.20794442296028137 | 3.5829685453341185e-07 | True |
| 4 | 8582 | 1801 | 59 | 0.4207165539264679 | 0.4207163453102112 | 4.958593969428852e-07 | True |
| 4 | 8582 | 1885 | 103 | 0.4002828598022461 | 0.4002828896045685 | 7.44531564564586e-08 | True |
| 5 | 1573 | 994 | 288 | 0.702568531036377 | 0.702568531036377 | 0.0 | True |
| 5 | 1573 | 261 | 64 | 0.48670557141304016 | 0.48670560121536255 | 6.123275371837427e-08 | True |
| 5 | 1237 | 2447 | 13 | 0.3902427554130554 | 0.3902427554130554 | 0.0 | True |
| 5 | 1237 | 1700 | 50 | 0.3714481294155121 | 0.3714485764503479 | 1.2034919559801155e-06 | True |
| 5 | 8498 | 2189 | 17 | 1.5036201477050781 | 1.5036200284957886 | 7.928151916075755e-08 | True |
| 5 | 8498 | 2299 | 116 | 0.9346557259559631 | 0.9346568584442139 | 1.211663524100402e-06 | True |
| 5 | 8686 | 279 | 5 | 0.7392499446868896 | 0.7392496466636658 | 4.031427070355498e-07 | True |
| 5 | 8686 | 163 | 52 | 0.5487387180328369 | 0.5487387776374817 | 1.0862117582857319e-07 | True |
| 6 | 7150 | 1386 | 5 | 2.221508264541626 | 2.2215094566345215 | 5.36614206904057e-07 | True |
| 6 | 7150 | 1869 | 7 | 0.93769770860672 | 0.9376981854438782 | 5.085190609153076e-07 | True |
| 6 | 1386 | 280 | 106 | 0.8501526713371277 | 0.8501524329185486 | 2.80442074864713e-07 | True |
| 6 | 1386 | 1316 | 97 | 0.7535697817802429 | 0.7535697817802429 | 0.0 | True |
| 6 | 8499 | 329 | 199 | 0.4815121293067932 | 0.4815121293067932 | 0.0 | True |
| 6 | 8499 | 1851 | 104 | 0.37978747487068176 | 0.3797871172428131 | 9.416526144631721e-07 | True |
| 6 | 8532 | 814 | 125 | 0.29865431785583496 | 0.29865434765815735 | 9.978868747540208e-08 | True |
| 6 | 8532 | 229 | 225 | 0.2680482864379883 | 0.2680482566356659 | 1.1118266333177974e-07 | True |
| 7 | 4305 | 2145 | 350 | 1.5614186525344849 | 1.5614190101623535 | 2.290403461440943e-07 | True |
| 7 | 4305 | 1659 | 301 | 0.9927200675010681 | 0.9927206635475159 | 6.004174462336684e-07 | True |
| 7 | 3830 | 1927 | 324 | 0.6438360214233398 | 0.6438356041908264 | 6.480415813103327e-07 | True |
| 7 | 3830 | 1191 | 69 | 0.5671564340591431 | 0.5671570301055908 | 1.0509383513257482e-06 | True |
| 7 | 8233 | 1393 | 188 | 0.46894755959510803 | 0.46894773840904236 | 3.813090198839308e-07 | True |
| 7 | 8233 | 2355 | 70 | 0.27342456579208374 | 0.27342453598976135 | 1.0899650622598944e-07 | True |
| 7 | 8272 | 1362 | 89 | 0.26846012473106384 | 0.2684599757194519 | 5.550605032600369e-07 | True |
| 7 | 8272 | 1160 | 89 | 0.253531277179718 | 0.25353118777275085 | 3.526466957357256e-07 | True |
| 8 | 4106 | 2012 | 25 | 1.1967276334762573 | 1.1967276334762573 | 0.0 | True |
| 8 | 4106 | 1237 | 126 | 0.5399690270423889 | 0.5399690270423889 | 0.0 | True |
| 8 | 2417 | 1541 | 232 | 1.2745413780212402 | 1.2745404243469238 | 7.482490037999826e-07 | True |
| 8 | 2417 | 564 | 150 | 0.9282442927360535 | 0.9282453060150146 | 1.091608070322676e-06 | True |
| 8 | 8603 | 588 | 146 | 1.2652426958084106 | 1.2652438879013062 | 9.421851629391466e-07 | True |
| 8 | 8603 | 1207 | 61 | 0.9512317776679993 | 0.9512320160865784 | 2.5064194100628104e-07 | True |
| 8 | 8287 | 918 | 52 | 0.3049011528491974 | 0.3049013912677765 | 7.819536819510917e-07 | True |
| 8 | 8287 | 118 | 300 | 0.2767988443374634 | 0.27679893374443054 | 3.230033975650712e-07 | True |
| 9 | 1199 | 1349 | 226 | 0.6458989381790161 | 0.6458989381790161 | 0.0 | True |
| 9 | 1199 | 353 | 72 | 0.37116310000419617 | 0.37116315960884094 | 1.6058882139608375e-07 | True |
| 9 | 6533 | 1190 | 88 | 0.5819104909896851 | 0.5819096565246582 | 1.4340092501790975e-06 | True |
| 9 | 6533 | 389 | 6 | 0.5504209399223328 | 0.5504209399223328 | 0.0 | True |
| 9 | 8491 | 1287 | 250 | 0.7474931478500366 | 0.7474934458732605 | 3.986969308469742e-07 | True |
| 9 | 8491 | 506 | 29 | 0.5082768797874451 | 0.508277416229248 | 1.0554125601834355e-06 | True |
| 9 | 8374 | 747 | 377 | 0.29372653365135193 | 0.2937265634536743 | 1.0146281991353948e-07 | True |
| 9 | 8374 | 1736 | 82 | 0.27948108315467834 | 0.27948108315467834 | 0.0 | True |
| 10 | 6424 | 1022 | 327 | 2.261580467224121 | 2.2615811824798584 | 3.162636694428994e-07 | True |
| 10 | 6424 | 1981 | 311 | 1.7505364418029785 | 1.7505367994308472 | 2.0429615751615097e-07 | True |
| 10 | 3594 | 293 | 135 | 2.178074359893799 | 2.178074359893799 | 0.0 | True |
| 10 | 3594 | 648 | 92 | 1.9303704500198364 | 1.9303717613220215 | 6.793007969248176e-07 | True |
| 10 | 8520 | 1686 | 60 | 1.4358172416687012 | 1.4358172416687012 | 0.0 | True |
| 10 | 8520 | 1391 | 316 | 0.8321147561073303 | 0.8321150541305542 | 3.58151591099188e-07 | True |
| 10 | 8478 | 2255 | 283 | 1.3274189233779907 | 1.3274191617965698 | 1.7961065260003932e-07 | True |
| 10 | 8478 | 838 | 208 | 1.1730711460113525 | 1.1730711460113525 | 0.0 | True |
| 11 | 8167 | 24 | 124 | 4.327548027038574 | 4.327549457550049 | 3.30559352703083e-07 | True |
| 11 | 8167 | 341 | 58 | 2.9538092613220215 | 2.953813076019287 | 1.2914500999017369e-06 | True |
| 11 | 1673 | 1390 | 33 | 0.9376563429832458 | 0.9376553297042847 | 1.0806506762997987e-06 | True |
| 11 | 1673 | 648 | 265 | 0.8612478971481323 | 0.861247718334198 | 2.0762191108771598e-07 | True |
| 11 | 8507 | 1018 | 28 | 1.5949015617370605 | 1.5949018001556396 | 1.49487958894399e-07 | True |
| 11 | 8507 | 334 | 81 | 1.0034866333007812 | 1.003485918045044 | 7.127705677075018e-07 | True |
| 11 | 8592 | 1898 | 168 | 1.9071478843688965 | 1.907147765159607 | 6.250657881742054e-08 | True |
| 11 | 8592 | 2016 | 55 | 1.6792500019073486 | 1.6792494058609009 | 3.5494801076486325e-07 | True |
| 12 | 6320 | 326 | 202 | 1.8903305530548096 | 1.8903305530548096 | 0.0 | True |
| 12 | 6320 | 131 | 54 | 1.4038255214691162 | 1.403826117515564 | 4.2458727145104056e-07 | True |
| 12 | 538 | 1141 | 372 | 7.010598659515381 | 7.010597229003906 | 2.0404983141743012e-07 | True |
| 12 | 538 | 2254 | 65 | 6.1980109214782715 | 6.198009490966797 | 2.308017027934161e-07 | True |
| 12 | 8388 | 1744 | 135 | 0.6261112093925476 | 0.6261113286018372 | 1.9039635093969642e-07 | True |
| 12 | 8388 | 1625 | 266 | 0.5400912761688232 | 0.540091335773468 | 1.1036031760816525e-07 | True |
| 12 | 8501 | 2449 | 305 | 1.4943736791610718 | 1.494375228881836 | 1.037036977946611e-06 | True |
| 12 | 8501 | 427 | 146 | 1.414182186126709 | 1.4141820669174194 | 8.429556723330147e-08 | True |
| 13 | 4614 | 1363 | 48 | 4.899542808532715 | 4.899542808532715 | 0.0 | True |
| 13 | 4614 | 2233 | 268 | 2.574738025665283 | 2.5747368335723877 | 4.6299580136887487e-07 | True |
| 13 | 4713 | 927 | 58 | 5.282950401306152 | 5.282950401306152 | 0.0 | True |
| 13 | 4713 | 389 | 323 | 4.849117755889893 | 4.849117755889893 | 0.0 | True |
| 13 | 8511 | 484 | 60 | 2.882081985473633 | 2.8820810317993164 | 3.3089770562148874e-07 | True |
| 13 | 8511 | 2486 | 151 | 2.1360292434692383 | 2.1360292434692383 | 0.0 | True |
| 13 | 8601 | 1625 | 64 | 1.3745965957641602 | 1.3745965957641602 | 0.0 | True |
| 13 | 8601 | 1831 | 252 | 1.1493881940841675 | 1.149388074874878 | 1.0371542892500928e-07 | True |
| 14 | 4235 | 1076 | 145 | 12.678433418273926 | 12.678437232971191 | 3.0088080599348546e-07 | True |
| 14 | 4235 | 2035 | 298 | 9.845398902893066 | 9.84539794921875 | 9.686497477781355e-08 | True |
| 14 | 4676 | 689 | 220 | 6.658744812011719 | 6.658740997314453 | 5.728853369997995e-07 | True |
| 14 | 4676 | 1826 | 23 | 6.183859825134277 | 6.183858394622803 | 2.3132986760066358e-07 | True |
| 14 | 8213 | 1319 | 218 | 2.040728807449341 | 2.0407285690307617 | 1.1683011394324183e-07 | True |
| 14 | 8213 | 867 | 136 | 1.8847535848617554 | 1.8847534656524658 | 6.324927062522347e-08 | True |
| 14 | 8579 | 2034 | 79 | 3.669987201690674 | 3.6699862480163574 | 2.59857668159419e-07 | True |
| 14 | 8579 | 978 | 171 | 3.322538137435913 | 3.322538137435913 | 0.0 | True |
| 15 | 4036 | 434 | 81 | 10.207719802856445 | 10.207719802856445 | 0.0 | True |
| 15 | 4036 | 273 | 22 | 7.59725284576416 | 7.59725284576416 | 0.0 | True |
| 15 | 2531 | 434 | 324 | 7.886922836303711 | 7.886923789978027 | 1.2091842866985617e-07 | True |
| 15 | 2531 | 1996 | 481 | 7.575536727905273 | 7.575531482696533 | 6.92387738140468e-07 | True |
| 15 | 8433 | 1870 | 303 | 3.601240873336792 | 3.601238489151001 | 6.620456322896603e-07 | True |
| 15 | 8433 | 1992 | 18 | 3.0175251960754395 | 3.0175249576568604 | 7.90112968772049e-08 | True |
| 15 | 8354 | 377 | 7 | 2.8060266971588135 | 2.806023359298706 | 1.1895325553393911e-06 | True |
| 15 | 8354 | 622 | 331 | 2.6169533729553223 | 2.616955518722534 | 8.199485837574745e-07 | True |

### midtrain_26B: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 5511 | 1910 | 157 | 0.23449259996414185 | 0.23449258506298065 | 6.354640272710658e-08 | True |
| 0 | 5511 | 2260 | 313 | 0.16518916189670563 | 0.16518913209438324 | 1.804133034244159e-07 | True |
| 0 | 7837 | 625 | 120 | 0.16799432039260864 | 0.16799432039260864 | 0.0 | True |
| 0 | 7837 | 1243 | 191 | 0.16091588139533997 | 0.16091588139533997 | 0.0 | True |
| 0 | 8206 | 2429 | 21 | 0.5311723351478577 | 0.531172513961792 | 3.366401495220888e-07 | True |
| 0 | 8206 | 2041 | 30 | 0.3429592549800873 | 0.3429594039916992 | 4.344877992784548e-07 | True |
| 0 | 8210 | 778 | 192 | 0.36106619238853455 | 0.36106622219085693 | 8.253977529866812e-08 | True |
| 0 | 8210 | 1182 | 226 | 0.30692094564437866 | 0.30692097544670105 | 9.710097277696548e-08 | True |
| 1 | 6897 | 685 | 120 | 0.3366149663925171 | 0.33661511540412903 | 4.426767280594363e-07 | True |
| 1 | 6897 | 766 | 108 | 0.1889936774969101 | 0.1889936774969101 | 0.0 | True |
| 1 | 2533 | 2340 | 4 | 0.5415936708450317 | 0.5415936708450317 | 0.0 | True |
| 1 | 2533 | 1309 | 11 | 0.45505255460739136 | 0.45505228638648987 | 5.894284050787771e-07 | True |
| 1 | 8560 | 2471 | 126 | 0.22492700815200806 | 0.22492697834968567 | 1.324977495257243e-07 | True |
| 1 | 8560 | 86 | 1 | 0.14836116135120392 | 0.14836114645004272 | 1.0043842376357036e-07 | True |
| 1 | 8526 | 1385 | 38 | 0.1995657980442047 | 0.19956573843955994 | 2.986716429344668e-07 | True |
| 1 | 8526 | 394 | 87 | 0.18318700790405273 | 0.18318703770637512 | 1.6268796968016846e-07 | True |
| 2 | 4266 | 546 | 59 | 0.7481515407562256 | 0.7481515407562256 | 0.0 | True |
| 2 | 4266 | 2319 | 4 | 0.4463401734828949 | 0.4463401734828949 | 0.0 | True |
| 2 | 5288 | 617 | 274 | 0.4798472225666046 | 0.47984763979911804 | 8.69511156480271e-07 | True |
| 2 | 5288 | 501 | 17 | 0.44990137219429016 | 0.44990143179893494 | 1.3248380302705617e-07 | True |
| 2 | 8629 | 426 | 25 | 0.25311464071273804 | 0.25311464071273804 | 0.0 | True |
| 2 | 8629 | 1692 | 52 | 0.21372447907924652 | 0.2137245088815689 | 1.3944271856966353e-07 | True |
| 2 | 8352 | 995 | 244 | 0.1331891566514969 | 0.13318918645381927 | 2.237593745388459e-07 | True |
| 2 | 8352 | 2067 | 117 | 0.12737810611724854 | 0.12737806141376495 | 3.509510774198077e-07 | True |
| 3 | 6554 | 1018 | 126 | 1.1925584077835083 | 1.1925582885742188 | 9.996096524307258e-08 | True |
| 3 | 6554 | 2452 | 246 | 0.7336149215698242 | 0.7336152791976929 | 4.8748717908719e-07 | True |
| 3 | 2598 | 1784 | 64 | 1.4489870071411133 | 1.4489872455596924 | 1.6454155760303756e-07 | True |
| 3 | 2598 | 721 | 149 | 0.9768628478050232 | 0.976862907409668 | 6.101639028377441e-08 | True |
| 3 | 8683 | 1076 | 68 | 0.5678933262825012 | 0.5678932666778564 | 1.0495746651148356e-07 | True |
| 3 | 8683 | 2034 | 24 | 0.3647671043872833 | 0.36476704478263855 | 1.6340466028457084e-07 | True |
| 3 | 8402 | 1179 | 0 | 0.23911552131175995 | 0.23911550641059875 | 6.231783328870338e-08 | True |
| 3 | 8402 | 1529 | 0 | 0.21949613094329834 | 0.21949611604213715 | 6.788803579274489e-08 | True |
| 4 | 3075 | 77 | 37 | 1.2957565784454346 | 1.2957557439804077 | 6.439982946925157e-07 | True |
| 4 | 3075 | 1020 | 324 | 0.8425066471099854 | 0.8425067067146301 | 7.074679467497366e-08 | True |
| 4 | 1336 | 2219 | 174 | 0.2752489149570465 | 0.2752489149570465 | 0.0 | True |
| 4 | 1336 | 1832 | 157 | 0.2565169632434845 | 0.2565167546272278 | 8.132649555649455e-07 | True |
| 4 | 8406 | 2355 | 0 | 0.30324670672416687 | 0.3032465875148773 | 3.931099230674053e-07 | True |
| 4 | 8406 | 2170 | 21 | 0.21476183831691742 | 0.2147619128227234 | 3.4692292892041815e-07 | True |
| 4 | 8214 | 486 | 7 | 0.16966921091079712 | 0.1696692258119583 | 8.782478042926645e-08 | True |
| 4 | 8214 | 2052 | 57 | 0.16024553775787354 | 0.16024549305438995 | 2.789686640079093e-07 | True |
| 5 | 646 | 1834 | 87 | 0.6112554669380188 | 0.6112552881240845 | 2.9253551746851466e-07 | True |
| 5 | 646 | 1912 | 163 | 0.3715476393699646 | 0.37154752016067505 | 3.2084523468625745e-07 | True |
| 5 | 3661 | 1089 | 213 | 0.35737183690071106 | 0.35737180709838867 | 8.339303579754472e-08 | True |
| 5 | 3661 | 1994 | 104 | 0.33830785751342773 | 0.33830776810646057 | 2.6427694532497607e-07 | True |
| 5 | 8551 | 1996 | 295 | 1.2625616788864136 | 1.262561321258545 | 2.832557605960079e-07 | True |
| 5 | 8551 | 232 | 232 | 0.9199042320251465 | 0.9199042320251465 | 0.0 | True |
| 5 | 8240 | 2322 | 140 | 0.2283409982919693 | 0.2283410280942917 | 1.3051673860858063e-07 | True |
| 5 | 8240 | 169 | 298 | 0.21113362908363342 | 0.21113371849060059 | 4.2346151842855125e-07 | True |
| 6 | 7150 | 1386 | 5 | 2.2356529235839844 | 2.235652208328247 | 3.1993147494382005e-07 | True |
| 6 | 7150 | 248 | 54 | 0.9053430557250977 | 0.9053415656089783 | 1.645913236934609e-06 | True |
| 6 | 1386 | 280 | 106 | 0.8594321012496948 | 0.8594326972961426 | 6.935352390109688e-07 | True |
| 6 | 1386 | 1892 | 209 | 0.7232616543769836 | 0.7232615351676941 | 1.6482180249617675e-07 | True |
| 6 | 8368 | 1393 | 239 | 0.2940746247768402 | 0.2940746247768402 | 0.0 | True |
| 6 | 8368 | 2230 | 45 | 0.25275754928588867 | 0.25275757908821106 | 1.1790873298105349e-07 | True |
| 6 | 8449 | 2253 | 2 | 0.3653290271759033 | 0.36532890796661377 | 3.263066460179821e-07 | True |
| 6 | 8449 | 2266 | 67 | 0.31273704767227173 | 0.31273743510246277 | 1.2388368884457876e-06 | True |
| 7 | 3942 | 913 | 73 | 1.2540298700332642 | 1.2540295124053955 | 2.8518289492008465e-07 | True |
| 7 | 3942 | 241 | 144 | 0.39450982213020325 | 0.3945099413394928 | 3.021706504215696e-07 | True |
| 7 | 2182 | 2213 | 0 | 0.4564569294452667 | 0.4564571678638458 | 5.223243721840596e-07 | True |
| 7 | 2182 | 508 | 75 | 0.3789297640323639 | 0.3789295554161072 | 5.505406978166264e-07 | True |
| 7 | 8233 | 1393 | 188 | 0.38761797547340393 | 0.38761815428733826 | 4.6131486577159795e-07 | True |
| 7 | 8233 | 617 | 272 | 0.2753306031227112 | 0.2753306031227112 | 0.0 | True |
| 7 | 8223 | 410 | 0 | 0.28931376338005066 | 0.28931379318237305 | 1.0301038581612914e-07 | True |
| 7 | 8223 | 2173 | 0 | 0.2771303951740265 | 0.27713045477867126 | 2.150779770582774e-07 | True |
| 8 | 980 | 92 | 209 | 0.9414299726486206 | 0.9414303302764893 | 3.7987729203712615e-07 | True |
| 8 | 980 | 1443 | 178 | 0.8098523020744324 | 0.8098524212837219 | 1.4719880309709226e-07 | True |
| 8 | 5500 | 1973 | 57 | 0.6518172025680542 | 0.6518172025680542 | 0.0 | True |
| 8 | 5500 | 793 | 207 | 0.477134644985199 | 0.4771343171596527 | 6.870713533594229e-07 | True |
| 8 | 8284 | 2110 | 0 | 0.4154353141784668 | 0.4154352843761444 | 7.173757591270283e-08 | True |
| 8 | 8284 | 1469 | 0 | 0.38195693492889404 | 0.3819569945335388 | 1.5605069400425144e-07 | True |
| 8 | 8327 | 784 | 0 | 0.5565570592880249 | 0.5565568208694458 | 4.2838119672142736e-07 | True |
| 8 | 8327 | 2363 | 0 | 0.5526017546653748 | 0.5526021718978882 | 7.550329145812927e-07 | True |
| 9 | 5684 | 1126 | 31 | 2.1676177978515625 | 2.167617082595825 | 3.299731797799475e-07 | True |
| 9 | 5684 | 2254 | 226 | 1.6401602029800415 | 1.6401606798171997 | 2.907259652665328e-07 | True |
| 9 | 3092 | 1007 | 143 | 0.7984108924865723 | 0.7984106540679932 | 2.9861639081479873e-07 | True |
| 9 | 3092 | 349 | 14 | 0.7131683826446533 | 0.7131683826446533 | 0.0 | True |
| 9 | 8543 | 861 | 113 | 0.7431607246398926 | 0.7431607246398926 | 0.0 | True |
| 9 | 8543 | 501 | 120 | 0.5523545742034912 | 0.5523548722267151 | 5.395505673266305e-07 | True |
| 9 | 8435 | 455 | 0 | 0.3412257730960846 | 0.34122589230537415 | 3.4935605382074557e-07 | True |
| 9 | 8435 | 1606 | 0 | 0.32796886563301086 | 0.32796892523765564 | 1.817387289502253e-07 | True |
| 10 | 2713 | 103 | 112 | 1.3064208030700684 | 1.3064210414886475 | 1.8249753719573554e-07 | True |
| 10 | 2713 | 484 | 129 | 1.1167056560516357 | 1.1167058944702148 | 2.1350172071711812e-07 | True |
| 10 | 5441 | 1496 | 124 | 2.0704281330108643 | 2.0704288482666016 | 3.454627214056187e-07 | True |
| 10 | 5441 | 998 | 402 | 1.9085628986358643 | 1.908562421798706 | 2.4984094500838404e-07 | True |
| 10 | 8633 | 129 | 170 | 1.0784904956817627 | 1.0784902572631836 | 2.2106692646451864e-07 | True |
| 10 | 8633 | 2155 | 66 | 0.6926279067993164 | 0.6926276087760925 | 4.302789722322047e-07 | True |
| 10 | 8615 | 931 | 157 | 0.8751022815704346 | 0.8751018643379211 | 4.767814256854414e-07 | True |
| 10 | 8615 | 673 | 147 | 0.7828339338302612 | 0.7828339338302612 | 0.0 | True |
| 11 | 129 | 125 | 80 | 5.838160991668701 | 5.838160991668701 | 0.0 | True |
| 11 | 129 | 1664 | 14 | 4.234217643737793 | 4.23421573638916 | 4.504606974167656e-07 | True |
| 11 | 213 | 253 | 151 | 2.4544379711151123 | 2.4544379711151123 | 0.0 | True |
| 11 | 213 | 1061 | 72 | 2.171701192855835 | 2.171701669692993 | 2.1956849301909423e-07 | True |
| 11 | 8479 | 2452 | 159 | 2.5510146617889404 | 2.5510151386260986 | 1.8692058706896462e-07 | True |
| 11 | 8479 | 1413 | 76 | 1.6674267053604126 | 1.6674267053604126 | 0.0 | True |
| 11 | 8699 | 991 | 176 | 1.298280119895935 | 1.298280119895935 | 0.0 | True |
| 11 | 8699 | 2226 | 35 | 1.1850805282592773 | 1.1850805282592773 | 0.0 | True |
| 12 | 2897 | 340 | 323 | 2.20188045501709 | 2.2018802165985107 | 1.0827952923525633e-07 | True |
| 12 | 2897 | 965 | 14 | 1.6921615600585938 | 1.6921617984771729 | 1.4089587231452467e-07 | True |
| 12 | 1888 | 340 | 157 | 0.992925226688385 | 0.992925226688385 | 0.0 | True |
| 12 | 1888 | 1625 | 216 | 0.9181389212608337 | 0.9181387424468994 | 1.9475694819757313e-07 | True |
| 12 | 8689 | 414 | 17 | 2.9900319576263428 | 2.9900319576263428 | 0.0 | True |
| 12 | 8689 | 2181 | 363 | 1.9622551202774048 | 1.9622551202774048 | 0.0 | True |
| 12 | 8355 | 488 | 20 | 0.5464027523994446 | 0.5464024543762207 | 5.454277500767144e-07 | True |
| 12 | 8355 | 1691 | 135 | 0.5285205245018005 | 0.5285206437110901 | 2.2555281020193404e-07 | True |
| 13 | 4614 | 1363 | 48 | 4.764978408813477 | 4.764978408813477 | 0.0 | True |
| 13 | 4614 | 1946 | 229 | 2.5001580715179443 | 2.5001606941223145 | 1.0489754227919282e-06 | True |
| 13 | 4393 | 452 | 223 | 1.5467286109924316 | 1.546729326248169 | 4.624313096825409e-07 | True |
| 13 | 4393 | 2449 | 24 | 1.41753351688385 | 1.41753351688385 | 0.0 | True |
| 13 | 8485 | 2336 | 140 | 2.1225368976593018 | 2.1225380897521973 | 5.616358880839398e-07 | True |
| 13 | 8485 | 676 | 93 | 1.5216846466064453 | 1.5216838121414185 | 5.483823660286211e-07 | True |
| 13 | 8618 | 2399 | 198 | 5.134793758392334 | 5.134796619415283 | 5.571836151243035e-07 | True |
| 13 | 8618 | 1030 | 46 | 4.324872016906738 | 4.324872016906738 | 0.0 | True |
| 14 | 7031 | 2333 | 89 | 5.549941062927246 | 5.549938201904297 | 5.155051047893002e-07 | True |
| 14 | 7031 | 73 | 61 | 3.6058237552642822 | 3.6058225631713867 | 3.3060209716779134e-07 | True |
| 14 | 1145 | 369 | 239 | 3.9235029220581055 | 3.92350172996521 | 3.038338238021473e-07 | True |
| 14 | 1145 | 1470 | 31 | 3.3864898681640625 | 3.3864877223968506 | 6.336257586612417e-07 | True |
| 14 | 8353 | 449 | 47 | 1.846266746520996 | 1.8462671041488647 | 1.937032497206799e-07 | True |
| 14 | 8353 | 2438 | 107 | 1.7008116245269775 | 1.7008118629455566 | 1.4017929773255781e-07 | True |
| 14 | 8661 | 1606 | 143 | 3.1864726543426514 | 3.1864728927612305 | 7.482210110186767e-08 | True |
| 14 | 8661 | 1956 | 74 | 2.846846580505371 | 2.846848487854004 | 6.699864495240583e-07 | True |
| 15 | 467 | 841 | 337 | 25.462398529052734 | 25.462411880493164 | 5.243591020874736e-07 | True |
| 15 | 467 | 682 | 62 | 18.236833572387695 | 18.23681640625 | 9.412893760956216e-07 | True |
| 15 | 7515 | 129 | 185 | 6.375539779663086 | 6.375532627105713 | 1.121874793388059e-06 | True |
| 15 | 7515 | 2369 | 392 | 5.687297344207764 | 5.687297821044922 | 8.384248780112764e-08 | True |
| 15 | 8249 | 614 | 240 | 2.9803154468536377 | 2.9803152084350586 | 7.999776646236037e-08 | True |
| 15 | 8249 | 1918 | 20 | 2.6349573135375977 | 2.6349565982818604 | 2.714486999959826e-07 | True |
| 15 | 8430 | 1021 | 61 | 2.8997340202331543 | 2.8997349739074707 | 3.288833768034937e-07 | True |
| 15 | 8430 | 1031 | 9 | 2.6792356967926025 | 2.6792359352111816 | 8.89875345371745e-08 | True |

### midtrain_51B: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 5631 | 364 | 277 | 0.1835804134607315 | 0.1835804134607315 | 0.0 | True |
| 0 | 5631 | 163 | 12 | 0.14962029457092285 | 0.14962030947208405 | 9.95931817711014e-08 | True |
| 0 | 7413 | 2385 | 503 | 0.16933907568454742 | 0.16933907568454742 | 0.0 | True |
| 0 | 7413 | 1362 | 200 | 0.14658275246620178 | 0.14658276736736298 | 1.0165698858249698e-07 | True |
| 0 | 8206 | 2429 | 21 | 0.5147151350975037 | 0.5147151947021484 | 1.1580122811834469e-07 | True |
| 0 | 8206 | 2041 | 30 | 0.3237953782081604 | 0.32379522919654846 | 4.602030231656998e-07 | True |
| 0 | 8693 | 491 | 69 | 0.15185090899467468 | 0.1518508940935135 | 9.813020740211856e-08 | True |
| 0 | 8693 | 205 | 174 | 0.13861097395420074 | 0.13861098885536194 | 1.0750347370599413e-07 | True |
| 1 | 1081 | 1577 | 111 | 0.26507309079170227 | 0.26507309079170227 | 0.0 | True |
| 1 | 1081 | 2040 | 304 | 0.18342112004756927 | 0.1834210902452469 | 1.6248032058667093e-07 | True |
| 1 | 7444 | 394 | 104 | 0.21260935068130493 | 0.212609201669693 | 7.008704530678922e-07 | True |
| 1 | 7444 | 1109 | 392 | 0.17729347944259644 | 0.17729347944259644 | 0.0 | True |
| 1 | 8631 | 1919 | 94 | 0.5325253009796143 | 0.5325251817703247 | 2.2385657419748537e-07 | True |
| 1 | 8631 | 673 | 230 | 0.33497071266174316 | 0.33497071266174316 | 0.0 | True |
| 1 | 8526 | 1190 | 13 | 0.18740494549274445 | 0.18740499019622803 | 2.385395084638986e-07 | True |
| 1 | 8526 | 1885 | 5 | 0.16857126355171204 | 0.16857141256332397 | 8.839680548088481e-07 | True |
| 2 | 2351 | 159 | 159 | 0.20131033658981323 | 0.20131038129329681 | 2.2206253458623976e-07 | True |
| 2 | 2351 | 1407 | 0 | 0.15414173901081085 | 0.15414176881313324 | 1.9334362372546675e-07 | True |
| 2 | 7011 | 1634 | 134 | 0.24162447452545166 | 0.24162432551383972 | 6.167074433628219e-07 | True |
| 2 | 7011 | 692 | 122 | 0.2187451720237732 | 0.2187451720237732 | 0.0 | True |
| 2 | 8619 | 2090 | 42 | 0.3863745927810669 | 0.38637489080429077 | 7.713323532270231e-07 | True |
| 2 | 8619 | 2125 | 76 | 0.2545751929283142 | 0.25457507371902466 | 4.682675015564041e-07 | True |
| 2 | 8351 | 1549 | 117 | 0.15410421788692474 | 0.15410415828227997 | 3.8678139763264643e-07 | True |
| 2 | 8351 | 1700 | 29 | 0.14115260541439056 | 0.14115259051322937 | 1.05567737485974e-07 | True |
| 3 | 6554 | 1018 | 126 | 1.013501524925232 | 1.0135011672973633 | 3.528636710030867e-07 | True |
| 3 | 6554 | 2452 | 246 | 0.7574176788330078 | 0.7574174404144287 | 3.147782072751539e-07 | True |
| 3 | 6054 | 2176 | 124 | 1.0014424324035645 | 1.0014419555664062 | 4.7615034351866534e-07 | True |
| 3 | 6054 | 375 | 87 | 0.865912914276123 | 0.8659127950668335 | 1.376689128726491e-07 | True |
| 3 | 8194 | 75 | 180 | 0.1843332201242447 | 0.1843332052230835 | 8.0838175472679e-08 | True |
| 3 | 8194 | 2270 | 364 | 0.15940992534160614 | 0.1594100147485733 | 5.608619850457369e-07 | True |
| 3 | 8359 | 1362 | 51 | 0.31485989689826965 | 0.31485989689826965 | 0.0 | True |
| 3 | 8359 | 345 | 23 | 0.20200274884700775 | 0.20200276374816895 | 7.376712088771353e-08 | True |
| 4 | 6495 | 483 | 19 | 0.6238706111907959 | 0.623869776725769 | 1.337561045330708e-06 | True |
| 4 | 6495 | 1337 | 220 | 0.37004223465919495 | 0.3700421154499054 | 3.221504963090815e-07 | True |
| 4 | 4508 | 205 | 347 | 0.27225714921951294 | 0.27225708961486816 | 2.1892774880755528e-07 | True |
| 4 | 4508 | 2253 | 62 | 0.22909493744373322 | 0.22909487783908844 | 2.6017443004400655e-07 | True |
| 4 | 8406 | 2355 | 0 | 0.2933512032032013 | 0.2933512032032013 | 0.0 | True |
| 4 | 8406 | 2170 | 21 | 0.19920259714126587 | 0.19920267164707184 | 3.7402025394479164e-07 | True |
| 4 | 8438 | 1721 | 0 | 0.16801179945468903 | 0.1680118590593338 | 3.5476463539375016e-07 | True |
| 4 | 8438 | 430 | 0 | 0.161437526345253 | 0.1614375114440918 | 9.230295787597603e-08 | True |
| 5 | 120 | 2306 | 89 | 0.5564466118812561 | 0.5564466714859009 | 1.0711655620271809e-07 | True |
| 5 | 120 | 1898 | 60 | 0.34092339873313904 | 0.3409235179424286 | 3.4966590733801004e-07 | True |
| 5 | 910 | 1995 | 7 | 0.6271919012069702 | 0.6271916627883911 | 3.801365716661025e-07 | True |
| 5 | 910 | 2322 | 110 | 0.5808559060096741 | 0.5808559060096741 | 0.0 | True |
| 5 | 8676 | 942 | 12 | 0.46969035267829895 | 0.46969038248062134 | 6.34509996165656e-08 | True |
| 5 | 8676 | 756 | 34 | 0.34967347979545593 | 0.34967347979545593 | 0.0 | True |
| 5 | 8511 | 378 | 38 | 0.8193729519844055 | 0.8193729519844055 | 0.0 | True |
| 5 | 8511 | 2331 | 104 | 0.6759647727012634 | 0.6759644746780396 | 4.408857323821841e-07 | True |
| 6 | 5058 | 578 | 13 | 0.6585535407066345 | 0.6585534811019897 | 9.050842656078387e-08 | True |
| 6 | 5058 | 721 | 131 | 0.5042816996574402 | 0.5042818188667297 | 2.3639424082166856e-07 | True |
| 6 | 6290 | 228 | 489 | 1.2073686122894287 | 1.2073673009872437 | 1.0860827188244399e-06 | True |
| 6 | 6290 | 1761 | 152 | 1.0528008937835693 | 1.0528013706207275 | 4.529224481273582e-07 | True |
| 6 | 8620 | 137 | 254 | 0.5420399904251099 | 0.5420399308204651 | 1.0996355587831081e-07 | True |
| 6 | 8620 | 461 | 124 | 0.3437555134296417 | 0.34375566244125366 | 4.3348137300196516e-07 | True |
| 6 | 8276 | 679 | 3 | 0.21980921924114227 | 0.21980921924114227 | 0.0 | True |
| 6 | 8276 | 792 | 7 | 0.21205373108386993 | 0.21205364167690277 | 4.2162411718058546e-07 | True |
| 7 | 1761 | 774 | 109 | 1.2913405895233154 | 1.2913408279418945 | 1.8462873469312396e-07 | True |
| 7 | 1761 | 2348 | 121 | 0.8118526339530945 | 0.8118523359298706 | 3.670902962102993e-07 | True |
| 7 | 8129 | 1875 | 149 | 0.641187310218811 | 0.641187310218811 | 0.0 | True |
| 7 | 8129 | 1691 | 95 | 0.5635588765144348 | 0.5635588765144348 | 0.0 | True |
| 7 | 8497 | 2228 | 155 | 0.5287508368492126 | 0.5287508368492126 | 0.0 | True |
| 7 | 8497 | 141 | 14 | 0.4000421166419983 | 0.40004169940948486 | 1.0429714674295655e-06 | True |
| 7 | 8383 | 640 | 435 | 0.23697657883167267 | 0.23697656393051147 | 6.288031191652965e-08 | True |
| 7 | 8383 | 1605 | 11 | 0.22386246919631958 | 0.22386230528354645 | 7.322030071444376e-07 | True |
| 8 | 5987 | 1302 | 30 | 0.8707391619682312 | 0.8707393407821655 | 2.0535878267146997e-07 | True |
| 8 | 5987 | 1142 | 274 | 0.44413161277770996 | 0.44413188099861145 | 6.039221117626312e-07 | True |
| 8 | 2417 | 2379 | 64 | 1.1061122417449951 | 1.1061122417449951 | 0.0 | True |
| 8 | 2417 | 2239 | 112 | 0.8443731665611267 | 0.8443731665611267 | 0.0 | True |
| 8 | 8637 | 640 | 236 | 1.2160892486572266 | 1.2160894870758057 | 1.960535210428165e-07 | True |
| 8 | 8637 | 1060 | 16 | 0.5910274386405945 | 0.5910274386405945 | 0.0 | True |
| 8 | 8259 | 98 | 51 | 0.2573569416999817 | 0.25735726952552795 | 1.2738166070018687e-06 | True |
| 8 | 8259 | 2231 | 3 | 0.24421460926532745 | 0.24421465396881104 | 1.8304999736102922e-07 | True |
| 9 | 492 | 778 | 25 | 1.7662805318832397 | 1.7662789821624756 | 8.773922014006554e-07 | True |
| 9 | 492 | 2077 | 149 | 1.283678412437439 | 1.2836782932281494 | 9.286538466003144e-08 | True |
| 9 | 1790 | 1402 | 479 | 0.38838934898376465 | 0.3883892893791199 | 1.5346621870900533e-07 | True |
| 9 | 1790 | 280 | 272 | 0.3370373845100403 | 0.33703795075416565 | 1.6800632552658045e-06 | True |
| 9 | 8520 | 889 | 191 | 0.9584348201751709 | 0.9584351181983948 | 3.109478261886229e-07 | True |
| 9 | 8520 | 1957 | 397 | 0.6623560190200806 | 0.6623563766479492 | 5.399329943154055e-07 | True |
| 9 | 8573 | 506 | 65 | 0.6503602266311646 | 0.6503603458404541 | 1.832973245739208e-07 | True |
| 9 | 8573 | 879 | 266 | 0.6088758707046509 | 0.6088762283325195 | 5.873575975977858e-07 | True |
| 10 | 4202 | 340 | 404 | 1.181942343711853 | 1.181942105293274 | 2.017176052368311e-07 | True |
| 10 | 4202 | 2363 | 116 | 0.7931588888168335 | 0.793158769607544 | 1.5029685884073928e-07 | True |
| 10 | 4027 | 1007 | 97 | 0.6885402798652649 | 0.6885403990745544 | 1.7313335622733413e-07 | True |
| 10 | 4027 | 406 | 341 | 0.6051563620567322 | 0.6051557660102844 | 9.849461810632475e-07 | True |
| 10 | 8423 | 211 | 151 | 0.3311232328414917 | 0.33112335205078125 | 3.6001487581466867e-07 | True |
| 10 | 8423 | 2058 | 81 | 0.3083723783493042 | 0.3083724081516266 | 9.66439424543309e-08 | True |
| 10 | 8406 | 2333 | 21 | 0.32469144463539124 | 0.3246913552284241 | 2.7535978739287243e-07 | True |
| 10 | 8406 | 2123 | 177 | 0.30676913261413574 | 0.3067690134048462 | 3.885961033137144e-07 | True |
| 11 | 319 | 1107 | 116 | 1.3264042139053345 | 1.3264045715332031 | 2.6962208420567314e-07 | True |
| 11 | 319 | 2145 | 181 | 0.9818816781044006 | 0.9818816781044006 | 0.0 | True |
| 11 | 5622 | 299 | 117 | 1.0904574394226074 | 1.0904570817947388 | 3.279613268003437e-07 | True |
| 11 | 5622 | 232 | 189 | 0.9838941097259521 | 0.9838941097259521 | 0.0 | True |
| 11 | 8479 | 2452 | 159 | 2.4139816761016846 | 2.413982629776001 | 3.9506278189581344e-07 | True |
| 11 | 8479 | 1413 | 76 | 1.639001488685608 | 1.639001488685608 | 0.0 | True |
| 11 | 8605 | 715 | 186 | 1.3432166576385498 | 1.3432166576385498 | 0.0 | True |
| 11 | 8605 | 2156 | 356 | 1.1932464838027954 | 1.1932462453842163 | 1.99806647107594e-07 | True |
| 12 | 2897 | 369 | 98 | 2.007970094680786 | 2.007970094680786 | 0.0 | True |
| 12 | 2897 | 296 | 213 | 1.5911955833435059 | 1.5911953449249268 | 1.4983612423092862e-07 | True |
| 12 | 8083 | 2045 | 305 | 1.1509456634521484 | 1.1509456634521484 | 0.0 | True |
| 12 | 8083 | 471 | 47 | 1.0874357223510742 | 1.087435245513916 | 4.384968678168732e-07 | True |
| 12 | 8214 | 1949 | 70 | 0.7455501556396484 | 0.7455502152442932 | 7.994719647567175e-08 | True |
| 12 | 8214 | 2139 | 91 | 0.6182180047035217 | 0.6182180643081665 | 9.641363454623935e-08 | True |
| 12 | 8424 | 590 | 174 | 0.5421915650367737 | 0.5421914458274841 | 2.1986562912076283e-07 | True |
| 12 | 8424 | 1672 | 48 | 0.5066222548484802 | 0.5066223740577698 | 2.3530211791906807e-07 | True |
| 13 | 1258 | 477 | 71 | 4.111429214477539 | 4.111429214477539 | 0.0 | True |
| 13 | 1258 | 8 | 175 | 1.873930811882019 | 1.8739302158355713 | 3.1807281462824506e-07 | True |
| 13 | 4658 | 434 | 353 | 4.454965591430664 | 4.454967498779297 | 4.2813992469018723e-07 | True |
| 13 | 4658 | 2170 | 188 | 2.9509828090667725 | 2.9509832859039307 | 1.6158588140129538e-07 | True |
| 13 | 8205 | 387 | 260 | 0.9531090259552002 | 0.9531084299087524 | 6.253706884755939e-07 | True |
| 13 | 8205 | 2348 | 157 | 0.7375013828277588 | 0.7375011444091797 | 3.2327882313577227e-07 | True |
| 13 | 8496 | 118 | 308 | 3.114915609359741 | 3.114912986755371 | 8.419503765163813e-07 | True |
| 13 | 8496 | 1492 | 75 | 2.430069923400879 | 2.4300715923309326 | 6.867827290233989e-07 | True |
| 14 | 262 | 4 | 184 | 16.305021286010742 | 16.30501937866211 | 1.1697921758918233e-07 | True |
| 14 | 262 | 362 | 214 | 9.040946006774902 | 9.04094409942627 | 2.109678159103277e-07 | True |
| 14 | 1145 | 1470 | 31 | 3.739894390106201 | 3.7398905754089355 | 1.0200013336517438e-06 | True |
| 14 | 1145 | 2363 | 76 | 3.2456133365631104 | 3.2456140518188477 | 2.2037613946401144e-07 | True |
| 14 | 8454 | 994 | 73 | 12.945328712463379 | 12.945328712463379 | 0.0 | True |
| 14 | 8454 | 1571 | 106 | 6.537574768066406 | 6.53757381439209 | 1.4587585614539973e-07 | True |
| 14 | 8273 | 387 | 228 | 1.657378911972046 | 1.6573790311813354 | 7.19263945556898e-08 | True |
| 14 | 8273 | 471 | 56 | 1.5752495527267456 | 1.575249433517456 | 7.56764471663749e-08 | True |
| 15 | 7395 | 1060 | 26 | 15.955168724060059 | 15.955168724060059 | 0.0 | True |
| 15 | 7395 | 1476 | 57 | 12.62187385559082 | 12.62187671661377 | 2.2667180657580955e-07 | True |
| 15 | 5120 | 472 | 158 | 4.937427520751953 | 4.9374308586120605 | 6.760322239451386e-07 | True |
| 15 | 5120 | 1961 | 378 | 4.468637466430664 | 4.468636512756348 | 2.1341501152654477e-07 | True |
| 15 | 8400 | 1541 | 325 | 3.405773878097534 | 3.405773639678955 | 7.000423035564038e-08 | True |
| 15 | 8400 | 1785 | 292 | 3.1104531288146973 | 3.1104536056518555 | 1.5330150896208286e-07 | True |
| 15 | 8242 | 657 | 243 | 3.0302815437316895 | 3.0302815437316895 | 0.0 | True |
| 15 | 8242 | 961 | 109 | 2.9537911415100098 | 2.9537923336029053 | 4.035806319394681e-07 | True |

### pretrain_1T: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 4931 | 212 | 143 | 1.7953925132751465 | 1.795392632484436 | 6.639734134421683e-08 | True |
| 0 | 4931 | 389 | 420 | 0.7430276870727539 | 0.7430276870727539 | 0.0 | True |
| 0 | 7837 | 673 | 414 | 0.46988826990127563 | 0.4698883593082428 | 1.9027282205165603e-07 | True |
| 0 | 7837 | 94 | 189 | 0.44049981236457825 | 0.44049981236457825 | 0.0 | True |
| 0 | 8356 | 24 | 84 | 0.7661274075508118 | 0.7661271095275879 | 3.8899955926349934e-07 | True |
| 0 | 8356 | 2429 | 21 | 0.499392569065094 | 0.4993927776813507 | 4.177400098371805e-07 | True |
| 0 | 8479 | 2266 | 67 | 1.1029369831085205 | 1.1029366254806519 | 3.242505003725638e-07 | True |
| 0 | 8479 | 633 | 5 | 0.7196639776229858 | 0.7196640372276306 | 8.282288210709363e-08 | True |
| 1 | 7828 | 952 | 22 | 1.8047181367874146 | 1.8047188520431519 | 3.9632545533005886e-07 | True |
| 1 | 7828 | 434 | 8 | 0.5478079319000244 | 0.5478079319000244 | 0.0 | True |
| 1 | 6433 | 389 | 247 | 0.6581045985221863 | 0.6581045985221863 | 0.0 | True |
| 1 | 6433 | 721 | 207 | 0.5515757203102112 | 0.5515758991241455 | 3.2418746464330464e-07 | True |
| 1 | 8391 | 970 | 0 | 0.8931317329406738 | 0.8931320905685425 | 4.004200673453164e-07 | True |
| 1 | 8391 | 1081 | 0 | 0.7205146551132202 | 0.7205148339271545 | 2.481752911716604e-07 | True |
| 1 | 8655 | 552 | 344 | 0.4959086775779724 | 0.4959086775779724 | 0.0 | True |
| 1 | 8655 | 2202 | 122 | 0.4226120710372925 | 0.4226120412349701 | 7.051933541449996e-08 | True |
| 2 | 2428 | 2218 | 149 | 1.4638386964797974 | 1.4638391733169556 | 3.257443319061117e-07 | True |
| 2 | 2428 | 392 | 41 | 1.0883221626281738 | 1.0883222818374634 | 1.0953492784058023e-07 | True |
| 2 | 5288 | 414 | 134 | 0.9129780530929565 | 0.9129776358604431 | 4.5700169025339437e-07 | True |
| 2 | 5288 | 720 | 164 | 0.8014187216758728 | 0.8014191389083862 | 5.206173778362027e-07 | True |
| 2 | 8655 | 2306 | 184 | 1.0806299448013306 | 1.0806297063827515 | 2.2062925449043963e-07 | True |
| 2 | 8655 | 299 | 10 | 0.5732948780059814 | 0.5732950568199158 | 3.1190568970041665e-07 | True |
| 2 | 8616 | 767 | 51 | 0.5930210947990417 | 0.593021035194397 | 1.0051015941614854e-07 | True |
| 2 | 8616 | 67 | 113 | 0.5241701602935791 | 0.5241702198982239 | 1.1371239587924472e-07 | True |
| 3 | 6554 | 1739 | 144 | 0.9151924252510071 | 0.9151918888092041 | 5.861519262808446e-07 | True |
| 3 | 6554 | 1173 | 90 | 0.6564023494720459 | 0.656402587890625 | 3.6322017935086016e-07 | True |
| 3 | 1706 | 1224 | 41 | 1.248170256614685 | 1.248170256614685 | 0.0 | True |
| 3 | 1706 | 2228 | 325 | 1.201250433921814 | 1.2012499570846558 | 3.9695066468892614e-07 | True |
| 3 | 8236 | 588 | 0 | 0.6394556760787964 | 0.6394556760787964 | 0.0 | True |
| 3 | 8236 | 952 | 17 | 0.5766658782958984 | 0.5766655802726746 | 5.168039849308227e-07 | True |
| 3 | 8683 | 2034 | 24 | 0.8560838103294373 | 0.8560837507247925 | 6.962477745310197e-08 | True |
| 3 | 8683 | 2088 | 41 | 0.7430622577667236 | 0.7430623173713684 | 8.021487318509839e-08 | True |
| 4 | 5393 | 232 | 43 | 0.809116780757904 | 0.809116780757904 | 0.0 | True |
| 4 | 5393 | 2090 | 10 | 0.6877729892730713 | 0.6877724528312683 | 7.799692796099737e-07 | True |
| 4 | 618 | 488 | 83 | 0.569729208946228 | 0.5697292685508728 | 1.0461925391825261e-07 | True |
| 4 | 618 | 1606 | 307 | 0.5316059589385986 | 0.5316058993339539 | 1.1212185223505942e-07 | True |
| 4 | 8337 | 1247 | 41 | 0.449758380651474 | 0.4497585594654083 | 3.9757777068469585e-07 | True |
| 4 | 8337 | 16 | 101 | 0.3643738627433777 | 0.3643738031387329 | 1.635810107965103e-07 | True |
| 4 | 8424 | 2110 | 0 | 0.4252921938896179 | 0.4252922236919403 | 7.007493392984854e-08 | True |
| 4 | 8424 | 683 | 35 | 0.4066426157951355 | 0.406642884016037 | 6.595986034685213e-07 | True |
| 5 | 3779 | 1629 | 365 | 1.4091522693634033 | 1.4091523885726929 | 8.45964571342138e-08 | True |
| 5 | 3779 | 1801 | 80 | 1.1320072412490845 | 1.1320067644119263 | 4.212315441347983e-07 | True |
| 5 | 4082 | 1445 | 109 | 2.850001335144043 | 2.8500003814697266 | 3.3462241039899373e-07 | True |
| 5 | 4082 | 2162 | 295 | 2.3922674655914307 | 2.392268419265747 | 3.9864870049991544e-07 | True |
| 5 | 8327 | 2355 | 0 | 1.0034681558609009 | 1.0034682750701904 | 1.1879728206074319e-07 | True |
| 5 | 8327 | 332 | 0 | 0.5941993594169617 | 0.5941994786262512 | 2.006217066066032e-07 | True |
| 5 | 8637 | 874 | 4 | 0.6801485419273376 | 0.6801481246948242 | 6.134432226310772e-07 | True |
| 5 | 8637 | 564 | 107 | 0.5785250663757324 | 0.5785250663757324 | 0.0 | True |
| 6 | 354 | 1351 | 24 | 1.2716846466064453 | 1.2716853618621826 | 5.624474111670559e-07 | True |
| 6 | 354 | 232 | 104 | 1.039800763130188 | 1.039800763130188 | 0.0 | True |
| 6 | 3628 | 283 | 161 | 1.4492994546890259 | 1.4492994546890259 | 0.0 | True |
| 6 | 3628 | 846 | 137 | 1.3584953546524048 | 1.3584930896759033 | 1.667268492091663e-06 | True |
| 6 | 8245 | 2355 | 0 | 0.7061348557472229 | 0.706134557723999 | 4.2204859518170756e-07 | True |
| 6 | 8245 | 1157 | 0 | 0.6261034607887268 | 0.6261035203933716 | 9.519935363446856e-08 | True |
| 6 | 8512 | 881 | 27 | 0.9362302422523499 | 0.9362298250198364 | 4.4565160854446556e-07 | True |
| 6 | 8512 | 126 | 128 | 0.8058128952980042 | 0.8058129549026489 | 7.396834317642404e-08 | True |
| 7 | 3942 | 1519 | 2 | 2.3549864292144775 | 2.3549859523773193 | 2.024797902390365e-07 | True |
| 7 | 3942 | 271 | 9 | 1.283031702041626 | 1.283031702041626 | 0.0 | True |
| 7 | 5904 | 86 | 112 | 1.43545663356781 | 1.4354565143585205 | 8.304624936978277e-08 | True |
| 7 | 5904 | 561 | 2 | 1.3227840662002563 | 1.3227834701538086 | 4.505999603291795e-07 | True |
| 7 | 8393 | 2030 | 0 | 0.9087136387825012 | 0.9087141156196594 | 5.247386391624909e-07 | True |
| 7 | 8393 | 1281 | 0 | 0.82597815990448 | 0.8259778618812561 | 3.6081247464390337e-07 | True |
| 7 | 8482 | 1586 | 21 | 2.0226473808288574 | 2.022646903991699 | 2.3574903006955303e-07 | True |
| 7 | 8482 | 1307 | 244 | 1.6311453580856323 | 1.631144642829895 | 4.3849907904230925e-07 | True |
| 8 | 3685 | 1625 | 208 | 1.6409099102020264 | 1.6409099102020264 | 0.0 | True |
| 8 | 3685 | 2211 | 111 | 1.1094183921813965 | 1.1094191074371338 | 6.447123486913843e-07 | True |
| 8 | 3784 | 1849 | 132 | 1.0864237546920776 | 1.0864237546920776 | 0.0 | True |
| 8 | 3784 | 1129 | 24 | 1.028967261314392 | 1.028967261314392 | 0.0 | True |
| 8 | 8288 | 1407 | 0 | 0.924115002155304 | 0.9241151213645935 | 1.289983273431885e-07 | True |
| 8 | 8288 | 1789 | 0 | 0.8590429425239563 | 0.8590428829193115 | 6.938494203824789e-08 | True |
| 8 | 8397 | 2349 | 0 | 1.5603916645050049 | 1.5603917837142944 | 7.639703047798413e-08 | True |
| 8 | 8397 | 484 | 0 | 1.516995906829834 | 1.5169960260391235 | 7.858247277667396e-08 | True |
| 9 | 7236 | 764 | 20 | 2.6105904579162598 | 2.6105892658233643 | 4.566372683593297e-07 | True |
| 9 | 7236 | 98 | 84 | 1.7447422742843628 | 1.7447428703308105 | 3.4162435136638465e-07 | True |
| 9 | 6108 | 1625 | 301 | 2.4851038455963135 | 2.485105276107788 | 5.756344859166706e-07 | True |
| 9 | 6108 | 2117 | 93 | 2.2715437412261963 | 2.2715437412261963 | 0.0 | True |
| 9 | 8337 | 1124 | 130 | 0.8969646692276001 | 0.8969650864601135 | 4.6516047704200463e-07 | True |
| 9 | 8337 | 1155 | 12 | 0.7343858480453491 | 0.7343858480453491 | 0.0 | True |
| 9 | 8460 | 896 | 328 | 1.0744712352752686 | 1.0744707584381104 | 4.4378773721286655e-07 | True |
| 9 | 8460 | 1874 | 42 | 0.9432343244552612 | 0.9432343244552612 | 0.0 | True |
| 10 | 65 | 1785 | 455 | 10.412165641784668 | 10.412171363830566 | 5.495538675906743e-07 | True |
| 10 | 65 | 452 | 34 | 7.171318531036377 | 7.171317100524902 | 1.9947677242592679e-07 | True |
| 10 | 1596 | 1391 | 313 | 8.341978073120117 | 8.341978073120117 | 0.0 | True |
| 10 | 1596 | 434 | 322 | 7.557908058166504 | 7.5579071044921875 | 1.2618231249529183e-07 | True |
| 10 | 8593 | 455 | 187 | 2.6841320991516113 | 2.684131622314453 | 1.7765040638418712e-07 | True |
| 10 | 8593 | 954 | 291 | 1.7468444108963013 | 1.7468446493148804 | 1.3648529749666175e-07 | True |
| 10 | 8194 | 2270 | 152 | 0.8146477341651917 | 0.8146477937698364 | 7.316615793017621e-08 | True |
| 10 | 8194 | 115 | 99 | 0.7833586931228638 | 0.783359169960022 | 6.087085806148535e-07 | True |
| 11 | 747 | 75 | 233 | 10.999449729919434 | 10.999454498291016 | 4.335100117836691e-07 | True |
| 11 | 747 | 2172 | 378 | 7.028768062591553 | 7.028762340545654 | 8.140894460426603e-07 | True |
| 11 | 2309 | 1135 | 79 | 4.536155700683594 | 4.536153793334961 | 4.2047688806737046e-07 | True |
| 11 | 2309 | 1618 | 182 | 4.166429042816162 | 4.166430473327637 | 3.433423346248728e-07 | True |
| 11 | 8295 | 1814 | 39 | 1.3202542066574097 | 1.3202540874481201 | 9.029267920500908e-08 | True |
| 11 | 8295 | 1751 | 210 | 1.174645185470581 | 1.1746454238891602 | 2.0297072005283735e-07 | True |
| 11 | 8299 | 1449 | 9 | 1.1992628574371338 | 1.199262261390686 | 4.970106795666782e-07 | True |
| 11 | 8299 | 1658 | 17 | 1.1157208681106567 | 1.115721344947815 | 4.2738033484180787e-07 | True |
| 12 | 6205 | 1784 | 17 | 6.7961201667785645 | 6.796122074127197 | 2.8065257617665173e-07 | True |
| 12 | 6205 | 94 | 323 | 5.712130069732666 | 5.712127685546875 | 4.1738996869992624e-07 | True |
| 12 | 7423 | 27 | 129 | 3.04311466217041 | 3.0431127548217773 | 6.267751447302156e-07 | True |
| 12 | 7423 | 1739 | 261 | 2.7316694259643555 | 2.7316689491271973 | 1.7455888097981996e-07 | True |
| 12 | 8333 | 1870 | 339 | 1.6111825704574585 | 1.6111825704574585 | 0.0 | True |
| 12 | 8333 | 118 | 435 | 1.3624426126480103 | 1.3624433279037476 | 5.249804510404544e-07 | True |
| 12 | 8501 | 1237 | 255 | 4.651230812072754 | 4.651234149932861 | 7.176294280554969e-07 | True |
| 12 | 8501 | 469 | 66 | 4.282090663909912 | 4.282088756561279 | 4.454246260799459e-07 | True |
| 13 | 5046 | 1974 | 159 | 35.49806594848633 | 35.49806594848633 | 0.0 | True |
| 13 | 5046 | 1113 | 48 | 17.862272262573242 | 17.862274169921875 | 1.0678085098999196e-07 | True |
| 13 | 7001 | 764 | 343 | 24.583066940307617 | 24.583057403564453 | 3.8793951898758336e-07 | True |
| 13 | 7001 | 867 | 101 | 20.84154510498047 | 20.841554641723633 | 4.57583308532318e-07 | True |
| 13 | 8237 | 1065 | 134 | 3.143796682357788 | 3.1437952518463135 | 4.5502671423920407e-07 | True |
| 13 | 8237 | 2300 | 181 | 2.5890109539031982 | 2.589010000228882 | 3.683546857801002e-07 | True |
| 13 | 8618 | 378 | 77 | 14.04815673828125 | 14.048155784606934 | 6.788608172398063e-08 | True |
| 13 | 8618 | 581 | 43 | 13.207986831665039 | 13.207991600036621 | 3.610218304124501e-07 | True |
| 14 | 5888 | 1021 | 50 | 15.562504768371582 | 15.562506675720215 | 1.2256051716616307e-07 | True |
| 14 | 5888 | 865 | 256 | 12.463746070861816 | 12.463748931884766 | 2.2954759612018654e-07 | True |
| 14 | 6925 | 572 | 231 | 13.755989074707031 | 13.755990982055664 | 1.386558699962563e-07 | True |
| 14 | 6925 | 1635 | 28 | 12.615540504455566 | 12.615535736083984 | 3.7797600351306014e-07 | True |
| 14 | 8475 | 614 | 434 | 13.228647232055664 | 13.22864818572998 | 7.209159785403499e-08 | True |
| 14 | 8475 | 1249 | 24 | 6.627888202667236 | 6.627885341644287 | 4.316643343602868e-07 | True |
| 14 | 8204 | 229 | 136 | 5.029248237609863 | 5.029247283935547 | 1.8962562024170062e-07 | True |
| 14 | 8204 | 278 | 30 | 4.816563606262207 | 4.816564083099365 | 9.899945213703188e-08 | True |
| 15 | 6299 | 666 | 59 | 28.660112380981445 | 28.66010856628418 | 1.3310126683789257e-07 | True |
| 15 | 6299 | 1849 | 363 | 23.970638275146484 | 23.970645904541016 | 3.182808252194268e-07 | True |
| 15 | 5902 | 1584 | 196 | 19.056137084960938 | 19.056129455566406 | 4.003641712501692e-07 | True |
| 15 | 5902 | 757 | 10 | 17.494997024536133 | 17.494993209838867 | 2.1804503654816393e-07 | True |
| 15 | 8259 | 2371 | 213 | 9.20541000366211 | 9.205410957336426 | 1.0359933083120229e-07 | True |
| 15 | 8259 | 2471 | 137 | 7.452627182006836 | 7.452630043029785 | 3.838945487741863e-07 | True |
| 15 | 8354 | 434 | 259 | 6.409968376159668 | 6.409965515136719 | 4.463396355993947e-07 | True |
| 15 | 8354 | 1570 | 431 | 6.194802761077881 | 6.19480037689209 | 3.848687170470594e-07 | True |

### pretrain_1T_b: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 4931 | 212 | 143 | 1.7953925132751465 | 1.795392632484436 | 6.639734134421683e-08 | True |
| 0 | 4931 | 389 | 420 | 0.7430276870727539 | 0.7430276870727539 | 0.0 | True |
| 0 | 7837 | 673 | 414 | 0.46988826990127563 | 0.4698883593082428 | 1.9027282205165603e-07 | True |
| 0 | 7837 | 94 | 189 | 0.44049981236457825 | 0.44049981236457825 | 0.0 | True |
| 0 | 8356 | 24 | 84 | 0.7661274075508118 | 0.7661271095275879 | 3.8899955926349934e-07 | True |
| 0 | 8356 | 2429 | 21 | 0.499392569065094 | 0.4993927776813507 | 4.177400098371805e-07 | True |
| 0 | 8479 | 2266 | 67 | 1.1029369831085205 | 1.1029366254806519 | 3.242505003725638e-07 | True |
| 0 | 8479 | 633 | 5 | 0.7196639776229858 | 0.7196640372276306 | 8.282288210709363e-08 | True |
| 1 | 7828 | 952 | 22 | 1.8047181367874146 | 1.8047188520431519 | 3.9632545533005886e-07 | True |
| 1 | 7828 | 434 | 8 | 0.5478079319000244 | 0.5478079319000244 | 0.0 | True |
| 1 | 6433 | 389 | 247 | 0.6581045985221863 | 0.6581045985221863 | 0.0 | True |
| 1 | 6433 | 721 | 207 | 0.5515757203102112 | 0.5515758991241455 | 3.2418746464330464e-07 | True |
| 1 | 8391 | 970 | 0 | 0.8931317329406738 | 0.8931320905685425 | 4.004200673453164e-07 | True |
| 1 | 8391 | 1081 | 0 | 0.7205146551132202 | 0.7205148339271545 | 2.481752911716604e-07 | True |
| 1 | 8655 | 552 | 344 | 0.4959086775779724 | 0.4959086775779724 | 0.0 | True |
| 1 | 8655 | 2202 | 122 | 0.4226120710372925 | 0.4226120412349701 | 7.051933541449996e-08 | True |
| 2 | 2428 | 2218 | 149 | 1.4638386964797974 | 1.4638391733169556 | 3.257443319061117e-07 | True |
| 2 | 2428 | 392 | 41 | 1.0883221626281738 | 1.0883222818374634 | 1.0953492784058023e-07 | True |
| 2 | 5288 | 414 | 134 | 0.9129780530929565 | 0.9129776358604431 | 4.5700169025339437e-07 | True |
| 2 | 5288 | 720 | 164 | 0.8014187216758728 | 0.8014191389083862 | 5.206173778362027e-07 | True |
| 2 | 8655 | 2306 | 184 | 1.0806299448013306 | 1.0806297063827515 | 2.2062925449043963e-07 | True |
| 2 | 8655 | 299 | 10 | 0.5732948780059814 | 0.5732950568199158 | 3.1190568970041665e-07 | True |
| 2 | 8616 | 767 | 51 | 0.5930210947990417 | 0.593021035194397 | 1.0051015941614854e-07 | True |
| 2 | 8616 | 67 | 113 | 0.5241701602935791 | 0.5241702198982239 | 1.1371239587924472e-07 | True |
| 3 | 6554 | 1739 | 144 | 0.9151924252510071 | 0.9151918888092041 | 5.861519262808446e-07 | True |
| 3 | 6554 | 1173 | 90 | 0.6564023494720459 | 0.656402587890625 | 3.6322017935086016e-07 | True |
| 3 | 1706 | 1224 | 41 | 1.248170256614685 | 1.248170256614685 | 0.0 | True |
| 3 | 1706 | 2228 | 325 | 1.201250433921814 | 1.2012499570846558 | 3.9695066468892614e-07 | True |
| 3 | 8236 | 588 | 0 | 0.6394556760787964 | 0.6394556760787964 | 0.0 | True |
| 3 | 8236 | 952 | 17 | 0.5766658782958984 | 0.5766655802726746 | 5.168039849308227e-07 | True |
| 3 | 8683 | 2034 | 24 | 0.8560838103294373 | 0.8560837507247925 | 6.962477745310197e-08 | True |
| 3 | 8683 | 2088 | 41 | 0.7430622577667236 | 0.7430623173713684 | 8.021487318509839e-08 | True |
| 4 | 5393 | 232 | 43 | 0.809116780757904 | 0.809116780757904 | 0.0 | True |
| 4 | 5393 | 2090 | 10 | 0.6877729892730713 | 0.6877724528312683 | 7.799692796099737e-07 | True |
| 4 | 618 | 488 | 83 | 0.569729208946228 | 0.5697292685508728 | 1.0461925391825261e-07 | True |
| 4 | 618 | 1606 | 307 | 0.5316059589385986 | 0.5316058993339539 | 1.1212185223505942e-07 | True |
| 4 | 8337 | 1247 | 41 | 0.449758380651474 | 0.4497585594654083 | 3.9757777068469585e-07 | True |
| 4 | 8337 | 16 | 101 | 0.3643738627433777 | 0.3643738031387329 | 1.635810107965103e-07 | True |
| 4 | 8424 | 2110 | 0 | 0.4252921938896179 | 0.4252922236919403 | 7.007493392984854e-08 | True |
| 4 | 8424 | 683 | 35 | 0.4066426157951355 | 0.406642884016037 | 6.595986034685213e-07 | True |
| 5 | 3779 | 1629 | 365 | 1.4091522693634033 | 1.4091523885726929 | 8.45964571342138e-08 | True |
| 5 | 3779 | 1801 | 80 | 1.1320072412490845 | 1.1320067644119263 | 4.212315441347983e-07 | True |
| 5 | 4082 | 1445 | 109 | 2.850001335144043 | 2.8500003814697266 | 3.3462241039899373e-07 | True |
| 5 | 4082 | 2162 | 295 | 2.3922674655914307 | 2.392268419265747 | 3.9864870049991544e-07 | True |
| 5 | 8327 | 2355 | 0 | 1.0034681558609009 | 1.0034682750701904 | 1.1879728206074319e-07 | True |
| 5 | 8327 | 332 | 0 | 0.5941993594169617 | 0.5941994786262512 | 2.006217066066032e-07 | True |
| 5 | 8637 | 874 | 4 | 0.6801485419273376 | 0.6801481246948242 | 6.134432226310772e-07 | True |
| 5 | 8637 | 564 | 107 | 0.5785250663757324 | 0.5785250663757324 | 0.0 | True |
| 6 | 354 | 1351 | 24 | 1.2716846466064453 | 1.2716853618621826 | 5.624474111670559e-07 | True |
| 6 | 354 | 232 | 104 | 1.039800763130188 | 1.039800763130188 | 0.0 | True |
| 6 | 3628 | 283 | 161 | 1.4492994546890259 | 1.4492994546890259 | 0.0 | True |
| 6 | 3628 | 846 | 137 | 1.3584953546524048 | 1.3584930896759033 | 1.667268492091663e-06 | True |
| 6 | 8245 | 2355 | 0 | 0.7061348557472229 | 0.706134557723999 | 4.2204859518170756e-07 | True |
| 6 | 8245 | 1157 | 0 | 0.6261034607887268 | 0.6261035203933716 | 9.519935363446856e-08 | True |
| 6 | 8512 | 881 | 27 | 0.9362302422523499 | 0.9362298250198364 | 4.4565160854446556e-07 | True |
| 6 | 8512 | 126 | 128 | 0.8058128952980042 | 0.8058129549026489 | 7.396834317642404e-08 | True |
| 7 | 3942 | 1519 | 2 | 2.3549864292144775 | 2.3549859523773193 | 2.024797902390365e-07 | True |
| 7 | 3942 | 271 | 9 | 1.283031702041626 | 1.283031702041626 | 0.0 | True |
| 7 | 5904 | 86 | 112 | 1.43545663356781 | 1.4354565143585205 | 8.304624936978277e-08 | True |
| 7 | 5904 | 561 | 2 | 1.3227840662002563 | 1.3227834701538086 | 4.505999603291795e-07 | True |
| 7 | 8393 | 2030 | 0 | 0.9087136387825012 | 0.9087141156196594 | 5.247386391624909e-07 | True |
| 7 | 8393 | 1281 | 0 | 0.82597815990448 | 0.8259778618812561 | 3.6081247464390337e-07 | True |
| 7 | 8482 | 1586 | 21 | 2.0226473808288574 | 2.022646903991699 | 2.3574903006955303e-07 | True |
| 7 | 8482 | 1307 | 244 | 1.6311453580856323 | 1.631144642829895 | 4.3849907904230925e-07 | True |
| 8 | 3685 | 1625 | 208 | 1.6409099102020264 | 1.6409099102020264 | 0.0 | True |
| 8 | 3685 | 2211 | 111 | 1.1094183921813965 | 1.1094191074371338 | 6.447123486913843e-07 | True |
| 8 | 3784 | 1849 | 132 | 1.0864237546920776 | 1.0864237546920776 | 0.0 | True |
| 8 | 3784 | 1129 | 24 | 1.028967261314392 | 1.028967261314392 | 0.0 | True |
| 8 | 8288 | 1407 | 0 | 0.924115002155304 | 0.9241151213645935 | 1.289983273431885e-07 | True |
| 8 | 8288 | 1789 | 0 | 0.8590429425239563 | 0.8590428829193115 | 6.938494203824789e-08 | True |
| 8 | 8397 | 2349 | 0 | 1.5603916645050049 | 1.5603917837142944 | 7.639703047798413e-08 | True |
| 8 | 8397 | 484 | 0 | 1.516995906829834 | 1.5169960260391235 | 7.858247277667396e-08 | True |
| 9 | 7236 | 764 | 20 | 2.6105904579162598 | 2.6105892658233643 | 4.566372683593297e-07 | True |
| 9 | 7236 | 98 | 84 | 1.7447422742843628 | 1.7447428703308105 | 3.4162435136638465e-07 | True |
| 9 | 6108 | 1625 | 301 | 2.4851038455963135 | 2.485105276107788 | 5.756344859166706e-07 | True |
| 9 | 6108 | 2117 | 93 | 2.2715437412261963 | 2.2715437412261963 | 0.0 | True |
| 9 | 8337 | 1124 | 130 | 0.8969646692276001 | 0.8969650864601135 | 4.6516047704200463e-07 | True |
| 9 | 8337 | 1155 | 12 | 0.7343858480453491 | 0.7343858480453491 | 0.0 | True |
| 9 | 8460 | 896 | 328 | 1.0744712352752686 | 1.0744707584381104 | 4.4378773721286655e-07 | True |
| 9 | 8460 | 1874 | 42 | 0.9432343244552612 | 0.9432343244552612 | 0.0 | True |
| 10 | 65 | 1785 | 455 | 10.412165641784668 | 10.412171363830566 | 5.495538675906743e-07 | True |
| 10 | 65 | 452 | 34 | 7.171318531036377 | 7.171317100524902 | 1.9947677242592679e-07 | True |
| 10 | 1596 | 1391 | 313 | 8.341978073120117 | 8.341978073120117 | 0.0 | True |
| 10 | 1596 | 434 | 322 | 7.557908058166504 | 7.5579071044921875 | 1.2618231249529183e-07 | True |
| 10 | 8593 | 455 | 187 | 2.6841320991516113 | 2.684131622314453 | 1.7765040638418712e-07 | True |
| 10 | 8593 | 954 | 291 | 1.7468444108963013 | 1.7468446493148804 | 1.3648529749666175e-07 | True |
| 10 | 8194 | 2270 | 152 | 0.8146477341651917 | 0.8146477937698364 | 7.316615793017621e-08 | True |
| 10 | 8194 | 115 | 99 | 0.7833586931228638 | 0.783359169960022 | 6.087085806148535e-07 | True |
| 11 | 747 | 75 | 233 | 10.999449729919434 | 10.999454498291016 | 4.335100117836691e-07 | True |
| 11 | 747 | 2172 | 378 | 7.028768062591553 | 7.028762340545654 | 8.140894460426603e-07 | True |
| 11 | 2309 | 1135 | 79 | 4.536155700683594 | 4.536153793334961 | 4.2047688806737046e-07 | True |
| 11 | 2309 | 1618 | 182 | 4.166429042816162 | 4.166430473327637 | 3.433423346248728e-07 | True |
| 11 | 8295 | 1814 | 39 | 1.3202542066574097 | 1.3202540874481201 | 9.029267920500908e-08 | True |
| 11 | 8295 | 1751 | 210 | 1.174645185470581 | 1.1746454238891602 | 2.0297072005283735e-07 | True |
| 11 | 8299 | 1449 | 9 | 1.1992628574371338 | 1.199262261390686 | 4.970106795666782e-07 | True |
| 11 | 8299 | 1658 | 17 | 1.1157208681106567 | 1.115721344947815 | 4.2738033484180787e-07 | True |
| 12 | 6205 | 1784 | 17 | 6.7961201667785645 | 6.796122074127197 | 2.8065257617665173e-07 | True |
| 12 | 6205 | 94 | 323 | 5.712130069732666 | 5.712127685546875 | 4.1738996869992624e-07 | True |
| 12 | 7423 | 27 | 129 | 3.04311466217041 | 3.0431127548217773 | 6.267751447302156e-07 | True |
| 12 | 7423 | 1739 | 261 | 2.7316694259643555 | 2.7316689491271973 | 1.7455888097981996e-07 | True |
| 12 | 8333 | 1870 | 339 | 1.6111825704574585 | 1.6111825704574585 | 0.0 | True |
| 12 | 8333 | 118 | 435 | 1.3624426126480103 | 1.3624433279037476 | 5.249804510404544e-07 | True |
| 12 | 8501 | 1237 | 255 | 4.651230812072754 | 4.651234149932861 | 7.176294280554969e-07 | True |
| 12 | 8501 | 469 | 66 | 4.282090663909912 | 4.282088756561279 | 4.454246260799459e-07 | True |
| 13 | 5046 | 1974 | 159 | 35.49806594848633 | 35.49806594848633 | 0.0 | True |
| 13 | 5046 | 1113 | 48 | 17.862272262573242 | 17.862274169921875 | 1.0678085098999196e-07 | True |
| 13 | 7001 | 764 | 343 | 24.583066940307617 | 24.583057403564453 | 3.8793951898758336e-07 | True |
| 13 | 7001 | 867 | 101 | 20.84154510498047 | 20.841554641723633 | 4.57583308532318e-07 | True |
| 13 | 8237 | 1065 | 134 | 3.143796682357788 | 3.1437952518463135 | 4.5502671423920407e-07 | True |
| 13 | 8237 | 2300 | 181 | 2.5890109539031982 | 2.589010000228882 | 3.683546857801002e-07 | True |
| 13 | 8618 | 378 | 77 | 14.04815673828125 | 14.048155784606934 | 6.788608172398063e-08 | True |
| 13 | 8618 | 581 | 43 | 13.207986831665039 | 13.207991600036621 | 3.610218304124501e-07 | True |
| 14 | 5888 | 1021 | 50 | 15.562504768371582 | 15.562506675720215 | 1.2256051716616307e-07 | True |
| 14 | 5888 | 865 | 256 | 12.463746070861816 | 12.463748931884766 | 2.2954759612018654e-07 | True |
| 14 | 6925 | 572 | 231 | 13.755989074707031 | 13.755990982055664 | 1.386558699962563e-07 | True |
| 14 | 6925 | 1635 | 28 | 12.615540504455566 | 12.615535736083984 | 3.7797600351306014e-07 | True |
| 14 | 8475 | 614 | 434 | 13.228647232055664 | 13.22864818572998 | 7.209159785403499e-08 | True |
| 14 | 8475 | 1249 | 24 | 6.627888202667236 | 6.627885341644287 | 4.316643343602868e-07 | True |
| 14 | 8204 | 229 | 136 | 5.029248237609863 | 5.029247283935547 | 1.8962562024170062e-07 | True |
| 14 | 8204 | 278 | 30 | 4.816563606262207 | 4.816564083099365 | 9.899945213703188e-08 | True |
| 15 | 6299 | 666 | 59 | 28.660112380981445 | 28.66010856628418 | 1.3310126683789257e-07 | True |
| 15 | 6299 | 1849 | 363 | 23.970638275146484 | 23.970645904541016 | 3.182808252194268e-07 | True |
| 15 | 5902 | 1584 | 196 | 19.056137084960938 | 19.056129455566406 | 4.003641712501692e-07 | True |
| 15 | 5902 | 757 | 10 | 17.494997024536133 | 17.494993209838867 | 2.1804503654816393e-07 | True |
| 15 | 8259 | 2371 | 213 | 9.20541000366211 | 9.205410957336426 | 1.0359933083120229e-07 | True |
| 15 | 8259 | 2471 | 137 | 7.452627182006836 | 7.452630043029785 | 3.838945487741863e-07 | True |
| 15 | 8354 | 434 | 259 | 6.409968376159668 | 6.409965515136719 | 4.463396355993947e-07 | True |
| 15 | 8354 | 1570 | 431 | 6.194802761077881 | 6.19480037689209 | 3.848687170470594e-07 | True |

### pretrain_210B: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 121 | 2346 | 272 | 0.4663863480091095 | 0.4663863480091095 | 0.0 | True |
| 0 | 121 | 125 | 30 | 0.3703739047050476 | 0.37037402391433716 | 3.218620103533353e-07 | True |
| 0 | 6507 | 734 | 105 | 0.4134253263473511 | 0.4134253263473511 | 0.0 | True |
| 0 | 6507 | 1594 | 118 | 0.36016660928726196 | 0.36016660928726196 | 0.0 | True |
| 0 | 8394 | 1449 | 308 | 0.3664305806159973 | 0.36643052101135254 | 1.6266285601815968e-07 | True |
| 0 | 8394 | 1195 | 9 | 0.3016337454319 | 0.30163371562957764 | 9.880301139722378e-08 | True |
| 0 | 8356 | 1307 | 31 | 0.34114402532577515 | 0.34114405512809753 | 8.735994235641563e-08 | True |
| 0 | 8356 | 990 | 220 | 0.32200494408607483 | 0.32200491428375244 | 9.255237515772082e-08 | True |
| 1 | 6910 | 584 | 149 | 0.8877602219581604 | 0.8877599835395813 | 2.6856190805178813e-07 | True |
| 1 | 6910 | 406 | 78 | 0.35907384753227234 | 0.35907384753227234 | 0.0 | True |
| 1 | 1786 | 755 | 100 | 0.32679322361946106 | 0.32679328322410583 | 1.823925359137743e-07 | True |
| 1 | 1786 | 2381 | 142 | 0.2870578467845917 | 0.2870578467845917 | 0.0 | True |
| 1 | 8556 | 2285 | 12 | 0.7175061702728271 | 0.7175067663192749 | 8.307196125257897e-07 | True |
| 1 | 8556 | 2213 | 47 | 0.4725247025489807 | 0.47252485156059265 | 3.153520041061354e-07 | True |
| 1 | 8493 | 2369 | 361 | 0.3575364351272583 | 0.3575364649295807 | 8.335464433740226e-08 | True |
| 1 | 8493 | 1473 | 14 | 0.33540838956832886 | 0.3354085087776184 | 3.554153481497699e-07 | True |
| 2 | 2428 | 1921 | 198 | 0.9102218747138977 | 0.9102218747138977 | 0.0 | True |
| 2 | 2428 | 2155 | 73 | 0.5892104506492615 | 0.589210569858551 | 2.0232039234779765e-07 | True |
| 2 | 4782 | 622 | 12 | 0.3896567225456238 | 0.3896567225456238 | 0.0 | True |
| 2 | 4782 | 77 | 97 | 0.3382488489151001 | 0.3382488787174225 | 8.810768309569516e-08 | True |
| 2 | 8207 | 407 | 14 | 0.3349556028842926 | 0.33495551347732544 | 2.6692184394948235e-07 | True |
| 2 | 8207 | 1352 | 69 | 0.2603076994419098 | 0.2603077292442322 | 1.1448882400171184e-07 | True |
| 2 | 8352 | 650 | 27 | 0.2478947639465332 | 0.2478947639465332 | 0.0 | True |
| 2 | 8352 | 25 | 242 | 0.22724014520645142 | 0.22724014520645142 | 0.0 | True |
| 3 | 7222 | 511 | 5 | 0.7847153544425964 | 0.7847152352333069 | 1.519140524980025e-07 | True |
| 3 | 7222 | 283 | 209 | 0.5699000954627991 | 0.5699000954627991 | 0.0 | True |
| 3 | 4212 | 2393 | 188 | 0.8533362150192261 | 0.8533360958099365 | 1.396979144358656e-07 | True |
| 3 | 4212 | 229 | 203 | 0.7865450978279114 | 0.7865450978279114 | 0.0 | True |
| 3 | 8473 | 1282 | 8 | 1.831908106803894 | 1.8319079875946045 | 6.507383700526558e-08 | True |
| 3 | 8473 | 1973 | 176 | 0.9886648654937744 | 0.9886648654937744 | 0.0 | True |
| 3 | 8359 | 32 | 20 | 0.2804829776287079 | 0.2804830074310303 | 1.062535867226368e-07 | True |
| 3 | 8359 | 1065 | 39 | 0.2724173367023468 | 0.2724173069000244 | 1.0939950719897987e-07 | True |
| 4 | 1819 | 2442 | 168 | 1.419657826423645 | 1.419657826423645 | 0.0 | True |
| 4 | 1819 | 793 | 159 | 0.8244949579238892 | 0.8244943618774414 | 7.229230961639532e-07 | True |
| 4 | 191 | 111 | 34 | 0.7564833164215088 | 0.7564837336540222 | 5.515422539672434e-07 | True |
| 4 | 191 | 1885 | 2 | 0.657924234867096 | 0.6579240560531616 | 2.7178499415254584e-07 | True |
| 4 | 8406 | 1301 | 253 | 0.5437445640563965 | 0.5437445044517517 | 1.0961883339252751e-07 | True |
| 4 | 8406 | 1258 | 137 | 0.33354929089546204 | 0.33354926109313965 | 8.934908033438357e-08 | True |
| 4 | 8208 | 2139 | 108 | 0.3144523799419403 | 0.3144523799419403 | 0.0 | True |
| 4 | 8208 | 819 | 2 | 0.30227991938591003 | 0.302280068397522 | 4.92959017063382e-07 | True |
| 5 | 3718 | 198 | 245 | 1.1170909404754639 | 1.1170908212661743 | 1.0671404200990349e-07 | True |
| 5 | 3718 | 707 | 23 | 0.8830445408821106 | 0.8830443620681763 | 2.024970723985758e-07 | True |
| 5 | 2926 | 1007 | 74 | 0.9054421186447144 | 0.9054422974586487 | 1.9748797923585056e-07 | True |
| 5 | 2926 | 1717 | 20 | 0.8263708353042603 | 0.826370894908905 | 7.212820471023143e-08 | True |
| 5 | 8610 | 2090 | 5 | 1.7632697820663452 | 1.7632702589035034 | 2.704277944605435e-07 | True |
| 5 | 8610 | 1247 | 10 | 1.0310789346694946 | 1.0310791730880737 | 2.312321308144909e-07 | True |
| 5 | 8637 | 1294 | 98 | 0.6946918368339539 | 0.6946918368339539 | 0.0 | True |
| 5 | 8637 | 1832 | 287 | 0.637153148651123 | 0.6371530890464783 | 9.354837985431898e-08 | True |
| 6 | 4186 | 1307 | 24 | 1.7857458591461182 | 1.7857465744018555 | 4.005361309624977e-07 | True |
| 6 | 4186 | 838 | 114 | 1.5749903917312622 | 1.5749903917312622 | 0.0 | True |
| 6 | 1079 | 2449 | 396 | 0.8117538690567017 | 0.8117547631263733 | 1.1014048786361964e-06 | True |
| 6 | 1079 | 111 | 105 | 0.7610043287277222 | 0.7610043287277222 | 0.0 | True |
| 6 | 8522 | 1760 | 8 | 3.1622583866119385 | 3.1622602939605713 | 6.031602733311253e-07 | True |
| 6 | 8522 | 1714 | 5 | 2.2174999713897705 | 2.2175002098083496 | 1.075168352548563e-07 | True |
| 6 | 8449 | 542 | 1 | 0.9427351355552673 | 0.9427351355552673 | 0.0 | True |
| 6 | 8449 | 2381 | 253 | 0.899890124797821 | 0.8998900055885315 | 1.3247093869105864e-07 | True |
| 7 | 4214 | 2319 | 6 | 1.7218719720840454 | 1.7218719720840454 | 0.0 | True |
| 7 | 4214 | 1896 | 34 | 1.3537746667861938 | 1.3537744283676147 | 1.7611393162464662e-07 | True |
| 7 | 5073 | 2371 | 194 | 1.712356686592102 | 1.7123582363128662 | 9.050221699103937e-07 | True |
| 7 | 5073 | 354 | 178 | 1.5728858709335327 | 1.5728856325149536 | 1.515803425458055e-07 | True |
| 7 | 8521 | 2255 | 11 | 1.4936884641647339 | 1.4936881065368652 | 2.3942600966147793e-07 | True |
| 7 | 8521 | 60 | 251 | 1.3749710321426392 | 1.3749704360961914 | 4.3349745836105187e-07 | True |
| 7 | 8388 | 1654 | 87 | 0.6157593727111816 | 0.6157594919204712 | 1.935971985710978e-07 | True |
| 7 | 8388 | 136 | 370 | 0.5864994525909424 | 0.5864995718002319 | 2.0325558536185796e-07 | True |
| 8 | 980 | 570 | 327 | 3.142954111099243 | 3.142953395843506 | 2.2757434948820424e-07 | True |
| 8 | 980 | 92 | 209 | 2.344338893890381 | 2.344338893890381 | 0.0 | True |
| 8 | 2632 | 1321 | 189 | 2.2044625282287598 | 2.204460859298706 | 7.570689146854714e-07 | True |
| 8 | 2632 | 155 | 16 | 1.9447474479675293 | 1.9447474479675293 | 0.0 | True |
| 8 | 8673 | 2422 | 207 | 2.245234489440918 | 2.2452340126037598 | 2.1237744228749194e-07 | True |
| 8 | 8673 | 2164 | 146 | 1.4346287250518799 | 1.4346288442611694 | 8.309417445023648e-08 | True |
| 8 | 8418 | 7 | 101 | 0.7607758641242981 | 0.7607757449150085 | 1.5669436317883034e-07 | True |
| 8 | 8418 | 591 | 114 | 0.7261798977851868 | 0.7261799573898315 | 8.207972288572278e-08 | True |
| 9 | 4647 | 1162 | 47 | 3.325857400894165 | 3.325857400894165 | 0.0 | True |
| 9 | 4647 | 707 | 118 | 2.2706096172332764 | 2.2706120014190674 | 1.0500201236356694e-06 | True |
| 9 | 6063 | 277 | 207 | 2.0686492919921875 | 2.068650484085083 | 5.762663106416564e-07 | True |
| 9 | 6063 | 2038 | 328 | 1.8628357648849487 | 1.8628369569778442 | 6.399345116618146e-07 | True |
| 9 | 8560 | 1487 | 113 | 2.473318099975586 | 2.473317861557007 | 9.639624563614195e-08 | True |
| 9 | 8560 | 2272 | 163 | 1.749430775642395 | 1.749429702758789 | 6.1327582714044e-07 | True |
| 9 | 8228 | 329 | 255 | 1.182681918144226 | 1.1826815605163574 | 3.0238719571658457e-07 | True |
| 9 | 8228 | 159 | 142 | 1.1287461519241333 | 1.1287459135055542 | 2.112242674716001e-07 | True |
| 10 | 6666 | 2233 | 303 | 17.831594467163086 | 17.831588745117188 | 3.2089367605205796e-07 | True |
| 10 | 6666 | 1921 | 65 | 7.890003204345703 | 7.890003204345703 | 0.0 | True |
| 10 | 7847 | 73 | 210 | 3.0248751640319824 | 3.024876594543457 | 4.729158715768576e-07 | True |
| 10 | 7847 | 2452 | 73 | 2.8963816165924072 | 2.89638090133667 | 2.4694803102161165e-07 | True |
| 10 | 8591 | 92 | 11 | 5.024823188781738 | 5.02482271194458 | 9.489630585762631e-08 | True |
| 10 | 8591 | 1680 | 168 | 3.540839672088623 | 3.540839672088623 | 0.0 | True |
| 10 | 8570 | 1182 | 275 | 3.301926612854004 | 3.3019261360168457 | 1.4441179775069961e-07 | True |
| 10 | 8570 | 2465 | 155 | 3.1451804637908936 | 3.1451804637908936 | 0.0 | True |
| 11 | 8136 | 1927 | 151 | 7.690424919128418 | 7.690423488616943 | 1.8601202009673866e-07 | True |
| 11 | 8136 | 1947 | 239 | 5.66998291015625 | 5.669981479644775 | 2.5229555313949856e-07 | True |
| 11 | 1358 | 2077 | 183 | 4.743950366973877 | 4.743947505950928 | 6.030887188737117e-07 | True |
| 11 | 1358 | 1386 | 300 | 4.3784379959106445 | 4.378438472747803 | 1.0890576928312777e-07 | True |
| 11 | 8593 | 673 | 93 | 6.476912975311279 | 6.476912975311279 | 0.0 | True |
| 11 | 8593 | 459 | 434 | 4.448263168334961 | 4.448263645172119 | 1.0719625619219174e-07 | True |
| 11 | 8636 | 755 | 5 | 4.148157596588135 | 4.14815616607666 | 3.448546592795733e-07 | True |
| 11 | 8636 | 2404 | 351 | 3.6321911811828613 | 3.6321897506713867 | 3.9384256038623877e-07 | True |
| 12 | 6320 | 1818 | 249 | 16.393251419067383 | 16.393245697021484 | 3.490488709141648e-07 | True |
| 12 | 6320 | 363 | 38 | 13.765702247619629 | 13.765708923339844 | 4.849531171573988e-07 | True |
| 12 | 39 | 590 | 374 | 13.638179779052734 | 13.638181686401367 | 1.3985360683850574e-07 | True |
| 12 | 39 | 1739 | 207 | 11.470098495483398 | 11.470096588134766 | 1.6628877542451446e-07 | True |
| 12 | 8357 | 1899 | 204 | 2.7771692276000977 | 2.7771682739257812 | 3.4339798487194515e-07 | True |
| 12 | 8357 | 5 | 32 | 2.4547677040100098 | 2.454768180847168 | 1.9424940185752933e-07 | True |
| 12 | 8599 | 14 | 67 | 6.0807623863220215 | 6.0807647705078125 | 3.920866561697227e-07 | True |
| 12 | 8599 | 1216 | 225 | 5.4882988929748535 | 5.488296031951904 | 5.21295032397183e-07 | True |
| 13 | 5632 | 1237 | 308 | 18.143999099731445 | 18.144004821777344 | 3.1536850652302964e-07 | True |
| 13 | 5632 | 382 | 116 | 13.147683143615723 | 13.14768123626709 | 1.4507108301728993e-07 | True |
| 13 | 5964 | 764 | 148 | 11.703280448913574 | 11.703283309936523 | 2.4446333331133165e-07 | True |
| 13 | 5964 | 652 | 281 | 10.461666107177734 | 10.461664199829102 | 1.823178653640906e-07 | True |
| 13 | 8573 | 1664 | 107 | 12.817179679870605 | 12.817184448242188 | 3.720297055303034e-07 | True |
| 13 | 8573 | 1771 | 175 | 7.402673244476318 | 7.402677059173584 | 5.153134738820774e-07 | True |
| 13 | 8331 | 797 | 14 | 4.530808448791504 | 4.530808448791504 | 0.0 | True |
| 13 | 8331 | 1232 | 98 | 4.1190667152404785 | 4.1190667152404785 | 0.0 | True |
| 14 | 1516 | 1020 | 288 | 18.907278060913086 | 18.90727424621582 | 2.0175814061311676e-07 | True |
| 14 | 1516 | 2093 | 120 | 15.273343086242676 | 15.273343086242676 | 0.0 | True |
| 14 | 7459 | 2020 | 200 | 23.656787872314453 | 23.65679168701172 | 1.6125170019761397e-07 | True |
| 14 | 7459 | 782 | 79 | 21.87436294555664 | 21.87436294555664 | 0.0 | True |
| 14 | 8414 | 2193 | 451 | 10.582883834838867 | 10.582889556884766 | 5.406887184758201e-07 | True |
| 14 | 8414 | 434 | 246 | 9.20580005645752 | 9.205803871154785 | 4.1437976517305897e-07 | True |
| 14 | 8615 | 522 | 286 | 25.434667587280273 | 25.434669494628906 | 7.499011442816551e-08 | True |
| 14 | 8615 | 1682 | 108 | 22.418701171875 | 22.418712615966797 | 5.10470776568983e-07 | True |
| 15 | 259 | 12 | 217 | 138.1961212158203 | 138.19615173339844 | 2.2082803668086186e-07 | True |
| 15 | 259 | 2185 | 78 | 63.68017578125 | 63.68014144897461 | 5.391360022083011e-07 | True |
| 15 | 6434 | 608 | 30 | 24.732072830200195 | 24.732078552246094 | 2.3136135566649075e-07 | True |
| 15 | 6434 | 67 | 20 | 22.388803482055664 | 22.388811111450195 | 3.407683013236889e-07 | True |
| 15 | 8511 | 1182 | 74 | 47.21849822998047 | 47.21849060058594 | 1.6157639097479522e-07 | True |
| 15 | 8511 | 387 | 47 | 35.35651397705078 | 35.35652542114258 | 3.2367704022809306e-07 | True |
| 15 | 8655 | 1482 | 24 | 20.305339813232422 | 20.305349349975586 | 4.6966676016166306e-07 | True |
| 15 | 8655 | 534 | 178 | 17.346254348754883 | 17.346263885498047 | 5.497868861093374e-07 | True |

### pretrain_21B: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 5689 | 344 | 2 | 37.15035629272461 | 37.15034866333008 | 2.0536531254598257e-07 | True |
| 0 | 5689 | 506 | 1 | 25.993896484375 | 25.993896484375 | 0.0 | True |
| 0 | 7404 | 2148 | 60 | 13.892102241516113 | 13.892102241516113 | 0.0 | True |
| 0 | 7404 | 2267 | 92 | 13.004422187805176 | 13.004416465759277 | 4.4000770013475236e-07 | True |
| 0 | 8384 | 2349 | 0 | 19.468902587890625 | 19.468904495239258 | 9.796898537048735e-08 | True |
| 0 | 8384 | 380 | 2 | 13.661520957946777 | 13.661521911621094 | 6.980733106817852e-08 | True |
| 0 | 8532 | 2369 | 184 | 19.250333786010742 | 19.250333786010742 | 0.0 | True |
| 0 | 8532 | 775 | 174 | 17.90966796875 | 17.90966796875 | 0.0 | True |
| 1 | 6910 | 584 | 86 | 21.259502410888672 | 21.259506225585938 | 1.7943492711622414e-07 | True |
| 1 | 6910 | 611 | 139 | 10.970415115356445 | 10.970414161682129 | 8.693147035715099e-08 | True |
| 1 | 2725 | 527 | 21 | 32.80422592163086 | 32.80422592163086 | 0.0 | True |
| 1 | 2725 | 1422 | 79 | 29.229652404785156 | 29.22965431213379 | 6.525389376509486e-08 | True |
| 1 | 8560 | 2111 | 25 | 11.767610549926758 | 11.767609596252441 | 8.104230781262435e-08 | True |
| 1 | 8560 | 2193 | 346 | 8.741355895996094 | 8.741354942321777 | 1.0909912921439027e-07 | True |
| 1 | 8366 | 215 | 10 | 4.169154167175293 | 4.169152736663818 | 3.431179124706205e-07 | True |
| 1 | 8366 | 1072 | 47 | 4.048315525054932 | 4.048314094543457 | 3.5335967904575924e-07 | True |
| 2 | 3152 | 2040 | 54 | 16.514066696166992 | 16.514066696166992 | 0.0 | True |
| 2 | 3152 | 751 | 112 | 10.996490478515625 | 10.996490478515625 | 0.0 | True |
| 2 | 4782 | 1785 | 174 | 10.122405052185059 | 10.122405052185059 | 0.0 | True |
| 2 | 4782 | 176 | 111 | 8.766121864318848 | 8.766120910644531 | 1.0879090333982634e-07 | True |
| 2 | 8379 | 141 | 131 | 4.591861724853516 | 4.591859817504883 | 4.1537588610060905e-07 | True |
| 2 | 8379 | 1207 | 75 | 3.997790813446045 | 3.997790813446045 | 0.0 | True |
| 2 | 8240 | 1142 | 122 | 4.1735920906066895 | 4.1735920906066895 | 0.0 | True |
| 2 | 8240 | 2230 | 453 | 3.9357049465179443 | 3.93570613861084 | 3.0289183556874575e-07 | True |
| 3 | 1997 | 1609 | 25 | 13.237310409545898 | 13.23730754852295 | 2.1613325220171341e-07 | True |
| 3 | 1997 | 1518 | 7 | 9.712821960449219 | 9.712817192077637 | 4.90935754968859e-07 | True |
| 3 | 3230 | 434 | 146 | 17.979028701782227 | 17.979028701782227 | 0.0 | True |
| 3 | 3230 | 2319 | 13 | 16.26650619506836 | 16.26650619506836 | 0.0 | True |
| 3 | 8444 | 2420 | 209 | 7.261552810668945 | 7.261548042297363 | 6.56660043155698e-07 | True |
| 3 | 8444 | 1840 | 218 | 5.1390299797058105 | 5.139031410217285 | 2.7836215788942843e-07 | True |
| 3 | 8694 | 334 | 42 | 28.14376449584961 | 28.14376449584961 | 0.0 | True |
| 3 | 8694 | 261 | 38 | 27.400619506835938 | 27.400615692138672 | 1.3921938022873188e-07 | True |
| 4 | 7660 | 178 | 68 | 34.01215744018555 | 34.012168884277344 | 3.3647062280600127e-07 | True |
| 4 | 7660 | 970 | 3 | 24.372386932373047 | 24.372398376464844 | 4.6955153915081606e-07 | True |
| 4 | 1254 | 769 | 282 | 18.814647674560547 | 18.814651489257812 | 2.0275145894881074e-07 | True |
| 4 | 1254 | 363 | 34 | 16.723247528076172 | 16.723247528076172 | 0.0 | True |
| 4 | 8677 | 1916 | 21 | 17.503931045532227 | 17.503944396972656 | 7.627681116291517e-07 | True |
| 4 | 8677 | 1141 | 17 | 13.098628997802734 | 13.098628044128418 | 7.280718589451054e-08 | True |
| 4 | 8221 | 1402 | 285 | 8.174278259277344 | 8.174280166625977 | 2.3333541779639898e-07 | True |
| 4 | 8221 | 2489 | 127 | 7.756537914276123 | 7.7565388679504395 | 1.2295102878965962e-07 | True |
| 5 | 4143 | 1157 | 31 | 30.119760513305664 | 30.1197509765625 | 3.1662745657786894e-07 | True |
| 5 | 4143 | 2396 | 31 | 20.805049896240234 | 20.805055618286133 | 2.7503158737781035e-07 | True |
| 5 | 6870 | 67 | 134 | 28.191539764404297 | 28.191537857055664 | 6.765677393828593e-08 | True |
| 5 | 6870 | 221 | 93 | 24.482677459716797 | 24.482694625854492 | 7.011544273928881e-07 | True |
| 5 | 8613 | 1861 | 178 | 56.78406524658203 | 56.78408432006836 | 3.358950481142785e-07 | True |
| 5 | 8613 | 1011 | 81 | 34.63279342651367 | 34.63279342651367 | 0.0 | True |
| 5 | 8413 | 1647 | 0 | 12.116676330566406 | 12.11667251586914 | 3.148303347842814e-07 | True |
| 5 | 8413 | 2134 | 0 | 11.750399589538574 | 11.750401496887207 | 1.6232202303235882e-07 | True |
| 6 | 4186 | 363 | 18 | 41.0693359375 | 41.0693359375 | 0.0 | True |
| 6 | 4186 | 2098 | 26 | 33.03372573852539 | 33.033729553222656 | 1.1547886834866258e-07 | True |
| 6 | 1386 | 886 | 65 | 33.8085823059082 | 33.808597564697266 | 4.513288645005815e-07 | True |
| 6 | 1386 | 2093 | 470 | 28.561996459960938 | 28.561994552612305 | 6.677924757417705e-08 | True |
| 6 | 8498 | 2465 | 175 | 34.828983306884766 | 34.828983306884766 | 0.0 | True |
| 6 | 8498 | 2404 | 467 | 22.27145767211914 | 22.271465301513672 | 3.4256377124344996e-07 | True |
| 6 | 8196 | 92 | 376 | 12.680198669433594 | 12.680196762084961 | 1.504194597053343e-07 | True |
| 6 | 8196 | 1739 | 250 | 12.199034690856934 | 12.1990385055542 | 3.127048461042644e-07 | True |
| 7 | 2427 | 874 | 131 | 42.13238525390625 | 42.13240432739258 | 4.527036913096826e-07 | True |
| 7 | 2427 | 1182 | 322 | 27.015705108642578 | 27.015710830688477 | 2.1180442544166518e-07 | True |
| 7 | 5073 | 316 | 181 | 52.01982879638672 | 52.01981735229492 | 2.19994799322944e-07 | True |
| 7 | 5073 | 685 | 276 | 46.57539749145508 | 46.57541275024414 | 3.276147898748356e-07 | True |
| 7 | 8364 | 192 | 218 | 20.50861167907715 | 20.50861930847168 | 3.7200931250912006e-07 | True |
| 7 | 8364 | 1721 | 383 | 16.890125274658203 | 16.89012908935547 | 2.25853698749561e-07 | True |
| 7 | 8393 | 667 | 344 | 16.177711486816406 | 16.17771339416504 | 1.1789978047060877e-07 | True |
| 7 | 8393 | 1182 | 275 | 15.233044624328613 | 15.233047485351562 | 1.8781688229609894e-07 | True |
| 8 | 3552 | 2474 | 66 | 51.84780502319336 | 51.84779739379883 | 1.4714980755380293e-07 | True |
| 8 | 3552 | 555 | 118 | 34.71082305908203 | 34.710819244384766 | 1.0989936075937821e-07 | True |
| 8 | 3562 | 1288 | 149 | 48.851966857910156 | 48.851966857910156 | 0.0 | True |
| 8 | 3562 | 584 | 107 | 46.07248306274414 | 46.07249069213867 | 1.6559547096386913e-07 | True |
| 8 | 8323 | 1539 | 174 | 22.91952896118164 | 22.919538497924805 | 4.160968220688434e-07 | True |
| 8 | 8323 | 387 | 253 | 20.94701385498047 | 20.94700813293457 | 2.7316761892898624e-07 | True |
| 8 | 8631 | 237 | 39 | 59.098106384277344 | 59.09810256958008 | 6.454855322809251e-08 | True |
| 8 | 8631 | 308 | 243 | 54.547393798828125 | 54.5473747253418 | 3.4966815093803377e-07 | True |
| 9 | 2816 | 640 | 198 | 126.72222137451172 | 126.72222137451172 | 0.0 | True |
| 9 | 2816 | 1895 | 180 | 95.34062194824219 | 95.34062194824219 | 0.0 | True |
| 9 | 4000 | 995 | 126 | 12.560007095336914 | 12.560007095336914 | 0.0 | True |
| 9 | 4000 | 1216 | 149 | 11.030311584472656 | 11.030306816101074 | 4.3229708839265027e-07 | True |
| 9 | 8335 | 248 | 193 | 36.67205810546875 | 36.672054290771484 | 1.040219028518645e-07 | True |
| 9 | 8335 | 1961 | 175 | 31.22709846496582 | 31.227100372314453 | 6.107991861467324e-08 | True |
| 9 | 8638 | 434 | 315 | 55.863372802734375 | 55.86336898803711 | 6.828619673744941e-08 | True |
| 9 | 8638 | 1918 | 43 | 49.69350051879883 | 49.693519592285156 | 3.8382255484114237e-07 | True |
| 10 | 2597 | 1007 | 147 | 115.08174896240234 | 115.08175659179688 | 6.629543433288065e-08 | True |
| 10 | 2597 | 648 | 34 | 78.29521179199219 | 78.29517364501953 | 4.87219739025619e-07 | True |
| 10 | 7782 | 1288 | 78 | 91.61176300048828 | 91.61176300048828 | 0.0 | True |
| 10 | 7782 | 75 | 209 | 84.97298431396484 | 84.97299194335938 | 8.978611958667139e-08 | True |
| 10 | 8520 | 1735 | 202 | 98.03472900390625 | 98.03472900390625 | 0.0 | True |
| 10 | 8520 | 613 | 154 | 62.55195999145508 | 62.55194854736328 | 1.8295336866244194e-07 | True |
| 10 | 8367 | 1458 | 159 | 33.1694450378418 | 33.16944885253906 | 1.1500636387714514e-07 | True |
| 10 | 8367 | 1363 | 114 | 31.81435775756836 | 31.814359664916992 | 5.995244811625211e-08 | True |
| 11 | 3111 | 1340 | 36 | 124.72367858886719 | 124.7236328125 | 3.6702226638451625e-07 | True |
| 11 | 3111 | 1297 | 70 | 99.3517074584961 | 99.35169982910156 | 7.679178069926135e-08 | True |
| 11 | 7807 | 1254 | 27 | 125.9404067993164 | 125.9404067993164 | 0.0 | True |
| 11 | 7807 | 1951 | 149 | 111.14038848876953 | 111.14035034179688 | 3.4323231342766684e-07 | True |
| 11 | 8353 | 60 | 45 | 51.12002182006836 | 51.12001037597656 | 2.2386711486852992e-07 | True |
| 11 | 8353 | 2399 | 95 | 43.350372314453125 | 43.350364685058594 | 1.759937487020461e-07 | True |
| 11 | 8699 | 2170 | 280 | 50.999366760253906 | 50.999393463134766 | 5.235924003704641e-07 | True |
| 11 | 8699 | 1793 | 337 | 47.051448822021484 | 47.05146408081055 | 3.243000894662872e-07 | True |
| 12 | 6205 | 1325 | 116 | 166.1769561767578 | 166.177001953125 | 2.754675993632291e-07 | True |
| 12 | 6205 | 2406 | 72 | 144.16600036621094 | 144.16598510742188 | 1.0584180058917897e-07 | True |
| 12 | 1626 | 1552 | 253 | 114.96318054199219 | 114.9631576538086 | 1.9909142636663324e-07 | True |
| 12 | 1626 | 1123 | 166 | 108.0714111328125 | 108.07140350341797 | 7.059586296947661e-08 | True |
| 12 | 8319 | 235 | 476 | 62.13923263549805 | 62.139244079589844 | 1.8416854073504242e-07 | True |
| 12 | 8319 | 1672 | 323 | 54.51235580444336 | 54.51235580444336 | 0.0 | True |
| 12 | 8562 | 92 | 217 | 137.21261596679688 | 137.21253967285156 | 5.560271901744212e-07 | True |
| 12 | 8562 | 1423 | 219 | 122.63780975341797 | 122.63780975341797 | 0.0 | True |
| 13 | 8015 | 1038 | 40 | 506.9392395019531 | 506.93939208984375 | 3.009983815317025e-07 | True |
| 13 | 8015 | 1571 | 78 | 316.2278137207031 | 316.2279357910156 | 3.8602016395627433e-07 | True |
| 13 | 6280 | 1370 | 223 | 176.97830200195312 | 176.97830200195312 | 0.0 | True |
| 13 | 6280 | 14 | 40 | 165.26515197753906 | 165.2650604248047 | 5.539748294149924e-07 | True |
| 13 | 8451 | 1031 | 133 | 188.45269775390625 | 188.45265197753906 | 2.4290640427593e-07 | True |
| 13 | 8451 | 180 | 345 | 157.99200439453125 | 157.99208068847656 | 4.828975086738051e-07 | True |
| 13 | 8491 | 802 | 147 | 164.0290985107422 | 164.02903747558594 | 3.7209956528538034e-07 | True |
| 13 | 8491 | 2193 | 457 | 155.56130981445312 | 155.56134033203125 | 1.9617717388340366e-07 | True |
| 14 | 1681 | 1176 | 376 | 435.66656494140625 | 435.6666564941406 | 2.101440453373169e-07 | True |
| 14 | 1681 | 751 | 178 | 299.908447265625 | 299.90838623046875 | 2.0351262795856482e-07 | True |
| 14 | 1145 | 531 | 22 | 294.92510986328125 | 294.92486572265625 | 8.278054897162759e-07 | True |
| 14 | 1145 | 49 | 49 | 265.6484069824219 | 265.6483459472656 | 2.2975916529414286e-07 | True |
| 14 | 8695 | 1224 | 71 | 242.50726318359375 | 242.50718688964844 | 3.1460478466057544e-07 | True |
| 14 | 8695 | 229 | 445 | 198.09120178222656 | 198.09117126464844 | 1.5405822091255618e-07 | True |
| 14 | 8696 | 2231 | 285 | 177.26101684570312 | 177.260986328125 | 1.7216181351122468e-07 | True |
| 14 | 8696 | 1854 | 278 | 157.8587646484375 | 157.8587646484375 | 0.0 | True |
| 15 | 6239 | 1970 | 59 | 745.588623046875 | 745.5889282226562 | 4.093085272718997e-07 | True |
| 15 | 6239 | 49 | 98 | 496.5940856933594 | 496.5940246582031 | 1.2290753758128413e-07 | True |
| 15 | 1199 | 2245 | 47 | 739.6640014648438 | 739.6635131835938 | 6.601392646296145e-07 | True |
| 15 | 1199 | 913 | 123 | 677.7374267578125 | 677.73779296875 | 5.403433882232158e-07 | True |
| 15 | 8513 | 1337 | 144 | 876.8866577148438 | 876.8869018554688 | 2.7841753874580276e-07 | True |
| 15 | 8513 | 1721 | 301 | 642.1463012695312 | 642.1461791992188 | 1.900973536072161e-07 | True |
| 15 | 8354 | 2425 | 15 | 143.85232543945312 | 143.852294921875 | 2.121451845270637e-07 | True |
| 15 | 8354 | 1574 | 111 | 136.89540100097656 | 136.89541625976562 | 1.1146312404162612e-07 | True |

### pretrain_2T: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 4977 | 1838 | 174 | 0.9495394825935364 | 0.9495396018028259 | 1.2554432094300858e-07 | True |
| 0 | 4977 | 378 | 26 | 0.6236703395843506 | 0.6236703395843506 | 0.0 | True |
| 0 | 6908 | 463 | 333 | 0.7132818698883057 | 0.7132818698883057 | 0.0 | True |
| 0 | 6908 | 2415 | 94 | 0.6164461374282837 | 0.6164461970329285 | 9.669075878072954e-08 | True |
| 0 | 8369 | 24 | 184 | 1.3672500848770142 | 1.3672500848770142 | 0.0 | True |
| 0 | 8369 | 991 | 314 | 0.6941977143287659 | 0.6941973567008972 | 5.151671653055521e-07 | True |
| 0 | 8554 | 1259 | 293 | 0.5412993431091309 | 0.5412994027137756 | 1.1011401645720044e-07 | True |
| 0 | 8554 | 485 | 73 | 0.4866529703140259 | 0.4866529703140259 | 0.0 | True |
| 1 | 2644 | 1785 | 301 | 1.287634253501892 | 1.2876347303390503 | 3.70320342835167e-07 | True |
| 1 | 2644 | 1840 | 242 | 0.701151430606842 | 0.7011514902114868 | 8.500966007272236e-08 | True |
| 1 | 3431 | 947 | 160 | 1.0507944822311401 | 1.0507946014404297 | 1.1344681720983679e-07 | True |
| 1 | 3431 | 412 | 62 | 0.9627031683921814 | 0.9627031683921814 | 0.0 | True |
| 1 | 8627 | 1372 | 176 | 4.145096302032471 | 4.145096302032471 | 0.0 | True |
| 1 | 8627 | 352 | 25 | 3.1467339992523193 | 3.1467339992523193 | 0.0 | True |
| 1 | 8546 | 1127 | 80 | 0.696428120136261 | 0.696428120136261 | 0.0 | True |
| 1 | 8546 | 615 | 32 | 0.6242983341217041 | 0.6242984533309937 | 1.9094923538197676e-07 | True |
| 2 | 4397 | 2411 | 401 | 0.7391983866691589 | 0.7391983866691589 | 0.0 | True |
| 2 | 4397 | 1953 | 60 | 0.5400062203407288 | 0.5400063395500183 | 2.207554006980948e-07 | True |
| 2 | 532 | 86 | 15 | 1.4376691579818726 | 1.4376693964004517 | 1.6583688797792843e-07 | True |
| 2 | 532 | 1672 | 41 | 1.202273964881897 | 1.2022731304168701 | 6.940722757291353e-07 | True |
| 2 | 8215 | 2322 | 72 | 0.6925687789916992 | 0.6925686001777649 | 2.581894242857792e-07 | True |
| 2 | 8215 | 168 | 75 | 0.4190122187137604 | 0.4190124571323395 | 5.690014955493058e-07 | True |
| 2 | 8243 | 1634 | 128 | 0.8844311237335205 | 0.8844312429428101 | 1.3478640263987257e-07 | True |
| 2 | 8243 | 353 | 147 | 0.8274149894714355 | 0.8274151682853699 | 2.1611154813668623e-07 | True |
| 3 | 1778 | 913 | 225 | 1.2150689363479614 | 1.2150682210922241 | 5.886544507132873e-07 | True |
| 3 | 1778 | 484 | 119 | 0.7233091592788696 | 0.7233093976974487 | 3.2962195493177897e-07 | True |
| 3 | 5000 | 1129 | 8 | 0.7877779006958008 | 0.7877779006958008 | 0.0 | True |
| 3 | 5000 | 2322 | 28 | 0.7152554392814636 | 0.7152554392814636 | 0.0 | True |
| 3 | 8487 | 846 | 34 | 1.7581558227539062 | 1.7581559419631958 | 6.7803597387663e-08 | True |
| 3 | 8487 | 990 | 208 | 1.4117110967636108 | 1.41171133518219 | 1.6888624000204015e-07 | True |
| 3 | 8278 | 932 | 10 | 0.6268296241760254 | 0.6268292665481567 | 5.705344081694441e-07 | True |
| 3 | 8278 | 2322 | 110 | 0.6035413146018982 | 0.6035415530204773 | 3.950327398196853e-07 | True |
| 4 | 7660 | 610 | 199 | 1.8594422340393066 | 1.859442949295044 | 3.846614453576877e-07 | True |
| 4 | 7660 | 2447 | 134 | 0.7655157446861267 | 0.7655150294303894 | 9.343449070377428e-07 | True |
| 4 | 1336 | 1438 | 14 | 0.916663408279419 | 0.9166631102561951 | 3.251173998931014e-07 | True |
| 4 | 1336 | 1785 | 167 | 0.7591136693954468 | 0.7591136693954468 | 0.0 | True |
| 4 | 8319 | 208 | 0 | 0.6140451431274414 | 0.6140449643135071 | 2.912064956909204e-07 | True |
| 4 | 8319 | 2355 | 0 | 0.5740208029747009 | 0.5740211009979248 | 5.191854064043181e-07 | True |
| 4 | 8681 | 841 | 413 | 0.8435567021369934 | 0.8435566425323486 | 7.065872943027231e-08 | True |
| 4 | 8681 | 1538 | 3 | 0.7429891228675842 | 0.7429890036582947 | 1.6044553800557692e-07 | True |
| 5 | 1503 | 1006 | 8 | 2.7662322521209717 | 2.7662312984466553 | 3.447556927568294e-07 | True |
| 5 | 1503 | 1088 | 163 | 1.919402837753296 | 1.9194022417068481 | 3.1053744218258633e-07 | True |
| 5 | 1862 | 1030 | 328 | 0.7674909234046936 | 0.7674906849861145 | 3.1064677357212956e-07 | True |
| 5 | 1862 | 1343 | 118 | 0.6966822743415833 | 0.6966817378997803 | 7.699949069114658e-07 | True |
| 5 | 8526 | 1787 | 28 | 2.590505361557007 | 2.5905065536499023 | 4.6017773720850847e-07 | True |
| 5 | 8526 | 2239 | 350 | 0.8247659206390381 | 0.8247658610343933 | 7.226855921642381e-08 | True |
| 5 | 8688 | 2218 | 70 | 1.3816423416137695 | 1.3816423416137695 | 0.0 | True |
| 5 | 8688 | 1216 | 144 | 0.9480717182159424 | 0.9480719566345215 | 2.5147736665978456e-07 | True |
| 6 | 3443 | 390 | 165 | 1.88722825050354 | 1.8872298002243042 | 8.2116233886742e-07 | True |
| 6 | 3443 | 1856 | 124 | 1.4714618921279907 | 1.4714622497558594 | 2.430425623426451e-07 | True |
| 6 | 1822 | 947 | 123 | 0.8463495373725891 | 0.8463484644889832 | 1.267660178899247e-06 | True |
| 6 | 1822 | 2254 | 8 | 0.7567383646965027 | 0.7567382454872131 | 1.5753038977823107e-07 | True |
| 6 | 8588 | 192 | 169 | 1.9375252723693848 | 1.9375250339508057 | 1.2305314542297674e-07 | True |
| 6 | 8588 | 932 | 23 | 1.0067896842956543 | 1.0067905187606812 | 8.288374820201469e-07 | True |
| 6 | 8658 | 1859 | 116 | 0.6604371666908264 | 0.6604368686676025 | 4.5125144208679606e-07 | True |
| 6 | 8658 | 1393 | 38 | 0.6272057294845581 | 0.6272055506706238 | 2.850961429722945e-07 | True |
| 7 | 4455 | 2038 | 10 | 1.2941551208496094 | 1.2941547632217407 | 2.7634080558871644e-07 | True |
| 7 | 4455 | 24 | 139 | 0.883198618888855 | 0.8831965327262878 | 2.36205370176332e-06 | True |
| 7 | 7935 | 889 | 304 | 1.2988593578338623 | 1.2988601922988892 | 6.42459879757209e-07 | True |
| 7 | 7935 | 2156 | 253 | 1.2488049268722534 | 1.2488045692443848 | 2.863760872148828e-07 | True |
| 7 | 8294 | 774 | 6 | 0.7363788485527039 | 0.7363787293434143 | 1.6188581432652224e-07 | True |
| 7 | 8294 | 1034 | 140 | 0.5693034529685974 | 0.5693033337593079 | 2.0939498773311814e-07 | True |
| 7 | 8537 | 271 | 394 | 1.0645594596862793 | 1.0645602941513062 | 7.838594822137804e-07 | True |
| 7 | 8537 | 1294 | 79 | 0.9292474389076233 | 0.9292474389076233 | 0.0 | True |
| 8 | 3037 | 363 | 234 | 3.185793399810791 | 3.1857943534851074 | 2.9935221677051944e-07 | True |
| 8 | 3037 | 310 | 56 | 2.261746406555176 | 2.261747121810913 | 3.1624046587702127e-07 | True |
| 8 | 2632 | 2136 | 266 | 0.9595610499382019 | 0.9595603942871094 | 6.832823118148891e-07 | True |
| 8 | 2632 | 1113 | 40 | 0.9112536907196045 | 0.9112541079521179 | 4.5786647305455805e-07 | True |
| 8 | 8556 | 1019 | 276 | 2.9673123359680176 | 2.9673094749450684 | 9.641799127577876e-07 | True |
| 8 | 8556 | 896 | 227 | 2.106731414794922 | 2.1067302227020264 | 5.658494894679566e-07 | True |
| 8 | 8346 | 1832 | 169 | 0.599003255367279 | 0.5990034341812134 | 2.9851913612144233e-07 | True |
| 8 | 8346 | 482 | 179 | 0.5598881840705872 | 0.5598881244659424 | 1.0645812230228465e-07 | True |
| 9 | 1900 | 1463 | 212 | 1.4346774816513062 | 1.4346774816513062 | 0.0 | True |
| 9 | 1900 | 2369 | 261 | 0.9167253971099854 | 0.9167249202728271 | 5.201526648071209e-07 | True |
| 9 | 6719 | 76 | 15 | 3.4408416748046875 | 3.4408438205718994 | 6.236169561727546e-07 | True |
| 9 | 6719 | 1069 | 13 | 2.9670894145965576 | 2.967090606689453 | 4.0177181369841e-07 | True |
| 9 | 8391 | 1160 | 112 | 0.8603986501693726 | 0.8603994250297546 | 9.005829820020569e-07 | True |
| 9 | 8391 | 1609 | 23 | 0.7641943693161011 | 0.7641946077346802 | 3.1198683041191463e-07 | True |
| 9 | 8310 | 1874 | 31 | 0.5847330093383789 | 0.5847328901290894 | 2.0386960826047033e-07 | True |
| 9 | 8310 | 1073 | 25 | 0.5524399280548096 | 0.5524399280548096 | 0.0 | True |
| 10 | 2713 | 485 | 240 | 2.3089654445648193 | 2.3089659214019775 | 2.0651550213779684e-07 | True |
| 10 | 2713 | 305 | 104 | 1.9339567422866821 | 1.9339573383331299 | 3.0820050661999276e-07 | True |
| 10 | 125 | 2367 | 85 | 1.702926516532898 | 1.7029259204864502 | 3.5001301698410165e-07 | True |
| 10 | 125 | 1524 | 288 | 1.5884017944335938 | 1.5884020328521729 | 1.5009966617834242e-07 | True |
| 10 | 8627 | 1587 | 62 | 3.201032876968384 | 3.2010326385498047 | 7.448176518804225e-08 | True |
| 10 | 8627 | 2240 | 41 | 1.9480046033859253 | 1.9480046033859253 | 0.0 | True |
| 10 | 8533 | 118 | 90 | 4.679548263549805 | 4.679548263549805 | 0.0 | True |
| 10 | 8533 | 1980 | 37 | 4.201279640197754 | 4.201278209686279 | 3.404942296442902e-07 | True |
| 11 | 7557 | 115 | 136 | 6.303983688354492 | 6.30397891998291 | 7.564060787212985e-07 | True |
| 11 | 7557 | 37 | 162 | 5.39935302734375 | 5.399354457855225 | 2.6494127488328453e-07 | True |
| 11 | 1167 | 27 | 190 | 1.620561122894287 | 1.620561122894287 | 0.0 | True |
| 11 | 1167 | 2346 | 91 | 1.3870997428894043 | 1.3871006965637207 | 6.875311752417274e-07 | True |
| 11 | 8646 | 1287 | 254 | 2.9504172801971436 | 2.9504168033599854 | 1.6161685379339404e-07 | True |
| 11 | 8646 | 603 | 104 | 2.080552577972412 | 2.0805511474609375 | 6.875632414939832e-07 | True |
| 11 | 8592 | 2217 | 15 | 4.139142990112305 | 4.139143466949463 | 1.1520190516302682e-07 | True |
| 11 | 8592 | 603 | 42 | 3.799168348312378 | 3.799168348312378 | 0.0 | True |
| 12 | 970 | 1154 | 373 | 29.908740997314453 | 29.908748626708984 | 2.5508912367575265e-07 | True |
| 12 | 970 | 1030 | 142 | 19.147430419921875 | 19.147430419921875 | 0.0 | True |
| 12 | 2306 | 333 | 171 | 3.276790142059326 | 3.2767889499664307 | 3.63798975163735e-07 | True |
| 12 | 2306 | 1617 | 207 | 2.970212936401367 | 2.970212459564209 | 1.6053972170118163e-07 | True |
| 12 | 8214 | 1949 | 70 | 2.2395968437194824 | 2.239595890045166 | 4.2582410270877355e-07 | True |
| 12 | 8214 | 2270 | 111 | 1.307396650314331 | 1.307396650314331 | 0.0 | True |
| 12 | 8513 | 1529 | 19 | 3.6609508991241455 | 3.660949945449829 | 2.6049907324198456e-07 | True |
| 12 | 8513 | 67 | 109 | 3.436375379562378 | 3.4363772869110107 | 5.550466471609399e-07 | True |
| 13 | 1874 | 652 | 437 | 21.69994354248047 | 21.699935913085938 | 3.5158591617137e-07 | True |
| 13 | 1874 | 1370 | 110 | 12.177855491638184 | 12.177855491638184 | 0.0 | True |
| 13 | 7795 | 1524 | 298 | 5.0554118156433105 | 5.055410861968994 | 1.8864423931898676e-07 | True |
| 13 | 7795 | 659 | 110 | 4.13669490814209 | 4.1366963386535645 | 3.4581024377547325e-07 | True |
| 13 | 8577 | 20 | 70 | 14.109756469726562 | 14.10975456237793 | 1.3517941552746467e-07 | True |
| 13 | 8577 | 1600 | 174 | 9.626587867736816 | 9.626587867736816 | 0.0 | True |
| 13 | 8349 | 1784 | 56 | 1.914094090461731 | 1.9140938520431519 | 1.2455948758717996e-07 | True |
| 13 | 8349 | 2302 | 95 | 1.8495653867721558 | 1.8495664596557617 | 5.800733586550394e-07 | True |
| 14 | 5888 | 171 | 173 | 12.170841217041016 | 12.170832633972168 | 7.052157443019351e-07 | True |
| 14 | 5888 | 1142 | 134 | 9.810742378234863 | 9.810745239257812 | 2.9162145319052825e-07 | True |
| 14 | 907 | 1996 | 254 | 8.555346488952637 | 8.555351257324219 | 5.573557527083866e-07 | True |
| 14 | 907 | 1849 | 117 | 7.510560512542725 | 7.510560512542725 | 0.0 | True |
| 14 | 8609 | 1551 | 326 | 16.724428176879883 | 16.72442626953125 | 1.1404567095748298e-07 | True |
| 14 | 8609 | 2126 | 113 | 11.66728687286377 | 11.667288780212402 | 1.6347833507451383e-07 | True |
| 14 | 8315 | 1243 | 43 | 4.279595375061035 | 4.279595851898193 | 1.1142108456838034e-07 | True |
| 14 | 8315 | 2286 | 59 | 4.151406764984131 | 4.151406764984131 | 0.0 | True |
| 15 | 1319 | 1591 | 58 | 28.00290298461914 | 28.002904891967773 | 6.811253225639246e-08 | True |
| 15 | 1319 | 1282 | 145 | 16.618375778198242 | 16.618375778198242 | 0.0 | True |
| 15 | 6966 | 49 | 88 | 15.86551570892334 | 15.86551570892334 | 0.0 | True |
| 15 | 6966 | 591 | 265 | 15.271498680114746 | 15.271493911743164 | 3.1223992365858763e-07 | True |
| 15 | 8284 | 149 | 299 | 9.795967102050781 | 9.7959623336792 | 4.867688439901961e-07 | True |
| 15 | 8284 | 1363 | 460 | 7.56996488571167 | 7.569964408874512 | 6.299066975900463e-08 | True |
| 15 | 8302 | 1577 | 137 | 5.782702922821045 | 5.782702445983887 | 8.245921752634387e-08 | True |
| 15 | 8302 | 881 | 206 | 5.479320526123047 | 5.479321479797363 | 1.740497406310765e-07 | True |

### pretrain_3T: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 4977 | 1838 | 174 | 0.6445115208625793 | 0.6445115208625793 | 0.0 | True |
| 0 | 4977 | 341 | 51 | 0.47932785749435425 | 0.47932758927345276 | 5.595771188667394e-07 | True |
| 0 | 7837 | 21 | 52 | 0.3748740255832672 | 0.3748740255832672 | 0.0 | True |
| 0 | 7837 | 2040 | 50 | 0.31494298577308655 | 0.31494298577308655 | 0.0 | True |
| 0 | 8442 | 1059 | 75 | 1.1582783460617065 | 1.158278465270996 | 1.0291938026477657e-07 | True |
| 0 | 8442 | 326 | 478 | 0.9650334715843201 | 0.9650338888168335 | 4.323503025680064e-07 | True |
| 0 | 8348 | 586 | 1 | 0.304543673992157 | 0.30454352498054504 | 4.892947208035393e-07 | True |
| 0 | 8348 | 86 | 232 | 0.28783196210861206 | 0.2878322899341583 | 1.138947682748815e-06 | True |
| 1 | 1770 | 952 | 127 | 0.5620822906494141 | 0.562082052230835 | 4.2417023818007215e-07 | True |
| 1 | 1770 | 1548 | 78 | 0.3708939552307129 | 0.37089401483535767 | 1.6070535508812438e-07 | True |
| 1 | 5078 | 1705 | 110 | 0.8027070760726929 | 0.8027070760726929 | 0.0 | True |
| 1 | 5078 | 484 | 66 | 0.6448813676834106 | 0.6448812484741211 | 1.8485460353586188e-07 | True |
| 1 | 8627 | 1372 | 176 | 3.121133804321289 | 3.1211328506469727 | 3.0555380710877044e-07 | True |
| 1 | 8627 | 650 | 50 | 2.011845588684082 | 2.011845588684082 | 0.0 | True |
| 1 | 8361 | 149 | 85 | 0.23131904006004333 | 0.23131908476352692 | 1.9325466494214794e-07 | True |
| 1 | 8361 | 1290 | 209 | 0.2157561480998993 | 0.2157561480998993 | 0.0 | True |
| 2 | 8115 | 353 | 70 | 0.9023187756538391 | 0.9023190140724182 | 2.6422876873951716e-07 | True |
| 2 | 8115 | 1971 | 40 | 0.41383567452430725 | 0.41383564472198486 | 7.20148702065192e-08 | True |
| 2 | 7207 | 1574 | 330 | 1.6891582012176514 | 1.689158320426941 | 7.057319407077898e-08 | True |
| 2 | 7207 | 1874 | 141 | 1.37144136428833 | 1.37144136428833 | 0.0 | True |
| 2 | 8331 | 1160 | 9 | 0.38211941719055176 | 0.38211938738822937 | 7.799216958617355e-08 | True |
| 2 | 8331 | 1577 | 49 | 0.3008829653263092 | 0.3008829951286316 | 9.904955023084983e-08 | True |
| 2 | 8305 | 769 | 0 | 0.33416324853897095 | 0.3341634273529053 | 5.351095164054768e-07 | True |
| 2 | 8305 | 1135 | 0 | 0.311624139547348 | 0.311624139547348 | 0.0 | True |
| 3 | 4538 | 131 | 173 | 0.672410249710083 | 0.6724100708961487 | 2.6592981651197236e-07 | True |
| 3 | 4538 | 2471 | 35 | 0.44974029064178467 | 0.4497406482696533 | 7.951875250981063e-07 | True |
| 3 | 1706 | 1518 | 173 | 1.3086332082748413 | 1.3086339235305786 | 5.465670080676023e-07 | True |
| 3 | 1706 | 2228 | 325 | 1.0721495151519775 | 1.0721495151519775 | 0.0 | True |
| 3 | 8448 | 345 | 29 | 0.8347923755645752 | 0.8347923755645752 | 0.0 | True |
| 3 | 8448 | 253 | 212 | 0.5700573921203613 | 0.5700573921203613 | 0.0 | True |
| 3 | 8419 | 692 | 177 | 0.3452322781085968 | 0.3452323079109192 | 8.632542284566059e-08 | True |
| 3 | 8419 | 1552 | 225 | 0.33020368218421936 | 0.3302037715911865 | 2.7076308347526583e-07 | True |
| 4 | 7660 | 610 | 199 | 1.0148518085479736 | 1.0148519277572632 | 1.1746472593012681e-07 | True |
| 4 | 7660 | 1319 | 38 | 0.5834015011787415 | 0.5834014415740967 | 1.0216745184056197e-07 | True |
| 4 | 7099 | 642 | 39 | 0.5509534478187561 | 0.5509534478187561 | 0.0 | True |
| 4 | 7099 | 2400 | 228 | 0.48080873489379883 | 0.48080873489379883 | 0.0 | True |
| 4 | 8497 | 713 | 28 | 0.9919856190681458 | 0.9919861555099487 | 5.407757861272625e-07 | True |
| 4 | 8497 | 2213 | 91 | 0.6974260210990906 | 0.697426438331604 | 5.982462667082704e-07 | True |
| 4 | 8438 | 2298 | 0 | 0.38970720767974854 | 0.3897073566913605 | 3.8236811894156834e-07 | True |
| 4 | 8438 | 1970 | 0 | 0.37236958742141724 | 0.3723694980144501 | 2.401027639829847e-07 | True |
| 5 | 3422 | 1785 | 168 | 3.0497326850891113 | 3.0497326850891113 | 0.0 | True |
| 5 | 3422 | 2370 | 19 | 1.9250411987304688 | 1.9250408411026 | 1.8577673500608358e-07 | True |
| 5 | 6870 | 1244 | 150 | 2.452202320098877 | 2.4522037506103516 | 5.833578505674419e-07 | True |
| 5 | 6870 | 1107 | 84 | 1.8321021795272827 | 1.8321024179458618 | 1.301338875995874e-07 | True |
| 5 | 8610 | 648 | 92 | 2.966858148574829 | 2.966855764389038 | 8.036062634679387e-07 | True |
| 5 | 8610 | 1290 | 90 | 1.5444514751434326 | 1.544451117515564 | 2.315565586928725e-07 | True |
| 5 | 8511 | 2316 | 115 | 1.1392053365707397 | 1.1392053365707397 | 0.0 | True |
| 5 | 8511 | 2399 | 4 | 0.9761868715286255 | 0.976186215877533 | 6.71645062694403e-07 | True |
| 6 | 2036 | 229 | 340 | 1.4160692691802979 | 1.416069507598877 | 1.6836646645087697e-07 | True |
| 6 | 2036 | 2212 | 124 | 0.7182954549789429 | 0.7182950377464294 | 5.808647549356493e-07 | True |
| 6 | 3628 | 1618 | 166 | 1.0100523233413696 | 1.0100517272949219 | 5.901144267280292e-07 | True |
| 6 | 3628 | 2029 | 93 | 0.8707152605056763 | 0.8707148432731628 | 4.79183646310991e-07 | True |
| 6 | 8561 | 2202 | 285 | 1.0775327682495117 | 1.0775322914123535 | 4.4252682818895893e-07 | True |
| 6 | 8561 | 1429 | 177 | 0.6939648985862732 | 0.6939659118652344 | 1.4601299910786058e-06 | True |
| 6 | 8546 | 1131 | 199 | 1.25489342212677 | 1.2548930644989014 | 2.8498664695065713e-07 | True |
| 6 | 8546 | 564 | 201 | 1.0959676504135132 | 1.0959675312042236 | 1.0877081043934379e-07 | True |
| 7 | 6547 | 2162 | 284 | 1.4724876880645752 | 1.4724867343902588 | 6.476619968617538e-07 | True |
| 7 | 6547 | 2181 | 225 | 0.7314497232437134 | 0.731449544429779 | 2.444651062730561e-07 | True |
| 7 | 6191 | 171 | 262 | 1.2044906616210938 | 1.2044910192489624 | 2.969121140142518e-07 | True |
| 7 | 6191 | 2084 | 163 | 1.1011055707931519 | 1.1011055707931519 | 0.0 | True |
| 7 | 8248 | 2355 | 0 | 0.7914441227912903 | 0.7914441823959351 | 7.531124820938094e-08 | True |
| 7 | 8248 | 482 | 0 | 0.564167320728302 | 0.564167320728302 | 0.0 | True |
| 7 | 8497 | 1700 | 79 | 0.7845773100852966 | 0.7845765948295593 | 9.116446883060221e-07 | True |
| 7 | 8497 | 229 | 281 | 0.6947929263114929 | 0.6947928071022034 | 1.7157527809564482e-07 | True |
| 8 | 3595 | 683 | 139 | 1.6510376930236816 | 1.6510400772094727 | 1.44405291356448e-06 | True |
| 8 | 3595 | 86 | 114 | 1.3488088846206665 | 1.348808765411377 | 8.838115681919398e-08 | True |
| 8 | 7701 | 1951 | 125 | 1.1057438850402832 | 1.1057432889938354 | 5.390456649300771e-07 | True |
| 8 | 7701 | 2017 | 164 | 0.9482206106185913 | 0.9482207298278809 | 1.2571893947022794e-07 | True |
| 8 | 8546 | 2442 | 221 | 3.0498626232147217 | 3.04986310005188 | 1.5634709398829007e-07 | True |
| 8 | 8546 | 564 | 89 | 0.8012295961380005 | 0.8012298345565796 | 2.9756586657652405e-07 | True |
| 8 | 8643 | 907 | 106 | 0.6107932925224304 | 0.6107929944992065 | 4.87928121551873e-07 | True |
| 8 | 8643 | 293 | 373 | 0.555686354637146 | 0.5556861162185669 | 4.290524269886848e-07 | True |
| 9 | 2018 | 802 | 133 | 4.205742835998535 | 4.20574426651001 | 3.4013289218853066e-07 | True |
| 9 | 2018 | 612 | 232 | 2.2616140842437744 | 2.2616124153137207 | 7.379375930394353e-07 | True |
| 9 | 3246 | 2273 | 79 | 0.9559210538864136 | 0.955920398235321 | 6.858841426953277e-07 | True |
| 9 | 3246 | 1594 | 115 | 0.8966026902198792 | 0.8966024518013 | 2.6591329883595786e-07 | True |
| 9 | 8497 | 2016 | 119 | 1.2565757036209106 | 1.256576418876648 | 5.692102236607218e-07 | True |
| 9 | 8497 | 180 | 235 | 1.0283591747283936 | 1.028358817100525 | 3.477655253543093e-07 | True |
| 9 | 8638 | 1067 | 0 | 0.6583858132362366 | 0.6583858728408813 | 9.053148408895588e-08 | True |
| 9 | 8638 | 2405 | 0 | 0.6328920125961304 | 0.6328919529914856 | 9.417822249153008e-08 | True |
| 10 | 5217 | 2334 | 36 | 1.5763683319091797 | 1.5763682126998901 | 7.562273812390271e-08 | True |
| 10 | 5217 | 2267 | 11 | 1.2124581336975098 | 1.2124576568603516 | 3.932813389184527e-07 | True |
| 10 | 3590 | 1951 | 52 | 3.125385284423828 | 3.125385046005249 | 7.628454011407285e-08 | True |
| 10 | 3590 | 149 | 224 | 2.8639495372772217 | 2.8639495372772217 | 0.0 | True |
| 10 | 8630 | 176 | 118 | 2.605343818664551 | 2.6053426265716553 | 4.5755684411697204e-07 | True |
| 10 | 8630 | 400 | 101 | 1.8951573371887207 | 1.895156979560852 | 1.8870616261489375e-07 | True |
| 10 | 8695 | 1970 | 185 | 0.7216814756393433 | 0.7216814160346985 | 8.259134644212172e-08 | True |
| 10 | 8695 | 228 | 181 | 0.6915316581726074 | 0.6915316581726074 | 0.0 | True |
| 11 | 747 | 1136 | 252 | 6.110573768615723 | 6.110572814941406 | 1.5606952023136993e-07 | True |
| 11 | 747 | 75 | 265 | 4.51897668838501 | 4.518977642059326 | 2.1103767117397407e-07 | True |
| 11 | 5622 | 1533 | 310 | 1.8132729530334473 | 1.813273549079895 | 3.2871303063158394e-07 | True |
| 11 | 5622 | 2442 | 409 | 1.670119047164917 | 1.6701184511184692 | 3.568885995077507e-07 | True |
| 11 | 8416 | 2435 | 2 | 0.9041481614112854 | 0.9041476249694824 | 5.93311833030975e-07 | True |
| 11 | 8416 | 49 | 2 | 0.9041481614112854 | 0.9041479229927063 | 2.636941480137667e-07 | True |
| 11 | 8215 | 11 | 4 | 0.7223036885261536 | 0.7223037481307983 | 8.252019991343631e-08 | True |
| 11 | 8215 | 757 | 43 | 0.6902854442596436 | 0.6902853846549988 | 8.634782215249923e-08 | True |
| 12 | 2562 | 865 | 115 | 2.096858024597168 | 2.0968594551086426 | 6.82216658366364e-07 | True |
| 12 | 2562 | 1962 | 150 | 1.6202070713043213 | 1.6202061176300049 | 5.886126121141478e-07 | True |
| 12 | 1888 | 1968 | 56 | 1.5966345071792603 | 1.596632957458496 | 9.706171056630953e-07 | True |
| 12 | 1888 | 340 | 157 | 1.3944305181503296 | 1.3944305181503296 | 0.0 | True |
| 12 | 8357 | 485 | 102 | 1.0673680305480957 | 1.0673683881759644 | 3.350558180655843e-07 | True |
| 12 | 8357 | 364 | 281 | 0.8884786367416382 | 0.8884786367416382 | 0.0 | True |
| 12 | 8316 | 272 | 27 | 1.086016297340393 | 1.0860164165496826 | 1.0976749597839336e-07 | True |
| 12 | 8316 | 1861 | 251 | 1.0349935293197632 | 1.034994125366211 | 5.75893888095755e-07 | True |
| 13 | 4596 | 2181 | 344 | 16.507427215576172 | 16.507427215576172 | 0.0 | True |
| 13 | 4596 | 995 | 246 | 8.637409210205078 | 8.637407302856445 | 2.2082415992968959e-07 | True |
| 13 | 7795 | 755 | 161 | 2.943234443664551 | 2.943232774734497 | 5.670394546052514e-07 | True |
| 13 | 7795 | 1691 | 133 | 2.655198574066162 | 2.655198574066162 | 0.0 | True |
| 13 | 8314 | 2180 | 123 | 2.080759048461914 | 2.080759048461914 | 0.0 | True |
| 13 | 8314 | 1551 | 326 | 1.5651516914367676 | 1.5651510953903198 | 3.808234377632442e-07 | True |
| 13 | 8252 | 1994 | 49 | 1.9157875776290894 | 1.9157875776290894 | 0.0 | True |
| 13 | 8252 | 1368 | 82 | 1.6846911907196045 | 1.6846909523010254 | 1.4152064213010078e-07 | True |
| 14 | 179 | 1104 | 64 | 10.4631986618042 | 10.463201522827148 | 2.7343674164028684e-07 | True |
| 14 | 179 | 456 | 337 | 8.101110458374023 | 8.101103782653809 | 8.240500174816325e-07 | True |
| 14 | 4850 | 1594 | 44 | 3.6470985412597656 | 3.647101402282715 | 7.84465491363035e-07 | True |
| 14 | 4850 | 637 | 22 | 3.358175754547119 | 3.3581762313842773 | 1.419929131337055e-07 | True |
| 14 | 8209 | 298 | 104 | 3.1549758911132812 | 3.154975414276123 | 1.511381305785084e-07 | True |
| 14 | 8209 | 1307 | 215 | 2.827193021774292 | 2.827193260192871 | 8.433049221094058e-08 | True |
| 14 | 8469 | 640 | 370 | 6.1203107833862305 | 6.120307922363281 | 4.674636714503263e-07 | True |
| 14 | 8469 | 2316 | 130 | 4.985744953155518 | 4.985743999481201 | 1.9128020493761158e-07 | True |
| 15 | 467 | 2489 | 294 | 37.47602844238281 | 37.476009368896484 | 5.089516451149396e-07 | True |
| 15 | 467 | 2302 | 155 | 24.999298095703125 | 24.99929428100586 | 1.5259217482912727e-07 | True |
| 15 | 6377 | 1720 | 107 | 11.507088661193848 | 11.507089614868164 | 8.28771155316107e-08 | True |
| 15 | 6377 | 610 | 160 | 11.12716293334961 | 11.12716293334961 | 0.0 | True |
| 15 | 8284 | 755 | 174 | 6.392274379730225 | 6.392273902893066 | 7.459585272421443e-08 | True |
| 15 | 8284 | 1363 | 460 | 5.470394134521484 | 5.470394134521484 | 0.0 | True |
| 15 | 8655 | 1493 | 48 | 7.44575834274292 | 7.445757865905762 | 6.40414496755564e-08 | True |
| 15 | 8655 | 333 | 101 | 6.257771015167236 | 6.257769584655762 | 2.2859760626302577e-07 | True |

### pretrain_4T: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2544 | 2379 | 154 | 0.4290291965007782 | 0.4290291965007782 | 0.0 | True |
| 0 | 2544 | 410 | 135 | 0.3238668143749237 | 0.3238668143749237 | 0.0 | True |
| 0 | 2988 | 1068 | 205 | 0.42736777663230896 | 0.4273677468299866 | 6.973460334922747e-08 | True |
| 0 | 2988 | 228 | 85 | 0.3440276086330414 | 0.3440276086330414 | 0.0 | True |
| 0 | 8206 | 2429 | 21 | 0.6165043115615845 | 0.6165047883987427 | 7.734530793390766e-07 | True |
| 0 | 8206 | 2041 | 30 | 0.36018338799476624 | 0.360183447599411 | 1.6548415824290244e-07 | True |
| 0 | 8447 | 1861 | 165 | 0.1660034954547882 | 0.1660035252571106 | 1.7952828225723783e-07 | True |
| 0 | 8447 | 2369 | 191 | 0.15412098169326782 | 0.15412098169326782 | 0.0 | True |
| 1 | 6897 | 685 | 120 | 0.3653813302516937 | 0.36538130044937134 | 8.156498408707943e-08 | True |
| 1 | 6897 | 1913 | 332 | 0.19915296137332916 | 0.19915296137332916 | 0.0 | True |
| 1 | 6205 | 1714 | 8 | 0.20021849870681763 | 0.20021852850914001 | 1.4884899537347553e-07 | True |
| 1 | 6205 | 691 | 254 | 0.18718819320201874 | 0.18718819320201874 | 0.0 | True |
| 1 | 8441 | 2213 | 0 | 0.18175189197063446 | 0.18175193667411804 | 2.4595883485364586e-07 | True |
| 1 | 8441 | 552 | 385 | 0.13919074833393097 | 0.13919071853160858 | 2.1411137409935388e-07 | True |
| 1 | 8361 | 952 | 17 | 0.12735889852046967 | 0.12735889852046967 | 0.0 | True |
| 1 | 8361 | 5 | 160 | 0.11522363871335983 | 0.11522361636161804 | 1.939857310562426e-07 | True |
| 2 | 408 | 2400 | 431 | 1.775759220123291 | 1.775758981704712 | 1.3426289803243092e-07 | True |
| 2 | 408 | 2188 | 170 | 1.1361520290374756 | 1.1361522674560547 | 2.0984742623180965e-07 | True |
| 2 | 7207 | 1874 | 141 | 0.7871043086051941 | 0.7871043086051941 | 0.0 | True |
| 2 | 7207 | 353 | 147 | 0.6948762536048889 | 0.6948763132095337 | 8.577735167401794e-08 | True |
| 2 | 8260 | 1393 | 50 | 0.3099304735660553 | 0.3099304735660553 | 0.0 | True |
| 2 | 8260 | 918 | 86 | 0.24461275339126587 | 0.24461261928081512 | 5.482561676991346e-07 | True |
| 2 | 8243 | 1393 | 239 | 0.3227307200431824 | 0.32273057103157043 | 4.61721189475047e-07 | True |
| 2 | 8243 | 2245 | 119 | 0.28407803177833557 | 0.2840779721736908 | 2.098178602627738e-07 | True |
| 3 | 6554 | 1018 | 126 | 1.2652604579925537 | 1.2652606964111328 | 1.8843438724058002e-07 | True |
| 3 | 6554 | 766 | 89 | 0.6943474411964417 | 0.6943474411964417 | 0.0 | True |
| 3 | 6045 | 5 | 419 | 0.6369470357894897 | 0.6369472146034241 | 2.807359549205433e-07 | True |
| 3 | 6045 | 727 | 189 | 0.5537967681884766 | 0.5537970662117004 | 5.381454732063826e-07 | True |
| 3 | 8400 | 280 | 164 | 0.24238243699073792 | 0.24238240718841553 | 1.229557832560044e-07 | True |
| 3 | 8400 | 232 | 72 | 0.19783739745616913 | 0.19783738255500793 | 7.5320244733552e-08 | True |
| 3 | 8359 | 1362 | 51 | 0.3377209007740021 | 0.3377208113670349 | 2.647362569452454e-07 | True |
| 3 | 8359 | 1393 | 188 | 0.21062476933002472 | 0.21062475442886353 | 7.074743032952243e-08 | True |
| 4 | 4143 | 854 | 251 | 0.7116641998291016 | 0.7116638422012329 | 5.025233371837788e-07 | True |
| 4 | 4143 | 1895 | 312 | 0.5296770930290222 | 0.5296773910522461 | 5.626507693067704e-07 | True |
| 4 | 2651 | 1686 | 7 | 0.34562021493911743 | 0.34561973810195923 | 1.379656448298668e-06 | True |
| 4 | 2651 | 1011 | 430 | 0.3052482008934021 | 0.3052480220794678 | 5.857984872730397e-07 | True |
| 4 | 8485 | 229 | 3 | 0.7928943634033203 | 0.7928943634033203 | 0.0 | True |
| 4 | 8485 | 983 | 150 | 0.26544392108917236 | 0.2654434144496918 | 1.9086497762388823e-06 | True |
| 4 | 8432 | 353 | 70 | 0.16881367564201355 | 0.16881361603736877 | 3.530794797797561e-07 | True |
| 4 | 8432 | 699 | 300 | 0.1605314463376999 | 0.16053149104118347 | 2.7847181721332695e-07 | True |
| 5 | 2123 | 2096 | 17 | 0.9022341966629028 | 0.9022341966629028 | 0.0 | True |
| 5 | 2123 | 1476 | 15 | 0.6416630148887634 | 0.6416633129119873 | 4.644544207189773e-07 | True |
| 5 | 6870 | 1764 | 62 | 1.9211071729660034 | 1.921107530593872 | 1.861571669113082e-07 | True |
| 5 | 6870 | 1136 | 239 | 1.4628914594650269 | 1.462891936302185 | 3.259552546553948e-07 | True |
| 5 | 8526 | 1372 | 35 | 0.36516502499580383 | 0.36516493558883667 | 2.4483989715092053e-07 | True |
| 5 | 8526 | 1597 | 333 | 0.30580875277519226 | 0.3058087229728699 | 9.745411835744203e-08 | True |
| 5 | 8477 | 406 | 66 | 1.7207649946212769 | 1.7207649946212769 | 0.0 | True |
| 5 | 8477 | 721 | 15 | 1.3991860151290894 | 1.3991869688034058 | 6.815922301212135e-07 | True |
| 6 | 4186 | 782 | 389 | 1.4999788999557495 | 1.49997878074646 | 7.947397763681743e-08 | True |
| 6 | 4186 | 1898 | 87 | 1.0298371315002441 | 1.0298374891281128 | 3.472664343840072e-07 | True |
| 6 | 1079 | 1029 | 170 | 0.6384523510932922 | 0.6384527087211609 | 5.601480956878586e-07 | True |
| 6 | 1079 | 2083 | 72 | 0.5618221163749695 | 0.5618219375610352 | 3.182750004217144e-07 | True |
| 6 | 8257 | 1548 | 85 | 0.30845290422439575 | 0.3084529638290405 | 1.932374244465825e-07 | True |
| 6 | 8257 | 783 | 81 | 0.23628440499305725 | 0.23628447949886322 | 3.1532257057518243e-07 | True |
| 6 | 8276 | 394 | 22 | 0.2487569898366928 | 0.24875707924365997 | 3.5941489411727074e-07 | True |
| 6 | 8276 | 1429 | 35 | 0.23085430264472961 | 0.23085428774356842 | 6.454790325818452e-08 | True |
| 7 | 7843 | 1487 | 119 | 2.1793341636657715 | 2.1793315410614014 | 1.2033970805586826e-06 | True |
| 7 | 7843 | 715 | 263 | 1.269420862197876 | 1.2694203853607178 | 3.756336234915258e-07 | True |
| 7 | 6164 | 2185 | 52 | 1.7726538181304932 | 1.7726564407348633 | 1.479479153398989e-06 | True |
| 7 | 6164 | 1698 | 293 | 1.4603058099746704 | 1.4603049755096436 | 5.714316968100968e-07 | True |
| 7 | 8285 | 1124 | 0 | 0.5991226434707642 | 0.5991226434707642 | 0.0 | True |
| 7 | 8285 | 210 | 0 | 0.5395445227622986 | 0.5395442843437195 | 4.418886098239571e-07 | True |
| 7 | 8275 | 1686 | 7 | 0.293050616979599 | 0.29305076599121094 | 5.084842116160778e-07 | True |
| 7 | 8275 | 1104 | 2 | 0.2710837125778198 | 0.27108344435691833 | 9.894393836452265e-07 | True |
| 8 | 2358 | 2355 | 70 | 0.7031024098396301 | 0.7031024694442749 | 8.477377397836793e-08 | True |
| 8 | 2358 | 417 | 43 | 0.5606490969657898 | 0.5606490969657898 | 0.0 | True |
| 8 | 2632 | 1176 | 236 | 0.513636589050293 | 0.5136370658874512 | 9.283551218280426e-07 | True |
| 8 | 2632 | 648 | 342 | 0.48848751187324524 | 0.4884869456291199 | 1.1591783036474887e-06 | True |
| 8 | 8255 | 1826 | 0 | 0.38036108016967773 | 0.38036125898361206 | 4.7011627542545524e-07 | True |
| 8 | 8255 | 1568 | 0 | 0.32589372992515564 | 0.3258937895298004 | 1.8289595442379133e-07 | True |
| 8 | 8657 | 1918 | 67 | 0.5514135956764221 | 0.5514135956764221 | 0.0 | True |
| 8 | 8657 | 1846 | 84 | 0.5031862258911133 | 0.5031862258911133 | 0.0 | True |
| 9 | 5684 | 1216 | 191 | 2.68109130859375 | 2.68109393119812 | 9.781853985020603e-07 | True |
| 9 | 5684 | 2302 | 50 | 1.9313558340072632 | 1.9313571453094482 | 6.789542154631575e-07 | True |
| 9 | 6063 | 1386 | 281 | 0.7342022657394409 | 0.7342022061347961 | 8.118286684304997e-08 | True |
| 9 | 6063 | 2448 | 73 | 0.6581603288650513 | 0.658160924911499 | 9.056249998867969e-07 | True |
| 9 | 8543 | 1597 | 299 | 0.8146631717681885 | 0.8146630525588989 | 1.463295429104068e-07 | True |
| 9 | 8543 | 2486 | 161 | 0.6132360696792603 | 0.6132360696792603 | 0.0 | True |
| 9 | 8696 | 1040 | 153 | 0.698614239692688 | 0.6986146569252014 | 5.972287561892096e-07 | True |
| 9 | 8696 | 287 | 112 | 0.6499212384223938 | 0.6499217748641968 | 8.253950959975766e-07 | True |
| 10 | 2107 | 2193 | 382 | 0.9836422204971313 | 0.9836417436599731 | 4.847668677358442e-07 | True |
| 10 | 2107 | 75 | 154 | 0.7534709572792053 | 0.7534708976745605 | 7.910675812990042e-08 | True |
| 10 | 1475 | 345 | 289 | 1.266487956047058 | 1.266487717628479 | 1.8825175396512317e-07 | True |
| 10 | 1475 | 863 | 27 | 1.1561528444290161 | 1.1561522483825684 | 5.15542949728479e-07 | True |
| 10 | 8630 | 176 | 26 | 1.9653316736221313 | 1.965331792831421 | 6.065606693809448e-08 | True |
| 10 | 8630 | 400 | 98 | 1.2302216291427612 | 1.2302217483520508 | 9.690066141484462e-08 | True |
| 10 | 8269 | 1038 | 130 | 0.40542399883270264 | 0.40542417764663696 | 4.410541429244771e-07 | True |
| 10 | 8269 | 1172 | 143 | 0.37171685695648193 | 0.37171661853790283 | 6.413983510289794e-07 | True |
| 11 | 6113 | 2239 | 323 | 2.324342966079712 | 2.324342727661133 | 1.0257461251670812e-07 | True |
| 11 | 6113 | 1674 | 83 | 1.1666138172149658 | 1.1666146516799927 | 7.152881395212433e-07 | True |
| 11 | 6919 | 163 | 75 | 0.8753055334091187 | 0.8753039240837097 | 1.8385870390509077e-06 | True |
| 11 | 6919 | 456 | 348 | 0.7775498628616333 | 0.7775499820709229 | 1.533140127014527e-07 | True |
| 11 | 8201 | 221 | 66 | 0.5669863820075989 | 0.5669862031936646 | 3.1537606545861517e-07 | True |
| 11 | 8201 | 1898 | 67 | 0.45404526591300964 | 0.45404547452926636 | 4.594613629422488e-07 | True |
| 11 | 8464 | 1209 | 83 | 2.3577442169189453 | 2.3577442169189453 | 0.0 | True |
| 11 | 8464 | 2123 | 18 | 2.02056622505188 | 2.020566463470459 | 1.1799592418478681e-07 | True |
| 12 | 104 | 1711 | 23 | 1.548767328262329 | 1.5487672090530396 | 7.697043150085722e-08 | True |
| 12 | 104 | 357 | 84 | 1.2057656049728394 | 1.205765724182129 | 9.886605577330804e-08 | True |
| 12 | 8083 | 1940 | 63 | 1.5607433319091797 | 1.5607433319091797 | 0.0 | True |
| 12 | 8083 | 1018 | 26 | 1.4194258451461792 | 1.4194263219833374 | 3.3593664637973253e-07 | True |
| 12 | 8357 | 485 | 102 | 0.8731898665428162 | 0.873189389705658 | 5.4608645435963e-07 | True |
| 12 | 8357 | 364 | 281 | 0.629417896270752 | 0.6294183731079102 | 7.575843664890132e-07 | True |
| 12 | 8295 | 2235 | 230 | 0.6035628914833069 | 0.6035624146461487 | 7.90037235442453e-07 | True |
| 12 | 8295 | 2139 | 294 | 0.5859290957450867 | 0.5859293937683105 | 5.086335975481419e-07 | True |
| 13 | 1258 | 477 | 71 | 5.4775285720825195 | 5.4775261878967285 | 4.3526670096568273e-07 | True |
| 13 | 1258 | 1568 | 185 | 2.319272994995117 | 2.319274663925171 | 7.195918968195683e-07 | True |
| 13 | 6762 | 1089 | 261 | 1.5928860902786255 | 1.592886209487915 | 7.483855266130758e-08 | True |
| 13 | 6762 | 910 | 170 | 1.5098638534545898 | 1.5098637342453003 | 7.895366809267518e-08 | True |
| 13 | 8577 | 907 | 116 | 8.35075569152832 | 8.350749015808105 | 7.994151022303453e-07 | True |
| 13 | 8577 | 1764 | 142 | 4.792029857635498 | 4.792026996612549 | 5.970377969703317e-07 | True |
| 13 | 8479 | 407 | 145 | 3.568892240524292 | 3.5688929557800293 | 2.004139349412276e-07 | True |
| 13 | 8479 | 2346 | 272 | 2.9552981853485107 | 2.955299139022827 | 3.2269986194093155e-07 | True |
| 14 | 1516 | 2473 | 79 | 6.915602684020996 | 6.915603160858154 | 6.895091866756486e-08 | True |
| 14 | 1516 | 1290 | 166 | 4.438659191131592 | 4.43865966796875 | 1.0742819794676782e-07 | True |
| 14 | 7355 | 1673 | 109 | 6.8150248527526855 | 6.815025806427002 | 1.399370269385074e-07 | True |
| 14 | 7355 | 1973 | 387 | 6.062071800231934 | 6.062072277069092 | 7.865910763130211e-08 | True |
| 14 | 8593 | 1849 | 98 | 6.08459997177124 | 6.08459997177124 | 0.0 | True |
| 14 | 8593 | 279 | 231 | 3.995051622390747 | 3.995054006576538 | 5.967847267988151e-07 | True |
| 14 | 8377 | 2389 | 144 | 1.940092921257019 | 1.94009268283844 | 1.2289028865024003e-07 | True |
| 14 | 8377 | 2071 | 90 | 1.8825623989105225 | 1.882562518119812 | 6.332288885604541e-08 | True |
| 15 | 3703 | 2449 | 203 | 27.93269157409668 | 27.93269157409668 | 0.0 | True |
| 15 | 3703 | 2429 | 23 | 21.427776336669922 | 21.427772521972656 | 1.7802581125026992e-07 | True |
| 15 | 4091 | 2410 | 81 | 6.534404277801514 | 6.534406661987305 | 3.64866587626834e-07 | True |
| 15 | 4091 | 328 | 122 | 6.192478656768799 | 6.192478656768799 | 0.0 | True |
| 15 | 8290 | 2231 | 358 | 4.121140003204346 | 4.121140003204346 | 0.0 | True |
| 15 | 8290 | 608 | 121 | 3.840723991394043 | 3.8407249450683594 | 2.4830587111783084e-07 | True |
| 15 | 8354 | 839 | 62 | 2.8630199432373047 | 2.8630199432373047 | 0.0 | True |
| 15 | 8354 | 326 | 324 | 2.655139446258545 | 2.6551413536071777 | 7.18361001905273e-07 | True |

### pretrain_4T_bf16r: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2544 | 2379 | 154 | 0.4288024306297302 | 0.4288024306297302 | 0.0 | True |
| 0 | 2544 | 410 | 135 | 0.3241761028766632 | 0.3241761028766632 | 0.0 | True |
| 0 | 5640 | 2084 | 7 | 0.18276242911815643 | 0.18276242911815643 | 0.0 | True |
| 0 | 5640 | 1597 | 52 | 0.16021095216274261 | 0.16021095216274261 | 0.0 | True |
| 0 | 8206 | 2429 | 21 | 0.6165443062782288 | 0.6165444850921631 | 2.90026089780283e-07 | True |
| 0 | 8206 | 2041 | 30 | 0.36020156741142273 | 0.36020150780677795 | 1.654758062374285e-07 | True |
| 0 | 8447 | 1861 | 165 | 0.16602782905101776 | 0.16602787375450134 | 2.692529549838677e-07 | True |
| 0 | 8447 | 2369 | 191 | 0.1541711986064911 | 0.1541711390018463 | 3.866133578394654e-07 | True |
| 1 | 6897 | 685 | 120 | 0.36425602436065674 | 0.36425602436065674 | 0.0 | True |
| 1 | 6897 | 1913 | 332 | 0.19935242831707 | 0.1993524134159088 | 7.474782885587609e-08 | True |
| 1 | 6205 | 1714 | 8 | 0.20013394951820374 | 0.2001340389251709 | 4.467356356995977e-07 | True |
| 1 | 6205 | 691 | 254 | 0.18711379170417786 | 0.18711374700069427 | 2.389106819673562e-07 | True |
| 1 | 8441 | 2213 | 0 | 0.18155324459075928 | 0.18155324459075928 | 0.0 | True |
| 1 | 8441 | 552 | 385 | 0.13919496536254883 | 0.1391950100660324 | 3.21157331122629e-07 | True |
| 1 | 8361 | 952 | 17 | 0.12739871442317963 | 0.1273987889289856 | 5.848238446249367e-07 | True |
| 1 | 8361 | 915 | 119 | 0.11543862521648407 | 0.11543862521648407 | 0.0 | True |
| 2 | 7059 | 2213 | 92 | 0.840923011302948 | 0.8409228920936584 | 1.4176005169138524e-07 | True |
| 2 | 7059 | 1362 | 50 | 0.8043981194496155 | 0.8043981194496155 | 0.0 | True |
| 2 | 7469 | 2031 | 60 | 0.27658289670944214 | 0.2765830159187317 | 4.310074519033397e-07 | True |
| 2 | 7469 | 1775 | 36 | 0.2219294011592865 | 0.221929132938385 | 1.2085866049660824e-06 | True |
| 2 | 8260 | 1393 | 50 | 0.31046605110168457 | 0.31046608090400696 | 9.599221004017083e-08 | True |
| 2 | 8260 | 918 | 86 | 0.2439766824245453 | 0.24397680163383484 | 4.886093554766207e-07 | True |
| 2 | 8243 | 1393 | 239 | 0.3225996494293213 | 0.3225993812084198 | 8.314358120467289e-07 | True |
| 2 | 8243 | 2245 | 119 | 0.28429892659187317 | 0.28429892659187317 | 0.0 | True |
| 3 | 272 | 128 | 68 | 0.8104250431060791 | 0.8104257583618164 | 8.825686513380182e-07 | True |
| 3 | 272 | 1710 | 165 | 0.4525774121284485 | 0.4525774419307709 | 6.585022051263343e-08 | True |
| 3 | 6045 | 5 | 419 | 0.6376869678497314 | 0.6376867890357971 | 2.8041020648285967e-07 | True |
| 3 | 6045 | 727 | 189 | 0.5540198087692261 | 0.5540199875831604 | 3.22757294045881e-07 | True |
| 3 | 8400 | 280 | 164 | 0.24159158766269684 | 0.24159155786037445 | 1.2335827863884256e-07 | True |
| 3 | 8400 | 232 | 72 | 0.19806155562400818 | 0.19806155562400818 | 0.0 | True |
| 3 | 8359 | 1362 | 51 | 0.3372085392475128 | 0.33720850944519043 | 8.837950086969848e-08 | True |
| 3 | 8359 | 1393 | 188 | 0.21035899221897125 | 0.21035905182361603 | 2.833472633931699e-07 | True |
| 4 | 4143 | 854 | 251 | 0.7121354341506958 | 0.7121352553367615 | 2.510954036986353e-07 | True |
| 4 | 4143 | 1895 | 312 | 0.5293275713920593 | 0.5293273329734802 | 4.5041783573554304e-07 | True |
| 4 | 8098 | 1993 | 53 | 1.5167793035507202 | 1.5167791843414307 | 7.859369472652813e-08 | True |
| 4 | 8098 | 552 | 286 | 1.3105149269104004 | 1.310514211654663 | 5.457822132487541e-07 | True |
| 4 | 8485 | 229 | 3 | 0.7931253910064697 | 0.7931253910064697 | 0.0 | True |
| 4 | 8485 | 983 | 150 | 0.26555976271629333 | 0.26555952429771423 | 8.97796325252306e-07 | True |
| 4 | 8432 | 353 | 70 | 0.1687965989112854 | 0.16879667341709137 | 4.413939999371455e-07 | True |
| 4 | 8432 | 699 | 300 | 0.1613454520702362 | 0.16134537756443024 | 4.6177815992485943e-07 | True |
| 5 | 2123 | 2096 | 17 | 0.9011985063552856 | 0.9011985063552856 | 0.0 | True |
| 5 | 2123 | 1476 | 15 | 0.6449065208435059 | 0.6449061036109924 | 6.469658779105147e-07 | True |
| 5 | 6870 | 1764 | 62 | 1.9155759811401367 | 1.9155768156051636 | 4.3562094903633176e-07 | True |
| 5 | 6870 | 1136 | 239 | 1.4645423889160156 | 1.464542031288147 | 2.4419086218258443e-07 | True |
| 5 | 8526 | 1372 | 35 | 0.3664575517177582 | 0.366457462310791 | 2.439763261638179e-07 | True |
| 5 | 8526 | 326 | 208 | 0.3067387044429779 | 0.3067387044429779 | 0.0 | True |
| 5 | 8511 | 1625 | 176 | 0.8798789381980896 | 0.8798789381980896 | 0.0 | True |
| 5 | 8511 | 1550 | 4 | 0.7338798642158508 | 0.7338798642158508 | 0.0 | True |
| 6 | 4186 | 782 | 389 | 1.4955720901489258 | 1.4955710172653198 | 7.173733804100315e-07 | True |
| 6 | 4186 | 1898 | 87 | 1.026108980178833 | 1.026109218597412 | 2.323521026587354e-07 | True |
| 6 | 1079 | 1029 | 170 | 0.6390939950942993 | 0.6390950083732605 | 1.585492852318429e-06 | True |
| 6 | 1079 | 2083 | 72 | 0.5590469837188721 | 0.5590465664863586 | 7.463281720120112e-07 | True |
| 6 | 8257 | 1548 | 85 | 0.3081224858760834 | 0.3081224858760834 | 0.0 | True |
| 6 | 8257 | 783 | 81 | 0.23680874705314636 | 0.23680871725082397 | 1.2584975326526624e-07 | True |
| 6 | 8516 | 1606 | 9 | 1.4092397689819336 | 1.4092397689819336 | 0.0 | True |
| 6 | 8516 | 1058 | 17 | 1.2329469919204712 | 1.2329479455947876 | 7.734917418637612e-07 | True |
| 7 | 7843 | 1487 | 119 | 2.177549123764038 | 2.1775500774383545 | 4.3795765891047303e-07 | True |
| 7 | 7843 | 715 | 263 | 1.2755229473114014 | 1.2755228281021118 | 9.345914928622444e-08 | True |
| 7 | 6164 | 2185 | 52 | 1.7683615684509277 | 1.7683614492416382 | 6.74122824639351e-08 | True |
| 7 | 6164 | 1698 | 293 | 1.4575053453445435 | 1.4575048685073853 | 3.271598006320891e-07 | True |
| 7 | 8233 | 1393 | 188 | 0.4103945195674896 | 0.4103943705558777 | 3.630935717550963e-07 | True |
| 7 | 8233 | 98 | 29 | 0.3086508810520172 | 0.3086507320404053 | 4.827836921462197e-07 | True |
| 7 | 8275 | 1686 | 7 | 0.29303741455078125 | 0.2930375337600708 | 4.068056965815304e-07 | True |
| 7 | 8275 | 1104 | 2 | 0.2708224058151245 | 0.27082228660583496 | 4.401751368834632e-07 | True |
| 8 | 3595 | 118 | 287 | 1.0813792943954468 | 1.0813779830932617 | 1.2126200232007281e-06 | True |
| 8 | 3595 | 1874 | 41 | 0.8726538419723511 | 0.8726538419723511 | 0.0 | True |
| 8 | 5182 | 1874 | 62 | 0.3586098849773407 | 0.3586098849773407 | 0.0 | True |
| 8 | 5182 | 715 | 71 | 0.286310076713562 | 0.286310076713562 | 0.0 | True |
| 8 | 8255 | 1826 | 0 | 0.38079163432121277 | 0.38079172372817993 | 2.3479236176619267e-07 | True |
| 8 | 8255 | 1568 | 0 | 0.3261125385761261 | 0.3261124789714813 | 1.827732384520898e-07 | True |
| 8 | 8356 | 316 | 197 | 0.2524797320365906 | 0.2524796724319458 | 2.360769488093105e-07 | True |
| 8 | 8356 | 18 | 93 | 0.24343428015708923 | 0.24343425035476685 | 1.224245096806569e-07 | True |
| 9 | 5684 | 1216 | 191 | 2.6777262687683105 | 2.677725076675415 | 4.451884830095596e-07 | True |
| 9 | 5684 | 2302 | 50 | 1.935088872909546 | 1.9350872039794922 | 8.624565398908943e-07 | True |
| 9 | 6063 | 1386 | 281 | 0.7319833040237427 | 0.731983482837677 | 2.4428690291598763e-07 | True |
| 9 | 6063 | 2448 | 73 | 0.6613964438438416 | 0.6613972187042236 | 1.1715520839162932e-06 | True |
| 9 | 8543 | 1597 | 299 | 0.8152254223823547 | 0.8152258396148682 | 5.118001744946138e-07 | True |
| 9 | 8543 | 2486 | 161 | 0.6126049160957336 | 0.6126049160957336 | 0.0 | True |
| 9 | 8696 | 1040 | 153 | 0.6981558799743652 | 0.6981564164161682 | 7.683696698196004e-07 | True |
| 9 | 8696 | 287 | 112 | 0.6510873436927795 | 0.6510872840881348 | 9.154631149383165e-08 | True |
| 10 | 2107 | 2193 | 382 | 0.9901700615882874 | 0.9901703000068665 | 2.407854855953996e-07 | True |
| 10 | 2107 | 75 | 154 | 0.7578096389770508 | 0.7578101754188538 | 7.078846393437878e-07 | True |
| 10 | 7436 | 306 | 62 | 0.8848463296890259 | 0.8848463296890259 | 0.0 | True |
| 10 | 7436 | 582 | 159 | 0.7663124799728394 | 0.7663124799728394 | 0.0 | True |
| 10 | 8630 | 176 | 26 | 1.9647027254104614 | 1.964705467224121 | 1.395536141018563e-06 | True |
| 10 | 8630 | 400 | 98 | 1.2304737567901611 | 1.2304741144180298 | 2.9064241856344753e-07 | True |
| 10 | 8615 | 2317 | 37 | 1.1046173572540283 | 1.1046171188354492 | 2.158381610934016e-07 | True |
| 10 | 8615 | 2189 | 21 | 0.9488074779510498 | 0.9488075971603394 | 1.2564117834338085e-07 | True |
| 11 | 6113 | 2239 | 323 | 2.32301664352417 | 2.3230152130126953 | 6.157990639443695e-07 | True |
| 11 | 6113 | 1674 | 83 | 1.1671236753463745 | 1.167124629020691 | 8.17115046632245e-07 | True |
| 11 | 6919 | 163 | 75 | 0.8765182495117188 | 0.8765188455581665 | 6.800160157372027e-07 | True |
| 11 | 6919 | 456 | 348 | 0.7807067632675171 | 0.7807072997093201 | 6.871232941973354e-07 | True |
| 11 | 8304 | 1460 | 2 | 0.5324354767799377 | 0.5324356555938721 | 3.3584150967475427e-07 | True |
| 11 | 8304 | 838 | 5 | 0.45484867691993713 | 0.4548487663269043 | 1.9656420189790598e-07 | True |
| 11 | 8585 | 463 | 386 | 1.06740403175354 | 1.06740403175354 | 0.0 | True |
| 11 | 8585 | 913 | 104 | 1.0144503116607666 | 1.014449954032898 | 3.525336475739927e-07 | True |
| 12 | 104 | 1711 | 23 | 1.5297044515609741 | 1.5297040939331055 | 2.337888657429253e-07 | True |
| 12 | 104 | 357 | 84 | 1.2039597034454346 | 1.2039599418640137 | 1.9802870346845294e-07 | True |
| 12 | 8083 | 1940 | 63 | 1.5639270544052124 | 1.5639275312423706 | 3.048973140147346e-07 | True |
| 12 | 8083 | 1018 | 26 | 1.4184848070144653 | 1.4184859991073608 | 8.403987759423749e-07 | True |
| 12 | 8357 | 485 | 102 | 0.8731285333633423 | 0.8731284737586975 | 6.826560179609529e-08 | True |
| 12 | 8357 | 364 | 281 | 0.629901647567749 | 0.6299015283584595 | 1.8925063938328515e-07 | True |
| 12 | 8678 | 2240 | 13 | 1.5400559902191162 | 1.5400561094284058 | 7.740581531313052e-08 | True |
| 12 | 8678 | 2447 | 51 | 1.4212632179260254 | 1.4212636947631836 | 3.3550235606529547e-07 | True |
| 13 | 1258 | 477 | 71 | 5.467405796051025 | 5.4674072265625 | 2.6164355234846456e-07 | True |
| 13 | 1258 | 1568 | 185 | 2.314375638961792 | 2.3143742084503174 | 6.180982250794385e-07 | True |
| 13 | 4614 | 1996 | 270 | 2.6619975566864014 | 2.6619977951049805 | 8.956378585048025e-08 | True |
| 13 | 4614 | 1946 | 229 | 2.1617021560668945 | 2.1617019176483154 | 1.1029205778068557e-07 | True |
| 13 | 8577 | 907 | 116 | 8.33981704711914 | 8.33981990814209 | 3.430558408001343e-07 | True |
| 13 | 8577 | 1764 | 142 | 4.778460502624512 | 4.77846097946167 | 9.97888667158027e-08 | True |
| 13 | 8479 | 407 | 145 | 3.5706043243408203 | 3.5706019401550293 | 6.677261254523844e-07 | True |
| 13 | 8479 | 2346 | 272 | 2.9574568271636963 | 2.9574575424194336 | 2.41848242968484e-07 | True |
| 14 | 3083 | 2452 | 96 | 7.904738903045654 | 7.904740333557129 | 1.809688456703113e-07 | True |
| 14 | 3083 | 1131 | 223 | 5.888040542602539 | 5.888043403625488 | 4.859040844773405e-07 | True |
| 14 | 4850 | 60 | 167 | 2.846932888031006 | 2.846932888031006 | 0.0 | True |
| 14 | 4850 | 241 | 151 | 2.3277981281280518 | 2.327796459197998 | 7.169565236539833e-07 | True |
| 14 | 8475 | 614 | 434 | 6.83355188369751 | 6.833556175231934 | 6.280093422669755e-07 | True |
| 14 | 8475 | 2411 | 69 | 2.824971914291382 | 2.824971914291382 | 0.0 | True |
| 14 | 8599 | 461 | 33 | 3.763302803039551 | 3.763303518295288 | 1.9006063948056174e-07 | True |
| 14 | 8599 | 449 | 326 | 3.450004816055298 | 3.450002670288086 | 6.219606424687581e-07 | True |
| 15 | 3703 | 2449 | 203 | 27.883533477783203 | 27.883533477783203 | 0.0 | True |
| 15 | 3703 | 2429 | 23 | 21.381563186645508 | 21.381553649902344 | 4.460264706005666e-07 | True |
| 15 | 1851 | 41 | 135 | 6.293060302734375 | 6.293056011199951 | 6.819471318212899e-07 | True |
| 15 | 1851 | 1148 | 392 | 5.428858757019043 | 5.428854465484619 | 7.905039743904819e-07 | True |
| 15 | 8290 | 2231 | 358 | 4.115748882293701 | 4.115750312805176 | 3.475701544288953e-07 | True |
| 15 | 8290 | 608 | 121 | 3.839446544647217 | 3.839447259902954 | 1.8629136491087883e-07 | True |
| 15 | 8227 | 2177 | 231 | 3.026803493499756 | 3.026806116104126 | 8.664600710780827e-07 | True |
| 15 | 8227 | 810 | 40 | 2.979447841644287 | 2.97944974899292 | 6.401684923471858e-07 | True |

### sft: passed

| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2470 | 2371 | 223 | 0.19719915091991425 | 0.19719931483268738 | 8.312042540126952e-07 | True |
| 0 | 2470 | 2193 | 2 | 0.15317195653915405 | 0.15317195653915405 | 0.0 | True |
| 0 | 7413 | 1445 | 14 | 0.16955241560935974 | 0.16955241560935974 | 0.0 | True |
| 0 | 7413 | 1348 | 148 | 0.15318302810192108 | 0.15318302810192108 | 0.0 | True |
| 0 | 8206 | 2429 | 21 | 0.49995988607406616 | 0.4999597370624542 | 2.980471355583943e-07 | True |
| 0 | 8206 | 2041 | 30 | 0.3228643536567688 | 0.3228644132614136 | 1.8461203319693582e-07 | True |
| 0 | 8515 | 1691 | 314 | 0.18649926781654358 | 0.18649925291538239 | 7.989930131256974e-08 | True |
| 0 | 8515 | 2453 | 18 | 0.17732493579387665 | 0.17732493579387665 | 0.0 | True |
| 1 | 1235 | 1918 | 75 | 0.25604644417762756 | 0.2560463547706604 | 3.4918261587363223e-07 | True |
| 1 | 1235 | 463 | 304 | 0.1867983341217041 | 0.1867983341217041 | 0.0 | True |
| 1 | 1357 | 846 | 56 | 0.2662366032600403 | 0.26623663306236267 | 1.1193923759080791e-07 | True |
| 1 | 1357 | 1931 | 27 | 0.2253042757511139 | 0.2253042310476303 | 1.9841382695695228e-07 | True |
| 1 | 8627 | 1372 | 176 | 1.6057233810424805 | 1.6057233810424805 | 0.0 | True |
| 1 | 8627 | 2370 | 19 | 1.1046005487442017 | 1.1045998334884644 | 6.475243363927779e-07 | True |
| 1 | 8361 | 1316 | 371 | 0.10957375168800354 | 0.10957370698451996 | 4.079762068276178e-07 | True |
| 1 | 8361 | 237 | 190 | 0.10344826430082321 | 0.1034482792019844 | 1.4404457430542966e-07 | True |
| 2 | 4266 | 546 | 59 | 0.6346736550331116 | 0.6346736550331116 | 0.0 | True |
| 2 | 4266 | 2217 | 76 | 0.39096859097480774 | 0.3909684121608734 | 4.573613800544245e-07 | True |
| 2 | 5747 | 602 | 12 | 0.33436983823776245 | 0.3343698978424072 | 1.7825963337341204e-07 | True |
| 2 | 5747 | 733 | 8 | 0.2702081799507141 | 0.2702081799507141 | 0.0 | True |
| 2 | 8515 | 1843 | 99 | 0.3522309958934784 | 0.3522309362888336 | 1.6922032833651087e-07 | True |
| 2 | 8515 | 1348 | 211 | 0.24580803513526917 | 0.24580803513526917 | 0.0 | True |
| 2 | 8313 | 778 | 28 | 0.13225707411766052 | 0.13225705921649933 | 1.1266816004557201e-07 | True |
| 2 | 8313 | 2389 | 0 | 0.12754230201244354 | 0.12754230201244354 | 0.0 | True |
| 3 | 6643 | 1835 | 256 | 0.5913618206977844 | 0.5913621187210083 | 5.03960880540609e-07 | True |
| 3 | 6643 | 280 | 4 | 0.40901979804039 | 0.4090200662612915 | 6.557650822143613e-07 | True |
| 3 | 4212 | 1362 | 94 | 0.37643963098526 | 0.37643963098526 | 0.0 | True |
| 3 | 4212 | 918 | 91 | 0.3619903326034546 | 0.36199039220809937 | 1.646581121288812e-07 | True |
| 3 | 8625 | 2354 | 92 | 0.8656414747238159 | 0.8656412363052368 | 2.7542416354025813e-07 | True |
| 3 | 8625 | 298 | 156 | 0.5213357210159302 | 0.5213356614112854 | 1.1433063642606089e-07 | True |
| 3 | 8278 | 1381 | 0 | 0.23223789036273956 | 0.23223790526390076 | 6.416335065123555e-08 | True |
| 3 | 8278 | 86 | 1 | 0.20930776000022888 | 0.20930781960487366 | 2.847703533558691e-07 | True |
| 4 | 4143 | 1895 | 312 | 0.6370136141777039 | 0.6370134949684143 | 1.8713774226735153e-07 | True |
| 4 | 4143 | 317 | 273 | 0.4524356424808502 | 0.45243558287620544 | 1.3174170904962125e-07 | True |
| 4 | 4317 | 897 | 463 | 0.2565929591655731 | 0.2565928101539612 | 5.807314917098837e-07 | True |
| 4 | 4317 | 637 | 267 | 0.23044872283935547 | 0.2304486185312271 | 4.526305334729332e-07 | True |
| 4 | 8250 | 1418 | 24 | 0.21660302579402924 | 0.21660329401493073 | 1.2383063463956995e-06 | True |
| 4 | 8250 | 1173 | 12 | 0.17625856399536133 | 0.1762586086988449 | 2.536244626543051e-07 | True |
| 4 | 8680 | 405 | 34 | 0.2239491045475006 | 0.22394908964633942 | 6.653815930166871e-08 | True |
| 4 | 8680 | 449 | 169 | 0.19529642164707184 | 0.19529642164707184 | 0.0 | True |
| 5 | 4780 | 230 | 146 | 0.7685494422912598 | 0.7685496211051941 | 2.3266419111967316e-07 | True |
| 5 | 4780 | 898 | 51 | 0.35307377576828003 | 0.35307374596595764 | 8.440820143848456e-08 | True |
| 5 | 6870 | 931 | 178 | 1.3845134973526 | 1.3845138549804688 | 2.5830580152247163e-07 | True |
| 5 | 6870 | 1136 | 239 | 1.212725043296814 | 1.2127243280410767 | 5.897921719833973e-07 | True |
| 5 | 8498 | 2189 | 17 | 1.3214082717895508 | 1.3214080333709717 | 1.8042764238086543e-07 | True |
| 5 | 8498 | 2090 | 58 | 0.734369695186615 | 0.7343688011169434 | 1.2174653685888579e-06 | True |
| 5 | 8511 | 378 | 37 | 0.7882740497589111 | 0.7882740497589111 | 0.0 | True |
| 5 | 8511 | 2316 | 115 | 0.6907760500907898 | 0.6907760500907898 | 0.0 | True |
| 6 | 4186 | 2422 | 68 | 1.0897352695465088 | 1.0897351503372192 | 1.0939288915590478e-07 | True |
| 6 | 4186 | 766 | 108 | 0.8496102094650269 | 0.8496102094650269 | 0.0 | True |
| 6 | 873 | 1947 | 175 | 0.3494586646556854 | 0.3494586646556854 | 0.0 | True |
| 6 | 873 | 1068 | 4 | 0.3178592920303345 | 0.31785914301872253 | 4.6879740713779683e-07 | True |
| 6 | 8587 | 1587 | 55 | 0.8386878371238708 | 0.8386868238449097 | 1.2081717610887249e-06 | True |
| 6 | 8587 | 1019 | 55 | 0.43221572041511536 | 0.4322156012058258 | 2.758097031646336e-07 | True |
| 6 | 8446 | 449 | 310 | 0.22862735390663147 | 0.22862732410430908 | 1.30353266476881e-07 | True |
| 6 | 8446 | 1736 | 154 | 0.21885719895362854 | 0.21885719895362854 | 0.0 | True |
| 7 | 4305 | 2145 | 350 | 1.3064062595367432 | 1.3064067363739014 | 3.649991376895372e-07 | True |
| 7 | 4305 | 1659 | 301 | 0.8067013621330261 | 0.8067015409469604 | 2.2166063269481033e-07 | True |
| 7 | 5904 | 353 | 70 | 0.8155080080032349 | 0.8155088424682617 | 1.0232456562856445e-06 | True |
| 7 | 5904 | 1647 | 127 | 0.7059141397476196 | 0.7059136033058167 | 7.599250004686204e-07 | True |
| 7 | 8497 | 2228 | 160 | 0.5081181526184082 | 0.5081181526184082 | 0.0 | True |
| 7 | 8497 | 763 | 4 | 0.38826796412467957 | 0.38826754689216614 | 1.0745993797565893e-06 | True |
| 7 | 8305 | 2266 | 8 | 0.22439728677272797 | 0.2243972271680832 | 2.656210582249992e-07 | True |
| 7 | 8305 | 649 | 205 | 0.21277746558189392 | 0.21277746558189392 | 0.0 | True |
| 8 | 980 | 2008 | 154 | 1.0320098400115967 | 1.0320099592208862 | 1.1551177607903593e-07 | True |
| 8 | 980 | 1443 | 178 | 0.6171543598175049 | 0.6171543002128601 | 9.657980021888847e-08 | True |
| 8 | 3784 | 1081 | 23 | 0.34627169370651245 | 0.3462718725204468 | 5.163977812108668e-07 | True |
| 8 | 3784 | 1029 | 20 | 0.2994111180305481 | 0.29941093921661377 | 5.972187522706754e-07 | True |
| 8 | 8520 | 192 | 95 | 1.073110818862915 | 1.0731110572814941 | 2.2217517045834514e-07 | True |
| 8 | 8520 | 1191 | 153 | 0.6090764403343201 | 0.6090763211250305 | 1.9572139333668472e-07 | True |
| 8 | 8327 | 1104 | 0 | 0.7973280549049377 | 0.7973282933235168 | 2.990219366230481e-07 | True |
| 8 | 8327 | 1244 | 0 | 0.7763993144035339 | 0.7763993144035339 | 0.0 | True |
| 9 | 2060 | 666 | 300 | 0.8887558579444885 | 0.8887556791305542 | 2.0119578704069768e-07 | True |
| 9 | 2060 | 1413 | 159 | 0.5894500017166138 | 0.5894500017166138 | 0.0 | True |
| 9 | 3092 | 1901 | 13 | 1.0187357664108276 | 1.0187348127365112 | 9.361351076994187e-07 | True |
| 9 | 3092 | 1007 | 143 | 0.9213177561759949 | 0.9213170409202576 | 7.76339902829416e-07 | True |
| 9 | 8543 | 166 | 0 | 0.9173321723937988 | 0.9173325896263123 | 4.5483253066221017e-07 | True |
| 9 | 8543 | 2322 | 0 | 0.8390878438949585 | 0.8390878438949585 | 0.0 | True |
| 9 | 8696 | 410 | 294 | 0.5661327242851257 | 0.5661330819129944 | 6.317032266663831e-07 | True |
| 9 | 8696 | 1942 | 259 | 0.5257480144500732 | 0.5257481336593628 | 2.2674225346428152e-07 | True |
| 10 | 6626 | 1745 | 8 | 4.178549766540527 | 4.178550720214844 | 2.2823093410128478e-07 | True |
| 10 | 6626 | 1316 | 29 | 3.477565050125122 | 3.477565050125122 | 0.0 | True |
| 10 | 7436 | 582 | 159 | 0.7594594955444336 | 0.7594594955444336 | 0.0 | True |
| 10 | 7436 | 2189 | 126 | 0.722667396068573 | 0.7226669192314148 | 6.598293499847868e-07 | True |
| 10 | 8520 | 1686 | 60 | 1.3630975484848022 | 1.3630980253219604 | 3.498188069761916e-07 | True |
| 10 | 8520 | 1752 | 143 | 0.8841413259506226 | 0.8841413259506226 | 0.0 | True |
| 10 | 8641 | 1020 | 51 | 0.8031936883926392 | 0.8031938076019287 | 1.4841910646651657e-07 | True |
| 10 | 8641 | 2486 | 218 | 0.7849204540252686 | 0.7849203944206238 | 7.593717869081266e-08 | True |
| 11 | 8136 | 75 | 90 | 2.1396214962005615 | 2.139620780944824 | 3.3429077926857846e-07 | True |
| 11 | 8136 | 886 | 26 | 1.4986649751663208 | 1.4986634254455566 | 1.0340675133134204e-06 | True |
| 11 | 8062 | 1135 | 1 | 6.032081127166748 | 6.032084941864014 | 6.32401518680627e-07 | True |
| 11 | 8062 | 2399 | 86 | 5.692551136016846 | 5.692551612854004 | 8.376510756068049e-08 | True |
| 11 | 8479 | 2452 | 159 | 3.1225976943969727 | 3.12259578704834 | 6.108211237825953e-07 | True |
| 11 | 8479 | 1300 | 192 | 1.5256049633026123 | 1.5256041288375854 | 5.469731987820939e-07 | True |
| 11 | 8251 | 2463 | 167 | 0.43425533175468445 | 0.43425554037094116 | 4.803999892664917e-07 | True |
| 11 | 8251 | 1691 | 246 | 0.41456660628318787 | 0.41456666588783264 | 1.4377579832051177e-07 | True |
| 12 | 1929 | 913 | 92 | 5.244166851043701 | 5.244165420532227 | 2.727814570439674e-07 | True |
| 12 | 1929 | 897 | 236 | 2.700767755508423 | 2.700767755508423 | 0.0 | True |
| 12 | 2306 | 1007 | 140 | 1.1466468572616577 | 1.1466470956802368 | 2.0792677151789974e-07 | True |
| 12 | 2306 | 1786 | 67 | 1.0797934532165527 | 1.0797933340072632 | 1.104000854938262e-07 | True |
| 12 | 8352 | 49 | 170 | 0.7864519953727722 | 0.786452054977417 | 7.57892981721516e-08 | True |
| 12 | 8352 | 2079 | 22 | 0.5906779766082764 | 0.5906780362129211 | 1.0090886597405511e-07 | True |
| 12 | 8438 | 2270 | 131 | 0.5349470376968384 | 0.534946858882904 | 3.3426474347075103e-07 | True |
| 12 | 8438 | 8 | 73 | 0.5241013169288635 | 0.5241014361381531 | 2.274546651577324e-07 | True |
| 13 | 4596 | 2181 | 344 | 14.638962745666504 | 14.638964653015137 | 1.3029260788146499e-07 | True |
| 13 | 4596 | 1259 | 362 | 4.380407810211182 | 4.380406856536865 | 2.177135914567445e-07 | True |
| 13 | 7637 | 1887 | 98 | 3.3878872394561768 | 3.3878886699676514 | 4.222429418397646e-07 | True |
| 13 | 7637 | 410 | 158 | 2.9109346866607666 | 2.910933494567871 | 4.095223781455926e-07 | True |
| 13 | 8515 | 378 | 71 | 4.507451057434082 | 4.507451057434082 | 0.0 | True |
| 13 | 8515 | 375 | 137 | 2.5616984367370605 | 2.5616989135742188 | 1.8614101931939025e-07 | True |
| 13 | 8430 | 199 | 137 | 0.7421175241470337 | 0.742117702960968 | 2.409509659965178e-07 | True |
| 13 | 8430 | 1438 | 340 | 0.6984119415283203 | 0.6984114646911621 | 6.827448527865548e-07 | True |
| 14 | 1765 | 1443 | 95 | 7.355968952178955 | 7.355968475341797 | 6.48231607967674e-08 | True |
| 14 | 1765 | 1606 | 155 | 4.921817779541016 | 4.921817779541016 | 0.0 | True |
| 14 | 4483 | 94 | 295 | 3.856917381286621 | 3.856916666030884 | 1.8544751328484168e-07 | True |
| 14 | 4483 | 2319 | 214 | 3.402834892272949 | 3.4028358459472656 | 2.8025876852615556e-07 | True |
| 14 | 8609 | 1625 | 197 | 6.1718549728393555 | 6.171855449676514 | 7.725994215702651e-08 | True |
| 14 | 8609 | 2173 | 92 | 4.772994518280029 | 4.772994041442871 | 9.990314390198702e-08 | True |
| 14 | 8273 | 2189 | 67 | 1.4100613594055176 | 1.4100605249404907 | 5.917934147257815e-07 | True |
| 14 | 8273 | 2190 | 45 | 1.334536075592041 | 1.3345361948013306 | 8.932638969530769e-08 | True |
| 15 | 2270 | 2040 | 386 | 9.260259628295898 | 9.260263442993164 | 4.1194279844689327e-07 | True |
| 15 | 2270 | 2416 | 1 | 6.068284511566162 | 6.068284034729004 | 7.857857641550466e-08 | True |
| 15 | 6434 | 1716 | 61 | 5.883737087249756 | 5.883726596832275 | 1.7829514345910894e-06 | True |
| 15 | 6434 | 427 | 35 | 5.381531238555908 | 5.381529808044434 | 2.6581866966793664e-07 | True |
| 15 | 8284 | 2125 | 77 | 3.484368085861206 | 3.4843668937683105 | 3.421259941924797e-07 | True |
| 15 | 8284 | 149 | 299 | 2.948031187057495 | 2.9480321407318115 | 3.234953282017809e-07 | True |
| 15 | 8367 | 463 | 386 | 2.6770992279052734 | 2.6770987510681152 | 1.7811710273295757e-07 | True |
| 15 | 8367 | 1343 | 225 | 2.590388774871826 | 2.590388298034668 | 1.8407937944632235e-07 | True |

## Gate 3 by layer

| Layer | Verdict | Primary units |
|---:|---|---:|
| 0 | passed | 8192 |
| 1 | passed | 8192 |
| 2 | passed | 8192 |
| 3 | passed | 8192 |
| 4 | passed | 8192 |
| 5 | passed | 8192 |
| 6 | passed | 8192 |
| 7 | passed | 8192 |
| 8 | passed | 8192 |
| 9 | passed | 8192 |
| 10 | passed | 8192 |
| 11 | passed | 8192 |
| 12 | passed | 8192 |
| 13 | passed | 8192 |
| 14 | passed | 8191 |
| 15 | passed | 8192 |
## Instruct saturation

- 500 → 1000 documents: primary median 0.5; all-neuron median 0.5; nested-sampling reference 0.5
- 1000 → 1500 documents: primary median 0.699999988079071; all-neuron median 0.699999988079071; nested-sampling reference 0.667
- 1500 → 2000 documents: primary median 0.800000011920929; all-neuron median 0.800000011920929; nested-sampling reference 0.75
- 2000 → 2500 documents: primary median 0.800000011920929; all-neuron median 0.800000011920929; nested-sampling reference 0.8
