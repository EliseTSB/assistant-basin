# Part 125 score

All gates passed: **True**

## Gates

| # | Gate | Result | Evidence |
|---:|---|---|---|
| 1 | Zero-vector hook has same token IDs at batch size one | PASS | `{"checks": 45, "expected": 45}` |
| 2 | Pushed path and no-hook twin are bitwise deterministic on both pilot units | PASS | `{"units": ["p92-02", "p92-08"], "rows": 2, "all_equal": true}` |
| 3 | Twin cache agrees with one-forward full re-encoding | PASS | `{"comparisons": 145, "required_comparisons": 145, "top1_agree": 140, "top1_required": 138, "max_top100_abs_diff": 0.25, "max_required": 1.0}` |
| 4 | DeepSeek and Luna each pass 11 of 12 known examples | PASS | `{"deepseek": {"correct": 12, "total": 12, "passed": true, "required_correct": 11, "reader": "ds"}, "luna": {"correct": 12, "total": 12, "passed": true, "required_correct": 11, "reader": "luna"}}` |
| 5 | DeepSeek/Luna agreement is at least 0.80 and kappa is at least 0.60 when defined | PASS | `{"agreement": {"n": 348, "expected_n": 348, "unresolved": [], "agreement": 0.9798850574712644, "expected_agreement": 0.5, "kappa": 0.9597701149425288, "single_class": false, "gate_method": "cohen_kappa", "gate_value": 0.9597701149425288, "threshold": 0.6, "kappa_threshold": 0.6, "agreement_threshold": 0.8, "gate_passed": true, "sample_policy_match": true}}` |
| 6 | Template ledger, paired call plans and hook coverage pass at every turn | PASS | `{"checks": 145, "expected": 145, "parameter_rows_match": true}` |
| 7 | All current DeepSeek labels and private joins match answer hashes | PASS | `{"missing_or_stale_deepseek": 0, "missing_or_stale_luna_sample": 0}` |
| 8 | Pilot pushes change logits and the target layer output by the intended amount | PASS | `{"checks": [{"unit_id": "p92-02", "layer": 3, "neuron": 2628, "question": "Who are you?", "target": "last prompt token at the hooked layer", "logits_differ": true, "plain_logit_sha256": "ca2046ec351c823e0df8ea652aab3995016416038b82d6f9b5fd9ba33fc51006", "hooked_logit_sha256": "82b84742beccfa890a08a347170e45ddbb3baec28d51489c3663ce08e94beed2", "intended_push_norm": 5.915412425994873, "observed_target_shift_norm": 5.915394306182861, "target_max_abs_error": 0.00390625, "tolerance": 0.05, "passed": true}, {"unit_id": "p92-08", "layer": 4, "neuron": 5154, "question": "Who are you?", "target": "last prompt token at the hooked layer", "logits_differ": true, "plain_logit_sha256": "ca2046ec351c823e0df8ea652aab3995016416038b82d6f9b5fd9ba33fc51006", "hooked_logit_sha256": "afc83429c103ba4eed07ebcf4e81da59929d4f78c22bca6e3933fa85a54c442d", "intended_push_norm": 7.234429836273193, "observed_target_shift_norm": 7.234312057495117, "target_max_abs_error": 0.00537109375, "tolerance": 0.05, "passed": true}], "gate_passed": true}` |

## Turn 5 reader counts

Missing labels are excluded from counts and are never treated as false.

| Arm | Coherent persona | Shows persona | Incoherent | Known | Missing |
|---|---:|---:|---:|---:|---:|
| arm1 | 11 | 23 | 15 | 29 | 0 |
| arm2 | 2 | 3 | 3 | 29 | 0 |
| arm3 | 3 | 4 | 2 | 29 | 0 |
| arm4 | 0 | 0 | 0 | 29 | 0 |

## Predictions

### 125.1 (medium)

At turn 5, the coherent-persona counts of arm 2 and arm 3 are within 5 units of each other.

Verdict: **held**. arm 2 2, arm 3 3 of 29 selected units; difference -1.

### 125.2 (medium)

At turn 5, arm 3 shows a coherent persona in at least 6 of 29 units, so text alone carries the persona across four turns.

Verdict: **not held**. arm 3 has 3 coherent-persona labels at turn 5, among 29 selected units.

### 125.3 (medium)

At turn 2, delta_2 is above the mean of its three delta_r_2 in at least 20 of 29 units.

Verdict: **not held**. 14/29 units have delta_2 above their three-direction mean.

### 125.4 (low)

The count in 125.3 is at least 5 units lower at turn 5 than at turn 2.

Verdict: **held**. The above-random count changes from 14/29 at turn 2 to 9/29 at turn 5.

### 125.5 (medium)

At turn 5, arm 1 has at least 6 more incoherent answers than arm 2. Pushing every turn breaks answers more than it holds a persona.

Verdict: **held**. arm 1 has 15 and arm 2 3 known incoherent answers; difference 12.

### 125.6 (medium)

At turn 5, arm 4 shows a coherent target persona in at most 2 of 29 units.

Verdict: **held**. arm 4 has 0 coherent-persona labels at turn 5, among 29 selected units.

## Interpretation

At turn 5, the real cache delta is no larger than the mean random-direction delta for 20 units (p92-02, p92-03, p92-04, p92-05, p92-07, p92-08, p92-10, p92-13, p92-18, p92-24, p92-29, p92-35, p92-36, p92-42, p92-43, p92-45, p92-46, p92-47, p92-48, p92-50). For these units, the kept cache changes the model but not toward this persona on this measure.
Arm 1 has at least 6 more incoherent answers than arm 2 at turn 5. Pushing every turn breaks answers more than it holds a persona.

## Reader-free per-turn summaries

Positive delta means arm 2 assigns more probability to the persona sentence than the twin. KL is `KL(arm 2 || twin)` for the next-token distributions.

| Turn | Comparisons | Mean delta | Median delta | Mean KL | Median KL |
|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 1.79408 | 1.79155 | 1.8958 | 1.90751 |
| 2 | 29 | -0.0515007 | -0.00217724 | 0.466297 | 0.403141 |
| 3 | 29 | -0.250651 | -0.0793092 | 0.0927099 | 0.0242045 |
| 4 | 29 | -0.620582 | -0.467837 | 0.180583 | 0.12512 |
| 5 | 29 | -0.885407 | -0.61348 | 0.220178 | 0.145171 |

## Cache-sanity distribution

{"n": 145, "min": 0.0, "median": 0.125, "p90": 0.15625, "max": 0.25, "top1_agree": 140, "comparisons": 145}

## Missing labels

None.
