VERDICT: SHIP

# Scope

Cold delta review of Part 213 changes in `936afadc..228ceaae`; Part 212 and `PROCEDURE.md` excluded. No regression introduced by this delta was measured. Two round-1 audit findings remain incompletely addressed; both are PRE-EXISTING follow-ups and do not hold this delta under the review contract.

# Findings by severity

## HIGH — PRE-EXISTING — F7 remains incomplete: a labelled subset of Stage 2 rows can count as the full audit

Location: `part213/score213.py:953-1001`, especially `complete = bool(audited) and labelled == len(audited)` at line 978. This is a pre-existing gap in the partial-audit fix, not a newly reachable path: the implementation checks every *present* row but does not verify the registered total of 84 slot rows (42 neuron and 42 random) over both `THIRD_FREE` prompts.

Measured with `exp_audit_contracts.py`, which imports the real scorer, builds only one prompt's 21 slot neuron/random pairs, assigns real seed-213 blind IDs, and labels all 42 observed rows (10 neuron hits, zero random hits):

```text
PARTIAL_STAGE2_AUDIT {"audit_complete": true, "audited_rows": 42, "clauses": {"difference_ci_above_0": "held", "neuron_ge_10_of_42": "held", "random_le_2_of_42": "held", "sign_test_p_lt_0.05": "held"}, "expected_audited_rows": 84, "hits": {"neuron": 10, "random": 0}, "labelled_rows": 42, "observed_rows": 42, "verdict": "held"}
```

The scorer can report `213.6: held` with only half the registered answers, comparing 10/21 vs 0/21 against fixed 10/42 and 2/42 thresholds. Smallest fix: require exactly 84 audited rows with the expected per-arm/per-prompt counts before `complete=True`; otherwise return `not read`.

## MEDIUM — PRE-EXISTING — F5 remains incomplete: Gate 3 passes with no audit or reader agreement

Location: `part213/score213.py:768-797` (`gate3`). The round-1 fix brief explicitly required Gates 3 and 4 not to pass without the audit. The new implementation sets `kappa_ok = True` whenever `records["labels"]` is empty, so four clean Stage 2 baseline answers produce `passed: true` with `kappa: null` and no reader data.

Measured with `exp_audit_contracts.py`, calling the real `score213.gate3` with the four Stage 2 baseline prompts, unbroken answers, and no labels:

```text
GATE3_WITHOUT_AUDIT {"baseline_broken": [], "kappa": null, "labels_present": false, "note": "kappa is `not read` until the frozen second reader labels the 40-item sample", "passed": true}
```

This marks reader agreement passed before evidence exists; `213.6` is reported `not read` instead of being voided by Gate 3. Smallest fix: make Gate 3 fail or remain unread until the required audit and kappa evidence exist, then enforce the registered agreement threshold. The headline's other audit clauses still prevent this absent-audit path from supporting a final headline.

# What I checked and found sound

- **F1 / read positions:** `exp_contract_audit.py` uses the real tokenizer and changed runner. Across all 39 units × 17 prompts (663 rows), `first_pos` includes opening and lead tokens, and every `content_pos` predicts the corresponding content token. Output: `"real_tokenizer_position_rows": 663`.
- **F2/F3 / row-local NP and off-diagonal contract:** The full smoke exercised Stage 1 on a tiny OLMo. The post-smoke audit found 6,474 instruct JSONL rows, exactly 936 unique `(push_unit, scored_unit, prompt_id)` offdiag keys, the exact `offdiag_8` peer set for every push and prompt, zero missing required logits, and zero missing `full_np_increment` values for the expected pairs. No `lp_by_unit` records remain.
- **F4 / manifests and pilot evidence:** The smoke ran `pilot`, `stage1 --pilot`, `stage2`, and `score213.py`; it wrote per-checkpoint manifests, merged pilot evidence, and Gate 1 and Gate 2 passed. `load_records` read the SFT/DPO/Instruct manifests. Stage 2 wrote 472 rows / 364 executed rows without overwriting `run-instruct.json`.
- **F5 / headline CI and Gate 4:** With no labels, the smoke produced `213.6: not read` and Gate 4 failed. The new headline branch requires a complete Stage 2 CI with `read` and upper bound `< 10/42`; the F5 fixture covers missing-CI and present-CI branches. Gate 3's absent-audit gap is listed above.
- **F6 / completion markers:** `test_f6_partial_output_is_not_a_finished_run` passed; it checks absent marker, matching marker, and digest mismatch after modifying the JSONL. The runner skips only when `is_complete` accepts the marker.
- **F8 / sparse logits:** The post-smoke audit found zero core rows missing own-token plus `offdiag_ou` logits. The complete smoke reported 11.0 MB of JSONL and at most 35 first-logit entries per row (vocabulary size 100,278).
- **Dose scaling and hook application:** `exp_hook_scaling.py` called the real `scale_vector` and `part92.common92.add_row_hook` wrapper for neuron and random_0..3 vectors at +0.8, −0.8, ±0.05, and ±0.025. Output: `hook deltas: 5 arm vectors x 6 doses x 5 rows x 7 positions = exact; hook removed after each pass`. The smoke also executed the runner's neuron/random, negative-dose, finite-difference, and offdiag code paths on its tiny model.
- **CLI and tests:** The committed smoke exited 0 with `SMOKE OK (793.7s)`; every prediction had a scored verdict without missing-file/field errors. The fixture suite passed: `35 passed in 39.03s`.

# Scope and limits

The smoke substitutes a randomly initialized tiny model, random push vectors, and a synthetic Part 106 reference. It validates plumbing and data shape, not scientific results on frozen production weights. The two findings above reproduce against the real scorer and target registry, but both are PRE-EXISTING rather than regressions from this delta.
