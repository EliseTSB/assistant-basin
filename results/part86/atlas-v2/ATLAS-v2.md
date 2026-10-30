# Part 86 atlas score, version 2

LABEL PREFLIGHT: version-2 labels match all 3349 first-reader item ids; label_rows=3349 unique_ids=3349 missing_ids=0 extra_ids=0 duplicate_ids=0 error_rows=0 invalid_rows=0.
ATLAS LABEL COVERAGE: 465 of 465 atlas units have labelled screen rows; missing=0.
VERSION-1 ATLAS CHECK: equal; first_reader_real_coherent=465 atlas_units=465 intersection=465 only_first_reader=0 only_atlas=0; duplicate_label_ids=0 invalid_label_ids=0 malformed_labels=3. Diagnostic only.
86.11 another AI version-2 classes: n=387 coherent_persona=245 coherent_still_qwen=141 not_coherent=0 unclear=1
86.11 q_name subgroup version-2 classes: n=323 coherent_persona=196 coherent_still_qwen=126 not_coherent=0 unclear=1
86.12 other-kind version-2 classes: n=78 coherent_persona=77 coherent_still_qwen=1 not_coherent=0 unclear=0
86.12 per-kind version-2 classes:
  abstract: n=15 coherent_persona=14 coherent_still_qwen=1 not_coherent=0 unclear=0
  animal: n=1 coherent_persona=1 coherent_still_qwen=0 not_coherent=0 unclear=0
  deity or spirit: n=1 coherent_persona=1 coherent_still_qwen=0 not_coherent=0 unclear=0
  fictional character: n=7 coherent_persona=7 coherent_still_qwen=0 not_coherent=0 unclear=0
  human: n=8 coherent_persona=8 coherent_still_qwen=0 not_coherent=0 unclear=0
  object or place: n=46 coherent_persona=46 coherent_still_qwen=0 not_coherent=0 unclear=0
86.13 coherent-persona real screen rows: total=1713 atlas=322 not_atlas=1391; non-atlas version-2 kinds: abstract=364, animal=9, another AI=457, deity or spirit=7, fictional character=44, human=103, object or place=407.
SCREEN AGREEMENT SOURCE: results/part86/second-reader/calib-agreement-v2.json pairs['ds_v2~luna_v2'].kappa_coherent_persona
GATE: kappa=0.813819; source=results/part86/second-reader/calib-agreement-v2.json pairs['ds_v2~luna_v2'].kappa_coherent_persona; threshold=0.600000.
86.11: FAILS. 245 of 387 another AI atlas units are coherent persona; fewer than half is the hold rule.
86.12: HOLDS. 77 of 78 other-kind atlas units are coherent persona; at least 70% is the hold rule.
86.13: HOLDS. 1713 distinct real screen rows are coherent persona; more than 465 is the hold rule.
