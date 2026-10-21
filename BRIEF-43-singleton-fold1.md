# Fold brief: Part 43 singleton, review round 1 (Luna)

Run size: code and tests only. Fix exactly the findings in review/part43/REVIEW-blind-C-luna.md (a blind Sol review of
this branch's build, labelled "C" there):
- C-B1: the tests can pass when singleton mode actually batches every record. Add a test that observes forward batch
  sizes (every confirmatory forward in singleton mode has batch size 1) and a sabotage that makes singleton batch and
  must fail it.
- C-B2: the singleton pilot still aborts on the original batched-logit check. Take run_validation_checks out of the
  singleton pilot's gate; keep it runnable as an explicit diagnostic command (and keep its receipts and tolerances). The
  pilot's gate becomes the singleton `validate` evidence, which reports numbers and never applies a budget here (Claude
  pre-registers the budget separately).
- C-N1: make _capture_norm_inputs singleton-row aware, with a tiny-model test.
Also, from the same review's shared findings: confirmatory commands must refuse `--exec-mode batched` (check this
branch still does), and the environment record must observe the attention backend from the loaded model rather than
assert one.
Keep every existing test green. Commit nothing. Report REPORT-43-singleton-fold1.md ending SUMMARY / FINDINGS / BLOCKERS.
