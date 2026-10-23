# Brief: fold the cold reviews into the Part 60 runner (the Luna build, /home/elisepc/assistant-basin/part60)

Domain: /home/elisepc/assistant-basin/part60 (write only there). No network, no API call, no Qwen load. Do not commit. Do not change a prediction, a method definition or a metric of PART60-DESIGN.md; Amendment 1 at its bottom is part of the design now (M6 = thinking ENABLED via {"type":"adaptive"}, max_tokens 4000, one smoke call that must show a non-empty thinking block or M6 is dropped).

## Reviews to read (all of them, first)
- This build was reviewed blind as "B": /home/elisepc/green/Journal/scratch/p60-rev-B-luna/REVIEW.md and p60-rev-B-haiku/REVIEW.md. Fix EVERY finding in them (high, medium and low) unless it is wrong; if you think one is wrong, say so with the evidence in the report.
- The other two builds' reviews (A: p60-rev-A-luna, p60-rev-A-haiku; C: p60-rev-C-luna, p60-rev-C-haiku) found defects in code that is NOT yours, but their finding CLASSES may apply to your code. For each of their high-severity findings, check whether the same defect exists in /home/elisepc/assistant-basin/part60 (grep the code), and fix it if so; list each as "checked: present/absent".
- Papercuts already filed against this build: /home/elisepc/green/Journal/ops/papercuts/2026-10-22/alert-203608-8pixy-0010.md, alert-203839-8pixy-0012.md, alert-203841-8pixy-0013.md. In particular: run60.sh preflight must not require M7.jsonl when M7 is skipped; a failed `pip freeze` must not fail a run; the runbook's destroy step must follow a CHECKED scp; a rejected thinking field lands as a per-request error inside the batch (not a 400 on submit) so the M6 smoke test and the runbook wording must account for that.

## Decisions (mine)
1. M6 per Amendment 1 (adaptive thinking, max_tokens 4000). gen60.py gets a `smoke` subcommand: one real batch-of-one or direct message call for M6 on one dev unit, passing only if the response contains a non-empty thinking block, writing the result to out/smoke-M6.json; `submit M6` refuses unless that file says pass; if the API errors or no thinking appears, `submit M6` marks M6 dropped (as the stop rule) and analyze60 treats M6 as dropped, with a test.
2. 60j: the test candidate must be chosen on dev and then compared on test against BOTH M0 and the dev leader's own result; fix the self-comparison bug (finding 1 of the B/Luna review) so that if the candidate is M1 the 5-point lead is required over M0 only; keep the prediction wording as registered; add a test where M1 wins.
3. Spend cap: write the `reserve` event BEFORE the POST (and settle after), idempotent on a rerun; a kill between POST success and the ledger append must not allow a double submit or an undercount: use a per-chunk manifest entry marked `posting` before the POST and a reconcile step that lists batches by an idempotency tag in each chunk's custom metadata, refusing to resubmit a chunk whose `posting` entry has no outcome until reconciled. Handle POST retries so that a retried POST can never create a second batch for the same chunk.
4. M3/M5 second-call rejection must not overwrite first-round evidence (keep round-1 and selection rows; record the second call's rejection beside them).
5. run60.sh: remove any stale success archive at start; a successful run must print RETURN_READY and exit 0 even if pip freeze fails.
6. Add a staging note and a path check at the top of gen60.py/score60.py: if the data files are not found relative to the checkout, print the exact fix instead of a late error.
7. Replace the two toy regression tests with tests that call the production functions (compute_thresholds, choose_m3_winners) on small fixtures, and add a sabotage for each that must go red.
8. Re-run ALL earlier sabotages plus the new ones; every new fix gets a test that fails without it. Run the full suite.

## Report
part60/REPORT-fold.md: SUMMARY / FINDINGS / BLOCKERS: every review item (B/Luna 1-7, B/Haiku all, the checked-present/absent list for A and C highs, the three papercuts) as fixed (where, which test) or not fixed (why); test count and sabotage results.
