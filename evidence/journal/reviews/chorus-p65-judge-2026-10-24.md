# Chorus judgment — Part 65 paid judge (2026-10-24)

**Verdict: HOLD first live judge POST until the terminal-release rule, cross-directory fencing, and durable pre-POST state are fixed and tested.** The simplest policy is **no replacement run after a released chunk**. Preserve partial results and report that the registered judge failed; no results for predictions 65a–65g. Recovery of a *known provider batch ID* may still GET results and settle its reservation, but it must not POST the rejected chunk again.

Evidence: I read the supplied judge65.py, analyze65.py, paid_ledger.py, PART65-DESIGN.md, RUNBOOK-65.md, and third review, in Council artifact art_ff32aff56db0bc5e (SHA-256 b384dfff87daa044add342283add4f9da68f4a8e39883efaece651a634eb3019). **[CHECKED]** means verified against supplied source or its written contract. **[INFERRED/PROPOSED]** means reasoning or a patch not executed. This was static review; I did not run tests, use credentials, or make API calls.

## 1. Policy

**Conclusion [PROPOSED]:** Supersede Amendment 2, item 6 *before the first judge call*: a validated provider 4xx that releases a chunk terminates this registered judge. No retry of that chunk and no new output-directory run using the same plan; the judge exits nonzero with a preserved audit trail. A future new judge needs its own expressly registered plan and run identity. The ability to poll/fetch/settle already known batches remains; there is no new POST on behalf of a failed judge.

- **[CHECKED]** The plan is 500 units x two trials, chunked by 100. analyze65.py:231–265 stops if more than 50 of the registered 1,000 are missing. An entire 100-trial rejected POST necessarily exceeds this threshold. **[INFERRED]** There is no scientific benefit to filling later chunks once one is released; report judge failure, not negative hypothesis evidence.
- **[CHECKED]** judge65.py:585–600 records a validated release then *raises* APIHttpError; run_chunks:635–643 catches only JudgeError, so the failing invocation bypasses main:696–714's intended report. A subsequent invocation sees a released state at 616–617 and keeps looping toward later chunks.
- **[CHECKED]** _prior_reservation at judge65.py:503–543 matches by plan SHA and chunk index but not run ownership. An active reservation can be adopted from another output directory, allowing a second POST. A released reservation returns None (541–542), permitting a fresh reserve on a new directory. The round-3 fake-API review demonstrated the first defect; I verified the branches.
- **[PROPOSED]** Give each manifest and chunk a generated judge_run_id and canonical resolved run_root. Put both in ledger reserve metadata alongside plan_sha256 and chunk_index. Before *every* POST, validate that every prior reserve for this plan belongs to the same run/root; for this chunk, an existing attempt must be unique, active, and of that same run. Any released/settled/retired attempt for the chunk is terminal. A ready chunk may adopt an active reservation only for the same run after a crash between reserve and local attempt-ID save.
- **[PROPOSED]** Protect plan lookup, reserve, and POST with a *shared judge lock keyed to the canonical ledger path* (in addition to the root lock). Two different directories can otherwise both observe no matching reservation before either reserves. Holding one lock through run_chunks is simplest for a one-shot job. **Assumption:** both processes use the same ledger and working filesystem lock; a different unrelated ledger cannot enforce global uniqueness. The current root lock is at judge65.py:633, whereas paid_ledger.py:225–238 locks only its individual ledger mutations, not the lookup-to-POST transaction.

## 2. Smallest code change (locations and tests)

**Conclusion [PROPOSED]:** Change state transitions and authorization at the POST boundary, then stamp provenance on answers. No statistical-method rewrite is necessary.

1. **judge65.py:290–326 — manifest/chunk identity.** Generate and store a random judge_run_id once, and the resolved run_root, in run.json and each chunk. Preserve on reopen and on --recover.
2. **judge65.py:503–543, 561–584 — fence the paid POST.** Inspect *all* previous reserve events for this plan, reject a foreign run/root, and refuse all terminal attempts on the same chunk. Include run identity/root in each new reserve. Check ownership and ledger active status again for both local ready and local reserved states, under the shared lock. The reserved path currently bypasses _prior_reservation entirely.
3. **judge65.py:585–605 — release outcome.** After successful validated ledger release *and successful saved state*, return "released" rather than raising APIHttpError. If release or save fails, keep the ambiguous fail-closed behavior and a nonzero error.
4. **judge65.py:627–644 — halt.** Count the released outcome, then break; do not call another POST. Likewise stop (not continue) when a no-batch-ID outcome is ambiguous. main:696–714 can then print a useful report and exit 2.
5. **judge65.py:696–711; RUNBOOK-65.md:76; PART65-DESIGN.md:143–145.** Replace “rerun the whole judge into a new directory” with “terminal failure; preserve partial artifacts; registered missing-trial stop.” A new study requires new authorization.
6. **judge65.py:415–435 and analyze65.py:239–256.** Add judge_run_id and plan_sha256 to each answer row; require both on every nonempty input row, and reject more than one identity pair, *even if the row keys are disjoint*. Optionally validate the identity against sibling run.json in analyze_files:596–610. Let an empty file still reach the registered missing-trial stop.

**Additional paid-path blocker [CHECKED]:** judge65.py:276–280 implements _atomic_json as write temp + os.replace, with **no fsync of file or containing directory**. But judge65.py:580–584 relies on the local posting marker surviving a machine crash *before* POST. paid_ledger.py:384–413 does fsync ledger events. **[INFERRED]** A power-loss recovery might retain the active ledger reservation but roll the chunk back to ready/reserved, reopening the same-run duplicate-POST path. Add flush/os.fsync on temp, os.replace, then fsync on parent directory. Require this for the posting marker and returned provider batch ID before considering them durable. This is separate from the cross-directory ownership fix.

**Minimum fake-API regression tests [PROPOSED]:**

1. A validated HTTP 400 on chunk 0 creates one POST, records released, leaves chunk 1 unposted, returns released=1 and exits 2 with the terminal message. --recover makes zero POSTs. Invalid Anthropic error body and HTTP 429 do *not* release; they halt with active reservation.
2. Directory A has a known pending chunk. Directory B, same plan but different run ID, must refuse before POST. Repeat after A's release and settlement; include copying a chunk already in local reserved state into B. Exactly one POST overall.
3. Barrier-interleave two fresh directories at preflight: one may POST, the other must reject. This detects a run-root check without a shared lock.
4. Inject interruptions after reserve, durable reserved, durable posting, and POST-before-batch-ID. Neither recovery nor a new directory may re-POST an ambiguous chunk. Examine fsync ordering by code review; ordinary kill tests do not simulate power loss.
5. Merge non-overlapping valid unit/trial rows from two runs with different judge_run_id but the *same plan hash*. analyze_rows must reject them on provenance, not score them; one-run data must pass.

These tests are specified, **not executed**. Include the same paid-path checks in the one-request probe.

## 3. Anything else I would block on

**Blocks [CHECKED]:** (a) durable posting state before external POST; (b) run ownership checks on the local reserved as well as ready branch, with cross-directory serialization; and (c) stop future spending on any released or ambiguous state. Current run_chunks:639–642 can count certain ambiguous errors and move to later chunks. These are all safety obligations of the terminal policy, not independent experimental redesigns.

**Budget condition [CHECKED/INFERRED]:** Amendment 2 says “under three cents”, but PART65_CAP_USD is $1.00 (judge65.py:45). Round-3's synthetic summaries estimated $0.03309; that review lacked the actual ranked output. Compute the plan reserve total from the real frozen requests and either correct the wording to the measured bound, or enforce a preflight whole-plan <= $0.03 limit **if** three cents is a hard authorization limit. I would not independently block a consciously authorized roughly three-cent run on the synthetic estimate.

**Not additional blockers [CHECKED]:** Known batch recovery is GET-only in _fetch_and_save:460–500; repeated settlement is idempotence-checked in paid_ledger.py:479–494; validated 4xx release rejects malformed provider bodies in paid_ledger.py:515–538; and the registered >50 missing-trial stop is implemented in analyze65.py:231–265. I have not verified Anthropic's live behavior, current pricing, model availability, or production filesystem lock semantics.

**GO criterion:** record the terminal-release amendment; fence one run against all other directories in the shared ledger; sync pre-POST state; stop on release/ambiguity; reject mixed-run answers; pass these fake-API tests; and verify actual reserve sum. Only then run the one-request probe, inspect its recorded outcome, and authorize the full job.
