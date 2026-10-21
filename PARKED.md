# vastrun is PARKED (round-3 stop rule, BRIEF-vastrun.md)

Round 3 (review/LUNA-R3.md, review/HAIKU-R3.md) found two ways a run reports success with a box still listed:
a lost rent reply whose box is listed without our label (Luna P1) and a label sweep that treats "no label match" as
destroyed while a tracked id is still listed (Haiku F1); plus a stop signal during cleanup setup that skips all
cleanup (Luna P1). The stop rule written before round 3 says: park, run GPU jobs by hand.

Reopen when one of these exists: Vast offers an idempotency or client-correlation key on rent (so a lost reply can be
resolved without guessing), or a design where cleanup treats ANY instance created during our rent window that is not
provably someone else's as ours, reviewed as a design first. Not reopened by another patch round.

## By-hand checklist (how Part 48 ran cleanly, 2026-10-21)
1. Before renting: GET v1 /instances/ (all pages via next_token) and note anything listed.
2. Rent one offer with a label; write the instance id and label into the run's cloud/<run>/README.md at once.
3. Upload, prefetch models (part43 forces HF offline), start the job under nohup, schedule a check (not a babysitting
   agent: those die at session boundaries).
4. On completion copy results back, then DELETE the instance, then GET v1 /instances/ (all pages) until it is gone.
5. Log spend in memory reference_vast_gpu_account.md and the README; destroy before ending any session that rented.
