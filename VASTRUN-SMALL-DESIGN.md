# vastrun, the smaller design (stop rule fired; written 2026-10-21 13:57 CT, nothing built yet)

Round 2 (Haiku and Luna independently): an unrelated label-less instance makes the listing "unreadable", so the label
sweep never destroys our box; and a run-record write failure after rent skips cleanup. Both are the class the stop rule
names. The fix is structural: cleanup must depend on nothing that can fail locally.

1. **One label, chosen before rent, is the only handle.** Cleanup = "destroy every listed instance whose label equals
   this run's label, then list again until none remain (retry with backoff up to a deadline)". It needs no recorded id.
2. **Listing parse:** a dict with an "instances" list; each item needs only an "id"; label read with .get (missing or
   null = not ours). Anything else is unreadable: retry, never "gone". No other strictness.
3. **Cleanup cannot be skipped:** everything after rent sits inside one try whose finally runs cleanup; SIGINT, SIGTERM
   and SIGHUP raise into that try; record writes, copies and prints are best-effort (caught, logged to stderr) so none
   can raise past cleanup. Cleanup itself catches and retries; the exit code is non-zero if anything with our label is
   still listed at the deadline.
4. **One job per rental.** Upload, one setup+job command via nohup, poll, copy its declared results to
   results_dir/<job>/ (namespaced), then cleanup. Several jobs = several runs (setup paid each time; acceptable).
5. **Spend:** rate = max(offer dph, any positive listed dph); cap and wall clock both trigger cleanup.
6. **Dead runner:** the box powers itself off at wall_clock+10 min (compute stops; disk may still bill until destroyed).
   Every new run first lists instances with a "vastrun-" label and refuses to start while any exist, naming them, so a
   dead run's box is noticed at the next run; "vastrun.py sweep LABEL" destroys one by label.
7. **Tests:** label-less and null-label neighbours; lost rent response; record write failure; signal at every phase
   (before rent returns, after, during job, during cleanup); unreadable listing then readable; a zero listed rate.
   Sabotages: cleanup outside finally; strict label parse; listing-lowers-rate.

Asking a reviewer: does this close the class, and is refusing to start while any vastrun-* instance exists the right
guard for a dead runner?

## Review and live probe (Prime, council orchestrator-vastrun-design-2026-10-21 #3; probe cloud/probe-label-2026-10-21/)
Live, 2026-10-21 14:0x CT, a GTX 1080 at $0.056/h for about a minute: a label set at create comes back EXACTLY in the
v1 listing ("vastrun-52b69190ab90-6ad90cab", 29 chars). The v1 listing is {"success", "instances", "total_instances",
"instances_found", "label_counts", "next_token"}: it is PAGINATED (next_token), so every sweep follows next_token to
the end before deciding. DELETE returned {"success": true} and the next listing no longer showed the box.
Folded from Prime:
- Label = vastrun-<uuid12>-<deadline epoch hex>; deadline = start + wall_clock + watchdog + 25 min.
- Keep the in-memory id as a second handle (DELETE by id when the rent reply gave one, whatever the listing says).
- Per-item parse: skip malformed neighbour items (logged); only a malformed item whose label matches ours is a hard error.
- DELETE then 404, or a listing that lags, confirms; the exit message distinguishes "deleted, listing lags" from "still there".
- Wall clock and spend cap are TIMERS that raise into the try (a watchdog thread / signal.alarm), so a hung API or ssh
  call cannot block cleanup; every network call also has its own timeout.
- Pre-run: list all vastrun-* (all pages); destroy those whose deadline has passed; refuse to start only while a
  non-expired one exists (naming label and expiry); an unreadable pre-rent listing means do not rent.
- A periodic local sweep (vastrun.py sweep-expired, for a user timer) applies the same expiry rule for the no-next-run case.
- Print the label at start and the cleanup outcome at exit. Reserve generated log paths; error messages name the step,
  never the command or env.
- Measured later, not assumed: the box-side poweroff watchdog (whether a stopped box bills storage).
