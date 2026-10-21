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
