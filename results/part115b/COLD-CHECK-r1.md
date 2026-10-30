# Part 115b cold check, round 1 (Haiku 5.5, 2026-10-29 20:5x CT, read-only; condensed by Claude, to fold)

1. Title overstates: only the direct-write threshold varies with the opening (100 plain, 50 quote and bold); the gradient
   threshold is 500 in all three; "instructed opening" is undefined. Fix the title to say exactly that.
2. A sentence says the plain and quote openings change the target to " Q"; only the plain one does (ids: plain 1167,
   quote and bold 48). The plain-versus-quote contrast changes the token and the quote mark together.
3. "Does the change track markdown context or the target token?" is not a registered prediction: label the section
   exploratory, state the facts (quote and bold share a threshold within each ranking; plain differs from quote only for
   direct-write, a contrast that also drops the quote mark), and delete the closing sentence that answers nothing.
4. Provenance: results/part115b/probe115b.json is pass 2's output (no 'pass' key; no quote, newline or colon openings).
   Pass 3's output, which showed 'My name is "' with Q first on 16 of 16 before registration, is now saved by Claude as
   results/part115b/probe115b-pass3.json (from the job folder's p115b3). Fix the README's provenance line and say in
   Method how the instruction and openings were chosen (passes 2 and 3, the same 16 phrasings).
5. Amendment 1 said "If the direct load fails or changes nothing, the run waits for more free memory instead", but after
   the first direct-load failure the warm-up was skipped and the run relaunched (b2e90a0 at 20:11:39, launch 20:11:42).
   Claude adds a dated note to the design; the README should say the fallback was not used and why, citing that note.
6. The comparison with Parts 115 and 116 treats the instruction as the only difference; the opening wording also changed
   ("I am **" against "My name is **"). Say both differ and cannot be separated here; restore the design's caveats (the
   bold opening is not Part 115's prompt; a set chosen for each opening could need fewer neurons).
7. 115b.7: quote a continuation ("Chat." and "Chat. I am a large language model trained by Google.") and write
   "failed (literal: 'Chat' is not on the list)".
8. Off-target scope: five of six probes end in "**", they run without the plain-text instruction, and they were chosen
   because their plain-model first choice was the answer (the unbolded versions put " **" or " typically" first).
9. The random controls are Part 115's (gradient) and Part 116's (direct) sets, unit for unit; say so.
10. The all-ones gate checks one prompt (held-out phrasing 1 with "My name is"); the direct load has no gate against the
    standard loader (the cold reader compared it with the probe: held-out log-odds within a mean 0.03).
11. Define "gradient ranking", "direct-write ranking", "flip", "withdrawal rule" and "log-odds" before use.
12. Amendment history: f97ef14 added Amendment 1 and the runner's --direct-load; b2e90a0 added the warm-up skip; both
    precede the successful launch; the amendment says "before any data", not "before scoring".
13. 115b.3: both thresholds are only bracketed in (200, 500]; plain 13 of 16 against quote 16 of 16 at 500.
14. The design header says 20:07 CT; the registration commit 091c8bb is 20:09:00. Note it; do not edit the header.
15. PREDICTIONS.md cells and commits: Claude fills and commits at publish.
