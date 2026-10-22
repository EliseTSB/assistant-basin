# Brief: the claim ledger for the public write-up (Elise 2026-10-22: make our results presentable, then publicize)

Domain: /home/elisepc/assistant-basin/publish (write only there; read the rest of /home/elisepc/assistant-basin). Do not commit.

We have ~50 numbered Parts in this repo (README.md "Results so far", PREDICTIONS.md, PART*-DESIGN.md, results/part*/). Before anything is
written for the public, I need one table of every result that could be a headline, each checked against the result files, not
against the README's own prose.

Write `publish/CLAIMS.md`: one row per Part or sub-part with a stated result (skip pure builds and designs). Columns:
1. Part id and one-sentence claim (plain words, no jargon beyond neuron, layer, steer, ablate).
2. Model(s) and size.
3. Kind: causal (we changed the model and measured), correlational (we only read it), or replication/null.
4. n (prompts, units, pairs, samples) and the effect with its interval or the raw counts; copy the numbers from the results
   file and give the path you read them from.
5. The control(s) that make it mean something (random sets, unadapted model, matched prompts, per-unit null) and whether
   the control behaved.
6. Registered prediction: held, failed, or none, from PREDICTIONS.md.
7. Strength: A (causal, controlled, n large enough that the headline needs no qualifier), B (real but small model, small n, or
   one control missing), C (suggestive, exploratory, or later superseded), N (null we should report as a null).
8. Caveats that must travel with the claim (judge model, hand reading, bf16, own-written prompts, single seed...).
9. Prior art: if PRIOR-ART.md or the README names the closest earlier work, name it; else "none recorded".

Rules: if the README number and the results file disagree, the results file wins and you list the disagreement in a section
"Mismatches" at the end with both numbers. If a claim has no results file you can find, mark it UNVERIFIED and say where you
looked. Do not rank by how interesting it is; I do that. Do not soften or strengthen wording: use what the data supports.
Also add a section "Superseded or withdrawn" listing every claim the repo itself later retracted (the README says so in
places; grep for "withdrawn", "failed", "superseded", "refuted").

Finish with `publish/REPORT-ledger.md` (SUMMARY / FINDINGS / BLOCKERS): counts per strength grade, the mismatches, anything you could
not verify.
