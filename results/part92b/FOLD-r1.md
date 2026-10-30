# Fold record, cold check round 1

## Blocking findings

1. Replaced the lead with the registered sentence that Part 92's split depends on the reader; identified DS as the only reader assigning unit kinds and stated the 21, 8, 7, 3 and one-unit-failure counts.
2. Added Part 86's v1 and v2 agreement numbers, distinguished the brief-based coherent-persona construct from Part 92b's shows_persona prompt, and said this run cannot identify which construct explains the gap; named Parts 112, 115, 117 and 117b as atlas descendants.
3. Replaced cross-reader class language with “Only DS gives unit-level kinds” and limited the classification comparison to the DS reading.
4. Retitled the write-up to name DS, removed “robust,” “survive” and “stable,” and reported the one-answer-per-slot, six-at-threshold and seven-single-answer margins with command-backed counts.
5. Used the specified sentence that all 27 disagreements run Haiku-yes/DS-no without showing which reader is wrong; added the p92-49 random-arm pattern and Luna's p92-06 solo yes.

## Minor findings

1. Replaced “mostly” with ten per-excerpt readings and both readers' coherence labels; four excerpts are identified as plain first-person topic or role claims.
2. Stated the lowest-question-index selection rule, included each selected question, removed the duplicate p92-40 excerpt, and added exact selected excerpts to NUMBERS.md.
3. Reported max_tokens 4,000 for both Part 92b readers, DS's ten first-attempt failures, their SSL and token-cap causes, and their recovery on retry.
4. Added 2,000-resample, seed-1 bootstrap intervals and by-arm kappas; described random-arm no-agreement dominance and identified neuron-arm kappa as relevant to unit kinds.
5. Added a short Terms section for the study's model, reader names, prompt labels, arms, kinds, push, dose, Gate A, Gate B and greedy decoding.
6. Named all 14 units omitted from the plot in the README caption and the regenerated figure.
7. Expanded Files to list reader labels and sidecars, sample, Gate A key and labels, labeller, launch record and supporting sources.
8. Replaced the inaccurate 92b.4 gloss by copying all five current Part 92b prediction rows verbatim from PREDICTIONS.md.
9. Removed the unregistered three-level account from the write-up.
10. Replaced “Cold check: pending” with the required round-one outcome line.
11. Left repository tracked/untracked and commit state out of the README; the brief forbids commits, and this is working-tree metadata rather than a finding about the write-up.
12. Added the Gate A and Gate B procedures, blinding, registration and amendment times, registered effect target, and the two Haiku error rows.
13. Corrected the route description: Luna used judge_client's cliproxy route, the same route calib86.py uses; the design named the script, while this runner calls the route directly.
