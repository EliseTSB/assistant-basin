# Build brief: Part 43, amendment 3 (Luna)

Run size, multiplied out: code only here; the new forward-pass sections add about 6,000 short passes on the A770 later.
part43.py (committed, built to amendments 1, 1b, 2; read REPORT-43-build.md) now needs Amendment 3 in PART43-DESIGN.md
and rows 43.11, 43.12 in PREDICTIONS.md (the design and REVIEW at review/part43/REVIEW-codex.md explain each item):
(1) `swap`: the four worlds A/B/C/D for an upstream U and downstream V (L7->L11 first; also L11->L29, L7->L29), V's
whole aligned trajectory restored (C) or transplanted from B (D); (2) record L11's gate, up and product in A and B and
transplant the two recombined products (clean gate x altered up, altered gate x clean up); (3) `attach`: write
data/part43-attach.json (both attachments, both orders, a held-out second wording) and measure L11 after the
search-engine phrase; (4) total derivative by finite perturbation of L7 beside the existing bypass derivative;
(5) the grid records realized activations and clipping, plus a second grid at fixed absolute levels taken from clean
trajectories (43.10 is scored on it, with a minimum slope magnitude and sign handling); (6) additive surrogate from
single-ablation final residuals, h_add(S) = h_clean + sum(h_single_i - h_clean), frozen and real final norm, plus the
raw " Google" minus " Alibaba" logit difference; (7) the eight competitor makers scored separately; (8) EN/ES/ZH
maker-question families for the swap; (9) third-person "Who created Qwen?" and company-country prompts;
(10) bootstrap over template clusters only. `score` fills 43.11 and 43.12 and applies the row clarifications in the
amendment. Add `swap` and `attach` to `pilot`'s timing report. Tests: tiny-model tests for every new path, including a
synthetic case where C = B and D = A (independent branches) and one where they differ. Keep all existing tests green.
Commit nothing. Report REPORT-43-amend3.md ending with SUMMARY / FINDINGS / BLOCKERS, with the A770 commands added.
