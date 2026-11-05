# COLD check — Part 188 README vs results.json, PART188-DESIGN.md, part188/analyze188.py

Checked 2026-11-04. Sources: `results/part188/README.md` (60 lines), `results/part188/results.json`,
`results/part188/summary.json`, `PART188-DESIGN.md` (incl. Amendments 1–2), `part188/analyze188.py` at HEAD,
`part188/manifest188.csv`, and git objects d479df9, 069f566, 0388f8e.

## Number-by-number

All README numbers were re-read from results.json or recomputed. Every one is a MATCH.

| README | README value | results.json | Status |
|---|---|---|---|
| L4 clip count | 3,138 | n_planned_rows 3138; manifest 3138 rows | MATCH |
| L5 folds | 19 groups, A and B, Chicken held out | manifest: 10 groups in A, 9 in B, Chicken in C | MATCH |
| L8 units | L13 N773 / L5 N1770 | layer 13 neuron 773 (idx 27397) / layer 5 neuron 1770 (idx 12010) | MATCH |
| L8–9 held-out AUC | 0.753 / 0.897 | 0.75336 / 0.89664 | MATCH |
| L11 probe | 0.918 / 0.965 | 0.91806 / 0.96508 | MATCH |
| L12 top-32 | 0.905 / 0.957 | 0.90520 / 0.95725 | MATCH |
| L17 Gate 3 | unit 31846, 28 other classes, 0.982 | gate3 unit_index 31846, n_negative 1120 (set E), auc 0.98247, passed | MATCH |
| L18 Gate 2 | each set 3,138 of 3,138 | 7 sets summing to 1120+678+533+400+269+100+38 = 3138, each fraction 1.0 | MATCH |
| L27 188.1 | 0.631 / 0.808, failed | 0.63051 / 0.80753, failed | MATCH |
| L28 188.2 | 0.753 / 0.897, failed | same, failed | MATCH |
| L29 188.3 | +0.123 / +0.089, failed | 0.122853 / 0.089114, failed | MATCH (recomputed) |
| L30 188.4 | 0.281 / 0.677, failed | 0.28093 / 0.67681, failed | MATCH |
| L31 188.5 | 0.647, held | 0.64739, held | MATCH |
| L32 188.6 | 0.918 / 0.965, held | same, held | MATCH |
| L33 188.7 | gap 0.152 / 0.061, failed | 0.151841 / 0.060609, failed | MATCH (recomputed) |
| L34 188.8 | 0.983 / 0.913, held | 0.98276 / 0.91276, held | MATCH |
| L35 188.9 | 0.632 / 0.634, held | 0.63185 / 0.63450, held | MATCH |
| L39–40 real vs human | 0.281 / 0.677 | real_vs_human 0.28093 / 0.67681 | MATCH |
| L41 vocal-only | 0.340 / 0.606 | real_vs_human_vocal 0.33958 / 0.60580 | MATCH |
| L42 verdict B→A | 0.808 | model_class_informed B_to_A 0.80753 | MATCH |
| L44–45 probe | "0.92 and 0.97" | 0.918 / 0.965 | MATCH (rounded) |
| L46 chicken | 0.983 / 0.913 | same | MATCH |
| L48 length | 0.63 | 0.632 / 0.634 | MATCH |
| L52 interval | 0.64 to 0.88 | A_to_B heldout group_clustered_95 [0.64485, 0.88065] | MATCH |
| L52 folds | 9 and 10 groups | 9 (B) and 10 (A) | MATCH (order A=10, B=9; README lists "9 and 10", harmless) |

All nine prediction verdicts in the README table agree with `predictions` in results.json, and each threshold in the
README table is the one in PART188-DESIGN.md "Predictions" (188.1 <0.75 both; 188.2 ≥0.85 both; 188.3 ≥0.10 above;
188.4 ≥0.80; 188.5 <0.70; 188.6 ≥0.90; 188.7 top-32 minus unit <0.05; 188.8 ≥0.80 both units; 188.9 <0.70).

## Findings (claims, Gate 2 account, rule logic)

1. **README L18–21, Gate 2 account — accurate.** Pre-fix code at d479df9 had
   `set_names = sorted({...} | {"P"})`; 069f566 removed `| {"P"}`. results.json gate2 does contain P (planned 0,
   recorded 0, fraction 0.0, verdict failed) and overall verdict "failed"; summary.json agrees. README's account is
   correct, including that the run's own verdict says "failed".
2. **README L20–21, "Re-applying `_headline_rule` with all three gates passed still withholds the headline" — true
   but not demonstrated in results.json.** `_headline_rule` (analyze188.py:323–342) appends "prediction 188.2 failed"
   and "prediction 188.3 failed" independently of gates, so gates all passed ⇒ still `headline_permitted=False`. The
   stored results.json `reading_rule` never shows this: it was produced with gate2 "failed" and its
   `withdrawal_reasons` are ["Gate 2 did not pass", "prediction 188.2 failed", "prediction 188.3 failed"]
   (`headline_permitted: false`). The README's claim is a re-derivation, not a stored artefact; it is correct, but a
   reader cannot verify it from results.json alone. Reading rule in DESIGN ("withdrawn if ... 188.2 or 188.3 fails")
   is consistent.
3. **README L18 "every real set recorded 3,138 of 3,138" — slightly loose.** 3,138 is the total, not "each set";
   each set is 100% of its own planned count (e.g. R 678/678). Minor wording, not a number error.
4. **README L17, Gate 3 "hen and rooster against 28 other classes" — MATCH**, but note the design says the earlier
   study found 0.94 "against the other ESC-50 classes"; here 0.982 is against set E (1,120 clips). README is right
   about this run.
5. **README L9 "the first is weak" — fair.** A→B 0.753 with group-clustered CI [0.645, 0.881] touches near 0.85 but
   is below the 0.85 bar; "weak" is a defensible description, not an overclaim.
6. **README L39–40 "The unit partly reads 'a human made this'." — MECHANISM CLAIM, over-reads.** The data show the
   A→B unit scores real animals against ESC-50 human sounds at 0.281, i.e. it places human sounds on the "real" side;
   B→A is 0.677. That is a direction-specific pattern, and the two units disagree (one below 0.5, one near 0.7). "A
   unit that only says 'a person is making this sound' would give exactly this" is a hypothesis, not a measurement.
   The README's own caveat (L54–55) partly corrects it, but the headline sentence in §1 still reads as a finding.
   The registered 188.4 ("it reads 'a human made this'") FAILED, and the design's reading rule ("If 188.4 holds, the
   result is reported as a unit that reads human-made sound") is not triggered; the README should say "consistent
   with" rather than "the unit partly reads".
7. **README L41 "The vocal-only subset is no better (0.340, 0.606)." — MATCH, but "no better" is ambiguous.** 0.340
   vs 0.281 (A→B) is slightly *higher*; 0.606 vs 0.677 (B→A) is lower. "No better" holds only for B→A; for A→B the
   vocal-only value moved toward 0.5. Not a number error; a wording issue.
8. **README L43 "The earlier chicken result, where the verdict sat near chance, does not generalise across animals."
   — CROSS-PART CLAIM, not checked by this part.** The earlier chicken study is cited in the design as "Journal, not
   yet public"; results.json contains no chicken verdict number, so "near chance" cannot be verified here. Also
   "does not generalise" compares one past study to this one; the data only show this run's verdict is 0.631/0.808.
9. **README L45 "The output embedding carries the signal ... a trained layer or a gated edit on several units is the
   route; a one-matrix edit on one unit is not supported by this part." — INFERENCE, not measured.** The probe result
   (188.6 held) shows the embedding separates the classes; it does not show that a gated edit on several units works
   or that a one-unit edit fails. "Not supported by this part" is honest; "is the route" is a recommendation beyond
   the data.
10. **README L47 "Chicken is the special case ... Why was not tested here." — fine.** 188.8 held on both units;
    the README correctly declines to explain. Note the chicken AUCs (0.983/0.913) are on 138 clips (C_R 100, C_I 38)
    and the clustered CI for B→A is [0.829, 0.974] — still above 0.80 but the README does not report it; the design
    says clustered intervals are "the honest one" and are reported beside predictions. Missing, not wrong.
11. **README L48 "Length is not the explanation (188.9): sequence length alone gives 0.63." — MATCH and rule-consistent.**
    `length_within_0_05_of_selected` is false in both directions (0.632 vs 0.753−0.05=0.703; 0.634 vs 0.897−0.05=0.847).
12. **README L52 "Folds are 9 and 10 animal groups" — MATCH** (B=9, A=10). "a fold's unit is chosen on a handful of
    animals" is fair.
13. **README L54–55 caveat on ESC-50 human sounds — good**, and it correctly limits the 0.28 reading. This is the
    sentence that should govern §1 (finding 6).
14. **README L56 "Cold check pending at writing" — now stale** (this file is that check). Not an error at writing.
15. **README L58–59 files and commits — MATCH.** 0388f8e exists ("part188 r2: final results"), d479df9 exists (review
    fold), 069f566 exists (Amendment 2). results.json's `directions`/`chicken`/`gate*` structure is as described.
    One nit: the README says results come "from the cloud branch `cloud/part188-r2`, 0388f8e" — correct.

## Things checked and found clean

- `_prediction_block` computes 188.3 as unit − class-informed and 188.7 as top-32 − unit; the README's +0.123/+0.089
  and 0.152/0.061 reproduce exactly.
- `prediction_verdict` uses `all(...)` over both folds for 188.1–188.4, 188.6–188.9 and a single pooled value for
  188.5; README verdicts follow.
- Headline rule in the README ("not permitted ... whatever Gate 2 says") matches `_headline_rule` and the design's
  reading rule.
- No prediction verdict in the README differs from results.json or from the design thresholds.

SUMMARY: Every number in the README matches results.json and the design thresholds, the Gate 2 phantom-P account is
accurate, and the headline-withheld conclusion is rule-correct; the only overclaims are interpretive mechanism
sentences in "What the numbers say" (§1 "the unit reads 'a human made this'", §3 "is the route", §2's cross-study
"does not generalise") that go beyond what 188.4-failed and a single probe can show.
