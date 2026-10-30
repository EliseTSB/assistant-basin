# Part 86 cold check, round 1

Reader: Claude Haiku 5.5 (a fresh subagent with none of this project's context), 2026-10-30, on `results/part86/README.md`
as first written by a Luna worker. Saved by Claude from the subagent's reply, which it could not write itself. Condensed
only where the reply listed absolute paths. Result: 7 blocking, 13 minor.

## Blocking

1. The random-direction total "41 ... all in layers 21 to 26" contradicts the label file. SCORE-v2's 86.4 line lists only
   layers 21 to 26; the changed random answers at L19 (2) and L20 (1) are never reported. From the label file (1,755
   coherent-persona items) and the 1,713 real persona rows, random persona rows must be at least 42. Fix: report all 17
   layers, restate totals and share, correct the 86.4 row with a dated note, regenerate the figure. (Claude checked: 42
   random coherent personas, 53 still Qwen, 5 not coherent, 100 changed; L19 has 1 persona.)
2. "Mostly coherent personas" is wrong: 41 (now 42) of 100 changed random answers are personas; most of the rest are still
   Qwen or an assistant. Fix the wording here and in the 86.4 row.
3. The headline and opening are one-sided. They name one held prediction and no failure; random directions rise with the
   real neurons in layers 21 to 26 (L24: 3.91% against 5.22%); ratios by layer from the report's counts, one-sided
   Fisher, unadjusted: L21 3.43 (p 0.009), L22 4.63 (0.0006), L23 2.38 (0.002), L24 1.34 (0.11), L25 2.07 (0.06), L26
   4.39 (0.07). The 4.63 rests on 3 random personas (Wald 95% interval about 1.5 to 14; Wilson for 3 of 512, 0.20% to
   1.71%). Amendment 1 says the count and the test carry the claim, not the ratio. Fix: lead with the registered test and
   its interval, add a table of layers 21 to 26, state the failures, retitle the figure.
4. The pilot's gate failures and Amendment 3's triage change are not disclosed (dry run failed gates 2 and 3; Amendment 3
   changed the triage rule after seeing layer 22's counts and moved the screen to batch 64; the rerun passed with 526
   changed). Fix: a protocol-history section, and define kept, broken and changed.
5. The registered triage miss-rate sample (Amendment 2 item 1: 200 kept and 200 broken answers per population, seed 8686,
   labelled blind with the rest) is missing, and the write-up is silent about it. Fix: report it if it exists; if not,
   say in Limits that it was not run.
6. "18 of 335" is 19 on the plain count: 11 and 7 where both readers marked the answer coherent, and 1 where both marked it
   not coherent. Fix: "19 of 335 (18 where both readers marked it coherent)", a NUMBERS command, and a dated note on the
   86.10 row. The verdict is unchanged.
7. The write-up, its artifacts and the registry change are not committed yet.

## Minor

1. Readers and blinding are not described (first reader Haiku 5.5 from the J-lens bundle; Opus 5.5 second read; DS is
   DeepSeek V4.1 Flash, Luna is GPT-6 Luna; blind to layer, neuron and condition).
2. The version 1 to version 2 shift is omitted (on the same 335 items DS went from 180 to 153 personas and 67 to 140
   still Qwen; Luna 149 to 156 and 58 to 133; version-2 labels agree with the first reader at kappa 0.109 and 0.051).
3. The replay result is not reported (`replay-final.json`: 320 answers on all 17 layers, 0 mismatches).
4. Name the clause applied for 86.4 and 86.4a (Amendment 1: fails when any read layer has personas at least as often;
   not read with fewer than 9 read layers; both apply).
5. Undefined terms: push, dose, random direction, residual norm, triage, kept, broken, changed, kind, kappa, gate,
   Fisher exact, one-sided, atlas, units, fresh, versions 1 and 2, the amendments, the later parts, replay check, top-20
   verification, "amended reading".
6. Garbled or vague sentences at lines 7, 19, 27, 30 and 53.
7. "Within changed answers": add the statistic (one-sided Fisher p about 0.011) and the change rates (5.6% against 3.3%),
   labelled descriptive.
8. Skeptic questions not answered: the unsteered canary answer, batch invariance, batch size and the random-direction
   construction, intervals on the figure (L24 intervals overlap), 17 layers scanned and only layer 22 registered.
9. 86.13 measures the reader change, not the neurons (Amendment 8 says so).
10. The atlas section should say how an atlas unit is labelled under version 2, and report the q_name subgroup (323 of
    387; 196 personas, 126 still Qwen) and the v1 check's malformed_labels=3.
11. NUMBERS gaps: no recomputation of 4.630 or 0.000584; no command for 18 or 19, 63%, 99%, 322, 1,391 or the 5; the 41
    came from a zero default in plot86.py; a claim at line 121 has no source; `rg` is not on PATH; line 84 rewrites the
    figure.
12. Cite V2-NOTES.md for "never ran".
13. Replace "Cold check: pending."

## Checked and correct

Design facts; the 15 copied prediction rows; 2.713%, 0.586%, 4.630, p 0.000584, 5.219%, 4.178%, 0.682%, 3.068%, 63%,
99%, 1,713 = 322 + 1,391, the kind mix, 519 and 17, 3,253 and 100, 48.2%, 17.6%, 52.7%; the calibration kappas and
counts; 3,349 unique ids; triage recount; all 14 links; style.

## Could not check

The blind key (not in the repository), the miss-rate labels (none exist), the first reader's labels and atlas list (not
in this worktree), the by-eye reading (no record), whether the four repeated answers are personas, and no model rerun.
