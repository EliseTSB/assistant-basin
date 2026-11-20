## Amendment 2 (DRAFT, 2026-11-20 14:32 CT, Journal Claude; to be appended to PART214-DESIGN.md only on Gemini 4 Neon's ruling, before any Stage 1 readout exists)

**What happened.** The first Stage 1 attempt (2026-11-20 14:13 CT) stopped at 14:20:46 CT on
`nat_prompt_fact_gen_you: no recorded donor token ids for question 0 (bank None)`. The two cross-phase arms had no
rule for which bank's recorded opening tokens they replay. No stage1.json was written. Stage 0 (sha 432d9987…), the
registration (public main 8d5c2db4) and every prediction are unchanged; the stopped attempt's logs are kept.

**The rule.** A cross arm's Stage 1 opening carries the first five recorded donor token ids of its **GEN** bank:
`nat_prompt_fact_gen_you` replays d_you[qi]'s, `nat_prompt_you_gen_fact` replays d_fact[qi]'s. The recipient
en_maker prompt and the scoring index are unchanged. Reason: the imported decode activations were recorded on that
donor's token history, so they stay paired with it. This is a GEN-history convention on a composite Stage 1
diagnostic, not a pure decode-only intervention: Stage 1 teacher-forces all five opening tokens, and in Stage 2 token
0 is predicted at prefill. The rule is fixed here before any Stage 1 data; no opening is chosen by outcome.

**What "exact 2×2 phase cross" (design line 278) means under this rule.** At a fixed GEN bank, each cross arm and its
matching pure arm (`nat_prompt_fact_gen_you` vs `nat_trace_you`; `nat_prompt_you_gen_fact` vs
`nat_trace_fact`) hold the opening tokens and the decode donor fixed and differ only in the prompt activation bank:
that contrast is exact. Across GEN banks the five-token history changes on 25 of 25 saved questions (token 0 on 24 of
25), so those contrasts are activation plus history, not activation-only factorial effects. An activation-only
estimand would need one common fixed opening for all four corners, which is a separate design and not run here.

**Unaffected.** 214.5 is read on Stage 2 free generation (design lines 225-242, 367); it needs no forced opening, and
its formula and cutoffs are unchanged. No other arm's opening changes.

**Mechanics.** `OPENING_BANK` (opening tokens) is kept separate from `NAT_BANK`/`needed_donor_banks`
(activation reads), so the crosses still load both banks. Stage 1 now resolves every planned arm's opening and donor
lookups for all 25 questions on the Stage 0 bytes **before** the model loads, and writes per-arm opening provenance
(group, opening bank, activation banks, per-question token ids) into stage1.json.

**Pins.** Appending this amendment changes the design file's hash, so `DESIGN_SHA256` and the calibration receipt's
publication design hash move to the amended file, with the receipt re-pinned and re-checked; the registered
PREDICTIONS.md section is unchanged.
