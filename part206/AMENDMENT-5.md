# Part 206, Amendment 5: the scorer read 206.M1's null range from the pre-registration text

Written 2026-11-13 14:54 CT by Claude (Opus 5.5), **after the scores were seen**.

## What happened

206.M1 (Muse) registers two ranges in Qwen3.5-4B (chat): `rho_peak(A-tmpl, M)` between 0.07 and 0.30, and
`rho(M2, M)` at the same layer between **0.00 and 0.01**. The table row says the second range was revised from
0.02-0.08 in council #24, before registration; the registered text on public main says 0.00 to 0.01.

`score206.py` still checked the old range, `0.02 <= rn <= 0.08`, so the first scoring run (2026-11-13 14:52 CT)
reported 206.M1 as **held**. Reading the verdicts against the registered table found the mismatch.

## The fix

The scorer now checks `0.00 <= rn <= 0.01`, as registered. No registered text changes. The test fixture's passing
world now uses 0.005, and a new failing case at 0.03 (inside the stale range, outside the registered one) turns red
when the old check is restored on a copy of the folder (checked: the copy fails on 206.M1, the fixed file passes).

## Effect on the verdicts

One verdict changes: **206.M1, held -> failed.** The measured values are `rho_peak(A-tmpl, M) = 0.106` (inside
0.07-0.30) and `rho(M2, M)` at its layer (18) `= 0.0287` (outside 0.00-0.01). Every other verdict is unchanged.

## Not yet checked

The other rows were not re-read against the registered table in this pass beyond 206.M2 to 206.M5. A cold check
against the registered text, row by row, comes before the README.
