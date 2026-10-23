# Retired reserve and actual-cost overrun fold

## SUMMARY
Retired unknown-outcome attempts remain counted at their full reserve as spent in aggregate and per-kind totals. Reserve-time sub-cap checks count retired attempts while holding the ledger lock. Part 63 reports the same amount in `spend_state`, including after creating a new reader process. Settlement above its reservation now durably records the actual billed amount as an overrun; that amount enters aggregate and sub-cap totals and constrains later reservations. Part 63 retry collection and retirement accept and consistently use an explicit ledger path.

## FINDINGS
- `release` remains the only transition that frees reserved headroom: it still requires a non-429 HTTP 4xx and a valid Anthropic error object.
- Four retired $1.12 attempts survive a Ledger restart as $4.48 counted spend. A further $3.52 reserve reaches the $8 aggregate exactly; another cent is refused. Two retired $1.12 Part 63 attempts exhaust the $3 sub-cap enough that a third $1.12 reserve is refused.
- Four $1.12 retires total $4.48: that exceeds the $3 Part 63 sub-cap, so four cannot be accepted under that sub-cap; it is still below the $8 aggregate. Tests demonstrate the $3 boundary with two $1.12 retires then a refused third, and the $8 boundary with four retires plus a $3.52 reserve and refused excess. A separate eight $1.00 retire case also proves the aggregate cap boundary.
- Copied production-code mutations that omit retired amounts from aggregate summary accounting go red. Existing copied-code mutations and all paidaction tests remain green.
- Part 63 kill/recovery coverage remains green. A Part 63 test verifies `spend_state` returns the retired reserve as effective spend and an explicit ledger path remains in use through automatic retry collection and retirement.
- Part 62 uses the shared library's reserve and release interfaces. Its source and tests contain no dependency on retirement freeing headroom.
- Parent-control message seq 1 was applied and disposed as `considered`: both requested additions (settlement overrun persistence and explicit retry/retire ledger-path threading) were implemented.

## BLOCKERS
Part 62 suite execution did not pass in this worker environment: its tests attempted to create `/home/elisepc/assistant-basin/part62/.test-ledgers`, which returned a read-only filesystem error, and three phase-hook tests require the unavailable `torch` module. No Part 62 source was changed for this task.

## UNVERIFIED
No provider API call or real charge was made. Existing legacy `settlement_overrun` diagnostic events retain their historical active-reservation interpretation when they lack the new `final: true` marker; new overruns are final and counted at the actual amount. The stated Part 63 assumptions about pricing and provider usage remain inherited and were not reviewed here.
