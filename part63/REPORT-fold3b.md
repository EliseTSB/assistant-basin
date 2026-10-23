# Part 63 fold 3b report

## SUMMARY

- Implemented run-all recovery for all four requested SIGKILL boundaries.
- A durable POST response ID is recovered and polled without another POST.
- A missing ID remains non-adoptable; run-all reports the chunk state and operator retirement/rerun commands.
- Explicit retirement records `retired-unknown` on the chunk and permits one new reserved attempt under the cap.
- Offline suite: 21 tests passed; all 12 copied-code sabotage mutations went red.
- No live API request, network access, GPU, paid run, commit, or paidaction edit was made.

## FINDINGS

| Item | Disposition | Evidence |
|---|---|---|
| Kill after reserve before submission metadata update | Fixed | SIGKILL test leaves the ready submission and ledger reserve; run-all recovers the reserve and continues without an extra reserve. |
| Kill after posting status before POST | Fixed | Rerun classifies the POST-window state as unknown and refuses with an exact retire command. After explicit retirement, it records `retired-unknown` and retries under the cap. |
| Kill after accepted POST before response saved | Fixed | No listing adoption is performed. The reserve remains spent after retirement, and a fresh attempt reserves again. Test checks total attempts stay at most one above the normal two-chunk run. |
| Kill after raw response saved before submission update | Fixed | Rerun recovers the saved batch ID, polls and collects it, and posts only the next chunk. The fake transport asserts the next POST starts at request `r00200`, with no duplicate POST for the saved-ID chunk. |
| Retire path regression protection | Added | `skip-retire` copied-code mutation makes the SIGKILL recovery test fail. |
| Other fold 3 items | Kept as-is | No work beyond the open run-all recovery and requested tests was changed. |

The decision to keep the no-ID attempt non-adoptable came from the binding parent answer in the brief and matches the DISPOSED section of papercut 004739. The referenced domain `HOLD.request` was not present at the stated path. The parent-control mailbox was empty at each required checkpoint; no messages changed, were declined, or deferred.

## TESTS

- `python3 -m unittest` from `part63/`: **21 passed**.
- The suite ran SIGKILL subprocesses at all four boundaries, reran each state to settled, exercised operator retirement for no-ID cases, checked the saved-ID case performs no second POST for that chunk, and verified no more than one extra reserve after retirement.
- All 12 copied-code sabotage mutations went red, including the new retire-path mutation.
- `python3 -m py_compile reader63.py test_part63.py` passed.
- `git diff --check` passed.

## BLOCKERS

None.

## UNVERIFIED ASSUMPTIONS

- Provider behavior, provider billing, and actual batch lifecycle remain unverified because no API or network call was allowed.
- `retire` continues to use the configured default shared ledger in normal CLI operation; tests passed an isolated explicit ledger to run-all and set that same ledger for retirement.
