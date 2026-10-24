# Brief: Part 65 judge fold after review round 4 (code and tests only)

Domain: /home/elisepc/green/worktrees/ab-part65/part65 (write only there). The review is
`/home/elisepc/green/Journal/reviews/cheap-p65-build-haiku-r4.md`; the contract is Amendment 4 at the end of
`PART65-DESIGN.md`. Do not commit, run on the GPU, call a paid API, or change `steer65.py` or
`paidaction/paid_ledger.py`. Python: `/home/elisepc/neuron-probe/.venv/bin/python`, suite from `part65/` with
`PYTHONPATH=/home/elisepc/green/worktrees/ab-part65`. The existing 44 tests stay green.

Fix exactly these findings, each with a test written from the property (not from the code), and name the mutant each test
catches. Run each mutant if you can and say which you ran.

- **F1.** The judge writes a completion marker (in `run.json` or a sibling file) only when every chunk is settled with its
  answers saved; a released or ambiguous outcome writes a failure marker. `analyze65` refuses unless the completion marker
  is present and matches the answers' run id and plan hash. Record `chunk_size` and the chunk count in `run.json`.
- **F2.** Remove the `--ledger` command-line option. The live ledger path is a module constant. Tests pass a ledger
  object or path through function arguments only. Nothing reachable from `main` can pick a different ledger.
- **F3.** Make the lock test deterministic: the first run blocks inside the critical section (an event or barrier in a
  wrapped `_prior_reservation`), the second run starts, and the test asserts the second one waits for the lock. It must
  fail every time with the lock replaced by a no-op; show that over at least 10 runs.
- **F4.** One test through the real `Ledger` with a small sub-cap: the refused chunk gets no POST.
- **F5.** A ledger cap refusal inside `run_chunks` becomes a `JudgeError` naming the sub-cap, stops the loop and makes
  `main` exit 2.
- **F7.** `_persist_answers` dedupes on (run id, custom id) and refuses to append rows whose run id differs from rows
  already in the file.
- **F8.** Check that an API key is present before the first reservation; with no key, refuse before any reserve.
- **F10.** Before running, check the chunk files are exactly indices 0..n-1 and their trials sum to the manifest's
  `request_count`; otherwise refuse.

Do not touch F6, F9, F11 or F12; say in the report that they were left, as the review allows.

Report: `part65/REPORT-fold-r4.md` (SUMMARY / FINDINGS one line per fix with its test and mutant / BLOCKERS).
