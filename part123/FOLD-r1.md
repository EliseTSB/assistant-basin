# Part 123 fold review round 1

- Finding 1: BUILD-NOTES gives Stage C its own fresh `main` output and excludes pilot files; the end-to-end stub test checks a fresh run and resume from that main folder.
- Finding 2: the scorer gates persona readings on readable marks and reports persona and coherent-persona counts; the synthetic scorer test and readability mutation probe cover this.
- Finding 3: Amendment 1 settles persona visibility; no code change was needed, and the end-to-end labeler test still exercises the reader main.
- Finding 4: BUILD-NOTES says reader and scorer run on the Linux repo root and points both at the main output; the end-to-end test drives both mains.
- Finding 5: a scorer comment clarifies that the three-of-six rule covers concepts while the question 0/1 restriction covers slots; the synthetic scorer test covers scoring.
- Finding 6 / Amendment 1 item 2: the runner checks same-batch-shape plain and zero-hook IDs for every layer and question before pushes; stub tests cover both a passing zero hook and a mismatch stop, and BUILD-NOTES gives the amended size meaning.
- Finding 7: no code change was needed because the scorer checks Luna's current sample keys; the gate 6 mutation test covers stale sample membership.
- Finding 8: Part 123 readers log and skip torn tails without truncating in place; the JSONL test checks unchanged inputs and raises on nonfinal damage.

The suite's copy-only red probe breaks the zero-hook input and readable-persona labels. Both tests that expect those broken copies to pass fail, as intended.
