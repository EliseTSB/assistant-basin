# Part 87 fold 2 notes

## SUMMARY

- Sol #1: corrected Windows `steps.txt` invocation, CRLF/resume/stop behavior, 12 GB memory precondition, and runner launch instructions.
- Replay steps: added pilot positive and negative controls in order, full-run replay after audit, and copied replay dependencies.
- Sol #2: scoring now derives registered index and kind from question text for both studies; mismatch returns NOT READ with a nonzero exit. Added the six-maker/six-guard N11 regression.
- Sol #4: generation and measurement verify the locally cached pinned snapshot before loading. Resolved revision is recorded in dose, phase, and arm-manifest records; tiny CPU fixtures require an explicit test-only allowance.
- Sol #6: combined blind-key scoring checks both answer digests. `blind-score` now requires both answer directories; changed 87B data is refused.

## FINDINGS

No disagreement with Sol's scoped findings was found. Sol's round-2 findings #3 (observer boundary checks) and #5 (independent measurement verification and source identity) remain outside this assignment and were not changed. The requested fixes do not establish those broader checks.

## BLOCKERS

None.

## TEST AND MUTATION SUMMARIES

Exact `part87/test_part87.py` summary:

~~~text
Ran 26 tests in 8.753s
OK
~~~

Exact `part87/mutate87.py` summary:

~~~text
MUTATE87: 18/18 killed; all killed
~~~

The added mutations were killed for question-metadata trust (N11), an unpinned model-load argument (N13), and omission of the 87B blind-key digest check (scoring probe).
