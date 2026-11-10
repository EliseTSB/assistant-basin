# Part 201, Amendment 4: the smoke, and the second code review

Written 2026-11-09 19:34 CT, before registration. It lists what changed after `5c4178a` (Amendment 3). The plan's
sha256 is unchanged (`94dc3fbd…`), and no prediction or bar changed.

## The smoke (commit ee17381)

The runner's whole path ran on Qwen3.5-0.8B on the host CPU and was scored. What it showed, and the pair counts
that were seen before registration, are in `part201/smoke08/RESULT.md`; the design lists them under "Our own data".

One wording fix came out of reading its report: for a smoke plan the scorer said the snapshot check held, when that
check is not made on a smoke. It now says "not checked (smoke plan)" and names the snapshot loaded.

## The second DeepSeek review (of 5c4178a; four findings, all in code Amendment 3 introduced)

The review is `reviews/cheap-part201-build-ds-r2.md` in the private notes repo; its verdict was ship with fixes. It
checked the Amendment 3 corrections on cases it drove itself and found them sound, then measured four faults.

| # | Fault it measured | What changed |
|---|---|---|
| F1 | A G4 failure in the bf16 process only made the ordered decision say "Float32 arm invalid" (rule 2). The float32 arm was intact. | The decision reads G4's float32 half, like its G1 and G3 neighbours. A bf16-only failure now ends at rule 4, with the bf16 readings withheld. |
| F2 | A load record with no revision and no snapshot on any launch passed G1: two absences compared equal. | A launch must record both, and they must agree. The kernel record must name the revision too. |
| F3 | For a pushed call, zero hook firings against zero decode forwards counted as "one firing per forward". | A pushed call with no decode forward is its own case (next row) and is never a pass. |
| F4 | A call whose rows all end at the first generated token has no decode forward. The runner stopped the launch and called it a wrong push and a wrong dtype. | The runner still stops, and now says what happened: the push was not exercised. This is a limit, written in the design: the first token is decoded before the push applies, as in Part 163's generator. No stored answer of Part 163 or Part 200 (3,416 rows each) is empty, so it has not happened on these questions. |

I kept F4 as a stop instead of writing such rows: a row that was never pushed must not be stored as a pushed row,
and a way to carry it through the gates would be new code in the runner and the scorer for a case the stored
answers say does not occur.

The review's one caveat stands and is not fixed: `load.json` is replaced atomically but not flushed to disk, so a
power loss (not a process crash) can leave it empty, and the next launch would fail on reading it.

## Evidence

Tests on the host: 104 pass (recorder 5, runner 42, scorer 46, shape floor 11), with
`python -I -m pytest -q tools/test_margin_recorder.py part201/test_part201.py part201/test_score201.py tools/test_shape_floor.py`.

Sabotage check on this state: 33 of 33 sabotages turned every named test red (`python -I part201/sabotage201.py`).
Its first run on these fixes came back 31 of 33, and both misses were its own to find: one new sabotage left its
test green because a second guard caught the same case (the test now isolates the revision rule on a smoke plan,
where the snapshot comparison is not made), and one older sabotage no longer matched the line it breaks.
