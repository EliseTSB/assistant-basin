# Part 219 Amendment 1 build report (mean-ablation)

Brief: `part219/BRIEF-build-a1.md`, against `PART219-DESIGN.md`'s "## Amendment 1". Nothing in the design,
`PREDICTIONS.md` or any threshold was changed. No commit.

## What shipped

**`part99/swap99.py` (additive; every default byte-identical).**
`--ablate-mode {zero,mean}` (default `zero`) and `--head-means <path>` (required with `mean`). In `mean` mode a
forward pre-hook on the model records the prefill call's `attention_mask` and each row's first real index (the mask
grows during `generate`, so only a call whose input length equals the mask width refreshes it); the `o_proj` pre-hook
then writes `mean_pos0` at each row's own first real prefill token, `mean_rest` at every other prefill position
(pads included) and `mean_rest` at every decode call. Means are stored float32 and cast to the model dtype once per
call. Helpers added: `load_head_means`, `file_sha256`, `ablation_receipt_problems`, `ABLATE_MODES`.
The receipt (first batch) records, per head, exact byte equality with `mean_pos0` at every row's first real prefill
token, with `mean_rest` at every other real prefill position, and with `mean_rest` at the first decode step, each with
`max_abs_diff_before > 0`. The means file's SHA-256 goes into the run's meta (`head_means_sha256`, `head_means`) and
into the intervention receipt. An empty head set installs the same hooks and changes nothing (R0b's mean half). A mean
run with no recorded decode step (a generation whose rows all stop on their first token, and every resumed run) takes
a two-token probe generate with `min_new_tokens=2`, so the receipt is complete rather than missing one of its three
checks.

**`part219/means219.py` (new).** Builds `results/part219/head-means.pt`: screen A's 24 `general` questions (the path
is an argument, and its SHA-256 must start `1eabf3aba5824428`, which was checked against the real file:
`1eabf3aba5824428431a25f399d4d8d59b09c171fc6c7370d455ccf4e7887af4`) plus `swap99.QUESTIONS[4]` and `[5]`, in Part 99's
chat frame, answered greedily by the unedited host with Part 99's 120-token cap and `common92.generate_batch`, then
teacher-forced back through layers 0-3's `o_proj` capture pre-hooks and summed in float32 into `pos0` (each sequence's
first real token) and `rest` (its later real tokens, pads excluded). Saves the eight heads (the four important and the
four R5 rig-check heads) plus a `meta` block (host, questions SHA-256, the 26 prompts and answers, `n_new_tokens`,
frame, layers, head list, versions). Heartbeat every 8 s.

**`part219/run219.py`.** `R2m`-`R7m` replace `R2`-`R7` in stage a (same heads, `ablate_mode="mean"`); `R2` stays the
reported zero-ablation pilot arm and is run only by `--pilot`. Stage a also runs `R0b` (zero wrapper) and `R0b-mean`
(mean wrapper), both with empty head sets; `run_checks` compares both against R0 token for token. `--means`
(default `results/part219/head-means.pt`). `--pilot-a1` runs R2m only, asserts `R0.jsonl` and `replay-R0.json` exist
and never reruns R0, records the R2m verdict in `manifest["pilot_a1"]` (its own `stages["a1"]` key, so the pilot's
stage record is untouched) and returns 2 on STOP. A main `--stage a` refuses to launch unless that verdict exists and
says `continue`; `pending_labels` gets its own message. `run_one` now asks `swap99.ablation_receipt_problems` about the
receipt (zero records: `all_zero`; mean records: the three equalities), and records `ablate_mode` and
`head_means_sha256` per run.

**`part219/score219.py`.** `219.1m` and `219.2m` computed by `build_predictions_a1` exactly as 219.1/219.2 with R2m in
place of R2 and R5m as the rig check (same 16 upstream units, net 7, 15 of 21, 13 concept, inconclusive flag, |net| ≤ 2).
The amendment's recorded verdicts for the original rows live in `build_pilot_record` / `scorecard["pilot_verdicts"]`:
219.2 failed, 219.1 "inconclusive: zeroing these heads breaks answers", computed from the pilot R2's own labels with
the same `_bars` rule. `build_predictions` uses that block for 219.1/219.2 **only when R5 was never scored** (the
amended design's state); whenever R5 exists the original logic runs unchanged, which is why every pre-existing scorer
test still passes. `R5m_rig_check` joins the controls, and `build_headlines` adds the amendment's mean headline
("the input-varying part of these four heads' output …"), never "these heads are needed". `--pilot-a1` prints the R2m
label criterion alone.

**`part219/steps219-pilot-a1.txt` (new).** means219 first, then `run219.py --pilot-a1`, in `steps219-pilot.txt`'s style.

## Tests

`~/neuron-probe/.venv/bin/python -I -m pytest part219 part99 -q` → **77 passed in 12.2 s** (63 before this brief; 14 new
tests). New tests: the mean hook on a left-padded two-prompt batch (pos0 at each row's own first real token, rest
elsewhere, other head untouched) and at decode steps; empty head set in mean mode identical to no hooks; the loader and
receipt refusals; `means219` against a plain numpy recomputation of the same captured slices; pads-vs-unpadded means;
the reference-file hash assertion; the 26 prompts; `--pilot-a1` refusing without R0.jsonl; stage a refusing without a
`continue` R2m verdict (missing and `stop`); the R2m pilot verdict; and the scorer's amended world (219.1m/219.2m held,
the pilot's 219.2 failed / 219.1 inconclusive, R5m failing freezes both).

Sabotage, shown red once and then restored: the mean hook was changed to write `mean_pos0` at local index 0 of every
call (prefill and decode) instead of each row's own first real token. Output:

```
FAILED part219/test_part219.py::test_mean_ablation_uses_each_rows_first_real_token_and_mean_rest_elsewhere
FAILED part219/test_part219.py::test_mean_ablation_writes_mean_rest_at_decode_steps
>               assert torch.equal(dirty[row, position, start:end], pos0), (layer, head, "pos0", row)
E               AssertionError: (0, 1, 'pos0', 0)
E                +  where False = <built-in method equal ...>(tensor([-50., -51., -52., -53., -54., -55., -56., -57.]),
E                    tensor([100., 101., 102., 103., 104., 105., 106., 107.]))
...
>                   assert torch.equal(dirty[row, 0, start:end], rest), (layer, head, "decode", row)
E                   AssertionError: (0, 1, 'decode', 0)
2 failed, 1 passed, 58 deselected in 2.88s
```

Restored (`grep -c SABOTAGE part99/swap99.py` → 0) and the suite is green again.

End-to-end smoke on a 16-head tiny OLMo 2 (scratch, not a test): `--pilot` ran R0 and R2; `--pilot-a1` ran R2m alone and
was `pending_labels`; stage a then refused; after a resumed `--pilot-a1` the R2m receipt held all three checks; with a
forced `continue`, stage a ran `R0, R0b, R0b-mean, R2m..R7m`, every run readable, `R0b_equals_r0` and
`R0b_mean_equals_r0` both identical on 18/18 items, and every mean reset receipt matched R0.

## Findings

1. **A plain forward does not place a left-padded row where `generate` does.** In transformers 5.20
   `Olmo2Model.forward` defaults `position_ids` to `arange(sequence_length)`; `generate` passes mask-derived positions
   instead. A teacher-forcing forward on a padded batch therefore differs from the same sequence decoded by
   `generate` (measured: ~1e-2 relative on the tiny model). `means219` passes `position_ids = cumsum(mask) - 1` (pads
   pinned to 1) and the padded forward then matches `generate`'s prefill **exactly** (max diff 0.0) and padded means
   match unpadded means exactly. Without this the reference means would have been read at the wrong RoPE positions for
   every padded row. Found because the pads-excluded test failed; it is a trap for anyone else who computes reference
   statistics with a plain forward.
2. **The mean decode receipt needed a probe.** A generation whose rows all stop on their first token never makes a
   decode call, and the first stage-a smoke run showed exactly that for R4m on the tiny model (`R4m` unreadable,
   "has no decode check"). Fixed in `swap99.run`: if a mean receipt still lacks `decode` after the loop, a two-token
   probe generate (`min_new_tokens=2`) is taken on the same first batch. Same probe for resumed runs.
3. **The real pilot labels would fail the amended scorecard's gate, before any mean run.** In `results/part219`,
   `labels-R0.jsonl` has 8 error rows and **3 items still missing** after the retry. `score219.readable_reasons` marks
   R0 "not read" for that, so `build_gate`'s `r0_slot_persona` fails and every prediction — 219.1m/219.2m included —
   comes out "not read: gate failed". This is pre-existing registered behaviour (not touched here), but it means the
   amended scorecard will report nothing until those 3 R0 items are labelled. The `pilot_verdicts` block still records
   the amendment's 219.2 failed / 219.1 inconclusive from R2's labels, which are complete.
4. **`steps219-pilot-a1.txt`'s questions path is repo-relative.** The design writes
   `Journal/notes/reports/attention-heads-screens-2026-11-18/a-olmo-stages/questions.jsonl` relative to the repo, but the Journal is not
   inside this worktree (it is at `/home/elisepc/green/Journal` here), and screen A's file would also need adding to
   `launch_arc.py --copy` if the job folder is to see it. The SHA-256 assertion in `means219` will refuse a wrong file
   rather than read it.
5. **Stage a's run list, decided in the brief's silence.** The brief says "add specs R2m to R7m" while the amendment
   says they *replace* R2-R7. Stage a therefore runs `R0, R0b, R0b-mean, R2m..R7m`; R3-R7 no longer run, and R2 runs
   only under `--pilot` (it is already on disk from the registered pilot). `R0` stays in stage a's spec list, where
   swap99's resume makes it a no-op on existing rows; `--pilot-a1` is the only path that reuses R0 without rerunning it.
6. `test_means219_reference_prompts_are_the_amendments_26` reads screen A's file from its absolute path and skips when
   it is not present, so it is one skipped test on a machine without the Journal.

SUMMARY
Built Amendment 1's mean-ablation end to end: `--ablate-mode mean` + `--head-means` in `swap99.py` (defaults untouched)
with the pos0/rest/decode receipt, `means219.py` producing `head-means.pt` from screen A's 24 general questions plus
Part 99's 4 and 5, R2m-R7m and the R0b mean half plus `--pilot-a1` in `run219.py`, and 219.1m/219.2m plus the recorded
pilot verdicts for 219.1/219.2 in `score219.py`. 77 tests pass (63 before, 14 new); the sabotage test was shown red and
restored; an end-to-end tiny-model run of the pilot, the R2m pilot, the refusals and a full stage a came back clean.
Two real defects were found and fixed on the way: the model's plain forward uses arange positions (so the reference
means needed `position_ids` from the mask to match `generate`), and a mean run can make no decode call at all.

FINDINGS
1. `generate` and a plain forward place left-padded rows at different RoPE positions in transformers 5.20;
   `means219` now passes mask-derived `position_ids` and matches `generate`'s prefill exactly. 2. A generation that
   stops on its first token makes no decode call, so the mean receipt takes a `min_new_tokens=2` probe when needed.
   3. Pre-existing: `labels-R0.jsonl` in `results/part219` has 3 missing items, so the amended scorecard would fail the
   gate and freeze all predictions until they are labelled (R2's labels are complete, so the pilot verdicts stand).
   4. `steps219-pilot-a1.txt`'s questions path is repo-relative and the Journal lives outside this repo. 5. Stage a
   runs R0, R0b, R0b-mean and R2m-R7m (R2 only under `--pilot`), per the amendment's "replace". 6. One new test skips
   without the Journal on disk.

BLOCKERS
none
