# Part 172 second cold line read

Repo: `/home/elisepc/green/worktrees/ab-part67`. Read: `PART172-DESIGN.md` (revised), `results/part172/COLD-BUILD.md` (first read, 19 findings), `part172/run172.py`, `score172.py`, `label172.py`, `prompts172.py`, `steps.txt`, `probe172.py` (imports), and the `run171.py` functions these call. Nothing loaded the model; no reader was called; `test_part172.py` was not run. Torch and numpy are not installed in this sandbox, so the hook and push code was read, not executed. Plan counts were checked by importing `run172` and calling `arm_specs`, `gate1_plan`, `planned_keys` and `gate6`. Line numbers are from the files as read.

## A. First-read findings (item 1)

1. **Sets from the wrong folder. FIXED.** `run172.py:36`, `:1040` now read `results/part171/sets_google.pt`. The preflight runs before the model load: `run172.py:1055` checks, `:1060` loads. The `weights_only` fallback is at `:200-204`.
2. **Gate 3 `{"set": "none"}` KeyError. FIXED.** `run172.py:939-940` builds explicit zero tensors for each layer; `set_pushes` is not called.
3. **Gate 3 second check was a real knockout. FIXED.** `run172.py:943` `knockout_hooks(layers, GATE3_UNITS, 1.0)`, with registration checked at `:944`.
4. **Gate 6 width mismatch. FIXED.** `run172.py:753-754` sums traces at hidden width (`actual.shape[-1]`). Pushes keep input width (`widths_of`, `:300`), which matches the input check in `run171.py:301-307`.
5. **Singles ignored dose. FIXED.** `run172.py:407` passes the spec dose; `:257` multiplies it in. Read: `n2_d0.5` and `n2_d1` now differ only by the dose factor.
6. **Haiku read an empty set; wrong population. FIXED.** Ids are `arm|prompt_id` in `score172.py:736` and `label172.py:146`. `score172.py:772` draws 10% of all eligible clean neither/target/both items. The reader join is still missing (finding 2).
7. **Hook removal unchecked. MOSTLY FIXED.** `run172.py:426-428` raises if any pre-hook is left after every arm; `:975` checks counts in Gate 3. `measure_gate6` (`:668-744`) is not counted (minor).
8. **Runner-up read the Qwen token's successor. FIXED.** `run172.py:824` stops the prefix before the family token; `:829` uses `logits_to_keep=1`.
9. **Frame arms identified themselves to the reader. FIXED on the reader path.** `label172.py:147` sends `reader_answer`, which excludes the phrase (`run172.py:383`).
10. **Duplicate control rows. FIXED.** `run172.py:129-130` and `:160-162`. Checked: 54 specs, 54 unique names.
11. **Frame scoring used first 40 and 12 words. FIXED.** `run172.py:380-381` and `score172.py:70-76` (12 tokens decoded, word window inside).
12. **Removed mass not recorded. FIXED.** `run172.py:838`, `:852`.
13. **Gate 1 only after the ladder. PARTLY.** The plan check is now before any generation (`run172.py:908-911`). Answer completeness is still at the end (`:1021-1027`).
14. **Kappa cut-off not applied. NOT FIXED.** `label172.py:177` computes `read`; nothing reads it. See finding 2.
15. **Known-example floor. Not a defect.** Design line 42 says "must get at least 9 right"; `label172.py:33` has `KNOWN_PASS = 9`. My first read was wrong on this point.
16. **Reader coverage not reported. FIXED in label output.** `label172.py:229-234` and `:184-185` count failures. Not carried into `score.json` (finding 2).
17. **Host memory for 21 weight matrices. FIXED.** `run172.py:1068-1070` copies only the four needed columns.
18. **Host CPU loops. NOT FIXED, CPU only.** `run172.py:771-774` decodes the whole vocabulary; the runner-up list comprehensions at `:834` and `:840` loop over it per prompt. No `.item()`, no device top-k, no reload.
19. **Reporting gaps. MOSTLY FIXED.** Exit is non-zero when readings stop (`score172.py:795-796`); the dose-axis SVG is written (`:766`); `qwen_retained` divides by clean answers (`:133-134`). Open: `bootstrap_outcomes` count (finding 7).

## B. Findings

**1. BLOCKER. A wrong or empty push can run as a silent null arm, and ten control sets are never checked.**
- `run172.py:219-222`: `if not rep: continue`. `set_pushes` (`:211-235`) returns `{}` with no error when none of an existing arm's layers match `widths`. That arm then runs as an unpushed arm.
- The ten control sets outside the ladder are not replayed (`run172.py:64`, `REPLAY_ARMS = LADDER`) and have no delivery check (`:547-551` and `:696-703` cover only the ladder, the n1 negatives and the singles). They enter the pooled twelve at `score172.py:163-175`. A silent null control would pull the pooled GS and strict share toward the baseline with no gate failing.
- The design calls a wrong key that silently pushes nothing the worst outcome (design lines 57-61, gates; 36-39 controls).
- Requires: raise when an arm's push is empty or covers fewer layers than `widths`; run the delivery check on all twelve control sets before the ladder.
- Not observed. Torch is not installed here, so this is read from the code, not run.

**2. MAJOR. The reader's labels never reach any output, and the kappa cut-off is never applied.**
- `label172.py:177` computes `read` per question. `label172.py:207-234` returns counts only. `score172.py` never opens `labels-*.jsonl`; the only `claim` there is an id type (`:737`). `score()` (`:743-781`) produces no maker/dodge or claim count.
- The design asks for the claim count beside the word-hit count (design line 42) and for a split with kappa under 0.6 to be reported as unread.
- Requires: join labels by `arm|prompt_id` in `score()`; report claim and maker counts per arm beside the word-hit counts; mark a split unread when kappa is below 0.6.

**3. MAJOR. Gate 6's omitted-dose self-test passes when the probe measures nothing.**
- `run172.py:563-567` requires `not _slope_matches_dose(...)`. `_slope_matches_dose` (`:587-592`) returns False for `None`, so the negation is True.
- Checked by import: with the probe's `delivery_slope` set to `None`, `gate6` returns `passed: True` and `omitted_dose_caught: True`. Gate 6 still fails if every delivery row is `None` (`:557`), so this does not open the gate by itself.
- The design requires that a run with the dose omitted fails this check, tested once before the ladder (design line 62). A `None` probe proves nothing.
- Requires: `omitted_dose_caught` must need a finite probe slope that differs from 0.5.

**4. MINOR. Gate 3's value checks cannot fail by value.** Adding an exact zero and multiplying by 1.0 are identities (`run172.py:939-944`). Gate 3 fails only on a leaked or unregistered hook (`:944`, `:975-978`). Say so in the write-up; it is a wiring check, not evidence about the push.

**5. MINOR. Gate 4's hashes are recorded and never compared** (`run172.py:612-647`, `:984`). A changed pushes file, hook source, tokenizer or snapshot passes Gate 4 if the replayed token ids still match. The design says "records" (line 60), so no gate fails. If these are meant to pin the run, register expected values.

**6. MINOR. Gate 6 checks proportional delivery only.** The reference `w1` is the intended write of the same push at d = 1 (`run172.py:703-717`). A push too large by a constant factor passes at every dose. Absolute size is checked only by Gate 4, for three sets.

**7. MINOR. `bootstrap_outcomes` reports zero-denominator replicates as 0 by construction** (`score172.py:254`). Not-testable replicates are counted under `not_testable` (`:244-252`), but the figure the design asks for is not.

**8. MINOR. 172.5 has two verdicts.** `verdict` uses the bootstrap interval (`score172.py:485-493`); `pass` uses the point estimate (`:496`). Design line 99 says verdicts are point estimates. They can disagree (point 0.6, interval 0.4 to 0.8: `pass` True, `verdict` inconclusive).

**9. MINOR. Empty denominators are recorded as failed, not unread.** 172.2 (`score172.py:408-409`), 172.3 (`:424`), 172.9 (`:543-544`) and 172.11 (`:558-559`) return `pass: False` with `read_status: unread`. The design says "unread, never favourable" (line 99). Not favourable, but the record reads as a failed bar. Use `None`.

**10. MINOR. Target and both answers get only the claim question.** `label172.py:143-146` sends them only as `claim`/`mention`. Design line 43 says they are "labelled the same way, with a second question", which reads as maker/dodge plus claim. Confirm which was meant.

**11. MINOR. Runner-up first-token marking may over-remove.** `run172.py:785-786` marks the first token of every multi-token spelling as family. A common fragment that begins a family spelling leaves the rest distribution, which changes `removed_mass` and `google_share`. Not verified; no tokenizer here.

## Scope

- Gates 1, 2, 5, 7 and the 172.1–172.12 and 172.W1 bars read the rows the design names, with the stated signs and bars. No bar passes by construction. Strict share, GS, family-hit and target-only counts match their definitions.
- `steps.txt`: `--copy` covers every import in the closure traced (`score165`→`triage86`; `score166`→`run165`→`run160`, `run164`, `run155`; `run163`; `run148`; `run153`; `common86`; `judge_client`; `neuron_report`). `--out-dir` is relative and the launch runs from the repo root. Part 171's sets and answers are not copied; `run172.py:1044-1047` checks them before the model load.
- Not run: `test_part172.py`, `tools/launch_arc.py`, any model or reader call.

Verdict: not ready to launch. Finding 1 must be closed before the 2,376-answer run; findings 2 and 3 before the reading.
