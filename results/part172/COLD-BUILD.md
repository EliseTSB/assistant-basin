# Part 172 cold line read

Read against PART172-DESIGN.md (Procedure v1). Files read in full: part172/run172.py, score172.py, label172.py, prompts172.py, probe172.py (for the imports), steps.txt. Beside them: part171/run171.py, prompts171.py, and the Part 171 saving code. Nothing was run that loads the model, and no reader was called. Only read-only grep, sed and a regex count were used. Paths are relative to the worktree `ab-part67`. Line numbers are from the files as checked.

## Blockers

1. **The pushes are read from the wrong folder. The stage fails after the model load.** `part172/run172.py:34-36, 780-781`, `146-149`. The design says the pushes come from `results/part171/sets_google.pt`. The code loads `out / SETS_FILE`, which is `results/part172/sets_google.pt`. `PART171_DIR` is only used for the answers file. `main` (`797-805`) has no flag to point elsewhere. `steps.txt` line 4 does not copy `results/part171/sets_google.pt` or `answers.jsonl` to the job, and line 5 excludes `*.pt` on the way back. The model loads at `run172.py:770`, before the sets load at `781`, so the run pays a full model load and then stops on `torch.load`. Also, `load_sets` uses `weights_only=True` (`149`). That will fail if the Part 171 file holds any non-tensor object, such as a numpy scalar. I could not check the file, because it is not in this checkout.

2. **Gate 3 crashes on its zero-push check.** `part172/run172.py:689`, `158-163`. The spec is `{"set": "none"}`, and `set_pushes` raises `KeyError` unless `"none"` is in `sets["arms"]`. Part 171 never saves `"none"` there. It appears only in `metrics` (`part171/run171.py:692`), and `materialize_arm` special-cases it (`run171.py:712`). The run172 code does not do that. The design asks for a zero-valued push, so the fix is an explicit zero dict for the 21 layers. Until that is fixed, Gate 3 never gets to give a result.

3. **The second gate-3 check is a real knockout, not a do-nothing hook.** `part172/run172.py:40, 333, 691`. The design says "a zero-unit hook on an empty list". The code zeroes L6 N753 and L7 N1169 (factor 0.0). That changes greedy output on some of the 8 prompts whenever those two neurons matter, and a correct build can then stop at gate 3. Taken literally, the design's empty list installs no hook, so the check passes by construction and cannot fail. Per the rule that a gate which cannot fail is a blocker, either form needs a fix. A useful form is an installed identity hook (factor 1.0) on real units, with a check that the hook is actually registered.

4. **Gate 6 sizes its arrays with the wrong width and will raise.** `part172/run172.py:242-243, 525-530`. `widths_of` returns `down_proj.weight.shape[1]`, the input width, which `validate_model` pins at 9216 (`part171/run171.py:1009`). The traced `actual` and `expected` are the output-side write, `effective @ W.T`, which is hidden-width (`run171.py:756-781`). So `np.zeros(9216) += hidden-width array` raises a broadcast error unless the hidden size is 9216. Part 171 sizes its own trace by `actual.shape[-1]` (`run171.py:844`). The Qwen3.5-4B hidden size is not in this checkout, so I could not confirm the mismatch. A 9216-wide residual would be unusual. Gate 6 runs after the baselines, the knockouts and the replays, so the run dies late and after most of the GPU time is spent.

5. **Singles ignore their dose.** `part172/run172.py:127, 325-330, 185-203, 642-644`. The spec carries `"dose": d`, but only the `set` branch (`326`) multiplies by dose. `single_push` takes no dose. So `n2_d0.5` through `n5_d0.5` push the same d = 1 vector as their `_d1` rows, and `_answer_record` still writes `dose: 0.5`. The `_d1` rows are correct, so 172.8 is not affected. The four d = 0.5 single arms are mislabelled data. The design says d multiplies the whole push and that each single is scaled alone to 4|v|.

6. **The second reader labels nothing, so the reader cut-off cannot be applied.** `part172/label172.py:175-178, 153-159, 121-128, 145-149`; `part172/score172.py:429-456, 486`. Haiku's filter keeps items whose id is in `sample["ids"]`. Those ids are the hand-read codes `S0001…` from `second_reader_sample(labels)`. The DeepSeek items have ids `arm|prompt_id`. The two sets never overlap, so Haiku writes an empty `labels-haiku.jsonl` with no error. `agreement` then returns `kappa: None, read: false`. The population is also wrong. The design asks for 10% of the clean `neither` answers, but the code samples about 11 labels from the hand-read sample.

## Major

7. **Gate 3 cannot see a leftover hook, and no gate or assertion checks hook removal.** `part172/run172.py:655-692`. Plain and zero-push both run with whatever hooks are still attached, so a stale hook affects both equally and the comparison still matches. Hook removal rests only on `finally` blocks (`run172.py:297-299, 346-348`; `run171.py:369-371`). The design's gate 3 is a do-nothing check and, as written, can rarely fail for a correct build (a zero push is the identity). Fix: assert no forward pre-hooks on the down-proj modules before and after each arm, and keep Gate 6's slope as the check that the hook is live.

8. **The runner-up reads the wrong position.** `part172/run172.py:583-599`. The prefix is `prompt + answer_ids[:answer_offset + 1]`, so it includes the Qwen-family token. The logits at `[-1]` are then the distribution after "Qwen", not the distribution that chose it. The design says "the next-token distribution there" and the stated purpose is a fallback rate. For the fallback reading the prefix should stop before the Qwen token. The design wording is ambiguous, so its owner should settle it before the code is fixed. As coded, the Google share measures what follows the word "Qwen", which is near zero by construction. The full-vocabulary logits are also materialised for every position at `599`. `logits_to_keep=1` would avoid that.

9. **The frame arms tell the reader which arm they are.** `part172/run172.py:307-313`, `part172/label172.py:121-128`. The forced phrase is the first tokens of `answer`, and `items_for` sends `answer` unchanged. "I was created by" (frame_plain), "I am a model trained by" (frame_n1) and "I am a large language model developed by" (frame_role) each identify the arm. The design requires the reader to be blind to arm and dose in what it sends. Fix: strip or rephrase the forced opening before it reaches the reader, or label the continuation alone.

## Minor

10. **Two control sets are generated twice.** `part172/run172.py:58, 123-124`. `rand100_d171` and `r100_d2171` run at d = 1 as controls and again as ladder rows `_d1`. The plan is 52 specs × 44 = 2,288 answers, against the design's 50 × 44 = 2,200. The design says two of the 12 control sets are ladder rows. The two copies match under greedy decoding, so the cost is about 88 generations.

11. **Frame scoring uses the first 40 tokens and a 12-word window.** `part172/score172.py:73, 77-78`. The design says the frame arms are "scored on the whole reply", and the frame window is "12 tokens". The code uses `first40` for hit, Qwen and frame, and 12 words for the window.

12. **Runner-up removed mass is not recorded.** `part172/run172.py:607-611`. The brief asks for the removed mass, and the row has no field for it.

13. **Gate 1 runs only after the ladder.** `part172/run172.py:749-754`. It is a completeness check, so it cannot stop an early failure. Its planned set comes from the same spec list, so a spec-level bug is not caught.

14. **The kappa cut-off is reported but not applied.** `part172/label172.py:149`. Nothing in `steps.txt` or `score172.py` reads `read`. The design says a kappa below 0.6 means the split is reported as unread.

15. **The known-example floor is not in the design.** `part172/label172.py:33, 118`. The design asks for 10 known examples, 5 of each, and gives no pass mark. The code uses 9 of 10.

16. **Reader coverage is not reported.** `part172/label172.py` `label_one`. An item that fails twice is dropped from kappa with no count. The design says every clean `neither` answer is labelled.

17. **Host memory for the single-neuron weights.** `part172/run172.py:782-788`. It builds float64 copies of all 21 down-proj matrices, about 0.2 GB each at a 2560 × 9216 shape (not checked against the config). Only layers 8, 18 and 20 are used.

18. **Host CPU loops at startup and in the runner-up.** `part172/run172.py:563-566` decodes every token in the vocabulary once, and `601-605` loops over the vocabulary per prompt. These cost CPU time only. There is no `.item()`, no device top-k and no model reload anywhere, so the GPU is not stalled.

19. **Reporting gaps.** `part172/score172.py:492-504` exits 0 when readings are stopped. Run172 itself exits non-zero on failed gates (`720-723, 753-754`). `score172.py:388` returns rows, not the dose-axis plot the design asks for. `arm_stats` at `121` divides `qwen_retained` by all answers including broken, but the clean counts use clean answers only.

## Hook removal, as checked

Every hook this code installs is removed in a `finally` block. Knockout and writer hooks (`ScaleUnits`) are removed in `run_spec` (`run172.py:346-348`) and in `measure_gate6` (`551-553`). Push hooks are removed in `run171.generate_batch` (`run171.py:369-371`) and in `generate_forced` (`run172.py:297-299`). The trace hooks are removed by `_TraceDownWrite.remove`. Nothing verifies this after the fact (finding 7).

## Scope and limits

- The 172.1 to 172.12 and 172.W1 bars match the design, and each reads the rows the design names. None is passed by construction. Clean, broken and GS match the design. The pool covers exactly the 12 control arm names.
- `steps.txt` `--copy` covers the imports I traced (`part165`, `part166`, `part160`, `part163`, `part164`, `part148`, `part153`, `part155`, `part86`, `part170`, `neuron_report`, `judge_client`). `--out-dir` is relative. The launch runs from the repo root, as its comment says.
- I did not run the tests (`test_part172.py`), the launcher (`tools/launch_arc.py`), or any step in `steps.txt`. The Part 171 sets file is not in this checkout, so findings 1 and 4 rest on the code and on `validate_model`. Whether `launch_arc.py` stops on a non-zero exit was not checked.

Verdict: not ready. Findings 1 to 6 are blockers, and as written the run stops at load, at gate 3, or at gate 6, or it silently mislabels the singles and empties the second reader.
