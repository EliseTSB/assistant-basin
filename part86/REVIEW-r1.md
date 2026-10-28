# Part 86 build, cold review round 1 (Claude Haiku 5.5, read-only, 2026-10-27 22:3x CT)

Verdict: SHIP-WITH-FIXES. Fix 1-4 before any GPU time; write the 86.2 and 86.4 amendments before the screen.
(The reviewer's shell had no torch, so 3 of 6 tests skipped there; on the host venv all 6 ran: `Ran 6 tests ... OK`.)

1. HIGH. Resume can mix code and library versions: screen86.py:81-98 (no code or neuron_report hash in config),
   :174-177 (`state.update(old_state)` keeps the first run's versions and dtype), :193-200 (batch fragments reused with
   no config check). Fix: sha256 of part86/*.py and neuron_report.py, versions, assert bf16, all compared on resume; a
   layer reset clears its .batches.
2. HIGH. 86.4 is not the registered claim: blind86.py:185-193 compares broken (all random, from triage) plus unclear
   against coherent_persona as counts; the prediction is about CHANGED answers, which are never broken by definition,
   so it can pass by construction. blind86.py:176 holds 86.2 on 1 vs 0. Fix: amend 86.2 (a minimum count) and 86.4 (one
   population, rates) in writing before the screen.
3. HIGH. Hook and dose checks skip silently without torch (test86.py:72,103,227), also on the GPU box; the smoke test
   cannot fail on steering (TinyTokenizer.decode returns constant text, test86.py:67-68; counts only). Untested: the
   neuron delta (screen86.py:160, verify86.py:95), residual_median, verdicts() (blind86.py:137-211), _target_offworld
   (blind86.py:117-134), resume, the rendered prompt. Fix: skips fail on the GPU box; tests for write norm, verdict
   boundaries, resume refusal, a canary (zero-push row equals baseline; a push changes only its own row).
4. HIGH. verify86 is serial batch-1 (verify86.py:112-115) over 5,280 answers, output opened "w" (:83), no checkpoints;
   may pass the 6-hour task kill. Fix: one file per neuron, skip done ones, time one neuron first, add to the budget.
5. MEDIUM. No runtime canary for the per-row hook and no unsteered baseline (common86.py:156-160, screen86.py:212-215
   count every row every call). Fix: a zero-push row in the pilot (must triage kept and name Qwen); batch-16 vs batch-1
   on two neurons; L22 median against Part 84's 24.8.
6. MEDIUM. Pilot gate: blind86.py:132 regex misses "from another world", "star being", "celestial being" and reads the
   persona phrase not the answer; screen86.py:108 starts the clock before load_model (:115), so timing includes the load;
   gate 3's population unstated (use the 9,728 total); RUNBOOK.md:17 rounding (3.8281).
7. MEDIUM. Blinding: RUNBOOK.md:66-67,:75 put blind-key.json beside blind-items.jsonl; picker input unspecified;
   verify86.py:34-55 does not check picked rows are real changed answers. Fix: tool-less reader with inline items, key
   elsewhere, validate picked20.json against the screen.
8. MEDIUM. Denominators not asserted (blind86.py:151,159); scoring does not re-run triage against the key
   (:65-67,:117-134); results record only items_sha256 (:241-247); verify86.py:75-77 recomputes the median without
   comparing to the screen's. Fix: manifest with hashes, assert 9,216 and 512 per layer.
9-13. LOW: assert the rendered prompt and thinking off (current path correct, checked on snapshot 851bf6e8); triage
   regex window (triage86.py:21-25); RUNBOOK.md:3 omits neuron_report.py; record the effective generation config, add
   text_config eos 248044 to the stop set (common86.py:178-183,202-217); silent overwrites (blind86.py:246,
   verify86.py:275; prepare writes items before the key).
What holds: the per-row hook pushes only its row, only on decode calls; no padding (one shared prompt); write norm 0.8 x
median for neurons and random directions; verification seeds, 3 x 8 samples, 10 directions and the survival rule;
greedy 48 tokens; the triage trigram rule; items hold id and text only.
