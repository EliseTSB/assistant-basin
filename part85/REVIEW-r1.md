# Part 85 build, cold review round 1 (Claude Haiku 5.5, read-only, 2026-10-27 21:4x CT)

Verdict: DO-NOT-SHIP as built. Items 2, 4, 5 and 6 are decided by Amendment 1 in PART85-DESIGN.md.

1. BLOCKER. part85.py:109 (used at 223, 287): `hashlib` is never imported. Every pilot or main session dies with
   NameError before the model loads. The suite stays green because no test calls `run()`, `_main_scope()` or
   `_template_sha256()`. Fix: import it, and add a smoke test that drives `run()` on the tiny Olmo2 model with a temp
   results root.
2. HIGH. score85.py:66, :125-126, :405-409 and CLOUD-RUN.md:63-65: pilot items hold only id and text, but the pilot
   gate needs a `responsive` judgement that requires the question. Fix per Amendment 1 item 1: the question goes in
   every item (pilot and main), `responsive` on all 96 pilot labels; update CLOUD-RUN and test_part85.py:200.
3. HIGH. part85.py:348-355 with `_load_completed` :250-258: a skipped (resumed) cell is checked for count, metadata and
   template SHA, never for `clamp_value` or `unit` against the current step1.json. Fix: compare against
   `_cell_value(cell, step1)` on skip, and record step1's SHA-256 in run.json.
4. HIGH. step0_weights.py:74-81: the reader requires `model.safetensors.index.json`. Measured since: Instruct ships one
   `model.safetensors`; SFT and DPO ship only `pytorch_model.bin`. Fix per Amendment 1 item 3: step 0 reads tensors
   from the checkpoint as downloaded (any format; torch.load with weights_only=True for .bin), records format and dtype.
5. HIGH. step0_weights.py:293-295: OLMo 2 applies `post_feedforward_layernorm` to the MLP output before the residual
   add, so the readout direction must be down column x `model.layers.10.post_feedforward_layernorm.weight`, then the
   final norm weight, then the output rows (Amendment 1 item 2).
6. MEDIUM. score85.py:250-254: the calibration check is skipped when only one onset exists. Fix per Amendment 1 item 4:
   one None and one non-None is disagreement; add a test where the absolute onset never occurs.
7. MEDIUM. score85.py:203-211, :264-331, :85-90: missing step files become None, then NOT READ verdicts, so a score.json
   missing step 1 looks complete; the chat-only exception is not tied to the pilot gate. Fix: main scoring requires all
   12 step0 and step1 files and the full 2,880 answers (or the gate's chat-only scope, checked against the recorded
   gate); NOT READ only under an explicit `--partial` flag.
8. MEDIUM. score85.py:49, :455, :64, :76: ids and order come from a public seed and `prepare` silently overwrites. Fix:
   ids from `secrets`, refuse to overwrite without `--force`, store the items file's SHA-256 in the key and check it
   when scoring.
9. MEDIUM. part85.py:292 hardcodes dtype float32; step1_acts.py:172 requests bf16 without asserting it; step 0 records a
   literal dtype source; only run.json records library versions; the WikiText load (step1_acts.py:37) is unpinned.
   Fix: record the actual parameter dtype, assert bf16 in step 1, record library versions and the dataset revision in
   every step JSON.
10. MEDIUM. test_part85.py:51 sets correlation 1.0 everywhere, so a same-unit gate ignoring correlation passes.
    Untested: score_payload (verdicts, 85.9), reader_agreement, _main_scope, _load_completed, collect_answers,
    validate_record_counts, all of step1_acts.py, BF16 decoding, read_rows, the z-score and token_sets; the Fisher
    test checks p < 0.05, not the value. Fix: add these, plus a score_payload test over all 12 checkpoints.
11. LOW. step0_weights.py:239-242: duplicate first tokens silently shrink the 16-form off-world set. Fix: require 16
    unique ids.
12. LOW. part85.py:131-142: the EOS set includes `convert_tokens_to_ids("<|im_end|>")`, which may be unk on base
    tokenizers; the effective generation_config is not recorded. Fix: record `model.generation_config.to_dict()`, drop
    ids equal to unk.

Checked and correct by the reviewer: the clamp hook covers every prompt and decode call for unit 6978 and control 7884;
layer 10 is 0-indexed as in part62; seeds 85000 + 100q + s; T 0.7, top-p 0.95, 64 tokens; the plain template; chat
only for SFT, DPO and Instruct; commit prefixes; the 85.1-85.9 thresholds; the passages leak no registered terms;
memory about 5.9 GB for fp32 1.48B, within 15 GB.
