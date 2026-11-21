# Part 215 fold 1 — v4.1 endpoint, `-ing` veto, admission, scorer admissibility, G0 reuse

## SUMMARY

- `DESIGN_VERSION`/`DESIGN_SHA256` are now v4.1 and `e64521c1…9e76` (the committed v4.1 bytes at `35e6310a`).
- 215.2, 215.3 and 215.4 (interaction included) read `attrib_google`; `attrib_any` is reported beside each; a red/green test shows an any-maker reading flips all three.
- `attrib215` has the v4.1 `-ing` veto (nine fixtures, G3 requires nine); name claims are unchanged; the rule is still labelled an attribution proxy.
- `full` admits only a whole preflight receipt (gates, counts, latencies, runtime, tool digest, pinned prompt bank and Part 214 receipt bytes, bound G0 reuse file) and refuses a label-only PASS with zero loader calls.
- `score_full` scores only an exact, unique, pin-matched, complete receipt with an evidenced PASS preflight; anything else is diagnostic-only (tables printed, all predictions and the headline `not_scored`).
- G0's 24 neutral reads are kept, digest-bound and reused; full runs the other 600, for 648 model forwards across preflight+full.
- Resume validates every saved row's/forward's pins and runtime and refuses to mix attempts.
- 39 CPU tests pass (`39 passed in 90.66s`); every new test is red on the pre-change tree (sabotaged copies for the Alden witnesses and the endpoint flip).

## CHANGES (file:line)

`part215/score215.py`
- `score215.py:52-66` — `DESIGN_VERSION = 'v4.1'`; `DESIGN_SHA256 = e64521c1…`; pinned reviewed bytes: prompt canonical `7118e649…`, prompt file `0ac54f8e…`, Part 214 stage1 `4f287bc2…`, stage2 `a1faffd9…`.
- `score215.py:287` — `_ing_between`: the v4.1 `-ing` veto (literal, `being` excepted). Applied in the participle route at `score215.py:322`; name claims untouched.
- `score215.py:369-402` — nine `FIXTURES`, `EXPECTED_FIXTURES = 9`, `g3_scorer_check` requires all nine and a red relaxed break.
- `score215.py:691` — `_endpoint_field`: the Google endpoint 215.2/3/4 read.
- `score215.py:737-793` — 215.2/3/4 verdicts read Google; observed carries Google and any-maker side by side.
- `score215.py:886-907` — `load_prompts` refuses a bank that is not the reviewed canonical/file bytes.
- `score215.py:986` — `_preflight_evidence_failures` (scorer-side gate evidence) and `score215.py:1026` `admissibility` (exact unique key sets, no off-plan, pin sets, completed run summary, PASS+evidence preflight, G0-bound neutral forwards).
- `score215.py:1149` — `score_full` is diagnostic-only unless admissible: all predictions `not_scored`, headline `not_scored`, contrasts/factorial marked `not_scored`; tables remain and `render_markdown` (`score215.py:1296`) labels them `— DIAGNOSTIC ONLY`.
- `score215.py:168-184` — AMBIGUITY 1 renamed: the task neutral seeds are "a new, disjoint task schedule, not a replay".

`part215/run215.py`
- `run215.py:59-73` — pinned Part 214 digests, `g0_neutral.jsonl`, `G0_REUSE_N=24`, preflight call counts, shared pin/runtime key lists.
- `run215.py:371-433` — `load_stage1_receipt` pinned by file digest + shape; `gate0` keeps the 24 complete neutral records.
- `run215.py:434-493` — `load_stage2_unchanged` pinned by file digest and validates all 48 keys, Part 214 row seeds and token ids, before any model load.
- `run215.py:538-640` — preflight validates the references BEFORE `rig_loader`, writes `g0_neutral.jsonl` with prompt/opening ids, model, revision, runtime, attempt id and `source: g0_reuse`, and binds its SHA-256 into the receipt (`attempt_id`, `g0_reuse_*`).
- `run215.py:655-806` — `_gate_failures` (G3 nine fixtures/red break; G0 both 24-question comparisons and rows; G1 48 rows/45 matches/exact seeds; G2 pilot 96/bound), `_latency_failures` (48 forwards + 144 generations, max ≤ 7.5 s), `_runtime_receipt_failures`.
- `run215.py:809-864` — `require_pass_preflight`: part/mode/result/full/PASS sentinel, design/version, prompt canonical+file, model, revision, runner `tool_sha256`, Part 214 pins, attempt id, G0 reuse binding, then all gate/latency/runtime evidence.
- `run215.py:866-947` — `load_g0_reuse`: digest + 24 keys + prompt/opening ids + model/revision/runtime/attempt id.
- `run215.py:949-971` — `_validate_saved`: every saved row/forward must carry this invocation's pins and runtime.
- `run215.py:973-1107` — `full`: verifies reuse before load, refuses run.json/row/forward identity mismatches before appending, writes 24 `g0_reuse` neutral records plus 600 `source: full` reads, and records the 648-forward accounting.

`part215/test_part215.py`
- New red/green and contract tests at lines 65-123 (`-ing`, nine fixtures), 216-260 (pins, including the production Part 214 receipts), 533-565 (Google vs any endpoint), 567-651 (admissibility and Alden's two witnesses), 873-952 (reference pins, tampered reference, thin PASS receipt — all with zero loader calls), 966-1026 (G0 reuse + 648 count), 1028-1076 (attempt mixing and saved-row identity), 1078-1163 (truncated run diagnostic, real preflight PASS path).

## TESTS (exact commands and full pass/fail lines)

Run from `/home/elisepc/assistant-basin`:

```
$ PYTHONDONTWRITEBYTECODE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider part215/test_part215.py
.......................................                                  [100%]
39 passed in 90.66s (0:01:30)
```

Prompt bank rebuilt and byte-identical (both pins still match):

```
$ /home/elisepc/neuron-probe/.venv/bin/python -I -B part215/build_prompts215.py
canonical SHA256: 7118e649be2cdfb4fa24d2b11ce4147e8a34794d559cf521dfebd9a17ccb0452
file SHA256: 0ac54f8e8e46b3bf06dfb7108a4743392bd0f218ae3f70778f2d5b4b98544d03
```

Old 21 tests: all still pass. Legitimately updated (reason in brackets):
- `test_attrib215_passes_all_six_fixtures` → `test_attrib215_passes_all_nine_fixtures` [v4.1 adds three fixtures].
- `test_g2_failed_arm_*`, `test_incomplete_arm_*`, `test_complete_synthetic_run_*`, `test_215_5_*` [their synthetic rows now carry pins and a PASS preflight, because admissibility is a new contract; the assertions are the same].
- `test_preflight_stop_on_g1_mismatch` [reference pins pointed at the synthetic fixtures, item C].
- `test_preflight_pass_then_full` [forwards_total 4 → 24 because the four neutral keys are now served by G0 reuse; the scorer half became `test_scorer_on_a_truncated_run_is_diagnostic_only`].
- `test_runner_refuses_a_stop_receipt` [unchanged].

## RED-GREEN WITNESSES (one line each)

- Google vs any endpoint: tracked `215.2/215.3/215.4 = holds/holds/holds`; a copy with `_endpoint_field` returning the any-maker entry gives `fails/fails/fails` (`test_215_2_215_3_215_4_read_google_not_any`).
- `-ing` veto: g3's relaxed matcher (self-predicate requirement dropped) is red on "I read a guide developed by Google."; the negation-break copy is red on "I was not trained by Google."; the tracked rule passes all nine.
- Alden witness 1 (384 identity rows, zero forwards, `deadline_reached=true`, preflight STOP): tracked card is diagnostic-only (`215.1 not_scored`, headline `not_scored`); a copy with the admissibility gate disabled scores `215.1 holds`, headline `rise`.
- Alden witness 2 (one off-plan variant substituted, 2,304 rows kept): tracked card is diagnostic-only with `off-plan` in the reason; the gate-disabled copy scores `215.2` and yields a headline.
- Thin PASS receipt (matching identity strings, no gates/latencies): `full` raises with `G3_missing`, `latency_missing`, `g0_reuse…` and the `rig_loader` counter is 0.
- Tampered Part 214 reference (changed bytes, or a self-consistent changed seed): preflight STOPs with 0 loader calls; production pins refuse a synthetic reference the same way.
- Attempt mixing: a second valid receipt with different bytes is refused with `refusing to mix attempts` and `rows.jsonl` is byte-unchanged; a saved row with another model or another `torch` is refused; a clean same-run resume adds 0 rows and 0 forwards.
- G0 reuse count: preflight G0 has 48 forward rows; full executes exactly 600 stubbed readout forwards and writes 624 forward rows (24 `g0_reuse` + 600 `full`); `model_forwards_total_across_preflight_and_full == 648`.

The remaining new tests fail on the pre-change tree (`daa1d7bd`) by construction: they assert fields and behaviours that did not exist there (nine fixtures, the `-ing` string, the reviewed-bytes pins, `g0_neutral.jsonl`, `source: g0_reuse`, `diagnostic_only`, the whole-receipt refusal, and the saved-row identity check). The two Alden witnesses and the endpoint flip additionally carry an explicit sabotaged-copy red run.

## OPEN QUESTIONS

1. **Forward key set.** Item D says "exact 648-set: 24 G0-reused neutral + 600 full", but 24 + 600 = 624; 648 is the preflight+full model-forward total (24 neutral + 24 zero + 600), and the 24 `zero` reads share `(arm, variant, question)` keys with the neutral reads, so they cannot also live in `forwards.jsonl`. I implemented the exact **624-key** set in `forwards.jsonl` and require the **648** model-forward accounting (receipt G0 48 forwards + full's 600). If 648 readout records were intended, that needs a different key shape and a re-review.
2. **The brief's "currently `d5379540…`" digest.** No file in the repo has that digest (grepped). The committed v4.1 `DESIGN.md` at `35e6310a` hashes to `e64521c1…`; that is the pin used. If the brief meant a different draft's bytes, the pin must be changed before registration.
3. **PASS sentinel required.** I read "the expected preflight artifact/mode" as requiring the `PASS` file beside `preflight.json` (stricter). A receipt copied out of its directory is refused; say if a receipt-only handoff should be allowed.
4. **Runner digest pins itself.** `full` requires the receipt's `tool_sha256` to equal the running `run215.py`. Any future edit to the runner invalidates old receipts; there is no data yet, so this is free now but will matter after a real preflight.
5. **G0 reuse runtime equality is strict.** Preflight and full must run with the same device/dtype/attn/versions/token cap/sampling for the reuse to be accepted. A library upgrade between the two refuses rather than silently mixing.
6. **Scorer evidence duplication.** `score_full` checks the preflight's gate evidence at a coarse level (it cannot import the torch-side validator). If the two validators drift, the scorer is the weaker one; tests pin both today.

## BLOCKERS

None. The real-model preflight/full still needs the Arc (xpu) and the Qwen3.5-4B weights; neither exists in this sandbox.
