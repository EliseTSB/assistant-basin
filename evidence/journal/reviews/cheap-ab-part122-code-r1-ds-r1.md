<!-- cheap-review.mjs cheap-review-ab-part122-code-r1-ds-r1-2026-10-30: /home/elisepc/green/worktrees/ab-part67 HEAD~1..HEAD, lane ds, df-run exit 0 -->
# Review: ab-part122-code-r1

Reviewed commit: `5cd012d "Part 122 build"` (6 files, +1906). All measurements below were run against a pinned export
of that commit (`git archive 5cd012d`), not the live worktree: the worktree HEAD moved to `20f37ec` mid-review
(three follow-up commits `4c2b74e`, `f3ca1f7`, `20f37ec` landed while I was reading). Several findings below are
already fixed by those later commits; I note where, but they are real defects **in the reviewed commit**.

## Verdict: SHIP-WITH-FIXES

The core of the change is sound and I verified it end to end on the real OLMo 2 1B and the real tokenizer: the hook
reads the right channel, frame C's mask is exact, the controls are seeded and non-self, the gates refuse incomplete
data, and the four reading branches fire as registered. The defects are in the **passage writer**, which is the one
piece that has never run against the real network and which the rest of the job depends on. None of them corrupts a
measurement; all of them either stop the run or waste paid calls. Fix the writer before the Linux passage run.

---

## Findings

### F1 — MEDIUM — `--dry-run` cannot produce a passage file; it dies on unit 34 of 39 (INTRODUCED)

`part122/write_passages.py:131-149` (`dry_run_client`) vs `part122/write_passages.py:35-83` (`validate_payload`).

The stub client builds `f"The {name} rests beside a window. {tail}"`. For a persona whose name ends in a period
(`Dr. Memento`, `p92-33`) that sentence contains three `.` characters, so `_valid_passage`'s
`len(_SENTENCE_END.findall(clean)) != 2` check rejects it. For `Street rat` (`p92-43`) the stub's ordinary sentence
reuses the word `street`, which the leak check rejects. The stub is never validated against its own validator.

Measured (pinned commit, real unit list):

```
$ cd pinned && python -I -B part122/write_passages.py --dry-run --out .../dryrun_pinned.json
  File ".../write_passages.py", line 175, in generate_passages
ValueError: could not generate valid passages for p92-33: third_person[0].persona must contain exactly two sentences
exit=1
$ ls .../dryrun_pinned.json
ls: cannot access ...: No such file or directory
```

and across all 39 units:

```
dry-run stub fails its own validator for 2/39 units
  ('p92-33', 'Dr. Memento', 'third_person[0].persona must contain exactly two sentences')
  ('p92-43', 'Street rat', 'third_person[1] ordinary passage leaks persona word(s): street')
```

**Why it matters.** BUILD-NOTES advertises `--dry-run` as the local smoke path ("uses a marked local stub and is not
study data"), and the only end-to-end test of the writer (`test_writer_dry_run_cli_uses_stub_and_marks_output`,
`test_part122.py:446`) runs on a synthetic 39-unit fixture whose personas are `Persona0..Persona38` — it never sees a
real persona, so the suite stays green while the real command fails. An operator who runs the documented smoke
command before the paid run gets a traceback and no file, and may conclude the writer is broken.

**Smallest fix.** Make the stub's sentences period-free and subject-free of persona words (e.g. `"The {name} rests
beside a window while the light fades"` with the name's trailing `.` stripped), and add a test that runs
`generate_passages(rows119.load_units(...), request_fn=dry_run_client)` over the real 39 units. (The later commit
`20f37ec` reworks the retry prompt but I did not see it fix this; verify before relying on it.)

---

### F2 — MEDIUM — a failed unit discards every earlier unit's paid call; there is no resume (INTRODUCED)

`part122/write_passages.py:152-176` (`generate_passages`) and `:179-188` (`write_output`).

`generate_passages` accumulates `rows` in memory and only `write_output` persists, once, after all 39 units succeed.
Any unit that exhausts its 4 attempts raises, and the whole run's 38 paid DeepSeek calls are lost. There is no
per-unit append, no checkpoint, and `write_output` uses mode `"x"` (refuse if exists), so a rerun starts from zero.

Measured:

```
raised: could not generate valid passages for u2: net
calls: 5 -> no partial file written; u1's paid call is lost on retry
```

**Why it matters.** The design budgets "39 DeepSeek calls"; a single transient failure near the end (a 300 s timeout
in `judge_client.call_prompt`, a rate limit, a malformed JSON) costs the whole batch and a full rerun. The brief's
"restart between two steps / retry after a partial success" question lands squarely here.

**Smallest fix.** Append each validated unit to a JSONL as it completes and load completed units on start (the same
pattern `read122._existing_rows` already uses for activations), or at minimum write the partial `rows` to a
`.partial.jsonl` in the `except` before re-raising. (The later commit `f3ca1f7` is titled "resumable" — check whether
it lands this.)

---

### F3 — LOW — the retry prompt never names the failed check (INTRODUCED; fixed after the reviewed commit)

`part122/write_passages.py:164-165`.

On a retry the writer appends a fixed string, `"Retry: fix the validation error and return the required JSON only."`,
and never includes `last_error`. The model is told something is wrong but not what, so attempts 2–4 are near-random
re-rolls rather than corrections. The later commit `20f37ec` ("a retry names the failed check") confirms this was a
real defect; it is present in `5cd012d`.

**Smallest fix.** `current_prompt += f"\n\nRetry: the previous answer failed this check: {last_error}. ..."`.

---

### F4 — LOW — `load_passages` accepts a file the writer marked "not study data" (INTRODUCED)

`part122/read122.py:142-165`.

`write_passages` stamps `model_id: "stub-local-not-study-data"` into every `--dry-run` file, but `load_passages`
never reads `model_id`, `seed`, or `schema_version` — it only checks per-unit persona/desc and the payload shape.

Measured:

```
load_passages accepted a stub file (model_id=stub-local-not-study-data) for 37 units; no study-data check
```

**Why it matters.** The marking exists precisely so a stub cannot be mistaken for study data, and the reader is the
only place that could enforce it. A dry-run file (once F1 is fixed) would run on the A770 and produce activations
indistinguishable from a real run.

**Smallest fix.** In `load_passages`, `if data.get("model_id") != write_passages.MODEL_ID: raise ValueError(...)`.

---

### F5 — LOW — Gate 1's "read by eye on a seeded sample of 24 pairs" is not implemented (INTRODUCED; a claim the code does not do)

`PART122-DESIGN.md` gate (1) says the within-2-token check is "read by eye on a seeded sample of 24 pairs before the
run, with any persona passage that does not concern its persona replaced". No code does this: `grep -n "eye\|sample\|24"`
over `part122/*.py` returns only `rng.sample` (random channels). `score122.evaluate_gate1` checks token counts only;
nothing checks that a persona passage actually concerns its persona.

**Why it matters.** The design makes the eye-check part of the gate that must pass "before any prediction is read".
The code silently drops it, so a passage that is structurally valid but off-topic passes gate 1. This is a
claim-vs-code gap, not a measurement bug.

**Smallest fix.** Either implement the seeded 24-pair sample as a file the operator must sign off (`gate1_eye.json`
with a `reviewed: true` flag that `evaluate_gate1` requires), or strike the clause from the design in a registered
amendment. Do not leave the design claiming a check that does not exist.

---

### F6 — LOW — Gate 2 has a 40% false-pass rate under a null (INTRODUCED; the threshold is registered, so this is a design note, not a code bug)

`part122/score122.py:99-118`, threshold `wins >= 9` of 16.

This is applied as registered (design: "at least 9 of 16"), so the code is correct. But the registered threshold is
weak: under a coin-flip null, gate 2 passes 40% of the time.

```
gate2 passes on pure coin flips: 7965/20000 = 0.398
```

**Why it matters.** Gate 2 is the check that "the capture reads the same channel". A 40% null pass rate means a
channel-capture bug has a 60% chance of being caught, not a 90%+ chance. On the real model it passed 11/16 on the
actual Part 95 passages (`gate2 pairs 16 space wins 11`), matching Part 95's recorded 11/16 — so the capture is
correct here — but the gate's power is lower than its role implies.

**Smallest fix.** None in code (it is registered). Note it in the design's Limits, or register a follow-up with a
stricter threshold (e.g. 12/16, null rate ≈ 4%).

---

## What I checked and found sound

- **The channel read is the right one.** `capture_batch` (`read122.py:259-286`) registers a
  `register_forward_pre_hook` on `layers[layer].mlp.down_proj`, so `inputs[0]` is the down_proj **input** (width 8192
  on OLMo 2 1B), not the 2048-wide output. Measured on the real model at L2 N2466:
  ```
  input shape torch.Size([1, 13, 8192]) output shape torch.Size([1, 13, 2048])
  manual input max at 2466: 0.0025942218489944935
  code unit max: 0.0025942218489944935 mean: -0.0063939448446035385
  manual mean over all tokens: -0.0063939448446035385
  MATCH input: True
  ```
  Layer index matches Part 95 (`push95.py:19` `LAYER, NEURON = 2, 2466`, same pre-hook on `down_proj`).
- **Frame C's mask is exact on the real tokenizer.** The trailing `<|endoftext|>` and the `<|assistant|>\n` header are
  excluded; only passage tokens are masked:
  ```
  RENDERED: '<|endoftext|><|user|>\nWho are you?\n<|assistant|>\nI am the Chicken ... home.<|endoftext|>'
  span 15 35; tokens in span: ['I',' am',' the',' Chicken', ... ,' home','.']
  token BEFORE span: '>\n'; token AT/after end: [(35, '<|endoftext|>')]; any eos inside span: False
  ```
  T and F spans are the whole bare passage (no BOS added by this tokenizer: `first id 791 bos? False`).
- **Padding does not leak into the stats.** Right-padding with `pad_token_id` and a `False` token mask gives the same
  max as a solo forward: batched `short` max `0.001322439406067133` == solo `0.001322439406067133`; `long`
  `0.02965623512864113` vs solo `0.02965623326599598` (bf16/float noise only).
- **The real job starts.** `read122.main` ran end to end on the real OLMo 2 1B (CPU) with a 2-unit limit:
  `rc 0`, 96 rows, frames `['C','F','T']`, `gate2 pairs 16 space wins 11`. Every import (`common92`, `rows119`,
  `write_passages`, `part85.checkpoints` via common92) is in the BUILD-NOTES `--copy` list, and the runtime copy tree
  imports cleanly under `python -I -B` (I reproduced the copy list in a bare directory; `--help` exits 0).
- **Resume after an interrupted write.** A truncated final JSONL line is dropped and the run rewrites the missing
  passages: `resume with truncated tail -> 0`, `rows after resume: 96 unique: 96`. A changed passages digest is
  refused (`ValueError: existing run_meta.json does not match this run configuration`).
- **Gates refuse incomplete data.** With only 5 of 39 units present, gate 1 fails (`120/936 complete`), gate 3 fails
  in every frame, and all seven predictions read `not read (gate 1)` with `branch: gate_stop`.
- **Controls are seeded, non-self, and cannot read focal channels.** 0 self-partners, 39 distinct partners,
  deterministic across two builds, 0 random channels hitting a focal channel in the same layer, no duplicate random
  channels within a unit.
- **Strict wins and 7-of-8.** All ties → not read; 7 → read; 6 → not read. Fisher matches the expected values
  (`fisher(9,9,3,18)=0.0346`).
- **The four reading branches** fire as registered on synthetic inputs (branch 1/3/4 reached; branch 2 reachable when
  122.4 is available and fails).
- **Predictions use Haiku kinds** (18 concept / 21 slot), as the design words them.
- **The J-lens list** has 11 rows, all present in `neurons.json`, all Haiku-concept, layer and neuron matching the
  unit data, no flags.
- **The suite catches real regressions.** Flipping `>` to `>=` in `_compare_series` turns
  `test_scorer_aggregates_gates_controls_tables_and_reading` red; widening the frame-C offset window turns
  `test_frame_c_token_mask_covers_only_assistant_passage_tokens` red. (Both sabotages were run in a mirrored tree.)
  Note the frame-C test silently falls back to `StubTokenizer` on this host (`Frame C token-mask tokenizer: stub
  tokenizer`), because it looks for the model under `/home/elisepc/neuron-probe/hf/...`, which does not exist here
  (the model is at `/mnt/e/...`). The stub does carry an `<END>` after the passage, so the test is not blind to an
  end-token inclusion — but it is not exercising the real tokenizer on this machine.
