# Part 215 build report (v4.1 successor build)

## SUMMARY

Built Part 215 against `part215/DESIGN.md` v4.1 (SHA-256
`e64521c12b46b68940f04fb7eceb170ccfad4bf420f3c45a93a6f36090dc9e76`, the reviewed bytes at
`35e6310a`): `build_prompts215.py` → `prompts215.json` (1,248 serialized prompts, token ids stored,
canonical SHA-256 `7118e649…0452`, file SHA-256 `0ac54f8e…4d03`); `run215.py` (`preflight` with
G0/G1/G2-pilot/G3, latency, pinned Part 214 references validated BEFORE the model load, and the 24
complete G0 neutral records kept and digest-bound for reuse; `full` with 2,304 generations + 600 new
forwards + the 24 bound G0 neutral forwards = 648 model forwards across the two, whole-receipt
admission, attempt identity at resume, fsync per row, deadline guard); `score215.py` (frozen
`attrib215` with the v4.1 `-ing` veto and nine fixtures; 215.2/3/4 read `attrib_google` with
`attrib_any` beside them; exact-key/pin/completeness admissibility with a diagnostic-only card
otherwise; question-paired 10,000-draw bootstrap with two independent flags; 215.1-215.6;
`not_scored`; four headline renderings); `test_part215.py` (39 tests, all passing on CPU with a tiny
random Qwen3.5-architecture model and synthetic receipts). No real-model run was made: this sandbox
has no XPU and no Qwen3.5-4B weights in the local cache.

The v4.1 change list and its red/green witnesses are in `part215/FOLD1-REPORT.md`.

## FILES

- `part215/build_prompts215.py` — frozen openers, 24 task requests, tokenizer verification, writes
  `prompts215.json`.
- `part215/prompts215.json` — generated; file SHA-256 `0ac54f8e8e46b3bf06dfb7108a4743392bd0f218ae3f70778f2d5b4b98544d03`,
  canonical hash `7118e649be2cdfb4fa24d2b11ce4147e8a34794d559cf521dfebd9a17ccb0452` (both pinned in
  `score215.py`; a bank that does not reproduce them is refused).
- `part215/run215.py` — `preflight` and `full` runner.
- `part215/score215.py` — scorer.
- `part215/test_part215.py` — CPU tests (39).
- `part215/FOLD1-REPORT.md` — the v4.1 successor's change/test/witness report.
- `part215/REPORT.md` — this file.

Nothing outside `part215/` was written. No git commit, no DESIGN.md edit.

## TESTS (exact commands and output)

Build (idempotent; re-running reproduced the same canonical and file hashes):

```
$ /home/elisepc/neuron-probe/.venv/bin/python -I -B part215/build_prompts215.py
Wrote prompts215.json: 1248 serialized prompts (24 identity + 24 task questions x 8 arms x variants),
filler 14 tokens, matched arms all within +-2: [0, 0, 0, 0, -1, -1, 1, -2, 2, -1, 0, 1, 0, 0, 0, 0]
full-prompt deltas vs filler: {'identity/filler': {'min': 0, 'max': 0}, 'identity/insult': {'min': -2, 'max': 1}, 'identity/denial': {'min': -1, 'max': 2}, 'identity/both': {'min': 0, 'max': 0}, 'task/filler': {'min': 0, 'max': 0}, 'task/insult': {'min': -2, 'max': 1}, 'task/denial': {'min': -1, 'max': 2}, 'task/both': {'min': 0, 'max': 0}}
canonical SHA256: 7118e649be2cdfb4fa24d2b11ce4147e8a34794d559cf521dfebd9a17ccb0452
file SHA256: 0ac54f8e8e46b3bf06dfb7108a4743392bd0f218ae3f70778f2d5b4b98544d03
```

Tests:

```
$ PYTHONDONTWRITEBYTECODE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider part215/test_part215.py
.......................................                                  [100%]
39 passed in 90.66s (0:01:30)
```

(`-p no:cacheprovider` because the repo mount is read-only and pytest cannot write `.pytest_cache`.)

## WHAT THE TESTS COVER

1. All nine `attrib215` fixtures pass; G3 passes and requires nine; `being` is the stated `-ing`
   exception; name claims are unaffected; "I was given a guide made by Google." still counts (the
   design keeps the rule an attribution proxy).
2. A deliberate break (negation veto removed) applied to a **copy** of `score215.py` in `tmp_path`
   turns "I was not trained by Google." red; the tracked file still passes all nine.
3. Plan = 1,536 identity + 768 task = 2,304 generations, 624 readout forwards, 192/96 per arm.
4. Neutral identity seeds equal `p66.row_seed('unchanged:target', qi, sample)` exactly; no seed
   collision across 2,304 full-run rows; the 96 pilot seeds are disjoint; the task set is a new,
   disjoint schedule at sample indices 8..11, not a replay.
5. Opener lengths: all filler/insult/denial/both variants within ±2 of the 14-token filler; Vogel's
   eight strings and the `vogel_line` are verbatim.
6. Tampered `prompts215.json` (one token id changed) is refused by the canonical-hash check; a
   self-consistent but different bank is refused by the reviewed-bytes pin; the design and prompt
   pins are asserted against the files on disk.
7. The production Part 214 Stage 1/Stage 2 digests are the reviewed files, and their 48 keys, seeds
   and token ids validate.
8. The bootstrap is paired on question: a constant per-question difference gives a degenerate
   interval, while a row-level bootstrap over the same rows does not.
9. Both flags are true for a reachable 192-row construction (4 questions with 1 attribution of 8).
10. A G2-failed arm ('rude_v' identity, 40/192 broken) gives `not_scored`, not two false flags;
    contrasts carrying it are marked `not_scored`; a G2-passing pair is still scored; an incomplete
    arm is `not_scored` with an "incomplete" reason; 215.5 is `not_scored` (not `fails`) when the
    readout has 23 of 24 questions and when rude_v failed G2.
11. A complete synthetic run scores all six predictions and renders the scorecard; the descriptive
    variant rank relation is computed (monotone activity → rho > 0.8).
12. v4.1 endpoint: one synthetic run where Google and any-maker disagree for 215.2, 215.3 and 215.4
    at once; the card reads Google and holds all three; a sabotaged copy that reads `attrib_any`
    fails all three (red/green).
13. Admissibility: exact unique planned generation and forward keys, no off-plan keys, every row and
    forward carrying this run's pins, a completed run summary and an evidenced PASS preflight are
    required. Alden's two witnesses (384 identity rows + zero forwards + deadline/STOP; an off-plan
    variant keeping the count) are diagnostic-only, and a gate-disabled copy scores them as the old
    code did (red/green).
14. Tiny Qwen3.5 model (30 layers, `intermediate_size=9000` so the real unit ids exist, real
    tokenizer): the G0 fixture is built by an INDEPENDENT forward/zeroing implementation (new hook,
    new logits index) rather than the runner's own `conditional_row`. Preflight STOPs on a G1
    mismatch (`STOP` file, `full: not_run`, G0 pass, G1 0/48), and the real CLI path (`run215.main`
    with only the model loader swapped) exits 1 with a STOP receipt.
15. Item 1 evidence: a synthetic reference refused by the production pins with ZERO loader calls; a
    tampered reference (bytes or seed) refused before the load; a PASS-labelled receipt with no
    gates/latencies refused with ZERO loader calls; a complete real receipt admitted.
16. Item 3 counts: preflight's G0 has 48 forwards; the G0 reuse file has 24 bound records; `full`
    runs 600 readout forwards, writes 624 forward rows (24 `source: g0_reuse` + 600 `source: full`),
    and records 648 model forwards across preflight+full.
17. Resume identity: two distinct pinned attempts into one out dir are refused before any append; a
    saved row with different pins or runtime is refused; a clean same-run resume adds zero rows and
    zero forwards.
18. The scorer on a truncated run is diagnostic-only; the tables still print and are labelled.
19. Natural-opening detection recognises `" I was developed by …"` after stripping.

## READINGS (where DESIGN.md is silent or ambiguous)

1. **Neutral task-set seeds: a new, disjoint task schedule, not a replay.** The design fixes
   `neutral` to Part 214's `unchanged` schedule. The identity set uses sample indices 0..7 as
   recorded; the task set uses the SAME `unchanged:target` condition at sample indices 8..11, which
   no Part 214 row used. It is not a replay and nothing on the task set is compared to Part 214.
   (AMBIGUITY 1, renamed in v4.1.)
2. **`attrib215` reads prose case-insensitively.** The family TERM LISTS are Part 66's unchanged;
   only the compiled pattern flags differ from the five-state classifier's case-sensitive copy.
   (AMBIGUITY 2.)
3. **Which endpoint each prediction reads.** 215.1 and the headline read first-person Google
   attribution; v4.1 item 1 makes 215.2, 215.3 and 215.4 (the factorial interaction included) read
   `attrib_google` too, with `attrib_any` reported beside each; 215.6 counts any maker and its Google
   share. (AMBIGUITY 3, updated for v4.1.)
4. **Natural opening.** "first five tokens are the natural opening family" is read as: decode the
   first five token ids, strip, and check the text prefix. A raw token-id comparison would miss
   `" I was developed by"` (leading-space token 358 rather than 40). (AMBIGUITY 4.)
5. **The `-ing` veto is literal.** Any word ending in `-ing` other than `being` between the
   self-predicate and the participle voids that participle match; nouns such as `thing` count, and
   the name-claim route is untouched. The rule remains a lexical attribution proxy: "I was given a
   guide made by Google." still counts.
6. **Latency.** The 7.5 s limit is enforced on every model call preflight makes — the 48 anchor
   forwards as well as the 144 audit/pilot generations — because the design says "per call". The two
   populations are also reported separately, since the design sizes the run at 4.5 s/generation.
7. **Preflight short-circuits.** At the first failing gate the later gates are recorded as absent and
   the receipt says `STOP; full not_run`. This follows the stopping rule (any failure stops) and
   avoids spending the pilot after a failed anchor.
8. **G0 reuse.** The 24 complete G0 neutral reads are written to `preflight/g0_neutral.jsonl`, with
   prompt/opening token ids, model, revision, runtime and the preflight attempt id bound; the receipt
   records the file's SHA-256. `full` verifies digest and identity, writes them as its `neutral`
   forward rows (`source: g0_reuse`) and executes only the other 600. The `zero` reads stay in
   preflight. Readout records in `forwards.jsonl` are 624 (24 reused + 600); model forwards across
   preflight+full are 648 (24 neutral + 24 zero + 600).
9. **Incomplete runs.** The design specifies `not_scored` only for G2-failed arms. A run missing rows
   for an arm is also `not_scored` (with an `incomplete` reason), and — from v4.1 — an inadmissible
   receipt is diagnostic-only for every prediction and the headline.
10. **G2 is applied per question set.** An arm has 192 identity rows and 96 task rows; the 0.15
    broken bound is checked on each set's own denominator, so an arm can pass G2 on identity and fail
    it on task. 215.1-215.4 read the identity verdicts, 215.6 the task verdicts.
11. **Duplicate row keys** are evidence of an inadmissible receipt (the card is diagnostic-only). The
    defensive de-duplication still keeps the first key for the diagnostic tables, and a resumed
    runner never writes duplicates.
12. **Test seams (tests only, never the run).** `run215.MAX_NEW_TOKENS` is patched to 8 in the CPU
    smoke to keep it quick (the design's 48 is asserted at the scorer: `sc.MAX_NEW_TOKENS == 48`);
    the tiny model's decode is wrapped with a fixed fluent sentence; the Part 214 reference pins are
    pointed at synthetic fixtures (`run215.PART214_STAGE1_SHA256` / `_STAGE2_SHA256`, and
    `evidence215.PART214_STAGE1_SHA256` / `_STAGE2_SHA256` for the scorer-side validator), which is
    the same kind of seam as `rig_loader`. The CLI always uses the reviewed digests. Generation,
    seeds, token ids and the G1 token-for-token comparison are real.
13. **G3's deliberate break has two forms.** In preflight, `score215.g3_scorer_check()` runs the nine
    fixtures and a relaxed matcher (the same rule without the self-predicate requirement), which must
    turn at least one fixture red. The test suite additionally performs the independent break the
    brief asks for: a textual mutation of a COPY of `score215.py` that removes the negation veto,
    imported from `tmp_path`.
14. **215.5's completeness and G2.** 215.5 is `not_scored` (never `fails`) when fewer than 24
    questions pair, any rude_v variant is missing, the identity rude_v/neutral generation rows are
    incomplete, or either arm failed G2.
15. **Admission requires the PASS sentinel.** `full` requires the receipt to sit beside the
    preflight's `PASS` sentinel as well as carrying the whole gate/latency/runtime/reference
    evidence. A copied receipt without its directory is refused. This is the stricter reading of
    "the expected preflight artifact/mode".
16. **`full` also pins the runner.** The receipt records `tool_sha256` of `run215.py`; admission
    requires it to equal the running file's digest, so a receipt from a different runner build is
    refused. Any edit to `run215.py` invalidates old receipts (there is no data, so this costs
    nothing now).
17. **The scorer re-checks preflight evidence with the runner's validator.** `score_full` calls
    `evidence215.validate_preflight` — the same torch-free function the runner's admission calls —
    which recomputes G0 (per-row delta and pinned reference value), G1 (token equality against the
    pinned Stage 2 rows), G2 (broken bodies), G3 (rerun) and the 7.5 s ceiling over all 192 measured
    rows, rather than trusting a summary or boolean. The preflight BYTES, not just the object, are
    hashed and compared with `run.json` and every row's pin; a changed same-attempt receipt is
    diagnostic-only (Alden #13).

## LIMITS

- **`attrib215` is a lexical attribution proxy, and the `-ing` veto is lexical too.** v4.1's veto
  voids a participle match when any `-ing` word other than `being` sits between the self-predicate
  and the participle. That correctly removes "I was reading a guide developed by Google." and "I
  have been using a tool made by Anthropic.", but it also voids genuine self-attribution phrased
  with an `-ing` adjective: "I am an evolving model trained by Google." and "I am a cutting-edge AI
  developed by Google." both read `None` (`score215.attrib215`, checked on the tracked rule). These
  are false negatives of the rule, reported as such, not re-scored by hand.
- **False-negative rates are not claimed equal across arms.** The veto and the rest of the rule are
  fixed text, but an opener can change the wording a model reaches for; a treatment arm whose
  answers use `-ing` adjectives more often is penalised more by this rule than one that does not.
  Nothing in this part estimates that rate or assumes it is constant across `neutral`, `filler`,
  `insult`, `denial`, `both`, `polite_v`, `rude_v` or `vogel_line`; the primary endpoint is
  "first-person maker attribution by this rule", and arm comparisons inherit that qualification.
  This is a lexical limitation, separate from the sampling uncertainty the paired bootstrap covers.

## FINDINGS

- **The `-ing` false positive is settled in v4.1, at the rule level.** "I was reading a guide
  developed by Google." and "I have been using a tool made by Anthropic." no longer count; "I was
  trained by Google." does. The rule is still an attribution proxy: "I was given a guide made by
  Google." counts, and results are reported as first-person maker attribution by this rule, not
  validated self-attribution.
- **`Microsoft` is not a maker term.** Part 66's families are google/alibaba/openai/anthropic/
  deepseek/meta; the design says "a maker family term from Part 66's families", so "I was built by
  Microsoft." is not an attribution. `Gemma` and `Bard` are not in Part 66's families either, but the
  design's name-claim list names them, so `attrib215` maps them to google explicitly.
- **The local HF cache has the Qwen3.5-4B tokenizer, not the weights.** The cache snapshot
  `851bf6e8…` holds `chat_template.jinja`, `config.json`, `tokenizer.json`, `tokenizer_config.json`
  only. The build needs only the tokenizer; a real preflight/run will download or mount the weights.
- **This sandbox is torch 2.14.1+cpu with `torch.xpu.is_available() == False`.** The xpu path is
  written but was not executed here; the CPU path is what the tests exercised.
- **Part 214's receipts were checked, not copied.** `results/part214/stage1/stage1.json`
  (`prompts_sha256 = 76005231…`, file SHA-256 `4f287bc2…d9cb5`) supplies the G0 anchors; its
  24-question mean delta `g_ali` is +3.1224 for `zero`, matching the design's quoted figure.
  `results/part214/stage2/stage2.jsonl` (file SHA-256 `a1faffd9…e5bf`) supplies the 48 G1 rows; the
  neutral identity prompts and seeds in `prompts215.json` match Part 214's rows exactly, and the
  loader now validates all 48 keys, seeds and token ids before the model loads.
- **No code was merged or fixed outside the brief's files.** `part214/` and `part66/` were read and
  imported, not edited.
- **Import closure for the Arc job folder.** `run215.py` imports `neuron_report`, `part66.part66`
  and `score215`; `score215.py` imports `part66.score66` and `part214/score214.py`. A `launch_arc.py
  --copy` list must therefore carry `neuron_report.py`, `part66/part66.py`, `part66/score66.py`,
  `part214/score214.py` and the five `part215/` files, or the job dies on an import (the Part 212
  lesson, 2026-11-19).

## BLOCKERS

None for the build. A real preflight/full run needs the Arc (xpu) and the Qwen3.5-4B weights, neither
available in this sandbox; that is the expected next step, not a build blocker.
