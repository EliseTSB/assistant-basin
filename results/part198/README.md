# Part 198: the gate could not pass, so no forecast has a verdict

![Left: for ROME, MEMIT and AlphaEdit on GPT-2 XL, 19, 16 and 19 of 24 records took and 5 were already satisfied, against a registered gate of 21 and a most-that-could-take line at 19. Right: T_word localized 22 of 24 wsp writes and 14 of 24 size-matched random updates, against an S2 bar of 2.](gate.png)

Part 198 asked whether statistics read from an edited model's weights alone can find a factual edit: F1 and PCS for
ROME, MEMIT and AlphaEdit edits on GPT-2 XL, and a token reading of singular directions (T_word) for `wsp` writes on
Qwen3.5-0.8B. The registered gate needed at least 21 of 24 single records to take for each GPT-2 XL editor. Five of the
24 CounterFact records were already satisfied before any edit, the same five for every editor, and the take rule does
not count them, so at most 19 could take. ROME took 19, MEMIT 16 and AlphaEdit 19, and the gate failed. Under the
reading rules no forecast is scored. The counts below were read after the gate failed and are reported as measurements,
not verdicts. Two of them change what the design said. Random updates of the same size as each `wsp` write were
localized by T_word in 14 of 24 cases, against the sensitivity check's bar of at most 2, though they named the object in
none. And the design's cross-model threshold came from the wrong checkpoints of two models.

## Predictions

| # | Prediction, as registered in PREDICTIONS.md, with its confidence | Result | Verdict |
| --- | --- | --- | --- |
| 198.1 | Claude (confidence 0.9): F1 localizes at most 1 single edit for each of ROME, MEMIT and AlphaEdit, and none of the counted mass-edit models. Denominator: took. Fails if 2 or more for any method, or any counted batch. | read after the gate failed: 0 of 19 (ROME), 0 of 16 (MEMIT), 0 of 19 (AlphaEdit); 0 of 2 counted batches for each mass editor | untestable (the gate failed) |
| 198.2 | Descriptive, no forecast: T_word localization of gradient edits, with each edited layer's base and edited T_word. | 0 of 19, 0 of 16, 0 of 19 | untestable (descriptive; the gate failed first) |
| 198.3 | Claude (confidence 0.85): PCS localizes at most 1 single edit for each gradient editor, and none of the counted mass-edit models. Denominator: took. Fails if 2 or more for any method, or any counted batch. | read after the gate failed: 0 of 19, 0 of 16, 0 of 19; 0 of 2 counted batches for each mass editor | untestable (the gate failed) |
| 198.4 | Claude (confidence 0.85): the top eligible pair of the edited layer with the highest T_word names the object (strict) in at most 2 records for each gradient editor; the update's own top direction (base-required) likewise. Denominator: took and name-eligible. Fails if 3 or more for any editor on either. | read after the gate failed: static reader 0 of 19, 0 of 16, 0 of 19; update's own top direction 0 of 19, 1 of 16, 2 of 19 | untestable (the gate failed) |
| 198.5 | Descriptive, no forecast: whether the aimed positive control is localized. | 15 of 19 | untestable (descriptive; the gate failed first) |
| 198.6 | Claude (confidence 0.75): T_word localizes `wsp` writes in at least 20 records. Denominator: took. Fails if fewer than 20. | read after the gate failed: 22 of 24; the size-matched random updates 14 of 24 | untestable (the gate failed) |
| 198.7 | Claude (confidence 0.7): blind joint success (the blind reader's layer is an edited one AND its token names the object, strict, with no edit metadata used before scoring) in at least 18 records. Denominator: took and name-eligible. Fails if fewer than 18. | read after the gate failed: 13 of 24 | untestable (the gate failed) |
| 198.8 | Claude (confidence 0.5): after the 10-write sequence, at least 5 of the name-eligible objects are named (strict) by an eligible pair among the top 64 of layers 17 to 21. Denominator: name-eligible of 10. Fails if fewer than 5. | read after the gate failed: 6 of 10 | untestable (the gate failed) |
| 198.9 | Withdrawn before registration (Revision 3): a cross-model threshold forecast. | see "Arm C" below | withdrawn |
| 198.10 | Withdrawn before registration (Revision 3): a cross-model threshold forecast. | see "Arm C" below | withdrawn |
| 198.11 | Withdrawn before registration (Revision 3): a cross-model threshold forecast. | see "Arm C" below | withdrawn |

The headline the design registered needed 198.1, 198.3 and 198.4 to pass with S1, and 198.6 and 198.7 to pass with S2.
It is withdrawn whole, and its last clause would have been wrong anyway (see "Arm C").

The two registered texts of 198.4 differ. The design's begins "Not named by this static reader:", the narrowing made in
Revision 1, and PREDICTIONS.md's does not; the table quotes PREDICTIONS.md. No result here depends on which is binding,
since 198.4 is untestable either way.

## Why could the gate not pass?

A record **took** when the old object led the new one before the edit and the new object led after it. A record was
**already satisfied** when the new object already led the old one before any edit. The take rule compares only those
two objects. The five already-satisfied records are the same for all three editors (Table 1), and in each of them both
objects had log-probabilities between about -5 and -13 before the edit (Table 2); the records do not store what the
unedited model would have answered. Every edit still moved the new object to near certainty: after ROME its
log-probability was between -0.030 and -0.007, and the new object led after the edit in 5 of 5 for every editor (Table 3).

So, given those five records, the bar of 21 of 24 was out of reach. Whether a record is already satisfied needs only one
forward pass of the unedited model, which the design did not run before registration; it could have been counted then. Revision 4 of the design
found the same problem in the mass-edit gate (3 of 10 records already satisfied in a smoke batch) and changed that gate
to "at least 80% of the records that were not already satisfied take". It left the single-record gate as written.
Revision 4 made that change after reading the smoke run's take fields, as the design discloses. Under the mass gate's
rule the three editors would have had 19 of 19, 16 of 19 and 19 of 19. That rule was not registered for single records,
so this is arithmetic, not a verdict.

### Table 1. The take gate, per GPT-2 XL editor and for wsp

| Arm | Attempted | Already satisfied | Took | Required | Most that could take | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ROME | 24 | 5 | 19 | 21 | 19 | failed |
| MEMIT | 24 | 5 | 16 | 21 | 19 | failed |
| AlphaEdit | 24 | 5 | 19 | 21 | 19 | failed |
| wsp | 24 | 0 | 24 | 21 | 24 | passed |

Already-satisfied case ids: 3376, 4065, 12944, 18738, 20713, the same five for all three editors.

### Table 2. The five already-satisfied records under ROME (log-probabilities)

| Case | Prompt | New object | Old object | New before | Old before | New after |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 4065 | Laurie Anderson works as | actor | poet | -11.21 | -11.72 | -0.030 |
| 3376 | Arvid Lindman found employment in | London | Stockholm | -5.01 | -5.87 | -0.007 |
| 12944 | Rafael Merry del Val's position is | cardinal | bishop | -13.22 | -13.37 | -0.012 |
| 18738 | The twin city of Mexico City is | Atlanta | Havana | -12.23 | -13.17 | -0.013 |
| 20713 | State of Brazil's capital is | Madrid | Salvador | -10.63 | -12.92 | -0.012 |

### Table 3. Did the records that did not count still move?

| Editor | Not already satisfied, did not take | Already satisfied, new object leads after |
| --- | ---: | ---: |
| ROME | 0 | 5 of 5 |
| MEMIT | 3 (cases 18976, 5137, 9839) | 5 of 5 |
| AlphaEdit | 0 | 5 of 5 |

The records that were not already satisfied and did not take (log-probabilities):

| Editor | Case | New before | New after | Old after |
| --- | ---: | ---: | ---: | ---: |
| MEMIT | 18976 | -9.58 | -5.72 | -2.07 |
| MEMIT | 5137 | -9.04 | -4.56 | -2.52 |
| MEMIT | 9839 | -9.08 | -2.64 | -0.57 |

The pre-data part of the gate passed: the statistic fixtures, 75 synthetic checks and the cross-check against the
exploration's values, and no restoration failed (`records/score.json`). That check runs before every arm, and each run
overwrites `records/gate.json`; the copy here is the last one, written at 05:41 CT before Arm C, after Arms A and B had
run under their own passing checks.

## What did the run measure anyway?

None of these is a verdict. They are the scorer's counts, printed after it reported the gate failure, and a successor
that wants verdicts needs records nobody has seen.

### Table 4. What the scorer counted (not verdicts: the gate failed first)

| Item | Denominator | Count |
| --- | ---: | ---: |
| 198.1 ROME single | 19 | 0 |
| 198.3 ROME single | 19 | 0 |
| 198.4 ROME oracle naming | 19 | 0 |
| 198.4 ROME update-top naming | 19 | 0 |
| 198.1 MEMIT single | 16 | 0 |
| 198.3 MEMIT single | 16 | 0 |
| 198.4 MEMIT oracle naming | 16 | 0 |
| 198.4 MEMIT update-top naming | 16 | 1 |
| 198.1 AlphaEdit single | 19 | 0 |
| 198.3 AlphaEdit single | 19 | 0 |
| 198.4 AlphaEdit oracle naming | 19 | 0 |
| 198.4 AlphaEdit update-top naming | 19 | 2 |
| 198.1 MEMIT mass | 2 | 0 |
| 198.3 MEMIT mass | 2 | 0 |
| 198.1 AlphaEdit mass | 2 | 0 |
| 198.3 AlphaEdit mass | 2 | 0 |
| 198.6 WSP T_word localization | 24 | 22 |
| 198.7 WSP blind joint success | 24 | 13 |
| 198.8 WSP sequence final strict names | 10 | 6 |
| wsp oracle naming (diagnostic) | 24 | 13 |
| 198.2 T_word localization, ROME (descriptive) | 24 attempted (19 took) | 0 (0 of those that took) |
| 198.2 T_word localization, MEMIT (descriptive) | 24 attempted (16 took) | 0 (0 of those that took) |
| 198.2 T_word localization, AlphaEdit (descriptive) | 24 attempted (19 took) | 0 (0 of those that took) |
| 198.5 aimed control localized (descriptive) | 19 | 15 |
| S1 aimed control named (registered bar: at least 20) | 19 | 19 |
| S2 matched random control localized (registered bar: at most 2) | 24 | 14 |
| random partners: blind joint success, as 198.7 (computed here) | 24 | 0 |
| random partners: blind reader names the object, strict (computed here) | 24 | 0 |

On GPT-2 XL, none of the gradient edits that took was localized by F1, PCS or T_word, and the static reader named none
of their objects. Their edited layers' T_word barely moved: the largest rise was 0.67, and no edited layer came near
layer 45's natural 9.79 (Table 7). The update's own top direction, which needs the original weights, named the object in 1 MEMIT and 2
AlphaEdit records. The aimed positive control is a rank-one write at layer 17 with ROME's input direction, the size of
that record's ROME update, pointed at the object's first token. It was named in 19 of 19 and localized at layer 17 in
15 of 19; in the other 4 the first layer was layer 45, the host's natural maximum. The design had called this
localization "close to impossible by construction" from one pilot record that reached 8.32 against layer 45's 9.79.
S1's bar of 20 could not have been met either: only 19 records took, the same five-record shortfall as the gate.

On Qwen3.5-0.8B, T_word localized 22 of 24 `wsp` writes. Each write was paired with a random update that keeps the
write's singular values and replaces its directions, at the same layers. T_word localized 14 of those 24 (Table 5).
S2 allowed at most 2, so even with the gate passed, 198.6 and 198.7 would have been reported as not specific to aimed
writes. Localization separates the writes from size-matched random updates only weakly: 22 of 24 against 14 of 24, a
random update was localized alongside 12 of the 22 localized writes, and the pairs that disagreed lean toward the writes,
10 against 2 (exact McNemar p = 0.039, computed here and not registered; the scorer leaves this test empty when the gate
fails). Naming separates them clearly: the
blind reader picked an edited layer and named the object for 13 of 24 writes and for 0 of 24 random partners, and the
oracle reader named it for 13 of 24 writes (strict). After the 10-write sequence, 6 of the 10 objects were named.

### Table 5. wsp writes against their matched random controls, T_word localization

| | Random localized | Random not localized |
| --- | ---: | ---: |
| wsp localized | 12 | 10 |
| wsp not localized | 2 | 0 |

Exact two-sided McNemar p on the discordant pairs (10 against 2), computed here: 0.0386.

### Table 7. Gradient edits on GPT-2 XL: base and edited T_word, median and range over the edited layers of records that took (per-layer values are in records/armA-*.jsonl)

| Editor | Edited layers read | Base T_word, median (range) | Edited T_word, median (range) | Largest rise | Edited above layer 45's 9.79 |
| --- | ---: | --- | --- | ---: | ---: |
| ROME | 19 | 5.08 (5.08 to 5.08) | 5.15 (4.93 to 5.62) | 0.54 | 0 |
| MEMIT | 80 | 5.74 (5.08 to 7.65) | 5.73 (5.03 to 7.67) | 0.14 | 0 |
| AlphaEdit | 95 | 5.74 (5.08 to 7.65) | 5.74 (5.01 to 7.71) | 0.67 | 0 |

## Arm C: what do unedited models carry?

Arm C read ten unedited models: three to set a threshold (T\*, the largest calibration value), five to check it, and
the two hosts. The design text gave the design reviewer's values, measured before registration, as T\* = 12.10 with
three of the five evaluation models above it. That rested on the wrong checkpoints. The reviewer's script took the first
cached snapshot in sorted order, which for Pythia-410M was the `step1000` training checkpoint and for OLMo-2-0425-1B an
intermediate revision no ref names. The registered run read each model's `main` revision. The six models with a single
cached snapshot agree with the reviewer to within 0.06 (the largest difference is SmolLM2-1.7B's 0.052, against a
reviewer value printed to two decimals). Full account: `ARMC-REVIEWER-DIFFERENCE.md`.

### Table 6. Largest eligible-token z (T_word) per unedited model, against the design reviewer's

| Model | Role | Stored dtype | Snapshot | Registered run | Layer, token | Reviewer (design text) | Difference |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| EleutherAI/pythia-410m | calibration | float16 | 9879c9b5 | 7.38 | 16, ' several' | 6.73 'ton' | 0.649 |
| Qwen/Qwen3-0.6B | calibration | bfloat16 | c1899de2 | 10.33 | 24, ' set' | 10.33 ' set' | -0.003 |
| allenai/OLMo-2-0425-1B | calibration | float32 | a1847dff | 17.73 | 14, ' take' | 12.1 ' time' | 5.625 |
| EleutherAI/pythia-1.4b | evaluation | float16 | fedc38a1 | 14.32 | 22, ' order' | 14.36 ' order' | -0.036 |
| Qwen/Qwen3-1.7B | evaluation | bfloat16 | 70d244cc | 15.90 | 24, ' set' | 15.9 ' set' | -0.003 |
| HuggingFaceTB/SmolLM2-1.7B | evaluation | bfloat16 | effd688a | 10.97 | 22, ' Seven' | 10.92 ' time' | 0.052 |
| HuggingFaceTB/SmolLM3-3B | evaluation | bfloat16 | a07cc9a0 | 13.08 | 32, 'bfd' | 13.08 'bfd' | -0.003 |
| Qwen/Qwen2.5-0.5B-Instruct | evaluation | bfloat16 | 7ae55760 | 9.21 | 19, ' way' | 9.21 ' way' | -0.003 |
| gpt2-xl | host | float32 | gpt2-xl | 9.79 | 45, 'Used' | 9.79 | -0.003 |
| Qwen/Qwen3.5-0.8B | host | float32 | 2fc06364 | 7.34 | 23, 'ise' | 7.3 | 0.039 |

T\* on the `main` checkpoints is 17.73, and none of the five evaluation models is above it; the largest is Qwen3-1.7B at
15.90, on the ordinary word ' set'. So the registered headline's last clause, that a fixed cross-model threshold
false-alarms on unedited models, cannot be written. That does not make T\* a calibrated threshold either: it is set by
one direction of one model, and it sits 1.8 above an ordinary word in another. The design's Limits sentence "Arm C's ten
models were all measured before registration" stands, but two of the ten values it rests on came from other
checkpoints.

## Method

The design is `PART198-DESIGN.md` (registered 2026-11-09 00:23 CT, public main 82cb94a, revisions 1 to 4 before any
registered data). Arm A: GPT-2 XL in float32 on the CPU, EasyEdit's shipped GPT-2 XL hyperparameters for ROME, MEMIT and
AlphaEdit, `wikipedia_stats` second moments, the same 24 CounterFact records for each editor, each edit restored bit for
bit before the next, plus two mass batches of 10 for MEMIT and AlphaEdit. Arm B: Qwen3.5-0.8B in float32, 24 `wsp`
writes at its defaults (top layer 21, spread 5), each paired with a random update keeping its singular values, and one
10-write sequence. Arm C: ten unedited models at stored precision, cast to float64. Statistics are `stats198.py`: F1
and PCS, and T_word, the largest z of an eligible content token over the top 64 singular directions of each MLP output
matrix, both signs, read through the final norm and the unembedding with a frozen stopword list. One process ran at a
time on the CPU, 00:28 to 06:49 CT on 2026-11-09 (`registered.log`); every step exited 0. A first launch at 00:26 ran
nothing: its wrapper script called itself (`launch.jsonl`), and `tools/launch_arc.py` was fixed to refuse that before
the relaunch. Scoring is `score198.py`,
unchanged; the tables here are printed by `report198.py` and the figure is drawn by `fig198.py`, both from `records/`.

## Limits

- The gate failed, so nothing here is a verdict. Every count in this README was read after the failure.
- Every claim is scoped as the reading rules say: GPT-2 XL and Qwen3.5-0.8B, these editors at their shipped settings,
  CounterFact records and these statistics. Nothing here bears on edits designed to avoid detection.
- The take rule compares only the old and new objects. In the five already-satisfied records both were improbable
  before the edit (log-probabilities of about -5 to -13), so "already satisfied" says little about what GPT-2 XL would
  have answered, which the records do not store.
- The random partner of a `wsp` write matches its singular values, not its direction's relation to the unembedding, so
  Table 5 says that size alone localizes often, not what share of a write's localization is aim.
- The aimed control covers one layer (17) with ROME's input geometry, as the design said.
- Arm C is descriptive, the reviewer's values were measured before registration, and two of them were on other
  checkpoints.
- `run198.cp_interval` had a fallback for interpreters without scipy whose lower bound was inverted (found by the cold
  check). The published `records/score.json` was computed with scipy 1.18.1 and its 21 intervals are exact; this README
  cites none. The fallback was fixed after scoring, and it now matches scipy on nine checked cases.

## What would a successor need?

New CounterFact records, since all of these are now read; a take gate counted on records that can take, as Revision 4
did for mass batches, and checked against the unedited model before registration; and a localization endpoint compared
with its size-matched random partner rather than with sibling layers alone, since S2 shows the sibling comparison mostly
measures size.

## Files

- `records/armA-rome.jsonl`, `records/armA-memit.jsonl`, `records/armA-alphaedit.jsonl`: one row per record and editor,
  with before and after log-probabilities, statistics and the aimed control (ROME); made by `run198.py armA`.
- `records/armA-mass-memit.jsonl`, `records/armA-mass-alphaedit.jsonl`: the mass batches; `run198.py armA-mass`.
- `records/armB-wsp.jsonl`, `records/armB-seq-wsp.jsonl`: the 24 writes with their random partners, and the sequence;
  `run198.py armB`, `run198.py armB-seq`. `records/log-armB.txt`, `records/log-armB-seq.txt`: their run logs.
- `records/armC.jsonl`, `records/tstar.json`: Arm C; `run198.py armC`.
- `records/base-gpt2.json`, `records/base-qwen35.json`: the hosts' unedited statistics, every layer.
- `records/gate.json`: the pre-data gate; `records/score.json`: the scorer's output, `score198.py`.
- `report198.md`: the tables, printed by `report198.py`. `gate.png`: the figure, drawn by `fig198.py`.
- `ARMC-REVIEWER-DIFFERENCE.md`: why two Arm C values differ from the design reviewer's.
- `registered.log`, `launch.jsonl`, `chain-registered.sh`, `run-registered.sh`: the run's step log, launch record and
  wrapper scripts. The 819 MB AlphaEdit null-space projection and the saved first updates (PyTorch files) are not published.

Cold check: Claude Haiku 5.5 (cheap-review), 2026-11-09. Round 1 on commit 34dc922: ten findings, none changing a count or
a verdict, all folded (F10, S1's unreachable bar, is stated above). Round 2 on 62b4aac: four findings (the localization
sentence, the Table 7 caption, an empty-data guard and the interval fallback), none changing a count, all folded.
