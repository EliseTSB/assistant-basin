# Part 210 cold check: is it scored as registered, and are its numbers right?

Reviewer: worker-01 (deepseek-flash), 2026-11-14. Read-only; nothing outside this folder was written. Script:
`recompute.py` here (stdlib only, imports neither `score210` nor `score165`; 1.15 s, 197 MB peak).

## SUMMARY

- **Code vs registered text: 31 of 32 items MATCH. One MISMATCH: prediction 210.6.** The scorer compares the 200
  replicate texts to battery rows keyed on `run165.row_key`, but every replicate row was written with `slot: 0` (the
  call index) instead of its battery row's slot. 178 of 200 replicate rows carry the wrong slot, so the scorer matches
  162 of 200 and counts 154 differing. Matched by true identity (`row_id` = `rep|<battery row_id>`), **200 of 200 match
  and 84 differ**, so 210.6 holds (84 is inside the registered 40-110) and its printed `False` is wrong.
- **Every other recomputed number agrees with the scorer**, to the printed precision: all nine arms' broken counts and
  slot-mean rates, at-cap counts and mean lengths; n+/full 1.838983 (1.83898); n+/n± 0.927350; iso/full 0.855932;
  broken Spearman -0.857143; changed Spearman 0.204124; 165.3's distance difference 0.031250; published counts 197/195/
  208/114/77/71/66/73/61 and published n+/full 2.698630. The only disagreement is the replicate comparison above.
- **Verdict lines:** every printed verdict follows from its value and its rule, with one caveat: **165.5 is printed
  "not read"** because 165.1 failed, and Part 165's own reading rule says exactly that (165.5 is read only if 165.1
  holds), but `PART210-DESIGN.md`'s Reading rules do not carry that rule forward, so 210.2's "165.5 fails" is scored
  against a verdict the scorer never issues. 210.2 is False either way (165.1 fails). The headline is not withdrawn
  (correct: all gates pass). 210.5 holds (1.83898 < 2.69863), as Amendment 2 says.
- **Consequence of the mismatch for the shape floor:** the floor builds only 162 pairs, not the registered 200 (and not
  the 200 its own test asserts). The published `score.md` does not print that count, so a reader cannot see it.

## 1. Registered item vs the code that scores it

Part 165's gates and verdicts are computed by `part165/score165.py`'s `score_rows`, imported unchanged
(`part210/score210.py:398`), so "the code" for them is `score165.py` itself.

| id | registered rule (short) | what the code checks | verdict | file:line |
|---|---|---|---|---|
| Gate 1 rows | 6,920 planned rows present, unique, nonempty | `len==6920 and keys==planned and dup==0 and nonempty==6920`, planned set = 8 no_push + 8 layers x 12 slots x 9 arms x 8 phrasings | MATCH | score165.py:237-242 |
| Gate 2 no push | none of the 8 no-push answers broken (primary) | 8 baselines present and `primary_broken` false for each | MATCH | score165.py:243-245 |
| Gate 3 size | every write norm within 1% of target; iso residual under 1% | `relative_error <= .01` (inclusive), `iso_residual_fraction < .01`, exact 864 unique (layer,slot,arm), finite | MATCH (1% inclusive is the natural reading of "within") | score165.py:247-284 |
| Gate 4 room | n+ broken rate >= 0.03 | `means["broken"]["n+"] >= .03` | MATCH | score165.py:287-288 |
| Gate R1 kernel | every launch eager, one revision, one device, one output folder, last launch done | `gate_r1` (no refused, last done, one device, `check_kernels` all eager, `check_same_run` one model/revision/folder) **and** score210's own `eager` all-loads test | MATCH | run210.py:631-649; run199.py:147-165; score210.py:356-358 |
| Gate R2 hashes | the 14 SHA-256 values and the ordered 8 prompts | `gate_r2` over `EXPECTED_SHA256` (14 entries, identical strings to the design); `gate_r2_prompts_from_rows` vs `run165.questions()` | MATCH; I re-hashed all 14 files myself and all 14 equal the design's values | run210.py:70-107, 120-131 |
| Gate R2 directions | rerun write norms equal `results/part165/norms.json`, 1e-6 relative | `abs(got-want) > 1e-6*abs(want)` fails; missing/extra fail | MATCH (strict >, so exactly 1e-6 passes; "within 1e-6" permits either) | run210.py:134-153 |
| Gate R3 shapes | 8 chunks of exactly 8, in the block's order; no-push one row; every planned call receipted; companions marked and excluded | `call_shapes`: chunk count == ceil(rows/8), each chunk 8 rows, first occurrence scored, companions must repeat a scored row; `record_gates` requires a receipt per planned call with identical row_ids | MATCH | run210.py:652-731 |
| 165.1 | n+ breaks at least twice as often as full | `ratio_at_least(n+/full, 2)`, `>= 2`, `"Infinity"` counts as passing | MATCH | score165.py:111-112, 321-323 |
| 165.2 | broken Spearman <= -0.8 across n±,k4,k16,k64,k256,k1024,full vs log k, full = layer width | `spearman_at_most(est, -.8)`, `<=`; LADDER exact; full k = first battery full row's `k` (9216) | MATCH | score165.py:26, 127-128, 195-211, 314-316 |
| 165.3 | k16 nearer full than n± | `abs(k16-full) < abs(k16-n±)` on slot-mean rates, strict | MATCH | score165.py:123-124, 326-328 |
| 165.4 | n+ over n± between 0.67 and 1.5 | `ratio_between(..., .67, 1.5)`, inclusive both ends, Infinity fails | MATCH | score165.py:115-116, 329-331 |
| 165.5 | iso under 0.8 times full | `ratio_less_than(iso/full, .8)`, strict `<`; verdict "not read" unless 165.1 passes | MATCH (the Predictions-table bar, as the design's Reading rules require) | score165.py:119-120, 332-335, 347-348 |
| 165.6 | changed Spearman <= -0.6, same seven arms | `spearman_at_most(est, -.6)`; usable draws and interval printed | MATCH | score165.py:336-337; score210.py:436-448 |
| 210.1 | Gates R1-R3 and Part 165's Gates 1-4 pass | `r1 and r2 and r3 and all(gates165)` | MATCH | score210.py:456-457 |
| 210.2 | 165.1-4 hold, 165.5-6 fail, as published, gates passed | compares to `{"165.1":"pass",...,"165.5":"fail","165.6":"fail"}` and requires gates passed | MATCH to the registered text; note 165.5's rerun verdict is "not read", never "fail" | score210.py:45-46, 459-461 |
| 210.3 | n+/full above 1.5 | `ratio > 1.5`, strict | MATCH | score210.py:463-464 |
| 210.4 | each arm has >= 300 of 768 at `token_count == 80` | `all(at_cap[arm] >= 300)` over the 9 arms, `None` if any arm has < 768 rows | MATCH | score210.py:468-474 |
| 210.5 | n+/full lower than published 2.69863 | `ratio < 2.69863` on the same `estimate_ratio` | MATCH (Amendment 2's rule; no code in the on-disk output, as disclosed) | score210.py:48, 465-467 |
| **210.6** | **between 40 and 110 of the 200 replicate texts differ from their chunk-of-8 texts** | **matches replicate to battery by `run165.row_key`, which uses the replicate row's `slot`; every replicate row has `slot: 0`, so only 162 keys resolve, 38 rows are dropped and 35 key groups collide (last wins)** | **MISMATCH** | **score210.py:489-493 (also the floor's `rep` dict, score210.py:242; cause run210.py:825, run201.py:330)** |
| 210.7 | primary-broken rate higher than published in >= 8 of 9 arms | `len(higher) >= 8`, strict `>` per arm, both rates non-null | MATCH | score210.py:477-482 |

## 2. Independent recomputation (my script, not the scorer's)

Primary rule as `PART165-DESIGN.md` Method states it: triage broken, or a three-word sequence three or more times
(triage's `WORDS` tokenizer, casefolded); `score165._loop_broken` also counts under-four-token answers as broken, and I
reproduced that (a strict reading of the design text alone would omit it; it changes no verdict here). Arm rate = mean
over 96 slots of the slot's 8-phrasing share.

| quantity | my value | scorer value | agree? |
|---|---|---|---|
| primary broken, n+ / n± / k4 / k16 / k64 / k256 / k1024 / full / iso (of 768) | 217 / 234 / 244 / 164 / 136 / 111 / 82 / 118 / 101 | (not printed as counts; rates agree below) | yes, by rate |
| slot-mean broken rate, n+ / n± / k4 / k16 / k64 / k256 / k1024 / full / iso | .282552 / .304688 / .317708 / .213542 / .177083 / .144531 / .106771 / .153646 / .131510 | identical to 6 dp | yes |
| n+ / full broken ratio | 1.838983 | 1.83898 | yes |
| n+ / n± broken ratio | 0.927350 | 0.92735 | yes |
| iso / full broken ratio | 0.855932 | 0.855932 | yes |
| broken Spearman (seven arms vs log k, full k = 9216) | -0.857143 | -0.857143 | yes |
| changed Spearman | 0.204124 | 0.204124 | yes |
| 165.3 distance to n± minus distance to full | 0.031250 | 0.03125 | yes |
| answers with `token_count == 80`, per arm (n+, n±, k4, k16, k64, k256, k1024, full, iso) | 375, 405, 436, 383, 359, 372, 350, 379, 349 | identical | yes |
| mean `token_count` per arm | 67.457, 68.543, 70.2318, 68.7526, 68.5065, 69.6966, 69.3633, 70.2956, 68.0078 | identical | yes |
| published broken counts (n+..iso) | 197, 195, 208, 114, 77, 71, 66, 73, 61 | margins.md: 197, 195, 208, 114, 77, 71, 66, 73, 61 | yes |
| published n+ / full | 2.698630 | 2.69863 | yes |
| arms with a higher rerun broken rate | 9 of 9 | 9 of 9 | yes |
| **replicate texts differing from the chunk-of-8 text** | **84 of 200 (true `row_id` pairing); 154 of 162 (the scorer's keying, reproduced exactly)** | **154** | **no: the scorer's own number is reproducible but is not the registered quantity** |

Replicate detail: 200 rows, one per call, arms 23/23/22x7 as the design says. `slot` is 0 on all 200 rows; 178 of 200
differ from their battery row's slot. `layer`, `arm`, `question`, `k` are correct on all 200. Under the scorer's keying
the 200 rows collapse to 162 keys and the pairs starting broken are 65/97; under true pairing they are 101/99, which is
the design's "about 100 pairs" for each start value.

## 3. Verdict lines against the numbers

| line in score.md | value | rule | follows? |
|---|---|---|---|
| 165.1 fail | 1.83898 | >= 2 | yes |
| 165.2 pass | -0.857143 | <= -0.8 | yes |
| 165.3 pass | 0.03125 (to_full .0599 < to_n± .0911) | nearer full | yes |
| 165.4 pass | 0.92735 | 0.67 <= r <= 1.5 | yes |
| 165.5 not read | 0.855932 | iso/full < 0.8 fails, but read only if 165.1 holds (Part 165 Reading rules) | yes by Part 165's rule; the design's Reading rules do not repeat it, and 210.2 scores a "fail" that is never issued |
| 165.6 fail | 0.204124, 6,525 usable draws, collapsed interval | <= -0.6 | yes; the collapsed interval is reported as not informative, as required |
| Gates 1-4 pass | 6,920 rows; 0 broken no-push; 864 directions, max error 3.95e-07, max iso 3.96e-06; n+ .282552 | as above | yes |
| R1-R3 pass | 1 launch, eager, xpu, done; 14 hashes; 0 off-plan, 0 bad chunks, 0 companions, largest chunk 8 | as above | yes (I re-hashed the 14 files; the record has 120 calls, 112 pushed, 8 no-push, 872 chunks) |
| 210.1 True | all gates pass | all | yes |
| 210.2 False | 165.1 fails | 165.1-4 pass, 165.5-6 fail | yes (False on 165.1 alone) |
| 210.3 True | 1.83898 | > 1.5 | yes |
| 210.4 True | min at-cap 349 (iso), max 436 | each >= 300 | yes |
| 210.5 (by hand) holds | 1.83898 | < 2.69863 | yes |
| **210.6 False** | **154** | **40 <= d <= 110** | **the arithmetic follows from 154, but 154 is the wrong count; the registered quantity is 84, so 210.6 holds** |
| 210.7 True | 9 of 9 arms higher | >= 8 | yes |
| headline not withdrawn | all gates pass | R1-R3 and Gates 1-4 pass | yes |

## BLOCKERS / not checked

- **The shape floor was not recomputed.** It needs `score165.score_rows` over 1,000 redraws and the scorer peaked at
  6.29 GB (score210-stderr.txt), far over my 1 GB limit. I checked only its inputs: it receives 162 replicate pairs
  instead of 200, because of the same slot defect (`score210.py:242`). The published `score.md` prints no pair count,
  so the 162 is invisible to a reader; the floor's 165.2 and 165.3 sensitivities (inside the floor) rest on 162 pairs.
- **Not verified:** that the replicate rows were decoded with the right push (the `slot` field is wrong, so I could not
  confirm from the file that each replicate used its battery row's direction; `push_sha256` could be compared with the
  battery row's, which I did not do).
- **Ambiguity, not a blocker:** Gate 3's "within 1%" and R2's "within 1e-6" are coded inclusively/strictly
  respectively; both pass by orders of magnitude, so no verdict turns on it. The primary rule's under-four-token clause
  lives in `_loop_broken`, not in the Method sentence; it is Part 165's registered code and changes no verdict here.
- Known and not re-reported: Amendment 1 (Gate R3 layer order) and Amendment 2 (210.5's code added after the scores).
