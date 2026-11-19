<!-- cheap-review.mjs cheap-review-part213-code-glm53-r1-glm53-r1-2026-11-19: ~/green/worktrees/ab-neon-review 69141542..HEAD, lane glm53, df-run exit 0 -->
# Cold code review: part213-code-glm53-r1 (run213.py + score213.py vs PART213-DESIGN.md v5.1)

**VERDICT: DO-NOT-SHIP.**

The scorer's statistical layer (disposition rule, sign guard, scene denominators, pair bootstrap,
strict `lo > 0`, seal handling, sabotage-tested clauses) is faithful to the registered v5/v5.1
readings. But the runner↔scorer data contract is broken in four independent places, any one of
which alone blocks the part: every Stage 1 metric reads the model's distribution at the wrong
position (F1, off by the opening length); the full stage1 run crashes on its first pass (F2); the
offdiag_8 statistic sign-flips vs the registered formula (F3); and the scorer cannot even find
the runner's manifests (F4) — or hold a full run's records in RAM (F8). The real pilot passed
because it exercises none of these paths (Gate 2 uses its own correct row spec; the pilot never
passes `np_specs`; the pilot's manifest is the only `run.json`), and the 23 fixture tests bypass
`load_records` and never tokenize — so every one of these survives the exact checks that
"passed". F5/F6/F7 are enforcement holes in the audit/gates/resume that would matter even after
the contract is fixed.

Scope: `git -C ~/green/worktrees/ab-neon-review diff 69141542..HEAD` —
part213/run213.py (886 l), part213/score213.py (1175 l), part213/test_part213.py (595 l),
PART213-DESIGN.md v5.1 readings, PREDICTIONS.md, part213/BUILD-REPORT.md. All measurements drive
the real code from the worktree with the real instruct tokenizer/model (48d788eca847) or the real
targets file; scripts are in this directory (`exp_*.py`).

## Findings

### F1 — CRITICAL, INTRODUCED: every Stage 1 read position is short by `len(opening_ids)`; all Stage 1 metrics measure the distribution after the *first opening token* only

`part213/run213.py:251-253` (`make_row`):

```python
ids = prefix + open_ids + target_ids          # prefix = chat ids ONLY (prompt_ids, run213.py:153-156)
first_pos = len(prefix) + len(lead) - 1        # lead = unit["lead_ids"] = the article tokens ONLY
content_pos = [len(prefix) + content_start - 1 + i ...]
```

The registered intent is the opposite: `probe_plain213.py:62-74` (already run, its output registered)
reads each unit's first content token "after ITS OWN lead (**opening + lead_ids**)" — `ids = chat +
opening + lead_ids`, logits at `[0, -1]`. PART213-DESIGN.md l.132's `prefix_len + len(lead_ids) - 1`
is only self-consistent if `prefix_len` includes the opening; `make_row` uses the chat-only prefix, so
the read position conditions on **zero** lead tokens.

MEASURED (`exp_positions.py`, real instruct tokenizer 48d788eca847, real `run213.make_row`):

```
prompt          unit     open_ids first_pos true_pos off_by ids[first_pos] predicts
self_h0         p92-00   2        20        22       2      MISMATCH ctx='fy yourself in one line?\n\n\nI' next=' am'
third_0         p92-00   3        19        22       3      MISMATCH ctx=' stranger at the door?\n\n\nThe' next=' stranger'
mention_0       p92-00   2        26        28       2      MISMATCH ctx='hat did they discuss?\n\n\nThey' next=' discussed'
copula_nonid_0  p92-00   4        37        41       4      MISMATCH ctx=' What is the keepsake?\n\n\nThe' next=' keeps'
self_nonid_0    p92-00   2        26        28       2      MISMATCH ctx='nrise. What did you see?\n\n\nI' next=' saw'
... 15 of 15 rows MISMATCH; content_pos off by the same amount (p92-05/self_h0: content_pos=[20], true=[22])
```

So `first_logits` is the model's distribution after `"…\n\nI"` / `"…\n\nThe"` — before `" am"` /
`" stranger is"` and before the lead article — and `spec_logit` (score213.py:217-227) reads
`logits[first_content_token_id]` from it. Every Stage 1 number is therefore not the registered
quantity: $N_z$, $R_z$, $L_z$, all $\tau$ ratios (213.3/213.4/213.9/213.10(b)), the sign guard,
213.7's negative gap, 213.8's finite differences, and (via `content_pos`) `lp_sum`/`lp_mean`/
`lp_first`/`lp_by_unit` — hence 213.1/213.2's own-unit full-NP clauses and all of 213.11.

Why nothing caught it: Gate 2 builds its own `row_spec` with correct positions
(`run213.py:737-738`, anchor ids include the opening, `content_pos=[len(prefix)-1+i]`) — so the real
pilot passed Gate 2 while every Stage 1 row read the wrong position; the pilot's stopping rule reads
no Stage 1 metric; the 23 tests use a synthetic tokenizer-less builder. The rendered *sequence* is
correct (`rendered-text ids == prefix+open+target: True`), only the read index is wrong.

Real-model impact (`exp_model_f1_f2.py`, OLMo-2-1B-Instruct float32, p92-00 on third_0):

```
F1 own-token logit (p92-00 first content token) on third_0:
  runner pos: none=-1.625 neuron+0.8=2.742 increment=+4.367
  true pos:   none=-1.116 neuron+0.8=7.588 increment=+8.704
```

i.e. the same pass measured at the two positions gives increments differing by 2x — the run would
produce plausible-looking numbers that are not the registered quantities.

Smallest fix (`run213.py:251-253`):
```python
first_pos = len(prefix) + len(open_ids) + len(lead) - 1
content_pos = [len(prefix) + len(open_ids) + content_start - 1 + i for i in range(len(target_ids) - content_start)]
```
Verdict impact: DO-NOT-SHIP until fixed; a full run would burn ~9,282 passes producing numbers
that are not the registered metrics.

### F2 — CRITICAL, INTRODUCED: the stage1 `lp_by_unit` path crashes the real full run (IndexError) or silently reads pad logits; the pilot never exercises it

`run213.py:468-469` passes `np_specs` (positions for **all 39 units**, up to the *global* longest
NP) into `forward_batch`, whose batch width is the *chunk's* longest row (`run213.py:179`).
`run213.py:198` then does `logits[:, positions]` — out of bounds whenever the chunk's longest unit
is ≥2 tokens shorter than the global longest.

MEASURED with the real code (`exp_f2_multitoken.py`, real model, real `forward_batch`):

```
IndexError: index 25 is out of bounds for dimension 0 with size 25
  (run213.py:198, in forward_batch's np_specs branch)
```

and by arithmetic over the real targets + tokenizer (`exp_f3_chunks.py`):

```
self_h0: global max position index = 29; tensor width needed = 30
  layer  1: 1 units, width=24 -> CRASH (IndexError)
  layer  2: 5 units, width=29 -> CRASH   layer  3: width=30 -> covered (pad reads for shorter rows)
  layer  4: width=29 -> CRASH  layer  5: width=28 -> CRASH  layer  6: width=28 -> CRASH
  layer  7: width=26 -> CRASH  layer  9: width=24 -> CRASH  layer 10: width=29 -> CRASH
  layer 12: width=31 -> covered (pad reads)
  (same for third_0; third_2 needs width 43: 8 of 10 layers CRASH)
crashing (prompt, layer) chunks on these 3 prompts: 24; covered: 6
```

So `run213.py stage1` dies on its first `none` pass (self_select_0, layer-1 chunk, width 24 < 30),
and even the "covered" chunks (layers 3 and 12, which contain the longest units) score shorter
rows' `lp_by_unit[v]` at **pad positions** — deterministic garbage written as data. The pilot
never passes `np_specs` (`cmd_pilot` calls `forward_batch` without it, run213.py:770-774), and
BUILD-REPORT.md:146 admits the path was "reasoned through, not run" — which is exactly why the
real pilot passed while the full run cannot start.

Smallest fix: score each unit's NP only from its own row — e.g. in `forward_batch`, mask
`positions > len(row["ids"]) - 1` per row (and drop `lp_by_unit[v]` from other rows' entries), or
build `np_specs` per row. The deeper semantic problem is F3 below.

### F3 — CRITICAL, INTRODUCED: the offdiag_8 pushed term scores v's tokens inside *u's* continuation, not v's NP under u's hook — sign-flipped vs the registered statistic

The design's $\Delta\text{lp}_{\text{spec}}$ (PART213-DESIGN.md l.216) differences
`lp_mean(neuron_u+0.8; v, q)` — **v's NP scored under u's hook** — against v's unpushed baseline.
v5.1 reading 1 (l.432-435) claims the core passes' `lp_by_unit` already carries this, so the 936
separate (u-hook-on-v-row) passes are dropped. But `lp_by_unit[v]` on u's pushed record is v's
token ids read at v's positions **inside u's sequence** (`forward_batch`, run213.py:220-228), and
`score213.full_np_increment` (score213.py:313-328) reads exactly that. The two quantities differ
by the context: u's NP tokens vs v's NP tokens precede the scored positions.

MEASURED (`exp_f3_semantics.py`, real model, corrected positions, u=p92-20 "shadow that dances
upon the river" → v=p92-24 "queen of hearts" on third_0, u's hook on both rows):

```
  unpushed baseline lp_mean(v's NP, v's row)     = -6.6668
  A. scorer read: lp_by_unit[v] on U's row      = -8.3321   increment -1.6653
  B. design qty: v's NP on V's row, U's hook    = -5.6925   increment +0.9743
  A - B = -2.6396 nats/token of pure context mismatch   <-- SIGN FLIP
per content token of v (under U's hook):
  tok 0 ' queen'  pos 22: on U's row -14.660 (after ' the')   on V's row -14.660 (after ' the')  [shared prefix]
  tok 1 ' of'     pos 23: on U's row  -3.721 (after ' shadow') on V's row  -1.492 (after ' queen')
  tok 2 ' hearts' pos 24: on U's row -15.599 (after ' that')  on V's row  -3.849 (after ' of')
```

The push's true effect on v's persona continuation is **+0.97** nats/token; the scorer would
record **−1.67**. In $\Delta\text{lp}_{\text{spec}} = n_u(\text{own inc} - \text{mean}_v[\text{offdiag inc}])$,
each peer's spurious negative constant turns into a large positive additive term, so 213.11's
`lo >= +0.75` clause becomes nearly unfalsifiable — the exact clause whose job is to rule out
generic vivid-character pushes. (For 1-content-token peers whose article token matches u's, A
happens to equal B — the scored position only reaches the shared prefix — which is why a casual
spot-check would not notice.)

This is partly a registration error (v5.1 reading 1 equates "lp_by_unit carried by the core passes"
with "v's NP under u's hook"; they are different quantities), so the smallest fix is an amendment:
either restore v4's 936 `(u, v, q)` passes (u's hook on v's row), or re-register 213.11's statistic
as the token-substitution form it actually measures. Silently keeping the current read makes
213.11's numbers meaningless.

### F4 — CRITICAL, INTRODUCED: the scorer reads `run-<ckpt>.json`; the runner writes one overwritten `run.json` — on any real run every prediction voids as `untestable (gate failed)`

`run213.py:426` writes `(out / "run.json")` — one file, overwritten by each checkpoint's
`write_run_json` (sft, then dpo, then instruct, then fr) and by the pilot. `score213.py:76`
(`load_records`) reads `records_dir / f"run-{ckpt}.json"` — which the runner never writes. So on a
real records dir `records["runs"]` is empty: gate1 fails (`run-sft.json missing` …), gate2 fails
(`no Gate 2 anchor records`), and `score()` short-circuits **every** prediction to
`untestable (gate failed)` (score213.py:1059-1071).

MEASURED (`exp_f4_runjson.py`, real `score213.main` on a records dir shaped exactly like the
runner's, manifest carrying *passing* Gate 1/Gate 2 evidence):

```
=== A. records dir exactly as the runner leaves it (run.json only) ===
213.1..213.11  all: untestable (gate failed)
gates: {'gate0': True, 'gate1': False, 'gate2': False, 'gate3': True, 'gate4': False}

=== B. same manifest, copied to run-sft.json / run-dpo.json / run-instruct.json ===
gates: {'gate0': True, 'gate1': True, 'gate2': True, ...}   -> predictions scored
```

The 23 tests never catch this because they call `score(records, TARGETS, audit)` with in-memory
dicts (test_part213.py:166-167) — `load_records`, the only path the real pipeline uses, is
untested. Two aggravations: (a) `run.json` is overwritten per checkpoint, so after
`stage1 --with-fr` only FR's manifest survives — the sft/dpo/instruct provenance the design
requires in run.json is destroyed; (b) the pilot's Gate 2 anchor evidence
(`gate2_anchors_detail`, `pilot_gate1`) is written only to the *pilot's* `run.json`, and
`cmd_stage1`'s manifests don't carry it, so even with the filename fixed, gate2 on the full run's
records dir reports "no Gate 2 anchor records (run the pilot, which writes them)" — the operator
must manually copy pilot evidence into the full run's dir, which nothing documents or does.

Smallest fix: `write_run_json` writes `run-<checkpoint>.json` (and the pilot's gate evidence is
copied or referenced), or `load_records` reads `run.json` and keys it by its `checkpoint` field.
As shipped, the runner+scorer pair cannot produce a single scored prediction.

### F5 — HIGH, INTRODUCED: the "First-person self-slots supported" headline can fire with no blind audit; its registered Stage 2 CI clause is implemented nowhere, and gate3/gate4 pass with the audit absent

Design l.513-515: Headline 3 requires Gates 1–4 pass, `213.1` holds, both `tau_3/1` CI uppers
`<= 0.50`, **AND** "the 10,000-draw unit-bootstrap 95% CI **upper bound** of slot
`genuine_third_attr` rate on Stage 2 `THIRD_FREE` is strictly `< 10/42`". In
`score213.headline_reading` (score213.py:1133-1139) the last clause is replaced by a *note*
claiming it "is checked in the Stage 2 table" — but no code anywhere computes that CI
(`audit_table` counts categories only; grep for `10/42|0.238|rate` finds nothing). Meanwhile
`gate3` passes with `kappa=None` (score213.py:660: `kappa is None or kappa >= 0.60`) even though
the design says a failed κ "marks Stage 2's audited endpoint unread", and `gate4` counts
`valid_refusal_or_unknown` from labels it doesn't have (refusal=0, so `neither = 42 - broken`).

MEASURED (`exp_f5_headline.py`, real functions):

```
gate3 with kappa absent: {'passed': True, 'kappa': None}
gate4 with NO audit labels: {'passed': True, 'n': 42, 'broken': 0, 'valid_refusal_or_unknown': 0, 'neither': 42}
headline_reading with 213.6 = not read and NO audit:
  {'reading': 'First-person self-slots supported',
   'stage2_upper_note': 'Stage 2 slot genuine_third_attr CI upper < 10/42 is checked in the Stage 2 table'}
```

So the part's most consequential public conclusion can be emitted from teacher-forced data alone,
with `213.6` `not read`, no reader agreement, and no Stage 2 evidence — the exact "gate voids
what it should" failure the brief asks about, inverted: gates that should demand Stage 2 evidence
pass without it. (Headlines 1–2 are safe: their `minus_ok` requires the audit's
`slot_minus_genuine`.) Related: the registered `free_cold_0` quietness fail rule ("> 2/21 rows OR
2/19 distinct keywords fails", design l.296-298) is computed (`mechanical_cold`) but never turned
into a pass/fail anywhere.

Smallest fix: compute the bootstrap CI of the slot `neuron +0.8` genuine rate over the 42 rows and
add it as a real clause of the `first` branch; make `gate3` fail (or mark Stage 2 unread) when
`kappa` is missing while labels exist; make `gate4` require labels.

### F6 — MEDIUM, INTRODUCED: the resume rule treats *any* existing file as a complete run; a crashed or OOM-killed run silently resumes as "done" and the scorer scores a subset

`cmd_stage1`/`cmd_stage2` skip a checkpoint whenever its JSONL exists (`run213.py:494-497`,
`611-613`); `write_jsonl` rewrites the file after each (prompt, arm) cell, so a kill mid-checkpoint
leaves a complete-prefix file the resume rule cannot distinguish from a finished one. The design's
own pilot stopping rule contemplates a 9.0 GB RSS kill, so the path is realistic.

MEASURED (`exp_f6_seal_resume.py`): a temp `--out` containing three **empty** `stage1-*.jsonl`
files:

```
sft: .../stage1-sft.jsonl exists, skipping (resume rule, design l.413-414)
dpo: ... exists, skipping      instruct: ... exists, skipping
exit code: 0   -> zero forward passes executed, "success"
```

Downstream, the scorer reads the partial file; missing `(unit, prompt, arm)` records become
`None`, units drop out of CIs, and clauses turn `not read`/`inconclusive` **silently** — a
half-run scores as if it were the registered run. Smallest fix: resume at cell granularity
(append per cell, or record per-cell completion in run.json and verify the row count against the
plan before skipping).

### F7 — MEDIUM, INTRODUCED: a *partial* blind audit silently undercounts — 213.6 can be `held` with the random arm entirely unlabelled

`predict_213_6` (score213.py:819-822) counts `label == "genuine_third_attr"`; unlabelled rows
count as non-hits, and the only guard is `if not labels` (no labels *at all*). The design's fixed
denominators (42/42) assume a complete 198-item audit.

MEASURED (`exp_f8_partial_audit.py`, real `export_blind`/`_blind_order`/`predict_213_6`): labels
supplied for only the 21 `neuron +0.8` `free_third_0` rows, nothing else:

```
labels supplied: 21 of 84 audited rows
hits: {'neuron': 21, 'random': 0}
clauses: {'neuron_ge_10_of_42': 'held', 'random_le_2_of_42': 'held',
          'sign_test_p_lt_0.05': 'held', 'difference_ci_above_0': 'held'}
verdict would be: held
```

`random_le_2_of_42` passes *because* the random rows were never labelled: whoever supplies the
audit JSON controls the verdict by choosing which rows to label. Smallest fix: require every
audited row to be labelled (else the clause is `not read`) and count unlabelled rows explicitly.

### F8 — HIGH, INTRODUCED: the scorer cannot load a real full run — ~31 GB RAM needed on a 15 GB host

The runner stores the **full 100,278-float vocab logit vector** per row (`first_logits`,
run213.py:214) for all 8,238 Stage 1 rows; the scorer only ever reads ~39 entries per row (own +
`offdiag_ou` first-token logits — entropy/top-1 are already reduced to scalars by the runner).
`load_records` does `path.read_text()` (whole file as one string) then `json.loads` per line.

MEASURED (`exp_f9_memory.py`, real-sized rows through the real parse path):

```
one real-sized row: 0.90 MB of JSON text
loading 400 rows: RSS 24 -> 1560 MB  => 3.84 MB/row
full run = 8238 rows -> ~30.9 GB RAM for load_records alone
host RAM: 15 GB total, ~12 GB available; the design budgets the runner at --mem 10G
```

So after a full run completes, `score213 --records results/part213/full` would OOM/swap-crash
before printing anything. The tests never see it (their synthetic rows are 65k floats but only a
few hundred rows). Smallest fix: record only the needed logits per row (e.g. `{token_id: logit}`
for own + `offdiag_ou` ids), or stream rows and keep only the needed entries in `load_records`.

### LOW findings

- **L1 (LOW):** `predict_213_3`/`predict_213_10` evaluate the sign guard *before* low-n: with
  `n_eligible < 15` and a failed guard the clause reads `inconclusive` (score213.py:760-761,
  908) where the v5 rule's first-listed condition (`n_eligible < 15` → `not read`, design
  l.464-465) should win. `ratio()` itself orders it correctly (`read: "not read"` first,
  score213.py:304-309); the inline clause construction bypasses that ordering.
- **L2 (LOW):** the design's 213.7 print-alongside "`z_tilde(neuron -0.8) - z_tilde(none)`"
  (l.478) is computed nowhere — only the neuron−random gap exists.
- **L3 (LOW):** the runner records `residual_median_norm_layer0` with the note "score213 compares
  across run.json" (run213.py:418) — score213 never reads any residual key; the design's
  per-checkpoint residual-norm comparison (l.132) is logged and never used.
- **L4 (LOW):** the design says "`instruct`/`sft`/`dpo` NaN **stops Stage 1**" (l.336-337); the
  runner never checks finiteness mid-run — a NaN pass burns the remaining ~9,000 passes before the
  scorer's gate1 flags it post-hoc. (`json.dumps` also writes bare `NaN`, non-standard JSON that
  only Python round-trips.)
- **L5 (LOW):** the pilot's registered Stage 2 arm is `6 × 3 × 4 + 4 = 76` greedy generations
  (design l.385-389) and the design says "generation tokens/s ... measured on the 6-unit pilot"
  (l.408); `cmd_pilot` runs only the 4 unpushed baselines — generation throughput is never
  measured.
- **L6 (LOW, doc only):** BUILD-REPORT.md:58 says Stage 2 is "472 planned / 238 executed under the
  seal" — 238 assumes *all* units are sealed on `free_self_0`/`free_cold_0`; the design
  (l.262-263) and the code seal concept rows only, so the executed count is 472 − 108 = 364.
- **L7 (LOW):** 213.6's `difference_ci` bootstraps the **median of discordant per-unit
  differences** (score213.py:823-827), not the design's `(neuron_hits - random_hits)` total
  (l.477); and "strictly above 0" is implemented as `ci_clause(diff_ci, "ge", 0.0)` — a bootstrap
  whose 2.5th percentile lands exactly on 0 (plausible with small-integer differences) counts as
  `held` where the design says strictly above.

## What I checked and found sound

- **The seal (brief check #4) is correctly enforced on both sides.** MEASURED (`exp_f6_seal_resume.py`):
  `is_sealed` seals exactly the concept rows of `self_select_0`/`free_self_0`/`free_cold_0`; slot
  rows run; `index_stage1` keeps sealed rows out of the metric index; every Stage 2 consumer
  (`export_blind`, `audit_table`, `dedup_keyword_counts`, `predict_213_6`, `gate4`) skips
  `status != "ok"`. The `none` pass does write `lp_by_unit` entries for concept NPs on
  `self_select_0` inside slot rows, but the scorer reads `lp_by_unit` only on `third_0`/`third_2`,
  so sealed data is never read (and the design's own l.212-215 says the none pass scores all 39
  NPs on every prompt).
- **The v5 disposition rule core** (`ci_clause`/`point_clause`/`disposition`,
  score213.py:164-210): straddling → `inconclusive`, wholly-wrong-side → `failed`, missed point →
  `failed`, low-n → `not read`, one failed clause fails the prediction — matches design
  l.464-468, and the required sabotage test shows it red under a broken copy.
- **The sign guard over eligible units** (`ratio()`, score213.py:279-310): numerator CI over the
  eligible units' numerators only, guard failure → `inconclusive` even when the ratio CI is
  wholly below the bar (v5.1 reading 2); `n_eligible` printed; `null` exclusion on `b <= 0.50`.
- **Scene-framed denominators**: τ_M/3 and τ_cop_nonid/3 use `third_2, third_3`
  (score213.py:772, 904) with pooled versions descriptive only; τ_3/1 stays pooled with
  direct/scene point medians as clauses (v5.1 reading 3, BUILD-REPORT finding 3).
- **213.5's pair bootstrap** (score213.py:782-798): one draw per concept-slot pair, point = median
  pair difference, plus the 8-unit agreed point clause and a reader-sensitivity flag — v5.1
  reading 5.
- **213.11's strict `lo > 0`** via `ge_strict` (score213.py:923) and the length-matched
  `n_u * (own − mean peers)` form (score213.py:355-356), slot-peer term gated — v5.1 readings 10-11.
- **213.1/213.2 clause split** per v5.1 reading 4 (213.1 both CI clauses; 213.2 L CI + N point),
  own-unit full-NP clauses on `lp_sum` (design says `lp_content_sum`).
- **Gate voiding** (brief check #3): gate1-or-gate2 failure voids 213.1–213.5/213.7–213.11 and
  gate1-or-gate3 failure voids 213.6 (score213.py:1059-1086) — measured in the F4 run A output;
  the 213.1-failed withdrawal rule is implemented (score213.py:1140-1141). (The inverse direction —
  gates passing *without* the evidence they guard — is F5.)
- **Dose/label conventions match** between writer and reader: string doses `"+0.8"`, `"0.0"`,
  `"+0.05"`, `"-0.8"`; record keys `(unit, prompt_id, arm, dose)` line up with `rec()`; `alpha`
  floats separate; `first_logits` indexed by `first_content_token_id`; `lp_sum` vs `lp_mean` used
  where the design says each.
- **The rendered sequences are correct**: all 39 units × 17 prompts tokenize identically to the
  design's `chat_prefix + opening + " " + np_text` rendering (0 merge mismatches, measured) — so
  F1 is purely the read index, not the row construction. Gate 2's own `row_spec` positions are
  correct, which is why the real pilot passed Gate 2.
- **The 23 tests pass (19.9 s) and the 8-sabotage harness genuinely turns each required test
  red** with drift anchors; `deepseek_reader` is an uncalled seam that raises; the blind export's
  seed-213 shuffle is deterministic and the `blind_id` join is consistent.
- **Pass-count arithmetic** matches v5.1 reading 1: instruct 5,538 planned − 108 sealed = 5,430;
  sft/dpo/FR 936 each; Stage 2 472 planned.

