<!-- cheap-review.mjs cheap-review-part203-design-r1-ds-ds-r1-2026-11-10: /home/elisepc/green/worktrees/ab-part203 0d214bc..HEAD, lane ds, df-run exit 0 -->
# Review: part203-design-r1-ds (cold, adversarial)

Verdict: **DO-NOT-SHIP**

Why: three deterministic INTRODUCED defects stop the pipeline before a single readout can be trusted — the
recorder cannot finish any run (HIGH-0), the download step rejects 2,190 of the 2,730 selected clips (HIGH-1),
and the analyzer will publish `status: "complete"` readouts on a recording whose V/D input gate failed (HIGH-2).
MEDIUM-1 additionally makes the E half of the ablation not the experiment the design names. The design itself is
mostly faithful to the code, but its sizing ("a few minutes") and one prediction (203.4) are not identified as
written.

Scope reviewed: `git -C /home/elisepc/green/worktrees/ab-part203 diff 0d214bc..HEAD` — PART203-DESIGN.md,
part203/{select203,record203,analyze203,test_part203}.py, download.sh, and the part196/analyze196.py hunk.
Test suite: 26 passed in 17.9 s on the real venv (`/home/elisepc/neuron-probe/.venv/bin/python -I -B -m pytest`).

## Findings

### HIGH-0 — the recorder can never finish a run: Gate 1 JSON write raises `TypeError` (INTRODUCED)
`part203/record203.py:220-221` builds `gate1_vd(...)` and writes it with `_write_json`, whose
`json.dumps(value, indent=2, sort_keys=True)` (line 265) has no `default=`. But `gate1_vd` embeds
`v_item(np.zeros(1, dtype=np.float32))` (line 253-254) inside `"v_vs_d"`, and `v_item` returns
`{"text": ..., "audio": [<np.ndarray>]}` (line 91). A numpy array is not JSON serialisable, so the write
raises **before** the `gate1["verdict"] != "passed"` check on line 222. Every recording run, complete or
not, ends in an exception and exits 1; the success summary (which is the only place `elapsed_seconds` appears)
is never printed. For a 20-clip smoke, `clips_processed` is not printed either: it has a `% 50` threshold.

MEASURED (real `gate1_vd` on a complete 300-row fixture whose records all say `v_same_samples=False`, then the real `_write_json`):

    $ python -B gate1_write_probe.py
    gate1_vd verdict: failed | v_d_input_mismatch_count: 300
    v_vs_d['v_item']['audio'][0] type: <class 'numpy.ndarray'>
    _write_json raised: TypeError Object of type ndarray is not JSON serializable
    bare json.dumps raised: TypeError Object of type ndarray is not JSON serializable

MEASURED through the real `run_recording` with only the model/audio functions stubbed (`limit=20`, 2,730-row
manifest): `run_recording raised: TypeError Object of type ndarray is not JSON serializable`,
`gate1.json exists: False`, `records written: 20`. The 20 `--limit` smoke rows in `steps.txt` step 3 are kept
(atomic writes) but the smoke's stated output is never produced.

Why it matters: on a fully successful 2,730-clip recording the operator sees a `TypeError`, not a success;
`gate1.json` (the artifact the design calls "the recorder's own Gate 1") is never written; and because the
analyzer does not read that file either (HIGH-2), nothing records that the V/D input check ran.
Smallest fix: `json.dumps(value, indent=2, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))`
in `_write_json`, or replace the `v_item`/`d_item` probes in `gate1_vd` with plain strings (as the rest of
`v_vs_d` already is).

### HIGH-1 — download.sh cannot download the selected clips; it rejects 2,190 of 2,730 (INTRODUCED)
`part203/download.sh:28` validates every manifest name with
`re.fullmatch(r"\d{4}_IEO_[A-Z]{3}_[A-Z]{2}\.wav", name)` — the sentence code is hardcoded to `IEO`.
`select_clips` pools all twelve CREMA-D sentence codes and takes the five lowest SHA-256 names per
(actor, emotion) cell, so only 540 of the 2,730 selected clips are `IEO`:

    $ awk -F_ '{print $2}' part203/download-list.txt | sort | uniq -c
        193 DFA / 540 IEO / 200 IOM / 210 ITH / 193 ITS / 202 IWL / 205 IWW
        197 MTI / 200 TAI / 210 TIE / 197 TSI / 183 WSI
    total: 2730

MEASURED (real script, manifest built by the real `select203.py`, temp data dir under my domain):

    $ /home/.../python -I -B part203/select203.py --tsv part203/prior/cremad-audiowav.tsv --out $D/manifest203.csv
    {'manifest': '.../manifest203.csv', 'clips': 2730, 'actors': 91,
     'folds': {'0': 19, '1': 18, '2': 18, '3': 18, '4': 18}}
    $ bash part203/download.sh $D/manifest203.csv $D/dl-test
    unsafe or unexpected CREMA-D filename: '1001_ITH_ANG_XX.wav'
    exit=1

Nothing is downloaded (the validation loop runs before the curl loop), so step 2 of `part203/steps.txt`
stops the run before step 3. The recorder then records 2,730 `record_error` rows and Gate 1 stops the part.
INTRODUCED by this change (`download.sh` is new in this diff).
Smallest fix: `r"\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\.wav"` (and keep the under-`$DATA_DIR` no-`/` guarantee).

### HIGH-2 — the analyzer never enforces the recorder's "V and D inputs are the same samples" gate; a false V-vs-D reading passes (INTRODUCED)
The design (Controls, line 78-79) makes `v_same_samples` part of the recorder's Gate 1, and Gates 1 and 2 must
pass "before any readout is read". `record203._record_one` writes `v_same_samples` per clip (line 154), and
`gate1_vd` fails on any mismatch (lines 240-246). But `analyze203.read_records` never reads that field, the
analyzer's `gate1` (lines 253-288) does not know it, and the analyzer never reads `<work-dir>/gate1.json`
(the recorder's gate file is written at most and ignored). `gate2` is read from disk; Gate 1 is recomputed in a
weaker form that only checks status, finiteness, fold disjointness and cell counts.

MEASURED (fixture where every record has `v_same_samples=False`, i.e. the two decoders returned different
samples on all 300 clips):

    $ python -B gate1_write_probe.py
    gate1_vd verdict: failed | v_d_input_mismatch_count: 300
    $ python -B f3_probe.py
    analyzer status: complete | gate1: passed | gate2: passed
    V_grid mean BA: 1.0
    gate1 problems: {}

So a run stopped by the recorder's Gate 1 (the recorder raises before finishing), and any run killed between the
last record write and the Gate 1 write, still produces a `status: "complete"` `results.json` with every readout.
That is exactly the false reading the gate exists to prevent (V and D were embedded from different sample arrays,
so `V_grid_minus_D_*`/`D_p196_minus_V_grid` are not the comparison the design names).
Smallest fix: have `read_records` copy `v_same_samples` into the output, extend `gate1` to fail when any ok
record has it False (or missing), and add a test that flips it.

### MEDIUM-1 — the three-actor ablation fits E with an inner-CV C, not the design's fixed C = 1.0 (INTRODUCED)
Design Method, line 66-67: "Three-actor ablation: ... for V (C by inner CV on the draw's actors) and E
(C = 1.0, the fold's selected layer)." `analyze203.py:422-427` calls `_fit_and_score(..., c=None)` for both
`name == "V"` and `name == "E"`, so E's 25 draw fits choose C from the grid. The baseline they are subtracted
from (`all_training_actors_mean`, line 462) is `E_p196`, which is C = 1.0. Prediction 203.7 is a difference of
these two drops, so E's drop mixes a regularisation change with the actor-count change; V's does not.

MEASURED (real `analyze()`, informative synthetic E, 2 layers x 8 units):

    fold 0 E draw C: [100.0, 100.0, 100.0, 100.0, 100.0]
    ... same for folds 1-4 (all 25 E draws chose C=100.0)
    E all-training (C=1.0) mean: 1.0

25/25 E draws used a C other than the design's 1.0. Fix: pass `c=P196_C` when `name == "E"`, keep `c=None` for V,
and assert the chosen C in the test.

At realistic 91-actor scale (the real 2,730-row manifest, synthetic records) the same code path chose C = 1.0
for all five E draws in fold 0 on that fixture, so the size of the error is data-dependent; the control flow is
nevertheless `c=None` for E, so the design's "C = 1.0" is not what runs.

### MEDIUM-2 — `--limit` cannot produce the timing smoke the design and steps.txt name (INTRODUCED)
Design Optimization (line 97-99) and `steps.txt` step 3 say to run with `--limit 20` and read
`elapsed_seconds` and `clips_processed` from the log. With `limit` set, `run_recording` breaks out of the loop
(line 201-202), then runs `gate1_vd` over the **whole** planned set, which fails because the un-run clips are
missing (line 222) and raises `RuntimeError("Gate 1 failed on successfully recorded clips")`; `elapsed_seconds`
is only in the return value (line 228), which is never reached. `clips_processed` is printed only every 50 clips
(line 211), so a 20-clip smoke prints neither number. (Today HIGH-0 fires first; after that fix this is next.)
MEASURED through the real `run_recording` (model/audio stubbed, `limit=20`): raised `TypeError` at the Gate 1
write, `records written: 20`, no summary. With HIGH-0 fixed the same control flow raises `RuntimeError`, never
returning the summary. Fix: treat a limited run as a timing run — skip the completeness half of Gate 1 (or write
`gate1.json` with `"limited": true` and only check the recorded rows), print the summary, and leave the full gate
to the un-limited run.

### MEDIUM-3 — the analysis plan cannot run in "a few minutes": the layer CV is ~4-9 CPU-hours (INTRODUCED)
Design Size (line 91) says "Probes, bootstraps and pitch features: a few minutes on CPU." The layer choice is
implemented (through the part196 helper, `analyze203.py:402-404`) as **leave-one-actor-out** CV: for each of the
five outer folds it fits `24 layers x ~72 training actors = ~1,728` six-class probes on 2,048 features and ~2,100
rows, i.e. ~8,640 fits for the study. MEASURED on the real `part196.analyze196.fit_probe` at that shape
(2,160 x 2,048, six classes, converged in 8-9 lbfgs iterations, random and separable data):

    random   fit: 3.8 s, 3.1 s (two runs);   random 2.32 s / 8 iter;  separable 1.60 s / 9 iter

so ~8,640 fits at 1.6-3.8 s is **4-9 hours** of CPU for the layer grid alone (plus the 2,000-draw bootstrap and
the per-clip recording). The design's own total-minutes field is still `{{CLAUDE: minutes}}`, but the "few minutes"
claim is wrong by roughly two orders of magnitude and the Optimization section does not account for it.
At realistic 2,730-row scale the whole `analyze()` with a 2-layer x 8-unit E took 57.7 s (5 outer folds, correct
n_train/n_test 2,160/570 and 2,190/540, 2,000 bootstrap draws), so nothing else in the analysis is slow: the layer
grid is the cost. Note also that the design calls both inner procedures "inner actor-grouped CV" (Method lines 55 and 62-63) while
the code uses 5 actor folds for C (`INNER_FOLDS = 5`) and 72 leave-one-out folds for the layer: the fold count is
never named and the two choices are not the same quantity. Fix: name the inner fold count for C and the layer in
the design; if 5-fold is intended for the layer too, pass the same `actor_folds` split (the cost drops ~14x), or
pre-register the LOO cost.

### MEDIUM-4 — `soundfile` is not importable by the recorder's interpreter, so readout V cannot be recorded (INTRODUCED)
`record203.decode_audio_v` imports `soundfile` (line 66); it is the only path that produces V. The exact preflight
`steps.txt` step 3 prescribes fails in this environment:

    $ PYTHONPATH=/home/elisepc/eg2-extra /home/elisepc/neuron-probe/.venv/bin/python -c \
        'import sentence_transformers, imageio_ffmpeg, soundfile'
    ModuleNotFoundError: No module named 'soundfile'

The venv has `site.ENABLE_USER_SITE False`, so the copy at `~/.local/lib/python3.12/site-packages/soundfile.py`
is not on the recorder's path, and `eg2-extra` has no `soundfile`. The recorder runs without `-I` but that does not
help. Every clip would therefore be written as `record_error: ModuleNotFoundError`, and Gate 1 stops the part —
it fails closed, which is the right direction, but the run cannot happen as specified. The three decode tests
monkeypatch a fake `soundfile` (`test_part203.py:355-379`), so the suite is green while the real dependency is
absent. Fix: add soundfile to the run environment (or to `eg2-extra`) and make step 3's preflight a hard
precondition in the design's "Changes before registration"; consider a test that imports the real package when
available (`pytest.importorskip` would at least show the skip).

### MEDIUM-5 — prediction 203.4's comparator bundles layer, marker and C effects (INTRODUCED, design)
Prediction 203.4 is `E_p196_minus_V_grid.mean_difference >= -0.02` with the prose "an intermediate layer at least
ties the final vector" and the rationale "Part 196's 37-point E over D". But `V_grid` is Venn's vector V at the
inner-CV C, so the quantity is `(E - V at grid C)`: a layer effect plus the marker/float16 difference of 203.2
plus the C effect that 203.3 expects to be worth +0.03. The named keys contain `V_grid_minus_V_p196` and
`D_p196_minus_V_grid`, so a reader can recover the same-item, same-C point estimate
`E - D = (E_p196_minus_V_grid) - (D_p196_minus_V_grid)` by arithmetic — but the design never says to, that
combination has no paired interval, and 203.4 as written would be read as a clean layer comparison even when
203.2 or 203.3 fails. Same for 203.7: E's all-training baseline is `E_p196` at C = 1.0 while its draws use grid C
(MEDIUM-1). Smallest fix: add `E_p196_minus_D_p196` (and/or `E_grid_minus_V_grid`) to `pairs` in
`analyze203.py:438-439` and state 203.4 against it; both models are already fitted in the fold loop, so it is free.

### LOW-1 — 203.2 attributes a V-vs-D difference to the text marker alone (design)
Prediction 203.2 ("the text marker does not matter", D_grid within 0.03 of V_grid) is measured between two vectors
that differ in the item's text field, the decoder (soundfile vs ffmpeg), the cap (30 s vs 10 s) and the storage
dtype (V float16, D float32). I measured the two decoders to be bit-identical on a 16-bit 16 kHz mono WAV
(`decode_compare.py`: `bit identical True`, `max abs diff 0.0`; a 12 s clip gives `v 192000 d 160000`), so on
CREMA-D the only real confounds left are the marker and the float16 rounding — but the design never states a
control that isolates the marker, and the claim as written is stronger than the comparison. Reword 203.2 as
"marker + float16 storage" or add a float32 V row.

### LOW-2 — `probe_scores`'s documented missing-class fill is not implemented; it raises instead (INTRODUCED)
`analyze203.py:52-77` and its docstring say "a class absent from the training rows gets MISSING_CLASS_SCORE".
Measured with a probe trained without class 5: `fit_probe_c`, then `probe_scores` ->
`ValueError: probe scores have shape (60, 7) for classes [0, 1, 2, 3, 4]`. `_score_probe` always expands to
`len(STYLES)=7` columns unless the classes are exactly `[0..6]`, so the "drop the seventh column" branch only
works for a full six-class probe. This is unreachable today (every actor carries all six emotions, and `analyze`
raises first if a fold lacks one), so it is a robustness/docstring defect, not a false-reading path. Fix: detect
the expansion by comparing `raw.shape[1] - (max(classes)+1)` or reconstruct from `model[-1].classes_` directly.

### LOW-3 — a test's monkeypatch of `BOOTSTRAPS` does nothing; the "8 draws" test runs 2,000 (INTRODUCED)
`test_analyze_runs_end_to_end_and_writes_results` sets `analyze203.BOOTSTRAPS = 8` and its comment says
"8 bootstrap draws", but `_text_cluster_bootstrap(..., n_boot: int = BOOTSTRAPS)` binds the default at function
definition. Measured on the same fixture: `balanced_accuracy_actor_cluster_95: {'n_boot': 2000}`. The test is not
wrong about the pipeline, but it does not exercise what it claims, and a regression that inflated `BOOTSTRAPS`
(e.g. 200,000) would still pass this suite (only slower). Fix: pass `n_boot` through `analyze` or read
`analyze203.BOOTSTRAPS` inside the function body.

### LOW-4 — a truncated `.npz` record crashes the analyzer instead of failing Gate 1 (INTRODUCED analyzer; PRE-EXISTING recorder half)
`read_records` catches `(OSError, ValueError, KeyError, EOFError)` (line 245) but not `zipfile.BadZipFile`.
MEASURED with one record truncated to half its bytes (all other records valid):
`truncated -> read_records RAISED: BadZipFile File is not a zip file`; an empty file is handled
(`record_error: EOFError`). The recorder's `_cached_status` (part196/record196.py:362-369) likewise misses
`EOFError`/`BadZipFile`: `_cached_status RAISED: EOFError No data left in file`. Because writes are atomic this
needs external corruption or a copied/partial work dir, but the design says a failed clip is "named and counted",
not a traceback. Fix: add `zipfile.BadZipFile` and `EOFError` to both catch tuples. (`read_records` is new code;
`_cached_status` is pre-existing part196 code that this change now drives with Part 203 records, so its half is a
follow-up, not a reason to hold this change.)

### LOW-5 — registration-blocking placeholders remain in the design and in `results.json` (INTRODUCED)
`PART203-DESIGN.md` still carries `{{CLAUDE: minutes}}` (line 89), `{{CLAUDE: measured ...}}` (line 97),
`{{CLAUDE: changes}}` (line 138), and its own header says the predictions are `{{CLAUDE: predictions}}` although
the table is filled. `analyze203.py:474` writes the literal string into every `results.json`
(`"predictions": "{{CLAUDE: predictions}}"`), and `test_part203.py:295` asserts that literal. The part is
explicitly DRAFT so shipping placeholders is defensible, but the size/optimization fields are exactly the ones a
preregistration is supposed to fix, and a results file that claims to carry the predictions carries a template.

### LOW-6 — `record203`'s docstring says the identity records whether V and D are the same computation; it does not (INTRODUCED)
Module docstring lines 7-8: "The recorder records both and states in its identity whether they are the same
computation." `recording_identity` (part196/record196.py:415-437) has model/manifest/data-dir fields only, and
`run_recording` adds only `study`; the V/D statement lives in `gate1_vd`'s `v_vs_d`, which is written to
`gate1.json` — the file that HIGH-0 shows is never written. Fix the docstring, or put the `v_vs_d` block in
`identity.json` (which is written before recording).

### LOW-7 — reading rule "both intervals are above chance" does not say which bound (design)
Reading rules line 126: "A gap between V and D is read only if both intervals are above chance (1/6)." It does not
say whether "above chance" means the point estimate, the 95% lower bound, or the whole interval. Given prediction
203.1 is stated against the interval's lower bound, say so explicitly for the V/D rule too; otherwise two readers
can reach different verdicts from the same `results.json`.

### LOW-8 — `gate2.json` and `results.json` are written non-atomically (INTRODUCED)
`record203.py:183-184` writes `gate2.json` with `write_text`, and `analyze203._write` writes `results.json` with
`write_text`. Both files are read back with `json.loads` (`analyze203.gate2` has no try/except, and a human or a
later run reads `results.json`), so a process killed mid-write leaves a file that raises `JSONDecodeError`
instead of a clean gate failure, and a truncated `results.json` is not obviously truncated. The recorder already
has `_write_json` and part196 already has `_atomic_json`; use one of them for both. (The same crash path exists
if a hand-edited/corrupt `gate2.json` is present.)

### Under-defined, not separately measured
- Method says "Intervals: ... for each readout", but `analyze203._text_cluster_bootstrap` only intervals balanced
  accuracy (`balanced_accuracy_actor_cluster_95`); `mean_macro_ovr_auc` gets no interval even though "each readout"
  could be read to include it. The predictions table only names the BA interval, so name that explicitly.
- The number of inner folds is not named for either inner procedure (see MEDIUM-3).

## What I checked and found sound

Everything below was driven through the real code from `/home/elisepc/green/worktrees/ab-part203`; no CREMA-D
audio was read, no model was run, no live service called.

- **Selection and folds.** `select203.py` on the committed `prior/cremad-audiowav.tsv` writes 2,730 rows,
  91 actors x 6 emotions x 5 clips, every (actor, emotion) cell exactly 5, folds 19/18/18/18/18. The rule is
  SHA-256 order within cell and round-robin by SHA-256 of the actor id; `test_selection_reproduces_the_committed_download_list`
  and the fold tests pass, and the committed `download-list.txt` is exactly the generated set (540 IEO + 2,190 other
  sentence codes). The fold assignment is also independently SHA-rule-checked in the test (line 77-80).
- **Emotion/label order.** `EMOTIONS = (ANG, DIS, FEA, HAP, NEU, SAD)` is used for selection, labels and
  per-emotion recall consistently; labels are the dataset's, never a model's.
- **V/D item shapes.** `v_item` and `d_item` reproduce the two prior implementations exactly
  (`{"text": "<|audio|>", "audio": [samples]}` vs `{"audio": {"array": ..., "sampling_rate": 16000}}`), matching
  `venn-server.py:346-369` and `record196.py:_record_one`. The design's line-number citations for both prior
  implementations (venn-server 346-369, 371-425, float16 at 414, renormalise at 425; venn-reader 14-15) are correct.
- **The two decoders really do agree on the input.** On a synthetic 2 s 16-bit 16 kHz mono WAV, `decode_audio_v`
  (soundfile) and `decode_audio_d` (ffmpeg, the real part196 path) return **bit-identical** float32
  (`max abs diff 0.0`, 32,000 samples), so `v_same_samples` should pass on CREMA-D's 16-bit PCM. A 12 s clip gives
  `v 192000 d 160000` — D's 10 s cap bites and `gate1_vd` reports the mismatch rather than dropping the clip, as
  the design says.
- **Probe scoring.** For a six-class probe, part196's `_score_probe` returns 7 columns (its seventh is the
  `-1e6` style-6 fill); `analyze203.probe_scores` drops that column and returns six columns in class order, and
  its argmax reproduces `model.predict` exactly (measured accuracy 1.0 both ways). Balanced accuracy and macro
  one-against-rest AUC are computed with `n_classes=6` via the part196 helpers. Missing-class handling is the only
  soft spot (LOW-2) and is unreachable given every actor carries all six emotions.
- **C selection.** Five actor-grouped inner folds, grid {0.01, 0.1, 1, 10, 100}, lowest mean held-out-actor log
  loss, ties to the lower C; measured on separable data (chose 100) and on the realistic 2,730-row fixture
  (chose 10, with identical losses for 10 and 100 and the tie broken down, as the rule says). The inner CV only
  ever sees outer-training rows (spy test checks the call count and fold sizes).
- **Layer choice.** Leave-one-training-actor-out, mean six-class balanced accuracy, ties to the earlier layer
  (`max(usable, key=lambda i: (means[i], -i))`); measured to pick the layer carrying the signal (layer 0) and to
  run on outer-training rows only. It is the cost, not the direction, that is wrong (MEDIUM-3).
- **Bootstrap.** 2,000 draws, seed 203, actors resampled with replacement within each held-out fold, fold BAs
  recomputed and averaged, paired differences computed from the same resample; `n_boot` reports usable draws.
  Verified at 2,730-row scale: `n_boot: 2000`, five folds, fold actor counts 19/18/18/18/18.
- **Gates.** Analyzer Gate 1 fails on a missing clip, a non-finite V, an actor in two folds, a cell that is not
  five, and a fold count other than five (all tested and measured); a missing `gate2.json` is `not_run` and fails
  the gate like `failed`; a manifest that differs from the recorded identity raises before any readout. The
  identity hash covers the full row set (manifest columns plus the recorder's synthetic fields), and both sides
  build it with `read_manifest_rows`, so an edited manifest is detected.
- **Resume.** Rows with `status: "ok"` are skipped, `record_error` rows are retried, each record and the
  `gate1.json` write is atomic (`_atomic_npz`/`_write_json` use fsync + `os.replace`), and the identity file
  refuses a resume under a different model/manifest/data dir. A restart after partial success keeps the completed
  rows. (`gate2.json` and `results.json` are plain `write_text` calls — LOW-8.)
- **The design's prior-art numbers.** `prior/sound-eval.json` gives family accuracy 0.564685 / CI 0.532051-0.600577,
  probe 0.567016, difference -0.002331 / CI -0.007550-0.002331, chosen C 100, log loss 1.7788/1.7080/1.4788/1.2507/1.1961,
  Jev 0.14627, chance 0.16667, 1,716 rows — every figure quoted in the design matches. Part 196's
  `results/part196/results.json` gives D 0.2583, E 0.6270, H 0.5177, matching the design's 25.8 / 62.7 / 51.8.
- **The part196 change is backward compatible.** `choose_intermediate_layer` gained `ba` and `n_classes` with
  part196's own defaults; `part196/test_part196.py` still passes 27/27, and the part203 suite passes 26/26.
- **No feature flags** exist in this change, so there is no OFF path to audit. The recorder refuses to start
  without Gate 2 passing (gate2 runs before any clip), and the analyzer refuses to compute readouts with
  `gate2 != passed`.
