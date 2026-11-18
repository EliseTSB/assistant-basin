# Cold adversarial review — Part 65 build (`afe30db..517dcc1`)

## Verdict

**SHIP-WITH-FIXES.** The core write side is faithful to the registered design: the neuron/control write
vectors, unit normalisation, dose × per-layer median, the every-position MLP hook, the 9 batched rollouts, the
blind 2AFC plan and the 65a–65f thresholds all match `PART65-DESIGN.md` when exercised against the real Part 64
inputs. But two paths cannot run as shipped in the repo's standard environment (the default corpus path does not
exist; the live judge ignores the MITM CA and the house credit-key name), and one registered pilot check that
would catch a silent no-op steering run is missing. Fix H1/H2 before the A770/live run; fix M1–M4 before the
numbers are trusted.

Method: modules copied to `p65copy/` (the worktree is read-only, so the suite's `TemporaryDirectory(dir=...)`
cannot run in place) and driven with `/home/elisepc/neuron-probe/.venv/bin/python` (torch 2.14.1, transformers
5.20.0). Scripts are in `scripts/`; outputs pasted below. No live API call, no GPU job, no writes outside
`/home/elisepc/green/Journal/scratch/review-p65-build-ds-r1-2026-10-24`.

---

## Findings

### H1 — INTRODUCED — the live judge cannot authenticate in the standard environment (TLS CA + key name)
`part65/judge65.py:323-346` (`api_call`), `:325-327` (key).

`api_call` calls `urllib.request.urlopen(request, timeout=timeout)` with **no SSL context**, and reads
`ANTHROPIC_API_KEY`. Every other reader in this repo does the opposite: `part63/reader63.py:160-182` and
`part64/gen64.py:534-561` build `ssl.create_default_context(cafile=~/.mitmproxy/ca-bundle-combined.crt)` and pass
`context=`; `part60/gen60.py:434-447` does the same via `SSL_CERT_FILE`; all three read `ANTHROPIC_CREDITS_KEY`
(only the Part 60 probes accept `ANTHROPIC_API_KEY` as a fallback). The design says "from the API credit", i.e. the
credit key.

Measured preconditions (no live call was made):

```
$ python - <<'EOF'   # compare the default trust store with the repo's bundle
default = ssl.create_default_context(); bundle = ssl.create_default_context(cafile="~/.mitmproxy/ca-bundle-combined.crt")
mitm certs in bundle: 1 in default: 0
bundle count 122 default count 120
EOF
HTTPS_PROXY=http://localhost:8080
localhost:8080 accepting connections: True
API-related env var names: ['ANTHROPIC_AUTH_TOKEN', ...]      # neither API key is set in a bare shell
```

`part60/probes/probe_batch.py:1` states the requirement outright: "Needs
SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt". So with `HTTPS_PROXY` pointing at the MITM proxy and the
default store not trusting its CA, the first live action (`--probe`) fails TLS verification; because that happens
on the POST, `process_chunk` lands in `unknown_no_batch` (`judge65.py:589-593`) and the reserve is stranded. Even
with `SSL_CERT_FILE` exported, a run configured like Parts 60/63/64 (only `ANTHROPIC_CREDITS_KEY`) is refused at
`judge65.py:325-327` before any call.

Why it matters: the judge is the primary measure; the one-request probe exists precisely to find this before
1,000 calls are planned. The build report does not mention either convention.

Smallest fix: copy `part63`'s `context()` helper and pass `context=`; accept
`os.environ.get("ANTHROPIC_CREDITS_KEY") or os.environ.get("ANTHROPIC_API_KEY")`.

### H2 — INTRODUCED — `steer65.py`'s default corpus path points at a drive that does not exist (and the default collection is WSL-only)
`part65/steer65.py:42` (`DEFAULT_CORPUS`), `:41` (`DEFAULT_COLLECTION`).

```
DEFAULT_CORPUS = E:\neuron-probe-gpu\qwen35\pile10k.jsonl
exists: False
canonical exists: True          # /mnt/c/Users/alexa/neuron-probe-gpu/qwen35/pile10k.jsonl (63 MB)
corpus_token_ids(default) RAISED: FileNotFoundError [Errno 2] ... 'E:\\neuron-probe-gpu\\qwen35\\pile10k.jsonl'
```

The registered Part 60 corpus is on **C:** — `part64/collect64.py:21`, `part64/gen64.py:44`,
`atlas_collect_fast2.py:29` and `part64/run64.sh:9` all use
`/mnt/c/Users/alexa/neuron-probe-gpu/qwen35/pile10k.jsonl` (and the env var there is `PART64_CORPUS`, not
`PART60_CORPUS`). `steer65`'s default is the only `E:\neuron-probe-gpu` reference in the repo (the only other
`E:` path is the HF cache `E:\neuron-probe\hf`). In addition, `DEFAULT_COLLECTION` is a WSL path; on the A770 the
step runs under Windows Python (`/mnt/c/.../job.py` uses `ctypes.windll`), where `/home/...` resolves to
`C:\home\...` and does not exist. So **both** defaults fail on the target machine.

Why it matters: `main` measures medians before the first unit (`steer65.py:589-594`), so a bare
`python steer65.py --out ...` on the A770 dies with `FileNotFoundError`; the build report says the module "reads
… the Part 60 corpus" without flagging that the default is wrong. It is loud, but it blocks the run and was never
exercised (no test touches `measure_layer_medians` or `main`).

Smallest fix: default to the registered C: path (or `PART64_CORPUS`), and document/require a Windows
`--collection` on the A770.

### M1 — INTRODUCED — the design's "a +0.8 push changes at least half the rollouts" check is absent; a dead steering path passes every built-in check
`part65/steer65.py:346-363` (checks and rollouts); design `PART65-DESIGN.md` "Checks before the full run".

The only rollout check is dose 0 (`:349-350`). I replaced `intervention_hook` with a no-op and ran the real
`process_unit` on the tiny model:

```
process_unit returned without raising: True
dose_zero_matches_baseline flag: True
rollouts equal to their prompt's baseline: 9/9
write_vector_norm: 1.0
```

Every built-in check passes with the intervention never applied: dose 0 trivially reproduces the baseline, the
normalised norms are 1, and the 9 rollouts are silent baselines. The design's third pilot check is the only one
that would catch this, and it is not implemented (the build brief omitted it, but the design is the registered
contract).

Why it matters: if the hook silently misses on the A770 (wrong layer, dtype/path change, a future refactor), the
collection looks valid, the judge measures noise, 65a fails, and the design's decision rule commits the false
conclusion "our atlas labels describe inputs only". This is the highest-consequence failure mode in the build.

Smallest fix: after the rollouts, count the `+0.8` rollouts whose `token_ids` differ from their prompt's baseline
and raise unless at least half differ (per the design); also assert at least one of the 9 differs.

### M2 — INTRODUCED — the judge is told the negative-dose lens list means the opposite of what it contains
`part65/judge65.py:171-173` (`format_write_summary`), `part65/steer65.py:310-322` (`lens_summary`).

`lens_summary` computes `mean_delta = logp(steered) − logp(base)` and takes `torch.topk(mean_delta, 12)` for
**both** doses, so the `-0.8` list is the 12 tokens whose log-probability **increased most** under the negative
push. `format_write_summary` labels that list "Tokens made less likely by a negative push". Measured on the real
code:

```
=== steer lens '-0.8' entries (mean_delta_logp) ===
  token='i'  +0.491490      token='l'  +0.408925      token='k'  +0.314258
  ... (top-12 largest deltas; the list is the increases, not the decreases)
=== format_write_summary text shown to the judge ===
Write words: amber, orchard
Tokens made more likely by a positive push: m, d, p, o, j, h, q, n, r, , g, s
Tokens made less likely by a negative push: i, l, k, e, , g, , s, , f, r, q
```

The `-0.8` line is false for every entry. One of the three lines the judge sees is systematically mislabelled for
all 1,000 trials.

Why it matters: the steer-lens half of every write summary carries inverted semantics, which can only depress or
bias 65a (the primary prediction). The design's "top 12 tokens per sign" is ambiguous; the code and the label
disagree, so one of them is wrong.

Smallest fix: relabel line 173 to "Tokens made more likely by a negative push" (matches the code), or take the
bottom-12 deltas for `-0.8` and keep the label.

### M3 — INTRODUCED — one malformed judge reply aborts the whole analysis
`part65/judge65.py:378-391` (`_extract_response`), `part65/analyze65.py:200-201`.

```
'A'            -> ('A', usage, None)
'A.'           -> (None, usage, 'response was not exactly A or B')
'A)'           -> (None, usage, 'response was not exactly A or B')
'Answer: B'    -> (None, usage, 'response was not exactly A or B')
```

`_extract_response` accepts only the exact strings `A`/`B` after `.strip()`. A trailing period (plausible for
Haiku even with "Reply with only A or B") yields `correct=None`, the batch still settles, and
`analyze_rows` then raises "one or more judge results have no valid A/B decision" for that unit
(`analyze65.py:200-201`) — with no retry path, because the batch is already settled. At 1,000 calls this is a
likely run-ender.

Why it matters: the paid batch is spent and the analysis cannot produce 65a–65f; there is no unit-drop or retry.

Smallest fix: accept a lone letter with optional trailing punctuation (`re.fullmatch(r"[AB][.)]?", answer)`), and
record/retry unparseable rows instead of making the whole analysis fail.

### M4 — INTRODUCED (test gap) — the suite stays green under four core regressions
`part65/test_part65.py:238-272`, `:405-426`.

I copied the suite into writable space and ran four sabotages; each is 10/10 OK:

```
===== MUTANT: layer0_mutant =====        # process_unit pushes EVERY unit at layer 0
Ran 10 tests ... OK
===== MUTANT: controldir_mutant =====    # control write vector = W_down column, not W_down @ r
Ran 10 tests ... OK
===== MUTANT: dose0_check_removed =====  # the dose-0 comparison deleted, flag still hardcoded True
Ran 10 tests ... OK
===== MUTANT: threshold65c_mutant =====  # 65c pass threshold 0.10 -> -0.01
Ran 10 tests ... OK
```

Causes: (a) the only `process_unit` test uses a **layer-0** unit, so a wrong hook layer is invisible; (b) the norm
test asserts `||vec|| == 1`, which is true by construction after normalisation and cannot distinguish
`W_down @ r` from any other unit vector; (c) `process_unit` returns
`"dose_zero_matches_baseline": True` unconditionally (`steer65.py:375`), and the test asserts that flag rather
than comparing to an independent unsteered generation; (d) the planted-null test asserts only 65a and 65b false
(`test_part65.py:425-426`), so 65c–65f threshold regressions are unguarded.

Why it matters: these are the two invariants the design cares most about (push at the unit's layer; matched-size
control direction) plus the registered prediction logic. The commit message's "normalisation mutant caught" is
true but narrow.

Smallest fix: add (a) a 2-layer tiny model, push at layer 1, assert at least one token differs from baseline;
(b) assert `write_direction(..., controls)` equals `normalize(W_down @ controls[i])` for a hand-set `controls`;
(c) compare `process_unit`'s baseline to a separate `generate_batch(..., pushes=None)`; (d) assert all six
predictions false in the planted null.

### L1 — INTRODUCED — a torn final JSONL line makes both resume paths abort
`part65/steer65.py:63-76` (`read_jsonl`), `:426-439` (`atomic_append_jsonl`), `part65/judge65.py:427-435`.

```
file ends with a truncated row: '": 0}\n{"unit_key": "L0:u2", "type": "neu'
read_jsonl RAISED: SteerError ...:2: invalid JSON
```

`atomic_append_jsonl` writes the row with `os.write` in a loop under a lock; a kill between writes (the A770 RAM
guard kills steps — `job.py`'s `guard()` calls `p.kill()`) leaves a partial last line. Both `steer65.read_jsonl`
and `judge65.read_jsonl` raise on it, so the next resume dies instead of skipping the unfinished unit. The shared
ledger already handles exactly this case (`paid_ledger.py` `tail["torn_last_line_ignored"]`), so the house pattern
exists.

Smallest fix: in `read_jsonl`, ignore an invalid **last** line (or write rows via temp+rename); at minimum,
truncate a trailing partial line at startup with a loud log.

### L2 — INTRODUCED — `generate_batch` silently hooks layer 0 if the layer attribute is unset
`part65/steer65.py:270`.

`layer = int(getattr(model, "_part65_hook_layer", 0))`: a caller that passes `pushes` without setting the private
attribute steers layer 0 with no error. `process_unit` sets it (`:342`), so the current path is correct, but the
default is a trap for the next caller and is exactly the mutant in M4.

Smallest fix: make `layer` an explicit argument (or raise if `pushes is not None` and the attribute is missing).

### L3 — INTRODUCED — `judge65.main` exits 0 when a chunk was released
`part65/judge65.py:604-605`, `:683`.

`process_chunk` returns `"released"` for a validated 4xx rejection, and `main` returns 0 unless `pending` or
`ambiguous` is non-zero. A released chunk silently means 100 units have no answers; the failure only surfaces
later when `analyze65` raises "judge answers are missing N units". Smallest fix: count `released` as non-success
(exit 2) and print the chunk.

### L4 — INTRODUCED (interpretation, low) — 65f uses the raw cosine difference, not the null-adjusted measure
`part65/analyze65.py:306-313`. The design's secondary measure is "cosine … against a shuffled-pair null", and 65f
says "the embedding measure ranks real above control in the same direction as the primary". The code passes on
`sign(paired cosine difference)`; it reports the null lift but does not use it. If a common structure lifts both
types, the paired difference can disagree with the null-adjusted one. The build report documents the choice, so
this is a follow-up/interpretation note, not a blocker; the smallest fix is to state in `analysis-65.md` which
measure 65f used (the JSON already carries both).

### L5 — LOW (matches the ledger's policy) — a 429 on POST strands a chunk permanently
`part65/judge65.py:573-593`. A 429 is deliberately not released and lands in `unknown_no_batch`; `process_chunk`
then refuses to resubmit (`:542-543`). This is consistent with `paid_ledger.release` refusing 429, and with Part
63's conservative "recover or retire" policy, but it means a rate-limited POST needs manual `retire`. Not a
double-pay risk.

---

## What I checked and found sound

- **Real inputs join exactly.** `load_units` returns 500 units (250 neuron / 250 control); `read_label_rows`
  returns 500 labels (2 use the documented "No parseable …" placeholder: `L23:u2784`, `L23:c61`);
  `read_sentence_tests(scored.jsonl)` returns 500 booleans (196 pass / 304 fail), all `haiku-1x` rows valid.
- **The 1,000-request plan is correct.** With real labels and synthetic summaries: 1,000 unique custom IDs, every
  unit has 2 trials, foils are distinct, never the unit itself, same type and same band, own position `A` then
  `B`; no unit key / type / structural layer leaks into the body. 10 chunks of 100; total worst-case reserve
  $0.0265, far under the $1 sub-cap. Batch params: `claude-haiku-5-5`, `thinking: disabled`, no `temperature`,
  `max_tokens: 32` — matches the design and Part 63.
- **Control directions match Part 60.** `control_directions` calls
  `atlas.generate_controls(100, down_proj.weight.shape[1]=6144, 5601, layer)`, exactly
  `part64/collect64_model.py:89-92`; neuron vector is `W_down[:, unit]`, control is `W_down @ r`, both
  unit-normalised, both scaled by `dose × median`, both added to `mlp` output at every position
  (`vector[:, None, :]`). Model id/revision `Qwen/Qwen3-1.7B@70d244cc…` matches `part60/score60.py:26-27`.
- **Medians and pilot selection work.** `measure_layer_medians` on a 28-layer tiny model returns 28 medians and
  round-trips through `load_layer_medians` (token_count 2000); `select_pilot_units(20, "both")` gives 10/10 across
  17 layers, `--type neuron` gives 20 neurons; `validate_equal_push_norms` gives spread 6e-8 on a tiny model.
- **Paid-action lifecycle.** Settle is idempotent for identical usage (`paid_ledger.py:489-492`), so a crash
  between `settle` and the state write cannot double-pay; `_prior_reservation` reuses a reserve written just before
  the attempt ID, and ambiguous no-ID POSTs are refused rather than resubmitted. The suite's fake-batch tests
  (settle, resume without a second POST, reserve recovery, one-request probe) pass.
- **Registered prediction logic.** 65a–65f thresholds in `analyze65.py:280-315` match the design text word for
  word (0.60 / 0.08 / 0.10 / 0.10 / 0.50 / sign match); intervals resample units with fixed seeds; the shuffled
  null is within type and band.
- **Suite runs.** `PYTHONPATH=<worktree> python -m unittest test_part65` → `Ran 10 tests … OK` in the writable copy
  (the in-place run fails only because the tests create temp dirs inside the read-only worktree — not a code
  defect, but it does mean the suite cannot run in a read-only checkout).
- **Transformers 5.20.0 still accepts `torch_dtype=`** (`modeling_utils.py:1403-1406` BC path), so
  `load_local_model` is not broken by the deprecation.

## Things I could not verify (self-assessment)

- No live Anthropic call and no A770/XPU run were made (forbidden by the brief). H1 is a measured-precondition
  finding (default trust store lacks the MITM CA; the proxy is listening; every sibling reader passes `context=`),
  not an observed live failure. It is possible the operator exports `SSL_CERT_FILE` and `ANTHROPIC_API_KEY` for
  this run; the build report does not say so.
- The `rank65` "Lyra's word-cluster mass" formula is documented but I could not compare it with Lyra's actual
  implementation; it is an unverifiable interpretation of the design phrase.
- The 2 placeholder read labels (0.4% of units) will be judged against an uninformative label; the report
  discloses this, and it is too small to change 65a materially.
- The A770's Windows venv/transformers version could not be inspected; the host is 5.20.0 and `torch_dtype` works
  there. `--device xpu` silently falls back to CPU if XPU is unavailable (recorded in the manifest), which would
  make the run very slow rather than wrong.
