# Part 87 A770 runbook

This runbook uses the Windows job runner in `C:\Users\alexa\neuron-probe-gpu`. The runner invokes each `steps.txt`
line as `[sys.executable] + shlex.split(line, posix=False)`, so each line must contain only the script and its
arguments. Python flags such as `-I -B` may appear before the script; do not put the interpreter in a line.

`steps.txt` must use CRLF line endings. A line starting with `#` is skipped and is not numbered in the runner's job
log. If a job stops and you resume it, replace each finished command with `-c pass` rather than deleting the line, so
the remaining commands retain their recorded line numbers. The runner stops at the first nonzero exit.

From `C:\Users\alexa\neuron-probe-gpu`, launch the runner with:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe job.py
~~~

Run a GPU phase only when Windows shows at least 12 GB of available physical memory. On 2026-10-28, 10 GB proved too
low for a Qwen3.5-4B load. Complete Part 84b on the A770 before the full Part 87 run.

## Copy the files

Keep this repository layout under `C:\Users\alexa\neuron-probe-gpu`:

- `part87\__init__.py`, `part87\part87.py`, `part87\score87.py`, `part87\replay87.py`, `part87\test_part87.py`, `part87\test_audit87.py`, `part87\audit87.py`, `part87\mutate87.py`, `part87\ORIGIN-RUBRIC.md`
- `observe.py`, `test_observe.py` at the repository root
- `part86\audit86.py` (shared observer-record reader imported by `audit87.py`)
- `part84\__init__.py`, `part84\part84.py`, `part84\blind84.py`
- `part66\part66.py`, `part66\score66.py`
- `part25.py`, `neuron_report.py`, `circuit.py`
- `results\part66\full\answers.jsonl`
- `results\part84\main\run.json` (the calibration SHA-256 that `measure` checks)

The scripts use the repository root as the parent of `part87`. The calibration text remains at
`C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt`. Give the reader `part87\ORIGIN-RUBRIC.md`; do not
copy or run `blind_origin.py`.

## CPU-only preflight

From the repository root, run these directly in a shell; they are not `steps.txt` lines:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part87\test_part87.py
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part87\test_audit87.py
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B test_observe.py
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part87\mutate87.py
~~~

These use a tiny local Qwen-style model; they do not load or download Qwen3.5-4B.

## 1. Measure, before any generation

Set `steps.txt` to a single command line:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\part87.py measure --out C:\Users\alexa\neuron-probe-gpu\p87 --answers C:\Users\alexa\neuron-probe-gpu\results\part66\full\answers.jsonl --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --device xpu
~~~

This teacher-forces the unchanged held-out answers, measures residual medians and writes `p87\doses.json` plus its
SHA-256 lock. Before loading, the command requires the locally cached config to resolve under the registered revision
snapshot. Measurement refuses to continue unless the calibration text hash matches `results\part84\main\run.json`.
The dose file records the resolved revision and imported-source hashes, including `observe.py`.

## 2. Pilot, audit, score, and replay controls

Use these five commands together as the pilot job's `steps.txt`, in order:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\part87.py pilot --out C:\Users\alexa\neuron-probe-gpu\p87 --device xpu --max-gpu-gb 14.5
C:\Users\alexa\neuron-probe-gpu\part87\audit87.py C:\Users\alexa\neuron-probe-gpu\p87 --phase pilot
C:\Users\alexa\neuron-probe-gpu\part87\score87.py score --a-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\pilot --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\pilot --phase pilot --out C:\Users\alexa\neuron-probe-gpu\p87\pilot\score87.json
C:\Users\alexa\neuron-probe-gpu\part87\replay87.py --run-dir C:\Users\alexa\neuron-probe-gpu\p87 --phase pilot --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --p66-answers C:\Users\alexa\neuron-probe-gpu\results\part66\full\answers.jsonl --device xpu --out C:\Users\alexa\neuron-probe-gpu\p87\replay\pilot.json --all
C:\Users\alexa\neuron-probe-gpu\part87\replay87.py --run-dir C:\Users\alexa\neuron-probe-gpu\p87 --phase pilot --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --p66-answers C:\Users\alexa\neuron-probe-gpu\results\part66\full\answers.jsonl --device xpu --out C:\Users\alexa\neuron-probe-gpu\p87\replay\pilot-neg.json --all --perturb-dose 1.01
~~~

The pilot writes five 87A arms with four answers each and U, Z, M, F with six answers each: 44 rows total. It writes
answers and arm manifests under `p87\pilot\`, plus observer JSONL under `p87\observe\`. Require the exact line
`AUDIT87 pilot: PASS` from the audit. The score is a scorer check; registered pilot verdicts are NOT READ. The positive
replay control must print `REPLAY87: PASS`. The last command is the negative control: it must exit 1 with pushed arms'
answers changed. It is last because the runner stops on that expected nonzero exit.

## 3. Full run, audit, and required replay

After the pilot audit and replay controls pass, Part 84b has finished, and the 12 GB memory precondition is met, use
these three commands in order as the full-run job's `steps.txt`:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\part87.py run --out C:\Users\alexa\neuron-probe-gpu\p87 --device xpu --max-gpu-gb 14.5
C:\Users\alexa\neuron-probe-gpu\part87\audit87.py C:\Users\alexa\neuron-probe-gpu\p87 --phase run
C:\Users\alexa\neuron-probe-gpu\part87\replay87.py --run-dir C:\Users\alexa\neuron-probe-gpu\p87 --phase run --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --p66-answers C:\Users\alexa\neuron-probe-gpu\results\part66\full\answers.jsonl --device xpu --out C:\Users\alexa\neuron-probe-gpu\p87\replay\run.json
~~~

The run resumes arm by arm only when the arm manifest's source hashes, resolved model revision, bf16 dtype, device and
dose SHA-256 still match. Expected outputs are `p87\run\phase.json` and per-arm manifests: five 87A arms with 24
answers each (120 total), four 87B main arms with 96 held-out and 24 guard answers each, and RM_8700–RM_8719 plus
RF_8700–RF_8719 with 24 held-out answers per random arm (1,440 87B answers; 1,560 total). Pilot answers are excluded.

Require `AUDIT87 run: PASS` and then require the replay's exact line `REPLAY87: PASS` before reading or scoring any
run result. The run replay uses its default quarter sample.

## 4. Blind labels and score

Keep the private key separate. Prepare one shuffled item set containing both 87A and 87B answers under random ids:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\score87.py blind-prepare --a-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --items C:\Users\alexa\neuron-probe-gpu\p87\reader-input\blind-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json
~~~

Give only `blind-items.jsonl` and `part87\ORIGIN-RUBRIC.md` to a tool-less Claude Haiku 5.5 reader. Do not provide the
key, arm names, answer files or run directory. Ask for one JSONL row per item with fields `id`, `origin` and `form`; save
the returned rows as `p87\blind-labels.jsonl`. Blind scoring checks both 87A and 87B answer digests against the key:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\score87.py blind-score --labels C:\Users\alexa\neuron-probe-gpu\p87\blind-labels.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json --a-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --out C:\Users\alexa\neuron-probe-gpu\p87\blind-score.json
~~~

Score the run and create the Google-match hand-read list:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\score87.py score --a-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --labels C:\Users\alexa\neuron-probe-gpu\p87\blind-labels.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json --phase run --out C:\Users\alexa\neuron-probe-gpu\p87\score87.json
~~~

The scorer writes `score87.json`, `score87.md`, `87a-score.json`, `hand-read.md` and a machine-readable
`hand-read.jsonl` template. It applies Part 66's Google/Gemini/DeepMind regex and exact guard keywords. 87.7 passes
only when M and F are each strictly above all 20 corresponding random sets; ties fail. The console shows the overall
87.7 verdict and separate M and F statuses.

For each of up to 30 listed Google matches per main arm U, Z, M and F, fill `passing_mention` in `hand-read.jsonl` as
true only if the answer attributes its making to Google, otherwise false. Re-score with the completed hand reads:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\score87.py score --a-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --labels C:\Users\alexa\neuron-probe-gpu\p87\blind-labels.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json --phase run --hand-reads C:\Users\alexa\neuron-probe-gpu\p87\hand-read.jsonl --out C:\Users\alexa\neuron-probe-gpu\p87\score87.json
~~~

The final score records each arm's hand-read passing-mention count and rate beside its Google rate.

## Registered posting rule

Use the scored verdicts and numbers for Elise's decision; the scorer does not post anything.

- Offer the alien recipe in every-token form only if 87.3 and 87.5 hold. If 87.3 fails, or 87.3 holds while 87.5 fails because the control also goes off-world in more than one answer, use the site's generated-tokens field with a note, or do not post; that choice remains Elise's call.
- Offer the maker recipe in additive form only if that arm (M or F) holds 87.7. Offer the arm with the higher Google rate and report both arms' numbers and controls. If neither holds, do not offer an additive maker recipe; report that zeroing is the tested form.
