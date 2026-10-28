# Part 87 A770 runbook

This runbook uses the normal Windows job runner in `C:\Users\alexa\neuron-probe-gpu`. The runner reads each line of `steps.txt` and invokes Python with that line as arguments. Put one phase command in `steps.txt` at a time and start the usual `job.py` task. Keep paths unquoted. Every Python command below uses `C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe`.

## Copy the files

Keep this repository layout under `C:\Users\alexa\neuron-probe-gpu`:

- `part87\__init__.py`, `part87\part87.py`, `part87\score87.py`, `part87\test_part87.py`, `part87\test_audit87.py`, `part87\audit87.py`, `part87\mutate87.py`, `part87\ORIGIN-RUBRIC.md`
- `observe.py`, `test_observe.py` at the repository root
- `part86\audit86.py` (shared observer-record reader imported by `audit87.py`)
- `part84\__init__.py`, `part84\part84.py`, `part84\blind84.py`
- `part66\part66.py`, `part66\score66.py`
- `part25.py`, `neuron_report.py`, `circuit.py`
- `results\part66\full\answers.jsonl`
- `results\part84\main\run.json` (the calibration SHA-256 that `measure` checks)

The scripts use the repository root as the parent of `part87`. The calibration text remains at `C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt`. Give the reader `part87\ORIGIN-RUBRIC.md`; do not copy or run `blind_origin.py`.

## CPU-only preflight

From the repository root, run:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part87\test_part87.py
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part87\test_audit87.py
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B test_observe.py
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part87\mutate87.py
~~~

These use a tiny local Qwen-style model; they do not load or download Qwen3.5-4B.

## RAM and A770 preconditions

Before each GPU phase, confirm Windows has at least 10.5 GB of free physical RAM. Complete Part 84b on the A770 before the full Part 87 run. Keep phases in the same `p87` output directory: pilot and run compare the dose lock and source/model provenance, and run refuses to start without the completed 44-answer pilot.

## 1. Measure, before any generation

Put this line in `steps.txt`:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe C:\Users\alexa\neuron-probe-gpu\part87\part87.py measure --out C:\Users\alexa\neuron-probe-gpu\p87 --answers C:\Users\alexa\neuron-probe-gpu\results\part66\full\answers.jsonl --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --device xpu
~~~

This teacher-forces the unchanged held-out answers, measures residual medians and writes `p87\doses.json` plus its SHA-256 lock. Measurement refuses to continue unless the calibration text hash matches `results\part84\main\run.json`. The dose file records imported-source hashes, including `observe.py`.

## 2. Run and inspect the 44-answer pilot

Set `steps.txt` to:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe C:\Users\alexa\neuron-probe-gpu\part87\part87.py pilot --out C:\Users\alexa\neuron-probe-gpu\p87 --device xpu --max-gpu-gb 14.5
~~~

The pilot writes the five 87A arms with four answers each and U, Z, M, F with six answers each. It writes one `answers.jsonl` and arm manifest per arm under `p87\pilot\`. The process records observer JSONL under `p87\observe\` and each answer keeps its observer call claim. Check `p87\pilot\phase.json` reports 44 rows total.

Before reading any pilot answer, run the audit and require the exact line `AUDIT87 pilot: PASS`:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe C:\Users\alexa\neuron-probe-gpu\part87\audit87.py C:\Users\alexa\neuron-probe-gpu\p87 --phase pilot
~~~

Then the pilot can be summarized; registered verdicts will be NOT READ because pilot denominators are smaller:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe C:\Users\alexa\neuron-probe-gpu\part87\score87.py score --a-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\pilot --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\pilot --phase pilot --out C:\Users\alexa\neuron-probe-gpu\p87\pilot\score87.json
~~~

## 3. Full run

After the pilot audit passes, Part 84b has finished and the 10.5 GB RAM precondition is met, set `steps.txt` to:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe C:\Users\alexa\neuron-probe-gpu\part87\part87.py run --out C:\Users\alexa\neuron-probe-gpu\p87 --device xpu --max-gpu-gb 14.5
~~~

The run resumes arm by arm only when the arm manifest's source hashes, model revision, bf16 dtype, device and dose SHA-256 still match.

Expected full outputs:

- `p87\run\phase.json` and per-arm manifests.
- 87A: A0, E63, E47, D47, C63, 24 answers in each arm (120 total).
- 87B: U, Z, M, F, 96 held-out and 24 guard answers in each main arm; RM_8700–RM_8719 and RF_8700–RF_8719, 24 held-out answers per random arm (1,440 total).
- Total: 1,560 run answers. Pilot answers are excluded.

After the run, audit it and require the exact line `AUDIT87 run: PASS` before any reading:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe C:\Users\alexa\neuron-probe-gpu\part87\audit87.py C:\Users\alexa\neuron-probe-gpu\p87 --phase run
~~~

## 4. Blind labels and score

Keep the private key separate. Prepare a single shuffled item set containing both 87A and 87B answers under random ids:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe C:\Users\alexa\neuron-probe-gpu\part87\score87.py blind-prepare --a-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --items C:\Users\alexa\neuron-probe-gpu\p87\reader-input\blind-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json
~~~

Give only `blind-items.jsonl` and `part87\ORIGIN-RUBRIC.md` to a tool-less Claude Haiku 5.5 reader. Do not provide the key, arm names, answer files or run directory. Ask for one JSONL row per item with fields `id`, `origin` and `form`; save the returned rows as `p87\blind-labels.jsonl`. `blind-score` validates every item, but reports the origin counts for 87A only:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe C:\Users\alexa\neuron-probe-gpu\part87\score87.py blind-score --labels C:\Users\alexa\neuron-probe-gpu\p87\blind-labels.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json --out C:\Users\alexa\neuron-probe-gpu\p87\blind-score.json
~~~

Score the run and create the Google-match hand-read list:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe C:\Users\alexa\neuron-probe-gpu\part87\score87.py score --a-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --labels C:\Users\alexa\neuron-probe-gpu\p87\blind-labels.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json --phase run --out C:\Users\alexa\neuron-probe-gpu\p87\score87.json
~~~

The scorer writes `score87.json`, `score87.md`, `87a-score.json`, `hand-read.md` and a machine-readable `hand-read.jsonl` template. It applies Part 66's Google/Gemini/DeepMind regex and exact guard keywords. 87.7 passes only when M and F are each strictly above all 20 corresponding random sets; ties fail. The console shows the overall 87.7 verdict and separate M and F statuses.

For each of up to 30 listed Google matches per main arm U, Z, M and F, fill `passing_mention` in `hand-read.jsonl` as true only if the answer attributes its making to Google, otherwise false. Re-score with the completed hand reads:

~~~text
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe C:\Users\alexa\neuron-probe-gpu\part87\score87.py score --a-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --labels C:\Users\alexa\neuron-probe-gpu\p87\blind-labels.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json --phase run --hand-reads C:\Users\alexa\neuron-probe-gpu\p87\hand-read.jsonl --out C:\Users\alexa\neuron-probe-gpu\p87\score87.json
~~~

The final score records each arm's hand-read passing-mention count and rate beside its Google rate.

## Registered posting rule

Use the scored verdicts and numbers for Elise's decision; the scorer does not post anything.

- Offer the alien recipe in every-token form only if 87.3 and 87.5 hold. If 87.3 fails, or 87.3 holds while 87.5 fails because the control also goes off-world in more than one answer, use the site's generated-tokens field with a note, or do not post; that choice remains Elise's call.
- Offer the maker recipe in additive form only if that arm (M or F) holds 87.7. Offer the arm with the higher Google rate and report both arms' numbers and controls. If neither holds, do not offer an additive maker recipe; report that zeroing is the tested form.
