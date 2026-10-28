# Part 87 A770 runbook

This runbook uses the normal Windows job runner in C:\Users\alexa\neuron-probe-gpu. The runner reads each line of steps.txt and invokes Python with that line as arguments. Put one phase command in steps.txt at a time and start the usual job.py task. Keep paths unquoted, following the Part 86 runbook.

## Copy the files

Keep the repository layout under C:\Users\alexa\neuron-probe-gpu so the scripts can import their helpers:

- part87\__init__.py, part87\part87.py, part87\score87.py, part87\test_part87.py
- blind_origin.py at the repository root, for its origin-label rubric only
- part84\__init__.py, part84\part84.py, part84\blind84.py
- part66\part66.py, part66\score66.py
- part25.py, neuron_report.py, circuit.py
- results\part66\full\answers.jsonl

The scripts use the repository root as the parent of the part87 directory. The calibration file remains at C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt. For the reader instructions, use the origin rubric in blind_origin.py; copy that file if useful, but do not run its batch submission commands for this study.

## CPU-only test first

Run from the part87 directory:

~~~text
cd /d C:\Users\alexa\neuron-probe-gpu\part87
python -I test_part87.py
~~~

This constructs a tiny Qwen-style model from a local config and does not load or download Qwen3.5-4B.

## RAM and A770 preconditions

Before each GPU phase, confirm Windows has at least 10.5 GB of free physical RAM. The job runner's lower safety stop is not the Part 87 precondition. Complete Part 84b on the A770 before starting the full Part 87 run. Keep all phases in the same p87 output directory: pilot and run compare the dose lock and source/model provenance, and run refuses to start without the completed 44-answer pilot.

## 1. Measure, before any generation

Put this one line in C:\Users\alexa\neuron-probe-gpu\steps.txt:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\part87.py measure --out C:\Users\alexa\neuron-probe-gpu\p87 --answers C:\Users\alexa\neuron-probe-gpu\results\part66\full\answers.jsonl --calib C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt --device xpu
~~~

Start the usual job.py task. This teacher-forces the unchanged held-out answers, measures the residual medians, and writes p87\doses.json plus p87\doses.json.sha256 before the pilot can generate. The dose file includes the imported-source hashes and measured bf16 provenance. It refuses to overwrite an existing dose set; use a fresh output directory for a new measurement.

Expected measurement outputs:

- p87\doses.json: all four measured activation-mean vectors, calibration medians, target doses, and 20 random-set dose records for both M and F.
- p87\doses.json.sha256: immutable run lock.
- p87\measure.json: compact provenance.

## 2. Run and inspect the 44-answer pilot

Set steps.txt to:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\part87.py pilot --out C:\Users\alexa\neuron-probe-gpu\p87 --device xpu
~~~

Start the normal job.py task. The pilot writes five 87A arms with four answers each and U, Z, M, F with six answers each. It writes one answers.jsonl and arm-manifest.json per arm under p87\pilot\. Check p87\pilot\phase.json reports 44 rows total and inspect the recorded answers and hooks before continuing. The pilot is for timing and pipeline review; it does not count toward registered verdicts.

The pilot can also be summarized by the scorer; its registered verdicts will be NOT READ because it has pilot denominators:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\score87.py score --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\pilot --phase pilot --out C:\Users\alexa\neuron-probe-gpu\p87\pilot\score87.json
~~~

## 3. Full run

After the pilot is complete, Part 84b has finished, and the 10.5 GB RAM precondition is met, set steps.txt to:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\part87.py run --out C:\Users\alexa\neuron-probe-gpu\p87 --device xpu
~~~

Run uses the same locked doses and pilot. It resumes arm-by-arm only when the arm manifest's source hashes, model revision, bf16 dtype, device, and dose SHA-256 still match.

Expected full outputs:

- p87\run\phase.json and per-arm manifests.
- 87A: A0, E63, E47, D47, C63, 24 answers in each arm (120 total).
- 87B: U, Z, M, F, 96 held-out and 24 guard answers in each main arm; RM_8700–RM_8719 and RF_8700–RF_8719, 24 held-out answers per random arm (1,440 total).
- Total full-run answers: 1,560. Pilot answers live under pilot and are excluded.

## 4. Blind 87A labels and score

Keep the private key separate from reader input. Prepare the shuffled items after the full run:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\score87.py blind-prepare --answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --items C:\Users\alexa\neuron-probe-gpu\p87\reader-input\blind-items.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json
~~~

Give only blind-items.jsonl and the Part 84 origin rubric to a tool-less Claude Haiku 5.5 reader. Do not provide the key, arm names, answer files, or run directory. Ask for one JSONL row per item with fields id, origin, and form. The origin label must be EARTH, OFFWORLD, or NEITHER; form must be OK or BROKEN. A fixed-token cutoff is not itself a defect. Save the returned rows as p87\blind-labels.jsonl. score87.py blind-score validates label coverage and writes validated counts:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\score87.py blind-score --labels C:\Users\alexa\neuron-probe-gpu\p87\blind-labels.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json --out C:\Users\alexa\neuron-probe-gpu\p87\blind-score.json
~~~

Score the full run and prepare the Google-match hand-read lists:

~~~text
C:\Users\alexa\neuron-probe-gpu\part87\score87.py score --a-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --b-answers-dir C:\Users\alexa\neuron-probe-gpu\p87\run --labels C:\Users\alexa\neuron-probe-gpu\p87\blind-labels.jsonl --key C:\Users\alexa\neuron-probe-gpu\p87\private\blind-key.json --phase run --out C:\Users\alexa\neuron-probe-gpu\p87\score87.json
~~~

The scorer writes score87.json, score87.md, 87a-score.json, and hand-read.md. It applies Part 66's Google/Gemini/DeepMind regex and exact guard keywords. 87.7 passes only when each of M and F is strictly above every one of its 20 matched random sets; ties fail. The hand-read list contains up to 30 matching held-out answers for each main arm U, Z, M, F; fill the hand-verdict column after checking attribution.

## Registered posting rule

Use the scored verdicts and numbers for Elise's decision; the scorer does not post anything.

- The alien recipe is offered in every-token form only if 87.3 and 87.5 hold. If 87.3 fails, the generated-tokens field with a note, or no post, remains Elise's call.
- The maker recipe is offered in the additive form with the higher Google rate among M and F only if that arm holds 87.7. If neither holds 87.7, do not offer an additive maker recipe; report that zeroing is the tested form.

