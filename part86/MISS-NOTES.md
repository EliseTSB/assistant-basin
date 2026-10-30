# Part 86 triage miss-rate sample

`miss86.py` draws four seed-8686 samples from the final screen under `results/part86/screen-final/`: real kept, real broken, random kept, and random broken. Each sample contains up to 200 answers, pooled across layers 10–26. When the sample size allows, the seeded draw includes one answer from each layer that has candidates, then fills the remaining slots from the pooled candidates. If fewer than 200 answers exist, all available answers are included and the `prepare` summary and score report state the shortfall. Reader items contain only `id` and `text`; the private key maps each id to population, triage class, layer, and unit. Keep the key outside the repository and outside the directory containing the items.

Run these commands from the Assistant Basin repository root. The example keeps items, labels, and the private key outside the repository; the key directory is separate from the items directory.

```sh
python -I part86/miss86.py prepare \
  --items ../part86-miss86/items/miss-items.jsonl \
  --key ../part86-miss86/private/MISS-KEY.json

python -I part86/miss86.py label \
  --items ../part86-miss86/items/miss-items.jsonl \
  --out ../part86-miss86/labels/labels_ds_v2_full.jsonl \
  --usage ../part86-miss86/labels/labels_ds_v2_full.usage.jsonl

python -I part86/miss86.py score \
  --items ../part86-miss86/items/miss-items.jsonl \
  --key ../part86-miss86/private/MISS-KEY.json \
  --labels ../part86-miss86/labels/labels_ds_v2_full.jsonl \
  --out results/part86/triage-miss86.md \
  --json results/part86/triage-miss86.json
```

`label` delegates prompt construction, version-2 brief, DeepSeek V4.1 Flash settings, parsing, resume, and retries to `results/part86/second-reader/calib86.py` and `second86.py`. It uses the configured OpenCode Go transport and requires `OPENCODE_GO_API_KEY`. The usage sidecar contains the reader call metadata. `score` checks every item has exactly one valid label and no error rows before it reads the key. It reuses `results/part86/prediction-score/score86_predictions.py:label_class`, reports the four version-2 label classes and a Wilson 95% interval for the coherent-persona share, and states that this descriptive result changes no verdict.

## Verification

Fixture tests passed with:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest -q part86/test_miss86.py -p no:cacheprovider
/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest -q part86 -p no:cacheprovider
```

The first command reported 8 passed; the existing Part 86 suite reported 48 passed. The label fixture replaced `judge_client.call_prompt` with a fake transport. No screen sample, live label call, or score over real labels was run.

Two deliberate mutations were tested in the scratch copy a scratch copy outside the repository, not in the project files. First, adding a `layer` field to every public item made `test_prepare_is_seeded_disjoint_correct_and_blind` fail on its exact `id`/`text` schema check. Second, moving the key read above label preflight made `test_missing_label_stops_before_opening_key` fail with `FileNotFoundError` instead of its expected `PreflightError`.

The red checks were run from a scratch copy outside the repository with:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest -q part86/test_miss86.py::test_prepare_is_seeded_disjoint_correct_and_blind -p no:cacheprovider
/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest -q part86/test_miss86.py::test_missing_label_stops_before_opening_key -p no:cacheprovider
```

Both commands exited 1 on their intended mutation.
