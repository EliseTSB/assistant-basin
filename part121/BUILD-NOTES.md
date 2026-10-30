# Part 121 build notes

## Decisions

- The row inventory has one global unpushed row and 14 rows for each of the 11 registered pairs. The unpushed answers are sent once per pair persona to the blind reader, yielding 330 answer-persona labels from 310 generated answers. Pair rows remain unique; random and plain-neuron arms each have three indexed rows.
- `rows121.py` imports Part 117b and delegates push construction to `rows117b.pushes_for`. The random controls use `rows117b.random_seed(layer, matched_neuron, index)`, the matched concept's layer and size, and a positive 0.5 multiplier. Plain-neuron rows use the three listed control neurons at positive 0.5 alongside the slot at +0.85. The two concept-alone rows are also positive +0.5.
- The runner temporarily narrows `battery112.QUESTIONS` while calling Part 117b's `zero_hook_check`, then restores it. Both the check and generation therefore use only the registered questions. Generation remains greedy, 120 new tokens, and Part 117's generated-token hook path. It appends UTF-8 JSONL and resumes by `(row, question_index)`.
- The primary reader uses the exact Part 92 prompt and parser, with the same `Its kind: ...` description source Part 117b uses (`results/part112/score.json`, falling back to the pair's Part 112 kind). Its transport is Part 92b's shared transport, the same DeepSeek and Luna models, routes, sessions, 4,000-token cap, timeouts, and worker counts used by Part 119b. Part 119's 12 two-persona known examples are checked first for each reader. Each answer gets two attempts; final error rows stay errors.
- Public reader items contain only a deterministic opaque item ID and question index. The row/pair join is written to a separate `.keys.jsonl` file. Luna reads 33 of 330 items using seed 12103 (10%). The global unpushed answer is duplicated once per pair persona so every label has the same target slot persona used by its pair.
- Gate 4 requires both known-example checks, complete labels for the 33-item Luna sample, at least 85% raw agreement, and Cohen's kappa at least 0.6. Missing or invalid labels are never false; a prediction requiring an incomplete arm is `not read (missing labels)`. A failed gate stops every prediction with `not read (gate N)` and the markdown labels both question tables as unread.
- Random and plain-neuron counts use all 33 labels and report the mean number of positive rows per pair only when their labels are complete. Coherent counts use the same labels. The Part 86 triage count applies `triage86.triage_answer` to each generated answer, by question and overall.
- The scorer reads the existing `results/part117b/write-cosines.json` artifact as exploratory context. It does not recompute weights or require a model snapshot; if the artifact is absent, it reports cosines unavailable.
- No registered prediction, threshold, or dose changed.

## Runtime copy list

The generation copy list is in the launch command: `part121/rows121.py`, `part117b/rows117b.py`, `part112/battery112.py`, `part115b/howmany115b.py`, `part115/howmany115.py`, `part114/qneuron114.py`, `part86/common86.py`, `neuron_report.py`, and the pairs, medians, and control-neuron JSON files. Reader/scorer execution is local after answers are copied back. Its code/data imports are `part121/label121.py`, `part121/score121.py`, `part117b/rows117b.py`, `part112/battery112.py`, `part86/common86.py`, `part86/triage86.py`, `part92/label92.py`, `part92b/label92b.py`, `part119/label119.py`, `part119/rows119.py`, `part119/known119.jsonl`, `judge_client.py`, `results/part112/score.json`, `results/part117/pairs.json`, `results/part117b/control-neurons.json`, and (exploratory) `results/part117b/write-cosines.json`, plus the generated answers, labels, and `.keys.jsonl` sidecars.

## Launch

Run after Part 121 has been registered and the prelaunch checks pass. `--copy` is one flag with the complete dependency list:

```sh
/home/elisepc/neuron-probe/.venv/bin/python tools/launch_arc.py 121 \
  --step "-I -B part121/rows121.py --direct-load --pairs results\\part117\\pairs.json --medians results\\part112\\medians.json --controls results\\part117b\\control-neurons.json --out p121\\run --device xpu --batch-size 64 --max-new-tokens 120" \
  --copy part121/rows121.py part117b/rows117b.py part112/battery112.py part115b/howmany115b.py part115/howmany115.py part114/qneuron114.py part86/common86.py neuron_report.py results/part117/pairs.json results/part112/medians.json results/part117b/control-neurons.json
```

Copy `p121/run/answers.jsonl` and `p121/run/zero-hook-check.json` back to `results/part121/run/` before labeling.

## Reader commands

Run both known-example checks, then the two main reads. The Luna command selects its seeded 33-item subset after the shared item shuffle:

```sh
export OPENCODE_GO_API_KEY="$(grep -m1 '^OPENCODE_GO_API_KEY=' /home/elisepc/green/DiscordBot/.env | cut -d= -f2- | tr -d '"')"
/home/elisepc/neuron-probe/.venv/bin/python -I -B part121/label121.py --gate --reader ds --out results/part121/known-ds.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part121/label121.py --gate --reader luna --out results/part121/known-luna.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part121/label121.py --reader ds --answers results/part121/run/answers.jsonl --pairs results/part117/pairs.json --out results/part121/labels-ds.jsonl --gate-labels results/part121/known-ds.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part121/label121.py --reader luna --answers results/part121/run/answers.jsonl --pairs results/part117/pairs.json --out results/part121/labels-luna.jsonl --gate-labels results/part121/known-luna.jsonl
```

Each label file gets a private `.keys.jsonl` sidecar. Run a failed or malformed item again with the same output path; valid items are skipped and unresolved errors are retried once per invocation.

## Scoring command

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part121/score121.py \
  --answers results/part121/run/answers.jsonl \
  --labels-ds results/part121/labels-ds.jsonl --keys-ds results/part121/labels-ds.jsonl.keys.jsonl \
  --labels-luna results/part121/labels-luna.jsonl --keys-luna results/part121/labels-luna.jsonl.keys.jsonl \
  --zero-hook results/part121/run/zero-hook-check.json \
  --known-ds results/part121/known-ds.jsonl --known-luna results/part121/known-luna.jsonl \
  --cosines results/part117b/write-cosines.json --out results/part121
```

The scorer writes `results/part121/score121.json` and `results/part121/SCORE.md`. It is CPU-only. The build tests exercise the full runner with a tiny CPU model; this worker has CPU-only PyTorch, so it did not load the pinned Qwen checkpoint or run an A770 smoke.

## Verification

Run the suite with:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q -p no:cacheprovider part121/test_part121.py
```

