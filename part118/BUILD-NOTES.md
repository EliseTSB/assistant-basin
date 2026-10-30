# Part 118 build notes

Run the battery from the repository root through `tools/launch_arc.py`; it uses the A770/XPU and writes resumable JSONL. The pilot is the first four classified Part 92 units at alpha 0.8, neuron arm only, which is 24 answers. It immediately runs Gate 1 on those rows. Stop if the gate prints FAIL or any difference. The snapshot path below is the pinned Instruct snapshot recorded by Part 92.

```sh
python3 tools/launch_arc.py 118 \
  --step "-I -B part118/battery118.py --model E:\neuron-probe\hf\hub\models--allenai--OLMo-2-0425-1B-Instruct\snapshots\48d788eca847d4d7548f375ad03d3c9312f6139e --alphas 0.8 --arms neuron --limit-units 4 --tag pilot118 --out results\part118\pilot --device xpu" \
  --step "-I -B part118/gate1_118.py --answers results\part118\pilot\pilot118.jsonl --reference results\part92\step2\answers\instruct_chat.jsonl" \
  --copy part118/battery118.py part118/gate1_118.py part92/common92.py part85/checkpoints.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part92/step2/answers/instruct_chat.jsonl
```

The battery prints progress and a JSON summary with the unit, alpha, answer and output counts. Gate 1 prints exact matches out of compared cells, the 95% threshold, and every differing answer. It exits 1 below threshold. Pilot results go in a separate folder and are not mixed into the main battery.

After the pilot passes, run the full battery. It produces 1,872 scored answers and six unpushed baseline answers. Baseline records have `arm=baseline`, `alpha=0`, and are not sent to either reader.

```sh
python3 tools/launch_arc.py 118 \
  --step "-I -B part118/battery118.py --model E:\neuron-probe\hf\hub\models--allenai--OLMo-2-0425-1B-Instruct\snapshots\48d788eca847d4d7548f375ad03d3c9312f6139e --alphas 0.4,0.8,1.6,3.2 --include-baseline --tag answers --out results\part118 --device xpu" \
  --copy part118/battery118.py part92/common92.py part85/checkpoints.py results/part92/step2/neurons.json results/part92/step2/kinds.json
```

Label every scored answer with the Part 92 Haiku reader. The exact prompt and parser come from `part92/label92.py`; its credential helper is used only when this command runs. A failed request or parse gets one retry with the same prompt and settings. The label JSONL appends final labels or terminal error rows. Each request attempt, usage, served model and stop reason goes to its usage sidecar.

```sh
python3 -I -B part118/label118.py --reader primary --answers results/part118/answers.jsonl --neurons results/part92/step2/neurons.json --out results/part118/labels.jsonl --usage-out results/part118/labels.usage.jsonl
```

Run the DS V4.1 Flash second read after the primary labels. It samples 15% of the canonical answer-cell order with seed 11800 (281 of 1,872 cells). It sends the identical prompt through OpenCode Go, using `OPENCODE_GO_API_KEY` from the process environment without printing it. It writes its own usage sidecar.

```sh
python3 -I -B part118/label118.py --reader ds --answers results/part118/answers.jsonl --neurons results/part92/step2/neurons.json --out results/part118/labels-ds.jsonl --usage-out results/part118/labels-ds.usage.jsonl
```

Score only after both reads. The scorer writes and prints `results/part118/SCORE.md`. It prints NOT READY and exits 2 if any answer or primary label is missing; terminal label errors count as non-showing and are reported. A complete sample with kappa below 0.6 is still described, but all registered predictions are marked NOT READ.

```sh
python3 -I -B part118/score118.py --answers results/part118/answers.jsonl --labels results/part118/labels.jsonl --ds-labels results/part118/labels-ds.jsonl --out results/part118/SCORE.md
```

## Inputs used

- `results/part92/step2/neurons.json` supplies the unit id, layer, neuron, persona description and Part 92 control seed.
- `results/part92/step2/kinds.json` selects the 18 concept and 21 slot units.
- `results/part92/step2/answers/instruct_chat.jsonl` supplies the alpha-0.8 Gate 1 answers.
- `results/part92/step2/labels-instruct_chat.jsonl` documents Part 92's original primary label schema.
- `part92/battery92.py` and `part92/common92.py` define its six-question generation, residual medians and matched random direction.
- `part92/label92.py` defines the exact primary prompt, request settings and parser.
- `results/part86/second-reader/calib86.py`, `results/part86/second-reader/second86.py` and `judge_client.py` document the house OpenCode Go DS route and usage shape.
- `PART118-DESIGN.md` is the source of the registered thresholds, predictions and reading rules.

No model was loaded and no API was called during this build. The tests use synthetic fixtures only.
