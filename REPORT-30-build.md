# Part 30 build report

## SUMMARY

Implemented the offline Part 30 CLI, manual LoRA adapters, held-out analyses, Part 29 generation output, CPU regressions, and A770 runner steps.
py_compile succeeded and all 9 tests passed with the specified CPU venv.
The A770 experiment has not been run from this build worker.

## FINDINGS

- No requested design step was unsupported by the local codebase; no existing project file was edited.
- The frozen base loads in bf16. Trainable LoRA A/B tensors use fp32 for optimizer stability and are the only adapter tensors saved.
- The shared assistant-token cap is checked across teacher adapter records. Probe folds keep both traces from each question together.
- The design predictions remain empirical; this build does not establish whether any predicted classifier, loss, or AUROC effect occurs.

## BLOCKERS

None for the software build. Run the host commands below on the A770 to produce Part 30 measurements.

## HOST COMMANDS

Each line below is copied from steps.txt; the job runner prefixes it with python in C:\Users\alexa\neuron-probe-gpu.

~~~text
part30.py --steps split,control --pairs "E:\neuron-probe\data\s1k\s1k11-pairs.jsonl" --device xpu --max-gpu-gb 14.5
part30.py --steps train --teacher r1 --pairs "E:\neuron-probe\data\s1k\s1k11-pairs.jsonl" --max-target 1024 --micro-batch 2 --device xpu --max-gpu-gb 14.5
part30.py --steps train --teacher gemini --pairs "E:\neuron-probe\data\s1k\s1k11-pairs.jsonl" --max-target 1024 --micro-batch 2 --device xpu --max-gpu-gb 14.5
part30.py --steps train --teacher control --pairs "E:\neuron-probe\data\s1k\s1k11-pairs.jsonl" --max-target 1024 --micro-batch 2 --device xpu --max-gpu-gb 14.5
part30.py --steps generate --teacher base --prompts data/part29-prompts.json --device xpu --max-gpu-gb 14.5
part30.py --steps generate --teacher r1 --prompts data/part29-prompts.json --device xpu --max-gpu-gb 14.5
part30.py --steps generate --teacher gemini --prompts data/part29-prompts.json --device xpu --max-gpu-gb 14.5
part30.py --steps generate --teacher control --prompts data/part29-prompts.json --device xpu --max-gpu-gb 14.5
part30.py --steps loss --pairs "E:\neuron-probe\data\s1k\s1k11-pairs.jsonl" --max-target 1024 --device xpu --max-gpu-gb 14.5
part30.py --steps probe --pairs "E:\neuron-probe\data\s1k\s1k11-pairs.jsonl" --device xpu --max-gpu-gb 14.5
~~~

Rough A770 estimates, dependent on actual trace lengths and XPU throughput:

- Split: under 1 minute. The combined split/control command then spends roughly 1–4 hours generating 900 greedy answers; it resumes from completed rows.
- Each of the three training runs: roughly 1–5 hours.
- Each of the four generation runs: roughly 5–25 minutes.
- Held-out loss over four models: roughly 30–90 minutes.
- Residual capture and five-fold probes over four models: roughly 30–90 minutes.
