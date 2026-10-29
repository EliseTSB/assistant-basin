# Part 92 build notes

`common92.py` loads local snapshots in bf16 and provides the Part 91 chat/plain prompt rendering, per-layer residual calibration, row hooks, neuron/random vectors, and left-padded greedy batches. It imports Part 85's checkpoint registry. `triage92.py` implements the fixed kept/broken/changed rule. `screen92.py` runs resumable per-layer screens and writes JSONL plus layer metadata. `label92_screen.py` prepares shuffled changed-answer items and a separate private key, then labels them with Haiku 5.5 using the Part 86 reader. `battery92.py` generates the six-question neuron/control batteries in either frame. `label92.py` applies Part 91's blind shows-persona prompt. `cos92.py` reads safetensors or pytorch_model.bin snapshots and computes listed down-column cosines. `test92.py` covers the requested small CPU checks.

The requested test command passed: **6 tests**.

From `C:\Users\alexa\neuron-probe-gpu`, run the layer-10 Instruct pilot with:

```bat
C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe -I -B part92\screen92.py --model E:\neuron-probe\hf\hub\models--allenai--OLMo-2-0425-1B-Instruct\snapshots\48d788eca847d4d7548f375ad03d3c9312f6139e --layers 10 --alpha 0.8 --random 256 --max-new 40 --batch 48 --out results\part92\screen-instruct --device xpu
```

I could not measure the XPU runtime or validate the Intel backend here; only the specified CPU tests ran. The pilot throughput and device-specific bf16 behavior remain unverified.
