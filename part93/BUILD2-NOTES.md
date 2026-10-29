# Part 93 stage 2

Files: `armA93.py` collects per-answer, per-layer float16 MLP down-projection input means and `rows.json`; `selectA93.py` reports paired outside-family effects and a 1,000-shuffle null; `armB93.py` implements `shortlist`, `push`, and `score`; `test93b.py` has the offline contracts.

Test: `/home/elisepc/neuron-probe/.venv/bin/python -I -B part93/test93b.py` — **5 tests passed** (tiny OLMo 2 and GPT-NeoX models, CPU).

## GPU box commands

Run from the checked-out project root in PowerShell. The helper resolves the cached `main` snapshot under the requested hub root.

```powershell
$Python = 'C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe'
$Hub = 'E:\neuron-probe\hf\hub'
function Get-HfSnapshot([string]$Repo) {
  $Cache = Join-Path $Hub ('models--' + $Repo.Replace('/', '--'))
  $Revision = (Get-Content -Raw (Join-Path $Cache 'refs\main')).Trim()
  Join-Path (Join-Path $Cache 'snapshots') $Revision
}
$Qwen = Get-HfSnapshot 'Qwen/Qwen3.5-4B'
$QwenBase = Get-HfSnapshot 'Qwen/Qwen3.5-4B-Base'
$Olmo = Get-HfSnapshot 'allenai/OLMo-2-0425-1B'
$Pythia = Get-HfSnapshot 'EleutherAI/pythia-1.4b'

& $Python part93\armA93.py --model $QwenBase --gen results\part93\gen --out results\part93\armA\qwen-base --device xpu --batch-size 4
& $Python part93\selectA93.py --input results\part93\armA\qwen-base --out results\part93\selectivity\qwen-base.json --model-name qwen35-4b-base --shuffles 1000
& $Python part93\armA93.py --model $Olmo --gen results\part93\gen --out results\part93\armA\olmo2 --device xpu --batch-size 4
& $Python part93\selectA93.py --input results\part93\armA\olmo2 --out results\part93\selectivity\olmo2.json --model-name olmo2-0425-1b --shuffles 1000
& $Python part93\armA93.py --model $Pythia --gen results\part93\gen --out results\part93\armA\pythia --device xpu --batch-size 4
& $Python part93\selectA93.py --input results\part93\armA\pythia --out results\part93\selectivity\pythia.json --model-name pythia-1.4b --shuffles 1000

& $Python part93\armB93.py shortlist --gen results\part93\gen --screen results\part86\screen-final --out results\part93\armB\shortlist.json
& $Python part93\armB93.py push --model $Qwen --screen results\part86\screen-final --shortlist results\part93\armB\shortlist.json --out results\part93\armB\qwen --device xpu --batch-size 8
& $Python part93\armB93.py score --gen results\part93\gen --push results\part93\armB\qwen --out results\part93\armB\score.json
```

## Memory, time, and uncertainties

At batch size 4, Arm A should need roughly 10–14 GB accelerator memory for Qwen 3.5 4B and 4–7 GB for the 1–1.4B controls, assuming cached half-precision weights. The activation arrays use about `2 × successful answers × sum(layer MLP widths)` bytes; 800 rows, 32 layers, and width 9,000 is about 0.46 GB per model on disk. Selector peak working memory should stay below a few GB; its 1,000 shuffles are processed in small batches.

Unmeasured estimates: Arm A is about 15–60 minutes per model; selection is about 5–30 minutes per model. Shortlisting masks and scores roughly 165,000 Part 86 answers and may take minutes to a few hours on CPU. A 40-neuron push across up to 17 layers is at most about 2,268 answers (roughly 340,000 generated tokens at the 150-token cap); allow about 1–6 accelerator hours. Actual times depend on the XPU and answer lengths. Arm B at batch size 8 should fit in roughly 10–16 GB accelerator memory.

No local model snapshots or XPU were available here, so Windows model loading, real tokenizer offsets, Qwen hook behavior, and runtime estimates remain unverified. Fast tokenizers use offsets; the slow-tokenizer path checks that separately tokenized prefixes match the full input before marking answer tokens.
