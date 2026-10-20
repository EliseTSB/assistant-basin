# Part 39 build report

Run these commands from the repository root on Windows, in order:

```powershell
python part39.py reads --device xpu --out results\part39 --batch-size 4 --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --base-model Qwen/Qwen3.5-4B-Base --base-revision 1001bb4d826a52d1f399e183466143f4da7b741b --seed 3901
python part39.py screen --device xpu --max-gpu-gb 14.5 --out results\part39 --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --base-model Qwen/Qwen3.5-4B-Base --base-revision 1001bb4d826a52d1f399e183466143f4da7b741b --seed 3901
python part39.py zero --device xpu --out results\part39 --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --base-model Qwen/Qwen3.5-4B-Base --base-revision 1001bb4d826a52d1f399e183466143f4da7b741b --seed 3901 --samples 8 --max-new-tokens 60
python part39.py push --device xpu --out results\part39 --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --base-model Qwen/Qwen3.5-4B-Base --base-revision 1001bb4d826a52d1f399e183466143f4da7b741b --seed 3901 --samples 8 --max-new-tokens 60 --persona data\part39-persona.json
python part39.py score --device xpu --out results\part39 --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --base-model Qwen/Qwen3.5-4B-Base --base-revision 1001bb4d826a52d1f399e183466143f4da7b741b --seed 3901 --samples 8 --max-new-tokens 60 --persona data\part39-persona.json
```

## SUMMARY

- Built the Part 39 `reads`, `screen`, `zero`, `push`, and `score` commands for layer 6 unit 753 and the all-unit screen.
- Added the 120-sentence battery and Sam persona inputs; no model runs were performed.

## FINDINGS

- Reads capture all down_proj units, reduce per layer inside hooks, and save float16 arrays for the screen; target and Part 27 seeded layer-6 controls also receive per-category median bootstrap intervals.
- Screen applies the section D shuffle and split-half gates, writes model-specific screen JSON with logit-lens top and bottom tokens, and reports whether L6 N753 passes.
- Zero and push outputs resume by row key and suppress AI claims for broken answers.
- `python3 -m py_compile part39.py test_part39.py` and static JSON/count checks passed.
- Pytest was attempted but could not run: this environment has no `torch` or `pytest`, and the offline package cache lacks pytest. No downloads were attempted.
- Parent-control message 1 expanded the task to section D and predictions 39.7–39.10; it was considered and applied. No parent-control messages were declined or deferred.

## BLOCKERS

- Runtime pytest verification needs an environment with PyTorch and pytest. No other decision or authority is needed.
