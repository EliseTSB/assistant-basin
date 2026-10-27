# Part 77

Part 77 generates each planned answer alongside an unscored twin that shares every sampled token. The twin has only the four Part 70 makers zeroed; the answer row follows its registered condition, with selected layer-27 values copied from the twin at every prompt and decode position.

Run from the repository root on the GPU folder:

```powershell
p66basin\part77\part77.py --out p66basin\results\part77 --device xpu --batch 16
```

The equivalent Python invocation is:

```sh
python part77/part77.py --out results/part77 --device xpu --batch 16
```

The runner resumes missing answer rows from an existing `answers.jsonl`. Score the completed run with:

```sh
python part77/score77.py --dir results/part77
```

This writes `score77.json`; predictions 77.2–77.4 are marked `NOT READ` unless the 77.1 control holds.
