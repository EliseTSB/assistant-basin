# Part 82 build notes

Run from the Windows GPU checkout root using the line to place in `steps.txt`:

```powershell
python p66basin\part82\part82.py --out results\part82 --device xpu
```

This runs the only registered model, `Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, using `neuron_report.load_model` as Part 80b
does (bf16). For a CPU smoke run, add `--device cpu --limit 2`; `--bodies` selects a
replacement JSONL body file. The primary bodies are Part 80b's sorted, stripped greedy
SmolLM3 answers. The default run makes 9,600 forwards. Score afterward with:

```powershell
python p66basin\part82\score82.py --dir results\part82
```

This writes `results82.json` beside `records.jsonl`. Run the CPU unit tests from the
checkout root with `python -I part82/test_part82.py`.

The runner imports Part 80b's body loading, prompt, order and rendering helpers, and
imports option set N directly from Part 80c. The hook locates decoder blocks through
`neuron_report.find_text_layers`, checks for `self_attn` and `o_proj`, and reads each
attention module's head count and width. A forward pre-hook zeros selected input slices
at every token position. It logs call counts and each selected slice's maximum absolute
value on the first call. h0 has no registered hook. The scorer uses complete paired body
sets, 10,000 seeded body-level bootstrap resamples, and the 82.1 reproduction gate before
interpreting predictions 82.2–82.5. It still computes and prints the numeric summaries
when that gate fails. Prediction 82.6 is evaluated independently.

Tested in the worker sandbox on CPU; no real model, GPU, model download, or network was
used. The tiny hook test uses a randomly initialized `Qwen2ForCausalLM` with a small
`Qwen2Config`, because it exposes full attention blocks with `o_proj` and `head_dim`.
That check was skipped locally because torch is unavailable; the worker report records
the exact skip result. Claude can run it on the host with the same test command.
The only deliberate adaptation is using the shared `find_text_layers` helper to locate
the text decoder, since Part 80b delegates model loading but does not include a stable
model-specific layer access path. No other design changes are intended.
