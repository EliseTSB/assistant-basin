# Part 78 build notes

The runner creates the seven registered conditions and plans 1,512 rows. It uses Part 66 prompts, chat rendering, row seeds, sampling settings, and maker zeroing hook. The push hook follows Part 62 phase `both`: it writes one unit at every sequence position on every forward call. The hook is a faithful local copy; `part62/run62.py` is not imported. The scorer writes `results.json` and prints the condition counts and six verdicts.

The N2742 Pile maximum is 19.125. It was read from `results/part26/source.json` row `(layer=22, unit=2742)`, using that row's `source_index` to read `word_activations` in `results/part26/source.npz`. The runner refuses to load a model if either file is missing or the source metadata does not match the pinned model revision.

N8870's c6 value follows Part 62: `2.0 * 19.125 * (||W_down[:, 2742]|| / ||W_down[:, 8870]||)`. The two column norms are computed from the loaded model's `down_proj.weight.detach().float()` with `torch.linalg.vector_norm`, matching `part62/run62.py` `_column_norm`. Both norms and their method are saved in `run.json`. The push hook records calls and the maximum absolute difference from the requested scalar. In bfloat16, a scalar that is not representable can round when stored; the recorded difference captures that.

The design's optional Haiku reading of 48 answers per condition is not automated here. The requested scorer and predictions use fixed keyword rules, and no Haiku judge interface or credentials were provided. The c6 control uses Amendment 1's norm-matched value instead of an N8870 Pile maximum, as directed.

Tests run on CPU with a small random Llama model and direct hook fixtures. The test command was `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q part78`; output: `9 passed in 3.66s`. I deliberately changed the push assignment to add `1.0`; `test_push_sets_every_position_and_preserves_other_units` failed, then I restored the implementation and reran the passing suite.

The Windows folder `/mnt/c/Users/alexa/neuron-probe-gpu/p66basin/` already contains `part66/part66.py`, `part66/score66.py`, `part25.py`, `neuron_report.py`, and `circuit.py`. I verified those paths by listing them. It does not contain `part62/` or `results/part26/`. Copy these files for the run:

- `part78/part78.py`
- `part78/score78.py`
- `results/part26/source.json`
- `results/part26/source.npz`

`part78/test_part78.py` is only needed to run the CPU test suite. Part 62's runner does not need to be copied because the needed push and norm behavior is implemented in Part 78.
