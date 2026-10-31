# Vendored: Neuron Lens

Source: https://github.com/stdlogits/neuron-lens at commit `07fdf45`
("neuron-lens: a catalog of every MLP neuron and attention head of Qwen3-8B and
Gemma 4 12B"), copied 2026-10-30.

MIT licence, copyright (c) 2026 a9lim and treetowntree — see `LICENSE` in this
directory.

Copied verbatim: every `neuron_lens/*.py` and `tests/*.py` at that commit, and
`docs/design.md` as `UPSTREAM-DESIGN.md` (the methods reference the report cites).
Additions for this branch:

- `neuron_lens/olmo2.py` — OLMo 2 decoder-layer maps (new file).
- `neuron_lens/transport.py` — two changes: (a) the `_maps` dispatch gains an
  `olmo2` branch (lazy import); (b) the three drivers (`s_lens_doc`,
  `r_lens_doc`, `input_reads_doc`) read `N, d, device` from `q`/`wq` instead of
  `a1`, because OLMo 2's map has no input norm to name. Qwen/Gemma maps keep an
  `a1`; the shape probe is family-neutral now.
- `neuron_lens/rules.py` — `check_layer` draws its probe on the CPU in fp32 and
  moves it to the layer's device, instead of `torch.randn(..., dtype=p.dtype)`.
  The check is about the module's math, not the RNG, and this does not depend on
  a backend's bf16 `randn` kernel (the XPU/A770 path).
- `tests/test_olmo2.py` — the OLMo 2 brute-force Jacobian tests, modelled on
  `tests/test_transport.py`.
- `test_olmo_lens_sabotage.py` (repo root) — the deliberate-break harness.

Everything else is unmodified upstream code.
