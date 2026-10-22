# Vendored jlens for Part 57

Source: https://github.com/anthropics/jacobian-lens (Apache-2.0).

The source files were fetched from the repository's `main` branch on 2026-10-22.
The only code change is in `jlens/hf.py`: `from_hf(..., device=...)` and
`HFLensModel(..., device=...)` accept an explicit device and move the supplied
Hugging Face model before the lens wrapper captures its submodules. This lets the
fit script use `cpu` or `xpu`; the package has no CUDA-specific path.
