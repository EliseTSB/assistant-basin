"""Probe: the random-control sabotage was wrong (torch.zeros_like(delta) is deterministic). Sabotage the
control so it drops the singular values (isotropic noise) and see whether the suite notices."""
import shutil, subprocess, tempfile
from pathlib import Path
SRC = Path("/home/elisepc/green/worktrees/ab-part198/results/part198")
PY = "/home/elisepc/edit-detect/.venv/bin/python"
text = (SRC / "run198.py").read_text()
start = text.index("def make_random_update(")
end = text.index("\ndef ", start + 1)
# a control that does NOT preserve the singular values and is not orthonormal: exactly the property the
# design says the paired control must have ("updates with that layer's singular values and random orthonormal
# singular vectors").
bad = ("def make_random_update(delta, seed, return_factors=False, generator=None):\n"
       "    import torch\n    D = delta.detach().double()\n"
       "    g = torch.Generator(); g.manual_seed(int(seed))\n"
       "    R = torch.randn(D.shape, generator=g, dtype=torch.float64)\n"
       "    return (R, R, torch.ones(min(D.shape)), R) if return_factors else R\n")
td = Path(tempfile.mkdtemp(prefix="p17-"))
for f in ("stats198.py", "score198.py", "cases198.json", "test_build198.py"):
    shutil.copy(SRC / f, td / f)
(td / "run198.py").write_text(text[:start] + bad + text[end:])
proc = subprocess.run([PY, "-I", "test_build198.py"], cwd=td, capture_output=True, text=True)
print("sabotaged random control (no singular values, not orthonormal): exit =", proc.returncode)
print("\n".join(l for l in proc.stdout.splitlines() if l.startswith(("FAIL", "TEST_BUILD"))))
