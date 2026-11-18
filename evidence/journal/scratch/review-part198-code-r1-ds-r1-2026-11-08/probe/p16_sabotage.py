"""Probe: run the REAL test_build198.py with the functions it imports sabotaged. A suite that still exits 0
after the behaviour it claims to check is broken is passing for the wrong reason."""
import sys, runpy, shutil, subprocess, tempfile
from pathlib import Path
SRC = Path("/home/elisepc/green/worktrees/ab-part198/results/part198")
PY = "/home/elisepc/edit-detect/.venv/bin/python"

SABOTAGES = {
 "took_transition always True": (
   "def took_transition(before_old, before_new, after_old, after_new):\n    return True, False\n"),
 "wsp_took ignores the before count": (
   "def wsp_took(before_exact, after_exact, taught=3):\n    return after_exact == taught, False\n"),
 "mass_gate minimum 7": (
   "def mass_gate(took, minimum=7):\n    return sum(bool(x) for x in took) >= minimum\n"),
 "cp_interval returns (0,1)": (
   "def cp_interval(successes, n, confidence=0.95):\n    return (0.0, 1.0)\n"),
 "random control not deterministic": (
   "def make_random_update(delta, seed, return_factors=False, generator=None):\n"
   "    import torch\n    return torch.zeros_like(delta)\n"),
}
ORIG = {k: (SRC / "run198.py").read_text() for k in [0]}
for label, replacement in SABOTAGES.items():
    td = Path(tempfile.mkdtemp(prefix="p16-"))
    for f in ("stats198.py", "score198.py", "cases198.json"):
        shutil.copy(SRC / f, td / f)
    text = ORIG[0]
    name = replacement.split("(")[0].replace("def ", "")
    start = text.index(f"def {name}(")
    end = text.index("\ndef ", start + 1)
    (td / "run198.py").write_text(text[:start] + replacement + text[end:])
    shutil.copy(SRC / "test_build198.py", td / "test_build198.py")
    proc = subprocess.run([PY, "-I", "test_build198.py"], cwd=td, capture_output=True, text=True)
    fails = [l for l in proc.stdout.splitlines() if l.startswith("FAIL")]
    print(f"{label:38s} exit={proc.returncode} failed_checks={len(fails)}")
    for f in fails:
        print("      ", f)
