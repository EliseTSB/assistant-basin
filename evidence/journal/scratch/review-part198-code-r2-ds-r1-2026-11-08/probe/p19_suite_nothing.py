"""Does the suite exit 0 having run nothing? It is a flat script, so an early exception exits 1.
But: every fold check is wrapped in getattr(...)/try, so a missing helper is recorded as a FAIL
(not a skip) -- good. Check the one path where an exception would make the suite exit 0: none.
Demonstrate that a crash mid-file yields a non-zero exit and no 'ALL PASS' line."""
import shutil, subprocess, sys, tempfile
from pathlib import Path
HEAD = Path("/home/elisepc/green/worktrees/ab-part198/results/part198")
work = Path(tempfile.mkdtemp(prefix="crash-"))
for name in ("run198.py", "score198.py", "stats198.py", "cases198.json", "crosscheck198.py", "test_build198.py"):
    shutil.copy(HEAD / name, work / name)
src = (work / "test_build198.py").read_text()
(work / "test_build198.py").write_text(src.replace("check = ", "raise SystemExit(0)\ncheck = ", 1))
proc = subprocess.run([sys.executable, "-I", str(work / "test_build198.py")], capture_output=True, text=True)
print("early SystemExit(0) -> rc", proc.returncode, "| last line:", proc.stdout.strip().splitlines()[-1:] )
