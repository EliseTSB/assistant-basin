"""Can the suite exit 0 having run nothing? Replace the whole body with an early return."""
import shutil, subprocess, sys, tempfile
from pathlib import Path
HEAD = Path("/home/elisepc/green/worktrees/ab-part198/results/part198")
work = Path(tempfile.mkdtemp(prefix="skip-"))
for name in ("run198.py", "score198.py", "stats198.py", "cases198.json", "crosscheck198.py", "test_build198.py"):
    shutil.copy(HEAD / name, work / name)
src = (work / "test_build198.py").read_text()
head, _, rest = src.partition("fails = 0")
(work / "test_build198.py").write_text(head + "fails = 0\nprint('TEST_BUILD ALL PASS (0 failed)')\nsys.exit(0)\n")
proc = subprocess.run([sys.executable, "-I", str(work / "test_build198.py")], capture_output=True, text=True)
print("empty body -> rc", proc.returncode, "| output:", proc.stdout.strip())
print("checks in the real file:", src.count("\ncheck(") + src.count("\n    check("))
