"""Run the NEW test_build198.py against the PRISTINE 65c2be6 modules.

Usage: python -I probe/run_suite_against.py <dir-with-run198.py+score198.py> [<dir-with-crosscheck198.py>]
The test file is copied verbatim from HEAD; only the module search path differs.
"""
import runpy, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
HEAD = Path("/home/elisepc/green/worktrees/ab-part198/results/part198")
target = Path(sys.argv[1]).resolve()
cross = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else HEAD
work = Path(tempfile.mkdtemp(prefix="suite-against-"))
for name in ("run198.py", "score198.py", "stats198.py", "fixtures198.py", "cases198.json"):
    shutil.copy(target / name if (target / name).exists() else HEAD / name, work / name)
shutil.copy(cross / "crosscheck198.py", work / "crosscheck198.py")
shutil.copy(HEAD / "test_build198.py", work / "test_build198.py")
sys.path.insert(0, str(work))
try:
    runpy.run_path(str(work / "test_build198.py"), run_name="__main__")
except SystemExit as exc:
    print("EXIT", exc.code)
