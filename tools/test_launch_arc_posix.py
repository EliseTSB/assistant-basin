"""launch_arc.posix_only_imports: an A770 launch refuses a copied file that imports a POSIX-only module outside a
try/except ImportError. Written 2026-11-14 after Part 205's qwen-chat launch died on `import fcntl` (Part 209 had on
2026-11-12). Break: make posix_only_imports return [] (every red case passes), or drop the `guarded` check (the guarded
cases go red)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_arc as la  # noqa: E402


def _w(tmp, name, text):
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p


def test_unguarded_imports_are_found_at_top_level_in_a_function_and_as_from(tmp_path):
    top = _w(tmp_path, "top.py", "import os\nimport fcntl\n")
    fn = _w(tmp_path, "fn.py", "def lock(f):\n    import fcntl\n    fcntl.flock(f, 2)\n")
    frm = _w(tmp_path, "frm.py", "from fcntl import flock\n")
    res = _w(tmp_path, "res.py", "import resource, json\n")
    found = la.posix_only_imports([top, fn, frm, res])
    assert [(Path(p).name, line, m) for p, line, m in found] == [
        ("fn.py", 2, "fcntl"), ("frm.py", 1, "fcntl"), ("res.py", 1, "resource"), ("top.py", 2, "fcntl")]


def test_guarded_imports_and_non_python_files_pass(tmp_path):
    ok1 = _w(tmp_path, "ok1.py", "try:\n    import fcntl\nexcept ImportError:\n    fcntl = None\n    import msvcrt\n")
    ok2 = _w(tmp_path, "ok2.py", "def f():\n    try:\n        import fcntl\n    except (OSError, ModuleNotFoundError):\n"
                                 "        return None\n")
    bad_handler = _w(tmp_path, "bad.py", "try:\n    import fcntl\nexcept KeyError:\n    pass\n")
    other = _w(tmp_path, "notes.txt", "import fcntl\n")
    found = la.posix_only_imports([ok1, ok2, bad_handler, other])
    assert [(Path(p).name, m) for p, _, m in found] == [("bad.py", "fcntl")]


def test_part_205s_pre_fix_common205_is_refused_and_the_fixed_one_passes(tmp_path):
    old = subprocess.run(["git", "-C", str(HERE.parent), "show", "a36063a6:part205/common205.py"],
                         capture_output=True, text=True, check=True).stdout
    p = _w(tmp_path, "common205_old.py", old)
    assert [m for _, _, m in la.posix_only_imports([p])] == ["fcntl", "fcntl"]
    assert la.posix_only_imports([HERE.parent / "part205" / "common205.py", HERE.parent / "part205" / "run205.py"]) == []
