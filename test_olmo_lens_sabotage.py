"""Deliberate-break harness for the OLMo 2 lens port (goal item 2).

Copies ``neuron_lens/`` and ``tests/test_olmo2.py`` to a throwaway directory,
applies ONE specific break to the copy, and runs the brute-force tests against
it. The break must turn them red **for its own reason** — a failing assertion in
``test_estimates_match_brute_force``, not an import or collection error — and
the unmodified copy must stay green.

Two breaks, one per lens direction:

1. ``jvp`` drops the post-feedforward norm's held-constant scale (``a_pf``), the
   term that exists only because OLMo 2 is post-norm. The S-lens (forward mode)
   must red.
2. ``vjp`` drops the same scale on the pull-back. The R-lens (reverse mode) must
   red.

Run directly (``python test_olmo_lens_sabotage.py``) or under pytest.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BREAKS = {
    # name: (old, new, the unique assertion message the break must hit)
    "jvp-post-ff-norm-dropped": (
        "    return d_h1 + m.a_pf * d_mlp, d_h1",
        "    return d_h1 + d_mlp, d_h1",
        "('s',",  # the S-lens (forward-mode) assertion, not the R-lens one
    ),
    "vjp-post-ff-norm-dropped": (
        "    gp = (m.a_pf * G).to(dt) @ m.wd  # [b, N, d_ff]",
        "    gp = G.to(dt) @ m.wd  # [b, N, d_ff]",
        "('r-out',",  # the R-lens (reverse-mode) assertion
    ),
}


def _copy(dst: Path) -> None:
    shutil.copytree(ROOT / "neuron_lens", dst / "neuron_lens",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "tests").mkdir()
    shutil.copy2(ROOT / "tests" / "test_olmo2.py", dst / "tests" / "test_olmo2.py")


def _run(dst: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(dst)
    env["OMP_NUM_THREADS"] = env.get("SABOTAGE_THREADS", "12")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_olmo2.py", "-q", "-x",
         "-p", "no:cacheprovider"],
        cwd=dst, env=env, text=True, capture_output=True,
    )


def _apply_break(dst: Path, old: str, new: str) -> None:
    path = dst / "neuron_lens" / "olmo2.py"
    text = path.read_text()
    if text.count(old) != 1:
        raise AssertionError(f"break target occurs {text.count(old)} times, expected 1")
    path.write_text(text.replace(old, new))


def check() -> list[str]:
    """Returns the harness's own report lines; raises AssertionError on a failed expectation."""
    lines = []
    with tempfile.TemporaryDirectory(prefix="olmo-lens-sabotage-") as tmp:
        control = Path(tmp) / "control"
        control.mkdir()
        _copy(control)
        got = _run(control)
        assert got.returncode == 0, f"CONTROL FAILED (not green):\n{got.stdout[-3000:]}"
        assert "9 passed" in got.stdout, f"CONTROL ran the wrong number of tests:\n{got.stdout[-1000:]}"
        lines.append("control: unmodified copy green (9 passed)")

        for name, (old, new, expect) in BREAKS.items():
            work = Path(tmp) / name
            work.mkdir()
            _copy(work)
            _apply_break(work, old, new)
            got = _run(work)
            out = got.stdout + got.stderr
            assert got.returncode != 0, f"BREAK {name} did NOT red the tests"
            assert "test_estimates_match_brute_force" in out, (
                f"BREAK {name} red for the wrong reason (test not named):\n{out[-2000:]}")
            assert "AssertionError" in out, (
                f"BREAK {name} red for the wrong reason (no assertion):\n{out[-2000:]}")
            assert "ImportError" not in out and "SyntaxError" not in out and "ERROR collecting" not in out, (
                f"BREAK {name} red for the wrong reason (import/collection):\n{out[-2000:]}")
            assert expect in out, (
                f"BREAK {name} red on the wrong assertion (wanted {expect}):\n{out[-2000:]}")
            failed = [ln for ln in out.splitlines() if ln.startswith("FAILED ")]
            lines.append(f"break {name}: red on {expect} ({failed[0] if failed else 'assertion'})")
    return lines


def test_sabotage_reds_and_control_greens():
    lines = check()
    assert len(lines) == 1 + len(BREAKS)


if __name__ == "__main__":
    for line in check():
        print(line)
    print("SABOTAGE HARNESS: OK")
