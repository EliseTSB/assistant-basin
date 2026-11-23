"""Deliberate-break harness for the OLMo 3 lens port.

Copies ``neuron_lens/`` and ``tests/test_olmo3.py`` to a throwaway directory,
applies ONE specific break at a time, and runs ``tests/test_olmo3.py``.
Every break must turn red for its own exact assertion while the unmodified
control stays green.

Four breaks covering OLMo 3's post-norm and hybrid-attention specifics:
1. ``jvp-post-ff-norm-dropped`` (``olmo2.py`` jvp drops ``a_pf`` -> S-lens reds)
2. ``vjp-post-ff-norm-dropped`` (``olmo2.py`` vjp drops ``a_pf`` -> R-lens reds)
3. ``sliding-window-mask-dropped`` (``olmo3.py`` ignores ``sliding_window`` -> brute-force reds)
4. ``yarn-layer-type-rope-dropped`` (``olmo3.py`` reuses sliding RoPE for full layers -> brute-force reds)
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BREAKS = {
    "jvp-post-ff-norm-dropped": (
        "neuron_lens/olmo2.py",
        "    return d_h1 + m.a_pf * d_mlp, d_h1",
        "    return d_h1 + d_mlp, d_h1",
        "('s',",
    ),
    "vjp-post-ff-norm-dropped": (
        "neuron_lens/olmo2.py",
        "    gp = (m.a_pf * G).to(dt) @ m.wd  # [b, N, d_ff]",
        "    gp = G.to(dt) @ m.wd  # [b, N, d_ff]",
        "('r-out',",
    ),
    "sliding-window-mask-dropped": (
        "neuron_lens/olmo3.py",
        '        sliding_mask = sliding_mask.masked_fill(too_old, float("-inf"))',
        "        sliding_mask = full_mask",
        "AssertionError",
    ),
    "yarn-layer-type-rope-dropped": (
        "neuron_lens/olmo3.py",
        "        cos_lt, sin_lt = model.rotary_emb(h[None], pos_ids, lt)",
        '        cos_lt, sin_lt = model.rotary_emb(h[None], pos_ids, "sliding_attention")',
        "AssertionError",
    ),
}


def _copy(dst: Path) -> None:
    shutil.copytree(ROOT / "neuron_lens", dst / "neuron_lens",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "tests").mkdir()
    shutil.copy2(ROOT / "tests" / "test_olmo3.py", dst / "tests" / "test_olmo3.py")


def _run(dst: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(dst)
    env["OMP_NUM_THREADS"] = env.get("SABOTAGE_THREADS", "12")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_olmo3.py", "-q", "-x",
         "-p", "no:cacheprovider"],
        cwd=dst, env=env, text=True, capture_output=True,
    )


def _apply_break(dst: Path, relpath: str, old: str, new: str) -> None:
    path = dst / relpath
    text = path.read_text()
    if text.count(old) != 1:
        raise AssertionError(f"break target in {relpath} occurs {text.count(old)} times, expected 1")
    path.write_text(text.replace(old, new))


def check() -> list[str]:
    lines = []
    with tempfile.TemporaryDirectory(prefix="olmo3-lens-sabotage-") as tmp:
        control = Path(tmp) / "control"
        control.mkdir()
        _copy(control)
        got = _run(control)
        assert got.returncode == 0, f"CONTROL FAILED (not green):\n{got.stdout[-3000:]}"
        assert "8 passed" in got.stdout, f"CONTROL ran unexpected test count:\n{got.stdout[-1000:]}"
        lines.append("control: unmodified copy green (8 passed)")

        for name, (relpath, old, new, expect) in BREAKS.items():
            work = Path(tmp) / name
            work.mkdir()
            _copy(work)
            _apply_break(work, relpath, old, new)
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


def test_olmo3_sabotage_reds_and_control_greens():
    lines = check()
    assert len(lines) == 1 + len(BREAKS)


if __name__ == "__main__":
    for line in check():
        print(line)
    print("OLMO 3 SABOTAGE HARNESS: OK")
