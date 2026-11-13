"""Size-section parsing in part_check.py prelaunch (2026-11-13, Claude Opus 5.5).

Papercut 2026-11-12 0003: a design that bolded its Size labels ("- **Total:** ...", "- **Pilot:** ...") was told it had
no Total line and no Pilot line. prelaunch now strips Markdown emphasis from the Size section before checking it.
Sabotage that turns this red: in part_check.py, read the Size checks from sec.get("Size", "") again instead of
size_text -> test_bold_size_labels_are_read fails.
"""
import sys
from pathlib import Path

_p = str(Path(__file__).resolve().parent)
if _p not in sys.path:
    sys.path.insert(0, _p)
import part_check as pc  # noqa: E402

FILLED = {s: "Filled text." for s in pc.SECTIONS}
FILLED["Controls"] = "Random controls. Gate: positive control."
FILLED["Predictions"] = "Z1.1 something."


def _design(tmp_path, size):
    body = dict(FILLED, Size=size)
    text = "# Part Z1\n\nProcedure: v1\nRegistered 2026-11-13 11:20 CT\n\n" + "\n".join(
        f"## {k}\n\n{v}\n" for k, v in body.items())
    (tmp_path / "PARTZ1-DESIGN.md").write_text(text, encoding="utf-8")
    (tmp_path / "PREDICTIONS.md").write_text("## Part Z1\n\nZ1.1 something.\n", encoding="utf-8")
    return pc.prelaunch(tmp_path, "z1", fetch=False)


def _size_fails(rep):
    return [m for m in rep.failed if m.startswith("Size")]


def test_bold_size_labels_are_read(tmp_path):
    rep = _design(tmp_path, "- **Total:** 120 generations, about 3 minutes on the CPU.\n"
                            "- **Smallest effect worth knowing:** 10 points.\n"
                            "- **Pilot:** not needed: under 1,000 generations and one GPU-hour.\n")
    assert _size_fails(rep) == []


def test_plain_size_labels_still_read(tmp_path):
    rep = _design(tmp_path, "- Total: 120 generations, about 3 minutes on the CPU.\n"
                            "- Smallest effect worth knowing: 10 points.\n"
                            "- Pilot: not needed: under 1,000 generations and one GPU-hour.\n")
    assert _size_fails(rep) == []


def test_missing_pilot_still_fails(tmp_path):
    rep = _design(tmp_path, "- **Total:** 120 generations, about 3 minutes on the CPU.\n"
                            "- **Smallest effect worth knowing:** 10 points.\n")
    assert "Size: no 'Pilot:' line" in _size_fails(rep)
