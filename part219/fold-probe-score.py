"""Probe score219's rule interactions on synthetic label worlds.

Uses the test module's own world builders (real code, real scorer) so the probe exercises exactly
the synthetic shape the tests use.
"""
import json
import sys
import tempfile
from pathlib import Path

WORKTREE = str(Path(__file__).resolve().parents[1])  # the worktree this copy lives in
sys.path.insert(0, WORKTREE)

import part219.test_part219 as t  # noqa: E402
import part219.score219 as score219  # noqa: E402


def card(tmp, show, **kwargs):
    return t.build(Path(tmp), show, **kwargs)


def report(name, card):
    verdicts = {key: value.get("verdict") for key, value in card["predictions"].items()}
    statuses = {tag: run["status"] for tag, run in card["runs"].items() if run["status"]}
    print(json.dumps({"case": name, "gate_passed": card["gate"]["passed"],
                      "gate_failures": card["gate"]["failures"],
                      "controls": {k: v.get("held") for k, v in card["controls"].items()},
                      "predictions": verdicts, "headlines": card["headlines"],
                      "nonempty_status": statuses}, indent=1))


with tempfile.TemporaryDirectory() as tmp:
    report("baseline", card(Path(tmp) / "baseline", t.baseline_show()))

with tempfile.TemporaryDirectory() as tmp:
    report("R5 receipt unreadable (manifest_readable R5=False)",
           card(Path(tmp) / "r5", t.baseline_show(), manifest_readable={"R5": False}))

with tempfile.TemporaryDirectory() as tmp:
    report("R9 receipt unreadable",
           card(Path(tmp) / "r9", t.baseline_show(), manifest_readable={"R9": False}))

# R0's own two-pass gate fails: pass 2 withholds three slot pushed labels on R0.
with tempfile.TemporaryDirectory() as tmp:
    show = t.baseline_show()
    pass2 = {"R0": set(show["R0"]) - set(t.SLOT[:3])}
    report("R0 two-pass gate fails (39/42 on R0)", card(Path(tmp) / "r0gate", show, pass2=pass2))

# R10 unreadable: the design says a broken R10 makes 219.5 harder, not unreadable.
with tempfile.TemporaryDirectory() as tmp:
    report("R10 receipt unreadable", card(Path(tmp) / "r10", t.baseline_show(),
                                         manifest_readable={"R10": False}))

# R8 receipt unreadable: shows the R8 path does consult status.
with tempfile.TemporaryDirectory() as tmp:
    report("R8 receipt unreadable", card(Path(tmp) / "r8", t.baseline_show(),
                                         manifest_readable={"R8": False}))
