"""Create single-edit mutants of the review copy (never the worktree). Each replacement must match exactly once."""
import shutil, sys
from pathlib import Path
BASE = Path("/home/elisepc/green/Journal/scratch/review-p65-build-haiku-r5-2026-10-24/mut")
EDITS = {
    "M1_no_break_on_release": [("part65/judge65.py",
        'if outcome in ("released", "ambiguous"):\n                    break',
        'if outcome in ("ambiguous",):\n                    break')],
    "M2_terminal_attempt_not_refused": [("part65/judge65.py",
        '    raise JudgeError(\n        f"Part 65 plan chunk already has a {status} paidaction attempt; refusing to resubmit"\n    )',
        '    return None')],
    "M3_posting_marker_not_durable": [("part65/judge65.py",
        '                state["status"] = "posting"\n                _save_chunk(path, state)',
        '                state["status"] = "posting"')],
    "M4_no_ledger_judge_lock": [("part65/judge65.py",
        '        with _lock_file(_ledger_judge_lock_path(ledger.path)):',
        '        if True:')],
}
for name, edits in EDITS.items():
    dest = BASE.parent / ("mutants/" + name)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "part65").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE / "part65", dest / "part65", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(BASE / "paidaction", dest / "paidaction", ignore=shutil.ignore_patterns("__pycache__"))
    for rel, old, new in edits:
        target = dest / rel
        text = target.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            sys.exit(f"{name}: expected 1 match in {rel}, found {count}")
        target.write_text(text.replace(old, new), encoding="utf-8")
    print("made", name)
