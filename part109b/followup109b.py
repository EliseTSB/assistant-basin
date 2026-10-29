#!/usr/bin/env python3
"""Part 109b (PART109B-DESIGN.md): Part 109's runner on all 18 of Part 92's concept neurons. Writes
results/part109b/answers.jsonl and label items for part92/label92.py. Written by Claude (Opus 5.5), 2026-10-29.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part109"))
import followup109 as f  # noqa: E402

UNITS = ["p92-00", "p92-01", "p92-08", "p92-11", "p92-15", "p92-28", "p92-29", "p92-30", "p92-38", "p92-39",
         "p92-40", "p92-41", "p92-43", "p92-45", "p92-48", "p92-49", "p92-51", "p92-52"]

if __name__ == "__main__":
    f.main(UNITS, ROOT / "results/part109b")
