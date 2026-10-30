#!/usr/bin/env python3
"""Part 117b's second reader: Part 117's (part117/second117.py: Opus 5.5 with Part 92's prompt on a seeded 10% sample
of the reader's items, then the agreement on shows_persona), pointed at results/part117b with its own sample seed.
Written by Claude (Opus 5.5), 2026-10-29.
Usage: second117b.py --label | --agree"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part117"))
import second117 as s  # noqa: E402

s.RES = HERE.parent / "results/part117b"
s.SEED = 1172

if __name__ == "__main__":
    s.label() if sys.argv[1:] == ["--label"] else s.agree()
