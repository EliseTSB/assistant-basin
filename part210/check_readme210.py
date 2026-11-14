#!/usr/bin/env python3
"""Check that every number on Part 210's results README appears in report210.md or score.md.

Extracts every number from results/part210/README.md (integers, decimals, scientific notation, and the same with
thousands separators), and reports each one that appears in neither results/part210/report210.md nor
results/part210/score.md. A number is "appearing" when the same value is printed in that file at any precision the
README uses: 0.882 matches 0.882 and 0.8820; 1.84 matches 1.83898 only through the rounded form the README prints
(1.84), so the README is expected to print a number at a precision the source prints it at, or a rounded form of it.

ALLOWED lists the numbers that are not results: dates, part numbers, model revision digits, counts of rows and
questions fixed by the design, the bar values quoted from the registered rules, and the design's own constants. Each
entry says why. A number not in the sources and not in ALLOWED is printed and the script exits 1.

Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I -B part210/check_readme210.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "part210"
README = RES / "README.md"
SOURCES = (RES / "report210.md", RES / "score.md")

# Numbers that are not results, with the reason each is allowed.
ALLOWED = {
    # dates and amendment times
    "2026": "year in dates", "11": "month in dates", "14": "day in dates", "05": "Amendment 1 time", "07": "Amendment 2 time", "04": "run time (04:07)", "33": "run time (01:33)",
    "42": "Amendment 2 time (07:42)", "09": "Amendment 3 time (09:50)", "50": "Amendment 3 time (09:50)",
    # part numbers, identifiers and section numbers
    "164": "part number", "165": "part number", "201": "part number", "202": "part number", "209": "part number",
    "210": "part number", "86": "part number (triage)", "3.5": "model name Qwen3.5", "4": "model name Qwen3.5-4B",
    "1": "section numbering", "2": "section numbering", "3": "section numbering", "5": "section numbering",
    "6": "section numbering", "7": "section numbering", "8": "section numbering", "9": "section numbering",
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a": "model revision",
    # fixed by the registered design: rows, cells, cap, bars, tolerances, scenario names
    "6920": "planned rows", "6912": "battery rows", "864": "directions", "768": "answers per arm", "96": "slots per arm",
    "80": "token cap", "0.8": "registered write size and 165.5's bar", "1e-6": "Gate R2's tolerance",
    "1e-2": "Gate 3's tolerance", "300": "210.4's bar", "39": "300 of 768 as a percent", "40": "210.6's lower bar",
    "110": "210.6's upper bar", "200": "replicate rows", "1000": "shape floor redraws", "22": "replicate rows per arm",
    "23": "replicate rows per arm", "0.67": "165.4's lower bar", "1.5": "165.4's and 210.3's bar",
    "0.95": "shape floor display threshold", "95": "95% interval", "0.5": "shared-change-0.5 scenario name",
    "0": "seed 0, or a count of zero", "64": "block size (calls of 64 rows)",
    # Amendment 3's own account of the first scoring, and the run's wall time: not results of the rerun, and not in
    # report210.md or score.md, which carry the corrected scoring only. Each is in PART210-DESIGN.md's Amendment 3 or
    # the run's log, named beside it in the README.
    "154": "the first scoring's printed count (Amendment 3; not in score.md)",
    "162": "keys the first pairing matched (Amendment 3; not in score.md)",
    "6911": "changed rows (design, Controls; not in score.md)", "112": "receipts in Amendment 1 (design; not in score.md)",
    "47": "minutes, the scorer rerun (score-r3-after-205q.log)", "1.57": "GB peak (score210-stderr.txt)",
}

NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])", re.IGNORECASE)


def numbers(text):
    out = set()
    for match in NUMBER.finditer(text):
        raw = match.group(0).lstrip("+-")
        out.add(raw)
        out.add(raw.replace(",", ""))
    return out


def main():
    readme = README.read_text(encoding="utf-8")
    # Thousands separators are stripped before extraction so 6,920 reads as 6920.
    found = numbers(readme.replace(",", ""))
    haystack = "\n".join(p.read_text(encoding="utf-8") for p in SOURCES)
    hay_numbers = numbers(haystack.replace(",", ""))

    def printed(value):
        """The value appears if the source prints it, or prints a number that rounds to it at the README's precision."""
        if value in hay_numbers:
            return True
        try:
            target = float(value)
        except ValueError:
            return False
        decimals = len(value.split(".")[1]) if "." in value and "e" not in value.lower() else 0
        for other in hay_numbers:
            try:
                if decimals and abs(round(float(other), decimals) - target) < 1e-12:
                    return True
                if not decimals and abs(float(other) - target) < 0.5:
                    return True
            except ValueError:
                continue
        return False

    unexplained = sorted(v for v in found if v not in ALLOWED and not printed(v))
    allowed_hit = sorted(v for v in found if v in ALLOWED)
    print(f"README numbers: {len(found)}; allowed by the list: {len(allowed_hit)}; "
          f"found in report210.md or score.md: {len(found) - len(allowed_hit) - len(unexplained)}; "
          f"unexplained: {len(unexplained)}")
    for value in unexplained:
        print(f"  UNEXPLAINED: {value}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
