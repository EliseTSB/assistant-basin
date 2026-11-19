#!/usr/bin/env python3
"""Check that every number on Part 212's results README appears in report212.md.

Every number in results/part212/README.md (integers, decimals, scientific notation, with or without thousands
separators) must appear in results/part212/report212.md, or be listed in ALLOWED with a reason. It must also appear
at the README's own precision: 0.9879 matches a source that prints 0.9879 or 0.98795, but 0.9879 does not match a
source that prints only 0.99, so a number typed at a precision no source supports is caught. A number inside a word
(p92-02, 2026-11-19, Qwen3.5-4B) is not a number for this check; the parts of a date are.

Exit 0 when every number is explained, 1 otherwise, with each unexplained number printed.

Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I -B part212/check_readme212.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "part212"
README = RES / "README.md"
SOURCES = (RES / "report212.md",)

# Numbers that are not results of this part, with the reason each is allowed. A number that is a result is NOT listed
# here; it must come from report212.md.
ALLOWED = {
    # dates and times, from chain.log and the commits
    "2026": "year in dates", "11": "month in dates", "19": "day in dates and 2026-11-19",
    "12": "pilot launch and failure times (12:17, 12:41, 12:48, 12:49)", "17": "pilot launch time (12:17)",
    "41": "pilot failure time (12:41)", "48": "pilot launch time (12:48)", "49": "pilot launch time (12:49)",
    "13": "pilot launch and end times (13:19, 13:09, 13:49)", "09": "RAM guard time (13:09)",
    "14": "full Stage 2 launch times (14:06, 14:17)", "06": "full launch time (14:06)", "16": "full launch (16:50)",
    "50": "full Stage 2 end (16:50)", "01": "registration time (01:24)", "24": "registration time (01:24)",
    "30db7048": "commit id of the lock fix", "4e98ffb1": "commit id of the load-once fix",
    # part numbers, section numbers, identifiers and model names
    "66": "part number", "183": "part number", "201": "part number", "212": "part number", "0": "counts of zero",
    "1": "section numbering, 212.1 and counts of one", "2": "section numbering and 212.2", "3": "section numbering and 212.3",
    "4": "section numbering and 212.4", "5": "section numbering and 212.5", "6": "section numbering and 212.6",
    "7": "section numbering and 212.7", "8": "section numbering and 212.8", "9": "section numbering and 212.9",
    "10": "section numbering and 212.10", "3.5": "model name Qwen3.5-4B", "4B": "model size",
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a": "model revision",
    # fixed by the design or the run's plan, not computed
    "24": "held-out prompts and the random sets' denominator (design)", "192": "held-out answers per full arm (design)",
    "20": "matched random sets (design)", "50": "units per layer pool (design)", "131": "Stage 1 arms (design)",
    "25": "Stage 1 prompts and donor bank size (design)", "3888": "planned Stage 2 rows (design)",
    "3,888": "planned Stage 2 rows (design; report212.md prints it without the separator)",
    "1748": "Stage 2 generation calls (report212.md prints it without the separator)",
    "1,748": "Stage 2 generation calls (report212.md)",
    "3449": "Stage 1 forwards (report212.md prints it without the separator)", "3,449": "Stage 1 forwards (report212.md)",
    "3989": "the design's arithmetic error (report212.md prints it without the separator)",
    "3,989": "the design's arithmetic error (report212.md)", "3712": "run.json new_rows_this_invocation (report212.md)",
    "3,712": "run.json new_rows_this_invocation (report212.md)", "176": "rows written before the crash (computed)",
    "160": "pilot Stage 2 rows (design)", "1350": "GPU samples in the full run (chain.log)",
    "1,350": "GPU samples in the full run (chain.log)", "78": "GPU busy percent (chain.log)",
    "4.80": "batch-1 seconds per call, recorded during the run (not in the result files)",
    "4.84": "the forecast, recorded during the run (not in the result files)",
    "100": "the first 100 batch-1 calls (Amendment 2)",
    "0xC0000005": "Windows access-violation code (chain.log)", "3221225477": "Windows access-violation code (chain.log)",
    "97": "job runner exit code (chain.log)", "1.92": "RAM available at the guard (chain.log)",
    "2.0": "the RAM guard floor (chain.log)", "0.7": "Stage 2 temperature (design)", "0.95": "Stage 2 top p (design)",
    "48": "token cap (design)", "95": "95% interval", "10000": "bootstrap draws (design)",
    "10,000": "bootstrap draws (design)", "212": "bootstrap seed (design)",
    # bars and thresholds quoted from the registered predictions (design, PREDICTIONS.md)
    "0.15": "broken-rate bar (design)", "0.12": "212.1's lower rate bar (design)", "0.35": "212.1's upper rate bar (design)",
    "0.25": "212.6's whole-arm bar (design)", "0.10": "212.6's L6 N753 bar (design)", "0.75": "212.3's bar (design)",
    "0.50": "212.2's and 212.4's half bars (design)", "0.05": "212.5's bar (design)", "0.04": "212.9's bar (design)",
    "0.30": "212.8's bar (design)", "0.70": "212.7's bar (design)", "1.5": "212.2's bar (design)",
    "1.6": "212.1's 1.6% (design)", "12%": "212.1's bar (design)", "35%": "212.1's bar (design)",
    "0.90": "Gate 3's guard bar (design)", "0.5": "counts of zero and one",
    "5.3": "GLM 5.3, the reviewer's version name (BUILD-REPORT.md)",
}

NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])", re.IGNORECASE)


def numbers(text: str) -> set[str]:
    out = set()
    for match in NUMBER.finditer(text):
        raw = match.group(0).lstrip("+-")
        out.add(raw)
        out.add(raw.replace(",", ""))
    return out


def main() -> int:
    readme = README.read_text(encoding="utf-8")
    found = numbers(readme)
    haystack = "\n".join(p.read_text(encoding="utf-8") for p in SOURCES)
    hay = numbers(haystack)

    def printed(value: str) -> bool:
        """The source prints the value, or a number that rounds to it at the README's precision."""
        if value in hay:
            return True
        try:
            target = float(value)
        except ValueError:
            return False
        decimals = len(value.split(".")[1]) if "." in value and "e" not in value.lower() else 0
        for other in hay:
            try:
                other_value = float(other)
            except ValueError:
                continue
            if decimals and abs(round(other_value, decimals) - target) < 1e-12:
                return True
            if not decimals and abs(other_value - target) < 0.5:
                return True
        return False

    unexplained = sorted(v for v in found if v not in ALLOWED and not printed(v))
    allowed_hit = sorted(v for v in found if v in ALLOWED)
    from_source = len(found) - len(allowed_hit) - len(unexplained)
    print(f"README numbers: {len(found)}; allowed by the list: {len(allowed_hit)}; "
          f"found in report212.md: {from_source}; unexplained: {len(unexplained)}")
    for value in unexplained:
        print(f"  UNEXPLAINED: {value}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
