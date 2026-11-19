#!/usr/bin/env python3
"""Check that every number on Part 213's results README appears in report213.md or score-table-audit.txt.

Every number in results/part213/README.md (integers, decimals, scientific notation, with or without thousands
separators) must appear in results/part213/report213.md or results/part213/score-table-audit.txt, or be listed in
ALLOWED with a reason. It must also appear at the README's own precision: 0.9232 matches a source that prints 0.9232 or
0.92324, but 0.9232 does not match a source that prints only 0.92, so a number typed at a precision no source supports
is caught. A number inside a word (p92-02, 2026-11-19, OLMo-2) is not a number for this check; the parts of a date are.

Exit 0 when every number is explained, 1 otherwise, with each unexplained number printed.

Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I -B part213/check_readme213.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "part213"
README = RES / "README.md"
SOURCES = (RES / "report213.md", RES / "score-table-audit.txt")

# Numbers that are not results of this part, with the reason each is allowed. A number that is a result is NOT listed
# here; it must come from report213.md or score-table-audit.txt.
ALLOWED = {
    "198": "sum of the five registered audit-table denominators (42 + 42 + 42 + 36 + 36), design l.258-265",
    # dates and times, from the run log and the commits
    "2026": "year in dates", "11": "month in dates", "19": "day in dates", "04": "Amendment 1 commit time (04:44)",
    "44": "Amendment 1 commit time (04:44)", "06": "full run start (06:24)", "24": "full run start (06:24)",
    "07": "FR failure and rerun times (07:12, 07:16)", "12": "FR failure time (07:12)", "54": "FR failure time (07:12:54)",
    "16": "rerun time (07:16)", "08": "Stage 2 end (08:24)", "09": "Stage 2 end (08:24)",
    # part numbers, section numbers, identifiers and model names
    "92": "part number", "99": "part number", "106": "part number", "180": "part number", "213": "part number",
    "0": "213.10 clause (a) and (b), and counts of zero", "1": "section numbering and 213.1", "2": "section numbering",
    "3": "section numbering and 213.3", "4": "section numbering and 213.4", "5": "section numbering and 213.5",
    "6": "section numbering and 213.6", "7": "section numbering and 213.7", "8": "section numbering and 213.8",
    "9": "section numbering and 213.9", "10": "section numbering and 213.10", "11": "section numbering and 213.11",
    "2.1": "OLMo 2 1B", "1B": "model size", "4.1": "DeepSeek V4.1 Flash",
    # fixed by the design or the run's plan, not computed
    "39": "units in the catalog (design)", "21": "slot units (design)", "18": "concept units (design)",
    "17": "Stage 1 prompts (design) and 213.8's bar", "472": "planned Stage 2 generations (design)",
    "364": "Stage 2 rows run (the seal leaves 108 sealed; design)", "108": "sealed concept rows (design)",
    "234": "exported Stage 2 answers (design)", "40": "second reader's sample size (design)",
    "42": "audited rows per slot arm-group (design)", "36": "audited rows per concept arm-group (design)",
    "10": "known examples per reader and 213.6's bar (design)", "60": "Gate 3's kappa bar 0.60 (design)",
    "0.05": "finite-difference step (design)", "0.8": "push dose (design)",
    "35": "off-diagonal set size, lower end (design)", "38": "off-diagonal set size, upper end (design)",
    "10000": "bootstrap draws (design)", "10,000": "bootstrap draws (design)", "213": "bootstrap seed (design)",
    "0.50": "213.3, 213.4 and 213.10(b)'s bar (design)", "0.75": "213.2, 213.9 and 213.11's bar (design)",
    "0.40": "213.8's context-contrast bar (design)", "0.30": "213.7's bar (design)", "0.25": "213.5 and 213.9's bar (design)",
    "0.65": "the transfer the design is powered for (design)", "0.60": "Gate 3's kappa bar (design)",
    "95": "95% interval", "25": "213.9's band edge",
    "936": "off-diagonal passes Amendment 1 restored (design; also in report213.md)",
    "6474": "Instruct Stage 1 records (report213.md prints it without the separator)",
    "6,474": "Instruct Stage 1 records (report213.md)",
    # commit ids the number pattern reads as numbers; not results, and not printed by any result file.
    "1e980869": "commit id of the independent 213.8 recomputation",
    "553904e1": "commit id of the audit label-key fix",
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
                if decimals and abs(round(float(other), decimals) - target) < 1e-12:
                    return True
                if not decimals and abs(float(other) - target) < 0.5:
                    return True
            except ValueError:
                continue
        return False

    unexplained = sorted(v for v in found if v not in ALLOWED and not printed(v))
    allowed_hit = sorted(v for v in found if v in ALLOWED)
    from_source = len(found) - len(allowed_hit) - len(unexplained)
    print(f"README numbers: {len(found)}; allowed by the list: {len(allowed_hit)}; "
          f"found in report213.md or score-table-audit.txt: {from_source}; unexplained: {len(unexplained)}")
    for value in unexplained:
        print(f"  UNEXPLAINED: {value}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
