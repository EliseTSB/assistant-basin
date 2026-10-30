#!/usr/bin/env python3
"""Count prediction verdicts for front-page Findings 7–11 from PREDICTIONS.md.

Each prediction row is counted once under its Part heading by the explicit verdict in its last table
cell. A failed result that also says another clause held is counted as failed.
Stopped, not tested, not read, not scorable, pending, and other verdicts stay
separate from held and failed.
"""
from collections import Counter
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PREDICTIONS.md"
FINDINGS = {
    7: ("86", "97", "98", "112"),
    8: ("92", "92b", "109", "109b", "109c", "112", "113", "117", "117b"),
    9: ("89", "90b", "91", "92", "94", "94b", "94c", "95", "96", "99", "104", "105", "106", "106b", "107", "108"),
    10: ("119", "119b", "119c"),
    11: ("110", "111", "114", "115", "115b", "116"),
}
SECTION = re.compile(r"^##\s+Part\s+(\d+[a-z]?)\b", re.IGNORECASE)
ROW_ID = re.compile(r"^\d+[a-z]?(?:\.[A-Za-z0-9]+)$", re.IGNORECASE)


def verdict_for(cell: str) -> str:
    value = re.sub(r"\*", "", cell).strip().lower()
    # Stopped/not-read states are distinct statuses, not prediction failures.
    for status in ("stopped", "not tested", "not scorable", "not read", "pending"):
        if re.search(rf"\b{re.escape(status)}\b", value):
            return status
    # Some rows explain that one clause held even though the prediction failed.
    if re.search(r"\bfailed\b", value):
        return "failed"
    if re.search(r"\bheld\b", value):
        return "held"
    return value or "no verdict"


def read_rows():
    rows = []
    current_part = None
    for line_number, line in enumerate(SOURCE.read_text(encoding="utf-8").splitlines(), 1):
        heading = SECTION.match(line)
        if heading:
            current_part = heading.group(1).lower()
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or current_part is None or not ROW_ID.match(cells[0]):
            continue
        rows.append((current_part, verdict_for(cells[-1]), line_number))
    return rows


def show(counter: Counter) -> str:
    order = ("held", "failed", "stopped", "not tested", "not read", "not scorable", "pending", "no verdict")
    keys = [key for key in order if counter[key]] + sorted(key for key in counter if key not in order)
    return ", ".join(f"{counter[key]} {key}" for key in keys) if keys else "0 predictions"


def main():
    rows = read_rows()
    print("Source: PREDICTIONS.md. One row counted once under its Part heading by its last-cell verdict.")
    for finding, parts in FINDINGS.items():
        selected = [(part, status, line) for part, status, line in rows if part in parts]
        if not selected:
            raise SystemExit(f"Finding {finding}: no prediction rows matched")
        total = Counter(status for _, status, _ in selected)
        print(f"Finding {finding} (Parts {', '.join(parts)}): {show(total)}")
        for part in parts:
            counts = Counter(status for row_part, status, _ in selected if row_part == part)
            print(f"  Part {part}: {show(counts)}")


if __name__ == "__main__":
    main()
