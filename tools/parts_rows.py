#!/usr/bin/env python3
"""Build PARTS.md rows from the records.

PARTS.md line 9 states the rule this script implements: from Part 128 on, rows
come from PREDICTIONS.md (verdict counts by the last table cell), the design
files (the question), the part folders (code files) and results/ (status `done`
when a results README exists).

Run with python3 -P, e.g.

    python3 -P tools/parts_rows.py 179 221          # print rows 179..221
    python3 -P tools/parts_rows.py --check 150 165 178
    python3 -P tools/parts_rows.py --append 179 221

`--check` compares the generated row with the row already in PARTS.md and
prints the fields that differ.  `--append` adds missing rows at the end of the
table; it never rewrites or reorders existing rows.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "PARTS.md"
PREDICTIONS = ROOT / "PREDICTIONS.md"

SECTION_RE = re.compile(r"^## Part ([0-9]+[A-Za-z0-9]*)\b([^\n]*)", re.M)
DATE_RE = re.compile(r"registered\s+(\d{4}-\d{2}-\d{2})")
ID_RE = re.compile(r"^([0-9]+)([A-Za-z]*)$")

# Extra verdict buckets, in the order PARTS.md writes them.  "not read" before
# "inconclusive" before "not testable" matches the rows already in the table.
EXTRA_ORDER = (
    "mixed",
    "untestable",
    "not_read",
    "withheld",
    "inconclusive",
    "not_testable",
    "pending",
    "withdrawn",
    "stopped",
    "other",
)
EXTRA_LABEL = {"not_read": "not read", "not_testable": "not testable"}


def id_key(pid: str) -> tuple[int, str]:
    m = ID_RE.match(pid)
    if not m:
        return (10**9, pid)
    return (int(m.group(1)), m.group(2).lower())


def split_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c) for c in cells)


def classify(cell: str) -> str | None:
    """Map a verdict cell to a bucket; None means the row is not a prediction."""
    text = cell.strip()
    if not text:
        return "open"
    low = text.lower()
    if "not a prediction" in low:
        return None
    if re.search(r"\bwithdrawn\b", low):
        return "withdrawn"
    if "not read" in low:
        return "not_read"
    if "withheld" in low:
        return "withheld"
    if "untestable" in low:
        return "untestable"
    if "not testable" in low:
        return "not_testable"
    if "inconclusive" in low:
        return "inconclusive"
    if "pending" in low:
        return "pending"
    if "mixed" in low:
        return "mixed"
    if "stopped" in low:
        return "stopped"
    if re.search(r"\bheld\b", low):
        return "held"
    if re.search(r"\bfailed\b", low):
        return "failed"
    if low.startswith("open"):
        return "open"
    return "other"


def parse_predictions() -> dict[str, dict]:
    """Return {part id: {date, question, counts, rows}} from PREDICTIONS.md."""
    text = PREDICTIONS.read_text(encoding="utf-8")
    heads = list(SECTION_RE.finditer(text))
    out: dict[str, dict] = {}
    for i, m in enumerate(heads):
        pid = m.group(1).lower()
        heading = m.group(2)
        body = text[m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        date = DATE_RE.search(heading)
        q = re.search(r"\):\s*(.+?)\s*$", heading)
        counts: dict[str, int] = {}
        rows = 0
        # Tables in this section, keyed by their header row.
        block: list[str] = []

        def flush(block: list[str]) -> None:
            nonlocal rows
            if len(block) < 2:
                return
            header = split_cells(block[0])
            names = [c.lower() for c in header]
            verdict_col = None
            if "verdict" in names:
                verdict_col = names.index("verdict")
            elif "result" in names:
                verdict_col = names.index("result")
            if verdict_col is None:
                # No verdict column: every prediction row is open.
                for line in block[1:]:
                    cells = split_cells(line)
                    if not cells or is_separator(cells):
                        continue
                    if not re.match(re.escape(pid) + r"(?:$|[.\w])", cells[0]):
                        continue
                    if re.search(r"\.A\d", cells[0]):
                        continue  # amendment records, not predictions
                    rows += 1
                    counts["open"] = counts.get("open", 0) + 1
                return
            for line in block[1:]:
                cells = split_cells(line)
                if not cells or is_separator(cells):
                    continue
                if not re.match(re.escape(pid) + r"(?:$|[.\w])", cells[0]):
                    continue
                if re.search(r"\.A\d", cells[0]):
                    continue  # amendment records, not predictions
                # PARTS.md's rule is the last table cell.  Some rows carry one
                # extra empty cell before the verdict (Parts 183 and 187).
                verdict = cells[-1] if cells else ""
                bucket = classify(verdict)
                if bucket is None:
                    continue
                # A few rows bundle several ids in one row (198.9-.11); count
                # each prediction once.
                n = len(
                    [
                        x
                        for x in re.split(r",\s*", cells[0])
                        if re.match(re.escape(pid) + r"[.\w]", x)
                    ]
                )
                rows += max(1, n)
                counts[bucket] = counts.get(bucket, 0) + max(1, n)

        for line in body.splitlines():
            if line.startswith("|"):
                block.append(line)
            else:
                flush(block)
                block = []
        flush(block)
        out[pid] = {
            "date": date.group(1) if date else "unknown",
            "question": q.group(1).strip() if q else None,
            "counts": counts if rows else None,
        }
    return out


def design_files(pid: str) -> list[Path]:
    """Design files for a part, in the order the table should name them."""
    files: list[Path] = []
    exact = ROOT / f"PART{pid.upper()}-DESIGN.md"
    if exact.is_file():
        files.append(exact)
    folder = ROOT / f"part{pid}" / "DESIGN.md"
    if folder.is_file():
        files.append(folder)
    # Versioned redesigns (Part 180 v5) are named beside the original.  Other
    # companion files such as PART165-READING-DESIGN.md are separate designs
    # and are not named in the existing row, so they are left out.
    for p in sorted(ROOT.glob(f"PART{pid.upper()}-V*DESIGN.md")):
        if p not in files:
            files.append(p)
    return files


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def question_for(pid: str, section: dict | None) -> str:
    for design in design_files(pid):
        try:
            first = design.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            continue
        m = re.match(
            rf"#\s*Part\s+{re.escape(pid)}\b[^:]*:\s*(.+)", first, re.IGNORECASE
        )
        if not m:
            continue
        q = m.group(1).strip()
        # Several design titles append a clause after the question; PARTS.md
        # keeps the question itself.
        end = q.find("?")
        if end != -1:
            q = q[: end + 1]
        return q
    if section and section.get("question"):
        return section["question"]
    return "unknown"


def code_for(pid: str) -> str:
    folder = ROOT / f"part{pid}"
    if folder.is_dir():
        names = sorted(
            p.name
            for p in folder.glob("*.py")
            if not p.name.startswith("test_")
            and p.name not in {"__init__.py", "conftest.py"}
        )
        if names:
            return "; ".join(f"`part{pid}/{n}`" for n in names)
    # Part 198 has no part folder: its scripts live in its results folder, as
    # Parts 119c, 122, 123, 124 and 126's figure scripts do.
    rfolder = ROOT / "results" / f"part{pid}"
    if rfolder.is_dir():
        names = sorted(
            p.name
            for p in rfolder.glob("*.py")
            if not p.name.startswith("test_")
            and p.name not in {"__init__.py", "conftest.py"}
        )
        if names:
            return "; ".join(f"`results/part{pid}/{n}`" for n in names)
    return "unknown"


def top_results(pid: str) -> list:
    # Part 218's runner writes top-level results218/ (the stopped v3 pilot) and results218-v4/ (the run); its
    # public runner defaults there, so those folders stay where they are rather than moving under results/.
    return sorted(p for p in ROOT.glob(f"results{pid}*") if p.is_dir() and
                  (p.name == f"results{pid}" or p.name.startswith(f"results{pid}-")))


def results_for(pid: str) -> str:
    folder = ROOT / "results" / f"part{pid}"
    if folder.is_dir():
        return f"`results/part{pid}/`"
    tops = top_results(pid)
    return "; ".join(f"`{p.name}/`" for p in tops) if tops else "unknown"


def status_for(pid: str, section: dict | None) -> str:
    folder = ROOT / "results" / f"part{pid}"
    if (folder / "README.md").is_file() or any((p / "README.md").is_file() for p in top_results(pid)):
        return "done"
    # Part 180 writes its result to results/part180/v5/RESULT.md, with no
    # README; the run is finished ("the both-head assay is PARKED").
    if folder.is_dir() and any(folder.rglob("RESULT.md")):
        return "done"
    has_design = bool(design_files(pid))
    if section is not None or has_design:
        return "registered, not run"
    return "unknown"


def format_counts(counts: dict[str, int] | None) -> str:
    if not counts:
        return "unknown"
    parts = [
        f"held {counts.get('held', 0)}",
        f"failed {counts.get('failed', 0)}",
        f"open {counts.get('open', 0)}",
    ]
    text = " / ".join(parts)
    extras = []
    for key in EXTRA_ORDER:
        if counts.get(key):
            label = EXTRA_LABEL.get(key, key)
            extras.append(f"{label} {counts[key]}")
    if extras:
        text += "; " + "; ".join(extras)
    return text


def build_rows(ids: list[str], predictions: dict[str, dict]) -> list[str]:
    rows = []
    for pid in ids:
        section = predictions.get(pid)
        designs = design_files(pid)
        design = "; ".join(f"`{rel(p)}`" for p in designs)
        if not design:
            design = "`PREDICTIONS.md`" if section is not None else "unknown"
        date = section["date"] if section else "unknown"
        cells = [
            pid,
            question_for(pid, section),
            status_for(pid, section),
            design,
            code_for(pid),
            results_for(pid),
            format_counts(section["counts"] if section else None),
            date,
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def scan_ids(min_id: str, max_id: str) -> list[str]:
    """Every part id at or above min_id, from every record the table reads."""
    found: set[str] = set(parse_predictions())
    for pattern, base in (
        ("PART*-DESIGN.md", ROOT),
        ("part*", ROOT),
        ("results/part*", ROOT),
    ):
        for p in ROOT.glob(pattern):
            name = p.name
            m = re.match(r"part([0-9]+[A-Za-z]*)", name, re.IGNORECASE)
            if not m:
                continue
            if base == ROOT and pattern == "part*" and not p.is_dir():
                continue
            found.add(m.group(1).lower())
    lo, hi = id_key(min_id), id_key(max_id)
    # PARTS.md lists every integer id in range, including ids with no record
    # (marked unknown).  Lettered ids are only listed when a record has one.
    lo_n = int(re.match(r"[0-9]+", min_id).group())
    hi_n = int(re.match(r"[0-9]+", max_id).group())
    for n in range(lo_n, hi_n + 1):
        found.add(str(n))
    return sorted(
        (pid for pid in found if lo <= id_key(pid) <= hi),
        key=id_key,
    )


def existing_rows() -> dict[str, str]:
    rows = {}
    for line in PARTS.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = split_cells(line)
        if len(cells) >= 8 and re.fullmatch(r"[0-9]+[A-Za-z]*", cells[0]):
            rows[cells[0].lower()] = line.strip()
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("from_id", nargs="?", help="lowest part id (e.g. 179)")
    ap.add_argument("to_id", nargs="?", help="highest part id (e.g. 221)")
    ap.add_argument("--check", nargs="+", metavar="ID", help="compare these rows")
    ap.add_argument("--append", action="store_true", help="append missing rows")
    args = ap.parse_args(argv)

    predictions = parse_predictions()

    if args.check:
        rows = build_rows([i.lower() for i in args.check], predictions)
        old = existing_rows()
        bad = 0
        for pid, row in zip((i.lower() for i in args.check), rows):
            want = old.get(pid)
            if want is None:
                print(f"{pid}: no existing row in PARTS.md")
                bad += 1
                continue
            if want == row:
                print(f"{pid}: exact match")
                continue
            bad += 1
            print(f"{pid}: DIFFERS")
            old_cells, new_cells = split_cells(want), split_cells(row)
            for i, (a, b) in enumerate(zip(old_cells, new_cells)):
                if a != b:
                    print(f"  col {i}: existing {a!r} != generated {b!r}")
            if len(old_cells) != len(new_cells):
                print(f"  column count {len(old_cells)} != {len(new_cells)}")
        return 1 if bad else 0

    if not args.from_id or not args.to_id:
        ap.error("give from_id and to_id, or --check")
    ids = scan_ids(args.from_id, args.to_id)
    rows = build_rows(ids, predictions)
    if args.append:
        old = existing_rows()
        add = [r for pid, r in zip(ids, rows) if pid not in old]
        text = PARTS.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        text += "\n".join(add) + "\n"
        PARTS.write_text(text, encoding="utf-8")
        print(f"appended {len(add)} rows", file=sys.stderr)
        return 0
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
