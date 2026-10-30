#!/usr/bin/env python3
"""Compare Part 118 alpha-0.8 neuron answers with the stored Part 92 replay."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with repo_path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            rows.append(row)
    return rows


def compare_answers(battery_rows: list[dict], stored_rows: list[dict],
                    alpha: float = 0.8) -> dict:
    stored = {}
    for row in stored_rows:
        arm = row.get("arm", row.get("unit"))
        if arm != "neuron":
            continue
        key = (str(row["id"]), int(row["question_index"]))
        if key in stored:
            raise ValueError(f"duplicate stored Part 92 answer cell {key}")
        stored[key] = str(row["answer"])

    candidates = {}
    for row in battery_rows:
        arm = row.get("arm", row.get("unit"))
        if arm != "neuron" or not math.isclose(float(row.get("alpha", -1)), alpha,
                                                rel_tol=0, abs_tol=1e-8):
            continue
        key = (str(row["id"]), int(row["question_index"]))
        if key in candidates:
            raise ValueError(f"duplicate Part 118 alpha-{alpha} neuron cell {key}")
        candidates[key] = str(row["answer"])

    differences = []
    matches = 0
    for key in sorted(candidates):
        if key not in stored:
            differences.append({"cell": key, "current": candidates[key],
                                "stored": None, "reason": "missing Part 92 reference"})
        elif candidates[key] == stored[key]:
            matches += 1
        else:
            differences.append({"cell": key, "current": candidates[key],
                                "stored": stored[key], "reason": "answer differs"})
    compared = len(candidates)
    rate = matches / compared if compared else 0.0
    return {"matches": matches, "compared": compared, "rate": rate,
            "differences": differences, "passed": compared > 0 and rate >= 0.95}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True, help="Part 118 battery JSONL")
    parser.add_argument("--reference", default="results/part92/step2/answers/instruct_chat.jsonl")
    parser.add_argument("--alpha", type=float, default=0.8)
    args = parser.parse_args(argv)
    result = compare_answers(read_jsonl(args.answers), read_jsonl(args.reference), args.alpha)
    percent = 100 * result["rate"]
    state = "PASS" if result["passed"] else "FAIL"
    print(f"Gate 1 {state}: {result['matches']} matches out of {result['compared']} cells "
          f"compared ({percent:.2f}%; threshold 95%)")
    for difference in result["differences"]:
        print(f"DIFFERENCE {difference['cell']}: {difference['reason']}\n"
              f"  Part 118: {difference['current']!r}\n"
              f"  Part 92:  {difference['stored']!r}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
