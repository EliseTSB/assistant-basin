#!/usr/bin/env python3
"""Exploratory census of apology openers in Part 92's stored screen. Reads only; no model."""
from __future__ import annotations

import collections
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENER = re.compile(r"^\W*(i'?m\s+(so\s+|very\s+|really\s+)?sorry|i\s+am\s+(so\s+|very\s+)?sorry|sorry\b"
                    r"|i\s+apologi[sz]e|apologies\b|my\s+apologies)", re.I)


def main() -> None:
    table = collections.defaultdict(lambda: [0, 0])
    for path in sorted(glob.glob(str(ROOT / "results/part92/screen-instruct/layer_*.jsonl"))):
        if "triage" in path:
            continue
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                cell = table[(int(row["layer"]), row["unit"])]
                cell[0] += 1
                cell[1] += bool(OPENER.search(row["answer"]))
    rows = [{"layer": layer, "unit": unit, "n": n, "apology_openers": hits, "percent": round(100 * hits / n, 2)}
            for (layer, unit), (n, hits) in sorted(table.items())]
    out = ROOT / "results/part139/census.json"
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
