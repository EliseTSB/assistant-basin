#!/usr/bin/env python3
"""Amendment 2: retry only the pairs whose reader error was a DNS failure, one at a time."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from part173 import label173  # noqa: E402
from part174 import label174  # noqa: E402

ANSWERS = ROOT / "results/part175/answers.jsonl"
PATH = ROOT / "results/part175/labels/labels-ds.jsonl"
MARK = "name resolution"
TRIES = 5


def main():
    items = {str(item["id"]): item for item in label173.items_for(ANSWERS)}
    prior, by_id = label174._valid_file(PATH)
    targets = [key for key, row in by_id.items() if MARK in str(row.get("error") or "")]
    updated = dict(by_id)
    done = 0
    for key in targets:
        for attempt in range(TRIES):
            row = label173.label_one(items[key], "ds", attempts=1)
            if MARK not in str(row.get("error") or ""):
                updated[key] = row
                done += 1
                break
            time.sleep(10)
    label174._write_checkpoint(PATH, prior, updated, targets)
    print({"targets": len(targets), "answered_by_a_model": done})
    return 0 if done == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
