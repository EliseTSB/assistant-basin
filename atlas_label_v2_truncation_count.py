#!/usr/bin/env python3
"""Read-only count of finished atlas labels whose 21 input windows had >12 marked-token types."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
JOBS = (
    ("label", Path("results/part56-pilot/label/atlas-labels.jsonl"),
     Path("results/part56-pilot/label/collection-summary.json"), Path("results/part56-pilot/pilot.jsonl")),
    ("label-stability", Path("results/part56-pilot/label-stability/atlas-labels.jsonl"),
     Path("results/part56-pilot/label-stability/collection-summary.json"),
     Path("results/part56-pilot/stability.jsonl")),
)


def marked_token(window: str) -> str:
    start = window.find("[[")
    end = window.find("]]", start + 2) if start >= 0 else -1
    return window[start + 2:end] if start >= 0 and end >= 0 else ""


def windows_for(record: dict[str, Any]) -> list[str]:
    windows: list[str] = []
    for section, limit in (("top", 12), ("random", 6), ("weak", 3)):
        items = record.get(section, [])
        if not isinstance(items, list):
            raise ValueError(f"{section} is not a list")
        windows.extend(item["window"] for item in items[:limit]
                       if isinstance(item, dict) and isinstance(item.get("window"), str))
    return windows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    total_labels = total_truncated = total_counted = 0
    group_counts: dict[tuple[str, int, str, str], list[int]] = defaultdict(lambda: [0, 0])
    missing: set[Path] = set()
    for name, labels_rel, summary_rel, windows_rel in JOBS:
        labels_path, summary_path, windows_path = (ROOT / p for p in (labels_rel, summary_rel, windows_rel))
        if not labels_path.is_file() or not summary_path.is_file():
            print(f"SKIPPED {name}: no finished labels/collection summary at {labels_path} / {summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("partial_results") is not False:
            print(f"SKIPPED {name}: collection summary does not mark results complete")
            continue
        labels = read_jsonl(labels_path)
        if not windows_path.is_file():
            missing.add(windows_path)
            print(f"MISSING windows: {windows_path}")
            total_labels += len(labels)
            continue
        records = read_jsonl(windows_path)
        total_labels += len(labels)
        for label in labels:
            try:
                idx = int(label["record_index"])
                record = records[idx]
                windows = windows_for(record)
                if not windows:
                    raise IndexError("record has no usable windows")
            except (KeyError, IndexError, TypeError, ValueError):
                missing.add(windows_path)
                continue
            total_counted += 1
            is_truncated = len({marked_token(window) for window in windows}) > 12
            key = (name, int(label["layer"]), str(label.get("model_arm", "trained")),
                   str(label.get("record_type", "unknown")))
            group_counts[key][0] += int(is_truncated)
            group_counts[key][1] += 1
            total_truncated += int(is_truncated)
        if len(labels) != int(summary.get("labels", len(labels))):
            print(f"WARNING {name}: {len(labels)} label rows differ from complete-summary labels={summary.get('labels')}")

    print("truncated / finished label records by label set, layer, arm, record type")
    for key in sorted(group_counts):
        truncated, denominator = group_counts[key]
        print(f"{key[0]} layer={key[1]} arm={key[2]} type={key[3]}: {truncated}/{denominator} ({truncated / denominator:.1%})")
    print(f"TOTAL: {total_truncated}/{total_labels} finished labels ({total_truncated / total_labels:.1%} of all finished labels)")
    print(f"Windows verified: {total_counted}/{total_labels}")
    if missing:
        print("Missing/unmatchable windows files:")
        for path in sorted(missing):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
