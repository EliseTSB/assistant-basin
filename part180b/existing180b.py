#!/usr/bin/env python3
"""Exploratory, before registration: the codebook's gold on Part 92's real perfect-day answers (full text) against
the labels two readers already gave those answers on the same prompt: Claude Haiku 5.5 in Part 92
(results/part92/step2/labels-instruct_chat.jsonl) and DeepSeek V4.1 Flash in Part 92b (results/part92b/labels-ds.jsonl).

No reader is called. Output: results/part180b/existing-readers.json and a markdown table on stdout. This is
description of stored data, computed after the codebook was written and before any Part 180b item was labelled.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PERFECT_DAY = 3


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_labels(path: Path) -> dict:
    out = {}
    for row in read_jsonl(path):
        if int(row["question_index"]) == PERFECT_DAY and "error" not in row:
            out[(row["id"], row["unit"])] = row
    return out


def compare(codebook: dict, labels: dict) -> dict:
    """Per codebook category: how the stored reader labelled the full real answers."""
    by_cat = defaultdict(Counter)
    direction = Counter()
    rows = []
    for key, entry in sorted(codebook.items()):
        pid, unit, form = key.split("/")
        if form != "full":
            continue
        label = labels.get((pid, unit))
        if label is None:
            by_cat[(unit, entry["category"], entry["boundary"])]["missing"] += 1
            continue
        gold = entry["shows_persona"]
        read = bool(label["shows_persona"])
        cell = by_cat[(unit, entry["category"], entry["boundary"])]
        cell["n"] += 1
        cell["reader_yes"] += read
        cell["gold_yes"] += gold
        cell["agree"] += (read == gold)
        if read != gold and not entry["boundary"]:
            direction["reader_yes_gold_no" if read else "reader_no_gold_yes"] += 1
        rows.append({"id": pid, "unit": unit, "category": entry["category"], "boundary": entry["boundary"],
                     "gold": gold, "reader": read, "coherent_gold": entry["coherent"],
                     "coherent_reader": bool(label["coherent"])})
    clear = [row for row in rows if not row["boundary"]]
    boundary = [row for row in rows if row["boundary"]]
    return {
        "items": len(rows),
        "clear": {"n": len(clear), "agree": sum(r["gold"] == r["reader"] for r in clear),
                  "disagreements": direction},
        "boundary": {"n": len(boundary), "reader_yes": sum(r["reader"] for r in boundary),
                     "gold_yes": sum(r["gold"] for r in boundary),
                     "agree": sum(r["gold"] == r["reader"] for r in boundary)},
        "coherent": {"n": len(rows), "agree": sum(r["coherent_gold"] == r["coherent_reader"] for r in rows),
                     "reader_yes": sum(r["coherent_reader"] for r in rows),
                     "gold_yes": sum(r["coherent_gold"] for r in rows)},
        "by_category": [{"unit": u, "category": c, "boundary": b, **dict(counter)}
                        for (u, c, b), counter in sorted(by_cat.items())],
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook", default=str(HERE / "codebook180b.json"))
    parser.add_argument("--out", default="results/part180b/existing-readers.json")
    args = parser.parse_args(argv)
    codebook = json.loads(Path(args.codebook).read_text(encoding="utf-8"))
    readers = {
        "haiku_part92": load_labels(ROOT / "results/part92/step2/labels-instruct_chat.jsonl"),
        "ds_part92b": load_labels(ROOT / "results/part92b/labels-ds.jsonl"),
    }
    report = {name: compare(codebook, labels) for name, labels in readers.items()}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("| Reader (stored labels) | Clear items agree | Reader yes, gold no | Reader no, gold yes | "
          "Boundary items: reader yes / gold yes / n | Coherent agree |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, result in report.items():
        clear, boundary, coh = result["clear"], result["boundary"], result["coherent"]
        print(f"| {name} | {clear['agree']} of {clear['n']} | {clear['disagreements'].get('reader_yes_gold_no', 0)} | "
              f"{clear['disagreements'].get('reader_no_gold_yes', 0)} | {boundary['reader_yes']} / "
              f"{boundary['gold_yes']} / {boundary['n']} | {coh['agree']} of {coh['n']} |")
    print()
    print("| Reader | Arm | Coded category | Boundary | n | Reader yes | Gold yes | Agree |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, result in report.items():
        for cell in result["by_category"]:
            print(f"| {name} | {cell['unit']} | {cell['category']} | {'yes' if cell['boundary'] else 'no'} | "
                  f"{cell.get('n', 0)} | {cell.get('reader_yes', 0)} | {cell.get('gold_yes', 0)} | {cell.get('agree', 0)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
