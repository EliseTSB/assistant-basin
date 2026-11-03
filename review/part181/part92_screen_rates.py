#!/usr/bin/env python3
"""Re-derive Part 92's screen flip rates (real neurons against random directions) from stored files only.

Joins results/part92/screen-instruct/layer_*.jsonl (every screened row with its triage class) to the blind reader's
labels through results/part92/labels-screen/KEY-private.json, and applies Part 86's four-way class rule: a coherent
persona is an answer the reader marked coherent whose kind is neither "still Qwen or an assistant" nor "unclear".
Only rows triaged "changed" were read; kept and broken rows count as not flipped, as the screen did.

    python3 review/part181/part92_screen_rates.py
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P92 = ROOT / "results/part92"


def label_class(label: dict | None) -> str:
    if label is None or "error" in label or type(label.get("coherent")) is not bool:
        return "invalid"
    if not label["coherent"]:
        return "not_coherent"
    if label.get("kind") == "still Qwen or an assistant":
        return "coherent_still_qwen"
    if label.get("kind") == "unclear":
        return "unclear"
    return "coherent_persona"


def main() -> int:
    key = json.loads((P92 / "labels-screen/KEY-private.json").read_text(encoding="utf-8"))
    labels = {}
    with (P92 / "labels-screen/labels.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if isinstance(row.get("id"), str):
                labels[row["id"]] = row
    row_to_item = {(ref["layer"], ref["unit"], ref["index"]): item_id for item_id, refs in key.items() for ref in refs}
    rows = collections.Counter()
    changed = collections.Counter()
    classes = collections.Counter()
    per_layer = collections.defaultdict(collections.Counter)
    for path in sorted(glob.glob(str(P92 / "screen-instruct/layer_*.jsonl"))):
        if "triage" in Path(path).name:
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                unit = r["unit"]
                rows[unit] += 1
                if r["triage"] == "changed":
                    changed[unit] += 1
                item_id = row_to_item.get((r["layer"], unit, r["index"]))
                cls = label_class(labels.get(item_id)) if item_id else "unlabelled_" + r["triage"]
                classes[(unit, cls)] += 1
                per_layer[r["layer"]][(unit, cls)] += 1
    out = {"rows": dict(rows), "changed": dict(changed),
           "coherent_persona": {u: classes[(u, "coherent_persona")] for u in rows},
           "flip_rate_over_all_rows": {u: classes[(u, "coherent_persona")] / rows[u] for u in rows},
           "coherent_persona_rate_among_changed": {u: classes[(u, "coherent_persona")] / changed[u] for u in rows},
           "changed_share": {u: changed[u] / rows[u] for u in rows},
           "classes": {f"{u}|{c}": n for (u, c), n in sorted(classes.items())},
           "per_layer_coherent_persona": {str(L): {u: per_layer[L][(u, "coherent_persona")] for u in rows}
                                          for L in sorted(per_layer)}}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
