#!/usr/bin/env python3
"""Join Part 86's version-2 reader labels to the atlas units, through the blind items' answer text.

The version-2 labels (results/part86/second-reader/labels_ds_v2_full.jsonl) are keyed by opaque item ids made with
secrets.token_urlsafe, and the id-to-screen-row key is private and lives in the J-lens study checkout, not here. The
blind items file (id, text) is public in that checkout. Every atlas unit's screen answer text is unique among the
165,376 screen rows, so an item's text identifies its screen row exactly. This tool rebuilds the join from stored
files only; it reads no model and calls no reader.

Inputs: one or more items files (id, text). The full join needs
  $JLENS_ROOT/data/qwen3.5-4b/labels/items.jsonl   (3,349 items; the J-lens study checkout)
and this checkout alone holds results/part86/second-reader/items_v2cal.jsonl (the 335-item version-2 calibration
sample), which joins about one atlas unit in eight. The output says how many units were joined.

Output: one JSON Lines row per atlas unit with its version-2 class, and a summary JSON. When every unit is joined,
the another-AI and other-kind class counts are checked against results/part86/atlas-v2/ATLAS-v2.md (245, 141, 1;
77, 1, 0) and the summary says whether they match.

    python3 tools/atlas_v2_units.py --items ~/jlens-persona-neurons/data/qwen3.5-4b/labels/items.jsonl
    python3 tools/atlas_v2_units.py            # falls back to the calibration items; partial coverage
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "results/part86/second-reader/labels_ds_v2_full.jsonl"
DEFAULT_SCREEN = ROOT / "results/part86/screen-final"
DEFAULT_UNITS = ROOT / "results/part112/units.json"
DEFAULT_OUT = ROOT / "results/part86/atlas-v2/atlas-units-v2.jsonl"
CALIBRATION_ITEMS = ROOT / "results/part86/second-reader/items_v2cal.jsonl"
JLENS_ITEMS = Path(os.environ.get("JLENS_ROOT", "~/jlens-persona-neurons")).expanduser() / "data/qwen3.5-4b/labels/items.jsonl"
EXPECTED = {"another AI": {"coherent_persona": 245, "coherent_still_qwen": 141, "unclear": 1, "not_coherent": 0},
            "other kind": {"coherent_persona": 77, "coherent_still_qwen": 1, "unclear": 0, "not_coherent": 0}}
STILL_QWEN = "still Qwen or an assistant"


def label_class(label: dict) -> str:
    """The version-2 four-way rule, as results/part86/prediction-score/score86_predictions.py applies it."""
    if "error" in label or type(label.get("coherent")) is not bool:
        return "invalid"
    if not label["coherent"]:
        return "not_coherent"
    kind = label.get("kind")
    if kind == STILL_QWEN:
        return "coherent_still_qwen"
    if kind == "unclear":
        return "unclear"
    return "coherent_persona"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_screen(screen_dir: Path) -> dict[tuple[int, int], str]:
    """(layer, neuron) -> screen answer text, for real-neuron rows."""
    answers = {}
    for path in sorted(glob.glob(str(Path(screen_dir) / "layer_*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("kind") == "neuron":
                    answers[(int(row["layer"]), int(row["id"]))] = row["answer"]
    if not answers:
        raise FileNotFoundError(f"no screen rows under {screen_dir}")
    return answers


def load_items(paths: list[Path]) -> tuple[dict[str, list[str]], list[str]]:
    """answer text -> item ids, over every items file given; a missing file is skipped and named."""
    by_text: dict[str, list[str]] = collections.defaultdict(list)
    used = []
    for path in paths:
        if not Path(path).is_file():
            continue
        used.append(str(path))
        for row in read_jsonl(Path(path)):
            item_id, text = row.get("id"), row.get("text")
            if isinstance(item_id, str) and isinstance(text, str) and item_id not in by_text[text]:
                by_text[text].append(item_id)
    return by_text, used


def join(units: list[dict], screen: dict, by_text: dict, labels: dict) -> list[dict]:
    rows = []
    for unit in units:
        key = (int(unit["layer"]), int(unit["neuron"]))
        text = screen.get(key)
        row = {"id": unit["id"], "layer": key[0], "neuron": key[1], "atlas_kind": unit["kind"],
               "q_name": bool(unit.get("q_name")), "atlas_persona": unit.get("persona"),
               "group": "another AI" if unit["kind"] == "another AI" else "other kind",
               "item_id": None, "v2_class": "unjoined", "v2_kind": None, "v2_persona": None, "v2_coherent": None}
        if text is None:
            row["v2_class"] = "no_screen_row"
        else:
            ids = [i for i in by_text.get(text, []) if i in labels]
            if len(ids) == 1:
                label = labels[ids[0]]
                row.update({"item_id": ids[0], "v2_class": label_class(label), "v2_kind": label.get("kind"),
                            "v2_persona": label.get("persona"), "v2_coherent": label.get("coherent")})
            elif len(ids) > 1:
                row["v2_class"] = "ambiguous"
        rows.append(row)
    return rows


def summarize(rows: list[dict], items_used: list[str]) -> dict:
    counts = {g: collections.Counter() for g in ("another AI", "other kind")}
    for row in rows:
        counts[row["group"]][row["v2_class"]] += 1
    joined = sum(r["v2_class"] not in ("unjoined", "no_screen_row", "ambiguous") for r in rows)
    complete = joined == len(rows)
    matches_atlas_v2 = None
    if complete:
        matches_atlas_v2 = all(counts[g][c] == n for g, exp in EXPECTED.items() for c, n in exp.items())
    subset_ai = sorted(r["id"] for r in rows if r["group"] == "another AI" and r["v2_class"] == "coherent_persona")
    subset_all = sorted(r["id"] for r in rows if r["v2_class"] == "coherent_persona")
    return {"units": len(rows), "joined": joined, "coverage": joined / len(rows) if rows else 0.0,
            "complete": complete, "items_files": items_used,
            "counts": {g: dict(c) for g, c in counts.items()},
            "expected_when_complete": EXPECTED, "matches_atlas_v2_md": matches_atlas_v2,
            "subset_another_ai_coherent": subset_ai, "subset_all_coherent": subset_all,
            "subset_sizes": {"another_ai_coherent": len(subset_ai), "all_coherent": len(subset_all)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", action="append", default=None,
                    help="items file (id, text); repeatable. Default: the J-lens items file if present, else the "
                         "335-item calibration sample in this checkout")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--screen", default=str(DEFAULT_SCREEN))
    ap.add_argument("--units", default=str(DEFAULT_UNITS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    item_paths = [Path(p).expanduser() for p in a.items] if a.items else [JLENS_ITEMS, CALIBRATION_ITEMS]
    labels = {}
    for row in read_jsonl(Path(a.labels)):
        if isinstance(row.get("id"), str):
            labels[row["id"]] = row
    by_text, used = load_items(item_paths)
    if not used:
        raise SystemExit("no items file found; pass --items")
    units = json.loads(Path(a.units).read_text(encoding="utf-8"))
    screen = load_screen(Path(a.screen))
    rows = join(units, screen, by_text, labels)
    summary = summarize(rows, used)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("subset_a")}, ensure_ascii=False))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
