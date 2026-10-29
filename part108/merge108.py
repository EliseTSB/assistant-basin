#!/usr/bin/env python3
"""Part 108: one unit list, kind map and push file from Part 92 (slot and concept) and Part 104 (slot).

Writes results/part108/neurons108.json, kinds108.json, pushes108.pt (the fixed pushes, untracked like the others) and
pushes108.json (which pushes, and their norms). Refuses if a unit lacks either push or a field swap99 and label92 read.
Written by Claude (Opus 5.5), 2026-10-29. Usage: python part108/merge108.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (  # neurons, kinds, pushes, kinds kept
    ("results/part92/step2/neurons.json", "results/part92/step2/kinds.json", "results/part99-pushes.pt", {"slot", "concept"}),
    ("results/part104/neurons.json", "results/part104/kinds.json", "results/part104/pushes104.pt", {"slot"}),
)
FIELDS = ("id", "layer", "neuron", "persona", "desc", "control_seed")


def main():
    import torch

    neurons, kinds, pushes, sources = [], {}, {}, {}
    for npath, kpath, ppath, keep in SOURCES:
        rows = json.loads((ROOT / npath).read_text(encoding="utf-8"))
        kmap = json.loads((ROOT / kpath).read_text(encoding="utf-8"))
        kmap = kmap.get("kinds", kmap)
        vecs = torch.load(ROOT / ppath, map_location="cpu")
        for row in rows:
            uid = str(row["id"])
            if kmap.get(uid) not in keep:
                continue
            missing = [f for f in FIELDS if f not in row]
            if missing or uid in kinds:
                raise SystemExit(f"{uid}: missing {missing} or duplicate id")
            for unit in ("neuron", "random"):
                if f"{uid}|{unit}" not in vecs:
                    raise SystemExit(f"{uid}: no {unit} push in {ppath}")
                pushes[f"{uid}|{unit}"] = vecs[f"{uid}|{unit}"].clone()
            neurons.append({f: row[f] for f in FIELDS})
            kinds[uid] = kmap[uid]
            sources[uid] = ppath
    out = ROOT / "results/part108"
    out.mkdir(parents=True, exist_ok=True)
    (out / "neurons108.json").write_text(json.dumps(neurons, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "kinds108.json").write_text(json.dumps({"kinds": kinds}, indent=1) + "\n", encoding="utf-8")
    torch.save(pushes, out / "pushes108.pt")
    desc = {"units": len(neurons), "kind_counts": {k: list(kinds.values()).count(k) for k in sorted(set(kinds.values()))},
            "source": sources, "norms": {k: round(float(v.float().norm()), 4) for k, v in sorted(pushes.items())}}
    (out / "pushes108.json").write_text(json.dumps(desc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"units": len(neurons), "kind_counts": desc["kind_counts"], "pushes": len(pushes)}))


if __name__ == "__main__":
    main()
