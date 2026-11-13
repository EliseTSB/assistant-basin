#!/usr/bin/env python3
"""amend4_catalogue.py: write part206/catalogue-amendment4.json, OLMo 2 1B's catalogued neuron sets after Amendment 4.

  python -I -B part206/amend4_catalogue.py [--check]

The registered prompts206.json gives OLMo the slot and concept sets of Part 112, which classified Qwen3.5-4B's persona
neurons (layers 10 to 26), not OLMo's (16 layers, 8,192 neurons per layer). Amendment 4 scores OLMo on Part 92's own
classification of OLMo 2 1B Instruct neurons (results/part92/step2/kinds.json: 21 slot, 18 concept), and keeps the
two unscored Part 92b sets prompts206.json already carries (they are Part 92 units, so already OLMo coordinates).
prompts206.json is left as registered so the recorded runs' plan hashes still match.
--check regenerates the sets and exits 1 if they differ from the file.
Written by Claude (Opus 5.5), 2026-11-13.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "catalogue-amendment4.json"


def build():
    kinds = json.loads((ROOT / "results/part92/step2/kinds.json").read_text())["kinds"]
    where = {u["id"]: (int(u["layer"]), int(u["neuron"]))
             for u in json.loads((ROOT / "results/part92/step2/neurons.json").read_text())}
    slot = sorted(where[i] for i, k in kinds.items() if k == "slot")
    concept = sorted(where[i] for i, k in kinds.items() if k == "concept")
    reg = json.loads((HERE / "prompts206.json").read_text())["catalogued"]["O"]
    part92 = set(where.values())
    for k in ("p92b_ds_slot", "p92b_ds_concept"):
        stray = [x for x in reg[k] if tuple(x) not in part92]
        if stray:
            raise SystemExit(f"{k} in prompts206.json holds units that are not Part 92 neurons: {stray[:3]}")
    return {"source": "Part 206 Amendment 4: slot and concept from results/part92/step2/kinds.json (Part 92, OLMo 2 1B "
                      "Instruct, one Haiku reader); p92b_ds_slot and p92b_ds_concept copied from prompts206.json",
            "O": {"slot": [list(x) for x in slot], "concept": [list(x) for x in concept],
                  "p92b_ds_slot": reg["p92b_ds_slot"], "p92b_ds_concept": reg["p92b_ds_concept"]}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    obj = build()
    text = json.dumps(obj, indent=1) + "\n"
    if a.check:
        same = OUT.exists() and OUT.read_text() == text
        print("catalogue-amendment4.json " + ("matches" if same else "DIFFERS"))
        return 0 if same else 1
    OUT.write_text(text)
    o = obj["O"]
    print(f"wrote {OUT}: " + ", ".join(f"{k} {len(v)}" for k, v in o.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
