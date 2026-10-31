#!/usr/bin/env python3
"""Part 132: join 200 fresh Qwen3-8B persona neurons to the Neuron Lens catalog.

Builds one card per arm: the pushed pole against its own persona (pushed), the same card against another neuron's
persona (mismatch), the opposite pole against its own persona (opposite), and Part 91's random same-layer control at the
same sign against the paired persona (control). Also writes the six KNOWN gate neurons from Neuron Lens' known.py, each
with a true and a mismatched name. The reader never sees the arm. Written by Claude (Opus 5.5), 2026-10-30.
Usage: join128.py CATALOG_DIR OUT_DIR
"""
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SEED = 132
KNOWN = [("Jesus", 8, 1931, 1), ("gods", 9, 3757, -1), ("ant", 13, 5115, 1),
         ("God", 18, 7807, 1), ("mother", 21, 1180, -1), ("political candidate", 27, 1343, 1)]
FIRE_COLS = ["read_R_z_str", "read_S_z_str"]
WRITE_COLS = ["write_R_z_str", "write_S_z_str"]


def carry_counts(tag):
    """Answers of six that show the persona, per persona neuron, from Part 91's stored labels."""
    labels = {r["id"]: r for r in map(json.loads, open(ROOT / "results/part132/labels.jsonl", encoding="utf-8")) if "error" not in r}
    counts = {}
    for i, line in enumerate(open(ROOT / f"results/part132/{tag}.jsonl", encoding="utf-8")):
        row = json.loads(line)
        if not row["cell"].startswith("p"):  # c00 etc. are the controls; "unpushed" has no label
            continue
        if row["cell"] != "unpushed":
            counts.setdefault(row["cell"], 0)
            if labels.get(f"{tag}:{i}", {}).get("shows_persona") is True:
                counts[row["cell"]] += 1
    return counts


def carry_class(n):
    return "carrier" if n >= 3 else "once" if n == 1 else "other"


def table(catalog, name, layer, wanted):
    """Rows of one catalog table for {(neuron, pole)} at one layer."""
    neurons = sorted({n for n, _ in wanted})
    rows = pq.read_table(Path(catalog) / "neurons" / name / f"L{layer:02d}.parquet",
                         filters=[("neuron", "in", neurons)]).to_pylist()
    return {(r["neuron"], r["pole"]): r for r in rows if (r["neuron"], r["pole"]) in wanted}


def pole_record(catalog, cache, layer, neuron, pole):
    key = (layer, neuron, pole)
    if key in cache:
        return cache[key]
    rec = {}
    for name in ("reads", "acts", "drivers", "legible", "labels"):
        rows = table(catalog, name, layer, {(neuron, pole)})
        if (neuron, pole) not in rows:
            raise KeyError(f"{name} has no row for L{layer} n{neuron} pole {pole}")
        rec[name] = rows[(neuron, pole)]
    cache[key] = rec
    return rec


def card(rec):
    contexts = rec["acts"].get("contexts") or []
    return {
        "fire": {
            "weights": " ; ".join(str(rec["reads"][c]) for c in FIRE_COLS),
            "drivers": str(rec["drivers"].get("drivers_str", "")),
            "contexts": [str(c) for c in contexts[:5]],
            "label": str(rec["labels"]["fires"]),
        },
        "write": {
            "weights": " ; ".join(str(rec["reads"][c]) for c in WRITE_COLS),
            "label": str(rec["labels"]["writes"]),
        },
    }


def facts(rec):
    return {"act_top": rec["acts"]["act_top"], "rate": rec["acts"]["rate"], "rate_chat": rec["acts"].get("rate_chat"),
            "dark": bool(rec["legible"]["dark"]), "silent": bool(rec["legible"]["silent"]),
            "cur_mode": rec["legible"]["cur_mode"], "writes_conf": rec["labels"]["writes_conf"],
            "fires_conf": rec["labels"]["fires_conf"],
            "w_norm": rec["reads"]["w_norm"]}


def main(catalog, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    neurons = json.load(open(ROOT / "part132/neurons132.json", encoding="utf-8"))
    chat = carry_counts("inst_chat")
    pushes = {}
    for line in open(ROOT / "results/part132/inst_chat.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if row["cell"].startswith("p"):
            pushes[row["cell"]] = row["c"]
    rng = random.Random(SEED)
    order = list(neurons)
    while True:
        rng.shuffle(order)
        if all(a["id"] != b["id"] and a["persona"] != b["persona"] for a, b in zip(neurons, order)):
            break
    cache, items, units = {}, [], []
    for p, other in zip(neurons, order):
        L, n, s, cn = int(p["L"]), int(p["n"]), int(p["s"]), int(p["control_n"])
        pushed, opposite, control = (pole_record(catalog, cache, L, n, s), pole_record(catalog, cache, L, n, -s),
                                     pole_record(catalog, cache, L, cn, s))
        arms = [("pushed", pushed, p), ("mismatch", pushed, other), ("opposite", opposite, p), ("control", control, p)]
        for arm, rec, target in arms:
            items.append({"id": p["id"], "arm": arm, "name": target["persona"], "desc": target["desc"], "card": card(rec)})
        units.append({"id": p["id"], "L": L, "n": n, "s": s, "persona": p["persona"], "chat": chat[p["id"]],
                      "class": carry_class(chat[p["id"]]), "c": pushes[p["id"]],
                      "pushed": facts(pushed), "opposite": facts(opposite), "control": facts(control)})
    gate = []
    for (name, L, n, s), (other, *_) in zip(KNOWN, KNOWN[1:] + KNOWN[:1]):
        rec = pole_record(catalog, cache, L, n, s)
        gate.append({"id": f"known-{L}-{n}", "arm": "known-true", "name": name, "desc": "", "card": card(rec)})
        gate.append({"id": f"known-{L}-{n}", "arm": "known-false", "name": other, "desc": "", "card": card(rec)})
    with open(out / "cards128.jsonl", "w", encoding="utf-8") as f:
        for it in items + gate:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    json.dump(units, open(out / "units128.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(len(items), "arm rows,", len(gate), "gate rows,", len(units), "units")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
