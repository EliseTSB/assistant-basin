#!/usr/bin/env python3
"""Part 151 scorer: does a unit's own lens list hit its own top windows more than the other units' lists do?"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORD = re.compile(r"[A-Za-z]+")
UNITS = ("L19N4979", "L19N5975", "L23N116", "L24N6835", "L24N8542")
CONTROL = "L22N2742"
ALIEN = ("alien", "extraterrestrial", "ufo", "martian")


def hit(entry, lens):
    """The marked token, or anything in the words after it, starts with a lens word."""
    words = [w.lower() for w in WORD.findall(entry["mark"] + " " + entry["after"])]
    return any(w.startswith(s) for w in words for s in lens)


def score(windows, lens):
    units = windows["units"]
    table = {u: {v: sum(hit(e, lens[v]) for e in units[u]["top"]) for v in UNITS} for u in UNITS}
    own = {u: table[u][u] for u in UNITS}
    beats = {u: all(own[u] > table[u][v] for v in UNITS if v != u) for u in UNITS}
    control = sum(any(re.search(rf"\b{a}", e["before"] + e["mark"] + e["after"], re.IGNORECASE) for a in ALIEN) for e in units[CONTROL]["top"])
    gates = {"gate1_docs": windows["docs"] == 2000, "gate2_full_top_lists": all(len(units[u]["top"]) == 20 for u in units),
             "gate3_alien_control": control >= 5}
    ok = all(gates.values())
    held = {"151.1": sum(beats.values()) >= 3, "151.2": own["L24N8542"] >= 5, "151.3": own["L24N6835"] >= 5,
            "151.4": own["L19N4979"] >= 5, "151.5": own["L19N5975"] >= 5}
    preds = {k: ("held" if v else "failed") if ok else "withheld" for k, v in held.items()}
    return {"gates": gates, "alien_control_windows": control, "predictions": preds, "hits": table, "beats_others": beats}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--lens", default=str(HERE.parent / "part150/lens_words.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    result = score(json.loads(Path(a.windows).read_text(encoding="utf-8")), json.loads(Path(a.lens).read_text(encoding="utf-8")))
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not all(result["gates"].values()):
        sys.exit(3)


if __name__ == "__main__":
    main()
