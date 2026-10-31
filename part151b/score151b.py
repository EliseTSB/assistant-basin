#!/usr/bin/env python3
"""Part 151b scorer: Part 151's five predictions on new documents, plus the two readings fixed before this data."""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from part151 import score151  # noqa: E402

AWARD = re.compile(r"award|nominat|oscar|prize|emmy|grammy|golden globe|winner|winning", re.IGNORECASE)


def marks_and_or(entry):
    return entry["mark"].strip().lower() in ("and", "or")


def mentions_award(entry):
    return bool(AWARD.search(entry["before"] + entry["mark"] + entry["after"]))


def score(windows, lens):
    r = score151.score(windows, lens)
    units = windows["units"]
    and_or = {u: sum(marks_and_or(e) for e in units[u]["top"]) for u in score151.UNITS}
    award = {u: sum(mentions_award(e) for e in units[u]["top"]) for u in score151.UNITS}
    gates = dict(r["gates"])
    gates["gate1_docs"] = windows["docs"] == 8000
    ok = all(gates.values())
    own = {u: r["hits"][u][u] for u in score151.UNITS}
    held = {"151b.1": sum(r["beats_others"].values()) >= 3, "151b.2": own["L24N8542"] >= 5, "151b.3": own["L24N6835"] >= 5,
            "151b.4": own["L19N4979"] >= 5, "151b.5": own["L19N5975"] >= 5,
            "151b.6": and_or["L19N4979"] >= 10 and all(and_or["L19N4979"] > v for u, v in and_or.items() if u != "L19N4979"),
            "151b.7": award["L24N8542"] >= 10 and all(award["L24N8542"] > v for u, v in award.items() if u != "L24N8542")}
    preds = {k: ("held" if v else "failed") if ok else "untestable" for k, v in held.items()}
    return {"gates": gates, "alien_control_windows": r["alien_control_windows"], "predictions": preds, "hits": r["hits"],
            "beats_others": r["beats_others"], "and_or_top": and_or, "award_top": award}


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
