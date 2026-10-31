#!/usr/bin/env python3
"""Score Part 156 (PART156-DESIGN.md): gates, predictions, a table and the curve. Prints every number."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(ROOT / "part86")):
    if path not in sys.path:
        sys.path.insert(0, path)

from part153.score153 import company_hit  # noqa: E402
from part155.run155 import REGISTERED_COORDS  # noqa: E402
import triage86  # noqa: E402

DOSES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
UNITS = {"L24N5017": "Google", "L21N7134": "Baidu", "L23N1743": "Huawei", "L23N2993": "Microsoft"}
PHRASINGS = 24
ARMS = ("neuron", "random")


def table(rows):
    hits = {u: {arm: {d: 0 for d in DOSES} for arm in ("neuron", "random")} for u in UNITS}
    broken = {u: {d: 0 for d in DOSES} for u in UNITS}
    for row in rows:
        if row["arm"] == "plain":
            continue
        company = UNITS[row["unit_id"]]
        hit = company_hit(row["answer"], company)
        hits[row["unit_id"]][row["arm"]][row["dose"]] += int(hit)
        if row["arm"] == "neuron":
            broken[row["unit_id"]][row["dose"]] += int(triage86.triage_answer(row["answer"]) == "broken")
    return hits, broken


def predictions(hits):
    neuron = {u: hits[u]["neuron"] for u in UNITS}
    random = {u: hits[u]["random"] for u in UNITS}
    r1 = sum(any(v[d] >= 12 for d in DOSES if d != 0.8) for v in neuron.values())
    r2 = sum(any(v[d] > v[0.8] for d in DOSES if d != 0.8) for v in neuron.values())
    r3 = max(v[0.5] for v in neuron.values())
    r4 = 0
    for v in neuron.values():
        at = [i for i, d in enumerate(DOSES) if v[d] >= 12]
        r4 += int(bool(at) and at == list(range(at[0], at[-1] + 1)) and len(at) <= 4)
    r5 = []
    for d in DOSES:
        n, r = sum(v[d] for v in neuron.values()), sum(v[d] for v in random.values())
        if n >= 24:
            r5.append({"dose": d, "neurons": n, "random": r, "ok": r < n / 4})
    return {
        "156.1": {"value": r1, "verdict": "held" if r1 >= 3 else "failed"},
        "156.2": {"value": r2, "verdict": "held" if r2 >= 2 else "failed"},
        "156.3": {"value": r3, "verdict": "held" if r3 <= 3 else "failed"},
        "156.4": {"value": r4, "verdict": "held" if r4 >= 3 else "failed"},
        "156.5": {"value": r5, "verdict": "held" if r5 and all(x["ok"] for x in r5) else "failed"},
    }


def score(rows):
    keys = {(r["unit_id"], r["arm"], float(r["dose"]), int(r["question_index"])) for r in rows}
    expected_keys = {
        (unit, arm, dose, question_index)
        for unit in UNITS
        for arm in ARMS
        for dose in DOSES
        for question_index in range(PHRASINGS)
    } | {("plain", "plain", 0.0, question_index) for question_index in range(PHRASINGS)}
    phrasing_counts = {q: 0 for q in range(PHRASINGS)}
    for row in rows:
        q = int(row["question_index"])
        if q in phrasing_counts:
            phrasing_counts[q] += 1
    phrasing_counts_valid = all(count == 57 for count in phrasing_counts.values())
    nonempty = sum(bool(str(r.get("answer", "")).strip()) for r in rows)
    plain = [r for r in rows if r["arm"] == "plain"]
    plain_hits = sum(any(company_hit(r["answer"], c) for c in UNITS.values()) for r in plain)
    hits, broken = table(rows)
    at_08 = sum(hits[u]["neuron"][0.8] for u in UNITS)
    gate1 = len(rows) == 1368 and keys == expected_keys and phrasing_counts_valid and nonempty == 1368
    gate2 = 59 <= at_08 <= 81 and plain_hits <= 2
    gates = {"gate1": {"passed": gate1, "rows": len(rows), "unique": len(keys),
                        "expected_unique": len(expected_keys), "matches_expected_keys": keys == expected_keys,
                        "57_rows_each_phrasing": phrasing_counts_valid, "nonempty": nonempty},
             "gate2": {"passed": gate2, "at_0_8": at_08, "plain_company_hits": plain_hits}}
    preds = predictions(hits)
    if not (gate1 and gate2):
        preds = {k: {**v, "verdict": "untestable"} for k, v in preds.items()}
    return {"gates": gates, "predictions": preds, "hits": hits, "broken": broken}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(args.answers, encoding="utf-8") if line.strip()]
    result = score(rows)
    out = Path(args.out)
    (out / "score.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    lines = ["# Part 156 score", "", "| Unit | " + " | ".join(f"+{d}" for d in DOSES) + " |", "|---|" + "---:|" * len(DOSES)]
    for u in UNITS:
        n, r, b = result["hits"][u]["neuron"], result["hits"][u]["random"], result["broken"][u]
        lines.append(f"| {u} | " + " | ".join(f"{n[d]}/{r[d]} ({b[d]})" for d in DOSES) + " |")
    lines += ["", "Each cell: neuron hits / random hits (broken neuron answers), of 24.", ""]
    lines += [f"- {k}: {v['verdict']} ({v['value']})" for k, v in result["predictions"].items()]
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for u in UNITS:
        ax.plot(DOSES, [result["hits"][u]["neuron"][d] for d in DOSES], marker="o", label=f"{u} ({UNITS[u]})")
    ax.set_xlabel("Dose"); ax.set_ylabel("Company hits (of 24)"); ax.set_ylim(0, 24); ax.legend(); ax.set_title("Part 156: the company peak on a finer ladder")
    fig.tight_layout(); fig.savefig(out / "fine156.png", dpi=150)
    print("\n".join(lines))
    return 0 if all(g["passed"] for g in result["gates"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
