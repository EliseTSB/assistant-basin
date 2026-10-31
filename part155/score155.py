#!/usr/bin/env python3
"""Score Part 155's registered gates and dose-ladder predictions."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PART86 = ROOT / "part86"
if str(PART86) not in sys.path:
    sys.path.insert(0, str(PART86))

from part153 import score153  # noqa: E402
from part153.battery153 import GUARDS, UNIT_COMPANIES  # noqa: E402
from part86 import triage86  # noqa: E402
from part155.run155 import DOSES, questions  # noqa: E402

UNITS = [(l, n, c) for l, n, c in UNIT_COMPANIES if (l, n) in {(24, 5017), (21, 7134), (23, 1743), (23, 2993)}]
UNIT_IDS = [f"L{l}N{n}" for l, n, _ in UNITS]
POSITIVE = (0.2, 0.4, 0.8, 1.2, 1.6)
NEGATIVE = (-0.2, -0.4, -0.8, -1.2, -1.6)


def load_answers(path):
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}") from exc
    return rows


def score_rows(rows):
    items = questions()
    expected = {(uid, arm, dose, qi) for uid in UNIT_IDS for arm in ("neuron", "random")
                for dose in DOSES for qi in range(len(items))}
    expected.update(("plain", "plain", 0.0, qi) for qi in range(len(items)))
    indexed = {}
    duplicates = 0
    for row in rows:
        try:
            key = (row.get("unit_id"), row.get("arm"), float(row.get("dose")), int(row.get("question_index", -1)))
        except (TypeError, ValueError):
            duplicates += 1
            continue
        if key in indexed:
            duplicates += 1
        indexed[key] = row
    n_expected = len(expected)
    nonempty = sum(isinstance(r.get("answer"), str) and bool(r["answer"].strip()) for r in indexed.values())
    complete = expected.issubset(indexed) and len(indexed) == n_expected and len(rows) == n_expected and duplicates == 0
    gate1 = {"passed": complete and nonempty == n_expected, "expected_rows": n_expected,
             "rows": len(rows), "unique_rows": len(indexed), "nonempty": nonempty, "duplicates": duplicates}

    maker_indices = [i for i, q in enumerate(items) if q["kind"] == "maker"]
    def get(uid, arm, dose, qi):
        return indexed.get((uid, arm, float(dose), qi), {})
    def company_hit(row, company):
        return score153.company_hit(row.get("answer", ""), company)
    def count_company(uid, arm, dose, indices):
        company = next(c for l, n, c in UNITS if f"L{l}N{n}" == uid)
        return sum(int(company_hit(get(uid, arm, dose, qi), company)) for qi in indices)
    def is_broken(row):
        answer = row.get("answer", "")
        return triage86.triage_answer(answer if isinstance(answer, str) else "") == "broken"
    neuron_08 = sum(count_company(uid, "neuron", 0.8, maker_indices) for uid in UNIT_IDS)
    plain_hits = 0
    companies = {company for _, _, company in UNITS}
    for qi in range(len(items)):
        answer = get("plain", "plain", 0.0, qi).get("answer", "")
        plain_hits += int(any(score153.company_hit(answer, company) for company in companies))
    gate2 = {"passed": 56 <= neuron_08 <= 76 and plain_hits <= 2,
             "neuron_maker_hits_at_positive_0_8": neuron_08, "expected_range": [56, 76],
             "plain_company_hits": plain_hits, "plain_max": 2, "plain_questions": len(items)}

    cell = {}
    for uid in UNIT_IDS:
        company = next(c for l, n, c in UNITS if f"L{l}N{n}" == uid)
        for arm in ("neuron", "random"):
            for dose in DOSES:
                stats = defaultdict(lambda: {"hits": 0, "n": 0, "broken": 0})
                for qi, item in enumerate(items):
                    row = get(uid, arm, dose, qi)
                    kind = item["kind"]
                    if kind == "guard":
                        hit = score153.guard_hit(row.get("answer", ""), dict(GUARDS)[item["question"]])
                        bucket = "guards"
                    elif kind in ("maker", "identity"):
                        hit = company_hit(row, company)
                        bucket = kind
                    else:
                        continue
                    stats[bucket]["n"] += 1
                    stats[bucket]["hits"] += int(hit)
                    stats[bucket]["broken"] += int(is_broken(row))
                cell[(uid, arm, float(dose))] = dict(stats)

    # Counts are maker-only for broken shares in 155.4; off-target categories remain separately tabulated.
    def maker_broken(dose):
        return sum(cell[(uid, "neuron", dose)]["maker"]["broken"] for uid in UNIT_IDS)
    broken08, broken16 = maker_broken(0.8), maker_broken(1.6)
    share08, share16 = broken08 / 96, broken16 / 96

    pred = {}
    monotonic_units = {}
    for uid in UNIT_IDS:
        a = count_company(uid, "neuron", 0.8, maker_indices)
        b = count_company(uid, "neuron", 0.4, maker_indices)
        monotonic_units[uid] = {"at_0_8": a, "at_0_4": b, "held": a >= b}
    pred["155.1"] = all(v["held"] for v in monotonic_units.values())

    step_hits = {}
    for uid in UNIT_IDS:
        counts = [count_company(uid, "neuron", d, maker_indices) for d in POSITIVE]
        step_hits[uid] = {"counts": dict(zip((str(d) for d in POSITIVE), counts)),
                          "qualifying_steps": [[POSITIVE[i], POSITIVE[i + 1]] for i in range(4)
                                               if counts[i] <= 3 and counts[i + 1] >= 12]}
    pred["155.2"] = sum(bool(v["qualifying_steps"]) for v in step_hits.values()) >= 2

    negative_counts = {uid: {str(d): count_company(uid, "neuron", d, maker_indices) for d in NEGATIVE}
                       for uid in UNIT_IDS}
    pred["155.3"] = all(n < 6 for unit in negative_counts.values() for n in unit.values())
    pred["155.4"] = broken16 >= 2 * broken08

    specificity = {}
    for dose in POSITIVE:
        neuron_total = sum(count_company(uid, "neuron", dose, maker_indices) for uid in UNIT_IDS)
        random_total = sum(count_company(uid, "random", dose, maker_indices) for uid in UNIT_IDS)
        if neuron_total >= 24:
            specificity[str(dose)] = {"neuron_hits": neuron_total, "random_hits": random_total,
                                      "held_at_dose": random_total < neuron_total / 2}
    pred["155.5"] = all(v["held_at_dose"] for v in specificity.values())

    target = "L24N5017"
    maker_threshold = {d: count_company(target, "neuron", d, maker_indices) for d in POSITIVE}
    identity_indices = [i for i, q in enumerate(items) if q["kind"] == "identity"]
    identity_threshold = {d: count_company(target, "neuron", d, identity_indices) for d in POSITIVE}
    maker_reach = next((d for d in POSITIVE if maker_threshold[d] >= 12), None)
    identity_reach = next((d for d in POSITIVE if identity_threshold[d] >= 4), None)
    pred["155.6"] = maker_reach is not None and identity_reach is not None and maker_reach <= identity_reach

    guard_indices = [i for i, q in enumerate(items) if q["kind"] == "guard"]
    guard_counts = {uid: {str(d): sum(score153.guard_hit(get(uid, "neuron", d, qi).get("answer", ""),
                                               dict(GUARDS)[items[qi]["question"]])
                                          for qi in guard_indices) for d in (0.4, -0.4)} for uid in UNIT_IDS}
    pred["155.7"] = all(n >= 5 for unit in guard_counts.values() for n in unit.values())

    all_passed = gate1["passed"] and gate2["passed"]
    predictions = {name: ("held" if value else "failed") if all_passed else "untestable" for name, value in pred.items()}
    per_unit = {}
    for uid in UNIT_IDS:
        per_unit[uid] = {"company": next(c for l, n, c in UNITS if f"L{l}N{n}" == uid), "doses": {}}
        for dose in DOSES:
            per_unit[uid]["doses"][str(dose)] = {
                "maker_neuron": cell[(uid, "neuron", dose)]["maker"]["hits"],
                "maker_neuron_broken": cell[(uid, "neuron", dose)]["maker"]["broken"],
                "maker_random": cell[(uid, "random", dose)]["maker"]["hits"],
                "identity_neuron": cell[(uid, "neuron", dose)]["identity"]["hits"],
                "guards_neuron": cell[(uid, "neuron", dose)]["guards"]["hits"],
            }
    return {"gates": {"gate1_rows": gate1, "gate2_positive_0_8_and_plain": gate2, "all_passed": all_passed},
            "predictions": predictions, "per_unit": per_unit,
            "summary": {"neuron_maker_hits_at_0_8": neuron_08, "plain_company_hits": plain_hits,
                        "maker_broken_at_0_8": broken08, "maker_broken_at_1_6": broken16,
                        "maker_broken_share_at_0_8": share08, "maker_broken_share_at_1_6": share16,
                        "155.1_units": monotonic_units, "155.2_steps": step_hits,
                        "155.3_negative_counts": negative_counts, "155.5_specificity": specificity,
                        "155.6_maker_counts": maker_threshold, "155.6_identity_counts": identity_threshold,
                        "155.6_maker_lowest_dose": maker_reach, "155.6_identity_lowest_dose": identity_reach,
                        "155.7_guards": guard_counts}}


def write_tables(result, path):
    lines = ["# Part 155 score", "", "| Unit | Dose | Maker neuron | Maker random | Identity neuron | Guards neuron | Broken maker |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for uid, unit in result["per_unit"].items():
        for dose in DOSES:
            row = unit["doses"][str(dose)]
            lines.append(f"| {uid} | {dose:+.1f} | {row['maker_neuron']}/24 | {row['maker_random']}/24 | {row['identity_neuron']}/8 | {row['guards_neuron']}/6 | {row['maker_neuron_broken']}/24 |")
    lines.extend(["", "## Gates", ""])
    for name, gate in result["gates"].items():
        if name != "all_passed":
            lines.append(f"- {name}: {'passed' if gate['passed'] else 'failed'} ({gate})")
    lines.extend(["", "## Predictions", ""])
    lines.extend(f"- {name}: {status}" for name, status in result["predictions"].items())
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_figure(result, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(9, 5))
    for uid, unit in result["per_unit"].items():
        ys = [unit["doses"][str(d)]["maker_neuron"] for d in POSITIVE]
        ax.plot(POSITIVE, ys, marker="o", label=uid)
    ax.set(xlabel="Dose", ylabel="Company hits on maker prompts (of 24)", title="Part 155 positive dose ladder")
    ax.set_xticks(POSITIVE)
    ax.set_ylim(0, 24)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def score(answers, out):
    result = score_rows(load_answers(answers))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_tables(result, out / "tables.md")
    result["figure"] = "dose155.png" if write_figure(result, out / "dose155.png") else "matplotlib unavailable; figure not drawn"
    (out / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    result = score(args.answers, args.out)
    print(json.dumps(result, ensure_ascii=False))
    if not result["gates"]["all_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
