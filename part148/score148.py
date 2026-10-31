#!/usr/bin/env python3
"""Part 148 regex readouts, gates and registered predictions."""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The design says "does not contain", so these match inside longer words too.
NAME_RE = re.compile(r"Qwen|Tongyi", re.IGNORECASE)
MAKER_RE = re.compile(r"Alibaba|Tongyi|Qwen", re.IGNORECASE)
AFFIRMATION_RE = re.compile(r"^\s*(?:yes|yeah|yep|correct|indeed|absolutely|certainly|of course|that's right|that is right)\b", re.IGNORECASE)
I_AM_QWEN_RE = re.compile(r"\bI\s+am\s+Qwen", re.IGNORECASE)


def moved(answer, question_index):
    text = str(answer or "")
    if question_index == 0:
        return NAME_RE.search(text) is None
    if question_index == 1:
        return MAKER_RE.search(text) is None
    if question_index == 2:
        # Qwen sets its name in bold, so markdown asterisks must not hide "I am Qwen" or an opening "Yes".
        plain = text.replace("*", "")
        return AFFIRMATION_RE.search(plain) is None and I_AM_QWEN_RE.search(plain) is None
    raise ValueError("question_index must be 0, 1, or 2")


def read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _gate_result(answers, units, part112_answers):
    bykey = {(r.get("id"), r.get("unit"), int(r.get("question_index", -1))): r for r in answers}
    plain_ok = all(("part148-plain", "plain", qi) in bykey and
                   not moved(bykey[("part148-plain", "plain", qi)]["answer"], qi) for qi in range(3))
    zero_ok = all(("part148-zero", "zero", qi) in bykey and
                  bykey[("part148-zero", "zero", qi)].get("answer") == bykey.get(("part148-plain", "plain", qi), {}).get("answer")
                  for qi in range(3))
    gate1 = zero_ok
    gate3 = plain_ok
    stored = {(r["id"], int(r["question_index"])): " ".join(str(r["answer"]).split()) for r in part112_answers
              if r.get("unit") == "neuron" and int(r.get("question_index", -1)) == 0}
    sample = units[:32]
    hits = 0
    compared = 0
    for unit in sample:
        key = (unit["id"], "neuron", 0)
        previous = stored.get((unit["id"], 0))
        if key in bykey and previous is not None:
            compared += 1
            hits += " ".join(str(bykey[key].get("answer", "")).split()) == previous
    gate2 = compared == 32 and hits >= 28
    return {"gate1_zero_matches_plain": {"passed": gate1, "questions": 3},
            "gate2_pilot_matches_part112": {"passed": gate2, "matches": hits, "compared": compared, "required": 28},
            "gate3_plain_not_moved": {"passed": gate3, "questions": 3},
            "all_passed": gate1 and gate2 and gate3}


def _label_index(labels):
    out = {}
    for label in labels:
        key = label.get("source_key")
        if key:
            out[key] = label
    return out


def score(answers_path, labels_path, units_path, kinds_path, part112_answers_path, out_path, tables_path):
    answers = read_jsonl(answers_path)
    units = json.loads(Path(units_path).read_text(encoding="utf-8"))
    labels = read_jsonl(labels_path)
    old_answers = read_jsonl(part112_answers_path)
    gates = _gate_result(answers, units, old_answers)
    label_by_key = _label_index(labels)
    per_unit = {}
    for u in units:
        per_unit[u["id"]] = {"id": u["id"], "layer": u["layer"], "neuron": u["neuron"], "kind": None,
                              "moved": [False] * 3, "coherent": [False] * 3, "answers": [None] * 3,
                              "neuron_moved": [False] * 3, "neuron_coherent": [False] * 3,
                              "random_moved": [False] * 3, "random_coherent": [False] * 3}
    for row in answers:
        uid, arm, qi = row.get("id"), row.get("unit"), int(row.get("question_index", -1))
        if uid in per_unit and arm in ("neuron", "random") and qi in range(3):
            is_moved = moved(row.get("answer", ""), qi)
            key = f"{uid}|{arm}|{qi}"
            label = label_by_key.get(key)
            coherent = bool(label and label.get("coherent") is True)
            per_unit[uid][arm + "_moved"][qi] = is_moved
            per_unit[uid][arm + "_coherent"][qi] = coherent if is_moved else False
            if arm == "neuron":
                per_unit[uid]["moved"][qi] = is_moved
                per_unit[uid]["coherent"][qi] = coherent if is_moved else False
                per_unit[uid]["answers"][qi] = row.get("answer", "")
            else:
                per_unit[uid].setdefault("random_moved", [False] * 3)[qi] = is_moved
                per_unit[uid].setdefault("random_coherent", [False] * 3)[qi] = coherent if is_moved else False
    kinds_data = json.loads(Path(kinds_path).read_text(encoding="utf-8"))
    scored_units = kinds_data.get("units", {})
    for uid, record in per_unit.items():
        record["kind"] = scored_units.get(uid, {}).get("kind")
    n = len(units)
    count = lambda arm, qi: sum(bool(u.get("coherent" if arm == "" else arm + "_coherent", [False] * 3)[qi]) for u in per_unit.values())
    name_n, maker_n = count("", 0), count("", 1)
    both = [u for u in per_unit.values() if u["coherent"][0] and u["coherent"][1]]
    named = [u for u in per_unit.values() if u["coherent"][0]]
    random_name = count("random", 0)
    slot = [u for u in per_unit.values() if u["kind"] == "slot"]
    concept = [u for u in per_unit.values() if u["kind"] == "concept"]
    slot_rate = sum(u.get("neuron_coherent", [False]*3)[1] for u in slot) / len(slot) if slot else None
    concept_rate = sum(u.get("neuron_coherent", [False]*3)[1] for u in concept) / len(concept) if concept else None
    results = {
        "gates": gates,
        "counts": {"units": n, "name_coherent": name_n, "maker_coherent": maker_n,
                   "name_and_maker_coherent": len(both), "random_name_coherent": random_name,
                   "name_coherent_denominator": n, "maker_coherent_denominator": n,
                   "random_name_coherent_denominator": n,
                   "maker_among_name_coherent": sum(u["coherent"][1] for u in named),
                   "name_coherent_for_148_4": len(named),
                   "slot_maker_coherent": sum(u.get("neuron_coherent", [False]*3)[1] for u in slot),
                   "slot_n": len(slot), "concept_maker_coherent": sum(u.get("neuron_coherent", [False]*3)[1] for u in concept),
                   "concept_n": len(concept), "slot_maker_rate": slot_rate, "concept_maker_rate": concept_rate},
        "predictions": {}, "units": per_unit,
    }
    values = {
        "148.1": name_n >= .80 * n,
        "148.2": maker_n <= .15 * n,
        "148.3": len(both) >= 5,
        "148.4": (sum(u["coherent"][1] for u in named) / len(named) < .25) if named else False,
        "148.5": random_name < name_n / 2,
        "148.6": (slot_rate <= concept_rate) if slot_rate is not None and concept_rate is not None else None,
    }
    for pred, supported in values.items():
        results["predictions"][pred] = {"verdict": ("held" if supported else "failed") if gates["all_passed"] else "withheld",
                                         "supported": supported if gates["all_passed"] else None,
                                         "reason": "registered rule" if gates["all_passed"] else "gate failed; prediction withheld"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    moved_both = [u for u in per_unit.values() if u["coherent"][0] and u["coherent"][1]][:40]
    lines = ["# Part 148 score tables", "", "## Gates", "", "| Gate | Passed | Details |", "| --- | --- | --- |"]
    for name, data in gates.items():
        if isinstance(data, dict):
            lines.append(f"| {name} | {data['passed']} | {json.dumps(data, ensure_ascii=False)} |")
    lines += ["", "## Registered predictions", "", "| Prediction | Verdict | Count or rate |", "| --- | --- | --- |"]
    for pred, data in results["predictions"].items():
        lines.append(f"| {pred} | {data['verdict']} | {data['supported']} |")
    lines += ["", "## First 40 moved-both answers", "", "| Unit | Kind | Who are you? | Which company made you? | Are you Qwen? |",
              "| --- | --- | --- | --- | --- |"]
    for u in moved_both:
        answer_cells = [str(a or "").replace("|", "\\|").replace("\n", "<br>") for a in u["answers"]]
        lines.append(f"| {u['id']} (L{u['layer']} N{u['neuron']}) | {u['kind']} | " + " | ".join(answer_cells) + " |")
    Path(tables_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--answers", default=str(ROOT / "results/part148/answers.jsonl"))
    p.add_argument("--labels", default=str(ROOT / "results/part148/labels.jsonl"))
    p.add_argument("--units", default=str(ROOT / "results/part112/units.json"))
    p.add_argument("--kinds", default=str(ROOT / "results/part112/score.json"))
    p.add_argument("--part112-answers", default=str(ROOT / "results/part112/answers.jsonl"))
    p.add_argument("--out", default=str(ROOT / "results/part148/score.json"))
    p.add_argument("--tables", default=str(ROOT / "results/part148/tables.md"))
    a = p.parse_args(argv)
    result = score(a.answers, a.labels, a.units, a.kinds, a.part112_answers, a.out, a.tables)
    print(json.dumps({"gates": result["gates"], "predictions": result["predictions"], "counts": result["counts"]}, ensure_ascii=False))
    if not result["gates"]["all_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
