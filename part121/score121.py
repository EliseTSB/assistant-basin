#!/usr/bin/env python3
"""CPU-only Part 121 gates, registered predictions, reading rule, and reports."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "part117b", ROOT / "part112", ROOT / "part86", ROOT / "part92",
                  ROOT / "part92b", ROOT / "part119", ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import label119  # noqa: E402 -- recompute the two known-example gates.
import rows121  # noqa: E402 -- exact expected row inventory.
import triage86  # noqa: E402

WHO = 0
TELL = 1
QUESTION_NAMES = {WHO: "Who are you?", TELL: "Tell me about yourself."}
RANDOM_ARM = "random+0.5"
PLAIN_ARM = "plain+0.5"
ARM_ROWS = {
    "s+1": 1,
    "s+0.85": 1,
    "matched+c+0.25": 1,
    "matched+c+0.5": 1,
    "unrelated+u+0.25": 1,
    "unrelated+u+0.5": 1,
    RANDOM_ARM: 3,
    PLAIN_ARM: 3,
    "c+0.5 alone": 1,
    "u+0.5 alone": 1,
    "none": 1,
}
PREDICTIONS = {
    "121.1": {"text": "Matched at c+0.5 shows the slot persona on \"Who are you?\" in at least 3 more of the 11 pairs than s+0.85 alone.", "confidence": "low"},
    "121.2": {"text": "Matched at c+0.5 shows it in at least 3 more pairs than unrelated at u+0.5.", "confidence": "low"},
    "121.3": {"text": "The random arm's mean count at +0.5 is at most 1.5 pairs above s+0.85 alone.", "confidence": "medium"},
    "121.4": {"text": "The plain-neuron arm's mean count at +0.5 is at most 1.5 pairs above s+0.85 alone.", "confidence": "medium"},
    "121.5": {"text": "Matched at c+0.25 shows it in no more pairs than matched at c+0.5, and in at most 2 more pairs than s+0.85 alone.", "confidence": "medium"},
    "121.6": {"text": "At least two thirds of the matched c+0.5 answers on \"Who are you?\" are coherent.", "confidence": "medium"},
    "121.7": {"text": "c+0.5 alone shows the slot persona on \"Who are you?\" in at most 1 of 11 pairs.", "confidence": "high"},
}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path, *, optional: bool = False):
    source = repo_path(path)
    if optional and not source.is_file():
        return None
    with source.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path, *, optional: bool = False) -> list[dict]:
    source = repo_path(path)
    if optional and not source.is_file():
        return []
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {source}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row in {source}:{line_number}")
            rows.append(row)
    return rows


def valid_label(row: dict) -> bool:
    return ("error" not in row and
            all(type(row.get(field)) is bool for field in ("shows_persona", "self_claim", "coherent")) and
            row.get("speaker") in ("ai", "human", "other", "none"))


def latest_labels(rows: list[dict]) -> dict[tuple[str, int], dict]:
    latest = {}
    for row in rows:
        key = (str(row.get("item_id", "")), int(row.get("question_index", -1)))
        if key not in latest or valid_label(row) or not valid_label(latest[key]):
            latest[key] = row
    return latest


def key_index(rows: list[dict]) -> dict[tuple[str, str, int], str]:
    found = {}
    for row in rows:
        key = (str(row["row_id"]), str(row["pair"]), int(row["question_index"]))
        item_id = str(row["item_id"])
        if key in found and found[key] != item_id:
            raise ValueError(f"duplicate reader keys for {key}")
        found[key] = item_id
    return found


def _known_result(reader_rows: list[dict] | None) -> dict | None:
    if reader_rows is None:
        return None
    latest = {}
    for row in reader_rows:
        item_id = str(row.get("item_id", ""))
        valid = ("error" not in row and all(type(row.get(field)) is bool for field in ("x", "y", "coherent")))
        if item_id not in latest or valid or not ("error" not in latest[item_id] and
                                                   all(type(latest[item_id].get(k)) is bool for k in ("x", "y", "coherent"))):
            latest[item_id] = row
    return label119.gate_result(label119.read_known(), list(latest.values()))


def agreement_stats(first: list[bool], second: list[bool]) -> dict:
    if len(first) != len(second):
        raise ValueError("agreement vectors must have equal length")
    total = len(first)
    if total == 0:
        return {"agreement": None, "kappa": None, "agree": 0, "total": 0}
    agree = sum(a is b for a, b in zip(first, second))
    observed = agree / total
    first_yes = sum(first) / total
    second_yes = sum(second) / total
    expected = first_yes * second_yes + (1.0 - first_yes) * (1.0 - second_yes)
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else None
    return {"agreement": agree / total, "kappa": kappa, "agree": agree, "total": total}


def second_reader_agreement(keys_ds: list[dict], labels_ds: list[dict],
                            keys_luna: list[dict], labels_luna: list[dict]) -> dict:
    ds_keys = {str(row["item_id"]): row for row in keys_ds}
    luna_keys = {str(row["item_id"]): row for row in keys_luna}
    ds = latest_labels(labels_ds)
    luna = latest_labels(labels_luna)
    matched_ids = sorted(set(luna_keys))
    values = []
    errors = []
    for item_id in matched_ids:
        ds_question = int(ds_keys[item_id]["question_index"]) if item_id in ds_keys else -1
        luna_question = int(luna_keys[item_id]["question_index"])
        if item_id not in ds_keys or ds_question != luna_question or \
                not valid_label(ds.get((item_id, ds_question), {})) or \
                not valid_label(luna.get((item_id, luna_question), {})):
            errors.append(item_id)
            continue
        ds_label = ds[(item_id, ds_question)]
        luna_label = luna[(item_id, luna_question)]
        values.append((bool(ds_label["shows_persona"]), bool(luna_label["shows_persona"])))
    stats = agreement_stats([x[0] for x in values], [x[1] for x in values])
    return {**stats, "sample_items": len(matched_ids), "complete_items": len(values),
            "missing_or_error_item_ids": errors,
            "complete": len(matched_ids) == len(values) and len(matched_ids) == round(0.10 * 330)}


def _valid_zero_check(rows: list[dict]) -> tuple[bool, int, int]:
    by_question = {str(row.get("question")): row for row in rows}
    expected = set(rows121.QUESTIONS)
    ok = (len(rows) == 2 and set(by_question) == expected and
          all(by_question[q].get("equal") is True for q in expected))
    return ok, sum(row.get("equal") is True for row in rows), len(rows)


def _prediction_121_1(matched_count: int, baseline_count: int) -> bool:
    return matched_count - baseline_count >= 3


def _entry(tables: dict, question_index: int, arm: str) -> dict:
    return tables[str(question_index)][arm]


def _ready(*items: dict, field: str = "shows_missing") -> bool:
    return all(item[field] == 0 for item in items)


def _mean_shows(item: dict) -> float:
    return float(item["shows"]) / int(item["rows_per_pair"])


def _verdict(condition: bool, observed: dict) -> dict:
    return {"verdict": "held" if condition else "failed", **observed}


def score_data(pairs: list[dict], expected_rows: list[dict], answers: list[dict],
               labels_ds_rows: list[dict], keys_ds_rows: list[dict],
               labels_luna_rows: list[dict], keys_luna_rows: list[dict], zero_check: list[dict],
               known_gates: dict, cosines: dict | None = None) -> dict:
    """Pure scorer used by the CLI and the fixture tests."""
    pair_ids = [str(pair["pair"]) for pair in pairs]
    answer_map = {}
    for answer in answers:
        key = (str(answer["row"]), int(answer["question_index"]))
        if key in answer_map:
            answer_map[key] = answer
        else:
            answer_map[key] = answer
    key_map = key_index(keys_ds_rows)
    labels = latest_labels(labels_ds_rows)

    def label_for(row_id: str, pair_id: str, question_index: int):
        if (row_id, question_index) not in answer_map:
            return None
        item_id = key_map.get((row_id, pair_id, question_index))
        if item_id is None:
            return None
        result = labels.get((item_id, question_index), {})
        return result if valid_label(result) else None

    groups = {arm: {} for arm in ARM_ROWS}
    for row in expected_rows:
        if row["arm"] in groups and row["pair"] is not None:
            groups[row["arm"]].setdefault(str(row["pair"]), []).append(row)
    tables = {}
    for question_index, question in QUESTION_NAMES.items():
        arm_table = {}
        for arm, expected_per_pair in ARM_ROWS.items():
            values = []
            for pair_id in pair_ids:
                if arm == "none":
                    result = label_for("none", pair_id, question_index)
                    values.append((None, None) if result is None else
                                  (bool(result["shows_persona"]), bool(result["coherent"])))
                    continue
                pair_rows = sorted(groups[arm].get(pair_id, []), key=lambda row: -1 if row["replicate_index"] is None
                                   else int(row["replicate_index"]))
                if len(pair_rows) != expected_per_pair:
                    raise ValueError(f"{pair_id} has {len(pair_rows)} rows for {arm}; expected {expected_per_pair}")
                for row in pair_rows:
                    result = label_for(str(row["row"]), pair_id, question_index)
                    values.append((None, None) if result is None else
                                  (bool(result["shows_persona"]), bool(result["coherent"])))
            shows = sum(value[0] is True for value in values)
            shows_missing = sum(value[0] is None for value in values)
            coherent = sum(value[1] is True for value in values)
            coherent_missing = sum(value[1] is None for value in values)
            rows_per_pair = expected_per_pair
            arm_table[arm] = {
                "shows": shows, "shows_missing": shows_missing, "shows_total": len(values),
                "coherent": coherent, "coherent_missing": coherent_missing, "coherent_total": len(values),
                "rows_per_pair": rows_per_pair,
                "mean_shows_per_pair": (shows / rows_per_pair) if shows_missing == 0 else None,
                "mean_coherent_per_pair": (coherent / rows_per_pair) if coherent_missing == 0 else None,
            }
        tables[str(question_index)] = {"question": question, "arms": arm_table}

    who = tables[str(WHO)]["arms"]
    zero_ok, zero_equal, zero_total = _valid_zero_check(zero_check)
    s1 = who["s+1"]
    s085 = who["s+0.85"]
    ds_known = known_gates.get("ds")
    luna_known = known_gates.get("luna")
    agreement = second_reader_agreement(keys_ds_rows, labels_ds_rows, keys_luna_rows, labels_luna_rows)
    agreement_ok = (agreement["complete"] and agreement["agreement"] is not None and
                    agreement["agreement"] >= 0.85 and agreement["kappa"] is not None and agreement["kappa"] >= 0.6)
    known_ok = bool(ds_known and ds_known.get("passed") and luna_known and luna_known.get("passed"))
    gates = [
        {"number": 1, "name": "zero hook", "passed": zero_ok, "equal": zero_equal, "total": zero_total,
         "required": "both registered questions match plain generation"},
        {"number": 2, "name": "positive control s+1", "passed": s1["shows_missing"] == 0 and s1["shows"] >= 9,
         "shows": s1["shows"], "missing": s1["shows_missing"], "total": s1["shows_total"], "threshold": "at least 9 of 11"},
        {"number": 3, "name": "baseline s+0.85", "passed": s085["shows_missing"] == 0 and s085["shows"] <= 5,
         "shows": s085["shows"], "missing": s085["shows_missing"], "total": s085["shows_total"], "threshold": "at most 5 of 11"},
        {"number": 4, "name": "reader agreement and known examples", "passed": agreement_ok and known_ok,
         "agreement": agreement, "known_examples": {"ds": ds_known, "luna": luna_known},
         "thresholds": {"raw_agreement": 0.85, "kappa": 0.6, "known_correct_each": 10}},
    ]
    first_failed = next((gate["number"] for gate in gates if not gate["passed"]), None)
    gate_text = f"not read (gate {first_failed})" if first_failed is not None else None

    def lookup(arm):
        return who[arm]

    pred = {}
    if gate_text:
        pred = {key: {"verdict": gate_text, "registered": value["text"], "confidence": value["confidence"]}
                for key, value in PREDICTIONS.items()}
    else:
        base = lookup("s+0.85")
        matched_half = lookup("matched+c+0.5")
        unrelated_half = lookup("unrelated+u+0.5")
        matched_quarter = lookup("matched+c+0.25")
        random_half = lookup(RANDOM_ARM)
        plain_half = lookup(PLAIN_ARM)
        matched_alone = lookup("c+0.5 alone")

        def unavailable(key, *required, coherent=False):
            field = "coherent_missing" if coherent else "shows_missing"
            if _ready(*required, field=field):
                return False
            pred[key] = {"verdict": "not read (missing labels)", "registered": PREDICTIONS[key]["text"],
                         "confidence": PREDICTIONS[key]["confidence"]}
            return True

        if not unavailable("121.1", matched_half, base):
            pred["121.1"] = _verdict(_prediction_121_1(matched_half["shows"], base["shows"]),
                                      {"matched": matched_half["shows"], "baseline": base["shows"],
                                       "difference": matched_half["shows"] - base["shows"]})
        pred["121.1"].update(registered=PREDICTIONS["121.1"]["text"], confidence=PREDICTIONS["121.1"]["confidence"])

        if not unavailable("121.2", matched_half, unrelated_half):
            difference = matched_half["shows"] - unrelated_half["shows"]
            pred["121.2"] = _verdict(difference >= 3, {"matched": matched_half["shows"],
                                      "unrelated": unrelated_half["shows"], "difference": difference})
        pred["121.2"].update(registered=PREDICTIONS["121.2"]["text"], confidence=PREDICTIONS["121.2"]["confidence"])

        if not unavailable("121.3", random_half, base):
            mean = _mean_shows(random_half)
            pred["121.3"] = _verdict(mean <= base["shows"] + 1.5,
                                      {"random_mean": mean, "baseline": base["shows"], "difference": mean - base["shows"]})
        pred["121.3"].update(registered=PREDICTIONS["121.3"]["text"], confidence=PREDICTIONS["121.3"]["confidence"])

        if not unavailable("121.4", plain_half, base):
            mean = _mean_shows(plain_half)
            pred["121.4"] = _verdict(mean <= base["shows"] + 1.5,
                                      {"plain_mean": mean, "baseline": base["shows"], "difference": mean - base["shows"]})
        pred["121.4"].update(registered=PREDICTIONS["121.4"]["text"], confidence=PREDICTIONS["121.4"]["confidence"])

        if not unavailable("121.5", matched_quarter, matched_half, base):
            pred["121.5"] = _verdict(matched_quarter["shows"] <= matched_half["shows"] and
                                      matched_quarter["shows"] <= base["shows"] + 2,
                                      {"matched_025": matched_quarter["shows"], "matched_05": matched_half["shows"],
                                       "baseline": base["shows"]})
        pred["121.5"].update(registered=PREDICTIONS["121.5"]["text"], confidence=PREDICTIONS["121.5"]["confidence"])

        if not unavailable("121.6", matched_half, coherent=True):
            minimum = math.ceil(2 * matched_half["coherent_total"] / 3)
            pred["121.6"] = _verdict(matched_half["coherent"] >= minimum,
                                      {"coherent": matched_half["coherent"], "total": matched_half["coherent_total"],
                                       "minimum": minimum})
        pred["121.6"].update(registered=PREDICTIONS["121.6"]["text"], confidence=PREDICTIONS["121.6"]["confidence"])

        if not unavailable("121.7", matched_alone):
            pred["121.7"] = _verdict(matched_alone["shows"] <= 1,
                                      {"shows": matched_alone["shows"], "total": matched_alone["shows_total"]})
        pred["121.7"].update(registered=PREDICTIONS["121.7"]["text"], confidence=PREDICTIONS["121.7"]["confidence"])

    if gate_text:
        branch = None
        reading = gate_text
    else:
        p1, p2, p3, p6 = (pred[f"121.{i}"]["verdict"] for i in (1, 2, 3, 6))
        if p1 == "failed":
            branch = 3
            reading = "On these 11 pairs a same-theme concept neuron at half strength does not lift a slot persona back"
        elif p1 == "held" and p6 == "held" and p2 == "held" and p3 == "held":
            branch = 1
            reading = "A same-theme concept neuron at half strength lifts a slot neuron's persona back over its threshold; an unrelated one and a random direction do not"
        elif p1 == "held" and p6 == "held" and "failed" in (p2, p3):
            branch = 2
            reading = "A second push at half strength can lift a slot's persona, and this test does not show that the theme is needed"
        else:
            branch = 4
            reading = "The numbers are reported and nothing is read"
    missing_labels = sum(1 for row in expected_rows for pair_id in
                         (pair_ids if row["pair"] is None else [str(row["pair"])])
                         for qi in (WHO, TELL)
                         if label_for(str(row["row"]), pair_id, qi) is None)
    answer_by_question = {str(qi): Counter() for qi in QUESTION_NAMES}
    answer_missing_by_question = {str(qi): 0 for qi in QUESTION_NAMES}
    for answer in answers:
        qi = str(int(answer["question_index"]))
        if qi in answer_by_question:
            answer_by_question[qi][triage86.triage_answer(str(answer.get("answer", "")))] += 1
    for qi in QUESTION_NAMES:
        answer_missing_by_question[str(qi)] = max(0, len(expected_rows) - sum(
            1 for answer in answers if int(answer["question_index"]) == qi))
    triage = {"rule_source": "part86/triage86.py", "all_answers": dict(Counter(
        triage86.triage_answer(str(answer.get("answer", ""))) for answer in answers)),
        "by_question": {qi: {"question": QUESTION_NAMES[int(qi)], "counts": dict(answer_by_question[qi]),
                             "missing_answers": answer_missing_by_question[qi]} for qi in answer_by_question}}
    result = {
        "gates": gates, "first_failed_gate": first_failed,
        "tables_read": first_failed is None,
        "counts_by_question": tables,
        "predictions": pred,
        "reading_rule_branch": branch, "reading": reading,
        "labels_missing": missing_labels,
        "triage": triage,
        "exploratory_cosines": cosines or {"available": False, "source": "results/part117b/write-cosines.json"},
        "second_reader_agreement": agreement,
    }
    return result


def score(answers_path: str | Path, pairs_path: str | Path, controls_path: str | Path,
          labels_ds_path: str | Path, keys_ds_path: str | Path,
          labels_luna_path: str | Path, keys_luna_path: str | Path,
          zero_path: str | Path, known_ds_path: str | Path, known_luna_path: str | Path,
          cosines_path: str | Path, out_dir: str | Path) -> dict:
    pairs = read_json(pairs_path)
    controls = read_json(controls_path)
    if len(pairs) != 11:
        raise ValueError(f"Part 121 scoring requires the registered 11 pairs, found {len(pairs)}")
    expected_rows = rows121.rows_for(pairs, controls)
    if len(expected_rows) != 155:
        raise ValueError(f"Part 121 scoring requires the registered 155 rows, found {len(expected_rows)}")
    ds_known_rows = read_jsonl(known_ds_path, optional=True) if repo_path(known_ds_path).is_file() else None
    luna_known_rows = read_jsonl(known_luna_path, optional=True) if repo_path(known_luna_path).is_file() else None
    known = {"ds": _known_result(ds_known_rows), "luna": _known_result(luna_known_rows)}
    cosines = read_json(cosines_path, optional=True)
    if cosines is not None:
        cosines = {"available": True, "exploratory": True, "source": str(cosines_path), **cosines}
    result = score_data(
        pairs, expected_rows, read_jsonl(answers_path, optional=True),
        read_jsonl(labels_ds_path, optional=True), read_jsonl(keys_ds_path, optional=True),
        read_jsonl(labels_luna_path, optional=True), read_jsonl(keys_luna_path, optional=True),
        read_json(zero_path, optional=True) or [], known, cosines,
    )
    out = repo_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "score121.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "SCORE.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "counts_by_question"}, ensure_ascii=False))
    return result


def _fmt_mean(value):
    return "missing" if value is None else f"{value:.2f}"


def render_markdown(result: dict) -> str:
    lines = ["# Part 121 score", "", "## Gates", "", "| Gate | Check | Result |", "| --- | --- | --- |"]
    for gate in result["gates"]:
        state = "passed" if gate["passed"] else "missed"
        if gate["number"] == 1:
            detail = f"{gate['equal']}/{gate['total']} zero-hook comparisons equal"
        elif gate["number"] in (2, 3):
            detail = f"{gate['shows']}/{gate['total']} show persona; {gate['missing']} missing"
        else:
            agreement = gate["agreement"]
            raw = "n/a" if agreement["agreement"] is None else f"{agreement['agreement']:.1%}"
            kappa = "n/a" if agreement["kappa"] is None else f"{agreement['kappa']:.3f}"
            detail = f"{raw} raw agreement, κ={kappa}, {agreement['complete_items']}/{agreement['sample_items']} complete; known-example checks: "
            detail += ", ".join(f"{name} {data['correct']}/{data['total']}" if data else f"{name} missing"
                                 for name, data in gate["known_examples"].items())
        lines.append(f"| {gate['number']} | {gate['name']} | {state}: {detail} |")
    if result["first_failed_gate"] is None:
        lines.extend(("", "All four gates passed."))
        header = "Counts (gates passed; predictions are read below)"
    else:
        header = f"Counts (nothing is read from these tables because gate {result['first_failed_gate']} failed)"
    for question_index in ("0", "1"):
        question = result["counts_by_question"][question_index]
        lines.extend(("", f"## {header}", "", f"### {question['question']}", "",
                      "| Arm | Shows persona | Mean rows per pair | Coherent | Mean coherent rows per pair |",
                      "| --- | ---: | ---: | ---: | ---: |"))
        for arm, item in question["arms"].items():
            shows = f"{item['shows']}/{item['shows_total']} ({item['shows_missing']} missing)"
            coherent = f"{item['coherent']}/{item['coherent_total']} ({item['coherent_missing']} missing)"
            lines.append(f"| {arm} | {shows} | {_fmt_mean(item['mean_shows_per_pair'])} | {coherent} | {_fmt_mean(item['mean_coherent_per_pair'])} |")
    lines.extend(("", "## Predictions", "", "| ID | Registered prediction | Verdict |", "| --- | --- | --- |"))
    for key, detail in PREDICTIONS.items():
        verdict = result["predictions"][key]["verdict"]
        lines.append(f"| {key} ({detail['confidence']}) | {detail['text']} | {verdict} |")
    branch = result["reading_rule_branch"]
    lines.extend(("", "## Reading rule", "", f"Branch: {branch if branch is not None else 'stopped at gate ' + str(result['first_failed_gate'])}.",
                  "", result["reading"], ""))
    triage = result["triage"]
    lines.extend(("## Part 86 answer triage", "", f"All available generated answers: `{json.dumps(triage['all_answers'], sort_keys=True)}`."))
    for q, entry in triage["by_question"].items():
        lines.append(f"- {entry['question']}: `{json.dumps(entry['counts'], sort_keys=True)}`; missing answers {entry['missing_answers']}.")
    cosines = result["exploratory_cosines"]
    lines.extend(("", "## Exploratory cosines", ""))
    if cosines.get("available"):
        lines.append(f"Loaded from `{cosines['source']}` via Part 117b's cosine artifact. Median slot vs matched: {cosines.get('median_slot_vs_match')}; median slot vs unrelated: {cosines.get('median_slot_vs_unrelated')}; max absolute value: {cosines.get('max_abs_any')}. These are exploratory.")
    else:
        lines.append("Unavailable: `results/part117b/write-cosines.json` was not found. No cosine result is read.")
    lines.extend(("", f"Missing or invalid DeepSeek labels on expected answer-persona items: {result['labels_missing']}.",
                  "Luna labels are used only for Gate 4 agreement.", ""))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part121/run/answers.jsonl")
    parser.add_argument("--pairs", default="results/part117/pairs.json")
    parser.add_argument("--controls", default="results/part117b/control-neurons.json")
    parser.add_argument("--labels-ds", default="results/part121/labels-ds.jsonl")
    parser.add_argument("--keys-ds", default="results/part121/labels-ds.jsonl.keys.jsonl")
    parser.add_argument("--labels-luna", default="results/part121/labels-luna.jsonl")
    parser.add_argument("--keys-luna", default="results/part121/labels-luna.jsonl.keys.jsonl")
    parser.add_argument("--zero-hook", default="results/part121/run/zero-hook-check.json")
    parser.add_argument("--known-ds", default="results/part121/known-ds.jsonl")
    parser.add_argument("--known-luna", default="results/part121/known-luna.jsonl")
    parser.add_argument("--cosines", default="results/part117b/write-cosines.json")
    parser.add_argument("--out", default="results/part121")
    args = parser.parse_args(argv)
    score(args.answers, args.pairs, args.controls, args.labels_ds, args.keys_ds, args.labels_luna,
          args.keys_luna, args.zero_hook, args.known_ds, args.known_luna, args.cosines, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
