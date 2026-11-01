#!/usr/bin/env python3
"""Host-side score and registered gates for Part 163."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART86 = ROOT / "part86"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PART86) not in sys.path:
    sys.path.insert(0, str(PART86))
from part153.battery153 import UNIT_COMPANIES  # noqa: E402
import triage86  # noqa: E402

NAME_TOKENS = ("qwen", "tongyi", "通义", "千问")
MAKER_TOKENS = ("alibaba", "阿里")
COMPANIES = sorted({name for _, _, name in UNIT_COMPANIES})
FIRST_NAME_PATTERNS = (
    re.compile(r"\b(?:i\s+am|i['’]m|my\s+name\s+is|this\s+is|call\s+me)\s+(?:(?:an?|the)\s+)?([^,.;:!?()\n]{1,60})", re.I),
)


def read_jsonl(path):
    rows = []
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{no}") from exc
    return rows


def name_gone(answer: str) -> bool:
    text = answer.casefold() if isinstance(answer, str) else ""
    return not any(token in text for token in NAME_TOKENS)


def maker_gone(answer: str) -> bool:
    text = answer.casefold() if isinstance(answer, str) else ""
    return not any(token in text for token in MAKER_TOKENS)


def triage(answer: str) -> str:
    return triage86.triage_answer(answer if isinstance(answer, str) else "")


def pct(values, p):
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    pos = (len(vals) - 1) * p
    low = int(math.floor(pos)); high = int(math.ceil(pos))
    if low == high:
        return vals[low]
    return vals[low] * (high - pos) + vals[high] * (pos - low)


def _mean(values):
    return sum(values) / len(values) if values else None


def bootstrap(values_by_pair, seed=163, reps=10_000):
    """Unit/pair bootstrap for an arm mean; values_by_pair maps pair ID to a rate."""
    ids = sorted(values_by_pair)
    values = [values_by_pair[key] for key in ids if values_by_pair[key] is not None]
    if not values:
        return {"mean": None, "interval_95": [None, None], "n_units": 0}
    mean = _mean(values)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        draws.append(_mean(sample))
    return {"mean": mean, "interval_95": [pct(draws, 0.025), pct(draws, 0.975)],
            "n_units": len(values)}


def paired_bootstrap(neuron_by_pair, random_by_pair, seed=163, reps=10_000):
    ids = sorted(set(neuron_by_pair) & set(random_by_pair))
    valid = [key for key in ids if neuron_by_pair[key] is not None and random_by_pair[key] is not None]
    nvals = [neuron_by_pair[k] for k in valid]
    rvals = [random_by_pair[k] for k in valid]
    nmean, rmean = _mean(nvals), _mean(rvals)
    ratio = None if rmean in (None, 0) else nmean / rmean
    if not valid:
        return {"neuron_mean": nmean, "random_mean": rmean, "ratio": ratio,
                "ratio_interval_95": [None, None], "n_pairs": 0}
    rng = random.Random(seed)
    ratios = []
    for _ in range(reps):
        sample = [valid[rng.randrange(len(valid))] for _ in valid]
        nm = _mean([neuron_by_pair[k] for k in sample])
        rm = _mean([random_by_pair[k] for k in sample])
        if rm is not None and rm > 0:
            ratios.append(nm / rm)
    return {"neuron_mean": nmean, "random_mean": rmean, "ratio": ratio,
            "ratio_interval_95": [pct(ratios, 0.025), pct(ratios, 0.975)],
            "n_pairs": len(valid), "bootstrap_replicates": reps, "bootstrap_seed": seed}


def _expected_questions():
    from part155 import run155 as part155
    import random as _random
    all_items = part155.questions()
    identity = [q for q in all_items if q["kind"] == "identity"]
    maker = [q for q in all_items if q["kind"] == "maker"]
    return identity + _random.Random(163).sample(maker, 8)


def _clean_name(raw):
    value = re.sub(r"\*|`|_|\[[^]]*\]", "", raw).strip(" \t\r\n-–—:'\"“”")
    value = re.sub(r"^(?:an?|the)\s+", "", value, flags=re.I)
    value = re.split(r"\s+(?:a|an|the|and|or|from|to|who|which|that|designed|developed|created|built|here)\b", value, 1, flags=re.I)[0]
    value = value.strip(" \t\r\n-–—:'\"“”")
    if not value or len(value) > 35 or len(value.split()) > 5:
        return None
    return value


def first_sentence_name(answer):
    first = re.split(r"(?<=[.!?])\s|\n", answer.strip(), maxsplit=1)[0]
    for pattern in FIRST_NAME_PATTERNS:
        found = pattern.search(first)
        if found:
            return _clean_name(found.group(1))
    return None


def score_rows(rows, units):
    questions = _expected_questions()
    units_by_arm = {arm: [u for u in units if u["arm"] == arm] for arm in ("neuron", "random")}
    pair_ids = {arm: {u["unit_id"]: u["pair_id"] for u in us} for arm, us in units_by_arm.items()}
    expected = {("no_push", "no_push", qi) for qi in range(16)}
    expected |= {("recipe", u["unit_id"], 0) for u in units}
    expected |= {("battery", u["unit_id"], qi) for u in units for qi in range(16)}
    indexed = {}
    duplicates = 0
    for row in rows:
        key = (row.get("stage"), row.get("unit_id"), int(row.get("question_index", -1)))
        if key in indexed:
            duplicates += 1
        indexed[key] = row
    nonempty = sum(isinstance(r.get("answer"), str) and bool(r["answer"].strip()) for r in indexed.values())
    units_ok = all(80 <= len(units_by_arm[a]) <= 120 for a in ("neuron", "random")) and len(units_by_arm["neuron"]) == len(units_by_arm["random"])
    complete = len(rows) == len(expected) and expected == set(indexed) and duplicates == 0 and nonempty == len(expected)
    gate3 = {"passed": bool(complete and units_ok), "expected_rows": len(expected), "rows": len(rows),
             "unique_rows": len(indexed), "nonempty": nonempty, "duplicates": duplicates,
             "units_per_arm": {a: len(units_by_arm[a]) for a in units_by_arm}}

    recipe = {a: [indexed.get(("recipe", u["unit_id"], 0), {}).get("answer", "") for u in units_by_arm[a]]
              for a in ("neuron", "random")}
    recipe_rates = {a: sum(triage(x) == "changed" for x in ans) / max(1, len(ans))
                    for a, ans in recipe.items()}
    gate1 = {"passed": all(len(recipe[a]) == len(units_by_arm[a]) and recipe_rates[a] >= 0.70
                            for a in ("neuron", "random")), "minimum_rate": 0.70,
             "changed_rate_by_arm": recipe_rates, "n_by_arm": {a: len(recipe[a]) for a in recipe}}
    baseline = [indexed.get(("no_push", "no_push", qi), {}).get("answer", "") for qi in range(16)]
    baseline = [answer if isinstance(answer, str) else "" for answer in baseline]
    identity_q = [i for i, q in enumerate(questions) if q["kind"] == "identity"]
    maker_q = [i for i, q in enumerate(questions) if q["kind"] == "maker"]
    qwen_hits = sum("qwen" in baseline[i].casefold() for i in identity_q)
    maker_hits = sum(any(t in baseline[i].casefold() for t in ("alibaba", "阿里", "tongyi", "通义")) for i in maker_q)
    gate2 = {"passed": len(baseline) == 16 and qwen_hits >= 7 and maker_hits >= 7,
             "identity_qwen_hits": qwen_hits, "identity_total": len(identity_q),
             "maker_alibaba_or_tongyi_hits": maker_hits, "maker_total": len(maker_q), "minimum_each": 7}
    gates = {"gate1_recipe": gate1, "gate2_no_push": gate2, "gate3_rows": gate3}
    all_gates = all(g["passed"] for g in gates.values())

    records = defaultdict(lambda: defaultdict(list))
    for unit in units:
        for qi, question in enumerate(questions):
            row = indexed.get(("battery", unit["unit_id"], qi), {})
            answer = row.get("answer", "")
            answer = answer if isinstance(answer, str) else ""
            records[unit["arm"]][unit["unit_id"]].append({"pair_id": unit["pair_id"], "layer": int(unit["layer"]),
                "question_index": qi, "kind": question["kind"], "question": question["question"],
                "answer": answer, "broken": triage(answer) == "broken"})
    unit_rates = {}
    metrics = ("name_gone_identity", "maker_gone_maker", "both_gone_identity",
               "name_gone_maker", "maker_gone_identity")
    for arm in ("neuron", "random"):
        unit_rates[arm] = {}
        for uid, cells in records[arm].items():
            unit_rates[arm][uid] = {"pair_id": cells[0]["pair_id"], "layer": cells[0]["layer"]}
            for metric in metrics:
                kind = "identity" if metric.endswith("identity") else "maker"
                eligible = [c for c in cells if c["kind"] == kind and not c["broken"]]
                pred = metric.split("_")[0]
                if metric == "both_gone_identity":
                    hits = sum(name_gone(c["answer"]) and maker_gone(c["answer"]) for c in eligible)
                elif pred == "name":
                    hits = sum(name_gone(c["answer"]) for c in eligible)
                else:
                    hits = sum(maker_gone(c["answer"]) for c in eligible)
                unit_rates[arm][uid][metric] = hits / len(eligible) if eligible else None
            unit_rates[arm][uid]["broken_rate"] = sum(c["broken"] for c in cells) / max(1, len(cells))
            unit_rates[arm][uid]["battery_n"] = len(cells)
            unit_rates[arm][uid]["nonbroken_n"] = sum(not c["broken"] for c in cells)

    def map_rates(metric, arm):
        return {unit_rates[arm][uid]["pair_id"]: unit_rates[arm][uid][metric] for uid in unit_rates[arm]}

    rates = {}
    for metric in (*metrics, "broken_rate"):
        rates[metric] = {arm: bootstrap(map_rates(metric, arm), seed=163 + (0 if arm == "neuron" else 1))
                         for arm in ("neuron", "random")}
    ratios = {"163.1_name_gone_identity": paired_bootstrap(map_rates("name_gone_identity", "neuron"),
              map_rates("name_gone_identity", "random"), seed=163),
              "163.2_maker_gone_maker": paired_bootstrap(map_rates("maker_gone_maker", "neuron"),
              map_rates("maker_gone_maker", "random"), seed=163)}

    identity_name_n = {a: sum((unit_rates[a][u]["name_gone_identity"] or 0) >= 0.5 for u in unit_rates[a]) for a in unit_rates}
    identity_both_n = {a: sum((unit_rates[a][u]["both_gone_identity"] or 0) >= 0.5 for u in unit_rates[a]) for a in unit_rates}
    predictions = {}
    for key, metric_key in (("163.1", "163.1_name_gone_identity"), ("163.2", "163.2_maker_gone_maker")):
        if not all_gates:
            verdict = "untestable"
        else:
            # Registered as a point statement, like every other part's rate prediction.
            estimate = ratios[metric_key]["ratio"]
            verdict = "untestable" if estimate is None else ("held" if estimate < 1.5 else "failed")
        wording = ("On identity phrasings, the neuron arm's name-gone rate is under 1.5 times the random arm's"
                   if key == "163.1" else
                   "On maker phrasings, the neuron arm's maker-gone rate is under 1.5 times the random arm's")
        predictions[key] = {"verdict": verdict, "prediction": wording,
                            "estimate_ratio": ratios[metric_key]["ratio"],
                            "ratio_interval_95": ratios[metric_key]["ratio_interval_95"]}
    for key, metric in (("163.3", "name_gone_identity"),):
        if not all_gates:
            verdict = "untestable"
        else:
            verdict = "held" if all(rates[metric][a]["mean"] is not None and rates[metric][a]["mean"] >= .15
                                     for a in ("neuron", "random")) else "failed"
        predictions[key] = {"verdict": verdict,
                            "prediction": "Each arm's name-gone rate on identity phrasings is at least 0.15",
                            "rate_by_arm": rates[metric]}
    both_pass = all(identity_both_n[a] < identity_name_n[a] for a in identity_name_n)
    predictions["163.4"] = {"verdict": "untestable" if not all_gates else ("held" if both_pass else "failed"),
                            "prediction": "In each arm fewer units reach both gone on at least half of identity phrasings than reach name gone on at least half",
                            "name_gone_units_at_least_half": identity_name_n,
                            "both_gone_units_at_least_half": identity_both_n}
    if not all_gates:
        broken_verdict = "untestable"
    else:
        ci = paired_bootstrap(map_rates("broken_rate", "neuron"), map_rates("broken_rate", "random"), seed=163)
        broken_verdict = "held" if ci["neuron_mean"] < ci["random_mean"] else "failed"
    predictions["163.5"] = {"verdict": broken_verdict,
                            "prediction": "The neuron arm's broken rate is below the random arm's",
                            "rate_by_arm": rates["broken_rate"]}

    exploratory = {"name_gone_on_maker": rates["name_gone_maker"],
                   "maker_gone_on_identity": rates["maker_gone_identity"],
                   "company_names_found": {}, "top_first_sentence_names": {}, "layer_rates_with_at_least_8_pairs": {}}
    for arm in ("neuron", "random"):
        all_answers = [c for cells in records[arm].values() for c in cells if not c["broken"]]
        exploratory["company_names_found"][arm] = {company: sum(company.casefold() in c["answer"].casefold()
            for c in all_answers) for company in COMPANIES}
        names = Counter(name for c in all_answers if (name := first_sentence_name(c["answer"])) is not None)
        exploratory["top_first_sentence_names"][arm] = [{"name": name, "count": count} for name, count in names.most_common(15)]
    layer_ids = sorted({int(u["layer"]) for u in units})
    for layer in layer_ids:
        members = {arm: [u for u in units if int(u["layer"]) == layer and u["arm"] == arm]
                   for arm in ("neuron", "random")}
        pairs_n = min(len(members["neuron"]), len(members["random"]))
        if pairs_n < 8:
            continue
        layer_data = {}
        for arm in ("neuron", "random"):
            layer_data[arm] = {}
            for metric in ("name_gone_identity", "maker_gone_maker"):
                vals = [unit_rates[arm][u["unit_id"]][metric] for u in members[arm]
                        if unit_rates[arm][u["unit_id"]][metric] is not None]
                layer_data[arm][metric] = _mean(vals)
        exploratory["layer_rates_with_at_least_8_pairs"][str(layer)] = {"pairs": pairs_n, "rates": layer_data}

    rng = random.Random(163)
    hand_read = []
    for arm in ("neuron", "random"):
        candidates = [c for cells in records[arm].values() for c in cells]
        for cell in rng.sample(candidates, min(40, len(candidates))):
            hand_read.append({"arm": arm, "unit_id": next(u for u in units if u["pair_id"] == cell["pair_id"] and u["arm"] == arm)["unit_id"],
                              "pair_id": cell["pair_id"], "question_index": cell["question_index"],
                              "question": cell["question"], "answer": cell["answer"]})
    return {"part": 163, "gates": gates, "all_gates_passed": all_gates,
            "predictions": predictions, "rates": rates, "ratios": ratios,
            "per_unit_rates": unit_rates, "exploratory": exploratory,
            "selected_unit_counts": {a: len(units_by_arm[a]) for a in units_by_arm},
            "hand_read_rows": hand_read}


def write_outputs(result, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    hand = result.pop("hand_read_rows")
    (out / "score.json").write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    with (out / "hand_read.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in hand:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    lines = ["# Part 163 score", "", "## Gates", "", "| Gate | Verdict | Detail |", "| --- | --- | --- |"]
    for name, gate in result["gates"].items():
        detail = json.dumps({k: v for k, v in gate.items() if k != "passed"}, ensure_ascii=True)
        lines.append(f"| {name} | {'passed' if gate['passed'] else 'failed'} | `{detail}` |")
    lines += ["", "## Predictions", "", "| Prediction | Verdict | Summary |", "| --- | --- | --- |"]
    for key, pred in result["predictions"].items():
        lines.append(f"| {key} | {pred['verdict']} | {pred['prediction']} |")
    lines += ["", "## Primary rates", "", "| Metric | Neuron mean (95% interval) | Random mean (95% interval) |", "| --- | --- | --- |"]
    for metric in result["rates"]:
        cells = []
        for arm in ("neuron", "random"):
            val = result["rates"][metric][arm]
            ci = val["interval_95"]
            cells.append("n/a" if val["mean"] is None else f"{val['mean']:.3f} ({ci[0]:.3f}, {ci[1]:.3f})")
        lines.append(f"| {metric} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Paired ratios", "", "| Contrast | Ratio | 95% interval | Pairs |", "| --- | ---: | --- | ---: |"]
    for key, val in result["ratios"].items():
        ratio = "n/a" if val["ratio"] is None else f"{val['ratio']:.3f}"
        interval = val["ratio_interval_95"]
        ci = "n/a" if interval[0] is None else f"({interval[0]:.3f}, {interval[1]:.3f})"
        lines.append(f"| {key} | {ratio} | {ci} | {val['n_pairs']} |")
    lines += ["", "Exploratory summaries and layer rates are in `score.json`. Hand-read answers are in `hand_read.jsonl`.", ""]
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--units", default=str(HERE / "units163.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    units = json.loads(Path(args.units).read_text(encoding="utf-8"))
    result = score_rows(read_jsonl(args.answers), units)
    write_outputs(result, args.out)
    print(json.dumps({"all_gates_passed": result["all_gates_passed"], "predictions": result["predictions"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
