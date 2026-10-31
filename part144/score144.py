#!/usr/bin/env python3
"""Gate and score Part 144's six-cell forced-token measurements."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CELLS = ("base-plain", "base-chat", "sft-chat", "dpo-chat", "instruct-chat", "instruct-plain")
READS = ("p_first", "p_m", "p_sorry")
ARMS = ("full", "prompt-only", "answer-only")
PREDICTIONS = {
    "144.1": "The plain model's P(\" sorry\" after \"I'm\") on Base in the plain frame is at least its value on Instruct in the chat frame.",
    "144.2": "The median full-arm shift on Base in the plain frame is at most half the median on Instruct in the chat frame.",
    "144.3": "The median full-arm shift on Base in the chat frame is at most half the median on Instruct in the chat frame.",
    "144.4": "The median full-arm shifts on SFT, DPO and Instruct, all chat, lie within 1.0 of each other.",
    "144.5": "Across units, the Spearman correlation of full-arm shift between SFT and Instruct is at least 0.7, and between Base in the plain frame and Instruct is under 0.4.",
    "144.6": "From Base chat to SFT the median full-arm shift changes by more, in absolute value, than the plain model's log-odds does.",
    "144.7": "On Set R, in each of SFT, DPO and Instruct, the median of full minus (prompt-only + answer-only - plain), in log-odds, has magnitude at most 0.5.",
    "144.8": "In every cell, the median full-arm shifts of Set F's neurons and random directions differ by at most 0.5.",
    "144.9": "The median full-arm shift on Instruct in the plain frame is at most half the median on Instruct in the chat frame.",
}
CONFIDENCE = {"144.1": "medium", "144.2": "medium", "144.3": "low", "144.4": "medium",
              "144.5": "medium", "144.6": "medium", "144.7": "medium", "144.8": "medium", "144.9": "low"}


def read_rows(path, allow_empty=False):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON {path}:{n}: {exc}") from exc
    if not rows and not allow_empty:
        raise ValueError(f"no rows in {path}")
    return rows


def logodds(p):
    p = min(1 - 1e-9, max(1e-9, float(p)))
    return math.log(p / (1 - p))


def _quantile(vals, frac):
    x = sorted(float(v) for v in vals)
    if len(x) == 1:
        return x[0]
    z = (len(x) - 1) * frac
    lo, hi = math.floor(z), math.ceil(z)
    return x[lo] + (x[hi] - x[lo]) * (z - lo)


def _stats(values):
    return {"n": len(values), "median": statistics.median(values), "q1": _quantile(values, .25), "q3": _quantile(values, .75)}


def _bootstrap(values):
    rng = random.Random(144)
    n = len(values)
    if not n:
        return None
    draws = [statistics.median([values[rng.randrange(n)] for _ in range(n)]) for _ in range(10000)]
    return {"low": _quantile(draws, .025), "high": _quantile(draws, .975), "resamples": 10000, "seed": 144}


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in order[i:j]:
            ranks[k] = rank
        i = j
    return ranks


def spearman(a, b):
    if len(a) != len(b) or len(a) < 2:
        return None
    x, y = _rank(a), _rank(b)
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((u-mx)*(v-my) for u, v in zip(x, y))
    den = math.sqrt(sum((u-mx)**2 for u in x) * sum((v-my)**2 for v in y))
    return num / den if den else 0.0


def score(rows_by_cell, part143_rows, units=None, pilot=False, gates_only=False):
    gates = {}
    indexed = {}
    for cell in CELLS:
        rows = rows_by_cell.get(cell, [])
        indexed[cell] = {(r.get("set"), r.get("unit_id"), r.get("arm")): r for r in rows if r.get("unit_id")}
        plain = next((r for r in rows if r.get("arm") == "plain"), None)
        if plain is None:
            gates.setdefault("missing_plain", []).append(cell)
        expected_first = "I" if cell.endswith("chat") else " I"
        if not all(r.get("token_checks", {}).get("im_ids") and
                   r["token_checks"].get("first") == expected_first and
                   r["token_checks"].get("m") == "'m" and
                   r["token_checks"].get("sorry") == " sorry" for r in rows):
            gates.setdefault("missing_token_checks", []).append(cell)
    noop_comps = []
    for cell in CELLS:
        rows = rows_by_cell.get(cell, [])
        plain = next((r for r in rows if r.get("arm") == "plain"), None)
        for r in rows:
            if r.get("arm") != "noop":
                continue
            for read, field in zip(("p_first", "p_m", "p_sorry"), READS):
                diff = abs(float(r[field]) - float(plain[field])) if plain else None
                noop_comps.append({"cell": cell, "unit_id": r.get("unit_id"), "readout": read,
                                   "difference": diff, "passed": diff is not None and diff <= .001})
    expected_noops = {u["id"] for u in ((units["sets"]["R"] + units["sets"]["S"] + units["stage_f"]["units"])[:12] if units else [])}
    noop_complete = True
    missing_noops = {}
    for cell in CELLS:
        present = {r.get("unit_id") for r in rows_by_cell.get(cell, []) if r.get("arm") == "noop"}
        if expected_noops and not expected_noops.issubset(present):
            noop_complete = False
            missing_noops[cell] = sorted(expected_noops - present)
    gate1 = {"passed": not gates.get("missing_plain") and not gates.get("missing_token_checks") and bool(noop_comps) and noop_complete and all(x["passed"] for x in noop_comps),
             "threshold": .001, "comparisons": noop_comps,
             "details": {"missing_plain": gates.get("missing_plain", []), "missing_token_checks": gates.get("missing_token_checks", []), "missing_noops": missing_noops}}
    # Gate 2 replays the two registered readouts and both registered arms on Instruct chat Set R.
    old = {(r.get("readout"), r.get("arm"), r.get("unit_id")): r for r in part143_rows
           if r.get("set") == "R" and r.get("readout") in ("im_after_i", "sorry_after_im") and r.get("arm") in ("full", "prompt-only")}
    gate2_comps = []
    current_rows = rows_by_cell.get("instruct-chat", [])
    for row in current_rows:
        if row.get("set") != "R" or row.get("arm") not in ("full", "prompt-only"):
            continue
        for readout, field in (("im_after_i", "p_m"), ("sorry_after_im", "p_sorry")):
            prior = old.get((readout, row["arm"], row["unit_id"]))
            if prior is None:
                gate2_comps.append({"unit_id": row["unit_id"], "arm": row["arm"], "readout": readout, "passed": False, "missing_prior": True})
            else:
                prev = float(prior.get("probability", prior.get("sorry_probability")))
                diff = abs(float(row[field]) - prev)
                gate2_comps.append({"unit_id": row["unit_id"], "arm": row["arm"], "readout": readout, "difference": diff, "passed": diff <= .05})
    r_ids = {r.get("unit_id") for r in current_rows if r.get("set") == "R" and r.get("arm") in ("full", "prompt-only")}
    expected2 = (min(12, len(units["sets"]["R"])) if pilot and units else
                 (len(units["sets"]["R"]) if units else len(r_ids))) * 4
    gate2 = {"passed": bool(gate2_comps) and sum(x["passed"] for x in gate2_comps) / expected2 >= .90,
             "threshold": .05, "required_fraction": .90, "matching": sum(x["passed"] for x in gate2_comps),
             "expected": expected2, "fraction": sum(x["passed"] for x in gate2_comps) / expected2 if expected2 else 0,
             "comparisons": gate2_comps}
    gate3_comps = []
    for cell in CELLS:
        rows = rows_by_cell.get(cell, [])
        base = next((r for r in rows if r.get("arm") == "plain"), None)
        ans = [r for r in rows if r.get("arm") == "answer-only"]
        if not ans:
            gate3_comps.append({"cell": cell, "unit_id": None, "difference": None,
                                "token_checks_recorded": False, "passed": False, "missing_answer_only": True})
        for r in ans:
            diff = abs(float(r["p_first"]) - float(base["p_first"])) if base else None
            check = bool(r.get("token_checks", {}).get("im_ids"))
            gate3_comps.append({"cell": cell, "unit_id": r.get("unit_id"), "difference": diff, "token_checks_recorded": check,
                                "passed": diff is not None and diff <= 1e-6 and check})
    gate3 = {"passed": bool(gate3_comps) and all(c["passed"] for c in gate3_comps), "tolerance": 1e-6, "comparisons": gate3_comps}
    gate_set = {"gate1_noop_and_token_checks": gate1, "gate2_matches_part143": gate2, "gate3_answer_only_first_and_tokens": gate3}
    if not all(g["passed"] for g in gate_set.values()):
        return {"gates": gate_set, "predictions": {k: {"verdict": "withheld", "reason": "gate failed", "confidence": CONFIDENCE[k], "text": PREDICTIONS[k]} for k in PREDICTIONS}}
    if gates_only:
        return {"gates": gate_set}

    # Map each cell's baseline log odds and derive arm shifts per unit.
    base = {cell: logodds(next(r for r in rows_by_cell[cell] if r["arm"] == "plain")["p_sorry"]) for cell in CELLS}
    shift_rows, gaps = {}, {}
    stats = {}
    for cell in CELLS:
        rows = rows_by_cell[cell]
        plain_row = next(r for r in rows if r["arm"] == "plain")
        stats[cell] = {"plain": {read: float(plain_row[read]) for read in READS}}
        per = defaultdict(dict)
        for row in rows:
            if row.get("arm") == "plain":
                continue
            per[(row.get("set"), row.get("unit_id"))][row["arm"]] = row
        cell_shifts = defaultdict(list)
        readme_shifts = defaultdict(list)
        cell_gaps = defaultdict(list)
        for (set_name, unit_id), arms in per.items():
            for read in READS:
                vals = {a: logodds(arms[a][read]) for a in ARMS if a in arms}
                if "full" in vals:
                    shift = vals["full"] - (base[cell] if read == "p_sorry" else logodds(plain_row[read]))
                    unit_type = arms["full"].get("unit")
                    if set_name == "F" and read == "p_sorry":
                        shift_rows.setdefault(cell, {}).setdefault(unit_id, shift)
                if all(a in vals for a in ARMS):
                    gap = vals["full"] - vals["prompt-only"] - vals["answer-only"] + logodds(plain_row[read])
                    cell_gaps[(set_name, read)].append(gap)
                    if read == "p_sorry":
                        gaps.setdefault(cell, {}).setdefault(set_name, []).append(gap)
            for arm in ARMS:
                if arm in arms:
                    for read in READS:
                        shift = logodds(arms[arm][read]) - logodds(plain_row[read])
                        cell_shifts[(set_name, arms[arm].get("unit"), read, arm)].append(shift)
                        if set_name == "F":
                            readme_shifts[(arm, read)].append(shift)
        # Bootstrap every unit-level median shift. The README's main table uses
        # Set F medians for all three readouts and all three arms.
        stats[cell]["shifts"] = {"|".join(map(str, key)): {**_stats(v), "bootstrap95": _bootstrap(v)} for key, v in cell_shifts.items()}
        stats[cell]["readme_main"] = {
            f"{arm}|{read}": {**_stats(values), "bootstrap95": _bootstrap(values)}
            for (arm, read), values in readme_shifts.items()
        }
        stats[cell]["additivity_gaps"] = {"|".join(key): _stats(v) for key, v in cell_gaps.items()}

    def med(cell, set_name="F", unit=None):
        vals = [v for k, v in shift_rows.get(cell, {}).items() if True]
        lookup = []
        for row in rows_by_cell[cell]:
            if row.get("set") == set_name and row.get("arm") == "full" and (unit is None or row.get("unit") == unit):
                plain_row = next(r for r in rows_by_cell[cell] if r.get("arm") == "plain")
                lookup.append((row["unit_id"], logodds(row["p_sorry"]) - logodds(plain_row["p_sorry"])))
        return statistics.median([v for _, v in lookup]) if lookup else None

    instruct = med("instruct-chat")
    pred_vals = {}
    def put(k, value, condition, detail):
        pred_vals[k] = {"value": value, "verdict": "held" if condition else "failed", "details": detail, "text": PREDICTIONS[k], "confidence": CONFIDENCE[k]}
    pbase = float(next(r for r in rows_by_cell["base-plain"] if r["arm"] == "plain")["p_sorry"])
    pinstruct = float(next(r for r in rows_by_cell["instruct-chat"] if r["arm"] == "plain")["p_sorry"])
    put("144.1", pbase, pbase >= pinstruct, {"instruct_chat_plain": pinstruct})
    unreadable = instruct < 1.0
    for k, cell in (("144.2", "base-plain"), ("144.3", "base-chat"), ("144.9", "instruct-plain")):
        value = med(cell)
        threshold = instruct / 2
        pred_vals[k] = {"value": value, "threshold": threshold, "verdict": "withheld" if unreadable else ("held" if value <= threshold else "failed"), "reason": "Instruct chat median shift under 1.0" if unreadable else None, "text": PREDICTIONS[k], "confidence": CONFIDENCE[k]}
    stage_meds = {c: med(c) for c in ("sft-chat", "dpo-chat", "instruct-chat")}
    spread = max(stage_meds.values()) - min(stage_meds.values())
    put("144.4", spread, spread <= 1.0, {"medians": stage_meds, "spread": spread})
    def shifts_by(cell):
        return {r["unit_id"]: logodds(r["p_sorry"]) - logodds(next(x for x in rows_by_cell[cell] if x["arm"] == "plain")["p_sorry"])
                for r in rows_by_cell[cell] if r.get("set") == "F" and r.get("arm") == "full"}
    a, b, d = shifts_by("sft-chat"), shifts_by("instruct-chat"), shifts_by("base-plain")
    common1 = sorted(set(a) & set(b)); common2 = sorted(set(d) & set(b))
    rho1, rho2 = spearman([a[x] for x in common1], [b[x] for x in common1]), spearman([d[x] for x in common2], [b[x] for x in common2])
    put("144.5", {"sft_instruct": rho1, "base_plain_instruct": rho2}, rho1 is not None and rho2 is not None and rho1 >= .7 and rho2 < .4, {"n_sft_instruct": len(common1), "n_base_instruct": len(common2)})
    shift_change = abs(med("sft-chat") - med("base-chat"))
    plain_change = abs(base["sft-chat"] - base["base-chat"])
    put("144.6", shift_change, shift_change > plain_change, {"plain_logodds_change": plain_change})
    gap_meds = {c: statistics.median(gaps.get(c, {}).get("R", [])) for c in ("sft-chat", "dpo-chat", "instruct-chat")}
    put("144.7", gap_meds, all(abs(v) <= .5 for v in gap_meds.values()), {})
    differences = {}
    for cell in CELLS:
        differences[cell] = abs(med(cell, "F", "neuron") - med(cell, "F", "random"))
    put("144.8", differences, all(v <= .5 for v in differences.values()), {})
    return {"gates": gate_set, "stats": stats, "predictions": pred_vals}


def render_tables(result):
    lines = ["# Part 144 tables", "", "Bootstrap intervals are 95% percentile intervals over units, with 10,000 resamples and seed 144.", "", "## Shift by cell", "", "| Cell | Set/unit/arm/read | n | Median log-odds shift | Q1 | Q3 | Bootstrap 95% |", "|---|---|---:|---:|---:|---:|---|"]
    if "stats" not in result:
        return "# Part 144 tables\n\nGates failed; shifts withheld.\n"
    for cell, stats in result["stats"].items():
        for key, s in stats["shifts"].items():
            ci = s.get("bootstrap95")
            ci_text = "" if not ci else f"{ci['low']:.6g} to {ci['high']:.6g}"
            lines.append(f"| {cell} | {key} | {s['n']} | {s['median']:.6g} | {s['q1']:.6g} | {s['q3']:.6g} | {ci_text} |")
    lines += ["", "## README main-table intervals", "", "Set F intervals combine neurons and matched random directions.", "", "| Cell | Full P(first) | Full P('m') | Full P(' sorry') | Prompt-only P(' sorry') | Answer-only P(' sorry') |", "|---|---:|---:|---:|---:|---:|"]
    for cell, s in result["stats"].items():
        vals = []
        for arm, read in (("full", "p_first"), ("full", "p_m"), ("full", "p_sorry"), ("prompt-only", "p_sorry"), ("answer-only", "p_sorry")):
            item = s["readme_main"][f"{arm}|{read}"]
            ci = item["bootstrap95"]
            vals.append(f"{item['median']:.6g} ({ci['low']:.6g} to {ci['high']:.6g})")
        lines.append(f"| {cell} | " + " | ".join(vals) + " |")
    lines += ["", "## Plain probabilities", "", "| Cell | P(first) | P('m') | P(' sorry') |", "|---|---:|---:|---:|"]
    for cell, s in result["stats"].items():
        p = s["plain"]
        lines.append(f"| {cell} | {p['p_first']:.8g} | {p['p_m']:.8g} | {p['p_sorry']:.8g} |")
    lines += ["", "## Additivity gaps", "", "| Cell | Set/read | n | Median log-odds gap | Q1 | Q3 |", "|---|---|---:|---:|---:|---:|"]
    for cell, s in result["stats"].items():
        for key, v in s["additivity_gaps"].items():
            lines.append(f"| {cell} | {key} | {v['n']} | {v['median']:.6g} | {v['q1']:.6g} | {v['q3']:.6g} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/part144")
    ap.add_argument("--part143", default="results/part143/results.jsonl")
    ap.add_argument("--units", default=str(ROOT / "part139/units139.json"))
    ap.add_argument("--out", default="results/part144")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.input)
    rows = {cell: read_rows(root / (f"pilot-{cell}.jsonl" if args.pilot else f"{cell}.jsonl")) for cell in CELLS}
    part143 = read_rows(args.part143)
    units = json.loads(Path(args.units).read_text(encoding="utf-8"))
    result = score(rows, part143, units, args.pilot, args.pilot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.pilot:
        (out / "score.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (out / "tables.md").write_text(render_tables(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not all(g["passed"] for g in result["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
