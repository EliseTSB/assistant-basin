#!/usr/bin/env python3
"""Part 202's report: every number the results README states, from the result files.

Reads:
- results/part202/score.md, part202/score202.py's printout (each line `label: value  <- file:key`), for the gates,
  verdicts, cap counts and the shape-floor stabilities (the shape floor took 15 minutes; it is read, not rerun);
- results/part202/cold/score164-direct.json, part164/score164.py run directly on Part 202's rows, for the bootstrap
  intervals score202 skips (its point estimates and verdicts are checked equal to score.md's here);
- results/part164/score.json and answers.jsonl, Part 164's published run;
- results/part202/answers.jsonl and replicate.jsonl, for the published-against-rerun sizes, where each differing
  replicate text first parts from its chunk-of-8 text (in bf16 steps, tools/margin_recorder.bf16_step) and the label
  changes under score202.floor_labels.

Writes results/part202/report202.json and report202.md. part202/fig202.py draws from report202.json only.

Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I -B part202/report202.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from tools.margin_recorder import bf16_step  # noqa: E402
from part202 import score202  # noqa: E402

RES = ROOT / "results" / "part202"
PUB = ROOT / "results" / "part164"
CAP = 80
SCENARIOS = ("independent", "shared-change-0.5", "shared-change-1", "upper-marginal-independent")
VERDICTS = ("164.1", "164.2", "164.3", "164.4", "164.5")
BARS = {"164.1": "< 0.7", "164.2": ">= 2", "164.3": "0.8 < 1.0 < 1.2", "164.4": "0.7 to 1.3", "164.5": "< 0.5"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_score_md(path):
    """label -> value text. Labels never contain ': ', values may, so split at the first ': '."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "  <- " not in line or ": " not in line:
            continue
        body = line.rsplit("  <- ", 1)[0]
        label, value = body.split(": ", 1)
        out[label] = value
    return out


def num(text):
    if text in ("null", "None"):
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


def interval(text):
    if text is None:
        return None
    values = json.loads(text) if isinstance(text, str) else text
    return [float(values[0]), float(values[1])]


def sizes(rows):
    out = {}
    for arm in ("neuron", "random"):
        battery = [r for r in rows if r.get("stage") == "battery" and r.get("arm") == arm]
        counts = [int(r["token_count"]) for r in battery]
        out[arm] = {"rows": len(battery), "at_cap": sum(c == CAP for c in counts),
                    "mean_tokens": sum(counts) / len(counts) if counts else None}
    return out


def estimate(pred):
    for key in ("estimate_ratio", "estimate_ratio_1.2_over_0.8"):
        if pred.get(key) is not None:
            return float(pred[key])
    return None


def main():
    md = read_score_md(RES / "score.md")
    direct = json.loads((RES / "cold" / "score164-direct.json").read_text(encoding="utf-8"))
    published = json.loads((PUB / "score.json").read_text(encoding="utf-8"))
    rows = read_jsonl(RES / "answers.jsonl")
    replicate = read_jsonl(RES / "replicate.jsonl")
    pub_rows = read_jsonl(PUB / "answers.jsonl")

    # ---- Part 164's five verdicts, published beside the rerun, with the rerun checked against score.md
    verdicts = {}
    for vid in VERDICTS:
        pub, run = published["predictions"][vid], direct["predictions"][vid]
        printed_verdict = md[f"{vid} verdict"]
        assert run["verdict"] == printed_verdict, (vid, run["verdict"], printed_verdict)
        printed_estimate = num(md[f"{vid} estimate"])
        if printed_estimate is not None:
            assert f"{estimate(run):.6g}" == f"{printed_estimate:.6g}", (vid, estimate(run), printed_estimate)
        entry = {"bar": BARS[vid],
                 "published": {"verdict": pub["verdict"], "estimate": estimate(pub),
                               "interval_95": interval(pub.get("ratio_interval_95"))},
                 "rerun": {"verdict": run["verdict"], "estimate": estimate(run),
                           "interval_95": interval(run.get("ratio_interval_95"))}}
        if vid == "164.3":
            entry["published"]["broken_rate_by_dose"] = pub["broken_rate_by_dose"]
            entry["rerun"]["broken_rate_by_dose"] = run["broken_rate_by_dose"]
        if vid == "164.4":
            for side, src in (("published", pub), ("rerun", run)):
                entry[side].update({k: src[k] for k in ("matched_random_dose", "neuron_broken_rate", "random_broken_rate")})
        verdicts[vid] = entry

    # ---- Part 164's four gates, published beside the rerun
    gates = {}
    for gid in ("gate1_rows", "gate2_no_push", "gate3_recipe", "gate4_measure_room"):
        pub, run = published["gates"][gid], direct["gates"][gid]
        assert str(run["passed"]) == md[f"{gid} passed"], gid
        keep = lambda g: {k: v for k, v in g.items() if not isinstance(v, (dict, list))}
        gates[gid] = {"published": keep(pub), "rerun": keep(run)}

    # ---- rates per cell
    rates = {}
    for kind in ("qwen_rate", "broken_rate"):
        rates[kind] = {}
        for dose, arms in direct["rates"][kind].items():
            for arm, value in arms.items():
                rates[kind][f"{arm}|{dose}"] = {"published": published["rates"][kind][dose][arm]["mean"],
                                                "rerun": value["mean"]}

    # ---- lengths before cross-run rates
    size = {"published": sizes(pub_rows), "rerun": sizes(rows)}
    assert size["rerun"]["neuron"]["at_cap"] == int(num(md["answers at the 80-token cap, neuron"]))
    assert size["rerun"]["random"]["at_cap"] == int(num(md["answers at the 80-token cap, random"]))

    # ---- the shape replicate: where each differing text first parts, and which labels change
    key = lambda r: (r.get("unit_id"), float(r.get("dose", 0.0)), int(r.get("question_index", -1)))
    battery = {key(r): r for r in rows if r.get("stage") == "battery"}
    bins = Counter()
    first_positions, flips = [], Counter()
    differ = 0
    for rep in replicate:
        bat = battery[key(rep)]
        a, b = score202.floor_labels(bat), score202.floor_labels(rep)
        for label in sorted(set(a) | set(b)):
            if a.get(label) != b.get(label):
                flips[label] += 1
        if rep.get("answer") == bat.get("answer"):
            continue
        differ += 1
        ta, tb = bat["tokens"], rep["tokens"]
        pos = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), min(len(ta), len(tb)))
        first_positions.append(pos)
        if pos >= len(bat["margin"]):
            bins["no recorded margin"] += 1
            continue
        steps = abs(float(bat["margin"][pos])) / bf16_step(float(bat["top1_logit"][pos]))
        if steps == 0:
            bins["0 (exact tie)"] += 1
        elif steps <= 1:
            bins["over 0, up to 1"] += 1
        elif steps <= 2:
            bins["over 1, up to 2"] += 1
        else:
            bins["over 2"] += 1
    assert differ == int(num(md["replicate texts that differ from the chunk-of-8 text"])), differ
    order = ["0 (exact tie)", "over 0, up to 1", "over 1, up to 2", "over 2", "no recorded margin"]
    replicate_out = {"rows": len(replicate), "differ": differ, "bar": [40, 100],
                     "first_difference_battery_margin_steps": {b: bins.get(b, 0) for b in order},
                     "first_difference_position_median": sorted(first_positions)[len(first_positions) // 2]
                     if first_positions else None,
                     "label_changes": dict(sorted(flips.items()))}

    # ---- stability per verdict and scenario, from score.md
    stability = {vid: {sc: num(md[f"stability {sc} {vid}"]) for sc in SCENARIOS} for vid in VERDICTS}
    gate_stability = {gid: {sc: num(md[f"stability {sc} gate:{gid}"]) for sc in SCENARIOS} for gid in gates}

    predictions = {
        "202.1": {"verdict": "held" if md["202.1 gates R1-R3 and Part 164's four gates pass"] == "True" else "failed",
                  "R1": md["R1 every launch eager"], "R2": md["R2 input hashes"],
                  "R3_bad_pushed_calls": int(num(md["R3 pushed calls not in chunks of 8"])),
                  "R3_bad_no_push_calls": int(num(md["R3 no-push calls not one row"]))},
        "202.2": {"verdict": "held" if md["202.2 verdicts as predicted (164.1 failed; 164.2, 164.3, 164.5 held; 164.4 failed), gates passed"] == "True" else "failed"},
        "202.3": {"verdict": "held" if md["202.3 neuron Qwen below random and neuron broken above random at +0.8"] == "True" else "failed",
                  "qwen_neuron": rates["qwen_rate"]["neuron|0.8"]["rerun"], "qwen_random": rates["qwen_rate"]["random|0.8"]["rerun"],
                  "broken_neuron": rates["broken_rate"]["neuron|0.8"]["rerun"], "broken_random": rates["broken_rate"]["random|0.8"]["rerun"]},
        "202.4": {"verdict": "held" if md["202.4 at least 480/1600 neuron and 1440/4800 random at the cap"] == "True" else "failed",
                  "neuron_at_cap": size["rerun"]["neuron"]["at_cap"], "random_at_cap": size["rerun"]["random"]["at_cap"]},
        "202.5": {"verdict": "held" if md["202.5 between 40 and 100 of the 200 differ"] == "True" else "failed",
                  "differ": differ, "of": len(replicate)},
    }

    report = {"source": "part202/report202.py", "predictions": predictions, "part164_gates": gates,
              "part164_verdicts": verdicts, "rates": rates, "sizes": size, "replicate": replicate_out,
              "stability": stability, "gate_stability": gate_stability,
              "withdrawn": md["headline reading withdrawn"] == "True"}
    (RES / "report202.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    f = lambda x, d=3: "null" if x is None else f"{x:.{d}f}"
    lines = ["# Part 202 report (made by part202/report202.py)", "", "## Predictions", "",
             "| # | Verdict | Numbers |", "| --- | --- | --- |"]
    for pid, p in predictions.items():
        lines.append(f"| {pid} | {p['verdict']} | " + ", ".join(f"{k} {v}" for k, v in p.items() if k != "verdict") + " |")
    lines += ["", "## Part 164's verdicts, published beside the eager rerun", "",
              "| Prediction | Bar | Published estimate (95% interval) | Published | Rerun estimate (95% interval) | Rerun |",
              "| --- | --- | --- | --- | --- | --- |"]
    for vid, v in verdicts.items():
        def cell(side):
            s = v[side]
            if vid == "164.3":
                return "broken by dose " + ", ".join(f"{d} {r:.4f}" for d, r in s["broken_rate_by_dose"].items())
            iv = s["interval_95"]
            return f"{f(s['estimate'])} ({f(iv[0])} to {f(iv[1])})" if iv else f(s["estimate"])
        lines.append(f"| {vid} | {v['bar']} | {cell('published')} | {v['published']['verdict']} | {cell('rerun')} | {v['rerun']['verdict']} |")
    lines += ["", "## Part 164's gates", "", "| Gate | Published | Rerun |", "| --- | --- | --- |"]
    for gid, g in gates.items():
        lines.append(f"| {gid} | {json.dumps(g['published'])} | {json.dumps(g['rerun'])} |")
    lines += ["", "## Rates per cell", "", "| Cell | Qwen published | Qwen rerun | Broken published | Broken rerun |",
              "| --- | --- | --- | --- | --- |"]
    for cell_id in rates["broken_rate"]:
        q = rates["qwen_rate"][cell_id]
        b = rates["broken_rate"][cell_id]
        lines.append(f"| {cell_id} | {f(q['published'], 4)} | {f(q['rerun'], 4)} | {f(b['published'], 4)} | {f(b['rerun'], 4)} |")
    lines += ["", "## Lengths (battery rows)", "", "| Run | Arm | Rows | At the 80-token cap | Mean token_count |",
              "| --- | --- | --- | --- | --- |"]
    for run in ("published", "rerun"):
        for arm, s in size[run].items():
            lines.append(f"| {run} | {arm} | {s['rows']} | {s['at_cap']} | {f(s['mean_tokens'])} |")
    lines += ["", "## Shape replicate", "", f"Differ: {differ} of {len(replicate)} (bar 40 to 100).", "",
              "Battery row's margin at the first differing position, in bf16 steps:", ""]
    for b, n in replicate_out["first_difference_battery_margin_steps"].items():
        lines.append(f"- {b}: {n}")
    lines += ["", "Label changes over the 200 pairs (score202.floor_labels): " +
              ", ".join(f"{k} {v}" for k, v in replicate_out["label_changes"].items()), "",
              "## Stability under the shape floor (share of 1,000 redraws keeping the verdict)", "",
              "| Verdict | " + " | ".join(SCENARIOS) + " |", "| --- |" + " --- |" * len(SCENARIOS)]
    for vid, st in stability.items():
        lines.append(f"| {vid} | " + " | ".join(f(st[sc]) for sc in SCENARIOS) + " |")
    for gid, st in gate_stability.items():
        lines.append(f"| {gid} | " + " | ".join(f(st[sc]) for sc in SCENARIOS) + " |")
    (RES / "report202.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
