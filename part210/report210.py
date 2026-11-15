#!/usr/bin/env python3
"""Part 210's report: every number the results README states, from the result files.

Reads:
- results/part210/score.md, part210/score210.py's printout (each line `label: value  <- file:key`), for the gates,
  verdicts, cap counts and the shape-floor stabilities (the shape floor took 47 minutes; it is read, not rerun);
- results/part210/score.json, score210's summary;
- results/part210/answers.jsonl and replicate.jsonl, the rerun's rows, for the point estimates, the per-arm broken
  counts, where each differing replicate text first parts from its chunk-of-8 text (in bf16 steps,
  tools/margin_recorder.bf16_step) and the label changes under score210.floor_labels;
- results/part165/score.json and answers.jsonl, Part 165's published run;
- results/part210/margins.json, part210/margins210.py's published distances.

The rerun's point estimates are recomputed from the rows with part165/score165.py's own score_rows, its bootstraps
replaced by their point estimates (score210._score_without_bootstrap, reps=0), and checked against score.md: a
disagreement is an error here, not a silent difference.

Writes results/part210/report210.json and report210.md. part210/fig210.py draws from report210.json only.

Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I -B part210/report210.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from tools.margin_recorder import bf16_step  # noqa: E402
from part210 import score210  # noqa: E402

RES = ROOT / "results" / "part210"
PUB = ROOT / "results" / "part165"
CAP = 80
ARMS = ("n+", "n±", "k4", "k16", "k64", "k256", "k1024", "full", "iso")
VERDICTS = ("165.1", "165.2", "165.3", "165.4", "165.5", "165.6")
SCENARIOS = ("independent", "shared-change-0.5", "shared-change-1", "upper-marginal-independent")
GATES = ("gate1_rows", "gate2_no_push", "gate3_size", "gate4_measure_room")
BARS = {"165.1": "n+ at least twice full (>= 2)", "165.2": "broken Spearman <= -0.8",
        "165.3": "k16 nearer full than n±", "165.4": "n+ over n± between 0.67 and 1.5",
        "165.5": "iso under 0.8 times full", "165.6": "changed Spearman <= -0.6"}
# The registered predictions, verbatim from PREDICTIONS.md, "Part 210", with the confidence from PART210-DESIGN.md.
PREDICTIONS = {
    "210.1": ("Gates R1 to R3 and Part 165's Gates 1 to 4 pass.", "medium"),
    "210.2": ("165.1 holds, 165.2 holds, 165.3 holds, 165.4 holds, 165.5 fails and 165.6 fails, as published. "
              "Registered rules: \"n+ breaks at least twice as often as full\"; \"Broken rate falls with k: Spearman "
              "at or below -0.8 across n±, k4, k16, k64, k256, k1024, full\"; \"k16's broken rate is nearer full's "
              "than n±'s\"; \"The n+ over n± broken ratio is between 0.67 and 1.5\"; \"iso's broken rate is under 0.8 "
              "times full's\"; \"Changed rate falls with k: Spearman at or below -0.6 across the same seven arms\".",
              "low"),
    "210.3": ("The n+ over full broken ratio is above 1.5, the same side of 1 as published.", "medium"),
    "210.4": ("In each arm at least 300 of 768 battery answers reach `token_count == 80`, against 50 to 84 published.",
              "low"),
    "210.5": ("The n+ over full broken ratio is lower on the rerun than the published 2.69863.", "low"),
    "210.6": ("Between 40 and 110 of the 200 shape-replicate texts differ from their chunk-of-8 texts.", "low"),
    "210.7": ("The primary-broken rate is higher on the rerun than published in at least 8 of the 9 arms.", "medium"),
}


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


def estimate(pred):
    """A verdict's estimate as score165 prints it: its ratio, its Spearman, or 165.3's distance margin."""
    if pred.get("estimate_ratio") is not None:
        return float(pred["estimate_ratio"])
    if pred.get("spearman") is not None:
        return float(pred["spearman"]["estimate"])
    if pred.get("distances") is not None:
        return float(pred["distances"]["to_n_pm"]) - float(pred["distances"]["to_full"])
    return None


def interval(pred):
    for key in ("ratio_interval_95",):
        if pred.get(key):
            return [float(pred[key][0]), float(pred[key][1])]
    if pred.get("spearman") and pred["spearman"].get("interval_95"):
        iv = pred["spearman"]["interval_95"]
        # score210's point-only scorer (reps=0) draws no bootstrap, so its interval is [null, null]; the published
        # file carries the real interval.
        return None if any(v is None for v in iv) else [float(v) for v in iv]
    return None


def sizes(rows):
    """Per arm: battery rows, answers at the 80-token cap, mean token_count."""
    out = {}
    for arm in ARMS:
        battery = [r for r in rows if r.get("stage") == "battery" and r.get("arm") == arm]
        counts = [int(r["token_count"]) for r in battery if isinstance(r.get("token_count"), (int, float))]
        out[arm] = {"rows": len(battery), "at_cap": sum(c == CAP for c in counts),
                    "mean_tokens": sum(counts) / len(counts) if counts else None}
    return out


def main():
    md = read_score_md(RES / "score.md")
    summary = json.loads((RES / "score.json").read_text(encoding="utf-8"))
    margins = json.loads((RES / "margins.json").read_text(encoding="utf-8"))
    published = json.loads((PUB / "score.json").read_text(encoding="utf-8"))
    rows = read_jsonl(RES / "answers.jsonl")
    replicate = read_jsonl(RES / "replicate.jsonl")
    pub_rows = read_jsonl(PUB / "answers.jsonl")
    norms = json.loads((RES / "norms.json").read_text(encoding="utf-8"))

    # ---- Part 165's six verdicts, published beside the rerun, the rerun checked against score.md
    rerun_score = score210._score_without_bootstrap(score210.part165_rows(rows), norms, reps=0)
    verdicts = {}
    for vid in VERDICTS:
        pub = published["predictions"][vid]
        run = rerun_score["predictions"][vid]
        printed_verdict = md[f"{vid} verdict"]
        assert run["verdict"] == printed_verdict, (vid, run["verdict"], printed_verdict)
        printed_estimate = num(md[f"{vid} estimate"])
        if printed_estimate is not None:
            assert f"{estimate(run):.6g}" == f"{printed_estimate:.6g}", (vid, estimate(run), printed_estimate)
        entry = {"bar": BARS[vid],
                 "published": {"verdict": pub["verdict"], "estimate": estimate(pub), "interval_95": interval(pub)},
                 "rerun": {"verdict": run["verdict"], "estimate": estimate(run), "interval_95": interval(run)}}
        if vid == "165.3":
            for side, src in (("published", pub), ("rerun", run)):
                entry[side]["distances"] = {k: float(v) for k, v in src["distances"].items()}
        if vid == "165.6":
            # score.md prints the eager usable-draw count from score165.score_rows on the eager rows (score210.py, the
            # verdict block), with score165's own bootstrap (10,000 draws, seed 165).
            entry["rerun"]["usable_draws"] = int(md["165.6 usable draws"])
            entry["rerun"]["interval_collapsed"] = md["165.6 interval collapsed (not informative)"] == "True"
            entry["published"]["usable_draws"] = int(pub["spearman"]["usable_draws"])
        verdicts[vid] = entry

    # ---- Part 165's four gates, published beside the rerun
    gates = {}
    for gid in GATES:
        pub, run = published["gates"][gid], rerun_score["gates"][gid]
        assert str(run["passed"]) == md[f"{gid} passed"], gid
        keep = lambda g: {k: v for k, v in g.items() if not isinstance(v, (dict, list))}
        gates[gid] = {"published": keep(pub), "rerun": keep(run)}

    # ---- per-arm broken rates and counts, published beside eager
    pub_broken = published["rates"]["broken"]
    broken = {}
    for arm in ARMS:
        battery = [r for r in rows if r.get("stage") == "battery" and r.get("arm") == arm]
        count = sum(bool(score210.score165.primary_broken(r.get("answer") or "")) for r in battery)
        broken[arm] = {"published_rate": float(pub_broken[arm]),
                       "published_count": sum(bool(score210.score165.primary_broken(r.get("answer") or ""))
                                              for r in pub_rows if r.get("stage") == "battery" and r.get("arm") == arm),
                       "rerun_rate": float(rerun_score["rates"]["broken"][arm]),
                       "rerun_count": count, "rows": len(battery)}
        assert f"{broken[arm]['rerun_rate']:.6g}" == f"{num(md[f'broken rate {arm}']):.6g}", arm

    # ---- lengths before cross-run rates
    size = {"published": sizes(pub_rows), "rerun": sizes(rows)}
    for arm in ARMS:
        assert size["rerun"][arm]["at_cap"] == int(num(md[f"answers at the 80-token cap, {arm}"])), arm
        assert f"{size['rerun'][arm]['mean_tokens']:.6g}" == f"{num(md[f'mean token_count, {arm}']):.6g}", arm

    # ---- the shape replicate: where each differing text first parts, and which labels change
    battery = {r.get("row_id"): r for r in rows if r.get("stage") == "battery"}
    bins = Counter()
    first_positions, flips = [], Counter()
    differ = 0
    for bat, rep in score210.replicate_pairs(list(battery.values()), replicate):
        a, b = score210.floor_labels(bat), score210.floor_labels(rep)
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
    assert len(replicate) == int(num(md["replicate rows"])), len(replicate)
    order = ["0 (exact tie)", "over 0, up to 1", "over 1, up to 2", "over 2", "no recorded margin"]
    # Where the first difference sits: the token index. A difference at token 0 is a different first token; any later
    # one means the two shapes agreed up to that token.
    position_bins = Counter()
    for pos in first_positions:
        if pos == 0:
            position_bins["token 0 (first token)"] += 1
        elif pos <= 9:
            position_bins["tokens 1 to 9"] += 1
        elif pos <= 19:
            position_bins["tokens 10 to 19"] += 1
        elif pos <= 39:
            position_bins["tokens 20 to 39"] += 1
        else:
            position_bins["token 40 or later"] += 1
    position_order = ["token 0 (first token)", "tokens 1 to 9", "tokens 10 to 19", "tokens 20 to 39",
                      "token 40 or later"]
    replicate_out = {"rows": len(replicate), "differ": differ, "bar": [40, 110],
                     "first_difference_battery_margin_steps": {b: bins.get(b, 0) for b in order},
                     "first_difference_position_bins": {b: position_bins.get(b, 0) for b in position_order},
                     "first_difference_position_median": sorted(first_positions)[len(first_positions) // 2]
                     if first_positions else None,
                     "label_changes": dict(sorted(flips.items()))}

    # ---- stability per verdict and scenario, from score.md
    stability = {vid: {sc: num(md[f"stability {sc} {vid}"]) for sc in SCENARIOS} for vid in VERDICTS[:5]}
    gate_stability = {gid: {sc: num(md[f"stability {sc} gate:{gid}"]) for sc in SCENARIOS} for gid in GATES}

    # ---- the registered predictions
    def held(label):
        value = md[label]
        if value == "True":
            return "held"
        if value == "False":
            return "failed"
        return "not read"

    at_cap = {arm: size["rerun"][arm]["at_cap"] for arm in ARMS}
    predictions = {
        "210.1": {"verdict": held("210.1 gates R1-R3 and Part 165's four gates pass"),
                  "R1": md["R1 every launch eager"], "R2": md["R2 input hashes"],
                  "R3_bad_pushed_calls": int(num(md["R3 pushed calls not in chunks of 8"])),
                  "R3_bad_no_push_calls": int(num(md["R3 no-push calls not one row"])),
                  "R3_companion_rows": int(num(md["R3 companion rows (discarded)"]))},
        "210.2": {"verdict": held("210.2 165.1-165.4 hold and 165.5-165.6 fail, as published, gates passed"),
                  "eager_verdicts": {vid: verdicts[vid]["rerun"]["verdict"] for vid in VERDICTS}},
        "210.3": {"verdict": held("210.3 n+ over full broken ratio above 1.5"),
                  "ratio": verdicts["165.1"]["rerun"]["estimate"]},
        "210.4": {"verdict": held("210.4 at least 300 of 768 at the cap in each arm"), "at_cap": at_cap,
                  "minimum": min(at_cap.values()), "maximum": max(at_cap.values())},
        "210.5": {"verdict": held("210.5 n+ over full broken ratio lower than the published 2.69863"),
                  "ratio": verdicts["165.1"]["rerun"]["estimate"],
                  "published": verdicts["165.1"]["published"]["estimate"]},
        "210.6": {"verdict": held("210.6 between 40 and 110 of the 200 differ"), "differ": differ,
                  "of": len(replicate)},
        "210.7": {"verdict": held("210.7 primary-broken rate higher in at least 8 of 9 arms"),
                  "arms_higher": md["210.7 arms higher"].split(", ")},
    }

    # ---- the published distances, from margins.json (part210/margins210.py)
    distances = {entry["id"]: {"estimate": entry.get("estimate"), "bar": entry.get("bar"),
                               "distance": entry.get("distance")} for entry in margins["distances"]}

    report = {"source": "part210/report210.py", "predictions": predictions, "part165_gates": gates,
              "part165_verdicts": verdicts, "broken_rates": broken, "sizes": size, "replicate": replicate_out,
              "stability": stability, "gate_stability": gate_stability, "published_distances": distances,
              "summary": summary, "withdrawn": md["headline reading withdrawn"] == "True",
              "score_md_meta": {"score_md_mtime": (RES / "score.md").stat().st_mtime,
                                "score_md_bytes": (RES / "score.md").stat().st_size}}
    (RES / "report210.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    f = lambda x, d=3: "null" if x is None else f"{x:.{d}f}"
    lines = ["# Part 210 report (made by part210/report210.py)", "", "## Predictions", "",
             "| # | Verdict | Numbers |", "| --- | --- | --- |"]
    for pid, p in predictions.items():
        body = ", ".join(f"{k} {json.dumps(v) if isinstance(v, (dict, list)) else v}"
                         for k, v in p.items() if k != "verdict")
        lines.append(f"| {pid} | {p['verdict']} | {body} |")
    lines += ["", "## Part 165's verdicts, published beside the eager rerun", "",
              "| Prediction | Bar | Published estimate (95% interval) | Published | Eager estimate (95% interval) | Eager |",
              "| --- | --- | --- | --- | --- | --- |"]
    for vid, v in verdicts.items():
        def cell(side):
            s = v[side]
            iv = s["interval_95"]
            if vid == "165.3":
                return f"to full {f(s['distances']['to_full'], 4)}, to n± {f(s['distances']['to_n_pm'], 4)}"
            where = f"{f(s['estimate'], 5)}"
            if iv and iv[0] is not None:
                where += f" ({f(iv[0])} to {f(iv[1])})"
            return where
        lines.append(f"| {vid} | {v['bar']} | {cell('published')} | {v['published']['verdict']} | "
                     f"{cell('rerun')} | {v['rerun']['verdict']} |")
    pub6 = verdicts["165.6"]["published"]["usable_draws"]
    run6 = verdicts["165.6"]["rerun"]["usable_draws"]
    lines += ["", f"165.6's interval collapsed to a point on both runs: published {pub6:,} usable draws of 10,000, eager "
              f"{run6:,} of 10,000 (score.md, '165.6 usable draws', from score165.score_rows on the eager rows).",
              "", "## Part 165's gates", "", "| Gate | Published | Eager |", "| --- | --- | --- |"]
    for gid, g in gates.items():
        lines.append(f"| {gid} | {json.dumps(g['published'])} | {json.dumps(g['rerun'])} |")
    lines += ["", "## Broken rates per arm, published beside eager", "",
              "| Arm | Published broken / 768 | Published rate | Eager broken / 768 | Eager rate | Change |",
              "| --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        b = broken[arm]
        lines.append(f"| {arm} | {b['published_count']} | {f(b['published_rate'], 4)} | {b['rerun_count']} | "
                     f"{f(b['rerun_rate'], 4)} | {b['rerun_rate'] - b['published_rate']:+.4f} |")
    lines += ["", "## Lengths (battery rows)", "",
              "| Run | Arm | Rows | At the 80-token cap | Mean token_count |", "| --- | --- | --- | --- | --- |"]
    for run in ("published", "rerun"):
        for arm in ARMS:
            s = size[run][arm]
            lines.append(f"| {run} | {arm} | {s['rows']} | {s['at_cap']} | {f(s['mean_tokens'])} |")
    lines += ["", "## Shape replicate", "", f"Differ: {differ} of {len(replicate)} (bar 40 to 110).", "",
              "Battery row's margin at the first differing position, in bf16 steps "
              "(tools/margin_recorder.bf16_step):", ""]
    for b, n in replicate_out["first_difference_battery_margin_steps"].items():
        lines.append(f"- {b}: {n}")
    lines += ["", f"Median first differing position: token {replicate_out['first_difference_position_median']}.",
              "", "Where the first difference sits (token index, 0-based):", ""]
    for b, n in replicate_out["first_difference_position_bins"].items():
        lines.append(f"- {b}: {n}")
    lines += ["", "Label changes over the 200 pairs (score210.floor_labels): " +
              (", ".join(f"{k} {v}" for k, v in replicate_out["label_changes"].items()) or "none"), "",
              "## Stability under the shape floor (share of 1,000 redraws keeping the verdict)", "",
              "| Verdict | " + " | ".join(SCENARIOS) + " |", "| --- |" + " --- |" * len(SCENARIOS)]
    for vid, st in stability.items():
        lines.append(f"| {vid} | " + " | ".join(f(st[sc]) for sc in SCENARIOS) + " |")
    lines.append("| 165.6 | floor not informative: changed is constant | | | |")
    for gid, st in gate_stability.items():
        lines.append(f"| {gid} | " + " | ".join(f(st[sc]) for sc in SCENARIOS) + " |")
    (RES / "report210.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
