#!/usr/bin/env python3
"""Part 209's report: every number the results README states, from the result files.

Reads:
- results/part209/scores.json, part209/score209.py's output, for the gates, R1, R1c, R2, R3 and the verdicts of
  209.1 to 209.15;
- results/part209/score.log, score209's printout, to check that each R1 s and R2 share taken from scores.json is the
  number the scorer printed;
- results/part209/r2-qwen-chat.jsonl, r2-instruct.jsonl and labels-ds.jsonl, labels-haiku.jsonl, to count the
  answers and labels directly.

Scores 209.16 by its registered rule (PART209-DESIGN.md): OLMo 2 1B at pretraining 3.0T, R1, THIRD's s is at least 0.5
and within 0.15 of ROOM-W's; not read if either arm's mean P3 is under 0.1. score209 prints no verdict for it.

Writes results/part209/report209.json and report209.md. part209/fig209.py draws from report209.json only.

Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I -B part209/report209.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "part209"

OLMO = ("pretrain-3.0T", "midtrain-26B", "midtrain-end", "sft", "dpo", "instruct")
QWEN = ("qwen-base", "qwen-chat")
PLACE_ARMS = ("ROOM-W", "ROOM-0", "FIELD-W", "FIELD-0", "NONE-W", "NONE-0")
R2_ARMS = PLACE_ARMS + ("NOBODY", "NOBODY-I", "TOLD-SIT", "TOLD-STAND", "TOLD-LIE")
R1_ARMS = ("ROOM-W", "ROOM-0", "FIELD-W", "FIELD-0", "NONE-W", "NONE-0", "NOBODY", "NOBODY-I", "THIRD", "OBJECT")
CATS = ("sitting", "standing", "lying", "other", "none", "unreadable")
P3_FLOOR = 0.1


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def last_labels(path):
    out = {}
    for r in read_jsonl(path):
        if "error" in r or "posture" not in r:
            continue
        out[r["row_id"]] = r
    return out


def log_values(path):
    """Parse score.log's R1 and R2 lines: (stage, arm) -> number."""
    r1, r2 = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[([^\]]+)\] R1 (\S+): s ([0-9.]+) .* mean P3 ([0-9.]+)", line)
        if m:
            r1[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))
        m = re.match(r"\[([^\]]+)\] R2 (\S+): share ([0-9.na]+) ", line)
        if m and m.group(3) != "nan":
            r2[(m.group(1), m.group(2))] = float(m.group(3))
    return r1, r2


def main():
    S = json.loads((RES / "scores.json").read_text(encoding="utf-8"))
    log_r1, log_r2 = log_values(RES / "score.log")
    rep = {"sources": {"scores": "results/part209/scores.json", "log": "results/part209/score.log"}}

    # ---- gates
    gates = {}
    for st in OLMO + QWEN:
        g = S["gates"][st]
        gates[st] = {"G0_mean_P3_told": g["G0"]["mean_P3_told"], "G0": g["G0"]["pass"],
                     "G1_r1": {a: g["G1_r1"][a] for a in ("TOLD-SIT", "TOLD-STAND", "TOLD-LIE")},
                     "G1_r1_pass": g["G1_r1"]["pass"], "G1_r3_pass": g["G1_r3"]["pass"],
                     "r3_heldout_accuracy": g["G1_r3"]["heldout_accuracy"], "r3_layer": g["G1_r3"]["chosen_layer"],
                     "read_r1": g["read_r1"], "read_r3": g["read_r3"]}
    g2 = S["gates"]["G2"]
    gates["G2"] = {"kappa_posture": g2["kappa_posture"], "kappa_says_no_body": g2["kappa_says_no_body"],
                   "kappa_valence": g2["kappa_valence"], "n_both": g2["n_both"], "pass": g2["pass"],
                   "checks": g2["checks"]}
    rep["gates"] = gates

    # ---- R1 (and its check against the printout)
    r1 = {}
    for st in OLMO + QWEN:
        r1[st] = {}
        for arm in R1_ARMS:
            v = S["r1"][st].get(arm)
            if v is None:
                continue
            r1[st][arm] = {"s": v["s"], "range": v["range"], "mean_P3": v["mean_P3"], "read": v["read"]}
            printed = log_r1.get((st, arm))
            assert printed is not None and abs(printed[0] - v["s"]) < 6e-4, (st, arm, printed, v["s"])
    rep["r1"] = r1
    rep["r1c_room_w"] = {st: S["r1c"][st]["ROOM-W"]["s"] for st in S["r1c"]}

    # ---- R2, with direct counts from the row and label files
    L_ds, L_hk = last_labels(RES / "labels-ds.jsonl"), last_labels(RES / "labels-haiku.jsonl")
    r2 = {}
    for st, fname in (("qwen-chat", "r2-qwen-chat.jsonl"), ("instruct", "r2-instruct.jsonl")):
        rows = [r for r in read_jsonl(RES / fname) if "answer" in r]
        arms = {}
        for arm in R2_ARMS:
            c = Counter()
            for r in rows:
                if r["arm"] != arm or r["row_id"] not in L_ds:
                    continue
                lab = L_ds[r["row_id"]]
                c["unreadable" if lab.get("unreadable") else lab["posture"]] += 1
            stored = S["r2"][st]["arms"][arm]
            assert dict(c) == {k: v for k, v in stored["counts"].items() if v}, (st, arm, dict(c), stored["counts"])
            named = sum(c[k] for k in ("sitting", "standing", "lying", "other"))
            arms[arm] = {"counts": {k: c.get(k, 0) for k in CATS}, "n": sum(c.values()), "named": named,
                         "sitting_share": stored["sitting_share"], "sitting_share_ci95": stored["sitting_share_ci95"],
                         "none_frac": stored["none_frac"], "none_frac_ci95": stored["none_frac_ci95"],
                         "negative_frac": stored["negative_frac"]}
            if stored["sitting_share"] == stored["sitting_share"]:  # not nan
                printed = log_r2.get((st, arm))
                assert printed is not None and abs(printed - stored["sitting_share"]) < 6e-4, (st, arm)
        r2[st] = {"n_answers": len(rows), "told_compliance": S["r2"][st]["told_compliance"],
                  "told_n": S["r2"][st]["told_n"], "arms": arms}
    common = sorted(set(L_ds) & set(L_hk))
    agree = sum(1 for i in common if (L_ds[i].get("posture"), bool(L_ds[i].get("unreadable")))
                == (L_hk[i].get("posture"), bool(L_hk[i].get("unreadable"))))
    rep["r2"] = r2
    rep["readers"] = {"labelled_ds": len(L_ds), "labelled_haiku": len(L_hk), "both": len(common),
                      "posture_agreement": agree}

    # ---- R3 (only the two registered, scored coordinates are stated; the rest are context)
    r3 = {}
    for st in OLMO + QWEN:
        v = S["r3"][st]
        r3[st] = {"layer": v.get("chosen_layer"), "heldout_accuracy": v.get("heldout_accuracy"),
                  "c_scored": v.get("c_scored"), "c_ci95": {a: v.get("c_ci95", {}).get(a) for a in ("ROOM-W", "NOBODY")},
                  "pole_gap_read": v.get("pole_gap_read")}
    rep["r3"] = r3

    # ---- verdicts: 209.1 to 209.15 from scores.json, 209.16 by its registered rule
    verdicts = {k: dict(v) for k, v in S["predictions"].items()}
    third, room = r1["pretrain-3.0T"]["THIRD"], r1["pretrain-3.0T"]["ROOM-W"]
    if third["mean_P3"] < P3_FLOOR or room["mean_P3"] < P3_FLOOR:
        verdicts["209.16"] = {"verdict": "not read", "reason": "mean P3 under 0.1"}
    else:
        gap = abs(third["s"] - room["s"])
        verdicts["209.16"] = {"verdict": "held" if third["s"] >= 0.5 and gap <= 0.15 else "failed",
                              "s_third": third["s"], "s_room_w": room["s"], "abs_gap": gap,
                              "p3_third": third["mean_P3"], "p3_room_w": room["mean_P3"],
                              "scored_by": "report209.py, the registered rule (score209 prints no verdict for it)"}
    rep["verdicts"] = verdicts
    rep["headline"] = S["headline"]
    rep["counts"] = {"held": sum(1 for v in verdicts.values() if v["verdict"] == "held"),
                     "failed": sum(1 for v in verdicts.values() if v["verdict"] == "failed"),
                     "not_read": sum(1 for v in verdicts.values() if v["verdict"] == "not read")}

    (RES / "report209.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    write_md(rep)
    print(f"wrote report209.json and report209.md; verdicts {rep['counts']}")


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def write_md(rep):
    L = ["# Part 209 report (made by part209/report209.py; every value from scores.json and the raw files)", ""]
    L += ["## Verdicts", "", "| # | Verdict | Numbers |", "| --- | --- | --- |"]
    for k in sorted(rep["verdicts"], key=lambda s: int(s.split(".")[1])):
        v = rep["verdicts"][k]
        nums = {a: b for a, b in v.items() if a not in ("verdict", "scored_by")}
        L.append(f"| {k} | {v['verdict']} | `{json.dumps(nums)[:300]}` |")
    L += ["", f"Totals: {rep['counts']}", ""]
    for st, title in (("qwen-chat", "Qwen3.5-4B"), ("instruct", "OLMo 2 1B Instruct")):
        R = rep["r2"][st]
        L += [f"## R2 free answers, {title} ({R['n_answers']} answers; told compliance "
              f"{R['told_compliance']:.3f} on {R['told_n']})", "",
              "| Arm | Sitting | Standing | Lying | Other | None | Unreadable | Sitting share (95% interval) | None fraction |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
        for arm, a in R["arms"].items():
            c = a["counts"]
            ci = a["sitting_share_ci95"]
            ci_t = "n/a" if not ci or ci[0] is None else f"{pct(ci[0])} to {pct(ci[1])}"
            L.append(f"| {arm} | {c['sitting']} | {c['standing']} | {c['lying']} | {c['other']} | {c['none']} | "
                     f"{c['unreadable']} | {pct(a['sitting_share'])} ({ci_t}) | {pct(a['none_frac'])} |")
        L.append("")
    L += ["## R1 forced choice, s = SIT / (SIT + STAND + LIE) after \"Right now I am\" (mean P3 in brackets)", "",
          "| Stage | " + " | ".join(R1_ARMS) + " |", "| --- |" + " ---: |" * len(R1_ARMS)]
    for st in OLMO + QWEN:
        cells = []
        for arm in R1_ARMS:
            v = rep["r1"][st].get(arm)
            cells.append("" if v is None else f"{v['s']:.3f} ({v['mean_P3']:.2f})")
        L.append(f"| {st} | " + " | ".join(cells) + " |")
    L += ["", "R1c (chat template, forced \"Right now I am\"), ROOM-W s: "
          + ", ".join(f"{k} {v:.3f}" for k, v in rep["r1c_room_w"].items()), ""]
    g = rep["gates"]["G2"]
    L += ["## Readers", "", f"Posture kappa {g['kappa_posture']:.3f}, says-no-body kappa {g['kappa_says_no_body']:.3f}, "
          f"valence kappa {g['kappa_valence']:.3f} on {g['n_both']} answers; same posture label on "
          f"{rep['readers']['posture_agreement']} of {rep['readers']['both']}.", ""]
    L += ["## R3 scored coordinates (c against each frame's own told poles)", ""]
    for st in OLMO + QWEN:
        v = rep["r3"][st]
        L.append(f"- {st}: layer {v['layer']}, held-out accuracy {v['heldout_accuracy']:.3f}, c {v['c_scored']}, "
                 f"intervals {v['c_ci95']}")
    (RES / "report209.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
