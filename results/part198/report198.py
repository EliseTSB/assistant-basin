#!/usr/bin/env python3
"""Print the tables in results/part198/README.md from the registered records (results/part198/records/).

Written 2026-11-09 by Journal Claude (Opus 5.5) after the gate failed. Every number in the README's tables comes from
this script's output; run it from the repository root:

    python3 results/part198/report198.py > results/part198/report198.md
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "records"
EDITORS = (("ROME", "armA-rome.jsonl"), ("MEMIT", "armA-memit.jsonl"), ("AlphaEdit", "armA-alphaedit.jsonl"))


def jsonl(name: str) -> list[dict]:
    rows = []
    with (REC / name).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cases(name: str) -> list[dict]:
    return [r for r in jsonl(name) if r.get("record_type") == "case" and r.get("run_mode", "registered") == "registered"]


def fmt(x: float | None, d: int = 2) -> str:
    return "none" if x is None else f"{x:.{d}f}"


def main() -> None:
    score = json.loads((REC / "score.json").read_text(encoding="utf-8"))
    gate = score["gate"]
    out: list[str] = []

    out.append("### Table 1. The take gate, per GPT-2 XL editor and for wsp\n")
    out.append("| Arm | Attempted | Already satisfied | Took | Required | Most that could take | Gate |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    sat_ids = {}
    for label, fname in EDITORS:
        cs = cases(fname)
        sat = sorted(r["case_id"] for r in cs if r.get("already_satisfied") is True)
        sat_ids[label] = sat
        req = gate["take_requirements"][f"armA_{label}"]
        out.append(f"| {label} | {req['attempted']} | {len(sat)} | {req['took']} | {req['required']} | "
                   f"{req['attempted'] - len(sat)} | {'passed' if req['passed'] else 'failed'} |")
    b = gate["take_requirements"]["armB_wsp"]
    out.append(f"| wsp | {b['attempted']} | 0 | {b['took']} | {b['required']} | {b['attempted']} | "
               f"{'passed' if b['passed'] else 'failed'} |")
    same = len({tuple(v) for v in sat_ids.values()}) == 1
    out.append("")
    out.append(f"Already-satisfied case ids: {sat_ids['ROME']} (the same five for all three editors: {same}).")
    out.append("")

    out.append("### Table 2. The five already-satisfied records under ROME (log-probabilities)\n")
    out.append("| Case | Prompt | New object | Old object | New before | Old before | New after |")
    out.append("| ---: | --- | --- | --- | ---: | ---: | ---: |")
    for r in cases("armA-rome.jsonl"):
        if r.get("already_satisfied") is True:
            out.append(f"| {r['case_id']} | {r['prompt']} | {r['target_new']} | {r['ground_truth']} | "
                       f"{fmt(r['before']['logp_new'])} | {fmt(r['before']['logp_old'])} | "
                       f"{fmt(r['after']['logp_new'], 3)} |")
    out.append("")

    out.append("### Table 3. Did the edits that did not count still move the answer?\n")
    out.append("| Editor | Not already satisfied, did not take | Already satisfied, new object leads after |")
    out.append("| --- | ---: | ---: |")
    for label, fname in EDITORS:
        cs = cases(fname)
        miss = [r["case_id"] for r in cs if r.get("already_satisfied") is not True and r.get("took") is not True]
        sat_after = sum(1 for r in cs if r.get("already_satisfied") is True and r["after"].get("new_leads") is True)
        out.append(f"| {label} | {len(miss)} {miss if miss else ''} | {sat_after} of {len(sat_ids[label])} |")
    out.append("")

    out.append("### Table 4. What the scorer counted (not verdicts: the gate failed first)\n")
    out.append("| Item | Denominator | Count |")
    out.append("| --- | ---: | ---: |")
    for f in score["forecasts"]:
        den = f.get("rate_denominator")
        out.append(f"| {f['forecast']} | {den} | {f['successes']} |")
    for k, v in score["descriptive"].items():
        if k == "198.2_gradient_tword_localization":
            for m, w in v.items():
                out.append(f"| 198.2 T_word localization, {m} (descriptive) | {w['took']} took of {w['attempted']} "
                           f"| {w['successes']} |")
        elif k == "198.5_aimed_control_localization":
            out.append(f"| 198.5 aimed control localized (descriptive) | {v['name_eligible']} | {v['successes']} |")
        elif k == "wsp_oracle_naming":
            out.append(f"| wsp oracle naming (diagnostic) | {v['rate_denominator']} | {v['successes']} |")
    s1, s2 = score["sensitivity_checks"]["S1"], score["sensitivity_checks"]["S2"]
    out.append(f"| S1 aimed control named (registered bar: at least {s1['threshold']}) | {s1['name_eligible']} | "
               f"{s1['successes']} |")
    out.append(f"| S2 matched random control localized (registered bar: at most {s2['threshold']}) | "
               f"{s2['denominator']} | {s2['successes']} |")
    out.append("")

    p = score["sensitivity_checks"]["paired_random_vs_real"]
    out.append("### Table 5. wsp writes against their matched random controls, T_word localization\n")
    out.append("| | Random localized | Random not localized |")
    out.append("| --- | ---: | ---: |")
    out.append(f"| wsp localized | {p['both']} | {p['real_only']} |")
    out.append(f"| wsp not localized | {p['random_only']} | {p['neither']} |")
    out.append("")

    out.append("### Table 6. Arm C, unedited models: largest eligible-token z (T_word), against the design reviewer's\n")
    out.append("| Model | Role | Stored dtype | Snapshot | Registered run | Layer, token | Reviewer (design text) |")
    out.append("| --- | --- | --- | --- | ---: | --- | --- |")
    tstar = json.loads((REC / "tstar.json").read_text(encoding="utf-8"))
    for r in jsonl("armC.jsonl"):
        if r.get("record_type") != "armC_model":
            continue
        snap = Path(str(r.get("snapshot", ""))).name[:8] or "local"
        ref = r.get("reviewer_reference") or {}
        tok = (r.get("top_eligible_pair") or {}).get("token")
        rv = ref.get("max_t_word")
        out.append(f"| {r['model_id']} | {r['role']} | {r.get('stored_dtype')} | {snap} | {fmt(r['max_t_word'])} | "
                   f"{r['top_t_word_layer']}, {tok!r} | {'' if rv is None else rv}"
                   f"{'' if ref.get('top_token') is None else ' ' + repr(ref['top_token'])} |")
    out.append("")
    ev = [r for r in jsonl("armC.jsonl") if r.get("record_type") == "armC_model" and r["role"] == "evaluation"]
    above = [r["model_id"] for r in ev if r["max_t_word"] is not None and r["max_t_word"] > tstar["t_star"]]
    out.append(f"T* (largest calibration value) = {tstar['t_star']:.2f}. Evaluation models above it: {len(above)} of "
               f"{len(ev)}{(' ' + str(above)) if above else ''}.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
