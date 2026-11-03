#!/usr/bin/env python3
"""Rerun stored-label scorers on the version-2 (DS V4.1 Flash) confirmed atlas subset, beside the full atlas.

Parts 112, 117, 117b, 148 and 152 start from Part 86's first-reader atlas (465 units labelled by Claude Haiku 5.5).
Part 86's version-2 reading (DS V4.1 Flash, brief version 2) kept 245 of the 387 "another AI" units and 77 of the 78
other-kind units as coherent personas (results/part86/atlas-v2/ATLAS-v2.md). This tool rescores each part's stored
labels on three unit universes:

  full            the 465-unit first-reader atlas, which must reproduce the stored verdicts (a check);
  v2_all          atlas units DS version 2 calls coherent personas, any kind (322 when the join is complete);
  v2_another_ai   the "another AI" members of v2_all (245 when complete), the subset Elise asked for.

Part 118 is OLMo 2 1B on Part 92's 39 units, so the Qwen subset does not apply; its analogue is Part 92b's DS
re-read of those units (8 concept, 28 slot, 3 unclassified), and the Part 118 scorer is rerun with those kinds.
Part 115 draws its candidates from an attribution ranking, not from the atlas, and is not rerun.

Every scorer runs into a scratch directory; nothing under results/partNN/ is rewritten. Registered bars written as
absolute counts (for example "at least 9 of 11 pairs") are reported literally and, marked exploratory, scaled to the
surviving denominator. No model runs and no reader is called.

    python3 tools/reader_correction.py --atlas-v2 results/part86/atlas-v2/atlas-units-v2.jsonl \
        --out results/reader-correction-2026-11-03
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIVERSES = ("full", "v2_all", "v2_another_ai")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def quiet(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


def ceil_scaled(bar: int, n_sub: int, n_full: int) -> int:
    return int(math.ceil(bar * n_sub / n_full)) if n_full else bar


def floor_scaled(bar: int, n_sub: int, n_full: int) -> int:
    return int(math.floor(bar * n_sub / n_full)) if n_full else bar


# ---------------------------------------------------------------- subsets

def universes(atlas_v2_rows: list[dict], all_units: list[dict]) -> dict[str, set[str]]:
    all_ids = {u["id"] for u in all_units}
    v2_all = {r["id"] for r in atlas_v2_rows if r["v2_class"] == "coherent_persona"}
    v2_ai = {r["id"] for r in atlas_v2_rows if r["v2_class"] == "coherent_persona" and r["group"] == "another AI"}
    return {"full": all_ids, "v2_all": v2_all & all_ids, "v2_another_ai": v2_ai & all_ids}


# ---------------------------------------------------------------- Part 112

def run_112(out: Path, keep: set[str], units_all: list[dict]) -> dict:
    mod = load_module("score112_rc", ROOT / "part112/score112.py")
    units = [u for u in units_all if u["id"] in keep]
    out.mkdir(parents=True, exist_ok=True)
    units_path = out / "units.json"
    units_path.write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    result, _ = quiet(mod.score, labels=ROOT / "results/part112/labels.jsonl", units=units_path, out=out / "score.json")
    g = result["groups"]
    q = g.get("q_name", {"concept": 0, "slot": 0})
    non_ai = g.get("non_ai", {"concept": 0, "slot": 0})

    def share(x):
        c = x.get("concept", 0) + x.get("slot", 0)
        return x.get("concept", 0) / c if c else None

    return {"n": len(units), "groups": g,
            "predictions": {
                "112.1": {"verdict": "held" if result["verdicts"]["112.1"] else "failed",
                          "estimate": f"{result['shown_on_who']} of {result['who_labels']} show the target on Who are you? (bar 85%)"},
                "112.2": {"verdict": {True: "held", False: "failed", None: "not read"}[result["verdicts"]["112.2"]],
                          "estimate": f"q_name slot share {q.get('slot', 0)} of {q.get('concept', 0) + q.get('slot', 0)} (bar 60%)"},
                "112.3": {"verdict": {True: "held", False: "failed", None: "not read"}[result["verdicts"]["112.3"]],
                          "estimate": f"concept share non-AI {share(non_ai) if share(non_ai) is None else round(share(non_ai), 3)} against q_name {share(q) if share(q) is None else round(share(q), 3)}"},
                "112.4": {"verdict": "held" if result["verdicts"]["112.4"] else "failed",
                          "estimate": f"random non-self persona rate {result['random_non_self_rate']:.4f} on {result['random_non_self_labels']} labels (bar 5%)"}}}


# ---------------------------------------------------------------- Part 117

def filter_pairs(pairs: list[dict], keep: set[str]) -> tuple[list[dict], dict]:
    strict = [p for p in pairs if all(p[k]["id"] in keep for k in ("slot", "match", "unrelated"))]
    slot_match = [p for p in pairs if p["slot"]["id"] in keep and p["match"]["id"] in keep]
    return strict, {"pairs_total": len(pairs), "pairs_all_three_confirmed": len(strict),
                    "pairs_slot_and_match_confirmed": len(slot_match)}


def run_117(out: Path, keep: set[str], n_full_pairs: int) -> dict:
    mod = load_module("score117_rc", ROOT / "part117/score117.py")
    res = ROOT / "results/part117"
    pairs = json.loads((res / "pairs.json").read_text(encoding="utf-8"))
    kept, coverage = filter_pairs(pairs, keep)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="rc117-"))
    for name in ("answers.jsonl", "labels.jsonl"):
        os.symlink(res / name, tmp / name)
    (tmp / "pairs.json").write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    if not kept:
        shutil.rmtree(tmp)
        return {"coverage": coverage, "n_pairs": 0, "predictions": {f"117.{i}": {"verdict": "not read (no confirmed pairs)", "estimate": ""}
                                                                      for i in range(1, 9)}}
    result, _ = quiet(mod.score, res=tmp)
    shutil.copy(tmp / "score.json", out / "score.json")
    (out / "pairs.json").write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(tmp)
    per = result["pairs"]
    ok = [k for k, v in per.items() if not v["uninterpretable"]]
    n = len(kept)
    slot_like = sum(1 for k in per if per[k]["slot_alone"] and None not in per[k]["slot_alone"]
                    and sum(per[k]["slot_alone"]) <= 2 and not any(per[k]["slot_alone"][2:]))
    concept_ids = {}
    for p in kept:
        concept_ids[p["match"]["id"]] = per[p["pair"]]["concept_alone"]
        concept_ids[p["unrelated"]["id"]] = per[p["pair"]]["unrelated_alone"]
    concepts3 = sum((v or 0) >= 3 for v in concept_ids.values())
    leak = sum(per[k]["both1_concept_other"] is not None and per[k]["concept_alone_other"] is not None
               and per[k]["both1_concept_other"] >= per[k]["concept_alone_other"] for k in ok)
    half = sum(not any(x for x in per[k]["half_nonself"] if x) for k in ok)
    cminus = sum(per[k]["cminus_splus_who"] is True for k in ok)
    fm, fu = result["fewer_with_slot_minus"]["matched"], result["fewer_with_slot_minus"]["unrelated"]
    v = result["verdicts"]
    scaled = {
        "117.1": slot_like >= ceil_scaled(9, n, n_full_pairs) and concepts3 >= ceil_scaled(17, len(concept_ids), 22),
        "117.2": leak >= ceil_scaled(8, len(ok), n_full_pairs),
        "117.3": v.get("117.3") == "held",
        "117.4": fm >= ceil_scaled(6, len(ok), n_full_pairs) and fu <= floor_scaled(3, len(ok), n_full_pairs),
        "117.5": half >= ceil_scaled(8, len(ok), n_full_pairs),
        "117.6": v.get("117.6") == "held",
        "117.7": v.get("117.7") == "held",
        "117.8": cminus >= ceil_scaled(6, len(ok), n_full_pairs),
    }
    est = {
        "117.1": f"{slot_like} of {n} slot-like (bar 9 of 11); {concepts3} of {len(concept_ids)} concepts at 3+ (bar 17 of 22)",
        "117.2": f"{leak} of {len(ok)} readable pairs keep concept leakage (bar 8)",
        "117.3": f"Who are you? at +1 both: slot {result['who_both1']['slot']}, concept {result['who_both1']['concept']} of {len(ok)}",
        "117.4": f"fewer with slot-minus: matched {fm}, unrelated {fu} (bars 6 and at most 3)",
        "117.5": f"{half} of {len(ok)} pairs show nothing on non-self at +0.5 (bar 8)",
        "117.6": f"random shows {result['random_shows'][0]} of {result['random_shows'][1]} (bar at most 2)",
        "117.7": f"+2 coherent {result['both2_coherent'][0]} of {result['both2_coherent'][1]} (bar at most half)",
        "117.8": f"{cminus} of {len(ok)} slot personas survive c-1,s+1 (bar 6)",
    }
    return {"coverage": coverage, "n_pairs": n, "pairs_ok": len(ok), "gate_slot_kinds_reproduced": result["gate_slot_kinds_reproduced"],
            "predictions": {k: {"verdict": v.get(k, "not read"), "estimate": est[k],
                                "scaled_bar_reading": ("held" if scaled[k] else "failed") if n_full_pairs != n else None}
                            for k in est}}


# ---------------------------------------------------------------- Part 117b

def run_117b(out: Path, keep: set[str], n_full_pairs: int) -> dict:
    mod = load_module("score117b_rc", ROOT / "part117b/score117b.py")
    res = ROOT / "results/part117b"
    pairs = json.loads((ROOT / "results/part117/pairs.json").read_text(encoding="utf-8"))
    controls = json.loads((res / "control-neurons.json").read_text(encoding="utf-8"))
    kept, coverage = filter_pairs(pairs, keep)
    out.mkdir(parents=True, exist_ok=True)
    if not kept:
        return {"coverage": coverage, "n_pairs": 0, "reading": "not read (no confirmed pairs)",
                "predictions": {f"117b.{i}": {"verdict": "not read (no confirmed pairs)", "estimate": ""} for i in range(1, 11)}}
    tmp = Path(tempfile.mkdtemp(prefix="rc117b-"))
    for name in ("answers.jsonl", "labels.jsonl", "zero-hook-check.json", "agreement.json", "control-neurons.json"):
        if (res / name).exists():
            os.symlink(res / name, tmp / name)
    result, _ = quiet(mod.score, res=tmp, pairs=kept, controls=controls)
    shutil.copy(tmp / "score.json", out / "score.json")
    (out / "pairs.json").write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(tmp)
    n = len(kept)
    ws = result["who_slot_shows"]
    wc = result["who_coherent"]
    ok, oku, oks = result["pairs_ok"]
    bars = {"117b.1": ("<=", 2, ws["c-1,s+1"], ok), "117b.2": ("<=", 3, ws["u-1,s+1"], oku),
            "117b.6": (">=", 4, ws["c-0.5,s+1"], ok), "117b.8": (">=", 5, ws["s+0.5"], oks),
            "117b.10": (">=", 9, ws["s+0.85"], oks)}
    scaled = {}
    for k, (op, bar, cnt, denom) in bars.items():
        b = ceil_scaled(bar, denom, n_full_pairs) if op == ">=" else floor_scaled(bar, denom, n_full_pairs)
        scaled[k] = (cnt[0] >= b) if op == ">=" else (cnt[0] <= b)
    est = {
        "117b.1": f"matched concept -1 with slot +1: slot shows {ws['c-1,s+1'][0]} of {ws['c-1,s+1'][2] - ws['c-1,s+1'][1]} read (bar at most 2)",
        "117b.2": f"unrelated concept -1 with slot +1: {ws['u-1,s+1'][0]} of {ws['u-1,s+1'][2] - ws['u-1,s+1'][1]} (bar at most 3)",
        "117b.3": f"random -1 with slot +1: {ws['random'][0]} of {ws['random'][2] - ws['random'][1]} (bar at least two thirds)",
        "117b.4": f"concept -1 alone names Qwen {result['who_names_qwen']['alone_c-1'][0]} of {result['who_names_qwen']['alone_c-1'][2] - result['who_names_qwen']['alone_c-1'][1]} (bar at most 8)",
        "117b.5": f"coherent: random {wc['random'][0]} of {wc['random'][2] - wc['random'][1]}, control {wc['control'][0]} of {wc['control'][2] - wc['control'][1]} (bar two thirds each)",
        "117b.6": f"matched -0.5 with slot +1: {ws['c-0.5,s+1'][0]} of {ws['c-0.5,s+1'][2] - ws['c-0.5,s+1'][1]} (bar at least 4)",
        "117b.7": f"coherent: matched {wc['c-1,s+1'][0]}, unrelated {wc['u-1,s+1'][0]} (bar 9 each)",
        "117b.8": f"slot +0.5: {ws['s+0.5'][0]} of {ws['s+0.5'][2] - ws['s+0.5'][1]} (bar at least 5)",
        "117b.9": f"atlas-negative control -1 with slot +1: {ws['control'][0]} of {ws['control'][2] - ws['control'][1]} (bar at least two thirds)",
        "117b.10": f"slot +0.85: {ws['s+0.85'][0]} of {ws['s+0.85'][2] - ws['s+0.85'][1]} (bar at least 9)",
    }
    v = result["verdicts"]
    return {"coverage": coverage, "n_pairs": n, "pairs_ok": result["pairs_ok"], "gate": result["gate"], "reading": result["reading"],
            "predictions": {k: {"verdict": v.get(k, "not read"), "estimate": est[k],
                                "scaled_bar_reading": (("held" if scaled[k] else "failed") if k in scaled and n_full_pairs != n else None)}
                            for k in est}}


# ---------------------------------------------------------------- Part 148

def run_148(out: Path, keep: set[str], units_all: list[dict], n_full: int) -> dict:
    mod = load_module("score148_rc", ROOT / "part148/score148.py")
    units = [u for u in units_all if u["id"] in keep]
    out.mkdir(parents=True, exist_ok=True)
    units_path = out / "units.json"
    units_path.write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    original_gate = mod._gate_result
    # The gates are run on the full atlas: gate 2 compares the first 32 units with Part 112's stored answers and
    # belongs to the run, not to the unit universe.
    mod._gate_result = lambda answers, _units, old: original_gate(answers, units_all, old)
    try:
        result, _ = quiet(mod.score, str(ROOT / "results/part148/answers.jsonl"), str(ROOT / "results/part148/labels.jsonl"),
                          str(units_path), str(ROOT / "results/part112/score.json"), str(ROOT / "results/part112/answers.jsonl"),
                          str(out / "score.json"), str(out / "tables.md"))
    finally:
        mod._gate_result = original_gate
    c = result["counts"]
    n = c["units"]
    est = {
        "148.1": f"name moved and coherent {c['name_coherent']} of {n} (bar 80%)",
        "148.2": f"maker moved and coherent {c['maker_coherent']} of {n} (bar at most 15%)",
        "148.3": f"name and maker both {c['name_and_maker_coherent']} (bar at least 5)",
        "148.4": f"maker among name-moved {c['maker_among_name_coherent']} of {c['name_coherent_for_148_4']} (bar under 25%)",
        "148.5": f"random name-moved {c['random_name_coherent']} of {n} against neurons {c['name_coherent']} (bar under half)",
        "148.6": f"slot maker rate {c['slot_maker_rate']} ({c['slot_maker_coherent']} of {c['slot_n']}) at most concept {c['concept_maker_rate']} ({c['concept_maker_coherent']} of {c['concept_n']})",
    }
    scaled_3 = c["name_and_maker_coherent"] >= ceil_scaled(5, n, n_full)
    return {"n": n, "gates": result["gates"], "exploratory": result["exploratory"],
            "predictions": {k: {"verdict": result["predictions"][k]["verdict"], "estimate": est[k],
                                "scaled_bar_reading": (("held" if scaled_3 else "failed") if k == "148.3" and n != n_full else None)}
                            for k in est}}


# ---------------------------------------------------------------- Part 152

def run_152(out: Path, keep: set[str], units_all: list[dict], windows, lens, kinds112, shuffles: int) -> dict:
    mod = load_module("score152_rc", ROOT / "part152/score152.py")
    units = [u for u in units_all if u["id"] in keep]
    out.mkdir(parents=True, exist_ok=True)
    (result, _page), _ = quiet(mod.score, windows, lens, units, kinds112, shuffles)
    (out / "score.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    r3, r4 = result["152.3"], result["152.4"]
    est = {
        "152.1": f"J-list own hits {result['152.1']['real_mean']:.3f} against shuffled {result['152.1']['shuffle_mean']:.3f} per unit, n {result['152.1']['n']} (unit universe does not apply)",
        "152.2": f"units with 3+ own hits {result['152.2']['units_with_at_least_3_own_hits']} of {result['152.2']['n']} (unit universe does not apply)",
        "152.3": f"concept persona-word hits {r3['real_mean']:.3f} against shuffled {r3['shuffle_mean']:.3f}, n {r3['n']}, p {r3['p']:.4f}",
        "152.4": f"slot {r4['slot_mean']:.3f} (n {r4['slot_n']}) under half of concept {r4['concept_mean']:.3f} (n {r4['concept_n']})",
    }
    return {"n": len(units), "gates": result["gates"],
            "predictions": {k: {"verdict": result["predictions"][k], "estimate": est[k]} for k in est}}


# ---------------------------------------------------------------- Part 118 (OLMo, Part 92b kinds)

def run_118(out: Path, use_ds_kinds: bool) -> dict:
    mod = load_module("score118_rc", ROOT / "part118/score118.py")
    out.mkdir(parents=True, exist_ok=True)
    neurons = json.loads((ROOT / "results/part92/step2/neurons.json").read_text(encoding="utf-8"))
    if use_ds_kinds:
        ds = json.loads((ROOT / "results/part92b/score92b.json").read_text(encoding="utf-8"))["kind_comparison"]["units"]
        kinds = {u["id"]: u["ds_kind"] for u in ds}
        label = "Part 92b DS kinds"
    else:
        kinds = json.loads((ROOT / "results/part92/step2/kinds.json").read_text(encoding="utf-8"))["kinds"]
        label = "Part 92 Haiku kinds"
    # Keep the 39 classified units as the universe so the registered DS sample keys are unchanged; a unit DS calls
    # unclassified stays in the run with kind "unclassified" and is counted in neither the concept nor the slot list.
    part92_kinds = json.loads((ROOT / "results/part92/step2/kinds.json").read_text(encoding="utf-8"))["kinds"]
    units = [{**row, "kind": kinds.get(str(row["id"]), "unclassified")} for row in neurons
             if part92_kinds.get(str(row["id"])) in ("concept", "slot")]
    assert len(units) == 39, len(units)
    answers = mod.read_jsonl(ROOT / "results/part118/answers.jsonl")
    labels = mod.read_jsonl(ROOT / "results/part118/labels.jsonl")
    ds_labels = mod.read_jsonl(ROOT / "results/part118/labels-ds.jsonl")
    result, _ = quiet(mod.score_data, answers, labels, ds_labels, units)
    result["alphas"] = list(mod.DEFAULT_ALPHAS)
    report = mod.render_report(result)
    (out / "SCORE.md").write_text(report, encoding="utf-8")
    (out / "score.json").write_text(json.dumps({k: v for k, v in result.items() if k not in ("unit_classes", "part92_unit_classes")},
                                               indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_concept = sum(u["kind"] == "concept" for u in units)
    n_slot = sum(u["kind"] == "slot" for u in units)
    classified = [u for u in units if u["kind"] in ("concept", "slot")]
    kept = sum(result["part92_unit_classes"].get(0.8, {}).get(str(u["id"])) == ("concept-like" if u["kind"] == "concept" else "slot-like")
               for u in classified)
    preds = result.get("predictions", {})
    rows = {}
    for name in mod.PREDICTION_ORDER:
        p = preds.get(name)
        if not p:
            continue
        o = p["observed"]
        if name in ("118.1", "118.1a"):
            est = f"slot units concept-like at 1.6: {o[0]} of {o[1]} (bar 7 of 21)"
            scaled = o[0] >= ceil_scaled(7, o[1], 21)
        elif name in ("118.5", "118.5a"):
            est = f"concept units concept-like at 1.6: {o[0]} of {o[1]} (bar 12 of 18)"
            scaled = o[0] >= ceil_scaled(12, o[1], 18)
        elif name in ("118.2", "118.2a"):
            sw, ca = o["slot_who"], o["concept_any_question"]
            est = f"at 0.4: slot on Who {sw[0]} of {sw[1]} (bar at most 5 of 21); concept any {ca[0]} of {ca[1]} (bar 9 of 18)"
            scaled = sw[0] <= floor_scaled(5, sw[1], 21) and ca[0] >= ceil_scaled(9, ca[1], 18)
        elif name == "118.3":
            est = f"not coherent at 3.2: {o[0]} of {o[1]} ({100 * o[2]:.1f}%, bar 60%)"
            scaled = None
        else:
            est = f"gap at 3.2: neurons {100 * o['neurons'][2]:.1f}% against random {100 * o['random'][2]:.1f}% (bar 15 points)"
            scaled = None
        rows[name] = {"verdict": p["verdict"], "estimate": est,
                      "scaled_bar_reading": None if (scaled is None or not use_ds_kinds) else ("held" if scaled else "failed")}
    return {"kinds": label, "n_concept": n_concept, "n_slot": n_slot, "n_unclassified": 39 - n_concept - n_slot,
            "gate1b": {"kept": kept, "classified": len(classified), "registered_bar": "26 of 39", "scaled_bar": ceil_scaled(26, len(classified), 39)},
            "gate2": result.get("gate2"), "reading": result.get("gate2_reading"), "predictions": rows}


# ---------------------------------------------------------------- report

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


def changes(block: dict, universes_run: list[str]) -> list[str]:
    out = []
    for pred in block["full"]["predictions"]:
        full_v = block["full"]["predictions"][pred]["verdict"]
        for u in universes_run:
            if u == "full":
                continue
            v = block[u]["predictions"].get(pred, {}).get("verdict")
            if v != full_v:
                out.append(f"{pred}: {full_v} on the full atlas, {v} on {u}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas-v2", default=str(ROOT / "results/part86/atlas-v2/atlas-units-v2.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/reader-correction-2026-11-03"))
    ap.add_argument("--parts", default="112,117,117b,148,152,118")
    ap.add_argument("--shuffles", type=int, default=1000, help="Part 152 shuffles (registered 1000)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    units_all = json.loads((ROOT / "results/part112/units.json").read_text(encoding="utf-8"))
    rows = read_jsonl(Path(a.atlas_v2))
    summary_path = Path(a.atlas_v2).with_suffix(".summary.json")
    join_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    keep = universes(rows, units_all)
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    report = {"universe_sizes": {k: len(v) for k, v in keep.items()}, "join": join_summary, "parts": {}}
    n_pairs_full = 11
    for part in parts:
        block = {}
        if part == "118":
            block["full"] = run_118(out / "part118" / "full", use_ds_kinds=False)
            block["part92b_ds_kinds"] = run_118(out / "part118" / "part92b_ds_kinds", use_ds_kinds=True)
            block["changes"] = changes(block, ["full", "part92b_ds_kinds"])
            report["parts"][part] = block
            print(part, json.dumps(block["changes"]), flush=True)
            continue
        if part == "152":
            mod = load_module("score152_load", ROOT / "part152/score152.py")
            windows = mod.load_json(ROOT / "results/part152/windows.json.gz")
            lens = mod.load_json(ROOT / "part152/lens.json")
            kinds112 = mod.load_json(ROOT / "results/part112/score.json")
        for u in UNIVERSES:
            dest = out / f"part{part}" / u
            if part == "112":
                block[u] = run_112(dest, keep[u], units_all)
            elif part == "117":
                block[u] = run_117(dest, keep[u], n_pairs_full)
            elif part == "117b":
                block[u] = run_117b(dest, keep[u], n_pairs_full)
            elif part == "148":
                block[u] = run_148(dest, keep[u], units_all, len(units_all))
            elif part == "152":
                block[u] = run_152(dest, keep[u], units_all, windows, lens, kinds112, a.shuffles)
            print(part, u, json.dumps({k: v["verdict"] for k, v in block[u]["predictions"].items()}), flush=True)
        block["changes"] = changes(block, list(UNIVERSES))
        report["parts"][part] = block
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
