#!/usr/bin/env python3
"""Recompute Part 200's scored and exploratory report from the stored result files; no model is loaded."""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT / "results" / "part200"
PUB = ROOT / "results" / "part163"
sys.path[:0] = [str(ROOT), str(ROOT / "part86")]
from part163 import score163  # noqa: E402
import triage86  # noqa: E402

TOKENIZER = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PREDICTIONS = [
    ("200.1", "Gates R1 to R3 and Part 163's three gates pass.", "medium"),
    ("200.2", "163.1, 163.2, 163.3 and 163.4 hold and 163.5 fails, as published.", "medium: 163.1 and 163.2 cannot fail unless a random rate falls below 2/3, and 163.3's bar is far below the published rates, so the live parts are 163.4 (one unit from flipping) and 163.5."),
    ("200.3", "The neuron arm's broken count over its 1,600 battery answers is at least twice the random arm's (published 343 and 83).", "medium: Part 199 kept its between-arm direction on eager while its rates moved."),
    ("200.4", "In each arm at least 480 of 1,600 battery answers (30%) reach the 80-token cap (published 124 neuron, 218 random).", "medium: on SDPA almost every Part 199 answer stopped early; on eager most ran to the cap."),
    ("200.5", "Each arm's broken count is higher on eager than published.", "low: Part 199's loop rates rose on eager, but Part 163's triage keeps any answer naming a model or identity, and a kernel can remove loops as well as add them."),
    ("200.6", "Under Part 161's rule on maker phrasings, the neuron arm's pooled Qwen rate stays below the random arm's (published 19 of 539 against 57 of 661).", "low: the published rates are small and the comparison was unregistered."),
    ("200.7", "At most 20 of the 200 batch-1 answers differ from the same units' chunked answers.", "low: rows are pushed independently, but eager batch 8 left the batch-1 path at a near-tie on 1 of 8 kernel-check prompts."),
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def qwen161(answer):
    # results/part163/extra163.py: the Part 161 rule used on maker phrasings.
    exact = re.search(r"\bQwen\b", answer if isinstance(answer, str) else "") is not None
    excluded = re.search(r"alibaba|tongyi|通义", answer if isinstance(answer, str) else "", re.I) is not None
    return exact and not excluded and triage86.triage_answer(answer or "") != "broken"


def answer_label(answer):
    return {"name_gone": score163.name_gone(answer),
            "broken": score163.triage(answer) == "broken",
            "qwen_part161_rule": qwen161(answer)}


def label_diff(a, b):
    la, lb = answer_label(a), answer_label(b)
    return {k: {"chunked": la[k], "batch1": lb[k], "changed": la[k] != lb[k]} for k in la}


def get_tokenizer():
    if TOKENIZER.exists():
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=True), "qwen3.5-4b-local-cache"
        except Exception as exc:  # fall back only when local tokenizer is not usable
            return None, f"characters (local tokenizer unusable: {type(exc).__name__}: {exc})"
    return None, "characters (Qwen3.5-4B tokenizer not present in local Hugging Face cache)"


def first_diff(a, b, tokenizer):
    if tokenizer is None:
        aa, bb = list(a), list(b)
        unit = "character"
    else:
        aa = tokenizer.encode(a, add_special_tokens=False)
        bb = tokenizer.encode(b, add_special_tokens=False)
        unit = "token"
    i = next((i for i, (x, y) in enumerate(zip(aa, bb)) if x != y), None)
    if i is None and len(aa) != len(bb):
        i = min(len(aa), len(bb))
    return i, len(aa), len(bb), unit


def build_report():
    score = read_json(NEW / "score.json")
    compare = read_json(NEW / "compare.json")
    pub_score = read_json(PUB / "score.json")
    new_rows, pub_rows = read_jsonl(NEW / "answers.jsonl"), read_jsonl(PUB / "answers.jsonl")
    batch_rows = read_jsonl(NEW / "batch1.jsonl")
    units = read_json(ROOT / "part163/units163.json")
    units_by_id = {u["unit_id"]: u for u in units}

    def battery_by_key(rows):
        return {(r["unit_id"], int(r["question_index"])): r for r in rows if r.get("stage") == "battery"}

    nb, pb = battery_by_key(new_rows), battery_by_key(pub_rows)
    batch = {r["unit_id"]: r for r in batch_rows}
    counts = {}
    for label, indexed in (("published", pb), ("rerun", nb)):
        counts[label] = {}
        for arm in ("neuron", "random"):
            rows = [r for (uid, _), r in indexed.items() if units_by_id[uid]["arm"] == arm]
            counts[label][arm] = {
                "battery_answers": len(rows),
                "broken": sum(score163.triage(r.get("answer", "")) == "broken" for r in rows),
                "at_cap": sum(int(r.get("token_count", 0)) == 80 for r in rows),
                "mean_tokens": statistics.mean(int(r.get("token_count", 0)) for r in rows),
            }

    # Recompute Part 161's pooled maker-Qwen counts using its published rule.
    qwen = {}
    for label, indexed in (("published", pb), ("rerun", nb)):
        qwen[label] = {}
        for arm in ("neuron", "random"):
            eligible = [r for (uid, _), r in indexed.items() if units_by_id[uid]["arm"] == arm and r.get("kind") == "maker"
                        and triage86.triage_answer(r.get("answer", "")) != "broken"
                        and not re.search(r"alibaba|tongyi|通义", r.get("answer", ""), re.I)]
            hits = sum(bool(re.search(r"\bQwen\b", r.get("answer", ""))) for r in eligible)
            qwen[label][arm] = {"qwen": hits, "eligible_maker_answers": len(eligible),
                                "maker_answers": sum(1 for (uid, _), r in indexed.items() if units_by_id[uid]["arm"] == arm and r.get("kind") == "maker"),
                                "rate": hits / len(eligible) if eligible else None}

    tokenizer, tokenizer_method = get_tokenizer()
    try:
        # Match common86.token_inputs: render the same prompt, then tokenize with add_special_tokens=False.
        from part86 import common86
        prompt = common86.build_prompt(tokenizer, "What is your name?") if tokenizer is not None else ""
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) if tokenizer is not None else 0
    except Exception as exc:
        tokenizer, tokenizer_method, prompt_len = None, f"characters (prompt tokenization unavailable: {type(exc).__name__}: {exc})", 0

    diffs = []
    for uid, br in batch.items():
        cr = nb[(uid, int(br["question_index"]))]
        if br.get("answer", "") == cr.get("answer", ""):
            continue
        pos, chunk_len, batch_len, unit = first_diff(cr.get("answer", ""), br.get("answer", ""), tokenizer)
        prompt_offset = prompt_len if tokenizer is not None else len("What is your name?")
        absolute = None if pos is None else prompt_offset + pos
        diffs.append({"unit_id": uid, "arm": units_by_id[uid]["arm"], "layer": units_by_id[uid]["layer"],
                      "first_answer_difference": pos, "first_difference_including_prompt": absolute,
                      "position_unit": unit, "chunked_length": chunk_len, "batch1_length": batch_len,
                      "before_or_at_64_keys": None if absolute is None else absolute <= 64,
                      "labels": label_diff(cr.get("answer", ""), br.get("answer", "")),
                      "chunked_excerpt": cr.get("answer", "")[:300], "batch1_excerpt": br.get("answer", "")[:300]})
    by_arm = {}
    for arm in ("neuron", "random"):
        vals = [r for r in diffs if r["arm"] == arm]
        positions = [r["first_difference_including_prompt"] for r in vals if r["first_difference_including_prompt"] is not None]
        by_arm[arm] = {"changed_rows": len(vals), "first_difference_before_or_at_64_keys": sum(r["before_or_at_64_keys"] is True for r in vals),
                       "first_difference_after_64_keys": sum(r["before_or_at_64_keys"] is False for r in vals),
                       "median_position_including_prompt": statistics.median(positions) if positions else None,
                       "minimum_position_including_prompt": min(positions) if positions else None,
                       "maximum_position_including_prompt": max(positions) if positions else None}
    label_changes = {label: sum(r["labels"][label]["changed"] for r in diffs) for label in ("name_gone", "broken", "qwen_part161_rule")}

    compare_scored = compare["scored"]
    report = {
        "source_files": {"score": "score.json", "compare": "compare.json", "answers": "answers.jsonl", "batch1": "batch1.jsonl",
                         "published_score": "../part163/score.json", "published_answers": "../part163/answers.jsonl"},
        "launch": read_jsonl(NEW / "launch.jsonl"),
        "load": read_json(NEW / "load.json"),
        "gates": {"r1_r3": compare_scored["record_r1_r3"], "r2": compare_scored["gate_r2"], "part163": score["gates"]},
        "part163_verdicts": {k: {"published": pub_score["predictions"][k], "rerun": score["predictions"][k]} for k in ("163.1", "163.2", "163.3", "163.4", "163.5")},
        "counts": counts,
        "part161_qwen": qwen,
        "batch1": {"compare": compare_scored["batch1"], "positions": {"method": tokenizer_method,
                    "prompt_tokens": prompt_len if tokenizer is not None else None,
                    "prompt_character_offset_used": len("What is your name?") if tokenizer is None else None,
                    "changed_rows_by_arm": by_arm, "label_changes": label_changes, "rows": diffs}},
        "predictions": compare_scored["predictions"],
        "exploratory": compare["exploratory"],
    }
    return report


def fmt(v, n=3):
    return "n/a" if v is None else f"{v:.{n}f}"


def render_md(r):
    md = ["# Part 200 report (computed by `part200/report200.py`)", "", "## Gates", "",
          "| Gate | Result | Evidence |", "| --- | --- | --- |"]
    md.append(f"| R1 and R3 | {'passed' if r['gates']['r1_r3']['passed'] else 'failed'} | {r['gates']['r1_r3'].get('why') or 'load record and observed call shapes'} |")
    md.append(f"| R2 | {'passed' if r['gates']['r2']['passed'] else 'failed'} | units hash matches design: {r['gates']['r2']['units_hash_matches_design']}; questions identical: {r['gates']['r2']['questions_identical']} |")
    for name, gate in r["gates"]["part163"].items():
        md.append(f"| Part 163 {name} | {'passed' if gate['passed'] else 'failed'} | `{json.dumps({k:v for k,v in gate.items() if k != 'passed'}, ensure_ascii=False)}` |")
    md += ["", "## Part 163 registered verdicts: published beside rerun", "",
           "| Prediction | Published verdict and estimate (95% interval) | Rerun verdict and estimate (95% interval) |",
           "| --- | --- | --- |"]
    for k, item in r["part163_verdicts"].items():
        cells = []
        for side in ("published", "rerun"):
            p = item[side]
            if p.get("estimate_ratio") is not None:
                ci = p.get("ratio_interval_95") or [None, None]
                val = f"{p['verdict']} ({p['estimate_ratio']:.3f}; {fmt(ci[0])} to {fmt(ci[1])})"
            elif p.get("rate_by_arm"):
                vals = p["rate_by_arm"]
                vals_s = ", ".join(f"{a} {d['mean']:.3f} ({fmt(d['interval_95'][0])} to {fmt(d['interval_95'][1])})" for a,d in vals.items())
                val = f"{p['verdict']} ({vals_s})"
            elif p.get("name_gone_units_at_least_half"):
                ng, bg = p["name_gone_units_at_least_half"], p["both_gone_units_at_least_half"]
                val = (f"{p['verdict']} (name gone at least half: neuron {ng['neuron']}, random {ng['random']}; "
                       f"both gone: neuron {bg['neuron']}, random {bg['random']})")
            else:
                val = f"{p['verdict']} (counts in score files)"
            cells.append(val)
        md.append(f"| {k} | {cells[0]} | {cells[1]} |")
    md += ["", "## Battery broken and at-cap counts", "", "| Arm | Published broken / 1,600 | Rerun broken / 1,600 | Published at cap / 1,600 | Rerun at cap / 1,600 |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ("neuron", "random"):
        md.append(f"| {arm} | {r['counts']['published'][arm]['broken']} / {r['counts']['published'][arm]['battery_answers']} | {r['counts']['rerun'][arm]['broken']} / {r['counts']['rerun'][arm]['battery_answers']} | {r['counts']['published'][arm]['at_cap']} / {r['counts']['published'][arm]['battery_answers']} | {r['counts']['rerun'][arm]['at_cap']} / {r['counts']['rerun'][arm]['battery_answers']} |")
    md += ["", "## Part 161 rule on maker phrasings", "", "| Arm | Published Qwen / eligible | Rerun Qwen / eligible | Published rate | Rerun rate |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ("neuron", "random"):
        p, n = r["part161_qwen"]["published"][arm], r["part161_qwen"]["rerun"][arm]
        md.append(f"| {arm} | {p['qwen']} / {p['eligible_maker_answers']} | {n['qwen']} / {n['eligible_maker_answers']} | {fmt(p['rate'])} | {fmt(n['rate'])} |")
    md += ["", "## Registered Part 200 predictions", "", "| # | Prediction | Value decided on | Result |", "| --- | --- | --- | --- |"]
    for key, verbatim, confidence in PREDICTIONS:
        value = r["predictions"][key]
        md.append(f"| {key} | {verbatim} ({confidence}) | `{json.dumps(value.get('value', value), ensure_ascii=False)}` | {value['result']} |")
    md += ["", "## Exploratory: compare.json", "", "```json", json.dumps(r["exploratory"], ensure_ascii=False, indent=2), "```", "",
           "## Exploratory: batch-1 first differences", "",
           f"Tokenizer: {r['batch1']['positions']['method']}. First positions include the rendered prompt ({r['batch1']['positions']['prompt_tokens']} prompt tokens); decoded answers were re-tokenized because token IDs were not stored.", "",
           "| Arm | Changed | At/before 64 keys | After 64 keys | Median absolute first-difference key |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm, d in r["batch1"]["positions"]["changed_rows_by_arm"].items():
        md.append(f"| {arm} | {d['changed_rows']} | {d['first_difference_before_or_at_64_keys']} | {d['first_difference_after_64_keys']} | {fmt(d['median_position_including_prompt'],1)} |")
    md.append(f"\nAmong changed rows, Part 163 label flips: name gone {r['batch1']['positions']['label_changes']['name_gone']}, broken {r['batch1']['positions']['label_changes']['broken']}, Qwen under the Part 161 predicate {r['batch1']['positions']['label_changes']['qwen_part161_rule']}. Exploratory rows and excerpts are in report200.json.")
    return "\n".join(md) + "\n"


def main():
    report = build_report()
    (NEW / "report200.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = render_md(report)
    (NEW / "report200.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
