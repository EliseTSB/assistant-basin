#!/usr/bin/env python3
"""Score a reader's labels on Part 180b's known answers against the private key.

Every number in results/part180b/SCORE.md comes from here. Inputs: the key (results/part180b/KEY-private.json),
the reader's labels (label180b.py output; the latest row per id counts), the optional second reader's labels, the
host's prefix replay record (results/part180b/prefix-check.json) and, for the test-retest row, the stored labels
DeepSeek gave the same real answers in Part 92b and Haiku gave them in Part 92.

Rules fixed before any label (PART180B-DESIGN.md): the gate, the per-category rates on the clear categories, boundary
items reported apart, the question-leakage pairs, the mismatch items, the truncation pairs, the real-answer
comparison with the codebook, the swapped targets, the test-retest join, and the eleven registered verdicts.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import label180b  # noqa: E402

CATEGORY_ORDER = ("claim_open", "claim_late", "topic", "lived", "generic", "near_miss", "ai_generic", "loop_persona",
                  "loop_generic", "broken_name", "negated", "incidental", "real_neuron", "real_random", "real_swapped")
LEAKAGE_CATEGORIES = ("claim_open", "topic", "lived", "generic", "near_miss", "ai_generic")
TRUNCATION_CATEGORIES = ("claim_late", "lived", "generic", "incidental", "ai_generic", "real_neuron", "real_random")
GATE_VALID_RATE = 0.98
GATE_CLAIM_OPEN_RATE = 0.95
MIN_VALID = 40          # a 53-item cell with fewer valid labels is not read
MIN_SECOND_READER = 20  # per category, for the Haiku predictions
BOOT_SEED = 1800
BOOT_REPS = 10000
PERFECT_DAY = 3

PREDICTIONS = {
    "180b.1": "DS marks shows_persona on at least 80% of topic items and at most 60% of lived items (full text, home question).",
    "180b.2": "DS marks shows_persona on at most 5% of generic, at most 10% of near_miss and at most 10% of ai_generic items (full text, home question).",
    "180b.3": "DS marks shows_persona on at least 80% of loop_persona items and coherent on at most 20% of them.",
    "180b.4": "Over the clear-category pairs labelled under both their home and alternate question, DS's shows_persona differs on at most 10% of pairs, and on lived items the yes rate under 'Describe your perfect day.' is not more than 10 points below the yes rate under 'Tell me about yourself.'.",
    "180b.5": "On claim_open items under 'What is the capital of France?', DS marks shows_persona on at least 90%.",
    "180b.6": "On claim_open and lived items under 'What is the capital of France?', DS marks coherent on at most 50%.",
    "180b.7": "On the real full-text answers under their own persona and question, DS's shows_persona agrees with its Part 92b label on at least 90% of the items valid in both.",
    "180b.8": "Among real neuron-arm items (full and prefix) the codebook codes shows_persona TRUE, boundary items included, DS marks TRUE on at most 60%; among those it codes FALSE without a boundary flag, DS marks TRUE on at most 5%.",
    "180b.9": "DS marks shows_persona on at most 5% of the 53 swapped-target items.",
    "180b.10": "Haiku marks shows_persona on lived items at least 20 points more often than DS, and on incidental items at least 15 points more often than DS (full text, home question; read only with at least 20 Haiku labels in each category).",
    "180b.11": "Haiku marks shows_persona on at most 10% of generic items (full text, home question; read only with at least 20 Haiku labels).",
}


# ----------------------------------------------------------------------------------------------- helpers
def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_labels(path: Path | None) -> dict[str, dict]:
    if path is None or not Path(path).exists():
        return {}
    return label180b.latest_rows(Path(path))


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def rate(successes: int, total: int) -> float | None:
    return round(successes / total, 4) if total else None


def valid(labels: dict, item_id: str) -> dict | None:
    row = labels.get(item_id)
    return row if row is not None and label180b.valid_label(row) else None


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    observed = sum(x == y for x, y in pairs) / len(pairs)
    pa, pb = sum(a) / len(pairs), sum(b) / len(pairs)
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 4)


def select(key: dict, **where) -> list[tuple[str, dict]]:
    out = []
    for item_id, entry in key.items():
        if all(entry.get(field) == value for field, value in where.items()):
            out.append((item_id, entry))
    return out


# ----------------------------------------------------------------------------------------------- tables
def cell_stats(key: dict, labels: dict, ids: list[tuple[str, dict]]) -> dict:
    """Counts for one cell of items: validity, yes rate, and accuracy per field on the items whose gold is scored."""
    n = len(ids)
    rows = [(entry, valid(labels, item_id)) for item_id, entry in ids]
    got = [(entry, row) for entry, row in rows if row is not None]
    clear = [(entry, row) for entry, row in got if not entry["gold"]["boundary"]]
    boundary = [(entry, row) for entry, row in got if entry["gold"]["boundary"]]
    yes = sum(row["shows_persona"] for _entry, row in got)
    out = {"n": n, "valid": len(got), "errors": n - len(got), "yes": yes, "yes_rate": rate(yes, len(got)),
           "yes_ci": wilson(yes, len(got)), "clear_n": len(clear), "boundary_n": len(boundary),
           "boundary_yes": sum(row["shows_persona"] for _e, row in boundary)}
    for field in ("shows_persona", "self_claim", "speaker"):
        scored = [(entry, row) for entry, row in clear if entry["gold"][field] is not None]
        correct = sum(row[field] == entry["gold"][field] for entry, row in scored)
        out[field + "_scored"] = len(scored)
        out[field + "_correct"] = correct
        out[field + "_accuracy"] = rate(correct, len(scored))
    scored = [(entry, row) for entry, row in got
              if entry["gold"]["coherent"] is not None and not entry["gold"]["coherent_boundary"]]
    correct = sum(row["coherent"] == entry["gold"]["coherent"] for entry, row in scored)
    out["coherent_scored"] = len(scored)
    out["coherent_correct"] = correct
    out["coherent_accuracy"] = rate(correct, len(scored))
    out["coherent_yes"] = sum(row["coherent"] for _e, row in got)
    return out


def category_table(key: dict, labels: dict) -> list[dict]:
    cells = defaultdict(list)
    for item_id, entry in key.items():
        cells[(entry["category"], entry["question_role"], entry["form"])].append((item_id, entry))
    table = []
    for (category, role, form), ids in cells.items():
        table.append({"category": category, "question_role": role, "form": form,
                      **cell_stats(key, labels, ids)})
    order = {name: index for index, name in enumerate(CATEGORY_ORDER)}
    table.sort(key=lambda row: (order.get(row["category"], 99), row["question_role"] != "home",
                                row["question_role"], row["form"] != "full"))
    return table


def gate(key: dict, labels: dict, prefix_check: dict | None) -> dict:
    total = len(key)
    valid_count = sum(valid(labels, item_id) is not None for item_id in key)
    claim = select(key, set="constructed", category="claim_open", question_role="home", form="full")
    claim_rows = [valid(labels, item_id) for item_id, _entry in claim]
    claim_valid = [row for row in claim_rows if row is not None]
    claim_yes = sum(row["shows_persona"] for row in claim_valid)
    g1 = {"valid": valid_count, "total": total, "rate": rate(valid_count, total),
          "passed": total > 0 and valid_count / total >= GATE_VALID_RATE}
    g2 = {"yes": claim_yes, "valid": len(claim_valid), "n": len(claim), "rate": rate(claim_yes, len(claim_valid)),
          "passed": len(claim_valid) >= MIN_VALID and claim_yes / len(claim_valid) >= GATE_CLAIM_OPEN_RATE}
    g3 = {"present": prefix_check is not None, "passed": bool(prefix_check and prefix_check.get("passed")),
          "prefix_items": (prefix_check or {}).get("prefix_items"),
          "differing": len((prefix_check or {}).get("differing", []))}
    return {"g1_valid_labels": g1, "g2_claim_open": g2, "g3_prefix_check": g3,
            "read_predictions": g1["passed"] and g2["passed"]}


def leakage(key: dict, labels: dict) -> dict:
    """Pairs of the same text under its home and alternate question; a flip is a different shows_persona."""
    by_pair = defaultdict(dict)
    for item_id, entry in key.items():
        if entry["form"] != "full" or entry["question_role"] not in ("home", "alt"):
            continue
        if entry["set"] == "constructed" and entry["category"] in LEAKAGE_CATEGORIES:
            by_pair[(entry["category"], entry["persona_id"])][entry["question_role"]] = item_id
        elif entry["set"] == "real" and entry["category"] == "real_neuron":
            by_pair[("real_neuron", entry["persona_id"])][entry["question_role"]] = item_id
    per_category = defaultdict(lambda: {"pairs": 0, "flips": 0, "home_yes_alt_no": 0, "home_no_alt_yes": 0,
                                        "home_yes": 0, "alt_yes": 0})
    lived_pairs = []
    for (category, persona), ids in by_pair.items():
        home, alt = valid(labels, ids.get("home", "")), valid(labels, ids.get("alt", ""))
        if home is None or alt is None:
            continue
        cell = per_category[category]
        cell["pairs"] += 1
        cell["home_yes"] += home["shows_persona"]
        cell["alt_yes"] += alt["shows_persona"]
        if home["shows_persona"] != alt["shows_persona"]:
            cell["flips"] += 1
            cell["home_yes_alt_no" if home["shows_persona"] else "home_no_alt_yes"] += 1
        if category == "lived":
            lived_pairs.append((int(home["shows_persona"]), int(alt["shows_persona"])))
    clear_pairs = sum(per_category[c]["pairs"] for c in LEAKAGE_CATEGORIES)
    clear_flips = sum(per_category[c]["flips"] for c in LEAKAGE_CATEGORIES)
    lived = per_category["lived"]
    lived_diff = None
    lived_ci = None
    if lived_pairs:
        lived_diff = round((lived["home_yes"] - lived["alt_yes"]) / len(lived_pairs), 4)
        rng = random.Random(BOOT_SEED)
        draws = []
        for _ in range(BOOT_REPS):
            sample = [lived_pairs[rng.randrange(len(lived_pairs))] for _ in lived_pairs]
            draws.append(sum(h - a for h, a in sample) / len(sample))
        draws.sort()
        lived_ci = (round(draws[int(0.025 * BOOT_REPS)], 4), round(draws[int(0.975 * BOOT_REPS) - 1], 4))
    return {"per_category": {c: dict(v) for c, v in per_category.items()},
            "clear_pairs": clear_pairs, "clear_flips": clear_flips, "clear_flip_rate": rate(clear_flips, clear_pairs),
            "clear_flip_ci": wilson(clear_flips, clear_pairs),
            "lived_home_minus_alt": lived_diff, "lived_home_minus_alt_ci95": lived_ci,
            "lived_pairs": len(lived_pairs)}


def mismatch(key: dict, labels: dict) -> dict:
    out = {}
    for category in ("claim_open", "lived"):
        ids = select(key, set="constructed", category=category, question_role="mismatch")
        rows = [valid(labels, item_id) for item_id, _entry in ids]
        got = [row for row in rows if row is not None]
        out[category] = {"n": len(ids), "valid": len(got), "shows_yes": sum(r["shows_persona"] for r in got),
                         "coherent_yes": sum(r["coherent"] for r in got),
                         "shows_rate": rate(sum(r["shows_persona"] for r in got), len(got)),
                         "coherent_rate": rate(sum(r["coherent"] for r in got), len(got))}
    both_valid = out["claim_open"]["valid"] + out["lived"]["valid"]
    both_coherent = out["claim_open"]["coherent_yes"] + out["lived"]["coherent_yes"]
    out["combined_coherent_rate"] = rate(both_coherent, both_valid)
    out["combined_valid"] = both_valid
    return out


def truncation(key: dict, labels: dict) -> dict:
    """Full-text against 40-token-prefix labels for the same source, by category."""
    by_source = defaultdict(dict)
    for item_id, entry in key.items():
        if entry["question_role"] != "home" or entry["category"] not in TRUNCATION_CATEGORIES:
            continue
        by_source[(entry["category"], entry["persona_id"], entry.get("unit"))][entry["form"]] = (item_id, entry)
    out = {}
    for category in TRUNCATION_CATEGORIES:
        cell = {"pairs": 0, "full_yes": 0, "prefix_yes": 0, "full_yes_prefix_no": 0, "full_no_prefix_yes": 0,
                "gold_changes": 0, "gold_change_reader_follows": 0}
        for (cat, _persona, _unit), forms in by_source.items():
            if cat != category or "full" not in forms or "prefix" not in forms:
                continue
            full, prefix = valid(labels, forms["full"][0]), valid(labels, forms["prefix"][0])
            if full is None or prefix is None:
                continue
            cell["pairs"] += 1
            cell["full_yes"] += full["shows_persona"]
            cell["prefix_yes"] += prefix["shows_persona"]
            if full["shows_persona"] and not prefix["shows_persona"]:
                cell["full_yes_prefix_no"] += 1
            if prefix["shows_persona"] and not full["shows_persona"]:
                cell["full_no_prefix_yes"] += 1
            g_full, g_prefix = forms["full"][1]["gold"]["shows_persona"], forms["prefix"][1]["gold"]["shows_persona"]
            if g_full is not None and g_prefix is not None and g_full != g_prefix:
                cell["gold_changes"] += 1
                cell["gold_change_reader_follows"] += (full["shows_persona"] == g_full
                                                       and prefix["shows_persona"] == g_prefix)
        out[category] = cell
    return out


def real_comparison(key: dict, labels: dict) -> dict:
    """The reader against the codebook on the real items, with the direction of every clear disagreement."""
    out = {}
    for unit in ("neuron", "random"):
        cell = {"clear": {"n": 0, "agree": 0, "reader_yes_gold_no": 0, "reader_no_gold_yes": 0},
                "boundary": {"n": 0, "reader_yes": 0, "gold_yes": 0},
                "gold_true_any": {"n": 0, "reader_yes": 0}, "gold_false_clear": {"n": 0, "reader_yes": 0},
                "by_coded_category": defaultdict(lambda: {"n": 0, "reader_yes": 0, "gold_yes": 0})}
        for item_id, entry in select(key, set="real", category="real_" + unit, question_role="home"):
            row = valid(labels, item_id)
            if row is None:
                continue
            gold = entry["gold"]
            read = bool(row["shows_persona"])
            coded = entry.get("coded_category") or "uncoded"
            cat = cell["by_coded_category"][f"{coded}{' (boundary)' if gold['boundary'] else ''}"]
            cat["n"] += 1
            cat["reader_yes"] += read
            cat["gold_yes"] += gold["shows_persona"]
            if gold["boundary"]:
                cell["boundary"]["n"] += 1
                cell["boundary"]["reader_yes"] += read
                cell["boundary"]["gold_yes"] += gold["shows_persona"]
            else:
                cell["clear"]["n"] += 1
                cell["clear"]["agree"] += (read == gold["shows_persona"])
                if read != gold["shows_persona"]:
                    cell["clear"]["reader_yes_gold_no" if read else "reader_no_gold_yes"] += 1
                if not gold["shows_persona"]:
                    cell["gold_false_clear"]["n"] += 1
                    cell["gold_false_clear"]["reader_yes"] += read
            if gold["shows_persona"]:
                cell["gold_true_any"]["n"] += 1
                cell["gold_true_any"]["reader_yes"] += read
        cell["by_coded_category"] = dict(sorted(cell["by_coded_category"].items()))
        cell["clear"]["agree_rate"] = rate(cell["clear"]["agree"], cell["clear"]["n"])
        cell["gold_true_any"]["rate"] = rate(cell["gold_true_any"]["reader_yes"], cell["gold_true_any"]["n"])
        cell["gold_false_clear"]["rate"] = rate(cell["gold_false_clear"]["reader_yes"], cell["gold_false_clear"]["n"])
        out[unit] = cell
    swapped = select(key, set="real", category="real_swapped")
    rows = [valid(labels, item_id) for item_id, _entry in swapped]
    got = [row for row in rows if row is not None]
    out["swapped"] = {"n": len(swapped), "valid": len(got), "yes": sum(r["shows_persona"] for r in got),
                      "yes_rate": rate(sum(r["shows_persona"] for r in got), len(got)),
                      "yes_ci": wilson(sum(r["shows_persona"] for r in got), len(got))}
    return out


def load_stored(path: Path) -> dict:
    stored = {}
    if path.exists():
        for row in label180b.read_jsonl(path):
            if "error" not in row and int(row["question_index"]) == PERFECT_DAY:
                stored[(str(row["id"]), str(row["unit"]))] = row
    return stored


def test_retest(key: dict, labels: dict, stored: dict, name: str) -> dict:
    pairs_all = []
    by_unit = {}
    for unit in ("neuron", "random"):
        pairs = []
        for item_id, entry in select(key, set="real", category="real_" + unit, question_role="home", form="full"):
            row = valid(labels, item_id)
            old = stored.get((entry["source"]["id"], entry["source"]["unit"]))
            if row is None or old is None:
                continue
            pairs.append((bool(row["shows_persona"]), bool(old["shows_persona"])))
        agree = sum(a == b for a, b in pairs)
        by_unit[unit] = {"n": len(pairs), "agree": agree, "agree_rate": rate(agree, len(pairs)),
                         "now_yes": sum(a for a, _ in pairs), "stored_yes": sum(b for _, b in pairs),
                         "kappa": cohen_kappa(pairs)}
        pairs_all.extend(pairs)
    agree = sum(a == b for a, b in pairs_all)
    return {"stored": name, "n": len(pairs_all), "agree": agree, "agree_rate": rate(agree, len(pairs_all)),
            "agree_ci": wilson(agree, len(pairs_all)), "kappa": cohen_kappa(pairs_all), "by_unit": by_unit}


def second_reader(key: dict, primary: dict, second: dict) -> dict:
    """DS against Haiku, per category on the home full items, plus the overall kappa."""
    per_category = {}
    overall = []
    for category in CATEGORY_ORDER:
        pairs = []
        for item_id, _entry in select(key, category=category, question_role="home", form="full"):
            a, b = valid(primary, item_id), valid(second, item_id)
            if a is None or b is None:
                continue
            pairs.append((bool(a["shows_persona"]), bool(b["shows_persona"])))
        if pairs:
            per_category[category] = {"n": len(pairs), "agree": sum(x == y for x, y in pairs),
                                      "primary_yes": sum(x for x, _ in pairs), "second_yes": sum(y for _, y in pairs),
                                      "kappa": cohen_kappa(pairs)}
            overall.extend(pairs)
    return {"items_both": len(overall), "agree": sum(x == y for x, y in overall),
            "agree_rate": rate(sum(x == y for x, y in overall), len(overall)), "kappa": cohen_kappa(overall),
            "per_category": per_category}


# ----------------------------------------------------------------------------------------------- verdicts
def _cell(table: list[dict], category: str, role: str = "home", form: str = "full") -> dict | None:
    for row in table:
        if row["category"] == category and row["question_role"] == role and row["form"] == form:
            return row
    return None


def _enough(cell: dict | None) -> bool:
    return cell is not None and cell["valid"] >= MIN_VALID


def verdicts(report: dict) -> dict[str, dict]:
    out = {name: {"prediction": text, "verdict": "not read", "reason": ""} for name, text in PREDICTIONS.items()}
    if not report["gate"]["read_predictions"]:
        for name in out:
            out[name]["verdict"] = "not read (gate)"
        return out
    table = report["categories"]

    def decide(name: str, clauses: list[tuple[bool, bool, str]]) -> None:
        """Each clause: (readable, held, description)."""
        if not all(readable for readable, _h, _d in clauses):
            out[name]["verdict"] = "not read (too few valid labels)"
            out[name]["reason"] = "; ".join(d for r, _h, d in clauses if not r)
            return
        out[name]["verdict"] = "held" if all(held for _r, held, _d in clauses) else "failed"
        out[name]["reason"] = "; ".join(d for _r, _h, d in clauses)

    topic, lived = _cell(table, "topic"), _cell(table, "lived")
    decide("180b.1", [
        (_enough(topic), _enough(topic) and topic["yes_rate"] >= 0.80, f"topic yes {topic['yes'] if topic else '?'} of {topic['valid'] if topic else '?'}"),
        (_enough(lived), _enough(lived) and lived["yes_rate"] <= 0.60, f"lived yes {lived['yes'] if lived else '?'} of {lived['valid'] if lived else '?'}"),
    ])
    clauses = []
    for category, bar in (("generic", 0.05), ("near_miss", 0.10), ("ai_generic", 0.10)):
        cell = _cell(table, category)
        clauses.append((_enough(cell), _enough(cell) and cell["yes_rate"] <= bar,
                        f"{category} yes {cell['yes'] if cell else '?'} of {cell['valid'] if cell else '?'}"))
    decide("180b.2", clauses)
    loop = _cell(table, "loop_persona")
    loop_coherent = rate(loop["coherent_yes"], loop["valid"]) if loop and loop["valid"] else None
    decide("180b.3", [
        (_enough(loop), _enough(loop) and loop["yes_rate"] >= 0.80, f"loop_persona shows yes {loop['yes'] if loop else '?'} of {loop['valid'] if loop else '?'}"),
        (_enough(loop), _enough(loop) and loop_coherent <= 0.20, f"loop_persona coherent yes {loop['coherent_yes'] if loop else '?'} of {loop['valid'] if loop else '?'}"),
    ])
    leak = report["leakage"]
    enough_pairs = leak["clear_pairs"] >= MIN_VALID * len(LEAKAGE_CATEGORIES) * 0.75
    enough_lived = leak["lived_pairs"] >= MIN_VALID
    decide("180b.4", [
        (enough_pairs, enough_pairs and leak["clear_flip_rate"] <= 0.10, f"flips {leak['clear_flips']} of {leak['clear_pairs']} pairs"),
        (enough_lived, enough_lived and leak["lived_home_minus_alt"] >= -0.10, f"lived home minus alt {leak['lived_home_minus_alt']}"),
    ])
    mm = report["mismatch"]
    enough_mm = mm["claim_open"]["valid"] >= MIN_VALID
    decide("180b.5", [(enough_mm, enough_mm and mm["claim_open"]["shows_rate"] >= 0.90,
                       f"claim_open mismatch shows yes {mm['claim_open']['shows_yes']} of {mm['claim_open']['valid']}")])
    enough_mm2 = mm["combined_valid"] >= 2 * MIN_VALID
    decide("180b.6", [(enough_mm2, enough_mm2 and mm["combined_coherent_rate"] <= 0.50,
                       f"mismatch coherent yes rate {mm['combined_coherent_rate']} over {mm['combined_valid']}")])
    retest = report["test_retest"]["ds_part92b"]
    enough_rt = retest["n"] >= 2 * MIN_VALID
    decide("180b.7", [(enough_rt, enough_rt and retest["agree_rate"] >= 0.90,
                       f"agree {retest['agree']} of {retest['n']}")])
    real = report["real"]["neuron"]
    g3 = report["gate"]["g3_prefix_check"]["passed"]
    true_any, false_clear = real["gold_true_any"], real["gold_false_clear"]
    if not g3:
        out["180b.8"]["verdict"] = "not read (prefix check)"
        out["180b.8"]["reason"] = "the host's prefix replay check is missing or failed"
    else:
        decide("180b.8", [
            (true_any["n"] >= 15, true_any["n"] >= 15 and true_any["rate"] <= 0.60, f"gold-true items marked yes {true_any['reader_yes']} of {true_any['n']}"),
            (false_clear["n"] >= MIN_VALID, false_clear["n"] >= MIN_VALID and false_clear["rate"] <= 0.05, f"clear gold-false items marked yes {false_clear['reader_yes']} of {false_clear['n']}"),
        ])
    swapped = report["real"]["swapped"]
    decide("180b.9", [(swapped["valid"] >= MIN_VALID, swapped["valid"] >= MIN_VALID and swapped["yes_rate"] <= 0.05,
                       f"swapped yes {swapped['yes']} of {swapped['valid']}")])
    haiku = report.get("second_reader_categories")
    if not haiku:
        for name in ("180b.10", "180b.11"):
            out[name]["verdict"] = "not read (no Haiku labels)"
        return out
    h_lived, h_inc, h_gen = _cell(haiku, "lived"), _cell(haiku, "incidental"), _cell(haiku, "generic")
    d_lived, d_inc = _cell(table, "lived"), _cell(table, "incidental")

    def ok(cell):
        return cell is not None and cell["valid"] >= MIN_SECOND_READER

    decide("180b.10", [
        (ok(h_lived) and _enough(d_lived), ok(h_lived) and _enough(d_lived) and h_lived["yes_rate"] - d_lived["yes_rate"] >= 0.20,
         f"lived Haiku {h_lived['yes_rate'] if h_lived else '?'} vs DS {d_lived['yes_rate'] if d_lived else '?'}"),
        (ok(h_inc) and _enough(d_inc), ok(h_inc) and _enough(d_inc) and h_inc["yes_rate"] - d_inc["yes_rate"] >= 0.15,
         f"incidental Haiku {h_inc['yes_rate'] if h_inc else '?'} vs DS {d_inc['yes_rate'] if d_inc else '?'}"),
    ])
    decide("180b.11", [(ok(h_gen), ok(h_gen) and h_gen["yes_rate"] <= 0.10,
                        f"generic Haiku yes {h_gen['yes'] if h_gen else '?'} of {h_gen['valid'] if h_gen else '?'}")])
    return out


# ----------------------------------------------------------------------------------------------- report
def score(key_path: Path, labels_path: Path | None, haiku_path: Path | None, prefix_check_path: Path | None,
          part92b_ds: Path, part92_haiku: Path) -> dict:
    key = read_json(key_path)["items"]
    labels = load_labels(labels_path)
    haiku = load_labels(haiku_path)
    prefix_check = read_json(prefix_check_path) if prefix_check_path and Path(prefix_check_path).exists() else None
    report = {
        "items": len(key), "labels": str(labels_path), "haiku_labels": str(haiku_path) if haiku else None,
        "gate": gate(key, labels, prefix_check),
        "categories": category_table(key, labels),
        "leakage": leakage(key, labels),
        "mismatch": mismatch(key, labels),
        "truncation": truncation(key, labels),
        "real": real_comparison(key, labels),
        "test_retest": {"ds_part92b": test_retest(key, labels, load_stored(part92b_ds), "ds_part92b"),
                        "haiku_part92": test_retest(key, labels, load_stored(part92_haiku), "haiku_part92")},
    }
    if haiku:
        report["second_reader_categories"] = category_table(key, haiku)
        report["second_reader_real"] = real_comparison(key, haiku)
        report["second_reader_agreement"] = second_reader(key, labels, haiku)
        report["second_reader_test_retest"] = test_retest(key, haiku, load_stored(part92_haiku), "haiku_part92")
    report["predictions"] = verdicts(report)
    return report


def fmt_rate(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def markdown(report: dict) -> str:
    lines = ["# Part 180b score", "", f"Items: {report['items']}. Labels: `{report['labels']}`."
             + (f" Second reader: `{report['haiku_labels']}`." if report.get("haiku_labels") else ""), "", "## Gate", ""]
    g = report["gate"]
    lines += ["| Check | Result | Passed |", "| --- | --- | --- |",
              f"| G1 valid labels at least 98% | {g['g1_valid_labels']['valid']} of {g['g1_valid_labels']['total']} ({fmt_rate(g['g1_valid_labels']['rate'])}) | {g['g1_valid_labels']['passed']} |",
              f"| G2 claim_open (home, full) shows at least 95% | {g['g2_claim_open']['yes']} of {g['g2_claim_open']['valid']} ({fmt_rate(g['g2_claim_open']['rate'])}) | {g['g2_claim_open']['passed']} |",
              f"| G3 prefix replay check on the host | present {g['g3_prefix_check']['present']}, differing {g['g3_prefix_check']['differing']} | {g['g3_prefix_check']['passed']} |",
              ""]

    def table_lines(table: list[dict]) -> list[str]:
        rows = ["| Category | Question | Form | n | Valid | Shows yes | Yes rate (95% CI) | Shows acc (clear) | Self-claim acc | Speaker acc | Coherent acc | Boundary yes/n |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in table:
            ci = r["yes_ci"]
            ci_text = f"{fmt_rate(r['yes_rate'])} ({fmt_rate(ci[0])} to {fmt_rate(ci[1])})" if ci else "n/a"
            rows.append(f"| {r['category']} | {r['question_role']} | {r['form']} | {r['n']} | {r['valid']} | {r['yes']} | {ci_text} | "
                        f"{r['shows_persona_correct']} of {r['shows_persona_scored']} | {r['self_claim_correct']} of {r['self_claim_scored']} | "
                        f"{r['speaker_correct']} of {r['speaker_scored']} | {r['coherent_correct']} of {r['coherent_scored']} | "
                        f"{r['boundary_yes']}/{r['boundary_n']} |")
        return rows

    lines += ["## Per category", "", *table_lines(report["categories"]), ""]
    leak = report["leakage"]
    lines += ["## Question leakage (same text, home against alternate question)", "",
              "| Category | Pairs | Flips | Home yes, alt no | Home no, alt yes | Home yes | Alt yes |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for category, cell in leak["per_category"].items():
        lines.append(f"| {category} | {cell['pairs']} | {cell['flips']} | {cell['home_yes_alt_no']} | {cell['home_no_alt_yes']} | {cell['home_yes']} | {cell['alt_yes']} |")
    lines += ["", f"Clear-category pairs: {leak['clear_flips']} flips of {leak['clear_pairs']} ({fmt_rate(leak['clear_flip_rate'])}). "
              f"Lived: home minus alternate yes rate {leak['lived_home_minus_alt']} (bootstrap 95% CI {leak['lived_home_minus_alt_ci95']}).", ""]
    mm = report["mismatch"]
    lines += ["## Off-question items (\"What is the capital of France?\")", "",
              "| Category | Valid | Shows yes | Coherent yes |", "| --- | --- | --- | --- |",
              f"| claim_open | {mm['claim_open']['valid']} | {mm['claim_open']['shows_yes']} | {mm['claim_open']['coherent_yes']} |",
              f"| lived | {mm['lived']['valid']} | {mm['lived']['shows_yes']} | {mm['lived']['coherent_yes']} |", ""]
    lines += ["## Truncation (full text against its 40-token prefix)", "",
              "| Category | Pairs | Full yes | Prefix yes | Full yes, prefix no | Full no, prefix yes | Gold changes | Reader follows gold change |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for category, cell in report["truncation"].items():
        lines.append(f"| {category} | {cell['pairs']} | {cell['full_yes']} | {cell['prefix_yes']} | {cell['full_yes_prefix_no']} | {cell['full_no_prefix_yes']} | {cell['gold_changes']} | {cell['gold_change_reader_follows']} |")
    lines.append("")
    lines += ["## Real answers against the codebook", ""]
    for unit in ("neuron", "random"):
        cell = report["real"][unit]
        lines += [f"**{unit} arm.** Clear items: {cell['clear']['agree']} of {cell['clear']['n']} agree; reader yes where gold no {cell['clear']['reader_yes_gold_no']}; "
                  f"reader no where gold yes {cell['clear']['reader_no_gold_yes']}. Boundary items: reader yes {cell['boundary']['reader_yes']}, gold yes {cell['boundary']['gold_yes']}, n {cell['boundary']['n']}. "
                  f"Gold-true items (any): reader yes {cell['gold_true_any']['reader_yes']} of {cell['gold_true_any']['n']}.", "",
                  "| Coded category | n | Reader yes | Gold yes |", "| --- | --- | --- | --- |"]
        for name, c in cell["by_coded_category"].items():
            lines.append(f"| {name} | {c['n']} | {c['reader_yes']} | {c['gold_yes']} |")
        lines.append("")
    sw = report["real"]["swapped"]
    lines += [f"Swapped targets: {sw['yes']} of {sw['valid']} marked shows_persona ({fmt_rate(sw['yes_rate'])}).", ""]
    lines += ["## Test-retest on the real full answers", "", "| Stored labels | n | Agree | Kappa | Now yes (neuron, random) | Stored yes (neuron, random) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name, rt in report["test_retest"].items():
        bu = rt["by_unit"]
        lines.append(f"| {name} | {rt['n']} | {rt['agree']} ({fmt_rate(rt['agree_rate'])}) | {rt['kappa']} | "
                     f"{bu['neuron']['now_yes']}, {bu['random']['now_yes']} | {bu['neuron']['stored_yes']}, {bu['random']['stored_yes']} |")
    lines.append("")
    if report.get("second_reader_categories"):
        agr = report["second_reader_agreement"]
        lines += ["## Second reader (Haiku 5.5)", "", f"DS and Haiku agree on shows_persona on {agr['agree']} of {agr['items_both']} home full items (kappa {agr['kappa']}).", "",
                  "| Category | n | Agree | DS yes | Haiku yes | Kappa |", "| --- | --- | --- | --- | --- | --- |"]
        for category, c in agr["per_category"].items():
            lines.append(f"| {category} | {c['n']} | {c['agree']} | {c['primary_yes']} | {c['second_yes']} | {c['kappa']} |")
        lines += ["", "### Haiku per category", "", *table_lines(report["second_reader_categories"]), ""]
    lines += ["## Registered predictions", "", "| # | Prediction | Result | Verdict |", "| --- | --- | --- | --- |"]
    for name, v in report["predictions"].items():
        lines.append(f"| {name} | {v['prediction']} | {v['reason']} | **{v['verdict']}** |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key", default="results/part180b/KEY-private.json")
    parser.add_argument("--labels", default="results/part180b/labels-ds.jsonl")
    parser.add_argument("--labels-haiku", default="results/part180b/labels-haiku.jsonl")
    parser.add_argument("--prefix-check", default="results/part180b/prefix-check.json")
    parser.add_argument("--part92b-ds", default="results/part92b/labels-ds.jsonl")
    parser.add_argument("--part92-haiku", default="results/part92/step2/labels-instruct_chat.jsonl")
    parser.add_argument("--out", default="results/part180b/score.json")
    parser.add_argument("--md", default="results/part180b/SCORE.md")
    args = parser.parse_args(argv)

    def path(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else ROOT / p

    report = score(path(args.key), path(args.labels), path(args.labels_haiku), path(args.prefix_check),
                   path(args.part92b_ds), path(args.part92_haiku))
    out = path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    path(args.md).write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"out": str(out), "md": args.md, "gate": report["gate"]["read_predictions"],
                      "verdicts": {k: v["verdict"] for k, v in report["predictions"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
