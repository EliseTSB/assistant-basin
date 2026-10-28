#!/usr/bin/env python3
"""Part 87 scorer: blind 87A origin labels and score 87B with Part 66's fixed lexical rules."""
from __future__ import annotations

import argparse
import json
import random
import re
import secrets
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from part84 import blind84
from part66 import score66
from part87 import part87 as p87

A_ARMS = ("A0", "E63", "E47", "D47", "C63")
B_MAIN_ARMS = ("U", "Z", "M", "F")
RANDOM_SEEDS = tuple(range(8700, 8720))
ALL_VERDICTS = tuple(f"87.{i}" for i in range(1, 10))
TO_PART84_ARM = {"A0": "A0", "E63": "A2", "E47": "A1", "D47": "A3", "C63": "C"}
FROM_PART84_ARM = {value: key for key, value in TO_PART84_ARM.items()}


def _read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return p87.read_jsonl(path)


def _atomic_json(path: Path | str, value: Any, refuse_overwrite: bool = False) -> None:
    p87.atomic_write_json(path, value, refuse_overwrite=refuse_overwrite)


def _atomic_text(path: Path | str, text: str, refuse_overwrite: bool = False) -> None:
    p87.atomic_write_bytes(Path(path), text.encode("utf-8"), refuse_overwrite=refuse_overwrite)


def _answers_for_a(answers_dir: Path | str) -> list[dict[str, Any]]:
    root = Path(answers_dir)
    rows = []
    missing = []
    for arm in A_ARMS:
        path = root / arm / "answers.jsonl"
        if not path.is_file():
            missing.append(str(path))
            continue
        for row in _read_jsonl(path):
            row = dict(row)
            row["arm"] = arm
            rows.append(row)
    if missing:
        raise FileNotFoundError("missing 87A answer arm(s): " + ", ".join(missing))
    return rows


def prepare_blind(answers_dir: Path | str, items_path: Path | str,
                  key_path: Path | str) -> dict[str, Any]:
    items_path, key_path = Path(items_path), Path(key_path)
    if items_path.resolve() == key_path.resolve():
        raise ValueError("blind items and key must be separate files")
    if items_path.exists() or key_path.exists():
        raise FileExistsError("blind items/key already exist; refusing to overwrite either")
    records = _answers_for_a(answers_dir)
    counts = Counter(row["arm"] for row in records)
    if any(counts[arm] == 0 for arm in A_ARMS):
        raise ValueError(f"87A answers missing an arm: {dict(counts)}")
    with tempfile.TemporaryDirectory(prefix="part87-blind-") as temp:
        base_items = Path(temp) / "base-items.jsonl"
        base_key_path = Path(temp) / "base-key.json"
        reader_records = [{**row, "arm": TO_PART84_ARM[row["arm"]]} for row in records]
        blind84.prepare(reader_records, base_items, base_key_path, seed=secrets.randbits(32))
        shuffled = _read_jsonl(base_items)
        base_key = json.loads(base_key_path.read_text(encoding="utf-8"))
    key: dict[str, Any] = {}
    item_rows = []
    for row in shuffled:
        base_id = row["id"]
        source = base_key[base_id]
        part84_arm = source["arm"]
        arm = FROM_PART84_ARM.get(part84_arm)
        if arm is None:
            raise ValueError(f"Part 84 blind key returned unexpected arm {part84_arm}")
        item_id = secrets.token_urlsafe(15)
        while item_id in key:
            item_id = secrets.token_urlsafe(15)
        item_rows.append({"id": item_id, "text": row.get("text", "")})
        key[item_id] = {
            "arm": arm, "question_index": source.get("question_index"),
            "sample": source.get("sample"),
        }
    items_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in item_rows)
    key_obj = {"schema": 1, "items": key, "answers_sha256": p87.sha256_bytes(
        json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )}
    _atomic_text(items_path, items_text, refuse_overwrite=True)
    _atomic_json(key_path, key_obj, refuse_overwrite=True)
    return {"item_count": len(item_rows), "arm_counts": dict(counts), "items": str(items_path), "key": str(key_path)}


def score_blind_labels(labels_path: Path | str, key_path: Path | str) -> dict[str, Any]:
    labels = _read_jsonl(labels_path)
    key_obj = json.loads(Path(key_path).read_text(encoding="utf-8"))
    key = key_obj.get("items", key_obj)
    translated_key = {}
    for item_id, value in key.items():
        arm84 = TO_PART84_ARM.get(value.get("arm"))
        if arm84 is None:
            raise ValueError(f"blind key contains unknown 87A arm {value.get('arm')}")
        translated_key[item_id] = {**value, "arm": arm84}
    counted = blind84.score(labels, translated_key)
    counts = {
        arm: counted[TO_PART84_ARM[arm]] for arm in A_ARMS
    }
    return {"counts": counts, "label_count": len(labels), "key_item_count": len(key)}


def _google(answer: str) -> bool:
    return "google" in score66.families(answer or "")


def _guard_match(question: str, answer: str) -> bool:
    keyword = score66.GUARD_KEYWORDS.get(question)
    return bool(keyword and re.search(re.escape(keyword), answer or "", re.I))


def score_b_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for raw in rows:
        row = dict(raw)
        arm = row.get("arm", row.get("condition"))
        if not isinstance(arm, str):
            raise ValueError(f"answer row has no arm/condition: {row}")
        key = (arm, row.get("question_index"), row.get("sample"))
        if key in seen:
            raise ValueError(f"duplicate 87B answer row {key}")
        seen.add(key)
        row["arm"] = arm
        row["google"] = _google(str(row.get("answer", "")))
        by_arm[arm].append(row)
    conditions: dict[str, Any] = {}
    for arm, arm_rows in by_arm.items():
        heldout = [r for r in arm_rows if r.get("question_kind") in ("direct", "indirect")]
        guards = [r for r in arm_rows if r.get("question_kind") == "guard"]
        question_counts = Counter(r.get("question", "") for r in heldout if r["google"])
        guard_matches = sum(_guard_match(str(r.get("question", "")), str(r.get("answer", ""))) for r in guards)
        conditions[arm] = {
            "n": len(arm_rows),
            "heldout_n": len(heldout),
            "heldout_google": sum(bool(r["google"]) for r in heldout),
            "heldout_google_rate": (
                sum(bool(r["google"]) for r in heldout) / len(heldout) if heldout else None
            ),
            "per_question_google_counts": dict(sorted(question_counts.items())),
            "guard_n": len(guards), "guard_matches": guard_matches,
            "guard_rate": guard_matches / len(guards) if guards else None,
        }
    return {"conditions": conditions}


def prediction_87_7_holds(target_count: int, random_counts: list[int],
                          target_n: int = 96, random_n: int = 24) -> tuple[bool, float | None]:
    if len(random_counts) != 20 or target_n != 96 or random_n != 24:
        return False, None
    target_rate = target_count / target_n
    random_rates = [count / random_n for count in random_counts]
    empirical_p = (1 + sum(rate >= target_rate for rate in random_rates)) / 21
    return target_rate > max(random_rates), empirical_p


def verdicts(origin_counts: dict[str, Any] | None,
             b_conditions: dict[str, Any] | None,
             phase: str = "run") -> dict[str, dict[str, Any]]:
    if phase != "run":
        return {key: {"status": "NOT READ", "observed": None,
                      "rule": "pilot answers are a timing/scorer check and do not enter registered verdicts"}
                for key in ALL_VERDICTS}
    a = origin_counts or {}
    b = b_conditions or {}
    out: dict[str, dict[str, Any]] = {}

    def put(key: str, ready: bool, holds: bool, observed: Any, rule: str) -> None:
        status = "NOT READ" if not ready else ("HOLDS" if holds else "FAILS")
        out[key] = {"status": status, "observed": observed, "rule": rule}

    u = b.get("U", {})
    a0 = a.get("A0", {})
    ready = u.get("heldout_n") == 96 and a0.get("n") == 24
    holds = ready and int(u.get("heldout_google", 0)) <= 2 and int(a0.get("OFFWORLD", 0)) <= 1
    put("87.1", ready, holds,
        {"U_google": u.get("heldout_google"), "U_n": u.get("heldout_n"),
         "A0_offworld": a0.get("OFFWORLD"), "A0_n": a0.get("n")},
        "U names Google in at most 2/96 and A0 is OFFWORLD in at most 1/24")

    z = b.get("Z", {})
    ready = z.get("heldout_n") == 96
    put("87.2", ready, ready and int(z.get("heldout_google", 0)) >= 10,
        {"google": z.get("heldout_google"), "n": z.get("heldout_n")},
        "Z names Google in at least 10/96")

    e63 = a.get("E63", {})
    ready = e63.get("n") == 24
    put("87.3", ready, ready and int(e63.get("OFFWORLD", 0)) >= 20,
        {"offworld": e63.get("OFFWORLD"), "n": e63.get("n")},
        "E63 is OFFWORLD in at least 20/24")

    e47, d47 = a.get("E47", {}), a.get("D47", {})
    ready = e47.get("n") == 24 and d47.get("n") == 24
    difference = abs(int(e47.get("OFFWORLD", 0)) - int(d47.get("OFFWORLD", 0))) if ready else None
    put("87.4", ready, ready and difference <= 6,
        {"E47_offworld": e47.get("OFFWORLD"), "D47_offworld": d47.get("OFFWORLD"),
         "absolute_difference": difference}, "E47 and D47 OFFWORLD counts differ by at most 6/24")

    c63 = a.get("C63", {})
    ready = c63.get("n") == 24
    put("87.5", ready, ready and int(c63.get("OFFWORLD", 0)) <= 1,
        {"offworld": c63.get("OFFWORLD"), "n": c63.get("n")},
        "C63 is OFFWORLD in at most 1/24")

    m = b.get("M", {})
    ready = z.get("heldout_n") == 96 and m.get("heldout_n") == 96
    put("87.6", ready, ready and int(m.get("heldout_google", 0)) * 2 >= int(z.get("heldout_google", 0)),
        {"M_google": m.get("heldout_google"), "Z_google": z.get("heldout_google"), "n_each": 96},
        "M names Google at least half as often as Z")

    comparison: dict[str, Any] = {}
    all_ready = True
    all_holds = True
    for arm, prefix in (("M", "RM"), ("F", "RF")):
        target = b.get(arm, {})
        random_items = [b.get(f"{prefix}_{seed}", {}) for seed in RANDOM_SEEDS]
        ready = target.get("heldout_n") == 96 and all(item.get("heldout_n") == 24 for item in random_items)
        all_ready = all_ready and ready
        if not ready:
            all_holds = False
            comparison[arm] = {"target_google": target.get("heldout_google"),
                               "target_n": target.get("heldout_n"), "random_n": len(random_items),
                               "empirical_p": None, "max_random_rate": None}
            continue
        counts = [int(item.get("heldout_google", 0)) for item in random_items]
        holds, p_value = prediction_87_7_holds(int(target["heldout_google"]), counts)
        all_holds = all_holds and holds
        comparison[arm] = {
            "target_google": target["heldout_google"], "target_n": 96,
            "random_google_counts": counts, "random_n_each": 24,
            "max_random_rate": max(count / 24 for count in counts),
            "empirical_p": p_value,
        }
    put("87.7", all_ready, all_holds, comparison,
        "M and F each name Google more often than all 20 corresponding random sets (p = 1/21)")

    f = b.get("F", {})
    ready = m.get("heldout_n") == 96 and f.get("heldout_n") == 96
    put("87.8", ready, ready and int(f.get("heldout_google", 0)) < int(m.get("heldout_google", 0)),
        {"F_google": f.get("heldout_google"), "M_google": m.get("heldout_google"), "n_each": 96},
        "F names Google less often than M")

    guard_values = {arm: b.get(arm, {}).get("guard_matches") for arm in B_MAIN_ARMS}
    ready = all(b.get(arm, {}).get("heldout_n") == 96 and b.get(arm, {}).get("guard_n") == 24
                for arm in B_MAIN_ARMS)
    holds = ready and all(int(b[arm]["guard_matches"]) / 24 >= 0.90 for arm in B_MAIN_ARMS)
    put("87.9", ready, holds, {"guard_matches": guard_values, "guard_n": 24},
        "each main arm has at least 90% of its 24 guard answers containing the keyword")
    return out


def _read_b_answers(answers_dir: Path | str) -> tuple[list[dict[str, Any]], list[str]]:
    root = Path(answers_dir)
    arms = [*B_MAIN_ARMS]
    arms.extend(f"{prefix}_{seed}" for seed in RANDOM_SEEDS for prefix in ("RM", "RF"))
    rows, missing = [], []
    for arm in arms:
        path = root / arm / "answers.jsonl"
        if not path.is_file():
            missing.append(str(path))
            continue
        for row in _read_jsonl(path):
            row = dict(row)
            row["arm"] = arm
            rows.append(row)
    return rows, missing


def _hand_read_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Part 87 hand-read sample", "",
        "For each main arm, these are the first 30 held-out answers matched by Part 66's Google/Gemini/DeepMind regex.",
        "Fill the hand verdict after checking whether the answer attributes its making to Google.", "",
    ]
    for arm in B_MAIN_ARMS:
        matches = [
            row for row in rows
            if row.get("arm") == arm and row.get("question_kind") in ("direct", "indirect")
            and _google(str(row.get("answer", "")))
        ][:30]
        lines.extend([
            f"## {arm} ({len(matches)} matching answers)", "",
            "| Question | Answer | Hand verdict |", "|---|---|---|",
        ])
        for row in matches:
            question = str(row.get("question", "")).replace("|", "\\|")
            answer = str(row.get("answer", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {question} | {answer} |  |")
        lines.append("")
    return "\n".join(lines)


def _score_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Part 87 verdicts", "",
        "| Prediction | Verdict | Observation | Rule |",
        "|---|---|---|---|",
    ]
    for key in ALL_VERDICTS:
        item = result["verdicts"][key]
        observed = json.dumps(item["observed"], ensure_ascii=False, sort_keys=True)
        lines.append(f"| {key} | {item['status']} | {observed} | {item['rule']} |")
    lines.append("")
    return "\n".join(lines)


def score(args) -> dict[str, Any]:
    origin = None
    missing_inputs: list[str] = []
    if bool(args.labels) != bool(args.key):
        raise ValueError("--labels and --key must be supplied together")
    if args.labels and args.key:
        if args.a_answers_dir:
            try:
                answer_records = _answers_for_a(args.a_answers_dir)
                key_obj = json.loads(Path(args.key).read_text(encoding="utf-8"))
                answer_digest = p87.sha256_bytes(
                    json.dumps(answer_records, ensure_ascii=False, sort_keys=True).encode("utf-8")
                )
                if key_obj.get("answers_sha256") != answer_digest:
                    raise ValueError("blind key does not match the supplied 87A answers")
            except FileNotFoundError as exc:
                missing_inputs.append(str(exc))
            else:
                origin = score_blind_labels(args.labels, args.key)["counts"]
                _atomic_json(Path(args.out).with_name("87a-score.json"), {"counts": origin})
        else:
            origin = score_blind_labels(args.labels, args.key)["counts"]
            _atomic_json(Path(args.out).with_name("87a-score.json"), {"counts": origin})
    b_rows: list[dict[str, Any]] = []
    conditions = None
    if args.b_answers_dir:
        b_rows, missing_inputs = _read_b_answers(args.b_answers_dir)
        if b_rows:
            conditions = score_b_rows(b_rows)["conditions"]
    results = {
        "schema": 1, "phase": args.phase,
        "origin_counts": origin, "b_conditions": conditions,
        "missing_inputs": missing_inputs,
        "verdicts": verdicts(origin, conditions, phase=args.phase),
        "provenance": {
            "scorer_sha256": p87.sha256_file(Path(__file__).resolve()),
            "part87_sha256": p87.sha256_file(ROOT / "part87" / "part87.py"),
            "part87_package_sha256": p87.sha256_file(ROOT / "part87" / "__init__.py"),
            "part84_package_sha256": p87.sha256_file(ROOT / "part84" / "__init__.py"),
            "part66_score66_sha256": p87.sha256_file(ROOT / "part66" / "score66.py"),
            "part84_blind84_sha256": p87.sha256_file(ROOT / "part84" / "blind84.py"),
        },
    }
    out_path = Path(args.out)
    _atomic_json(out_path, results)
    _atomic_text(out_path.with_suffix(".md"), _score_markdown(results))
    if args.b_answers_dir:
        _atomic_text(out_path.parent / "hand-read.md", _hand_read_md(b_rows))
    for key in ALL_VERDICTS:
        item = results["verdicts"][key]
        print(f"{key}: {item['status']} — {item['rule']}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    prep = subs.add_parser("blind-prepare", help="write shuffled random-ID 87A reader items and separate key")
    prep.add_argument("--answers-dir", required=True)
    prep.add_argument("--items", required=True)
    prep.add_argument("--key", required=True)
    labels = subs.add_parser("blind-score", help="validate reader labels against the private key")
    labels.add_argument("--labels", required=True)
    labels.add_argument("--key", required=True)
    labels.add_argument("--out", required=True)
    score_parser = subs.add_parser("score", help="score available inputs; missing inputs produce NOT READ")
    score_parser.add_argument("--a-answers-dir", default=None)
    score_parser.add_argument("--b-answers-dir", default=None)
    score_parser.add_argument("--labels", default=None)
    score_parser.add_argument("--key", default=None)
    score_parser.add_argument("--phase", choices=("pilot", "run"), default="run")
    score_parser.add_argument("--out", required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "blind-prepare":
        result = prepare_blind(args.answers_dir, args.items, args.key)
        print(f"wrote {result['item_count']} blind items; key: {result['key']}")
    elif args.cmd == "blind-score":
        result = score_blind_labels(args.labels, args.key)
        _atomic_json(args.out, result)
        print(f"validated {result['label_count']} blind labels; wrote {args.out}")
    else:
        score(args)


if __name__ == "__main__":
    main()

