#!/usr/bin/env python3
"""Recompute Part 117 counts from answers and labels, check score.json, and draw figures."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NONSELF = (4, 5)
QUESTIONS = tuple(range(6))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact_pairs(values: Counter) -> str:
    return ",".join(f"{key}:{values[key]}" for key in sorted(values)) or "none"


def scorer_verdicts_in_memory(pairs: list[dict], label_map: dict) -> dict:
    """Run score117's verdict rules without writing any project files."""
    import contextlib
    import importlib.util
    import io

    class MemoryFile:
        def __init__(self, value: str = ""):
            self.value = value

        def read_text(self, encoding: str = "utf-8") -> str:
            return self.value

        def write_text(self, value: str, encoding: str = "utf-8") -> int:
            self.value = value
            return len(value)

    class MemoryRes:
        def __init__(self, pair_rows: list[dict]):
            self.pairs = MemoryFile(json.dumps(pair_rows))

        def __truediv__(self, name: str) -> MemoryFile:
            return self.pairs if name == "pairs.json" else MemoryFile()

    source = ROOT / "part117/score117.py"
    spec = importlib.util.spec_from_file_location("part117_score_for_check", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load score117.py for the label substitution check")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer.load_labels = lambda _path: label_map
    with contextlib.redirect_stdout(io.StringIO()):
        result = scorer.score(res=MemoryRes(pairs), labels=object())
    return result["verdicts"]


def calculate():
    answers = read_jsonl(HERE / "answers.jsonl")
    items = read_jsonl(HERE / "items.jsonl")
    first_rows = read_jsonl(HERE / "labels.jsonl")
    second_rows = read_jsonl(HERE / "labels_second.jsonl")
    pairs = read_json(HERE / "pairs.json")
    personas = read_json(HERE / "personas.json")
    part112 = read_json(ROOT / "results/part112/score.json")
    score_file = read_json(HERE / "score.json")
    agreement_file = read_json(HERE / "agreement.json")

    answer_row_ids = {row["row"] for row in answers}
    answer_keys = {(row["row"], int(row["question_index"])) for row in answers}
    answers_per_row = Counter(row["row"] for row in answers)
    assert len(answer_keys) == len(answers), "duplicate answer row/question keys"
    assert set(answers_per_row.values()) == {6}, "each row must contain six answers"

    labels = {}
    error_rows = [row for row in first_rows if "error" in row]
    for row in first_rows:
        if "error" in row:
            continue
        key = (row["id"], int(row["question_index"]))
        if key in labels:
            raise ValueError(f"duplicate successful first-reader label: {key}")
        labels[key] = row
    item_keys = {(row["id"], int(row["question_index"])) for row in items}
    persona_ids = {row["id"] for row in personas}
    assert {row["id"] for row in items} == persona_ids, "persona lookup and answer items disagree"
    missing_keys = sorted(item_keys - labels.keys())
    extra_keys = sorted(labels.keys() - item_keys)
    assert not extra_keys, f"first-reader labels without matching items: {extra_keys}"

    second = {}
    for row in second_rows:
        if "error" not in row:
            second[(row["id"], int(row["question_index"]))] = row
    paired_reader_keys = item_keys & labels.keys() & second.keys()
    agree_n = len(paired_reader_keys)
    agree_hits = sum(labels[key]["shows_persona"] == second[key]["shows_persona"] for key in paired_reader_keys)

    slot_ids, matched_ids, unrelated_ids = set(), set(), set()
    match_scores = Counter()
    for pair in pairs:
        name = pair["pair"]
        slot = pair["slot"]["id"]
        match = pair["match"]["id"]
        unrelated = pair["unrelated"]["id"]
        slot_ids.add(slot)
        matched_ids.add(match)
        unrelated_ids.add(unrelated)
        match_scores[int(pair["match_score"])] += 1
        for unit in (pair["slot"], pair["match"], pair["unrelated"]):
            assert part112["units"][unit["id"]]["kind"] == unit["part112_kind"]

    def label(row: str, role: str, qi: int):
        return labels.get((f"{row}|{role}", qi))

    def vals(row: str, role: str, questions=QUESTIONS, field="shows_persona"):
        return [None if (entry := label(row, role, q)) is None else entry[field] for q in questions]

    def count(row: str, role: str, questions=QUESTIONS, field="shows_persona"):
        values = vals(row, role, questions, field)
        valid = [value for value in values if value is not None]
        return {
            "hits": sum(bool(value) for value in valid),
            "valid": len(valid),
            "n": len(values),
            "count": sum(bool(value) for value in valid) if len(valid) == len(values) else None,
            "values": values,
        }

    none_rates = {}
    none_valid = {}
    all_unit_ids = slot_ids | matched_ids | unrelated_ids
    for uid in sorted(all_unit_ids):
        got = count("none", uid, QUESTIONS, "shows_persona")
        none_rates[uid] = got["hits"]
        none_valid[uid] = got["valid"]
    bad_units = {uid for uid, hits in none_rates.items() if hits > 2}
    none_complete = sum(none_valid[uid] == 6 for uid in all_unit_ids)

    slot_reproduced = 0
    slot_patterns = {}
    for pair in pairs:
        name, uid = pair["pair"], pair["slot"]["id"]
        pattern = vals(f"{name}:matched:s+1", "s")
        same_part112_kind = part112["units"][uid]["kind"] == "slot"
        reproduced = (None not in pattern and sum(pattern) <= 2 and
                      not any(pattern[q] for q in range(2, 6)) and same_part112_kind)
        slot_patterns[name] = pattern
        slot_reproduced += reproduced
    concept_sources = {}
    for pair in pairs:
        concept_sources.setdefault(pair["match"]["id"], []).append((f"{pair['pair']}:matched:c+1", "c"))
        concept_sources.setdefault(pair["unrelated"]["id"], []).append((f"{pair['pair']}:unrelated:c+1", "c"))
    concept_counts = {}
    for uid, sources in concept_sources.items():
        rows = [count(row, role) for row, role in sources]
        complete_counts = [entry["count"] for entry in rows if entry["count"] is not None]
        if len(set(complete_counts)) > 1:
            raise ValueError(f"repeated concept unit has different alone counts: {uid}: {complete_counts}")
        concept_counts[uid] = complete_counts[0] if complete_counts else None
    concept_ge3 = sum(value is not None and value >= 3 for value in concept_counts.values())

    matched_ok = [pair for pair in pairs if not ({pair["slot"]["id"], pair["match"]["id"]} & bad_units)]
    unrelated_ok = [pair for pair in pairs if not ({pair["slot"]["id"], pair["unrelated"]["id"]} & bad_units)]

    def matched_row(pair: dict, condition: str) -> str:
        return f"{pair['pair']}:matched:{condition}"

    def unrelated_row(pair: dict, condition: str) -> str:
        return f"{pair['pair']}:unrelated:{condition}"

    def paired_comparison(source_pairs, first, second, questions=QUESTIONS):
        rows = []
        for pair in source_pairs:
            a = count(first(pair), "c", questions)["count"]
            b = count(second(pair), "c", questions)["count"]
            if a is not None and b is not None:
                rows.append((pair["pair"], a, b))
        return rows

    comp1172 = []
    for pair in matched_ok:
        together = count(matched_row(pair, "both+1"), "c", range(2, 6))["count"]
        alone = count(matched_row(pair, "c+1"), "c", range(2, 6))["count"]
        if together is not None and alone is not None:
            comp1172.append((pair["pair"], together, alone))
    who_both = []
    for pair in matched_ok:
        concept = vals(matched_row(pair, "both+1"), "c", (0,))[0]
        slot = vals(matched_row(pair, "both+1"), "s", (0,))[0]
        if concept is not None and slot is not None:
            who_both.append((pair["pair"], concept, slot))
    who_concept = sum(concept is True for _, concept, _ in who_both)
    who_slot = sum(slot is True for _, _, slot in who_both)
    who_overlap = sum(concept is True and slot is True for _, concept, slot in who_both)

    matched_reductions = paired_comparison(
        matched_ok,
        lambda pair: matched_row(pair, "c+1"),
        lambda pair: matched_row(pair, "c+1,s-1"),
    )
    unrelated_reductions = paired_comparison(
        unrelated_ok,
        lambda pair: unrelated_row(pair, "c+1"),
        lambda pair: unrelated_row(pair, "c+1,s-1"),
    )
    fewer_matched = sum(after < before for _, before, after in matched_reductions)
    fewer_unrelated = sum(after < before for _, before, after in unrelated_reductions)
    matched_deltas = Counter(after - before for _, before, after in matched_reductions)
    unrelated_deltas = Counter(after - before for _, before, after in unrelated_reductions)
    p02_reduction = next((before, after) for name, before, after in matched_reductions if name == "p02")

    half_pairs = []
    for pair in matched_ok:
        values = [value for role in ("c", "s") for value in vals(matched_row(pair, "both+0.5"), role, NONSELF)]
        if None not in values:
            half_pairs.append((pair["pair"], values))
    half_pass = sum(not any(values) for _, values in half_pairs)
    half_capital_any = sum(any(vals(matched_row(pair, "both+0.5"), role, (4,))[0] for role in ("c", "s"))
                           for pair in matched_ok)
    half_poem_any = sum(any(vals(matched_row(pair, "both+0.5"), role, (5,))[0] for role in ("c", "s"))
                        for pair in matched_ok)

    random_values = []
    random_positives = []
    random_answer_values = []
    random_answer_positives = []
    for pair in pairs:
        row = f"{pair['pair']}:random:both+1"
        for qi in QUESTIONS:
            answer_values = []
            for role in ("c", "s"):
                value = vals(row, role, (qi,))[0]
                if value is not None:
                    random_values.append(value)
                    answer_values.append(value)
                    if value:
                        random_positives.append((pair["pair"], role, qi))
            if len(answer_values) == 2:
                has_persona = any(answer_values)
                random_answer_values.append(has_persona)
                if has_persona:
                    random_answer_positives.append((pair["pair"], qi))
    coherent_values = []
    for pair in matched_ok:
        coherent_values.extend(value for value in vals(matched_row(pair, "both+2"), "c", QUESTIONS, "coherent")
                               if value is not None)
    minus_slot = [vals(matched_row(pair, "c-1,s+1"), "s", (0,))[0] for pair in matched_ok]
    alone_slot = [vals(matched_row(pair, "s+1"), "s", (0,))[0] for pair in matched_ok]
    minus_slot_valid = [value for value in minus_slot if value is not None]
    alone_slot_valid = [value for value in alone_slot if value is not None]
    minus_slot_hits = sum(value is True for value in minus_slot_valid)
    alone_slot_hits = sum(value is True for value in alone_slot_valid)

    # Cold-check text counts use question 0 and a mechanical opener rule.
    def condition_name(row_id: str) -> str:
        if row_id == "none":
            return "none"
        _, group, condition = row_id.split(":", 2)
        return f"{group}:{condition}"

    bold_prefix = re.compile(
        r"^\s*(?:(?:I am|I'm|I’m|I are|I is)\s+(?:(?:a|an|the)\s+)?)?\*\*",
        re.IGNORECASE,
    )
    q0_bold = Counter()
    q0_qwen = Counter()
    q0_totals = Counter()
    for answer in answers:
        if int(answer["question_index"]) != 0:
            continue
        group = condition_name(answer["row"])
        q0_totals[group] += 1
        q0_bold[group] += bool(bold_prefix.match(answer["answer"]))
        q0_qwen[group] += "qwen" in answer["answer"].casefold()

    def who_q0(setname: str, condition: str, role: str) -> tuple[int, int]:
        entries = [label(f"{pair['pair']}:{setname}:{condition}", role, 0) for pair in pairs]
        valid = [bool(entry["shows_persona"]) for entry in entries if entry is not None]
        return sum(valid), len(valid)

    neg_who_q0 = {
        "matched_alone": who_q0("matched", "c+1", "c"),
        "matched_slot_minus": who_q0("matched", "c+1,s-1", "c"),
        "unrelated_alone": who_q0("unrelated", "c+1", "c"),
        "unrelated_slot_minus": who_q0("unrelated", "c+1,s-1", "c"),
    }
    coherent_by_role = {}
    for role in ("c", "s"):
        coherent_by_role[role] = [
            value for pair in matched_ok
            for value in vals(matched_row(pair, "both+2"), role, QUESTIONS, "coherent")
            if value is not None
        ]
    plus2_loop_example = next(
        answer["answer"] for answer in answers
        if answer["row"] == "p01:matched:both+2" and int(answer["question_index"]) == 0
    )
    coherent_agree_hits = sum(labels[key]["coherent"] == second[key]["coherent"] for key in paired_reader_keys)
    shows_disagreements = sum(labels[key]["shows_persona"] != second[key]["shows_persona"] for key in paired_reader_keys)
    second_overlay = {key: dict(row) for key, row in labels.items()}
    for key in paired_reader_keys:
        second_overlay[key]["shows_persona"] = second[key]["shows_persona"]
        second_overlay[key]["coherent"] = second[key]["coherent"]
    second_overlay_verdicts_unchanged = (
        scorer_verdicts_in_memory(pairs, labels) == scorer_verdicts_in_memory(pairs, second_overlay)
    )
    matched_by_concept = {}
    for pair in pairs:
        matched_by_concept.setdefault(pair["match"]["id"], []).append(pair["pair"])
    shared_matched_concepts = sorted(
        (unit, names) for unit, names in matched_by_concept.items() if len(names) > 1
    )

    verdicts = {
        "117.1": "held" if slot_reproduced >= 9 and concept_ge3 >= 17 else "failed",
        "117.2": "held" if sum(together >= alone for _, together, alone in comp1172) >= 8 else "failed",
        "117.3": "held" if who_slot > who_concept else "failed",
        "117.4": "held" if fewer_matched >= 6 and fewer_unrelated <= 3 else "failed",
        "117.5": "held" if half_pass >= 8 else "failed",
        "117.6": "held" if sum(random_values) <= 2 else "failed",
        "117.7": "held" if coherent_values and sum(coherent_values) <= len(coherent_values) / 2 else "failed",
        "117.8": "held" if minus_slot_hits >= 6 else "failed",
    }

    # Check the independent calculation against the submitted score and agreement summaries.
    assert score_file["gate_slot_kinds_reproduced"] == (slot_reproduced >= 9)
    assert score_file["uninterpretable_units"] == sorted(bad_units)
    assert score_file["pairs_ok"] == len(matched_ok)
    assert score_file["who_both1"] == {"slot": who_slot, "concept": who_concept}
    assert score_file["fewer_with_slot_minus"] == {"matched": fewer_matched, "unrelated": fewer_unrelated}
    assert score_file["random_shows"] == [sum(random_values), len(random_values)]
    assert score_file["both2_coherent"] == [sum(coherent_values), len(coherent_values)]
    assert score_file["labels_missing"] == sum(1 for key in missing_keys if not key[0].startswith("none|"))
    assert score_file["verdicts"] == verdicts
    assert agreement_file["n"] == agree_n and agreement_file["agree"] == agree_hits

    who_conditions = [
        ("Matched: concept +1 alone", "matched", "c+1"),
        ("Matched: slot +1 alone", "matched", "s+1"),
        ("Matched: both +0.5", "matched", "both+0.5"),
        ("Matched: both +1", "matched", "both+1"),
        ("Matched: both +2", "matched", "both+2"),
        ("Matched: concept +1, slot -1", "matched", "c+1,s-1"),
        ("Matched: concept -1, slot +1", "matched", "c-1,s+1"),
        ("Unrelated: concept +1", "unrelated", "c+1"),
        ("Unrelated: both +1", "unrelated", "both+1"),
        ("Unrelated: concept +1, slot -1", "unrelated", "c+1,s-1"),
        ("Random directions: both +1", "random", "both+1"),
    ]
    who_rows = []
    for title, setname, condition in who_conditions:
        concept_hits = slot_hits = concept_valid = slot_valid = 0
        for pair in pairs:
            if setname == "matched":
                row = matched_row(pair, condition)
            elif setname == "unrelated":
                row = unrelated_row(pair, condition)
            else:
                row = f"{pair['pair']}:random:{condition}"
            for role in ("c", "s"):
                value = vals(row, role, (0,))[0]
                if value is not None:
                    if role == "c":
                        concept_valid += 1
                        concept_hits += value is True
                    else:
                        slot_valid += 1
                        slot_hits += value is True
        who_rows.append({"title": title, "concept_hits": concept_hits, "concept_valid": concept_valid,
                         "slot_hits": slot_hits, "slot_valid": slot_valid})

    pair_plot = {}
    for setname in ("matched", "unrelated"):
        rows = []
        for pair in pairs:
            alone = (matched_row(pair, "c+1") if setname == "matched" else unrelated_row(pair, "c+1"))
            neg = (matched_row(pair, "c+1,s-1") if setname == "matched" else unrelated_row(pair, "c+1,s-1"))
            a = count(alone, "c")["count"]
            b = count(neg, "c")["count"]
            rows.append((pair["pair"], a, b))
        pair_plot[setname] = rows

    return {
        "answers": answers, "items": items, "labels": labels, "pairs": pairs,
        "answer_row_ids": answer_row_ids, "persona_ids": persona_ids,
        "item_keys": item_keys, "missing_keys": missing_keys,
        "error_rows": error_rows, "second_rows": second_rows, "agree_n": agree_n, "agree_hits": agree_hits,
        "slot_ids": slot_ids, "matched_ids": matched_ids, "unrelated_ids": unrelated_ids,
        "match_scores": match_scores, "none_rates": none_rates, "none_valid": none_valid,
        "bad_units": bad_units, "none_complete": none_complete, "slot_patterns": slot_patterns,
        "slot_reproduced": slot_reproduced, "concept_counts": concept_counts, "concept_ge3": concept_ge3,
        "matched_ok": matched_ok, "unrelated_ok": unrelated_ok, "comp1172": comp1172, "who_both": who_both,
        "who_concept": who_concept, "who_slot": who_slot, "who_overlap": who_overlap,
        "matched_reductions": matched_reductions, "unrelated_reductions": unrelated_reductions,
        "fewer_matched": fewer_matched, "fewer_unrelated": fewer_unrelated,
        "matched_deltas": matched_deltas, "unrelated_deltas": unrelated_deltas, "p02_reduction": p02_reduction,
        "half_pairs": half_pairs, "half_pass": half_pass, "half_capital_any": half_capital_any,
        "half_poem_any": half_poem_any, "random_values": random_values, "random_positives": random_positives,
        "random_answer_values": random_answer_values, "random_answer_positives": random_answer_positives,
        "coherent_values": coherent_values, "minus_slot_hits": minus_slot_hits, "minus_slot_valid": len(minus_slot_valid),
        "alone_slot_hits": alone_slot_hits, "alone_slot_valid": len(alone_slot_valid),
        "q0_bold": q0_bold, "q0_qwen": q0_qwen, "q0_totals": q0_totals,
        "neg_who_q0": neg_who_q0, "coherent_by_role": coherent_by_role,
        "coherent_agree_hits": coherent_agree_hits, "shows_disagreements": shows_disagreements,
        "second_overlay_verdicts_unchanged": second_overlay_verdicts_unchanged,
        "plus2_loop_example": plus2_loop_example,
        "shared_matched_concepts": shared_matched_concepts,
        "verdicts": verdicts, "who_rows": who_rows, "pair_plot": pair_plot,
        "score_file": score_file, "agreement_file": agreement_file,
    }


def number_lines(data: dict) -> list[str]:
    scores = data["match_scores"]
    matched_deltas = data["matched_deltas"]
    unrelated_deltas = data["unrelated_deltas"]
    pair_results = {name: sum(a >= b for _, a, b in data["comp1172"])
                    for name in ("117.2",)}
    lines = [
        f"data.answers={len(data['answers'])}",
        f"data.row_variants={len(data['answer_row_ids'])}",
        f"data.questions_per_row={len(QUESTIONS)}",
        f"data.pairs={len(data['pairs'])}",
        f"data.slot_units={len(data['slot_ids'])}",
        f"data.matched_concept_units={len(data['matched_ids'])}",
        f"data.unrelated_concept_units={len(data['unrelated_ids'])}",
        f"data.distinct_concept_units={len(data['matched_ids'] | data['unrelated_ids'])}",
        f"pairing.theme_score_counts={compact_pairs(scores)}",
        f"data.label_items={len(data['items'])}",
        f"data.persona_targets={len(data['persona_ids'])}",
        f"labels.error_attempts={len(data['error_rows'])}",
        f"labels.successful_unique_items={len(data['labels'])}/{len(data['item_keys'])}",
        f"labels.unlabelled_items={len(data['missing_keys'])}",
        f"labels.unlabelled_keys={','.join(f'{i}#{q}' for i,q in data['missing_keys']) or 'none'}",
        f"gate.unpushed_personas_over_2={len(data['bad_units'])}/{len(data['none_rates'])}",
        f"gate.unpushed_personas_complete={data['none_complete']}/{len(data['none_rates'])}",
        f"gate.unpushed_max_shows={max(data['none_rates'].values(), default=0)}/6",
        f"gate.slot_kind_reproduction={data['slot_reproduced']}/{len(data['pairs'])} (registered threshold 9)",
        f"gate.uninterpretable_units={len(data['bad_units'])}/{len(data['none_rates'])}",
        f"prediction.117.1.slot_patterns={data['slot_reproduced']}/{len(data['pairs'])}",
        f"prediction.117.1.concepts_at_least_3={data['concept_ge3']}/{len(data['concept_counts'])}",
        f"prediction.117.1.verdict={data['verdicts']['117.1']}",
        f"prediction.117.2.met={pair_results['117.2']}/{len(data['comp1172'])} evaluable pairs",
        f"prediction.117.2.verdict={data['verdicts']['117.2']}",
        f"prediction.117.3.concept_who={data['who_concept']}/{len(data['who_both'])}",
        f"prediction.117.3.slot_who={data['who_slot']}/{len(data['who_both'])}",
        f"prediction.117.3.both_personas_who={data['who_overlap']}/{len(data['who_both'])}",
        f"prediction.117.3.verdict={data['verdicts']['117.3']}",
        f"prediction.117.4.matched_concept_decreased={data['fewer_matched']}/{len(data['matched_reductions'])}",
        f"prediction.117.4.matched_delta_histogram={compact_pairs(matched_deltas)}",
        f"prediction.117.4.unrelated_concept_decreased={data['fewer_unrelated']}/{len(data['unrelated_reductions'])}",
        f"prediction.117.4.unrelated_delta_histogram={compact_pairs(unrelated_deltas)}",
        f"prediction.117.4.p02_matched_count={data['p02_reduction'][0]}->{data['p02_reduction'][1]}",
        f"prediction.117.4.verdict={data['verdicts']['117.4']}",
        f"prediction.117.5.neither_capital_nor_poem={data['half_pass']}/{len(data['half_pairs'])}",
        f"prediction.117.5.capital_any_persona={data['half_capital_any']}/{len(data['matched_ok'])}",
        f"prediction.117.5.poem_any_persona={data['half_poem_any']}/{len(data['matched_ok'])}",
        f"prediction.117.5.verdict={data['verdicts']['117.5']}",
        f"prediction.117.6.random_persona_labels={sum(data['random_values'])}/{len(data['random_values'])}",
        f"prediction.117.6.random_answer_texts_with_persona={sum(data['random_answer_values'])}/{len(data['random_answer_values'])}",
        f"prediction.117.6.random_positive={','.join(f'{p}:{r}:q{q}' for p,r,q in data['random_positives']) or 'none'}",
        f"prediction.117.6.withdrawal_threshold=6; triggered={sum(data['random_values']) >= 6}",
        f"prediction.117.6.verdict={data['verdicts']['117.6']}",
        f"prediction.117.7.coherent_at_both_plus_2={sum(data['coherent_values'])}/{len(data['coherent_values'])}",
        f"prediction.117.7.verdict={data['verdicts']['117.7']}",
        f"prediction.117.8.slot_persona_at_c_minus1_s_plus1={data['minus_slot_hits']}/{data['minus_slot_valid']}",
        f"prediction.117.8.slot_persona_at_slot_plus1_alone={data['alone_slot_hits']}/{data['alone_slot_valid']}",
        f"prediction.117.8.verdict={data['verdicts']['117.8']}",
        f"second_reader.agreement={data['agree_hits']}/{data['agree_n']} ({data['agree_hits']/data['agree_n']*100:.1f}%)",
        "verification.score_json=all checked summaries and verdicts agree",
        f"cold.naming.bold_opener_rule=bold marker at answer start after optional first-person lead-in and article",
        f"cold.naming.bold.matched_concept_plus1={data['q0_bold']['matched:c+1']}/{data['q0_totals']['matched:c+1']}",
        f"cold.naming.bold.matched_slot_plus1={data['q0_bold']['matched:s+1']}/{data['q0_totals']['matched:s+1']}",
        f"cold.naming.bold.matched_concept_minus1_slot_plus1={data['q0_bold']['matched:c-1,s+1']}/{data['q0_totals']['matched:c-1,s+1']}",
        f"cold.naming.bold.unrelated_concept_plus1={data['q0_bold']['unrelated:c+1']}/{data['q0_totals']['unrelated:c+1']}",
        f"cold.naming.bold.random_both_plus1={data['q0_bold']['random:both+1']}/{data['q0_totals']['random:both+1']}",
        f"cold.naming.qwen.unpushed={data['q0_qwen']['none']}/{data['q0_totals']['none']}",
        f"cold.naming.qwen.matched_both_half={data['q0_qwen']['matched:both+0.5']}/{data['q0_totals']['matched:both+0.5']}",
        f"cold.naming.qwen.random_both_plus1={data['q0_qwen']['random:both+1']}/{data['q0_totals']['random:both+1']}",
        f"cold.naming.qwen.zero_in_other_pushed_conditions={sum(data['q0_qwen'][g] == 0 for g in data['q0_qwen'] if g not in {'none', 'matched:both+0.5', 'random:both+1'})}/9 (each 0/11)",
        f"cold.117.4.who_concept_alone_matched={data['neg_who_q0']['matched_alone'][0]}/{data['neg_who_q0']['matched_alone'][1]}",
        f"cold.117.4.who_concept_slot_minus_matched={data['neg_who_q0']['matched_slot_minus'][0]}/{data['neg_who_q0']['matched_slot_minus'][1]}",
        f"cold.117.4.who_concept_alone_unrelated={data['neg_who_q0']['unrelated_alone'][0]}/{data['neg_who_q0']['unrelated_alone'][1]}",
        f"cold.117.4.who_concept_slot_minus_unrelated={data['neg_who_q0']['unrelated_slot_minus'][0]}/{data['neg_who_q0']['unrelated_slot_minus'][1]}",
        f"cold.117.7.coherent_plus2_concept={sum(data['coherent_by_role']['c'])}/{len(data['coherent_by_role']['c'])}",
        f"cold.117.7.coherent_plus2_slot={sum(data['coherent_by_role']['s'])}/{len(data['coherent_by_role']['s'])}",
        f"cold.117.7.coherent_plus2_both_targets={sum(data['coherent_by_role']['c']) + sum(data['coherent_by_role']['s'])}/{len(data['coherent_by_role']['c']) + len(data['coherent_by_role']['s'])}",
        f"cold.117.7.loop_example=p01:matched:both+2:q0:{data['plus2_loop_example'].replace(chr(10), ' / ')}",
        f"cold.second_reader.coherent_agreement={data['coherent_agree_hits']}/{data['agree_n']} ({data['coherent_agree_hits']/data['agree_n']*100:.1f}%)",
        f"cold.second_reader.shows_disagreements={data['shows_disagreements']}/{data['agree_n']}",
        f"cold.second_reader.verdicts_after_sample_overlay={len(data['verdicts'])}/{len(data['verdicts'])} unchanged={data['second_overlay_verdicts_unchanged']}",
        f"cold.limits.shared_matched_concept_count={len(data['shared_matched_concepts'])}",
        f"cold.limits.shared_matched_concepts={';'.join(','.join(names)+':'+unit for unit,names in data['shared_matched_concepts']) or 'none'}",
    ]
    for row in data["who_rows"]:
        lines.append(f"figure.who.{row['title']}=concept {row['concept_hits']}/{row['concept_valid']}; slot {row['slot_hits']}/{row['slot_valid']}")
    return lines


def make_figures(data: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = data["who_rows"]
    fig, ax = plt.subplots(figsize=(11, 7.3))
    y = np.arange(len(rows))
    height = 0.36
    concepts = [row["concept_hits"] for row in rows]
    slots = [row["slot_hits"] for row in rows]
    bars_c = ax.barh(y - height/2, concepts, height, label="Concept persona", color="#2878B5")
    bars_s = ax.barh(y + height/2, slots, height, label="Slot persona", color="#D47A23")
    for bars, counts, valids in ((bars_c, concepts, [r["concept_valid"] for r in rows]),
                                 (bars_s, slots, [r["slot_valid"] for r in rows])):
        for rect, hits, valid in zip(bars, counts, valids):
            ax.text(rect.get_width() + 0.12, rect.get_y() + rect.get_height()/2,
                    f"{hits}/{valid}", va="center", fontsize=8)
    ax.set_yticks(y, [row["title"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 12.6)
    ax.set_xticks(range(12))
    ax.set_xlabel('Pairs whose "Who are you?" answer received a persona label')
    ax.set_title('Persona labels on “Who are you?” by condition')
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig-117-who.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), sharey=True)
    for ax, setname, title in zip(axes, ("matched", "unrelated"), ("Matched theme", "Unrelated theme")):
        pair_rows = data["pair_plot"][setname]
        for idx, (pair, before, after) in enumerate(pair_rows):
            if before is None or after is None:
                continue
            ax.plot([0, 1], [before, after], color="#9CA3AF", alpha=0.7, linewidth=1)
            ax.scatter(0, before, color="#2878B5", s=30, zorder=3)
            ax.scatter(1, after, color="#D47A23", marker="s", s=28, zorder=3)
        ax.set_xticks([0, 1], ["Concept +1", "Concept +1,\nslot -1"])
        ax.set_ylim(-0.25, 6.25)
        ax.set_yticks(range(7))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Concept persona labels across six answers")
    fig.suptitle("Per-pair concept counts with and without a negative slot push")
    fig.tight_layout()
    fig.savefig(HERE / "fig-117-negative.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers", action="store_true", help="print recomputed counts without drawing figures")
    args = parser.parse_args()
    data = calculate()
    if not args.numbers:
        make_figures(data)
    print("\n".join(number_lines(data)))


if __name__ == "__main__":
    main()
