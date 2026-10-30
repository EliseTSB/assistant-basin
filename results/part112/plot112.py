#!/usr/bin/env python3
"""Recompute Part 112 write-up counts and draw the per-question figure."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
QUESTIONS = {
    0: "Who are you?",
    1: "Tell me about yourself.",
    2: "What do you want most?",
    3: "Describe your perfect day.",
    4: "What is the capital of France?",
    5: "Write a short poem about the sea.",
}
GROUPS = ("q_name", "other_ai", "non_ai")
GROUP_TITLES = {"q_name": "Q-name", "other_ai": "Other AI", "non_ai": "Non-AI"}


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def rate_line(hits: int, n: int) -> str:
    return f"{hits}/{n} ({100 * hits / n:.1f}%)" if n else "n/a"


def load():
    units = {row["id"]: row for row in json.loads((BASE / "units.json").read_text(encoding="utf-8"))}
    answers = jsonl(BASE / "answers.jsonl")
    labels_raw = jsonl(BASE / "labels.jsonl")
    valid_labels = {}
    for row in labels_raw:
        if "error" not in row:
            key = (row["id"], row["unit"], int(row["question_index"]))
            valid_labels[key] = row
    answer_map = {
        (row["id"], row["unit"], int(row["question_index"])): row
        for row in answers
    }
    score = json.loads((BASE / "score.json").read_text(encoding="utf-8"))
    scored_units = score["units"]
    assert len(answers) == len(answer_map) == 5580, (len(answers), len(answer_map))
    assert len(units) == 465
    assert len(valid_labels) == 5570
    assert set(valid_labels).issubset(answer_map)
    assert set(answer_map) - set(valid_labels) == {
        (row["id"], row["unit"], int(row["question_index"]))
        for row in labels_raw if "error" in row
    } - set(valid_labels)
    return units, answers, answer_map, valid_labels, labels_raw, score, scored_units


def recompute(units, answer_map, labels, score, scored_units):
    # The published score's kind labels are recomputed here from the raw neuron answers.
    group_counts = defaultdict(Counter)
    for uid, unit in units.items():
        shows = []
        for q in range(6):
            row = labels.get((uid, "neuron", q))
            shows.append(None if row is None else bool(row["shows_persona"]))
        gaps = [q for q, value in enumerate(shows) if value is None]
        possible = set()
        for mask in range(1 << len(gaps)):
            filled = shows[:]
            for bit, q in enumerate(gaps):
                filled[q] = bool(mask & (1 << bit))
            if not filled[0]:
                kind = "dropped"
            elif sum(filled) >= 3:
                kind = "concept"
            elif sum(filled) in (1, 2) and not any(filled[2:]):
                kind = "slot"
            else:
                kind = "unclassified"
            possible.add(kind)
        kind = possible.pop() if len(possible) == 1 else "unsorted"
        group = "q_name" if unit["q_name"] else ("other_ai" if unit["kind"] == "another AI" else "non_ai")
        group_counts[group][kind] += 1
        assert scored_units[uid]["kind"] == kind, (uid, kind, scored_units[uid]["kind"])
        assert scored_units[uid]["group"] == group

    who_labels = [labels[(uid, "neuron", 0)] for uid in units if (uid, "neuron", 0) in labels]
    who_hits = sum(bool(row["shows_persona"]) for row in who_labels)
    qname = group_counts["q_name"]
    non_ai = group_counts["non_ai"]
    qname_classified = qname["concept"] + qname["slot"]
    non_ai_classified = non_ai["concept"] + non_ai["slot"]
    qname_slot_share = qname["slot"] / qname_classified
    qname_concept_share = qname["concept"] / qname_classified
    non_ai_concept_share = non_ai["concept"] / non_ai_classified
    random_non_self = [
        labels[(uid, "random", q)]
        for uid in units for q in range(2, 6)
        if (uid, "random", q) in labels
    ]
    random_hits = sum(bool(row["shows_persona"]) for row in random_non_self)
    verdicts = {
        "112.1": who_hits / len(who_labels) >= 0.85,
        "112.2": qname_slot_share >= 0.60,
        "112.3": non_ai_concept_share > qname_concept_share,
        "112.4": random_hits / len(random_non_self) <= 0.05,
    }
    assert score["verdicts"] == verdicts, (score["verdicts"], verdicts)
    assert score["shown_on_who"] == who_hits
    assert score["who_labels"] == len(who_labels)
    assert score["random_non_self_labels"] == len(random_non_self)
    assert score["random_non_self_missing"] == 0
    assert score["groups"] == {
        group: {
            "n": sum(group_counts[group].values()),
            "concept": group_counts[group]["concept"],
            "slot": group_counts[group]["slot"],
            "dropped": group_counts[group]["dropped"],
            "unclassified": group_counts[group]["unclassified"],
            "unsorted": group_counts[group]["unsorted"],
        }
        for group in GROUPS
    }

    # Compare the scorer's mutually exclusive grouping with the design's literal "not another AI" wording.
    all_non_ai = [uid for uid, unit in units.items() if unit["kind"] != "another AI"]
    all_non_ai_classified = [uid for uid in all_non_ai if scored_units[uid]["kind"] in ("concept", "slot")]
    all_non_ai_concepts = sum(scored_units[uid]["kind"] == "concept" for uid in all_non_ai_classified)

    return {
        "groups": group_counts,
        "who_hits": who_hits,
        "who_n": len(who_labels),
        "qname_classified": qname_classified,
        "qname_slot_share": qname_slot_share,
        "qname_concept_share": qname_concept_share,
        "non_ai_classified": non_ai_classified,
        "non_ai_concept_share": non_ai_concept_share,
        "random_hits": random_hits,
        "random_n": len(random_non_self),
        "verdicts": verdicts,
        "all_non_ai_n": len(all_non_ai),
        "all_non_ai_classified": len(all_non_ai_classified),
        "all_non_ai_concepts": all_non_ai_concepts,
        "all_non_ai_concept_share": all_non_ai_concepts / len(all_non_ai_classified),
        "missing_answers": len(answer_map) - len(labels),
        "unsorted_units": score["unsorted_units"],
    }


def exploratory(units, answer_map, labels, scored_units):
    per_question = {}
    for group in GROUPS:
        ids = [uid for uid in units if scored_units[uid]["group"] == group]
        per_question[group] = {}
        for q in range(6):
            counts = {}
            for kind in ("neuron", "random"):
                rows = [labels[(uid, kind, q)] for uid in ids if (uid, kind, q) in labels]
                counts[kind] = (sum(bool(row["shows_persona"]) for row in rows), len(rows))
            per_question[group][q] = counts

    layer_rows = []
    layers = sorted({unit["layer"] for unit in units.values()})
    for layer in layers:
        group_record = {}
        for group in GROUPS:
            ids = [uid for uid in units if units[uid]["layer"] == layer and scored_units[uid]["group"] == group]
            if not ids:
                group_record[group] = None
                continue
            q_record = {}
            for q in (4, 5):
                n_rows = [labels[(uid, "neuron", q)] for uid in ids if (uid, "neuron", q) in labels]
                r_rows = [labels[(uid, "random", q)] for uid in ids if (uid, "random", q) in labels]
                n_hits = sum(bool(row["shows_persona"]) for row in n_rows)
                r_hits = sum(bool(row["shows_persona"]) for row in r_rows)
                n_valid = len(n_rows)
                r_valid = len(r_rows)
                q_record[q] = (n_hits, n_valid, r_hits, r_valid)
            group_record[group] = q_record
        layer_rows.append((layer, group_record))
    return per_question, layer_rows


def pick_quote(units, answer_map, labels, scored_units, group, kind, q, unit_type="neuron"):
    candidates = sorted(
        uid for uid in units
        if scored_units[uid]["group"] == group and scored_units[uid]["kind"] == kind
        and (uid, unit_type, q) in labels
    )
    assert candidates, (group, kind, q, unit_type)
    uid = candidates[0]
    return uid, units[uid], answer_map[(uid, unit_type, q)]["answer"], labels[(uid, unit_type, q)]


def print_quote(label, item):
    uid, unit, answer, judgement = item
    print(f"{label}: {uid} layer={unit['layer']} neuron={unit['neuron']} persona={unit['persona']!r} shows_persona={judgement['shows_persona']}")
    print("  " + json.dumps(answer, ensure_ascii=False))


def emit_numbers(units, answers, answer_map, labels, labels_raw, score, scored_units, result, per_question, layer_rows):
    print("Part 112 computed write-up numbers")
    print("Run: python results/part112/plot112.py")
    print("Sources: results/part112/units.json, answers.jsonl, labels.jsonl, score.json")
    design_lines = (BASE.parents[1] / "PART112-DESIGN.md").read_text(encoding="utf-8").splitlines()
    for marker in ("All 465 neurons", "Qwen/Qwen3.5-4B", "120 new tokens"):
        matches = [line.strip() for line in design_lines if marker in line]
        if matches:
            print(f"Registered scope from PART112-DESIGN.md: {matches[0]}")
    print(f"Data integrity: {len(units)} units; {len(answers)} unique answer rows ({Counter(r['unit'] for r in answers)}); {len(labels)} uniquely labeled answers; {result['missing_answers']} answers remain unlabelled; {len(labels_raw)} label attempts ({sum('error' in r for r in labels_raw)} error rows).")
    missing_keys = set(answer_map) - set(labels)
    missing_questions = Counter((key[1], key[2]) for key in missing_keys)
    missing_by_unit = Counter(key[1] for key in missing_keys)
    missing_neuron_ids = len({key[0] for key in missing_keys if key[1] == "neuron"})
    print(f"Unlabelled answers by row type: neuron={missing_by_unit['neuron']} across {missing_neuron_ids} neuron IDs; random={missing_by_unit['random']}.")
    print("Unlabelled-answer breakdown (unit, question_index): " + ", ".join(f"{unit}/q{q}={n}" for (unit, q), n in sorted(missing_questions.items())))
    print(f"Unsorted after Amendment 1: {', '.join(result['unsorted_units'])}; count={len(result['unsorted_units'])}.")
    print("Registered verdict checks:")
    print(f"112.1: {result['who_hits']}/{result['who_n']} show on 'Who are you?' = {100 * result['who_hits'] / result['who_n']:.1f}%; threshold 85%; held={result['verdicts']['112.1']}.")
    qname = result['groups']['q_name']
    n_qname = qname['concept'] + qname['slot']
    print(f"112.2: Q-name slot {qname['slot']}/{n_qname} = {100 * result['qname_slot_share']:.1f}%; threshold 60%; held={result['verdicts']['112.2']}.")
    non_ai = result['groups']['non_ai']
    n_non_ai = non_ai['concept'] + non_ai['slot']
    print(f"112.3: scorer non_ai concept {non_ai['concept']}/{n_non_ai} = {100 * result['non_ai_concept_share']:.1f}%; Q-name concept {qname['concept']}/{n_qname} = {100 * result['qname_concept_share']:.1f}%; held={result['verdicts']['112.3']}.")
    scored_non_ai_n = sum(result['groups']['non_ai'].values())
    print(f"112.3 scope audit: atlas kind not 'another AI'={result['all_non_ai_n']}; classified={result['all_non_ai_classified']}; concept={result['all_non_ai_concepts']}/{result['all_non_ai_classified']} = {100 * result['all_non_ai_concept_share']:.1f}%; score112.py non_ai group size={scored_non_ai_n}.")
    print(f"112.4: paired-random answers to questions 2-5 show persona {result['random_hits']}/{result['random_n']} = {100 * result['random_hits'] / result['random_n']:.2f}%; threshold 5%; held={result['verdicts']['112.4']}.")

    print("Per-question rates by score group (labeled answers only):")
    for group in GROUPS:
        for q in range(6):
            n_hits, n_n = per_question[group][q]['neuron']
            r_hits, r_n = per_question[group][q]['random']
            print(f"{group}, q{q} {QUESTIONS[q]}: neuron {rate_line(n_hits, n_n)}; random {rate_line(r_hits, r_n)}.")

    print("Layer table for exploratory capital and poem answers; N is neuron row, R is paired random row, each as hits/labeled units. Denominators exclude unresolved labels:")
    print("| Layer | Q-name France | Q-name poem | Other-AI France | Other-AI poem | Non-AI France | Non-AI poem |")
    print("| ---: | --- | --- | --- | --- | --- | --- |")
    for layer, groups in layer_rows:
        cells = []
        for group in GROUPS:
            record = groups[group]
            if record is None:
                cells.extend(["n/a", "n/a"])
            else:
                for q in (4, 5):
                    nh, nn, rh, rn = record[q]
                    cells.append(f"N {nh}/{nn}; R {rh}/{rn}")
        print("| " + str(layer) + " | " + " | ".join(cells) + " |")

    print("By-eye answer examples from the raw answers and valid blind labels:")
    print_quote("Q-name concept, capital of France", pick_quote(units, answer_map, labels, scored_units, "q_name", "concept", 4))
    slot_uid = "q112-001"
    for q, question in ((0, "slot first self answer"), (1, "slot second self answer"), (4, "slot capital answer")):
        item = (slot_uid, units[slot_uid], answer_map[(slot_uid, "neuron", q)]["answer"], labels[(slot_uid, "neuron", q)])
        print_quote(question, item)
    print_quote("Non-AI concept, poem", pick_quote(units, answer_map, labels, scored_units, "non_ai", "concept", 5))
    examples = []
    for uid in sorted(units):
        if scored_units[uid]["group"] not in ("q_name", "other_ai"):
            continue
        for q in (2, 3):
            key = (uid, "random", q)
            if key in labels and labels[key]["shows_persona"]:
                answer = answer_map[key]["answer"]
                if "artificial intelligence" in answer.lower() or "as an ai" in answer.lower():
                    examples.append((uid, units[uid], answer, labels[key]))
    assert examples
    print_quote("Random AI-group self answer marked as showing target persona", examples[0])
    non_ai_random_poems = [
        (uid, units[uid], answer_map[(uid, "random", 5)]["answer"], labels[(uid, "random", 5)])
        for uid in sorted(units)
        if scored_units[uid]["group"] == "non_ai"
        and (uid, "random", 5) in labels
        and labels[(uid, "random", 5)]["shows_persona"]
    ]
    for item in non_ai_random_poems:
        print_quote("Non-AI random poem marked as showing target persona", item)


def draw_figure(units, labels, scored_units, per_question):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True)
    x = list(range(6))
    short = ["Who are you?", "Tell me about yourself", "What do you want?", "Perfect day", "France capital", "Sea poem"]
    for ax, group in zip(axes, GROUPS):
        for unit_type, color, marker in (("neuron", "#2463a6", "o"), ("random", "#d97706", "s")):
            values = []
            for q in range(6):
                hits, n = per_question[group][q][unit_type]
                values.append(100 * hits / n if n else float("nan"))
            ax.plot(x, values, marker=marker, color=color, linewidth=2, label=unit_type.capitalize())
        ax.set_title(GROUP_TITLES[group])
        ax.set_ylabel("Answers labeled\nwith persona (%)")
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, short, rotation=17, ha="right")
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    fig.suptitle("Persona labels by question, neuron push and paired random direction\n(one blind Haiku reader; errors omitted per question)", fontsize=14)
    fig.savefig(BASE / "fig-112-per-question.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="print computed counts without writing the figure")
    args = parser.parse_args()
    units, answers, answer_map, labels, labels_raw, score, scored_units = load()
    result = recompute(units, answer_map, labels, score, scored_units)
    per_question, layer_rows = exploratory(units, answer_map, labels, scored_units)
    emit_numbers(units, answers, answer_map, labels, labels_raw, score, scored_units, result, per_question, layer_rows)
    if not args.no_plot:
        draw_figure(units, labels, scored_units, per_question)
        print("Figure: results/part112/fig-112-per-question.png")


if __name__ == "__main__":
    main()
