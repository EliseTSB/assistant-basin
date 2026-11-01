"""Part 165 secondary tables, added by Amendment 4 before any answer was read.

Two descriptive measures per arm, with no verdicts. Malla et al.'s repetition
measure is the fraction of distinct adjacent word pairs, which collapses when
a model repeats itself. Refusal is scored apart from breakage, because a dense
push may send the model to its default answer rather than break it.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ARMS = ("plain", "n+", "n±", "k4", "k16", "k64", "k256", "k1024", "full", "iso")
WORDS = re.compile(r"\w+", re.UNICODE)
REFUSAL = re.compile(r"(?:\bi'?m sorry\b|\bi am sorry\b|\bsorry\b|\bi can(?:'|no)t\b|\bi cannot\b)", re.IGNORECASE)
SENTENCE_END = re.compile(r"[.!?\n]")


def distinct_pair_fraction(answer):
    """Distinct adjacent word pairs over all pairs, or None under two words."""
    words = [word.casefold() for word in WORDS.findall(answer)]
    pairs = list(zip(words, words[1:]))
    if not pairs:
        return None
    return len(set(pairs)) / len(pairs)


def first_sentence(answer):
    text = answer.lstrip()
    match = SENTENCE_END.search(text)
    return text if match is None else text[:match.start()]


def opens_with_refusal(answer):
    return REFUSAL.search(first_sentence(answer)) is not None


def summarise(rows):
    pair_values = defaultdict(list)
    refusals = defaultdict(int)
    totals = defaultdict(int)
    short = defaultdict(int)
    for row in rows:
        arm = row["arm"]
        totals[arm] += 1
        fraction = distinct_pair_fraction(row["answer"])
        if fraction is None:
            short[arm] += 1
        else:
            pair_values[arm].append(fraction)
        if opens_with_refusal(row["answer"]):
            refusals[arm] += 1
    result = {}
    for arm in ARMS:
        total = totals[arm]
        values = pair_values[arm]
        result[arm] = {
            "answers": total,
            "mean_distinct_pair_fraction": sum(values) / len(values) if values else None,
            "under_two_words": short[arm],
            "refusal_openers": refusals[arm],
            "refusal_rate": refusals[arm] / total if total else None,
        }
    return result


def write_tables(result, out):
    out = Path(out)
    (out / "secondary.json").write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    lines = ["# Part 165 secondary tables", "",
             "Descriptive, no verdicts. Added by Amendment 4 before any answer was read.", "",
             "| Arm | Answers | Mean distinct word pairs | Under two words | Refusal openers | Refusal rate |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        row = result[arm]
        pairs = "n/a" if row["mean_distinct_pair_fraction"] is None else f"{row['mean_distinct_pair_fraction']:.4f}"
        rate = "n/a" if row["refusal_rate"] is None else f"{row['refusal_rate']:.4f}"
        lines.append(f"| {arm} | {row['answers']} | {pairs} | {row['under_two_words']} | "
                     f"{row['refusal_openers']} | {rate} |")
    (out / "secondary-tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.answers, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    write_tables(summarise(rows), args.out)


if __name__ == "__main__":
    main()
