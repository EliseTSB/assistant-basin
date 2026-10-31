#!/usr/bin/env python3
"""Part 150 scorer: gates, word rates and the registered predictions."""
import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABSTRACT = re.compile(r"\b[a-z]{3,}(?:ness|ity|ism|ance|ence|ship|ment|tion|sion)\b", re.IGNORECASE)
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
TRIAGE_WORD = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
ANAPHORS = {"them", "these", "those", "they", "which", "themselves"}
UNITS = ("L19N4979", "L19N5975", "L23N116", "L24N6835", "L24N8542")
NEURON_DOSES = (0.4, 0.8, 1.5, 2.5, -0.8, -1.5)
TEST_QUESTIONS = (1, 2, 3, 4, 5)  # "Who are you?" chose these units, so the write tests leave it out.


def words_of(text):
    return [w.lower() for w in WORD.findall(text)]


def abstract_rate(text):
    """Distinct abstract nouns per 100 words, so a looping answer does not count the same word again."""
    words = words_of(text)
    return 100.0 * len({m.lower() for m in ABSTRACT.findall(text)}) / len(words) if words else 0.0


def lens_rate(text, lens):
    """Distinct answer words that start with a lens word, per 100 words."""
    words = words_of(text)
    return 100.0 * len({w for w in set(words) if any(w.startswith(s) for s in lens)}) / len(words) if words else 0.0


def whole_lens_rate(text, lens):
    """Distinct exact lens-word matches per 100 words, excluding lens fragments under five letters."""
    words = words_of(text)
    eligible = {word.lower() for word in lens if len(word) >= 5}
    return 100.0 * len(set(words) & eligible) / len(words) if words else 0.0


def anaphor_rate(text):
    """Occurrences of plural anaphors per 100 words."""
    words = words_of(text)
    return 100.0 * sum(w in ANAPHORS for w in words) / len(words) if words else 0.0


def broken(text):
    """Part 86's rule: under four words, or a three-word sequence repeated at least three times."""
    words = [w.casefold() for w in TRIAGE_WORD.findall(text)]
    if len(words) < 4:
        return True
    seen = {}
    for i in range(len(words) - 2):
        key = tuple(w.lower() for w in words[i:i + 3])
        seen[key] = seen.get(key, 0) + 1
    return max(seen.values(), default=0) >= 3


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def cell(rows, unit, dose, fn):
    """Mean of fn over the five test questions."""
    vals = [fn(r["answer"]) for r in rows if r["unit"] == unit and abs(r["dose"] - dose) < 1e-9 and r["question_index"] in TEST_QUESTIONS]
    if len(vals) != len(TEST_QUESTIONS):
        raise ValueError(f"expected {len(TEST_QUESTIONS)} answers for {unit} at {dose}, got {len(vals)}")
    return sum(vals) / len(vals)


def maybe_cell(rows, unit, dose, fn):
    """Cell mean when a fixture or partial run has all five test answers; otherwise None."""
    vals = [fn(r["answer"]) for r in rows if r["unit"] == unit and abs(r["dose"] - dose) < 1e-9 and r["question_index"] in TEST_QUESTIONS]
    return sum(vals) / len(vals) if len(vals) == len(TEST_QUESTIONS) else None


def unbroken_cell(rows, unit, dose, fn):
    """Mean of fn over tested answers that pass the Part 86 broken-answer rule."""
    vals = [fn(r["answer"]) for r in rows if r["unit"] == unit and abs(r["dose"] - dose) < 1e-9
            and r["question_index"] in TEST_QUESTIONS and not broken(r["answer"])]
    return sum(vals) / len(vals) if vals else None


def broken_cell(rows, unit, dose):
    """Number of the six generated answers classified broken by the Part 86 rule."""
    return sum(broken(r["answer"]) for r in rows if r["unit"] == unit and abs(r["dose"] - dose) < 1e-9)


def medians(meta, unit):
    out = {}
    for a in meta["activations"]:
        out.setdefault(a["class"], []).append(a["activations"][unit])
    return {k: st.median(v) for k, v in out.items()}


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def layer_of(unit):
    return int(re.match(r"L(\d+)N", unit).group(1))


def score(in_dir, lens_path=None, chat="instruct", base="base"):
    in_dir = Path(in_dir)
    lens = json.loads(Path(lens_path or HERE / "lens_words.json").read_text(encoding="utf-8"))
    rows = {t: read_jsonl(in_dir / f"answers-{t}.jsonl") for t in (chat, base)}
    meta = {t: json.loads((in_dir / f"meta-{t}.json").read_text(encoding="utf-8")) for t in (chat, base)}
    expected_rows = 6 * (1 + 5 * 6 + 3 * 2 * 4)
    gates = {
        "gate1_zero_matches_no_hook": all(meta[t]["gate1_zero_matches_no_hook"] for t in (chat, base)),
        "gate3_rows": all(len(rows[t]) == expected_rows and len(meta[t]["activations"]) == 120 for t in (chat, base)),
        "gate4_marks_are_the_word": all(a["token"].lstrip("Ġ ").lower() == a["marked"].strip().lower()
                                        for t in (chat, base) for a in meta[t]["activations"]),
    }
    all_passed = all(gates.values())
    # Amendment 1: the Gate 2 text came from a Qwen3-1.7B atlas, so it is reported and no longer stops the reading.
    gates["gate2_reported_only"] = [a["activation"] for a in meta[chat]["atlas_check"]]
    table = {}
    for t in (chat, base):
        table[t] = {}
        for unit in UNITS:
            rand = [f"random-L{layer_of(unit)}-{i}" for i in range(4)]
            fn = lambda text, u=unit: lens_rate(text, lens[u])
            exact_fn = lambda text, u=unit: whole_lens_rate(text, lens[u])
            exact = {"plain": cell(rows[t], "unpushed", 0.0, exact_fn),
                     **{str(d): cell(rows[t], unit, d, exact_fn) for d in NEURON_DOSES},
                     "random_0.8": [maybe_cell(rows[t], r, 0.8, exact_fn) for r in rand],
                     "random_1.5": [maybe_cell(rows[t], r, 1.5, exact_fn) for r in rand]}
            exact_unbroken = {"plain": unbroken_cell(rows[t], "unpushed", 0.0, exact_fn),
                              **{str(d): unbroken_cell(rows[t], unit, d, exact_fn) for d in NEURON_DOSES},
                              "random_0.8": [unbroken_cell(rows[t], r, 0.8, exact_fn) for r in rand],
                              "random_1.5": [unbroken_cell(rows[t], r, 1.5, exact_fn) for r in rand]}
            breaks = {"plain": broken_cell(rows[t], "unpushed", 0.0),
                      **{str(d): broken_cell(rows[t], unit, d) for d in NEURON_DOSES},
                      "random_0.8": [broken_cell(rows[t], r, 0.8) for r in rand],
                      "random_1.5": [broken_cell(rows[t], r, 1.5) for r in rand]}
            table[t][unit] = {
                "lens": {"plain": cell(rows[t], "unpushed", 0.0, fn), **{str(d): cell(rows[t], unit, d, fn) for d in NEURON_DOSES},
                         "random_0.8": [cell(rows[t], r, 0.8, fn) for r in rand],
                         "random_1.5": [maybe_cell(rows[t], r, 1.5, fn) for r in rand]},
                "whole_lens": exact,
                "whole_lens_unbroken": exact_unbroken,
                "broken": breaks,
                "abstract": {"plain": cell(rows[t], "unpushed", 0.0, abstract_rate), **{str(d): cell(rows[t], unit, d, abstract_rate) for d in NEURON_DOSES},
                             "random_0.8": [cell(rows[t], r, 0.8, abstract_rate) for r in rand],
                             "random_1.5": [maybe_cell(rows[t], r, 1.5, abstract_rate) for r in rand]},
                "anaphor": {"plain": cell(rows[t], "unpushed", 0.0, anaphor_rate), **{str(d): cell(rows[t], unit, d, anaphor_rate) for d in NEURON_DOSES},
                            "random_0.8": [cell(rows[t], r, 0.8, anaphor_rate) for r in rand],
                            "random_1.5": [maybe_cell(rows[t], r, 1.5, anaphor_rate) for r in rand]},
            }
    m = {t: {u: medians(meta[t], u) for u in UNITS} for t in (chat, base)}
    a = m[chat]["L19N4979"]
    b = m[base]["L19N4979"]
    n116 = m[chat]["L23N116"]
    diff_chat = a["relative_that"] - a["dem_that"]
    diff_base = b["relative_that"] - b["dem_that"]
    abstract = table[chat]["L19N4979"]["abstract"]
    cos = {u: cosine(meta[chat]["columns"][u], meta[base]["columns"][u]) for u in UNITS}
    beat = lambda t, u: table[t][u]["lens"]["0.8"] > max(table[t][u]["lens"]["random_0.8"])
    held = {
        "150.1": diff_chat >= 5,
        "150.2": a["relative_that"] - a["comp_that"] >= 3,
        "150.3": a["free_what"] - a["question_what"] >= 3,
        "150.4": n116["of_pron"] - n116["of_noun"] >= 10,
        "150.5": n116["verb_them"] - n116["verb_other"] >= 10,
        "150.6": abstract["0.8"] >= abstract["plain"] + 1 and abstract["0.8"] > max(abstract["random_0.8"]),
        "150.7": sum(beat(chat, u) for u in UNITS) >= 3,
        "150.8": abstract["1.5"] >= abstract["0.8"] >= abstract["0.4"],
        "150.9": abstract["-0.8"] <= abstract["plain"],
        "150.10": table[chat]["L23N116"]["anaphor"]["0.8"] > max(table[chat]["L23N116"]["anaphor"]["random_0.8"]),
        "150.11": diff_chat > 0 and diff_base >= diff_chat / 2,
        "150.12": sum(cos[u] >= 0.9 for u in UNITS) >= 4,
        "150.13": sum(beat(base, u) for u in UNITS) >= 3,
    }
    preds = {k: ("held" if v else "failed") if all_passed else "withheld" for k, v in held.items()}
    broken_counts = {t: {f"{u}@{d}": sum(broken(r["answer"]) for r in rows[t] if r["unit"] == u and abs(r["dose"] - d) < 1e-9)
                         for u, d in sorted({(r["unit"], r["dose"]) for r in rows[t]})} for t in (chat, base)}
    return {"gates": gates, "all_passed": all_passed, "predictions": preds, "activation_medians": m, "rates": table,
            "column_cosine": cos, "broken": broken_counts, "atlas_check": {t: meta[t]["atlas_check"] for t in (chat, base)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    result = score(a.input)
    Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("gates", "predictions", "column_cosine")}, ensure_ascii=False))
    if not result["all_passed"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
