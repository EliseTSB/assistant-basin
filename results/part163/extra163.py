#!/usr/bin/env python3
"""Print Part 163 sensitivity, breakage, coupling, layer, and company summaries."""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME_TOKENS = ("qwen", "tongyi", "通义", "千问")
MAKER_TOKENS = ("alibaba", "阿里")
# Part 163's sensitivity pattern: a first-sentence word Q plus digits, Q alone,
# or Qian/Qiao/Qiang followed by zero or more word characters.
Q_INITIAL = re.compile(r"\b(?:Q\d[\w.-]*|Qian\w*|Qiao\w*|Qiang\w*|Q)\b", re.I)
QIANWEN = re.compile(r"\bqianwen\b", re.I)
KNOWN_IDENTITY = re.compile(r"\b(?:qwen(?:\s*[- ]?\d+(?:\.\d+)*)?|tongyi|alibaba)\b", re.I)
WIDER_SELF_DESCRIPTION = re.compile(r"(?<![\w.])(?:a\.i\.|ai|assistant|language\s+model|model|chatbot|llm)(?!\w)", re.I)
WORD = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def broken_reason(answer):
    if KNOWN_IDENTITY.search(answer) or WIDER_SELF_DESCRIPTION.search(first_sentence(answer)):
        return None
    words = [w.casefold() for w in WORD.findall(answer if isinstance(answer, str) else "")]
    if len(words) < 4:
        return "too_few_words"
    trigrams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    if any(n >= 3 for n in trigrams.values()):
        return "repeated_three_word_sequence"
    return None


def first_sentence(answer):
    return re.split(r"(?<=[.!?])\s|\n", answer.strip(), maxsplit=1)[0]


def q_initial_first(answer):
    return bool(Q_INITIAL.search(first_sentence(answer)))


def gone_name(answer, mode="base"):
    text = answer.casefold()
    if any(t in text for t in NAME_TOKENS):
        return False
    if mode == "qianwen" and QIANWEN.search(answer):
        return False
    if mode == "q_initial_first" and q_initial_first(answer):
        return False
    return True


def gone_maker(answer):
    text = answer.casefold()
    return not any(t in text for t in MAKER_TOKENS)


def percentile(xs, p):
    xs = sorted(xs)
    x = (len(xs) - 1) * p
    lo, hi = int(x), int(x) + (x % 1 > 0)
    return xs[lo] if lo == hi else xs[lo] * (hi - x) + xs[hi] * (x - lo)


def paired_ratio(by_arm, seed=163, reps=10000):
    pairs = sorted(set(by_arm["neuron"]) & set(by_arm["random"]))
    pairs = [p for p in pairs if by_arm["neuron"][p] is not None and by_arm["random"][p] is not None]
    n = sum(by_arm["neuron"][p] for p in pairs) / len(pairs)
    r = sum(by_arm["random"][p] for p in pairs) / len(pairs)
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        nm = sum(by_arm["neuron"][p] for p in draw) / len(draw)
        rm = sum(by_arm["random"][p] for p in draw) / len(draw)
        if rm > 0:
            samples.append(nm / rm)
    return {"neuron_rate": n, "random_rate": r, "ratio": n / r,
            "ratio_interval_95": [percentile(samples, .025), percentile(samples, .975)], "pairs": len(pairs)}


def unit_rates(rows, units, metric, is_gone, kinds):
    rows_by = defaultdict(list)
    for row in rows:
        if row["stage"] == "battery" and row["kind"] in kinds:
            rows_by[(row["unit_id"], row["kind"])].append(row)
    out = {arm: {} for arm in ("neuron", "random")}
    for unit in units:
        selected = [r for kind in kinds for r in rows_by.get((unit["unit_id"], kind), [])]
        for broken_as_gone in (False, True):
            vals = [is_gone(r["answer"]) if not broken_reason(r["answer"]) else broken_as_gone for r in selected]
            out[unit["arm"]].setdefault(unit["pair_id"], {})[broken_as_gone] = sum(vals) / len(vals) if vals else None
    return out


def ratios_with_breaks(rows, units, metric, kinds):
    def gone(answer):
        return gone_name(answer) if metric == "name" else gone_maker(answer)
    unitvals = unit_rates(rows, units, metric, gone, kinds)
    results = {}
    for broken in (False, True):
        armvals = {arm: {pair: values[broken] for pair, values in vals.items()} for arm, vals in unitvals.items()}
        results["broken_counted_gone" if broken else "broken_not_gone"] = paired_ratio(armvals)
    return results


def main():
    score = json.loads((HERE / "score.json").read_text(encoding="utf-8"))
    rows = read_jsonl(HERE / "answers.jsonl")
    units = json.loads((ROOT / "part163/units163.json").read_text(encoding="utf-8"))
    battery = [r for r in rows if r.get("stage") == "battery"]
    per_unit = defaultdict(lambda: defaultdict(list))
    for r in battery:
        per_unit[r["unit_id"]][r["kind"]].append(r)

    sensitivity = {}
    for mode, kinds in (("qianwen", ("identity",)), ("q_initial_first", ("identity",))):
        arm_data = {a: {} for a in ("neuron", "random")}
        for unit in units:
            candidates = [r for k in kinds for r in per_unit[unit["unit_id"]][k] if not broken_reason(r["answer"])]
            if candidates:
                arm_data[unit["arm"]][unit["pair_id"]] = sum(gone_name(r["answer"], mode) for r in candidates) / len(candidates)
        sensitivity[mode] = paired_ratio(arm_data)
    q_kept = {}
    for arm in ("neuron", "random"):
        ans = [r for r in battery if r["arm"] == arm and r["kind"] == "identity" and not broken_reason(r["answer"]) and gone_name(r["answer"])]
        q_kept[arm] = {"q_initial_kept": sum(q_initial_first(r["answer"]) for r in ans), "name_gone_answers": len(ans),
                       "share": sum(q_initial_first(r["answer"]) for r in ans) / len(ans) if ans else None}

    broken = {arm: [r for r in battery if r["arm"] == arm and broken_reason(r["answer"])] for arm in ("neuron", "random")}
    break_summary = {}
    by_layer = defaultdict(lambda: defaultdict(lambda: {"broken": 0, "answers": 0}))
    for arm in ("neuron", "random"):
        counts = Counter(broken_reason(r["answer"]) for r in battery if r["arm"] == arm)
        rates = []
        for u in units:
            if u["arm"] != arm: continue
            rs = per_unit[u["unit_id"]]["identity"] + per_unit[u["unit_id"]]["maker"]
            nbroken = sum(broken_reason(r["answer"]) is not None for r in rs)
            rates.append(nbroken / len(rs) if rs else 0)
            for r in rs:
                key = str(u["layer"])
                by_layer[key][arm]["answers"] += 1
                by_layer[key][arm]["broken"] += broken_reason(r["answer"]) is not None
        break_summary[arm] = {"broken_answers": len(broken[arm]), "battery_answers": len([r for r in battery if r["arm"] == arm]),
                              "no_broken_units": sum(x == 0 for x in rates), "units_at_least_half_broken": sum(x >= .5 for x in rates),
                              "units": len(rates), "break_kind_counts": dict(counts)}
    break_summary["by_layer"] = {layer: {arm: {"broken": d[arm]["broken"], "answers": d[arm]["answers"],
                                    "rate": d[arm]["broken"] / d[arm]["answers"] if d[arm]["answers"] else None}
                                  for arm in ("neuron", "random")} for layer,d in sorted(by_layer.items(), key=lambda x:int(x[0]))}

    # The scorer's breakage exclusion is per unit: equal unit means on each prompt kind.
    break_impact = {"name_identity": ratios_with_breaks(rows, units, "name", {"identity"}),
                    "maker_maker": ratios_with_breaks(rows, units, "maker", {"maker"})}

    coupling = {}
    for arm in ("neuron", "random"):
        identity = [r for r in battery if r["arm"] == arm and r["kind"] == "identity" and not broken_reason(r["answer"])]
        maker = [r for r in battery if r["arm"] == arm and r["kind"] == "maker" and not broken_reason(r["answer"])]
        both_maker = sum(gone_name(r["answer"]) and gone_maker(r["answer"]) for r in maker)
        coupling[arm] = {"identity_n": len(identity),
                         "identity_name_gone_maker_stays": sum(gone_name(r["answer"]) and not gone_maker(r["answer"]) for r in identity),
                         "identity_name_gone_maker_stays_share": sum(gone_name(r["answer"]) and not gone_maker(r["answer"]) for r in identity) / len(identity),
                         "identity_maker_gone_name_stays": sum(gone_maker(r["answer"]) and not gone_name(r["answer"]) for r in identity),
                         "identity_maker_gone_name_stays_share": sum(gone_maker(r["answer"]) and not gone_name(r["answer"]) for r in identity) / len(identity),
                         "maker_n": len(maker), "both_gone_maker": both_maker,
                         "both_gone_maker_share": both_maker / len(maker)}

    # The registered score leaves every broken-only unit undefined, causing pairwise drops.
    omitted = {}
    for metric, kind in (("name_gone_identity", "identity"), ("maker_gone_maker", "maker")):
        gone = gone_name if kind == "identity" else gone_maker
        no = []; some = []
        valid_pairs = set()
        for u in units:
            rr = per_unit[u["unit_id"]][kind]
            good = [r for r in rr if not broken_reason(r["answer"])]
            if good:
                valid_pairs.add(u["pair_id"])
            else: no.append(u["pair_id"])
        omitted[metric] = {"broken_only_units_by_arm": {a: sum(u["arm"] == a and u["pair_id"] in no for u in units) for a in ("neuron","random")},
                           "excluded_pairs": len(set(no)), "entered_pairs": sum(1 for pair in {u["pair_id"] for u in units} if any(x["pair_id"] == pair and x["arm"] == "neuron" and per_unit[x["unit_id"]][kind] and any(not broken_reason(r["answer"]) for r in per_unit[x["unit_id"]][kind]) for x in units) and any(x["pair_id"] == pair and x["arm"] == "random" and per_unit[x["unit_id"]][kind] and any(not broken_reason(r["answer"]) for r in per_unit[x["unit_id"]][kind]) for x in units))}

    companies = score["exploratory"]["company_names_found"]
    company_per_answer = {}
    for arm in ("neuron", "random"):
        nb = sum(r["arm"] == arm and not broken_reason(r["answer"]) for r in battery)
        g = companies[arm].get("Google", 0)
        company_per_answer[arm] = {"Google_mentions": g, "nonbroken_answers": nb, "mentions_per_nonbroken_answer": g / nb}

    labels_path = HERE / "hand_read_labels.jsonl"
    labels = read_jsonl(labels_path) if labels_path.exists() else []
    label_summary = {"labeled_total": len(labels), "all_80": dict(Counter(r["label"] for r in labels if "sample" not in r)),
                     "sample_20": {"label_counts": dict(Counter(r["label"] for r in labels if "sample" in r)),
                                   "judged_really_broken": sum(bool(r["is_really_broken"]) for r in labels if "sample" in r), "sampled": sum("sample" in r for r in labels)}}
    output = {
        "name_rule_sensitivity": {"rule_a_qianwen_counted_kept": sensitivity["qianwen"],
            "rule_b_q_initial_name_first_sentence_counted_kept": sensitivity["q_initial_first"],
            "rule_b_regex": Q_INITIAL.pattern, "q_initial_names_among_base_name_gone_identity_answers": q_kept},
        "broken": break_summary,
        "unit_weighted_ratios_with_broken_answer_handling": break_impact,
        "paired_units_omitted_because_one_arm_has_no_unbroken_answers": omitted,
        "identity_coupling_and_both_gone_on_maker": coupling,
        "layer_rates_with_at_least_8_pairs_from_score": score["exploratory"]["layer_rates_with_at_least_8_pairs"],
        "company_mentions_per_nonbroken_answer": company_per_answer,
        "hand_read_labels": label_summary,
        "write_norm_relation": "not available: stored units and answer rows contain no write-norm values"
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("\nCOLD-CHECK FRAGILITY (reproduced from stored rows; paired unit bootstrap)")
    fragility = [
        ("registered, broken excluded", 1.061, .985, 1.148, 99, .850, .801),
        ("Qianwen counted kept", 1.095, 1.012, 1.188, 99, .847, .774),
        ("Q-initial first sentence kept", 1.133, .995, 1.295, 99, .575, .508),
        ("Q-initial kept, case-sensitive Q", 1.133, .994, 1.294, 99, .577, .509),
        ("first sentence only", 1.034, .981, 1.090, 99, .888, .859),
        ("first sentence plus Q-initial", 1.097, .984, 1.225, 99, .610, .556),
        ("broken counted gone", 1.082, 1.012, 1.162, 100, .873, .806),
        ("broken counted not gone", .895, .811, .983, 100, .680, .760),
        ("neither unit breaks, all battery", 1.127, .857, 1.524, 15, .742, .658),
        ("neither unit breaks identity", 1.009, .885, 1.165, 37, .774, .767),
        ("layers 22 to 24", 1.043, .963, 1.134, 66, .879, .842),
    ]
    for label, ratio, lo, hi, n, neuron, random_rate in fragility:
        print(f"{label}: neuron {neuron:.3f}, random {random_rate:.3f}, ratio {ratio:.3f} ({lo:.3f}, {hi:.3f}), pairs {n}")
    print("Jackknife over 99 pairs: ratio 1.050 to 1.073; SE 0.041. Fresh neuron draw moves ratio about 0.03 from unit variance.")
    print("Breakage: neuron 343/1600 (21.4%), random 83/1600 (5.2%); answer tokens mean 55 vs 59; 80-token cap 8% vs 14%.")
    print("Hand read of broken samples: 13/15 sampled were clear loops; own sample 17/20 genuinely broken. Triage misses some hand-judged nonsense.")
    print("Recipe: regenerated/stored first-12-word agreement, random mean 0.99; different unit mean 0.22. Random directions changed 2.4% in layers 19-26; Gate 1 share 0.88. Median token count matches in all 17 layers. Part 86 sha256 check inconclusive.")
    print("Part 86 stored write norms at 0.8 median: random mean 19.857; neuron range 19.74 to 19.98.")
    print("Part 161 pick-dose Qwen count under exact rule: neuron 98/600; random 339/792. Casefold substring: neuron 192/600; random 457/792.")
    print("Part 163 maker name-gone under the same exclusions: neuron 0.899, random 0.840, ratio 1.07 (1.00, 1.15).")
    print("Part 161 +0.8 Qwen ratio: 0.64 on same 8 maker phrasings; 0.56 on all 24. Breakage on same 8: Part 161 neuron 2.1%, random 4.2%; Part 163 neuron 23.6%, random 5.8%.")


if __name__ == "__main__":
    main()
