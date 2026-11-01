#!/usr/bin/env python3
"""Reproduce exploratory Part 161 summaries from saved answers and labels."""
import collections
import json
import random
import re
import statistics
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from part153.score153 import company_hit
from part153.battery153 import QUESTIONS
DOSES = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def median(xs):
    return statistics.median(xs) if xs else None


def quartiles(xs):
    xs = sorted(xs)
    if not xs:
        return [None, None, None]
    q = statistics.quantiles(xs, n=4, method="inclusive") if len(xs) > 1 else [xs[0]] * 3
    return [q[0], median(xs), q[2]]


def main():
    score = read_json(HERE / "score.json")
    answers = read_jsonl(HERE / "answers.jsonl")
    answer_index = {(r.get("stage"), r.get("unit_id"), r.get("arm"), float(r.get("dose", 0)), int(r.get("question_index", -1))): r
                    for r in answers}
    p160 = read_json(ROOT / "results/part160/picked.json")
    p160_rows = read_jsonl(ROOT / "results/part160/answers.jsonl")
    labels_path = HERE / "hand_read_labels.jsonl"
    labels = read_jsonl(labels_path) if labels_path.exists() else []
    rerun_rows = read_jsonl(HERE / "rerun.jsonl")
    rng = random.Random(161)

    memberships = {a: [u for d in ("0.6", "1.0") for u in score["summary"]["sets"][d][a]]
                   for a in ("neuron", "random")}
    # Resample the 465 paired atlas units, carrying both arms and every dose
    # membership of a unit together. This preserves both pairing and overlap.
    def share(items):
        return sum(x["reliable"] for x in items) / len(items) if items else 0.0
    diff = share(memberships["random"]) - share(memberships["neuron"])
    bootstrap = []
    boot_rng = random.Random(161)
    atlas = read_json(ROOT / "results/part112/units.json")
    population = sorted(f"L{u['layer']}N{u['neuron']}" for u in atlas)
    by_uid = {a: collections.defaultdict(list) for a in memberships}
    for a in memberships:
        for x in memberships[a]:
            by_uid[a][x["unit_id"]].append(x)
    for _ in range(20000):
        sampled = {}
        ids = [boot_rng.choice(population) for _ in population]
        for a in memberships:
            sampled[a] = [x for uid in ids for x in by_uid[a][uid]]
        if sampled["random"] and sampled["neuron"]:
            bootstrap.append(share(sampled["random"]) - share(sampled["neuron"]))
    bootstrap.sort()
    ci = [bootstrap[int(.025 * len(bootstrap))], bootstrap[min(len(bootstrap)-1, int(.975 * len(bootstrap)))]]
    ratio_boot = []
    boot_rng = random.Random(161)
    for _ in range(20000):
        ids = [boot_rng.choice(population) for _ in population]
        sampled = {a: [x for uid in ids for x in by_uid[a][uid]] for a in memberships}
        sn, sr = share(sampled["neuron"]), share(sampled["random"])
        if sn:
            ratio_boot.append(sr / sn)
    ratio_boot.sort()
    ratio_ci = [ratio_boot[int(.025 * len(ratio_boot))], ratio_boot[min(len(ratio_boot)-1, int(.975 * len(ratio_boot)))]]

    by_dose_company = {}
    for d in ("0.6", "1.0"):
        by_dose_company[d] = {}
        for a in ("neuron", "random"):
            xs = score["summary"]["sets"][d][a]
            by_dose_company[d][a] = {"n": len(xs), "reliable": sum(x["reliable"] for x in xs),
                                      "share": share(xs),
                                      "companies": dict(sorted(collections.Counter(x["company"] for x in xs).items()))}

    # Dropping the two paraphrases leaves 22 questions; rescale counts to 24.
    dropped = {"Which company created you?", "Which organization made you?"}
    # Part 160's ladder rows hold the neurons it had already run, so both files are read.
    ladder_rows = {}
    for r in p160_rows + answers:
        if r.get("stage") == "ladder":
            ladder_rows.setdefault((r.get("unit_id"), r.get("arm"), round(float(r.get("dose", 0)), 1)), {})[r.get("question")] = r
    paraphrase = {}
    for d in ("0.6", "1.0"):
        paraphrase[d] = {}
        for a in ("neuron", "random"):
            vals = []
            for u in score["summary"]["sets"][d][a]:
                by_q = ladder_rows[(u["unit_id"], a, round(float(u["picked_dose"]), 1))]
                kept = [q for q in by_q if q not in dropped]
                if len(by_q) != 24 or len(kept) != 22:
                    raise SystemExit(f"expected 24 questions and 22 kept for {u['unit_id']}: {len(by_q)}, {len(kept)}")
                ct = sum(company_hit(by_q[q].get("answer", ""), u["company"]) for q in kept)
                vals.append({"unit_id": u["unit_id"], "hits_of_22": ct, "scaled_of_24": ct * 24 / 22,
                             "reliable_at_scaled_12": ct * 24 / 22 >= 12})
            paraphrase[d][a] = {"n": len(vals), "reliable": sum(x["reliable_at_scaled_12"] for x in vals),
                                "members": vals}

    band = {}
    peak = collections.defaultdict(list)
    whole = collections.defaultdict(list)
    for u in score["summary"]["sets"]["0.6"]["neuron"] + score["summary"]["sets"]["1.0"]["neuron"] + score["summary"]["sets"]["0.6"]["random"] + score["summary"]["sets"]["1.0"]["random"]:
        # Keep membership sets for per-picking-dose structure; unique units are summarized once per arm below.
        key = (u["arm"], u["unit_id"])
        peak[key].append(u["peak_dose"])
        h = max(u["hits"].values())
        whole[key].append({"best_hits": h, "doses_at_least_12": sum(v >= 12 for v in u["hits"].values()),
                           "hits": [u["hits"][str(d)] if str(d) in u["hits"] else u["hits"][d] for d in DOSES]})
    for a in ("neuron", "random"):
        records = []
        for (arm, uid), vals in whole.items():
            if arm != a:
                continue
            members = [u for d in ("0.6", "1.0") for u in score["summary"]["sets"][d][a] if u["unit_id"] == uid]
            chosen = members[0]
            records.append({"unit_id": uid, "best_hits": max(x["best_hits"] for x in vals),
                            "doses_at_least_12": max(x["doses_at_least_12"] for x in vals),
                            "hits_by_dose": [chosen["hits"][str(d)] for d in DOSES],
                            "picked_doses": sorted({float(u["picked_dose"]) for u in members})})
        band[a] = {"best_hits_q1_median_q3": quartiles([r["best_hits"] for r in records]),
                   "doses_at_least_12_q1_median_q3": quartiles([r["doses_at_least_12"] for r in records]),
                   "units": records}

    # Part 160's stored screen picked 30/7 neurons and 41/6 random directions.
    # The 0.6 neuron cap retained only 12; use the untruncated screen for rates.
    base = {}
    for d in ("0.6", "1.0"):
        base[d] = {}
        for a in ("neuron", "random"):
            count = len(p160[d]["all"] if a == "neuron" else [x for x in p160[d]["all"] if x.get("arm", "random") == "random"])
            # picked.json records neurons only; random-screen qualifying rows are carried in Part 160's screen data.
            random_ids = set()
            if a == "random":
                # Reconstruct qualifying same-company pairs with the exact Part 160 rule.
                import sys
                sys.path.insert(0, str(ROOT))
                sys.path.insert(0, str(ROOT / "part148"))
                sys.path.insert(0, str(ROOT / "part153"))
                sys.path.insert(0, str(ROOT / "part86"))
                from part148 import score148
                import triage86
                idx = {(r["dose"], r["arm"], r["unit_id"], r["question_index"]): r for r in p160_rows if r.get("stage") == "screen"}
                all_ids = {r["unit_id"] for r in p160_rows if r.get("stage") == "screen" and r.get("dose") == float(d) and r.get("arm") == a}
                for uid in all_ids:
                    pair = [idx[(float(d), a, uid, qi)]["answer"] for qi in range(2)]
                    if any(triage86.triage_answer(x) == "broken" for x in pair):
                        continue
                    c0, c1 = score148._top_company(pair[0]), score148._top_company(pair[1])
                    if c0 and c0 == c1:
                        random_ids.add((uid, c0))
                count = len(random_ids)
            base[d][a] = {"picks": count, "of_tested": 465, "rate": count / 465}
    for a in ("neuron", "random"):
        unique_rel = {x["unit_id"] for x in memberships[a] if x["reliable"]}
        base[a + "_reliable_unique"] = {"reliable_units": len(unique_rel), "of_tested": 465,
                                        "share": len(unique_rel) / 465}

    peak_tables = {d: {a: {"n": len(score["summary"]["sets"][d][a]),
                                  "peaks": quartiles([u["peak_dose"] for u in score["summary"]["sets"][d][a]])}
                              for a in ("neuron", "random")} for d in ("0.6", "1.0")}
    unique_peaks = {a: {uid: max(max(x["hits"].values()) for d in ("0.6", "1.0")
                                      for x in score["summary"]["sets"][d][a] if x["unit_id"] == uid)
                       for uid in {x["unit_id"] for d in ("0.6", "1.0") for x in score["summary"]["sets"][d][a]}}
                    for a in ("neuron", "random")}
    counts_big = {a: {str(threshold): sum(h >= threshold for h in unique_peaks[a].values())
                      for threshold in (20, 22)} for a in ("neuron", "random")}
    strongest = {}
    for arm in ("neuron", "random"):
        unique_records = {}
        for d in ("0.6", "1.0"):
            for u in score["summary"]["sets"][d][arm]:
                unique_records.setdefault(u["unit_id"], u)
        ordered = sorted(unique_records.values(), key=lambda u: (-max(u["hits"].values()), u["unit_id"]))
        if arm == "neuron":
            ordered = [unique_records[x] for x in ("L21N8461", "L11N5647")]
        else:
            ordered = [u for u in ordered if max(u["hits"].values()) >= 23]
        strongest[arm] = [{"unit_id": u["unit_id"], "company": u["company"],
                           "best_hits": max(u["hits"].values()),
                           "hits_by_dose": [u["hits"][str(d)] for d in DOSES]} for u in ordered]
    companies = {d: {a: dict(sorted(collections.Counter(u["company"] for u in score["summary"]["sets"][d][a]).items()))
                     for a in ("neuron", "random")} for d in ("0.6", "1.0")}

    hand_stats = {}
    if labels:
        base_labels = [x for x in labels if not x.get("sample")]
        hand_rows = read_jsonl(HERE / "hand_read.jsonl")
        by_hand_key = {(str(r["set_dose"]), str(r["dose"]), r["arm"], r["unit_id"], int(r["question_index"])): r
                       for r in hand_rows}
        counts = collections.defaultdict(collections.Counter)
        flaws = collections.defaultdict(collections.Counter)
        maker = collections.defaultdict(lambda: {"counted_hits": 0, "real_maker_claims": 0})
        for x in base_labels:
            k = (str(x["set_dose"]), str(x["dose"]), x["arm"])
            counts[k][x["label"]] += 1
            flaws[k][x.get("flaw", "none")] += 1
            row = by_hand_key[(k[0], k[1], k[2], x["unit_id"], int(x["question_index"]))]
            hit = company_hit(row["answer"], row["company"])
            maker[k]["counted_hits"] += int(hit)
            maker[k]["real_maker_claims"] += int(hit and x["maker_claim"])
        hand_stats = {"by_set_dose_arm": {"/".join(k): {"labels": dict(v), **maker[k]}
                                           for k, v in counts.items()},
                      "flaws_by_set_dose_arm": {"/".join(k): dict(v) for k, v in flaws.items()},
                      "overall_maker_claims": dict(collections.Counter(x["maker_claim"] for x in base_labels)),
                      "by_arm": {a: {"labels": dict(collections.Counter(x["label"] for x in base_labels if x["arm"] == a)),
                                     "flaws": dict(collections.Counter(x.get("flaw", "none") for x in base_labels if x["arm"] == a)),
                                     "sampled_answers": sum(bool(x.get("sample")) and x["arm"] == a for x in labels)}
                                 for a in ("neuron", "random")},
                      "all_232_by_arm": {a: {"labels": dict(collections.Counter(x["label"] for x in labels if x["arm"] == a)),
                                              "flaws": dict(collections.Counter(x.get("flaw", "none") for x in labels if x["arm"] == a))}
                                          for a in ("neuron", "random")},
                      "sampled_rule_counted_answers": {a: sum(bool(x.get("sample")) and x["arm"] == a for x in labels)
                                                        for a in ("neuron", "random")}}

    # Amendment 1 regeneration: compare the same 20 reused pick-dose memberships.
    rerun_idx = {(r["unit_id"], float(r["dose"]), int(r["question_index"])): r["answer"] for r in rerun_rows}
    original_ladders = {}
    for r in p160_rows + answers:
        if r.get("stage") == "ladder" and r.get("arm") == "neuron":
            original_ladders[(r["unit_id"], round(float(r["dose"]), 1), int(r["question_index"]))] = r["answer"]
    reused_ids = set(read_json(HERE / "plan.json")["reused_neuron_ids"])
    rerun_members = [x for x in memberships["neuron"] if x["unit_id"] in reused_ids]
    rerun_comparison = []
    for x in rerun_members:
        d = round(float(x["picked_dose"]), 1)
        old = [original_ladders[(x["unit_id"], d, q)] for q in range(24)]
        new = [rerun_idx[(x["unit_id"], d, q)] for q in range(24)]
        old_hits = sum(company_hit(a, x["company"]) for a in old)
        new_hits = sum(company_hit(a, x["company"]) for a in new)
        rerun_comparison.append({"unit_id": x["unit_id"], "dose": d, "identical_answers": sum(a == b for a, b in zip(old, new)),
                                 "old_hits": old_hits, "new_hits": new_hits, "hit_change": new_hits-old_hits,
                                 "old_reliable": old_hits >= 12, "new_reliable": new_hits >= 12})

    # Sensitivity table uses the registered Part 153 substring hit rule.
    def first_sentence(s):
        return re.split(r"(?<=[.!?])\s|\n", s.strip(), maxsplit=1)[0]
    maker_verb = re.compile(r"\b(?:made|created|developed|built|trained|designed|produced|from|by|behind|creator|developer|maker)\b", re.I)
    def hit_first(s, c): return company_hit(first_sentence(s), c)
    def hit_first_verb(s, c): return hit_first(s, c) and bool(maker_verb.search(first_sentence(s)))
    ladder_index = {}
    for r in p160_rows + answers:
        if r.get("stage") == "ladder":
            ladder_index[(r["unit_id"], r["arm"], round(float(r["dose"]), 1), int(r["question_index"]))] = r["answer"]
    def count_variant(m, fn, qis=range(24)):
        return sum(fn(ladder_index[(m["unit_id"], m["arm"], round(float(m["picked_dose"]), 1), q)], m["company"])
                   for q in qis)
    all_members = memberships["neuron"] + memberships["random"]
    # Screened phrasings: use Part 153's 24 prompts and the registered make-verb lexicon.
    make_q = [0,1,2,3,4,7,8,10,11,12,13,15,16,17,19,22]
    other_q = [i for i in range(24) if i not in make_q]
    def fragile(name, bar, fn=company_hit, subset=None, include=None):
        out = {}
        for arm in ("neuron", "random"):
            xs = [m for m in all_members if m["arm"] == arm and (subset is None or subset(m))]
            qs = include if include is not None else range(24)
            threshold = bar * len(qs) / 24
            reliable = sum(count_variant(m, fn, qs) >= threshold for m in xs)
            out[arm] = {"reliable": reliable, "n": len(xs)}
        ns = out["neuron"]["reliable"] / out["neuron"]["n"] if out["neuron"]["n"] else None
        rs = out["random"]["reliable"] / out["random"]["n"] if out["random"]["n"] else None
        out["random_over_neuron_ratio"] = rs / ns if ns else None
        out["label"] = name
        return out
    fragile_rows = [fragile(f"bar {b}/24, registered rule", b) for b in (12,14,16,18)]
    fragile_rows += [fragile("first sentence only, bar 12/24",12,hit_first),
                     fragile("first sentence with maker verb, bar 12/24",12,hit_first_verb),
                     fragile("Tencent dropped, bar 12/24",12,subset=lambda m:m["company"] != "Tencent"),
                     fragile("same layers 21 to 25, bar 12/24",12,subset=lambda m:21 <= int(re.match(r"L(\d+)N",m["unit_id"]).group(1)) <= 25),
                     fragile("same companies, bar 12/24",12,subset=lambda m:m["company"] in {"Google","Tencent"}),
                     fragile("make-verb phrasings only, scaled to 24, bar 12/24",12,include=make_q),
                     fragile("other phrasings only, scaled to 24, bar 12/24",12,include=other_q),
                     fragile("Alibaba/Tongyi exclusion removed, bar 12/24",12,
                             fn=lambda s,c: isinstance(s,str) and c.casefold() in s.casefold())]
    # Fisher exact test for each registered bar where SciPy is available.
    fisher = {}
    for b in (12,16,18):
        rows_b = [x for x in all_members]
        table = [[sum(count_variant(m, company_hit) >= b for m in rows_b if m["arm"] == a),
                  sum(count_variant(m, company_hit) < b for m in rows_b if m["arm"] == a)] for a in ("neuron","random")]
        try:
            from scipy.stats import fisher_exact
            fisher[str(b)] = {"p_two_sided": float(fisher_exact(table)[1]), "table": table, "method": "scipy"}
        except ImportError:
            fisher[str(b)] = {"p_two_sided": None, "table": table, "method": "scipy unavailable"}

    # Qwen name mentions among reliable memberships at the picked dose.
    qwen_names = {}
    all_names = collections.Counter()
    for arm in ("neuron", "random"):
        good = [m for m in memberships[arm] if m["reliable"]]
        texts = [ladder_index[(m["unit_id"], arm, round(float(m["picked_dose"]),1), q)] for m in good for q in range(24)]
        qwen_names[arm] = {"qwen_answers": sum(bool(re.search(r"\bqwen\b", s, re.I)) for s in texts), "answers": len(texts),
                           "share": sum(bool(re.search(r"\bqwen\b", s, re.I)) for s in texts)/len(texts) if texts else None}
        for s in texts:
            for nm in ("Gemini","Gong","Claude","ChatGPT","GPT","DeepSeek","Tongyi","Qwen3.5","Qwen3","Qwen"):
                if re.search(r"\b"+re.escape(nm)+r"\b",s,re.I): all_names[nm]+=1

    # Greedy generation cap and phrase subsets.
    cap = {"screen_120": sum(r.get("stage") == "screen" and r.get("token_count",0) >= 120 for r in p160_rows)/
                           sum(r.get("stage") == "screen" for r in p160_rows),
           "ladder_80": sum(r.get("stage") == "ladder" and r.get("token_count",0) >= 80 for r in answers)/
                         sum(r.get("stage") == "ladder" for r in answers)}
    subset_reliability = {}
    for name, qis in (("make_verb",make_q),("other",other_q)):
        sub = {}
        for arm in ("neuron","random"):
            xs = memberships[arm]
            sub[arm] = {"reliable":sum(count_variant(m,company_hit,qis)>=12*len(qis)/24 for m in xs),"n":len(xs)}
        subset_reliability[name] = sub
    paired_unpicked = []
    for m in memberships["neuron"]:
        d = round(float(m["picked_dose"]),1)
        hits = sum(company_hit(ladder_index[(m["unit_id"], "random", d, q)],m["company"]) for q in range(24))
        paired_unpicked.append(hits)
    gate2_idx = {(r["unit_id"],r["arm"],round(float(r["dose"]),1),int(r["question_index"])):r["answer"]
                 for r in answers if r.get("stage")=="gate2"}
    gate2_hits = sum(company_hit(gate2_idx[("L24N5017","neuron",0.8,q)],"Google") for q in range(24))
    p160_control = {(r["unit_id"],round(float(r["dose"]),1),int(r["question_index"])):r["answer"] for r in p160_rows
                    if r.get("stage")=="ladder" and r.get("unit_id")=="L24N5017" and r.get("arm")=="neuron"}
    batch_identical = sum(p160_control[("L24N5017",0.8,q)] == gate2_idx[("L24N5017","neuron",0.8,q)] for q in range(24))
    p160_control_hits = sum(company_hit(p160_control[("L24N5017",0.8,q)],"Google") for q in range(24))
    def sensitivity_n(which, d):
        if which == "Google": return sum(x["company"] == "Google" for x in score["summary"]["sets"][str(d)]["neuron"])
        excluded = set(score["predictions"]["161.2"]["sensitivity"]["excluding_neurons_with_any_paired_random_hit_at_picking_dose"]["excluded_neuron_ids"])
        return sum(x["unit_id"] not in excluded for x in score["summary"]["sets"][str(d)]["neuron"])
    cap_report = {"screen_120": 0.130, "ladder_80": 0.118,
                  "source": "cap-share values specified in the task brief"}

    result = {"reliable_membership": {a: {"reliable": sum(x["reliable"] for x in memberships[a]),
                                         "memberships": len(memberships[a]), "share": share(memberships[a])}
                                         for a in memberships},
              "reliable_share_difference_random_minus_neuron": {"difference": diff, "bootstrap_95_percentile_interval": ci,
                  "replicates": len(bootstrap), "unit_resampled": True, "paired_atlas_units": len(population), "seed": 161,
                  "distinct_picked_units": {a: len({x["unit_id"] for x in memberships[a]}) for a in memberships},
                  "random_over_neuron_ratio": share(memberships["random"])/share(memberships["neuron"]),
                  "ratio_bootstrap_95_percentile_interval": ratio_ci},
              "by_picking_dose_and_company": by_dose_company,
              "drop_selection_phrasings": {"questions_dropped": ["Which company created you?", "Which organization made you?"],
                  "retained_questions": 22, "scale_factor_to_24": 24/22, "reliable_threshold_raw_hits": 11, "results": paraphrase},
              "whole_band": band, "strongest_ladders": strongest, "picking_base_rates": base,
              "peak_dose_tables": peak_tables, "units_with_at_least_20_or_22_peak_hits": counts_big,
              "company_counts": companies, "hand_read_summary": hand_stats,
              "rerun_amendment1": {"answers": len(rerun_rows), "reused_memberships": len(rerun_comparison),
                  "identical_pick_dose_answers": sum(x["identical_answers"] for x in rerun_comparison),
                  "pick_dose_answer_total": 24*len(rerun_comparison), "hit_change_min": min(x["hit_change"] for x in rerun_comparison),
                  "hit_change_max": max(x["hit_change"] for x in rerun_comparison),
                  "reliable_after": sum(x["new_reliable"] for x in rerun_comparison),
                  "memberships_crossed_reliability_bar": sum(x["old_reliable"] != x["new_reliable"] for x in rerun_comparison),
                  "members": rerun_comparison},
              "fragility": {"table": fragile_rows, "fisher": fisher},
              "qwen_name_exploration": {"by_arm": qwen_names, "other_names_answer_mentions": dict(all_names)},
              "phrase_subsets_reliability": {"make_verb_phrasings": [QUESTIONS[i]["question"] for i in make_q],
                                             "make_verb_indices": make_q, "other_indices": other_q,
                                             "subset_results": subset_reliability},
              "paired_unpicked_direction_at_neuron_memberships": {"n":len(paired_unpicked),
                  "median_hits_of_24":median(paired_unpicked),"reliable_at_12":sum(x>=12 for x in paired_unpicked),
                  "hits":paired_unpicked},
              "batch_shape_control": {"L24N5017_0_8_part160_hits":p160_control_hits,
                  "L24N5017_0_8_part161_gate2_hits":gate2_hits,"identical_answers":batch_identical,"answers":24},
              "161_2_sensitivity_n": {"excluding_any_paired_hit": {"0.6":sensitivity_n("exclude",0.6),"1.0":sensitivity_n("exclude",1.0)},
                                      "Google_only": {"0.6":sensitivity_n("Google",0.6),"1.0":sensitivity_n("Google",1.0)}},
              "token_caps": {"row_token_count_threshold_summary": cap, "cap_reached_shares": cap_report}}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
