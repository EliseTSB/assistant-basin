"""Print Part 199's tables from the records, no model.

Reads results/part170 (published) and results/part199 (the eager rerun): score.json, compare.json, stage0.json,
lists.jsonl and answers.jsonl. Writes results/part199/report199.json and results/part199/report199.md.
Every number in results/part199/README.md comes from these two files or from compare.json and score.json.
Usage: python -I -B part199/report199.py  (from the repository root)
Written 2026-11-09 by Journal Claude (Opus 5.5).
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "results" / "part170"
NEW = ROOT / "results" / "part199"
LIST_FIELDS = ("b5", "b50", "b200")
PATH_TOKENS = 40  # run170.LIST_PATH_TOKENS: the plain path Stage 1 lists are read from


def jsonl(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def stage0_rows(folder):
    return {r["prompt_id"]: r for r in jsonl(folder / "answers.jsonl") if r.get("stage") == "stage0"}


def lists(folder):
    return {str(r["prompt_id"]): r for r in jsonl(folder / "lists.jsonl")}


def plain_paths(pub, new):
    """Stage 0 plain answers: where the two runs' greedy tokens first differ, in keys (prompt plus answer index)."""
    same40, same_all, first_key = [], [], {}
    for pid, a in pub.items():
        b = new[pid]
        x, y = list(map(int, a["token_ids"])), list(map(int, b["token_ids"]))
        if x[:PATH_TOKENS] == y[:PATH_TOKENS]:
            same40.append(pid)
        if x == y:
            same_all.append(pid)
        k = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
        if k is None and len(x) != len(y):
            k = min(len(x), len(y))
        if k is not None:
            first_key[pid] = len(a["prompt_ids"]) + k
    keys = sorted(first_key.values())
    by_group = {}
    for g in ("p8", "m96", "q96"):
        ids = [p for p in pub if pub[p]["group"] == g]
        by_group[g] = {"prompts": len(ids), "same_first_40": sum(p in same40 for p in ids),
                       "same_whole_answer": sum(p in same_all for p in ids)}
    return same40, {
        "prompts": len(pub), "same_first_40_tokens": len(same40), "same_whole_answer": len(same_all),
        "differing_answers": len(keys),
        "first_difference_key_median": statistics.median(keys) if keys else None,
        "first_difference_key_min": keys[0] if keys else None,
        "first_difference_key_max": keys[-1] if keys else None,
        "first_difference_at_or_below_64_keys": sum(k <= 64 for k in keys),
        "by_group": by_group,
    }


def list_agreement(la, lb, same40):
    out = {}
    for f in LIST_FIELDS:
        rows = {}
        for label, ids in (("same_path", same40), ("changed_path", [p for p in la if p not in same40])):
            ordered = sum(list(map(int, la[p][f])) == list(map(int, lb[p][f])) for p in ids)
            as_set = sum(set(map(int, la[p][f])) == set(map(int, lb[p][f])) for p in ids)
            overlap = [len(set(map(int, la[p][f])) & set(map(int, lb[p][f]))) for p in ids]
            rows[label] = {"prompts": len(ids), "ordered_equal": ordered, "set_equal": as_set,
                           "overlap_median": statistics.median(overlap) if overlap else None,
                           "overlap_min": min(overlap) if overlap else None}
        rows["all"] = {"prompts": len(la),
                       "ordered_equal": rows["same_path"]["ordered_equal"] + rows["changed_path"]["ordered_equal"],
                       "set_equal": rows["same_path"]["set_equal"] + rows["changed_path"]["set_equal"]}
        out[f] = rows
    return out


def exploratory(pub0_frozen):
    """Not pre-registered. (1) The no-push b0 and b50 loop counts on the identity prompts (P8 and M96) and on Q96,
    restricted to the prompts Part 170 left eligible, so both runs are read on the same population. (2) Mean answer
    length and the share stopping before the 80-token cap in the dose-1 push cells."""
    import sys
    sys.path.insert(0, str(ROOT))
    from part166 import score166

    def rows(folder):
        return jsonl(folder / "answers.jsonl")

    out = {"same_population": {}, "dose1_lengths": {}}
    for label, folder in (("published", PUB), ("eager", NEW)):
        rs = rows(folder)
        for setname, groups in (("identity", ("p8", "m96")), ("q96", ("q96",))):
            for ban in ("b0", "b50"):
                sel = [r for r in rs if r["stage"] == "no_push" and r["ban"] == ban and r["group"] in groups
                       and r["prompt_id"] not in pub0_frozen]
                out["same_population"].setdefault(f"{setname}_{ban}", {})[label] = {
                    "rows": len(sel), "loops": sum(bool(score166.loop_phrase(r["answer"])) for r in sel)}
        for ban in ("b0", "b50", "b200"):
            sel = [r for r in rs if r["stage"] == "push" and float(r["dose"]) == 1.0 and r["ban"] == ban]
            out["dose1_lengths"].setdefault(ban, {})[label] = {
                "rows": len(sel), "mean_tokens": statistics.mean(int(r["token_count"]) for r in sel),
                "stopped_before_cap": sum(bool(r["ended_eos_before_80"]) for r in sel)}
    return out


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def r3(v):
    return "n/a" if v is None else f"{v:.3f}"


def main():
    compare = json.loads((NEW / "compare.json").read_text(encoding="utf-8"))
    pub0, new0 = stage0_rows(PUB), stage0_rows(NEW)
    same40, paths = plain_paths(pub0, new0)
    agree = list_agreement(lists(PUB), lists(NEW), same40)
    cells = compare["cells_and_predictions"]["cells"]
    verdicts = [p for p in compare["cells_and_predictions"]["predictions"] if p["prediction"] != "exploratory"]
    frozen_ids = {f"{g}:{i}" for g in ("m96", "q96") for i in compare["frozen"][g]["published"]}
    explo = exploratory(frozen_ids)
    report = {"plain_paths": paths, "list_agreement": agree, "frozen": compare["frozen"], "exploratory": explo,
              "lists_registered_count": {k: v for k, v in compare["lists"].items() if k != "differing_prompt_ids"},
              "gates": compare["gates"], "gate_r2": compare["gate_r2"], "noise_limited": compare["noise_limited"],
              "loop_onset": compare["loop_onset"]}
    (NEW / "report199.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md = ["# Part 199 report (printed by part199/report199.py)", ""]
    md += ["## Verdicts: published (SDPA) beside the eager rerun", "",
           "| Prediction | Published | Published 95% interval | Eager | Eager 95% interval | Changed |",
           "| --- | --- | --- | --- | --- | --- |"]
    for p in verdicts:
        pi = p.get("published_interval_95") or [None, None]
        ri = p.get("rerun_interval_95") or [None, None]
        md.append(f"| {p['prediction']} | {'holds' if p['published_pass'] else 'fails'} ({r3(p['published_estimate'])}) "
                  f"| {r3(pi[0])} to {r3(pi[1])} | {'holds' if p['rerun_pass'] else 'fails'} ({r3(p['rerun_estimate'])}) "
                  f"| {r3(ri[0])} to {r3(ri[1])} | {'yes' if p.get('changed') else 'no'} |")
    md += ["", "## Cells: loop rate (broken rate), published beside eager", "",
           "| Stage | Group | Dose | Ban | Published rows | Published loop (broken) | Eager rows | Eager loop (broken) |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in cells:
        k = c["cell"]
        md.append(f"| {k['stage']} | {k['group']} | {k['dose']} | {k['ban']} | {c['published_rows']} "
                  f"| {pct(c['published_loop_rate'])} ({pct(c['published_broken_rate'])}) | {c['rerun_rows']} "
                  f"| {pct(c['rerun_loop_rate'])} ({pct(c['rerun_broken_rate'])}) |")
    fr = compare["frozen"]
    md += ["", "## Stage 0 frozen prompts", "",
           "| Set | Published frozen | Eager frozen | Overlap |", "| --- | --- | --- | --- |"]
    for g in ("p8", "m96", "q96"):
        md.append(f"| {g} | {fr[g]['published_count']} | {fr[g]['rerun_count']} | {fr[g]['overlap']} |")
    md += ["", "## Stage 0 plain answers: published (SDPA, one row at a time) against eager", "",
           f"- Prompts: {paths['prompts']}; identical through the first {PATH_TOKENS} answer tokens (the Stage 1 list "
           f"path): {paths['same_first_40_tokens']}; identical whole answer: {paths['same_whole_answer']}.",
           f"- Answers that differ: {paths['differing_answers']}; the first differing token is generated over "
           f"{paths['first_difference_key_min']} to {paths['first_difference_key_max']} keys "
           f"(median {paths['first_difference_key_median']}); {paths['first_difference_at_or_below_64_keys']} at 64 "
           f"keys or fewer."]
    for g, v in paths["by_group"].items():
        md.append(f"- {g}: {v['same_first_40']} of {v['prompts']} identical through {PATH_TOKENS} tokens, "
                  f"{v['same_whole_answer']} identical whole.")
    md += ["", "## Stage 1 ban lists: published against eager", "",
           f"Registered count (`--compare`, ordered equality on every list): "
           f"{compare['lists']['identical_on_every_list']} of {compare['lists']['shared']}.", "",
           "| List | Path | Prompts | Same order | Same tokens (as a set) | Shared tokens, median | Shared tokens, min |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for f in LIST_FIELDS:
        for label in ("same_path", "changed_path"):
            v = agree[f][label]
            md.append(f"| {f} | {label.replace('_', ' ')} | {v['prompts']} | {v['ordered_equal']} | {v['set_equal']} "
                      f"| {v['overlap_median']} | {v['overlap_min']} |")
    lo = compare["loop_onset"]
    md += ["", "## Loop onset on the eager answers (descriptive)", "",
           f"- All loops: {lo['total']['loops']}; established before t0: {lo['total']['before']}; after: "
           f"{lo['total']['after']}.",
           f"- Dose-1 b50: {lo['dose1_b50']['loops']}; before t0: {lo['dose1_b50']['before']}; after: "
           f"{lo['dose1_b50']['after']}."]
    md += ["", "## Exploratory (not pre-registered)", "",
           "No-push loops on the prompts Part 170 left eligible (its 19 frozen prompts removed from both runs):", "",
           "| Set and ban | Published loops / rows | Eager loops / rows |", "| --- | --- | --- |"]
    for key, v in explo["same_population"].items():
        md.append(f"| {key} | {v['published']['loops']} / {v['published']['rows']} "
                  f"| {v['eager']['loops']} / {v['eager']['rows']} |")
    md += ["", "Dose-1 push cells, answer length:", "",
           "| Ban | Published mean tokens | Published stopped before 80 | Eager mean tokens | Eager stopped before 80 |",
           "| --- | --- | --- | --- | --- |"]
    for ban, v in explo["dose1_lengths"].items():
        md.append(f"| {ban} | {v['published']['mean_tokens']:.2f} | {v['published']['stopped_before_cap']} of "
                  f"{v['published']['rows']} | {v['eager']['mean_tokens']:.2f} | {v['eager']['stopped_before_cap']} of "
                  f"{v['eager']['rows']} |")
    g = compare["gates"]
    md += ["", "## Gates", "",
           f"- Part 170 gates 1 to 8, published: {g['published']['passed']}; eager: {g['rerun']['passed']}.",
           f"- Gate 6 dose-1 b0 loop rate: published {r3(g['published']['gate6_dose1_b0_loop_rate'])}, eager "
           f"{r3(g['rerun']['gate6_dose1_b0_loop_rate'])}; b50 mean blocked steps: published "
           f"{g['published']['gate6_dose1_b50_mean_blocked_steps']:.2f}, eager "
           f"{g['rerun']['gate6_dose1_b50_mean_blocked_steps']:.2f}.",
           f"- Gate 8 repeat: published {g['published']['gate8_differing_answers']} of {g['published']['gate8_rows']} "
           f"changed; eager {g['rerun']['gate8_differing_answers']} of {g['rerun']['gate8_rows']}. Noise-limited: "
           f"{compare['noise_limited']}.",
           f"- Gate R2: {compare['gate_r2']['passed']} ({compare['gate_r2']['rerun_directions']} directions, worst "
           f"relative write error {compare['gate_r2']['rerun_worst_relative_write_error']:.4f})."]
    (NEW / "report199.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
