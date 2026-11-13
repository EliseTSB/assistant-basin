#!/usr/bin/env python3
"""Part 210: Part 165's published margins, answer lengths and cap counts, from results/part165/.

Usage: margins210.py [--results results/part165] [--out results/part210]

Reads ONLY `results/part165/score.json` and `results/part165/answers.jsonl` (Part 165's own published files; nothing
under `results/part165/` is written). Prints, and writes to `margins.md` and `margins.json`:

  * every registered verdict (165.1 to 165.6) and every gate (Gates 1 to 4): the published estimate, its bar, and the
    distance to the bar in the units the bar is read in (rows, slots, rate points or Spearman units), plus whether the
    published verdict read a point estimate or an interval;
  * each arm's count of answers at the 80-token cap, mean `token_count`, primary-broken count and loop-only count.

The arithmetic is Part 165's own where it can be: the broken rule, the loop rule, the changed rule and the Spearman
come from `part165/score165.py`, imported rather than rewritten. The distances are computed from the published
numbers and say in `unit` what one unit of distance is; a distance is not a prediction of independent row flips.

Exit codes: 0 written; 2 bad input (a missing file, a row that is not JSON, a score file without a verdict).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part165 import score165  # noqa: E402  (Part 165's own rules, imported, not copied)

TOKEN_CAP = 80
ARMS = score165.ARMS
LADDER = score165.LADDER
# The ladder's x axis, as score165._bootstrap_spearman builds it: log k with full at the layer's width. The widths
# are the same in every layer (one model, one down_proj width), so the published `k` field gives it. `full` is read
# from the rows (`full_k` in build_report), so it is not listed here.
LADDER_K = {"n±": 1, "k4": 4, "k16": 16, "k64": 64, "k256": 256, "k1024": 1024}
# The bars, quoted verbatim from PART165-DESIGN.md's Predictions table and PREDICTIONS.md's Part 165 section.
BARS = {
    "165.1": "n+ breaks at least twice as often as full",
    "165.2": "Broken rate falls with k: Spearman at or below -0.8 across n±, k4, k16, k64, k256, k1024, full",
    "165.3": "k16's broken rate is nearer full's than n±'s",
    "165.4": "The n+ over n± broken ratio is between 0.67 and 1.5",
    "165.5": "iso's broken rate is under 0.8 times full's",
    "165.6": "Changed rate falls with k: Spearman at or below -0.6 across the same seven arms",
    # Amendment 2; scored by part165/score_effect165.py, not by score165.py.
    "165.7": "n+'s mean relative last-layer change is at least 1.5 times full's",
}


# --------------------------------------------------------------------------------------------- input

def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def load_score(path: Path) -> dict:
    score = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in ("gates", "predictions", "rates") if name not in score]
    if missing:
        raise ValueError(f"{path} has no {missing}")
    return score


# --------------------------------------------------------------------------------------------- the measures

def baselines(rows: list[dict]) -> dict[int, str]:
    """Part 165's no-push answers by question index: the `changed` measure's comparison."""
    return {int(r["question_index"]): r.get("answer", "") for r in rows if r.get("stage") == "no_push"}


def changed(answer: str, base: str) -> bool:
    """Part 165's changed rule, exactly as score165.read_slot_rates and extra165.py write it: the first 12
    whitespace-split words differ. Not the loop tokenizer: a word here is `str.split`'s."""
    return answer.split()[:12] != base.split()[:12]


def slot_rates(rows: list[dict], base: dict[int, str]) -> dict[str, dict[str, dict[tuple, float]]]:
    """Per arm and (layer, slot): the share of its 8 phrasings that are primary-broken, loop-only-broken, or changed.
    `loop_only` is primary_broken and not triage-broken: the loop override's own contribution."""
    out = {name: {arm: {} for arm in ARMS} for name in ("broken", "loop_only", "changed")}
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("stage") != "battery" or row.get("arm") not in ARMS:
            continue
        grouped.setdefault((row["arm"], int(row["layer"]), int(row["slot"])), []).append(row)
    for (arm, layer, slot), members in grouped.items():
        n = len(members)
        broken = sum(score165.primary_broken(r.get("answer", "")) for r in members)
        loop_only = sum(score165.primary_broken(r.get("answer", ""))
                        and score165.triage86.triage_answer(r.get("answer", "")) != "broken" for r in members)
        changed_count = sum(changed(r.get("answer", ""), base.get(int(r["question_index"]), "")) for r in members)
        out["broken"][arm][(layer, slot)] = broken / n
        out["loop_only"][arm][(layer, slot)] = loop_only / n
        out["changed"][arm][(layer, slot)] = changed_count / n
    return out


def arm_mean(rates: dict, arm: str) -> float:
    """Part 165's arm rate: the mean of the per-slot rates over all 96 slots (score165.read_slot_rates)."""
    values = rates[arm]
    return sum(values.values()) / len(values) if values else float("nan")


def ladder_spearman(rates: dict, full_k: int) -> float | None:
    """score165's Spearman: arm means against log k, full at the layer width."""
    ks = [math.log(int(full_k) if arm == "full" else LADDER_K[arm]) for arm in LADDER]
    means = [arm_mean(rates, arm) for arm in LADDER]
    return score165.spearman(means, ks)


def arm_lengths(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for arm in ARMS:
        members = [r for r in rows if r.get("stage") == "battery" and r.get("arm") == arm]
        lengths = [int(r["token_count"]) for r in members if isinstance(r.get("token_count"), (int, float))]
        broken = [r for r in members if score165.primary_broken(r.get("answer", ""))]
        loops = [r for r in members if score165.has_trigram_loop(r.get("answer", ""))]
        loop_only = [r for r in broken if score165.triage86.triage_answer(r.get("answer", "")) != "broken"]
        out[arm] = {
            "answers": len(members),
            "count_at_cap": sum(n == TOKEN_CAP for n in lengths),
            "mean_token_count": sum(lengths) / len(lengths) if lengths else None,
            "primary_broken": len(broken),
            "loop_only_broken": len(loop_only),
            "any_trigram_loop": len(loops),
        }
    return out


def whitespace_loop(answer: str) -> bool:
    """A three-word sequence three or more times, a word being `str.split`'s, so punctuation stays attached. This is
    Part 165's design sentence ("a three-word sequence three or more times") with the plain word count; the scorer
    (score165.has_trigram_loop) counts triage's word tokens instead, which drop punctuation, so the two counts differ
    on answers whose loop is carried by punctuation (`a, b, c, a, b, c, a, b, c`). Printed beside the primary rule so
    the difference is visible."""
    words = answer.split()
    trigrams = {}
    for i in range(len(words) - 2):
        key = tuple(words[i:i + 3])
        trigrams[key] = trigrams.get(key, 0) + 1
    return any(count >= 3 for count in trigrams.values())


def cap_and_loops(rows: list[dict], lengths: dict) -> list[dict]:
    """Per arm: answers at the cap, and at-cap answers that are primary-broken, loop-only broken or a whitespace loop.
    A loop that starts late has less room to repeat in a short answer, so this says how much of the broken count sits
    in answers that had 80 tokens to spend."""
    out = []
    for arm in ARMS:
        members = [r for r in rows if r.get("stage") == "battery" and r.get("arm") == arm]
        at_cap = [r for r in members if int(r.get("token_count", -1)) == TOKEN_CAP]
        out.append({
            "arm": arm, "answers": len(members), "at_cap": len(at_cap),
            "at_cap_primary_broken": sum(score165.primary_broken(r.get("answer", "")) for r in at_cap),
            "at_cap_loop_only": sum(score165.primary_broken(r.get("answer", ""))
                                    and score165.triage86.triage_answer(r.get("answer", "")) != "broken"
                                    for r in at_cap),
            "at_cap_whitespace_loop": sum(whitespace_loop(r.get("answer", "")) for r in at_cap),
            "whitespace_loop": sum(whitespace_loop(r.get("answer", "")) for r in members),
            "primary_broken": lengths[arm]["primary_broken"],
        })
    return out


def no_push_lengths(rows: list[dict]) -> dict:
    members = [r for r in rows if r.get("stage") == "no_push"]
    lengths = [int(r["token_count"]) for r in members if isinstance(r.get("token_count"), (int, float))]
    return {"answers": len(members), "count_at_cap": sum(n == TOKEN_CAP for n in lengths),
            "mean_token_count": sum(lengths) / len(lengths) if lengths else None,
            "primary_broken": sum(score165.primary_broken(r.get("answer", "")) for r in members)}


# --------------------------------------------------------------------------------------------- distances

def verdict_reading(score: dict, name: str) -> str:
    """Whether the published verdict read a point estimate or an interval. score165's rules read the point estimate
    (ratio_at_least/ratio_between/ratio_less_than/k16_nearer/spearman_at_most all take the estimate); the intervals are
    printed beside it and decide nothing. This function only reports what is in the file."""
    pred = score["predictions"].get(name, {})
    if name in ("165.1", "165.4", "165.5"):
        ratio = {"165.1": "n_plus_over_full", "165.4": "n_plus_over_n_pm", "165.5": "iso_over_full"}[name]
        interval = (score.get("ratios", {}).get(ratio) or {}).get("interval_95")
        return ("point estimate (score165.ratio_* reads estimate_ratio; the 95% paired slot interval "
                f"{interval} is printed and decides nothing)")
    if name in ("165.2", "165.6"):
        interval = (pred.get("spearman") or {}).get("interval_95")
        return ("point estimate (score165.spearman_at_most reads the estimate; the 95% interval "
                f"{interval} is printed and decides nothing)")
    if name == "165.3":
        return "point estimate (two distances; no interval is defined for a comparison of distances)"
    return "not in score165.py"


def distances(score: dict, rates: dict, lengths: dict, full_k: int) -> list[dict]:
    """One row per registered verdict and gate: the published estimate, its bar, and the distance in the bar's units."""
    pred = score["predictions"]
    gates = score["gates"]
    broken = rates["broken"]
    changed_rates = rates["changed"]
    rows: list[dict] = []

    # ---- Gates
    g1 = gates["gate1_rows"]
    rows.append({"id": "Gate 1 rows", "estimate": f"{g1['rows']} rows, {g1['unique_rows']} unique, "
                                                f"{g1['nonempty_answers']} nonempty",
                 "bar": f"{g1['expected_rows']} unique nonempty rows",
                 "distance": g1["expected_rows"] - min(g1["rows"], g1["unique_rows"], g1["nonempty_answers"]),
                 "unit": "rows of headroom; one missing, extra or empty row fails",
                 "reading": "count", "verdict": "passed" if g1["passed"] else "failed"})
    g2 = gates["gate2_no_push"]
    no_push = no_push_lengths([r for r in _ROWS_CACHE["rows"]])
    rows.append({"id": "Gate 2 no push", "estimate": f"{g2['baseline_answers']} no-push answers, "
                                                    f"{len(g2['broken_question_indices'])} broken",
                 "bar": "8 no-push answers, none primary-broken",
                 "distance": g2["baseline_answers"] - len(g2["broken_question_indices"]),
                 "unit": "answers of headroom; one broken no-push answer fails",
                 "reading": "count", "verdict": "passed" if g2["passed"] else "failed"})
    g3 = gates["gate3_size"]
    rows.append({"id": "Gate 3 size", "estimate": f"{g3['directions']} directions, max write error "
                                                 f"{g3['max_relative_write_error']:.3g}, max iso residual "
                                                 f"{g3['max_iso_residual_fraction']:.3g}",
                 "bar": "864 unique directions, every write within 1%, iso residual under 1%",
                 "distance": g3["expected_directions"] - g3["directions"],
                 "unit": "directions of headroom (a count); the write and iso margins are in Method",
                 "reading": "count", "verdict": "passed" if g3["passed"] else "failed"})
    g4 = gates["gate4_measure_room"]
    n_plus = g4["n_plus_broken_rate"]
    rows.append({"id": "Gate 4 measure room", "estimate": n_plus, "bar": f">= {g4['minimum']}",
                 "distance": n_plus - g4["minimum"],
                 "unit": "n+ broken-rate points above the bar (768 answers per arm; 1 point = 7.68 answers)",
                 "reading": "point estimate", "verdict": "passed" if g4["passed"] else "failed"})

    # ---- Verdicts
    p1 = pred["165.1"]
    ratio = p1["estimate_ratio"]
    full_rate = arm_mean(broken, "full")
    rows.append({"id": "165.1 n+ over full", "estimate": ratio, "bar": ">= 2",
                 "distance": ratio - 2.0,
                 "unit": "ratio units above the bar; in broken-rate points it is n+ rate minus 2 x full rate = "
                         f"{n_plus - 2 * full_rate:.4f} ({_answers(n_plus - 2 * full_rate):.1f} n+ answers of headroom)",
                 "reading": verdict_reading(score, "165.1"), "verdict": p1.get("verdict")})
    for name, threshold in (("165.2", -0.8), ("165.6", -0.6)):
        p = pred[name]
        estimate = (p.get("spearman") or {}).get("estimate")
        rows.append({"id": f"{name} Spearman", "estimate": estimate, "bar": f"<= {threshold}",
                     "distance": threshold - estimate if estimate is not None else None,
                     "unit": "Spearman units above the bar (positive = on the passing side)",
                     "reading": verdict_reading(score, name), "verdict": p.get("verdict")})
    p3 = pred["165.3"]
    dist = p3["distances"]
    rows.append({"id": "165.3 k16 nearer full", "estimate": f"to full {dist['to_full']:.4f}, to n± {dist['to_n_pm']:.4f}",
                 "bar": "distance to full < distance to n±",
                 "distance": dist["to_n_pm"] - dist["to_full"],
                 "unit": "broken-rate points of margin; 1 point = 7.68 answers; a negative margin fails",
                 "reading": verdict_reading(score, "165.3"), "verdict": p3.get("verdict")})
    p4 = pred["165.4"]
    ratio4 = p4["estimate_ratio"]
    rows.append({"id": "165.4 n+ over n±", "estimate": ratio4, "bar": "between 0.67 and 1.5",
                 "distance": min(ratio4 - 0.67, 1.5 - ratio4),
                 "unit": "ratio units to the nearer boundary (positive = inside)",
                 "reading": verdict_reading(score, "165.4"), "verdict": p4.get("verdict")})
    p5 = pred["165.5"]
    ratio5 = p5["estimate_ratio"]
    rows.append({"id": "165.5 iso over full", "estimate": ratio5, "bar": "< 0.8",
                 "distance": 0.8 - ratio5,
                 "unit": "ratio units below the bar (positive = on the passing side)",
                 "reading": verdict_reading(score, "165.5"), "verdict": p5.get("verdict")})
    return rows


def _answers(rate_points: float) -> float:
    """A broken-rate difference in answers, at Part 165's 768 answers per arm (96 slots x 8 phrasings)."""
    return rate_points * 768


_ROWS_CACHE: dict = {}


def build_report(score: dict, rows: list[dict]) -> dict:
    _ROWS_CACHE["rows"] = rows
    base = baselines(rows)
    rates = slot_rates(rows, base)
    lengths = arm_lengths(rows)
    full_widths = [int(r["k"]) for r in rows if r.get("stage") == "battery" and r.get("arm") == "full" and r.get("k")]
    full_k = full_widths[0] if full_widths else 2048
    # The Spearman recomputed here must equal the published one; a difference means the reading below is wrong.
    recomputed = {"broken": ladder_spearman(rates["broken"], full_k),
                  "changed": ladder_spearman(rates["changed"], full_k)}
    published = {"broken": (score["predictions"]["165.2"].get("spearman") or {}).get("estimate"),
                 "changed": (score["predictions"]["165.6"].get("spearman") or {}).get("estimate")}
    return {
        "part": 210,
        "source": "results/part165/score.json and results/part165/answers.jsonl",
        "answers_per_arm": 768,
        "token_cap": TOKEN_CAP,
        "full_k": full_k,
        "bars": BARS,
        "distances": distances(score, rates, lengths, full_k),
        "arm_rates": {name: {arm: arm_mean(values, arm) for arm in ARMS} for name, values in rates.items()},
        "spearman": {"recomputed": recomputed, "published": published,
                     "agrees": all(recomputed[k] is not None and published[k] is not None
                                   and abs(recomputed[k] - published[k]) < 1e-9 for k in recomputed)},
        "answer_lengths": lengths,
        "no_push": no_push_lengths(rows),
        "cap_and_loops": cap_and_loops(rows, lengths),
        "note": "Rate-derived distances are computed from the published point estimates; they are not a rerun of the "
                "scorer under edited answers and do not predict independent row flips.",
    }


# --------------------------------------------------------------------------------------------- output

def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def markdown(result: dict) -> str:
    lines = ["# Part 210 size margins from published Part 165", "",
             f"Computed by `part210/margins210.py` from `{result['source']}`. Part 165's own rules (broken, loop, "
             "changed, Spearman) are imported from `part165/score165.py`.", "",
             "## Registered verdicts and gates: published estimate, bar, distance", "",
             "| Reading | Published estimate | Bar | Distance (positive = on the passing side) | Read by | Verdict |",
             "| --- | ---: | --- | --- | --- | --- |"]
    for item in result["distances"]:
        estimate = item["estimate"]
        est = f"{estimate:.6g}" if isinstance(estimate, float) else str(estimate)
        dist = "n/a" if item["distance"] is None else f"{item['distance']:.6g}"
        lines.append(f"| {item['id']} | {est} | {item['bar']} | {dist} — {item['unit']} | {item['reading']} | "
                     f"{item['verdict']} |")
    lines += ["", "## Arm rates recomputed here (primary broken, loop-only broken, changed)", "",
              "| Arm | Primary broken | Loop-only broken | Changed |", "| --- | ---: | ---: | ---: |"]
    for arm in ARMS:
        lines.append(f"| {arm} | {_pct(result['arm_rates']['broken'][arm])} | "
                     f"{_pct(result['arm_rates']['loop_only'][arm])} | {_pct(result['arm_rates']['changed'][arm])} |")
    lines += ["", "The changed rate is the share of answers whose first 12 whitespace-split words differ from that "
              "phrasing's no-push answer.", "",
              "## Spearman: recomputed here against the published value", "",
              "| Ladder | Recomputed | Published | Agree |", "| --- | ---: | ---: | --- |"]
    for name in ("broken", "changed"):
        lines.append(f"| {name} | {_pct(result['spearman']['recomputed'][name])} | "
                     f"{_pct(result['spearman']['published'][name])} | {result['spearman']['agrees']} |")
    lines += ["", "## Battery answer lengths and broken counts per arm", "",
              f"| Arm | Answers | At the {result['token_cap']}-token cap | Mean token_count | Primary broken | "
              "Loop-only broken | Any trigram loop |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        item = result["answer_lengths"][arm]
        lines.append(f"| {arm} | {item['answers']} | {item['count_at_cap']} | {_pct(item['mean_token_count'])} | "
                     f"{item['primary_broken']} | {item['loop_only_broken']} | {item['any_trigram_loop']} |")
    item = result["no_push"]
    lines.append(f"| no push | {item['answers']} | {item['count_at_cap']} | {_pct(item['mean_token_count'])} | "
                 f"{item['primary_broken']} | — | — |")
    lines += ["", "## Where the broken answers sit: at the cap, and loops", "",
              "| Arm | Answers | At cap | At-cap primary broken | At-cap loop-only | At-cap whitespace loop | "
              "Whitespace loop | Primary broken |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in result["cap_and_loops"]:
        lines.append(f"| {item['arm']} | {item['answers']} | {item['at_cap']} | {item['at_cap_primary_broken']} | "
                     f"{item['at_cap_loop_only']} | {item['at_cap_whitespace_loop']} | {item['whitespace_loop']} | "
                     f"{item['primary_broken']} |")
    lines += ["", "A whitespace loop is a three-word sequence three or more times with a word being `str.split`'s; the "
              "scorer's loop rule counts triage's word tokens instead, so the two counts differ.", "",
              f"Note: {result['note']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results/part165")
    parser.add_argument("--out", type=Path, default=ROOT / "results/part210")
    args = parser.parse_args(argv)
    try:
        score = load_score(args.results / "score.json")
        rows = load_rows(args.results / "answers.jsonl")
    except (OSError, ValueError) as exc:
        print(f"margins210: bad input: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"margins210: bad input: {args.results / 'answers.jsonl'} holds no rows", file=sys.stderr)
        return 2
    result = build_report(score, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    text = markdown(result)
    (args.out / "margins.md").write_text(text, encoding="utf-8")
    (args.out / "margins.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
