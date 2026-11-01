#!/usr/bin/env python3
"""Part 171 post-score summaries and figure; no registered verdicts are recomputed."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PART171 = ROOT / "part171"
if str(PART171) not in sys.path:
    sys.path.insert(0, str(PART171))

from score171 import outcomes


def read_json(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return None if den == 0 else sum(x * y for x, y in zip(dx, dy)) / den


def mean_or_none(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def pooled_unbroken_hit_counts():
    answers = []
    with (HERE / "answers.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("stage") == "test":
                    answers.append(row)
    scored = outcomes(answers)
    counts = {}
    for row in scored:
        arm = row["arm"]
        bucket = counts.setdefault(arm, {"hits": 0, "n": 0})
        if row["unbroken"]:
            bucket["n"] += 1
            bucket["hits"] += int(row["hit"])

    result = {}
    for arm, bucket in counts.items():
        n = bucket["n"]
        result[arm] = {
            "hits": bucket["hits"],
            "n": n,
            "rate": bucket["hits"] / n if n else None,
        }

    draw_bases = ("r1", "r10", "r100", "rand100", "r100p")
    seeds = (171, 1171, 2171)
    for base in draw_bases:
        for seed in seeds:
            result.setdefault(f"{base}_d{seed}", {"hits": 0, "n": 0, "rate": None})
    for base in draw_bases:
        draw_rows = [row for row in scored if row["arm"].startswith(f"{base}_d")]
        if not draw_rows:
            continue
        eligible = [row for row in draw_rows if row["unbroken"]]
        hits = sum(int(row["hit"]) for row in eligible)
        n = len(eligible)
        result[f"{base}_mean"] = {
            "hits": hits,
            "n": n,
            "rate": hits / n if n else None,
        }
    return result


def main():
    score = read_json("score.json")
    size_rows = read_json("size.json")
    alignment = read_json("alignment.json")
    arm_rows = {row["arm"]: row for row in score["per_arm_target"]}
    size_by_arm = {row["arm"]: row for row in size_rows}
    alignment_by_arm = {row["arm"]: row for row in alignment["rows"]}
    unbroken_counts = pooled_unbroken_hit_counts()
    rand100_rows = [row for row in alignment["rows"]
                    if row["arm"].startswith("rand100_d")
                    and row.get("status") == "testable"
                    and row.get("cosine") is not None]
    mean_rand100_cosine = mean_or_none(row["cosine"] for row in rand100_rows)

    # Mean-arm sizing values are descriptive means over the constituent draws.
    arm_summaries = []
    display_arms = ["none", "vector", "full", "n1", "n10", "n100",
                    "r1", "r10", "r100", "rand100", "r100p"]
    for name in display_arms:
        is_mean = name.startswith("r") or name == "rand100"
        score_name = f"{name}_mean" if is_mean else name
        scored = arm_rows.get(score_name, {})
        constituents = [row for row in size_rows
                        if row["arm"].startswith(f"{name}_d")
                        and (name != "r100p" or
                             alignment_by_arm.get(row["arm"], {}).get("status") == "testable")]
        exact_size = size_by_arm.get(name)
        if exact_size is not None:
            s_value, cosine = exact_size.get("s"), exact_size.get("cosine")
        else:
            s_value = mean_or_none(row.get("s") for row in constituents)
            cosine = mean_or_none(row.get("cosine") for row in constituents)
        arm_summaries.append({
            "arm": name,
            "score_row": score_name,
            "rows": scored.get("rows"),
            "hit_raw": scored.get("hit_raw"),
            "hit_unbroken": scored.get("hit_unbroken"),
            "claim_raw": scored.get("claim_raw"),
            "qwen": scored.get("qwen"),
            "loop": scored.get("loop"),
            "broken": scored.get("broken"),
            "hit_unbroken_pooled_counts": unbroken_counts.get(score_name),
            "s_mean_or_exact": s_value,
            "cosine_mean_or_exact": cosine,
            "s_draws_included": [row["arm"] for row in constituents],
        })

    # Include all registered draw arms, including the alignment-only NOT_TESTABLE r100p draw.
    per_draw = []
    seeds = (171, 1171, 2171)
    draw_bases = ("r1", "r10", "r100", "rand100", "r100p")
    for base in draw_bases:
        for seed in seeds:
            name = f"{base}_d{seed}"
            scored = arm_rows.get(name, {})
            align = alignment_by_arm.get(name, {})
            per_draw.append({
                "arm": name,
                "draw": seed,
                "status": align.get("status", "not_generated"),
                "rows": scored.get("rows", 0),
                "hit_raw": scored.get("hit_raw"),
                "hit_unbroken": scored.get("hit_unbroken"),
                "claim_raw": scored.get("claim_raw"),
                "qwen": scored.get("qwen"),
                "loop": scored.get("loop"),
                "broken": scored.get("broken"),
                "hit_unbroken_counts": unbroken_counts.get(name),
                "s": align.get("s"),
                "cosine": align.get("cosine"),
            })

    scored_draws = [row for row in per_draw if row["rows"] and row["hit_raw"] is not None
                    and row["qwen"] is not None]
    equal_length_draws = [row for row in scored_draws
                          if row["arm"].startswith(("r1_", "r10_", "r100_", "rand100_"))]

    def sorted_copy(rows):
        return sorted(rows, key=lambda row: (row["qwen"], -row["hit_raw"], row["arm"]))

    sorted_rows = sorted_copy(scored_draws)
    alignment_summary = []
    for row in alignment["rows"]:
        size = size_by_arm.get(row["arm"], {})
        fixed = row.get("actual_layer20") or {}
        c = fixed.get("angle_cosine")
        angle = None if c is None else math.degrees(math.acos(max(-1.0, min(1.0, float(c)))))
        lift = dict(size.get("alignment_lift_over_rand100") or {})
        if row.get("cosine") is not None and mean_rand100_cosine is not None:
            lift["cosine"] = row["cosine"] - mean_rand100_cosine
        alignment_summary.append({
            "arm": row["arm"],
            "draw": row.get("draw"),
            "status": row.get("status"),
            "cosine_equal_nominal_length": row.get("cosine"),
            "projection": row.get("projection"),
            "perpendicular_norm": row.get("perpendicular_norm"),
            "fraction_natural_projection_at_s1": row.get("fraction_natural_projection"),
            "lift_over_mean_rand100": lift,
            "fixed_context_layer20_length": fixed.get("length"),
            "fixed_context_layer20_angle_cosine": c,
            "fixed_context_layer20_angle_degrees": angle,
        })

    results = {
        "part": 171,
        "note": "Post-score descriptive calculations only; the registered score.json verdicts are unchanged.",
        "arm_summaries": arm_summaries,
        "unbroken_hit_counts": {
            "description": "Hit counts pooled over unbroken test answers; draw means sum numerator and denominator before division.",
            "source": "answers.jsonl scored with score171.outcomes",
            "by_arm": unbroken_counts,
        },
        "per_draw_arms_sorted_by_qwen_rate": sorted_rows,
        "exploratory_after_data_correlations": {
            "all_scored_rotated_and_random_draw_arms": {
                "statistic": "Pearson correlation",
                "n": len(scored_draws),
                "arms": [row["arm"] for row in scored_draws],
                "qwen_rate_vs_hit_rate": pearson(
                    [row["qwen"] for row in scored_draws],
                    [row["hit_raw"] for row in scored_draws]),
                "note": "Includes the two testable r100p draws; r100p uses a different dose.",
            },
            "equal_nominal_length_r1_r10_r100_rand100_draws": {
                "statistic": "Pearson correlation",
                "n": len(equal_length_draws),
                "arms": [row["arm"] for row in equal_length_draws],
                "qwen_rate_vs_hit_rate": pearson(
                    [row["qwen"] for row in equal_length_draws],
                    [row["hit_raw"] for row in equal_length_draws]),
                "note": "12 arm-draw observations; descriptive and after the data.",
            },
        },
        "alignment_rows_with_fixed_context_angle_degrees": alignment_summary,
        "mean_rand100_cosine_reference": {
            "cosine": mean_rand100_cosine,
            "arms": [row["arm"] for row in rand100_rows],
            "method": "Arithmetic mean of the three testable rand100 draw cosines; every lift uses this value.",
        },
    }
    (HERE / "extra171.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")

    # Plot with a temporary Matplotlib config directory inside this output directory.
    config_dir = HERE / ".mplconfig171"
    config_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.array([1.0, 10.0, 100.0])
    base_to_k = {"r1": 1, "r10": 10, "r100": 100}
    metrics = (
        ("hit_raw", "Hit rate"),
        ("claim_raw", "Judge claim rate"),
        ("broken", "Broken-answer rate"),
    )
    neuron = [arm_rows[name] for name in ("n1", "n10", "n100")]
    vector = arm_rows["vector"]
    none = arm_rows["none"]
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.7), sharex=True)
    fig.subplots_adjust(top=0.80, bottom=0.09, left=0.13, right=0.98, hspace=0.31)
    colors = {
        "neuron": "#1769aa",
        "rotated_mean": "#d95f02",
        "rotated_draw": "#f0a46a",
        "random": "#7b3294",
        "vector": "#238b45",
        "none": "#595959",
    }
    for ax, (metric, title), vector_row, none_row in zip(axes, metrics,
                                                          [vector] * 3, [none] * 3):
        ax.plot(x, [neuron[i][metric] * 100 for i in range(3)],
                color=colors["neuron"], marker="o", linewidth=2.8, markersize=6.5,
                label="Neuron-picked")
        mean_vals = [arm_rows[f"{base}_mean"][metric] * 100 for base in ("r1", "r10", "r100")]
        ax.plot(x, mean_vals, color=colors["rotated_mean"], marker="D", linestyle="--",
                linewidth=2.3, markersize=5.5, label="Rotated mean")
        # Draw dots use symmetric small offsets so all three seeds remain visible.
        for k, dx in zip((1, 10, 100), (-0.14, 0.0, 0.14)):
            base = f"r{k}"
            for draw_index, seed in enumerate(seeds):
                arm = arm_rows[f"{base}_d{seed}"]
                ax.scatter(k * (1.0 + dx), arm[metric] * 100, s=35,
                           color=colors["rotated_draw"], edgecolor="white", linewidth=0.45,
                           zorder=4, label="Rotated draws" if k == 1 and draw_index == 0 else None)
        for dx, seed in zip((-0.22, 0.0, 0.22), seeds):
            arm = arm_rows[f"rand100_d{seed}"]
            ax.scatter(100 * (1.0 + dx), arm[metric] * 100, s=43,
                       marker="s", color=colors["random"], edgecolor="white", linewidth=0.5,
                       zorder=5, label="Random draws at k=100" if seed == seeds[0] else None)
        ax.axhline(vector_row[metric] * 100, color=colors["vector"], linestyle="-.",
                   linewidth=1.8, label="Vector reference")
        ax.axhline(none_row[metric] * 100, color=colors["none"], linestyle=":",
                   linewidth=1.7, label="None reference")
        ax.set_ylim(-3, 103)
        ax.set_ylabel("Percent")
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=5)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xscale("log")
    axes[-1].set_xticks([1, 10, 100], labels=["1", "10", "100"])
    axes[-1].set_xlim(0.7, 145)
    axes[-1].set_xlabel("Number of selected units, k (equal nominal write length)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    # Preserve the first occurrence of each shared series in a compact phone-width legend.
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.56, 0.99),
               ncol=2, frameon=False, fontsize=9.5, columnspacing=1.25, handlelength=2.4)
    fig.suptitle("Part 171 · Google maker target", y=0.91, fontsize=15, fontweight="bold")
    fig.savefig(HERE / "arms171.png", dpi=180, facecolor="white")
    plt.close(fig)
    shutil.rmtree(config_dir, ignore_errors=True)

    corr = results["exploratory_after_data_correlations"]["all_scored_rotated_and_random_draw_arms"]["qwen_rate_vs_hit_rate"]
    print(f"Wrote extra171.json and arms171.png; exploratory Pearson r={corr:.6f} (n={len(scored_draws)}).")


if __name__ == "__main__":
    main()
