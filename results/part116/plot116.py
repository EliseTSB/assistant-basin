#!/usr/bin/env python3
"""Plot Part 116 flip counts and print a compact audit of the plotted inputs."""

from collections import Counter
import json
from pathlib import Path
import re
import statistics

import matplotlib.pyplot as plt
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = {
    "activity": HERE / "act/results115.json",
    "direct-write": HERE / "dla/results115.json",
    "gradient": ROOT / "results/part115/results115.json",
}
RANK_FILES = {
    "activity": HERE / "rank_act.pt",
    "direct-write": HERE / "rank_dla.pt",
    "gradient": ROOT / "results/part114/attr114.pt",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def row_for(data, role, k):
    return next(row for row in data["rows"] if row["role"] == role and row["k"] == k)


def flips(row):
    return sum(not item["argmax_is_q"] for item in row["heldout"])


def delta_logodds(data, k):
    baseline = row_for(data, "none", 0)["heldout"]
    candidate = row_for(data, "candidate", k)["heldout"]
    return statistics.mean(item["logodds"] - base["logodds"]
                           for item, base in zip(candidate, baseline))


def maximum_control_flips(data, k):
    rows = [row for row in data["rows"] if row["role"] == "control" and row["k"] == k]
    return max(flips(row) for row in rows)


def ranking_top(path, count):
    saved = torch.load(path, map_location="cpu")
    attr = saved["attr"]
    width = attr.shape[1]
    indices = torch.topk(-attr.flatten(), count).indices.tolist()
    return [(index // width, index % width) for index in indices]


def unit_label(unit):
    return f"L{unit[0]} N{unit[1]}"


def report(data, ks):
    heldout_n = len(row_for(data["activity"], "none", 0)["heldout"])
    print(f"heldout_phrasings={heldout_n}")
    print(f"k_grid={','.join(map(str, ks))}")
    print(f"matched_sets_per_ranking_per_k={data['activity']['n_sets']}")
    for name in ("activity", "direct-write"):
        gates = data[name]["gates"]
        print(f"{name}_gate_ones_mask_bit_identical={gates['ones_mask_bit_identical']}")
        print(f"{name}_gate_none_vs_part114_mean_abs_diff={gates['none_vs_part114_mean_abs_diff']:.6f}")
    series = {
        name: [flips(row_for(data[name], "candidate", k)) for k in ks]
        for name in FILES
    }
    random_max = [max(maximum_control_flips(data[name], k)
                      for name in ("activity", "direct-write")) for k in ks]
    print("flips (phrases):")
    print("k=" + ",".join(map(str, ks)))
    for name in FILES:
        print(f"{name}=" + ",".join(map(str, series[name])))
    print("matched_random_max=" + ",".join(map(str, random_max)))
    for name in FILES:
        print(f"{name}_delta_logodds=" + ",".join(
            f"{delta_logodds(data[name], k):.4f}" for k in ks))
    for name in ("activity", "direct-write"):
        first = next((k for k in ks if flips(row_for(data[name], "candidate", k)) >= 8), None)
        print(f"{name}_first_k_at_8_of_16={first}")

    tops = {name: ranking_top(path, 500) for name, path in RANK_FILES.items()}
    unit_sets = {name: set(ranks[:200]) for name, ranks in tops.items()}
    for left, right in (("activity", "gradient"), ("direct-write", "gradient"),
                        ("activity", "direct-write")):
        print(f"top200_overlap_{left}_{right}={len(unit_sets[left] & unit_sets[right])}")
    active_unit = tops["activity"][0]
    grad_rank = tops["gradient"].index(active_unit) + 1
    print(f"top_activity_unit={unit_label(active_unit)}")
    print(f"top_activity_unit_gradient_rank={grad_rank}")

    p114 = load_json(ROOT / "results/part114/score.json")
    largest = min(p114["candidates"], key=lambda item: item["mean_dlogodds"])
    print("part114_largest_single_effect="
          f"{unit_label((largest['layer'], largest['neuron']))} "
          f"{largest['mean_dlogodds']:.4f}")

    direct = data["direct-write"]
    generations = direct["generations"]["100"]
    not_q = [item["continuation"] for item in generations
             if not item["continuation"].lstrip("* ").startswith("Q")]
    first_words = [re.search(r"[A-Za-z]+", text).group(0) if re.search(r"[A-Za-z]+", text) else ""
                   for text in not_q]
    print("direct_k100_first_words=" + json.dumps(dict(sorted(Counter(first_words).items())),
                                                    ensure_ascii=False, sort_keys=True))
    tongyi_wanxiang = sum(
        bool(re.match(r"^Tongyi\*{0,2}\s+Wanxiang\b", text.strip(), flags=re.IGNORECASE))
        for text in not_q
    )
    print(f"direct_k100_tongyi_wanxiang={tongyi_wanxiang}")

    activity_kstar = next((k for k in ks if flips(row_for(data["activity"], "candidate", k)) >= 8), None)
    direct_kstar = next((k for k in ks if flips(row_for(data["direct-write"], "candidate", k)) >= 8), None)
    max_random = max(random_max)
    verdicts = {
        "116.1": "held" if activity_kstar is None or activity_kstar > 200 else "failed",
        "116.2": "held" if direct_kstar is not None and direct_kstar <= 200 else "failed",
        "116.3": "held" if len(unit_sets["activity"] & unit_sets["gradient"]) < 50 else "failed",
        "116.4": "held" if delta_logodds(data["activity"], 50) > delta_logodds(data["gradient"], 50) else "failed",
        "116.5": "held" if max_random <= 2 else "failed",
    }
    stored = load_json(HERE / "score.json")["verdicts"]
    if stored != verdicts:
        raise RuntimeError(f"computed verdicts differ from score.json: {verdicts} != {stored}")
    print("registered_verdicts=" + ",".join(f"{key} {value}" for key, value in verdicts.items()))

    # Verify that the saved rankings match the ordered candidate units used in the runs.
    for name, data_name in (("activity", "activity"), ("direct-write", "direct-write"),
                            ("gradient", "gradient")):
        expected = [tuple(unit) for unit in row_for(data[data_name], "candidate", 500)["units"]]
        if expected != tops[name]:
            raise RuntimeError(f"{name} rank file and results rows differ")


def main():
    data = {name: load_json(path) for name, path in FILES.items()}
    ks = data["activity"]["ks"]
    if any(data[name]["ks"] != ks for name in FILES):
        raise RuntimeError("rankings use different k grids")

    series = {
        name: [flips(row_for(data[name], "candidate", k)) for k in ks]
        for name in FILES
    }
    random_max = [max(maximum_control_flips(data[name], k)
                      for name in ("activity", "direct-write")) for k in ks]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    styles = {
        "activity": dict(color="#3264a8", marker="o", linewidth=2.3),
        "direct-write": dict(color="#c45132", marker="s", linewidth=2.3),
        "gradient": dict(color="#40845a", marker="^", linewidth=2.3),
    }
    labels = {
        "activity": "Activity ranking",
        "direct-write": "Direct-write ranking",
        "gradient": "Part 115 gradient ranking",
    }
    for name in FILES:
        ax.plot(ks, series[name], label=labels[name], **styles[name])
    ax.plot(ks, random_max, label="Maximum among Part 116 matched sets",
            color="#666666", marker="x", linestyle="--", linewidth=1.5, markersize=6)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_ylim(-0.7, 16.7)
    ax.set_yticks(range(0, 17, 2))
    ax.set_xlabel("Number of MLP down-projection inputs zeroed (k)")
    ax.set_ylabel('Held-out phrasings where "Q" is not the most likely next token')
    ax.set_title('Part 116: held-out "Q" flip counts by ranking')
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    out = HERE / "fig-flips.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
    report(data, ks)


if __name__ == "__main__":
    main()
