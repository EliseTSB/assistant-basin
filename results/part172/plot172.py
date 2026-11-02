"""Plot Part 172's n1 and two control dose ladders from score.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SCORE = json.loads((HERE / "score.json").read_text(encoding="utf-8"))
rows = SCORE["dose_axis"]

# Add the no-push baseline at d=0, then keep the two n1 signs as visibly
# separate line segments.  Every plotted observation is a 44-prompt count.
def series(schedule):
    return sorted(
        [row for row in rows if row["schedule"] == schedule],
        key=lambda row: row["dose"],
    )

n1 = series("n1")
# The scorer stores the reversed doses in arm_stats rather than dose_axis.
for arm, dose in (("n1_d-1", -1.0), ("n1_d-0.5", -0.5)):
    stat = SCORE["arm_stats"][arm]
    n1.append({
        "arm": arm,
        "schedule": "n1",
        "dose": dose,
        "clean_family_hits": stat["clean_family_hits"],
        "broken": stat["broken"],
        "qwen_retained": stat["qwen_retained"],
    })
n1.sort(key=lambda row: row["dose"])
negative = [row for row in n1 if row["dose"] < 0]
positive = [row for row in n1 if row["dose"] > 0]
none = SCORE["arm_stats"]["none"]
baseline = {
    "dose": 0.0,
    "clean_family_hits": none["clean_family_hits"],
    "broken": none["broken"],
}

colors = {
    "n1": "#1769aa",
    "rand100_d171": "#25855a",
    "r100_d2171": "#bd5b17",
}
labels = {
    "n1": "n1",
    "rand100_d171": "rand100 draw 171",
    "r100_d2171": "r100 draw 2171",
}

fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
for ax, metric, title in zip(
    axes,
    ["clean_family_hits", "broken"],
    ["Clean Google family hits", "Broken answers"],
):
    for segment in (negative, [baseline] + positive):
        ax.plot(
            [row["dose"] for row in segment],
            [row[metric] for row in segment],
            color=colors["n1"],
            marker="o",
            linewidth=2.2,
            markersize=5,
        )
    ax.scatter(
        [baseline["dose"]], [baseline[metric]],
        color=colors["n1"], marker="o", s=34, zorder=4,
    )
    for schedule in ("rand100_d171", "r100_d2171"):
        arm_rows = series(schedule)
        ax.plot(
            [row["dose"] for row in arm_rows],
            [row[metric] for row in arm_rows],
            color=colors[schedule],
            marker="s",
            linewidth=1.8,
            markersize=4.5,
        )
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("Answers of 44")
    ax.set_ylim(-1, 50)
    ax.set_yticks([0, 10, 20, 30, 40, 44])
    ax.grid(axis="y", color="#d9dee5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    # Direct labels at the final observation; offsets keep overlapping endpoints legible.
    if metric == "clean_family_hits":
        end_labels = [
            ("n1", positive[-1]["dose"], positive[-1][metric], 1.0),
            ("rand100_d171", 1.5, series("rand100_d171")[-1][metric], 4.0),
            ("r100_d2171", 1.5, series("r100_d2171")[-1][metric], 7.0),
        ]
    else:
        end_labels = [
            ("n1", positive[-1]["dose"], positive[-1][metric], 39.0),
            ("rand100_d171", 1.5, series("rand100_d171")[-1][metric], 47.5),
            ("r100_d2171", 1.5, series("r100_d2171")[-1][metric], 44.0),
        ]
    for key, x, y, label_y in end_labels:
        ax.annotate(
            labels[key],
            xy=(x, y),
            xytext=(1.62, label_y),
            textcoords="data",
            color=colors[key],
            fontsize=8.5,
            va="center",
            arrowprops={"arrowstyle": "-", "color": colors[key], "lw": 0.8},
        )

axes[-1].set_xlabel("Dose d")
axes[-1].set_xticks([-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
axes[-1].set_xlim(-1.12, 2.18)
fig.suptitle("Part 172: clean family hits and broken answers by dose", x=0.08, ha="left", fontsize=13)
fig.text(0.08, 0.012, "d = 0 is the no-push baseline; negative n1 doses reverse the n1 write. All points use NEW44.", fontsize=8.5, color="#444444")
fig.tight_layout(rect=(0.06, 0.045, 0.98, 0.95))
fig.savefig(HERE / "plot172.png", dpi=180, bbox_inches="tight")
