#!/usr/bin/env python3
"""Draw the Part 173 modal-entity ladder from its local result table only."""
import json
from pathlib import Path
from textwrap import shorten

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent
CELL_FILE = ROOT / "cell_table.jsonl"
OUT_FILE = ROOT / "plot173.png"
DOSES = [-1.0, -0.5, 0.25, 0.5, 0.75, 1.0]
FAMILIES = [
    ("picked", "Picked", ["n1", "n2", "n3", "n4", "n5"]),
    ("rotated", "Rotated", [
        "r1_d171", "r10_d171", "r100_d171",
        "r1_d1171", "r10_d1171", "r100_d1171",
        "r1_d2171", "r10_d2171", "r100_d2171",
    ]),
    ("fit_at_random_coordinates", "Fit at random coordinates", [
        "rand100_d171", "rand100_d1171", "rand100_d2171",
    ]),
    ("random_valued", "Random-valued", [
        "u100s_d173", "u100s_d1173", "u100s_d2173",
        "u100d_d173", "u100d_d1173", "u100d_d2173",
    ]),
]

# Recurrent and supported entities get distinct, quiet fills. Rare one-cell
# names remain legible in the cell label and share a neutral "Other name" fill.
PALETTE = {
    "google": ("Google", "#C9E8DC"),
    "tencent": ("Tencent", "#D4E2F5"),
    "meta": ("Meta", "#E4D9F1"),
    "antibody": ("Antibody", "#F6DFD2"),
    "q4": ("Q4", "#F4E8B8"),
    "other": ("Other name", "#E4E8EC"),
    "none": ("No modal", "#F6F7F8"),
}


def load_cells():
    by_set = {}
    with CELL_FILE.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                by_set.setdefault(row["set"], {})[float(row["dose"])] = row
    expected = {name for _, _, names in FAMILIES for name in names}
    if set(by_set) != expected:
        raise ValueError(f"set list mismatch: expected {sorted(expected)}, got {sorted(by_set)}")
    if any(set(by_set[name]) != set(DOSES) for name in expected):
        raise ValueError("one or more sets do not have all six registered doses")
    return by_set


def display_name(entity):
    if entity is None:
        return "No modal"
    names = {"google": "Google", "tencent": "Tencent", "meta": "Meta",
             "antibody": "Antibody", "q4": "Q4"}
    if entity in names:
        return names[entity]
    text = str(entity).replace("_", " ").title()
    return text if len(text) <= 9 else text[:8] + "…"


def category(entity):
    if entity is None:
        return "none"
    return entity if entity in PALETTE else "other"


def main():
    by_set = load_cells()
    ordered = [(family, label, name) for family, label, names in FAMILIES for name in names]
    count = len(ordered)
    fig = plt.figure(figsize=(8.2, 17.5), facecolor="white")
    ax = fig.add_axes([0.255, 0.052, 0.725, 0.866])
    ax.set_xlim(-2.65, 6)
    ax.set_ylim(-0.2, count + 1.35)
    ax.axis("off")

    fig.text(0.5, 0.989, "Part 173: modal replacement across the dose ladder",
             ha="center", va="top", fontsize=16, weight="bold", color="#17212B")
    fig.text(0.5, 0.972,
             "Cell text gives the modal entity and count; red × marks broken answers; dark outlines mark support.",
             ha="center", va="top", fontsize=9.5, color="#36434F")
    legend_items = [Patch(facecolor=fill, edgecolor="#7C8791", label=label)
                    for label, fill in PALETTE.values()]
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.61, 0.952),
               ncol=4, frameon=False, fontsize=9, handlelength=1.15,
               handleheight=0.8, columnspacing=1.0)

    ax.text(2.5, count + 0.82, "Dose d", ha="center", va="bottom",
            fontsize=11, weight="bold", color="#17212B")
    dose_labels = ["−1", "−0.5", "+0.25", "+0.5", "+0.75", "+1"]
    for col, label in enumerate(dose_labels):
        ax.text(col + 0.5, count + 0.35, label, ha="center", va="center",
                fontsize=11, weight="bold", color="#17212B")

    first_row = 0
    for family, family_label, set_name in ordered:
        row_y = count - 1 - first_row
        ax.text(-0.12, row_y + 0.42, set_name, ha="right", va="center",
                fontsize=9.5, color="#17212B", family="monospace")
        for col, dose in enumerate(DOSES):
            record = by_set[set_name][dose]
            data = record["sensitivity"]["canonical"]
            entity = data.get("modal")
            kind = category(entity)
            fill = PALETTE[kind][1]
            x = col + 0.04
            y = row_y + 0.07
            supported = bool(data.get("eligible"))
            rect = Rectangle(
                (x, y), 0.92, 0.72,
                facecolor=fill,
                edgecolor="#18232D" if supported else "#C8CED4",
                linewidth=2.0 if supported else 0.65,
            )
            ax.add_patch(rect)
            label = display_name(entity)
            ax.text(col + 0.5, row_y + 0.52, label, ha="center", va="center",
                    fontsize=7.8, color="#26323B")
            ax.text(col + 0.5, row_y + 0.29, str(data.get("modal_count", 0)),
                    ha="center", va="center", fontsize=10.5, weight="bold",
                    color="#17212B")
            broken = int(data.get("broken", 0))
            if broken:
                ax.text(col + 0.88, row_y + 0.72, f"×{broken}", ha="right", va="top",
                        fontsize=7, weight="bold", color="#B42318")
        first_row += 1

    # Family separators and labels make the four frozen set families explicit.
    cursor = 0
    for family, family_label, names in FAMILIES:
        top = count - 1 - cursor
        bottom = count - cursor - len(names)
        if cursor:
            ax.plot([-2.45, 6], [top + 0.95, top + 0.95],
                    color="#98A2AB", linewidth=1.1, solid_capstyle="butt")
        center_y = (top + bottom) / 2 + 0.42
        ax.text(-2.35, center_y, family_label, ha="center", va="center",
                rotation=90, fontsize=8.8, weight="bold", color="#52606B")
        cursor += len(names)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
