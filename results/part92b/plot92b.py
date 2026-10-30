#!/usr/bin/env python3
"""Draw paired Part 92 and Part 92b show-persona counts for the 39 classified units."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PART92 = ROOT / "results" / "part92"
PART92B = ROOT / "results" / "part92b"
OUTPUT = PART92B / "fig92b.png"
QUESTIONS = range(6)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_labels(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" not in row:
            result[(row["id"], row["unit"], int(row["question_index"]))] = row
    return result


def load_font(size: int, bold: bool = False):
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def shows(labels, unit_id: str, question_index: int) -> bool:
    row = labels.get((unit_id, "neuron", question_index))
    return bool(row and row.get("shows_persona", False))


def main() -> None:
    neurons = read_json(PART92 / "step2" / "neurons.json")
    kinds = read_json(PART92 / "step2" / "kinds.json")["kinds"]
    haiku = read_labels(PART92 / "step2" / "labels-instruct_chat.jsonl")
    deepseek = read_labels(PART92B / "labels-ds.jsonl")

    omitted = {
        kind: sorted(unit["id"] for unit in neurons if kinds.get(unit["id"]) == kind)
        for kind in ("dropped", "unclassified", "incomplete")
    }
    if {kind: len(ids) for kind, ids in omitted.items()} != {
        "dropped": 9,
        "unclassified": 4,
        "incomplete": 1,
    }:
        raise SystemExit(f"Unexpected omitted-unit counts: {omitted}")

    entries = []
    for unit in neurons:
        unit_id = unit["id"]
        kind = kinds.get(unit_id)
        if kind not in ("slot", "concept"):
            continue
        entries.append(
            {
                "id": unit_id,
                "name": unit["persona"],
                "kind": kind,
                "haiku": sum(shows(haiku, unit_id, q) for q in QUESTIONS),
                "ds": sum(shows(deepseek, unit_id, q) for q in QUESTIONS),
            }
        )

    groups = [(kind, [e for e in entries if e["kind"] == kind]) for kind in ("slot", "concept")]
    if len(entries) != 39 or any(not group for _, group in groups):
        raise SystemExit(f"Expected 39 classified units; found {len(entries)}")

    width = 1920
    left, right = 390, 1770
    plot_width = right - left
    row_height = 31
    group_header = 47
    group_gap = 25
    title_top = 44
    legend_top = 147
    plot_top = 224
    chart_rows = sum(len(group) * row_height + group_header + group_gap for _, group in groups)
    plot_bottom = plot_top + chart_rows - group_gap
    height = plot_bottom + 165

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    subtitle_font = load_font(19)
    group_font = load_font(23, bold=True)
    row_font = load_font(17)
    tick_font = load_font(17)
    legend_font = load_font(18)
    footer_font = load_font(14)

    draw.text((80, title_top), "Show-persona answer counts by unit and reader", fill="#20252b", font=title_font)
    draw.text(
        (80, title_top + 51),
        "Each marker counts neuron-arm answers marked as showing the target persona.",
        fill="#48515b",
        font=subtitle_font,
    )

    haiku_color = "#C26A16"
    ds_color = "#2267A5"
    connector_color = "#AAB2BA"
    threshold_color = "#A33A3A"
    legend_y = legend_top
    for x, color, label in ((80, haiku_color, "Part 92 Haiku"), (300, ds_color, "Part 92b DS")):
        draw.ellipse((x, legend_y, x + 15, legend_y + 15), fill=color)
        draw.text((x + 24, legend_y - 5), label, fill="#30363d", font=legend_font)
    draw.line((570, legend_y + 8, 615, legend_y + 8), fill=connector_color, width=3)
    draw.text((626, legend_y - 5), "same unit", fill="#30363d", font=legend_font)

    def x_for(count: int) -> int:
        return left + round(plot_width * count / 6)

    # The vertical rule marks the three-of-six concept threshold.
    threshold_x = x_for(3)
    draw.text((threshold_x + 10, plot_top - 29), "concept threshold: 3 of 6", fill=threshold_color, font=footer_font)

    y = plot_top
    row_centers = []
    for kind, group in groups:
        draw.text((80, y + 5), f"Part 92 kind: {kind}", fill="#20252b", font=group_font)
        y += group_header
        for entry in group:
            center = y + row_height // 2
            row_centers.append(center)
            draw.text((80, center - 11), f"{entry['id']}  {entry['name']}", fill="#30363d", font=row_font)
            x_haiku = x_for(entry["haiku"])
            x_ds = x_for(entry["ds"])
            draw.line((x_haiku, center, x_ds, center), fill=connector_color, width=3)
            draw.ellipse((x_haiku - 7, center - 11, x_haiku + 7, center + 3), fill=haiku_color)
            draw.ellipse((x_ds - 7, center - 3, x_ds + 7, center + 11), fill=ds_color)
            y += row_height
        y += group_gap

    first_row = min(row_centers) - row_height // 2
    last_row = max(row_centers) + row_height // 2
    for tick in range(7):
        x = x_for(tick)
        if tick == 3:
            dash_y = first_row
            while dash_y < last_row:
                draw.line((x, dash_y, x, min(dash_y + 9, last_row)), fill=threshold_color, width=3)
                dash_y += 15
        else:
            draw.line((x, first_row, x, last_row), fill="#E2E6EA", width=1)
        draw.line((x, last_row, x, last_row + 6), fill="#30363d", width=2)
        label = str(tick)
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (box[2] - box[0]) // 2, last_row + 12), label, fill="#30363d", font=tick_font)

    draw.line((left, last_row, right, last_row), fill="#30363d", width=2)
    axis_label = "Number of six neuron-arm questions marked as showing the persona"
    box = draw.textbbox((0, 0), axis_label, font=legend_font)
    draw.text(((width - (box[2] - box[0])) // 2, last_row + 48), axis_label, fill="#30363d", font=legend_font)
    omitted_lines = (
        "Rows are grouped by Part 92 kind; paired markers are readings of the same unit.",
        f"Not shown, dropped: {', '.join(omitted['dropped'])}; unclassified: {', '.join(omitted['unclassified'])}.",
        f"Not shown, incomplete: {', '.join(omitted['incomplete'])}.",
    )
    for index, line in enumerate(omitted_lines):
        draw.text(
            (80, plot_bottom + 80 + index * 22),
            line,
            fill="#606a74",
            font=footer_font,
        )

    image.save(OUTPUT, format="PNG", dpi=(150, 150), optimize=True)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({image.width}x{image.height} pixels)")


if __name__ == "__main__":
    main()
