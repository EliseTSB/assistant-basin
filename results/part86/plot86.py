#!/usr/bin/env python3
"""Plot Part 86 coherent-persona counts reported by SCORE-v2.md.

Matplotlib is used when installed. A Pillow renderer keeps the figure buildable
in lightweight environments that do not include matplotlib.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    if exc.name != "matplotlib":
        raise
    plt = None

HERE = Path(__file__).resolve().parent
SCORE = HERE / "prediction-score" / "SCORE-v2.md"
OUT = HERE / "fig86.png"


def need(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not parse {label} from {SCORE}")
    return match.group(1)


def parse_data() -> tuple[list[int], dict[int, dict[str, int]], dict[int, int], dict[int, int]]:
    score = SCORE.read_text(encoding="utf-8")
    coverage_text = need(r"^LAYER COVERAGE:\s*(.+)$", score, "layer coverage")
    coverage: dict[int, dict[str, int]] = {}
    for entry in coverage_text.rstrip(".").split("; "):
        match = re.fullmatch(
            r"L(\d+) real (\d+)/(\d+), random (\d+)/(\d+)", entry
        )
        if match is None:
            raise ValueError(f"Unrecognized layer coverage entry: {entry}")
        layer, real_changed, real_read, random_changed, random_read = map(
            int, match.groups()
        )
        coverage[layer] = {
            "real_changed": real_changed,
            "real_read": real_read,
            "random_changed": random_changed,
            "random_read": random_read,
        }

    real_text = need(
        r"^86\.3: FAILS\. Known coherent-persona labels by layer are (.+?)\.",
        score,
        "real coherent-persona counts",
    )
    real_counts = {
        int(layer): int(count)
        for layer, count in re.findall(r"L(\d+)=(\d+)", real_text)
    }

    random_text = need(
        r"^RANDOM CHANGED BY VERSION-2 CLASS:\s*(.+)$",
        score,
        "random version-2 class counts",
    )
    random_counts = {}
    for entry in random_text.rstrip(".").split("; "):
        match = re.fullmatch(
            r"L(\d+) changed=(\d+) coherent_persona=(\d+) "
            r"still_qwen_or_assistant=(\d+) not_coherent=(\d+) unclear=(\d+)",
            entry,
        )
        if match is None:
            raise ValueError(f"Unrecognized random class entry: {entry}")
        layer, changed, persona, still_qwen, not_coherent, unclear = map(int, match.groups())
        if changed != persona + still_qwen + not_coherent + unclear:
            raise ValueError(f"Random class totals do not match changed count in layer {layer}")
        random_counts[layer] = persona

    layers = list(range(10, 27))
    if sorted(coverage) != layers or sorted(real_counts) != layers:
        raise ValueError("Expected complete Part 86 coverage for layers 10 through 26")
    if set(random_counts) != set(layers):
        raise ValueError("Expected random counts for layers 10 through 26")
    return layers, coverage, real_counts, random_counts


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval in percentage points."""
    if total == 0:
        return 0.0, 0.0
    rate = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (rate + z2 / (2 * total)) / denominator
    half_width = z * ((rate * (1 - rate) / total + z2 / (4 * total * total)) ** 0.5) / denominator
    return 100 * (center - half_width), 100 * (center + half_width)


def draw_with_matplotlib(
    layers: list[int], real_counts: dict[int, int], random_counts: dict[int, int]
) -> None:
    real_rates = [100 * real_counts[layer] / 9216 for layer in layers]
    random_rates = [100 * random_counts[layer] / 512 for layer in layers]
    real_intervals = [wilson_interval(real_counts[layer], 9216) for layer in layers]
    random_intervals = [wilson_interval(random_counts[layer], 512) for layer in layers]
    real_yerr = [
        [rate - interval[0] for rate, interval in zip(real_rates, real_intervals)],
        [interval[1] - rate for rate, interval in zip(real_rates, real_intervals)],
    ]
    random_yerr = [
        [rate - interval[0] for rate, interval in zip(random_rates, random_intervals)],
        [interval[1] - rate for rate, interval in zip(random_rates, random_intervals)],
    ]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 11})
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    ax.errorbar(
        layers,
        real_rates,
        yerr=real_yerr,
        color="#0072B2",
        fmt="-o",
        markersize=5,
        linewidth=2.2,
        elinewidth=1.1,
        capsize=2,
        label="Real neurons (9,216 per layer)",
    )
    ax.errorbar(
        layers,
        random_rates,
        yerr=random_yerr,
        color="#D55E00",
        fmt="-s",
        markersize=5,
        linewidth=2.2,
        elinewidth=1.1,
        capsize=2,
        label="Random directions (512 per layer)",
    )
    ax.set_title("Registered L22 test: coherent-persona rates")
    ax.set_xlabel("MLP layer")
    ax.set_ylabel("Coherent-persona answers (% of all pushes)")
    ax.set_xticks(layers)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT, dpi=220, facecolor="white")


def draw_with_pillow(
    layers: list[int], real_counts: dict[int, int], random_counts: dict[int, int]
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1892, 1144
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    font_path = next((path for path in font_candidates if Path(path).is_file()), None)

    def font(size: int):
        return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()

    title_font = font(34)
    legend_font = font(24)
    label_font = font(25)
    tick_font = font(22)
    title = "Registered L22 test: coherent-persona rates"
    draw.text((width // 2, 62), title, fill="#202124", font=title_font, anchor="mt")

    left, right, top, bottom = 190, 70, 250, 930
    plot_width, plot_height = width - left - right, bottom - top
    max_rate = 6.0
    real_rates = [100 * real_counts[layer] / 9216 for layer in layers]
    random_rates = [100 * random_counts[layer] / 512 for layer in layers]
    real_intervals = [wilson_interval(real_counts[layer], 9216) for layer in layers]
    random_intervals = [wilson_interval(random_counts[layer], 512) for layer in layers]

    for percent in range(7):
        y = bottom - (percent / max_rate) * plot_height
        draw.line((left, y, width - right, y), fill="#d9d9d9", width=2)
        draw.text((left - 24, y), str(percent), fill="#454545", font=tick_font, anchor="rm")

    def point(index: int, rate: float) -> tuple[float, float]:
        x = left + index * plot_width / (len(layers) - 1)
        y = bottom - (rate / max_rate) * plot_height
        return x, y

    series = (
        (real_rates, real_intervals, "#0072B2", "circle"),
        (random_rates, random_intervals, "#D55E00", "square"),
    )
    for rates, intervals, color, marker in series:
        for index, (lower, upper) in enumerate(intervals):
            x, lower_y = point(index, lower)
            _, upper_y = point(index, upper)
            cap = 7
            draw.line((x, upper_y, x, lower_y), fill=color, width=2)
            draw.line((x - cap, upper_y, x + cap, upper_y), fill=color, width=2)
            draw.line((x - cap, lower_y, x + cap, lower_y), fill=color, width=2)
        coords = [point(index, rate) for index, rate in enumerate(rates)]
        draw.line(coords, fill=color, width=6, joint="curve")
        for x, y in coords:
            if marker == "circle":
                draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline="white", width=2)
            else:
                draw.rectangle((x - 8, y - 8, x + 8, y + 8), fill=color, outline="white", width=2)

    for index, layer in enumerate(layers):
        x, _ = point(index, 0)
        draw.text((x, bottom + 20), str(layer), fill="#454545", font=tick_font, anchor="mt")

    draw.line((left, top, left, bottom), fill="#555555", width=2)
    draw.line((left, bottom, width - right, bottom), fill="#555555", width=2)
    draw.line((left + 3, 190, left + 45, 190), fill="#0072B2", width=6)
    draw.ellipse((left + 18, 181, left + 36, 199), fill="#0072B2", outline="white", width=2)
    draw.text((left + 58, 190), "Real neurons (9,216 per layer)", fill="#202124", font=legend_font, anchor="lm")
    second_x = left + 570
    draw.line((second_x, 190, second_x + 42, 190), fill="#D55E00", width=6)
    draw.rectangle((second_x + 12, 182, second_x + 28, 198), fill="#D55E00", outline="white", width=2)
    draw.text((second_x + 55, 190), "Random directions (512 per layer)", fill="#202124", font=legend_font, anchor="lm")
    draw.text((width // 2, 1042), "MLP layer", fill="#202124", font=label_font, anchor="mm")

    vertical = Image.new("RGBA", (620, 48), (255, 255, 255, 0))
    ImageDraw.Draw(vertical).text(
        (310, 2), "Coherent-persona answers (% of all pushes)",
        fill="#202124", font=label_font, anchor="mt"
    )
    vertical = vertical.rotate(90, expand=True)
    image.paste(vertical, (30, (height - vertical.height) // 2), vertical)
    image.save(OUT, format="PNG", dpi=(220, 220), optimize=True)


def main() -> None:
    layers, coverage, real_counts, random_counts = parse_data()
    if plt is None:
        draw_with_pillow(layers, real_counts, random_counts)
    else:
        draw_with_matplotlib(layers, real_counts, random_counts)
    real_rates = [100 * real_counts[layer] / 9216 for layer in layers]
    random_rates = [100 * random_counts[layer] / 512 for layer in layers]
    real_changed = sum(coverage[layer]["real_changed"] for layer in layers)
    random_changed = sum(coverage[layer]["random_changed"] for layer in layers)
    real_personas = sum(real_counts.values())
    random_personas = sum(random_counts.values())
    payload = {
        "figure": "results/part86/fig86.png",
        "summary": {
            "l22": {
                "real_changed": coverage[22]["real_changed"],
                "random_changed": coverage[22]["random_changed"],
                "real_personas": real_counts[22],
                "random_personas": random_counts[22],
                "real_persona_share_of_changed_percent": round(100 * real_counts[22] / coverage[22]["real_changed"], 3),
                "random_persona_share_of_changed_percent": round(100 * random_counts[22] / coverage[22]["random_changed"], 3),
                "real_wilson_95_percent": [round(value, 3) for value in wilson_interval(real_counts[22], 9216)],
                "random_wilson_95_percent": [round(value, 3) for value in wilson_interval(random_counts[22], 512)],
            },
            "pooled": {
                "real_changed": real_changed,
                "random_changed": random_changed,
                "real_personas": real_personas,
                "random_personas": random_personas,
                "real_persona_share_of_changed_percent": round(100 * real_personas / real_changed, 3),
                "random_persona_share_of_changed_percent": round(100 * random_personas / random_changed, 3),
            },
        },
        "layers": [
            {
                "layer": layer,
                "real_changed": coverage[layer]["real_changed"],
                "random_changed": coverage[layer]["random_changed"],
                "real_personas": real_counts[layer],
                "random_personas": random_counts[layer],
                "real_rate_percent_all_pushes": round(real_rates[index], 6),
                "random_rate_percent_all_pushes": round(random_rates[index], 6),
                "real_wilson_95_percent": [round(value, 6) for value in wilson_interval(real_counts[layer], 9216)],
                "random_wilson_95_percent": [round(value, 6) for value in wilson_interval(random_counts[layer], 512)],
            }
            for index, layer in enumerate(layers)
        ],
    }
    print(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
