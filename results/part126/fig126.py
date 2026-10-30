#!/usr/bin/env python3
"""Recompute Part 126's figure and number ledger from the saved run files."""
from __future__ import annotations

import json
import math
import string
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAYERS = (11, 15)
HEADS = 16
REGIONS = ("role", "template", "body", "question", "options", "other")
COPIES = ("A", "U")
BODY_TOKEN_SEED = 12601
OCCURRENCE_SPLITS = (
    "header",
    "question_header",
    "option_labels",
    "generation_prefix",
    "all_role_words",
)
SPLIT_LABELS = {
    "header": "Header role word",
    "question_header": "Question-header role word",
    "option_labels": "Option-label role words",
    "generation_prefix": "Generation-prefix role word",
    "all_role_words": "All five role-word occurrences",
}
REGION_LABELS = {
    "role": "Header role token",
    "template": "Copied-message template",
    "body": "Copied body",
    "question": "Question and answer format",
    "options": "Option descriptions",
    "other": "Other, including final token",
}
FIGURE_REGION_LABELS = (
    "Header role",
    "Copy template",
    "Copied body",
    "Question + format",
    "Option text",
    "Other incl. final",
    "Self mass (unregistered)",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / den) if den else float("nan")


def head_key(layer: int, head: int) -> str:
    return f"L{layer}h{head}"


def clean_token(token: str) -> str:
    value = token.strip()
    while value and (value[0] in string.punctuation or unicodedata.category(value[0]).startswith("P")):
        value = value[1:]
    while value and (value[-1] in string.punctuation or unicodedata.category(value[-1]).startswith("P")):
        value = value[:-1]
    return value.lower()


def load_data():
    with np.load(HERE / "reads126.npz", allow_pickle=False) as z:
        raw_attention = z["attention"].astype(np.float32)
        region_mass = z["region_mass"].astype(np.float32)
        region_ids = z["region_ids"].astype(np.int64)
        input_ids = z["input_ids"].astype(np.int64)
        lengths = z["lengths"].astype(np.int64)
        layer_ids = tuple(int(x) for x in z["layer_ids"].tolist())
        region_names = tuple(str(x) for x in z["region_names"].tolist())
        row_indexes = z["row_indexes"].astype(np.int64)
    rows = [json.loads(line) for line in (HERE / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    run = read_json(HERE / "run.json")
    part83 = read_json(ROOT / "results/part83/main/results83.json")
    lens = read_json(HERE / "lens/lens126.json")
    with np.load(HERE / "lens/lens126-vectors.npz", allow_pickle=False) as z:
        vectors = {key: z[key].astype(np.float64) for key in z.files if z[key].dtype.kind in "biufc"}

    if layer_ids != LAYERS or region_names != REGIONS:
        raise ValueError(f"Unexpected capture schema: layers={layer_ids}, regions={region_names}")
    if len(rows) != 600 or len(row_indexes) != 600 or row_indexes.tolist() != list(range(600)):
        raise ValueError("Expected the complete 600-row capture")
    if [int(row["row_index"]) for row in rows] != row_indexes.tolist():
        raise ValueError("rows.jsonl does not match the tensor row order")
    if raw_attention.shape[:3] != (600, 2, HEADS) or region_ids.shape[0] != 600:
        raise ValueError(f"Unexpected raw capture shapes: attention={raw_attention.shape}, regions={region_ids.shape}")
    return raw_attention, region_mass, region_ids, input_ids, lengths, rows, run, part83, lens, vectors


def role_word_masses(attention, region_ids, input_ids, lengths, labels, run):
    """Return final-position attention mass for each of the five structural role-word occurrences."""
    tokenizer = AutoTokenizer.from_pretrained(
        run["model_id"], revision=run["revision"], local_files_only=True
    )
    token_ids = np.unique(input_ids)
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    token_text = {int(token_id): str(token) for token_id, token in zip(token_ids, tokens)}

    masses = {name: np.zeros((len(lengths), len(LAYERS), HEADS), dtype=np.float64)
              for name in OCCURRENCE_SPLITS}
    occurrence_counts = {name: np.zeros(len(lengths), dtype=np.int64) for name in OCCURRENCE_SPLITS}
    for row_i, length in enumerate(lengths):
        n = int(length)
        for pos in range(n):
            value = token_text[int(input_ids[row_i, pos])].replace("Ġ", "").replace("Ċ", "").lower()
            if value not in ("assistant", "user"):
                continue
            if pos == 1:
                split = "header"
            elif int(region_ids[row_i, pos]) == 4:
                split = "option_labels"
            elif pos > n - 8:
                split = "generation_prefix"
            else:
                split = "question_header"
            masses[split][row_i] += attention[row_i, :, :, pos]
            occurrence_counts[split][row_i] += 1
            masses["all_role_words"][row_i] += attention[row_i, :, :, pos]
            occurrence_counts["all_role_words"][row_i] += 1

    expected_counts = {"header": 1, "question_header": 1, "option_labels": 2,
                       "generation_prefix": 1, "all_role_words": 5}
    for name, expected in expected_counts.items():
        if not np.all(occurrence_counts[name] == expected):
            raise ValueError(f"Expected {expected} {name} occurrence(s) in every row")
    copy_means = {
        copy: {name: masses[name][labels == copy].mean(axis=0) for name in OCCURRENCE_SPLITS}
        for copy in COPIES
    }
    pooled = {name: masses[name].mean(axis=0).reshape(-1) for name in OCCURRENCE_SPLITS}
    return masses, copy_means, pooled, occurrence_counts


def reconstruct_region_mass(attention, region_ids, lengths):
    reconstructed = np.zeros((len(lengths), len(LAYERS), HEADS, len(REGIONS)), dtype=np.float64)
    for row_i, length in enumerate(lengths):
        ids = region_ids[row_i, :length]
        weights = attention[row_i, :, :, :length].astype(np.float64)
        for region_i in range(len(REGIONS)):
            reconstructed[row_i, :, :, region_i] = weights[..., ids == region_i].sum(axis=-1)
    return reconstructed


def compute():
    attention, region_mass_saved, region_ids, input_ids, lengths, rows, run, part83, lens, vectors = load_data()
    reconstructed = reconstruct_region_mass(attention, region_ids, lengths)
    reconstruction_error = float(np.max(np.abs(reconstructed - region_mass_saved)))
    labels = np.asarray([row["rendering"] for row in rows])
    if set(labels.tolist()) != set(COPIES):
        raise ValueError(f"Expected copies A and U, got {set(labels.tolist())}")
    if any(int(np.count_nonzero(labels == copy)) != 300 for copy in COPIES):
        raise ValueError("Expected 300 prompt rows in each copy")

    # The capture stores per-row region totals in reads126.npz. Average them equally by copy.
    copy_means = {
        copy: region_mass_saved[labels == copy].mean(axis=0).astype(np.float64)
        for copy in COPIES
    }
    pooled = region_mass_saved.mean(axis=0).astype(np.float64)
    region_sum_error = float(np.max(np.abs(region_mass_saved.sum(axis=-1) - 1.0)))
    self_attention = np.asarray([
        attention[row_i, :, :, int(length) - 1].astype(np.float64)
        for row_i, length in enumerate(lengths)
    ])
    self_attention_by_copy = {
        copy: self_attention[labels == copy].mean(axis=0) for copy in COPIES
    }
    occurrence_masses, occurrence_means, occurrence_pooled, occurrence_counts = role_word_masses(
        attention, region_ids, input_ids, lengths, labels, run
    )

    drops = {}
    for layer in LAYERS:
        for head in range(HEADS):
            key = head_key(layer, head)
            row = part83["singles"][key]
            drop = float(row["drop"])
            ci = [float(x) for x in row["drop_ci95"]]
            drops[key] = {
                "drop": drop,
                "absolute_drop": abs(drop),
                "ci95": ci,
                "matters": abs(drop) >= 0.05 and (ci[0] > 0 or ci[1] < 0),
            }
    order = [head_key(layer, head) for layer in LAYERS for head in range(HEADS)]
    pooled_role = np.asarray([pooled[i, h, 0] for i in range(2) for h in range(HEADS)])
    absolute_drops = np.asarray([drops[key]["absolute_drop"] for key in order], dtype=np.float64)
    rho = spearman(pooled_role, absolute_drops)
    rng = np.random.default_rng(12602)
    null = np.asarray([spearman(pooled_role, rng.permutation(absolute_drops)) for _ in range(10000)])
    null_p95 = float(np.quantile(null, 0.95, method="linear"))
    exceedances = int(np.count_nonzero(null >= rho))
    empirical_p = (exceedances + 1) / (len(null) + 1)

    occurrence_stats = {}
    occurrence_rng = np.random.default_rng(12602)
    # Preserve the cold check's split order and seed progression for comparable two-sided p-values.
    shuffle_order = ("header", "generation_prefix", "question_header", "option_labels", "all_role_words")
    for name in shuffle_order:
        split_rho = spearman(occurrence_pooled[name], absolute_drops)
        split_null = np.asarray([
            spearman(occurrence_pooled[name], occurrence_rng.permutation(absolute_drops))
            for _ in range(10000)
        ])
        two_sided_exceedances = int(np.count_nonzero(np.abs(split_null) >= abs(split_rho)))
        occurrence_stats[name] = {
            "rho": split_rho,
            "two_sided_p": (two_sided_exceedances + 1) / (len(split_null) + 1),
            "permutations": len(split_null),
        }

    assistant_role = copy_means["A"][:, :, 0]
    assistant_order = sorted(
        [(head_key(layer, head), float(assistant_role[li, head]), layer, head)
         for li, layer in enumerate(LAYERS) for head in range(HEADS)],
        key=lambda row: (-row[1], row[2], row[3]),
    )
    assistant_rank = next(i + 1 for i, row in enumerate(assistant_order) if row[0] == "L11h5")

    role_readout = lens["readouts"]["L11h5"]["A"]["role_token"]
    top_tokens = role_readout["top20_written_tokens"]
    first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    second_person = {"you", "your", "yours"}
    token_matches = []
    for token in top_tokens:
        normalized = clean_token(str(token))
        family = "first-person" if normalized in first_person else (
            "second-person" if normalized in second_person else ("assistant-role" if normalized == "assistant" else None)
        )
        if family:
            token_matches.append({"token": token, "normalized": normalized, "family": family})

    cosine_by_head = {}
    role_writes = vectors["role_writes"]
    if role_writes.shape != (2, HEADS, 2, 2560):
        raise ValueError(f"Unexpected role write shape: {role_writes.shape}")
    for layer_i, layer in enumerate(LAYERS):
        for head in range(HEADS):
            a = role_writes[layer_i, head, 0]
            u = role_writes[layer_i, head, 1]
            denominator = float(np.linalg.norm(a) * np.linalg.norm(u))
            cosine_by_head[head_key(layer, head)] = float(np.dot(a, u) / denominator) if denominator else float("nan")
    mattering = [key for key in order if drops[key]["matters"]]
    below = [key for key in mattering if cosine_by_head[key] < 0.90]

    overlaps = {}
    for key, copies in lens["readouts"].items():
        overlaps[key] = {}
        for copy in COPIES:
            item = copies[copy]
            role_set = set(item["role_token"]["top20_written_tokens"])
            body_set = set(item["random_body_token_control"]["top20_written_tokens"])
            overlaps[key][copy] = len(role_set & body_set)
    overlap_values = np.asarray([overlaps[key][copy] for key in overlaps for copy in COPIES], dtype=np.float64)
    zero_overlap_pairs = int(np.count_nonzero(overlap_values == 0))

    head_order_by_drop = sorted(order, key=lambda key: (-drops[key]["absolute_drop"], key))
    region_leaders = {
        copy: {
            head_key(layer, head): {
                "region": REGIONS[int(np.argmax(copy_means[copy][li, head]))],
                "mass": float(np.max(copy_means[copy][li, head])),
            }
            for li, layer in enumerate(LAYERS) for head in range(HEADS)
        }
        for copy in COPIES
    }
    return {
        "attention": attention,
        "region_mass": region_mass_saved,
        "reconstruction_error": reconstruction_error,
        "region_sum_error": region_sum_error,
        "self_attention_by_copy": self_attention_by_copy,
        "occurrence_masses": occurrence_masses,
        "occurrence_means": occurrence_means,
        "occurrence_pooled": occurrence_pooled,
        "occurrence_counts": occurrence_counts,
        "occurrence_stats": occurrence_stats,
        "rows": rows,
        "run": run,
        "part83": part83,
        "lens": lens,
        "vectors": vectors,
        "copy_means": copy_means,
        "pooled": pooled,
        "drops": drops,
        "head_order": order,
        "head_order_by_drop": head_order_by_drop,
        "rho": rho,
        "null_p95": null_p95,
        "empirical_p": empirical_p,
        "exceedances": exceedances,
        "permutations": len(null),
        "assistant_order": assistant_order,
        "assistant_rank": assistant_rank,
        "token_matches": token_matches,
        "cosine_by_head": cosine_by_head,
        "mattering": mattering,
        "below": below,
        "overlaps": overlaps,
        "overlap_values": overlap_values,
        "zero_overlap_pairs": zero_overlap_pairs,
        "region_leaders": region_leaders,
    }


def make_figure(data):
    ordered = data["head_order_by_drop"]
    labels = [f"L{key[1:key.index('h')]} h{key.split('h')[1]}" for key in ordered]
    values = {}
    for copy in COPIES:
        registered = np.asarray([
            [data["copy_means"][copy][li, int(key.split("h")[1]), ri] for ri in range(len(REGIONS))]
            for key in ordered
            for li, layer in enumerate(LAYERS) if key.startswith(f"L{layer}h")
        ], dtype=np.float64)
        self_column = np.asarray([
            data["self_attention_by_copy"][copy][LAYERS.index(int(key.split("h")[0][1:])),
                                                   int(key.split("h")[1])]
            for key in ordered
        ], dtype=np.float64)[:, None]
        values[copy] = np.concatenate([registered, self_column], axis=1)
    # The comprehension above preserves the supplied drop order while selecting the layer slot.
    fig = plt.figure(figsize=(16, 17), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(3.8, 1.6), hspace=0.24, wspace=0.27)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    image = None
    for axis, copy in zip(axes, COPIES):
        image = axis.imshow(values[copy], aspect="auto", interpolation="nearest", cmap="viridis", norm=Normalize(0, 1))
        axis.set_title(f"{copy} copy")
        column_labels = FIGURE_REGION_LABELS
        axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(ordered)), labels if copy == "A" else [])
        axis.tick_params(axis="y", labelsize=7)
        axis.tick_params(axis="x", labelsize=8)
        axis.set_xlabel("Mean attention mass by region")
        cmap = plt.get_cmap("viridis")
        for ri in range(len(column_labels)):
            for hi in range(len(ordered)):
                mass = values[copy][hi, ri]
                red, green, blue = cmap(Normalize(0, 1)(mass))[:3]
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "black" if luminance > 0.52 else "white"
                axis.text(ri, hi, f"{mass * 100:.2f}", ha="center", va="center", fontsize=5.2, color=color)
    axes[0].set_ylabel("Heads, sorted by Part 83 absolute drop")
    scatter = fig.add_subplot(grid[1, :])
    x = np.asarray([data["pooled"][i, h, 0] * 100 for i in range(2) for h in range(HEADS)])
    y = np.asarray([data["drops"][key]["absolute_drop"] for key in data["head_order"]])
    scatter.scatter(x, y, s=28, color="#276FBF", edgecolor="white", linewidth=0.4, alpha=0.9)
    scatter.set_xlabel("Mean header-token attention mass pooled across copies (%)")
    scatter.set_ylabel("Part 83 absolute drop")
    scatter.set_title(f"Header-token mass vs absolute drop: Spearman rho={data['rho']:.3f}, empirical p={data['empirical_p']:.3f}")
    scatter.grid(True, alpha=0.25)
    fig.suptitle("Part 126: final-position attention mass to the copied-message header token", fontsize=14)
    fig.subplots_adjust(top=0.95, bottom=0.085, left=0.12, right=0.92)
    color_axis = fig.add_axes([0.94, 0.58, 0.014, 0.33])
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("Attention mass (0 to 1)")
    place_scatter_labels(scatter, x, y, data["head_order"])
    fig.savefig(HERE / "fig126.png", dpi=200, facecolor="white")
    plt.close(fig)


def place_scatter_labels(axis, x, y, keys):
    """Place every scatter label beside its point without overlapping labels or markers."""
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    points = axis.transData.transform(np.column_stack([x, y]))
    axes_box = axis.get_window_extent(renderer).expanded(0.995, 0.98)
    occupied = []
    directions = ((1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0))
    for point_i, key in enumerate(keys):
        layer = int(key.split("h")[0][1:])
        head = int(key.split("h")[1])
        label = f"{layer}.{head}"
        candidates = [(dx * radius, dy * radius) for radius in (7, 11, 15, 20, 26, 32)
                      for dx, dy in directions]
        accepted = None
        for dx, dy in candidates:
            annotation = axis.annotate(
                label,
                (x[point_i], y[point_i]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                fontsize=6,
                arrowprops=(
                    {"arrowstyle": "-", "color": "#4a4a4a", "lw": 0.35, "shrinkA": 1, "shrinkB": 2}
                    if max(abs(dx), abs(dy)) >= 15 else None
                ),
            )
            figure.canvas.draw()
            box = annotation.get_window_extent(renderer).expanded(1.04, 1.12)
            label_collision = any(box.overlaps(other) for other in occupied)
            point_collision = any(box.contains(float(px), float(py)) for j, (px, py) in enumerate(points) if j != point_i)
            inside_axes = (axes_box.contains(box.x0, box.y0) and axes_box.contains(box.x1, box.y1))
            if inside_axes and not label_collision and not point_collision:
                accepted = (annotation, box)
                break
            annotation.remove()
        if accepted is None:
            # Keep the label visible if an unusually dense cluster exhausts the candidate positions.
            annotation = axis.annotate(label, (x[point_i], y[point_i]), xytext=(34, 34),
                                       textcoords="offset points", fontsize=6, ha="left", va="bottom",
                                       arrowprops={"arrowstyle": "-", "color": "#4a4a4a", "lw": 0.35})
            figure.canvas.draw()
            accepted = (annotation, annotation.get_window_extent(renderer).expanded(1.04, 1.12))
        occupied.append(accepted[1])


def fmt_mass(mass: float) -> str:
    return f"{mass * 100:.3f}%"


def display_token(token) -> str:
    value = str(token)
    return value.replace("`", "grave-accent symbol").replace("*", r"\*").replace("_", r"\_")


def token_short(tokens, n=5):
    return ", ".join(display_token(token) for token in tokens[:n])


def write_numbers(data):
    run = data["run"]
    gates = run["gates"]
    lens_gate = data["lens"]["gates"]["lens_shape"]
    lines = [
        "# Part 126 numbers",
        "",
        "This file is generated by [fig126.py](fig126.py). The script reads the saved capture, row metadata, Part 83 results, and lens readouts. It does not read score126.json. Regional masses are averaged from reads126.npz, checked by rebuilding them from its saved final-position attention and token-region arrays. The lens token rankings are the saved output of the lens calculation, and their source is stated below.",
        "",
        "## Scope and capture",
        "",
        "| Item | Value | Source |",
        "| --- | --- | --- |",
        f"| Model | {run['model_id']} at revision {run['revision']} | [run.json](run.json) |",
        f"| Device and dtype | {run['device']}, {run['dtype']} | [run.json](run.json) |",
        f"| Bodies and prompt rows | {run['body_count']} bodies; {len(data['rows'])} rows; {sum(row['rendering'] == 'A' for row in data['rows'])} A rows; {sum(row['rendering'] == 'U' for row in data['rows'])} U rows; {len(run['orders'])} option orders per copy | [run.json](run.json), [rows.jsonl](rows.jsonl) |",
        f"| Heads and regions | {len(data['head_order'])} heads in layers 11 and 15; {len(REGIONS)} registered regions plus one unregistered self-attention split | [reads126.npz](reads126.npz) |",
        f"| Loaded layers | Layers 0 through 15 only; 16 of {data['lens']['gates']['lens_shape']['snapshot_layers']} decoder layers | [PART126-DESIGN.md](../../PART126-DESIGN.md), [lens/lens126.json](lens/lens126.json) |",
        f"| Lens fit and body-token control | WikiText fit; one uniformly selected copied-body token per prompt; seed {BODY_TOKEN_SEED} plus row index | [lens/lens126.json](lens/lens126.json), [PART126-DESIGN.md](../../PART126-DESIGN.md), [reads126.py](../../part126/reads126.py) |",
        f"| Rebuilt regional masses | maximum difference from saved regional masses = {data['reconstruction_error']:.8f}; maximum saved regional sum error from 1 = {data['region_sum_error']:.8f} | [reads126.npz](reads126.npz), [rows.jsonl](rows.jsonl) |",
        "",
        "## Gates",
        "",
        "All four gates passed on the final scored run. The numeric gate values below are the recorded run measurements; the capture and lens shape fields are also checked by this script.",
        "",
        "| Gate | Final result | Value and source |",
        "| --- | --- | --- |",
        f"| Recomputed head output | {'passed' if gates['recomputation']['max_abs_difference'] <= gates['recomputation']['tolerance'] and gates['recomputation']['region_mass_sum_max_abs_difference'] <= gates['recomputation']['region_mass_tolerance'] else 'failed'} | max error {gates['recomputation']['max_abs_difference']:.6f} vs {gates['recomputation']['tolerance']:.2f}; recorded region-mass sum error {gates['recomputation']['region_mass_sum_max_abs_difference']:.11f} vs {gates['recomputation']['region_mass_tolerance']:.2f}; saved float16 masses recompute to {data['region_sum_error']:.8f}; [run.json](run.json), [reads126.npz](reads126.npz) |",
        f"| No-op hook | {'passed' if gates['zero_effect']['bitwise_equal'] and gates['zero_effect']['max_abs_difference'] == 0 else 'failed'} | bitwise equal = {str(gates['zero_effect']['bitwise_equal']).lower()}; max logit difference = {gates['zero_effect']['max_abs_difference']:.1f}; [run.json](run.json) |",
        f"| Copy-style positive control | {'passed' if gates['positive_control']['max_rate_over_uniform'] >= 3 else 'failed'} | {gates['positive_control']['max_rate_over_uniform']:.3f} times uniform for {gates['positive_control']['head']}; threshold at least 3.0; [run.json](run.json) |",
        f"| J-lens shape | {'passed' if lens_gate['passed'] else 'failed'} | {lens_gate['lens_shape'][0]} matrices of {lens_gate['lens_shape'][1]} by {lens_gate['lens_shape'][2]}; hidden size {lens_gate['snapshot_hidden_size']}; selected layers 11 and 15; [lens/lens126.json](lens/lens126.json) |",
        "",
        "## Registered predictions",
        "",
        "| Prediction | Recomputed evidence | Verdict | Source |",
        "| --- | --- | --- | --- |",
        f"| 126.1 | Spearman rho {data['rho']:.6f}; shuffled-null 95th percentile {data['null_p95']:.6f}; empirical p {data['empirical_p']:.6f} ({data['exceedances']} of {data['permutations']} null values at least as large as rho); threshold rho at least 0.40 and above null 95th. | {'held' if data['rho'] >= 0.40 and data['rho'] > data['null_p95'] else 'failed'} | [reads126.npz](reads126.npz), [rows.jsonl](rows.jsonl), [results83.json](../part83/main/results83.json); 10,000 shuffles, seed 12602 |",
        f"| 126.2 | L11 h5 ranks {data['assistant_rank']} of 32 in the Assistant-copy role-token mass order; mass {fmt_mass(data['assistant_order'][data['assistant_rank'] - 1][1])}; top five: {', '.join(f'{key} ({fmt_mass(mass)})' for key, mass, _, _ in data['assistant_order'][:5])}. | {'held' if data['assistant_rank'] <= 5 else 'failed'} | [reads126.npz](reads126.npz), [rows.jsonl](rows.jsonl) |",
        f"| 126.3 | L11 h5 Assistant-copy role-write top 20: {'; '.join(display_token(x) for x in data['lens']['readouts']['L11h5']['A']['role_token']['top20_written_tokens'])}. Registered token matches: {len(data['token_matches'])}. | {'held' if data['token_matches'] else 'failed'} | [lens/lens126.json](lens/lens126.json), [lens/lens126-vectors.npz](lens/lens126-vectors.npz) |",
        f"| 126.4 | {len(data['below'])} of {len(data['mattering'])} Part 83 mattering heads have Assistant/User role-write cosine below 0.90; threshold at least 10 of 20. | {'held' if len(data['below']) >= 10 else 'failed'} | [results83.json](../part83/main/results83.json), [lens/lens126-vectors.npz](lens/lens126-vectors.npz) |",
        "",
        "## Part 83 absolute-drop ranking",
        "",
        "| Rank | Head | Absolute drop | Signed drop | 95% interval |",
        "| ---: | --- | ---: | ---: | --- |",
    ]
    for rank, key in enumerate(data["head_order_by_drop"], 1):
        drop = data["drops"][key]
        lines.append(f"| {rank} | {key} | {drop['absolute_drop']:.6f} | {drop['drop']:.6f} | {drop['ci95'][0]:.6f} to {drop['ci95'][1]:.6f} |")
    lines.extend([
        "",
        "## Regional attention by head",
        "",
        "Values are mean fractions of final-position attention, shown as percentages. Each copy averages 300 prompt rows. Rows use Part 83 absolute-drop order, from largest to smallest. The six named regions are the registered, mutually exclusive regions. Self-attention is the query position attending to itself; it is an unregistered split inside the registered Other region, so it overlaps that region and is not added to the six-region total.",
        "",
    ])
    for copy in COPIES:
        lines.extend([
            f"### {copy} copy",
            "",
            "| Head | Header role token | Template | Copied body | Question and answer format | Option descriptions | Other, incl. final token | Self-attention, unregistered | Largest registered region |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for key in data["head_order_by_drop"]:
            layer = int(key.split("h")[0][1:])
            head = int(key.split("h")[1])
            layer_i = LAYERS.index(layer)
            masses = data["copy_means"][copy][layer_i, head]
            leader = data["region_leaders"][copy][key]
            self_mass = data["self_attention_by_copy"][copy][layer_i, head]
            lines.append(
                f"| {key} | " + " | ".join(fmt_mass(float(masses[i])) for i in range(len(REGIONS)))
                + f" | {fmt_mass(float(self_mass))} | {REGION_LABELS[leader['region']]} ({fmt_mass(leader['mass'])}) |"
            )
        lines.append("")

    lines.extend([
        "## Role-word occurrence splits",
        "",
        "Every saved prompt row contains five exact structural role-word occurrences: one copied-message header word, one question-message header word, two option-label words, and one generation-prefix word. The all-five column sums those four categories. The occurrence-split analysis is unregistered except that header mass is the registered predictor. The two-sided p values in this table are unregistered summaries and are uncorrected for multiplicity.",
        "",
        "| Split | Spearman rho with Part 83 absolute drop | Two-sided permutation p |",
        "| --- | ---: | ---: |",
    ])
    for name in OCCURRENCE_SPLITS:
        stat = data["occurrence_stats"][name]
        lines.append(f"| {SPLIT_LABELS[name]} | {stat['rho']:+.4f} | {stat['two_sided_p']:.4f} |")
    lines.extend([
        "",
        "For each head, the following values are mean attention masses over 300 prompts per copy, shown as percentages. The all-five value is not the sum of any unrelated region: it sums the five role-word positions in each prompt.",
        "",
    ])
    for copy in COPIES:
        lines.extend([
            f"### {copy} copy: role-word occurrence attention mass",
            "",
            "| Head | Header | Question header | Option labels | Generation prefix | All five occurrences |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for key in data["head_order_by_drop"]:
            layer = int(key.split("h")[0][1:])
            head = int(key.split("h")[1])
            layer_i = LAYERS.index(layer)
            cells = [fmt_mass(float(data["occurrence_means"][copy][name][layer_i, head]))
                     for name in OCCURRENCE_SPLITS]
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend([
        "## Lens leads for three heads",
        "",
        "These are lens rankings, not tested claims. For each copy, the table gives the top five written and suppressed tokens for the role write, the largest-mass registered-region write, and the same-head random-body-token control. The writes and masses come from [lens/lens126.json](lens/lens126.json); the corresponding vectors are in [lens/lens126-vectors.npz](lens/lens126-vectors.npz).",
        "",
        "| Head | Copy | Role-token write: top / suppressed | Largest-mass region write: top / suppressed | Random body-token control: top / suppressed |",
        "| --- | --- | --- | --- | --- |",
    ])
    for key in ("L11h5", "L11h7", "L15h1"):
        for copy in COPIES:
            reads = data["lens"]["readouts"][key][copy]
            role = reads["role_token"]
            region = reads["most_attended_region"]
            body = reads["random_body_token_control"]
            layer = int(key.split("h")[0][1:])
            head = int(key.split("h")[1])
            mass = float(data["copy_means"][copy][LAYERS.index(layer), head, 0])
            lines.append(
                f"| {key} ({fmt_mass(mass)} role mass) | {copy} | "
                f"{token_short(role['top20_written_tokens'])} / {token_short(role['top20_suppressed_tokens'])} | "
                f"{REGION_LABELS[region['name']]} at {fmt_mass(region['mean_attention_mass'])}: "
                f"{token_short(region['top20_written_tokens'])} / {token_short(region['top20_suppressed_tokens'])} | "
                f"{token_short(body['top20_written_tokens'])} / {token_short(body['top20_suppressed_tokens'])} |"
            )
    lines.extend([
        "",
        "## Role-write cosine for Part 83 mattering heads",
        "",
        "| Head | Role-write cosine, A vs U | Below 0.90? |",
        "| --- | ---: | --- |",
    ])
    for key in data["mattering"]:
        lines.append(f"| {key} | {data['cosine_by_head'][key]:.6f} | {'yes' if key in data['below'] else 'no'} |")
    lines.extend([
        "",
        "## Amendment and gate history",
        "",
        "Amendment 2 says the run loaded decoder layers 0 through 15, or 16 of 32, because loading all layers crossed the memory guard. Layers 11 and 15 are computed from the input and earlier layers, so higher layers cannot change their capture. No answer generation is used. Amendment 3 records that the scorer and lens script first failed gate 4 because of their own shape and tensor-layout defects. The defects were fixed before a prediction was read, after seeing the other three gates, four lines marked not read, and three rows of the attention table. No prediction or threshold changed. Source: [PART126-DESIGN.md](../../PART126-DESIGN.md).",
        "",
        "## Sources",
        "",
        "- Capture and per-prompt metadata: [reads126.npz](reads126.npz), [rows.jsonl](rows.jsonl), and [run.json](run.json).",
        "- Causal drops and confidence intervals: [Part 83 results](../part83/main/results83.json).",
        "- Lens token readouts and mean writes: [lens/lens126.json](lens/lens126.json) and [lens/lens126-vectors.npz](lens/lens126-vectors.npz).",
        "- The registered scoring rules and code: [PART126-DESIGN.md](../../PART126-DESIGN.md) and [score126.py](../../part126/score126.py).",
        "",
    ])
    (HERE / "NUMBERS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    data = compute()
    make_figure(data)
    write_numbers(data)
    print(
        f"wrote fig126.png and NUMBERS.md; rho={data['rho']:.6f}; p={data['empirical_p']:.6f}; "
        f"L11h5 assistant role rank={data['assistant_rank']}; {len(data['below'])}/{len(data['mattering'])} cosines < .90; "
        f"raw-mass reconstruction max error={data['reconstruction_error']:.8f}"
    )


if __name__ == "__main__":
    main()
