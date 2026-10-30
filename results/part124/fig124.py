#!/usr/bin/env python3
"""Recompute Part 124 write-up values from the frozen run files and draw its figure."""
from __future__ import annotations

import json
import gzip
import math
import os
import random
import re
import sys
import statistics
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-part124")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PART_CODE = REPO / "part124"
sys.path.insert(0, str(PART_CODE))

import matplotlib.pyplot as plt
import numpy as np

import score124
import passages124

FRAMES = ("T", "F", "C")
KINDS = ("concept", "slot")
FRAME_COLORS = {"T": "#276FBF", "F": "#F28E2B", "C": "#2A9D70"}
KIND_COLORS = {"concept": "#276FBF", "slot": "#E07A24"}
FRAME_MARKERS = {"T": "o", "F": "s", "C": "^"}
CONTENT_STOPWORDS = {
    "a", "an", "the", "of", "your", "from", "in", "on", "to", "and",
    "or", "for", "with", "my", "me", "i", "we", "you", "at", "by",
    "as", "is", "am", "it", "its", "our", "this", "that", "into",
    "under", "over", "before", "after", "up", "down", "about", "off",
    "out", "than", "then", "but", "if", "not", "so", "too", "very",
    "can", "will", "would", "who", "what", "which", "where", "when",
    "why", "how",
}
PART122_COLD_UNIT_IDS = (
    "p92-00", "p92-01", "p92-07", "p92-09", "p92-28", "p92-35", "p92-37",
)
PREDICTION_TEXT = {
    "124.1": "In T, at least 9 of 39 units read their persona, and the count is above the 95th percentile of the permutation null.",
    "124.2": "In T, the concept read rate is at least 15 percentage points above the slot read rate.",
    "124.3": "In T, at most 4 of 21 slot units read their persona.",
    "124.4": "Of the units that read in T, at least half also read in C.",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_activation_jsonl(path: Path) -> tuple[list[dict], int]:
    """Read plain or gzipped activations, accepting only a torn final line."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        lines = stream.readlines()
    uncompressed_bytes = sum(len(line.encode("utf-8")) for line in lines)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                print(f"skipping torn final JSONL line: {path}:{index + 1}",
                      file=sys.stderr, flush=True)
                break
            raise ValueError(f"invalid activation JSONL at {path}:{index + 1}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"activation row at {path}:{index + 1} is not an object")
        rows.append(row)
    return rows, uncompressed_bytes


def load_and_recompute():
    manifest = read_json(PART_CODE / "passages124.json")
    read_set = read_json(HERE / "read_set.json")
    run_meta = read_json(HERE / "run_meta.json")
    activation_path = HERE / "activations.jsonl"
    if not activation_path.exists():
        activation_path = HERE / "activations.jsonl.gz"
    if not activation_path.exists():
        raise FileNotFoundError(f"missing {HERE / 'activations.jsonl'} and its .gz copy")
    activation_rows, activation_bytes = read_activation_jsonl(activation_path)
    result = score124.score_results(manifest, read_set, run_meta, activation_rows)
    reference = read_json(HERE / "score124.json")
    check_reference(result, reference)
    return (manifest, read_set, run_meta, activation_rows, result, reference,
            activation_bytes, activation_path)


def check_reference(result: dict, reference: dict) -> None:
    """Fail closed if a fresh raw-file score diverges from the saved score."""
    assert result["all_gates_passed"] == reference["all_gates_passed"]
    assert result["persistence_T_to_C"] == reference["persistence_T_to_C"]
    assert result["permutations"]["all_units"]["null_counts"] == \
        reference["permutations"]["all_units"]["null_counts"]
    for kind in KINDS:
        assert result["kind_counts"][kind]["reads"] == reference["kind_counts"][kind]["reads"]
        assert result["kind_counts"][kind]["permutation"]["null_counts"] == \
            reference["kind_counts"][kind]["permutation"]["null_counts"]
    for frame in FRAMES:
        assert result["instrument"]["by_frame"][frame]["repeat"] == \
            reference["instrument"]["by_frame"][frame]["repeat"]
        assert result["gates"]["random_channels"]["by_frame"][frame]["reads"] == \
            reference["gates"]["random_channels"]["by_frame"][frame]["reads"]
        for unit_id, row in result["focal"][frame].items():
            saved = reference["focal"][frame][unit_id]
            assert (row["reads"], row["own_rank"], row["top3"]) == \
                (saved["reads"], saved["own_rank"], saved["top3"])


def fisher_two_sided(a: int, n_a: int, b: int, n_b: int) -> float:
    """Two-sided Fisher exact p, summing tables no more likely than observed."""
    total_n = n_a + n_b
    total_success = a + b
    denominator = math.comb(total_n, n_a)
    low = max(0, total_success - n_b)
    high = min(n_a, total_success)
    probabilities = {
        x: math.comb(total_success, x) *
           math.comb(total_n - total_success, n_a - x) / denominator
        for x in range(low, high + 1)
    }
    observed = probabilities[a]
    return min(1.0, sum(p for p in probabilities.values() if p <= observed + 1e-15))


def cmh_two_sided(ids: list[str], labels: dict[str, str], units: dict[str, dict],
                  focal: dict) -> float:
    """Length-stratified Cochran-Mantel-Haenszel association test."""
    score = 0.0
    variance = 0.0
    lengths = sorted({int(units[unit_id]["slot_token_count"]) for unit_id in ids})
    for length in lengths:
        members = [unit_id for unit_id in ids
                   if int(units[unit_id]["slot_token_count"]) == length]
        concept = [unit_id for unit_id in members if labels[unit_id] == "concept"]
        slot = [unit_id for unit_id in members if labels[unit_id] == "slot"]
        a = sum(bool(focal["T"][unit_id]["reads"]) for unit_id in concept)
        b = len(concept) - a
        c = sum(bool(focal["T"][unit_id]["reads"]) for unit_id in slot)
        d = len(slot) - c
        n = a + b + c + d
        if n <= 1:
            continue
        score += a - ((a + b) * (a + c) / n)
        variance += ((a + b) * (c + d) * (a + c) * (b + d) /
                     (n * n * (n - 1)))
    if variance == 0:
        return 1.0
    statistic = score * score / variance
    return math.erfc(math.sqrt(statistic / 2))


def content_tokens(phrase: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", phrase.casefold())
            if word not in CONTENT_STOPWORDS}


def top_three(phrase_scores: dict[str, float]) -> list[str]:
    return sorted(phrase_scores,
                  key=lambda phrase: (-float(phrase_scores[phrase]),
                                      phrase.casefold(), phrase))[:3]


def cold_check_fix_count() -> int:
    text = (HERE / "COLD-CHECK-r1.md").read_text(encoding="utf-8")
    marker = "Fixes, in priority order:"
    if marker not in text:
        raise ValueError("cold check has no numbered priority list")
    priority_text = text.split(marker, 1)[1]
    return sum(bool(re.match(r"^\d+\.\s", line))
               for line in priority_text.splitlines())


def nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def one_token_null(units: list[dict], focal: dict, count: int, seed: int) -> list[int]:
    """Repeat the registered within-stratum shuffle and retain its one-token counts."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for unit in units:
        groups[int(unit["length_stratum"])].append(unit)
    rng = random.Random(seed)
    counts = []
    for _ in range(count):
        assigned = {}
        for members in groups.values():
            targets = [str(unit["persona"]) for unit in members]
            rng.shuffle(targets)
            assigned.update((str(unit["id"]), phrase)
                            for unit, phrase in zip(members, targets))
        counts.append(sum(
            assigned[str(unit["id"])] in focal["T"][str(unit["id"])]["top3"]
            for unit in units if int(unit["slot_token_count"]) == 1
        ))
    return counts


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def ratio(n: int, d: int) -> str:
    return f"{n}/{d}"


def number_source(raw: str = "activations.jsonl; read_set.json; run_meta.json; frozen passages124.json") -> str:
    return raw


def build_summaries(manifest: dict, read_set: dict, run_meta: dict,
                    result: dict, reference: dict, part92b: dict,
                    part92: list[dict], activation_rows: list[dict],
                    activation_bytes: int) -> dict:
    units = manifest["phrases"]
    by_id = {str(unit["id"]): unit for unit in units}
    focal = result["focal"]
    counts = {}
    random_by_kind = {}
    for frame in FRAMES:
        counts[frame] = {}
        for kind in KINDS:
            members = [unit for unit in units if unit["kind"] == kind]
            reads = sum(bool(focal[frame][str(unit["id"])]["reads"]) for unit in members)
            counts[frame][kind] = {"reads": reads, "units": len(members),
                                   "rate": reads / len(members)}
        controls = defaultdict(lambda: {"reads": 0, "observed": 0})
        for control in result["random_controls"][frame]:
            kind = by_id[str(control["unit_id"])]["kind"]
            controls[kind]["reads"] += int(bool(control["reads"]))
            controls[kind]["observed"] += 1
        random_by_kind[frame] = {
            kind: {**controls[kind],
                   "rate": controls[kind]["reads"] / controls[kind]["observed"]}
            for kind in KINDS
        }

    p92b_ids = {
        str(unit["id"]): unit
        for unit in part92b["kind_comparison"]["units"]
    }
    agreed_ids = {unit_id for unit_id, row in p92b_ids.items() if row["same"] is True}
    def label_counts(frame: str, group: str) -> dict:
        if group == "agreed":
            classifier = lambda unit_id: p92b_ids[unit_id]["part92_kind"]
            include = agreed_ids
        elif group == "deepseek":
            classifier = lambda unit_id: p92b_ids[unit_id]["ds_kind"]
            include = {u for u, row in p92b_ids.items() if row["ds_kind"] in KINDS}
        else:
            raise ValueError(group)
        output = {}
        for kind in KINDS:
            ids = [unit_id for unit_id in include if classifier(unit_id) == kind]
            reads = sum(bool(focal[frame][unit_id]["reads"]) for unit_id in ids)
            output[kind] = {"reads": reads, "units": len(ids),
                            "rate": reads / len(ids) if ids else None}
        return output
    alternate_counts = {
        group: {frame: label_counts(frame, group) for frame in FRAMES}
        for group in ("agreed", "deepseek")
    }

    label_maps = {
        "Haiku labels": {unit_id: unit["kind"] for unit_id, unit in by_id.items()},
        "DeepSeek labels": {
            unit_id: row["ds_kind"] for unit_id, row in p92b_ids.items()
            if row["ds_kind"] in KINDS
        },
        "29 agreed units": {
            unit_id: row["part92_kind"] for unit_id, row in p92b_ids.items()
            if row["same"] is True
        },
    }
    label_tests = {}
    for label_name, labels in label_maps.items():
        concept_ids = [unit_id for unit_id, kind in labels.items() if kind == "concept"]
        slot_ids = [unit_id for unit_id, kind in labels.items() if kind == "slot"]
        concept_reads = sum(bool(focal["T"][unit_id]["reads"]) for unit_id in concept_ids)
        slot_reads = sum(bool(focal["T"][unit_id]["reads"]) for unit_id in slot_ids)
        label_tests[label_name] = {
            "concept_reads": concept_reads,
            "concept_units": len(concept_ids),
            "slot_reads": slot_reads,
            "slot_units": len(slot_ids),
            "fisher_p": fisher_two_sided(
                concept_reads, len(concept_ids), slot_reads, len(slot_ids)),
            "length_adjusted_p": cmh_two_sided(
                list(labels), labels, by_id, focal),
            "unit_ids": list(labels),
        }

    part92_by_id = {str(row["id"]): row for row in part92}
    if any(unit_id not in part92_by_id for unit_id in by_id):
        raise ValueError("Part 92 source answers are missing a Part 124 unit")
    phrase_origin = {}
    for unit_id, unit in by_id.items():
        phrase = str(unit["persona"])
        answer = str(part92_by_id[unit_id].get("screen_answer", ""))
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(phrase) +
                             r"(?![A-Za-z0-9])", re.IGNORECASE)
        phrase_origin[unit_id] = {
            "verbatim": bool(pattern.search(answer)),
            "content_words_present": content_tokens(phrase) <=
                                     set(re.findall(r"[a-z0-9]+", answer.casefold())),
            "screen_answer": answer,
        }

    phrases = [str(unit["persona"]) for unit in units]
    token_sets = {
        phrase: set(re.findall(r"[a-z0-9]+", phrase.casefold()))
        for phrase in phrases
    }
    phrase_overlaps = {
        phrase: {
            other for other in phrases
            if other != phrase and token_sets[phrase] & token_sets[other]
        }
        for phrase in phrases
    }
    content_token_sets = {phrase: content_tokens(phrase) for phrase in phrases}
    content_phrase_overlaps = {
        phrase: {
            other for other in phrases
            if other != phrase and content_token_sets[phrase] & content_token_sets[other]
        }
        for phrase in phrases
    }
    shared_reads = {}
    shared_reads_content = {}
    for frame in FRAMES:
        rows = []
        content_rows = []
        for unit_id, row in focal[frame].items():
            if not row["reads"]:
                continue
            peers = sorted(phrase_overlaps[row["persona"]])
            top3_peers = [phrase for phrase in row["top3"]
                          if phrase != row["persona"] and phrase in peers]
            rows.append({
                "unit_id": unit_id,
                "persona": row["persona"],
                "overlap_phrases": peers,
                "top3_overlap_phrases": top3_peers,
            })
            content_peers = sorted(content_phrase_overlaps[row["persona"]])
            content_top3_peers = [phrase for phrase in row["top3"]
                                  if phrase != row["persona"] and phrase in content_peers]
            content_rows.append({
                "unit_id": unit_id,
                "persona": row["persona"],
                "overlap_phrases": content_peers,
                "top3_overlap_phrases": content_top3_peers,
            })
        shared_reads[frame] = rows
        shared_reads_content[frame] = content_rows

    template_robustness = {}
    for unit_id, row in focal["T"].items():
        if not row["reads"]:
            continue
        template_hits = 0
        for template_index in range(8):
            template_scores = {
                phrase: values[template_index]
                for phrase, values in row["template_maxima"].items()
            }
            template_hits += int(row["persona"] in top_three(template_scores))
        ordered_mean = sorted(
            row["phrase_scores"],
            key=lambda phrase: (-float(row["phrase_scores"][phrase]),
                                phrase.casefold(), phrase),
        )
        fourth = ordered_mean[3]
        template_robustness[unit_id] = {
            "templates_top3": template_hits,
            "margin_over_fourth": float(row["phrase_scores"][row["persona"]] -
                                         row["phrase_scores"][fourth]),
            "fourth_phrase": fourth,
        }

    diagnostics = {}
    for frame in FRAMES:
        diagnostics[frame] = {}
        for kind in KINDS:
            rows = [row for row in focal[frame].values() if row["kind"] == kind]
            slot_means, post_maxima, slot_maxima = [], [], []
            post_mean_gt = 0
            post_any_gt = 0
            for row in rows:
                own = row["persona"]
                means = row["template_mean_slot_activation"][own]
                posts = row["template_post_slot_maxima"][own]
                maxima = row["template_maxima"][own]
                slot_means.extend(float(x) for x in means)
                post_maxima.extend(float(x) for x in posts)
                slot_maxima.extend(float(x) for x in maxima)
                post_mean_gt += int(sum(posts) / len(posts) > sum(maxima) / len(maxima))
                post_any_gt += int(any(post > maximum for post, maximum in zip(posts, maxima)))
            total_pairs = len(rows) * 8
            diagnostics[frame][kind] = {
                "units": len(rows),
                "mean_slot_activation": sum(slot_means) / len(slot_means),
                "post_quote_maximum": sum(post_maxima) / len(post_maxima),
                "slot_token_maximum": sum(slot_maxima) / len(slot_maxima),
                "units_mean_post_gt_slot": post_mean_gt,
                "units_any_template_post_gt_slot": post_any_gt,
                "unit_template_pairs": total_pairs,
            }

    lengths = {}
    for frame in FRAMES:
        lengths[frame] = {}
        for unit in units:
            length = int(unit["slot_token_count"])
            row = lengths[frame].setdefault(length, {"reads": 0, "units": 0})
            row["units"] += 1
            row["reads"] += int(bool(focal[frame][str(unit["id"])]["reads"]))
        for row in lengths[frame].values():
            row["rate"] = row["reads"] / row["units"]

    one_null = one_token_null(
        units, focal, result["permutations"]["permutations"],
        result["permutations"]["seed"],
    )
    expected_one = sum(one_null) / len(one_null)
    registered_null = result["permutations"]["all_units"]["null_counts"]
    registered_null_mean = sum(registered_null) / len(registered_null)
    shared_phrase_table = []
    for phrase in phrases:
        if content_phrase_overlaps[phrase]:
            shared_phrase_table.append((phrase, sorted(content_phrase_overlaps[phrase])))

    phrase_popularity = Counter(
        phrase for row in focal["T"].values() for phrase in row["top3"]
    )
    hubs = {phrase for phrase, count in phrase_popularity.items() if count >= 4}
    t_readers = [row for row in focal["T"].values() if row["reads"]]
    hub_summary = {
        "threshold": 4,
        "hub_phrases": sorted(
            ((phrase, phrase_popularity[phrase]) for phrase in hubs),
            key=lambda item: (-item[1], item[0].casefold(), item[0]),
        ),
        "readers_with_other_hub": sum(
            any(phrase != row["persona"] and phrase in hubs for phrase in row["top3"])
            for row in t_readers
        ),
        "readers_owning_hub": sum(row["persona"] in hubs for row in t_readers),
        "readers": len(t_readers),
    }

    ruler_phrases = set(passages124.FAMILIES["king or ruler"])
    ruler_ids = [str(unit["id"]) for unit in units if unit["persona"] in ruler_phrases]
    ruler_observed = sum(bool(focal["T"][unit_id]["reads"]) for unit_id in ruler_ids)
    groups: dict[int, list[dict]] = defaultdict(list)
    for unit in units:
        groups[int(unit["length_stratum"])].append(unit)
    rng = random.Random(result["permutations"]["seed"])
    ruler_all_read = 0
    for _iteration in range(result["permutations"]["permutations"]):
        assigned = {}
        for members in groups.values():
            targets = [str(unit["persona"]) for unit in members]
            rng.shuffle(targets)
            assigned.update((str(unit["id"]), phrase)
                            for unit, phrase in zip(members, targets))
        ruler_all_read += int(all(
            assigned[unit_id] in focal["T"][unit_id]["top3"]
            for unit_id in ruler_ids
        ))

    random_max_blocks = []
    for row in activation_rows:
        if row["frame"] == "T" and row["side"] == "persona":
            random_max_blocks.extend(
                np.asarray(values, dtype=np.float64)
                for values in row["random_max"].values()
            )
    random_max_values = np.concatenate(random_max_blocks)
    random_channel_scale = {
        "mean": float(random_max_values.mean()),
        "median": float(np.median(random_max_values)),
        "observations": int(random_max_values.size),
        "channels": int(sum(len(pool) for pool in read_set["random_pools"].values())),
    }

    part122_cold_rows = [
        {
            "unit_id": unit_id,
            "phrase": by_id[unit_id]["persona"],
            "rank_T": focal["T"][unit_id]["own_rank"],
            "read_T": bool(focal["T"][unit_id]["reads"]),
        }
        for unit_id in PART122_COLD_UNIT_IDS if unit_id in by_id
    ]
    if len(part122_cold_rows) != len(PART122_COLD_UNIT_IDS):
        raise ValueError("a Part 122 cold-check unit is missing from Part 124")

    content_overlap_summary = {}
    for frame in FRAMES:
        rows = shared_reads_content[frame]
        content_overlap_summary[frame] = {
            "reads": len(rows),
            "own_phrase_overlap": sum(bool(row["overlap_phrases"]) for row in rows),
            "top3_overlap": sum(bool(row["top3_overlap_phrases"]) for row in rows),
        }
    all_word_overlap_summary = {}
    for frame in FRAMES:
        rows = shared_reads[frame]
        all_word_overlap_summary[frame] = {
            "reads": len(rows),
            "own_phrase_overlap": sum(bool(row["overlap_phrases"]) for row in rows),
            "top3_overlap": sum(bool(row["top3_overlap_phrases"]) for row in rows),
        }

    origin_summary = {
        "verbatim": sum(row["verbatim"] for row in phrase_origin.values()),
        "phrases": len(phrase_origin),
        "content_words_present": sum(
            row["content_words_present"] for row in phrase_origin.values()),
        "readers_verbatim": sum(
            phrase_origin[unit_id]["verbatim"] for unit_id in focal["T"]
            if focal["T"][unit_id]["reads"]),
        "readers": len(t_readers),
        "nonreaders_verbatim": sum(
            phrase_origin[unit_id]["verbatim"] for unit_id in focal["T"]
            if not focal["T"][unit_id]["reads"]),
        "nonreaders": len(units) - len(t_readers),
    }

    return {
        "units": units,
        "by_id": by_id,
        "activation_bytes": activation_bytes,
        "cold_fix_count": cold_check_fix_count(),
        "counts": counts,
        "random_by_kind": random_by_kind,
        "alternate_counts": alternate_counts,
        "agreed_units": len(agreed_ids),
        "deepseek_classified_units": sum(
            row["ds_kind"] in KINDS for row in p92b_ids.values()
        ),
        "deepseek_unclassified": sum(
            row["ds_kind"] not in KINDS for row in p92b_ids.values()
        ),
        "p92b_kinds": p92b_ids,
        "p92b_failed_predictions": {
            key: row["verdict"] for key, row in part92b.get("verdicts", {}).items()
            if row.get("verdict") == "failed"
        },
        "shared_reads": shared_reads,
        "shared_reads_content": shared_reads_content,
        "shared_phrase_table": shared_phrase_table,
        "content_overlap_summary": content_overlap_summary,
        "all_word_overlap_summary": all_word_overlap_summary,
        "phrase_origin": phrase_origin,
        "origin_summary": origin_summary,
        "label_tests": label_tests,
        "template_robustness": template_robustness,
        "thin_read_count": sum(
            row["templates_top3"] < 6 for row in template_robustness.values()),
        "hub_summary": hub_summary,
        "ruler_cluster": {
            "units": len(ruler_ids),
            "reads": ruler_observed,
            "permutations_all_read": ruler_all_read,
            "permutations": result["permutations"]["permutations"],
            "rate": ruler_all_read / result["permutations"]["permutations"],
            "post_hoc": True,
        },
        "registered_null_mean": registered_null_mean,
        "random_channel_scale": random_channel_scale,
        "part122_cold_rows": part122_cold_rows,
        "diagnostics": diagnostics,
        "lengths": lengths,
        "one_token_null": one_null,
        "one_token_expected": expected_one,
        "one_token_zero_rate": sum(value == 0 for value in one_null) / len(one_null),
        "fisher_t": fisher_two_sided(
            result["kind_counts"]["concept"]["reads"],
            result["kind_counts"]["concept"]["units"],
            result["kind_counts"]["slot"]["reads"],
            result["kind_counts"]["slot"]["units"],
        ),
        "mean_rate_difference_pp": 100 * result["concept_minus_slot_rate"],
    }


def draw_figure(result: dict, summary: dict, run_meta: dict) -> None:
    units = summary["units"]
    focal = result["focal"]
    permutation = result["permutations"]["all_units"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 13),
                             gridspec_kw={"width_ratios": [1.0, 1.7, 2.2]})
    ax = axes[0]
    null = permutation["null_counts"]
    bins = np.arange(min(null) - 0.5, max(null) + 1.5, 1)
    ax.hist(null, bins=bins, color="#98AFC7", edgecolor="white")
    ax.axvline(permutation["observed"], color="#B33A3A", linewidth=2.3,
               label=f"True map: {permutation['observed']}")
    ax.axvline(permutation["p95"], color="#333333", linewidth=1.4,
               linestyle="--", label=f"Null p95: {permutation['p95']}")
    ax.set_title("T permutation null")
    ax.set_xlabel("Top-three own-phrase reads")
    ax.set_ylabel("Permutations")
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.04, 0.96,
        f"{permutation['permutations']:,} shuffles\nmedian {permutation['median']:.0f}\n"
        f"tail {permutation['tail_count']}/{permutation['permutations']}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
    )

    ax = axes[1]
    positions = {}
    x_values = []
    labels = []
    actual_handles = []
    random_handles = []
    width = 0.20
    group_centers = {"concept": 0.0, "slot": 2.15}
    offsets = {"T": -0.50, "F": 0.0, "C": 0.50}
    for kind in KINDS:
        center = group_centers[kind]
        for frame in FRAMES:
            x = center + offsets[frame]
            positions[(kind, frame)] = x
            actual = summary["counts"][frame][kind]["rate"]
            random_rate = summary["random_by_kind"][frame][kind]["rate"]
            ax.bar(x - width / 2, actual, width=width,
                   color=FRAME_COLORS[frame], alpha=0.9)
            ax.bar(x + width / 2, random_rate, width=width,
                   facecolor="white", edgecolor=FRAME_COLORS[frame],
                   linewidth=1.3, hatch="///")
            x_values.append(x)
            labels.append(frame)
        # The read count p95 is converted using the registered kind denominator.
        kind_n = result["kind_counts"][kind]["units"]
        p95_rate = result["permutations"]["by_kind"][kind]["p95"] / kind_n
        ax.hlines(p95_rate, center - 0.70, center + 0.70, color="#222222",
                  linestyle=":", linewidth=1.8)
        ax.text(center, min(0.99, p95_rate + 0.025),
                f"T p95 {result['permutations']['by_kind'][kind]['p95']}/{kind_n}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(
        [group_centers[k] + offsets[f] for k in KINDS for f in FRAMES],
        [f for k in KINDS for f in FRAMES],
    )
    ax.axvline((group_centers["concept"] + group_centers["slot"]) / 2,
               color="#aaaaaa", linewidth=0.8)
    ax.set_xlim(-0.85, 3.0)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Read rate")
    ax.set_xlabel("Frame; concept group first, slot group second")
    ax.set_title("Read rates by kind and frame")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=FRAME_COLORS["T"], label="T focal"),
        Patch(facecolor=FRAME_COLORS["F"], label="F focal"),
        Patch(facecolor=FRAME_COLORS["C"], label="C focal"),
        Patch(facecolor="white", edgecolor="#555555", hatch="///",
              label="Random-channel rate"),
        plt.Line2D([0], [0], color="#222222", linestyle=":",
                   label="T kind permutation p95"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=7, loc="upper left")

    ax = axes[2]
    sorted_units = sorted(
        units,
        key=lambda unit: (
            focal["T"][str(unit["id"])]["own_rank"],
            0 if unit["kind"] == "concept" else 1,
            str(unit["persona"]).casefold(),
        ),
    )
    y_positions = np.arange(len(sorted_units))
    for index, unit in enumerate(sorted_units):
        unit_id = str(unit["id"])
        color = KIND_COLORS[str(unit["kind"])]
        for frame in FRAMES:
            ax.scatter(
                focal[frame][unit_id]["own_rank"], index,
                color=color, marker=FRAME_MARKERS[frame], s=26,
                edgecolors="white", linewidths=0.25, zorder=3,
            )
    ax.set_yticks(
        y_positions,
        [str(unit["id"]) for unit in sorted_units],
        fontsize=6.3,
    )
    ax.invert_yaxis()
    unit_count = len(units)
    ax.set_xlim(0, unit_count + 1)
    ax.set_xticks([tick for tick in [1, 3, 5, 10, 15, 20, 25, 30, 35, unit_count]
                    if tick <= unit_count])
    ax.set_xlabel("Own phrase rank (1 is highest)")
    ax.set_title(f"Own-phrase rank for all {unit_count} units")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    rank_legend = [
        Patch(facecolor=KIND_COLORS[kind], label=kind.title())
        for kind in KINDS
    ] + [
        Line2D([0], [0], color="#333333", marker=FRAME_MARKERS[frame],
               linestyle="none", label=frame)
        for frame in FRAMES
    ]
    ax.legend(handles=rank_legend, frameon=False, fontsize=7,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.subplots_adjust(right=0.88, wspace=0.3)

    fig.suptitle(
        f'Quoted phrase preference in OLMo 2 1B Instruct, revision '
        f'{run_meta["model_revision"][:8]}',
        fontsize=15,
    )
    fig.savefig(HERE / "fig124.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def format_read_list(result: dict) -> str:
    lines = [
        "| Frame | Unit | Kind | Own phrase (rank) | Its top three, in rank order |",
        "| --- | --- | --- | --- | --- |",
    ]
    for frame in FRAMES:
        rows = [
            row for row in result["focal"][frame].values()
            if row["reads"]
        ]
        rows.sort(key=lambda row: (row["own_rank"], row["unit_id"]))
        for row in rows:
            top = "; ".join(row["top3"])
            lines.append(
                f'| {frame} | {row["unit_id"]} | {row["kind"]} | '
                f'{row["persona"]} ({row["own_rank"]}) | {top} |'
            )
    return "\n".join(lines)


def format_numbers(result: dict, summary: dict, run_meta: dict) -> str:
    src = number_source()
    unit_count = len(summary["units"])
    construction = result["gates"]["construction"]["details"]
    all_perm = result["permutations"]["all_units"]
    concept = result["kind_counts"]["concept"]
    slot = result["kind_counts"]["slot"]
    concept_perm = result["permutations"]["by_kind"]["concept"]
    slot_perm = result["permutations"]["by_kind"]["slot"]
    one_token = summary["lengths"]["T"].get(1, {"reads": 0, "units": 0})
    persistence = result["persistence_T_to_C"]
    agreed_concept_n = summary["alternate_counts"]["agreed"]["T"]["concept"]["units"]
    agreed_slot_n = summary["alternate_counts"]["agreed"]["T"]["slot"]["units"]
    deepseek_concept_n = summary["alternate_counts"]["deepseek"]["T"]["concept"]["units"]
    deepseek_slot_n = summary["alternate_counts"]["deepseek"]["T"]["slot"]["units"]
    lines = [
        "# Part 124 numbers",
        "",
        "fig124.py calls the Part 124 scorer on raw activation rows, the read set, run metadata, and",
        "the frozen passage manifest, then checks the result against score124.json. This is an internal",
        "consistency check, not an independent scoring method. Added summaries use Part 92 and 92b source files.",
        "",
        "## Run and gates",
        "",
        "| Item | Recomputed value | Source |",
        "| --- | --- | --- |",
        f'| Model | OLMo 2 1B Instruct, revision {run_meta["model_revision"]} | run_meta.json |',
        f'| LF-normalized activation log size | {summary["activation_bytes"]:,} bytes ({summary["activation_bytes"] / (1024 * 1024):.1f} MiB) | activations.jsonl or activations.jsonl.gz |',
        f'| Dtype and device | {run_meta["parameter_dtype"]}; {run_meta["device"]} | run_meta.json |',
        f'| Construction gate | pass; {construction["frame_template_checks"]} frame/template checks; {construction["matched_persona_neutral_pairs"]} matched phrase-neutral pairs; {construction["slots_checked"]} slots; {construction["quote_boundary_merges"]} quote-boundary merges | frozen passages124.json |',
        f'| Do-nothing gate | pass; identical logits, shape {run_meta["do_nothing"]["logit_shape"]} | run_meta.json |',
        f'| Random-channel gate | pass; each frame is at or below 15% | {src} |',
        f'| Instrument gate | 10,000 random channels; T: {result["instrument"]["by_frame"]["T"]["repeat"]}/{result["instrument"]["by_frame"]["T"]["channels"]} ({percent(result["instrument"]["by_frame"]["T"]["rate"])}); the check says nothing about the 39 focal units | {src} |',
        f'| All gates | {sum(bool(gate["passed"]) for gate in result["gates"].values())}/{len(result["gates"])} passed | {src} |',
        f'| Cold-check priority items | {summary["cold_fix_count"]} | COLD-CHECK-r1.md |',
        "",
        "## Phrase origin",
        "",
        "These labels came from the answers the same units gave when pushed to write in Part 92. Part 124 reads those write-side choices in quoted text.",
        "",
        "| Measure | Recomputed value | Source |",
        "| --- | ---: | --- |",
        f'| Phrases that appear verbatim in their unit\'s Part 92 answer | {summary["origin_summary"]["verbatim"]}/{summary["origin_summary"]["phrases"]} | Part 92 neurons.json; passages124.json |',
        f'| Phrases whose content words all appear in that answer | {summary["origin_summary"]["content_words_present"]}/{summary["origin_summary"]["phrases"]} | Part 92 neurons.json; passages124.json |',
        f'| T readers with a verbatim phrase | {summary["origin_summary"]["readers_verbatim"]}/{summary["origin_summary"]["readers"]} | raw Part 124 activations; Part 92 neurons.json |',
        f'| T non-readers with a verbatim phrase | {summary["origin_summary"]["nonreaders_verbatim"]}/{summary["origin_summary"]["nonreaders"]} | raw Part 124 activations; Part 92 neurons.json |',
        "",
        "## Registered T result",
        "",
        "| Measure | Value | Source |",
        "| --- | --- | --- |",
        f'| Focal own-phrase reads | {all_perm["observed"]}/{unit_count} ({percent(all_perm["observed"] / unit_count)}) | {src} |',
        f'| Length-stratified null, seed {result["permutations"]["seed"]} | median {all_perm["median"]:.1f}; nearest-rank p95 {all_perm["p95"]}; tail {all_perm["tail_count"]}/{all_perm["permutations"]} | {src} |',
        f'| Registered null mean | {summary["registered_null_mean"]:.4f} reads ({percent(summary["registered_null_mean"] / unit_count)} of {unit_count}); this differs from the unstratified 3/39 rate because labels are reassigned only within length strata | {src} |',
        f'| Concept own-phrase reads | {concept["reads"]}/{concept["units"]}; rate {percent(concept["rate"])}; null median {concept_perm["median"]:.1f}, p95 {concept_perm["p95"]}, tail {concept_perm["tail_count"]}/{concept_perm["permutations"]} | {src} |',
        f'| Slot own-phrase reads | {slot["reads"]}/{slot["units"]}; rate {percent(slot["rate"])}; null median {slot_perm["median"]:.1f}, p95 {slot_perm["p95"]}, tail {slot_perm["tail_count"]}/{slot_perm["permutations"]} | {src} |',
        f'| Concept minus slot rate | {summary["mean_rate_difference_pp"]:.2f} percentage points | {src} |',
        f'| After-the-fact Fisher exact test, concept vs slot | two-sided p = {summary["fisher_t"]:.6f} | {src} |',
        "",
        "## Frame rates and controls",
        "",
        "| Frame | Kind | Focal reads | Random pseudo-target reads | Kind T permutation p95 | Source |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for kind in KINDS:
        p95 = result["permutations"]["by_kind"][kind]["p95"]
        n_kind = result["kind_counts"][kind]["units"]
        p95_text = f"{p95}/{n_kind} ({percent(p95 / n_kind)})"
        for frame in FRAMES:
            focal = summary["counts"][frame][kind]
            random_row = summary["random_by_kind"][frame][kind]
            lines.append(
                f'| {frame} | {kind} | {focal["reads"]}/{focal["units"]} ({percent(focal["rate"])}) | '
                f'{random_row["reads"]}/{random_row["observed"]} ({percent(random_row["rate"])}) | '
                f'{p95_text if frame == "T" else "T-only test"} | {src} |'
            )
    lines.extend([
        "",
        "| Frame | Random reads | Chance by construction | Gate | Instrument repeat |",
        "| --- | ---: | ---: | --- | ---: |",
    ])
    for frame in FRAMES:
        random_gate = result["gates"]["random_channels"]["by_frame"][frame]
        instrument = result["instrument"]["by_frame"][frame]
        lines.append(
            f'| {frame} | {random_gate["reads"]}/{random_gate["observed"]} ({percent(random_gate["rate"])}) | '
            f'{percent(result["chance_rate"])} | {"pass" if random_gate["passed"] else "fail"} | '
            f'{instrument["repeat"]}/{instrument["channels"]} ({percent(instrument["rate"])}) |'
        )

    lines.extend([
        "",
        "## Neutral foil",
        "",
        "A neutral win is a strict own-phrase-over-neutral slot-maximum comparison in one matched template.",
        "",
        "| Frame | Kind | Own-phrase wins | Share of comparisons | Source |",
        "| --- | --- | ---: | ---: | --- |",
    ])
    for frame in FRAMES:
        for kind in KINDS:
            row = result["neutral"][frame][kind]
            lines.append(
                f'| {frame} | {kind} | {row["wins"]}/{row["comparisons"]} | '
                f'{percent(row["share"])} | {src} |'
            )

    lines.extend([
        "",
        "## Reads by phrase token length",
        "",
        "| Frame | Slot tokens | Reads / phrases | Rate | Source |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for frame in FRAMES:
        for length in sorted(summary["lengths"][frame]):
            row = summary["lengths"][frame][length]
            lines.append(
                f'| {frame} | {length} | {row["reads"]}/{row["units"]} | '
                f'{percent(row["rate"])} | {src} |'
            )
    one_null = summary["one_token_null"]
    lines.extend([
        "",
        f'One-token T group: observed {one_token["reads"]}/{one_token["units"]}; its within-length permutation mean is '
        f'{summary["one_token_expected"]:.3f}, median {statistics.median(one_null):.1f}, '
        f'p95 {nearest_rank(one_null, 0.95)}, and {sum(v == 0 for v in one_null)}/'
        f'{len(one_null)} permutations ({percent(summary["one_token_zero_rate"])}) also give zero. '
        f'Source: {src}.',
        "",
        "## Part 92b labels, after the fact",
        "",
        f'Part 92b agrees with {summary["agreed_units"]} units: {agreed_concept_n} concept and {agreed_slot_n} slot. '
        f'DeepSeek classifies {summary["deepseek_classified_units"]} units: {deepseek_concept_n} concept and {deepseek_slot_n} slot; '
        f'{summary["deepseek_unclassified"]} are unclassified. Read decisions remain the raw Part 124 scores.',
        f'Part 92b verdicts for predictions 92b.2 and 92b.3: {summary["p92b_failed_predictions"].get("92b.2", "missing")}, {summary["p92b_failed_predictions"].get("92b.3", "missing")}. Source: results/part92b/score92b.json.',
        "",
        "| Label set | Frame | Kind | Reads / units | Rate | Source |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ])
    for group, label in (("agreed", "Haiku and DeepSeek agree"),
                         ("deepseek", "DeepSeek kind")):
        for frame in FRAMES:
            for kind in KINDS:
                row = summary["alternate_counts"][group][frame][kind]
                lines.append(
                    f'| {label} | {frame} | {kind} | {row["reads"]}/{row["units"]} | '
                    f'{percent(row["rate"])} | raw Part 124 activations + Part 92b labels |'
                )

    lines.extend([
        "",
        "## Label-conditional concept and slot comparison",
        "",
        "| Labels | Concept reads | Slot reads | Fisher p | Length-adjusted p (CMH) | Source |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for label_name in ("Haiku labels", "DeepSeek labels", "29 agreed units"):
        row = summary["label_tests"][label_name]
        lines.append(
            f'| {label_name} | {row["concept_reads"]}/{row["concept_units"]} | '
            f'{row["slot_reads"]}/{row["slot_units"]} | {row["fisher_p"]:.3f} | '
            f'{row["length_adjusted_p"]:.3f} | raw Part 124 activations + labels in score92b.json |'
        )
    lines.extend([
        "",
        f'Part 92b predictions 92b.2 and 92b.3: {summary["p92b_failed_predictions"].get("92b.2", "missing")}, {summary["p92b_failed_predictions"].get("92b.3", "missing")} (score92b.json).',
        "",
        "## Robustness and phrase hubs",
        "",
        "Templates counts how many of the eight templates put the unit's own phrase in its top three. Margin is its mean score minus the fourth-highest phrase score. Thin marks the three reads with fewer than six template hits.",
        f'Thin reads flagged: {summary["thin_read_count"]}. Source: raw Part 124 activations.',
        "",
        "| Unit | Templates in top three | Margin over fourth | Note | Source |",
        "| --- | ---: | ---: | --- | --- |",
    ])
    reader_ids = sorted(
        summary["template_robustness"],
        key=lambda unit_id: (
            result["focal"]["T"][unit_id]["own_rank"], unit_id),
    )
    for unit_id in reader_ids:
        robustness = summary["template_robustness"][unit_id]
        thin = "thin" if robustness["templates_top3"] < 6 else ""
        lines.append(
            f'| {unit_id} | {robustness["templates_top3"]}/8 | '
            f'{robustness["margin_over_fourth"]:.5f} | {thin} | raw Part 124 activations |'
        )
    hub_summary = summary["hub_summary"]
    lines.extend([
        "",
        f'Using four or more appearances in the top threes as the hub rule, {hub_summary["readers_with_other_hub"]}/{hub_summary["readers"]} T readers had another hub phrase in their top three and {hub_summary["readers_owning_hub"]}/{hub_summary["readers"]} owned a hub phrase. The king-or-ruler family read {summary["ruler_cluster"]["reads"]}/{summary["ruler_cluster"]["units"]}; under the registered length-stratified permutation all four read in {summary["ruler_cluster"]["permutations_all_read"]}/{summary["ruler_cluster"]["permutations"]} runs. This is post hoc. Source: raw Part 124 activations, frozen passages, and registered permutation seed.',
        "",
        "| Hub phrase | T top-three appearances | Source |",
        "| --- | ---: | --- |",
    ])
    for phrase, appearances in hub_summary["hub_phrases"]:
        lines.append(f'| {phrase} | {appearances} | raw Part 124 activations |')
    lines.extend([
        "",
        "## Part 122 cold-check cases",
        "",
        "The Part 122 cold check's named units are shown here by their Part 124 T rank.",
        "",
        "| Unit | Phrase | T rank | Top three read | Source |",
        "| --- | --- | ---: | --- | --- |",
    ])
    for row in summary["part122_cold_rows"]:
        lines.append(
            f'| {row["unit_id"]} | {row["phrase"]} | {row["rank_T"]} | '
            f'{"yes" if row["read_T"] else "no"} | Part 122 COLD-CHECK-r1.md; raw Part 124 activations |'
        )
    scale = summary["random_channel_scale"]
    slot_scale = summary["diagnostics"]["T"]["slot"]
    lines.extend([
        "",
        "## Slot scale and random-channel scale",
        "",
        "| T measure | Value | Source |",
        "| --- | ---: | --- |",
        f'| Slot-unit own-phrase maximum mean | {slot_scale["slot_token_maximum"]:.6f} over {slot_scale["unit_template_pairs"]} unit-template values | raw Part 124 activations |',
        f'| Random-channel slot maxima | mean {scale["mean"]:.6f}; median {scale["median"]:.6f} across {scale["observations"]:,} channel-phrase-template values from {scale["channels"]:,} channels | raw Part 124 activations and read_set.json |',
    ])

    lines.extend([
        "",
        "## Shared content words, after the fact",
        "",
        "Words are lowercase alphanumeric tokens split on punctuation. Articles, pronouns, prepositions, conjunctions, and other listed function words are removed before checking overlap. A top-three overlap requires a different phrase that shares a content word with the unit's own phrase.",
        "",
        "| Frame | Reads | Own phrases sharing a content word | Read top threes containing a content-word phrase | Source |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for frame in FRAMES:
        row = summary["content_overlap_summary"][frame]
        lines.append(
            f'| {frame} | {row["reads"]} | '
            f'{row["own_phrase_overlap"]}/{row["reads"]} | '
            f'{row["top3_overlap"]}/{row["reads"]} | raw Part 124 activations + frozen phrases |'
        )
    lines.extend([
        "",
        "Content-word phrase pairs in the frozen set:",
        "",
    ])
    for phrase, overlaps in summary["shared_phrase_table"]:
        lines.append(f'- {phrase}: {", ".join(overlaps)}')
    all_words = summary["all_word_overlap_summary"]
    lines.extend([
        "",
        f'[^1]: Before removing function words, own-phrase overlaps were T {all_words["T"]["own_phrase_overlap"]}/{all_words["T"]["reads"]}, F {all_words["F"]["own_phrase_overlap"]}/{all_words["F"]["reads"]}, C {all_words["C"]["own_phrase_overlap"]}/{all_words["C"]["reads"]}; top-three overlaps were T {all_words["T"]["top3_overlap"]}/{all_words["T"]["reads"]}, F {all_words["F"]["top3_overlap"]}/{all_words["F"]["reads"]}, C {all_words["C"]["top3_overlap"]}/{all_words["C"]["reads"]}. Source: raw Part 124 activations + frozen phrases.',
        "Content-word tokens exclude: " + ", ".join(sorted(CONTENT_STOPWORDS)) + ".",
    ])
    lines.extend([
        "",
        "## Mean-slot and post-quote diagnostics, after the fact",
        "",
        "Values average each unit's own persona phrase across the eight templates, then average across units and templates. Post-quote maxima are compared with that same unit's slot-token maxima.",
        "",
        "| Frame | Kind | Mean slot activation | Mean slot-token maximum | Mean post-quote maximum | Units with mean post > slot | Units with any template post > slot | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for frame in FRAMES:
        for kind in KINDS:
            row = summary["diagnostics"][frame][kind]
            lines.append(
                f'| {frame} | {kind} | {row["mean_slot_activation"]:.6f} | '
                f'{row["slot_token_maximum"]:.6f} | {row["post_quote_maximum"]:.6f} | '
                f'{row["units_mean_post_gt_slot"]}/{row["units"]} | '
                f'{row["units_any_template_post_gt_slot"]}/{row["units"]} | {src} |'
            )

    lines.extend([
        "",
        "## Registered phrase families",
        "",
        f'The T check found {len(result["family_crowding_near_misses_T"])} family-crowding near misses. These do not count as reads. Source: frozen passages124.json and raw Part 124 activations.',
        "",
    ])
    for family, family_phrases in passages124.FAMILIES.items():
        lines.append(f'- {family}: {", ".join(family_phrases)}')

    lines.extend([
        "",
        "## All unit ranks and own-phrase template maxima",
        "",
        "The eight maxima are for the unit's own phrase, in template order. Ranks and verdicts are recomputed separately for each frame.",
        "",
        "| Unit | Kind | Own phrase | Tokens / stratum | Frame | Own rank | Read | Own-phrase slot maxima, templates 1 to 8 | Source |",
        "| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |",
    ])
    sorted_units = sorted(
        summary["units"],
        key=lambda unit: (
            result["focal"]["T"][str(unit["id"])]["own_rank"],
            str(unit["id"]),
        ),
    )
    for unit in sorted_units:
        unit_id = str(unit["id"])
        for frame in FRAMES:
            row = result["focal"][frame][unit_id]
            maxima = row["template_maxima"][row["persona"]]
            maxima_text = ", ".join(f"{float(value):.6g}" for value in maxima)
            lines.append(
                f'| {unit_id} | {row["kind"]} | {row["persona"]} | '
                f'{row["slot_token_count"]} / {row["length_stratum"]} | {frame} | '
                f'{row["own_rank"]} | {"yes" if row["reads"] else "no"} | '
                f'{maxima_text} | activations.jsonl + passages124.json |'
            )

    lines.extend([
        "",
        "## Every own-phrase read and its top three",
        "",
        format_read_list(result),
        "",
        "## Registered predictions",
        "",
        "| # | Prediction, verbatim | Result | Verdict | Source |",
        "| --- | --- | --- | --- | --- |",
    ])
    for key, text in PREDICTION_TEXT.items():
        status = result["predictions"][key]["status"]
        if key == "124.1":
            value = (f'T {all_perm["observed"]}/{unit_count}; null median '
                     f'{all_perm["median"]:.0f}, p95 {all_perm["p95"]}; tail '
                     f'{all_perm["tail_count"]}/{all_perm["permutations"]}')
        elif key == "124.2":
            value = (f'T concept {concept["reads"]}/{concept["units"]} vs slot '
                     f'{slot["reads"]}/{slot["units"]}; rate difference '
                     f'{summary["mean_rate_difference_pp"]:.2f} points')
        elif key == "124.3":
            value = f'T slot reads {slot["reads"]}/{slot["units"]}'
        else:
            value = (f'{persistence["also_read_in_C"]}/{persistence["reads_in_T"]} '
                     f'T readers also read in C ({percent(persistence["rate"])})')
        lines.append(
            f'| {key} ({result["predictions"][key]["confidence"]}) | {text} | {value} | '
            f'{"held" if status is True else "failed" if status is False else "untestable"} | {src} |'
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    (manifest, read_set, run_meta, activation_rows, result, reference,
     activation_bytes, activation_path) = load_and_recompute()
    part92b_path = HERE.parent / "part92b" / "score92b.json"
    part92b = read_json(part92b_path)
    part92 = read_json(REPO / "results" / "part92" / "step2" / "neurons.json")
    summary = build_summaries(
        manifest, read_set, run_meta, result, reference, part92b, part92,
        activation_rows, activation_bytes,
    )
    draw_figure(result, summary, run_meta)
    (HERE / "NUMBERS.md").write_text(
        format_numbers(result, summary, run_meta), encoding="utf-8"
    )
    print(f"Loaded activation source: {activation_path.name}.")
    print(f"Recomputed {len(activation_rows)} activation records.")
    print(f"Wrote {HERE / 'fig124.png'} and {HERE / 'NUMBERS.md'}.")
    print(
        f'T reads {result["permutations"]["all_units"]["observed"]}/39; '
        f'concept {result["kind_counts"]["concept"]["reads"]}/18; '
        f'slot {result["kind_counts"]["slot"]["reads"]}/21.'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
