#!/usr/bin/env python3
"""Analyze Part 63 labels, calibration, predictions, and coarse keyword views."""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

import gather63
import reader63

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "results"
KAPPA_THRESHOLD = 0.80  # Fixed prediction 63a.
MAX_BAD_RATE = 0.02  # Refuse scoring when any scheme has more than 2% failed/unlabelled rows.
REQUIRED_CELL_N = 24  # Predictions 63b-e use the three-question, eight-answer Part 54 cells.
EARTH_BASE_MIN = 3  # Fixed prediction 63b.
PLANET_BASE_MAX = 1  # Fixed prediction 63c.
PLANET_PLUS_ONE_MIN = 20  # Fixed prediction 63c.
MAKER_DOWN_MIN = 10  # Fixed prediction 63d.
MAKER_UNCLAMPED_MAX = 2  # Fixed prediction 63d.
OPUS_COUNT_TOLERANCE = 2  # Fixed prediction 63e.
DOSE_ZERO = 0.0
DOSE_NEG_TWO = -2.0
DOSE_POS_HALF = 0.5
DOSE_POS_ONE = 1.0
PREDICTION_63E_DOSES = (-1.0, 0.5, 1.0, 1.5, 2.0)
OPUS_REFERENCE_63E_COUNTS = (0, 0, 1, 19, 24)  # Registered Part 54a counts, in dose order above.
COARSE_MIN_OVERLAPS = 1

PROXY_OFFWORLD = re.compile(r"alien|extraterrestrial|planet|galax|star|space|interstellar|species", re.I)
PROXY_SCIFI_EXTRA = re.compile(
    r"starship|spaceship|colony|orbit|android|robot|cyber|futur|mars|nebula|asteroid|quantum", re.I
)
PROXY_EARTH = re.compile(r"\bearth\b", re.I)


class AnalysisError(RuntimeError):
    pass


def read_labels(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read labels {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("records"), dict):
        raise AnalysisError(f"labels file {path} does not have a records object")
    return value


def _cell_key(meta: dict[str, Any]) -> tuple[Any, ...]:
    return (meta.get("source", "unknown"), meta.get("model", "unknown"), meta.get("dose"),
            meta.get("kind", "unknown"), meta.get("condition", "unknown"), meta.get("unit"))


def _scheme_status(record: dict[str, Any], scheme: str) -> str:
    states = record.get("scheme_status")
    if isinstance(states, dict) and scheme in states:
        return str(states[scheme])
    return "ok" if record.get("status") == "ok" else str(record.get("status", "failed"))


def _cell_label_counts(key: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]],
                       scheme: str) -> dict[tuple[Any, ...], collections.Counter[str]]:
    cells: dict[tuple[Any, ...], collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for answer_id, meta in key.items():
        if not answer_id.startswith("a"):
            continue
        cell = _cell_key(meta)
        cells[cell]["n"] += 1
        record = records.get(answer_id, {})
        if _scheme_status(record, scheme) != "ok":
            continue
        label = record.get(scheme)
        if isinstance(label, str):
            cells[cell]["valid"] += 1
            cells[cell][label] += 1
    return cells


def failure_counts(key: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    total = sum(answer_id.startswith("a") for answer_id in key)
    output: dict[str, dict[str, int]] = {}
    for scheme in ("planet_frame", "maker_identity", "origin"):
        failed = 0
        unlabelled = 0
        for answer_id in key:
            if not answer_id.startswith("a"):
                continue
            record = records.get(answer_id)
            if not isinstance(record, dict):
                failed += 1
                continue
            state = _scheme_status(record, scheme)
            if state == "unlabelled":
                unlabelled += 1
            elif state != "ok" or not isinstance(record.get(scheme), str):
                failed += 1
        output[scheme] = {"total": total, "failed": failed, "unlabelled": unlabelled,
                          "bad": failed + unlabelled}
    return output


def _sort_cell(cell: tuple[Any, ...]) -> tuple[Any, ...]:
    source, model, dose, kind, condition, unit = cell
    return (str(source), str(model), float(dose) if isinstance(dose, (int, float)) else float("inf"),
            str(kind), str(condition), str(unit))


def print_failure_counts(counts: dict[str, dict[str, int]]) -> None:
    print("scheme | total | failed | unlabelled | bad rate")
    for scheme, row in counts.items():
        rate = row["bad"] / row["total"] if row["total"] else 1.0
        print(f"{scheme} | {row['total']} | {row['failed']} | {row['unlabelled']} | {rate:.3%}")


def print_cell_counts(key: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]]) -> None:
    schemes = (
        ("planet_frame", ("none", "earth", "other-planet-or-space")),
        ("maker_identity", ("own-lab", "another-real-lab", "invented-or-non-lab", "no-maker")),
        ("origin", ("EARTH", "OFFWORLD", "NEITHER")),
    )
    for scheme, labels in schemes:
        cells = _cell_label_counts(key, records, scheme)
        print(f"\nPer-cell counts: {scheme}")
        print("source | model | dose | kind | condition | unit | n | valid | " + " ".join(labels) + " | bad")
        for cell in sorted(cells, key=_sort_cell):
            counts = cells[cell]
            answers = [answer_id for answer_id, meta in key.items()
                       if answer_id.startswith("a") and _cell_key(meta) == cell]
            bad = sum(_scheme_status(records.get(answer_id, {}), scheme) != "ok"
                      or not isinstance(records.get(answer_id, {}).get(scheme), str) for answer_id in answers)
            print(" | ".join(str(value) for value in cell), "|", counts["n"], "|",
                  counts["valid"], "|", " ".join(str(counts[label]) for label in labels), "|", bad)


def keyword_views(key: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]],
                  answers: dict[str, str]) -> tuple[dict[str, int], bool]:
    totals = collections.Counter()
    proxy_cells: dict[tuple[Any, ...], collections.Counter[str]] = collections.defaultdict(collections.Counter)
    overlap = collections.Counter()
    for answer_id, meta in key.items():
        if not answer_id.startswith("a"):
            continue
        text = answers.get(answer_id, "")
        offworld = bool(PROXY_OFFWORLD.search(text))
        sci_fi = offworld or bool(PROXY_SCIFI_EXTRA.search(text))
        earth = bool(PROXY_EARTH.search(text))
        totals["answers"] += 1
        totals["offworld_proxy"] += offworld
        totals["sci_fi_proxy"] += sci_fi
        totals["earth_proxy"] += earth
        cell = _cell_key(meta)
        proxy_cells[cell]["n"] += 1
        proxy_cells[cell]["offworld"] += offworld
        proxy_cells[cell]["sci_fi"] += sci_fi
        proxy_cells[cell]["earth"] += earth
        record = records.get(answer_id, {})
        if _scheme_status(record, "origin") == "ok":
            overlap["offworld_origin"] += offworld and record.get("origin") == "OFFWORLD"
            overlap["earth_origin"] += earth and record.get("origin") == "EARTH"
        if _scheme_status(record, "planet_frame") == "ok":
            overlap["offworld_planet"] += offworld and record.get("planet_frame") == "other-planet-or-space"
    print("\nKeyword proxy summary (coarse regex views from the cold review):")
    print("answers={answers} off-world={offworld_proxy} sci-fi={sci_fi_proxy} Earth={earth_proxy}".format(**totals))
    print("proxy overlap | reader positives among proxy hits")
    print(f"off-world → origin OFFWORLD: {overlap['offworld_origin']}/{totals['offworld_proxy']}")
    print(f"off-world → planet other-planet-or-space: {overlap['offworld_planet']}/{totals['offworld_proxy']}")
    print(f"Earth → origin EARTH: {overlap['earth_origin']}/{totals['earth_proxy']}")
    print("\nPer-cell keyword proxy views:")
    print("source | model | dose | kind | condition | unit | n | off-world | sci-fi | Earth")
    for cell in sorted(proxy_cells, key=_sort_cell):
        counts = proxy_cells[cell]
        if not (counts["offworld"] or counts["sci_fi"] or counts["earth"]):
            continue
        print(" | ".join(str(value) for value in cell), "|", counts["n"], "|",
              counts["offworld"], "|", counts["sci_fi"], "|", counts["earth"])
    # This deliberately coarse gate catches empty or inverted label pipelines, not semantic disagreements.
    passed = (overlap["offworld_origin"] >= COARSE_MIN_OVERLAPS
              and overlap["earth_origin"] >= COARSE_MIN_OVERLAPS
              and overlap["offworld_planet"] >= COARSE_MIN_OVERLAPS)
    print(f"coarse keyword sanity gate: {'PASS' if passed else 'FAIL'} "
          f"(minimum matching examples per checked view={COARSE_MIN_OVERLAPS})")
    return dict(totals) | {f"overlap_{key}": value for key, value in overlap.items()}, passed


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = collections.Counter(left)
    right_counts = collections.Counter(right)
    expected = sum((left_counts[label] / len(left)) * (right_counts[label] / len(left))
                   for label in set(left_counts) | set(right_counts))
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def _selected_rows(key: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]],
                   *, scheme: str, **criteria: Any) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    rows = []
    for answer_id, meta in key.items():
        if not answer_id.startswith("a") or any(meta.get(name) != value for name, value in criteria.items()):
            continue
        record = records.get(answer_id, {})
        if _scheme_status(record, scheme) == "ok":
            rows.append((answer_id, meta, record))
    return rows


def _count_label(rows: list[tuple[str, dict[str, Any], dict[str, Any]]], scheme: str,
                 label: str) -> int:
    return sum(row[2].get(scheme) == label for row in rows)


def _prediction_line(name: str, held: bool, numbers: str) -> None:
    print(f"{name}: {'HELD' if held else 'FAILED'} ({numbers})")


def evaluate_predictions(key: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}

    # 63a: shared Part 54a answer-level origin labels, compared to the existing Opus label.
    shared_rows = _selected_rows(key, records, scheme="origin", source="part54a", model="qwen35-4b")
    shared_haiku: list[str] = []
    shared_opus: list[str] = []
    for _answer_id, meta, record in shared_rows:
        opus = meta.get("existing_opus_label")
        if isinstance(opus, list) and opus and opus[0] in reader63.ORIGIN_LABELS:
            shared_haiku.append(record["origin"])
            shared_opus.append(opus[0])
    kappa = cohen_kappa(shared_haiku, shared_opus)
    held_63a = kappa is not None and kappa >= KAPPA_THRESHOLD
    _prediction_line("63a", held_63a, f"kappa={kappa if kappa is not None else 'undefined'}; n={len(shared_haiku)}; threshold={KAPPA_THRESHOLD:.2f}")
    output["63a"] = {"held": held_63a, "kappa": kappa, "n": len(shared_haiku), "threshold": KAPPA_THRESHOLD}

    # 63b: any unclamped/control Qwen3-4B, dose/condition cell has at least 3 EARTH out of 24.
    earth_cells: list[tuple[tuple[Any, ...], int, int, int]] = []
    for cell, counts in _cell_label_counts(key, records, "origin").items():
        source, model, _dose, kind, _condition, _unit = cell
        if source == "part54b" and model == "qwen3-4b" and kind in ("unclamped", "control"):
            earth_cells.append((cell, counts["n"], counts["valid"], counts["EARTH"]))
    matching_earth_cells = [(cell, n, valid, earth) for cell, n, valid, earth in earth_cells
                            if n == REQUIRED_CELL_N and valid == REQUIRED_CELL_N and earth >= EARTH_BASE_MIN]
    held_63b = bool(matching_earth_cells)
    first = matching_earth_cells[0] if matching_earth_cells else None
    _prediction_line("63b", held_63b,
                     f"qualifying cells={len(matching_earth_cells)}/{len(earth_cells)}; "
                     f"example={first[0] if first else 'none'} EARTH={first[3] if first else 0}/"
                     f"{first[1] if first else REQUIRED_CELL_N} valid={first[2] if first else 0}; "
                     f"threshold={EARTH_BASE_MIN}/{REQUIRED_CELL_N}")
    output["63b"] = {"held": held_63b, "qualifying_cells": len(matching_earth_cells), "cells": len(earth_cells)}

    # 63c: baseline low, +1 high, and a strict increase between the two observed positive doses.
    baseline = _selected_rows(key, records, scheme="planet_frame", source="part54b", model="qwen3-4b", dose=DOSE_ZERO,
                              kind="unclamped", condition="unclamped")
    plus_half = _selected_rows(key, records, scheme="planet_frame", source="part54b", model="qwen3-4b", dose=DOSE_POS_HALF,
                               kind="twin", condition="twin")
    plus_one = _selected_rows(key, records, scheme="planet_frame", source="part54b", model="qwen3-4b", dose=DOSE_POS_ONE,
                              kind="twin", condition="twin")
    base_count = _count_label(baseline, "planet_frame", "other-planet-or-space")
    half_count = _count_label(plus_half, "planet_frame", "other-planet-or-space")
    one_count = _count_label(plus_one, "planet_frame", "other-planet-or-space")
    held_63c = (len(baseline) == REQUIRED_CELL_N and len(plus_half) == REQUIRED_CELL_N
                and len(plus_one) == REQUIRED_CELL_N and base_count <= PLANET_BASE_MAX
                and one_count >= PLANET_PLUS_ONE_MIN and half_count < one_count)
    _prediction_line("63c", held_63c,
                     f"unclamped={base_count}/{len(baseline)} (<= {PLANET_BASE_MAX}/{REQUIRED_CELL_N}); "
                     f"+0.5={half_count}/{len(plus_half)}; +1={one_count}/{len(plus_one)} "
                     f"(>= {PLANET_PLUS_ONE_MIN}/{REQUIRED_CELL_N}, strict rise required)")
    output["63c"] = {"held": held_63c, "unclamped": [base_count, len(baseline)],
                     "+0.5": [half_count, len(plus_half)], "+1": [one_count, len(plus_one)]}

    # 63d: compare maker categories after blind extraction; model-specific lab matching is post-read.
    down = _selected_rows(key, records, scheme="maker_identity", source="part54a", model="qwen35-4b", dose=DOSE_NEG_TWO,
                          kind="twin", condition="twin")
    unclamped = _selected_rows(key, records, scheme="maker_identity", source="part54a", model="qwen35-4b", dose=DOSE_ZERO,
                               kind="unclamped", condition="unclamped")
    down_count = sum(row[2].get("maker_identity") in ("another-real-lab", "invented-or-non-lab") for row in down)
    un_count = sum(row[2].get("maker_identity") in ("another-real-lab", "invented-or-non-lab") for row in unclamped)
    held_63d = (len(down) == REQUIRED_CELL_N and len(unclamped) == REQUIRED_CELL_N
                and down_count >= MAKER_DOWN_MIN and un_count <= MAKER_UNCLAMPED_MAX)
    _prediction_line("63d", held_63d,
                     f"-2 another-real/invented={down_count}/{len(down)} (>= {MAKER_DOWN_MIN}/{REQUIRED_CELL_N}); "
                     f"unclamped={un_count}/{len(unclamped)} (<= {MAKER_UNCLAMPED_MAX}/{REQUIRED_CELL_N})")
    output["63d"] = {"held": held_63d, "dose_minus_2": [down_count, len(down)],
                     "unclamped": [un_count, len(unclamped)]}

    # 63e: Haiku OFFWORLD counts stay within two of the frozen Part 54a Opus counts.
    comparisons = []
    for dose_index, dose in enumerate(PREDICTION_63E_DOSES):
        rows = _selected_rows(key, records, scheme="origin", source="part54a", model="qwen35-4b", dose=dose,
                              kind="twin", condition="twin")
        haiku_count = _count_label(rows, "origin", "OFFWORLD")
        opus_count = sum(isinstance(meta.get("existing_opus_label"), list)
                         and meta["existing_opus_label"]
                         and meta["existing_opus_label"][0] == "OFFWORLD" for _answer_id, meta, _record in rows)
        within = len(rows) == REQUIRED_CELL_N and abs(haiku_count - opus_count) <= OPUS_COUNT_TOLERANCE
        comparisons.append({"dose": dose, "n": len(rows), "haiku_offworld": haiku_count,
                            "opus_offworld": opus_count,
                            "registered_opus_reference": OPUS_REFERENCE_63E_COUNTS[dose_index],
                            "reference_matches": opus_count == OPUS_REFERENCE_63E_COUNTS[dose_index],
                            "within": within})
        print(f"63e dose {dose:+g}: {'within' if within else 'outside'}; "
              f"Haiku={haiku_count}/{len(rows)} Opus={opus_count}/{len(rows)} "
              f"(registered reference {OPUS_REFERENCE_63E_COUNTS[dose_index]}) "
              f"difference={abs(haiku_count - opus_count)} tolerance={OPUS_COUNT_TOLERANCE}")
    held_63e = len(comparisons) == len(PREDICTION_63E_DOSES) and all(row["within"] for row in comparisons)
    _prediction_line("63e", held_63e, f"within-tolerance cells={sum(row['within'] for row in comparisons)}/{len(comparisons)}")
    output["63e"] = {"held": held_63e, "comparisons": comparisons}
    return output


def answer_text_map(answer_rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["id"]: row["text"] for row in answer_rows}


def analyze(data_dir: Path, labels_path: Path) -> dict[str, Any]:
    answer_rows, key, _manifest = reader63.load_bundle(data_dir)
    labels = read_labels(labels_path)
    if labels.get("scoring_gate_passed") is not True or labels.get("calibration", {}).get("passed") is not True:
        raise AnalysisError("calibration gate failed; refusing to score any scheme")
    records = labels["records"]
    failures = failure_counts(key, records)
    print("Per-scheme unlabelled/failed counts:")
    print_failure_counts(failures)
    blocked = {scheme: row for scheme, row in failures.items()
               if not row["total"] or row["bad"] / row["total"] > MAX_BAD_RATE}
    if blocked:
        raise AnalysisError("per-scheme failed plus unlabelled rate exceeds 2%; refusing to score")
    print_cell_counts(key, records)
    proxy_counts, proxy_passed = keyword_views(key, records, answer_text_map(answer_rows))
    if not proxy_passed:
        raise AnalysisError("coarse keyword sanity gate failed; refusing to score predictions")
    predictions = evaluate_predictions(key, records)
    return {"failures": failures, "keyword_proxy": proxy_counts, "predictions": predictions}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--labels", type=Path, required=True,
                        help="labels63-BATCH_ID.json written by reader63.py collect")
    args = parser.parse_args(argv)
    try:
        result = analyze(args.data_dir, args.labels)
    except (AnalysisError, gather63.GatherError, OSError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    held = sum(bool(row.get("held")) for row in result["predictions"].values())
    print(f"RETURN_READY: Part 63 analysis complete; {held}/5 predictions held")


if __name__ == "__main__":
    main()
