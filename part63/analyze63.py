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
REQUIRED_CELL_N = 24  # Predictions 63b-e use the three-question, eight-answer Part 54 cells.
EARTH_BASE_MIN = 3  # Fixed prediction 63b.
PLANET_BASE_MAX = 1  # Fixed prediction 63c.
PLANET_PLUS_ONE_MIN = 20  # Fixed prediction 63c.
MAKER_DOWN_MIN = 10  # Fixed prediction 63d.
MAKER_UNCLAMPED_MAX = 2  # Fixed prediction 63d.
OPUS_COUNT_TOLERANCE = 2  # Fixed prediction 63e.
DOSE_NEG_TWO = -2.0
DOSE_POS_HALF = 0.5
DOSE_POS_ONE = 1.0
PREDICTION_63E_DOSES = (-1.0, 0.5, 1.0, 1.5, 2.0)
OPUS_REFERENCE_63E_COUNTS = (0, 0, 1, 19, 24)  # Registered Part 54a counts, in dose order above.
COARSE_MIN_OVERLAPS = 1
DUPLICATE_AGREEMENT_THRESHOLD = 0.90

PROXY_OFFWORLD = re.compile(
    r"\b(?:alien|extraterrestrial|planet|galax(?:y|ies)|star|space|interstellar|species)\b", re.I
)
PROXY_SCIFI_EXTRA = re.compile(
    r"\b(?:starship|spaceship|colony|orbit|androids?|robots?|cyber\w*|futur\w*|mars|nebula|asteroid|quantum)\b", re.I
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
    print("scheme | total | failed | unlabelled | bad rate | hand-label gate")
    for scheme, row in counts.items():
        rate = row["bad"] / row["total"] if row["total"] else 1.0
        print(f"{scheme} | {row['total']} | {row['failed']} | {row['unlabelled']} | {rate:.3%} | not gated by hand labels")


def print_cell_counts(key: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]]) -> None:
    schemes = (
        ("planet_frame", ("none", "earth", "other-planet-or-space")),
        ("maker_identity", ("own-lab", "another-real-lab", "invented-or-non-lab", "no-maker")),
        ("origin", ("EARTH", "OFFWORLD", "NEITHER")),
    )
    for scheme, labels in schemes:
        cells = _cell_label_counts(key, records, scheme)
        print(f"\nPer-cell counts: {scheme}")
        print("source | model | dose | kind | condition | unit | n | valid | " + " ".join(labels)
              + " | bad | hand-label gate")
        for cell in sorted(cells, key=_sort_cell):
            counts = cells[cell]
            answers = [answer_id for answer_id, meta in key.items()
                       if answer_id.startswith("a") and _cell_key(meta) == cell]
            bad = sum(_scheme_status(records.get(answer_id, {}), scheme) != "ok"
                      or not isinstance(records.get(answer_id, {}).get(scheme), str) for answer_id in answers)
            print(" | ".join(str(value) for value in cell), "|", counts["n"], "|",
                  counts["valid"], "|", " ".join(str(counts[label]) for label in labels), "|", bad,
                  "| not gated by hand labels")


def _exact_scheme_label(record: dict[str, Any], scheme: str) -> Any:
    if _scheme_status(record, scheme) != "ok":
        return None
    if scheme == "maker_identity":
        name, kind = record.get("maker_name"), record.get("maker_kind")
        return (name, kind) if (name is None or isinstance(name, str)) and isinstance(kind, str) else None
    value = record.get(scheme)
    return value if isinstance(value, str) else None


def duplicate_consistency(pairs: list[dict[str, str]], records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compare exact parsed scheme labels for each answer and its seeded repeat."""
    output: dict[str, dict[str, Any]] = {}
    for scheme in ("planet_frame", "maker_identity", "origin"):
        matches = valid = 0
        for pair in pairs:
            left = _exact_scheme_label(records.get(pair["answer_id"], {}), scheme)
            right = _exact_scheme_label(records.get(pair["duplicate_id"], {}), scheme)
            if left is None or right is None:
                continue
            valid += 1
            matches += left == right
        rate = matches / valid if valid else None
        output[scheme] = {"matches": matches, "n": valid, "agreement": rate,
                          "threshold": DUPLICATE_AGREEMENT_THRESHOLD,
                          "unreliable_reader": rate is None or rate < DUPLICATE_AGREEMENT_THRESHOLD}
    return output


def print_duplicate_consistency(counts: dict[str, dict[str, Any]]) -> None:
    print("\nDuplicate consistency (exact label match; seeded repeated answers):")
    print("scheme | exact agreement | matches | n | reader status | hand-label gate")
    for scheme, row in counts.items():
        rate = row["agreement"]
        shown = "unavailable" if rate is None else f"{rate:.1%}"
        status = "unreliable reader" if row["unreliable_reader"] else "agreement >= 90%"
        print(f"{scheme} | {shown} | {row['matches']} | {row['n']} | {status} | not gated by hand labels")


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
    print("source | model | dose | kind | condition | unit | n | off-world | sci-fi | Earth | hand-label gate")
    for cell in sorted(proxy_cells, key=_sort_cell):
        counts = proxy_cells[cell]
        if not (counts["offworld"] or counts["sci_fi"] or counts["earth"]):
            continue
        print(" | ".join(str(value) for value in cell), "|", counts["n"], "|",
              counts["offworld"], "|", counts["sci_fi"], "|", counts["earth"],
              "| not gated by hand labels")
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
                   *, scheme: str, text_by_id: dict[str, str] | None = None,
                   dedupe_by_text: bool = False, **criteria: Any) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    rows = []
    seen_texts: set[str] = set()
    for answer_id, meta in key.items():
        if not answer_id.startswith("a") or any(meta.get(name) != value for name, value in criteria.items()):
            continue
        if dedupe_by_text:
            if text_by_id is None or answer_id not in text_by_id:
                raise AnalysisError("text-deduplicated selection requires every answer text")
            text = text_by_id[answer_id]
            if text in seen_texts:
                continue
            seen_texts.add(text)
        record = records.get(answer_id, {})
        if _scheme_status(record, scheme) == "ok":
            rows.append((answer_id, meta, record))
    return rows


def _count_label(rows: list[tuple[str, dict[str, Any], dict[str, Any]]], scheme: str,
                 label: str) -> int:
    return sum(row[2].get(scheme) == label for row in rows)


def _prediction_line(name: str, held: bool, numbers: str) -> None:
    print(f"{name}: {'HELD' if held else 'FAILED'} ({numbers})")


def evaluate_predictions(key: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]],
                         text_by_id: dict[str, str],
                         reliability: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {}

    # 63a: shared Part 54a answer-level origin labels, compared to the existing Opus label.
    shared_rows = _selected_rows(key, records, scheme="origin", source="part54a", model="qwen35-4b")
    expected_63a = sum(meta.get("source") == "part54a" and meta.get("model") == "qwen35-4b"
                       for answer_id, meta in key.items() if answer_id.startswith("a"))
    shared_haiku: list[str] = []
    shared_opus: list[str] = []
    for _answer_id, meta, record in shared_rows:
        opus = meta.get("existing_opus_label")
        if isinstance(opus, list) and opus and opus[0] in reader63.ORIGIN_LABELS:
            shared_haiku.append(record["origin"])
            shared_opus.append(opus[0])
    if len(shared_rows) != expected_63a or len(shared_haiku) != expected_63a or len(shared_opus) != expected_63a:
        raise AnalysisError(f"63a full paired denominator is incomplete: {len(shared_haiku)}/{expected_63a}")
    kappa = cohen_kappa(shared_haiku, shared_opus)
    held_63a = kappa is not None and kappa >= KAPPA_THRESHOLD
    _prediction_line("63a", held_63a, f"kappa={kappa if kappa is not None else 'undefined'}; n={len(shared_haiku)}; threshold={KAPPA_THRESHOLD:.2f}")
    output["63a"] = {"held": held_63a, "kappa": kappa, "n": len(shared_haiku), "threshold": KAPPA_THRESHOLD}

    # Registered 63b counts the 24 registered rows; distinct texts are a sensitivity view.
    earth_cells: list[tuple[tuple[Any, ...], int, int, int]] = []
    for cell, counts in _cell_label_counts(key, records, "origin").items():
        source, model, _dose, kind, _condition, _unit = cell
        if source == "part54b" and model == "qwen3-4b" and kind == "control":
            earth_cells.append((cell, counts["n"], counts["valid"], counts["EARTH"]))
    unclamped_63b = _selected_rows(key, records, scheme="origin", source="part54b", model="qwen3-4b",
                                   dose=-3.0, kind="unclamped")
    distinct_63b = _selected_rows(key, records, scheme="origin", text_by_id=text_by_id,
                                  dedupe_by_text=True, source="part54b", model="qwen3-4b",
                                  dose=-3.0, kind="unclamped")
    baseline_earth = _count_label(unclamped_63b, "origin", "EARTH")
    earth_cells.append((("part54b", "qwen3-4b", "unclamped", "unclamped", "unclamped", None),
                        len(unclamped_63b), len(unclamped_63b), baseline_earth))
    matching_earth_cells = [(cell, n, valid, earth) for cell, n, valid, earth in earth_cells
                            if n > 0 and valid == n and earth >= EARTH_BASE_MIN
                            and n == REQUIRED_CELL_N]
    held_63b = bool(matching_earth_cells)
    first = matching_earth_cells[0] if matching_earth_cells else None
    _prediction_line("63b", held_63b,
                     f"qualifying cells={len(matching_earth_cells)}/{len(earth_cells)}; "
                     f"example={first[0] if first else 'none'} EARTH={first[3] if first else 0}/"
                     f"{first[1] if first else REQUIRED_CELL_N} valid={first[2] if first else 0}; "
                     f"threshold=at least {EARTH_BASE_MIN} of {REQUIRED_CELL_N} rows")
    output["63b"] = {"held": held_63b, "qualifying_cells": len(matching_earth_cells), "cells": len(earth_cells),
                     "unclamped_rows": [baseline_earth, len(unclamped_63b)],
                     "distinct_text_sensitivity": [_count_label(distinct_63b, "origin", "EARTH"), len(distinct_63b)]}

    # 63c: the real key carries source doses on copied baselines, so KIND selects and text dedupes them.
    baseline = _selected_rows(key, records, scheme="planet_frame", source="part54b", model="qwen3-4b",
                              dose=-3.0, kind="unclamped")
    distinct_baseline = _selected_rows(key, records, scheme="planet_frame", text_by_id=text_by_id,
                                       dedupe_by_text=True, source="part54b", model="qwen3-4b", dose=-3.0,
                                       kind="unclamped")
    plus_half = _selected_rows(key, records, scheme="planet_frame", source="part54b", model="qwen3-4b", dose=DOSE_POS_HALF,
                               kind="twin", condition="twin")
    plus_one = _selected_rows(key, records, scheme="planet_frame", source="part54b", model="qwen3-4b", dose=DOSE_POS_ONE,
                              kind="twin", condition="twin")
    base_count = _count_label(baseline, "planet_frame", "other-planet-or-space")
    half_count = _count_label(plus_half, "planet_frame", "other-planet-or-space")
    one_count = _count_label(plus_one, "planet_frame", "other-planet-or-space")
    held_63c = (len(baseline) > 0 and len(plus_half) == REQUIRED_CELL_N
                and len(plus_one) == REQUIRED_CELL_N and base_count <= PLANET_BASE_MAX
                and one_count >= PLANET_PLUS_ONE_MIN and half_count <= one_count)
    distinct_base_count = _count_label(distinct_baseline, "planet_frame", "other-planet-or-space")
    _prediction_line("63c", held_63c,
                     f"unclamped rows={base_count}/{len(baseline)} (<= {PLANET_BASE_MAX}/{REQUIRED_CELL_N}); "
                     f"+0.5={half_count}/{len(plus_half)}; +1={one_count}/{len(plus_one)} "
                     f"(>= {PLANET_PLUS_ONE_MIN}/{REQUIRED_CELL_N}, strict rise required)")
    output["63c"] = {"held": held_63c, "unclamped": [base_count, len(baseline)],
                     "distinct_text_sensitivity": [distinct_base_count, len(distinct_baseline)],
                     "+0.5": [half_count, len(plus_half)], "+1": [one_count, len(plus_one)]}

    # 63d uses the registered own-lab aliases; strict exact-name remains sensitivity.
    down = _selected_rows(key, records, scheme="maker_identity", source="part54a", model="qwen35-4b", dose=DOSE_NEG_TWO,
                          kind="twin", condition="twin")
    unclamped = _selected_rows(key, records, scheme="maker_identity", source="part54a", model="qwen35-4b",
                               dose=-3.0, kind="unclamped")
    distinct_unclamped = _selected_rows(key, records, scheme="maker_identity", text_by_id=text_by_id,
                                         dedupe_by_text=True, source="part54a", model="qwen35-4b",
                                         dose=-3.0, kind="unclamped")

    def maker_counts(rows: list[tuple[str, dict[str, Any], dict[str, Any]]], field: str) -> int:
        return sum(row[2].get(field) in ("another-real-lab", "invented-or-non-lab") for row in rows)

    registered_down = maker_counts(down, "maker_identity")
    registered_unclamped = maker_counts(unclamped, "maker_identity")
    strict_down = maker_counts(down, "maker_identity_alias")
    strict_unclamped = maker_counts(unclamped, "maker_identity_alias")
    registered_distinct_unclamped = maker_counts(distinct_unclamped, "maker_identity")
    registered_held = (len(down) == REQUIRED_CELL_N and len(unclamped) == REQUIRED_CELL_N
                       and registered_down >= MAKER_DOWN_MIN and registered_unclamped <= MAKER_UNCLAMPED_MAX)
    strict_held = (len(down) == REQUIRED_CELL_N and len(unclamped) == REQUIRED_CELL_N
                   and strict_down >= MAKER_DOWN_MIN and strict_unclamped <= MAKER_UNCLAMPED_MAX)
    _prediction_line("63d registered (own-lab aliases)", registered_held,
                     f"-2 another-real/invented={registered_down}/{len(down)} (>= {MAKER_DOWN_MIN}/{REQUIRED_CELL_N}); "
                     f"unclamped={registered_unclamped}/{len(unclamped)} (<= {MAKER_UNCLAMPED_MAX}/{REQUIRED_CELL_N})")
    _prediction_line("63d strict exact-name sensitivity", strict_held,
                     f"-2 another-real/invented={strict_down}/{len(down)}; unclamped={strict_unclamped}/{len(unclamped)}")
    output["63d"] = {
        "registered_reading": "own-lab aliases: alibaba, alibaba cloud, tongyi, tongyi lab, qwen team, 通义, 阿里",
        "held": registered_held,
        "registered_aliases": {"held": registered_held, "dose_minus_2": [registered_down, len(down)],
                               "unclamped": [registered_unclamped, len(unclamped)],
                               "distinct_text_sensitivity": [registered_distinct_unclamped, len(distinct_unclamped)]},
        "strict_exact_name_sensitivity": {"held": strict_held, "dose_minus_2": [strict_down, len(down)],
                                          "unclamped": [strict_unclamped, len(unclamped)]},
    }

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
    scheme_for_prediction = {"63a": "origin", "63b": "origin", "63c": "planet_frame",
                             "63d": "maker_identity", "63e": "origin"}
    print("\nPrediction table | reader reliability")
    print("prediction | result | reader status | hand-label gate")
    for name, scheme in scheme_for_prediction.items():
        unreliable = bool(reliability and reliability.get(scheme, {}).get("unreliable_reader"))
        status = "unreliable reader" if unreliable else "agreement >= 90%"
        output[name]["reader_status"] = status
        output[name]["unreliable_reader"] = unreliable
        print(f"{name} | {'HELD' if output[name].get('held') else 'FAILED'} | {status} | not gated by hand labels")
    if "63d" in output:
        output["63d"]["registered_aliases"]["reader_status"] = output["63d"]["reader_status"]
        output["63d"]["strict_exact_name_sensitivity"]["unreliable_reader"] = output["63d"]["unreliable_reader"]
    return output


def answer_text_map(answer_rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["id"]: row["text"] for row in answer_rows}


def analyze(data_dir: Path, labels_path: Path) -> dict[str, Any]:
    answer_rows, key, _manifest = reader63.load_bundle(data_dir)
    labels = read_labels(labels_path)
    if labels.get("calibration", {}).get("passed") is not True:
        raise AnalysisError("calibration gate failed; refusing to score any scheme")
    records = labels["records"]
    failures = failure_counts(key, records)
    override = labels.get("gate_override")
    actual_failed = sorted(reader63._failed_or_unlabelled_ids(records))
    override_valid = (isinstance(override, dict)
                      and override.get("amendment") == 3
                      and override.get("rule") == reader63.AMENDMENT_3_RULE
                      and override.get("residual_failed_ids") == actual_failed
                      and labels.get("residual_failed_ids") == actual_failed
                      and labels.get("failed_or_unlabelled_ids") == actual_failed
                      and labels.get("scoring_gate_passed") is True
                      and reader63._amendment3_gate_passes(
                          labels.get("calibration", {}).get("passed") is True,
                          len(actual_failed), labels.get("request_count", 0)))
    if override is not None and not override_valid:
        raise AnalysisError("gate_override is invalid or its residual failure list/gate does not match labels")
    if override_valid:
        print(f"Amendment 3 gate override: {override['rule']} "
              f"({len(actual_failed)} residual IDs; strict gate={labels.get('strict_gate')!r})")
    print("Per-scheme unlabelled/failed counts:")
    print_failure_counts(failures)
    blocked = {scheme: row for scheme, row in failures.items() if not row["total"] or row["bad"] > 0}
    if blocked and not override_valid:
        raise AnalysisError("failed or unlabelled answers remain after the one retry; refusing to score")
    if labels.get("scoring_gate_passed") is not True and not override_valid:
        raise AnalysisError("reader scoring gate failed after collection; refusing to score")
    duplicate_meta = labels.get("duplicate_consistency", {})
    duplicate_pairs = duplicate_meta.get("pairs") if isinstance(duplicate_meta, dict) else None
    if (not isinstance(duplicate_pairs, list)
            or (not override_valid and len(duplicate_pairs) != reader63.DUPLICATE_COUNT)):
        raise AnalysisError("labels do not contain the required seeded duplicate pairs")
    agreements = duplicate_consistency(duplicate_pairs, records)
    if not override_valid and any(row["n"] != reader63.DUPLICATE_COUNT for row in agreements.values()):
        raise AnalysisError("duplicate consistency is missing one or more valid pair labels")
    print_duplicate_consistency(agreements)
    print_cell_counts(key, records)
    proxy_counts, proxy_passed = keyword_views(key, records, answer_text_map(answer_rows))
    if not proxy_passed:
        raise AnalysisError("coarse keyword sanity gate failed; refusing to score predictions")
    predictions = evaluate_predictions(key, records, answer_text_map(answer_rows), agreements)
    return {"failures": failures, "keyword_proxy": proxy_counts,
            "duplicate_consistency": agreements, "predictions": predictions}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--labels", type=Path, required=True,
                        help="labels JSON written by reader63.py collect or reparse")
    args = parser.parse_args(argv)
    try:
        result = analyze(args.data_dir, args.labels)
    except (AnalysisError, gather63.GatherError, OSError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    held = sum(bool(row.get("held")) for row in result["predictions"].values())
    print(f"RETURN_READY: Part 63 analysis complete; {held}/5 predictions held")


if __name__ == "__main__":
    main()
