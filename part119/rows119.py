#!/usr/bin/env python3
"""Deterministic Part 119 rows and blinded-reader sample."""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_NEURONS = "results/part92/step2/neurons.json"
DEFAULT_HAIKU_KINDS = "results/part92/step2/kinds.json"
DEFAULT_DS_KINDS = "results/part92b/score92b.json"
PILOT_SEED = 11901
READER_SEED = 11900
UNIT_COUNT = 39
ROW_COUNT = 2302


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _read_json(path: str | Path):
    with repo_path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_units(neurons_path: str | Path = DEFAULT_NEURONS,
               kinds_path: str | Path = DEFAULT_HAIKU_KINDS,
               kinds_ds_path: str | Path = DEFAULT_DS_KINDS) -> list[dict]:
    """Load the 39 Haiku-classified units and their Part 92b reader kinds."""
    neurons = _read_json(neurons_path)
    haiku_data = _read_json(kinds_path)
    ds_data = _read_json(kinds_ds_path)
    haiku = haiku_data.get("kinds", haiku_data)
    comparison = ds_data.get("kind_comparison", {})
    comparisons = comparison.get("units", [])
    if not isinstance(neurons, list) or not isinstance(haiku, dict) or not isinstance(comparisons, list):
        raise ValueError("Part 92 and Part 92b inputs have unexpected JSON shapes")
    ds_kinds = {}
    for row in comparisons:
        unit_id = str(row["id"])
        if unit_id in ds_kinds:
            raise ValueError(f"duplicate Part 92b kind row for {unit_id!r}")
        ds_kinds[unit_id] = str(row["ds_kind"])

    units = []
    seen = set()
    for source in neurons:
        unit_id = str(source["id"])
        haiku_kind = haiku.get(unit_id)
        if haiku_kind not in ("slot", "concept"):
            continue
        if unit_id in seen:
            raise ValueError(f"duplicate classified Part 92 unit id {unit_id!r}")
        seen.add(unit_id)
        ds_kind = ds_kinds.get(unit_id)
        if ds_kind is None:
            raise ValueError(f"Part 92b has no kind for classified unit {unit_id!r}")
        if ds_kind not in ("slot", "concept", "unclassified"):
            raise ValueError(f"unexpected Part 92b kind {ds_kind!r} for {unit_id!r}")
        for field in ("layer", "neuron", "persona", "desc", "control_seed"):
            if field not in source:
                raise ValueError(f"Part 92 unit {unit_id!r} is missing {field!r}")
        group = ("agreed_slot" if haiku_kind == ds_kind == "slot" else
                 "agreed_concept" if haiku_kind == ds_kind == "concept" else "disputed")
        units.append({**source, "id": unit_id, "layer": int(source["layer"]),
                      "neuron": int(source["neuron"]),
                      "control_seed": int(source["control_seed"]),
                      "haiku_kind": haiku_kind, "ds_kind": ds_kind, "group": group})

    counts = {group: sum(unit["group"] == group for unit in units)
              for group in ("agreed_slot", "agreed_concept", "disputed")}
    if len(units) != UNIT_COUNT or counts != {
        "agreed_slot": 21, "agreed_concept": 8, "disputed": 10
    }:
        raise ValueError(f"expected 39 units split 21/8/10, found {len(units)} and {counts}")
    if sum(unit["haiku_kind"] == "slot" for unit in units) != 21 or \
            sum(unit["haiku_kind"] == "concept" for unit in units) != 18:
        raise ValueError("Part 92 Haiku kinds must contain 21 slots and 18 concepts")
    return units


def _row(row_id: str, row_type: str, pushes: list[tuple[str, str]]) -> dict:
    return {"row_id": row_id, "row_type": row_type,
            "pushes": [{"unit_id": unit_id, "arm": arm} for unit_id, arm in pushes],
            "unit_ids": [unit_id for unit_id, _arm in pushes]}


def all_rows(units: list[dict] | None = None) -> list[dict]:
    units = load_units() if units is None else units
    ordered = sorted(units, key=lambda unit: unit["id"])
    rows = [_row("plain", "plain", [])]
    rows.extend(_row(f"single:{unit['id']}", "single", [(unit["id"], "neuron")])
                for unit in ordered)
    rows.extend(_row(f"random:{unit['id']}", "random", [(unit["id"], "random")])
                for unit in ordered)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            a, b = sorted((first["id"], second["id"]))
            rows.append(_row(f"pair:{a}+{b}", "pair", [(a, "neuron"), (b, "neuron")]))
    for first in ordered:
        for second in ordered:
            if first["id"] == second["id"]:
                continue
            rows.append(_row(f"control:{first['id']}+R{second['id']}", "control",
                             [(first["id"], "neuron"), (second["id"], "random")]))
    if len(rows) != ROW_COUNT:
        raise AssertionError(f"expected {ROW_COUNT} rows, got {len(rows)}")
    if len({row["row_id"] for row in rows}) != ROW_COUNT:
        raise AssertionError("row ids are not unique")
    return rows


def pilot_rows(units: list[dict] | None = None) -> list[dict]:
    """Return plain, all singles, and seeded samples of 20 pairs and 40 controls."""
    rows = all_rows(units)
    plain_singles = [row for row in rows if row["row_type"] in ("plain", "single", "random")]
    pairs = [row for row in rows if row["row_type"] == "pair"]
    controls = [row for row in rows if row["row_type"] == "control"]
    rng = random.Random(PILOT_SEED)
    selected = plain_singles + rng.sample(pairs, 20) + rng.sample(controls, 40)
    if len(selected) != 139 or len({row["row_id"] for row in selected}) != len(selected):
        raise AssertionError("pilot row counts or ids are invalid")
    return selected


def pair_class(row: dict, units_by_id: dict[str, dict], scheme: str = "agreed") -> str:
    """Classify a real pair under the agreed DS/Haiku grouping or Haiku alone."""
    if row["row_type"] != "pair":
        raise ValueError("pair_class requires a real pair row")
    first, second = (units_by_id[unit_id] for unit_id in row["unit_ids"])
    if scheme == "agreed":
        left = first["group"]
        right = second["group"]
        if "disputed" in (left, right):
            return "involving_disputed"
        left = "slot" if left == "agreed_slot" else "concept"
        right = "slot" if right == "agreed_slot" else "concept"
    elif scheme == "haiku":
        left, right = first["haiku_kind"], second["haiku_kind"]
    else:
        raise ValueError("scheme must be 'agreed' or 'haiku'")
    if left == right:
        return f"{left}_x_{right}"
    return "slot_x_concept"


def reader_sample(units: list[dict] | None = None) -> list[dict]:
    """Select the registered 148 rows and assign X/Y order from seed 11900."""
    units = load_units() if units is None else units
    by_id = {unit["id"]: unit for unit in units}
    pairs_and_controls = [row for row in all_rows(units)
                          if row["row_type"] in ("pair", "control")]
    groups = {name: [] for name in ("concept_x_concept", "slot_x_concept", "slot_x_slot", "control")}
    for row in pairs_and_controls:
        if row["row_type"] == "control":
            groups["control"].append(row)
        else:
            category = pair_class(row, by_id, "agreed")
            if category in groups:
                groups[category].append(row)
    expected = {"concept_x_concept": 28, "slot_x_concept": 168,
                "slot_x_slot": 210, "control": 1482}
    if {name: len(rows) for name, rows in groups.items()} != expected:
        raise AssertionError("reader-sample source classes do not match the registered design")

    rng = random.Random(READER_SEED)
    chosen = groups["concept_x_concept"] + rng.sample(groups["slot_x_concept"], 40)
    chosen += rng.sample(groups["slot_x_slot"], 40)
    chosen += rng.sample(groups["control"], 40)
    items = []
    for row in chosen:
        a, b = row["unit_ids"]
        if rng.randrange(2):
            x_id, y_id = b, a
        else:
            x_id, y_id = a, b
        items.append({"row_id": row["row_id"], "row_type": row["row_type"],
                      "sample_class": ("control" if row["row_type"] == "control"
                                       else pair_class(row, by_id, "agreed")),
                      "x_unit_id": x_id, "y_unit_id": y_id})
    counts = {name: sum(item["sample_class"] == name for item in items) for name in expected}
    if len(items) != 148 or counts != {
        "concept_x_concept": 28, "slot_x_concept": 40,
        "slot_x_slot": 40, "control": 40,
    }:
        raise AssertionError(f"reader sample counts are invalid: {counts}")
    return items


def rows_by_id(rows: list[dict] | None = None) -> dict[str, dict]:
    return {row["row_id"]: row for row in (all_rows() if rows is None else rows)}


def interaction_rows(units: list[dict] | None = None, stage: str = "main") -> list[dict]:
    """Build Path 2's directed real/control rows, keyed by first, second, and kind."""
    units = load_units() if units is None else units
    if stage not in ("pilot", "main"):
        raise ValueError("stage must be 'pilot' or 'main'")
    rows = []
    if stage == "pilot":
        for row in pilot_rows(units):
            if row["row_type"] not in ("pair", "control"):
                continue
            first, second = row["unit_ids"]
            kind = "real" if row["row_type"] == "pair" else "control"
            pushes = [(first, "neuron"),
                      (second, "neuron" if kind == "real" else "random")]
            rows.append({"key": f"{kind}:{first}+{second}", "first": first,
                         "second": second, "kind": kind,
                         "pushes": [{"unit_id": unit_id, "arm": arm} for unit_id, arm in pushes]})
    else:
        ordered = sorted(units, key=lambda unit: unit["id"])
        for first in ordered:
            for second in ordered:
                if first["id"] == second["id"]:
                    continue
                a, b = first["id"], second["id"]
                rows.append({"key": f"real:{a}+{b}", "first": a, "second": b,
                             "kind": "real", "pushes": [
                                 {"unit_id": a, "arm": "neuron"},
                                 {"unit_id": b, "arm": "neuron"},
                             ]})
                rows.append({"key": f"control:{a}+{b}", "first": a, "second": b,
                             "kind": "control", "pushes": [
                                 {"unit_id": a, "arm": "neuron"},
                                 {"unit_id": b, "arm": "random"},
                             ]})
    expected = 60 if stage == "pilot" else 2964
    if len(rows) != expected or len({row["key"] for row in rows}) != expected:
        raise AssertionError(f"expected {expected} unique Path 2 rows, found {len(rows)}")
    return rows


if __name__ == "__main__":
    units = load_units()
    print(json.dumps({"units": len(units), "rows": len(all_rows(units)),
                      "pilot_rows": len(pilot_rows(units)),
                      "reader_items": len(reader_sample(units))}, sort_keys=True))
