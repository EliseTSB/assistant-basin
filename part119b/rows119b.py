#!/usr/bin/env python3
"""Dose-tagged Part 119b answer and fixed-path row inventories."""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "part119") not in sys.path:
    sys.path.insert(0, str(ROOT / "part119"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rows119  # noqa: E402 -- Part 92 unit and pair inventory is shared

DOSES = (0.2, 0.4, 0.6, 0.8)
SINGLE_DOSES = (0.2, 0.4, 0.6, 0.8, 1.13, 1.6)
PILOT_SEED = 11902
NEW_ANSWER_COUNT = 5577


def dose_text(dose: float) -> str:
    return f"{float(dose):g}"


def _push(unit_id: str, arm: str) -> dict:
    return {"unit_id": unit_id, "arm": arm}


def _tagged(row_id: str, row_type: str, dose: float, pushes: list[dict],
            source_row_id: str | None = None) -> dict:
    reused = source_row_id is not None
    return {"row_id": row_id, "row_type": row_type, "dose": float(dose),
            "pushes": pushes, "reused": reused,
            "source_row_id": source_row_id}


def _source_id(row: dict) -> str:
    if row["row_type"] in ("single", "random"):
        return f"{row['row_type']}:{row['pushes'][0]['unit_id']}"
    if row["row_type"] == "pair":
        a, b = row["unit_ids"]
        return f"pair:{a}+{b}"
    if row["row_type"] == "control":
        return f"control:{row['pushes'][0]['unit_id']}+R{row['pushes'][1]['unit_id']}"
    return "plain"


def _source_rows(units: list[dict]) -> list[dict]:
    """Build the Part 119 row shape for an explicitly reduced test inventory."""
    ordered = sorted(units, key=lambda unit: unit["id"])
    result = [{"row_id": "plain", "row_type": "plain", "pushes": [], "unit_ids": []}]
    result.extend({"row_id": f"single:{unit['id']}", "row_type": "single",
                  "pushes": [_push(unit["id"], "neuron")], "unit_ids": [unit["id"]]}
                 for unit in ordered)
    result.extend({"row_id": f"random:{unit['id']}", "row_type": "random",
                  "pushes": [_push(unit["id"], "random")], "unit_ids": [unit["id"]]}
                 for unit in ordered)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            a, b = sorted((first["id"], second["id"]))
            result.append({"row_id": f"pair:{a}+{b}", "row_type": "pair",
                           "pushes": [_push(a, "neuron"), _push(b, "neuron")],
                           "unit_ids": [a, b]})
    for first in ordered:
        for second in ordered:
            if first["id"] == second["id"]:
                continue
            result.append({"row_id": f"control:{first['id']}+R{second['id']}",
                           "row_type": "control",
                           "pushes": [_push(first["id"], "neuron"),
                                      _push(second["id"], "random")],
                           "unit_ids": [first["id"], second["id"]]})
    return result


def _registered_inventory(units: list[dict] | None) -> bool:
    if units is None:
        return True
    unit_ids = [str(unit["id"]) for unit in units]
    registered_ids = [str(unit["id"]) for unit in rows119.load_units()]
    return len(unit_ids) == len(registered_ids) and set(unit_ids) == set(registered_ids)


def all_answer_rows(units: list[dict] | None = None) -> list[dict]:
    """All planned rows, including the Part 119 0.8 rows marked for reuse."""
    registered_inventory = _registered_inventory(units)
    units = rows119.load_units() if units is None else units
    source = rows119.all_rows(units) if registered_inventory else _source_rows(units)
    single_rows = [row for row in source if row["row_type"] in ("single", "random")]
    pair_rows = [row for row in source if row["row_type"] == "pair"]
    control_rows = [row for row in source if row["row_type"] == "control"]
    result = []

    for row in single_rows:
        arm = row["pushes"][0]["arm"]
        unit_id = row["pushes"][0]["unit_id"]
        for dose in SINGLE_DOSES:
            suffix = f"{row['row_type']}@{dose_text(dose)}:{unit_id}"
            source_id = _source_id(row) if dose == 0.8 else None
            result.append(_tagged(suffix, row["row_type"], dose,
                                  [_push(unit_id, arm)], source_id))

    for row in pair_rows:
        a, b = row["unit_ids"]
        for dose in DOSES:
            source_id = _source_id(row) if dose == 0.8 else None
            result.append(_tagged(f"pair@{dose_text(dose)}:{a}+{b}", "pair", dose,
                                  [_push(a, "neuron"), _push(b, "neuron")], source_id))

    for row in control_rows:
        a, b = (push["unit_id"] for push in row["pushes"])
        for dose in (0.4, 0.8):
            source_id = _source_id(row) if dose == 0.8 else None
            result.append(_tagged(f"control@{dose_text(dose)}:{a}+{b}", "control", dose,
                                  [_push(a, "neuron"), _push(b, "random")], source_id))

    for row in pair_rows:
        a, b = row["unit_ids"]
        for dose in (0.4, 0.8):
            result.append(_tagged(f"randpair@{dose_text(dose)}:{a}+{b}", "randpair", dose,
                                  [_push(a, "random"), _push(b, "random")]))

    ids = [row["row_id"] for row in result]
    if len(set(ids)) != len(ids):
        raise AssertionError("planned answer row ids are not unique")
    if registered_inventory and len(result) != 7878:
        raise AssertionError(f"expected 7,878 unique planned answers, found {len(result)}")
    new_count = sum(not row["reused"] for row in result)
    if registered_inventory and new_count != NEW_ANSWER_COUNT:
        raise AssertionError(f"expected {NEW_ANSWER_COUNT:,} new answers, found {new_count:,}")
    reused = sum(row["reused"] for row in result)
    if registered_inventory and reused != 2301:
        raise AssertionError(f"expected 2,301 Part 119 answers reused, found {reused}")
    return result


def pilot_answer_rows(units: list[dict] | None = None) -> list[dict]:
    """Plain, all single doses, seeded 20-pair cells, and 20 random pairs."""
    registered_inventory = _registered_inventory(units)
    units = rows119.load_units() if units is None else units
    all_rows = all_answer_rows(units)
    lookup = {row["row_id"]: row for row in all_rows}
    base = rows119.all_rows(units) if registered_inventory else _source_rows(units)
    pairs = [row for row in base if row["row_type"] == "pair"]
    rng = random.Random(PILOT_SEED)
    chosen: list[dict] = []
    for row in all_rows:
        if row["row_type"] in ("single", "random"):
            chosen.append(row)
    pair_sample_size = min(20, len(pairs))
    for dose in (0.4, 0.6):
        for row in rng.sample(pairs, pair_sample_size):
            a, b = row["unit_ids"]
            chosen.append(lookup[f"pair@{dose_text(dose)}:{a}+{b}"])
    for row in rng.sample(pairs, pair_sample_size):
        a, b = row["unit_ids"]
        chosen.append(lookup[f"randpair@0.8:{a}+{b}"])
    # The plain answer is read from Part 119 and is not copied to the new answer file.
    chosen.insert(0, {"row_id": "plain", "row_type": "plain", "dose": 0.0,
                      "pushes": [], "reused": True, "source_row_id": "plain"})
    if len({row["row_id"] for row in chosen}) != len(chosen):
        raise AssertionError("pilot answer row ids are not unique")
    if registered_inventory and len(chosen) != 529:
        raise AssertionError("pilot answer rows must be 468 singles, 40 real pairs, 20 random pairs, and plain")
    return chosen


def all_path2_rows(units: list[dict] | None = None) -> list[dict]:
    """Directed real and matched-random second pushes for every ordered pair."""
    registered_inventory = _registered_inventory(units)
    units = rows119.load_units() if units is None else units
    ordered = sorted(units, key=lambda unit: unit["id"])
    rows = []
    for first in ordered:
        for second in ordered:
            if first["id"] == second["id"]:
                continue
            a, b = first["id"], second["id"]
            rows.append({"key": f"real:{a}+{b}", "first": a, "second": b,
                         "kind": "real", "pushes": [_push(a, "neuron"), _push(b, "neuron")]})
            rows.append({"key": f"control:{a}+{b}", "first": a, "second": b,
                         "kind": "control", "pushes": [_push(a, "neuron"), _push(b, "random")]})
    if len({row["key"] for row in rows}) != 2 * len(ordered) * (len(ordered) - 1):
        raise AssertionError("Path 2 row keys are not unique")
    if registered_inventory and len(rows) != 2964:
        raise AssertionError("Path 2 must have 1,482 directed real and 1,482 control rows")
    return rows


def pilot_path2_rows(units: list[dict] | None = None) -> list[dict]:
    """Two seeded first units, each with eight second units in both row classes."""
    registered_inventory = _registered_inventory(units)
    units = rows119.load_units() if units is None else units
    ordered = sorted(units, key=lambda unit: unit["id"])
    rng = random.Random(PILOT_SEED)
    firsts = rng.sample(ordered, min(2, len(ordered)))
    rows = []
    for first in firsts:
        available = [unit for unit in ordered if unit["id"] != first["id"]]
        seconds = rng.sample(available, min(8, len(available)))
        for second in seconds:
            a, b = first["id"], second["id"]
            rows.append({"key": f"real:{a}+{b}", "first": a, "second": b,
                         "kind": "real", "pushes": [_push(a, "neuron"), _push(b, "neuron")]})
            rows.append({"key": f"control:{a}+{b}", "first": a, "second": b,
                         "kind": "control", "pushes": [_push(a, "neuron"), _push(b, "random")]})
    if len({row["key"] for row in rows}) != len(rows):
        raise AssertionError("pilot Path 2 row keys are not unique")
    if registered_inventory and len(rows) != 32:
        raise AssertionError("pilot Path 2 must have two first units and eight second units each")
    return rows


def path2_key(kind: str, dose: float, first: str, second: str) -> str:
    return f"{kind}@{dose_text(dose)}:{first}+{second}"


if __name__ == "__main__":
    units = rows119.load_units()
    rows = all_answer_rows(units)
    print({"planned_answers": len(rows), "new_answers": sum(not row["reused"] for row in rows),
           "reused_answers": sum(row["reused"] for row in rows),
           "pilot_answers": len(pilot_answer_rows(units)),
           "path2_rows": len(all_path2_rows(units)),
           "pilot_path2_rows": len(pilot_path2_rows(units))})
