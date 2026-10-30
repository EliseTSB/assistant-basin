#!/usr/bin/env python3
"""Deterministic 312-row Path 2 sample for Part 119c."""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "part119b") not in sys.path:
    sys.path.insert(0, str(ROOT / "part119b"))
if str(ROOT / "part119") not in sys.path:
    sys.path.insert(0, str(ROOT / "part119"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rows119b  # noqa: E402

SEED = 11906
DOSES = (0.8, 0.4, 0.2, 0.1, 0.05)
FIRST_UNIT_COUNT = 39
SECOND_UNITS_PER_CLASS = 4
ROW_COUNT = FIRST_UNIT_COUNT * SECOND_UNITS_PER_CLASS * 2


def draw_path2_rows(units: list[dict] | None = None,
                    path2_rows: list[dict] | None = None,
                    seed: int = SEED) -> list[dict]:
    """Take four matched real/control second units for each first unit.

    The candidate pool is Part 119b's directed Path 2 inventory. Sampling the
    shared second-unit IDs gives four real and four random-control rows per
    first unit while keeping the two classes matched.
    """
    source = (rows119b.all_path2_rows(units) if path2_rows is None else path2_rows)
    by_first: dict[str, dict[str, set[str]]] = {}
    for row in source:
        kind = str(row["kind"])
        if kind not in ("real", "control"):
            continue
        by_first.setdefault(str(row["first"]), {"real": set(), "control": set()})[
            kind].add(str(row["second"]))

    ordered_firsts = sorted(by_first)
    if units is not None:
        expected_firsts = sorted(str(unit["id"]) for unit in units)
        if ordered_firsts != expected_firsts:
            raise ValueError("Path 2 candidate rows do not cover the requested first units")
    rng = random.Random(seed)
    lookup = {(str(row["kind"]), str(row["first"]), str(row["second"])): row
              for row in source}
    selected = []
    for first in ordered_firsts:
        candidates = by_first[first]
        seconds = candidates["real"] & candidates["control"]
        if candidates["real"] != candidates["control"]:
            raise ValueError(f"real/control Path 2 second units differ for {first}")
        chosen = rng.sample(sorted(seconds), min(SECOND_UNITS_PER_CLASS, len(seconds)))
        for second in chosen:
            for kind in ("real", "control"):
                row = lookup[(kind, first, second)]
                selected.append({**row, "pushes": [dict(push) for push in row["pushes"]]})

    expected = len(ordered_firsts) * min(SECOND_UNITS_PER_CLASS,
                                        max(0, len(ordered_firsts) - 1)) * 2
    if len(selected) != expected or len({(row["kind"], row["first"], row["second"])
                                         for row in selected}) != len(selected):
        raise AssertionError("Part 119c Path 2 sample has an invalid size or duplicate row")
    if units is None and len(selected) != ROW_COUNT:
        raise AssertionError(f"expected {ROW_COUNT} sampled rows, found {len(selected)}")
    return selected


if __name__ == "__main__":
    rows = draw_path2_rows()
    print({"seed": SEED, "rows": len(rows),
           "first_units": len({row["first"] for row in rows}),
           "real": sum(row["kind"] == "real" for row in rows),
           "control": sum(row["kind"] == "control" for row in rows)})
