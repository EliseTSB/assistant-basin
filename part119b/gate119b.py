"""Small, dependency-free helpers shared by the Part 119b runner and scorer."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from statistics import median

PART119_INTERACTIONS_PATH = "results/part119/interaction.jsonl"
PART119_INTERACTIONS_ROW_COUNT = 2964
PART119_INTERACTIONS_SHA256 = "89cee0f0114d464c3405d21982beb753a2eeb977c771562198bd06c370104c56"


def interaction_share(position: dict) -> float | None:
    n_pair = float(position.get("nPair", 0.0))
    if n_pair <= 0:
        return None
    return (float(position.get("nI", 0.0)) / n_pair) ** 2


def _position_map(row: dict) -> dict[int, dict]:
    return {int(position["pos"]): position for position in row.get("positions", [])}


def path2_index(interactions: list[dict]) -> dict[tuple[str, str, str], dict[float, dict]]:
    result = defaultdict(dict)
    for row in interactions:
        key = (str(row["kind"]), str(row["first"]), str(row["second"]))
        dose = float(row["dose"])
        if dose in result[key]:
            raise ValueError(f"duplicate Path 2 row/dose {key} at {dose}")
        result[key][dose] = row
    return result


def _median_or_none(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def gate2_result(interactions: list[dict], reference: list[dict],
                 expected: dict[str, set[tuple[str, str, str]]]) -> dict:
    """Check Part 119 reproduction medians on the same selected rows."""
    index = path2_index(interactions)
    reference_by_key = {(str(row["kind"]), str(row["first"]), str(row["second"])): row
                        for row in reference}
    class_rows = {"real": [], "control": []}
    relative_npair_by_row = []
    compared_rows = {"real": 0, "control": 0}
    position_pairs = 0
    missing = []
    mismatch_positions = 0
    for kind in ("real", "control"):
        for key in expected.get(kind, set()):
            current = index.get(key, {}).get(.8)
            old = reference_by_key.get(key)
            if current is None or old is None:
                missing.append({"kind": kind, "first": key[1], "second": key[2]})
                continue
            compared_rows[kind] += 1
            current_pos = _position_map(current)
            old_pos = _position_map(old)
            common = sorted(set(current_pos) & set(old_pos))
            if set(current_pos) != set(old_pos):
                mismatch_positions += 1
            old_shares, new_shares = [], []
            row_relative_npair = []
            for pos in common:
                before, after = old_pos[pos], current_pos[pos]
                old_pair, new_pair = float(before["nPair"]), float(after["nPair"])
                if old_pair > 0:
                    row_relative_npair.append(abs(new_pair - old_pair) / old_pair)
                old_share = interaction_share(before)
                new_share = interaction_share(after)
                if old_share is not None:
                    old_shares.append(old_share)
                if new_share is not None:
                    new_shares.append(new_share)
                position_pairs += 1
            if old_shares and new_shares:
                class_rows[kind].append({"old": float(median(old_shares)),
                                         "new": float(median(new_shares))})
            if row_relative_npair:
                relative_npair_by_row.append(float(median(row_relative_npair)))

    observed = {}
    class_ok = {}
    for kind in ("real", "control"):
        expected_count = len(expected.get(kind, set()))
        new_values = [row["new"] for row in class_rows[kind]]
        old_values = [row["old"] for row in class_rows[kind]]
        value = _median_or_none(new_values)
        reference_median = _median_or_none(old_values)
        difference = (abs(value - reference_median)
                      if value is not None and reference_median is not None else None)
        observed[kind] = {"median_row_share_0.8": value,
                          "reference_median_row_share_0.8": reference_median,
                          "absolute_median_difference": difference,
                          "rows_compared": compared_rows[kind],
                          "rows_with_share_medians": len(class_rows[kind]),
                          "expected_rows": expected_count}
        if expected_count == 0:
            class_ok[kind] = True
        else:
            class_ok[kind] = (compared_rows[kind] == expected_count
                              and len(class_rows[kind]) == expected_count
                              and value is not None and reference_median is not None
                              and difference <= .01)

    median_relative = _median_or_none(relative_npair_by_row)
    complete = not missing and mismatch_positions == 0
    reasons = []
    if missing:
        reasons.append(f"{len(missing)} selected rows are missing from the pilot or Part 119 reference")
    if mismatch_positions:
        reasons.append(f"{mismatch_positions} selected rows have different position lists")
    failed_classes = [kind for kind, passed in class_ok.items() if not passed]
    if failed_classes:
        reasons.append("selected-row median share differs by more than 0.01 for "
                       + ", ".join(failed_classes))
    if median_relative is None:
        reasons.append("no row-level relative nPair differences were available")
    elif median_relative >= .01:
        reasons.append("median row relative nPair difference is at least 0.01")
    passed = bool(complete and all(class_ok.values())
                  and median_relative is not None and median_relative < .01)
    return {"passed": passed,
            "complete": complete, "missing_rows": missing[:25],
            "missing_row_count": len(missing), "position_list_mismatch_rows": mismatch_positions,
            "position_pairs_compared": position_pairs, "row_medians": observed,
            "class_thresholds_passed": class_ok,
            "median_relative_nPair_difference": median_relative,
            "median_relative_nPair_difference_over_rows": median_relative,
            "relative_nPair_rows_compared": len(relative_npair_by_row),
            "share_median_difference_threshold": .01,
            "nPair_threshold": .01,
            "failure_reasons": reasons,
            "reason": "; ".join(reasons) if reasons else None,
            "expected_rows": {kind: len(expected.get(kind, set()))
                              for kind in ("real", "control")},
            "rows_compared": compared_rows}


def validate_part119_reference(path: str | Path, rows: list[dict], *,
                               test_only_allow_unregistered_part119_reference: bool = False) -> dict:
    """Attest the registered Part 119 interaction input used by pilot Gate 2.

    The explicit test-only escape hatch supports the tiny synthetic pilot tests;
    it is deliberately not exposed through the runner's command line.
    """
    path = Path(path).expanduser()
    resolved_path = path.resolve()
    registered_path = (Path(__file__).resolve().parents[1]
                       / PART119_INTERACTIONS_PATH).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    if not test_only_allow_unregistered_part119_reference:
        if resolved_path != registered_path:
            raise ValueError(
                "pilot Gate 2 requires the registered Part 119 path "
                f"{PART119_INTERACTIONS_PATH!r}; got {str(path)!r}")
        if len(rows) != PART119_INTERACTIONS_ROW_COUNT:
            raise ValueError(
                f"registered Part 119 reference expected {PART119_INTERACTIONS_ROW_COUNT:,} rows, "
                f"found {len(rows):,}")
        if digest != PART119_INTERACTIONS_SHA256:
            raise ValueError(
                "registered Part 119 reference sha256 mismatch: "
                f"expected {PART119_INTERACTIONS_SHA256}, found {digest}")

    recorded_path = (PART119_INTERACTIONS_PATH if resolved_path == registered_path
                     else str(path))
    return {"path": recorded_path, "row_count": len(rows), "sha256": digest}
