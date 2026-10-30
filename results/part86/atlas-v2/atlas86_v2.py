#!/usr/bin/env python3
"""Measure the Amendment 8 atlas predictions using the Part 86 v2 scorer."""
from __future__ import annotations

import argparse
import collections
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path


CLASS_NAMES = (
    "coherent_persona",
    "coherent_still_qwen",
    "not_coherent",
    "unclear",
)
EXPECTED_ATLAS_TOTAL = 465
EXPECTED_ANOTHER_AI = 387
EXPECTED_OTHER_KIND = 78
KAPPA_SOURCE = (
    "results/part86/second-reader/calib-agreement-v2.json "
    "pairs['ds_v2~luna_v2'].kappa_coherent_persona"
)


def load_scorer():
    candidates = [
        Path(__file__).resolve().parents[1] / "prediction-score" / "score86_predictions.py",
        Path.cwd() / "results" / "part86" / "prediction-score" / "score86_predictions.py",
    ]
    scorer_path = next((path for path in candidates if path.is_file()), None)
    if scorer_path is None:
        raise RuntimeError("cannot find results/part86/prediction-score/score86_predictions.py")
    spec = importlib.util.spec_from_file_location("score86_predictions_for_atlas_v2", scorer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scorer at {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = load_scorer()


def classify(label: dict) -> str:
    """Use the scorer's registered four-way class rule."""
    return SCORER.label_class(label)


def row_identity(reference: dict) -> tuple[int, str, int]:
    return (int(reference["layer"]), reference["unit"], int(reference["index"]))


def row_item_map(key: dict) -> dict[tuple[int, str, int], str]:
    """Rebuild the item-to-screen-row join returned internally by the scorer."""
    rows = {}
    for item_id, references in key.items():
        for reference in references:
            rows[row_identity(reference)] = item_id
    return rows


def read_atlas(path: Path) -> list[dict]:
    try:
        units = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read atlas units at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid atlas units JSON at {path}: {exc}") from exc
    if not isinstance(units, list):
        raise ValueError("atlas units must be a JSON list")

    seen = set()
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("atlas contains a non-object unit")
        layer, neuron, kind = unit.get("layer"), unit.get("neuron"), unit.get("kind")
        if type(layer) is not int or type(neuron) is not int or not isinstance(kind, str):
            raise ValueError("atlas unit must have integer layer/neuron and string kind fields")
        if type(unit.get("q_name")) is not bool:
            raise ValueError("atlas unit must have a boolean q_name field")
        identity = (layer, "neuron", neuron)
        if identity in seen:
            raise ValueError(f"duplicate atlas screen row {identity}")
        seen.add(identity)

    another_ai = sum(unit["kind"] == "another AI" for unit in units)
    other_kind = len(units) - another_ai
    if (len(units), another_ai, other_kind) != (
        EXPECTED_ATLAS_TOTAL,
        EXPECTED_ANOTHER_AI,
        EXPECTED_OTHER_KIND,
    ):
        raise ValueError(
            "atlas count check failed: "
            f"total={len(units)} another_ai={another_ai} other_kind={other_kind}; "
            f"expected {EXPECTED_ATLAS_TOTAL}, {EXPECTED_ANOTHER_AI}, {EXPECTED_OTHER_KIND}"
        )
    return units


def empty_class_counts() -> collections.Counter:
    return collections.Counter({name: 0 for name in CLASS_NAMES})


def labels_for_units(units: list[dict], rows: dict, labels: dict) -> list[dict]:
    return [labels[rows[(unit["layer"], "neuron", unit["neuron"])]] for unit in units]


def count_classes(label_rows: list[dict]) -> collections.Counter:
    counts = empty_class_counts()
    for label in label_rows:
        counts[classify(label)] += 1
    return counts


def format_classes(counts: collections.Counter, total: int) -> str:
    return (
        f"n={total} "
        + " ".join(f"{name}={counts[name]}" for name in CLASS_NAMES)
    )


def first_reader_check(label_dir: Path, key: dict, atlas_rows: set) -> None:
    """Print a diagnostic comparison. Its result is deliberately not returned."""
    try:
        first_labels = SCORER.read_jsonl(label_dir / "labels.jsonl")
        labels_by_id = {}
        duplicate_ids = 0
        invalid_ids = 0
        for label in first_labels:
            item_id = label.get("id")
            if not isinstance(item_id, str):
                invalid_ids += 1
            elif item_id in labels_by_id:
                duplicate_ids += 1
            else:
                labels_by_id[item_id] = label

        coherent_rows = set()
        malformed_labels = 0
        for item_id, references in key.items():
            label = labels_by_id.get(item_id)
            if label is None:
                continue
            try:
                is_persona = classify(label) == "coherent_persona"
            except (KeyError, TypeError, ValueError):
                malformed_labels += 1
                continue
            if is_persona:
                for reference in references:
                    identity = row_identity(reference)
                    if identity[1] == "neuron":
                        coherent_rows.add(identity)

        intersection = coherent_rows & atlas_rows
        only_first_reader = coherent_rows - atlas_rows
        only_atlas = atlas_rows - coherent_rows
        status = "equal" if coherent_rows == atlas_rows else "MISMATCH"
        print(
            "VERSION-1 ATLAS CHECK: "
            f"{status}; first_reader_real_coherent={len(coherent_rows)} "
            f"atlas_units={len(atlas_rows)} intersection={len(intersection)} "
            f"only_first_reader={len(only_first_reader)} only_atlas={len(only_atlas)}; "
            f"duplicate_label_ids={duplicate_ids} invalid_label_ids={invalid_ids} "
            f"malformed_labels={malformed_labels}. Diagnostic only."
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"VERSION-1 ATLAS CHECK: unavailable ({exc}). Diagnostic only.")


def score(root: Path) -> int:
    part86 = root / "results" / "part86"
    screen_counts, screen_rows, _n2742_answer, screen_problems = SCORER.load_screen(root)
    if screen_problems:
        raise ValueError("screen data check failed: " + "; ".join(screen_problems))

    # This performs the version-2 complete-label preflight before opening the key.
    # Keep every later data read, including units.json and the v1 diagnostic, below it.
    items, labels, key, _coverage, _classes, _expected, _label_errors = SCORER.load_labels_v2(
        screen_rows, screen_counts, root
    )
    rows = row_item_map(key)

    units = read_atlas(root / "results" / "part112" / "units.json")
    atlas_rows = {(unit["layer"], "neuron", unit["neuron"]) for unit in units}
    missing_atlas_rows = atlas_rows - rows.keys()
    if missing_atlas_rows:
        raise ValueError(
            "atlas label coverage failed: "
            f"missing_labelled_screen_rows={len(missing_atlas_rows)}"
        )
    missing_labels = [identity for identity in atlas_rows if rows[identity] not in labels]
    if missing_labels:
        raise ValueError(f"atlas label coverage failed: missing_v2_labels={len(missing_labels)}")
    print(
        "ATLAS LABEL COVERAGE: "
        f"{len(units)} of {len(units)} atlas units have labelled screen rows; missing=0."
    )

    first_reader_check(SCORER.J_LENS_LABEL_DIR, key, atlas_rows)

    another_ai_units = [unit for unit in units if unit["kind"] == "another AI"]
    q_name_units = [unit for unit in another_ai_units if unit["q_name"]]
    other_units = [unit for unit in units if unit["kind"] != "another AI"]
    another_ai_counts = count_classes(labels_for_units(another_ai_units, rows, labels))
    q_name_counts = count_classes(labels_for_units(q_name_units, rows, labels))
    other_counts = count_classes(labels_for_units(other_units, rows, labels))

    kind_counts: dict[str, collections.Counter] = collections.defaultdict(empty_class_counts)
    kind_totals = collections.Counter()
    for unit in other_units:
        item_id = rows[(unit["layer"], "neuron", unit["neuron"])]
        kind_counts[unit["kind"]][classify(labels[item_id])] += 1
        kind_totals[unit["kind"]] += 1

    coherent_real_rows = set()
    coherent_non_atlas_kinds = collections.Counter()
    for identity, item_id in rows.items():
        if identity[1] != "neuron":
            continue
        label = labels[item_id]
        if classify(label) != "coherent_persona":
            continue
        coherent_real_rows.add(identity)
        if identity not in atlas_rows:
            coherent_non_atlas_kinds[label["kind"]] += 1
    coherent_atlas_count = len(coherent_real_rows & atlas_rows)
    coherent_non_atlas_count = len(coherent_real_rows - atlas_rows)

    print(
        "86.11 another AI version-2 classes: "
        + format_classes(another_ai_counts, len(another_ai_units))
    )
    print(
        "86.11 q_name subgroup version-2 classes: "
        + format_classes(q_name_counts, len(q_name_units))
    )
    print(
        "86.12 other-kind version-2 classes: "
        + format_classes(other_counts, len(other_units))
    )
    print("86.12 per-kind version-2 classes:")
    for kind in sorted(kind_totals):
        print(f"  {kind}: " + format_classes(kind_counts[kind], kind_totals[kind]))
    kind_text = ", ".join(
        f"{kind}={count}" for kind, count in sorted(coherent_non_atlas_kinds.items())
    ) or "none"
    print(
        "86.13 coherent-persona real screen rows: "
        f"total={len(coherent_real_rows)} atlas={coherent_atlas_count} "
        f"not_atlas={coherent_non_atlas_count}; non-atlas version-2 kinds: {kind_text}."
    )

    kappa = SCORER._screen_kappa(part86, "v2")
    gate_passed = kappa is not None and kappa >= 0.6
    kappa_text = "missing" if kappa is None else f"{kappa:.6f}"
    print(f"GATE: kappa={kappa_text}; source={KAPPA_SOURCE}; threshold=0.600000.")

    if not gate_passed:
        verdict_8611 = verdict_8612 = verdict_8613 = "NOT READ"
    else:
        verdict_8611 = (
            "HOLDS"
            if another_ai_counts["coherent_persona"] * 2 < len(another_ai_units)
            else "FAILS"
        )
        verdict_8612 = (
            "HOLDS"
            if other_counts["coherent_persona"] * 10 >= len(other_units) * 7
            else "FAILS"
        )
        verdict_8613 = "HOLDS" if len(coherent_real_rows) > EXPECTED_ATLAS_TOTAL else "FAILS"

    print(
        f"86.11: {verdict_8611}. "
        f"{another_ai_counts['coherent_persona']} of {len(another_ai_units)} "
        "another AI atlas units are coherent persona; fewer than half is the hold rule."
    )
    print(
        f"86.12: {verdict_8612}. "
        f"{other_counts['coherent_persona']} of {len(other_units)} other-kind atlas units "
        "are coherent persona; at least 70% is the hold rule."
    )
    print(
        f"86.13: {verdict_8613}. "
        f"{len(coherent_real_rows)} distinct real screen rows are coherent persona; "
        f"more than {EXPECTED_ATLAS_TOTAL} is the hold rule."
    )
    return 0


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the Part 86 Amendment 8 atlas predictions.")
    parser.add_argument("--root", type=Path, default=None, help="Assistant Basin repository root")
    args = parser.parse_args(argv)
    score_root = Path(root) if root is not None else (args.root or Path.cwd())

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        try:
            code = score(score_root)
        except SCORER.NotReady as exc:
            print(str(exc))
            code = 2
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}")
            code = 1

    report = captured.getvalue()
    report_path = score_root / "results" / "part86" / "atlas-v2" / "ATLAS-v2.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Part 86 atlas score, version 2\n\n" + report, encoding="utf-8")
    sys.stdout.write(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
