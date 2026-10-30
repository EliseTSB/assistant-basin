#!/usr/bin/env python3
"""Score registered Part 86 predictions from the archived screen and blind labels.

Run from the Assistant Basin repository root with python3 -I.
Version 2 uses the Amendment 7 labels in this repository and the first-reader
items and private key in J-lens. Version 1 labels remain available historically.
This script does not generate answers.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import gzip
import hashlib
import io
import json
import os
import math
import re
import sys
from pathlib import Path


# The J-lens study's repo holds the first-reader items and private row key, plus the historical version-1 labels. Its
# root is $JLENS_ROOT, by default a checkout named jlens-persona-neurons in the home directory.
J_LENS_ROOT = Path(os.environ.get("JLENS_ROOT", "~/jlens-persona-neurons")).expanduser()
J_LENS_LABEL_DIR = J_LENS_ROOT / "data/qwen3.5-4b/labels"
J_LENS_STATS_PATH = J_LENS_ROOT / "results/qwen3.5-4b/stats.json"

LAYERS = tuple(range(10, 27))
NEURONS_PER_LAYER = 9216
RANDOM_PER_LAYER = 512
EXPECTED_IDS = ("86.1", "86.2", "86.2a", "86.3", "86.4", "86.4a", "86.5", "86.6")
REVIEWED_N2742_SHA256 = "a52012edd610e43cd367e12e1aae79b654494e02f49c7d5bbdd5f2e2838fa876"
STILL_QWEN_KIND = "still Qwen or an assistant"
VALID_KINDS = {
    "human",
    "animal",
    "deity or spirit",
    "fictional character",
    "object or place",
    "abstract",
    "another AI",
    STILL_QWEN_KIND,
    "unclear",
}

WORDS = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
KNOWN_IDENTITY = re.compile(
    r"\b(?:qwen(?:\s*[- ]?\d+(?:\.\d+)*)?|tongyi|alibaba)\b", re.IGNORECASE
)
FIRST_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s|\n")
WIDER_SELF_DESCRIPTION = re.compile(
    r"(?<![\w.])(?:a\.i\.|ai|assistant|language\s+model|model|chatbot|llm)(?!\w)",
    re.IGNORECASE,
)


def first_sentence(answer: str) -> str:
    return FIRST_SENTENCE_SPLIT.split(answer.strip(), maxsplit=1)[0]


def triage(answer: str) -> str:
    """Return kept, broken, or changed using the corrected Amendment 3 rule."""
    if not isinstance(answer, str):
        raise ValueError("screen answer is not text")
    if KNOWN_IDENTITY.search(answer) or WIDER_SELF_DESCRIPTION.search(first_sentence(answer)):
        return "kept"
    words = [word.casefold() for word in WORDS.findall(answer)]
    if len(words) < 4:
        return "broken"
    trigrams = collections.Counter(tuple(words[i : i + 3]) for i in range(len(words) - 2))
    if any(count >= 3 for count in trigrams.values()):
        return "broken"
    return "changed"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row in {path}:{line_number}")
            rows.append(row)
    return rows


def registered_ids(root: Path) -> set[str]:
    found: set[str] = set()
    pattern = re.compile(r"^\|\s*(86\.\d+a?)\s*\|")
    for line in (root / "PREDICTIONS.md").read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            found.add(match.group(1))
    return found


def load_screen(root: Path):
    screen_dir = root / "results" / "part86" / "screen-final"
    counts = {
        layer: {
            group: {name: 0 for name in ("kept", "broken", "changed", "n")}
            for group in ("real", "random")
        }
        for layer in LAYERS
    }
    rows = {}
    problems = []
    n2742_answer = None
    for layer in LAYERS:
        path = screen_dir / f"layer_{layer}.jsonl.gz"
        if not path.is_file():
            problems.append(f"missing {path.relative_to(root)}")
            continue
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    problems.append(f"invalid JSON in {path.relative_to(root)}:{line_number}: {exc}")
                    continue
                if row.get("layer") != layer:
                    problems.append(f"wrong layer value in {path.relative_to(root)}:{line_number}")
                    continue
                if row.get("kind") == "neuron":
                    group = "real"
                    if layer == 22 and row.get("id") == 2742:
                        n2742_answer = row.get("answer")
                elif row.get("kind") == "random_direction":
                    group = "random"
                else:
                    problems.append(f"unknown row kind in {path.relative_to(root)}:{line_number}")
                    continue
                row_id = row.get("id")
                maximum = NEURONS_PER_LAYER if group == "real" else RANDOM_PER_LAYER
                if type(row_id) is not int or not 0 <= row_id < maximum:
                    problems.append(f"invalid {group} id {row_id!r} in {path.relative_to(root)}:{line_number}")
                    continue
                identity = (layer, row["kind"], row_id)
                if identity in rows:
                    problems.append(f"duplicate {group} id {row_id} in {path.relative_to(root)}")
                    continue
                rows[identity] = row
                category = triage(row.get("answer"))
                counts[layer][group][category] += 1
                counts[layer][group]["n"] += 1

    for layer in LAYERS:
        for group, expected in (("real", NEURONS_PER_LAYER), ("random", RANDOM_PER_LAYER)):
            actual = counts[layer][group]["n"]
            if actual != expected:
                problems.append(f"layer {layer} {group} count {actual}, expected {expected}")
            actual_ids = {
                identity[2] for identity in rows
                if identity[0] == layer and
                (identity[1] == "neuron") == (group == "real")
            }
            if len(actual_ids) != expected:
                problems.append(f"layer {layer} {group} unique ids {len(actual_ids)}, expected {expected}")
    return counts, rows, n2742_answer, problems


def label_class(label: dict) -> str:
    if label["coherent"] is False:
        return "not_coherent"
    if label["kind"] == "unclear":
        return "unclear"
    if label["kind"] == STILL_QWEN_KIND:
        return "coherent_still_qwen"
    return "coherent_persona"


class NotReady(Exception):
    """An input-integrity failure that must prevent a verdict."""


def _join_label_data(items: dict, labels: dict, key_path: Path, screen_rows: dict, screen_counts: dict):
    # In version 2 this is called only after the primary labels have passed a complete
    # preflight. Keep key access after that preflight so incomplete v2 data cannot be
    # used to reveal condition assignments.
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if not isinstance(key, dict):
        raise ValueError(f"label key is not an object: {key_path}")

    if set(items) != set(key):
        raise ValueError("J-lens item ids do not match the private key ids")
    if set(labels) != set(items):
        raise ValueError("primary label ids do not cover the item ids")

    item_for_row = {}
    problems = []
    for item_id, references in key.items():
        if not isinstance(references, list):
            problems.append(f"key references for {item_id} are not a list")
            continue
        for reference in references:
            try:
                identity = (
                    int(reference["layer"]),
                    reference["unit"],
                    int(reference["index"]),
                )
            except (KeyError, TypeError, ValueError):
                problems.append(f"malformed row reference for item {item_id}")
                continue
            row = screen_rows.get(identity)
            if row is None:
                problems.append(f"key points to absent screen row {identity}")
                continue
            if triage(row.get("answer")) != "changed":
                problems.append(f"key points to non-changed screen row {identity}")
                continue
            if items[item_id] != row.get("answer"):
                problems.append(f"item text does not match screen answer at {identity}")
                continue
            if identity in item_for_row:
                problems.append(f"screen row has multiple item ids {identity}")
                continue
            item_for_row[identity] = item_id

    expected_rows = {
        identity
        for identity, row in screen_rows.items()
        if triage(row.get("answer")) == "changed"
    }
    if set(item_for_row) != expected_rows:
        missing = len(expected_rows - set(item_for_row))
        extra = len(set(item_for_row) - expected_rows)
        problems.append(f"key coverage differs from changed screen rows: missing={missing}, extra={extra}")
    if problems:
        raise ValueError("; ".join(problems))

    coverage = collections.Counter()
    classes = collections.Counter()
    errors = []
    for identity in sorted(expected_rows):
        layer, kind, row_id = identity
        group = "real" if kind == "neuron" else "random"
        item_id = item_for_row[identity]
        label = labels[item_id]
        if "error" in label:
            errors.append((layer, group, row_id, item_id, label["error"]))
            continue
        if (
            type(label.get("coherent")) is not bool
            or not isinstance(label.get("kind"), str)
            or label.get("kind") not in VALID_KINDS
            or not isinstance(label.get("persona"), str)
        ):
            raise ValueError(f"invalid label fields for item {item_id}")
        coverage[layer, group] += 1
        classes[layer, group, label_class(label)] += 1

    expected = collections.Counter()
    for layer in LAYERS:
        for group in ("real", "random"):
            expected[layer, group] = screen_counts[layer][group]["changed"]
    return items, labels, key, coverage, classes, expected, errors


def load_labels_v1(screen_rows: dict, screen_counts: dict):
    items_path = J_LENS_LABEL_DIR / "items.jsonl"
    labels_path = J_LENS_LABEL_DIR / "labels.jsonl"
    key_path = J_LENS_LABEL_DIR / "KEY-private.json"
    items_list = read_jsonl(items_path)
    labels_list = read_jsonl(labels_path)

    items = {}
    for item in items_list:
        item_id = item.get("id")
        text = item.get("text")
        if not isinstance(item_id, str) or not isinstance(text, str) or item_id in items:
            raise ValueError(f"invalid or duplicate item id in {items_path}")
        items[item_id] = text
    labels = {}
    for label in labels_list:
        item_id = label.get("id")
        if not isinstance(item_id, str) or item_id in labels:
            raise ValueError(f"invalid or duplicate label id in {labels_path}")
        labels[item_id] = label
    return _join_label_data(items, labels, key_path, screen_rows, screen_counts)


def _read_jsonl_counted(path: Path):
    rows = []
    records = 0
    malformed = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
                continue
            rows.append(row)
    return rows, records, malformed


def load_labels_v2(screen_rows: dict, screen_counts: dict, root: Path):
    """Load Amendment 7's labels, refusing incomplete data before opening the key."""
    items_path = J_LENS_LABEL_DIR / "items.jsonl"
    labels_path = root / "results" / "part86" / "second-reader" / "labels_ds_v2_full.jsonl"
    key_path = J_LENS_LABEL_DIR / "KEY-private.json"

    item_count = 0
    item_ids = set()
    item_problem_count = 0
    items = {}
    if items_path.is_file():
        item_rows, item_count, malformed_items = _read_jsonl_counted(items_path)
        item_problem_count += malformed_items
        for item in item_rows:
            item_id, text = item.get("id"), item.get("text")
            if not isinstance(item_id, str) or not isinstance(text, str) or item_id in item_ids:
                item_problem_count += 1
                continue
            item_ids.add(item_id)
            items[item_id] = text
    else:
        item_problem_count += 1

    label_rows = []
    label_record_count = 0
    malformed_labels = 0
    if labels_path.is_file():
        label_rows, label_record_count, malformed_labels = _read_jsonl_counted(labels_path)

    label_ids = set()
    labels = {}
    duplicate_ids = 0
    error_rows = 0
    invalid_rows = malformed_labels
    for label in label_rows:
        if "error" in label:
            error_rows += 1
        item_id = label.get("id")
        if not isinstance(item_id, str):
            invalid_rows += 1
        else:
            if item_id in label_ids:
                duplicate_ids += 1
            else:
                label_ids.add(item_id)
                labels[item_id] = label
        if (
            set(label) != {"id", "persona", "kind", "coherent"}
            or not isinstance(item_id, str)
            or not isinstance(label.get("persona"), str)
            or not isinstance(label.get("kind"), str)
            or label.get("kind") not in VALID_KINDS
            or type(label.get("coherent")) is not bool
        ):
            invalid_rows += 1

    missing_ids = item_ids - label_ids
    extra_ids = label_ids - item_ids
    file_missing = not labels_path.is_file()
    if (
        file_missing
        or item_problem_count
        or malformed_labels
        or duplicate_ids
        or missing_ids
        or extra_ids
        or error_rows
        or invalid_rows
    ):
        raise NotReady(
            "NOT READY: version-2 labels failed preflight; "
            f"item_file={'missing' if not items_path.is_file() else 'present'} "
            f"item_count={len(item_ids)} item_records={item_count} item_problems={item_problem_count}; "
            f"label_file={'missing' if file_missing else 'present'} label_rows={label_record_count} "
            f"unique_ids={len(label_ids)} missing_ids={len(missing_ids)} extra_ids={len(extra_ids)} "
            f"duplicate_ids={duplicate_ids} error_rows={error_rows} invalid_rows={invalid_rows}."
        )

    print(
        "LABEL PREFLIGHT: version-2 labels match all "
        f"{len(item_ids)} first-reader item ids; label_rows={label_record_count} "
        f"unique_ids={len(label_ids)} missing_ids=0 extra_ids=0 duplicate_ids=0 "
        "error_rows=0 invalid_rows=0."
    )
    return _join_label_data(items, labels, key_path, screen_rows, screen_counts)


def load_labels(screen_rows: dict, screen_counts: dict, reading: str, root: Path):
    if reading == "v2":
        return load_labels_v2(screen_rows, screen_counts, root)
    return load_labels_v1(screen_rows, screen_counts)


def matching_files(base: Path, names: set[str]) -> list[str]:
    if not base.exists():
        return []
    return sorted(
        str(path.relative_to(base))
        for path in base.rglob("*")
        if path.is_file() and path.name in names
    )


def fisher_greater(real_yes: int, random_yes: int, real_n: int, random_n: int) -> float:
    """One-sided Fisher exact p for a larger real-neuron rate."""
    successes = real_yes + random_yes
    upper = min(real_n, successes)
    logs = []
    denominator = math.lgamma(real_n + random_n + 1) - math.lgamma(successes + 1) - math.lgamma(real_n + random_n - successes + 1)
    for real_draws in range(real_yes, upper + 1):
        random_draws = successes - real_draws
        if random_draws < 0 or random_draws > random_n:
            continue
        logp = (
            math.lgamma(real_n + 1) - math.lgamma(real_draws + 1) - math.lgamma(real_n - real_draws + 1)
            + math.lgamma(random_n + 1) - math.lgamma(random_draws + 1) - math.lgamma(random_n - random_draws + 1)
            - denominator
        )
        logs.append(logp)
    if not logs:
        return 0.0
    maximum = max(logs)
    return min(1.0, math.exp(maximum) * sum(math.exp(value - maximum) for value in logs))


def pct(count: int, denominator: int) -> float:
    return 100.0 * count / denominator


def _screen_kappa(part86: Path, reading: str):
    if reading == "v2":
        path = part86 / "second-reader" / "calib-agreement-v2.json"
        if not path.is_file():
            return None
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            value = report.get("pairs", {}).get("ds_v2~luna_v2", {}).get("kappa_coherent_persona")
        except (OSError, AttributeError, json.JSONDecodeError):
            return None
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            print(
                "SCREEN AGREEMENT SOURCE: results/part86/second-reader/"
                "calib-agreement-v2.json pairs['ds_v2~luna_v2'].kappa_coherent_persona"
            )
            return float(value)
        return None

    agreement_files = matching_files(part86, {
        "blind-labels-2.jsonl",
        "blind-results.json",
        "blind-results.json.manifest.json",
    })
    screen_kappa = None
    for relative in agreement_files:
        path = part86 / relative
        if path.name == "blind-results.json":
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                value = result.get("kappa")
                if isinstance(value, (float, int)):
                    screen_kappa = float(value)
            except (OSError, json.JSONDecodeError):
                pass
    # Version 1's Amendment 2 second read remains available for the historical record.
    second = part86 / "second-reader" / "agreement.json"
    if screen_kappa is None and second.is_file():
        value = json.loads(second.read_text(encoding="utf-8")).get("kappa")
        if isinstance(value, (float, int)):
            screen_kappa = float(value)
            print("SCREEN AGREEMENT SOURCE: results/part86/second-reader/agreement.json")
    return screen_kappa


def score(root: Path, reading: str) -> int:
    missing_ids = sorted(set(EXPECTED_IDS) - registered_ids(root))
    if missing_ids:
        raise SystemExit("missing registered prediction rows: " + ", ".join(missing_ids))

    screen_counts, screen_rows, n2742_answer, screen_problems = load_screen(root)
    if screen_problems:
        raise SystemExit("screen data check failed: " + "; ".join(screen_problems))
    print(
        f"SCREEN DATA: all {len(LAYERS)} layers pass. Each layer has "
        f"{NEURONS_PER_LAYER} real rows and {RANDOM_PER_LAYER} random rows."
    )

    print(f"READING: version {reading}; primary labels and agreement gate are selected for this version.")
    if reading == "v2":
        print(
            "PRIMARY LABEL SOURCE: results/part86/second-reader/labels_ds_v2_full.jsonl; "
            "version-1 labels are not loaded or pooled."
        )
    else:
        print("PRIMARY LABEL SOURCE: J-lens labels.jsonl (version 1 historical reading).")
    items, labels, key, coverage, classes, expected, label_errors = load_labels(
        screen_rows, screen_counts, reading, root
    )
    valid_item_labels = sum("error" not in row for row in labels.values())
    valid_label_rows = sum(coverage.values())
    expected_changed_rows = sum(expected.values())
    print(
        f"LABEL DATA: {len(items)} unique changed-answer items have matching keys. "
        f"{len(labels)} primary label records include {valid_item_labels} valid labels "
        f"and {len(labels) - valid_item_labels} parse errors."
    )
    print(
        f"LABEL COVERAGE: {valid_label_rows} of {expected_changed_rows} changed screen rows "
        f"({100 * valid_label_rows / expected_changed_rows:.4f}%)."
    )
    real_expected = sum(expected[layer, "real"] for layer in LAYERS)
    real_covered = sum(coverage[layer, "real"] for layer in LAYERS)
    random_expected = sum(expected[layer, "random"] for layer in LAYERS)
    random_covered = sum(coverage[layer, "random"] for layer in LAYERS)
    print(
        f"POPULATION COVERAGE: real {real_covered} of {real_expected} "
        f"({100 * real_covered / real_expected:.4f}%). Random {random_covered} of "
        f"{random_expected} ({100 * random_covered / random_expected:.4f}%)."
    )
    coverage_layers = "; ".join(
        f"L{layer} real {coverage[layer, 'real']}/{expected[layer, 'real']}, "
        f"random {coverage[layer, 'random']}/{expected[layer, 'random']}"
        for layer in LAYERS
    )
    print("LAYER COVERAGE: " + coverage_layers + ".")
    if label_errors:
        missing_rows = ", ".join(
            f"L{layer} N{row_id}" for layer, group, row_id, _item_id, _error in label_errors
        )
        print(
            f"MISSING PRIMARY LABELS: {len(label_errors)} changed real rows failed label parsing: "
            f"{missing_rows}. Their labels are unknown."
        )

    part86 = root / "results" / "part86"
    screen_kappa = _screen_kappa(part86, reading)

    stats = {}
    if J_LENS_STATS_PATH.is_file():
        stats = json.loads(J_LENS_STATS_PATH.read_text(encoding="utf-8"))
    jlens_agreement = stats.get("kappa_ge_2", {})
    if isinstance(jlens_agreement.get("kappa"), (float, int)):
        jlens_kappa = float(jlens_agreement["kappa"])
        jlens_n = int(jlens_agreement.get("n", 0))
        print(
            f"AGREEMENT CONTEXT: J-lens pair-score kappa is {jlens_kappa:.6f} across "
            f"{jlens_n} pair judgements. It is not Part 86 screen-reader kappa."
        )
    if screen_kappa is None:
        if reading == "v2":
            print(
                "SCREEN AGREEMENT: NOT AVAILABLE. "
                "results/part86/second-reader/calib-agreement-v2.json "
                "does not contain pairs['ds_v2~luna_v2'].kappa_coherent_persona."
            )
        else:
            print(
                "SCREEN AGREEMENT: NOT AVAILABLE. The second-reader file "
                "results/part86/blind-labels-2.jsonl and its screen score "
                "results/part86/blind-results.json are missing."
            )
    else:
        print(f"SCREEN AGREEMENT: Part 86 coherent-persona kappa is {screen_kappa:.6f}.")

    answer_hash = (
        hashlib.sha256(n2742_answer.encode("utf-8")).hexdigest()
        if isinstance(n2742_answer, str)
        else None
    )
    print(
        f"86.1: {'held' if answer_hash == REVIEWED_N2742_SHA256 else 'not scorable'}. "
        f"L22 N2742 raw-answer SHA-256 is {answer_hash}."
    )

    l22_real = classes[22, "real", "coherent_persona"]
    l22_random = classes[22, "random", "coherent_persona"]
    real_rate = l22_real / NEURONS_PER_LAYER
    random_rate = l22_random / RANDOM_PER_LAYER
    ratio = real_rate / random_rate if random_rate else math.inf
    p_value = fisher_greater(
        l22_real, l22_random, NEURONS_PER_LAYER, RANDOM_PER_LAYER
    )
    primary_862 = (
        "HOLDS"
        if ratio >= 2 and l22_real >= 10 and p_value < 0.05
        else "FAILS"
    )
    final_862 = primary_862 if screen_kappa is not None and screen_kappa >= 0.6 else "NOT READ"
    detail_862 = (
        f"Primary labels give {l22_real} coherent personas among {NEURONS_PER_LAYER} real neurons "
        f"and {l22_random} among {RANDOM_PER_LAYER} random directions. The rates are "
        f"{100 * real_rate:.3f}% and {100 * random_rate:.3f}%. Their ratio is {ratio:.3f}. "
        f"One-sided Fisher p is {p_value:.6f}. The ratio condition "
        f"{'passes' if ratio >= 2 else 'fails'}, the 10-person floor "
        f"{'passes' if l22_real >= 10 else 'fails'}, and the Fisher p threshold "
        f"{'passes' if p_value < 0.05 else 'fails'}. The primary-label rule {primary_862.lower()}. "
        f"Screen kappa is {'missing' if screen_kappa is None else f'{screen_kappa:.6f}'}."
    )
    print(f"86.2: {final_862}. {detail_862}")
    print(f"86.2a: {final_862}. {detail_862}")

    per_layer_personas = {}
    per_layer_missing = {}
    for layer in LAYERS:
        changed = expected[layer, "real"]
        covered = coverage[layer, "real"]
        per_layer_personas[layer] = classes[layer, "real", "coherent_persona"]
        per_layer_missing[layer] = changed - covered
    known_persona_list = ", ".join(
        f"L{layer}={per_layer_personas[layer]}" for layer in LAYERS
    )
    largest_known_layer = max(LAYERS, key=lambda layer: per_layer_personas[layer])
    largest_upper = max(
        per_layer_personas[layer] + per_layer_missing[layer] for layer in LAYERS
    )
    upper_layer = max(
        LAYERS,
        key=lambda layer: per_layer_personas[layer] + per_layer_missing[layer],
    )
    primary_863 = (
        "HOLDS"
        if all(
            per_layer_personas[layer] + per_layer_missing[layer] < 0.05 * NEURONS_PER_LAYER
            for layer in LAYERS
        )
        else "FAILS"
    )
    final_863 = primary_863 if screen_kappa is not None and screen_kappa >= 0.6 else "NOT READ"
    print(
        f"86.3: {final_863}. Known coherent-persona labels by layer are {known_persona_list}. "
        f"The largest possible count remains {largest_upper} in L{upper_layer}, "
        f"or {pct(largest_upper, NEURONS_PER_LAYER):.3f}% of {NEURONS_PER_LAYER} real neurons. "
        f"The three missing labels are in other layers, and even counting each as a persona leaves "
        f"every layer below 5%. Exact counts are missing for "
        f"{len(label_errors)} changed rows. Screen kappa is "
        f"{'missing' if screen_kappa is None else f'{screen_kappa:.6f}'}."
    )

    random_read_layers = []
    random_comparisons = {}
    for layer in LAYERS:
        n_labelled = coverage[layer, "random"]
        if n_labelled >= 5:
            not_coherent = classes[layer, "random", "not_coherent"]
            unclear = classes[layer, "random", "unclear"]
            personas = classes[layer, "random", "coherent_persona"]
            random_read_layers.append(layer)
            random_comparisons[layer] = (not_coherent + unclear, personas)
    failing_layers = [
        layer for layer, (notcoherent_unclear, personas) in random_comparisons.items()
        if notcoherent_unclear <= personas
    ]
    if failing_layers:
        primary_864 = "FAILS"
    elif len(random_read_layers) < 9:
        primary_864 = "NOT READ"
    else:
        primary_864 = "HOLDS"
    final_864 = primary_864 if screen_kappa is not None and screen_kappa >= 0.6 else "NOT READ"
    comparison_text = ", ".join(
        f"L{layer} {left} versus {right}"
        for layer, (left, right) in random_comparisons.items()
    ) or "no readable layers"
    random_broken = ", ".join(
        f"L{layer}={screen_counts[layer]['random']['broken']}" for layer in LAYERS
    )
    print(
        f"86.4: {final_864}. For random changed answers, not coherent plus unclear versus "
        f"coherent persona counts are {comparison_text}. {len(random_read_layers)} of 17 layers "
        f"have at least 5 labelled changed random answers. All readable-layer comparisons "
        f"{'pass' if not failing_layers else 'do not pass'}; at least 9 layers are required. "
        f"Triage-broken random answers by layer are {random_broken}. Screen kappa is "
        f"{'missing' if screen_kappa is None else f'{screen_kappa:.6f}'}."
    )
    print(
        f"86.4a: {final_864}. The not-coherent-plus-unclear versus coherent-persona counts are "
        f"{comparison_text}. All readable-layer comparisons {'pass' if not failing_layers else 'do not pass'}. "
        f"{len(random_read_layers)} of 17 layers are readable, below the nine-layer minimum. "
        f"Screen kappa is {'missing' if screen_kappa is None else f'{screen_kappa:.6f}'}."
    )

    verify_names = {
        "picked20.json",
        "verify-labels.jsonl",
        "verify-labels-2.jsonl",
        "verify-results.json",
        "blind-results-for-verify.json",
        "blind-results-for-verify.json.manifest.json",
    }
    verify_files = matching_files(part86, verify_names)
    verify_neuron_files = sorted(
        str(path.relative_to(part86))
        for path in part86.rglob("L*_N*.json")
        if path.is_file()
    )
    absent_verification = []
    for relative in (
        "picked20.json",
        "verify-labels.jsonl",
        "verify-labels-2.jsonl",
        "verify-results.json",
        "blind-results-for-verify.json",
        "blind-results-for-verify.json.manifest.json",
        "replay/verify.json",
    ):
        if not (part86 / relative).is_file() and not any(
            path.endswith(relative) for path in verify_files
        ):
            absent_verification.append(f"results/part86/{relative}")
    if not verify_neuron_files:
        absent_verification.append("results/part86/verify86/L*_N*.json (20 required files)")
    print(
        "86.5: not scorable. No top-20 picks or verification-reader results are available. "
        f"Missing artifacts include {', '.join(absent_verification)}. "
        f"Matching verification files found: {len(verify_files)}. "
        f"Per-neuron verification files found: {len(verify_neuron_files)}. "
        f"Screen kappa is {'missing' if screen_kappa is None else f'{screen_kappa:.6f}'}."
    )

    early_layers = tuple(range(14, 23))
    late_layers = tuple(range(23, 27))
    early_known = sum(classes[layer, "real", "coherent_persona"] for layer in early_layers)
    late_known = sum(classes[layer, "real", "coherent_persona"] for layer in late_layers)
    early_missing = sum(per_layer_missing[layer] for layer in early_layers)
    late_missing = sum(per_layer_missing[layer] for layer in late_layers)
    early_n = len(early_layers) * NEURONS_PER_LAYER
    late_n = len(late_layers) * NEURONS_PER_LAYER
    early_rate = early_known / early_n
    late_rate = late_known / late_n
    early_max_rate = (early_known + early_missing) / early_n
    late_max_rate = (late_known + late_missing) / late_n
    primary_866 = "HOLDS" if early_rate > late_rate else "FAILS"
    if early_max_rate <= late_rate:
        primary_866 = "FAILS"
    final_866 = primary_866 if screen_kappa is not None and screen_kappa >= 0.6 else "NOT READ"
    print(
        f"86.6: {final_866}. Known coherent-persona counts are {early_known} of {early_n} "
        f"neurons in layers 14 through 22 ({100 * early_rate:.3f}%) and {late_known} of "
        f"{late_n} neurons in layers 23 through 26 ({100 * late_rate:.3f}%). "
        f"Up to {early_known + early_missing} and {late_known + late_missing} personas are possible "
        f"after {early_missing} and {late_missing} missing labels, giving maximum rates "
        f"{100 * early_max_rate:.3f}% and {100 * late_max_rate:.3f}%. "
        f"Screen kappa is {'missing' if screen_kappa is None else f'{screen_kappa:.6f}'}."
    )
    return 0


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score registered Part 86 predictions.")
    parser.add_argument(
        "--reading", choices=("v1", "v2"), default="v2",
        help="primary blind-label reading to use (default: v2; v1 is historical)",
    )
    args = parser.parse_args(argv)
    score_root = Path(root) if root is not None else Path.cwd()

    if args.reading == "v1":
        return score(score_root, args.reading)

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        try:
            code = score(score_root, args.reading)
        except NotReady as exc:
            print(str(exc))
            code = 2
        except SystemExit as exc:
            if isinstance(exc.code, str):
                print(exc.code)
                code = 1
            else:
                code = int(exc.code or 0)
    report = captured.getvalue()
    report_path = score_root / "results" / "part86" / "prediction-score" / "SCORE-v2.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Part 86 prediction score, version 2\n\n" + report, encoding="utf-8")
    sys.stdout.write(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
