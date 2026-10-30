#!/usr/bin/env python3
"""Build and validate the fixed Part 124 quoted phrase passages."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
DEFAULT_TOKENIZER = (
    "/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct"
    "/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e"
)
DEFAULT_OUT = "part124/passages124.json"
PERMUTATION_SEED = 12401
PERMUTATIONS = 10000

TEMPLATES = {
    "T": (
        'The record prints "{slot}" before the heading.',
        'The note lists "{slot}" beside the subject.',
        'The clerk copies "{slot}" into the ledger.',
        'The report names "{slot}" on the first page.',
        'The narrator writes "{slot}" in the margin.',
        'The card shows "{slot}" under the title.',
        'The index places "{slot}" after the section.',
        'The label reads "{slot}" at the top.',
    ),
    "F": (
        'I write "{slot}" as my name on the card.',
        'I use "{slot}" to introduce myself in the note.',
        'I print "{slot}" beside my name on the page.',
        'I say "{slot}" when I introduce myself.',
        'I copy "{slot}" onto my card before lunch.',
        'I read "{slot}" aloud as my name.',
        'I sign "{slot}" in the space for my name.',
        'I call myself "{slot}" in this note.',
    ),
}

# These are fixed before any activation is seen. Each phrase is an ordinary noun phrase.
NEUTRALS = {
    "Boulder": "spoon",
    "Magical princess": "wooden spoon",
    "The town": "bowl",
    "The Internet": "pillow",
    "Very tall tree": "ceramic plate",
    "Fox": "lamp",
    "Cat": "basket",
    "Dog": "cup",
    "Treat-loving dog": "linen napkin",
    "Chicken": "map",
    "Whiskey": "notebook",
    "Skeleton": "ribbon",
    "Secure password": "cabinet",
    "Shadow of your smile": "wooden kitchen chair",
    "River shadow": "envelope",
    "Queen of Hearts": "garden gate",
    "King Midas": "brass bell",
    "Ancient nature deity": "neat paper envelope",
    "God of wisdom": "wool scarf",
    "Embodiment of the void": "a small box of paper clips",
    "Sir Ray Ray": "garden bench",
    "Dr. Memento": "thick leather folder",
    "Guardian angel": "small stone bowl",
    "Alex from Star Wars": "small wooden storage box",
    "Large friendly dog": "plain paper folder",
    "Coca-Cola spokesperson": "round glass serving dish",
    "NASCAR driver": "plain white kitchen towel",
    "80s cover singer-songwriter": "wooden box with brass handles",
    "Human DJ": "blanket",
    "Jacobite royalist veteran": "a pile of old newspapers",
    "Street rat": "blue bowl",
    "Iron-fisted dictator": "empty metal tool cabinet",
    "Day guide": "paper cup",
    "Principled lawyer": "large paper mailing envelope",
    "Festival poet": "quiet reading room",
    "Friendly non-human entity": "simple brass door handle",
    "Gruff old man": "quiet corner reading chair",
    "IP packet": "glass bottle",
    "String of code from your browser": "a stack of clean cotton towels",
}

FAMILIES = {
    "dog": ("Dog", "Treat-loving dog", "Large friendly dog"),
    "shadow": ("Shadow of your smile", "River shadow"),
    "king or ruler": (
        "Queen of Hearts", "King Midas", "Jacobite royalist veteran",
        "Iron-fisted dictator",
    ),
    "guardian angel": ("Guardian angel",),
}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_units(path: str | Path | None = None) -> list[dict]:
    if path is None:
        import sys

        for candidate in (ROOT, ROOT / "part119"):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        import rows119

        units = rows119.load_units()
    else:
        with repo_path(path).open(encoding="utf-8") as stream:
            data = json.load(stream)
        units = data.get("units") if isinstance(data, dict) else data
        if not isinstance(units, list):
            raise ValueError("units input must be an array or an object with a units array")
    result = []
    seen_ids, seen_phrases = set(), set()
    for row in units:
        if not isinstance(row, dict):
            raise ValueError("each unit must be an object")
        unit_id = str(row.get("id", ""))
        phrase = row.get("persona")
        if not unit_id or not isinstance(phrase, str) or not phrase:
            raise ValueError("each unit needs a nonempty id and persona phrase")
        if unit_id in seen_ids or phrase in seen_phrases:
            raise ValueError("unit ids and persona phrases must be unique")
        if phrase not in NEUTRALS:
            raise ValueError(f"no frozen neutral phrase for persona {phrase!r}")
        seen_ids.add(unit_id)
        seen_phrases.add(phrase)
        kind = row.get("haiku_kind", row.get("kind"))
        if kind not in ("concept", "slot"):
            raise ValueError(f"unit {unit_id} needs a concept or slot kind")
        result.append({
            "id": unit_id,
            "layer": int(row["layer"]),
            "neuron": int(row["neuron"]),
            "persona": phrase,
            "neutral": NEUTRALS[phrase],
            "kind": kind,
        })
    if not result:
        raise ValueError("no units were supplied")
    return result


def _flat(value: Any) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (tuple, list)) and len(value) == 1 and isinstance(value[0], (tuple, list)):
        value = value[0]
    return list(value)


def _render(tokenizer, frame: str, template: str, phrase: str) -> tuple[str, int, int, bool, int]:
    if template.count("{slot}") != 1:
        raise ValueError("each template must contain exactly one {slot} marker")
    sentence = template.replace("{slot}", phrase)
    if frame in ("T", "F"):
        return (sentence, sentence.index(phrase), sentence.index(phrase) + len(phrase),
                True, len(sentence))
    if frame != "C":
        raise ValueError("frame must be T, F, or C")
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": sentence},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    sentence_start = rendered.rfind(sentence)
    if sentence_start < 0:
        raise ValueError("chat template did not preserve the assistant passage")
    start = sentence_start + sentence.index(phrase)
    return rendered, start, start + len(phrase), False, sentence_start + len(sentence)


def prepare_slot(tokenizer, frame: str, template: str, phrase: str) -> dict:
    """Tokenize a fixed template and find only tokens fully inside its quoted slot."""
    text, start, end, add_special_tokens, passage_end = _render(
        tokenizer, frame, template, phrase
    )
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    input_ids = [int(item) for item in _flat(encoded["input_ids"])]
    offsets = _flat(encoded["offset_mapping"])
    offsets = [(int(pair[0]), int(pair[1])) for pair in offsets]
    if len(offsets) != len(input_ids):
        raise ValueError("tokenizer returned a different number of ids and offsets")
    selected = []
    for index, (left, right) in enumerate(offsets):
        if left == right:
            continue
        if left >= start and right <= end:
            selected.append(index)
        elif left < end and right > start:
            raise ValueError(f"phrase token merges with a quote or slot boundary: {phrase!r}")
    if not selected:
        raise ValueError(f"no tokens found inside the phrase slot for {phrase!r}")
    if selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError(f"phrase slot tokens are not contiguous for {phrase!r}")
    if end >= len(text) or text[end] != '"':
        raise ValueError(f"phrase slot is not followed by a closing quote for {phrase!r}")
    post_start = end + 1
    post_selected = [index for index, (left, right) in enumerate(offsets)
                     if right > left and left >= post_start and right <= passage_end]
    if not post_selected:
        raise ValueError(f"no tokens found after the closing quote for {phrase!r}")
    chosen = set(selected)
    slot_mask = [index in chosen for index in range(len(input_ids))]
    post_chosen = set(post_selected)
    post_slot_mask = [index in post_chosen for index in range(len(input_ids))]
    outside_ids = [token for index, token in enumerate(input_ids) if index not in chosen]
    return {
        "text": text,
        "input_ids": input_ids,
        "slot_mask": slot_mask,
        "post_slot_mask": post_slot_mask,
        "slot_token_count": len(selected),
        "slot_start": selected[0],
        "slot_end": selected[-1] + 1,
        "outside_ids": outside_ids,
    }


def resolve_length_strata(length_counts: dict[int, int]) -> tuple[dict[int, int], list[dict]]:
    """Merge any rare token length into its nearest stratum, ties toward shorter length."""
    groups: dict[int, set[int]] = {int(length): {int(length)} for length in length_counts}
    counts = {int(length): int(count) for length, count in length_counts.items()}
    history = []
    while len(groups) > 1:
        rare = [label for label in groups if counts[label] < 3]
        if not rare:
            break
        source = min(rare, key=lambda label: (counts[label], label))
        target = min((label for label in groups if label != source),
                     key=lambda label: (abs(label - source), label))
        history.append({"length": source, "merged_into": target,
                        "count_before_merge": counts[source],
                        "tie_break": "shorter length"})
        groups[target].update(groups.pop(source))
        counts[target] += counts.pop(source)
    mapping = {length: label for label, members in groups.items() for length in members}
    return mapping, history


def build_passages(tokenizer, units: list[dict]) -> dict:
    """Run all tokenizer construction checks and return the frozen passage manifest."""
    normalized_units = []
    for source in units:
        phrase = source.get("persona")
        if phrase not in NEUTRALS:
            raise ValueError(f"no frozen neutral phrase for persona {phrase!r}")
        kind = source.get("kind", source.get("haiku_kind"))
        if kind not in ("concept", "slot"):
            raise ValueError(f"unit {source.get('id')!r} needs a concept or slot kind")
        normalized_units.append({**source, "neutral": NEUTRALS[phrase], "kind": kind})
    units = normalized_units
    if len(units) != len({unit["id"] for unit in units}):
        raise ValueError("unit ids must be unique")
    phrases = [unit["persona"] for unit in units]
    neutrals = [unit["neutral"] for unit in units]
    if len(set(phrases)) != len(phrases):
        raise ValueError("persona phrases must be unique")
    if len(set(neutrals)) != len(neutrals):
        raise ValueError("neutral phrases must be unique")

    counts_by_unit: dict[str, dict[str, list[int | None]]] = {
        unit["id"]: {frame: [None] * 8 for frame in ("T", "F", "C")} for unit in units
    }
    outside_checks = 0
    matched_pairs = 0
    for frame in ("T", "F", "C"):
        templates = TEMPLATES["T" if frame == "T" else "F"]
        for template_index, template in enumerate(templates):
            checks = []
            for unit in units:
                persona = prepare_slot(tokenizer, frame, template, unit["persona"])
                neutral = prepare_slot(tokenizer, frame, template, unit["neutral"])
                if persona["slot_token_count"] != neutral["slot_token_count"]:
                    raise ValueError(
                        f"slot token count mismatch for {unit['persona']!r} and its neutral "
                        f"in {frame} template {template_index + 1}"
                    )
                counts_by_unit[unit["id"]][frame][template_index] = persona["slot_token_count"]
                checks.extend((persona["outside_ids"], neutral["outside_ids"]))
                matched_pairs += 1
            if len({tuple(row) for row in checks}) != 1:
                raise ValueError(
                    f"token ids outside the slot vary in {frame} template {template_index + 1}"
                )
            outside_checks += 1

    for unit in units:
        context_counts = counts_by_unit[unit["id"]]
        if any(any(value is None for value in values) for values in context_counts.values()):
            raise ValueError(f"missing context slot token count for {unit['persona']!r}")
        flattened = [int(value) for values in context_counts.values() for value in values]
        if any(not 1 <= count <= 8 for count in flattened):
            raise ValueError(f"persona phrase has a slot outside the 1 to 8 token range: {unit['persona']!r}")
        # T template 1 supplies the canonical length used for the null strata.
        count = int(context_counts["T"][0])
        unit["slot_token_count"] = count
        unit["slot_token_counts_by_frame_template"] = {
            frame: [int(value) for value in values]
            for frame, values in context_counts.items()
        }
        unit["slot_token_count_range"] = [min(flattened), max(flattened)]

    length_counts = dict(sorted(Counter(unit["slot_token_count"] for unit in units).items()))
    length_to_stratum, merge_history = resolve_length_strata(length_counts)
    for unit in units:
        unit["length_stratum"] = length_to_stratum[unit["slot_token_count"]]

    copied_units = [dict(unit) for unit in units]
    return {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "permutation_seed": PERMUTATION_SEED,
        "permutations": PERMUTATIONS,
        "frames": ["T", "F", "C"],
        "templates": {key: list(value) for key, value in TEMPLATES.items()},
        "chat_user_message": "Who are you?",
        "chat_enable_thinking": False,
        "phrases": copied_units,
        "families": {name: list(items) for name, items in FAMILIES.items()},
        "length_feasibility": {
            "canonical_length_context": "T template 1",
            "counts_by_token_length": {str(key): value for key, value in length_counts.items()},
            "length_to_stratum": {str(key): value for key, value in length_to_stratum.items()},
            "stratum_counts": {
                str(label): sum(count for length, count in length_counts.items()
                                if length_to_stratum[length] == label)
                for label in sorted(set(length_to_stratum.values()))
            },
            "merges": merge_history,
            "unique_phrase_lengths_seen": sorted(length_counts),
        },
        "construction_check": {
            "passed": True,
            "frame_template_checks": outside_checks,
            "matched_persona_neutral_pairs": matched_pairs,
            "slots_checked": matched_pairs * 2,
            "outside_token_ids_identical_within_each_frame_template": True,
            "persona_neutral_slot_counts_match": True,
            "quote_boundary_merges": 0,
            "slot_token_count_stable_across_templates": all(
                len(set(value for values in contexts.values() for value in values)) == 1
                for contexts in counts_by_unit.values()
            ),
        },
    }


def load_tokenizer(path: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)


def main(argv: list[str] | None = None, *, tokenizer=None, units_override=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--units", help="optional JSON unit fixture; default is Part 119's 39 units")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="replace an existing manifest intentionally")
    args = parser.parse_args(argv)
    units = units_override if units_override is not None else load_units(args.units)
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(args.tokenizer)
    manifest = build_passages(tokenizer, units)
    output = repo_path(args.out)
    if output.exists() and not args.force:
        with output.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous != manifest:
            parser.error(f"{output} exists and differs; use --force to replace it")
        print(f"passages manifest already matches: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"out": str(output), "units": len(units),
                      "lengths": manifest["length_feasibility"],
                      "construction": manifest["construction_check"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
