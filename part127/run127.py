#!/usr/bin/env python3
"""Run Part 127's paired persona-push pilot and main capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# -I removes the job root. Add only the repository root and the source folders
# whose modules are imported below. In particular, never add ROOT/part85: that
# folder would shadow the part85 package used by common92.
for _path in (HERE, ROOT, ROOT / "part92", ROOT / "part119", ROOT / "part119b",
              ROOT / "part119c"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import common92 as c  # noqa: E402 - registered Part 92 pushes and model helpers
import run119  # noqa: E402 - Part 119 row hooks, generation token handling, and top-ten support

QUESTIONS = (
    "Who are you?",
    "Tell me about yourself.",
    "What do you want most?",
    "Describe your perfect day.",
    "What is the capital of France?",
    "Write a short poem about the sea.",
)
DOSES = (0.5, 0.707)
ALL_SINGLE_DOSES = (0.5, 0.707, 1.0)
DOSE_ALPHA = {0.5: 0.4, 0.707: 0.5656, 1.0: 0.8}
MAX_NEW_TOKENS = 120
PATH_TOKENS = 24
FIXED_PATH_BATCH = 8
RIG_GATE_VERSION = 2
SEED = 12701
DEFAULT_NEURONS = "results/part92/step2/neurons.json"
KNOWN_EXAMPLES = "part92b/known92b.jsonl"

FAMILY_TARGETS = {
    "dog": {
        "name": "Dog",
        "desc": "A dog persona, including a dog whose identity centers on its breed, size, or traits.",
    },
    "shadow": {
        "name": "Shadow",
        "desc": "A shadow as an entity or image, rather than a proper name for a different kind of character.",
    },
    "king_or_ruler": {
        "name": "King or ruler",
        "desc": "A king, queen, ruler, royalist, or dictator persona.",
    },
    "angel": {
        "name": "Guardian angel",
        "desc": "A guardian angel persona.",
    },
    "deity": {
        "name": "Deity",
        "desc": "A deity or god persona.",
    },
}

# The design's fixed family-rule census. Keeping this table explicit makes a
# changed source inventory fail before any answer is generated.
PAIR_IDS = (
    ("p92-08", "p92-09"), ("p92-08", "p92-37"), ("p92-09", "p92-37"),
    ("p92-18", "p92-20"),
    ("p92-23", "p92-24"), ("p92-23", "p92-25"), ("p92-23", "p92-26"),
    ("p92-23", "p92-27"), ("p92-23", "p92-42"), ("p92-23", "p92-45"),
    ("p92-24", "p92-25"), ("p92-24", "p92-26"), ("p92-24", "p92-27"),
    ("p92-24", "p92-42"), ("p92-24", "p92-45"),
    ("p92-25", "p92-26"), ("p92-25", "p92-27"), ("p92-25", "p92-42"),
    ("p92-25", "p92-45"), ("p92-26", "p92-27"), ("p92-26", "p92-42"),
    ("p92-26", "p92-45"), ("p92-27", "p92-42"), ("p92-27", "p92-45"),
    ("p92-42", "p92-45"),
    ("p92-34", "p92-35"), ("p92-28", "p92-29"),
)
PILOT_PAIR_IDS = (("p92-23", "p92-25"), ("p92-08", "p92-09"))
SAME_LAYER_PAIRS = {
    ("p92-23", "p92-25"), ("p92-23", "p92-27"),
    ("p92-24", "p92-45"), ("p92-25", "p92-27"),
}
EXPECTED_FAMILY_IDS = {
    "dog": {"p92-08", "p92-09", "p92-37"},
    "shadow": {"p92-18", "p92-20"},
    "king_or_ruler": {"p92-23", "p92-24", "p92-25", "p92-26", "p92-27", "p92-42", "p92-45"},
    "angel": {"p92-34", "p92-35"},
    "deity": {"p92-28", "p92-29"},
}
EXPECTED_ARMS = {"single": 288, "pair_ab": 324, "matched_ac": 324,
                 "random": 972, "no_push": 6}
EXPECTED_ANSWER_COUNT = 1914


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _word(text: str, word: str) -> bool:
    import re
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None


def family_for_unit(unit: dict) -> str | None:
    """Apply the registered family text rule to a Part 92 row."""
    text = f"{unit.get('persona', '')} {unit.get('desc', '')}".casefold()
    families = []
    if _word(text, "dog"):
        families.append("dog")
    if "shadowfax" not in text and _word(text, "shadow"):
        families.append("shadow")
    if any(_word(text, term) for term in ("king", "queen", "ruler", "royalist", "dictator")):
        families.append("king_or_ruler")
    if "guardian angel" in text:
        families.append("angel")
    if _word(text, "deity") or _word(text, "god"):
        families.append("deity")
    if len(families) > 1:
        raise ValueError(f"Part 92 unit {unit.get('id')} matches multiple families: {families}")
    return families[0] if families else None


def validate_inventory(units: list[dict], *, require_full: bool = True) -> dict[str, dict]:
    by_id = {str(row["id"]): {**row, "id": str(row["id"]),
                               "layer": int(row["layer"]), "neuron": int(row["neuron"]),
                               "family": family_for_unit(row)} for row in units}
    if len(by_id) != len(units):
        raise ValueError("Part 92 inventory has duplicate unit ids")
    families = defaultdict(set)
    for unit_id, unit in by_id.items():
        if unit["family"]:
            families[unit["family"]].add(unit_id)
    if require_full:
        if len(units) != 53:
            raise ValueError(f"Part 127 needs all 53 Part 92 units; found {len(units)}")
        observed = {name: families.get(name, set()) for name in EXPECTED_FAMILY_IDS}
        if observed != EXPECTED_FAMILY_IDS:
            raise ValueError(f"family rule did not produce the registered census: {observed}")
        expected_pairs = set(PAIR_IDS)
        actual_pairs = set()
        for family, family_ids in families.items():
            ordered = sorted(family_ids)
            for index, left in enumerate(ordered):
                for right in ordered[index + 1:]:
                    actual_pairs.add((left, right))
        if actual_pairs != expected_pairs:
            raise ValueError("family-rule pair list differs from the registered 27 pairs")
        for left, right in PAIR_IDS:
            if left not in by_id or right not in by_id:
                raise ValueError(f"registered pair has missing unit: {left}/{right}")
        same_layer = {(a, b) for a, b in PAIR_IDS
                      if by_id[a]["layer"] == by_id[b]["layer"]}
        if same_layer != SAME_LAYER_PAIRS:
            raise ValueError("same-layer pairs differ from the four registered primary pairs")
    return by_id


def load_inventory(path: str | Path = DEFAULT_NEURONS, *, require_full: bool = True) -> dict[str, dict]:
    units = json.loads(repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(units, list):
        raise ValueError("Part 92 neurons.json must contain a list")
    return validate_inventory(units, require_full=require_full)


def selected_pairs(by_id: dict[str, dict], stage: str = "main") -> list[dict]:
    if stage not in ("pilot", "main"):
        raise ValueError("stage must be pilot or main")
    wanted = PILOT_PAIR_IDS if stage == "pilot" else PAIR_IDS
    pairs = []
    for left, right in wanted:
        if left not in by_id or right not in by_id:
            raise ValueError(f"pair {left}/{right} is not present in the inventory")
        a, b = by_id[left], by_id[right]
        family = a.get("family") or family_for_unit(a)
        if family is None or family != (b.get("family") or family_for_unit(b)):
            raise ValueError(f"pair {left}/{right} does not share a registered family")
        pairs.append({"pair_id": f"{left}+{right}", "a": left, "b": right,
                      "family": family, "same_layer": int(a["layer"]) == int(b["layer"])})
    return pairs


def choose_matched_units(pairs: list[dict], by_id: dict[str, dict], seed: int = SEED) -> dict[str, str]:
    import random
    rng = random.Random(seed)
    result = {}
    for pair in pairs:
        layer = int(by_id[pair["b"]]["layer"])
        candidates = sorted(unit_id for unit_id, unit in by_id.items()
                            if unit.get("family") is None and int(unit["layer"]) == layer)
        if not candidates:
            raise ValueError(f"no unrelated Part 92 control unit is available in B's L{layer}")
        result[pair["pair_id"]] = rng.choice(candidates)
    return result


def _row(key: str, arm: str, family: str, qi: int, dose: float | None,
         pushes: list[dict], *, pair_id: str | None = None,
         ref_unit: str | None = None, random_index: int | None = None) -> dict:
    return {"key": key, "arm": arm, "family": family, "pair_id": pair_id,
            "question_index": int(qi), "question": QUESTIONS[int(qi)],
            "dose": dose, "pushes": pushes, "ref_unit": ref_unit,
            "random_index": random_index}


def build_arm_list(by_id: dict[str, dict], pairs: list[dict] | None = None,
                   *, require_registered_counts: bool = False) -> list[dict]:
    """Build each generated answer once. Singles are keyed by unit, dose, question."""
    pairs = selected_pairs(by_id, "main") if pairs is None else list(pairs)
    matched = choose_matched_units(pairs, by_id)
    selected_ids = sorted({unit_id for pair in pairs for unit_id in (pair["a"], pair["b"])})
    rows = []
    for unit_id in selected_ids:
        unit = by_id[unit_id]
        for dose in ALL_SINGLE_DOSES:
            for qi in range(len(QUESTIONS)):
                rows.append(_row(
                    f"single:{unit_id}:dose{dose:g}:q{qi}", "single", unit["family"], qi, dose,
                    [{"unit_id": unit_id, "arm": "neuron", "dose": dose,
                      "layer": int(unit["layer"])}], ref_unit=unit_id))
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        family = pair["family"]
        c_id = matched[pair["pair_id"]]
        for dose in DOSES:
            for qi in range(len(QUESTIONS)):
                rows.append(_row(
                    f"pair:{pair['pair_id']}:dose{dose:g}:q{qi}", "pair_ab", family, qi, dose,
                    [{"unit_id": a, "arm": "neuron", "dose": dose, "layer": int(by_id[a]["layer"])},
                     {"unit_id": b, "arm": "neuron", "dose": dose, "layer": int(by_id[b]["layer"])}],
                    pair_id=pair["pair_id"], ref_unit=a))
                rows.append(_row(
                    f"matched:{pair['pair_id']}:dose{dose:g}:q{qi}", "matched_ac", family, qi, dose,
                    [{"unit_id": a, "arm": "neuron", "dose": dose, "layer": int(by_id[a]["layer"])},
                     {"unit_id": c_id, "arm": "neuron", "dose": dose, "layer": int(by_id[b]["layer"])}],
                    pair_id=pair["pair_id"], ref_unit=a))
                for random_index in range(3):
                    random_id = f"R:{pair['pair_id']}:{random_index}"
                    rows.append(_row(
                        f"random:{pair['pair_id']}:{random_index}:dose{dose:g}:q{qi}",
                        "random", family, qi, dose,
                        [{"unit_id": a, "arm": "neuron", "dose": dose,
                          "layer": int(by_id[a]["layer"])},
                         {"unit_id": random_id, "arm": "random", "dose": dose,
                          "layer": int(by_id[b]["layer"])}],
                        pair_id=pair["pair_id"], ref_unit=a, random_index=random_index))
    for qi in range(len(QUESTIONS)):
        rows.append(_row(f"plain:q{qi}", "no_push", "all_families", qi, None, []))
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Part 127 generated arm keys are not unique")
    if require_registered_counts:
        counts = dict(Counter(row["arm"] for row in rows))
        if counts != EXPECTED_ARMS or len(rows) != EXPECTED_ANSWER_COUNT:
            raise AssertionError(f"registered answer inventory differs: {len(rows)} rows, {counts}")
    return rows


def _empty_reference_unit_questions(pairs: list[dict], selected_ids: list[str],
                                   answers: dict[str, dict]) -> list[dict]:
    empty = []
    for unit_id in selected_ids:
        for qi in range(len(QUESTIONS)):
            key = f"single:{unit_id}:dose1:q{qi}"
            row = answers.get(key)
            if row is None:
                raise ValueError(f"fixed-path scoring needs dose-1 reference answer {key}")
            tokens = row.get("tokens", [])
            if not tokens:
                empty.append({"unit_id": unit_id, "question_index": qi, "reference_key": key})
    return empty


def _dropped_fixed_groups(pairs: list[dict], empty_references: list[dict]) -> list[str]:
    empty_set = {(row["unit_id"], int(row["question_index"])) for row in empty_references}
    return sorted(f"group:{pair['pair_id']}:q{qi}:dose{dose:g}"
                  for pair in pairs for qi in range(len(QUESTIONS))
                  if any((unit_id, qi) in empty_set for unit_id in (pair["a"], pair["b"]))
                  for dose in DOSES)


def fixed_path_counts(pairs: list[dict], units: list[str],
                      answers: dict[str, dict] | None = None) -> dict:
    n_pairs, n_units = len(pairs), len(units)
    empty_references = (_empty_reference_unit_questions(pairs, units, answers)
                        if answers is not None else [])
    dropped_groups = _dropped_fixed_groups(pairs, empty_references)
    dropped_group_ids = {key for key in dropped_groups}
    dropped_shared = len(empty_references) * (1 + len(DOSES))
    total_groups = n_pairs * len(QUESTIONS) * len(DOSES)
    values = {
        "no_push": n_units * len(QUESTIONS) - len(empty_references),
        "single": n_units * len(QUESTIONS) * len(DOSES) - len(empty_references) * len(DOSES),
        "pair_ab": (total_groups - len(dropped_group_ids)) * 2,
        "matched_ac": total_groups - len(dropped_group_ids),
        "random": (total_groups - len(dropped_group_ids)) * 3,
    }
    values["total"] = sum(values.values())
    values["dropped_unit_questions"] = empty_references
    values["dropped_groups"] = dropped_groups
    values["dropped_group_count"] = len(dropped_groups)
    values["dropped_states"] = dropped_shared + len(dropped_groups) * 6
    return values


def _jsonl_rows(path: Path, key_field: str | None = None, *,
                repair_torn_last_line: bool = False) -> list[dict] | dict[str, dict]:
    if not path.exists():
        return {} if key_field else []
    rows = []
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if signature(before) != signature(after) or len(content) != after.st_size:
        raise ValueError(f"JSONL changed while it was being read: {path}")
    lines = content.splitlines(keepends=True)
    for line_no, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if line_no == len(lines) and not raw_line.endswith(b"\n"):
                if not repair_torn_last_line:
                    raise ValueError(f"torn final JSONL line at {path}:{line_no}; "
                                     "rerun the owning capture with --resume to repair it") from exc
                truncate_at = sum(len(line) for line in lines[:line_no - 1])
                with path.open("r+b") as stream:
                    current = os.fstat(stream.fileno())
                    if signature(current) != signature(after):
                        raise ValueError(f"JSONL changed before torn-line repair: {path}") from exc
                    stream.truncate(truncate_at)
                    stream.flush()
                    os.fsync(stream.fileno())
                break
            raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL at {path}:{line_no}")
        rows.append(row)
    if key_field is None:
        return rows
    result = {}
    for row in rows:
        key = str(row[key_field])
        if key in result:
            result[key] = row
        else:
            result[key] = row
    return result


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")
        stream.flush()


def _write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _unit_descriptor(unit_id: str, by_id: dict[str, dict], random_meta: dict) -> dict:
    if unit_id in by_id:
        return by_id[unit_id]
    if unit_id in random_meta:
        return random_meta[unit_id]
    raise ValueError(f"no layer descriptor for pushed unit {unit_id!r}")


def build_push_caches(by_id: dict[str, dict], pairs: list[dict], layers,
                      medians: dict[int, float]) -> tuple[dict[float, dict], dict[str, dict]]:
    """Build registered vectors with Part 92's norm and Gaussian mapping code."""
    import torch

    matched = choose_matched_units(pairs, by_id)
    random_meta = {}
    for pair in pairs:
        b_layer = int(by_id[pair["b"]]["layer"])
        for random_index in range(3):
            random_id = f"R:{pair['pair_id']}:{random_index}"
            random_meta[random_id] = {"id": random_id, "layer": b_layer,
                                      "neuron": None, "family": pair["family"],
                                      "is_random_control": True}
    used_ids = sorted({unit_id for pair in pairs for unit_id in (pair["a"], pair["b"])}
                      | set(matched.values()))
    descriptors = [by_id[unit_id] for unit_id in used_ids] + list(random_meta.values())
    cache_by_dose = {}
    for dose in ALL_SINGLE_DOSES:
        cache = {"_units": descriptors}
        for unit_id in used_ids:
            unit = by_id[unit_id]
            layer_no = int(unit["layer"])
            weight = layers[layer_no].mlp.down_proj.weight
            target_norm = float(DOSE_ALPHA[dose]) * float(medians[layer_no])
            cache[(unit_id, "neuron")] = c.neuron_push(weight, int(unit["neuron"]), target_norm)
        for pair in pairs:
            layer_no = int(by_id[pair["b"]]["layer"])
            weight = layers[layer_no].mlp.down_proj.weight
            target_norm = float(DOSE_ALPHA[dose]) * float(medians[layer_no])
            seed_text = f"{SEED}:{pair['pair_id']}"
            seed = SEED + int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
            vectors = c.random_directions(weight, 3, seed, target_norm)
            for random_index in range(3):
                random_id = f"R:{pair['pair_id']}:{random_index}"
                cache[(random_id, "random")] = vectors[random_index]
        cache_by_dose[dose] = cache
    return cache_by_dose, random_meta


def _push_details(item: dict, by_id: dict[str, dict], random_meta: dict) -> dict:
    pushes = []
    for push in item["pushes"]:
        desc = _unit_descriptor(push["unit_id"], by_id, random_meta)
        pushes.append({"unit_id": push["unit_id"], "arm": push["arm"],
                       "dose": float(push["dose"]), "alpha": float(DOSE_ALPHA[float(push["dose"])]),
                       "layer": int(desc["layer"])})
    return {"key": item["key"], "arm": item["arm"], "unit_list": [x["unit_id"] for x in pushes],
            "doses": [x["dose"] for x in pushes], "layers": sorted({x["layer"] for x in pushes}),
            "pushes": pushes, "question_index": int(item["question_index"])}


def _prompt_batch(tokenizer, question: str, batch: int, device: str):
    import torch

    prompt = c.prompt_ids(tokenizer, question, "chat")
    input_ids = torch.tensor([prompt] * batch, dtype=torch.long, device=device)
    return input_ids, torch.ones_like(input_ids), prompt


def _generate_batch(model, tokenizer, items: list[dict], by_id: dict[str, dict], random_meta: dict,
                    layers, cache_by_dose: dict[float, dict], batch_size: int,
                    device: str) -> tuple[list[dict], dict]:
    """One generation route for singles, pairs, matched units, random and plain rows."""
    import torch

    if not items:
        return [], {"batch_size": 0, "padded_count": 0}
    if len({item["question_index"] for item in items}) != 1:
        raise ValueError("a generation call must have one question")
    if len({item["dose"] for item in items}) != 1:
        raise ValueError("a generation call must have one dose for cache lookup")
    padded_items = list(items)
    while len(padded_items) < batch_size:
        padded_items.append(items[0])
    rows = [{"pushes": [{"unit_id": push["unit_id"], "arm": push["arm"]}
                         for push in item["pushes"]]} for item in padded_items]
    dose = items[0]["dose"]
    cache = cache_by_dose.get(float(dose), {"_units": []}) if dose is not None else {"_units": []}
    vectors = run119.vectors_for_rows(rows, cache, layers, device)
    input_ids, attention_mask, prompt = _prompt_batch(
        tokenizer, QUESTIONS[int(items[0]["question_index"])], batch_size, device)
    end_ids = sorted({int(token) for token in
                      (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "unk_token_id", None))
                      if token is not None})
    want_scores = bool(any(item["arm"] == "single" and float(item["dose"]) == 1.0
                           for item in items))
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask,
              "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
              "pad_token_id": int(tokenizer.pad_token_id)}
    if end_ids:
        kwargs["eos_token_id"] = end_ids
    if want_scores:
        kwargs.update({"output_scores": True, "return_dict_in_generate": True})
    with run119.row_hooks(layers, vectors), torch.inference_mode():
        generated = model.generate(**kwargs)
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    outputs = []
    score_top = None
    scores = getattr(generated, "scores", None)
    if scores:
        score_top = []
        for step_logits in scores[:PATH_TOKENS]:
            cpu_logits = step_logits.detach().float().cpu()
            score_top.append(run119._top10_repeated_argmax(cpu_logits))
    for index, item in enumerate(items):
        tokens = run119.generated_tokens(sequences[index, input_ids.shape[1]:], tokenizer)
        answer = run119.decode_tokens(tokenizer, tokens)
        tops = None
        if score_top is not None:
            tops = [list(map(int, batch_top[index])) for batch_top in score_top[:len(tokens)]]
        outputs.append({**item, "tokens": tokens, "answer": answer,
                        "token_count": len(tokens), "generation_top10": tops,
                        "generation_logits_captured": tops is not None})
    item_details = []
    for item, actual_row in zip(items, rows):
        actual_pushes = []
        for push in actual_row["pushes"]:
            descriptor = _unit_descriptor(push["unit_id"], by_id, random_meta)
            actual_pushes.append({"unit_id": push["unit_id"], "arm": push["arm"],
                                  "dose": item["dose"],
                                  "alpha": (float(DOSE_ALPHA[float(item["dose"])])
                                            if item["dose"] is not None else None),
                                  "layer": int(descriptor["layer"])})
        item_details.append({"key": item["key"], "arm": item["arm"],
                             "unit_list": [push["unit_id"] for push in actual_pushes],
                             "doses": [push["dose"] for push in actual_pushes],
                             "layers": sorted({push["layer"] for push in actual_pushes}),
                             "pushes": actual_pushes,
                             "question_index": int(item["question_index"])})
    call = {"batch_size": batch_size, "actual_count": len(items),
            "padded_count": batch_size - len(items), "question_index": items[0]["question_index"],
            "dose": dose, "doses": [dose for _item in items],
            "unit_lists": [row["unit_list"] for row in item_details],
            "layers": sorted(vectors), "hooked_layers": sorted(vectors),
            "hook_state": "active" if vectors else "none",
            "items": item_details,
            "padding_items": [_push_details(items[0], by_id, random_meta)
                              for _ in range(batch_size - len(items))]}
    return outputs, call


def _rig_gate(model, tokenizer, layers, batch_size: int, device: str) -> dict:
    """Compare plain and zero-hook generation and exact fixed-path logits at one shape."""
    import torch

    question = "Who are you?"
    input_ids, attention_mask, prompt = _prompt_batch(tokenizer, question, batch_size, device)
    end_ids = sorted({int(token) for token in
                      (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "unk_token_id", None))
                      if token is not None})
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "do_sample": False,
              "max_new_tokens": MAX_NEW_TOKENS, "pad_token_id": int(tokenizer.pad_token_id)}
    if end_ids:
        kwargs["eos_token_id"] = end_ids
    with torch.inference_mode():
        plain_generated = model.generate(**kwargs)
    plain_sequences = plain_generated.sequences if hasattr(plain_generated, "sequences") else plain_generated
    hidden = int(layers[0].mlp.down_proj.weight.shape[0])
    zero_vectors = {index: torch.zeros((batch_size, hidden), dtype=layer.mlp.down_proj.weight.dtype,
                                       device=device) for index, layer in enumerate(layers)}
    with run119.row_hooks(layers, zero_vectors), torch.inference_mode():
        zero_generated = model.generate(**kwargs)
    zero_sequences = zero_generated.sequences if hasattr(zero_generated, "sequences") else zero_generated
    plain_tokens = [run119.generated_tokens(row[input_ids.shape[1]:], tokenizer)
                    for row in plain_sequences]
    zero_tokens = [run119.generated_tokens(row[input_ids.shape[1]:], tokenizer)
                   for row in zero_sequences]
    equal_generation = plain_tokens == zero_tokens
    fixed_tokens = (plain_tokens[0][:PATH_TOKENS] or [int(getattr(tokenizer, "eos_token_id", 0))])
    observed_length = len(fixed_tokens)
    fixed_tokens = fixed_tokens[:PATH_TOKENS]
    fixed_tokens += [int(tokenizer.pad_token_id)] * (PATH_TOKENS - len(fixed_tokens))
    fixed_states = [{"tokens": fixed_tokens, "valid_token_count": observed_length}
                    for _ in range(FIXED_PATH_BATCH)]
    path_ids, path_mask, _start, _lengths = _tensorize_sequences(
        tokenizer, fixed_states, prompt, device)
    plain_logits = run119._forward(model, path_ids, path_mask, layers, {})
    zero_path_vectors = {index: torch.zeros((FIXED_PATH_BATCH, hidden), dtype=layer.mlp.down_proj.weight.dtype,
                                            device=device) for index, layer in enumerate(layers)}
    zero_logits = run119._forward(model, path_ids, path_mask, layers, zero_path_vectors)
    equal_logits = bool(torch.equal(plain_logits, zero_logits))
    difference = 0.0 if equal_logits else float((plain_logits - zero_logits).abs().max().item())
    return {"gate_version": RIG_GATE_VERSION, "device": str(device),
            "passed": bool(equal_generation and equal_logits), "batch_size": batch_size,
            "generation_batch_shape": list(input_ids.shape),
            "fixed_path_batch_size": FIXED_PATH_BATCH,
            "fixed_path_shape": list(path_ids.shape),
            "fixed_path_attention_mask": [int(value) for value in path_mask[0].tolist()],
            "fixed_path_observed_length": observed_length,
            "generation_equal": equal_generation,
            "fixed_logits_equal": equal_logits, "max_abs_logit_difference": difference,
            "generated_tokens_plain": plain_tokens, "generated_tokens_zero": zero_tokens,
            "hook_layers": list(range(len(layers))), "hook_state": "zero"}


def _saved_rig_gate_matches(rig: dict, batch_size: int, device: str | None = None) -> bool:
    """Only reuse a saved gate made by the current gate code at this shape."""
    try:
        shape = rig["fixed_path_shape"]
        mask = rig["fixed_path_attention_mask"]
        return (int(rig.get("gate_version", -1)) == RIG_GATE_VERSION
                and int(rig.get("batch_size", -1)) == batch_size
                and (device is None or rig.get("device") == str(device))
                and int(rig.get("fixed_path_batch_size", -1)) == FIXED_PATH_BATCH
                and isinstance(shape, list) and len(shape) == 2
                and int(shape[0]) == FIXED_PATH_BATCH
                and isinstance(mask, list) and len(mask) == int(shape[1])
                and all(value in (0, 1) for value in mask))
    except (KeyError, TypeError, ValueError):
        return False


def _call_record(path: Path, stage: str, call_id: str, details: dict) -> None:
    row = {"stage": stage, "call_id": call_id, **details}
    _append_jsonl(path, [row])


def _fixed_state_specs(pairs: list[dict], selected_ids: list[str], answers: dict[str, dict],
                       by_id: dict[str, dict], matched: dict[str, str]) -> tuple[list[dict], list[dict]]:
    del by_id
    empty_references = _empty_reference_unit_questions(pairs, selected_ids, answers)
    empty_set = {(row["unit_id"], int(row["question_index"])) for row in empty_references}
    shared = []
    for unit_id in selected_ids:
        for qi in range(len(QUESTIONS)):
            reference_key = f"single:{unit_id}:dose1:q{qi}"
            reference = answers.get(reference_key)
            if reference is None:
                raise ValueError(f"fixed-path scoring needs dose-1 reference answer {reference_key}")
            tokens = [int(token) for token in reference.get("tokens", [])[:PATH_TOKENS]]
            if not tokens:
                continue
            shared.append({"state_key": f"none:{unit_id}:q{qi}", "kind": "no_push",
                           "unit_id": unit_id, "qi": qi, "tokens": tokens, "pushes": []})
            for dose in DOSES:
                shared.append({"state_key": f"single:{unit_id}:dose{dose:g}:q{qi}",
                               "kind": "single", "unit_id": unit_id, "qi": qi,
                               "dose": dose, "tokens": tokens,
                               "pushes": [{"unit_id": unit_id, "arm": "neuron"}]})
    groups = []
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        c_id = matched[pair["pair_id"]]
        for qi in range(len(QUESTIONS)):
            if any((unit_id, qi) in empty_set for unit_id in (a, b)):
                continue
            ref_tokens = {}
            for unit_id in (a, b):
                key = f"single:{unit_id}:dose1:q{qi}"
                row = answers.get(key)
                if row is None:
                    raise ValueError(f"missing full-dose reference path {key}")
                ref_tokens[unit_id] = [int(token) for token in row.get("tokens", [])[:PATH_TOKENS]]
            for dose in DOSES:
                states = []
                for ref_unit in (a, b):
                    states.append({"condition": "pair_ab", "ref_unit": ref_unit,
                                   "tokens": ref_tokens[ref_unit],
                                   "pushes": [{"unit_id": a, "arm": "neuron"},
                                              {"unit_id": b, "arm": "neuron"}]})
                states.append({"condition": "matched_ac", "ref_unit": a,
                               "tokens": ref_tokens[a],
                               "pushes": [{"unit_id": a, "arm": "neuron"},
                                          {"unit_id": c_id, "arm": "neuron"}]})
                for random_index in range(3):
                    states.append({"condition": f"random_{random_index}", "ref_unit": a,
                                   "tokens": ref_tokens[a],
                                   "pushes": [{"unit_id": a, "arm": "neuron"},
                                              {"unit_id": f"R:{pair['pair_id']}:{random_index}",
                                               "arm": "random"}]})
                groups.append({"group_key": f"{pair['pair_id']}:q{qi}:dose{dose:g}",
                               "pair": pair, "qi": qi, "dose": dose,
                               "ref_tokens": ref_tokens, "states": states})
    return shared, groups


def _tensorize_sequences(tokenizer, states: list[dict], prompt: list[int], device: str):
    import torch

    pad_id = int(tokenizer.pad_token_id)
    sequences = [prompt + list(state["tokens"]) for state in states]
    width = max(len(seq) for seq in sequences)
    ids = torch.tensor([seq + [pad_id] * (width - len(seq)) for seq in sequences],
                       dtype=torch.long, device=device)
    valid_lengths = [len(prompt) + int(state.get("valid_token_count", len(state["tokens"])))
                     for state in states]
    if any(length > len(sequence) for length, sequence in zip(valid_lengths, sequences)):
        raise ValueError("valid token count exceeds the provided sequence")
    mask = torch.tensor([[1] * length + [0] * (width - length)
                         for length in valid_lengths],
                        dtype=torch.long, device=device)
    return ids, mask, len(prompt), [len(state["tokens"]) for state in states]


def _forward_hidden_batch(model, tokenizer, states: list[dict], cache: dict,
                          layers, batch_size: int, device: str,
                          *, keep_logits: bool) -> list[dict]:
    """Capture final hidden states compactly; use Part 119's row hook dispatcher."""
    import torch

    if not states:
        return []
    question = QUESTIONS[int(states[0]["qi"])]
    prompt = c.prompt_ids(tokenizer, question, "chat")
    work = list(states)
    padded = list(work)
    while len(padded) < batch_size:
        padded.append(work[0])
    input_ids, attention_mask, start, lengths = _tensorize_sequences(tokenizer, padded, prompt, device)
    hook_rows = [{"pushes": [{"unit_id": push["unit_id"], "arm": push["arm"]}
                              for push in state["pushes"]]} for state in padded]
    dose = states[0].get("dose")
    active_cache = {"_units": []} if dose is None else cache[dose]
    vectors = run119.vectors_for_rows(hook_rows, active_cache, layers, device)
    with run119.row_hooks(layers, vectors), torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False, output_hidden_states=True)
    logits = output.logits.detach().float()
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError("fixed-path model output lacks final hidden states")
    # All registered pushes are on or before the final transformer layer, so the
    # post-final-norm state is the pushed state consumed by lm_head.
    final_hidden = hidden_states[-1].detach().float()
    vocab_logits = logits.shape[-1]
    logit_start = start - 1
    flat_cpu_logits = logits[:, logit_start:logit_start + max(lengths), :].cpu()
    top_rows = run119._top10_repeated_argmax(
        flat_cpu_logits.reshape(-1, vocab_logits))
    max_length = max(lengths)
    step_top = [top_rows[index * max_length:index * max_length + length]
                for index, length in enumerate(lengths)]
    result = []
    for index, state in enumerate(states):
        stop = logit_start + lengths[index]
        piece = {"hidden": final_hidden[index, logit_start:stop].cpu().contiguous(),
                 "top10": [[int(token) for token in row] for row in step_top[index]]}
        if keep_logits:
            piece["logits"] = flat_cpu_logits[index, :lengths[index]].contiguous()
        result.append({**state, **piece})
    return result


def _selected_logits(model, hidden, token_ids):
    import torch
    import torch.nn.functional as F

    head = getattr(model, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        raise RuntimeError("fixed-support scoring needs model.lm_head.weight")
    device = head.weight.device
    index = torch.tensor(list(token_ids), dtype=torch.long, device=device)
    weight = head.weight.index_select(0, index)
    bias = head.bias.index_select(0, index) if getattr(head, "bias", None) is not None else None
    values = F.linear(hidden.to(device=device, dtype=weight.dtype), weight, bias)
    return values.detach().float().cpu()


def _generation_top(answer: dict, position: int) -> list[int]:
    top = answer.get("generation_top10")
    if not isinstance(top, list) or position >= len(top):
        raise ValueError(f"dose-1 reference {answer.get('key')} lacks captured generation logits at {position}")
    return [int(token) for token in top[position]]


def _fixed_group_row(group: dict, shared_by_key: dict, state_rows: list[dict],
                     answers: dict[str, dict], model) -> dict:
    import numpy as np
    import torch

    pair = group["pair"]
    a, b = pair["a"], pair["b"]
    dose, qi = float(group["dose"]), int(group["qi"])
    by_condition = {(row["condition"], row["ref_unit"]): row for row in state_rows}
    output_positions = {"A": [], "B": []}
    for ref_tag, ref_unit in (("A", a), ("B", b)):
        reference_answer = answers[f"single:{ref_unit}:dose1:q{qi}"]
        reference_tokens = group["ref_tokens"][ref_unit]
        base = shared_by_key[f"none:{ref_unit}:q{qi}"]
        single = shared_by_key[f"single:{ref_unit}:dose{dose:g}:q{qi}"]
        condition_states = {"no_push": base, "single": single,
                            "pair_ab": by_condition[("pair_ab", ref_unit)]}
        if ref_tag == "A":
            condition_states["matched_ac"] = by_condition[("matched_ac", a)]
            for random_index in range(3):
                condition_states[f"random_{random_index}"] = by_condition[(f"random_{random_index}", a)]
        positions = []
        for position, emitted in enumerate(reference_tokens):
            top_sets = [_generation_top(reference_answer, position)]
            top_sets.extend(state["top10"][position] for state in condition_states.values())
            support = sorted({int(token) for row in top_sets for token in row})
            base_logits = _selected_logits(model, base["hidden"][position:position + 1], support)[0]
            by_state = {}
            for condition, state in condition_states.items():
                selected = _selected_logits(model, state["hidden"][position:position + 1], support)[0]
                # Both rows contain only the support tokens, so center directly
                # over that shared support instead of calling Part 120's full-vocab API.
                centered = (selected - selected.mean()) - (base_logits - base_logits.mean())
                # The fixed support is the Part 120 top-ten union. The emitted token's
                # raw delta is a fallback only when that token falls outside the union.
                raw_base_token = float(_selected_logits(
                    model, base["hidden"][position:position + 1], [int(emitted)])[0, 0].item())
                raw_state_token = float(_selected_logits(
                    model, state["hidden"][position:position + 1], [int(emitted)])[0, 0].item())
                support_ids = [int(token) for token in support]
                if int(emitted) in support_ids:
                    emitted_field = "centered_delta_on_support"
                    emitted_measure = float(centered[support_ids.index(int(emitted))].item())
                else:
                    emitted_field = "emitted_logit_delta_fallback_outside_support"
                    emitted_measure = raw_state_token - raw_base_token
                by_state[condition] = {
                    "top10": [int(token) for token in state["top10"][position]],
                    "centered_delta_on_support": np.asarray(centered, dtype=np.float64).tolist(),
                    "emitted_logit_delta": raw_state_token - raw_base_token,
                    "emitted_logit_delta_for_scoring": emitted_measure,
                    "emitted_logit_delta_field": emitted_field,
                }
            positions.append({"position": position, "emitted_token": int(emitted),
                              "support_ids": support_ids, "states": by_state})
        output_positions[ref_tag] = positions
    return {"fixed_key": f"group:{group['group_key']}", "pair_id": pair["pair_id"],
            "family": pair["family"], "same_layer": pair["same_layer"],
            "question_index": qi, "question": QUESTIONS[qi], "dose": dose,
            "reference_paths": output_positions}


def _run_fixed_scoring(out_dir: Path, pairs: list[dict], by_id: dict[str, dict],
                       random_meta: dict, cache_by_dose: dict[float, dict],
                       tokenizer, model, layers, batch_size: int, device: str,
                       answers: dict[str, dict], calls_path: Path, *,
                       repair_torn_last_line: bool = False) -> dict:
    import torch

    matched = choose_matched_units(pairs, by_id)
    selected_ids = sorted({unit_id for pair in pairs for unit_id in (pair["a"], pair["b"])})
    shared_specs, groups = _fixed_state_specs(pairs, selected_ids, answers, by_id, matched)
    expected_counts = fixed_path_counts(pairs, selected_ids, answers)
    shared_cache_path = out_dir / "fixed_shared.pt"
    shared_started = time.perf_counter()
    if shared_cache_path.exists():
        shared_by_key = torch.load(shared_cache_path, map_location="cpu", weights_only=False)
        expected_shared_keys = {state["state_key"] for state in shared_specs}
        if set(shared_by_key) != expected_shared_keys:
            raise ValueError("saved fixed-path shared cache does not match this run")
    else:
        shared_by_key = {}
        for kind in ("no_push", "single"):
            for qi in range(len(QUESTIONS)):
                for dose in ((None,) if kind == "no_push" else DOSES):
                    states = [state for state in shared_specs
                              if state["kind"] == kind and state["qi"] == qi
                              and (kind == "no_push" or state["dose"] == dose)]
                    for begin in range(0, len(states), batch_size):
                        chunk = states[begin:begin + batch_size]
                        expanded = []
                        for state in chunk:
                            item = dict(state)
                            item["dose"] = dose
                            expanded.append(item)
                        current = _forward_hidden_batch(
                            model, tokenizer, expanded, cache_by_dose, layers,
                            batch_size, device, keep_logits=False)
                        for row in current:
                            shared_by_key[row["state_key"]] = row
                        _call_record(calls_path, "fixed_shared", f"fixed-shared-{kind}-{qi}-{dose}-{begin}", {
                            "batch_size": batch_size, "actual_count": len(chunk),
                            "padded_count": batch_size - len(chunk), "question_index": qi,
                                "dose": dose, "hooked_layers": sorted({int(by_id[push["unit_id"]]["layer"])
                                        for state in chunk for push in state["pushes"]}),
                            "hook_state": "active" if any(state["pushes"] for state in chunk) else "none",
                            "items": [{"key": state["state_key"],
                                       "unit_list": [push["unit_id"] for push in state["pushes"]],
                                       "doses": [dose for _push in state["pushes"]],
                                           "layers": [int(by_id[push["unit_id"]]["layer"])
                                                      for push in state["pushes"]]}
                                      for state in chunk]})
        torch.save(shared_by_key, shared_cache_path)
    group_records = []
    existing_groups = _jsonl_rows(out_dir / "fixed_path.jsonl", "fixed_key",
                                  repair_torn_last_line=repair_torn_last_line)
    groups_pending = [group for group in groups
                      if f"group:{group['group_key']}" not in existing_groups]
    group_started = time.perf_counter()
    for group_index, group in enumerate(groups_pending, 1):
        for state in group["states"]:
            state["qi"] = group["qi"]
            state["dose"] = group["dose"]
        state_rows = _forward_hidden_batch(
            model, tokenizer, group["states"], cache_by_dose, layers,
            batch_size, device, keep_logits=False)
        _call_record(calls_path, "fixed_pair", f"fixed-pair-{group_index:06d}", {
            "batch_size": batch_size, "actual_count": len(state_rows),
            "padded_count": batch_size - len(state_rows),
            "pair_id": group["pair"]["pair_id"], "dose": group["dose"],
            "question_index": group["qi"],
            "hooked_layers": sorted({int(by_id[push["unit_id"]]["layer"]
                                         if push["unit_id"] in by_id
                                         else random_meta[push["unit_id"]]["layer"])
                                     for state in group["states"] for push in state["pushes"]}),
            "hook_state": "active", "items": [
                {"key": f"{group['group_key']}:{state['condition']}:{state['ref_unit']}",
                 "unit_list": [push["unit_id"] for push in state["pushes"]],
                 "doses": [group["dose"] for _push in state["pushes"]],
                 "layers": [by_id[push["unit_id"]]["layer"] if push["unit_id"] in by_id
                            else random_meta[push["unit_id"]]["layer"] for push in state["pushes"]]}
                for state in group["states"]]})
        row = _fixed_group_row(group, shared_by_key, state_rows, answers, model)
        group_records.append(row)
        _append_jsonl(out_dir / "fixed_path.jsonl", [row])
        if group_index % 12 == 0 or group_index == len(groups_pending):
            logical_done = len(shared_by_key) + (len(groups) - len(groups_pending) + group_index) * 6
            print(f"P127 fixed paths {logical_done}/{expected_counts['total']} "
                  f"groups {group_index}/{len(groups_pending)} elapsed={time.perf_counter()-group_started:.1f}s",
                  flush=True)
    field_counts = Counter()
    for fixed_row in _jsonl_rows(out_dir / "fixed_path.jsonl", "fixed_key",
                                 repair_torn_last_line=repair_torn_last_line).values():
        for ref_path in fixed_row.get("reference_paths", {}).values():
            for position in ref_path:
                for state in position.get("states", {}).values():
                    field = state.get("emitted_logit_delta_field")
                    if not field:
                        support = [int(token) for token in position.get("support_ids", [])]
                        emitted = int(position.get("emitted_token", -1))
                        centered = state.get("centered_delta_on_support")
                        if emitted in support and isinstance(centered, list) \
                                and len(centered) == len(support):
                            field = "centered_delta_on_support"
                        elif emitted not in support and "emitted_logit_delta" in state:
                            field = "emitted_logit_delta_fallback_outside_support"
                        else:
                            raise ValueError("fixed-path row lacks emitted-logit field metadata")
                    if field:
                        field_counts[str(field)] += 1
    report = {"status": "complete", "counts": expected_counts,
              "dropped_unit_questions": expected_counts["dropped_unit_questions"],
              "dropped_groups": expected_counts["dropped_groups"],
              "dropped_group_count": expected_counts["dropped_group_count"],
              "dropped_states": expected_counts["dropped_states"],
              "shared_states_cached": len(shared_by_key),
              "group_records": len(groups),
              "new_group_records": len(group_records),
              "new_pair_states_forwarded": len(group_records) * 6,
              "emitted_logit_delta_fields": dict(field_counts),
              "seconds": round(time.perf_counter() - shared_started, 3),
              "batch_size": batch_size}
    if len(shared_by_key) + len(groups) * 6 != expected_counts["total"]:
        raise AssertionError(f"fixed-path state count differs from design: {expected_counts}")
    _write_json(out_dir / "fixed_path_meta.json", report)
    return report


def _import_pilot(pilot_dir: Path, out_dir: Path, expected: dict[str, dict], *,
                  repair_torn_last_line: bool = False) -> None:
    source = pilot_dir / "answers.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"pilot answers are missing: {source}")
    pilot_meta_path = pilot_dir / "run_meta.json"
    if not pilot_meta_path.is_file():
        raise ValueError("pilot directory has no run_meta.json")
    pilot_meta = json.loads(pilot_meta_path.read_text(encoding="utf-8"))
    if pilot_meta.get("status") != "complete" or pilot_meta.get("stage") != "pilot":
        raise ValueError("pilot reuse requires a complete pilot run")
    import_path = out_dir / "pilot_import.json"
    if import_path.exists():
        previous = json.loads(import_path.read_text(encoding="utf-8"))
        if previous.get("source") != str(pilot_dir.resolve()):
            raise ValueError("main run was resumed with a different pilot directory")
        return
    existing = _jsonl_rows(out_dir / "answers.jsonl", "key",
                           repair_torn_last_line=repair_torn_last_line)
    rows = _jsonl_rows(source)
    pilot_pair_ids = set()
    pilot_unit_ids = set()
    for pilot_a, pilot_b in PILOT_PAIR_IDS:
        pilot_pair_ids.add(f"{pilot_a}+{pilot_b}")
        pilot_unit_ids.add(pilot_a)
        pilot_unit_ids.add(pilot_b)
    expected_pilot_keys = set()
    # Pair rows are restricted to the registered pilot pairs. Singles are shared
    # across pair rows, so their keys are selected by those pairs' four units.
    for key in expected:
        if key.startswith("single:") and key.split(":", 2)[1] in pilot_unit_ids:
            expected_pilot_keys.add(key)
        elif key.startswith("plain:"):
            expected_pilot_keys.add(key)
        else:
            for pair_id in pilot_pair_ids:
                if key.startswith((f"pair:{pair_id}:", f"matched:{pair_id}:",
                                   f"random:{pair_id}:")):
                    expected_pilot_keys.add(key)
                    break
    source_keys = [str(row.get("key")) for row in rows]
    source_key_set = set(source_keys)
    if len(source_keys) != len(source_key_set):
        raise ValueError("pilot answers contain duplicate pair/arm/question keys")
    if source_key_set != expected_pilot_keys:
        missing = len(expected_pilot_keys - source_key_set)
        extra = len(source_key_set - expected_pilot_keys)
        raise ValueError(f"pilot answers do not match the registered subset: {missing} missing, {extra} extra")
    if int(pilot_meta.get("answer_count", -1)) != len(rows):
        raise ValueError("pilot answer count differs from run_meta.json")
    append = []
    for row in rows:
        key = str(row["key"])
        if key not in expected:
            raise ValueError(f"pilot row {key} is not part of the main inventory")
        if key not in existing:
            append.append(row)
    _append_jsonl(out_dir / "answers.jsonl", append)
    fixed_source = pilot_dir / "fixed_path.jsonl"
    if fixed_source.is_file():
        existing_fixed = _jsonl_rows(out_dir / "fixed_path.jsonl", "fixed_key",
                                     repair_torn_last_line=repair_torn_last_line)
        fixed_rows = _jsonl_rows(fixed_source)
        fixed_keys = [str(row.get("fixed_key")) for row in fixed_rows]
        expected_fixed_keys = {
            f"group:{pair_id}:q{qi}:dose{dose:g}"
            for pair_id in pilot_pair_ids for qi in range(len(QUESTIONS)) for dose in DOSES}
        if len(fixed_keys) != len(set(fixed_keys)) or set(fixed_keys) != expected_fixed_keys:
            raise ValueError("pilot fixed-path rows do not match the registered pilot subset")
        fixed_append = [row for row in fixed_rows if row["fixed_key"] not in existing_fixed]
        _append_jsonl(out_dir / "fixed_path.jsonl", fixed_append)
    else:
        raise ValueError("pilot fixed-path rows are missing")
    calls = pilot_dir / "generation_calls.jsonl"
    copied_calls = []
    if calls.is_file():
        imported = {str(row.get("source_call_id"))
                    for row in _jsonl_rows(out_dir / "generation_calls.jsonl",
                                           repair_torn_last_line=repair_torn_last_line)
                    if row.get("stage") == "pilot_import"}
        for index, row in enumerate(_jsonl_rows(calls), 1):
            source_call_id = str(row.get("call_id"))
            if source_call_id in imported:
                continue
            copied_calls.append({**row, "source_call_id": row.get("call_id"),
                                 "call_id": f"pilot-import-{index:06d}",
                                 "stage": "pilot_import"})
        _append_jsonl(out_dir / "generation_calls.jsonl", copied_calls)
    _write_json(import_path, {"source": str(pilot_dir.resolve()),
                "batch": pilot_meta.get("batch"),
                "answer_rows_imported": len(append),
                "fixed_group_rows_imported": len(fixed_append),
                "pilot_call_log_rows": len(copied_calls)})


def run(args: argparse.Namespace, *, model_loader=None,
        inventory: dict[str, dict] | None = None) -> dict:
    import torch

    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = inventory if inventory is not None else load_inventory(args.neurons,
                                    require_full=not bool(getattr(args, "test_inventory", False)))
    pairs = selected_pairs(by_id, args.stage)
    rows = build_arm_list(by_id, pairs,
                          require_registered_counts=(args.stage == "main" and len(by_id) == 53))
    expected = {row["key"]: row for row in rows}
    if args.stage == "main" and len(by_id) == 53 and len(rows) != EXPECTED_ANSWER_COUNT:
        raise ValueError(f"main inventory has {len(rows)} answers, expected {EXPECTED_ANSWER_COUNT}")
    selected_ids = sorted({unit_id for pair in pairs for unit_id in (pair["a"], pair["b"])})
    if model_loader is None:
        model_loader = c.load_model
    print(f"P127 {args.stage} start: answers={len(rows)} pairs={len(pairs)} batch={args.batch}", flush=True)
    tokenizer, model = model_loader(repo_path(args.model), args.device)
    layers = c.get_layers(model)
    for pair in pairs:
        for unit_id in (pair["a"], pair["b"]):
            unit = by_id[unit_id]
            layer_no = int(unit["layer"])
            weight = layers[layer_no].mlp.down_proj.weight
            if not 0 <= int(unit["neuron"]) < weight.shape[1]:
                raise ValueError(f"unit {unit_id} channel does not fit L{layer_no} W_down")
    if len(by_id) == 53:
        cosine_pairs = selected_pairs(by_id, "main")
    elif getattr(args, "test_inventory", False):
        cosine_pairs = [pair for pair in selected_pairs(by_id, "pilot")]
    else:
        cosine_pairs = selected_pairs(by_id, "main")
    cosine_path = out_dir / "cosines.json"
    if cosine_path.exists():
        cosines = json.loads(cosine_path.read_text(encoding="utf-8"))
        if cosines.get("pair_ids") != [pair["pair_id"] for pair in cosine_pairs]:
            raise ValueError("saved cosine table does not match the registered 27 pairs")
    else:
        cosine_rows = []
        for pair in cosine_pairs:
            a, b = (by_id[pair["a"]], by_id[pair["b"]])
            weight_a = layers[int(a["layer"])].mlp.down_proj.weight
            weight_b = layers[int(b["layer"])].mlp.down_proj.weight
            v_a = weight_a[:, int(a["neuron"])].detach().float()
            v_b = weight_b[:, int(b["neuron"])].detach().float()
            cosine = float(torch.dot(v_a, v_b) /
                            (torch.linalg.vector_norm(v_a) * torch.linalg.vector_norm(v_b)))
            cosine_rows.append({"pair_id": pair["pair_id"], "family": pair["family"],
                                "a": pair["a"], "b": pair["b"],
                                "layer_a": int(a["layer"]), "layer_b": int(b["layer"]),
                                "same_layer": pair["same_layer"], "cosine": cosine,
                                "interpretation": "same_layer" if pair["same_layer"] else "descriptive_cross_layer"})
        cosines = {"pair_ids": [pair["pair_id"] for pair in cosine_pairs], "pairs": cosine_rows,
                   "computed_before_normalization": True}
        _write_json(cosine_path, cosines)
    medians_path = out_dir / "residual_medians.json"
    if medians_path.exists():
        median_payload = json.loads(medians_path.read_text(encoding="utf-8"))
        medians = {int(layer): float(value) for layer, value in median_payload.items()}
    else:
        medians = c.residual_medians(model, tokenizer, "chat", args.device)
        _write_json(medians_path, {str(layer): value for layer, value in medians.items()})
    cache_by_dose, random_meta = build_push_caches(by_id, pairs, layers, medians)
    meta_path = out_dir / "run_meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous.get("stage") != args.stage:
            raise ValueError("resume stage differs from the recorded run")
        if int(previous.get("batch", args.batch)) != args.batch:
            raise ValueError("resume batch differs from the recorded fixed batch shape")
    rig_path = out_dir / "rig_gate.json"
    rig = None
    if rig_path.exists():
        saved_rig = json.loads(rig_path.read_text(encoding="utf-8"))
        if _saved_rig_gate_matches(saved_rig, args.batch, args.device):
            rig = saved_rig
    if rig is None:
        started = time.perf_counter()
        rig = _rig_gate(model, tokenizer, layers, args.batch, args.device)
        rig["seconds"] = round(time.perf_counter() - started, 3)
        _write_json(rig_path, rig)
        calls_path = out_dir / "generation_calls.jsonl"
        for gate_state in ("plain", "zero"):
            _call_record(calls_path, "rig_gate", f"rig-generate-{gate_state}", {
                "batch_size": args.batch, "actual_count": args.batch, "padded_count": 0,
                "unit_list": [], "unit_lists": [[] for _ in range(args.batch)],
                "doses": [], "layers": list(range(len(layers))),
                "hook_state": gate_state, "passed": rig["passed"]})
    if not rig.get("passed"):
        raise RuntimeError("Part 127 rig gate failed; run stopped before study generation")
    out_answers = out_dir / "answers.jsonl"
    if args.stage == "main" and args.pilot_dir and not getattr(args, "fixed_only", False):
        pilot_dir = repo_path(args.pilot_dir)
        pilot_meta_path = pilot_dir / "run_meta.json"
        if not pilot_meta_path.is_file():
            raise ValueError("pilot directory has no run_meta.json")
        pilot_meta = json.loads(pilot_meta_path.read_text(encoding="utf-8"))
        if (pilot_meta.get("status") != "complete" or pilot_meta.get("stage") != "pilot"
                or int(pilot_meta.get("batch", -1)) != args.batch):
            raise ValueError("pilot reuse requires a complete pilot with the selected fixed batch size")
        _import_pilot(pilot_dir, out_dir, expected,
                      repair_torn_last_line=bool(getattr(args, "resume", False)))
    repair_torn = bool(getattr(args, "resume", False))
    existing = _jsonl_rows(out_answers, "key", repair_torn_last_line=repair_torn)
    if getattr(args, "fixed_only", False) and len(existing) != len(expected):
        missing_answers = sorted(set(expected) - set(existing))
        raise ValueError(f"--fixed-only requires a complete answers.jsonl: "
                         f"{len(missing_answers)} rows missing; first={missing_answers[0] if missing_answers else 'unknown'}")
    pending = ([] if getattr(args, "fixed_only", False)
               else [item for item in rows if item["key"] not in existing])
    matched = choose_matched_units(pairs, by_id)
    started = time.perf_counter()
    total = len(pending)
    grouped = defaultdict(list)
    for item in pending:
        layer_set = tuple(sorted({int(push["layer"]) for push in item["pushes"]}))
        grouped[(item["question_index"], item["dose"], layer_set)].append(item)
    call_index = len(_jsonl_rows(out_dir / "generation_calls.jsonl",
                                 repair_torn_last_line=repair_torn))
    done = 0
    for group_key, tasks in grouped.items():
        question_index, dose, _layer_set = group_key
        for begin in range(0, len(tasks), args.batch):
            chunk = tasks[begin:begin + args.batch]
            result_rows, call = _generate_batch(
                model, tokenizer, chunk, by_id, random_meta, layers,
                cache_by_dose, args.batch, args.device)
            call_index += 1
            _call_record(out_dir / "generation_calls.jsonl", args.stage,
                         f"generation-{call_index:06d}", call)
            saved = []
            for row in result_rows:
                saved.append({"key": row["key"], "arm": row["arm"], "pair_id": row["pair_id"],
                              "family": row["family"], "question_index": row["question_index"],
                              "question": row["question"], "dose": row["dose"],
                              "unit_ids": [push["unit_id"] for push in row["pushes"]],
                              "pushes": row["pushes"], "ref_unit": row["ref_unit"],
                              "random_index": row["random_index"], "answer": row["answer"],
                              "tokens": row["tokens"], "token_count": row["token_count"],
                              "generation_top10": row["generation_top10"],
                              "generation_logits_captured": row["generation_logits_captured"]})
            _append_jsonl(out_answers, saved)
            done += len(saved)
            print(f"P127 {args.stage} generated {done}/{total} question={question_index} dose={dose} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    answers = _jsonl_rows(out_answers, "key", repair_torn_last_line=repair_torn)
    missing = sorted(set(expected) - set(answers))
    if missing:
        raise RuntimeError(f"incomplete answer capture: {len(missing)} rows missing; first={missing[0]}")
    for key, row in answers.items():
        if key in expected and row.get("key") != key:
            raise ValueError(f"answer key mismatch in {key}")
    fixed_path = out_dir / "fixed_path.jsonl"
    fixed_existing = _jsonl_rows(fixed_path, "fixed_key", repair_torn_last_line=repair_torn)
    fixed_counts_for_answers = fixed_path_counts(pairs, selected_ids, answers)
    expected_fixed_groups = (len(pairs) * len(QUESTIONS) * len(DOSES)
                             - fixed_counts_for_answers["dropped_group_count"])
    if getattr(args, "fresh_fixed", False):
        for stale_path in (fixed_path, out_dir / "fixed_path_meta.json",
                           out_dir / "fixed_shared.pt"):
            if stale_path.exists():
                superseded = stale_path.with_name(stale_path.name + ".superseded")
                suffix = 1
                while superseded.exists():
                    superseded = stale_path.with_name(stale_path.name + f".superseded.{suffix}")
                    suffix += 1
                os.replace(stale_path, superseded)
    if len(fixed_existing) < expected_fixed_groups or not (out_dir / "fixed_path_meta.json").exists():
        fixed_report = _run_fixed_scoring(
            out_dir, pairs, by_id, random_meta, cache_by_dose, tokenizer, model, layers,
            FIXED_PATH_BATCH, args.device, answers, out_dir / "generation_calls.jsonl",
            repair_torn_last_line=repair_torn)
    else:
        fixed_report = json.loads((out_dir / "fixed_path_meta.json").read_text(encoding="utf-8"))
    expected_counts = dict(Counter(row["arm"] for row in rows))
    pilot_import_path = out_dir / "pilot_import.json"
    imported_answer_count = (int(json.loads(pilot_import_path.read_text(encoding="utf-8"))
                                   .get("answer_rows_imported", 0))
                             if pilot_import_path.is_file() else 0)
    generated_by_run = len(answers) - imported_answer_count
    if generated_by_run < 0:
        raise ValueError("pilot-import answer count exceeds the completed answer table")
    run_meta = {"status": "complete", "stage": args.stage, "batch": args.batch,
                "device": args.device, "model_revision": "48d788eca847d4d7548f375ad03d3c9312f6139e",
                "questions": list(QUESTIONS), "pairs": [pair["pair_id"] for pair in pairs],
                "selected_units": selected_ids, "answer_count": len(rows),
                "arm_counts": expected_counts, "new_answers": generated_by_run,
                "answers_path": str(out_answers), "rig_gate_passed": bool(rig["passed"]),
                "fixed_path": fixed_report, "pilot_dir": args.pilot_dir,
                "generated_at_elapsed_seconds": round(time.perf_counter() - started, 3)}
    if args.stage == "main" and len(by_id) == 53 and expected_counts != EXPECTED_ARMS:
        raise AssertionError(f"main arm counts differ from design: {expected_counts}")
    _write_json(meta_path, run_meta)
    signature = hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest()[:16]
    print(f"P127 {args.stage} complete: answers={len(rows)} fixed_paths={fixed_report['counts']['total']}",
          flush=True)
    print(f"PART127_RUN_COMPLETE stage={args.stage} inventory={signature}", flush=True)
    return run_meta


def main(argv: list[str] | None = None, *, model_loader=None,
         inventory: dict[str, dict] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--model", required=True, help="local pinned OLMo 2 1B Instruct snapshot")
    parser.add_argument("--neurons", default=DEFAULT_NEURONS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pilot-dir", help="pilot directory whose registered rows should be reused")
    parser.add_argument("--resume", action="store_true",
                        help="resume this output as its owner; repair a torn final JSONL row if needed")
    parser.add_argument("--fixed-only", action="store_true",
                        help="score a complete existing answers.jsonl without generating answers")
    parser.add_argument("--fresh-fixed", action="store_true",
                        help="supersede prior fixed-path rows/meta and rebuild all fixed groups")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--test-inventory", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.pilot_dir and args.stage != "main":
        parser.error("--pilot-dir is only valid for the main stage")
    if args.fixed_only and args.stage != "main":
        parser.error("--fixed-only requires --stage main")
    if args.fresh_fixed and not args.fixed_only:
        parser.error("--fresh-fixed requires --fixed-only")
    try:
        run(args, model_loader=model_loader, inventory=inventory)
    except (ValueError, RuntimeError, OSError, AssertionError) as exc:
        print(f"run127: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
