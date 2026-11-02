#!/usr/bin/env python3
"""Host-side Part 173 scoring, entity tables, prediction verdicts, and prompt bootstrap."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86"), str(ROOT / "part171"), str(ROOT / "part172")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part165 import score165
from part166 import score166
import names173
import run171
import run173

DESIGN = run173.DESIGN
PROMPTS = DESIGN["prompt_count"]
REPS = DESIGN["bootstrap"]["replicates"]
SEED = DESIGN["bootstrap"]["seed"]
MODES = ("canonical", "generic", "raw")
QWEN_ALIASES = names173.ALIAS_GROUPS["qwen"]
CELL_SPECS = tuple({"set": name, "family": run173.FAMILY_BY_SET[name], "dose": float(dose),
                    "arm": run173.dose_arm(name, dose)}
                   for name in run173.SET_NAMES for dose in run173.DOSES)
CELL_INDEX = {(row["set"], row["dose"]): index for index, row in enumerate(CELL_SPECS)}
SET_INDEX = {name: index for index, name in enumerate(run173.SET_NAMES)}
FIT_BUILT_SETS = tuple(run173.SET_FAMILIES["rotated"] + run173.SET_FAMILIES["fit_at_random_coordinates"])
PICKED_AND_FIT_BUILT = tuple(run173.SET_FAMILIES["picked"] + FIT_BUILT_SETS)
WORD_HIT_GROUPS = names173.ALIAS_GROUPS
NO_ENTITY_CHANCE_CATEGORY = "__no_entity__"
CHANCE_DRAWS = DESIGN["chance_simulation"]["draws"]
CHANCE_SEED = DESIGN["chance_simulation"]["seed"]


def read_jsonl(path):
    return run171.read_jsonl(path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def prompt_index(prompt_id):
    match = re.search(r"new44b:(\d+)$", str(prompt_id))
    return int(match.group(1)) if match else None


def _first40(row):
    return str(row.get("first40", row.get("score_text", "")))


def score_answer(row):
    """Part 172's hit, Qwen, loop, broken and one-cell rule, with broken answers first."""
    first40 = _first40(row)
    answer = str(row.get("answer", row.get("reader_answer", first40)))
    hit = bool(run171.hit_match(first40, "google"))
    qwen = bool(run171._word_match(first40, QWEN_ALIASES))
    broken = bool(score165.primary_broken(answer))
    loop = bool(score166.loop_phrase(answer))
    if broken:
        cell = "broken"
    elif hit and qwen:
        cell = "both"
    elif hit:
        cell = "hit"
    elif qwen:
        cell = "qwen"
    else:
        cell = "neither"
    raw_hits = {entity: bool(run171._word_match(first40, aliases))
                for entity, aliases in WORD_HIT_GROUPS.items()}
    return {**row, "score_text": first40, "hit": hit, "qwen": qwen, "loop": loop, "broken": broken,
            "cell": cell, "clean": not broken, "raw_word_hits": raw_hits}


def score_rows(rows):
    return [score_answer(row) for row in rows if not str(row.get("arm", "")).startswith("replay:")]


def arm_stats(scored, arm):
    rows = [row for row in scored if row.get("arm") == arm]
    cells = {name: sum(row["cell"] == name for row in rows)
             for name in ("hit", "both", "qwen", "neither", "broken")}
    return {"arm": arm, "answers": len(rows), "cells": cells, "broken": cells["broken"],
            "clean": len(rows) - cells["broken"], "hit": cells["hit"], "qwen": cells["qwen"],
            "both": cells["both"], "clean_google_word_hit": cells["hit"] + cells["both"],
            "raw_word_hits": {name: sum(row["raw_word_hits"][name] for row in rows)
                              for name in WORD_HIT_GROUPS}}


def _reader_id_map(scored, reader_rows):
    import label173
    by_id = {str(row.get("id")): row for row in reader_rows if row.get("id") is not None}
    joined = {}
    for row in scored:
        pair = label173.pair_id(row.get("question", ""), row.get("answer", ""))
        joined[(str(row.get("arm")), str(row.get("prompt_id")))] = by_id.get(pair)
    return joined


def _reader_for(row, reader_by_row):
    return reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))


def _primary_entity(reader_row, mode="canonical"):
    if not reader_row or not reader_row.get("relation") == "claim":
        return None
    return names173.normalize_name(reader_row.get("primary"), mode=mode)


def _canonical_entity(reader_row):
    if not reader_row or reader_row.get("relation") != "claim":
        return None
    return names173.normalize_name(reader_row.get("primary"), mode="canonical")


def _distribution(values, known_by_entity=None):
    counts = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    ordered = {key: counts[key] for key in sorted(counts)}
    known, invented = {}, {}
    for entity, count in ordered.items():
        if known_by_entity is not None and known_by_entity.get(entity, False):
            known[entity] = count
        elif entity in names173.KNOWN_ENTITIES:
            known[entity] = count
        else:
            invented[entity] = count
    return ordered, known, invented


def _modal(counts):
    candidates = [(entity, int(count)) for entity, count in counts.items()
                  if entity is not None and entity != names173.QWEN_ENTITY]
    if not candidates:
        return None, 0, None, 0
    ordered = sorted(candidates, key=lambda item: (-item[1], item[0]))
    runner = ordered[1] if len(ordered) > 1 else (None, 0)
    return ordered[0][0], ordered[0][1], runner[0], runner[1]


def _half_modal(rows, reader_by_row, mode, parity):
    values = []
    for row in rows:
        index = prompt_index(row.get("prompt_id"))
        if index is None or index % 2 != parity:
            continue
        entity = _primary_entity(_reader_for(row, reader_by_row), mode=mode)
        if entity is not None and _canonical_entity(_reader_for(row, reader_by_row)) != names173.QWEN_ENTITY:
            values.append(entity)
    counts = {entity: values.count(entity) for entity in set(values)}
    if not counts:
        return None, 0
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def summarize_cell(rows, reader_by_row, mode="canonical"):
    clean = [row for row in rows if row["clean"]]
    claims, none = [], 0
    raw_names, primary_entities, mention_entities, non_qwen_primary_entities = [], [], [], []
    named = unnamed = mixed = qwen_primary = 0
    mention_only = 0
    known_by_entity = {}
    for row in clean:
        record = _reader_for(row, reader_by_row)
        if not record or not record.get("relation") in ("claim", "none"):
            continue
        if record["relation"] == "none":
            none += 1
        else:
            claims.append(row)
        names = list(record.get("names", []))
        entities_for_answer = []
        for name in names:
            resolved = names173.resolve_entity(name, mode=mode)
            entity = resolved["entity"]
            if entity is None:
                continue
            raw_names.append(entity)
            entities_for_answer.append(entity)
            known_by_entity[entity] = resolved["known"]
        mention_entities.extend(sorted(set(entities_for_answer)))
        entity = _primary_entity(record, mode=mode)
        canonical = _canonical_entity(record)
        if record["relation"] == "claim" and entity is not None:
            primary_entities.append(entity)
            known_by_entity[entity] = names173.is_known_entity(canonical)
            named += 1
            qwen_primary += int(canonical == names173.QWEN_ENTITY)
            if canonical != names173.QWEN_ENTITY:
                non_qwen_primary_entities.append(entity)
        elif record["relation"] == "claim" and len(names) > 1:
            mixed += 1
        elif record["relation"] == "claim":
            unnamed += 1
        else:
            unnamed += 1
            mention_only += int(bool(names))

    full, known_dist, invented_dist = _distribution(primary_entities, known_by_entity)
    mentions, known_mentions, invented_mentions = _distribution(mention_entities, known_by_entity)
    non_qwen_counts = {entity: non_qwen_primary_entities.count(entity)
                       for entity in set(non_qwen_primary_entities)}
    modal, modal_count, runner_up, runner_up_count = _modal(non_qwen_counts)
    invented_modal = bool(modal is not None and modal not in names173.KNOWN_ENTITIES)
    even_entity, even_count = _half_modal(rows, reader_by_row, mode, parity=0)
    odd_entity, odd_count = _half_modal(rows, reader_by_row, mode, parity=1)
    broken = sum(not row["clean"] for row in rows)
    eligible = (modal_count >= DESIGN["cell"]["modal_count_min"] and
                broken <= DESIGN["cell"]["broken_max"])
    stable = bool(eligible and modal is not None and even_entity == modal and odd_entity == modal and
                  even_count >= DESIGN["cell"]["stable_half_min"] and
                  odd_count >= DESIGN["cell"]["stable_half_min"])
    return {"answers": len(rows), "clean": len(clean), "broken": broken,
            "relation_claim": len(claims), "relation_none": none,
            "named": named, "unnamed": unnamed, "mixed": mixed, "qwen_primary": qwen_primary,
            "mention_only": mention_only, "full_name_distribution": full,
            "known_primary_distribution": known_dist, "invented_primary_distribution": invented_dist,
            "mention_distribution": mentions, "known_mention_distribution": known_mentions,
            "invented_mention_distribution": invented_mentions,
            "modal": modal, "modal_count": modal_count, "runner_up": runner_up,
            "runner_up_count": runner_up_count, "invented_modal": invented_modal,
            "invented_modal_count": modal_count if invented_modal else 0,
            "modal_count_over_clean": (modal_count / len(clean)) if clean else None,
            "eligible": bool(eligible), "stable": stable,
            "stability_halves": {"even_indexed": {"modal": even_entity, "count": even_count},
                                 "odd_indexed": {"modal": odd_entity, "count": odd_count}},
            "raw_word_hits": {name: sum(row["raw_word_hits"][name] for row in rows)
                              for name in WORD_HIT_GROUPS}}


def build_cell_tables(scored, reader_by_row):
    controls = {}
    for mode in MODES:
        controls[mode] = {
            "none": summarize_cell([row for row in scored if row.get("arm") == "none"],
                                   reader_by_row, mode),
            "zero4": summarize_cell([row for row in scored if row.get("arm") == "zero4"],
                                    reader_by_row, mode),
        }
    cells = []
    for spec in CELL_SPECS:
        rows = [row for row in scored if row.get("arm") == spec["arm"]]
        entry = {**spec, "sensitivity": {mode: summarize_cell(rows, reader_by_row, mode)
                                          for mode in MODES}}
        for mode in MODES:
            data = entry["sensitivity"][mode]
            none_distribution = controls[mode]["none"]["full_name_distribution"]
            zero4_modal = controls[mode]["zero4"]["modal"]
            baseline = int(none_distribution.get(data["modal"], 0)) if data["modal"] is not None else 0
            data["baseline"] = baseline
            data["modal_count_multiple_of_baseline"] = (
                data["modal_count"] / baseline if baseline else None)
            data["none_entity_distribution"] = dict(none_distribution)
            data["zero4_modal_entity"] = zero4_modal
            data["modal_matches_zero4"] = (
                bool(data["modal"] == zero4_modal) if data["eligible"] else None)
        entry.update(entry["sensitivity"]["canonical"])
        entry["generic_only_bar_passes_do_not_count"] = None
        cells.append(entry)

    for mode in MODES:
        table = cross_set_modal_table(cells, mode)
        counts = {entity: values["set_count"] for entity, values in table.items()}
        for cell in cells:
            data = cell["sensitivity"][mode]
            data["supported_sets_with_modal_entity"] = (
                counts.get(data["modal"], 0) if data["eligible"] and data["modal"] is not None else None)
            if mode == "canonical":
                cell.update(data)
    return cells


def cross_set_modal_table(cells, mode="canonical"):
    """Count distinct sets with each entity modal in at least one supported cell."""
    entity_sets = {}
    for cell in cells:
        data = cell.get("sensitivity", {}).get(mode, cell)
        if not data.get("eligible") or data.get("modal") is None:
            continue
        entity_sets.setdefault(data["modal"], set()).add(cell["set"])
    return {entity: {"modal_entity": entity, "set_count": len(set_names),
                     "sets": sorted(set_names)}
            for entity, set_names in sorted(entity_sets.items())}


def reader_entity_frequency_counts(rows, reader_by_row, mode="canonical", include_no_entity=False):
    """Count primary reader entities on clean answers using the registered resolver."""
    counts = {}
    for row in rows:
        if not row.get("clean"):
            continue
        entity = _primary_entity(_reader_for(row, reader_by_row), mode=mode)
        if entity is None:
            if include_no_entity:
                counts[NO_ENTITY_CHANCE_CATEGORY] = counts.get(NO_ENTITY_CHANCE_CATEGORY, 0) + 1
            continue
        counts[entity] = counts.get(entity, 0) + 1
    return {entity: int(counts[entity]) for entity in sorted(counts)}


def simulate_chance(entity_frequencies, draws=CHANCE_DRAWS, seed=CHANCE_SEED,
                    answer_count=None, expected_cells=None, threshold=None):
    """Vectorized multinomial chance concentration summary; descriptive only."""
    settings = DESIGN["chance_simulation"]
    answer_count = int(settings["answer_count"] if answer_count is None else answer_count)
    expected_cells = int(settings["expected_cells"] if expected_cells is None else expected_cells)
    threshold = int(settings["threshold"] if threshold is None else threshold)
    draws, seed = int(draws), int(seed)
    if draws < 1 or answer_count < 1 or threshold < 1 or expected_cells < 0:
        raise ValueError("chance simulation dimensions and threshold must be positive")
    frequencies = {str(entity): int(count) for entity, count in entity_frequencies.items()
                   if int(count) > 0}
    entities = sorted(frequencies)
    weights = np.asarray([frequencies[entity] for entity in entities], dtype=np.float64)
    started = time.perf_counter()
    if weights.size:
        probabilities = weights / weights.sum()
        samples = np.random.default_rng(seed).multinomial(answer_count, probabilities, size=draws)
        keep = [index for index, entity in enumerate(entities)
                if entity not in (names173.QWEN_ENTITY, NO_ENTITY_CHANCE_CATEGORY)]
        reaches = (samples[:, keep].max(axis=1) >= threshold) if keep else np.zeros(draws, dtype=bool)
        probability = float(np.mean(reaches))
    else:
        probability = 0.0
    elapsed = time.perf_counter() - started
    return {"seed": seed, "draws": draws, "answers_per_draw": answer_count,
            "threshold": threshold, "expected_cells": expected_cells,
            "entity_frequencies": frequencies,
            "probability_top_non_qwen_reaches_10": probability,
            "expected_cells_reaching_10": probability * expected_cells,
            "elapsed_seconds": float(elapsed)}


def chance_simulations(scored, reader_by_row):
    none_rows = [row for row in scored if row.get("arm") == "none"]
    # Fixed 2026-11-02 after the run, on a cold check: rows carry arm names like `u100d_d173_d0.5`, so comparing them
    # with set names left this pool empty. No verdict depends on this simulation.
    random_sets = set(run173.SET_FAMILIES["random_valued"])
    random_arms = {cell["arm"] for cell in CELL_SPECS if cell["set"] in random_sets}
    random_rows = [row for row in scored if row.get("arm") in random_arms]
    none_frequencies = reader_entity_frequency_counts(none_rows, reader_by_row,
                                                       include_no_entity=True)
    random_frequencies = reader_entity_frequency_counts(random_rows, reader_by_row,
                                                         include_no_entity=True)
    return {"reported_only": True, "verdict_dependency": False,
            "none_entity_frequencies": simulate_chance(none_frequencies),
            "random_valued_pooled_clean_entity_frequencies": simulate_chance(random_frequencies)}


def set_summary(cells, mode="canonical"):
    by_set = {}
    for cell in cells:
        by_set.setdefault(cell["set"], []).append(cell)
    result = {}
    for name in run173.SET_NAMES:
        candidates = [cell for cell in by_set.get(name, []) if cell["sensitivity"][mode]["eligible"]]
        best = sorted(candidates, key=lambda cell: (-cell["sensitivity"][mode]["modal_count"],
                                                     cell["sensitivity"][mode]["broken"],
                                                     abs(float(cell["dose"])),
                                                     0 if float(cell["dose"]) > 0 else 1))
        selected = best[0] if best else None
        result[name] = {"set": name, "family": run173.FAMILY_BY_SET[name],
                        "eligible_cells": len(candidates),
                        "best_cell": ({"arm": selected["arm"], "dose": selected["dose"],
                                       "modal": selected["sensitivity"][mode]["modal"],
                                       "modal_count": selected["sensitivity"][mode]["modal_count"],
                                       "broken": selected["sensitivity"][mode]["broken"]}
                                      if selected else None)}
    return result


def family_table(cells, summaries=None, mode="canonical"):
    summaries = summaries or set_summary(cells, mode)
    table = {}
    for family, sets in run173.SET_FAMILIES.items():
        supported = [name for name in sets if summaries[name]["eligible_cells"] > 0]
        google_supported = [name for name in sets if any(
            cell["set"] == name and cell["sensitivity"][mode]["eligible"] and
            cell["sensitivity"][mode]["modal"] == "google" for cell in cells)]
        table[family] = {"sets_total": len(sets), "sets_with_supported_replacement": len(supported),
                         "sets_with_supported_google_replacement": len(google_supported),
                         "supported_sets": supported, "supported_google_sets": google_supported,
                         "specificity_informative": not (family == "random_valued" and not supported),
                         "specificity_status": ("uninformative: no supported cell in this family"
                                                if family == "random_valued" and not supported else "read")}
    return table


# ---------------------------------------------------------------- Vectorized prompt bootstrap

def _count_top(code_row, draws):
    """Vectorized per-replicate bincount for one cell; ties follow sorted entity-code order."""
    valid = np.asarray(code_row, dtype=np.int32) >= 0
    if not valid.any():
        reps = int(draws.shape[0])
        return np.full(reps, -1, dtype=np.int32), np.zeros(reps, dtype=np.int16)
    entities, local_ids = np.unique(np.asarray(code_row, dtype=np.int32)[valid], return_inverse=True)
    local = np.full(PROMPTS, -1, dtype=np.int16)
    local[valid] = local_ids.astype(np.int16)
    sampled = local[draws]
    n_entities = len(entities)
    rep_offsets = np.arange(draws.shape[0], dtype=np.int64)[:, None] * n_entities
    valid_sample = sampled >= 0
    flattened = (rep_offsets + sampled)[valid_sample]
    counts = np.bincount(flattened, minlength=draws.shape[0] * n_entities).reshape(draws.shape[0], n_entities)
    winners = np.argmax(counts, axis=1)
    return entities[winners].astype(np.int32), counts[np.arange(draws.shape[0]), winners].astype(np.int16)


def bootstrap_entity_grid(entity_codes, broken, qwen_flags=None, reps=REPS, seed=SEED):
    """Bootstrap cell name codes over whole prompts; no loop runs once per replicate."""
    entity_codes = np.asarray(entity_codes, dtype=np.int32)
    broken = np.asarray(broken, dtype=np.int8)
    if entity_codes.ndim != 2 or entity_codes.shape[1] != PROMPTS or broken.shape != entity_codes.shape:
        raise ValueError(f"expected equally shaped cell-by-prompt matrices with {PROMPTS} prompts")
    qwen_flags = np.zeros_like(entity_codes, dtype=np.int8) if qwen_flags is None else np.asarray(qwen_flags, dtype=np.int8)
    if qwen_flags.shape != entity_codes.shape:
        raise ValueError("Qwen code matrix shape differs from entity code matrix")
    rng = np.random.default_rng(int(seed))
    random_draws = rng.integers(0, PROMPTS, size=(int(reps), PROMPTS), dtype=np.int16)
    draws = np.vstack((np.arange(PROMPTS, dtype=np.int16)[None, :], random_draws))
    nrows, ncells = draws.shape[0], entity_codes.shape[0]
    modal_ids = np.full((nrows, ncells), -1, dtype=np.int32)
    modal_counts = np.zeros((nrows, ncells), dtype=np.int16)
    broken_counts = np.zeros((nrows, ncells), dtype=np.int16)
    stable = np.zeros((nrows, ncells), dtype=bool)
    qwen_counts = np.zeros((nrows, ncells), dtype=np.int16)
    google_counts = np.zeros((nrows, ncells), dtype=np.int16)
    tencent_counts = np.zeros((nrows, ncells), dtype=np.int16)
    indices = np.arange(PROMPTS)
    even = indices % 2 == 0
    odd = ~even
    for cell_index in range(ncells):
        codes = entity_codes[cell_index]
        top_ids, top_counts = _count_top(codes, draws)
        even_codes = np.where(even, codes, -1)
        odd_codes = np.where(odd, codes, -1)
        even_ids, even_counts = _count_top(even_codes, draws)
        odd_ids, odd_counts = _count_top(odd_codes, draws)
        modal_ids[:, cell_index] = top_ids
        modal_counts[:, cell_index] = top_counts
        broken_counts[:, cell_index] = broken[cell_index][draws].sum(axis=1)
        qwen_counts[:, cell_index] = qwen_flags[cell_index][draws].sum(axis=1)
        # Entity ID 0 is not assumed to be a particular company. Specific counts are patched by
        # bootstrap_mode below from its mode-specific target ids.
        stable[:, cell_index] = ((top_ids >= 0) &
                                 (top_counts >= DESIGN["cell"]["modal_count_min"]) &
                                 (broken_counts[:, cell_index] <= DESIGN["cell"]["broken_max"]) &
                                 (even_ids == top_ids) & (odd_ids == top_ids) &
                                 (even_counts >= DESIGN["cell"]["stable_half_min"]) &
                                 (odd_counts >= DESIGN["cell"]["stable_half_min"]))
    eligible = ((modal_counts >= DESIGN["cell"]["modal_count_min"]) &
                (broken_counts <= DESIGN["cell"]["broken_max"]) & (modal_ids >= 0))
    return {"draws": draws, "modal_ids": modal_ids, "modal_counts": modal_counts,
            "broken_counts": broken_counts, "eligible": eligible, "stable": stable,
            "qwen_counts": qwen_counts, "google_counts": google_counts, "tencent_counts": tencent_counts}


def _code_matrices(scored, reader_by_row, mode):
    code_map = np.full((len(CELL_SPECS), PROMPTS), -1, dtype=np.int32)
    broken = np.zeros((len(CELL_SPECS), PROMPTS), dtype=np.int8)
    qwen = np.zeros_like(broken)
    entity_names = set()
    rows_by_cell_prompt = {}
    for row in scored:
        cell_index = CELL_INDEX.get((str(row.get("set")), float(row.get("dose", float("nan")))))
        if cell_index is None:
            cell_index = next((i for i, spec in enumerate(CELL_SPECS) if spec["arm"] == row.get("arm")), None)
        prompt = prompt_index(row.get("prompt_id"))
        if cell_index is None or prompt is None or not 0 <= prompt < PROMPTS:
            continue
        rows_by_cell_prompt[(cell_index, prompt)] = row
        broken[cell_index, prompt] = int(bool(row["broken"]))
        record = _reader_for(row, reader_by_row)
        canonical = _canonical_entity(record)
        qwen[cell_index, prompt] = int(canonical == names173.QWEN_ENTITY)
        entity = _primary_entity(record, mode=mode)
        if record and record.get("relation") == "claim" and entity is not None and canonical != names173.QWEN_ENTITY:
            entity_names.add(entity)
    ordered_names = sorted(entity_names)
    name_to_id = {name: index for index, name in enumerate(ordered_names)}
    for (cell_index, prompt), row in rows_by_cell_prompt.items():
        record = _reader_for(row, reader_by_row)
        entity = _primary_entity(record, mode=mode)
        if record and record.get("relation") == "claim" and entity in name_to_id and \
                _canonical_entity(record) != names173.QWEN_ENTITY:
            code_map[cell_index, prompt] = name_to_id[entity]
    return code_map, broken, qwen, ordered_names, name_to_id


def _specific_count(codes, draws, entity_id):
    if entity_id is None:
        return np.zeros(draws.shape[0], dtype=np.int16)
    mask = np.asarray(codes) == int(entity_id)
    return mask[draws].sum(axis=1).astype(np.int16)


def bootstrap_mode(scored, reader_by_row, mode="canonical", reps=REPS, seed=SEED):
    codes, broken, qwen, entity_names, name_to_id = _code_matrices(scored, reader_by_row, mode)
    grid = bootstrap_entity_grid(codes, broken, qwen, reps=reps, seed=seed)
    draws = grid["draws"]
    modal_ids = grid["modal_ids"]
    # The design counts an invented name as a supported replacement; it is reported apart, never excluded.
    grid["stable"] &= grid["eligible"]
    for cell_index in range(len(CELL_SPECS)):
        grid["google_counts"][:, cell_index] = _specific_count(codes[cell_index], draws, name_to_id.get("google"))
        grid["tencent_counts"][:, cell_index] = _specific_count(codes[cell_index], draws, name_to_id.get("tencent"))
    estimates = derive_bootstrap_estimates(grid, entity_names, mode)
    return {"grid": grid, "estimates": estimates, "entity_names": entity_names, "name_to_id": name_to_id}


def _set_cells(name):
    return [index for index, cell in enumerate(CELL_SPECS) if cell["set"] == name]


def derive_bootstrap_estimates(grid, entity_names, mode="canonical"):
    modal_ids = grid["modal_ids"]
    modal_counts = grid["modal_counts"]
    broken_counts = grid["broken_counts"]
    eligible = grid["eligible"]
    stable = grid["stable"]
    google_id = entity_names.index("google") if "google" in entity_names else -1
    nrows = modal_ids.shape[0]
    set_supported = np.zeros((nrows, len(run173.SET_NAMES)), dtype=bool)
    set_google = np.zeros_like(set_supported)
    best_entities = np.full_like(set_supported, -1, dtype=np.int32)
    max_google_by_set = np.zeros((nrows, len(run173.SET_NAMES)), dtype=np.int16)
    for set_pos, name in enumerate(run173.SET_NAMES):
        cell_indices = _set_cells(name)
        set_supported[:, set_pos] = eligible[:, cell_indices].any(axis=1)
        set_google[:, set_pos] = (eligible[:, cell_indices] & (modal_ids[:, cell_indices] == google_id)).any(axis=1)
        # Best cell is reselected in every replicate: modal count, broken, |dose|, then positive sign.
        tie_order = sorted(cell_indices, key=lambda idx: (abs(CELL_SPECS[idx]["dose"]),
                                                           0 if CELL_SPECS[idx]["dose"] > 0 else 1))
        counts = np.where(eligible[:, cell_indices], modal_counts[:, cell_indices], -1)
        maximum = counts.max(axis=1)
        possible = (counts == maximum[:, None]) & (maximum[:, None] >= 0)
        damaged = np.where(possible, broken_counts[:, cell_indices], np.iinfo(np.int16).max)
        min_damage = damaged.min(axis=1)
        possible &= damaged == min_damage[:, None]
        reorder = [cell_indices.index(idx) for idx in tie_order]
        first = np.argmax(possible[:, reorder], axis=1)
        chosen = np.asarray(tie_order, dtype=np.int32)[first]
        has = maximum >= 0
        best_entities[has, set_pos] = modal_ids[np.flatnonzero(has), chosen[has]]

        damage_eligible = broken_counts[:, cell_indices] <= DESIGN["cell"]["broken_max"]
        google_candidates = np.where(damage_eligible, grid["google_counts"][:, cell_indices], 0)
        max_google_by_set[:, set_pos] = google_candidates.max(axis=1)

    sorted_best = np.sort(np.where(best_entities >= 0, best_entities, np.iinfo(np.int32).max), axis=1)
    present = sorted_best != np.iinfo(np.int32).max
    distinct_best = (present[:, 0].astype(int) +
                     np.sum((sorted_best[:, 1:] != sorted_best[:, :-1]) & present[:, 1:], axis=1))
    best_count = np.sum(best_entities >= 0, axis=1)
    google_best_count = np.sum(best_entities == google_id, axis=1) if google_id >= 0 else np.zeros(nrows, dtype=int)
    google_best_share = np.divide(google_best_count, best_count, out=np.full(nrows, np.nan), where=best_count > 0)

    fit_indices = [SET_INDEX[name] for name in FIT_BUILT_SETS]
    all13 = [SET_INDEX[name] for name in ("n1", *FIT_BUILT_SETS)]
    n1_value = max_google_by_set[:, SET_INDEX["n1"]]
    fit_median = np.median(max_google_by_set[:, fit_indices], axis=1)
    all13_max = max_google_by_set[:, all13].max(axis=1)

    qualifying_pairs = np.zeros(nrows, dtype=np.int16)
    different_pairs = np.zeros(nrows, dtype=np.int16)
    for name in run173.SET_NAMES:
        for magnitude in DESIGN["predictions"]["173.7"]["absolute_doses"]:
            positive = CELL_INDEX[(name, float(magnitude))]
            negative = CELL_INDEX[(name, -float(magnitude))]
            pair_ok = eligible[:, positive] & eligible[:, negative]
            qualifying_pairs += pair_ok.astype(np.int16)
            different_pairs += (pair_ok & (modal_ids[:, positive] != modal_ids[:, negative])).astype(np.int16)
    different_share = np.divide(different_pairs, qualifying_pairs, out=np.full(nrows, np.nan),
                                where=qualifying_pairs > 0)

    random_set_positions = [SET_INDEX[name] for name in run173.SET_FAMILIES["random_valued"]]
    fit_picked_positions = [SET_INDEX[name] for name in PICKED_AND_FIT_BUILT]
    random_supported = set_supported[:, random_set_positions].sum(axis=1)
    random_google = set_google[:, random_set_positions].sum(axis=1)
    fit_picked_google = set_google[:, fit_picked_positions].sum(axis=1)
    supported_cells = eligible.sum(axis=1)
    stable_cells = (stable & eligible).sum(axis=1)
    stable_share = np.divide(stable_cells, supported_cells, out=np.full(nrows, np.nan),
                             where=supported_cells > 0)

    fit_built_positions = [SET_INDEX[name] for name in FIT_BUILT_SETS]
    output = {
        "173.1": {"supported_sets": set_supported[:, fit_built_positions].sum(axis=1)},
        "173.2": {"supported_sets": set_supported[:, random_set_positions].sum(axis=1)},
        "173.3": {"best_entity_count": distinct_best, "google_best_share": google_best_share,
                  "best_set_count": best_count},
        "173.4": {},
        "173.5": {},
        "173.6": {"n1_google_claim_max": n1_value, "fit_google_claim_median": fit_median,
                  "maximum_among_n1_and_12_fit_controls": all13_max},
        "173.7": {"qualifying_pairs": qualifying_pairs, "different_pairs": different_pairs,
                  "different_share": different_share},
        "173.8": {"random_supported_sets": random_supported, "random_google_sets": random_google,
                  "fit_picked_google_sets": fit_picked_google},
        "173.9": {"supported_cells": supported_cells, "stable_cells": stable_cells,
                  "stable_share": stable_share},
    }
    for set_name, dose in DESIGN["predictions"]["173.4"]["cells"]:
        idx = CELL_INDEX[(set_name, float(dose))]
        output["173.4"][f"{set_name}@{dose:g}"] = {
            "modal_count": modal_counts[:, idx], "broken": broken_counts[:, idx],
            "is_google": modal_ids[:, idx] == google_id,
            "passes": ((modal_ids[:, idx] == google_id) &
                       (modal_counts[:, idx] >= DESIGN["predictions"]["173.4"]["modal_count_min"]) &
                       (broken_counts[:, idx] <= DESIGN["predictions"]["173.4"]["broken_max"]))}
    set_name, dose = DESIGN["predictions"]["173.5"]["cell"]
    idx = CELL_INDEX[(set_name, float(dose))]
    tencent_id = entity_names.index("tencent") if "tencent" in entity_names else -1
    output["173.5"] = {"modal_count": modal_counts[:, idx], "broken": broken_counts[:, idx],
                       "is_tencent": modal_ids[:, idx] == tencent_id,
                       "passes": ((modal_ids[:, idx] == tencent_id) &
                                  (modal_counts[:, idx] >= DESIGN["predictions"]["173.5"]["modal_count_min"]) &
                                  (broken_counts[:, idx] <= DESIGN["predictions"]["173.5"]["broken_max"]))}
    return output


def _interval(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return [None, None]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _point(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _prediction_interval_map(estimate_arrays):
    return {name: _interval(values[1:]) for name, values in estimate_arrays.items()
            if isinstance(values, np.ndarray) and values.ndim == 1}


# ---------------------------------------------------------------- Prediction verdicts: named rules, kept separate from table calculation.

def verdict_173_1(supported_fit_sets, reader_read=True):
    if not reader_read:
        return "unread"
    return "held" if supported_fit_sets >= DESIGN["predictions"]["173.1"]["minimum_sets"] else "failed"


def verdict_173_2(supported_random_sets, reader_read=True):
    if not reader_read:
        return "unread"
    return "held" if supported_random_sets >= DESIGN["predictions"]["173.2"]["minimum_sets"] else "failed"


def verdict_173_3(best_entity_count, google_best_share, best_set_count, reader_read=True):
    if not reader_read:
        return "unread"
    if best_set_count <= 0 or google_best_share is None:
        return "not read"
    bar = DESIGN["predictions"]["173.3"]
    return "held" if best_entity_count >= bar["minimum_best_entities"] and \
        google_best_share <= bar["google_best_share_max"] else "failed"


def verdict_173_4(cells, reader_read=True):
    if not reader_read:
        return "unread"
    bar = DESIGN["predictions"]["173.4"]
    if len(cells) != 2:
        return "not read"
    return "held" if all(row.get("modal") == bar["entity"] and
                         row.get("modal_count", 0) >= bar["modal_count_min"] and
                         row.get("broken", PROMPTS + 1) <= bar["broken_max"] for row in cells) else "failed"


def verdict_173_5(cell, reader_read=True):
    if not reader_read:
        return "unread"
    bar = DESIGN["predictions"]["173.5"]
    return "held" if cell.get("modal") == bar["entity"] and \
        cell.get("modal_count", 0) >= bar["modal_count_min"] and \
        cell.get("broken", PROMPTS + 1) <= bar["broken_max"] else "failed"


def verdict_173_6(n1_google_max, fit_median, max_among_13, reader_read=True):
    if not reader_read:
        return "unread"
    if any(value is None for value in (n1_google_max, fit_median, max_among_13)):
        return "not read"
    bar = DESIGN["predictions"]["173.6"]
    return "held" if n1_google_max - fit_median >= bar["minimum_above_fit_median"] and \
        n1_google_max < max_among_13 else "failed"


def verdict_173_7(pair_count, different_share, reader_read=True):
    if not reader_read:
        return "unread"
    bar = DESIGN["predictions"]["173.7"]
    if pair_count < bar["minimum_pairs"] or different_share is None:
        return "not read"
    return "held" if different_share >= bar["different_entity_share_min"] else "failed"


def verdict_173_8(random_google_sets, fit_picked_google_sets, random_supported_sets, reader_read=True):
    if not reader_read:
        return "unread"
    if random_supported_sets <= 0:
        return "not read"
    bar = DESIGN["predictions"]["173.8"]
    return "held" if random_google_sets <= bar["random_google_sets_max"] and \
        fit_picked_google_sets >= bar["fit_and_picked_google_sets_min"] else "failed"


def verdict_173_9(supported_cells, stable_share, reader_read=True):
    if not reader_read:
        return "unread"
    bar = DESIGN["predictions"]["173.9"]
    if supported_cells < bar["minimum_supported_cells"] or stable_share is None:
        return "not read"
    return "held" if stable_share >= bar["stable_supported_share_min"] else "failed"


def verdict_173_w1(none_entity_distribution, reader_read=True):
    if not reader_read:
        return "unread"
    minimum = DESIGN["weaver_predictions"]["173.W1"]["minimum_known_non_qwen_count"]
    return "held" if any(entity in names173.KNOWN_ENTITIES and entity != names173.QWEN_ENTITY and
                         int(count) >= minimum
                         for entity, count in none_entity_distribution.items()) else "failed"


def verdict_173_w2(random_supported_sets, reader_read=True):
    if not reader_read:
        return "unread"
    maximum = DESIGN["weaver_predictions"]["173.W2"]["maximum_random_valued_sets"]
    return "held" if int(random_supported_sets) <= maximum else "failed"


def _under_two_times_baseline(cell):
    baseline = int(cell.get("baseline", 0) or 0)
    return baseline > 0 and int(cell.get("modal_count", 0)) < 2 * baseline


def _at_least_three_times_baseline(cell):
    baseline = int(cell.get("baseline", 0) or 0)
    return baseline == 0 or int(cell.get("modal_count", 0)) >= 3 * baseline


def verdict_173_w3(w1_verdict, supported_cells, reader_read=True):
    if not reader_read:
        return "unread"
    supported_cells = list(supported_cells)
    if w1_verdict != "held" or not supported_cells:
        return "not testable"
    minimum = DESIGN["weaver_predictions"]["173.W3"]["minimum_under_two_times_cells"]
    if sum(_under_two_times_baseline(cell) for cell in supported_cells) >= minimum:
        return "held"
    if all(_at_least_three_times_baseline(cell) for cell in supported_cells):
        return "failed"
    return "inconclusive"


def weaver_prediction_table(none_entity_distribution, set_summaries, cells, reader_read=True):
    known_non_qwen = {entity: int(count) for entity, count in none_entity_distribution.items()
                      if entity in names173.KNOWN_ENTITIES and entity != names173.QWEN_ENTITY}
    known_non_qwen = dict(sorted(known_non_qwen.items()))
    qualifying = {entity: count for entity, count in known_non_qwen.items()
                  if count >= DESIGN["weaver_predictions"]["173.W1"]["minimum_known_non_qwen_count"]}
    w1 = verdict_173_w1(none_entity_distribution, reader_read)

    random_sets = list(run173.SET_FAMILIES["random_valued"])
    supported_random = [name for name in random_sets if set_summaries.get(name, {}).get("eligible_cells", 0) > 0]
    w2 = verdict_173_w2(len(supported_random), reader_read)

    supported = [cell["sensitivity"]["canonical"] for cell in cells
                 if cell["sensitivity"]["canonical"].get("eligible")]
    under_two = sum(_under_two_times_baseline(cell) for cell in supported)
    at_least_three = sum(_at_least_three_times_baseline(cell) for cell in supported)
    w3 = verdict_173_w3(w1, supported, reader_read)
    supported_details = [{"set": cell["set"], "dose": cell["dose"], "modal": summary["modal"],
                          "modal_count": summary["modal_count"], "baseline": summary["baseline"],
                          "modal_count_multiple_of_baseline": summary["modal_count_multiple_of_baseline"],
                          "under_2_times_baseline": _under_two_times_baseline(summary),
                          "at_least_3_times_baseline": _at_least_three_times_baseline(summary)}
                         for cell in cells
                         if (summary := cell["sensitivity"]["canonical"]).get("eligible")]
    return {
        "173.W1": {"prediction": "`none` already names at least one known non-Qwen entity on at least 3 of 44",
                   "falsifier": "No known non-Qwen entity reaches 3 of 44 in `none`",
                   "estimate": {"known_non_qwen_entity_distribution": known_non_qwen,
                                "qualifying_entities": qualifying},
                   "verdict": w1, "read_status": "read" if reader_read else "unread"},
        "173.W2": {"prediction": "At most 1 of the 6 random-valued sets has a supported modal replacement",
                   "falsifier": "At least 2 of 6 do, which makes 173.2 right and value-randomness irrelevant to whether a lead forms",
                   "estimate": {"supported_sets": len(supported_random),
                                "supported_set_names": supported_random},
                   "verdict": w2, "read_status": "read" if reader_read else "unread"},
        "173.W3": {"prediction": "If 173.W1 holds, at least 3 supported cells have modal_count under 2 times the `none` baseline",
                   "falsifier": "Every supported cell is at 3 times baseline or more",
                   "estimate": {"supported_cells": len(supported), "under_2_times_baseline": int(under_two),
                                "at_least_3_times_baseline": int(at_least_three),
                                "cells": supported_details},
                   "verdict": w3, "read_status": "read" if reader_read else "unread"},
    }


def _derive_prediction_table(cells, summaries, estimates, estimate_arrays, reader_read, mode):
    lookup = {(cell["set"], float(cell["dose"])): cell["sensitivity"][mode] for cell in cells}
    point = lambda key, field: _point(estimates[key][field][0])
    interval = lambda key: _prediction_interval_map(estimates[key])
    result = {}

    fit_count = int(point("173.1", "supported_sets"))
    result["173.1"] = {"prediction": "At least 5 of 12 fit-built control sets have a supported replacement",
        "bar": ">= 5 of 12", "estimate": {"supported_sets": fit_count},
        "interval_95": interval("173.1"), "verdict": verdict_173_1(fit_count, reader_read),
        "read_status": "read" if reader_read else "unread"}

    random_count = int(point("173.2", "supported_sets"))
    result["173.2"] = {"prediction": "At least 2 of 6 random-valued sets have a supported replacement",
        "bar": ">= 2 of 6", "estimate": {"supported_sets": random_count},
        "interval_95": interval("173.2"), "verdict": verdict_173_2(random_count, reader_read),
        "read_status": "read" if reader_read else "unread"}

    distinct = int(point("173.3", "best_entity_count"))
    share = point("173.3", "google_best_share")
    best_count = int(point("173.3", "best_set_count"))
    result["173.3"] = {"prediction": "Best cells span at least 3 entities; Google is modal on at most half",
        "bar": "entities >= 3 and Google share <= 0.50",
        "estimate": {"distinct_best_entities": distinct, "google_best_share": share,
                     "sets_with_best_cell": best_count}, "interval_95": interval("173.3"),
        "verdict": verdict_173_3(distinct, share, best_count, reader_read),
        "read_status": "read" if reader_read and best_count else "not read" if reader_read else "unread",
        "best_cells": {name: row["best_cell"] for name, row in summaries.items()}}

    bar4 = DESIGN["predictions"]["173.4"]
    cells4 = []
    for set_name, dose in bar4["cells"]:
        cell = lookup[(set_name, float(dose))]
        cells4.append({"set": set_name, "dose": dose, "modal": cell["modal"],
                       "modal_count": cell["modal_count"], "broken": cell["broken"]})
    intervals4 = {}
    for set_name, dose in bar4["cells"]:
        cell_key = f"{set_name}@{dose:g}"
        sample = estimate_arrays["173.4"][cell_key]
        intervals4[f"{cell_key}.modal_count"] = _interval(sample["modal_count"][1:])
        intervals4[f"{cell_key}.broken"] = _interval(sample["broken"][1:])
    result["173.4"] = {"prediction": "r100_d2171 at 0.75 and n2 at 0.5 each have a supported Google modal",
        "bar": "each modal count >= 30 of 44 and broken <= 4", "estimate": {"cells": cells4},
        "interval_95": intervals4,
        "verdict": verdict_173_4(cells4, reader_read),
        "read_status": "read" if reader_read else "unread",
        "raw_word_hit_fallback": {f"{row['set']}@{row['dose']:g}": row["raw_word_hits"]
                                   for row in cells if (row["set"], row["dose"]) in bar4["cells"]}}

    set5, dose5 = DESIGN["predictions"]["173.5"]["cell"]
    cell5 = lookup[(set5, float(dose5))]
    result["173.5"] = {"prediction": "n1 at -0.5 has Tencent as modal",
        "bar": "modal count >= 10 of 44 and broken <= 8",
        "estimate": {"set": set5, "dose": dose5, "modal": cell5["modal"],
                     "modal_count": cell5["modal_count"], "broken": cell5["broken"]},
        "interval_95": {"modal_count": _interval(estimates["173.5"]["modal_count"][1:]),
                        "broken": _interval(estimates["173.5"]["broken"][1:])},
        "verdict": verdict_173_5(cell5, reader_read),
        "read_status": "read" if reader_read else "unread",
        "raw_word_hit_fallback": cell5["raw_word_hits"]}

    n1_value = point("173.6", "n1_google_claim_max")
    fit_median = point("173.6", "fit_google_claim_median")
    max13 = point("173.6", "maximum_among_n1_and_12_fit_controls")
    result["173.6"] = {"prediction": "n1 Google claim capacity clears the fit-control median by 5 and is not highest",
        "bar": "n1 max >= median + 5 and n1 max < maximum of the 13 sets",
        "estimate": {"n1_google_claim_max": n1_value, "fit_control_median": fit_median,
                     "maximum_among_13": max13}, "interval_95": interval("173.6"),
        "verdict": verdict_173_6(n1_value, fit_median, max13, reader_read),
        "read_status": "read" if reader_read else "unread",
        "per_set_maxima": per_set_maxima(cells, mode)}

    pair_count = int(point("173.7", "qualifying_pairs"))
    different = int(point("173.7", "different_pairs"))
    different_share = point("173.7", "different_share")
    result["173.7"] = {"prediction": "At least 75% of supported paired positive and negative cells differ",
        "bar": "different entity share >= 0.75; at least 4 set-and-size pairs",
        "estimate": {"qualifying_pairs": pair_count, "different_pairs": different,
                     "different_entity_share": different_share}, "interval_95": interval("173.7"),
        "verdict": verdict_173_7(pair_count, different_share, reader_read),
        "read_status": "unread" if not reader_read else "not read" if pair_count < 4 else "read"}

    random_supported = int(point("173.8", "random_supported_sets"))
    random_google = int(point("173.8", "random_google_sets"))
    fit_google = int(point("173.8", "fit_picked_google_sets"))
    result["173.8"] = {"prediction": "Google has supported cells in at most 1 random set and at least 4 fit-built/picked sets",
        "bar": "random-valued <= 1; fit-built plus picked >= 4",
        "estimate": {"random_supported_sets": random_supported, "random_google_sets": random_google,
                     "fit_and_picked_google_sets": fit_google}, "interval_95": interval("173.8"),
        "verdict": verdict_173_8(random_google, fit_google, random_supported, reader_read),
        "read_status": "unread" if not reader_read else "not read" if random_supported == 0 else "read",
        "specificity_status": ("uninformative: no random-valued set has a supported cell"
                               if random_supported == 0 else "read")}

    supported = int(point("173.9", "supported_cells"))
    stable_count = int(point("173.9", "stable_cells"))
    stable_share = point("173.9", "stable_share")
    result["173.9"] = {"prediction": "At least 60% of supported cells are stable across prompt parity halves",
        "bar": "stable supported share >= 0.60; at least 5 supported cells",
        "estimate": {"supported_cells": supported, "stable_cells": stable_count,
                     "stable_share": stable_share}, "interval_95": interval("173.9"),
        "verdict": verdict_173_9(supported, stable_share, reader_read),
        "read_status": "unread" if not reader_read else "not read" if supported < 5 else "read"}

    for prediction in result.values():
        prediction["generic_only_bar_passes_do_not_count"] = False
    return result


def _max_google_claim(cell, mode):
    data = cell["sensitivity"][mode]
    if data["broken"] > DESIGN["cell"]["broken_max"]:
        return 0
    return int(data.get("google_claim_count", 0))


def per_set_maxima(cells, mode="canonical"):
    maxima = {}
    for cell in cells:
        name = str(cell["set"])
        maxima[name] = max(maxima.get(name, 0), _max_google_claim(cell, mode))
    return maxima


def assert_bootstrap_point_entity_count(estimate_arrays, summaries):
    plain = len({row["best_cell"]["modal"] for row in summaries.values()
                 if row.get("best_cell") and row["best_cell"].get("modal") is not None})
    boot = int(estimate_arrays["173.3"]["best_entity_count"][0])
    assert boot == plain, f"bootstrap point has {boot} best entities; plain set count has {plain}"
    return plain


def _reader_gate(out_dir, items, first_rows, second_rows):
    import label173
    directory = Path(out_dir)
    sample = read_json(directory / "second_reader.json", {}) or {}
    known = {reader: read_json(directory / f"known-{reader}.json")
             for reader in ("ds", "haiku")}
    missing_known = [reader for reader, data in known.items() if not isinstance(data, dict)]
    known_failed = [reader for reader, data in known.items()
                    if isinstance(data, dict) and not data.get("passed")]
    sample_problem = None
    first_path = directory / "labels-ds.jsonl"
    if first_path.exists() and sample:
        expected_sha = hashlib.sha256(first_path.read_bytes()).hexdigest()
        if sample.get("first_reader_sha256") != expected_sha:
            sample_problem = "first-reader file changed after the sample was frozen"
    if not sample or missing_known or known_failed or sample_problem:
        status = "unread" if missing_known or not sample or sample_problem else "failed"
        return {"passed": False, "status": status,
                "known_checks": {reader: (None if value is None else {
                    "relation_correct": value.get("relation_correct"),
                    "primary_correct": value.get("primary_correct"), "passed": value.get("passed")})
                                  for reader, value in known.items()},
                "missing_known": missing_known, "failed_known": known_failed,
                "sample_problem": sample_problem, "reason": "known checks or second-reader sample missing"}
    agreement = label173.gate9(first_rows, second_rows, sample, items)
    agreement["known_checks"] = {reader: {"relation_correct": known[reader].get("relation_correct"),
                                        "primary_correct": known[reader].get("primary_correct"),
                                        "passed": known[reader].get("passed")}
                                 for reader in known}
    if known_failed:
        agreement["passed"] = False
        agreement["status"] = "failed"
    return agreement


def _reader_disagreements(first_rows, second_rows, sample, items):
    import label173
    first = {str(row.get("id")): row for row in first_rows if label173.valid(row)}
    second = {str(row.get("id")): row for row in second_rows if label173.valid(row)}
    item_counts = {str(item["id"]): int(item.get("answer_count", 1)) for item in items}
    out = []
    for item_id in sorted(set(sample.get("ids", []))):
        if item_id not in first or item_id not in second:
            continue
        f, s = first[item_id], second[item_id]
        fp = names173.normalize_name(f.get("primary"))
        sp = names173.normalize_name(s.get("primary"))
        if f.get("relation") != s.get("relation") or fp != sp or (f.get("primary") is None) != (s.get("primary") is None):
            out.append({"pair_id": item_id, "relation_first": f.get("relation"),
                        "relation_second": s.get("relation"), "primary_first": f.get("primary"),
                        "primary_second": s.get("primary"), "resolved_primary_first": fp,
                        "resolved_primary_second": sp,
                        "null_primary_disagreement": (f.get("primary") is None) != (s.get("primary") is None),
                        "answer_count": item_counts.get(item_id, 1)})
    return out


def _attach_claim_counts(cells, scored, reader_by_row):
    for cell in cells:
        rows = [row for row in scored if row.get("arm") == cell["arm"] and row["clean"]]
        for mode in MODES:
            data = cell["sensitivity"][mode]
            counts = {}
            for row in rows:
                record = _reader_for(row, reader_by_row)
                entity = _primary_entity(record, mode)
                if entity is not None and _canonical_entity(record) != names173.QWEN_ENTITY:
                    counts[entity] = counts.get(entity, 0) + 1
            data["google_claim_count"] = counts.get("google", 0)
            data["tencent_claim_count"] = counts.get("tencent", 0)


def _prediction_sensitivity(scored, cells, reader_by_row, reader_read, reps, seed):
    all_modes = {}
    for mode in MODES:
        summaries = set_summary(cells, mode)
        tables = family_table(cells, summaries, mode)
        boot = bootstrap_mode(scored, reader_by_row, mode=mode, reps=reps, seed=seed)
        # The identity draw is the point estimate from the same resampling path. The ordinary
        # tables remain the audit source; compare their distinct-best point with it explicitly.
        arrays = boot["estimates"]
        assert_bootstrap_point_entity_count(arrays, summaries)
        predictions = _derive_prediction_table(cells, summaries, arrays, arrays,
                                               reader_read, mode)
        all_modes[mode] = {"set_summary": summaries, "family_table": tables,
                           "predictions": predictions, "bootstrap": {
                               "seed": seed, "replicates": reps,
                               "over": "44 prompt indices sampled together across every set and dose",
                               "entity_code_count": len(boot["entity_names"]),
                               "cell_count": len(CELL_SPECS)}}
    for key in all_modes["canonical"]["predictions"]:
        canonical_verdict = all_modes["canonical"]["predictions"][key]["verdict"]
        generic_verdict = all_modes["generic"]["predictions"][key]["verdict"]
        all_modes["canonical"]["predictions"][key]["generic_only_bar_passes_do_not_count"] = \
            generic_verdict == "held" and canonical_verdict != "held"
    return all_modes


def _make_items(scored):
    import label173
    grouped = {}
    for row in scored:
        if not row["clean"]:
            continue
        key = label173.pair_id(row.get("question", ""), row.get("answer", ""))
        grouped[key] = grouped.get(key, 0) + 1
    return [{"id": key, "answer_count": count} for key, count in sorted(grouped.items())]


def _raw_fallback(scored):
    cells = []
    for spec in CELL_SPECS:
        rows = [row for row in scored if row.get("arm") == spec["arm"]]
        cells.append({**spec, "answers": len(rows), "broken": sum(row["broken"] for row in rows),
                      "clean": sum(row["clean"] for row in rows),
                      "raw_word_hits": {name: sum(row["raw_word_hits"][name] for row in rows)
                                        for name in WORD_HIT_GROUPS},
                      "hit": sum(row["hit"] for row in rows), "qwen": sum(row["qwen"] for row in rows),
                      "loop": sum(row["loop"] for row in rows)})
    return cells


def score(answers_path="results/part173/answers.jsonl", gates_path="results/part173/gates.json",
          labels_dir="results/part173/labels", out_dir="results/part173", reps=REPS, seed=SEED):
    import label173
    answers = read_jsonl(answers_path)
    scored = score_rows(answers)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gates = read_json(gates_path, {}) or {}
    reading_gates_passed = bool(gates.get("passed"))
    labels_path = Path(labels_dir)
    first_rows = read_jsonl(labels_path / "labels-ds.jsonl") if (labels_path / "labels-ds.jsonl").exists() else []
    second_rows = read_jsonl(labels_path / "labels-haiku.jsonl") if (labels_path / "labels-haiku.jsonl").exists() else []
    sample = read_json(labels_path / "second_reader.json", {}) or {}
    items = _make_items(scored)
    reader_gate = _reader_gate(labels_path, items, first_rows, second_rows)
    reader_read = bool(reading_gates_passed and reader_gate.get("passed"))
    reader_by_row = _reader_id_map(scored, first_rows)
    cells = build_cell_tables(scored, reader_by_row)
    _attach_claim_counts(cells, scored, reader_by_row)
    none_entity_distribution = reader_entity_frequency_counts(
        [row for row in scored if row.get("arm") == "none"], reader_by_row)
    cross_set_reading = cross_set_modal_table(cells)
    chance = chance_simulations(scored, reader_by_row)
    if reading_gates_passed and first_rows:
        modes = _prediction_sensitivity(scored, cells, reader_by_row, reader_read, reps, seed)
        predictions = modes["canonical"]["predictions"]
        set_summaries = modes["canonical"]["set_summary"]
        families = modes["canonical"]["family_table"]
        bootstrap_info = modes["canonical"]["bootstrap"]
    else:
        predictions = {key: {"verdict": "unread", "read_status": "unread",
                             "generic_only_bar_passes_do_not_count": False,
                             "raw_word_hit_fallback": "reported in word_hit_cells"}
                       for key in DESIGN["predictions"]}
        set_summaries = set_summary(cells)
        families = family_table(cells, set_summaries)
        modes = {"canonical": {"set_summary": set_summaries, "family_table": families,
                                "predictions": predictions},
                 "generic": {"set_summary": set_summary(cells, "generic"),
                             "family_table": family_table(cells, mode="generic")},
                 "raw": {"set_summary": set_summary(cells, "raw"),
                         "family_table": family_table(cells, mode="raw")}}
        bootstrap_info = {"seed": seed, "replicates": reps, "status": "not run; reader/generation gates unread"}
    predictions.update(weaver_prediction_table(none_entity_distribution, set_summaries, cells, reader_read))
    for cell in cells:
        cell["generic_only_bar_passes_do_not_count"] = any(
            predictions.get(key, {}).get("generic_only_bar_passes_do_not_count", False)
            for key in predictions)

    baseline = {arm: arm_stats(scored, arm) for arm in ("none", "role", "zero4")}
    disagreements = _reader_disagreements(first_rows, second_rows, sample, items)
    word_cells = _raw_fallback(scored)
    invented_modal_cells = [{"set": cell["set"], "dose": cell["dose"],
                              "modal": cell["modal"], "modal_count": cell["modal_count"],
                              "broken": cell["broken"]}
                             for cell in cells if cell["invented_modal"]]
    result = {"part": 173, "gates": gates, "reader_gate9": reader_gate,
              "readings_stopped": not reading_gates_passed,
              "prediction_read_status": "read" if reader_read else "unread",
              "word_hit_cells": word_cells, "baseline_arms": baseline,
              "none_entity_distribution": none_entity_distribution,
              "cross_set_modal_table": cross_set_reading,
              "chance_simulations": chance,
              "cells": cells, "invented_modal_cells": invented_modal_cells,
              "set_summary": set_summaries, "family_table": families,
              "predictions": predictions, "sensitivity_reads": modes,
              "reader_disagreements": disagreements,
              "reader_failures": {"ds": {"pairs": sum(not label173.valid(row) for row in first_rows),
                                            "weighted_answers": sum(int(row.get("answer_count", 1))
                                                                     for row in first_rows if not label173.valid(row))},
                                  "haiku": {"pairs": sum(not label173.valid(row) for row in second_rows),
                                              "weighted_answers": sum(int(row.get("answer_count", 1))
                                                                       for row in second_rows if not label173.valid(row))}},
              "bootstrap": bootstrap_info,
              "reader_population": {"unique_clean_pairs": len(items),
                                    "first_reader_pairs": len(first_rows),
                                    "second_reader_sample_pairs": len(sample.get("ids", [])),
                                    "second_reader_attempted_pairs": len(second_rows)}}
    run171.write_json(out / "score.json", result)
    (out / "cell_table.jsonl").write_text("".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
                                                           for row in cells),
                                          encoding="utf-8", newline="\n")
    run171.write_json(out / "family_table.json", families)
    run171.write_json(out / "cross_set_modal_table.json", cross_set_reading)
    run171.write_json(out / "prediction_table.json", predictions)
    (out / "scored_answers.jsonl").write_text("".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
                                                                for row in scored),
                                               encoding="utf-8", newline="\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part173/answers.jsonl")
    parser.add_argument("--gates", default="results/part173/gates.json")
    parser.add_argument("--labels-dir", default="results/part173/labels")
    parser.add_argument("--out", default="results/part173")
    parser.add_argument("--bootstrap-reps", type=int, default=REPS)
    args = parser.parse_args(argv)
    result = score(args.answers, args.gates, args.labels_dir, args.out, args.bootstrap_reps, SEED)
    print(json.dumps({"readings_stopped": result["readings_stopped"],
                      "reader_gate9": result["reader_gate9"].get("status"),
                      "predictions": {key: row["verdict"] for key, row in result["predictions"].items()},
                      "bootstrap": result["bootstrap"]}, ensure_ascii=True, allow_nan=False), flush=True)
    if result["readings_stopped"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
