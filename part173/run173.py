#!/usr/bin/env python3
"""Part 173 runner: six random-valued controls, a signed dose ladder, and held-out maker prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86"), str(ROOT / "part171"), str(ROOT / "part172")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import common86
import run171
import run172
from part171.prompts171 import DRAW_SEEDS as SOURCE_DRAW_SEEDS, TARGETS
from part172.prompts172 import NEW44 as NEW44
from part173.prompts173 import NEW44B


# This is the single frozen inventory of Part 173's arms, doses, thresholds and registered bars.
# Derived tuples below are conveniences only; edits to the draft design belong in this constant.
DESIGN = {
    "part": 173,
    "prompts": "NEW44B",
    "prompt_count": 44,
    "batch_size": 44,
    "max_new_tokens": 80,
    "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "set_families": {
        "picked": ("n1", "n2", "n3", "n4", "n5"),
        "rotated": ("r1_d171", "r10_d171", "r100_d171", "r1_d1171", "r10_d1171",
                    "r100_d1171", "r1_d2171", "r10_d2171", "r100_d2171"),
        "fit_at_random_coordinates": ("rand100_d171", "rand100_d1171", "rand100_d2171"),
        "random_valued": ("u100s_d173", "u100s_d1173", "u100s_d2173",
                          "u100d_d173", "u100d_d1173", "u100d_d2173"),
    },
    "source_draw_seeds": (171, 1171, 2171),
    "random_value_seeds": (173, 1173, 2173),
    "width": 9216,
    "control_arms": ("none", "role", "zero4"),
    "zero4_knockout": {"source_part": 172, "dose": 1.0},
    "doses": (0.25, 0.5, 0.75, 1.0, -0.5, -1.0),
    "replay_arms": ("n1", "rand100_d171", "r100_d2171"),
    "gate2": {"qwen_min": 37, "hit_max": 1},
    "gate3": {"prompts": 8},
    "gate4": {"matching_rows_min": 43, "rows_per_arm": 44},
    "gate5": {"google_min": 38},
    "gate6": {"relative_tolerance": 0.05, "omitted_dose": 0.5},
    "gate7": {"casefold_whitespace": True},
    "gate8": {"cosine_sigma_multiplier": 4, "length_relative_tolerance": 0.01},
    "gate9": {"known_examples": 20, "minimum_relation": 18, "minimum_primary": 18,
              "relation_kappa_min": 0.6, "replacement_agreement_min": 0.80,
              "replacement_pairs_min": 30},
    "pilot": {"batches": 3, "minimum_answers_per_second": 1.0},
    "cell": {"modal_count_min": 10, "broken_max": 8, "stable_half_min": 4},
    "bootstrap": {"replicates": 10_000, "seed": 173},
    "chance_simulation": {"draws": 10_000, "seed": 173, "answer_count": 44,
                          "expected_cells": 138, "threshold": 10},
    "second_reader": {"fraction": 0.10, "seed": 173, "minimum_entity_pairs": 5},
    "reader_errors": {"maximum_fraction": 0.005},
    "predictions": {
        "173.1": {"family": "fit_built", "minimum_sets": 5},
        "173.2": {"family": "random_valued", "minimum_sets": 2},
        "173.3": {"minimum_best_entities": 3, "google_best_share_max": 0.50},
        "173.4": {"cells": (("r100_d2171", 0.75), ("n2", 0.5)),
                  "entity": "google", "modal_count_min": 30, "broken_max": 4},
        "173.5": {"cell": ("n1", -0.5), "entity": "tencent", "modal_count_min": 10, "broken_max": 8},
        "173.6": {"minimum_above_fit_median": 5, "not_highest_of": 13},
        "173.7": {"absolute_doses": (0.5, 1.0), "different_entity_share_min": 0.75,
                  "minimum_pairs": 4},
        "173.8": {"random_google_sets_max": 1, "fit_and_picked_google_sets_min": 4},
        "173.9": {"stable_supported_share_min": 0.60, "minimum_supported_cells": 5},
    },
    "weaver_predictions": {
        "173.W1": {"minimum_known_non_qwen_count": 3},
        "173.W2": {"maximum_random_valued_sets": 1},
        "173.W3": {"minimum_under_two_times_cells": 3, "failure_multiple": 3},
    },
}

SET_FAMILIES = DESIGN["set_families"]
SET_NAMES = tuple(name for names in SET_FAMILIES.values() for name in names)
FAMILY_BY_SET = {name: family for family, names in SET_FAMILIES.items() for name in names}
DOSES = DESIGN["doses"]
PROMPTS = DESIGN["prompt_count"]
BATCH_SIZE = DESIGN["batch_size"]
MAX_NEW_TOKENS = DESIGN["max_new_tokens"]
REPLAY_ARMS = DESIGN["replay_arms"]
PART171_SETS = Path("results/part171/sets_google.pt")
PART171_ANSWERS = Path("results/part171/answers.jsonl")


def answer_key(row):
    """Keep Part 172's resumable row key: one arm and one prompt id."""
    return run172.answer_key(row)


def read_jsonl(path):
    return run171.read_jsonl(path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, value):
    run171.write_json(path, value)


def hash_value(value):
    """Hash nested data without relying on torch's abbreviated tensor repr."""
    import torch

    digest = hashlib.sha256()

    def feed(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().to(device="cpu").contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            raw = array.tobytes()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            digest.update(len(item).to_bytes(8, "big"))
            for key in sorted(item, key=lambda value: (type(value).__name__, str(value))):
                feed(key)
                feed(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"tuple\0" if isinstance(item, tuple) else b"list\0")
            digest.update(len(item).to_bytes(8, "big"))
            for value_item in item:
                feed(value_item)
        elif isinstance(item, bytes):
            digest.update(b"bytes\0")
            digest.update(len(item).to_bytes(8, "big"))
            digest.update(item)
        else:
            encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False, default=repr).encode("utf-8")
            digest.update(b"json\0")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)

    feed(value)
    return digest.hexdigest()


def make_run_key(sets_artifact_sha256, source_sets_sha256, *, model_id,
                 max_new_tokens, numpy_version=None, model_revision=None):
    """Return the provenance digest that must match every resumed answer row."""
    inputs = {"sets_artifact_sha256": str(sets_artifact_sha256),
              "source_sets_sha256": str(source_sets_sha256),
              "model_revision": str(model_revision or DESIGN["model_revision"]),
              "model_id": str(model_id), "max_new_tokens": int(max_new_tokens),
              "planned_specs_sha256": hash_value(planned_specs()),
              "planned_keys_sha256": hash_value(sorted(planned_keys())),
              "design_sha256": hash_value(DESIGN), "new44b_sha256": hash_value(list(NEW44B)),
              "numpy_version": str(numpy_version or np.__version__)}
    return hash_value(inputs)


def _versioned_path(path, version):
    path = Path(path)
    token = str(version or "unkeyed")[:12]
    candidate = path.with_name(f"{path.stem}.{token}{path.suffix}")
    index = 1
    while candidate.exists():
        index += 1
        candidate = path.with_name(f"{path.stem}.{token}.{index}{path.suffix}")
    return candidate


def verify_resume_key(out_dir, expected_key, answers_path=None):
    """Check existing provenance; preserve mismatched run files before refusing the append."""
    out = Path(out_dir)
    stage_path = out / "stage0.json"
    answer_path = Path(answers_path) if answers_path is not None else out / "answers.jsonl"
    stage = read_json(stage_path, {}) or {}
    answer_rows = read_jsonl(answer_path) if answer_path.is_file() else []
    stored_stage_key = stage.get("run_key")
    row_keys = [row.get("run_key") for row in answer_rows]
    mismatch = ((stored_stage_key is not None and stored_stage_key != expected_key) or
                any(key != expected_key for key in row_keys) or
                (stage_path.is_file() and stored_stage_key is None and bool(answer_rows)))
    if not mismatch:
        return True
    old_key = stored_stage_key or next((key for key in row_keys if key), None)
    if answer_path.is_file():
        os.replace(answer_path, _versioned_path(answer_path, old_key))
    if stage_path.is_file():
        os.replace(stage_path, _versioned_path(stage_path, old_key))
    raise ValueError(f"run key mismatch: existing answers and stage metadata were preserved under versioned names; "
                     f"expected {expected_key}, found {old_key or 'unkeyed'}")


def new_prompts():
    return [{"prompt_id": f"new44b:{index}", "question_index": index, "question": question}
            for index, question in enumerate(NEW44B)]


def part171_prompts():
    return run171.test_catalog()


def dose_arm(set_name, dose):
    return f"{set_name}_d{float(dose):g}"


def planned_specs():
    specs = [{"arm": "none"}, {"arm": "role", "role": True},
             {"arm": "zero4", "knockout": float(DESIGN["zero4_knockout"]["dose"])}]
    specs.extend({"arm": dose_arm(name, dose), "set": name, "dose": float(dose)}
                 for name in SET_NAMES for dose in DOSES)
    return specs


def planned_keys(specs=None, prompts=None):
    specs = planned_specs() if specs is None else list(specs)
    prompts = new_prompts() if prompts is None else list(prompts)
    return {(str(spec["arm"]), str(prompt["prompt_id"])) for spec in specs for prompt in prompts}


def gate1(answers, expected=None):
    expected = planned_keys() if expected is None else set(expected)
    rows = [row for row in answers if not str(row.get("arm", "")).startswith("replay:")]
    keys = [answer_key(row) for row in rows]
    try:
        valid = all(isinstance(row.get("answer"), str) and
                    isinstance(row.get("token_ids"), list) and bool(row["token_ids"]) and
                    all(int(token) >= 0 for token in row["token_ids"]) for row in rows)
    except (TypeError, ValueError):
        valid = False
    return {"passed": bool(len(keys) == len(expected) and len(set(keys)) == len(keys) and
                           set(keys) == expected and valid),
            "planned": len(expected), "answers": len(keys), "unique_keys": len(set(keys)),
            "missing": len(expected - set(keys)), "unexpected": len(set(keys) - expected),
            "finite_answers": bool(valid)}


def _first40(row):
    # Part 172 scores the first 40 generated tokens; the runner stores this token-bounded text.
    return str(row.get("first40", row.get("score_text", "")))


def gate2(answers, prompts=None):
    prompts = new_prompts() if prompts is None else list(prompts)
    rows = [row for row in answers if row.get("arm") == "none"]
    by_prompt = {str(row.get("prompt_id")): row for row in rows}
    qwen_aliases = ("Qwen", "Tongyi", "Alibaba", "通义")
    qwen_hits = sum(run171._word_match(_first40(by_prompt.get(p["prompt_id"], {})), qwen_aliases)
                    for p in prompts)
    target_hits = sum(run171.hit_match(_first40(by_prompt.get(p["prompt_id"], {})), "google")
                      for p in prompts)
    complete = len(rows) == len(prompts) == PROMPTS and len(by_prompt) == len(rows) and \
        set(by_prompt) == {p["prompt_id"] for p in prompts}
    return {"passed": bool(complete and qwen_hits >= DESIGN["gate2"]["qwen_min"] and
                           target_hits <= DESIGN["gate2"]["hit_max"]),
            "qwen_prompts": int(qwen_hits), "qwen_denominator": len(prompts),
            "hit_rows": int(target_hits), "hit_denominator": len(prompts), "complete": bool(complete)}


def gate5(answers, prompts=None):
    prompts = new_prompts() if prompts is None else list(prompts)
    rows = [row for row in answers if row.get("arm") == "role"]
    by_prompt = {str(row.get("prompt_id")): row for row in rows}
    hits = sum(run171.hit_match(_first40(by_prompt.get(p["prompt_id"], {})), "google") for p in prompts)
    complete = len(rows) == len(prompts) == PROMPTS and len(by_prompt) == len(rows) and \
        set(by_prompt) == {p["prompt_id"] for p in prompts}
    minimum = DESIGN["gate5"]["google_min"]
    return {"passed": bool(complete and hits >= minimum), "role_hits": int(hits),
            "role_denominator": len(prompts), "minimum": minimum, "complete": bool(complete)}


def gate7(prompts=None, others=None):
    from part170.prompts170 import M96
    import prompts171
    prompts = list(NEW44B if prompts is None else prompts)
    others = list(prompts171.FIT_PROMPTS) + list(M96) + list(NEW44) if others is None else list(others)
    fold = lambda text: " ".join(str(text).casefold().split())
    reference = {fold(text) for text in others}
    overlap = [str(text) for text in prompts if fold(text) in reference]
    return {"passed": len(prompts) == PROMPTS and not overlap, "prompts": len(prompts),
            "overlap": overlap, "compared_against": {"p8": len(prompts171.FIT_PROMPTS),
                                                       "m96": len(M96), "new44": len(NEW44)}}


# ---------------------------------------------------------------- random-valued set builders

def _layer_count(source_sets, layer):
    counts = source_sets.get("layer_counts", {}).get("n100", {})
    value = counts.get(str(layer), counts.get(int(layer))) if isinstance(counts, dict) else None
    if value is None:
        rep = source_sets.get("arms", {}).get("n100", {}).get(int(layer),
              source_sets.get("arms", {}).get("n100", {}).get(str(layer), {}))
        value = len(rep.get("coords", [])) if rep.get("kind") == "sparse" else 0
    return int(value)


def u100_stream_entropy(family, seed, layer):
    if family == "u100s":
        return [int(seed), 20_000, int(layer)]
    if family == "u100d":
        return [int(seed), 30_000, int(layer)]
    raise ValueError(f"unknown random-valued family {family!r}")


def draw_u100_raw(counts, width=DESIGN["width"], seeds=None):
    """Build all six raw controls from the registered, independent NumPy streams."""
    seeds = DESIGN["random_value_seeds"] if seeds is None else tuple(map(int, seeds))
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    out = {}
    for family in ("u100s", "u100d"):
        for seed in seeds:
            set_name = f"{family}_d{seed}"
            layers = {}
            for layer, raw_k in sorted(counts.items(), key=lambda item: int(item[0])):
                layer = int(layer)
                k = int(raw_k)
                if not 0 <= k <= width:
                    raise ValueError(f"{set_name}/L{layer}: k={k} is outside [0,{width}]")
                if family == "u100s":
                    rep_rng = np.random.default_rng(np.random.SeedSequence(u100_stream_entropy(family, seed, layer)))
                    coords = np.sort(rep_rng.choice(width, size=k, replace=False)) if k else np.asarray([], dtype=np.int64)
                    values = rep_rng.standard_normal(k).astype(np.float32)
                    layers[layer] = {"kind": "sparse", "coords": coords.astype(np.int64).tolist(),
                                     "values": values.tolist()}
                else:
                    if k:
                        dense_rng = np.random.default_rng(np.random.SeedSequence(
                            u100_stream_entropy(family, seed, layer)))
                        g = dense_rng.standard_normal(width)
                        norm = float(np.linalg.norm(g))
                        if not math.isfinite(norm) or norm <= 0:
                            raise ValueError(f"{set_name}/L{layer}: dense Gaussian has invalid norm")
                        values = (math.sqrt(k) * g / norm).astype(np.float32)
                    else:
                        values = np.zeros(width, dtype=np.float32)
                    layers[layer] = {"kind": "dense", "values": values.tolist()}
            out[set_name] = layers
    if set(out) != set(SET_FAMILIES["random_valued"]):
        raise ValueError("random-valued seed list no longer matches the frozen Part 173 set inventory")
    return out


def _rep_vector(rep, width):
    if rep is None:
        return np.zeros(int(width), dtype=np.float32)
    if rep["kind"] == "dense":
        value = np.asarray(rep["values"], dtype=np.float32)
    elif rep["kind"] == "sparse":
        value = np.zeros(int(width), dtype=np.float32)
        if rep.get("coords"):
            coords = np.asarray(rep["coords"], dtype=np.int64)
            values = np.asarray(rep["values"], dtype=np.float32)
            if coords.size != values.size or (coords < 0).any() or (coords >= width).any():
                raise ValueError("invalid sparse representation")
            value[coords] = values
    else:
        raise ValueError(f"unsupported Part 171 representation {rep.get('kind')!r}")
    if value.shape != (int(width),):
        raise ValueError(f"representation has shape {value.shape}, expected {(int(width),)}")
    return value


def _rep_vectors(reps, widths):
    result = {}
    for layer, width in widths.items():
        rep = reps.get(int(layer), reps.get(str(layer)))
        if rep is None:
            raise ValueError(f"push representation is missing layer {layer}")
        result[int(layer)] = _rep_vector(rep, width)
    return result


def _tensor_vectors(pushes, widths):
    return {int(layer): value.detach().to(device="cpu", dtype=__import__("torch").float32).numpy().copy()
            for layer, value in run172.set_pushes({"arms": {"temporary": pushes}}, "temporary", 1.0,
                                                   widths, "cpu").items()}


def _nominal_writes(layers, vectors_by_name):
    """Compute Σ W_l x_l on the CPU, accumulating in float64 only after weights leave the device."""
    import torch
    names = tuple(vectors_by_name)
    first_weight = layers[min(run171.LAYERS)].mlp.down_proj.weight
    output_width = int(first_weight.shape[0])
    totals = {name: np.zeros(output_width, dtype=np.float64) for name in names}
    for layer in run171.LAYERS:
        module_weight = layers[int(layer)].mlp.down_proj.weight.detach().to(
            device="cpu", dtype=torch.float32).numpy()
        weight = np.asarray(module_weight, dtype=np.float64)
        columns = []
        for name in names:
            x = np.asarray(vectors_by_name[name].get(int(layer), np.zeros(weight.shape[1])), dtype=np.float32)
            if x.shape != (weight.shape[1],):
                raise ValueError(f"{name}/L{layer} vector width {x.shape} != {(weight.shape[1],)}")
            columns.append(x.astype(np.float64))
        writes = weight @ np.stack(columns, axis=1)
        for index, name in enumerate(names):
            totals[name] += writes[:, index]
    return totals


def _scaled_rep(rep, scale):
    factor = np.float32(scale)
    if rep["kind"] == "sparse":
        return {"kind": "sparse", "coords": list(map(int, rep["coords"])),
                "values": (np.asarray(rep["values"], dtype=np.float32) * factor).tolist()}
    return {"kind": "dense", "values": (np.asarray(rep["values"], dtype=np.float32) * factor).tolist()}


def build_u100_sets(source_sets, layers, widths=None):
    """Scale six value-randomized controls to n1's measured nominal write length; never redraw."""
    widths = run172.widths_of(layers) if widths is None else {int(k): int(v) for k, v in widths.items()}
    if len(widths) != len(run171.LAYERS):
        raise ValueError("Part 173 requires widths for every Part 171 layer 0 through 20")
    counts = {int(layer): _layer_count(source_sets, layer) for layer in widths}
    raw = draw_u100_raw(counts, width=next(iter(widths.values())))
    raw_vectors = {name: _rep_vectors(reps, widths) for name, reps in raw.items()}
    n1_pushes = run172.set_pushes(source_sets, "n1", 1.0, widths, "cpu")
    n1_vectors = {int(layer): tensor.detach().cpu().numpy().astype(np.float32, copy=True)
                  for layer, tensor in n1_pushes.items()}
    unscaled = _nominal_writes(layers, {"n1": n1_vectors, **raw_vectors})
    reference_norm = float(np.linalg.norm(unscaled["n1"]))
    if not math.isfinite(reference_norm) or reference_norm <= 0:
        raise ValueError("n1 has a zero or non-finite summed nominal write")

    scaled_reps, scales = {}, {}
    for name, reps in raw.items():
        raw_norm = float(np.linalg.norm(unscaled[name]))
        if not math.isfinite(raw_norm) or raw_norm <= 0:
            raise ValueError(f"{name} has a zero or non-finite unscaled write; no draw is replaced")
        scale = reference_norm / raw_norm
        scales[name] = float(scale)
        scaled_reps[name] = {int(layer): _scaled_rep(rep, scale) for layer, rep in reps.items()}
    scaled_vectors = {name: _rep_vectors(reps, widths) for name, reps in scaled_reps.items()}
    actual_writes = _nominal_writes(layers, {"n1": n1_vectors, **scaled_vectors})
    v = np.asarray(source_sets["v"].detach().cpu().numpy() if hasattr(source_sets["v"], "detach")
                   else source_sets["v"], dtype=np.float64).reshape(-1)
    vnorm = float(np.linalg.norm(v))
    if not math.isfinite(vnorm) or vnorm <= 0:
        raise ValueError("saved Part 171 fit vector has a zero or non-finite norm")
    final_reference_norm = float(np.linalg.norm(actual_writes["n1"]))
    metrics, gate_rows, streams = {}, [], {}
    for name in SET_FAMILIES["random_valued"]:
        write = actual_writes[name]
        norm = float(np.linalg.norm(write))
        cosine = float(write @ v / (norm * vnorm)) if norm > 0 else None
        seed_text = name.rsplit("d", 1)[-1]
        family = name.split("_", 1)[0]
        streams[name] = {str(layer): u100_stream_entropy(family, int(seed_text), layer)
                         for layer in widths if counts[int(layer)] > 0}
        metrics[name] = {"scale": scales[name], "nominal_write_norm": norm,
                         "n1_nominal_write_norm": final_reference_norm, "cosine_with_fit": cosine,
                         "layer_counts": {str(layer): counts[int(layer)] for layer in widths},
                         "stream_entropy": streams[name]}
        gate_rows.append({"set": name, "norm": norm, "reference_norm": final_reference_norm,
                          "cosine": cosine, "fit_vector_size": int(v.size)})
    return {"part": 173, "arms": scaled_reps, "metrics": metrics, "layer_counts": {
                name: {str(layer): counts[int(layer)] for layer in widths}
                for name in SET_FAMILIES["random_valued"]},
            "source_set_norm": final_reference_norm, "seed_streams": streams}, gate_rows


def gate8(rows):
    expected = set(SET_FAMILIES["random_valued"])
    by_name = {str(row.get("set")): row for row in rows}
    failures = []
    if len(rows) != len(expected) or set(by_name) != expected:
        failures.append({"kind": "set_coverage", "expected": sorted(expected),
                         "observed": sorted(by_name)})
    fit_sizes = {int(row["fit_vector_size"]) for row in rows
                 if row.get("fit_vector_size") is not None}
    fit_size = next(iter(fit_sizes)) if len(fit_sizes) == 1 else 0
    cosine_max = (float(DESIGN["gate8"]["cosine_sigma_multiplier"]) / math.sqrt(fit_size)
                  if fit_size > 0 else None)
    if len(fit_sizes) != 1 or fit_size <= 0:
        failures.append({"kind": "fit_vector_size", "observed": sorted(fit_sizes)})
    for name in sorted(expected):
        row = by_name.get(name, {})
        try:
            norm, reference = float(row.get("norm")), float(row.get("reference_norm"))
            cosine = float(row.get("cosine"))
            finite = all(math.isfinite(value) for value in (norm, reference, cosine))
        except (TypeError, ValueError):
            norm = reference = cosine = float("nan")
            finite = False
        if not finite or norm <= 0 or reference <= 0:
            failures.append({"kind": "nonpositive_or_nonfinite", "set": name,
                             "norm": norm, "reference_norm": reference, "cosine": cosine})
            continue
        cosine_ok = cosine_max is not None and abs(cosine) <= cosine_max
        norm_ok = abs(norm / reference - 1.0) <= DESIGN["gate8"]["length_relative_tolerance"]
        if not cosine_ok:
            failures.append({"kind": "cosine", "set": name, "cosine": cosine,
                             "maximum_absolute_cosine": cosine_max})
        if not norm_ok:
            failures.append({"kind": "length", "set": name, "norm": norm,
                             "reference_norm": reference})
    return {"passed": not failures, "rows": len(rows), "failures": failures,
            "fit_vector_size": fit_size or None, "absolute_cosine_max": cosine_max,
            "length_relative_tolerance": DESIGN["gate8"]["length_relative_tolerance"]}


def _save_artifact(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    return digest


def _load_artifact(path):
    import pickle
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _source_hash(source_path, source_sets):
    if isinstance(source_sets, dict):
        return hash_value(source_sets)
    path = Path(source_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merge_sets(source_sets, random_sets):
    return {**source_sets,
            "arms": {**source_sets["arms"], **random_sets["arms"]},
            "metrics": {**source_sets.get("metrics", {}), **random_sets["metrics"]}}


def _set_pushes(sets, source_sets, set_name, dose, widths, device):
    if set_name in source_sets.get("arms", {}):
        return run172.set_pushes(source_sets, set_name, dose, widths, device)
    if set_name in SET_FAMILIES["picked"]:
        if set_name == "n1":
            return run172.set_pushes(source_sets, set_name, dose, widths, device)
        by_name = {name: (layer, unit) for name, layer, unit in run172.NEXT_SINGLES}
        if set_name not in by_name:
            raise KeyError(f"Part 172 defines no picked-neuron construction for {set_name}")
        layer, unit = by_name[set_name]
        return run172.single_push(source_sets, layer, unit, widths, device, dose)[0]
    if set_name not in sets.get("arms", {}):
        raise KeyError(f"no saved or generated push for {set_name}")
    return run172.set_pushes(sets, set_name, dose, widths, device)


def hook_counts(layers):
    return {int(layer): len(layers[int(layer)].mlp.down_proj._forward_pre_hooks) for layer in run171.LAYERS}


def run_spec(model, tok, layers, spec, questions, device, max_new_tokens, sets, source_sets, widths):
    if spec.get("knockout") is not None:
        # Keep the zero4 arm's four-unit hook construction owned by Part 172.
        return run172.run_spec(model, tok, layers, spec, questions, device,
                               max_new_tokens, sets, widths, seed=172)
    before = hook_counts(layers)
    pushes = None
    if spec.get("set"):
        pushes = _set_pushes(sets, source_sets, spec["set"], float(spec["dose"]), widths, device)
    roles = [TARGETS["google"]] * len(questions) if spec.get("role") else None
    try:
        rows = run171.generate_batch(model, tok, layers, questions, roles, device,
                                     max_new_tokens=max_new_tokens, pushes=pushes, seed=172)
    finally:
        after = hook_counts(layers)
        if after != before:
            raise RuntimeError(f"forward pre-hooks leaked after {spec.get('arm')}: {before} -> {after}")
    return rows


def _record(spec, prompt, answer, run_key=None):
    row = {"arm": str(spec["arm"]), "prompt_id": str(prompt["prompt_id"]),
           "question_index": int(prompt["question_index"]), "question": str(prompt["question"]), **answer}
    if run_key is not None:
        row["run_key"] = str(run_key)
    row["score_text"] = str(answer.get("first40", ""))
    if spec.get("set") is not None:
        row["set"] = str(spec["set"])
        row["dose"] = float(spec["dose"])
    if spec.get("knockout") is not None:
        row["knockout"] = float(spec["knockout"])
    return row


def _replay_record(arm, prompt, answer, run_key=None):
    row = {"arm": f"replay:{arm}", "prompt_id": str(prompt["prompt_id"]),
            "question_index": int(prompt["question_index"]), "question": str(prompt["question"]),
            "dose": 1.0, **answer, "score_text": str(answer.get("first40", ""))}
    if run_key is not None:
        row["run_key"] = str(run_key)
    return row


def gate3(model, tok, layers, args, sets, source_sets, widths, prompts=None):
    import torch
    prompts = new_prompts() if prompts is None else list(prompts)
    checks = prompts[:DESIGN["gate3"]["prompts"]]
    questions = [prompt["question"] for prompt in checks]
    before = hook_counts(layers)
    plain = run171.generate_batch(model, tok, layers, questions, [None] * len(questions), args.device,
                                  args.max_new_tokens, seed=172)
    zero_pushes = {layer: torch.zeros(width, dtype=torch.float32, device=args.device)
                   for layer, width in widths.items()}
    zero = run171.generate_batch(model, tok, layers, questions, [None] * len(questions), args.device,
                                 args.max_new_tokens, pushes=zero_pushes, seed=172)
    pushed = run_spec(model, tok, layers, {"arm": "gate3_d1", "set": "n1", "dose": 1.0},
                      questions, args.device, args.max_new_tokens, sets, source_sets, widths)
    post = run171.generate_batch(model, tok, layers, questions, [None] * len(questions), args.device,
                                 args.max_new_tokens, seed=172)
    after = hook_counts(layers)
    same = (len(plain) == len(zero) == len(post) == len(checks) and
            [row.get("token_ids") for row in plain] == [row.get("token_ids") for row in zero] ==
            [row.get("token_ids") for row in post])
    return {"passed": bool(same and before == after), "prompts": len(checks),
            "matching_zero_push": sum(a.get("token_ids") == b.get("token_ids")
                                       for a, b in zip(plain, zero)),
            "matching_post_d1_hooks_removed": sum(a.get("token_ids") == b.get("token_ids")
                                                   for a, b in zip(plain, post)),
            "d1_pushed_rows": len(pushed), "hook_counts_restored": bool(before == after),
            "plain": [{"prompt_id": p["prompt_id"], "token_ids": row["token_ids"]}
                      for p, row in zip(checks, plain)],
            "zero_push": [{"prompt_id": p["prompt_id"], "token_ids": row["token_ids"]}
                          for p, row in zip(checks, zero)],
            "post_d1": [{"prompt_id": p["prompt_id"], "token_ids": row["token_ids"]}
                        for p, row in zip(checks, post)]}


def _replay_stage(model, tok, layers, args, out, sets, source_sets, widths,
                  reference_path=PART171_ANSWERS, run_key=None):
    answer_path = Path(out) / "answers.jsonl"
    run172.repair_tail(answer_path)
    answers = read_jsonl(answer_path)
    done = {answer_key(row) for row in answers}
    reference = read_jsonl(reference_path)
    prompts = part171_prompts()
    for name in REPLAY_ARMS:
        arm = f"replay:{name}"
        if all((arm, str(prompt["prompt_id"])) in done for prompt in prompts):
            continue
        spec = {"arm": arm, "set": name, "dose": 1.0}
        generated = run_spec(model, tok, layers, spec, [p["question"] for p in prompts], args.device,
                             args.max_new_tokens, sets, source_sets, widths)
        if len(generated) != len(prompts):
            raise RuntimeError(f"{arm} generated {len(generated)} rows for {len(prompts)} replay prompts")
        rows = [_replay_record(name, prompt, answer, run_key) for prompt, answer in zip(prompts, generated)]
        run172.append_answers(answer_path, rows, done)
    replay_rows = [row for row in read_jsonl(answer_path) if str(row.get("arm", "")).startswith("replay:")]
    gate = run172.gate4(replay_rows, reference, minimum=DESIGN["gate4"]["matching_rows_min"])
    gate["artifacts"] = run172.artifact_hashes(source_sets, tok, model, args.model)
    gate["artifacts"]["pushes"]["sha256"] = (_source_hash(source_sets, source_sets)
                                                if isinstance(source_sets, dict)
                                                else hashlib.sha256(Path(source_sets).read_bytes()).hexdigest())
    return gate


def _sum_trace_writes(traces):
    import torch
    if not traces or any(trace.actual is None or trace.expected is None for trace in traces):
        return None, None
    hidden = int(traces[0].actual.shape[-1])
    actual, expected = np.zeros(hidden, dtype=np.float64), np.zeros(hidden, dtype=np.float64)
    for trace in traces:
        actual += trace.actual.reshape(-1).detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)
        expected += trace.expected.reshape(-1).detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)
    return actual, expected


def measure_gate6(model, tok, layers, sets, source_sets, device, widths):
    import torch
    context = NEW44B[0]
    inputs = run171.encode_batch(tok, [context], device, [None])

    def observed(set_name, dose):
        pushes = _set_pushes(sets, source_sets, set_name, dose, widths, device)
        traces = []
        for layer, push in sorted(pushes.items()):
            if int(np.count_nonzero(push.detach().to(device="cpu", dtype=torch.float32).numpy())):
                traces.append(run171._TraceDownWrite(layers[int(layer)].mlp.down_proj, push))
        try:
            with torch.inference_mode():
                model(**inputs, use_cache=False)
        finally:
            for trace in traces:
                trace.remove()
        return _sum_trace_writes(traces)[0]

    rows = []
    references = {}
    tolerance = DESIGN["gate6"]["relative_tolerance"]
    for name in SET_NAMES:
        reference = observed(name, 1.0)
        references[name] = reference
        denominator = float(reference @ reference) if reference is not None else 0.0
        for dose in DOSES:
            actual = reference if float(dose) == 1.0 else observed(name, float(dose))
            slope = float(actual @ reference / denominator) if actual is not None and denominator > 0 else None
            rows.append({"kind": "delivery", "set": name, "dose": float(dose),
                         "delivery_slope": slope, "reference_norm": math.sqrt(denominator) if denominator > 0 else None,
                         "actual_norm": float(np.linalg.norm(actual)) if actual is not None else None,
                         "tolerance": tolerance})
    # This intentional unit-dose push is labelled as d=0.5. Gate 6 must diagnose the
    # omitted-dose path while keeping it outside the registered delivery grid.
    omitted_pushes = _set_pushes(sets, source_sets, "n1", 1.0, widths, device)
    omitted_trace = []
    for layer, push in sorted(omitted_pushes.items()):
        if int(np.count_nonzero(push.detach().to(device="cpu", dtype=torch.float32).numpy())):
            omitted_trace.append(run171._TraceDownWrite(layers[int(layer)].mlp.down_proj, push))
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for trace in omitted_trace:
            trace.remove()
    omitted = _sum_trace_writes(omitted_trace)[0]
    n1 = references.get("n1")
    denom = float(n1 @ n1) if n1 is not None else 0.0
    rows.append({"kind": "delivery", "probe": "omitted_dose", "set": "n1",
                 "dose": DESIGN["gate6"]["omitted_dose"],
                 "delivery_slope": float(omitted @ n1 / denom) if omitted is not None and denom > 0 else None})
    return rows


def gate6(rows):
    delivery = [row for row in rows if row.get("kind") == "delivery"]
    probes = [row for row in delivery if row.get("probe") == "omitted_dose"]
    planned = [row for row in delivery if row.get("probe") != "omitted_dose"]
    expected = {(name, float(dose)) for name in SET_NAMES for dose in DOSES}
    observed_keys = {(str(row.get("set")), float(row.get("dose"))) for row in planned}
    failures = []
    if len(planned) != len(expected) or observed_keys != expected:
        failures.append({"kind": "coverage", "missing": sorted(expected - observed_keys),
                         "unexpected": sorted(observed_keys - expected), "rows": len(planned)})
    tolerance = DESIGN["gate6"]["relative_tolerance"]
    for row in planned:
        try:
            dose, slope = float(row["dose"]), float(row["delivery_slope"])
            valid = math.isfinite(slope) and abs(slope - dose) <= tolerance * abs(dose)
        except (KeyError, TypeError, ValueError):
            dose, slope, valid = row.get("dose"), row.get("delivery_slope"), False
        if not valid:
            failures.append({"kind": "signed_dose", "set": row.get("set"), "dose": dose, "slope": slope})
    expected_dose = DESIGN["gate6"]["omitted_dose"]
    try:
        omitted_slope = float(probes[0]["delivery_slope"]) if len(probes) == 1 else float("nan")
        omission_detected = math.isfinite(omitted_slope) and \
            abs(omitted_slope - expected_dose) > tolerance * abs(expected_dose)
    except (TypeError, ValueError):
        omitted_slope, omission_detected = None, False
    if not omission_detected:
        failures.append({"kind": "omitted_dose_not_caught", "rows": probes})
    return {"passed": not failures, "failures": failures, "delivery_rows": len(delivery),
            "expected_delivery_rows": len(expected), "omitted_dose_caught": bool(omission_detected),
            "omitted_probe_failures": ([{"kind": "signed_dose", "set": "n1",
                                           "dose": expected_dose, "slope": omitted_slope,
                                           "detected": bool(omission_detected)}]
                                        if probes and omission_detected else []),
            "omitted_probe_failure": ({"set": "n1", "dose": expected_dose,
                                        "slope": omitted_slope, "detected": bool(omission_detected),
                                        "failing": bool(omission_detected)} if probes else None),
            "omitted_probe_slope": omitted_slope, "omitted_probe_expected_dose": expected_dose,
            "tolerance": tolerance}


def _answer_stage(args, tok, model, layers, out, sets, source_sets, widths, pilot=None,
                  run_key=None, on_arm_complete=None):
    answer_path = Path(out) / "answers.jsonl"
    run172.repair_tail(answer_path)
    answers = read_jsonl(answer_path)
    done = {answer_key(row) for row in answers}
    if len(done) != len(answers):
        raise ValueError("answers.jsonl contains duplicate Part 173 row keys")
    prompts = new_prompts()
    specs = planned_specs()
    questions = [prompt["question"] for prompt in prompts]
    pilot_path = Path(out) / "pilot.json"
    pilot = pilot or read_json(pilot_path, {"batches": 0, "answers": 0, "elapsed_seconds": 0.0})
    for spec in specs:
        if all((spec["arm"], prompt["prompt_id"]) in done for prompt in prompts):
            if on_arm_complete is not None:
                on_arm_complete(spec["arm"], answers)
            continue
        started = time.perf_counter()
        generated = run_spec(model, tok, layers, spec, questions, args.device, args.max_new_tokens,
                             sets, source_sets, widths)
        elapsed = time.perf_counter() - started
        if len(generated) != len(prompts):
            raise RuntimeError(f"{spec['arm']} generated {len(generated)} rows for {len(prompts)} prompts")
        run172.append_answers(answer_path, [_record(spec, prompt, answer, run_key)
                                            for prompt, answer in zip(prompts, generated)], done)
        answers = read_jsonl(answer_path)
        if on_arm_complete is not None:
            on_arm_complete(spec["arm"], answers)
        if int(pilot.get("batches", 0)) < DESIGN["pilot"]["batches"]:
            pilot = run172.pilot_update(pilot_path, pilot, len(prompts), elapsed)
    return read_jsonl(answer_path), pilot


def preflight_inputs(sets_path=PART171_SETS, answers_path=PART171_ANSWERS):
    if not isinstance(sets_path, dict) and not Path(sets_path).is_file():
        raise FileNotFoundError(f"Part 171 sets are required before Part 173: {sets_path}")
    if not Path(answers_path).is_file():
        raise FileNotFoundError(f"Part 171 replay answers are required before Part 173: {answers_path}")


def run(args, loader=None, config=None):
    config = config or {}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_path = config.get("sets_path", PART171_SETS)
    reference_path = config.get("answers_path", PART171_ANSWERS)
    preflight_inputs(source_path, reference_path)
    source_sets = source_path if isinstance(source_path, dict) else run172.load_sets(source_path)
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common86.REVISION, args.device)
    else:
        tok, model = loader(args.device)
    model.requires_grad_(False)
    model.eval()
    layers = common86.find_layers(model)
    strict = bool(config.get("strict_model", loader is None))
    if strict:
        run171.validate_model(layers)
    widths = run172.widths_of(layers)
    if "unit_weight_columns" not in source_sets and "weights" not in source_sets:
        source_sets["unit_weight_columns"] = run172.single_weight_columns(layers)

    random_sets, g8_rows = build_u100_sets(source_sets, layers, widths)
    source_sha = _source_hash(source_path, source_sets)
    random_sets["source_sha256"] = source_sha
    artifact_path = Path(config.get("artifact_path", out / "sets_u100.pt"))
    candidate_path = artifact_path.with_name(f".{artifact_path.name}.candidate")
    artifact_hash = _save_artifact(candidate_path, random_sets)
    run_key = make_run_key(artifact_hash, source_sha, model_id=args.model,
                           max_new_tokens=args.max_new_tokens, numpy_version=np.__version__)
    try:
        verify_resume_key(out, run_key)
    except Exception:
        candidate_path.unlink(missing_ok=True)
        Path(str(candidate_path) + ".sha256").unlink(missing_ok=True)
        raise
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate_path, artifact_path)
    os.replace(Path(str(candidate_path) + ".sha256"), Path(str(artifact_path) + ".sha256"))
    run_key_inputs = {"sets_artifact_sha256": artifact_hash, "source_sets_sha256": source_sha,
                      "model_revision": DESIGN["model_revision"], "model_id": str(args.model),
                      "max_new_tokens": int(args.max_new_tokens),
                      "design_sha256": hash_value(DESIGN), "new44b_sha256": hash_value(list(NEW44B)),
                      "numpy_version": np.__version__}
    stage0 = {"part": 173, "stage": "stage0", "artifact": str(artifact_path),
              "artifact_sha256": artifact_hash, "source_sha256": random_sets["source_sha256"],
              "run_key": run_key, "run_key_inputs": run_key_inputs,
              "numpy_version": np.__version__,
              "set_names": list(SET_FAMILIES["random_valued"]),
              "random_set_metrics": random_sets["metrics"],
              "seed_streams": random_sets["seed_streams"]}
    write_json(out / "stage0.json", stage0)

    sets = _merge_sets(source_sets, random_sets)
    gate3_result = gate3(model, tok, layers, args, sets, source_sets, widths)
    stage0["gate3"] = gate3_result
    write_json(out / "stage0.json", stage0)
    if not gate3_result["passed"]:
        write_json(out / "gates.json", {"passed": False, "gate3": gate3_result})
        raise SystemExit("Part 173 Gate 3 failed before planned answers")

    # Keep Part 172's replay keys and exact token-id comparison for the three registered arms.
    gate4_result = _replay_stage(model, tok, layers, args, out, sets, source_sets, widths,
                                 reference_path=reference_path, run_key=run_key)
    stage0["gate4"] = gate4_result
    write_json(out / "stage0.json", stage0)
    if not gate4_result["passed"]:
        write_json(out / "gates.json", {"passed": False, "gate3": gate3_result,
                                        "gate4": gate4_result})
        raise SystemExit("Part 173 Gate 4 replay failed before planned answers")

    gate6_rows = measure_gate6(model, tok, layers, sets, source_sets, args.device, widths)
    gate6_result = gate6(gate6_rows)
    stage0["gate6"] = gate6_result
    stage0["gate6_rows"] = gate6_rows
    write_json(out / "stage0.json", stage0)
    if not gate6_result["passed"]:
        write_json(out / "gates.json", {"passed": False, "gate3": gate3_result,
                                        "gate4": gate4_result, "gate6": gate6_result})
        raise SystemExit("Part 173 Gate 6 dose delivery failed before planned answers")

    gate7_result = gate7()
    stage0["gate7"] = gate7_result
    write_json(out / "stage0.json", stage0)
    if not gate7_result["passed"]:
        write_json(out / "gates.json", {"passed": False, "gate3": gate3_result,
                                        "gate4": gate4_result, "gate6": gate6_result,
                                        "gate7": gate7_result})
        raise SystemExit("Part 173 Gate 7 prompt disjointness failed before planned answers")

    gate8_result = gate8(g8_rows)
    stage0["gate8"] = gate8_result
    stage0["gate8_rows"] = g8_rows
    write_json(out / "stage0.json", stage0)
    if not gate8_result["passed"]:
        write_json(out / "gates.json", {"passed": False, "gate3": gate3_result,
                                        "gate4": gate4_result, "gate6": gate6_result,
                                        "gate7": gate7_result, "gate8": gate8_result})
        raise SystemExit(f"Part 173 Gate 8 failed before planned answers: {gate8_result['failures']}")

    def on_arm_complete(arm, partial_answers):
        if arm == "none":
            result = gate2(partial_answers)
            stage0["gate2"] = result
            write_json(out / "stage0.json", stage0)
            if not result["passed"]:
                write_json(out / "gates.json", {"passed": False, "gate2": result})
                raise SystemExit("Part 173 Gate 2 failed after none arm, before ladder")
        elif arm == "role":
            result = gate5(partial_answers)
            stage0["gate5"] = result
            write_json(out / "stage0.json", stage0)
            if not result["passed"]:
                write_json(out / "gates.json", {"passed": False, "gate2": stage0.get("gate2"),
                                                "gate5": result})
                raise SystemExit("Part 173 Gate 5 failed after role arm, before ladder")

    answers, pilot = _answer_stage(args, tok, model, layers, out, sets, source_sets, widths,
                                   run_key=run_key, on_arm_complete=on_arm_complete)
    g1 = gate1(answers)
    g2 = stage0["gate2"]
    g5 = stage0["gate5"]
    gates = {"gate1": g1, "gate2": g2, "gate3": gate3_result, "gate4": gate4_result,
             "gate5": g5, "gate6": gate6_result, "gate7": gate7_result, "gate8": gate8_result}
    gates["passed"] = all(gates[name]["passed"] for name in gates if name != "passed")
    write_json(out / "gates.json", gates)
    write_json(out / "stage0.json", stage0)
    result = {"stage": "test", "planned_answers": len(planned_keys()),
              "answers": len([row for row in answers if not str(row.get("arm", "")).startswith("replay:")]),
              "replay_answers": sum(str(row.get("arm", "")).startswith("replay:") for row in answers),
              "gates": {name: value["passed"] for name, value in gates.items() if name != "passed"},
              "pilot": pilot}
    if not gates["passed"]:
        raise SystemExit(f"Part 173 post-ladder gates failed: " + ", ".join(
            name for name, value in gates.items() if name != "passed" and not value["passed"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test",), default="test")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=run171.MODEL_DEFAULT)
    parser.add_argument("--out-dir", default="results/part173")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)
    if args.batch_size != BATCH_SIZE:
        parser.error("registered generation batches hold one arm's 44 prompts")
    result = run(args)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
