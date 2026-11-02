#!/usr/bin/env python3
"""Part 174 runner: 60 dense random pushes, two signs, and the NEW44B maker prompts."""
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
for entry in (str(ROOT), str(HERE), str(ROOT / "part86"), str(ROOT / "part171"),
              str(ROOT / "part172"), str(ROOT / "part173")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import common86
import label173
import names173
import prompts173
import run171
import run172
import run173
import score173
from part171.prompts171 import TARGETS
from part173.prompts173 import NEW44B


DESIGN = {
    "part": 174,
    "prompts": "NEW44B",
    "prompt_count": 44,
    "batch_size": 44,
    "max_new_tokens": 80,
    "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "width": 9216,
    "families": {
        "u100d": {"seeds": tuple(range(174001, 174041)), "stream": 30000},
        "f100d": {"seeds": tuple(range(174101, 174121)), "layers": tuple(range(21)),
                  "k_per_layer": 100 / 21, "stream": 40000},
    },
    "control_arms": ("none", "role"),
    "signs": (0.5, -0.5),
    "replays": (("u100d_d2173", 0.5), ("n3", -0.5)),
    "gate1": {"rows_per_arm": 44},
    "gate2": {"qwen_min": 37, "hit_max": 1},
    "gate3": {"prompts": 8},
    "gate4": {"matching_rows_min": 43, "rows_per_arm": 44},
    "gate5": {"google_min": 38},
    "gate6": {"relative_tolerance": 0.05},
    "gate8": {"absolute_cosine_max": 0.079, "length_relative_tolerance": 0.01},
    "gate9": {"known_examples": 20, "minimum_relation": 18, "minimum_primary": 18,
              "relation_kappa_min": 0.6, "replacement_agreement_min": 0.80,
              "replacement_pairs_min": 30},
    "cell": {"modal_count_min": 10, "broken_max": 8, "stable_half_min": 4},
    "pilot": {"batches": 3, "minimum_answers_per_second": 1.0},
    "bootstrap": {"replicates": 10_000, "seed": 174},
    "reader_errors": {"maximum_fraction": 0.005},
    "predictions": {
        "174.1": {"u100d_supported_min": 20},
        "174.2": {"google_share_min": 0.80, "not_read_below": 10},
        "174.3": {"u100d_other_led_max": 4},
        "174.4": {"both_supported_share_max": 0.25, "not_read_below_draws": 8},
        "174.5": {"f100d_share_min_of_u100d": 0.5, "not_read_below_u100d_supported": 10},
        "174.6": {"absolute_spearman_max": 0.30, "projection": "cf"},
        "174.7": {"u100d_median_broken_max": 8},
        "174.8": {"stable_share_min": 0.60, "not_read_below_supported": 5},
        "174.9": {"absolute_spearman_max": 0.30, "projection": "cr"},
        "174.10": {"google_share_min": 0.80, "not_read_below": 5},
        "174.W1": {"u100d_supported_min": 30, "falsifier_below": 20},
        "174.W2": {"known_non_qwen_falsifier_share": 0.50},
        "174.W3": {"share_absolute_difference_max": 0.20, "falsifier_f_below_u_fraction": 0.50},
    },
}

SET_NAMES = tuple(f"{family}_d{seed}" for family, data in DESIGN["families"].items()
                  for seed in data["seeds"])
FAMILY_BY_SET = {name: family for family, data in DESIGN["families"].items()
                 for name in (f"{family}_d{seed}" for seed in data["seeds"])}
PROMPTS = DESIGN["prompt_count"]
BATCH_SIZE = DESIGN["batch_size"]
MAX_NEW_TOKENS = DESIGN["max_new_tokens"]
PART171_SETS = Path("results/part171/sets_google.pt")
PART171_ANSWERS = Path("results/part171/answers.jsonl")
PART173_ANSWERS = Path("results/part173/answers.jsonl")
PART173_SETS = Path("results/part173/sets_u100.pt")
PROJECTION_TOKEN_IDS = {" Google": 4923, " Alibaba": 52540}
NEW44B_SHA256 = "a6952e5b71e3f9170beba488a6422b1934b61622be54073109f50ef172451802"
PART173_SOURCE_MODULES = {
    "label173": label173,
    "names173": names173,
    "prompts173": prompts173,
    "run173": run173,
    "score173": score173,
}


def read_jsonl(path):
    return run171.read_jsonl(path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, value):
    run171.write_json(path, value)


def prompts():
    return [{"prompt_id": f"new44b:{index}", "question_index": index, "question": question}
            for index, question in enumerate(NEW44B)]


def gate7(questions=NEW44B, others=None):
    """Check prompt disjointness and pin NEW44B to Part 173's frozen tuple."""
    source_sha = run173.hash_value(tuple(prompts173.NEW44B))
    assert source_sha == NEW44B_SHA256, "Part 173 NEW44B prompts changed from the registered tuple"
    question_sha = run173.hash_value(tuple(questions))
    result = run173.gate7(questions, others=others)
    result.update({"new44b_sha256": question_sha,
                   "part173_new44b_sha256": source_sha,
                   "sha256_matches": question_sha == source_sha})
    result["passed"] = bool(result["passed"] and result["sha256_matches"])
    return result


def dose_arm(set_name, dose):
    return f"{set_name}_d{float(dose):g}"


def planned_specs():
    specs = [{"arm": "none"}, {"arm": "role", "role": True}]
    specs.extend({"arm": dose_arm(name, dose), "set": name, "dose": float(dose)}
                 for name in SET_NAMES for dose in DESIGN["signs"])
    return specs


def planned_keys(specs=None, prompt_rows=None):
    specs = planned_specs() if specs is None else list(specs)
    prompt_rows = prompts() if prompt_rows is None else list(prompt_rows)
    return {(str(spec["arm"]), str(prompt["prompt_id"]))
            for spec in specs for prompt in prompt_rows}


def answer_key(row):
    return run172.answer_key(row)


def gate1(answers, expected=None):
    expected = planned_keys() if expected is None else set(expected)
    rows = [row for row in answers if not str(row.get("arm", "")).startswith("replay:")]
    keys = [answer_key(row) for row in rows]
    try:
        valid = all(isinstance(row.get("answer"), str) and isinstance(row.get("token_ids"), list) and
                    bool(row["token_ids"]) and all(int(token) >= 0 for token in row["token_ids"])
                    for row in rows)
    except (TypeError, ValueError):
        valid = False
    return {"passed": bool(len(keys) == len(expected) and len(set(keys)) == len(keys) and
                           set(keys) == expected and valid),
            "planned": len(expected), "answers": len(keys), "unique_keys": len(set(keys)),
            "missing": len(expected - set(keys)), "unexpected": len(set(keys) - expected),
            "valid_answer_and_token_ids": bool(valid)}


def _set_pushes(sets, source_sets, set_name, dose, widths, device):
    return run173._set_pushes(sets, source_sets, set_name, dose, widths, device)


def run_spec(model, tok, layers, spec, questions, device, max_new_tokens, sets,
             source_sets, widths):
    before = run173.hook_counts(layers)
    pushes = (_set_pushes(sets, source_sets, spec["set"], float(spec["dose"]), widths, device)
              if spec.get("set") else None)
    roles = [TARGETS["google"]] * len(questions) if spec.get("role") else None
    try:
        rows = run171.generate_batch(model, tok, layers, questions, roles, device,
                                     max_new_tokens=max_new_tokens, pushes=pushes, seed=172)
    finally:
        after = run173.hook_counts(layers)
        if after != before:
            raise RuntimeError(f"forward pre-hooks leaked after {spec.get('arm')}: {before} -> {after}")
    return rows


def draw_f100d_raw(seeds=None, widths=None):
    """Use the registered f100d streams, retaining tensors compactly in the saved artifact."""
    import torch
    seeds = DESIGN["families"]["f100d"]["seeds"] if seeds is None else tuple(map(int, seeds))
    widths = ({layer: DESIGN["width"] for layer in DESIGN["families"]["f100d"]["layers"]}
              if widths is None else {int(layer): int(width) for layer, width in widths.items()})
    k = float(DESIGN["families"]["f100d"]["k_per_layer"])
    result = {}
    for seed in seeds:
        layer_reps = {}
        for layer in DESIGN["families"]["f100d"]["layers"]:
            width = widths[int(layer)]
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 40000, int(layer)]))
            gaussian = rng.standard_normal(width)
            norm = float(np.linalg.norm(gaussian))
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError(f"f100d_d{seed}/L{layer}: Gaussian has invalid norm")
            values = (math.sqrt(k) * gaussian / norm).astype(np.float32)
            layer_reps[int(layer)] = {"kind": "dense", "values": torch.from_numpy(values.copy())}
        result[f"f100d_d{seed}"] = layer_reps
    return result


def draw_u100d_raw(counts, seeds=None, widths=None):
    """Apply Part 173's imported u100d stream rule to Part 174's new seed interval."""
    import torch
    seeds = DESIGN["families"]["u100d"]["seeds"] if seeds is None else tuple(map(int, seeds))
    widths = ({layer: DESIGN["width"] for layer in run171.LAYERS}
              if widths is None else {int(layer): int(width) for layer, width in widths.items()})
    result = {}
    for seed in seeds:
        layer_reps = {}
        for layer, width in widths.items():
            k = int(counts[int(layer)])
            if not 0 <= k <= width:
                raise ValueError(f"u100d_d{seed}/L{layer}: k={k} outside [0,{width}]")
            if k:
                entropy = run173.u100_stream_entropy("u100d", seed, layer)
                rng = np.random.default_rng(np.random.SeedSequence(entropy))
                gaussian = rng.standard_normal(width)
                norm = float(np.linalg.norm(gaussian))
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError(f"u100d_d{seed}/L{layer}: Gaussian has invalid norm")
                values = (math.sqrt(k) * gaussian / norm).astype(np.float32)
            else:
                values = np.zeros(width, dtype=np.float32)
            layer_reps[int(layer)] = {"kind": "dense", "values": torch.from_numpy(values.copy())}
        result[f"u100d_d{seed}"] = layer_reps
    return result


def _scaled_compact_rep(rep, factor):
    import torch
    values = rep["values"]
    if isinstance(values, torch.Tensor):
        scaled = values.detach().to(device="cpu", dtype=torch.float32) * np.float32(factor)
    else:
        scaled = torch.as_tensor(np.asarray(values, dtype=np.float32) * np.float32(factor))
    return {"kind": "dense", "values": scaled.contiguous()}


def _layer_count(source_sets, layer):
    return run173._layer_count(source_sets, layer)


def build_sets(source_sets, layers, widths=None):
    """Draw and globally scale all 60 sets to the written length of n1; no draw is replaced."""
    import torch
    widths = run172.widths_of(layers) if widths is None else {int(k): int(v) for k, v in widths.items()}
    if set(widths) != set(map(int, run171.LAYERS)):
        raise ValueError("Part 174 requires widths for every Part 171 layer")
    counts = {int(layer): _layer_count(source_sets, layer) for layer in widths}
    raw = draw_u100d_raw(counts, widths=widths)
    raw.update(draw_f100d_raw(widths=widths))

    raw_vectors = {name: run173._rep_vectors(reps, widths) for name, reps in raw.items()}
    n1_pushes = run172.set_pushes(source_sets, "n1", 1.0, widths, "cpu")
    n1_vectors = {int(layer): push.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
                  for layer, push in n1_pushes.items()}
    writes = run173._nominal_writes(layers, {"n1": n1_vectors, **raw_vectors})
    reference_norm = float(np.linalg.norm(writes["n1"]))
    if not math.isfinite(reference_norm) or reference_norm <= 0:
        raise ValueError("n1 has a zero or non-finite summed nominal write")

    scales, scaled = {}, {}
    for name in SET_NAMES:
        norm = float(np.linalg.norm(writes[name]))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"{name} has a zero or non-finite write; no draw is replaced")
        factor = reference_norm / norm
        scales[name] = float(factor)
        scaled[name] = {int(layer): _scaled_compact_rep(rep, factor)
                        for layer, rep in raw[name].items()}
    scaled_vectors = {name: run173._rep_vectors(reps, widths) for name, reps in scaled.items()}
    writes = run173._nominal_writes(layers, {"n1": n1_vectors, **scaled_vectors})
    v_value = source_sets["v"]
    v = np.asarray(v_value.detach().to(device="cpu", dtype=torch.float32).numpy()
                   if hasattr(v_value, "detach") else v_value, dtype=np.float64).reshape(-1)
    vnorm = float(np.linalg.norm(v))
    if not math.isfinite(vnorm) or vnorm <= 0:
        raise ValueError("saved Part 171 fit vector has a zero or non-finite norm")

    layer_counts = {}
    streams = {}
    for name in SET_NAMES:
        family = FAMILY_BY_SET[name]
        seed = int(name.rsplit("d", 1)[1])
        if family == "u100d":
            layer_counts[name] = {str(layer): counts[int(layer)] for layer in widths}
            streams[name] = {str(layer): run173.u100_stream_entropy("u100d", seed, layer)
                             for layer in widths if counts[int(layer)] > 0}
        else:
            layer_counts[name] = {str(layer): DESIGN["families"]["f100d"]["k_per_layer"]
                                  for layer in widths}
            streams[name] = {str(layer): [seed, 40000, int(layer)] for layer in widths}
    artifact = {"part": 174, "arms": scaled, "layer_counts": layer_counts,
                "seed_streams": streams, "global_scales": scales}
    n1_norm_final = float(np.linalg.norm(writes["n1"]))
    metrics = {}
    for name in SET_NAMES:
        write = writes[name]
        norm = float(np.linalg.norm(write))
        cosine = float(write @ v / (norm * vnorm))
        metrics[name] = {"family": FAMILY_BY_SET[name], "seed": int(name.rsplit("d", 1)[1]),
                         "global_scale": scales[name], "nominal_write_norm": norm,
                         "n1_nominal_write_norm": n1_norm_final, "cf": cosine,
                         "layer_counts": layer_counts[name], "stream_entropy": streams[name]}
    artifact["metrics"] = metrics
    # Return the actual nominal writes for the output-embedding projection calculated at snapshot.
    artifact["source_n1_nominal_write_norm"] = n1_norm_final
    return artifact, writes, v


def _artifact_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_artifact(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    digest = _artifact_sha(path)
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


def output_projection_vectors(tok, model, source_sets, writes):
    """Compute cf/cr with the registered, frozen one-token maker ids."""
    import torch
    token_ids = {}
    for text, expected in PROJECTION_TOKEN_IDS.items():
        encoded = run172.single_token_ids(tok, text)
        if encoded != [expected]:
            raise ValueError(f'projection token {text!r} must encode to frozen id {expected}; got {encoded}')
        token_ids[text] = encoded[0]
    output = model.get_output_embeddings()
    weight = output.weight
    google_id = token_ids[" Google"]
    alibaba_id = token_ids[" Alibaba"]
    if max(google_id, alibaba_id) >= weight.shape[0]:
        raise ValueError("projection token id is outside the output embedding")
    google_row = weight[google_id].detach().to(device="cpu", dtype=torch.float32)
    alibaba_row = weight[alibaba_id].detach().to(device="cpu", dtype=torch.float32)
    e = (google_row - alibaba_row).to(dtype=torch.float64).numpy()
    e_norm = float(np.linalg.norm(e))
    if not math.isfinite(e_norm) or e_norm <= 0:
        raise ValueError('output embedding difference for " Google" and " Alibaba" has zero or invalid norm')
    e = e / e_norm
    v_value = source_sets["v"]
    v = np.asarray(v_value.detach().to(device="cpu", dtype=torch.float32).numpy()
                   if hasattr(v_value, "detach") else v_value, dtype=np.float64).reshape(-1)
    if v.size != e.size:
        raise ValueError(f"fit vector width {v.size} differs from output embedding width {e.size}")
    v_norm = float(np.linalg.norm(v))
    if not math.isfinite(v_norm) or v_norm <= 0:
        raise ValueError("saved Part 171 fit vector has a zero or non-finite norm")
    v = v / v_norm
    result = {}
    for name in SET_NAMES:
        w = np.asarray(writes[name], dtype=np.float64).reshape(-1)
        w_norm = float(np.linalg.norm(w))
        if w.size != v.size or not math.isfinite(w_norm) or w_norm <= 0:
            raise ValueError(f"{name} has invalid nominal write size or norm")
        result[name] = {"cf": float(w @ v / w_norm), "cr": float(w @ e / w_norm),
                        "write_norm": w_norm}
    values_cf = np.asarray([result[name]["cf"] for name in SET_NAMES], dtype=np.float64)
    values_cr = np.asarray([result[name]["cr"] for name in SET_NAMES], dtype=np.float64)
    # Sample standard deviation (ddof=1) across the 60 preregistered directions.
    stats = {"cf_sd": float(np.std(values_cf, ddof=1)), "cr_sd": float(np.std(values_cr, ddof=1)),
             "sd_ddof": 1, "draws": len(SET_NAMES), "google_token_id": google_id,
             "alibaba_token_id": alibaba_id, "e_norm_before_unit_scaling": e_norm,
             "cf_test": "read" if np.std(values_cf, ddof=1) >= 0.01 else "not read",
             "cr_test": "read" if np.std(values_cr, ddof=1) >= 0.01 else "not read"}
    return result, stats


def gate8(rows, fit_vector_size):
    expected = set(SET_NAMES)
    by_name = {str(row.get("set")): row for row in rows}
    failures = []
    if len(rows) != len(expected) or set(by_name) != expected:
        failures.append({"kind": "set_coverage", "expected": len(expected), "observed": len(by_name)})
    try:
        fit_vector_size = int(fit_vector_size)
    except (TypeError, ValueError):
        fit_vector_size = 0
    cosine_max = float(DESIGN["gate8"]["absolute_cosine_max"])
    if fit_vector_size <= 0:
        failures.append({"kind": "fit_vector_size", "value": fit_vector_size})
    for name in sorted(expected):
        row = by_name.get(name, {})
        try:
            norm, reference, cosine = float(row["norm"]), float(row["reference_norm"]), float(row["cf"])
            finite = all(math.isfinite(value) for value in (norm, reference, cosine))
        except (KeyError, TypeError, ValueError):
            norm = reference = cosine = float("nan")
            finite = False
        if not finite or norm <= 0 or reference <= 0:
            failures.append({"kind": "nonpositive_or_nonfinite", "set": name})
            continue
        if abs(cosine) > cosine_max:
            failures.append({"kind": "cosine", "set": name, "absolute_cosine": abs(cosine),
                             "maximum": cosine_max})
        ratio_error = abs(norm / reference - 1.0)
        if ratio_error > DESIGN["gate8"]["length_relative_tolerance"] + 1e-12:
            failures.append({"kind": "length", "set": name, "norm": norm,
                             "reference_norm": reference, "relative_error": ratio_error})
    return {"passed": not failures, "rows": len(rows), "failures": failures,
            "fit_vector_size": fit_vector_size or None, "absolute_cosine_max": cosine_max,
            "eligibility_screen": True,
            "readings_conditional_on_all_60_draws_passing": bool(not failures and len(rows) == 60),
            "length_relative_tolerance": DESIGN["gate8"]["length_relative_tolerance"]}


def gate6(rows):
    expected = {(name, float(sign)) for name in SET_NAMES for sign in DESIGN["signs"]}
    observed = []
    for row in rows:
        try:
            observed.append((str(row.get("set")), float(row.get("dose", float("nan")))))
        except (TypeError, ValueError):
            observed.append((str(row.get("set")), float("nan")))
    failures = []
    if len(observed) != len(expected) or set(observed) != expected:
        failures.append({"kind": "coverage", "expected": len(expected), "observed": len(observed),
                         "missing": sorted(expected - set(observed)),
                         "unexpected": sorted(set(observed) - expected)})
    tolerance = DESIGN["gate6"]["relative_tolerance"]
    for row in rows:
        try:
            dose = float(row["dose"])
            slope = float(row["delivery_slope"])
            valid = math.isfinite(slope) and abs(slope / dose - 1.0) <= tolerance + 1e-12
        except (KeyError, TypeError, ValueError):
            dose, slope, valid = row.get("dose"), row.get("delivery_slope"), False
        if not valid:
            failures.append({"kind": "delivery_slope", "set": row.get("set"),
                             "dose": dose, "slope": slope})
    return {"passed": not failures, "rows": len(rows), "expected_rows": len(expected),
            "tolerance": tolerance, "failures": failures}


def gate4(replay_rows, reference_rows, minimum=None):
    minimum = int(DESIGN["gate4"]["matching_rows_min"] if minimum is None else minimum)
    replay_rows = list(replay_rows)
    reference_rows = list(reference_rows)
    references, reference_raw_counts = {}, {}
    expected_ids = {str(row["prompt_id"]) for row in prompts()}

    def tokens(row):
        try:
            values = row.get("token_ids")
            if not isinstance(values, list):
                return None
            return list(map(int, values))
        except (TypeError, ValueError):
            return None

    for set_name, dose in DESIGN["replays"]:
        arm = dose_arm(set_name, dose)
        saved = [row for row in reference_rows if row.get("arm") == arm]
        references[arm] = {str(row.get("prompt_id")): row for row in saved}
        reference_raw_counts[arm] = len(saved)
    result, all_ok = {}, True
    for set_name, dose in DESIGN["replays"]:
        arm = dose_arm(set_name, dose)
        replay_arm = f"replay:{arm}"
        rows = [row for row in replay_rows if row.get("arm") == replay_arm]
        by_prompt = {}
        duplicates = []
        for row in rows:
            pid = str(row.get("prompt_id"))
            if pid in by_prompt:
                duplicates.append(pid)
            by_prompt[pid] = row
        ref = references[arm]
        reference_duplicates = reference_raw_counts[arm] - len(ref)
        matches = sum(pid in ref and tokens(row) is not None and tokens(ref[pid]) is not None and
                      tokens(row) == tokens(ref[pid]) for pid, row in by_prompt.items())
        missing = sorted(set(ref) - set(by_prompt))
        unexpected = sorted(set(by_prompt) - set(ref))
        wrong_reference_ids = sorted(set(ref) ^ expected_ids)
        passed = (reference_raw_counts[arm] == DESIGN["gate4"]["rows_per_arm"] and
                  len(ref) == DESIGN["gate4"]["rows_per_arm"] and reference_duplicates == 0 and
                  not wrong_reference_ids and
                  len(rows) == DESIGN["gate4"]["rows_per_arm"] and not duplicates and
                  not missing and not unexpected and matches >= minimum)
        result[arm] = {"replay_arm": replay_arm, "reference_rows": len(ref),
                       "reference_raw_rows": reference_raw_counts[arm],
                       "reference_duplicate_rows": reference_duplicates,
                       "wrong_reference_prompt_ids": wrong_reference_ids, "rows": len(rows),
                       "matching": int(matches), "minimum_matching": minimum,
                       "missing_prompts": missing, "unexpected_prompts": unexpected,
                       "duplicate_prompts": duplicates, "passed": bool(passed)}
        all_ok = all_ok and passed
    return {"passed": bool(all_ok), "minimum_matching": minimum, "per_arm": result}


def _record(spec, prompt, answer, run_key=None):
    row = {"arm": str(spec["arm"]), "prompt_id": str(prompt["prompt_id"]),
           "question_index": int(prompt["question_index"]), "question": str(prompt["question"]), **answer}
    if run_key is not None:
        row["run_key"] = str(run_key)
    if spec.get("set") is not None:
        row["set"] = str(spec["set"])
        row["dose"] = float(spec["dose"])
    return row


def _replay_record(spec, prompt, answer, run_key=None):
    replay = {**spec, "arm": f"replay:{spec['arm']}"}
    return _record(replay, prompt, answer, run_key)


def gate3(model, tok, layers, args, sets, source_sets, widths, prompt_rows=None):
    import torch
    prompt_rows = prompts() if prompt_rows is None else list(prompt_rows)
    checks = prompt_rows[:DESIGN["gate3"]["prompts"]]
    questions = [row["question"] for row in checks]
    before = run173.hook_counts(layers)
    plain = run171.generate_batch(model, tok, layers, questions, [None] * len(questions), args.device,
                                  args.max_new_tokens, seed=172)
    zero = {layer: torch.zeros(width, dtype=torch.float32, device=args.device)
            for layer, width in widths.items()}
    zeroed = run171.generate_batch(model, tok, layers, questions, [None] * len(questions), args.device,
                                   args.max_new_tokens, pushes=zero, seed=172)
    pushed = run_spec(model, tok, layers, {"arm": "gate3_push", "set": "u100d_d174001", "dose": 0.5},
                      questions, args.device, args.max_new_tokens, sets, source_sets, widths)
    after = run171.generate_batch(model, tok, layers, questions, [None] * len(questions), args.device,
                                  args.max_new_tokens, seed=172)
    hooks_restored = before == run173.hook_counts(layers)
    zero_matches = sum(a.get("token_ids") == b.get("token_ids") for a, b in zip(plain, zeroed))
    after_matches = sum(a.get("token_ids") == b.get("token_ids") for a, b in zip(plain, after))
    passed = (len(plain) == len(zeroed) == len(after) == len(checks) and
              zero_matches == len(checks) and after_matches == len(checks) and hooks_restored and
              len(pushed) == len(checks))
    return {"passed": bool(passed), "prompts": len(checks), "matching_zero_push": zero_matches,
            "matching_after_push_hooks_removed": after_matches, "pushed_rows": len(pushed),
            "hook_counts_restored": bool(hooks_restored)}


def _sum_actual_traces(traces):
    import torch
    if not traces or any(trace.actual is None for trace in traces):
        return None
    total = None
    for trace in traces:
        value = trace.actual.reshape(-1).detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)
        total = value.copy() if total is None else total + value
    return total


def measure_gate6(model, tok, layers, sets, source_sets, device, widths):
    import torch
    context = NEW44B[0]
    inputs = run171.encode_batch(tok, [context], device, [None])

    def observed(set_name, dose):
        pushes = _set_pushes(sets, source_sets, set_name, dose, widths, device)
        traces = []
        for layer, push in sorted(pushes.items()):
            values = push.detach().to(device="cpu", dtype=torch.float32).numpy()
            if np.count_nonzero(values):
                traces.append(run171._TraceDownWrite(layers[int(layer)].mlp.down_proj, push))
        try:
            with torch.inference_mode():
                model(**inputs, use_cache=False)
        finally:
            for trace in traces:
                trace.remove()
        return _sum_actual_traces(traces)

    rows = []
    tolerance = DESIGN["gate6"]["relative_tolerance"]
    for name in SET_NAMES:
        reference = observed(name, 1.0)
        denom = float(reference @ reference) if reference is not None else 0.0
        for dose in DESIGN["signs"]:
            actual = observed(name, dose)
            slope = float(actual @ reference / denom) if actual is not None and denom > 0 else None
            rows.append({"set": name, "dose": float(dose), "delivery_slope": slope,
                         "reference_norm": math.sqrt(denom) if denom > 0 else None,
                         "actual_norm": float(np.linalg.norm(actual)) if actual is not None else None,
                         "tolerance": tolerance})
    return rows


def _resume_key(artifact_sha, source_sha, reference_sha, reference_sets_sha, args):
    inputs = {"part": 174, "artifact_sha256": artifact_sha, "part171_sets_sha256": source_sha,
              "part173_reference_sha256": reference_sha, "design_sha256": run173.hash_value(DESIGN),
              "part173_sets_sha256": reference_sets_sha,
              "new44b_sha256": NEW44B_SHA256,
              "part173_modules_sha256": {
                  name: _artifact_sha(Path(module.__file__).resolve())
                  for name, module in sorted(PART173_SOURCE_MODULES.items())},
              "model_id": str(args.model),
              "model_revision": DESIGN["model_revision"], "max_new_tokens": int(args.max_new_tokens),
              "numpy_version": np.__version__}
    return run173.hash_value(inputs), inputs


def preflight_inputs(sets_path=PART171_SETS, reference_answers=PART173_ANSWERS,
                    reference_sets=PART173_SETS):
    for path, label in ((sets_path, "Part 171 sets"), (reference_answers, "Part 173 answers"),
                        (reference_sets, "Part 173 sets_u100 artifact")):
        if not isinstance(path, dict) and not Path(path).is_file():
            raise FileNotFoundError(f"{label} are required before Part 174: {path}")


def _replay_stage(model, tok, layers, args, out, sets, source_sets, widths,
                  reference_answers, reference_sets, run_key):
    answer_path = Path(out) / "answers.jsonl"
    run172.repair_tail(answer_path)
    answers = read_jsonl(answer_path) if answer_path.is_file() else []
    done = {answer_key(row) for row in answers}
    prompt_rows = prompts()
    for set_name, dose in DESIGN["replays"]:
        base_arm = dose_arm(set_name, dose)
        replay_arm = f"replay:{base_arm}"
        if all((replay_arm, row["prompt_id"]) in done for row in prompt_rows):
            continue
        spec = {"arm": base_arm, "set": set_name, "dose": float(dose)}
        generated = run_spec(model, tok, layers, spec, [row["question"] for row in prompt_rows],
                             args.device, args.max_new_tokens, sets, source_sets, widths)
        records = [_replay_record(spec, prompt, answer, run_key)
                   for prompt, answer in zip(prompt_rows, generated)]
        run172.append_answers(answer_path, records, done)
        answers = read_jsonl(answer_path)
    reference_rows = read_jsonl(reference_answers)
    gate = gate4(answers, reference_rows)
    part173_artifact_path = (Path(args.part173_sets) if not isinstance(reference_sets, dict)
                             else None)
    part173_sets_sha = (_artifact_sha(part173_artifact_path) if part173_artifact_path is not None
                        else run173.hash_value(reference_sets))
    gate["artifacts"] = {
        "part174_sets": {"path": str(args.out_dir / Path("sets_174.pt")),
                         "sha256": _artifact_sha(args.out_dir / "sets_174.pt")},
        "part173_sets_u100": {"path": str(part173_artifact_path) if part173_artifact_path else "in-memory fixture",
                              "sha256": part173_sets_sha},
        "part171_sets": {"path": str(args.part171_sets),
                         "sha256": (_artifact_sha(args.part171_sets)
                                    if not isinstance(args.part171_sets, dict) else
                                    run173.hash_value(args.part171_sets))},
        "hook_source": {"path": str(Path(run171.__file__).resolve()),
                        "sha256": _artifact_sha(Path(run171.__file__).resolve())},
        "tokenizer_model_metadata": run172.artifact_hashes(args.part171_sets, tok, model, args.model),
        "snapshot": DESIGN["model_revision"],
    }
    return gate


def run(args, loader=None, config=None):
    config = config or {}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_path = config.get("sets_path", args.part171_sets)
    reference_answers = config.get("answers_path", args.part173_answers)
    reference_sets = config.get("part173_sets_path", args.part173_sets)
    preflight_inputs(source_path, reference_answers, reference_sets)
    source_sets = source_path if isinstance(source_path, dict) else run172.load_sets(source_path)
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common86.REVISION, args.device)
    else:
        tok, model = loader(args.device)
    model.requires_grad_(False)
    model.eval()
    layers = common86.find_layers(model)
    if bool(config.get("strict_model", loader is None)):
        run171.validate_model(layers)
    widths = run172.widths_of(layers)
    if "unit_weight_columns" not in source_sets and "weights" not in source_sets:
        source_sets["unit_weight_columns"] = run172.single_weight_columns(layers)

    sets_artifact, writes, v = build_sets(source_sets, layers, widths)
    prior_random_sets = reference_sets if isinstance(reference_sets, dict) else _load_artifact(reference_sets)
    if "arms" not in prior_random_sets or "u100d_d2173" not in prior_random_sets["arms"]:
        raise ValueError("Part 173 sets_u100.pt must contain the u100d_d2173 replay arm")
    projections, projection_stats = output_projection_vectors(tok, model, source_sets, writes)
    for name in SET_NAMES:
        sets_artifact["metrics"][name].update(projections[name])
    artifact_path = Path(config.get("artifact_path", out / "sets_174.pt"))
    candidate_path = artifact_path.with_name(f".{artifact_path.name}.candidate")
    artifact_sha = _save_artifact(candidate_path, sets_artifact)
    source_sha = (_artifact_sha(source_path) if not isinstance(source_path, dict)
                  else run173.hash_value(source_sets))
    reference_sha = (_artifact_sha(reference_answers) if not isinstance(reference_answers, dict)
                     else run173.hash_value(reference_answers))
    reference_sets_sha = (_artifact_sha(reference_sets) if not isinstance(reference_sets, dict)
                          else run173.hash_value(reference_sets))
    run_key, key_inputs = _resume_key(artifact_sha, source_sha, reference_sha, reference_sets_sha, args)
    try:
        run173.verify_resume_key(out, run_key)
    except Exception:
        candidate_path.unlink(missing_ok=True)
        Path(str(candidate_path) + ".sha256").unlink(missing_ok=True)
        raise
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate_path, artifact_path)
    os.replace(Path(str(candidate_path) + ".sha256"), Path(str(artifact_path) + ".sha256"))
    Path(str(artifact_path) + ".sha256").write_text(
        f"{artifact_sha}  {artifact_path.name}\n", encoding="utf-8", newline="\n")

    fit_size = int(v.size)
    g8_rows = [{"set": name, "norm": projections[name]["write_norm"],
                "reference_norm": sets_artifact["source_n1_nominal_write_norm"],
                "cf": projections[name]["cf"], "fit_vector_size": fit_size}
               for name in SET_NAMES]
    gate8_result = gate8(g8_rows, fit_size)
    stage0 = {"part": 174, "stage": "stage0", "artifact": str(artifact_path),
              "artifact_sha256": artifact_sha, "part171_sets_sha256": source_sha,
              "part173_reference_sha256": reference_sha, "part173_sets_sha256": reference_sets_sha,
              "run_key": run_key,
              "run_key_inputs": key_inputs, "numpy_version": np.__version__,
              "set_names": list(SET_NAMES), "random_set_metrics": sets_artifact["metrics"],
              "projection_standard_deviations": projection_stats,
              "projection_tests": {"cf": projection_stats["cf_test"], "cr": projection_stats["cr_test"]},
              "gate8": gate8_result,
              "eligibility_screen": {
                  "gate": 8, "absolute_cf_max": DESIGN["gate8"]["absolute_cosine_max"],
                  "readings_conditional_on_all_60_draws_passing": bool(gate8_result["passed"]),
              },
              "gate8_rows": g8_rows}
    write_json(out / "stage0.json", stage0)
    if not gate8_result["passed"]:
        write_json(out / "gates.json", {"passed": False, "gate8": gate8_result})
        raise SystemExit(f"Part 174 Gate 8 failed before answers: {gate8_result['failures']}")

    sets = {**source_sets,
            "arms": {**source_sets["arms"], **prior_random_sets["arms"], **sets_artifact["arms"]},
            "metrics": {**source_sets.get("metrics", {}), **prior_random_sets.get("metrics", {}),
                        **sets_artifact["metrics"]}}
    gate3_result = gate3(model, tok, layers, args, sets, source_sets, widths)
    stage0["gate3"] = gate3_result
    write_json(out / "stage0.json", stage0)
    if not gate3_result["passed"]:
        write_json(out / "gates.json", {"passed": False, "gate3": gate3_result,
                                        "gate8": gate8_result})
        raise SystemExit("Part 174 Gate 3 failed before planned answers")

    replay_gate = _replay_stage(model, tok, layers, args, out, sets, source_sets, widths,
                                reference_answers, reference_sets, run_key)
    stage0["gate4"] = replay_gate
    write_json(out / "stage0.json", stage0)
    if not replay_gate["passed"]:
        write_json(out / "gates.json", {"passed": False, "gate3": gate3_result,
                                        "gate4": replay_gate, "gate8": gate8_result})
        raise SystemExit("Part 174 Gate 4 replay failed before planned answers")

    delivery_rows = measure_gate6(model, tok, layers, sets, source_sets, args.device, widths)
    gate6_result = gate6(delivery_rows)
    stage0["gate6"] = gate6_result
    stage0["gate6_rows"] = delivery_rows
    write_json(out / "stage0.json", stage0)
    if not gate6_result["passed"]:
        write_json(out / "gates.json", {"passed": False, "gate3": gate3_result,
                                        "gate4": replay_gate, "gate6": gate6_result,
                                        "gate8": gate8_result})
        raise SystemExit("Part 174 Gate 6 delivery failed before planned answers")

    gate7_result = gate7()
    stage0["gate7"] = gate7_result
    write_json(out / "stage0.json", stage0)
    if not gate7_result["passed"]:
        write_json(out / "gates.json", {"passed": False, "gate3": gate3_result,
                                        "gate4": replay_gate, "gate6": gate6_result,
                                        "gate7": gate7_result, "gate8": gate8_result})
        raise SystemExit("Part 174 Gate 7 prompt disjointness failed before planned answers")

    answer_path = out / "answers.jsonl"
    run172.repair_tail(answer_path)
    answers = read_jsonl(answer_path) if answer_path.is_file() else []
    done = {answer_key(row) for row in answers}
    if len(done) != len(answers):
        raise ValueError("answers.jsonl contains duplicate Part 174 row keys")
    pilot_path = out / "pilot.json"
    pilot = read_json(pilot_path, {"batches": 0, "answers": 0, "elapsed_seconds": 0.0})
    prompt_rows = prompts()
    for spec in planned_specs():
        if all((spec["arm"], prompt["prompt_id"]) in done for prompt in prompt_rows):
            # Re-check early gates on resume before allowing the next arm to run.
            if spec["arm"] == "none":
                stage0["gate2"] = run173.gate2(answers, prompt_rows)
                if not stage0["gate2"]["passed"]:
                    raise SystemExit("Part 174 Gate 2 failed after resumed none arm, before draws")
            elif spec["arm"] == "role":
                stage0["gate5"] = run173.gate5(answers, prompt_rows)
                if not stage0["gate5"]["passed"]:
                    raise SystemExit("Part 174 Gate 5 failed after resumed role arm, before draws")
            write_json(out / "stage0.json", stage0)
            continue
        started = time.perf_counter()
        generated = run_spec(model, tok, layers, spec, [p["question"] for p in prompt_rows],
                             args.device, args.max_new_tokens, sets, source_sets, widths)
        elapsed = time.perf_counter() - started
        if len(generated) != len(prompt_rows):
            raise RuntimeError(f"{spec['arm']} generated {len(generated)} answers, expected {len(prompt_rows)}")
        run172.append_answers(answer_path,
                              [_record(spec, prompt, answer, run_key)
                               for prompt, answer in zip(prompt_rows, generated)], done)
        answers = read_jsonl(answer_path)
        if spec["arm"] == "none":
            stage0["gate2"] = run173.gate2(answers, prompt_rows)
            write_json(out / "stage0.json", stage0)
            if not stage0["gate2"]["passed"]:
                write_json(out / "gates.json", {"passed": False, "gate2": stage0["gate2"]})
                raise SystemExit("Part 174 Gate 2 failed after none arm, before draws")
        elif spec["arm"] == "role":
            stage0["gate5"] = run173.gate5(answers, prompt_rows)
            write_json(out / "stage0.json", stage0)
            if not stage0["gate5"]["passed"]:
                write_json(out / "gates.json", {"passed": False, "gate2": stage0["gate2"],
                                                "gate5": stage0["gate5"]})
                raise SystemExit("Part 174 Gate 5 failed after role arm, before draws")
        if int(pilot.get("batches", 0)) < DESIGN["pilot"]["batches"]:
            pilot = run172.pilot_update(pilot_path, pilot, len(prompt_rows), elapsed)

    gate1_result = gate1(answers)
    gate2_result = stage0.get("gate2", run173.gate2(answers, prompt_rows))
    gate5_result = stage0.get("gate5", run173.gate5(answers, prompt_rows))
    gates = {"gate1": gate1_result, "gate2": gate2_result, "gate3": gate3_result,
             "gate4": replay_gate, "gate5": gate5_result, "gate6": gate6_result,
             "gate7": gate7_result, "gate8": gate8_result}
    gates["passed"] = all(value["passed"] for key, value in gates.items() if key != "passed")
    write_json(out / "gates.json", gates)
    write_json(out / "stage0.json", stage0)
    result = {"stage": "test", "planned_answers": len(planned_keys()),
              "answers": len([row for row in answers if not str(row.get("arm", "")).startswith("replay:")]),
              "replay_answers": sum(str(row.get("arm", "")).startswith("replay:") for row in answers),
              "gates": {key: value["passed"] for key, value in gates.items() if key != "passed"},
              "pilot": pilot}
    if not gates["passed"]:
        raise SystemExit("Part 174 gates failed: " + ", ".join(
            name for name, value in gates.items() if name != "passed" and not value["passed"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test",), default="test")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=run171.MODEL_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=Path("results/part174"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--part171-sets", type=Path, default=PART171_SETS)
    parser.add_argument("--part173-answers", type=Path, default=PART173_ANSWERS)
    parser.add_argument("--part173-sets", type=Path, default=PART173_SETS)
    args = parser.parse_args(argv)
    if args.batch_size != BATCH_SIZE:
        parser.error("registered generation batches hold one arm's 44 prompts")
    result = run(args)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
