#!/usr/bin/env python3
"""Part 176: one-process, resumable five-rung census over twenty fresh f100d draws."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86"), str(ROOT / "part171"),
              str(ROOT / "part172"), str(ROOT / "part173"), str(ROOT / "part174"),
              str(ROOT / "part175")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import common86
import run171
import run172
import run173
import run174
import run175
import score173
from part171.prompts171 import TARGETS
from part173.prompts173 import NEW44B

SEEDS = tuple(range(176101, 176121))
DOSES = (0.20, 0.225, 0.25, 0.275, 0.30)
REPLAYS = (("f100d_d174104", 0.225), ("f100d_d174115", -0.225))
PART174_STAGE0 = Path("results/part174/stage0.json")
PART174_SET_HASH = Path("results/part174/sets_174.pt.sha256")
PART174_ANSWERS = Path("results/part174/answers.jsonl")
PART175_ANSWERS = Path("results/part175/answers.jsonl")
PART171_ANSWERS = Path("results/part171/answers.jsonl")
PART171_SUMMARY = Path("results/part171/sets.json")
MODEL_DEFAULT = run171.MODEL_DEFAULT
BATCH_SIZE = 44
MAX_NEW_TOKENS = 80


def prompts():
    return [{"prompt_id": f"new44b:{index}", "question_index": index, "question": question}
            for index, question in enumerate(NEW44B)]


def build_fresh_sets(layers, widths, source_sets=None, fit_vector=None, seeds=None,
                     reference_norm=None):
    """Use Part 174's f100d stream and globally match each new draw to n1's written length."""
    seeds = SEEDS if seeds is None else tuple(map(int, seeds))
    raw = run174.draw_f100d_raw(seeds=seeds, widths=widths)
    raw_vectors = {name: run173._rep_vectors(reps, widths) for name, reps in raw.items()}
    writes = run173._nominal_writes(layers, raw_vectors)
    if reference_norm is None:
        if source_sets is None:
            raise ValueError("Part 171 n1 written length is required to scale fresh draws")
        n1_pushes = run172.set_pushes(source_sets, "n1", 1.0, widths, "cpu")
        n1_vectors = {int(layer): push.detach().to(device="cpu", dtype=__import__("torch").float32)
                      .numpy().copy() for layer, push in n1_pushes.items()}
        n1_write = run173._nominal_writes(layers, {"n1": n1_vectors})["n1"]
        reference_norm = float(np.linalg.norm(n1_write))
    else:
        reference_norm = float(reference_norm)
    if not math.isfinite(reference_norm) or reference_norm <= 0:
        raise ValueError("Part 171 n1 has a zero or non-finite written length")
    scaled, scales = {}, {}
    for name, reps in raw.items():
        raw_norm = float(np.linalg.norm(writes[name]))
        if not math.isfinite(raw_norm) or raw_norm <= 0:
            raise ValueError(f"{name} has a zero or non-finite write; no draw is replaced")
        factor = reference_norm / raw_norm
        scales[name] = factor
        scaled[name] = {int(layer): run174._scaled_compact_rep(rep, factor)
                        for layer, rep in reps.items()}
    scaled_vectors = {name: run173._rep_vectors(reps, widths) for name, reps in scaled.items()}
    scaled_writes = run173._nominal_writes(layers, scaled_vectors)
    fit = np.asarray(fit_vector.detach().to(device="cpu", dtype=__import__("torch").float32).numpy()
                     if hasattr(fit_vector, "detach") else fit_vector, dtype=np.float64).reshape(-1)
    fit_norm = float(np.linalg.norm(fit))
    if not math.isfinite(fit_norm) or fit_norm <= 0:
        raise ValueError("Part 171 fit vector has a zero or non-finite norm")
    metrics = {}
    for name, write in scaled_writes.items():
        write = np.asarray(write, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(write))
        metrics[name] = {"seed": int(name.rsplit("d", 1)[1]), "global_scale": float(scales[name]),
                         "write_norm": norm, "nominal_write_norm": norm,
                         "length_relative_error": abs(norm / reference_norm - 1.0),
                         "reference_norm": reference_norm,
                         "cf": float(write @ fit / (norm * fit_norm))}
    return scaled, metrics, reference_norm


def gate8_fresh(metrics, seeds=None):
    seeds = SEEDS if seeds is None else tuple(map(int, seeds))
    expected = {f"f100d_d{seed}" for seed in seeds}
    failures = []
    if set(metrics) != expected:
        failures.append({"kind": "draw_coverage", "expected": sorted(expected),
                         "observed": sorted(metrics)})
    for name in sorted(expected):
        row = metrics.get(name, {})
        try:
            rel = float(row["length_relative_error"])
            cf = float(row["cf"])
            valid = math.isfinite(rel) and math.isfinite(cf)
        except (KeyError, TypeError, ValueError):
            rel = cf = float("nan")
            valid = False
        if not valid:
            failures.append({"kind": "nonfinite", "set": name})
            continue
        if rel > float(run174.DESIGN["gate8"]["length_relative_tolerance"]) + 1e-12:
            failures.append({"kind": "length", "set": name, "relative_error": rel})
        if abs(cf) > float(run174.DESIGN["gate8"]["absolute_cosine_max"]):
            failures.append({"kind": "cf", "set": name, "absolute_cf": abs(cf)})
    return {"passed": not failures, "draws": len(metrics), "expected_draws": len(expected),
            "failures": failures,
            "absolute_cf_max": run174.DESIGN["gate8"]["absolute_cosine_max"],
            "length_relative_tolerance": run174.DESIGN["gate8"]["length_relative_tolerance"]}


def gate8_replays(stage0, artifact, layers, widths, fit_vector, tok, model):
    """Recompute metrics from the rebuilt tensors before comparing the two frozen replay draws."""
    try:
        metrics = run175._recomputed_part174_metrics(stage0, artifact, layers, widths, fit_vector,
                                                     tok=tok, model=model)
    except Exception as exc:
        return {"passed": False, "tolerance": 1e-6, "error": repr(exc), "draws": []}
    source = stage0.get("random_set_metrics", {})
    rows, failures = [], []
    for set_name, _dose in REPLAYS:
        actual, expected = metrics.get(set_name, {}), source.get(set_name, {})
        fields = {}
        row_passed = True
        for field in ("global_scale", "nominal_write_norm", "cf", "cr"):
            try:
                wanted, got = float(expected[field]), float(actual[field])
                error = abs(wanted - got)
                passed = math.isfinite(wanted) and math.isfinite(got) and error <= 1e-6
            except (KeyError, TypeError, ValueError):
                wanted = got = error = None
                passed = False
            fields[field] = {"expected": wanted, "rebuilt": got, "absolute_error": error,
                             "passed": bool(passed)}
            if not passed:
                row_passed = False
                failures.append({"set": set_name, "field": field, "expected": wanted,
                                 "rebuilt": got, "absolute_error": error})
        rows.append({"set": set_name, "passed": bool(row_passed), "fields": fields})
    return {"passed": not failures and len(rows) == len(REPLAYS), "tolerance": 1e-6,
            "draws": rows, "failures": failures,
            "source": "metrics recomputed from independently rebuilt Part 174 draw tensors"}


def _append_jsonl(path, records, done):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            key = run174.answer_key(record)
            if key in done:
                continue
            handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
            handle.flush()
            done.add(key)


def _read_jsonl(path):
    return run174.read_jsonl(path) if Path(path).is_file() else []


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _fingerprint_value(value):
    """Return stable shape/content summaries for the arrays and tensors in a run plan."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        array = value.detach().to(device="cpu").contiguous().numpy()
        return {"tensor_dtype": str(array.dtype), "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"array_dtype": str(array.dtype), "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): _fingerprint_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__qualname__, "repr": repr(value)}


def _input_fingerprint(args, prompt_rows, seeds, doses, sets, reference_answers):
    replay_arms = {run175.dose_arm(name, dose) for name, dose in REPLAYS}
    replay_reference = [{"arm": row.get("arm"), "prompt_id": row.get("prompt_id"),
                         "token_ids": row.get("token_ids")}
                        for row in reference_answers if row.get("arm") in replay_arms]
    material = {"part": 176, "model": str(args.model),
                "model_revision": str(common86.REVISION),
                "batch_size": int(args.batch_size),
                "max_new_tokens": int(args.max_new_tokens),
                "prompts": _fingerprint_value(prompt_rows),
                "seeds": list(map(int, seeds)), "doses": list(map(float, doses)),
                "replays": _fingerprint_value(REPLAYS),
                "push_sets": _fingerprint_value(sets.get("arms", {})),
                "set_metrics": _fingerprint_value(sets.get("metrics", {})),
                "replay_reference": _fingerprint_value(replay_reference)}
    payload = json.dumps(material, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_answer_rows(rows):
    return all(isinstance(row.get("answer"), str) and isinstance(row.get("token_ids"), list) and
               bool(row["token_ids"]) and all(int(value) >= 0 for value in row["token_ids"])
               for row in rows)


def gate1(answers, prompt_rows, seeds=None, doses=None):
    seeds = SEEDS if seeds is None else tuple(map(int, seeds))
    doses = DOSES if doses is None else tuple(map(float, doses))
    expected = {(arm, str(prompt["prompt_id"]))
                for arm in ("none", "role") for prompt in prompt_rows}
    for name, dose in REPLAYS:
        expected.update((f"replay:{run175.dose_arm(name, dose)}", str(prompt["prompt_id"]))
                        for prompt in prompt_rows)
    for seed in seeds:
        name = f"f100d_d{seed}"
        for magnitude in doses:
            for dose in (magnitude, -magnitude):
                expected.update((run175.dose_arm(name, dose), str(prompt["prompt_id"]))
                                for prompt in prompt_rows)
    keys = [(str(row.get("arm")), str(row.get("prompt_id"))) for row in answers]
    observed = set(keys)
    try:
        valid = _valid_answer_rows(answers)
    except (TypeError, ValueError):
        valid = False
    return {"passed": bool(len(keys) == len(expected) and len(observed) == len(keys) and
                           observed == expected and valid),
            "planned": len(expected), "answers": len(keys), "unique_keys": len(observed),
            "missing": len(expected - observed), "unexpected": len(observed - expected),
            "valid_answer_and_token_ids": bool(valid)}


def gate4_replays(replay_rows, reference_rows, prompt_rows, replays=None, minimum=43):
    """Compare replay rows without collapsing duplicates behind prompt-id dictionaries."""
    replays = REPLAYS if replays is None else tuple(replays)
    replay_rows, reference_rows = list(replay_rows), list(reference_rows)
    expected_ids = [str(row["prompt_id"]) for row in prompt_rows]
    expected_set = set(expected_ids)
    prompt_duplicates = sorted({pid for pid in expected_ids if expected_ids.count(pid) > 1})
    per_arm, all_passed = {}, not prompt_duplicates
    for set_name, dose in replays:
        arm = run175.dose_arm(set_name, dose)
        replay_arm = f"replay:{arm}"
        references = [row for row in reference_rows if row.get("arm") == arm]
        actual = [row for row in replay_rows if row.get("arm") == replay_arm]
        ref_by_prompt, actual_by_prompt = {}, {}
        ref_dupes, actual_dupes = [], []
        for rows, indexed, duplicates in ((references, ref_by_prompt, ref_dupes),
                                          (actual, actual_by_prompt, actual_dupes)):
            for row in rows:
                pid = str(row.get("prompt_id"))
                if pid in indexed:
                    duplicates.append(pid)
                else:
                    indexed[pid] = row
        matches = sum(pid in ref_by_prompt and
                      isinstance(row.get("token_ids"), list) and
                      isinstance(ref_by_prompt[pid].get("token_ids"), list) and
                      row["token_ids"] == ref_by_prompt[pid]["token_ids"]
                      for pid, row in actual_by_prompt.items())
        missing = sorted(expected_set - set(actual_by_prompt))
        unexpected = sorted(set(actual_by_prompt) - expected_set)
        missing_reference = sorted(expected_set - set(ref_by_prompt))
        unexpected_reference = sorted(set(ref_by_prompt) - expected_set)
        passed = (len(references) == len(expected_ids) and len(actual) == len(expected_ids) and
                  not ref_dupes and not actual_dupes and not missing and not unexpected and
                  not missing_reference and not unexpected_reference and
                  matches >= min(int(minimum), len(expected_ids)))
        per_arm[arm] = {"replay_arm": replay_arm, "rows": len(actual),
                        "reference_rows": len(references), "matching": int(matches),
                        "minimum_matching": min(int(minimum), len(expected_ids)),
                        "duplicate_prompts": actual_dupes,
                        "duplicate_reference_prompts": ref_dupes,
                        "missing_prompts": missing, "unexpected_prompts": unexpected,
                        "missing_reference_prompts": missing_reference,
                        "unexpected_reference_prompts": unexpected_reference,
                        "passed": bool(passed)}
        all_passed = all_passed and passed
    return {"passed": bool(all_passed), "per_arm": per_arm,
            "prompt_id_duplicates": prompt_duplicates,
            "comparison": "Part 175 token_ids at identical prompt batch shape"}


def _run_batch(model, tok, layers, spec, prompt_rows, args, sets, config):
    runner = config.get("run_spec_fn", run174.run_spec)
    generated = runner(model, tok, layers, spec, [row["question"] for row in prompt_rows],
                       args.device, args.max_new_tokens, sets, {}, run172.widths_of(layers))
    if len(generated) != len(prompt_rows):
        raise RuntimeError(f"{spec['arm']} generated {len(generated)} answers, expected {len(prompt_rows)}")
    maker = run174._replay_record if spec.get("replay") else run175._record
    base_spec = dict(spec)
    base_spec.pop("replay", None)
    records = [maker(base_spec, prompt, answer) for prompt, answer in zip(prompt_rows, generated)]
    input_hash = config.get("input_hash")
    if input_hash is not None:
        for row in records:
            row["input_hash"] = str(input_hash)
    return records


def _ensure_model_controls(model, tok, layers, args, sets, prompt_rows, path, done, config):
    generate = config.get("generate_fn", run171.generate_batch)
    answers = _read_jsonl(path)
    gates = {}
    for name, role in (("none", None), ("role", TARGETS["google"])):
        if all((name, str(row["prompt_id"])) in done for row in prompt_rows):
            continue
        generated = generate(model, tok, layers, [row["question"] for row in prompt_rows],
                             [role] * len(prompt_rows) if role else None, args.device,
                             args.max_new_tokens, seed=172)
        records = [run175._record({"arm": name}, prompt, answer)
                   for prompt, answer in zip(prompt_rows, generated)]
        if config.get("input_hash") is not None:
            for row in records:
                row["input_hash"] = str(config["input_hash"])
        _append_jsonl(path, records, done)
        answers = _read_jsonl(path)
        if name == "none":
            gates["gate2"] = (config["gate2"] if "gate2" in config else
                               run173.gate2(answers, prompt_rows))
            if not gates["gate2"].get("passed"):
                return answers, gates
        else:
            gates["gate5"] = (config["gate5"] if "gate5" in config else
                               run173.gate5(answers, prompt_rows))
            if not gates["gate5"].get("passed"):
                return answers, gates
    if "gate2" not in gates:
        gates["gate2"] = (config["gate2"] if "gate2" in config else run173.gate2(answers, prompt_rows))
    if "gate5" not in gates:
        gates["gate5"] = (config["gate5"] if "gate5" in config else run173.gate5(answers, prompt_rows))
    return answers, gates


def measure_gate6(model, tok, layers, sets, device, widths, seeds=None, doses=None):
    seeds = SEEDS if seeds is None else tuple(map(int, seeds))
    doses = DOSES if doses is None else tuple(map(float, doses))
    signed = tuple(value for magnitude in doses for value in (magnitude, -magnitude))
    previous_names, previous_design = run174.SET_NAMES, run174.DESIGN
    try:
        run174.SET_NAMES = tuple(f"f100d_d{seed}" for seed in seeds)
        run174.DESIGN = {**previous_design, "signs": signed}
        rows = run174.measure_gate6(model, tok, layers, sets, {}, device, widths)
        gate = run174.gate6(rows)
    finally:
        run174.SET_NAMES, run174.DESIGN = previous_names, previous_design
    gate["doses"] = list(doses)
    gate["draws"] = len(seeds)
    return gate, rows


def run(args, loader=None, config=None):
    """Execute all arms in one model process and append completed answers for resumability."""
    config = dict(config or {})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pilot_path = out / "pilot.json"
    prior_pilot = run175.read_json(pilot_path, {}) or {}
    retry_pilot_timing = (prior_pilot.get("status") == "stopped" and
                          "timing is incomplete" in str(prior_pilot.get("reason", "")))
    if prior_pilot.get("status") == "stopped" and not retry_pilot_timing:
        return {"stage": "pilot_stopped", "answers": len(_read_jsonl(out / "answers.jsonl")),
                "pilot_answers_per_second": prior_pilot.get("answers_per_second"),
                "gates": run175.read_json(out / "gates.json", {}) or {}}
    prompt_rows = list(config.get("prompt_rows", prompts()))
    seeds = tuple(map(int, config.get("fresh_seeds", SEEDS)))
    doses = tuple(map(float, config.get("doses", DOSES)))
    stage0 = config.get("part174_stage0")
    if stage0 is None:
        stage0 = run175.read_json(getattr(args, "part174_stage0", PART174_STAGE0))
    reference_answers = config.get("part175_answers")
    if reference_answers is None:
        reference_answers = _read_jsonl(getattr(args, "part175_answers", PART175_ANSWERS))

    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common86.REVISION, args.device)
    else:
        tok, model = loader(args.device)
    model.requires_grad_(False)
    model.eval()
    layers = config.get("layers") or common86.find_layers(model)
    if bool(config.get("strict_model", loader is None)):
        run171.validate_model(layers)
    widths = config.get("widths") or run172.widths_of(layers)
    fit_vector = config.get("fit_vector")
    if fit_vector is None:
        fit_vector, fit_diagnostics = run175._fit_vector(model, tok, layers, args, config)
    else:
        fit_diagnostics = config.get("fit_vector_diagnostics", {"source": "fixture"})

    if "replay_sets" in config:
        replay_artifact = {"arms": config["replay_sets"]}
        replay_metrics = config.get("replay_metrics", {})
        replay_g8 = config.get("gate8_replay", {"passed": True, "fixture": True})
    else:
        replay_artifact = run175.rebuild_part174_sets(stage0, widths=widths)
        replay_metrics = None
        replay_g8 = gate8_replays(stage0, replay_artifact, layers, widths, fit_vector, tok, model)

    if "fresh_sets" in config:
        fresh_sets = config["fresh_sets"]
        fresh_metrics = config.get("fresh_metrics", {})
        reference_norm = config.get("reference_norm", 1.0)
    else:
        fresh_sets, fresh_metrics, reference_norm = build_fresh_sets(
            layers, widths, config.get("part171_sets"), fit_vector, seeds=seeds,
            reference_norm=run175._reference_norm(stage0))
    fresh_g8 = gate8_fresh(fresh_metrics, seeds)
    gate8 = {"passed": bool(fresh_g8["passed"] and replay_g8.get("passed")),
             "replay_rebuild": replay_g8, "fresh_draw_screens": fresh_g8,
             "fit_vector": fit_diagnostics, "reference_n1_written_norm": reference_norm}
    sets = {"arms": {**replay_artifact["arms"], **fresh_sets},
            "metrics": {**(replay_artifact.get("metrics", {})), **fresh_metrics}}
    input_hash = _input_fingerprint(args, prompt_rows, seeds, doses, sets, reference_answers)
    if prior_pilot and prior_pilot.get("input_hash") != input_hash:
        raise ValueError("Part 176 pilot fingerprint mismatch; refusing to resume different inputs")
    if not gate8["passed"]:
        gates = {"gate8": gate8, "input_hash": input_hash}
        _write_json(out / "gates.json", {**gates, "passed": False})
        raise SystemExit("Part 176 Gate 8 failed before generation")

    config["input_hash"] = input_hash
    gate7 = config.get("gate7", run174.gate7())
    gate3 = config.get("gate3")
    if gate3 is None:
        gate3 = run174.gate3(model, tok, layers, args, sets, {}, widths, prompt_rows)
    if not gate3.get("passed") or not gate7.get("passed"):
        _write_json(out / "gates.json", {"gate3": gate3, "gate7": gate7, "gate8": gate8,
                                         "input_hash": input_hash, "passed": False})
        raise SystemExit("Part 176 Gate 3, 7 or 8 failed before generation")

    residuals = config.get("residual_norms")
    if residuals is None:
        residuals = run175.measure_gate3_residual_norms(model, tok, layers, args.device)
    weights = config.get("layer_weights")
    if weights is None:
        weights = {int(layer): layers[int(layer)].mlp.down_proj.weight.detach().to(
            device="cpu", dtype=__import__("torch").float32).numpy().astype(np.float64)
                   for layer in run171.LAYERS}
    layer_norm_rows = []
    for seed in seeds:
        name = f"f100d_d{seed}"
        for magnitude in doses:
            for dose in (magnitude, -magnitude):
                layer_norm_rows.append(run175.measure_written_layer_norms(
                    sets, name, "f100d", dose, layers, widths, residuals, weights))
    layer_doc = run175.layer_norms_document(residuals, layer_norm_rows)
    layer_doc["part"] = 176
    _write_json(out / "layer_norms.json", layer_doc)

    answer_path = out / "answers.jsonl"
    run172.repair_tail(answer_path)
    answers = _read_jsonl(answer_path)
    if any(row.get("input_hash") != input_hash for row in answers):
        raise ValueError("Part 176 answers fingerprint mismatch; refusing to mix inputs in answers.jsonl")
    done = {run174.answer_key(row) for row in answers}
    if len(done) != len(answers):
        raise ValueError("Part 176 answers.jsonl contains duplicate row keys")

    answers, control_gates = _ensure_model_controls(model, tok, layers, args, sets,
                                                     prompt_rows, answer_path, done, config)
    if not control_gates.get("gate2", {}).get("passed") or not control_gates.get("gate5", {}).get("passed"):
        gates = {"gate2": control_gates.get("gate2"), "gate3": gate3, "gate5": control_gates.get("gate5"),
                 "gate7": gate7, "gate8": gate8, "input_hash": input_hash, "passed": False}
        _write_json(out / "gates.json", gates)
        raise SystemExit("Part 176 Gate 2 or 5 failed before dose ladder")

    gate6_result, gate6_rows = (config.get("gate6_result"), config.get("gate6_rows"))
    if gate6_result is None:
        gate6_result, gate6_rows = measure_gate6(model, tok, layers, sets, args.device, widths,
                                                 seeds=seeds, doses=doses)
    if not gate6_result.get("passed"):
        gates = {"gate2": control_gates["gate2"], "gate3": gate3, "gate5": control_gates["gate5"],
                 "gate6": gate6_result, "gate7": gate7, "gate8": gate8,
                 "gate6_rows": gate6_rows, "input_hash": input_hash, "passed": False}
        _write_json(out / "gates.json", gates)
        raise SystemExit("Part 176 Gate 6 delivery failed before dose ladder")

    # Replays are generated with exactly the same 44-question batch and token cap as the census.
    for set_name, dose in REPLAYS:
        base_arm = run175.dose_arm(set_name, dose)
        replay_arm = f"replay:{base_arm}"
        if all((replay_arm, str(row["prompt_id"])) in done for row in prompt_rows):
            continue
        spec = {"arm": base_arm, "set": set_name, "dose": float(dose), "replay": True}
        records = _run_batch(model, tok, layers, spec, prompt_rows, args, sets, config)
        _append_jsonl(answer_path, records, done)
    answers = _read_jsonl(answer_path)
    replay_gate = config.get("gate4")
    if replay_gate is None:
        replay_gate = gate4_replays(answers, reference_answers, prompt_rows,
                                   minimum=min(43, len(prompt_rows)))

    minimum_rate = float(config.get("pilot_minimum_rate", 1.0))
    pilot_elapsed = 0.0 if retry_pilot_timing else float(prior_pilot.get("elapsed_seconds", 0) or 0)
    pilot_answer_count = 0 if retry_pilot_timing else int(prior_pilot.get("answers", 0) or 0)
    pilot_target = 2 * 2 * len(doses) * len(prompt_rows)
    for seed_index, seed in enumerate(seeds):
        name = f"f100d_d{seed}"
        for magnitude in doses:
            for dose in (magnitude, -magnitude):
                arm = run175.dose_arm(name, dose)
                already_done = all((arm, str(prompt["prompt_id"])) in done for prompt in prompt_rows)
                if already_done and not (retry_pilot_timing and seed_index < 2):
                    continue
                spec = {"arm": arm, "set": name, "dose": float(dose)}
                batch_started = time.perf_counter()
                records = _run_batch(model, tok, layers, spec, prompt_rows, args, sets, config)
                _append_jsonl(answer_path, records, done)
                if seed_index < 2:
                    pilot_elapsed += max(time.perf_counter() - batch_started, 0.0)
                    pilot_answer_count += len(records)
                    _write_json(pilot_path, {"answers": pilot_answer_count,
                                              "elapsed_seconds": pilot_elapsed,
                                              "answers_per_second": (pilot_answer_count / pilot_elapsed
                                                                     if pilot_elapsed > 0 else None),
                                              "minimum_answers_per_second": minimum_rate,
                                              "input_hash": input_hash,
                                              "status": "running"})
            # At two complete draws only throughput is inspected; no content statistic is read.
        if seed_index == 1:
            if pilot_answer_count != pilot_target or pilot_elapsed <= 0:
                pilot = {"answers": pilot_answer_count, "expected_answers": pilot_target,
                         "elapsed_seconds": pilot_elapsed, "answers_per_second": None,
                         "minimum_answers_per_second": minimum_rate, "input_hash": input_hash,
                         "status": "stopped",
                         "reason": "pilot timing is incomplete; refusing to infer throughput from skipped rows"}
                _write_json(pilot_path, pilot)
                pilot_gates = {"gate1": {"passed": False, "reason": "pilot timing incomplete"},
                               "gate2": control_gates["gate2"], "gate3": gate3, "gate4": replay_gate,
                               "gate5": control_gates["gate5"], "gate6": gate6_result,
                               "gate6_rows": gate6_rows, "gate7": gate7, "gate8": gate8,
                               "pilot_status": "stopped", "input_hash": input_hash, "passed": False}
                _write_json(out / "gates.json", pilot_gates)
                return {"stage": "pilot_stopped", "answers": len(_read_jsonl(answer_path)),
                        "pilot_answers_per_second": None, "gates": pilot_gates}
            rate = pilot_answer_count / pilot_elapsed
            _write_json(pilot_path, {"answers": pilot_answer_count, "expected_answers": pilot_target,
                                     "elapsed_seconds": pilot_elapsed, "answers_per_second": rate,
                                     "minimum_answers_per_second": minimum_rate,
                                     "input_hash": input_hash,
                                     "status": "continue" if rate >= minimum_rate else "stopped"})
            if rate < minimum_rate:
                pilot_gates = {"gate1": {"passed": False, "reason": "pilot stopped before all planned answers"},
                               "gate2": control_gates["gate2"], "gate3": gate3, "gate4": replay_gate,
                               "gate5": control_gates["gate5"], "gate6": gate6_result,
                               "gate6_rows": gate6_rows, "gate7": gate7, "gate8": gate8,
                               "pilot_status": "stopped", "input_hash": input_hash, "passed": False}
                _write_json(out / "gates.json", pilot_gates)
                return {"stage": "pilot_stopped", "answers": len(_read_jsonl(answer_path)),
                        "pilot_answers_per_second": rate, "gates": pilot_gates}

    answers = _read_jsonl(answer_path)
    gate1_result = gate1(answers, prompt_rows, seeds, doses)
    gates = {"gate1": gate1_result, "gate2": control_gates["gate2"], "gate3": gate3,
             "gate4": replay_gate, "gate5": control_gates["gate5"], "gate6": gate6_result,
             "gate6_rows": gate6_rows, "gate7": gate7, "gate8": gate8,
             "fresh_seeds": list(seeds), "doses": list(doses), "input_hash": input_hash}
    gates["passed"] = all(bool(gates[f"gate{index}"].get("passed"))
                          for index in range(1, 9))
    _write_json(out / "gates.json", gates)
    layer_doc = run175.layer_norms_document(residuals, layer_norm_rows, status="complete")
    layer_doc["part"] = 176
    _write_json(out / "layer_norms.json", layer_doc)
    return {"stage": "complete" if gates["passed"] else "generation_gate_failure",
            "answers": len(answers), "planned_answers": len(gates.get("gate1", {}).get("planned", []))
            if isinstance(gates.get("gate1", {}).get("planned"), list) else gates["gate1"]["planned"],
            "pilot": run175.read_json(out / "pilot.json", {}), "gates": gates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=Path("results/part176"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--part171-answers", type=Path, default=PART171_ANSWERS)
    parser.add_argument("--part171-summary", type=Path, default=PART171_SUMMARY)
    parser.add_argument("--part171-sets", type=Path, default=run174.PART171_SETS)
    parser.add_argument("--part174-stage0", type=Path, default=PART174_STAGE0)
    parser.add_argument("--part174-set-hash", type=Path, default=PART174_SET_HASH)
    parser.add_argument("--part174-answers", type=Path, default=PART174_ANSWERS)
    parser.add_argument("--part175-answers", type=Path, default=PART175_ANSWERS)
    args = parser.parse_args(argv)
    if args.batch_size != BATCH_SIZE:
        parser.error("registered generation batches hold one arm's 44 prompts")
    result = run(args)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return 0 if result.get("stage") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
