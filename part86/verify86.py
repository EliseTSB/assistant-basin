#!/usr/bin/env python3
"""Run Part 86's resumable, per-neuron verification and score blind persona claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import secrets
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common86 as c
import blind86
import triage86

SAMPLES = 8
DIRECTIONS = 10
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
TOP_P = 0.95
ANSWERS_PER_NEURON = 24 * (DIRECTIONS + 1)


def _relative_close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    scale = max(abs(left), abs(right))
    return left == right if scale == 0 else abs(left - right) <= tolerance * scale


def _key_is_inside_items_folder(items_path: Path, key_path: Path) -> bool:
    try:
        key_path.resolve().relative_to(items_path.parent.resolve())
        return True
    except ValueError:
        return False


def seed_for(question_index: int, sample_index: int) -> int:
    if not 0 <= question_index < 3 or not 0 <= sample_index < SAMPLES:
        raise ValueError("verification seed index outside design range")
    return c.VERIFY_SEED + 100 * question_index + sample_index


def load_picked(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("picked file must be a JSON list of exactly 20 neurons")
    seen_neurons, seen_personas = set(), set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"layer", "neuron", "persona"}:
            raise ValueError(f"picked row {i} must contain layer, neuron, persona")
        if type(row["layer"]) is not int or not 10 <= row["layer"] <= 26:
            raise ValueError(f"picked row {i} has invalid layer")
        if type(row["neuron"]) is not int or not 0 <= row["neuron"] < 9216:
            raise ValueError(f"picked row {i} has invalid neuron")
        if not isinstance(row["persona"], str) or not row["persona"].strip():
            raise ValueError(f"picked row {i} has invalid persona")
        neuron_key = (row["layer"], row["neuron"])
        persona_key = row["persona"].strip().casefold()
        if neuron_key in seen_neurons:
            raise ValueError(f"duplicate picked neuron {neuron_key}")
        if persona_key in seen_personas:
            raise ValueError(f"duplicate picked persona {row['persona']!r}")
        seen_neurons.add(neuron_key)
        seen_personas.add(persona_key)
    return rows


def validate_picked_against_screen(rows, screen_dir, blind_key, blind_labels):
    changed_real = set()
    for row in triage86.read_screen(screen_dir):
        if row["kind"] == "neuron" and triage86.triage_answer(row.get("answer", "")) == "changed":
            changed_real.add((int(row["layer"]), int(row["id"])))
    for picked in rows:
        key = (picked["layer"], picked["neuron"])
        if key not in changed_real:
            raise ValueError(f"picked L{key[0]} N{key[1]} is absent or its screen answer was not triaged changed")
    key_rows = blind_key.get("rows")
    if not isinstance(key_rows, dict):
        raise ValueError("blind key has no screen-label mapping")
    label_rows = blind86._validate_labels(blind_labels, key_rows)
    by_neuron = {}
    for item_id, info in key_rows.items():
        if info.get("source") == "real" and info.get("triage_category", "changed") == "changed":
            neuron_key = (int(info["layer"]), int(info["item_index"]))
            by_neuron[neuron_key] = label_rows[item_id]
    for picked in rows:
        key = (picked["layer"], picked["neuron"])
        label = by_neuron.get(key)
        if label is None:
            raise ValueError(f"picked L{key[0]} N{key[1]} has no blind label")
        if label["coherent"] is not True or label["kind"] in ("still Qwen or an assistant", "unclear"):
            raise ValueError(f"picked L{key[0]} N{key[1]} blind label is not a coherent persona")


def _one_push(width, device, unit=None, value=None, vector=None):
    import torch
    row = torch.zeros((1, width), dtype=torch.float32, device=device)
    if vector is not None:
        row[0] = vector.to(device=device, dtype=torch.float32)
    else:
        row[0, int(unit)] = float(value)
    return row


def _screen_metadata(screen_dir, layer_no):
    path = Path(screen_dir) / f"layer_{layer_no}.meta.json"
    if not path.is_file():
        raise ValueError(f"screen metadata missing for L{layer_no}: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError(f"screen metadata for L{layer_no} is not complete")
    return meta


def _verify_config(selected, median, calibration_hash, actual_dtype, screen_meta, tokenizer, model):
    screen_config = screen_meta["config"]
    expected_generation_code = c.generation_code_sha256()
    expected_code = c.verification_code_sha256()
    expected_versions = c.dependency_versions()
    if screen_config.get("code_sha256") != expected_generation_code:
        raise ValueError("screen generation-path hashes differ from the current generation code")
    if screen_config.get("versions") != expected_versions:
        raise ValueError("verification torch/transformers versions differ from the screen provenance")
    if screen_config.get("actual_dtype") != actual_dtype or screen_config.get("dose") != c.DOSE:
        raise ValueError("verification dtype or dose differs from the screen provenance")
    if screen_config.get("model_id") != c.MODEL_ID or screen_config.get("revision") != c.REVISION:
        raise ValueError("verification model or revision differs from the screen provenance")
    if int(screen_config.get("layer", -1)) != int(selected["layer"]):
        raise ValueError("verification layer differs from the screen metadata")
    if screen_config.get("calibration_sha256") != calibration_hash:
        raise ValueError("verification calibration text differs from the screen provenance")
    screen_median = float(screen_meta.get("median", float("nan")))
    config_median = float(screen_config.get("median", float("nan")))
    if not _relative_close(config_median, screen_median):
        raise ValueError("screen median metadata is internally inconsistent")
    if not _relative_close(float(median), screen_median):
        raise ValueError(
            f"L{selected['layer']} residual median {median:.9g} differs from screen median {screen_median:.9g}"
        )
    return {
        "model_id": c.MODEL_ID,
        "revision": c.REVISION,
        "layer": selected["layer"],
        "neuron": selected["neuron"],
        "persona": selected["persona"].strip(),
        "calibration_sha256": calibration_hash,
        "median": screen_median,
        "dose": c.DOSE,
        "actual_dtype": actual_dtype,
        "versions": expected_versions,
        "code_sha256": expected_code,
        "generation_code_sha256": expected_generation_code,
        "questions": list(c.VERIFY_QUESTIONS),
        "samples": SAMPLES,
        "directions": DIRECTIONS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "mode": "generated_tokens",
        "generation_config": c.effective_generation_config(
            model, tokenizer, MAX_NEW_TOKENS, True, TEMPERATURE, TOP_P,
        ),
        "random_seed": c.RANDOM_VERIFY_SEED + selected["layer"] * 10000 + selected["neuron"],
    }


def _case_key(record):
    return (record["condition"], record.get("direction"), int(record["question_index"]), int(record["sample"]))


def _expected_case_keys():
    expected = {("neuron", None, qi, sample)
                for qi in range(3) for sample in range(SAMPLES)}
    expected.update(("random_direction", direction, qi, sample)
                    for direction in range(DIRECTIONS)
                    for qi in range(3) for sample in range(SAMPLES))
    return expected


def _load_neuron_state(path: Path, config: dict):
    if not path.exists():
        return {"status": "running", "config": config, "records": []}, False
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("config") != config:
        raise ValueError(f"resume provenance differs for {path}; use a new output directory")
    records = state.get("records")
    if not isinstance(records, list):
        raise ValueError(f"invalid answer list in {path}")
    keys = [_case_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate answer cases in {path}")
    expected_keys = _expected_case_keys()
    if any(key not in expected_keys for key in keys):
        raise ValueError(f"unexpected answer case in {path}")
    if state.get("status") == "complete":
        if len(records) != ANSWERS_PER_NEURON or set(keys) != expected_keys:
            raise ValueError(f"completed verification file {path} has {len(records)} answers, expected {ANSWERS_PER_NEURON}")
        return state, True
    return state, False


def _case_plan(selected, random_vectors):
    cases = []
    for qi, question in enumerate(c.VERIFY_QUESTIONS):
        for sample in range(SAMPLES):
            cases.append(("neuron", None, question, qi, sample, seed_for(qi, sample), None))
    for direction in range(DIRECTIONS):
        for qi, question in enumerate(c.VERIFY_QUESTIONS):
            for sample in range(SAMPLES):
                cases.append(("random_direction", direction, question, qi, sample,
                              seed_for(qi, sample), random_vectors[direction]))
    return cases


def verdict_865(survivors: int) -> dict:
    if type(survivors) is not int or not 0 <= survivors <= 20:
        raise ValueError("86.5 survivor count must be an integer from 0 to 20")
    return {"status": "HOLDS" if survivors >= 10 else "FAILS", "detail": f"{survivors} of 20 survive"}


def run(args):
    import torch
    picked = load_picked(args.picked)
    blind_key = json.loads(Path(args.blind_key).read_text(encoding="utf-8"))
    blind_labels = [json.loads(line) for line in Path(args.blind_labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_picked_against_screen(picked, args.screen_dir, blind_key, blind_labels)
    calibration = Path(args.calib).read_text(encoding="utf-8")
    calibration_hash = c.sha256_text(calibration)
    tokenizer, model = c.load_model(args.device)
    model.requires_grad_(False)
    model.eval()
    actual_dtype = c.require_bfloat16(model)
    for question in c.VERIFY_QUESTIONS:
        c.assert_rendered_prompt(tokenizer, question)
    layers = c.find_layers(model)
    medians = {}
    screen_metas = {}
    for layer_no in sorted({row["layer"] for row in picked}):
        layer = layers[layer_no]
        screen_metas[layer_no] = _screen_metadata(args.screen_dir, layer_no)
        medians[layer_no] = c.residual_median(model, tokenizer, layer, calibration, args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    completed_at_start = set()
    candidate_configs = {}
    for selected in picked:
        layer_no = selected["layer"]
        config = _verify_config(
            selected, medians[layer_no], calibration_hash, actual_dtype, screen_metas[layer_no], tokenizer, model,
        )
        candidate_configs[(layer_no, selected["neuron"])] = config
        path = out_dir / f"L{layer_no}_N{selected['neuron']}.json"
        _state, complete = _load_neuron_state(path, config)
        if complete:
            completed_at_start.add((layer_no, selected["neuron"]))
    remaining = [row for row in picked if (row["layer"], row["neuron"]) not in completed_at_start]
    if args.max_neurons is not None and args.max_neurons < 1:
        raise ValueError("--max-neurons must be positive")
    if args.max_neurons is not None:
        remaining = remaining[:args.max_neurons]
    elapsed_per_neuron = []
    for index, selected in enumerate(remaining):
        layer_no, neuron = selected["layer"], selected["neuron"]
        layer = layers[layer_no]
        down = layer.mlp.down_proj
        width = int(down.in_features)
        if width != 9216:
            raise ValueError(f"L{layer_no} down_proj width {width}; expected 9216")
        median = medians[layer_no]
        config = candidate_configs[(layer_no, neuron)]
        path = out_dir / f"L{layer_no}_N{neuron}.json"
        state, _complete = _load_neuron_state(path, config)
        records = state["records"]
        completed_cases = {_case_key(record) for record in records}
        neuron_delta = c.neuron_delta(down.weight, neuron, median)
        random_seed = c.RANDOM_VERIFY_SEED + layer_no * 10000 + neuron
        random_vectors = c.scaled_random_directions(down.weight.detach(), DIRECTIONS, random_seed, c.DOSE * median)
        cases = _case_plan(selected, random_vectors)
        started = time.monotonic()
        for condition, direction, question, qi, sample, seed, vector in cases:
            plan_record = {"condition": condition, "direction": direction, "question_index": qi, "sample": sample}
            if _case_key(plan_record) in completed_cases:
                continue
            if condition == "neuron":
                pushes = _one_push(width, down.weight.device, unit=neuron, value=neuron_delta)
            else:
                pushes = _one_push(width, down.weight.device, vector=vector)
            generated, hook_counts = c.generate_batch(
                model, tokenizer, layer, question, pushes, args.device, MAX_NEW_TOKENS,
                do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, seed=seed,
            )
            rec = {
                "layer": layer_no,
                "condition": condition,
                "neuron": neuron,
                "direction": direction,
                "persona": selected["persona"].strip(),
                "question": question,
                "question_index": qi,
                "sample": sample,
                "seed": seed,
                "answer": generated[0]["answer"],
                "token_count": generated[0]["token_count"],
                "median": median,
                "dose": c.DOSE,
                "mode": "generated_tokens",
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "hook_fire_counts": hook_counts,
            }
            records.append(rec)
            completed_cases.add(_case_key(rec))
            state.update({"status": "running", "config": config, "records": records})
            c.atomic_json(path, state)
        elapsed = time.monotonic() - started
        elapsed_per_neuron.append(elapsed)
        if len(records) != ANSWERS_PER_NEURON:
            raise RuntimeError(f"L{layer_no} N{neuron}: wrote {len(records)} answers, expected {ANSWERS_PER_NEURON}")
        state.update({"status": "complete", "config": config, "records": records, "elapsed_seconds": elapsed})
        c.atomic_json(path, state)
        projected = elapsed * len(picked)
        print(
            f"L{layer_no} N{neuron} complete in {elapsed:.1f}s; projected full verification "
            f"{projected / 3600:.2f}h for 20 neurons (6h task limit)",
            flush=True,
        )
    print(f"completed {len(elapsed_per_neuron)} new neurons; {len(completed_at_start)} already complete in {out_dir}")


def _read_verification_files(input_path):
    source = Path(input_path)
    if source.is_dir():
        files = sorted(source.glob("L*_N*.json"))
        if not files:
            raise ValueError(f"no per-neuron verification JSON files under {source}")
        rows = []
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                raise ValueError(f"verification file is incomplete: {path}")
            rows.extend(payload.get("records", []))
        return rows
    raise ValueError("verification input must be the per-neuron output directory")


def _write_reader_chunks(path: Path, items: list[dict], chunk_size: int = 500) -> None:
    if not 1 <= chunk_size <= 500:
        raise ValueError("reader chunks must contain between 1 and 500 items")
    path.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(items), chunk_size), 1):
        encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                          for row in items[start:start + chunk_size])
        (path / f"chunk-{index:04d}.jsonl").write_text(encoded, encoding="utf-8")


def prepare(input_path, items_path, key_path, force=False, chunk_size=500):
    items_path, key_path = Path(items_path), Path(key_path)
    if _key_is_inside_items_folder(items_path, key_path):
        raise ValueError("verification key must live outside the folder containing reader items")
    primary_chunks = items_path.parent / f"{items_path.stem}-reader1-chunks"
    second_chunks = items_path.parent / f"{items_path.stem}-reader2-chunks"
    if not force and (items_path.exists() or key_path.exists() or primary_chunks.exists() or second_chunks.exists()):
        raise FileExistsError("items or key already exists; pass --force to overwrite")
    if force:
        import shutil
        for path in (primary_chunks, second_chunks):
            if path.exists():
                shutil.rmtree(path)
    records = _read_verification_files(input_path)
    secrets.SystemRandom().shuffle(records)
    ids = set()
    items, mapping = [], {}
    for row in records:
        item_id = secrets.token_urlsafe(16)
        while item_id in ids:
            item_id = secrets.token_urlsafe(16)
        ids.add(item_id)
        items.append({"id": item_id, "persona": row["persona"], "text": row["answer"]})
        mapping[item_id] = {
            "layer": row["layer"], "condition": row["condition"], "neuron": row["neuron"],
            "direction": row["direction"], "question_index": row["question_index"],
            "sample": row["sample"], "seed": row["seed"], "persona": row["persona"],
        }
    stable_ids = sorted(mapping, key=lambda item_id: (
        int(mapping[item_id]["layer"]), int(mapping[item_id]["neuron"]),
        mapping[item_id]["condition"], -1 if mapping[item_id]["direction"] is None else int(mapping[item_id]["direction"]),
        int(mapping[item_id]["question_index"]), int(mapping[item_id]["sample"]),
    ))
    second_count = max(1, int(round(len(stable_ids) * 0.10))) if stable_ids else 0
    second_ids = set(random.Random(c.SECOND_READER_SEED).sample(stable_ids, second_count))
    items_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in items).encode("utf-8")
    key = {
        "items_sha256": hashlib.sha256(data).hexdigest(),
        "rows": mapping,
        "second_reader_ids": sorted(second_ids),
        "manifest": {
            "code_sha256": c.verification_code_sha256(),
            "items_sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(mapping),
        },
    }
    key["manifest"]["manifest_sha256"] = c.canonical_sha256(key["manifest"])
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    items_path.write_bytes(data)
    public_by_id = {row["id"]: row for row in items}
    _write_reader_chunks(primary_chunks, items, chunk_size)
    _write_reader_chunks(second_chunks, [public_by_id[item_id] for item_id in sorted(second_ids)], chunk_size)
    return len(items)


def _binary_kappa(first: dict, second: dict, item_ids) -> float:
    ids = list(item_ids)
    if not ids:
        raise ValueError("kappa needs at least one double-read item")
    left = [bool(first[item_id]) for item_id in ids]
    right = [bool(second[item_id]) for item_id in ids]
    observed = sum(a == b for a, b in zip(left, right)) / len(ids)
    p_left = sum(left) / len(ids)
    p_right = sum(right) / len(ids)
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    return (1.0 if observed == 1 else 0.0) if expected == 1 else (observed - expected) / (1 - expected)


def score(labels, key, items_path, labels2=None):
    digest = hashlib.sha256(Path(items_path).read_bytes()).hexdigest()
    if digest != key.get("items_sha256"):
        raise ValueError("items file SHA-256 does not match the key")
    mapping = key.get("rows", {})
    seen = {}
    for row in labels:
        if not isinstance(row, dict) or set(row) != {"id", "claims_persona"}:
            raise ValueError("each label must have exactly id and claims_persona")
        item_id = row["id"]
        if item_id not in mapping:
            raise ValueError(f"unknown label id {item_id!r}")
        if item_id in seen:
            raise ValueError(f"duplicate label for {item_id}")
        if type(row["claims_persona"]) is not bool:
            raise ValueError(f"claims_persona must be a JSON boolean for {item_id}")
        seen[item_id] = row["claims_persona"]
    if set(seen) != set(mapping):
        raise ValueError(f"missing labels for {len(set(mapping) - set(seen))} ids")

    neurons = {}
    for item_id, info in mapping.items():
        key_n = (int(info["layer"]), int(info["neuron"]))
        group = neurons.setdefault(key_n, {"persona": info["persona"], "neuron_claims": [], "directions": {}})
        claim = seen[item_id]
        if info["condition"] == "neuron":
            group["neuron_claims"].append((int(info["question_index"]), int(info["sample"]), claim))
        elif info["condition"] == "random_direction":
            direction = int(info["direction"])
            group["directions"].setdefault(direction, []).append((int(info["question_index"]), int(info["sample"]), claim))
        else:
            raise ValueError(f"invalid condition in key: {info['condition']!r}")

    per_neuron = []
    for (layer, neuron), group in sorted(neurons.items()):
        neuron_claims = group["neuron_claims"]
        directions = group["directions"]
        expected_samples = {(q, s) for q in range(3) for s in range(SAMPLES)}
        if len(neuron_claims) != 24 or {(q, s) for q, s, _ in neuron_claims} != expected_samples:
            raise ValueError(f"L{layer} N{neuron}: expected exactly one neuron label for each of 24 samples")
        if set(directions) != set(range(DIRECTIONS)):
            raise ValueError(f"L{layer} N{neuron}: expected ten random directions")
        direction_counts = {}
        for d, samples in directions.items():
            if len(samples) != 24 or {(q, s) for q, s, _ in samples} != expected_samples:
                raise ValueError(f"L{layer} N{neuron} D{d}: expected exactly one label for each of 24 samples")
            direction_counts[str(d)] = sum(int(claim) for _, _, claim in samples)
        neuron_count = sum(int(claim) for _, _, claim in neuron_claims)
        survives = neuron_count >= 12 and all(n <= 1 for n in direction_counts.values())
        per_neuron.append({
            "layer": layer, "neuron": neuron, "persona": group["persona"],
            "neuron_claims": neuron_count, "neuron_total": 24,
            "random_direction_claims": direction_counts, "survives": survives,
        })
    if len(per_neuron) != 20:
        raise ValueError(f"expected 20 neurons in verification key, got {len(per_neuron)}")
    survivors = sum(row["survives"] for row in per_neuron)
    result = {
        "survivors": survivors,
        "tested_neurons": len(per_neuron),
        "per_neuron": per_neuron,
    }
    verdict = verdict_865(survivors)
    if labels2 is None:
        result["reader_agreement"] = {"method": "Cohen's kappa", "items": 0, "kappa": None,
                                      "threshold": 0.6, "status": "NOT READ: second-reader labels are required"}
        verdict = {"status": "NOT READ", "detail": "second-reader agreement is required"}
    else:
        second_seen = {}
        for row in labels2:
            if not isinstance(row, dict) or set(row) != {"id", "claims_persona"}:
                raise ValueError("each second-reader label must have exactly id and claims_persona")
            item_id = row["id"]
            if item_id not in mapping or item_id in second_seen or type(row["claims_persona"]) is not bool:
                raise ValueError(f"invalid second-reader label for {item_id!r}")
            second_seen[item_id] = row["claims_persona"]
        second_ids = set(key.get("second_reader_ids", []))
        primary_ids = set(seen)
        if set(second_seen) not in (second_ids, primary_ids):
            raise ValueError("second-reader labels must cover the seed-8687 sample or the full item set")
        scope = "full second reading" if set(second_seen) == primary_ids else "seed-8687 10% sample"
        agreement_kappa = _binary_kappa(seen, second_seen, second_seen)
        result["reader_agreement"] = {
            "method": "Cohen's kappa", "items": len(second_seen), "scope": scope,
            "kappa": agreement_kappa, "threshold": 0.6,
        }
        if not c.kappa_reads(agreement_kappa):
            verdict = {
                "status": "NOT READ", "detail": f"Cohen's kappa {agreement_kappa:.4f} is below 0.6"
            }
    result["manifest"] = {
        "code_sha256": c.verification_code_sha256(),
        "items_sha256": digest,
        "key_sha256": c.canonical_sha256(key),
        "labels_sha256": c.canonical_sha256(labels),
        "labels2_sha256": c.canonical_sha256(labels2) if labels2 is not None else None,
    }
    result["manifest"]["manifest_sha256"] = c.canonical_sha256(result["manifest"])
    result["verdict_86.5"] = verdict
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="command", required=True)
    run_p = sp.add_parser("run", help="generate resumable per-neuron verification files")
    run_p.add_argument("--picked", required=True)
    run_p.add_argument("--screen-dir", required=True)
    run_p.add_argument("--blind-key", required=True, help="private key for the primary screen labels")
    run_p.add_argument("--blind-labels", required=True, help="primary screen labels used to validate each pick")
    run_p.add_argument("--calib", required=True)
    run_p.add_argument("--out", required=True, help="directory for one resumable JSON file per neuron")
    run_p.add_argument("--device", choices=("xpu", "cpu", "cuda"), default="xpu")
    run_p.add_argument("--max-neurons", type=int, default=None, help="maximum new neuron files in this task")
    prep = sp.add_parser("prepare", help="shuffle per-neuron answers for blind persona-claim reading")
    prep.add_argument("--input", required=True, help="directory written by run")
    prep.add_argument("--items", required=True)
    prep.add_argument("--key", required=True)
    prep.add_argument("--chunk-size", type=int, default=500)
    prep.add_argument("--force", action="store_true")
    score_p = sp.add_parser("score", help="apply the registered survival rule")
    score_p.add_argument("--items", required=True)
    score_p.add_argument("--key", required=True)
    score_p.add_argument("--labels", required=True)
    score_p.add_argument("--labels2", help="second-reader JSONL for the seed-8687 overlap sample or full reread")
    score_p.add_argument("--out", required=True)
    score_p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "prepare":
        count = prepare(args.input, args.items, args.key, args.force, args.chunk_size)
        print(f"{count} verification answers blinded; reader chunks are at most {args.chunk_size}")
    else:
        if Path(args.out).exists() and not args.force:
            raise FileExistsError(f"{args.out} exists; pass --force to overwrite")
        key = json.loads(Path(args.key).read_text(encoding="utf-8"))
        labels = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
        labels2 = ([json.loads(line) for line in Path(args.labels2).read_text(encoding="utf-8").splitlines() if line.strip()]
                   if args.labels2 else None)
        result = score(labels, key, args.items, labels2)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        agreement = result["reader_agreement"]
        if agreement["kappa"] is None:
            print("reader agreement NOT READ: second-reader labels are required", flush=True)
        else:
            print(f"reader agreement Cohen's kappa={agreement['kappa']:.4f} on {agreement['items']} items", flush=True)
        print(f"86.5 {result['verdict_86.5']['status']}: {result['verdict_86.5']['detail']}")


if __name__ == "__main__":
    main()
