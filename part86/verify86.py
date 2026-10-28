#!/usr/bin/env python3
"""Run Part 86's 20-neuron verification and score blind persona claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common86 as c

SAMPLES = 8
DIRECTIONS = 10
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
TOP_P = 0.95


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


def _one_push(width, device, unit=None, value=None, vector=None):
    import torch
    row = torch.zeros((1, width), dtype=torch.float32, device=device)
    if vector is not None:
        row[0] = vector.to(device=device, dtype=torch.float32)
    else:
        row[0, int(unit)] = float(value)
    return row


def run(args):
    import torch
    picked = load_picked(args.picked)
    calibration = Path(args.calib).read_text(encoding="utf-8")
    tokenizer, model = c.load_model(args.device)
    layers = c.find_layers(model)
    medians = {}
    for layer_no in sorted({row["layer"] for row in picked}):
        layer = layers[layer_no]
        medians[layer_no] = c.residual_median(model, tokenizer, layer, calibration, args.device)
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists; pass --force to overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for selected in picked:
            layer_no, neuron = selected["layer"], selected["neuron"]
            layer = layers[layer_no]
            down = layer.mlp.down_proj
            width = int(down.in_features)
            if width != 9216:
                raise ValueError(f"L{layer_no} down_proj width {width}; expected 9216")
            median = medians[layer_no]
            col_norm = float(torch.linalg.vector_norm(down.weight[:, neuron].detach().float()).item())
            if col_norm <= 0:
                raise ValueError(f"L{layer_no} N{neuron} has zero down_proj column norm")
            neuron_delta = c.DOSE * median / col_norm
            cases = []
            for qi, question in enumerate(c.VERIFY_QUESTIONS):
                for sample in range(SAMPLES):
                    cases.append(("neuron", None, question, qi, sample, seed_for(qi, sample), None))
            random_seed = c.RANDOM_VERIFY_SEED + layer_no * 10000 + neuron
            random_vectors = c.scaled_random_directions(down.weight.detach(), DIRECTIONS, random_seed, c.DOSE * median)
            for direction in range(DIRECTIONS):
                for qi, question in enumerate(c.VERIFY_QUESTIONS):
                    for sample in range(SAMPLES):
                        cases.append(("random_direction", direction, question, qi, sample,
                                      seed_for(qi, sample), random_vectors[direction]))
            for condition, direction, question, qi, sample, seed, vector in cases:
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
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                f.flush()
                total += 1
    print(f"wrote {total} answers to {out_path}")


def prepare(input_path, items_path, key_path, force=False):
    items_path, key_path = Path(items_path), Path(key_path)
    if not force and (items_path.exists() or key_path.exists()):
        raise FileExistsError("items or key already exists; pass --force to overwrite")
    records = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
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
            "layer": row["layer"],
            "condition": row["condition"],
            "neuron": row["neuron"],
            "direction": row["direction"],
            "question_index": row["question_index"],
            "sample": row["sample"],
            "seed": row["seed"],
            "persona": row["persona"],
        }
    items_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in items).encode("utf-8")
    items_path.write_bytes(data)
    key = {"items_sha256": hashlib.sha256(data).hexdigest(), "rows": mapping}
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(items)


def score(labels, key, items_path):
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
            "layer": layer,
            "neuron": neuron,
            "persona": group["persona"],
            "neuron_claims": neuron_count,
            "neuron_total": 24,
            "random_direction_claims": direction_counts,
            "survives": survives,
        })
    if len(per_neuron) != 20:
        raise ValueError(f"expected 20 neurons in verification key, got {len(per_neuron)}")
    survivors = sum(row["survives"] for row in per_neuron)
    verdict = "HOLDS" if survivors >= 10 else "FAILS"
    return {
        "survivors": survivors,
        "tested_neurons": len(per_neuron),
        "verdict_86.5": {"status": verdict, "detail": f"{survivors} of 20 survive"},
        "per_neuron": per_neuron,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="command", required=True)
    run_p = sp.add_parser("run", help="generate the 5,280 verification answers")
    run_p.add_argument("--picked", required=True)
    run_p.add_argument("--calib", required=True)
    run_p.add_argument("--out", required=True)
    run_p.add_argument("--device", choices=("xpu", "cpu", "cuda"), default="xpu")
    run_p.add_argument("--force", action="store_true")
    prep = sp.add_parser("prepare", help="shuffle verification answers for blind reading")
    prep.add_argument("--input", required=True)
    prep.add_argument("--items", required=True)
    prep.add_argument("--key", required=True)
    prep.add_argument("--force", action="store_true")
    score_p = sp.add_parser("score", help="apply the registered survival rule")
    score_p.add_argument("--items", required=True)
    score_p.add_argument("--key", required=True)
    score_p.add_argument("--labels", required=True)
    score_p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.command == "run":
        if Path(args.out).exists() and not args.force:
            raise FileExistsError(f"{args.out} exists; pass --force to overwrite")
        run(args)
    elif args.command == "prepare":
        n = prepare(args.input, args.items, args.key, args.force)
        print(f"{n} verification answers blinded")
    else:
        key = json.loads(Path(args.key).read_text(encoding="utf-8"))
        labels = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
        result = score(labels, key, args.items)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"86.5 {result['verdict_86.5']['status']}: {result['verdict_86.5']['detail']}")


if __name__ == "__main__":
    main()
