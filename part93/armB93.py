#!/usr/bin/env python3
"""Shortlist Part 86 neurons, steer held-out answers, and score style moves."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import armA93
import classify93

FAMILIES = ("claude", "gpt", "deepseek", "qwen")
OUTSIDE = ("claude", "gpt", "deepseek")
HELDOUT_SEED = 93020
RANDOM_SEED = 93000
MAX_NEW_TOKENS = 150
NAME_QUESTION = "Who are you?"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_no}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_family_answers(gen_dir: Path) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for family in FAMILIES:
        path = gen_dir / f"{family}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing family generation file: {path}")
        for row in read_jsonl(path):
            prompt_id = str(row.get("prompt_id") or row.get("id") or "")
            if prompt_id:
                latest[(family, prompt_id)] = row
    output = []
    for (family, prompt_id), row in sorted(latest.items()):
        if row.get("error") or not isinstance(row.get("answer"), str):
            continue
        output.append({"family": family, "prompt_id": prompt_id, "answer": row["answer"]})
    if not output or {row["family"] for row in output} != set(FAMILIES):
        raise ValueError("successful generation data must include all four Part 93 families")
    return output


def fit_classifiers(rows: list[dict[str, str]]):
    return classify93.fit_all([row["answer"] for row in rows], [row["family"] for row in rows])


def predict_batched(models: dict[str, Any], texts: list[str], classifier: str = "a",
                    batch_size: int = 128) -> tuple[list[str], np.ndarray]:
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    names: list[str] | None = None
    chunks = []
    for begin in range(0, len(texts), batch_size):
        result = classify93.predict_proba(models, texts[begin:begin + batch_size],
                                          classifier=classifier, masked=True)
        if names is None:
            names = result["class_names"]
        elif names != result["class_names"]:
            raise RuntimeError("classifier class order changed during prediction")
        chunks.append(np.asarray(result["probabilities"], dtype=np.float32))
    if not chunks or names is None:
        raise ValueError("no texts to score")
    return names, np.concatenate(chunks, axis=0)


def rank_layer_candidates(layer: int, neuron_ids: list[int], neuron_scores: np.ndarray,
                          random_scores: np.ndarray, class_names: list[str],
                          threshold: float = 3.0) -> list[dict[str, Any]]:
    """Rank each neuron by its strongest probability z-score over layer controls."""
    neuron_scores = np.asarray(neuron_scores, dtype=np.float64)
    random_scores = np.asarray(random_scores, dtype=np.float64)
    if neuron_scores.ndim != 2 or random_scores.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")
    if neuron_scores.shape != (len(neuron_ids), len(class_names)):
        raise ValueError("neuron score shape does not match ids/classes")
    if random_scores.shape[1] != len(class_names) or random_scores.shape[0] < 2:
        raise ValueError("at least two random direction scores are required")
    target_columns = [class_names.index(name) for name in OUTSIDE if name in class_names]
    if len(target_columns) != len(OUTSIDE):
        raise ValueError("classifier is missing an outside family")
    means = random_scores.mean(axis=0)
    deviations = random_scores.std(axis=0, ddof=1)
    z_scores = np.zeros((len(neuron_ids), len(class_names)), dtype=np.float64)
    usable = deviations > 1e-12
    z_scores[:, usable] = (neuron_scores[:, usable] - means[usable]) / deviations[usable]
    output = []
    for row_no, neuron_id in enumerate(neuron_ids):
        family_column = max(target_columns, key=lambda index: z_scores[row_no, index])
        z_value = float(z_scores[row_no, family_column])
        if z_value >= threshold:
            output.append({
                "layer": int(layer),
                "id": int(neuron_id),
                "target_family": class_names[family_column],
                "move_z": z_value,
                "family_z": {class_names[index]: float(z_scores[row_no, index])
                             for index in target_columns},
            })
    output.sort(key=lambda row: (-row["move_z"], row["layer"], row["id"]))
    return output


def layer_files(screen_dir: Path) -> list[Path]:
    files = sorted(screen_dir.glob("layer_*.jsonl.gz"))
    if not files:
        files = sorted(screen_dir.glob("layer_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no Part 86 layer_*.jsonl.gz files found in {screen_dir}")
    return files


def shortlist(gen_dir: Path, screen_dir: Path, out_path: Path,
              limit: int = 40, threshold: float = 3.0, prediction_batch_size: int = 128) -> dict:
    if limit < 1:
        raise ValueError("shortlist limit must be positive")
    training_rows = read_family_answers(gen_dir)
    models = fit_classifiers(training_rows)
    class_names = list(models["class_names"])
    all_candidates = []
    scored_rows = 0
    for path in layer_files(screen_dir):
        rows = read_jsonl(path)
        layer_match = re.search(r"layer_(\d+)", path.name)
        if not layer_match:
            raise ValueError(f"cannot read layer number from {path.name}")
        layer_no = int(layer_match.group(1))
        neurons = [row for row in rows if row.get("kind") == "neuron" and isinstance(row.get("answer"), str)]
        randoms = [row for row in rows if row.get("kind") in {"random", "random_direction"}
                   and isinstance(row.get("answer"), str)]
        if not neurons or len(randoms) < 2:
            raise ValueError(f"layer {layer_no} needs neuron and random-direction answers")
        all_answers = neurons + randoms
        names, probabilities = predict_batched(models, [row["answer"] for row in all_answers],
                                               "a", prediction_batch_size)
        if names != class_names:
            raise RuntimeError("Part 93 classifier classes differ from training metadata")
        neuron_scores = probabilities[:len(neurons)]
        random_scores = probabilities[len(neurons):]
        neuron_ids = [int(row["id"]) for row in neurons]
        all_candidates.extend(rank_layer_candidates(
            layer_no, neuron_ids, neuron_scores, random_scores, class_names, threshold,
        ))
        scored_rows += len(all_answers)
    all_candidates.sort(key=lambda row: (-row["move_z"], row["layer"], row["id"]))
    result = {
        "families": list(FAMILIES),
        "classifier": "A, masked; probability minus layer random-direction mean in random-direction SD units",
        "minimum_move_z": float(threshold),
        "limit": int(limit),
        "screen_answers_scored": int(scored_rows),
        "shortlist": all_candidates[:limit],
    }
    write_json(out_path, result)
    return result


def select_heldout_prompts(prompts: list[dict[str, str]], count: int = 20,
                           seed: int = HELDOUT_SEED) -> list[dict[str, str]]:
    if count < 1 or count > len(prompts):
        raise ValueError("held-out count must be between one and the number of prompts")
    return random.Random(seed).sample(prompts, count)


def read_screen_medians(screen_dir: Path, layers: list[int]) -> dict[int, float]:
    medians = {}
    for layer_no in sorted(set(layers)):
        path = screen_dir / f"layer_{layer_no}.jsonl.gz"
        if not path.is_file():
            path = screen_dir / f"layer_{layer_no}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing Part 86 screen rows for layer {layer_no}: {path}")
        values = [float(row["median"]) for row in read_jsonl(path) if row.get("median") is not None]
        if not values:
            raise ValueError(f"Part 86 layer {layer_no} has no residual median")
        if max(values) - min(values) > 1e-6:
            raise ValueError(f"Part 86 layer {layer_no} contains inconsistent residual medians")
        medians[layer_no] = values[0]
    return medians


def load_push_model(model_dir: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("XPU was requested but torch.xpu is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True, torch_dtype="auto",
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return tokenizer, model


def generate_pushed_batch(model: Any, tokenizer: Any, projection: Any, question: str,
                          pushes, device: str, max_new_tokens: int = MAX_NEW_TOKENS) -> list[dict[str, Any]]:
    import torch
    import part86.common86 as common86

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention = encoded.get("attention_mask")
    if attention is None:
        attention = torch.ones_like(input_ids)
    else:
        attention = attention.to(device)
    batch_size = int(pushes.shape[0])
    if input_ids.shape[0] != 1:
        raise ValueError("chat template must produce exactly one prompt row")
    input_ids = input_ids.expand(batch_size, -1).contiguous()
    attention = attention.expand(batch_size, -1).contiguous()
    hook = common86.AddRowsHook(projection, pushes)
    hook.begin_sequence()
    eos = common86.eos_ids(tokenizer, model)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer needs a pad or EOS token for generation")
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention,
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
        "pad_token_id": int(pad_id),
    }
    if eos:
        kwargs["eos_token_id"] = eos
    try:
        with torch.inference_mode():
            sequences = model.generate(**kwargs)
    finally:
        hook.remove()
    generated = sequences[:, input_ids.shape[1]:]
    stop_ids = set(eos)
    answers = []
    for row_no in range(batch_size):
        token_ids = generated[row_no].tolist()
        if stop_ids:
            for token_no, token_id in enumerate(token_ids):
                if int(token_id) in stop_ids:
                    token_ids = token_ids[:token_no + 1]
                    break
        fired = hook.counts["per_row_generated_calls"][row_no]
        if fired < max(0, len(token_ids) - 1):
            raise RuntimeError(f"generated-token push hook missed steps for row {row_no}")
        answers.append({
            "answer": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
            "token_count": len(token_ids),
        })
    return answers


def push(model_dir: Path, screen_dir: Path, shortlist_path: Path, out_dir: Path,
         prompts_path: Path = armA93.DEFAULT_PROMPTS, device: str = "xpu",
         batch_size: int = 8) -> dict:
    import torch
    import part86.common86 as common86

    with shortlist_path.open("r", encoding="utf-8") as handle:
        shortlist_data = json.load(handle)
    candidates = shortlist_data.get("shortlist", [])
    if not isinstance(candidates, list):
        raise ValueError("shortlist JSON needs a shortlist array")
    if not candidates:
        raise ValueError("shortlist is empty; there are no units to push")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    prompts = armA93.load_prompts(prompts_path)
    heldout = select_heldout_prompts(prompts)
    heldout_ids = [row["id"] for row in heldout]
    prompt_map = {row["id"]: row["prompt"] for row in prompts}
    layers_by_no = armA93.decoder_layers  # retain one supported layer finder for both arms
    tokenizer, model = load_push_model(model_dir, device)
    layers = layers_by_no(model)
    layer_numbers = sorted({int(row["layer"]) for row in candidates})
    if any(number < 0 or number >= len(layers) for number in layer_numbers):
        raise ValueError("shortlist contains a layer outside the loaded model")
    medians = read_screen_medians(screen_dir, layer_numbers)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "heldout.json", {
        "seed": HELDOUT_SEED,
        "prompt_ids": heldout_ids,
    })
    output_path = out_dir / "pushes.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for layer_no in layer_numbers:
            projection = armA93.down_projection(layers[layer_no], layer_no)
            weight = projection.weight.detach()
            median = medians[layer_no]
            neuron_candidates = [row for row in candidates if int(row["layer"]) == layer_no]
            unit_rows = []
            for row in neuron_candidates:
                neuron_id = int(row["id"])
                if neuron_id < 0 or neuron_id >= int(weight.shape[1]):
                    raise ValueError(f"neuron {neuron_id} is outside layer {layer_no} width")
                vector = torch.zeros(int(weight.shape[1]), dtype=torch.float32, device=weight.device)
                vector[neuron_id] = common86.neuron_delta(weight, neuron_id, median, dose=common86.DOSE)
                unit_rows.append({
                    "condition": "neuron", "unit_id": neuron_id,
                    "target_family": row.get("target_family"), "push": vector,
                })
            random_vectors = common86.scaled_random_directions(
                weight, count=4, seed=RANDOM_SEED + layer_no,
                target_norm=common86.DOSE * median,
            )
            for direction_no, vector in enumerate(random_vectors):
                unit_rows.append({
                    "condition": "random", "unit_id": direction_no,
                    "target_family": None, "push": vector.float(),
                })

            for prompt_no, prompt_id in enumerate([*heldout_ids, "name_check"]):
                question = NAME_QUESTION if prompt_id == "name_check" else prompt_map[prompt_id]
                for begin in range(0, len(unit_rows), batch_size):
                    current = unit_rows[begin:begin + batch_size]
                    pushes = torch.stack([row["push"] for row in current], dim=0)
                    answers = generate_pushed_batch(
                        model, tokenizer, projection, question, pushes, device,
                    )
                    for unit, answer in zip(current, answers):
                        record = {
                            "layer": layer_no,
                            "condition": unit["condition"],
                            "unit_id": int(unit["unit_id"]),
                            "target_family": unit["target_family"],
                            "prompt_id": prompt_id,
                            "answer": answer["answer"],
                            "token_count": int(answer["token_count"]),
                            "dose": float(common86.DOSE),
                            "median": float(median),
                            "random_seed": RANDOM_SEED + layer_no,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
    run = {
        "model": str(model_dir),
        "device": device,
        "families": list(FAMILIES),
        "heldout_seed": HELDOUT_SEED,
        "random_seed_base": RANDOM_SEED,
        "dose": float(common86.DOSE),
        "max_new_tokens": MAX_NEW_TOKENS,
        "greedy": True,
        "chat_template": {"tokenize": False, "enable_thinking": False},
        "shortlist_size": len(candidates),
        "layers": layer_numbers,
        "push_rows": str(output_path),
    }
    write_json(out_dir / "run.json", run)
    return run


def _style_rule(target: np.ndarray, random_controls: np.ndarray,
                class_names: list[str]) -> dict[str, Any]:
    target_mean = np.asarray(target, dtype=np.float64).mean(axis=0)
    random_means = np.asarray(random_controls, dtype=np.float64).mean(axis=1)
    baseline = random_means.mean(axis=0)
    rises = target_mean - baseline
    candidates = []
    for family in OUTSIDE:
        index = class_names.index(family)
        other_indices = [class_names.index(name) for name in OUTSIDE if name != family]
        all_random_below = bool(np.all(target_mean[index] > random_means[:, index]))
        strongest_other_rise = max(0.0, max(float(rises[other]) for other in other_indices))
        passes = bool(all_random_below and rises[index] > 0.0
                      and rises[index] >= 2.0 * strongest_other_rise)
        candidates.append({
            "family": family,
            "rise": float(rises[index]),
            "above_all_random_means": all_random_below,
            "twice_other_rises": bool(rises[index] >= 2.0 * strongest_other_rise),
            "passes": passes,
        })
    passing = [row["family"] for row in candidates if row["passes"]]
    return {
        "mean_probabilities": {name: float(target_mean[index]) for index, name in enumerate(class_names)},
        "random_direction_mean_probabilities": [
            {name: float(random_means[direction, index]) for index, name in enumerate(class_names)}
            for direction in range(random_means.shape[0])
        ],
        "rises_over_random_mean": {name: float(rises[index]) for index, name in enumerate(class_names)},
        "candidates": candidates,
        "moves_family": passing[0] if len(passing) == 1 else None,
    }


def _says_qwen(answer: str) -> bool:
    return re.search(r"\bqwen(?:[\s._-]*\d+(?:\.\d+)*)?", answer, flags=re.IGNORECASE) is not None


def score(gen_dir: Path, push_dir: Path, out_path: Path,
          prompts_path: Path = armA93.DEFAULT_PROMPTS,
          prediction_batch_size: int = 128) -> dict:
    prompts = armA93.load_prompts(prompts_path)
    expected_heldout = [row["id"] for row in select_heldout_prompts(prompts)]
    heldout_path = push_dir / "heldout.json"
    if heldout_path.is_file():
        with heldout_path.open("r", encoding="utf-8") as handle:
            saved_heldout = json.load(handle)
        heldout_ids = saved_heldout.get("prompt_ids", [])
        if heldout_ids != expected_heldout:
            raise ValueError("push held-out IDs do not match deterministic seed 93020 selection")
    else:
        heldout_ids = expected_heldout
    heldout_set = set(heldout_ids)
    prompt_ids = {row["id"] for row in prompts}
    training_rows = read_family_answers(gen_dir)
    by_prompt = defaultdict(set)
    for row in training_rows:
        by_prompt[row["prompt_id"]].add(row["family"])
    training_ids = prompt_ids - heldout_set
    incomplete = [prompt_id for prompt_id in training_ids
                  if by_prompt.get(prompt_id, set()) != set(FAMILIES)]
    if incomplete:
        raise ValueError(f"expected all four family answers for each of 180 training prompts; incomplete: {incomplete[:8]}")
    train_rows = [row for row in training_rows if row["prompt_id"] in training_ids]
    models = fit_classifiers(train_rows)
    class_names = list(models["class_names"])

    pushed_rows = read_jsonl(push_dir / "pushes.jsonl")
    scored_rows = [row for row in pushed_rows
                   if row.get("prompt_id") in heldout_set and isinstance(row.get("answer"), str)]
    if not scored_rows:
        raise ValueError("no held-out pushed answers found")
    for row in scored_rows:
        row["_probabilities"] = {}
    for classifier in ("a", "b"):
        names, probabilities = predict_batched(
            models, [row["answer"] for row in scored_rows], classifier, prediction_batch_size,
        )
        if names != class_names:
            raise RuntimeError("classifier class order changed during held-out scoring")
        for row, probability in zip(scored_rows, probabilities):
            row["_probabilities"][classifier] = probability

    scored_lookup = {
        (int(row["layer"]), row["condition"], int(row["unit_id"]), row["prompt_id"]): row
        for row in scored_rows
    }

    expected_prompt_order = list(heldout_ids)
    neurons: dict[tuple[int, int], dict[str, Any]] = {}
    random_by_layer: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in pushed_rows:
        layer_no = int(row.get("layer", -1))
        unit_id = int(row.get("unit_id", -1))
        if row.get("condition") == "neuron":
            neurons.setdefault((layer_no, unit_id), {"unit": row, "heldout": {}, "name": None})
            record = neurons[(layer_no, unit_id)]
        elif row.get("condition") == "random":
            random_by_layer[layer_no].setdefault(unit_id, {"heldout": {}, "name": None})
            record = random_by_layer[layer_no][unit_id]
        else:
            continue
        prompt_id = row.get("prompt_id")
        if prompt_id == "name_check":
            record["name"] = row
        elif prompt_id in heldout_set:
            record["heldout"][prompt_id] = scored_lookup.get(
                (layer_no, row["condition"], unit_id, prompt_id)
            )

    results = []
    for (layer_no, neuron_id), neuron_data in sorted(neurons.items()):
        controls = random_by_layer.get(layer_no, {})
        if len(controls) != 4 or set(controls) != {0, 1, 2, 3}:
            raise ValueError(f"layer {layer_no} needs random directions 0 through 3")
        if set(neuron_data["heldout"]) != heldout_set or any(
                row is None for row in neuron_data["heldout"].values()):
            raise ValueError(f"neuron {layer_no}:{neuron_id} is missing held-out answers")
        if neuron_data["name"] is None:
            raise ValueError(f"neuron {layer_no}:{neuron_id} is missing its name-check answer")
        if any(set(control["heldout"]) != heldout_set or control["name"] is None
               or any(answer is None for answer in control["heldout"].values())
               for control in controls.values()):
            raise ValueError(f"layer {layer_no} has incomplete random direction answers")

        classifier_results = {}
        moves_by_classifier = {}
        for classifier in ("a", "b"):
            target_probs = np.stack([
                neuron_data["heldout"][prompt_id]["_probabilities"][classifier]
                for prompt_id in expected_prompt_order
            ])
            control_probs = np.stack([
                np.stack([
                    controls[direction]["heldout"][prompt_id]["_probabilities"][classifier]
                    for prompt_id in expected_prompt_order
                ])
                for direction in range(4)
            ])
            detail = _style_rule(target_probs, control_probs, class_names)
            moves_by_classifier[classifier] = detail["moves_family"]
            classifier_results[classifier] = detail
        moves_family = moves_by_classifier["a"] if moves_by_classifier["a"] == moves_by_classifier["b"] else None
        name_answer = neuron_data["name"]["answer"]
        results.append({
            "layer": layer_no,
            "neuron_id": neuron_id,
            "shortlist_target_family": neuron_data["unit"].get("target_family"),
            "classifier_a": classifier_results["a"],
            "classifier_b": classifier_results["b"],
            "moves_style": moves_family is not None,
            "moves_family": moves_family,
            "name_answer": name_answer,
            "name_still_says_qwen": _says_qwen(name_answer),
        })
    name_checks = []
    for (layer_no, unit_id), data in sorted(neurons.items()):
        answer = data["name"]["answer"] if data["name"] else ""
        name_checks.append({"layer": layer_no, "condition": "neuron", "unit_id": unit_id,
                            "answer": answer, "still_says_qwen": _says_qwen(answer)})
    for layer_no, controls in sorted(random_by_layer.items()):
        for direction, data in sorted(controls.items()):
            answer = data["name"]["answer"] if data["name"] else ""
            name_checks.append({"layer": layer_no, "condition": "random", "unit_id": direction,
                                "answer": answer, "still_says_qwen": _says_qwen(answer)})
    output = {
        "training_prompts": len(training_ids),
        "heldout_seed": HELDOUT_SEED,
        "heldout_prompt_ids": heldout_ids,
        "classes": class_names,
        "move_rule": "both masked classifiers; target family exceeds every layer random-direction mean over 20 prompts, with target rise at least twice the largest other positive outside-family rise",
        "name_check": {
            "units": len(name_checks),
            "answers_still_saying_qwen": sum(row["still_says_qwen"] for row in name_checks),
            "rows": name_checks,
        },
        "units": results,
    }
    write_json(out_path, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    short = sub.add_parser("shortlist", help="fit classifier A and rank Part 86 neurons")
    short.add_argument("--gen", required=True, type=Path)
    short.add_argument("--screen", required=True, type=Path)
    short.add_argument("--out", required=True, type=Path)
    short.add_argument("--limit", type=int, default=40)
    short.add_argument("--threshold", type=float, default=3.0)
    short.add_argument("--batch-size", type=int, default=128)

    push_parser = sub.add_parser("push", help="steer held-out prompts and run name checks")
    push_parser.add_argument("--model", required=True, type=Path)
    push_parser.add_argument("--screen", required=True, type=Path)
    push_parser.add_argument("--shortlist", required=True, type=Path)
    push_parser.add_argument("--out", required=True, type=Path)
    push_parser.add_argument("--prompts", type=Path, default=armA93.DEFAULT_PROMPTS)
    push_parser.add_argument("--device", choices=("xpu", "cpu", "cuda"), default="xpu")
    push_parser.add_argument("--batch-size", type=int, default=8)

    score_parser = sub.add_parser("score", help="refit on non-held-out answers and apply the style rule")
    score_parser.add_argument("--gen", required=True, type=Path)
    score_parser.add_argument("--push", required=True, type=Path)
    score_parser.add_argument("--out", required=True, type=Path)
    score_parser.add_argument("--prompts", type=Path, default=armA93.DEFAULT_PROMPTS)
    score_parser.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args(argv)
    if args.command == "shortlist":
        result = shortlist(args.gen, args.screen, args.out, args.limit, args.threshold, args.batch_size)
        print(f"scored {result['screen_answers_scored']} screen answers; shortlisted {len(result['shortlist'])} units")
    elif args.command == "push":
        result = push(args.model, args.screen, args.shortlist, args.out,
                      args.prompts, args.device, args.batch_size)
        print(f"pushed {result['shortlist_size']} shortlisted units across {len(result['layers'])} layers to {args.out}")
    else:
        result = score(args.gen, args.push, args.out, args.prompts, args.batch_size)
        print(f"scored {len(result['units'])} units; {result['name_check']['answers_still_saying_qwen']} still say Qwen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
