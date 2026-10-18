#!/usr/bin/env python3
"""Build and run Part 17's natural-residual self-report experiment.

All model loads are offline (HF cache only). ``capture`` writes passage activations
and Qwen text features; ``fit`` learns the input-only ridge and freezes FIT-derived
labels; the remaining commands train/evaluate readers or replay FIT-derived clamps.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


OLMO_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
QWEN_MODEL = "Qwen/Qwen3-0.6B"
GROUP_NAMES = ("G1", "G2", "G3", "G4", "GL")
ANSWER_WORDS = ("lower", "higher")
DEFAULT_IDENTITY = "results/part2-olmo2-1b/identity-instruct-chat.json"
DEFAULT_CORPUS = "data/wikitext103-train0.parquet"
DEFAULT_OUT_DIR = "results/part17"
GROUP_SEED = 7
GL_SEED = 17017
DEFAULT_SEED = 42
CAPTURE_BATCH_SIZE = 4
QWEN_BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
SMOKE_COUNTS = (12, 16, 8)
DEFAULT_COUNTS = (4000, 6000, 2000)
SMOKE_EPOCHS = 1
DEFAULT_EPOCHS = 3
DEFAULT_LR = 3e-4
DEFAULT_RANK = 8
DEFAULT_MIN_TOKENS = 40
DEFAULT_MAX_TOKENS = 120
TARGET_PASSAGE_TOKENS = 80


@dataclass
class PromptExample:
    """The subset of selfreport.Example used by the shared batch helper."""

    passage_id: str
    text: str
    prompt_ids: list[int]
    span_positions: list[int]
    label: int | None = None
    train_label: int | None = None
    train_clamp: np.ndarray | None = None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_by_hash(
    passages: Sequence[dict[str, Any]],
    fit_count: int,
    train_count: int,
    test_count: int,
) -> list[dict[str, Any]]:
    """Assign exact split sizes by sorting stable passage hashes, independent of row order."""
    if min(fit_count, train_count, test_count) < 0:
        raise ValueError("Split counts must be nonnegative")
    required = fit_count + train_count + test_count
    records = []
    seen: set[str] = set()
    for raw in passages:
        text = str(raw["text"]).strip()
        passage_id = str(raw.get("passage_id") or sha256_text(text))
        if passage_id in seen:
            continue
        seen.add(passage_id)
        records.append({**raw, "text": text, "passage_id": passage_id})
    if len(records) < required:
        raise ValueError(f"Need {required} unique passages for the requested split; found {len(records)}")
    records.sort(key=lambda row: row["passage_id"])
    chosen = records[:required]
    for index, row in enumerate(chosen):
        if index < fit_count:
            row["split"] = "FIT"
        elif index < fit_count + train_count:
            row["split"] = "TRAIN"
        else:
            row["split"] = "TEST"
    return chosen


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Centered closed-form ridge, solved in sample space for wide Qwen features."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim not in (1, 2) or len(x) != len(y):
        raise ValueError("Ridge expects aligned [rows, features] and [rows, outputs] arrays")
    if alpha < 0 or not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite and nonnegative")
    one_dimensional = y.ndim == 1
    if one_dimensional:
        y = y[:, None]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    if xc.shape[1] <= xc.shape[0]:
        gram = xc.T @ xc
        gram.flat[:: len(gram) + 1] += float(alpha)
        weights = np.linalg.solve(gram, xc.T @ yc)
    else:
        gram = xc @ xc.T
        gram.flat[:: len(gram) + 1] += float(alpha)
        dual = np.linalg.solve(gram, yc)
        weights = xc.T @ dual
    bias = y_mean - x_mean @ weights
    if one_dimensional:
        return weights[:, 0], np.asarray([bias[0]])
    return weights, bias


def ridge_predict(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    prediction = np.asarray(x, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)
    return prediction + np.asarray(bias, dtype=np.float64)


def select_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = 0.25,
) -> tuple[float, dict[str, Any]]:
    """Choose alpha using only a deterministic held-out quarter of FIT."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 4:
        raise ValueError("At least four FIT rows are needed for held-out alpha selection")
    order = np.random.default_rng(seed).permutation(len(x))
    validation_count = max(1, int(math.ceil(len(x) * validation_fraction)))
    validation_count = min(validation_count, len(x) - 2)
    validation = order[:validation_count]
    fitting = order[validation_count:]
    centered = x[fitting] - x[fitting].mean(axis=0)
    gram_scale = float(np.trace(centered @ centered.T) / max(1, len(fitting)))
    base = max(gram_scale, 1e-8)
    candidates = [float(base * factor) for factor in np.logspace(-6, 2, 9)]
    scores = []
    for alpha in candidates:
        weights, bias = ridge_fit(x[fitting], y[fitting], alpha)
        error = ridge_predict(x[validation], weights, bias) - y[validation]
        scores.append(float(np.mean(error * error)))
    best = int(np.argmin(scores))
    return candidates[best], {
        "validation_rows": int(validation_count),
        "fitting_rows": int(len(fitting)),
        "candidate_alphas": candidates,
        "validation_mse": scores,
        "selected_alpha": candidates[best],
    }


def fit_threshold_labels(
    residual_scalar: np.ndarray,
    splits: Sequence[str],
) -> tuple[float, np.ndarray]:
    """Threshold all labels at the FIT-only median; higher=1 and lower=0."""
    residual_scalar = np.asarray(residual_scalar, dtype=np.float64)
    splits = np.asarray(splits)
    if residual_scalar.ndim != 1 or len(residual_scalar) != len(splits):
        raise ValueError("Residuals and split names must be aligned one-dimensional arrays")
    fit = splits == "FIT"
    if not np.any(fit):
        raise ValueError("Cannot label without FIT residuals")
    threshold = float(np.median(residual_scalar[fit]))
    return threshold, (residual_scalar > threshold).astype(np.int64)


def fit_activation_quantiles(
    activations: np.ndarray,
    splits: Sequence[str],
    low: float = 0.20,
    high: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group, per-unit natural-mean clamp bounds estimated from FIT rows only."""
    activations = np.asarray(activations, dtype=np.float64)
    splits = np.asarray(splits)
    if activations.ndim != 3 or activations.shape[0] != len(splits):
        raise ValueError("Activations must have shape [passage, group, unit]")
    if not 0 <= low < high <= 1:
        raise ValueError("Require 0 <= low < high <= 1")
    fit = activations[splits == "FIT"]
    if len(fit) == 0:
        raise ValueError("Cannot estimate clamp quantiles without FIT activations")
    return (
        np.quantile(fit, low, axis=0).astype(np.float32),
        np.quantile(fit, high, axis=0).astype(np.float32),
    )


def shuffled_training_labels(labels: Sequence[int], seed: int) -> list[int]:
    """Reproducibly permute labels, retrying only if the permutation is unchanged."""
    values = np.asarray(labels, dtype=np.int64)
    if len(values) < 2 or len(np.unique(values)) < 2:
        return values.tolist()
    rng = np.random.default_rng(seed)
    for _ in range(20):
        shuffled = values[rng.permutation(len(values))]
        if not np.array_equal(shuffled, values):
            return shuffled.tolist()
    return np.roll(values, 1).tolist()


def _torch_helpers():
    import torch
    import torch.nn.functional as F
    import selfreport as sr
    import selfreport_multi as srm

    return torch, F, sr, srm


def _device(value: str):
    torch, _, _, _ = _torch_helpers()
    if value == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("--device xpu was requested but torch.xpu is unavailable")
    if value not in ("cpu", "xpu"):
        raise ValueError("--device must be cpu or xpu")
    return torch.device(value)


def _load_model(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch, _, _, _ = _torch_helpers()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, local_files_only=True
    )
    model.to(_device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def _as_flat_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value and isinstance(value[0], list) and len(value) == 1:
        value = value[0]
    return list(value)


def _encode_with_offsets(tokenizer: Any, rendered: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = _as_flat_list(encoded["input_ids"])
    offsets = _as_flat_list(encoded["offset_mapping"])
    return [int(x) for x in ids], [(int(a), int(b)) for a, b in offsets]


def _span_positions(rendered: str, marker: str, offsets: Sequence[tuple[int, int]]) -> list[int]:
    start = rendered.find(marker)
    if start < 0:
        raise ValueError("Passage text was not found in its rendered chat prompt")
    end = start + len(marker)
    positions = [i for i, (left, right) in enumerate(offsets) if right > start and left < end and right > left]
    if not positions:
        raise ValueError("No tokens overlap the passage text in the rendered prompt")
    return positions


def _capture_prompt(tokenizer: Any, text: str) -> tuple[list[int], list[int]]:
    messages = [{"role": "user", "content": text}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids, offsets = _encode_with_offsets(tokenizer, rendered)
    return ids, _span_positions(rendered, text, offsets)


def _question(group: str) -> str:
    return (
        f"While you read my first message, was group {group} of your neurons more or less active "
        "than its words would suggest? Answer higher or lower."
    )


def _reader_prompt(tokenizer: Any, text: str, group: str, passage_id: str) -> PromptExample:
    # This mirrors selfreport's first-user / assistant / final-question message structure.
    messages = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "Okay."},
        {"role": "user", "content": _question(group)},
    ]
    rendered = _render_generation_prompt(tokenizer, messages)
    ids, offsets = _encode_with_offsets(tokenizer, rendered)
    return PromptExample(passage_id, text, ids, _span_positions(rendered, text, offsets))


def _render_generation_prompt(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    """Render an assistant prefix with thinking disabled when the template supports it."""
    apply_template = tokenizer.apply_chat_template
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        parameters = inspect.signature(apply_template).parameters.values()
    except (TypeError, ValueError):
        supports_enable_thinking = True
    else:
        supports_enable_thinking = any(
            parameter.name == "enable_thinking"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    if supports_enable_thinking:
        kwargs["enable_thinking"] = False
    return apply_template(messages, **kwargs)


def _answer_targets(tokenizer: Any, prefix_ids: Sequence[int]) -> dict[str, list[int]]:
    """Build full prefix-plus-answer targets and ensure their first answer tokens differ."""
    prefix = [int(token) for token in prefix_ids]
    targets: dict[str, list[int]] = {}
    first_answer_tokens: dict[str, int] = {}
    for answer in ANSWER_WORDS:
        answer_ids = [int(token) for token in tokenizer.encode(answer, add_special_tokens=False)]
        if not answer_ids:
            raise ValueError(f"Answer {answer!r} produced no tokens")
        targets[answer] = prefix + answer_ids
        first_answer_tokens[answer] = targets[answer][len(prefix)]
    if first_answer_tokens["higher"] == first_answer_tokens["lower"]:
        raise ValueError(f"Higher/lower answer tokens are not distinct: {first_answer_tokens}")
    return targets


def _answer_token_ids(tokenizer: Any, example: PromptExample, group: str) -> dict[str, int]:
    del group  # The example already contains the exact prefix shared by training and eval.
    targets = _answer_targets(tokenizer, example.prompt_ids)
    answer_position = len(example.prompt_ids)
    return {answer: targets[answer][answer_position] for answer in ANSWER_WORDS}


def _prompt_logit_position(example: PromptExample) -> int:
    """Return the final prefix position whose logits predict the first answer token."""
    if not example.prompt_ids:
        raise ValueError("Cannot score an empty generation prefix")
    return len(example.prompt_ids) - 1


def _paired_prompt_logits(logits: Any, batch: Sequence[PromptExample]) -> list[Any]:
    """Select paired low/high logits using prompt positions from this batch."""
    paired_examples = [*batch, *batch]
    if int(logits.shape[0]) != len(paired_examples):
        raise ValueError("Paired logits and transplant batch sizes differ")
    return [logits[row, _prompt_logit_position(example)] for row, example in enumerate(paired_examples)]


def _test_label_summary(labels: Sequence[int]) -> dict[str, Any]:
    values = [int(label) for label in labels]
    lower = sum(label == 0 for label in values)
    higher = sum(label == 1 for label in values)
    return {
        "majority_class_rate": max(lower, higher) / len(values) if values else None,
        "label_counts": {"lower": int(lower), "higher": int(higher)},
    }


def _lora_drift_metrics(base: Sequence[float], adapted: Sequence[float]) -> dict[str, float | None]:
    base_values = np.asarray(base, dtype=np.float64)
    adapted_values = np.asarray(adapted, dtype=np.float64)
    if base_values.ndim != 1 or adapted_values.ndim != 1 or len(base_values) != len(adapted_values):
        raise ValueError("Base and adapter activations must be aligned one-dimensional arrays")
    if not len(base_values):
        return {"pearson_correlation": None, "mean_absolute_difference": None}
    correlation = None
    if len(base_values) >= 2 and np.std(base_values) > 0 and np.std(adapted_values) > 0:
        correlation = float(np.corrcoef(base_values, adapted_values)[0, 1])
    return {
        "pearson_correlation": correlation,
        "mean_absolute_difference": float(np.mean(np.abs(adapted_values - base_values))),
    }


def _pad_id(tokenizer: Any) -> int:
    return int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0))


def _split_indices(manifest: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray([i for i, row in enumerate(manifest["passages"]) if row["split"] == name], dtype=np.int64)


def _paths(out_dir: str | Path) -> dict[str, Path]:
    root = Path(out_dir)
    return {
        "root": root,
        "capture": root / "capture.npz",
        "manifest": root / "capture-manifest.json",
        "fit_npz": root / "fit.npz",
        "fit_json": root / "fit.json",
    }


def _run_suffix(args: argparse.Namespace) -> str:
    tag = str(getattr(args, "run_tag", "") or "").strip()
    if not tag and bool(getattr(args, "shuffle_labels", False)):
        tag = "shuffled"
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in tag):
        raise ValueError("--run-tag may contain only letters, digits, underscore, and hyphen")
    return f"-{tag}" if tag else ""


def _read_capture(out_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paths = _paths(out_dir)
    if not paths["capture"].is_file() or not paths["manifest"].is_file():
        raise FileNotFoundError(f"Expected {paths['capture']} and {paths['manifest']}; run capture first")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    with np.load(paths["capture"], allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return arrays, manifest


def _read_fit(out_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    paths = _paths(out_dir)
    arrays, manifest = _read_capture(out_dir)
    if not paths["fit_npz"].is_file() or not paths["fit_json"].is_file():
        raise FileNotFoundError(f"Expected {paths['fit_npz']} and {paths['fit_json']}; run fit first")
    with np.load(paths["fit_npz"], allow_pickle=False) as loaded:
        fit = {key: loaded[key] for key in loaded.files}
    fit_json = json.loads(paths["fit_json"].read_text(encoding="utf-8"))
    return {**arrays, **fit}, manifest, fit_json


def _manifest_records(
    corpus: str,
    passages: Sequence[dict[str, Any]],
    groups: dict[str, list[tuple[int, int]]],
    fit_activations: np.ndarray,
    timings: dict[str, float],
    config: dict[str, Any],
    model_details: dict[str, Any],
) -> dict[str, Any]:
    split_counts = {name: sum(row["split"] == name for row in passages) for name in ("FIT", "TRAIN", "TEST")}
    fit = np.asarray([row["split"] == "FIT" for row in passages])
    per_group = {}
    for gi, name in enumerate(GROUP_NAMES):
        values = fit_activations[fit, gi]
        per_group[name] = {
            "unit_count": int(values.shape[1]),
            "fit_natural_mean": float(values.mean()),
            "fit_natural_std": float(values.std()),
        }
    return {
        "task": "part17-natural-residual-self-report",
        "config": config,
        "models": model_details,
        "sizes": {"total": len(passages), "splits": split_counts},
        "seeds": {"split": "passage SHA-256 ascending", "groups": GROUP_SEED, "GL": GL_SEED, "run": config["seed"]},
        "timings_seconds": timings,
        "per_group": per_group,
        "group_units": {name: [f"L{layer}.{neuron}" for layer, neuron in units] for name, units in groups.items()},
        "passages": [
            {
                "passage_id": str(row["passage_id"]),
                "split": row["split"],
                "text_hash": sha256_text(row["text"]),
                "text": row["text"],
                "olmo_token_count": int(row["olmo_token_count"]),
            }
            for row in passages
        ],
        "corpus": str(corpus),
    }


def _corpus_lines(corpus: str) -> list[str]:
    """The WikiText `text` column, from the parquet or from a JSON-lines export of it.

    The Windows GPU venv has no pyarrow (2026-10-18), so the parquet is exported once in WSL to JSON lines
    ({"text": ...} per row, same order); both paths yield the same list.
    """
    if str(corpus).endswith(".jsonl"):
        with open(corpus, encoding="utf-8") as src:
            return [json.loads(line)["text"] for line in src if line.strip()]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("capture requires pyarrow to read the WikiText parquet (or pass a .jsonl export)") from exc
    return pq.read_table(corpus, columns=["text"]).column("text").to_pylist()


def _passages_from_parquet(
    corpus: str,
    tokenizer: Any,
    counts: tuple[int, int, int],
    minimum: int = DEFAULT_MIN_TOKENS,
    maximum: int = DEFAULT_MAX_TOKENS,
    target: int = TARGET_PASSAGE_TOKENS,
) -> list[dict[str, Any]]:
    required = sum(counts)
    lines = _corpus_lines(corpus)
    newline_ids = [int(x) for x in tokenizer("\n", add_special_tokens=False)["input_ids"]]
    stream: list[int] = []
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    batch_no = 0

    def add_chunk(ids: list[int]) -> None:
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not text:
            return
        actual_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if actual_tokens < minimum or actual_tokens > maximum:
            return
        passage_id = sha256_text(text)
        if passage_id in seen:
            return
        seen.add(passage_id)
        candidates.append({"passage_id": passage_id, "text": text, "olmo_token_count": actual_tokens})

    for line in lines:
        if not isinstance(line, str) or not line.strip() or line.strip().startswith("="):
            continue
        stream.extend(int(x) for x in tokenizer(line, add_special_tokens=False)["input_ids"])
        stream.extend(newline_ids)
        while len(stream) >= target and len(candidates) < required:
            add_chunk(stream[:target])
            del stream[:target]
            batch_no += 1
            if batch_no % 1000 == 0:
                print(f"passage build {batch_no} chunks; {len(candidates)}/{required} unique", flush=True)
        if len(candidates) >= required:
            break
    if len(candidates) < required and len(stream) >= minimum:
        add_chunk(stream[: min(maximum, len(stream))])
    if len(candidates) < required:
        raise RuntimeError(f"WikiText supplied only {len(candidates)} unique passages; need {required}")
    return candidates


def _fit_capture(args: argparse.Namespace) -> dict[str, Any]:
    torch, _, sr, srm = _torch_helpers()
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    sr.DEVICE = args.device
    tokenizer, model = _load_model(args.olmo_model, args.device)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected the selfreport model.model.layers OLMo hook path")
    layers = model.model.layers
    neuron_count = int(layers[0].mlp.down_proj.in_features)
    g1 = sr.select_target_units("A", args.identity, neuron_count)
    groups_all = srm.make_groups(g1, neuron_count, GROUP_SEED)
    groups = {name: list(groups_all[name]) for name in ("G1", "G2", "G3", "G4")}
    last_layer = len(layers) - 1
    used = {unit for values in groups.values() for unit in values}
    pool = [index for index in range(neuron_count) if (last_layer, index) not in used]
    if len(pool) < 11:
        raise ValueError("The last OLMo layer has fewer than 11 available units for GL")
    gl_rng = np.random.default_rng(GL_SEED)
    groups["GL"] = [(last_layer, int(n)) for n in gl_rng.choice(pool, size=11, replace=False)]
    unit_order = [unit for group in GROUP_NAMES for unit in groups[group]]
    group_slices = {}
    at = 0
    for name in GROUP_NAMES:
        group_slices[name] = slice(at, at + len(groups[name]))
        at += len(groups[name])

    counts = SMOKE_COUNTS if args.smoke else (args.fit_count, args.train_count, args.test_count)
    passages = _passages_from_parquet(args.corpus, tokenizer, counts, args.min_tokens, args.max_tokens)
    passages = split_by_hash(passages, *counts)
    examples = []
    for row in passages:
        ids, spans = _capture_prompt(tokenizer, row["text"])
        examples.append(PromptExample(row["passage_id"], row["text"], ids, spans))

    hooks = sr.NeuronHooks(model, unit_order)
    olmo_activations = np.empty((len(examples), len(GROUP_NAMES), 11), dtype=np.float32)
    model.eval()
    capture_started = time.perf_counter()
    with torch.inference_mode():
        batches = list(sr.chunks(examples, args.capture_batch_size))
        for batch_i, batch in enumerate(batches, 1):
            input_ids, attention_mask, _, spans = sr.padded_batch(batch, _pad_id(tokenizer))
            hooks.prepare(spans, capture=True)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for row_i, example in enumerate(batch):
                captured = hooks.captures[row_i]
                if captured is None:
                    raise RuntimeError(f"No OLMo activation capture for {example.passage_id}")
                passage_means = captured.float().numpy().mean(axis=0)
                index = (batch_i - 1) * args.capture_batch_size + row_i
                for group_i, group in enumerate(GROUP_NAMES):
                    olmo_activations[index, group_i] = passage_means[group_slices[group]]
            print(f"OLMo capture batch {batch_i}/{len(batches)} rows {len(batch)}", flush=True)
    hooks.close()
    olmo_capture_seconds = time.perf_counter() - capture_started
    del hooks, model, tokenizer
    gc.collect()
    if args.device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()

    tokenizer, model = _load_model(args.qwen_model, args.device)
    sr.DEVICE = args.device
    qwen_features: list[np.ndarray] = []
    qwen_started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        batches = list(sr.chunks(passages, args.qwen_batch_size))
        for batch_i, batch in enumerate(batches, 1):
            q_examples = []
            for row in batch:
                ids, spans = _capture_prompt(tokenizer, row["text"])
                q_examples.append(PromptExample(row["passage_id"], row["text"], ids, spans))
            input_ids, attention_mask, _, spans = sr.padded_batch(q_examples, _pad_id(tokenizer))
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            final_hidden = out.hidden_states[-1]
            for row_i, example in enumerate(q_examples):
                positions = torch.as_tensor(spans[row_i], dtype=torch.long, device=final_hidden.device)
                qwen_features.append(final_hidden[row_i].index_select(0, positions).float().mean(0).cpu().numpy())
            print(f"Qwen feature batch {batch_i}/{len(batches)} rows {len(batch)}", flush=True)
    qwen_capture_seconds = time.perf_counter() - qwen_started
    model_details = {
        "olmo": args.olmo_model,
        "qwen": args.qwen_model,
        "hidden_size": int(np.asarray(qwen_features).shape[1]),
        "activation": "MLP down_proj input (post-SiLU*up), passage-token mean",
        "qwen_feature": "final hidden state, passage-token mean",
    }
    timings = {
        "total_seconds": time.perf_counter() - started,
        "olmo_capture_seconds": olmo_capture_seconds,
        "qwen_feature_seconds": qwen_capture_seconds,
    }
    config = {
        "device": args.device,
        "seed": args.seed,
        "identity": args.identity,
        "group_seed": GROUP_SEED,
        "gl_seed": GL_SEED,
        "passage_token_range": [args.min_tokens, args.max_tokens],
        "target_passage_tokens": TARGET_PASSAGE_TOKENS,
        "smoke": bool(args.smoke),
        "split_counts": {"FIT": counts[0], "TRAIN": counts[1], "TEST": counts[2]},
        "capture_batch_size": args.capture_batch_size,
        "qwen_batch_size": args.qwen_batch_size,
        "activation_hook_writes": False,
    }
    manifest = _manifest_records(args.corpus, passages, groups, olmo_activations, timings, config, model_details)
    paths = _paths(args.out_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["capture"],
        passage_ids=np.asarray([row["passage_id"] for row in passages], dtype="U64"),
        splits=np.asarray([row["split"] for row in passages], dtype="U5"),
        olmo_activations=olmo_activations,
        qwen_features=np.asarray(qwen_features, dtype=np.float32),
    )
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(paths["capture"]), "manifest": str(paths["manifest"]), "sizes": manifest["sizes"], "timings_seconds": timings}, indent=2), flush=True)
    return manifest


def _fit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    arrays, manifest = _read_capture(args.out_dir)
    features = arrays["qwen_features"].astype(np.float64)
    activations = arrays["olmo_activations"].astype(np.float64)
    splits = arrays["splits"].astype(str)
    fit_indices = np.flatnonzero(splits == "FIT")
    train_indices = np.flatnonzero(splits == "TRAIN")
    test_indices = np.flatnonzero(splits == "TEST")
    if min(len(fit_indices), len(train_indices), len(test_indices)) == 0:
        raise ValueError("FIT, TRAIN, and TEST must all be nonempty")
    predicted = np.empty_like(activations, dtype=np.float32)
    residuals = np.empty_like(activations, dtype=np.float32)
    ridge_weights = np.empty((len(GROUP_NAMES), features.shape[1], activations.shape[2]), dtype=np.float32)
    ridge_biases = np.empty((len(GROUP_NAMES), activations.shape[2]), dtype=np.float32)
    alphas: dict[str, float] = {}
    alpha_selection: dict[str, Any] = {}
    per_group: dict[str, Any] = {}
    threshold_values = []
    label_matrix = np.zeros((len(splits), len(GROUP_NAMES)), dtype=np.int8)
    residual_scalar = np.zeros((len(splits), len(GROUP_NAMES)), dtype=np.float64)
    for gi, group in enumerate(GROUP_NAMES):
        alpha, selection = select_ridge_alpha(features[fit_indices], activations[fit_indices, gi], args.seed)
        weights, bias = ridge_fit(features[fit_indices], activations[fit_indices, gi], alpha)
        prediction = ridge_predict(features, weights, bias)
        resid = activations[:, gi] - prediction
        predicted[:, gi] = prediction.astype(np.float32)
        residuals[:, gi] = resid.astype(np.float32)
        ridge_weights[gi] = weights.astype(np.float32)
        ridge_biases[gi] = bias.astype(np.float32)
        residual_scalar[:, gi] = resid.mean(axis=1)
        alpha_selection[group] = selection
        alphas[group] = alpha
        y_test = activations[test_indices, gi]
        pred_test = prediction[test_indices]
        ss_res = float(np.sum((y_test - pred_test) ** 2))
        centered_test = y_test - y_test.mean(axis=0, keepdims=True)
        ss_tot = float(np.sum(centered_test ** 2))
        pooled_r2 = None if ss_tot <= 1e-12 else float(1 - ss_res / ss_tot)
        per_unit_r2 = []
        for unit_i in range(y_test.shape[1]):
            target = y_test[:, unit_i]
            denom = float(np.sum((target - target.mean()) ** 2))
            per_unit_r2.append(None if denom <= 1e-12 else float(1 - np.sum((target - pred_test[:, unit_i]) ** 2) / denom))
        per_group[group] = {
            "alpha": alpha,
            "test_r2": pooled_r2,
            "test_r2_mean_per_unit": None if all(value is None for value in per_unit_r2) else float(np.nanmean([np.nan if value is None else value for value in per_unit_r2])),
            "test_n": int(len(test_indices)),
            "residual_mean_fit": float(resid[fit_indices].mean()),
            "residual_std_fit": float(resid[fit_indices].std()),
        }
        threshold, labels = fit_threshold_labels(residual_scalar[:, gi], splits)
        threshold_values.append(threshold)
        label_matrix[:, gi] = labels
        print(f"fit ridge group {group}: alpha={alpha:.6g} test R2={pooled_r2}", flush=True)
    low, high = fit_activation_quantiles(activations, splits)
    for gi, group in enumerate(GROUP_NAMES):
        train_labels = label_matrix[train_indices, gi]
        test_labels = label_matrix[test_indices, gi]
        per_group[group]["fit_median_residual_threshold"] = threshold_values[gi]
        per_group[group]["train_label_counts"] = {"lower": int(np.sum(train_labels == 0)), "higher": int(np.sum(train_labels == 1))}
        per_group[group]["test_label_counts"] = {"lower": int(np.sum(test_labels == 0)), "higher": int(np.sum(test_labels == 1))}
        per_group[group]["fit_quantile20_mean"] = float(low[gi].mean())
        per_group[group]["fit_quantile80_mean"] = float(high[gi].mean())
    result = {
        "task": "part17-natural-residual-fit",
        "config": {
            "seed": args.seed,
            "alpha_validation_fraction": 0.25,
            "ridge": "centered dual closed-form, one 11-output model per group",
            "label_definition": "mean of 11 unit residuals > FIT-only median means higher (1); otherwise lower (0)",
            "r2_definition": "pooled over TEST passages and 11 units",
            "quantiles": [0.20, 0.80],
        },
        "sizes": {"FIT": len(fit_indices), "TRAIN": len(train_indices), "TEST": len(test_indices), "features": int(features.shape[1])},
        "seeds": {
            "alpha_validation": args.seed,
            "split": "passage SHA-256 ascending",
            "groups": GROUP_SEED,
            "GL": GL_SEED,
        },
        "timings_seconds": {"fit_total": time.perf_counter() - started},
        "alpha_selection": alpha_selection,
        "per_group": per_group,
    }
    paths = _paths(args.out_dir)
    np.savez_compressed(
        paths["fit_npz"],
        predicted=predicted,
        residuals=residuals,
        residual_scalar=residual_scalar.astype(np.float32),
        labels=label_matrix,
        thresholds=np.asarray(threshold_values, dtype=np.float64),
        alphas=np.asarray([alphas[group] for group in GROUP_NAMES], dtype=np.float64),
        low20=low,
        high80=high,
        ridge_weights=ridge_weights,
        ridge_biases=ridge_biases,
    )
    paths["fit_json"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(paths["fit_npz"]), "report": str(paths["fit_json"]), "per_group": per_group}, indent=2), flush=True)
    return result


def _passages_for_split(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [row for row in manifest["passages"] if row["split"] == split]


def _build_examples(tokenizer: Any, passages: Sequence[dict[str, Any]], group: str) -> list[PromptExample]:
    return [_reader_prompt(tokenizer, row["text"], group, row["passage_id"]) for row in passages]


def _adapter_path(args: argparse.Namespace, arm: str) -> Path:
    if getattr(args, "adapter", None):
        return Path(args.adapter)
    return Path(args.out_dir) / f"lora-{arm}-{args.group}{_run_suffix(args)}.pt"


def _adapter_sha256(path: str | Path) -> str:
    adapter = Path(path)
    if not adapter.is_file():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter}")
    digest = hashlib.sha256()
    with adapter.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eval_output_path(args: argparse.Namespace, arm: str, transplant: bool = False) -> Path:
    suffix = _run_suffix(args)
    if transplant:
        return Path(args.out_dir) / f"transplant-{args.group}{suffix}.json"
    return Path(args.out_dir) / f"eval-{arm}-{args.group}{suffix}.json"


def _target_name(arm: str, group: str) -> str:
    return f"residual17-{arm}-{group}"


def _train_reader(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    torch, F, sr, _ = _torch_helpers()
    model_id = args.olmo_model if arm == "self" else args.qwen_model
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    sr.DEVICE = args.device
    arrays, manifest, fit_report = _read_fit(args.out_dir)
    passages = _passages_for_split(manifest, "TRAIN")
    group_i = GROUP_NAMES.index(args.group)
    labels = arrays["labels"][:, group_i]
    ids_by_passage = {str(passage_id): i for i, passage_id in enumerate(arrays["passage_ids"].astype(str))}
    true_labels = [int(labels[ids_by_passage[row["passage_id"]]]) for row in passages]
    effective_labels = shuffled_training_labels(true_labels, args.seed) if args.shuffle_labels else true_labels
    for row, label in zip(passages, effective_labels):
        row["label"] = int(label)
    tokenizer, model = _load_model(model_id, args.device)
    examples = _build_examples(tokenizer, passages, args.group)
    for example, label in zip(examples, effective_labels):
        example.train_label = int(label)
    answer_ids = _answer_token_ids(tokenizer, examples[0], args.group)
    sr.install_lora(model, rank=args.rank)
    model.to(_device(args.device))
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params or any("lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise RuntimeError("Only LoRA parameters may be trainable")
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    epochs = args.epochs if args.epochs is not None else (SMOKE_EPOCHS if args.smoke else DEFAULT_EPOCHS)
    rng = np.random.default_rng(args.seed)
    hooks = sr.NeuronHooks(model, [])
    losses: list[float] = []
    step_times: list[float] = []
    total_steps = epochs * math.ceil(len(examples) / args.train_batch_size)
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(examples)).tolist()
        for indices in sr.chunks(order, args.train_batch_size):
            batch = [examples[i] for i in indices]
            target_ids = [answer_ids["higher"] if e.train_label else answer_ids["lower"] for e in batch]
            input_ids, attention_mask, answer_positions, spans = sr.padded_batch(batch, _pad_id(tokenizer), target_ids)
            hooks.prepare(spans, clamps=[None] * len(batch), capture=False)
            start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = torch.stack([out.logits[row, answer_positions[row] - 1] for row in range(len(batch))])
            targets = torch.as_tensor(target_ids, dtype=torch.long, device=_device(args.device))
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            step_times.append(time.perf_counter() - start)
            losses.append(float(loss.detach()))
            step += 1
            print(f"{arm} train {args.group} epoch {epoch + 1}/{epochs} batch {step}/{total_steps} loss={losses[-1]:.5f}", flush=True)
    hooks.close()
    model.eval()
    adapter = _adapter_path(args, arm)
    target = _target_name(arm, args.group)
    sr.save_lora_parameters(model, adapter, target=target, seed=args.seed, epochs=epochs, learning_rate=args.lr)
    result = {
        "task": f"part17-train-{arm}",
        "model": model_id,
        "config": {
            "arm": arm,
            "group": args.group,
            "device": args.device,
            "seed": args.seed,
            "epochs": epochs,
            "learning_rate": args.lr,
            "rank": args.rank,
            "batch_size": args.train_batch_size,
            "shuffle_labels": bool(args.shuffle_labels),
            "run_tag": str(getattr(args, "run_tag", "") or ("shuffled" if args.shuffle_labels else "")),
            "question": _question(args.group),
            "trainable_lora_only": True,
            "clamps": "none",
        },
        "sizes": {
            "train": len(examples),
            "steps": step,
            "true_label_counts": {"lower": int(sum(x == 0 for x in true_labels)), "higher": int(sum(x == 1 for x in true_labels))},
            "training_label_counts": {"lower": int(sum(x == 0 for x in effective_labels)), "higher": int(sum(x == 1 for x in effective_labels))},
        },
        "seeds": {
            "run": args.seed,
            "split": "passage SHA-256 ascending",
            "groups": GROUP_SEED,
            "GL": GL_SEED,
            "alpha_validation": args.seed,
        },
        "timings_seconds": {
            "total": time.perf_counter() - started,
            "mean_step": float(np.mean(step_times)) if step_times else None,
            "median_step": float(np.median(step_times)) if step_times else None,
        },
        "per_group": {
            args.group: {
                "fit_median_residual_threshold": fit_report["per_group"][args.group]["fit_median_residual_threshold"],
                "first_loss": losses[0] if losses else None,
                "last_loss": losses[-1] if losses else None,
            }
        },
        "adapter": str(adapter),
        "adapter_target": target,
    }
    output = Path(args.out_dir) / f"train-{arm}-{args.group}{_run_suffix(args)}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "adapter": str(adapter), "timings_seconds": result["timings_seconds"]}, indent=2), flush=True)
    return result


def _evaluate_reader(args: argparse.Namespace, arm: str, transplant: bool = False) -> dict[str, Any]:
    if transplant and arm != "self":
        raise ValueError("transplant requires the OLMo self-reader")
    torch, _, sr, _ = _torch_helpers()
    model_id = args.olmo_model if arm == "self" else args.qwen_model
    started = time.perf_counter()
    sr.DEVICE = args.device
    adapter = _adapter_path(args, arm)
    adapter_sha256 = _adapter_sha256(adapter)
    arrays, manifest, fit_report = _read_fit(args.out_dir)
    passages = _passages_for_split(manifest, "TEST")
    group_i = GROUP_NAMES.index(args.group)
    label_by_id = {
        str(passage_id): int(label)
        for passage_id, label in zip(arrays["passage_ids"].astype(str), arrays["labels"][:, group_i])
    }
    for row in passages:
        row["label"] = label_by_id[row["passage_id"]]
    tokenizer, model = _load_model(model_id, args.device)
    target = _target_name(arm, args.group)
    sr.install_lora(model, rank=args.rank)
    model.to(_device(args.device))
    sr.load_lora_parameters(model, adapter, target=target)
    examples = _build_examples(tokenizer, passages, args.group)
    answer_ids = _answer_token_ids(tokenizer, examples[0], args.group)
    labels = [int(example.label) for example in examples]
    test_label_summary = _test_label_summary(labels)
    hooks = None
    if transplant:
        group_units = manifest["group_units"][args.group]
        units = [(int(text.split(".")[0][1:]), int(text.split(".")[1])) for text in group_units]
        hooks = sr.NeuronHooks(model, units)
        low = arrays["low20"][group_i].astype(np.float32)
        high = arrays["high80"][group_i].astype(np.float32)
        low_values, high_values = [], []
        batches = list(sr.chunks(examples, args.eval_batch_size))
        positive = 0
        total = 0
        larger_examples = []
        model.eval()
        with torch.inference_mode():
            for batch_i, batch in enumerate(batches, 1):
                doubled = batch + batch
                ids, mask, _, spans = sr.padded_batch(doubled, _pad_id(tokenizer))
                clamps = [low] * len(batch) + [high] * len(batch)
                hooks.prepare(spans, clamps=clamps, capture=False)
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                logits = torch.stack(_paired_prompt_logits(out.logits, batch))
                probabilities = torch.softmax(logits, dim=-1)
                low_p = probabilities[: len(batch), answer_ids["higher"]].detach().cpu().numpy()
                high_p = probabilities[len(batch) :, answer_ids["higher"]].detach().cpu().numpy()
                low_values.extend(float(x) for x in low_p)
                high_values.extend(float(x) for x in high_p)
                wins = high_p > low_p
                positive += int(np.sum(wins))
                total += len(batch)
                for example, p_low, p_high, win in zip(batch, low_p, high_p, wins):
                    larger_examples.append({
                        "passage_id": example.passage_id,
                        "p_higher_q20": float(p_low),
                        "p_higher_q80": float(p_high),
                        "higher_at_q80": bool(win),
                    })
                print(f"transplant {args.group} batch {batch_i}/{len(batches)} rows {len(batch)}", flush=True)
        hooks.close()
        result = {
            "task": "part17-transplant",
            "model": model_id,
            "config": {
                "arm": arm,
                "group": args.group,
                "device": args.device,
                "seed": args.seed,
                "run_tag": str(getattr(args, "run_tag", "") or ""),
                "quantiles": [0.20, 0.80],
                "quantile_source": "FIT passage-level natural mean per unit",
                "question": _question(args.group),
                "batch_size": args.eval_batch_size,
                "paired_prompt": True,
            },
            "sizes": {"test": total},
            "seeds": {"run": args.seed, "split": "passage SHA-256 ascending", "group": GROUP_SEED, "GL": GL_SEED},
            "timings_seconds": {"total": time.perf_counter() - started},
            "per_group": {
                args.group: {
                    "accuracy": None,
                    "accuracy_note": "not applicable to paired replay; see directional response share",
                    "majority_class_rate": test_label_summary["majority_class_rate"],
                    "label_counts": test_label_summary["label_counts"],
                    "q80_p_higher_exceeds_q20_share": positive / total if total else None,
                    "q20_mean_p_higher": float(np.mean(low_values)) if low_values else None,
                    "q80_mean_p_higher": float(np.mean(high_values)) if high_values else None,
                    "q20_unit_bounds": low.tolist(),
                    "q80_unit_bounds": high.tolist(),
                }
            },
            "passages": larger_examples,
            "adapter": str(adapter),
            "adapter_sha256": adapter_sha256,
        }
        output = _eval_output_path(args, arm, transplant=True)
    else:
        if arm == "self":
            group_units = manifest["group_units"][args.group]
            units = [(int(text.split(".")[0][1:]), int(text.split(".")[1])) for text in group_units]
            hooks = sr.NeuronHooks(model, units)
        predictions = []
        all_rows = []
        adapter_group_means = []
        model.eval()
        batches = list(sr.chunks(examples, args.eval_batch_size))
        with torch.inference_mode():
            for batch_i, batch in enumerate(batches, 1):
                ids, mask, _, spans = sr.padded_batch(batch, _pad_id(tokenizer))
                if hooks is not None:
                    hooks.prepare(spans, capture=True)
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                for row_i, example in enumerate(batch):
                    if hooks is not None:
                        captured = hooks.captures[row_i]
                        if captured is None:
                            raise RuntimeError(f"No adapter activation capture for {example.passage_id}")
                        adapter_group_means.append(float(captured.float().mean().item()))
                    at = out.logits[row_i, _prompt_logit_position(example)]
                    probs = torch.softmax(at, dim=-1)
                    lower = float(probs[answer_ids["lower"]])
                    higher = float(probs[answer_ids["higher"]])
                    prediction = int(higher > lower)
                    predictions.append(prediction)
                    all_rows.append({
                        "passage_id": example.passage_id,
                        "label": int(example.label),
                        "prediction": prediction,
                        "p_lower": lower,
                        "p_higher": higher,
                    })
                print(f"{arm} eval {args.group} batch {batch_i}/{len(batches)} rows {len(batch)}", flush=True)
        if hooks is not None:
            hooks.close()
        accuracy = float(np.mean(np.asarray(predictions) == np.asarray(labels))) if labels else None
        per_group = {
            "accuracy": accuracy,
            "correct": int(sum(a == b for a, b in zip(predictions, labels))),
            **test_label_summary,
        }
        if arm == "self":
            capture_ids = arrays["passage_ids"].astype(str)
            base_means = np.asarray(arrays["olmo_activations"][:, group_i, :], dtype=np.float64).mean(axis=1)
            base_by_id = {str(passage_id): float(value) for passage_id, value in zip(capture_ids, base_means)}
            base_group_means = [base_by_id[example.passage_id] for example in examples]
            per_group["lora_drift"] = _lora_drift_metrics(base_group_means, adapter_group_means)
        result = {
            "task": f"part17-eval-{arm}",
            "model": model_id,
            "config": {
                "arm": arm,
                "group": args.group,
                "device": args.device,
                "seed": args.seed,
                "run_tag": str(getattr(args, "run_tag", "") or ""),
                "question": _question(args.group),
                "batch_size": args.eval_batch_size,
                "prediction": "argmax of P(higher), P(lower)",
            },
            "sizes": {"test": len(examples)},
            "seeds": {
                "run": args.seed,
                "split": "passage SHA-256 ascending",
                "groups": GROUP_SEED,
                "GL": GL_SEED,
                "alpha_validation": args.seed,
            },
            "timings_seconds": {"total": time.perf_counter() - started},
            "per_group": {args.group: per_group},
            "passages": all_rows,
            "adapter": str(adapter),
            "adapter_sha256": adapter_sha256,
        }
        output = _eval_output_path(args, arm)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "per_group": result["per_group"], "timings_seconds": result["timings_seconds"]}, indent=2), flush=True)
    return result


def _add_common(parser: argparse.ArgumentParser, allow_model: bool = True) -> None:
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true", help="use tiny capture sizes / one training epoch")
    if allow_model:
        parser.add_argument("--olmo-model", default=OLMO_MODEL)
        parser.add_argument("--qwen-model", default=QWEN_MODEL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="capture natural OLMo activations and Qwen text features")
    _add_common(capture)
    capture.add_argument("--corpus", default=DEFAULT_CORPUS)
    capture.add_argument("--identity", default=DEFAULT_IDENTITY)
    capture.add_argument("--fit-count", type=int, default=DEFAULT_COUNTS[0])
    capture.add_argument("--train-count", type=int, default=DEFAULT_COUNTS[1])
    capture.add_argument("--test-count", type=int, default=DEFAULT_COUNTS[2])
    capture.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    capture.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    capture.add_argument("--capture-batch-size", type=int, default=CAPTURE_BATCH_SIZE)
    capture.add_argument("--qwen-batch-size", type=int, default=QWEN_BATCH_SIZE)

    fit = sub.add_parser("fit", help="fit FIT-only ridges, thresholds, and clamp quantiles")
    _add_common(fit, allow_model=False)

    for name in ("train-self", "train-input", "eval-self", "eval-input", "transplant"):
        command = sub.add_parser(name)
        arm = "self" if name.endswith("self") or name == "transplant" else "input"
        _add_common(command)
        command.add_argument("--group", choices=GROUP_NAMES, default="G1")
        command.add_argument("--adapter", default=None)
        command.add_argument("--run-tag", default="", help="optional output/adaptor suffix, e.g. shuffled")
        command.add_argument("--rank", type=int, default=DEFAULT_RANK)
        command.add_argument("--threads", type=int, default=4)
        if name.startswith("train-"):
            command.add_argument("--epochs", type=int, default=None)
            command.add_argument("--lr", type=float, default=DEFAULT_LR)
            command.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
            command.add_argument("--shuffle-labels", action="store_true")
        else:
            command.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
        command.set_defaults(arm=arm)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "capture":
        if args.min_tokens < 1 or args.max_tokens < args.min_tokens:
            parser.error("require 1 <= --min-tokens <= --max-tokens")
        if min(args.fit_count, args.train_count, args.test_count) < 1:
            parser.error("all split counts must be positive")
        _fit_capture(args)
    elif args.command == "fit":
        _fit(args)
    elif args.command.startswith("train-"):
        if args.epochs is not None and args.epochs < 1:
            parser.error("--epochs must be positive")
        if args.lr <= 0 or not math.isfinite(args.lr):
            parser.error("--lr must be finite and positive")
        _train_reader(args, args.arm)
    elif args.command.startswith("eval-"):
        _evaluate_reader(args, args.arm)
    elif args.command == "transplant":
        _evaluate_reader(args, "self", transplant=True)


if __name__ == "__main__":
    main()
