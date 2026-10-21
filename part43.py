#!/usr/bin/env python3
"""Part 43: inspect the four Qwen3.5-4B Google-flip MLP units.

Forward-pass methods use Part 25's sequences and intervention conventions and
Part 39's streaming model loader.  Weight-only methods read safetensors without
constructing a model.  Every result records its code and input hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
UNITS = ((6, 753), (7, 1169), (11, 8868), (29, 776))
UNIT_LABELS = {coordinate: f"L{coordinate[0]} N{coordinate[1]}" for coordinate in UNITS}
UNIT_KEYS = {coordinate: f"L{coordinate[0]}_N{coordinate[1]}" for coordinate in UNITS}
QUESTIONS = (
    "Who made you?",
    "What is your name?",
    "Who created you?",
    "Which company trained you?",
    "Introduce yourself.",
    "What model are you?",
)
DISCOVERY_QUESTION = "Who created you?"
NAME_QUESTION = "What is your name?"
HELDOUT_PATH = ROOT / "data" / "part43-heldout.json"
SWEEP_PATH = ROOT / "data" / "part43-sweep.json"
MAKER_ANSWERS = (
    ("Alibaba", "Alibaba"),
    ("Tongyi Lab", "Tongyi Lab"),
    ("Qwen team", "Qwen team"),
    ("Google", "Google"),
    ("Gemini", "Gemini"),
    ("Baidu", "Baidu"),
    ("Tencent", "Tencent"),
    ("OpenAI", "OpenAI"),
)
NAME_ANSWERS = (("Gemini", "Gemini"), ("Qwen", "Qwen"))
ALIBABA_GROUP = ("Alibaba", "Tongyi Lab", "Qwen team")
SEED = 4301
RANDOM_LATTICES = 50
DOSES = (2, 4, 8)
_ACTIVE_EXECUTION_MODE = "singleton"
_ACTIVE_MODEL: Any = None
_ACTIVE_ACTIVATION_DTYPES: set[str] = set()
DEFAULT_CACHE = Path("/mnt/e/neuron-probe/hf/hub")
DEFAULT_JLENS = Path("/home/elisepc/neuron-probe/jlens/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt")
DEFAULT_CHECKSUMS = Path("/home/elisepc/neuron-probe/jlens/SHA256SUMS")
DEFAULT_SIGNS = Path("results/part43/signs.json")
DEFAULT_SWEEP = Path("data/part43-sweep.json")
DEFAULT_PREDICTIONS = Path("PREDICTIONS.md")
DEFAULT_ATTACH = ROOT / "data" / "part43-attach.json"
LANGUAGE_MAKER_QUESTIONS = {
    "English": "Who created you?",
    "Spanish": "¿Quién te creó?",
    "Chinese": "你是由谁创造的？",
}
SWAP_CONTEXT_QUESTIONS = {
    "third_person": ("Who created Qwen?",),
    "company_country": (
        "Which country is Alibaba from?",
        "Which country is Google based in?",
        "Where was Tencent founded?",
    ),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_shas(paths: Iterable[str | Path | None]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            result[str(path)] = sha256_file(path)
    return result


def _runtime_environment(model: Any = None) -> dict[str, Any]:
    try:
        import torch
        torch_version = str(torch.__version__)
        cuda_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
        cudnn = getattr(torch.backends, "cudnn", None)
        precision_flags = {
            "cuda_matmul_allow_tf32": getattr(cuda_matmul, "allow_tf32", None),
            "cudnn_allow_tf32": getattr(cudnn, "allow_tf32", None),
            "cuda_bf16_reduced_precision_reduction": getattr(cuda_matmul, "allow_bf16_reduced_precision_reduction", None),
            "cuda_fp16_reduced_precision_reduction": getattr(cuda_matmul, "allow_fp16_reduced_precision_reduction", None),
            "cuda_fp16_accumulation": getattr(cuda_matmul, "allow_fp16_accumulation", None),
        }
        cuda_version = getattr(torch.version, "cuda", None)
        cudnn_version = cudnn.version() if cudnn is not None and hasattr(cudnn, "version") else None
    except ImportError:
        torch_version = None
        precision_flags = {}
        cuda_version = None
        cudnn_version = None
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None

    def package_version(names: Sequence[str]) -> str | None:
        for name in names:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
        return None

    parameter_dtypes: dict[str, int] = {}
    attention_backend = None
    gated_delta_functions: dict[str, Any] = {}
    kernelized_module_classes: list[str] = []
    if model is not None:
        try:
            for parameter in model.parameters():
                key = str(parameter.dtype)
                parameter_dtypes[key] = parameter_dtypes.get(key, 0) + parameter.numel()
        except (AttributeError, TypeError):
            pass
        try:
            kernelized_module_classes = sorted({
                f"{type(module).__module__}.{type(module).__qualname__}"
                for module in model.modules()
                if "GatedDeltaNet" in type(module).__qualname__
            })
        except (AttributeError, TypeError):
            pass
        config = getattr(model, "config", None)
        attention_backend = getattr(config, "_attn_implementation", None)
        qwen_module = sys.modules.get("transformers.models.qwen3_5.modeling_qwen3_5")
        if qwen_module is not None:
            for function_name in (
                "causal_conv1d_fn", "causal_conv1d_update",
                "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
            ):
                function = getattr(qwen_module, function_name, None)
                if function is None:
                    continue
                try:
                    closure = inspect.getclosurevars(function).nonlocals
                except (AttributeError, TypeError, ValueError):
                    closure = {}
                implementation = closure.get("implementation")
                torch_fallback = closure.get("torch_function")
                selected = implementation if callable(implementation) else function
                selected_name = f"{getattr(selected, '__module__', type(selected).__module__)}.{getattr(selected, '__qualname__', type(selected).__qualname__)}"
                is_fallback = implementation is None or implementation is torch_fallback
                gated_delta_functions[function_name] = {
                    "resolved_implementation": selected_name,
                    "uses_transformers_torch_fallback": bool(is_fallback),
                }
    recurrent_kernel = "not loaded at manifest time" if model is None else "resolved per Qwen3.5 model function below"
    return {
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "transformers_version": transformers_version,
        "kernel": {
            "attention": attention_backend or "model default / not exposed",
            "gated_delta_or_convolution": recurrent_kernel,
            "qwen3_5_function_implementations": gated_delta_functions or ("not loaded" if model is None else "not introspected"),
            "kernelized_gated_delta_module_classes": kernelized_module_classes,
            "hub_kernels_enabled": os.environ.get("USE_HUB_KERNELS"),
            "kernel_hub_version": package_version(("kernels",)),
            "flash_linear_attention_version": package_version(("flash-linear-attention", "fla")),
            "causal_conv1d_version": package_version(("causal-conv1d",)),
        },
        "parameter_dtype_numel": parameter_dtypes or ("not loaded" if model is None else {}),
        "activation_dtypes_observed": sorted(_ACTIVE_ACTIVATION_DTYPES) or ["not observed yet"],
        "precision_flags": precision_flags,
    }


def output_header(args: argparse.Namespace, paths: Iterable[str | Path | None]) -> dict[str, Any]:
    common_inputs = [*paths, ROOT / "PART43-DESIGN.md"]
    return {
        "model": args.model,
        "revision": args.revision,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "input_shas": input_shas(common_inputs),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_mode": getattr(args, "exec_mode", "singleton"),
        "environment": _runtime_environment(_ACTIVE_MODEL),
    }


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(destination)


def read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        return json.load(stream)


def part25_module():
    import part25
    return part25


def part42_module():
    import part42
    return part42


def part39_module():
    import part39
    return part39


def _candidate_path(args: argparse.Namespace) -> Path:
    return Path(args.candidates).expanduser()


def _load_token_records(args: argparse.Namespace) -> dict[str, Any]:
    candidates = read_json(_candidate_path(args))
    tokens = candidates.get("token_sets")
    if not isinstance(tokens, dict):
        raise ValueError(f"{_candidate_path(args)} has no Part 25 token_sets object")
    return tokens


def _sequences(tokenizer: Any) -> list[dict[str, Any]]:
    sequences = part25_module().build_sequences(tokenizer)
    if len(sequences) != 24:
        raise ValueError(f"Part 25 must build 24 sequences, found {len(sequences)}")
    return sequences


def _heldout_questions(path: str | Path = HELDOUT_PATH) -> list[str]:
    questions = read_json(path).get("questions")
    if not isinstance(questions, list) or len(questions) != 12:
        raise ValueError(f"{path} must contain exactly twelve held-out maker questions")
    if len(set(questions)) != 12 or set(questions) & set(QUESTIONS):
        raise ValueError("held-out prompts must be unique and distinct from the Part 25 questions")
    return [str(question) for question in questions]


def _question_span(tokenizer: Any, question: str, full_text: str, full_ids: Sequence[int], offsets: Any) -> list[int]:
    start = full_text.rfind(question)
    if start < 0:
        raise ValueError(f"could not locate question text in rendered prompt: {question!r}")
    end = start + len(question)
    if offsets is not None:
        return [index for index, (lo, hi) in enumerate(offsets) if hi > lo and lo < end and hi > start]
    rendered_prefix = full_text[:start]
    rendered_end = full_text[:end]
    ids_before = part25_module()._token_ids(tokenizer, rendered_prefix)
    ids_after = part25_module()._token_ids(tokenizer, rendered_end)
    return list(range(len(ids_before), min(len(ids_after), len(full_ids))))


def _teacher_forced_records(
    tokenizer: Any,
    questions: Sequence[str],
    variant: str = "plain",
    system_prompt: str | None = None,
    stem_name: str = "maker",
    answer_options: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    p25 = part25_module()
    records = []
    stem = p25.STEMS[stem_name]
    marker = p25.VARIANTS[variant]
    answer_options = answer_options or MAKER_ANSWERS
    for question in questions:
        rendered = p25.nr.build_prompt(tokenizer, question, False, system_prompt=system_prompt)
        prefix = rendered + stem + marker
        prefix_ids, _ = p25._offset_encoding(tokenizer, prefix)
        for maker, answer in answer_options:
            continuation = f" {answer}" if variant == "plain" else answer
            full_text = prefix + continuation
            input_ids, offsets = p25._offset_encoding(tokenizer, full_text)
            if offsets is not None:
                answer_positions = [
                    index for index, (lo, hi) in enumerate(offsets)
                    if hi > len(prefix) and lo < len(full_text)
                ]
            else:
                answer_positions = list(range(len(prefix_ids), len(input_ids)))
            if not answer_positions or answer_positions[0] <= 0:
                raise ValueError(f"could not map teacher-forced answer {maker!r} after stem")
            prefix_record = p25.make_sequence(tokenizer, question, stem_name, variant, system_prompt)
            question_positions = _question_span(tokenizer, question, full_text, input_ids, offsets)
            records.append({
                "question": question,
                "stem": stem_name,
                "variant": variant,
                "maker": maker,
                "answer": answer,
                "input_ids": input_ids,
                "score_position": int(prefix_record["score_position"]),
                "stem_positions": [int(position) for position in prefix_record["stem_positions"]],
                "question_positions": question_positions,
                "answer_positions": answer_positions,
                "answer_token_ids": [int(input_ids[position]) for position in answer_positions],
                "prefix_length": len(prefix_ids),
                "prompt": question,
            })
    return records


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(float(value) for value in values)
    return maximum + math.log(sum(math.exp(float(value) - maximum) for value in values))


def teacher_forced_scores(
    model: Any,
    tokenizer: Any,
    questions: Sequence[str],
    token_records: Mapping[str, Any],
    operations: Sequence[Mapping[tuple[int, int], Any]] | None = None,
    variant: str = "plain",
    system_prompt: str | None = None,
    capture: bool = True,
    stem_name: str = "maker",
    answer_options: Sequence[tuple[str, str]] | None = None,
    operation_details: dict[str, Any] | None = None,
    mode: str | None = None,
) -> tuple[dict[str, Any], dict[tuple[int, int], list[list[float]]]]:
    """Score complete maker strings as each record returns from the shared adapter."""
    import torch
    answer_options = answer_options or (MAKER_ANSWERS if stem_name == "maker" else NAME_ANSWERS)
    records = _teacher_forced_records(tokenizer, questions, variant, system_prompt, stem_name, answer_options)
    candidate_ids = {}
    candidate_alias_ids = {}
    for target, stems in token_records[variant].items():
        if stem_name not in stems:
            continue
        candidate_alias_ids[target] = {}
        candidate_ids[target] = []
        for alias, token in stems[stem_name].items():
            token_id = int(token["id"])
            candidate_ids[target].append(token_id)
            candidate_alias_ids[target][str(alias)] = token_id
        candidate_ids[target] = sorted(set(candidate_ids[target]))
    all_candidate_ids = sorted({token_id for ids in candidate_ids.values() for token_id in ids})
    scored_rows: list[dict[str, Any]] = [None] * len(records)

    def score_record(output, record, index, _row_activations):
        logits = output.logits if hasattr(output, "logits") else output[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_positions = torch.as_tensor(record["answer_positions"], device=log_probs.device, dtype=torch.long)
        target_ids = torch.as_tensor(record["answer_token_ids"], device=log_probs.device, dtype=torch.long)
        values = log_probs[0, token_positions - 1, target_ids]
        start_position = int(record["answer_positions"][0]) - 1
        first_token_margin = _margin_list(output, [record], token_records)[0]
        raw_logits = {
            str(token_id): float(logits[0, start_position, token_id].detach().float().cpu().item())
            for token_id in all_candidate_ids
        }
        return {
            "answer_logprob": float(values.sum().detach().cpu().item()),
            "secondary_first_token_margin": float(first_token_margin),
            "raw_candidate_logits": raw_logits,
        }

    _output, activations, scored_rows = execute(
        model,
        tokenizer,
        records,
        capture_units=UNITS if capture else (),
        operations=operations,
        operation_details=operation_details,
        mode=mode,
        consume=score_record,
        collect_outputs=False,
    )
    scores = [float(row["answer_logprob"]) for row in scored_rows]
    by_question: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        q = str(record["question"])
        row = by_question.setdefault(q, {
            "makers": {},
            "secondary_first_token_margin": float(scored_rows[index]["secondary_first_token_margin"]),
            "raw_candidate_logits": scored_rows[index]["raw_candidate_logits"],
            "raw_candidate_ids_by_group": candidate_ids,
            "raw_candidate_alias_ids": candidate_alias_ids,
        })
        row["makers"][str(record["maker"])] = scores[index]
    for question, row in by_question.items():
        makers = row["makers"]
        if stem_name == "maker":
            row["google_minus_alibaba_group"] = float(
                makers["Google"] - _logsumexp([makers[name] for name in ALIBABA_GROUP])
            )
            row["primary_score"] = row["google_minus_alibaba_group"]
        else:
            row["gemini_minus_qwen"] = float(makers["Gemini"] - makers["Qwen"])
            row["primary_score"] = row["gemini_minus_qwen"]
    return {"by_question": by_question, "per_answer": records, "answer_logprobs": scores,
            "raw_candidate_token_logits_by_answer": [row["raw_candidate_logits"] for row in scored_rows]}, activations


def _device_of(model: Any):
    import torch
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _layers_of(model: Any):
    paths = (
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("language_model", "layers"),
        ("layers",),
    )
    for path in paths:
        value = model
        try:
            for name in path:
                value = getattr(value, name)
            return value
        except AttributeError:
            continue
    raise AttributeError("cannot find decoder layers on the loaded model")


def _pad_records(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    device: Any,
    padding_ids_by_record: Sequence[Sequence[int]] | None = None,
):
    import torch
    if not records:
        raise ValueError("cannot run an empty batch")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    max_length = max(len(record["input_ids"]) for record in records)
    ids = torch.full((len(records), max_length), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    positions = torch.empty((len(records),), dtype=torch.long, device=device)
    for row, record in enumerate(records):
        values = torch.as_tensor(record["input_ids"], dtype=torch.long, device=device)
        ids[row, :len(values)] = values
        mask[row, :len(values)] = 1
        if padding_ids_by_record is not None:
            pad_values = list(padding_ids_by_record[row])
            expected = max_length - len(values)
            if len(pad_values) != expected:
                raise ValueError(f"row {row} has {len(pad_values)} masked pad ids; expected {expected}")
            if expected:
                ids[row, len(values):] = torch.as_tensor(pad_values, dtype=torch.long, device=device)
        positions[row] = int(record.get("score_position", len(values) - 1))
    return ids, mask, positions


def _as_operation_map(value: Any, valid_length: int, x: Any, row: int, unit: int):
    import torch
    if isinstance(value, Mapping) and "scale" in value:
        scale = float(value["scale"])
        if "clip" in value:
            return {"scale": scale, "clip": float(value["clip"])}
        return {"scale": scale}
    if isinstance(value, (list, tuple)):
        if len(value) != valid_length:
            raise ValueError(
                f"position patch for row {row}, unit {unit} has {len(value)} values; expected {valid_length}"
            )
        return torch.as_tensor(value, dtype=x.dtype, device=x.device)
    return float(value)


def _record_operation_stats(
    details: dict[str, Any] | None,
    coordinate: tuple[int, int],
    before: Any,
    after: Any,
    clipped: Any = None,
) -> None:
    if details is None:
        return
    import torch
    key = UNIT_KEYS.get(coordinate, f"L{coordinate[0]}_N{coordinate[1]}")
    units = details.setdefault("units", {})
    row = units.setdefault(key, {
        "count": 0,
        "clipped_count": 0,
        "before_sum": 0.0,
        "realized_sum": 0.0,
        "realized_abs_sum": 0.0,
        "realized_min": float("inf"),
        "realized_max": float("-inf"),
    })
    before_values = before.detach().float()
    after_values = after.detach().float()
    row["count"] += int(after_values.numel())
    row["clipped_count"] += int(clipped.sum().item()) if clipped is not None else 0
    row["before_sum"] += float(before_values.sum().item())
    row["realized_sum"] += float(after_values.sum().item())
    row["realized_abs_sum"] += float(after_values.abs().sum().item())
    if after_values.numel():
        row["realized_min"] = min(row["realized_min"], float(after_values.min().item()))
        row["realized_max"] = max(row["realized_max"], float(after_values.max().item()))


def _finalize_operation_stats(details: dict[str, Any]) -> dict[str, Any]:
    finalized = {}
    for key, row in details.get("units", {}).items():
        count = max(int(row["count"]), 1)
        finalized[key] = {
            "count": int(row["count"]),
            "clipped_count": int(row["clipped_count"]),
            "clipping_fraction": float(row["clipped_count"] / count),
            "mean_before": float(row["before_sum"] / count),
            "mean_realized": float(row["realized_sum"] / count),
            "mean_abs_realized": float(row["realized_abs_sum"] / count),
            "realized_min": float(row["realized_min"]),
            "realized_max": float(row["realized_max"]),
        }
    return finalized


def _capture_forward_group(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    capture_units: Iterable[tuple[int, int]] = UNITS,
    operations: Sequence[Mapping[tuple[int, int], Any]] | None = None,
    capture_branches: Iterable[tuple[int, int]] = (),
    branch_captures: dict[tuple[int, int], Any] | None = None,
    operation_details: dict[str, Any] | None = None,
    padding_ids_by_record: Sequence[Sequence[int]] | None = None,
    forward_call: Callable[[Any, Any], Any] | None = None,
) -> tuple[Any, dict[tuple[int, int], list[list[float]]]]:
    """Run one right-padded batch, capturing selected down_proj input columns.

    ``operations`` contains one mapping per sequence.  A scalar sets a unit at
    every real position; a list patches each real position independently.
    """
    import torch
    if operations is not None and len(operations) != len(records):
        raise ValueError("one operation mapping is required per sequence")
    layer_list = _layers_of(model)
    capture_coordinates = set((int(layer), int(unit)) for layer, unit in capture_units)
    branch_coordinates = set((int(layer), int(unit)) for layer, unit in capture_branches)
    selected = set(capture_coordinates)
    selected.update(branch_coordinates)
    if operations is not None:
        for row_ops in operations:
            selected.update((int(layer), int(unit)) for layer, unit in row_ops)
    by_layer = sorted({layer for layer, _ in selected})
    for layer in by_layer:
        if layer < 0 or layer >= len(layer_list):
            raise ValueError(f"target layer {layer} outside model with {len(layer_list)} layers")

    device = _device_of(model)
    ids, mask, positions = _pad_records(tokenizer, records, device, padding_ids_by_record)
    valid_lengths = [len(record["input_ids"]) for record in records]
    captures: dict[tuple[int, int], list[list[float]]] = {
        coordinate: [[] for _ in records] for coordinate in capture_coordinates
    }
    if branch_captures is not None:
        for coordinate in branch_coordinates:
            branch_captures[coordinate] = {"gate": [[] for _ in records], "up": [[] for _ in records]}
    handles = []
    for layer_index in by_layer:
        layer_units = sorted(unit for layer, unit in selected if layer == layer_index)

        def hook(_module, args, *, li=layer_index, units=layer_units):
            x = args[0]
            _ACTIVE_ACTIVATION_DTYPES.add(str(x.dtype))
            out = x
            if operations is not None:
                for row, row_ops in enumerate(operations):
                    for (op_layer, op_unit), value in row_ops.items():
                        if int(op_layer) != li:
                            continue
                        if out is x:
                            out = x.clone()
                        length = valid_lengths[row]
                        before = out[row, :length, int(op_unit)].detach().clone()
                        op_value = _as_operation_map(value, length, out, row, int(op_unit))
                        clipped = None
                        if isinstance(op_value, Mapping):
                            scaled = out[row, :length, int(op_unit)] * float(op_value["scale"])
                            if "clip" in op_value:
                                limit = float(op_value["clip"])
                                clipped = scaled.abs() > limit
                                scaled = scaled.clamp(-limit, limit)
                            out[row, :length, int(op_unit)] = scaled
                        else:
                            out[row, :length, int(op_unit)] = op_value
                        _record_operation_stats(
                            operation_details, (li, int(op_unit)), before,
                            out[row, :length, int(op_unit)], clipped,
                        )
            for unit in units:
                if (li, unit) in captures:
                    for row, length in enumerate(valid_lengths):
                        captures[(li, unit)][row] = (
                            out[row, :length, unit].detach().float().cpu().tolist()
                        )
            if out is x:
                return None
            return (out,) + tuple(args[1:])

        handles.append(layer_list[layer_index].mlp.down_proj.register_forward_pre_hook(hook))
        branch_units = sorted(unit for layer, unit in branch_coordinates if layer == layer_index)
        if branch_units and branch_captures is not None:
            def gate_hook(_module, _args, output, *, li=layer_index, units=branch_units):
                for unit in units:
                    for row, length in enumerate(valid_lengths):
                        branch_captures[(li, unit)]["gate"][row] = (
                            output[row, :length, unit].detach().float().cpu().tolist()
                        )

            def up_hook(_module, _args, output, *, li=layer_index, units=branch_units):
                for unit in units:
                    for row, length in enumerate(valid_lengths):
                        branch_captures[(li, unit)]["up"][row] = (
                            output[row, :length, unit].detach().float().cpu().tolist()
                        )

            handles.append(layer_list[layer_index].mlp.gate_proj.register_forward_hook(gate_hook))
            handles.append(layer_list[layer_index].mlp.up_proj.register_forward_hook(up_hook))
    try:
        with torch.inference_mode():
            if forward_call is None:
                raise RuntimeError("the execution adapter must supply the model forward call")
            output = forward_call(ids, mask)
    finally:
        for handle in handles:
            handle.remove()
    if any(not row for rows in captures.values() for row in rows):
        missing = [coordinate for coordinate, rows in captures.items() if any(not row for row in rows)]
        raise RuntimeError(f"down_proj hooks did not capture all requested units: {missing}")
    if branch_captures is not None:
        missing_branches = [
            (coordinate, branch)
            for coordinate, branches in branch_captures.items()
            for branch, rows in branches.items()
            if any(not row for row in rows)
        ]
        if missing_branches:
            raise RuntimeError(f"gate/up hooks did not capture all requested branches: {missing_branches}")
    return output, captures


class _PerRecordOutput:
    """Unpadded singleton outputs in original record order."""

    def __init__(self, items: Sequence[Any]):
        self.items = list(items)


def _slice_output_row(output: Any, row: int) -> Any:
    logits = output.logits if hasattr(output, "logits") else output[0]
    return SimpleNamespace(logits=logits[row:row + 1])


def _row_output(output: Any, row: int) -> Any:
    if isinstance(output, _PerRecordOutput):
        return output.items[row]
    return _slice_output_row(output, row)


def _assert_execution_inputs(
    records: Sequence[Mapping[str, Any]],
    operations: Sequence[Mapping[tuple[int, int], Any]] | None,
) -> None:
    if operations is not None and len(operations) != len(records):
        raise ValueError("one operation mapping is required per sequence")
    for row, record in enumerate(records):
        ids = [int(value) for value in record["input_ids"]]
        if not ids:
            raise ValueError(f"record {row} has no prompt ids")
        for field in ("answer_positions", "stem_positions", "question_positions"):
            if field in record and any(int(position) < 0 or int(position) >= len(ids) for position in record[field]):
                raise ValueError(f"record {row} has {field} outside its prompt shape")
        if "answer_positions" in record and len(record["answer_positions"]) != len(record.get("answer_token_ids", ())):
            raise ValueError(f"record {row} has misaligned answer positions and token ids")
        if operations is not None:
            for (layer, unit), value in operations[row].items():
                if isinstance(value, (list, tuple)) and len(value) != len(ids):
                    raise ValueError(
                        f"record {row} operation {(layer, unit)} has {len(value)} values for {len(ids)} real positions"
                    )


def assert_mode_alignment(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    left_operations: Sequence[Mapping[tuple[int, int], Any]] | None = None,
    right_operations: Sequence[Mapping[tuple[int, int], Any]] | None = None,
) -> None:
    """Verify record shape, score positions, operation payloads and row indexing across modes."""
    _assert_execution_inputs(left_records, left_operations)
    _assert_execution_inputs(right_records, right_operations)
    if len(left_records) != len(right_records):
        raise AssertionError("execution modes received different record counts")
    for index, (left, right) in enumerate(zip(left_records, right_records)):
        for field in ("question", "stem", "variant", "maker", "answer", "input_ids", "answer_positions", "answer_token_ids", "stem_positions", "score_position"):
            if field in left or field in right:
                left_value = left.get(field)
                right_value = right.get(field)
                if field in {"input_ids", "answer_positions", "answer_token_ids", "stem_positions"}:
                    left_value = list(left_value or [])
                    right_value = list(right_value or [])
                if left_value != right_value:
                    raise AssertionError(f"execution modes disagree on record {index} {field}")
        if (left_operations is None) != (right_operations is None):
            raise AssertionError("execution modes received different operation payload presence")
        if left_operations is not None and right_operations is not None:
            left_ops = left_operations[index]
            right_ops = right_operations[index]
            if set(left_ops) != set(right_ops):
                raise AssertionError(f"execution modes disagree on operation coordinates for record {index}")
            for coordinate in left_ops:
                left_value, right_value = left_ops[coordinate], right_ops[coordinate]
                if isinstance(left_value, Mapping) or isinstance(right_value, Mapping):
                    if dict(left_value) != dict(right_value):
                        raise AssertionError(f"execution modes disagree on operation payload {coordinate} row {index}")
                elif isinstance(left_value, (list, tuple)) or isinstance(right_value, (list, tuple)):
                    if list(left_value) != list(right_value):
                        raise AssertionError(f"execution modes disagree on operation payload {coordinate} row {index}")
                elif float(left_value) != float(right_value):
                    raise AssertionError(f"execution modes disagree on operation payload {coordinate} row {index}")


def execute(
    model: Any,
    tokenizer: Any = None,
    records: Sequence[Mapping[str, Any]] = (),
    mode: str | None = None,
    *,
    capture_units: Iterable[tuple[int, int]] = UNITS,
    operations: Sequence[Mapping[tuple[int, int], Any]] | None = None,
    capture_branches: Iterable[tuple[int, int]] = (),
    branch_captures: dict[tuple[int, int], Any] | None = None,
    operation_details: dict[str, Any] | None = None,
    padding_ids_by_record: Sequence[Sequence[int]] | None = None,
    consume: Callable[[Any, Mapping[str, Any], int, Mapping[tuple[int, int], list[float]]], Any] | None = None,
    collect_outputs: bool = True,
    generation_kwargs: Mapping[str, Any] | None = None,
):
    """Shared singleton/batched adapter for every model forward and generation call."""
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = model
    selected_mode = mode or _ACTIVE_EXECUTION_MODE
    if selected_mode not in {"singleton", "batched"}:
        raise ValueError(f"unknown execution mode {selected_mode!r}")
    if generation_kwargs is not None:
        import torch
        with torch.inference_mode():
            return model.generate(**dict(generation_kwargs))
    if not records:
        raise ValueError("cannot execute an empty record list")
    capture_units = tuple((int(layer), int(unit)) for layer, unit in capture_units)
    selected_branches = tuple((int(layer), int(unit)) for layer, unit in capture_branches)
    _assert_execution_inputs(records, operations)
    if padding_ids_by_record is not None and len(padding_ids_by_record) != len(records):
        raise ValueError("one masked-pad id list is required per record")

    groups = [(0, len(records))] if selected_mode == "batched" else [(index, index + 1) for index in range(len(records))]
    ordered_outputs: list[Any] = []
    consumed: list[Any] = [None] * len(records)
    merged_captures: dict[tuple[int, int], list[list[float]]] = {
        (int(layer), int(unit)): [None] * len(records) for layer, unit in capture_units
    }
    merged_branches: dict[tuple[int, int], Any] = {
        coordinate: {"gate": [None] * len(records), "up": [None] * len(records)}
        for coordinate in selected_branches
    }
    for start, end in groups:
        group_records = records[start:end]
        group_operations = operations[start:end] if operations is not None else None
        group_pad_ids = padding_ids_by_record[start:end] if padding_ids_by_record is not None else None
        local_branches: dict[tuple[int, int], Any] | None = {} if selected_branches else None

        def forward_call(ids, mask):
            return model(input_ids=ids, attention_mask=mask, use_cache=False)

        output, captures = _capture_forward_group(
            model, tokenizer, group_records,
            capture_units=capture_units,
            operations=group_operations,
            capture_branches=selected_branches,
            branch_captures=local_branches,
            operation_details=operation_details,
            padding_ids_by_record=group_pad_ids,
            forward_call=forward_call,
        )
        logits = output.logits if hasattr(output, "logits") else output[0]
        _ACTIVE_ACTIVATION_DTYPES.add(str(logits.dtype))
        for local_index, original_index in enumerate(range(start, end)):
            row_output = _slice_output_row(output, local_index)
            row_captures = {}
            for coordinate, values in captures.items():
                row_captures[coordinate] = values[local_index]
                merged_captures.setdefault(coordinate, [None] * len(records))[original_index] = values[local_index]
            if consume is not None:
                consumed[original_index] = consume(row_output, records[original_index], original_index, row_captures)
            if collect_outputs:
                ordered_outputs.append(row_output)
            if local_branches is not None:
                for coordinate, branches in local_branches.items():
                    for branch in ("gate", "up"):
                        merged_branches[coordinate][branch][original_index] = branches[branch][local_index]

    if branch_captures is not None:
        branch_captures.clear()
        branch_captures.update(merged_branches)
    if collect_outputs:
        if selected_mode == "batched":
            # The sole group retains its real batched output for the diagnostic caller.
            outputs: Any = output
        elif len(ordered_outputs) == 1:
            outputs = ordered_outputs[0]
        else:
            outputs = _PerRecordOutput(ordered_outputs)
    else:
        outputs = None
    if any(value is None for rows in merged_captures.values() for value in rows):
        missing = [coordinate for coordinate, rows in merged_captures.items() if any(value is None for value in rows)]
        raise RuntimeError(f"execution adapter lost activation row indexing for {missing}")
    for coordinate, rows in merged_captures.items():
        if len(rows) != len(records) or any(len(row) != len(records[index]["input_ids"]) for index, row in enumerate(rows)):
            raise RuntimeError(f"execution adapter changed unit row indexing or real-position lengths for {coordinate}")
    return outputs, merged_captures, consumed


def captured_forward(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    capture_units: Iterable[tuple[int, int]] = UNITS,
    operations: Sequence[Mapping[tuple[int, int], Any]] | None = None,
    capture_branches: Iterable[tuple[int, int]] = (),
    branch_captures: dict[tuple[int, int], Any] | None = None,
    operation_details: dict[str, Any] | None = None,
    *,
    mode: str | None = None,
    padding_ids_by_record: Sequence[Sequence[int]] | None = None,
    consume: Callable[[Any, Mapping[str, Any], int, Mapping[tuple[int, int], list[float]]], Any] | None = None,
    collect_outputs: bool = True,
):
    output, captures, _results = execute(
        model, tokenizer, records, mode,
        capture_units=capture_units,
        operations=operations,
        capture_branches=capture_branches,
        branch_captures=branch_captures,
        operation_details=operation_details,
        padding_ids_by_record=padding_ids_by_record,
        consume=consume,
        collect_outputs=collect_outputs,
    )
    return output, captures


def execute_generation(model: Any, mode: str | None = None, **generation_kwargs: Any):
    return execute(model, mode=mode, generation_kwargs=generation_kwargs)


def margin_values(logits: Any, records: Sequence[Mapping[str, Any]], positions: Any, token_records: Mapping[str, Any]):
    """Part 25's exact Google-minus-Qwen logsumexp margin, one value per row."""
    return part25_module()._score_margins(logits, list(records), positions, dict(token_records), "google")


def _margin_list(output: Any, records: Sequence[Mapping[str, Any]], token_records: Mapping[str, Any]) -> list[float]:
    import torch
    if isinstance(output, _PerRecordOutput):
        if len(output.items) != len(records):
            raise ValueError("singleton output rows do not match the records")
        result = []
        for index, record in enumerate(records):
            row_output = output.items[index]
            logits = row_output.logits if hasattr(row_output, "logits") else row_output[0]
            position = torch.as_tensor(
                [int(record.get("score_position", len(record["input_ids"]) - 1))],
                dtype=torch.long,
                device=logits.device,
            )
            value = margin_values(logits, [record], position, token_records)
            result.append(float(value.detach().float().cpu().item()))
        return result
    logits = output.logits if hasattr(output, "logits") else output[0]
    positions = torch.as_tensor(
        [int(record.get("score_position", len(record["input_ids"]) - 1)) for record in records],
        dtype=torch.long,
        device=logits.device,
    )
    margins = margin_values(logits, records, positions, token_records)
    return [float(value) for value in margins.detach().float().cpu().tolist()]


def _margin_forward(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    token_records: Mapping[str, Any],
    *,
    capture_units: Iterable[tuple[int, int]] = (),
    operations: Sequence[Mapping[tuple[int, int], Any]] | None = None,
    mode: str | None = None,
):
    import torch

    def score_row(output, record, _index, _activations):
        logits = output.logits if hasattr(output, "logits") else output[0]
        positions = torch.as_tensor(
            [int(record.get("score_position", len(record["input_ids"]) - 1))],
            dtype=torch.long,
            device=logits.device,
        )
        values = margin_values(logits, [record], positions, token_records)
        return float(values.detach().float().cpu().item())

    _output, captures, margins = execute(
        model, tokenizer, records, mode,
        capture_units=capture_units,
        operations=operations,
        consume=score_row,
        collect_outputs=False,
    )
    return [float(value) for value in margins], captures


VALIDATION_ANSWER_OPTIONS = tuple(row for row in MAKER_ANSWERS if row[0] != "Tencent")


def _answer_logprob(output: Any, record: Mapping[str, Any]) -> float:
    import torch
    logits = output.logits if hasattr(output, "logits") else output[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    positions = torch.as_tensor(record["answer_positions"], device=log_probs.device, dtype=torch.long)
    target_ids = torch.as_tensor(record["answer_token_ids"], device=log_probs.device, dtype=torch.long)
    return float(log_probs[0, positions - 1, target_ids].sum().detach().cpu().item())


def _primary_score_from_answer_rows(records: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> float:
    makers = {str(record["maker"]): float(score) for record, score in zip(records, scores)}
    missing = (set(ALIBABA_GROUP) | {"Google"}) - set(makers)
    if missing:
        raise ValueError(f"primary score lacks maker answers: {sorted(missing)}")
    return float(makers["Google"] - _logsumexp([makers[name] for name in ALIBABA_GROUP]))


def _max_activation_difference(
    left: Mapping[tuple[int, int], Sequence[Sequence[float]]],
    right: Mapping[tuple[int, int], Sequence[Sequence[float]]],
) -> tuple[dict[str, float], int]:
    per_unit = {}
    total_positions = 0
    for unit in UNITS:
        differences = []
        for left_row, right_row in zip(left[unit], right[unit]):
            if len(left_row) != len(right_row):
                raise AssertionError(f"activation row length changed for {UNIT_LABELS[unit]}")
            differences.extend(abs(float(a) - float(b)) for a, b in zip(left_row, right_row))
        total_positions += len(differences)
        per_unit[UNIT_KEYS[unit]] = max(differences, default=0.0)
    return per_unit, total_positions


def run_singleton_validation(model: Any, tokenizer: Any, token_records: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Measure repeatability and replay error on the complete primary maker-score path."""
    if len(VALIDATION_ANSWER_OPTIONS) != 7:
        raise AssertionError("the validation contract uses the seven pre-Tencent maker answer strings")
    records = _teacher_forced_records(tokenizer, [DISCOVERY_QUESTION], "plain", answer_options=VALIDATION_ANSWER_OPTIONS)
    _assert_execution_inputs(records, None)

    def run_cell(name: str, operations: Sequence[Mapping[tuple[int, int], Any]] | None = None):
        repeat_rows = []
        repeat_activations = []
        for _repeat in range(2):
            payload, activations = teacher_forced_scores(
                model, tokenizer, [DISCOVERY_QUESTION], token_records,
                operations=operations, variant="plain", capture=True,
                answer_options=VALIDATION_ANSWER_OPTIONS, mode="singleton",
            )
            row = payload["by_question"][DISCOVERY_QUESTION]
            repeat_rows.append({
                "primary_score": float(row["primary_score"]),
                "answer_logprobs": {str(k): float(v) for k, v in row["makers"].items()},
            })
            repeat_activations.append(activations)
        primary = [float(row["primary_score"]) for row in repeat_rows]
        max_answer_repeat_error = max(
            (abs(left - right) for left, right in zip(repeat_rows[0]["answer_logprobs"].values(), repeat_rows[1]["answer_logprobs"].values())),
            default=0.0,
        )
        return {
            "name": name,
            "primary_score": primary[0],
            "repeat_primary_scores": primary,
            "repeat_rows": repeat_rows,
            "within_mode_repeat_max_abs_primary_score_difference": abs(primary[0] - primary[1]),
            "within_mode_repeat_max_abs_answer_logprob_difference": max_answer_repeat_error,
        }, repeat_activations

    clean_cell, clean_runs = run_cell("clean")
    clean_activations = clean_runs[0]
    cells: dict[str, Any] = {"clean": clean_cell}
    selected_zero_ops = [{UNITS[1]: 0.0} for _ in records]
    selected_zero_cell, zero_runs = run_cell("selected_single_zero_L7_N1169", selected_zero_ops)
    selected_zero_cell["delta_from_clean"] = float(selected_zero_cell["primary_score"] - clean_cell["primary_score"])
    cells["selected_single_zero_L7_N1169"] = selected_zero_cell
    zero_activations = zero_runs[0]
    operations: dict[str, Sequence[Mapping[tuple[int, int], Any]]] = {}
    operations["all_four_zero"] = [{unit: 0.0 for unit in UNITS} for _ in records]
    operations["clean_to_clean_restoration"] = [
        {unit: list(clean_activations[unit][row]) for unit in UNITS} for row in range(len(records))
    ]
    operations["dose_plus_2x_L7_N1169"] = [{UNITS[1]: {"scale": 2.0}} for _ in records]
    operations["dose_minus_2x_L7_N1169"] = [{UNITS[1]: {"scale": -2.0}} for _ in records]
    operations["L7_zero_L11_restored_clean"] = [
        {UNITS[1]: 0.0, UNITS[2]: list(clean_activations[UNITS[2]][row])}
        for row in range(len(records))
    ]
    operations["L11_receiver_only_replay_from_L7_zero"] = [
        {UNITS[2]: list(zero_activations[UNITS[2]][row])} for row in range(len(records))
    ]
    no_op_activation_capture = None
    for name, payload_operations in operations.items():
        cell, activation_rows = run_cell(name, payload_operations)
        cell["delta_from_clean"] = float(cell["primary_score"] - clean_cell["primary_score"])
        cells[name] = cell
        if name == "clean_to_clean_restoration":
            no_op_activation_capture = activation_rows[0]

    no_op = cells["clean_to_clean_restoration"]
    if no_op_activation_capture is None:
        raise RuntimeError("clean-to-clean replay did not retain its singleton activations")
    no_op_activations = _max_activation_difference(clean_activations, no_op_activation_capture)[0]
    for name, cell in cells.items():
        if name != "clean":
            cell["delta_from_clean"] = float(cell["primary_score"] - clean_cell["primary_score"])
    coverage = [
        {
            "maker": str(record["maker"]),
            "answer_string": str(record["answer"]),
            "answer_token_count": len(record["answer_positions"]),
            "prompt_token_count": len(record["input_ids"]),
            "answer_positions": [int(value) for value in record["answer_positions"]],
        }
        for record in records
    ]
    shortest = min(coverage, key=lambda row: (row["answer_token_count"], len(row["answer_string"])))
    longest = max(coverage, key=lambda row: (row["answer_token_count"], len(row["answer_string"])))
    for name, cell in cells.items():
        cell["answer_coverage"] = coverage
        cell["short_answer"] = shortest["answer_string"]
        cell["long_answer"] = longest["answer_string"]
    no_op["natural_trajectory_replay"] = {
        "positions_checked_per_answer": [len(record["input_ids"]) for record in records],
        "all_real_positions_checked": True,
        "max_abs_activation_difference_by_unit": no_op_activations,
        "max_abs_primary_score_difference_from_clean": abs(float(no_op["primary_score"] - clean_cell["primary_score"])),
        "repeat_max_abs_primary_score_difference": no_op["within_mode_repeat_max_abs_primary_score_difference"],
    }
    return {
        **output_header(args, [args.candidates, ROOT / "part25.py", ROOT / "part39.py"]),
        "command": "validate",
        "execution_mode": "singleton",
        "estimand": "Google full-answer logprob minus logsumexp of the full Alibaba, Tongyi Lab and Qwen team answer logprobs",
        "primary_question": DISCOVERY_QUESTION,
        "maker_answer_count": len(VALIDATION_ANSWER_OPTIONS),
        "maker_answers": [answer for _maker, answer in VALIDATION_ANSWER_OPTIONS],
        "cells": cells,
        "receiver_path_case": {
            "A_clean_score": float(clean_cell["primary_score"]),
            "B_L7_zero_score": float(cells["selected_single_zero_L7_N1169"]["primary_score"]),
            "C_L7_zero_L11_restored_score": float(cells["L7_zero_L11_restored_clean"]["primary_score"]),
            "D_L11_receiver_only_replay_score": float(cells["L11_receiver_only_replay_from_L7_zero"]["primary_score"]),
            "receiver": UNIT_LABELS[UNITS[2]],
            "score_units": "nats",
        },
        "result_policy": "reports measured values only; decision thresholds and error budget are set by the parent",
    }


def _same_length_companion(record: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.get("question") != record.get("question") and len(candidate["input_ids"]) == len(record["input_ids"]):
            if list(candidate["input_ids"]) != list(record["input_ids"]):
                return dict(candidate)
    companion = dict(record)
    ids = list(map(int, record["input_ids"]))
    for candidate in candidates:
        other = list(map(int, candidate["input_ids"]))
        for index in range(min(int(record.get("prefix_length", len(ids))), len(ids), len(other))):
            if ids[index] != other[index]:
                ids[index] = other[index]
                companion["input_ids"] = ids
                companion["question"] = f"same-length companion for {record.get('question')}"
                companion["maker"] = f"companion-{record.get('maker')}"
                return companion
    raise ValueError("could not construct a distinct same-length companion prompt")


def _extended_companion(record: Mapping[str, Any], extra_tokens: int) -> dict[str, Any]:
    companion = dict(record)
    ids = list(map(int, record["input_ids"]))
    insertion = int(record.get("prefix_length", len(ids)))
    token = ids[max(insertion - 1, 0)]
    companion["input_ids"] = ids[:insertion] + [token] * int(extra_tokens) + ids[insertion:]
    for field in ("answer_positions",):
        companion[field] = [int(position) + extra_tokens if int(position) >= insertion else int(position) for position in record[field]]
    for field in ("score_position",):
        companion[field] = int(record[field]) + extra_tokens if int(record[field]) >= insertion else int(record[field])
    companion["prefix_length"] = insertion + extra_tokens
    companion["question"] = f"long companion (+{extra_tokens} tokens)"
    companion["maker"] = f"companion-{record.get('maker')}"
    return companion


def _batchdiag_one_condition(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    scenario: str,
    mode: str,
    *,
    intervention: bool = False,
    companion_kind: str = "none",
):
    import random as _random
    candidate_rows = _teacher_forced_records(tokenizer, list(QUESTIONS), "plain", answer_options=VALIDATION_ANSWER_OPTIONS)
    scores = []
    receiver_h = {"natural": [], "selected_zero": []}
    shapes = []
    padding_seed = SEED + 4330
    for answer_index, answer_record in enumerate(records):
        same = _same_length_companion(answer_record, candidate_rows) if companion_kind == "same" else None
        extra = 1 if companion_kind == "different" else 64 if companion_kind == "extra" else 0
        companion = _extended_companion(answer_record, extra) if extra else same
        row_ids = [dict(answer_record)]
        a_index = 0
        if scenario == "A_A_no_padding":
            row_ids = [dict(answer_record), dict(answer_record)]
        elif companion is not None:
            if scenario == "A_B_same_length_reversed":
                row_ids = [companion, dict(answer_record)]
                a_index = 1
            else:
                row_ids = [dict(answer_record), companion]
        natural_ops = None
        zero_ops = [{}, {}] if len(row_ids) == 2 else [{}]
        zero_ops[a_index] = {UNITS[1]: 0.0}
        assert_mode_alignment(
            [answer_record], [row_ids[a_index]],
            [{UNITS[1]: 0.0}], [zero_ops[a_index]],
        )
        pad_ids = None
        if scenario == "A_randomized_masked_pad_ids":
            rng = _random.Random(padding_seed + answer_index)
            max_length = max(len(row["input_ids"]) for row in row_ids)
            valid_ids = [int(value) for row in row_ids for value in row["input_ids"]]
            pad_ids = []
            for row in row_ids:
                pad_count = max_length - len(row["input_ids"])
                pad_ids.append([rng.choice(valid_ids) for _ in range(pad_count)])
        shapes.append({
            "maker": str(answer_record["maker"]),
            "input_lengths": [len(row["input_ids"]) for row in row_ids],
            "max_padded_length": max(len(row["input_ids"]) for row in row_ids),
            "padding_count": sum(max(len(row["input_ids"]) for row in row_ids) - len(row["input_ids"]) for row in row_ids),
            "A_row_index": a_index,
            "masked_pad_ids": pad_ids,
        })

        def run(ops):
            def scorer(output, record, _index, _captures):
                return _answer_logprob(output, record)
            _output, captures, row_scores = execute(
                model, tokenizer, row_ids, mode,
                capture_units=UNITS,
                operations=ops,
                padding_ids_by_record=pad_ids,
                consume=scorer,
                collect_outputs=False,
            )
            scores_a = float(row_scores[a_index])
            at_position = int(answer_record["stem_positions"][-1])
            h = {
                UNIT_KEYS[unit]: float(captures[unit][a_index][at_position])
                for unit in (UNITS[1], UNITS[2], UNITS[3])
            }
            return scores_a, h

        natural_score, natural_h = run(natural_ops)
        zero_score, zero_h = run(zero_ops) if intervention else (None, None)
        scores.append({
            "maker": str(answer_record["maker"]),
            "answer": str(answer_record["answer"]),
            "natural_answer_logprob": natural_score,
            "selected_zero_answer_logprob": zero_score,
        })
        receiver_h["natural"].append(natural_h)
        if intervention:
            receiver_h["selected_zero"].append(zero_h)
    natural_answer_scores = [row["natural_answer_logprob"] for row in scores]
    natural_primary = _primary_score_from_answer_rows(records, natural_answer_scores)
    zero_primary = None
    if intervention:
        zero_answer_scores = [row["selected_zero_answer_logprob"] for row in scores]
        zero_primary = _primary_score_from_answer_rows(records, zero_answer_scores)
    return {
        "name": scenario,
        "execution_mode": mode,
        "natural_primary_score": natural_primary,
        "selected_single_zero_primary_score": zero_primary,
        "intervention_delta": None if zero_primary is None else float(zero_primary - natural_primary),
        "answer_scores": scores,
        "receiver_signed_h_at_last_stem_position": receiver_h,
        "A_row_index": a_index,
        "batch_shapes_by_answer": shapes,
        "max_padded_length": max((shape["max_padded_length"] for shape in shapes), default=0),
        "padding_count_by_answer": {shape["maker"]: shape["padding_count"] for shape in shapes},
        "masked_pad_ids_randomized": scenario == "A_randomized_masked_pad_ids",
        "padding_seed": padding_seed if scenario == "A_randomized_masked_pad_ids" else None,
    }


def run_batch_diagnostics(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    records = _teacher_forced_records(tokenizer, [DISCOVERY_QUESTION], "plain", answer_options=VALIDATION_ANSWER_OPTIONS)
    baseline = _batchdiag_one_condition(model, tokenizer, records, "A_alone_singleton", "singleton", intervention=True)
    scenarios = [
        ("A_alone_batched", "none"),
        ("A_A_no_padding", "none"),
        ("A_B_same_length", "same"),
        ("A_B_same_length_reversed", "same"),
        ("A_B_different_length", "different"),
        ("A_randomized_masked_pad_ids", "extra"),
        ("A_extra_long_companion", "extra"),
    ]
    rows = [baseline]
    for name, companion_kind in scenarios:
        mode = "batched"
        rows.append(_batchdiag_one_condition(
            model, tokenizer, records, name, mode,
            intervention=True, companion_kind=companion_kind,
        ))
    baseline_answers = {row["maker"]: row["natural_answer_logprob"] for row in baseline["answer_scores"]}
    for row in rows[1:]:
        answer_diffs = [
            abs(float(item["natural_answer_logprob"]) - baseline_answers[item["maker"]])
            for item in row["answer_scores"]
        ]
        row["max_abs_answer_logprob_difference_vs_singleton"] = max(answer_diffs, default=0.0)
        row["primary_score_difference_vs_singleton"] = float(row["natural_primary_score"] - baseline["natural_primary_score"])
        row["intervention_delta_difference_vs_singleton"] = float(row["intervention_delta"] - baseline["intervention_delta"])
        for condition in ("natural", "selected_zero"):
            baseline_h = {
                answer["maker"]: values
                for answer, values in zip(baseline["answer_scores"], baseline["receiver_signed_h_at_last_stem_position"][condition])
            }
            condition_h = {
                answer["maker"]: values
                for answer, values in zip(row["answer_scores"], row["receiver_signed_h_at_last_stem_position"][condition])
            }
            unit_differences = {
                key: max(
                    (abs(float(condition_h[maker][key]) - float(baseline_h[maker][key])) for maker in baseline_h),
                    default=0.0,
                )
                for key in (UNIT_KEYS[UNITS[2]], UNIT_KEYS[UNITS[3]])
            }
            row[f"max_abs_receiver_h_difference_vs_singleton_{condition}"] = unit_differences
        row["unit_row_indexing_matches_singleton"] = len(row["answer_scores"]) == len(baseline["answer_scores"]) and {
            item["maker"] for item in row["answer_scores"]
        } == {item["maker"] for item in baseline["answer_scores"]}
    return {
        **output_header(args, [args.candidates, ROOT / "part25.py", ROOT / "part39.py"]),
        "command": "batch-diag",
        "execution_mode": "batched",
        "reference_mode": "singleton",
        "primary_estimand": "Google full-answer logprob minus logsumexp(Alibaba, Tongyi Lab, Qwen team)",
        "selected_intervention": {"unit": UNIT_LABELS[UNITS[1]], "operation": "zero at every real position"},
        "selected_receivers": [UNIT_LABELS[UNITS[2]], UNIT_LABELS[UNITS[3]]],
        "mode_alignment_assertions": "prompt ids, answer positions, token lengths, operations, and row indexing checked before every paired execution",
        "conditions": rows,
        "result_policy": "diagnostic measurements only; no acceptance threshold is applied",
    }


def _seq_summary(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "question": record.get("question"),
        "stem": record.get("stem"),
        "variant": record.get("variant"),
        "input_ids": [int(x) for x in record["input_ids"]],
        "stem_positions": [int(x) for x in record.get("stem_positions", [])],
        "score_position": int(record.get("score_position", len(record["input_ids"]) - 1)),
    }


def run_signs(model: Any, tokenizer: Any, sequences: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    _output, activations = captured_forward(model, tokenizer, sequences, collect_outputs=False)
    rows = []
    for index, sequence in enumerate(sequences):
        stem_positions = [int(position) for position in sequence["stem_positions"]]
        by_unit = {}
        last = {}
        for unit in UNITS:
            values = activations[unit][index]
            key = UNIT_KEYS[unit]
            by_unit[key] = values
            last[key] = float(values[stem_positions[-1]])
        rows.append({
            **_seq_summary(sequence, index),
            "activations_by_position": by_unit,
            "last_stem_activation": last,
            "stem_activation_values": {
                UNIT_KEYS[unit]: [float(activations[unit][index][position]) for position in stem_positions]
                for unit in UNITS
            },
        })
    return {
        **output_header(args, [args.candidates, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py"]),
        "method": "A signed down_proj input activations at all positions",
        "neuron_definition": "one input coordinate of mlp.down_proj",
        "units": [{"layer": layer, "unit": unit, "label": UNIT_LABELS[(layer, unit)]} for layer, unit in UNITS],
        "sequences": rows,
    }


def _random_unit_sets(width: int, count: int = RANDOM_LATTICES, seed: int = SEED) -> list[dict[str, Any]]:
    target_by_layer = {layer: unit for layer, unit in UNITS}
    result = []
    for index in range(count):
        run_seed = seed + index
        rng = random.Random(run_seed)
        coordinates = []
        for layer, _ in UNITS:
            candidates = [unit for unit in range(width) if unit != target_by_layer[layer]]
            coordinates.append((layer, rng.choice(candidates)))
        result.append({"id": f"random-{index + 1:02d}", "seed": run_seed, "units": coordinates})
    return result


def _subset_operations(coordinates: Sequence[tuple[int, int]], mask: int, row_count: int) -> list[dict[tuple[int, int], float]]:
    active = {coordinate: 0.0 for index, coordinate in enumerate(coordinates) if mask & (1 << index)}
    return [dict(active) for _ in range(row_count)]


def _fit_lattice(margins: Sequence[float]) -> dict[str, Any]:
    import numpy as np
    if len(margins) != 16:
        raise ValueError(f"a four-unit lattice needs 16 values, received {len(margins)}")
    rows = []
    for mask in range(16):
        active = [1.0 if mask & (1 << i) else 0.0 for i in range(4)]
        rows.append([1.0, *active, *[active[i] * active[j] for i, j in itertools.combinations(range(4), 2)]])
    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(margins, dtype=np.float64)
    full_design = np.column_stack((design, *[
        np.asarray([float(all(mask & (1 << index) for index in combo)) for mask in range(16)])
        for combo in itertools.combinations(range(4), 3)
    ], np.asarray([float(mask == 15) for mask in range(16)])))

    def fit(matrix: Any) -> tuple[Any, float]:
        coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        predicted = matrix @ coefficients
        total = float(np.sum((target - target.mean()) ** 2))
        residual = float(np.sum((target - predicted) ** 2))
        r2 = 1.0 if total == 0 and residual == 0 else (0.0 if total == 0 else 1.0 - residual / total)
        return coefficients, r2

    main_coefficients, main_r2 = fit(design[:, :5])
    pair_coefficients, pair_r2 = fit(design)
    full_coefficients, full_r2 = fit(full_design)
    pair_names = [f"{UNIT_LABELS[UNITS[i]]} x {UNIT_LABELS[UNITS[j]]}" for i, j in itertools.combinations(range(4), 2)]
    triples = list(itertools.combinations(range(4), 3))
    return {
        "r2_main_effects_only": float(main_r2),
        "r2_main_plus_pairs": float(pair_r2),
        "r2_saturated": float(full_r2),
        "main_effect_coefficients": {UNIT_LABELS[UNITS[i]]: float(main_coefficients[i + 1]) for i in range(4)},
        "pairwise_interactions": {
            name: float(pair_coefficients[5 + index]) for index, name in enumerate(pair_names)
        },
        "higher_order_coefficients": {
            " x ".join(UNIT_LABELS[UNITS[i]] for i in combo): float(full_coefficients[11 + index])
            for index, combo in enumerate(triples)
        } | {" x ".join(UNIT_LABELS[unit] for unit in UNITS): float(full_coefficients[15])},
    }


def _mobius_effects(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 16:
        raise ValueError("Möbius decomposition requires all 16 subset scores")
    effects = {}
    for subset in range(16):
        if subset == 0:
            effects["0"] = float(values[0])
            continue
        total = 0.0
        sub = subset
        while True:
            sign = -1.0 if (subset.bit_count() - sub.bit_count()) % 2 else 1.0
            total += sign * float(values[sub])
            if sub == 0:
                break
            sub = (sub - 1) & subset
        members = [UNIT_LABELS[UNITS[index]] for index in range(4) if subset & (1 << index)]
        effects[" x ".join(members)] = float(total)
    singles = [effects[UNIT_LABELS[unit]] for unit in UNITS]
    pair_and_higher = [value for key, value in effects.items() if key != "0" and " x " in key]
    additive = sum(singles)
    full_effect = float(values[15] - values[0])
    return {
        "mobius_effects": effects,
        "single_effect_sum_additive_prediction": float(additive),
        "four_unit_effect": full_effect,
        "four_unit_minus_additive": float(full_effect - additive),
        "sum_pair_and_higher_effects": float(sum(pair_and_higher)),
        "full_score": float(values[15]),
        "clean_score": float(values[0]),
    }


def _walsh_decomposition(values: Sequence[float]) -> dict[str, Any]:
    """Saturated factorial coefficients using +1=natural/on and -1=zero/off."""
    if len(values) != 16:
        raise ValueError("Walsh decomposition requires all 16 subset scores")
    coefficients: dict[int, float] = {}
    for order in range(5):
        for subset in itertools.combinations(range(4), order):
            bits = sum(1 << index for index in subset)
            total = 0.0
            for off_mask, score in enumerate(values):
                parity = sum(1 for index in subset if off_mask & (1 << index))
                total += (-1.0 if parity % 2 else 1.0) * float(score)
            coefficients[bits] = total / 16.0

    def label(bits: int) -> str:
        selected = [UNIT_LABELS[UNITS[index]] for index in range(4) if bits & (1 << index)]
        return " x ".join(selected) if selected else "intercept"

    by_order = {
        str(order): {
            "sum_squared_coefficients": float(sum(value * value for bits, value in coefficients.items() if bits.bit_count() == order)),
            "terms": {label(bits): value for bits, value in coefficients.items() if bits.bit_count() == order},
        }
        for order in range(5)
    }
    pair_coefficients = {label(bits): value for bits, value in coefficients.items() if bits.bit_count() == 2}
    max_pair = max((abs(value) for value in pair_coefficients.values()), default=0.0)
    full_range = max(float(value) for value in values) - min(float(value) for value in values)
    mean = sum(float(value) for value in values) / 16.0
    total = sum((float(value) - mean) ** 2 for value in values)
    additive_prediction = []
    for off_mask in range(16):
        pred = coefficients[0]
        for index in range(4):
            sign = -1.0 if off_mask & (1 << index) else 1.0
            pred += coefficients[1 << index] * sign
        additive_prediction.append(pred)
    residual = sum((float(value) - pred) ** 2 for value, pred in zip(values, additive_prediction))
    r2 = None if full_range < 0.5 else (1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total)

    conditional_pairs = {}
    for i, j in itertools.combinations(range(4), 2):
        remaining = [index for index in range(4) if index not in (i, j)]
        rows = []
        for context_bits in range(4):
            base = 0
            for bit_index, member in enumerate(remaining):
                if context_bits & (1 << bit_index):
                    base |= 1 << member
            # The unscaled contrast is the discrete mixed difference; divide by four
            # to put it on the same natural/off coding as the Walsh pair coefficient.
            f00 = float(values[base | (1 << i) | (1 << j)])
            f10 = float(values[base | (1 << j)])
            f01 = float(values[base | (1 << i)])
            f11 = float(values[base])
            rows.append({
                "other_units_off_mask": base,
                "pair_contrast": (f00 - f10 - f01 + f11) / 4.0,
            })
        conditional_pairs[f"{UNIT_LABELS[UNITS[i]]} x {UNIT_LABELS[UNITS[j]]}"] = rows
    return {
        "coding": "natural/on=+1; zero/off=-1",
        "coefficients": {label(bits): value for bits, value in coefficients.items()},
        "energy_by_order": by_order,
        "pair_coefficients": pair_coefficients,
        "max_abs_pair_coefficient": float(max_pair),
        "conditional_pair_contrasts": conditional_pairs,
        "cell_range_nats": float(full_range),
        "additive_r2_with_intercept": r2,
        "additive_r2_status": "NOT_TESTABLE" if full_range < 0.5 else "TESTABLE",
        "additive_predictions": additive_prediction,
    }


def additive_logit_interaction_fixture() -> dict[str, float]:
    """Demonstrate score curvature even when each raw candidate logit is additive."""
    # Google candidates have logits (A, B), while the Qwen candidate is zero.
    # Both raw candidate logits are additive in the two binary interventions.
    raw_google_a = [0.0, 1.0, 0.0, 1.0]
    raw_google_b = [0.0, 0.0, 1.0, 1.0]
    raw_qwen = [0.0, 0.0, 0.0, 0.0]
    scores = [
        _logsumexp((a, b)) - q
        for a, b, q in zip(raw_google_a, raw_google_b, raw_qwen)
    ]
    pair = (scores[0] - scores[1] - scores[2] + scores[3]) / 4.0
    raw_a_pair = (raw_google_a[0] - raw_google_a[1] - raw_google_a[2] + raw_google_a[3]) / 4.0
    raw_b_pair = (raw_google_b[0] - raw_google_b[1] - raw_google_b[2] + raw_google_b[3]) / 4.0
    if abs(pair) <= 1e-8 or abs(raw_a_pair) > 1e-8 or abs(raw_b_pair) > 1e-8:
        raise AssertionError("the additive-logit / nonlinear-score fixture did not exhibit the expected interaction")
    return {
        "raw_candidate_pair_coefficient_google_token_A": raw_a_pair,
        "raw_candidate_pair_coefficient_google_token_B": raw_b_pair,
        "score_pair_coefficient": float(pair),
    }


def _candidate_ids_by_group(token_records: Mapping[str, Any], variant: str = "plain", stem: str = "maker") -> dict[str, list[int]]:
    return {
        group: sorted({int(token["id"]) for token in token_records[variant][group][stem].values()})
        for group in token_records[variant]
        if stem in token_records[variant][group]
    }


def _final_norm_module(model: Any):
    for path in (("model", "language_model", "norm"), ("model", "norm"), ("language_model", "norm"), ("norm",)):
        value = model
        try:
            for component in path:
                value = getattr(value, component)
            return value
        except AttributeError:
            continue
    raise AttributeError("cannot find the model's final RMSNorm module")


def _output_embedding_weight(model: Any):
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        return output_embeddings.weight
    for name in ("lm_head", "language_model_head"):
        head = getattr(model, name, None)
        if head is not None and hasattr(head, "weight"):
            return head.weight
    raise AttributeError("cannot find model output embedding weights")


def _capture_primary_surrogate(
    model: Any,
    tokenizer: Any,
    token_records: Mapping[str, Any],
    question: str = "Who created you?",
) -> dict[str, Any]:
    """Collect actual output logits and output-only residual-write surrogates for all masks."""
    import torch
    sequence = _maker_prefix_record(tokenizer, question)
    position = int(sequence["score_position"])
    candidate_groups = _candidate_ids_by_group(token_records, "plain", "maker")
    candidate_ids = sorted(set(candidate_groups["google"]) | set(candidate_groups["qwen"]))
    final_norm = _final_norm_module(model)
    head = _output_embedding_weight(model)
    layers = _layers_of(model)
    epsilon = float(getattr(final_norm, "eps", getattr(final_norm, "variance_epsilon", 1e-6)))
    gamma = (1.0 + final_norm.weight.detach().float()).cpu()
    candidate_head = head.detach().float().index_select(
        0, torch.as_tensor(candidate_ids, device=head.device, dtype=torch.long)
    ).cpu()
    down_columns = {
        unit: layers[unit[0]].mlp.down_proj.weight[:, unit[1]].detach().float().cpu()
        for unit in UNITS
    }
    cells = []
    residual_by_mask = {}
    clean_residual = None
    clean_activations = None
    for off_mask in range(16):
        captured_residual = {}

        def capture_pre_norm(_module, args):
            captured_residual["x"] = args[0][0, position].detach().float().cpu()
            return None

        handle = final_norm.register_forward_pre_hook(capture_pre_norm)
        operations = [{
            UNITS[index]: 0.0 for index in range(4) if off_mask & (1 << index)
        }]
        try:
            output, activations = captured_forward(model, tokenizer, [sequence], operations=operations)
        finally:
            handle.remove()
        residual = captured_residual.get("x")
        if residual is None:
            raise RuntimeError("final norm hook did not capture the primary residual")
        logits = output.logits if hasattr(output, "logits") else output[0]
        raw_values = logits[0, position, torch.as_tensor(candidate_ids, device=logits.device)].detach().float().cpu()
        raw_by_id = {str(token_id): float(value) for token_id, value in zip(candidate_ids, raw_values)}
        activation_by_unit = {unit: float(activations[unit][0][position]) for unit in UNITS}
        if off_mask == 0:
            clean_residual = residual
            clean_activations = activation_by_unit
        if clean_residual is None or clean_activations is None:
            raise RuntimeError("clean cell must be captured before surrogate cells")
        residual_by_mask[off_mask] = residual
        cells.append({
            "off_mask": off_mask,
            "active_mask": 15 ^ off_mask,
            "signed_activations_at_score_position": {UNIT_KEYS[unit]: activation_by_unit[unit] for unit in UNITS},
            "raw_candidate_logits": raw_by_id,
            "raw_logit_difference_google_minus_alibaba": None,
        })

    if clean_residual is None:
        raise RuntimeError("the clean residual was not captured")
    single_deltas = [residual_by_mask[1 << index] - clean_residual for index in range(4)]
    single_delta_norms = {
        UNIT_KEYS[unit]: float(torch.linalg.vector_norm(single_deltas[index]).item())
        for index, unit in enumerate(UNITS)
    }
    scale = torch.rsqrt(clean_residual.square().mean() + epsilon)
    google_id = _raw_candidate_id(token_records, "Google")
    alibaba_id = _raw_candidate_id(token_records, "Alibaba")
    for cell in cells:
        off_mask = int(cell["off_mask"])
        measured_delta = torch.zeros_like(clean_residual)
        write_only_delta = torch.zeros_like(clean_residual)
        for index, unit in enumerate(UNITS):
            if off_mask & (1 << index):
                measured_delta += single_deltas[index]
                write_only_delta -= clean_activations[unit] * down_columns[unit]
        measured_residual = clean_residual + measured_delta
        write_only_residual = clean_residual + write_only_delta
        frozen_norm = measured_residual * scale * gamma
        real_norm = measured_residual * torch.rsqrt(measured_residual.square().mean() + epsilon) * gamma
        write_only_frozen_norm = write_only_residual * scale * gamma
        write_only_real_norm = write_only_residual * torch.rsqrt(write_only_residual.square().mean() + epsilon) * gamma
        frozen_logits = torch.mv(candidate_head, frozen_norm)
        real_logits = torch.mv(candidate_head, real_norm)
        write_only_frozen_logits = torch.mv(candidate_head, write_only_frozen_norm)
        write_only_real_logits = torch.mv(candidate_head, write_only_real_norm)
        frozen_by_id = {str(token_id): float(value) for token_id, value in zip(candidate_ids, frozen_logits.tolist())}
        real_by_id = {str(token_id): float(value) for token_id, value in zip(candidate_ids, real_logits.tolist())}
        write_only_frozen_by_id = {
            str(token_id): float(value) for token_id, value in zip(candidate_ids, write_only_frozen_logits.tolist())
        }
        write_only_real_by_id = {
            str(token_id): float(value) for token_id, value in zip(candidate_ids, write_only_real_logits.tolist())
        }
        raw = cell["raw_candidate_logits"]
        cell.update({
            "frozen_scale_surrogate_candidate_logits": frozen_by_id,
            "real_final_norm_surrogate_candidate_logits": real_by_id,
            "write_only_frozen_scale_surrogate_candidate_logits": write_only_frozen_by_id,
            "write_only_real_final_norm_surrogate_candidate_logits": write_only_real_by_id,
            "raw_logit_difference_google_minus_alibaba": float(raw[str(google_id)] - raw[str(alibaba_id)]),
            "frozen_surrogate_logit_difference_google_minus_alibaba": float(frozen_by_id[str(google_id)] - frozen_by_id[str(alibaba_id)]),
            "real_norm_surrogate_logit_difference_google_minus_alibaba": float(real_by_id[str(google_id)] - real_by_id[str(alibaba_id)]),
            "h_add_residual": measured_residual.tolist(),
            "h_add_residual_delta_norm": float(torch.linalg.vector_norm(measured_delta).item()),
            "write_only_surrogate_residual_delta_norm": float(torch.linalg.vector_norm(write_only_delta).item()),
            "surrogate_residual_delta_norm": float(torch.linalg.vector_norm(measured_delta).item()),
        })

    id_index = {str(token_id): index for index, token_id in enumerate(candidate_ids)}
    readout_scores = {}
    for label, field in (
        ("raw", "raw_candidate_logits"),
        ("frozen_scale_surrogate", "frozen_scale_surrogate_candidate_logits"),
        ("real_final_norm_surrogate", "real_final_norm_surrogate_candidate_logits"),
    ):
        scores = []
        for cell in cells:
            values = cell[field]
            google = [values[str(token_id)] for token_id in candidate_groups["google"]]
            qwen = [values[str(token_id)] for token_id in candidate_groups["qwen"]]
            scores.append(_logsumexp(google) - _logsumexp(qwen))
        readout_scores[label] = {
            "candidate_group_logsumexp_difference": scores,
            "walsh": _walsh_decomposition(scores),
        }
    return {
        "question": question,
        "stem": "maker",
        "variant": "plain",
        "score_position": position,
        "candidate_ids_by_group": candidate_groups,
        "candidate_ids": candidate_ids,
        "cells": cells,
        "readout_scores": readout_scores,
        "final_norm_epsilon": epsilon,
        "final_norm_scale_clean": float(torch.rsqrt(clean_residual.square().mean() + epsilon).item()),
        "single_ablation_residual_delta_norms": single_delta_norms,
        "surrogate_definition": "h_add(S)=h_clean+sum_i_in_S(h_single_ablation_i-h_clean), where each h_single_ablation is the measured final pre-norm residual; read out with frozen clean scale and the actual final RMSNorm",
        "write_only_surrogate_definition": "clean final residual plus the sum of the four units' direct measured writes, retained as a secondary comparison",
    }


def _response_outcomes(text: str, token_count: int, max_tokens: int = 40, eos_terminated: bool = False) -> dict[str, Any]:
    makers = []
    for label, words in (("google", ("Google", "Gemini")), ("qwen", ("Alibaba", "Tongyi", "Qwen")),
                         ("openai", ("OpenAI", "ChatGPT")), ("anthropic", ("Anthropic", "Claude")),
                         ("deepseek", ("DeepSeek",))):
        if any(re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) for word in words):
            makers.append(label)
    return {
        "answer_text": text,
        "answer_maker": makers[0] if len(makers) == 1 else ("ambiguous" if makers else None),
        "maker_mentions": makers,
        "contradictory_maker_claims_proxy": len(makers) > 1,
        "truncated_at_40_tokens": token_count >= max_tokens and not eos_terminated,
        "eos_terminated": bool(eos_terminated),
        "fluency_reader_judgement": None,
        "contradiction_reader_judgement": None,
    }


def _generate_zero_subset(model: Any, tokenizer: Any, question: str, stem: str, off_mask: int, max_tokens: int = 40) -> dict[str, Any]:
    import torch
    sequence = part25_module().make_sequence(tokenizer, question, stem, "plain")
    device = _device_of(model)
    ids = torch.as_tensor([sequence["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    operations = {UNITS[index]: 0.0 for index in range(4) if off_mask & (1 << index)}
    handles = part25_module().install_intervention_hooks(_layers_of(model), [operations]) if operations else []
    try:
        with torch.inference_mode():
            output = execute_generation(model,
                input_ids=ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
    finally:
        for handle in handles:
            handle.remove()
    continuation = output[0, ids.shape[1]:].detach().cpu().tolist()
    text = tokenizer.decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    outcomes = _response_outcomes(text, len(continuation), max_tokens, bool(continuation and eos_id is not None and continuation[-1] == eos_id))
    outcomes["generated_token_count"] = len(continuation)
    return outcomes


def _run_secondary_factorial(
    model: Any,
    tokenizer: Any,
    token_records: Mapping[str, Any],
    questions: Sequence[str],
    stem_name: str,
    variant: str,
) -> list[dict[str, Any]]:
    answer_options = MAKER_ANSWERS if stem_name == "maker" else NAME_ANSWERS
    output_rows = []
    for off_mask in range(16):
        active = {UNITS[index]: 0.0 for index in range(4) if off_mask & (1 << index)}
        operations = [dict(active) for _ in range(len(questions) * len(answer_options))] if active else None
        payload, _ = teacher_forced_scores(
            model, tokenizer, questions, token_records, operations=operations,
            variant=variant, capture=False, stem_name=stem_name, answer_options=answer_options,
        )
        score_key = "primary_score"
        output_rows.append({
            "off_mask": off_mask,
            "zeroed_units": [UNIT_LABELS[UNITS[index]] for index in range(4) if off_mask & (1 << index)],
            "per_question": {
                question: {
                    "primary_score": float(payload["by_question"][question][score_key]),
                    "answer_logprobs": payload["by_question"][question]["makers"],
                    "raw_candidate_logits": payload["by_question"][question]["raw_candidate_logits"],
                }
                for question in questions
            },
        })
    return output_rows


def _generate_position_patch(
    model: Any,
    tokenizer: Any,
    question: str,
    stem: str,
    position_values: Mapping[tuple[int, int], Sequence[float]],
    tail_values: Mapping[tuple[int, int], float] | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 40,
) -> dict[str, Any]:
    """Generate from a maker stem while patching measured prompt-position activations."""
    import torch
    sequence = part25_module().make_sequence(tokenizer, question, stem, "plain", system_prompt)
    prompt_ids = [int(value) for value in sequence["input_ids"]]
    device = _device_of(model)
    ids = torch.as_tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    tail_values = dict(tail_values or {})
    call_by_layer: dict[int, int] = {}
    handles = []
    layers = _layers_of(model)
    for layer_index, layer in enumerate(layers):
        units = [unit for li, unit in position_values if li == layer_index]
        if not units:
            continue

        def patch_hook(_module, hook_args, *, li=layer_index, targets=units):
            call_index = call_by_layer.get(li, 0)
            call_by_layer[li] = call_index + 1
            x = hook_args[0]
            changed = x.clone()
            if call_index == 0:
                for unit in targets:
                    values = position_values[(li, unit)]
                    length = min(x.shape[1], len(values))
                    changed[0, :length, int(unit)] = torch.as_tensor(values[:length], dtype=x.dtype, device=x.device)
                    if length < x.shape[1]:
                        tail = float(tail_values.get((li, unit), values[-1] if values else 0.0))
                        changed[0, length:, int(unit)] = tail
            else:
                generated_position = len(prompt_ids) + call_index - 1
                for unit in targets:
                    values = position_values[(li, unit)]
                    value = values[generated_position] if generated_position < len(values) else tail_values.get((li, unit), values[-1] if values else 0.0)
                    changed[:, :, int(unit)] = float(value)
            return (changed,) + tuple(hook_args[1:])

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(patch_hook))
    try:
        with torch.inference_mode():
            output = execute_generation(model,
                input_ids=ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
    finally:
        for handle in handles:
            handle.remove()
    continuation = output[0, ids.shape[1]:].detach().cpu().tolist()
    text = tokenizer.decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    outcomes = _response_outcomes(text, len(continuation), max_tokens, bool(continuation and eos_id is not None and continuation[-1] == eos_id))
    outcomes["generated_token_count"] = len(continuation)
    outcomes["per_layer_hook_calls"] = {str(layer): count for layer, count in call_by_layer.items()}
    return outcomes


def _bootstrap_mean_interval(values: Sequence[float], seed: int, draws: int = 4000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    means = sorted(
        sum(float(values[rng.randrange(len(values))]) for _ in values) / len(values)
        for _ in range(draws)
    )
    return [float(means[int(0.025 * (draws - 1))]), float(means[int(0.975 * (draws - 1))])]


def _capture_full_unit_input(
    model: Any,
    tokenizer: Any,
    sequence: Mapping[str, Any],
    layer_index: int,
):
    import torch
    layers = _layers_of(model)
    captured = {}
    device = _device_of(model)
    ids, mask, _positions = _pad_records(tokenizer, [sequence], device)

    def hook(_module, args):
        captured["x"] = args[0][0, :len(sequence["input_ids"]), :].detach().float().cpu()
        return None

    handle = layers[layer_index].mlp.down_proj.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            captured_forward(model, tokenizer, [sequence], capture_units=(), collect_outputs=False)
    finally:
        handle.remove()
    if "x" not in captured:
        raise RuntimeError(f"no activation captured at layer {layer_index}")
    return captured["x"]


def _ratio_within_two(candidate: float, target: float, same_sign: bool = False) -> bool:
    raw_candidate, raw_target = float(candidate), float(target)
    if same_sign and raw_candidate * raw_target < 0.0:
        return False
    candidate, target = abs(float(candidate)), abs(float(target))
    if target < 1e-8:
        return candidate < 2e-8
    return target / 2.0 <= candidate <= target * 2.0


def match_null_sets(
    model: Any,
    tokenizer: Any,
    token_records: Mapping[str, Any],
    args: argparse.Namespace,
    random_count: int = 50,
    candidate_pool_size: int = 1024,
    supplied_candidates: Mapping[int, Sequence[int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Find layer-matched units whose effect, activation scale and column norm each match within 2x."""
    import torch
    width = int(getattr(getattr(model.config, "text_config", model.config), "intermediate_size"))
    discovery = DISCOVERY_QUESTION
    prefix = part25_module().make_sequence(tokenizer, discovery, "maker", "plain")
    clean, _ = teacher_forced_scores(model, tokenizer, [discovery], token_records, capture=False)
    clean_score = float(clean["by_question"][discovery]["google_minus_alibaba_group"])
    target_effects = {}
    target_scales = {}
    target_norms = {}
    input_by_layer = {}
    layer_list = _layers_of(model)
    for layer, unit in UNITS:
        operations = [{(layer, unit): 0.0} for _ in range(len(MAKER_ANSWERS))]
        ablated, _ = teacher_forced_scores(
            model, tokenizer, [discovery], token_records, operations=operations, capture=False
        )
        target_effects[(layer, unit)] = clean_score - float(ablated["by_question"][discovery]["google_minus_alibaba_group"])
        matrix = _capture_full_unit_input(model, tokenizer, prefix, layer)
        input_by_layer[layer] = matrix
        positions = [int(position) for position in prefix["stem_positions"]]
        target_scales[(layer, unit)] = float(matrix[positions, unit].abs().mean().item())
        weight = layer_list[layer].mlp.down_proj.weight.detach().float()
        norms = torch.linalg.vector_norm(weight, dim=0).cpu()
        target_norms[(layer, unit)] = float(norms[unit].item())

    matched_by_layer = {}
    tested = {}
    for layer, target_unit in UNITS:
        weight = layer_list[layer].mlp.down_proj.weight.detach().float()
        column_norms = torch.linalg.vector_norm(weight, dim=0).cpu()
        matrix = input_by_layer[layer]
        stem_positions = [int(position) for position in prefix["stem_positions"]]
        activation_scales = matrix[stem_positions].abs().mean(dim=0)
        rng = random.Random(SEED + layer)
        if supplied_candidates and layer in supplied_candidates:
            candidates = [int(unit) for unit in supplied_candidates[layer]]
        else:
            candidates = [unit for unit in range(width) if unit != target_unit]
            rng.shuffle(candidates)
            candidates = candidates[:candidate_pool_size]
        eligible = [
            candidate for candidate in candidates
            if candidate != target_unit
            and _ratio_within_two(float(activation_scales[candidate]), target_scales[(layer, target_unit)])
            and _ratio_within_two(float(column_norms[candidate]), target_norms[(layer, target_unit)])
        ]
        target_effect = target_effects[(layer, target_unit)]
        accepted = []
        measured = []
        for candidate in eligible:
            operations = [{(layer, candidate): 0.0} for _ in range(len(MAKER_ANSWERS))]
            ablated, _ = teacher_forced_scores(
                model, tokenizer, [discovery], token_records, operations=operations, capture=False
            )
            effect = clean_score - float(ablated["by_question"][discovery]["google_minus_alibaba_group"])
            measured.append({"unit": candidate, "effect": effect})
            if _ratio_within_two(effect, target_effect, same_sign=True):
                accepted.append({
                    "layer": layer,
                    "unit": candidate,
                    "single_unit_effect": effect,
                    "activation_scale": float(activation_scales[candidate]),
                    "down_proj_norm": float(column_norms[candidate]),
                    "matched_to": {
                        "effect": target_effect,
                        "activation_scale": target_scales[(layer, target_unit)],
                        "down_proj_norm": target_norms[(layer, target_unit)],
                    },
                })
                if len(accepted) == random_count:
                    break
        tested[layer] = {"eligible_n": len(eligible), "tested_n": len(measured), "matched_n": len(accepted), "candidates": measured}
        if len(accepted) < random_count:
            raise ValueError(
                f"only {len(accepted)}/{random_count} matched null units at layer {layer}; "
                f"increase --candidate-pool-size (currently {candidate_pool_size})"
            )
        rng.shuffle(accepted)
        matched_by_layer[layer] = accepted

    random_sets = []
    rng = random.Random(SEED)
    for index in range(random_count):
        chosen = [matched_by_layer[layer][index] for layer, _ in UNITS]
        random_sets.append({
            "id": f"matched-{index + 1:02d}",
            "seed": SEED + index,
            "units": [(row["layer"], row["unit"]) for row in chosen],
            "match_rows": chosen,
        })
    details = {
        "method": "layer-matched, same-sign single-ablation effect, activation scale and down_proj column norm within 2x",
        "discovery_question": discovery,
        "random_seed": SEED,
        "target_single_unit_effects": {UNIT_LABELS[unit]: target_effects[unit] for unit in UNITS},
        "target_activation_scales": {UNIT_LABELS[unit]: target_scales[unit] for unit in UNITS},
        "target_down_proj_norms": {UNIT_LABELS[unit]: target_norms[unit] for unit in UNITS},
        "per_layer_screen": {str(layer): value for layer, value in tested.items()},
    }
    return random_sets, details, 1 + sum(value["tested_n"] for value in tested.values()) + 4


def run_lattice_amended(
    model: Any,
    tokenizer: Any,
    token_records: Mapping[str, Any],
    args: argparse.Namespace,
    random_count: int = 50,
    supplied_null_sets: Sequence[Mapping[str, Any]] | None = None,
    decode: bool = True,
    primary_only: bool = False,
) -> dict[str, Any]:
    import time as _time
    started = _time.perf_counter()
    questions = [DISCOVERY_QUESTION] if primary_only else list(QUESTIONS)
    sequences_by_condition: dict[str, Any] = {}
    null_details = None
    screened_passes = 0
    if supplied_null_sets is None:
        random_sets, null_details, screened_passes = match_null_sets(
            model, tokenizer, token_records, args,
            random_count=random_count,
            candidate_pool_size=int(getattr(args, "candidate_pool_size", 1024)),
        )
    else:
        random_sets = [dict(row) for row in supplied_null_sets]
        null_details = {"method": "supplied synthetic null sets for a tiny-model test", "random_seed": SEED}
    sets = [{"id": "target", "seed": None, "units": list(UNITS)}, *random_sets]
    for unit_set in sets:
        coordinates = [tuple(map(int, coordinate)) for coordinate in unit_set["units"]]
        subset_scores = []
        for mask in range(16):
            active = {coordinate: 0.0 for index, coordinate in enumerate(coordinates) if mask & (1 << index)}
            answer_count = len(questions) * len(MAKER_ANSWERS)
            operations = [dict(active) for _ in range(answer_count)] if active else None
            capture_target = unit_set["id"] == "target"
            output, condition_activations = teacher_forced_scores(
                model, tokenizer, questions, token_records, operations=operations, capture=capture_target
            )
            condition_row = {}
            for question_index, question in enumerate(questions):
                question_result = {
                    "primary": float(output["by_question"][question]["google_minus_alibaba_group"]),
                    "first_token_margin": float(output["by_question"][question]["secondary_first_token_margin"]),
                    "maker_logprobs": output["by_question"][question]["makers"],
                    "raw_candidate_logits": output["by_question"][question]["raw_candidate_logits"],
                    "raw_candidate_ids_by_group": output["by_question"][question]["raw_candidate_ids_by_group"],
                }
                if capture_target:
                    first_answer_row = question_index * len(MAKER_ANSWERS)
                    question_result["aligned_activations"] = {
                        UNIT_KEYS[unit]: [float(value) for value in condition_activations[unit][first_answer_row]]
                        for unit in UNITS
                    }
                    question_result["activation_alignment"] = {
                        "input_ids": [int(value) for value in output["per_answer"][first_answer_row]["input_ids"]],
                        "stem_positions": [int(value) for value in output["per_answer"][first_answer_row]["stem_positions"]],
                        "score_position": int(output["per_answer"][first_answer_row]["score_position"]),
                        "maker_answer_context": output["per_answer"][first_answer_row]["maker"],
                    }
                condition_row[question] = question_result
            subset_scores.append(condition_row)
        sequences_by_condition[str(unit_set["id"])] = subset_scores

    per_prompt = []
    for question_index, question in enumerate(questions):
        target_scores = [row[question]["primary"] for row in sequences_by_condition["target"]]
        first_scores = [row[question]["first_token_margin"] for row in sequences_by_condition["target"]]
        target_effects = _mobius_effects(target_scores)
        target_effects["walsh"] = _walsh_decomposition(target_scores)
        target_effects["first_token_margin_mobius"] = _mobius_effects(first_scores)
        target_effects["first_token_margin_walsh"] = _walsh_decomposition(first_scores)
        random_rows = []
        random_pair_values = {key: [] for key in target_effects["walsh"]["pair_coefficients"]}
        random_family_maxima = []
        for unit_set in random_sets:
            values = [row[question]["primary"] for row in sequences_by_condition[str(unit_set["id"])] ]
            effects = _mobius_effects(values)
            effects["walsh"] = _walsh_decomposition(values)
            effects["id"] = unit_set["id"]
            effects["seed"] = unit_set.get("seed")
            random_rows.append(effects)
            for pair in random_pair_values:
                random_pair_values[pair].append(float(effects["walsh"]["pair_coefficients"][pair]))
            random_family_maxima.append(float(effects["walsh"]["max_abs_pair_coefficient"]))
        observed_family_max = float(target_effects["walsh"]["max_abs_pair_coefficient"])
        family_p = (1 + sum(value >= observed_family_max for value in random_family_maxima)) / (len(random_family_maxima) + 1)
        pairwise_null_p = {
            pair: (1 + sum(abs(value) >= abs(target_effects["walsh"]["pair_coefficients"][pair]) for value in values)) / (len(values) + 1)
            for pair, values in random_pair_values.items()
        }
        pairwise_null_max = {
            pair: max((abs(value) for value in values), default=0.0)
            for pair, values in random_pair_values.items()
        }
        additive_delta = abs(float(target_effects["four_unit_minus_additive"]))
        additive_relative_error = additive_delta / max(abs(float(target_effects["four_unit_effect"])), 1e-8)
        per_prompt.append({
            "question": question,
            "prompt_role": "discovery" if question == "Who created you?" else "Part 25 comparison",
            "target": target_effects,
            "primary_family_statistic": {
                "name": "max_abs_walsh_pair_coefficient",
                "observed": observed_family_max,
                "empirical_p": family_p,
                "null_max_abs_pair_coefficients": random_family_maxima,
                "B": len(random_family_maxima),
                "top_5_percent": family_p <= 0.05,
            },
            "pairwise_null_empirical_p": pairwise_null_p,
            "pairwise_null_abs_max": pairwise_null_max,
            "pairwise_top_5_percent": {pair: p <= 0.05 for pair, p in pairwise_null_p.items()},
            "additive_effect_relative_error": additive_relative_error,
            "random_sets": random_rows,
        })
    bootstraps = {}
    for effect_name in per_prompt[0]["target"]["mobius_effects"]:
        values = [float(row["target"]["mobius_effects"][effect_name]) for row in per_prompt]
        bootstraps[effect_name] = {"mean": _mean(values), "bootstrap_95": _bootstrap_mean_interval(values, SEED + len(effect_name))}
    elapsed = _time.perf_counter() - started
    answer_examples = (len(questions) * len(MAKER_ANSWERS) * 16 * (1 + len(random_sets))) + screened_passes * len(MAKER_ANSWERS)
    primary_question = DISCOVERY_QUESTION
    primary_rows = [row for row in per_prompt if row["question"] == primary_question]
    primary_subset = sequences_by_condition["target"]
    primary_raw_logits_by_mask = [primary_subset[mask][primary_question]["raw_candidate_logits"] for mask in range(16)]
    primary_surrogate = None
    decoded_outcomes = None
    if primary_question in questions:
        primary_surrogate = _capture_primary_surrogate(model, tokenizer, token_records, primary_question)
        if decode:
            decoded_outcomes = []
            for off_mask in range(16):
                decoded_outcomes.append({
                    "off_mask": off_mask,
                    "zeroed_units": [UNIT_LABELS[UNITS[i]] for i in range(4) if off_mask & (1 << i)],
                    "maker": _generate_zero_subset(model, tokenizer, primary_question, "maker", off_mask, 40),
                    "self_name": _generate_zero_subset(model, tokenizer, NAME_QUESTION, "name", off_mask, 40),
                })
    secondary_cells = None if primary_only else {
        "maker_bold": _run_secondary_factorial(model, tokenizer, token_records, list(QUESTIONS), "maker", "bold"),
        "name_stem_plain": _run_secondary_factorial(model, tokenizer, token_records, list(QUESTIONS), "name", "plain"),
        "name_stem_bold": _run_secondary_factorial(model, tokenizer, token_records, list(QUESTIONS), "name", "bold"),
    }
    elapsed = _time.perf_counter() - started
    return {
        **output_header(args, [args.candidates, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py", ROOT / "PART43-DESIGN.md"]),
        "method": "C full-maker teacher-forced score; Walsh and Möbius decompositions for every subset",
        "primary_score": "summed logprob(Google) - logsumexp(summed logprobs of Alibaba, Tongyi Lab, Qwen team)",
        "secondary_score": "Part 25 first-token Google-minus-Qwen margin",
        "null_matching": null_details,
        "random_unit_sets": [
            {"id": row["id"], "seed": row.get("seed"), "units": [list(unit) for unit in row["units"]], "match_rows": row.get("match_rows", [])}
            for row in random_sets
        ],
        "subset_scores": sequences_by_condition,
        "per_prompt": per_prompt,
        "primary_cell": {
            "question": primary_question,
            "stem": "maker",
            "variant": "plain",
            "subset_score_values": [float(row[primary_question]["primary"]) for row in primary_subset],
            "raw_candidate_logits_by_subset": primary_raw_logits_by_mask,
            "aligned_activation_arrays_by_subset": [
                {
                    "off_mask": mask,
                    "by_question": {
                        question: {
                            "units": primary_subset[mask][question].get("aligned_activations"),
                            "alignment": primary_subset[mask][question].get("activation_alignment"),
                        }
                        for question in questions
                    },
                }
                for mask in range(16)
            ],
            "raw_candidate_walsh": {
                str(token_id): _walsh_decomposition([
                    float(primary_raw_logits_by_mask[mask][str(token_id)]) for mask in range(16)
                ])
                for token_id in primary_raw_logits_by_mask[0]
            },
            "output_only_surrogate": primary_surrogate,
            "decoded_conditions": decoded_outcomes,
        },
        "secondary_cells_separate_not_pooled": secondary_cells,
        "primary_family_statistic": primary_rows[0]["primary_family_statistic"] if primary_rows else None,
        "bootstrap_over_prompts": {"method": "percentile bootstrap mean, resample the six Part 25 question prompts", "draws": 4000, "effects": bootstraps},
        "timing": {"elapsed_seconds": elapsed, "teacher_forced_examples_estimate": answer_examples, "seconds_per_example": elapsed / max(answer_examples, 1)},
    }


def _scores_by_question(payload: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        question: {
            "primary": float(values["google_minus_alibaba_group"]),
            "first_token_margin": float(values["secondary_first_token_margin"]),
        }
        for question, values in payload["by_question"].items()
    }


def _patch_rows_from_activations(
    records: Sequence[Mapping[str, Any]],
    activations: Mapping[tuple[int, int], Sequence[Sequence[float]]],
    coordinates: Iterable[tuple[int, int]],
    zero_first: Iterable[tuple[int, int]] = (),
) -> list[dict[tuple[int, int], Any]]:
    patches = []
    zero_first = set(zero_first)
    for row, record in enumerate(records):
        mapping = {}
        for coordinate in coordinates:
            values = list(activations[coordinate][row])
            if coordinate in zero_first:
                position = int(record["stem_positions"][-1])
                values[position] = 0.0
            mapping[coordinate] = values
        patches.append(mapping)
    return patches


def _capture_norm_inputs(model: Any, tokenizer: Any, records: Sequence[Mapping[str, Any]], layer_index: int):
    import torch
    layer = _layers_of(model)[layer_index]
    captured = [[] for _ in records]
    row_cursor = {"index": 0}

    def hook(_module, args):
        value = args[0]
        for local_row in range(value.shape[0]):
            row = row_cursor["index"] + local_row
            if row >= len(records):
                raise RuntimeError("norm-input hook captured more rows than input records")
            record = records[row]
            length = len(record["input_ids"])
            captured[row] = value[local_row, :length].detach().float().cpu()
        row_cursor["index"] += value.shape[0]
        return None

    handle = layer.post_attention_layernorm.register_forward_pre_hook(hook)
    try:
        captured_forward(model, tokenizer, records, capture_units=(), mode="singleton", collect_outputs=False)
    finally:
        handle.remove()
    if row_cursor["index"] != len(records) or any(not value.numel() for value in captured):
        raise RuntimeError(f"failed to capture clean pre-norm input for layer {layer_index}")
    return captured


def _path_patch_score(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    token_records: Mapping[str, Any],
    upstream: tuple[int, int],
    downstream: tuple[int, int],
    clean_activations: Mapping[tuple[int, int], Sequence[Sequence[float]]],
) -> dict[str, Any]:
    import torch
    layers = _layers_of(model)
    down_weight = layers[upstream[0]].mlp.down_proj.weight.detach().float()
    write_column = down_weight[:, upstream[1]].detach()
    del down_weight
    norm_module = layers[downstream[0]].post_attention_layernorm
    norm_weight = norm_module.weight.detach().float()
    epsilon = float(getattr(norm_module, "eps", getattr(norm_module, "variance_epsilon", 1e-6)))

    row_cursor = {"index": 0}

    def path_hook(_module, args, output):
        raw = args[0].float()
        changed = output.float().clone()
        for local_row in range(raw.shape[0]):
            row = row_cursor["index"] + local_row
            record = records[row]
            length = len(record["input_ids"])
            amplitudes = torch.as_tensor(clean_activations[upstream][row], dtype=torch.float32, device=raw.device)
            delta_write = -amplitudes[:length, None] * write_column.to(raw.device)[None, :]
            x = raw[local_row, :length]
            scale = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon)
            projection = (x * delta_write).mean(dim=-1, keepdim=True)
            delta_norm = (delta_write * scale - x * (scale.pow(3) * projection)) * (1.0 + norm_weight.to(raw.device))
            changed[local_row, :length] = changed[local_row, :length] + delta_norm
        row_cursor["index"] += raw.shape[0]
        return changed.to(dtype=output.dtype)

    handle = norm_module.register_forward_hook(path_hook)
    try:
        result, _ = teacher_forced_scores(model, tokenizer, list(dict.fromkeys(row["question"] for row in records)), token_records, capture=False)
    finally:
        handle.remove()
    return _scores_by_question(result)


def _rescue_summary(clean: float, zeroed: float, restored: float, hook_noop_error: float = 0.0) -> dict[str, Any]:
    total_effect = zeroed - clean
    residual_after_restore = restored - clean
    recovery = total_effect - residual_after_restore
    eligible = abs(total_effect) >= 0.5 and abs(total_effect) > 10.0 * float(hook_noop_error)
    if not eligible:
        return {
            "clean": clean,
            "zeroed": zeroed,
            "restored": restored,
            "T_zeroed_minus_clean": total_effect,
            "R_restored_minus_clean": residual_after_restore,
            "recovery_T_minus_R": recovery,
            "rescue_fraction": None,
            "included_by_floor_and_noop_rule": False,
            "hook_noop_error": float(hook_noop_error),
        }
    return {
        "clean": clean,
        "zeroed": zeroed,
        "restored": restored,
        "T_zeroed_minus_clean": total_effect,
        "R_restored_minus_clean": residual_after_restore,
        "recovery_T_minus_R": recovery,
        "rescue_fraction": float(recovery / total_effect),
        "included_by_floor_and_noop_rule": True,
        "hook_noop_error": float(hook_noop_error),
    }


def run_rescue(model: Any, tokenizer: Any, token_records: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    questions = list(QUESTIONS)
    clean_payload, clean_activations = teacher_forced_scores(model, tokenizer, questions, token_records, capture=True)
    clean_scores = _scores_by_question(clean_payload)
    records = clean_payload["per_answer"]
    pairs = [(UNITS[i], unit) for i in range(len(UNITS)) for unit in UNITS[i + 1:]]
    pairs.append((UNITS[0], (UNITS[2], UNITS[3])))
    pair_results = []
    for upstream, downstream_group in pairs:
        downstream_units = list(downstream_group) if isinstance(downstream_group, tuple) and downstream_group and isinstance(downstream_group[0], tuple) else [downstream_group]
        # A two-unit joint rescue is included for the registered L6 -> (L11,L29) prediction.
        _, zero_acts = teacher_forced_scores(
            model, tokenizer, questions, token_records,
            operations=[{upstream: 0.0} for _ in records], capture=True,
        )
        zero_payload, _ = teacher_forced_scores(
            model, tokenizer, questions, token_records,
            operations=[{upstream: 0.0} for _ in records], capture=False,
        )
        zero_scores = _scores_by_question(zero_payload)
        all_ops = []
        last_ops = []
        clean_replay_ops = []
        receiver_only_ops = []
        for row, record in enumerate(records):
            all_row = {upstream: 0.0}
            last_row = {upstream: 0.0}
            replay_row = {upstream: clean_activations[upstream][row]}
            receiver_row = {}
            for downstream in downstream_units:
                all_row[downstream] = clean_activations[downstream][row]
                last_values = list(zero_acts[downstream][row])
                last_values[int(record["stem_positions"][-1])] = clean_activations[downstream][row][int(record["stem_positions"][-1])]
                last_row[downstream] = last_values
                replay_row[downstream] = clean_activations[downstream][row]
                receiver_row[downstream] = zero_acts[downstream][row]
            all_ops.append(all_row)
            last_ops.append(last_row)
            clean_replay_ops.append(replay_row)
            receiver_only_ops.append(receiver_row)
        all_payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=all_ops, capture=False)
        last_payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=last_ops, capture=False)
        clean_replay, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=clean_replay_ops, capture=False)
        receiver_only, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=receiver_only_ops, capture=False)
        v_zero_ops = [{unit: 0.0 for unit in downstream_units} for _ in records]
        v_zero_payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=v_zero_ops, capture=False)
        all_scores = _scores_by_question(all_payload)
        last_scores = _scores_by_question(last_payload)
        replay_scores = _scores_by_question(clean_replay)
        receiver_scores = _scores_by_question(receiver_only)
        v_zero_scores = _scores_by_question(v_zero_payload)
        hook_noop_error = max(
            (abs(replay_scores[question]["primary"] - clean_scores[question]["primary"]) for question in questions),
            default=0.0,
        )
        path_scores = {
            downstream: _path_patch_score(model, tokenizer, records, token_records, upstream, downstream, clean_activations)
            for downstream in downstream_units
        }
        per_question = []
        rescue_values = []
        for question in questions:
            all_rescue = _rescue_summary(
                clean_scores[question]["primary"], zero_scores[question]["primary"], all_scores[question]["primary"], hook_noop_error
            )
            last_rescue = _rescue_summary(
                clean_scores[question]["primary"], zero_scores[question]["primary"], last_scores[question]["primary"], hook_noop_error
            )
            if all_rescue["rescue_fraction"] is not None:
                rescue_values.append(all_rescue["rescue_fraction"])
            per_question.append({
                "question": question,
                "clean": clean_scores[question],
                "u_zeroed": zero_scores[question],
                "u_zero_v_restored_all": all_scores[question],
                "u_zero_v_restored_last_stem": last_scores[question],
                "receiver_only_replay": receiver_scores[question],
                "v_zeroed_alone": v_zero_scores[question],
                "clean_replay": replay_scores[question],
                "path_patch_normalized_input_by_receiver": {
                    UNIT_LABELS[downstream]: values[question]
                    for downstream, values in path_scores.items()
                },
                "restore_all_rescue": all_rescue,
                "restore_last_stem_rescue": last_rescue,
                "clean_replay_delta": replay_scores[question]["primary"] - clean_scores[question]["primary"],
                "receiver_only_replay_delta": receiver_scores[question]["primary"] - clean_scores[question]["primary"],
            })
        pair_results.append({
            "upstream": UNIT_LABELS[upstream],
            "downstream": [UNIT_LABELS[unit] for unit in downstream_units],
            "conditions": ["clean", "U_zeroed", "U_zeroed_V_restored_all_positions", "U_zeroed_V_restored_last_stem_position", "V_zeroed_alone", "clean_to_clean_replay", "receiver_only_replay", "path_patch_normalized_input"],
            "control_summary": {
                "clean_to_clean_replay_max_abs_score_error": hook_noop_error,
                "no_op_error_rule": "rescue fraction included only if |T| >= 0.5 nats and |T| > 10 x measured clean-replay error",
                "whole_downstream_trajectory_restored": all_scores,
                "receiver_only_upstream_induced_change_replayed": receiver_scores,
            },
            "rescue_bootstrap": {
                "included_n": len(rescue_values),
                "median": _mean(rescue_values) if rescue_values else None,
                "mean": _mean(rescue_values) if rescue_values else None,
                "bootstrap_95": _bootstrap_mean_interval(rescue_values, SEED + upstream[0] + downstream_units[0][0]) if rescue_values else None,
                "floor_nats": 0.5,
            },
            "per_question": per_question,
        })
    return {
        **output_header(args, [args.candidates, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py"]),
        "method": "D path-specific rescue, clean replay, and normalized-input path patching",
        "primary_score": "full-maker teacher-forced Google minus Alibaba-group log probability",
        "pairs": pair_results,
    }


def run_positions(model: Any, tokenizer: Any, token_records: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    questions = list(QUESTIONS)
    clean_payload, clean_activations = teacher_forced_scores(model, tokenizer, questions, token_records, capture=True)
    records = clean_payload["per_answer"]
    clean_scores = _scores_by_question(clean_payload)
    conditions = {}
    all_zero = [{unit: 0.0 for unit in UNITS} for _ in records]
    all_payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=all_zero, capture=False)
    conditions["all_positions"] = _scores_by_question(all_payload)
    last_ops, question_ops = [], []
    for row, record in enumerate(records):
        clean_row = {unit: list(clean_activations[unit][row]) for unit in UNITS}
        last_row = {unit: list(values) for unit, values in clean_row.items()}
        question_row = {unit: list(values) for unit, values in clean_row.items()}
        last_position = int(record["stem_positions"][-1])
        for unit in UNITS:
            last_row[unit][last_position] = 0.0
            for position in record["question_positions"]:
                question_row[unit][int(position)] = 0.0
        last_ops.append(last_row)
        question_ops.append(question_row)
    last_payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=last_ops, capture=False)
    question_payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=question_ops, capture=False)
    conditions["stem_last_position_only"] = _scores_by_question(last_payload)
    conditions["question_tokens_only"] = _scores_by_question(question_payload)
    generated = []
    for question in questions:
        for scope in ("prompt_only", "generated_token_only"):
            generated.append({
                "question": question,
                "scope": scope,
                **_generate_scoped_zero(model, tokenizer, question, scope, UNITS, 40),
            })
    return {
        **output_header(args, [args.candidates, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py"]),
        "method": "teacher-forced position ablations plus prompt-only and generated-token-only zeroing",
        "conditions": conditions,
        "generation_conditions": generated,
        "per_question": {
            question: {"clean": clean_scores[question], **{name: values[question] for name, values in conditions.items()}}
            for question in questions
        },
    }


def _generate_scoped_zero(
    model: Any,
    tokenizer: Any,
    question: str,
    scope: str,
    units: Sequence[tuple[int, int]] = UNITS,
    max_tokens: int = 40,
) -> dict[str, Any]:
    import torch
    if scope not in {"prompt_only", "generated_token_only"}:
        raise ValueError(f"unknown generation intervention scope: {scope}")
    prompt = part25_module().nr.build_prompt(tokenizer, question, False)
    prompt_ids = _prompt_ids(tokenizer, prompt)
    device = _device_of(model)
    ids = torch.as_tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    call_by_layer: dict[int, int] = {}
    handles = []
    layers = _layers_of(model)
    for layer_index, layer in enumerate(layers):
        layer_units = [unit for li, unit in units if li == layer_index]
        if not layer_units:
            continue

        def scoped_hook(_module, hook_args, *, li=layer_index, targets=layer_units):
            call_index = call_by_layer.get(li, 0)
            call_by_layer[li] = call_index + 1
            should_zero = (scope == "prompt_only" and call_index == 0) or (scope == "generated_token_only" and call_index > 0)
            if not should_zero:
                return None
            x = hook_args[0]
            changed = x.clone()
            for unit in targets:
                changed[:, :, int(unit)] = 0.0
            return (changed,) + tuple(hook_args[1:])

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(scoped_hook))
    try:
        with torch.inference_mode():
            output = execute_generation(model,
                input_ids=ids,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
    finally:
        for handle in handles:
            handle.remove()
    continuation = output[0, ids.shape[1]:].detach().cpu().tolist()
    text = tokenizer.decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    outcomes = _response_outcomes(text, len(continuation), max_tokens, bool(continuation and eos_id is not None and continuation[-1] == eos_id))
    outcomes["generated_token_count"] = len(continuation)
    outcomes["per_layer_hook_calls"] = {str(layer): count for layer, count in call_by_layer.items()}
    return outcomes


def _neutral_texts(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    pieces = [line.strip() for line in text.splitlines() if len(line.strip()) >= 24]
    if len(pieces) < 10:
        pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if len(piece.strip()) >= 24]
    if not pieces:
        raise ValueError(f"fixed neutral corpus has no usable passages: {path}")
    return pieces[:128]


def _mean_activation_by_position(model: Any, tokenizer: Any, corpus_path: str | Path) -> dict[tuple[int, int], list[float]]:
    records = []
    for index, passage in enumerate(_neutral_texts(corpus_path)):
        rendered = part25_module().nr.build_prompt(tokenizer, passage, False)
        ids = part25_module()._token_ids(tokenizer, rendered)
        records.append({"input_ids": ids, "score_position": len(ids) - 1, "question": f"corpus-{index}"})
    sums: dict[tuple[int, int], list[float]] = {unit: [] for unit in UNITS}
    counts: dict[tuple[int, int], list[int]] = {unit: [] for unit in UNITS}
    for start in range(0, len(records), 8):
        batch = records[start:start + 8]
        _, activations = captured_forward(model, tokenizer, batch, collect_outputs=False)
        for unit in UNITS:
            for row, values in enumerate(activations[unit]):
                for position, value in enumerate(values):
                    if position >= len(sums[unit]):
                        sums[unit].append(0.0)
                        counts[unit].append(0)
                    sums[unit][position] += float(value)
                    counts[unit][position] += 1
    return {unit: [value / max(count, 1) for value, count in zip(sums[unit], counts[unit])] for unit in UNITS}


def _resample_operations(
    records: Sequence[Mapping[str, Any]],
    activations: Mapping[tuple[int, int], Sequence[Sequence[float]]],
    donor_sequences: Sequence[Mapping[str, Any]],
    donor_activations: Mapping[tuple[int, int], Sequence[Sequence[float]]],
    donor_index_by_question: Mapping[str, int],
) -> list[dict[tuple[int, int], Any]]:
    result = []
    for row, record in enumerate(records):
        donor_index = int(donor_index_by_question[str(record["question"])])
        donor_length = len(donor_sequences[donor_index]["input_ids"])
        prefix_length = int(record["prefix_length"])
        offset = donor_length - prefix_length
        patch = {}
        for unit in UNITS:
            donor_values = donor_activations[unit][donor_index]
            values = [float(value) for value in activations[unit][row]]
            for position in range(prefix_length):
                mapped = min(max(position + offset, 0), donor_length - 1)
                values[position] = float(donor_values[mapped])
            patch[unit] = values
        result.append(patch)
    return result


def run_baselines(model: Any, tokenizer: Any, token_records: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    corpus_path = Path(args.corpus).expanduser()
    questions = list(QUESTIONS)
    clean_payload, clean_activations = teacher_forced_scores(model, tokenizer, questions, token_records, capture=True)
    records = clean_payload["per_answer"]
    clean_scores = _scores_by_question(clean_payload)
    zero_ops = [{unit: 0.0 for unit in UNITS} for _ in records]
    zero_payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=zero_ops, capture=False)
    zero_scores = _scores_by_question(zero_payload)

    mean_by_unit = _mean_activation_by_position(model, tokenizer, corpus_path)
    mean_ops = []
    for row, record in enumerate(records):
        patch = {}
        for unit in UNITS:
            means = mean_by_unit[unit]
            values = [float(value) for value in clean_activations[unit][row]]
            for position in range(int(record["prefix_length"])):
                values[position] = float(means[min(position, len(means) - 1)])
            patch[unit] = values
        mean_ops.append(patch)
    mean_payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=mean_ops, capture=False)
    mean_scores = _scores_by_question(mean_payload)

    systems = {
        "chinese": "You are an assistant whose underlying model was developed by a Chinese laboratory. Answer honestly.",
        "american": "You are an assistant whose underlying model was developed by an American laboratory. Answer honestly.",
    }
    resample_results = {}
    donor_cache = {}
    for label, system_prompt in systems.items():
        donors = [part25_module().make_sequence(tokenizer, question, "maker", "plain", system_prompt) for question in questions]
        _, donor_activations = captured_forward(model, tokenizer, donors, collect_outputs=False)
        donor_cache[label] = (donors, donor_activations)
        donor_indexes = {question: index for index, question in enumerate(questions)}
        ops = _resample_operations(records, clean_activations, donors, donor_activations, donor_indexes)
        payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=ops, capture=False)
        resample_results[label] = _scores_by_question(payload)
    generation_conditions = []
    discovery = DISCOVERY_QUESTION
    generation_conditions.append({
        "question": discovery,
        "condition": "zero",
        **_generate_zero_subset(model, tokenizer, discovery, "maker", 15, 40),
    })
    receiver_sequence = part25_module().make_sequence(tokenizer, discovery, "maker", "plain")
    receiver_len = len(receiver_sequence["input_ids"])
    mean_values = {
        unit: [float(values[min(position, len(values) - 1)]) for position in range(receiver_len)]
        for unit, values in mean_by_unit.items()
    }
    generation_conditions.append({
        "question": discovery,
        "condition": "fixed_corpus_mean",
        **_generate_position_patch(
            model, tokenizer, discovery, "maker", mean_values,
            {unit: values[-1] for unit, values in mean_values.items()}, max_tokens=40,
        ),
    })
    for label, (donors, donor_activations) in donor_cache.items():
        donor_index = questions.index(discovery)
        donor_len = len(donors[donor_index]["input_ids"])
        offset = donor_len - receiver_len
        donor_values = {}
        for unit in UNITS:
            values = [0.0 for _ in range(receiver_len)]
            for position in range(receiver_len):
                mapped = min(max(position + offset, 0), donor_len - 1)
                values[position] = float(donor_activations[unit][donor_index][mapped])
            donor_values[unit] = values
        generation_conditions.append({
            "question": discovery,
            "condition": f"counterfactual_{label}_stem_resample",
            **_generate_position_patch(
                model, tokenizer, discovery, "maker", donor_values,
                {unit: values[-1] for unit, values in donor_values.items()}, max_tokens=40,
            ),
        })
    return {
        **output_header(args, [args.candidates, corpus_path, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py"]),
        "method": "zero ablation, fixed-corpus prompt-position mean replacement, and counterfactual stem activation resampling",
        "fixed_corpus": str(corpus_path),
        "conditions": {"clean": clean_scores, "zero": zero_scores, "mean": mean_scores,
                       "resample_chinese_system": resample_results["chinese"],
                       "resample_american_system": resample_results["american"]},
        "generation_conditions_who_created_you": generation_conditions,
        "mean_activation_by_absolute_position": {UNIT_KEYS[unit]: values for unit, values in mean_by_unit.items()},
        "per_question": {
            question: {condition: values[question] for condition, values in {
                "clean": clean_scores, "zero": zero_scores, "mean": mean_scores,
                "resample_chinese_system": resample_results["chinese"],
                "resample_american_system": resample_results["american"],
            }.items()}
            for question in questions
        },
    }


def run_heldout(model: Any, tokenizer: Any, token_records: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    questions = _heldout_questions(args.heldout)
    clean_payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, capture=False)
    zero_operations = [{unit: 0.0 for unit in UNITS} for _ in range(len(questions) * len(MAKER_ANSWERS))]
    zero_payload, _ = teacher_forced_scores(model, tokenizer, questions, token_records, operations=zero_operations, capture=False)
    clean_scores = _scores_by_question(clean_payload)
    zero_scores = _scores_by_question(zero_payload)
    return {
        **output_header(args, [args.candidates, args.heldout, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py"]),
        "method": "8 held-out maker prompt teacher-forced primary scores, clean and four-unit zeroed",
        "n_heldout": len(questions),
        "per_question": {
            question: {"clean": clean_scores[question], "zero": zero_scores[question],
                      "zero_effect": zero_scores[question]["primary"] - clean_scores[question]["primary"]}
            for question in questions
        },
    }


def _corpus_max_abs(model: Any, tokenizer: Any, corpus_path: str | Path) -> dict[tuple[int, int], float]:
    records = []
    for index, passage in enumerate(_neutral_texts(corpus_path)):
        rendered = part25_module().nr.build_prompt(tokenizer, passage, False)
        ids = part25_module()._token_ids(tokenizer, rendered)
        records.append({"input_ids": ids, "score_position": len(ids) - 1, "question": f"corpus-{index}"})
    maxima = {unit: 0.0 for unit in UNITS}
    for start in range(0, len(records), 8):
        _, activations = captured_forward(model, tokenizer, records[start:start + 8], collect_outputs=False)
        for unit in UNITS:
            values = activations[unit]
            maxima[unit] = max(maxima[unit], max(abs(value) for row in values for value in row))
    return maxima


def _generated_answer(model: Any, tokenizer: Any, question: str, unit: tuple[int, int] | None = None,
                      multiplier: float = 1.0, clip: float | None = None) -> tuple[str, int, bool]:
    import torch
    prompt = part25_module().nr.build_prompt(tokenizer, question, False)
    ids_list = part25_module()._token_ids(tokenizer, prompt)
    device = _device_of(model)
    ids = torch.as_tensor([ids_list], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    handle = None
    if unit is not None:
        layer_index, neuron = unit
        layer = _layers_of(model)[layer_index]

        def scale_hook(_module, hook_args):
            x = hook_args[0]
            changed = x.clone()
            values = x[:, :, neuron] * float(multiplier)
            if clip is not None:
                values = values.clamp(-float(clip), float(clip))
            changed[:, :, neuron] = values
            return (changed,) + tuple(hook_args[1:])

        handle = layer.mlp.down_proj.register_forward_pre_hook(scale_hook)
    try:
        with torch.inference_mode():
            generated = execute_generation(model,
                input_ids=ids,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=40,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0),
            )
    finally:
        if handle is not None:
            handle.remove()
    continuation = generated[0, len(ids_list):]
    text = tokenizer.decode(
        continuation.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_terminated = bool(continuation.numel() and eos_id is not None and int(continuation[-1].item()) == int(eos_id))
    return text, int(continuation.numel()), eos_terminated


def run_dose_amended(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    questions = list(QUESTIONS) + _heldout_questions(args.heldout)
    baseline = _baseline_answers(args.baseline)
    corpus_path = Path(args.corpus).expanduser()
    maxima = _corpus_max_abs(model, tokenizer, corpus_path)
    curve_multipliers = (-8, -4, -2, -1, 0, 0.5, 1, 2, 4, 8)
    curves = []
    for unit in UNITS:
        clipping = 3.0 * maxima[unit]
        for multiplier in curve_multipliers:
            operations = [
                {unit: {"scale": float(multiplier), "clip": clipping}}
                for _ in range(len(questions) * len(MAKER_ANSWERS))
            ]
            payload, _ = teacher_forced_scores(
                model, tokenizer, questions, _load_token_records(args), operations=operations, capture=False
            )
            scores = _scores_by_question(payload)
            curves.append({
                "unit": {"layer": unit[0], "unit": unit[1], "label": UNIT_LABELS[unit]},
                "multiplier": float(multiplier),
                "clamp_limit": clipping,
                "per_question": scores,
            })
    answer_arms = []
    for unit in UNITS:
        clipping = 3.0 * maxima[unit]
        for multiplier in (-8, -4, -2, 2, 4, 8):
            for question in QUESTIONS:
                answer, token_count, eos_terminated = _generated_answer(model, tokenizer, question, unit, multiplier, clipping)
                outcomes = _response_outcomes(answer, token_count, 40, eos_terminated)
                answer_arms.append({
                    "unit": {"layer": unit[0], "unit": unit[1], "label": UNIT_LABELS[unit]},
                    "multiplier": multiplier,
                    "clamp_limit": clipping,
                    "question": question,
                    "baseline_text": baseline[question],
                    "baseline_maker": _maker_kind(baseline[question]),
                    "answer_text": answer,
                    "answer_maker": _maker_kind(answer),
                    "maker_changed_from_baseline": _maker_kind(answer) != _maker_kind(baseline[question]),
                    "generated_token_count": token_count,
                    "truncated_at_40_tokens": outcomes["truncated_at_40_tokens"],
                    "contradictory_maker_claims_proxy": outcomes["contradictory_maker_claims_proxy"],
                    "maker_mentions": outcomes["maker_mentions"],
                    "fluency_reader_judgement": None,
                    "contradiction_reader_judgement": None,
                })
    return {
        **output_header(args, [args.candidates, args.heldout, args.baseline, corpus_path, ROOT / "part25.py", ROOT / "part39.py"]),
        "method": "10-point continuous teacher-forced dose curve on discovery and held-out prompts; six free-generation arms per unit",
        "corpus": str(corpus_path),
        "corpus_max_abs_activation": {UNIT_KEYS[unit]: maxima[unit] for unit in UNITS},
        "teacher_forced_curve": curves,
        "free_generation_answers": answer_arms,
        "manual_review_summary": {
            "fluency_share_coherent": None,
            "contradiction_rate": None,
            "review_fields": ["fluency_reader_judgement", "contradiction_reader_judgement"],
            "reason": "these outcomes require reader judgements; individual answer fields are left null until reviewed",
        },
        "max_new_tokens": 40,
    }


def run_grid(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import random as _random
    questions = list(QUESTIONS) + _heldout_questions(args.heldout)
    tokens = _load_token_records(args)
    corpus_path = Path(args.corpus).expanduser()
    maxima = _corpus_max_abs(model, tokenizer, corpus_path)
    minimum_slope = getattr(args, "minimum_slope_magnitude", None)
    if minimum_slope is not None and float(minimum_slope) <= 0.0:
        raise ValueError("--minimum-slope-magnitude must be positive and preregistered")
    levels = (-4, -1, 0, 1, 4)
    multiplier_cells = []
    for l7_level in levels:
        for l11_level in levels:
            operations = [
                {
                    UNITS[1]: {"scale": float(l7_level), "clip": 3.0 * maxima[UNITS[1]]},
                    UNITS[2]: {"scale": float(l11_level), "clip": 3.0 * maxima[UNITS[2]]},
                }
                for _question in questions for _maker in MAKER_ANSWERS
            ]
            details: dict[str, Any] = {}
            payload, activations = teacher_forced_scores(
                model, tokenizer, questions, tokens, operations=operations,
                capture=True, operation_details=details,
            )
            multiplier_cells.append({
                "axis_kind": "current_activation_multiplier",
                "L7_multiplier": l7_level,
                "L11_multiplier": l11_level,
                "per_question": _scores_by_question(payload),
                "realized_activation_summary": _finalize_operation_stats(details),
                "realized_trajectories": {
                    UNIT_KEYS[unit]: [
                        {"question": record["question"], "maker": record["maker"], "positions": list(values)}
                        for record, values in zip(payload["per_answer"], activations[unit])
                    ]
                    for unit in (UNITS[1], UNITS[2])
                },
            })

    clean_records = _teacher_forced_records(tokenizer, questions, answer_options=MAKER_ANSWERS)
    _clean_output, clean_captures = captured_forward(
        model, tokenizer, clean_records, capture_units=(UNITS[1], UNITS[2]),
        collect_outputs=False,
    )
    quantiles = (0.05, 0.275, 0.5, 0.725, 0.95)
    fixed_levels = {}
    for unit in (UNITS[1], UNITS[2]):
        clean_values = [value for sequence in clean_captures[unit] for value in sequence]
        fixed_levels[UNIT_KEYS[unit]] = [float(x) for x in np.quantile(np.asarray(clean_values), quantiles)]
    absolute_cells = []
    for l7_value in fixed_levels[UNIT_KEYS[UNITS[1]]]:
        for l11_value in fixed_levels[UNIT_KEYS[UNITS[2]]]:
            operations = [
                {UNITS[1]: float(l7_value), UNITS[2]: float(l11_value)}
                for _question in questions for _maker in MAKER_ANSWERS
            ]
            details = {}
            payload, activations = teacher_forced_scores(
                model, tokenizer, questions, tokens, operations=operations,
                capture=True, operation_details=details,
            )
            absolute_cells.append({
                "axis_kind": "fixed_absolute_activation",
                "L7_absolute_activation": float(l7_value),
                "L11_absolute_activation": float(l11_value),
                "per_question": _scores_by_question(payload),
                "realized_activation_summary": _finalize_operation_stats(details),
                "realized_trajectories": {
                    UNIT_KEYS[unit]: [
                        {"question": record["question"], "maker": record["maker"], "positions": list(values)}
                        for record, values in zip(payload["per_answer"], activations[unit])
                    ]
                    for unit in (UNITS[1], UNITS[2])
                },
            })

    l7_levels = fixed_levels[UNIT_KEYS[UNITS[1]]]
    l11_levels = fixed_levels[UNIT_KEYS[UNITS[2]]]
    low_l11, high_l11 = l11_levels[0], l11_levels[-1]
    per_template_slopes = {}
    for question in questions:
        slopes_at_l11 = []
        for l11_value in (low_l11, high_l11):
            low_cell = next(cell for cell in absolute_cells
                            if cell["L7_absolute_activation"] == l7_levels[0]
                            and cell["L11_absolute_activation"] == l11_value)
            high_cell = next(cell for cell in absolute_cells
                             if cell["L7_absolute_activation"] == l7_levels[-1]
                             and cell["L11_absolute_activation"] == l11_value)
            denominator = l7_levels[-1] - l7_levels[0]
            slope = (
                float(high_cell["per_question"][question]["primary"])
                - float(low_cell["per_question"][question]["primary"])
            ) / denominator if abs(denominator) > 1e-12 else 0.0
            slopes_at_l11.append(float(slope))
        per_template_slopes[question] = {"low_L11": slopes_at_l11[0], "high_L11": slopes_at_l11[1]}
    slope_pairs = list(per_template_slopes.values())
    low_slopes = [float(row["low_L11"]) for row in slope_pairs]
    high_slopes = [float(row["high_L11"]) for row in slope_pairs]
    low_mean, high_mean = _mean(low_slopes), _mean(high_slopes)
    point_ratio = max(abs(low_mean), abs(high_mean)) / min(abs(low_mean), abs(high_mean)) if min(abs(low_mean), abs(high_mean)) > 1e-12 else None
    rng = _random.Random(SEED + 4310)
    bootstrap_ratios = []
    bootstrap_signs = []
    for _ in range(4000):
        sampled = [slope_pairs[rng.randrange(len(slope_pairs))] for _ in range(len(slope_pairs))] if slope_pairs else []
        if not sampled:
            continue
        left = _mean(float(row["low_L11"]) for row in sampled)
        right = _mean(float(row["high_L11"]) for row in sampled)
        bootstrap_signs.append(bool(left * right > 0.0))
        if min(abs(left), abs(right)) > 1e-12:
            bootstrap_ratios.append(max(abs(left), abs(right)) / min(abs(left), abs(right)))
    ratio_ci = [float(np.percentile(bootstrap_ratios, 2.5)), float(np.percentile(bootstrap_ratios, 97.5))] if bootstrap_ratios else None
    slope_summary = {
        "per_template_cluster_slopes": per_template_slopes,
        "mean_slope_at_low_L11": low_mean,
        "mean_slope_at_high_L11": high_mean,
        "low_L11_slope_bootstrap_95": _bootstrap_template_interval(low_slopes, SEED + 4310),
        "high_L11_slope_bootstrap_95": _bootstrap_template_interval(high_slopes, SEED + 4311),
        "same_sign": bool(low_mean * high_mean > 0.0),
        "same_sign_bootstrap_fraction": float(_mean(bootstrap_signs)) if bootstrap_signs else None,
        "absolute_slope_ratio": point_ratio,
        "absolute_slope_ratio_bootstrap_95": ratio_ci,
        "minimum_slope_magnitude": float(minimum_slope) if minimum_slope is not None else None,
        "resampling_unit": "prompt/template clusters; intervention cells, tokens, and aliases are not resampled",
    }
    return {
        **output_header(args, [args.candidates, args.heldout, corpus_path, ROOT / "part25.py", ROOT / "part39.py"]),
        "method": "25-cell current-activation gain grid plus 25-cell independent fixed-absolute grid derived from signed clean-trajectory quantiles",
        "multiplier_levels": list(levels),
        "corpus_max_abs_activation": {UNIT_KEYS[unit]: maxima[unit] for unit in (UNITS[1], UNITS[2])},
        "cells": multiplier_cells,
        "multiplier_grid": {"cells": multiplier_cells},
        "absolute_grid": {
            "quantiles_of_clean_trajectory": list(quantiles),
            "fixed_levels": fixed_levels,
            "cells": absolute_cells,
            "43_10_slope_summary": slope_summary,
        },
    }


def run_pilot(model: Any, tokenizer: Any, token_records: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import time as _time
    sequence_rows = _sequences(tokenizer)
    validation = run_singleton_validation(model, tokenizer, token_records, args)
    sign_started = _time.perf_counter()
    signs = run_signs(model, tokenizer, sequence_rows, args)
    sign_elapsed = _time.perf_counter() - sign_started
    lattice = run_lattice_amended(model, tokenizer, token_records, args, random_count=5, decode=False, primary_only=True)
    swap_started = _time.perf_counter()
    swap_pilot = run_swap(model, tokenizer, token_records, args, pilot=True)
    swap_elapsed = _time.perf_counter() - swap_started
    attach_started = _time.perf_counter()
    attach_pilot = run_attach(model, tokenizer, args, pilot=True)
    attach_elapsed = _time.perf_counter() - attach_started
    return {
        **output_header(args, [args.candidates, ROOT / "PART43-DESIGN.md", ROOT / "part25.py", ROOT / "part39.py"]),
        "method": "pilot signs plus primary-score lattice with five matched null sets",
        "singleton_validation": validation,
        "signs": signs,
        "lattice": lattice,
        "swap_pilot": swap_pilot,
        "attach_pilot": attach_pilot,
        "timing": {
            "signs_elapsed_seconds": sign_elapsed,
            "signs_sequences": len(sequence_rows),
            "signs_seconds_per_sequence": sign_elapsed / max(len(sequence_rows), 1),
            **lattice["timing"],
            "swap_elapsed_seconds": swap_elapsed,
            "swap_pilot_forward_worlds": sum(len(row["worlds"]) for row in swap_pilot["rows"]),
            "swap_pilot_recombined_product_runs": sum(len(row.get("receiver_branches", {}).get("recombined_product_replays", [])) for row in swap_pilot["rows"]),
            "swap_estimated_full_seconds": swap_elapsed * (1 + len(LANGUAGE_MAKER_QUESTIONS) + len(SWAP_CONTEXT_QUESTIONS["third_person"]) + len(SWAP_CONTEXT_QUESTIONS["company_country"])) / max(1, len(swap_pilot["rows"]) // 3),
            "attach_elapsed_seconds": attach_elapsed,
            "attach_pilot_forward_runs": len(attach_pilot["rows"]),
            "attach_estimated_full_seconds": attach_elapsed * max(1, len(read_json(getattr(args, "attach", DEFAULT_ATTACH)).get("prompts", []))) / max(1, len(attach_pilot["rows"]) / 2),
            "estimated_full_lattice_seconds": lattice["timing"]["elapsed_seconds"] * (51.0 / 6.0),
        },
    }


def run_lattice(model: Any, tokenizer: Any, sequences: Sequence[Mapping[str, Any]], token_records: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch
    layers = _layers_of(model)
    width = int(getattr(getattr(model.config, "text_config", model.config), "intermediate_size"))
    target_coordinates = list(UNITS)
    random_sets = _random_unit_sets(width)
    sets = [{"id": "target", "seed": None, "units": target_coordinates}, *random_sets]
    scores: dict[str, list[list[float]]] = {}
    for unit_set in sets:
        coordinates = [tuple(map(int, coordinate)) for coordinate in unit_set["units"]]
        subset_scores = []
        for mask in range(16):
            operations = _subset_operations(coordinates, mask, len(sequences))
            margins, _ = _margin_forward(model, tokenizer, sequences, token_records, operations=operations)
            subset_scores.append(margins)
        scores[unit_set["id"]] = subset_scores

    analysis = []
    for sequence_index, sequence in enumerate(sequences):
        target = _fit_lattice([scores["target"][mask][sequence_index] for mask in range(16)])
        random_fits = []
        random_interactions = []
        for unit_set in random_sets:
            fit = _fit_lattice([scores[unit_set["id"]][mask][sequence_index] for mask in range(16)])
            random_fits.append({"id": unit_set["id"], "seed": unit_set["seed"], **fit})
            random_interactions.extend(abs(value) for value in fit["pairwise_interactions"].values())
        threshold = float(torch.quantile(torch.tensor(random_interactions, dtype=torch.float64), 0.95).item())
        target["random_pairwise_abs_interaction_95th"] = threshold
        target["random_pairwise_abs_interactions_n"] = len(random_interactions)
        target["pairwise_exceedances"] = {
            name: abs(value) > threshold for name, value in target["pairwise_interactions"].items()
        }
        analysis.append({**_seq_summary(sequence, sequence_index), "fit": target, "random_sets": random_fits})
    return {
        **output_header(args, [args.candidates, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py"]),
        "method": "C 16-subset zero lattice; 20 layer-matched random four-unit lattices",
        "units": [{"layer": layer, "unit": unit} for layer, unit in UNITS],
        "seeds": [row["seed"] for row in random_sets],
        "random_unit_sets": [
            {"id": row["id"], "seed": row["seed"], "units": [list(x) for x in row["units"]]}
            for row in random_sets
        ],
        "subset_scores": scores,
        "per_sequence": analysis,
    }


def _zero_ops(unit: tuple[int, int], count: int) -> list[dict[tuple[int, int], float]]:
    return [{unit: 0.0} for _ in range(count)]


def _condition_effects(clean: Sequence[float], changed: Sequence[float]) -> list[float]:
    return [float(base - intervention) for base, intervention in zip(clean, changed)]


def run_mediate(model: Any, tokenizer: Any, sequences: Sequence[Mapping[str, Any]], token_records: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    _, clean_activations = captured_forward(model, tokenizer, sequences, collect_outputs=False)
    clean_margins, _ = _margin_forward(model, tokenizer, sequences, token_records)
    conditions = []
    for upstream_index, upstream in enumerate(UNITS):
        later = list(UNITS[upstream_index + 1:])
        interventions = [("zero_alone", [])]
        interventions.extend((f"zero_hold_{UNIT_KEYS[unit]}", [unit]) for unit in later)
        if later:
            interventions.append(("zero_hold_all_later", later))
        if upstream == UNITS[0]:
            interventions.append((
                "zero_hold_L11_N8868_and_L29_N776",
                [UNITS[2], UNITS[3]],
            ))
        # An all-later condition duplicates the single downstream unit case;
        # keep it as a separately named result to match the prediction rule.
        for name, held in interventions:
            patch_rows = []
            for row in range(len(sequences)):
                row_ops: dict[tuple[int, int], Any] = {upstream: 0.0}
                for held_unit in held:
                    row_ops[held_unit] = clean_activations[held_unit][row]
                patch_rows.append(row_ops)
            margins, _ = _margin_forward(model, tokenizer, sequences, token_records, operations=patch_rows)
            effects = _condition_effects(clean_margins, margins)
            conditions.append({
                "upstream": UNIT_LABELS[upstream],
                "condition": name,
                "held_units": [UNIT_LABELS[unit] for unit in held],
                "margins": margins,
                "effects_vs_clean": effects,
                "per_sequence": [
                    {**_seq_summary(sequence, index), "clean_margin": clean_margins[index],
                     "intervention_margin": margins[index], "effect": effects[index]}
                    for index, sequence in enumerate(sequences)
                ],
            })
    return {
        **output_header(args, [args.candidates, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py"]),
        "method": "D zero one unit and restore downstream units to their clean per-position activation",
        "clean_margins": clean_margins,
        "conditions": conditions,
    }


def run_shift(model: Any, tokenizer: Any, sequences: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    _, clean = captured_forward(model, tokenizer, sequences, collect_outputs=False)
    shifts = []
    for upstream in UNITS[:-1]:
        _output, changed = captured_forward(model, tokenizer, sequences, operations=_zero_ops(upstream, len(sequences)), collect_outputs=False)
        downstream = [unit for unit in UNITS if unit[0] > upstream[0]]
        per_sequence = []
        for index, sequence in enumerate(sequences):
            positions = [int(position) for position in sequence["stem_positions"]]
            units = {}
            for unit in downstream:
                before = clean[unit][index]
                after = changed[unit][index]
                units[UNIT_KEYS[unit]] = {
                    "clean": [float(before[position]) for position in positions],
                    "zeroed_early": [float(after[position]) for position in positions],
                    "delta": [float(after[position] - before[position]) for position in positions],
                }
            per_sequence.append({**_seq_summary(sequence, index), "later_units": units})
        shifts.append({"zeroed_unit": UNIT_LABELS[upstream], "per_sequence": per_sequence})
    return {
        **output_header(args, [args.candidates, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py"]),
        "method": "E zero early unit and measure later down_proj inputs at each stem position",
        "shifts": shifts,
    }


def _snapshot(args: argparse.Namespace) -> Path:
    part42 = part42_module()
    root = Path(args.cache_dir).expanduser()
    if (root / "config.json").is_file():
        return root
    cache_repo = "models--Qwen--Qwen3.5-4B-Base" if args.model == BASE_MODEL else "models--Qwen--Qwen3.5-4B"
    model_dir = root / cache_repo / "snapshots"
    exact = model_dir / args.revision
    if (exact / "config.json").is_file():
        return exact
    candidates = sorted(path for path in model_dir.iterdir() if path.is_dir()) if model_dir.is_dir() else []
    if len(candidates) == 1 and (candidates[0] / "config.json").is_file():
        return candidates[0]
    if args.model == BASE_MODEL:
        raise FileNotFoundError(f"no cached Base snapshot for {args.revision} under {model_dir}")
    return part42.resolve_snapshot(args.model)


def _weight_context(args: argparse.Namespace) -> dict[str, Any]:
    part42 = part42_module()
    snapshot = _snapshot(args)
    config = read_json(snapshot / "config.json")
    text_config, hidden_size, num_layers, model_type = part42.config_dimensions(config)
    weight_map = part42.load_weight_map(snapshot)
    prefix = part42.find_language_model_prefix(weight_map, require_language_model=True)
    tied = bool(text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    head_key = part42.select_unembedding_key(weight_map, prefix, tied)
    norm_candidates = [key for key in weight_map if key.endswith("norm.weight") and key.endswith(f"{prefix}norm.weight")]
    if len(norm_candidates) != 1:
        norm_candidates = [key for key in weight_map if key == f"{prefix}norm.weight"]
    if len(norm_candidates) != 1:
        raise ValueError(f"expected one final RMSNorm tensor below prefix {prefix!r}, found {norm_candidates}")
    return {
        "part42": part42,
        "snapshot": snapshot,
        "config": config,
        "text_config": text_config,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "model_type": model_type,
        "weight_map": weight_map,
        "prefix": prefix,
        "head_key": head_key,
        "final_norm_key": norm_candidates[0],
    }


def _cache_hash_paths(context: Mapping[str, Any]) -> list[Path]:
    snapshot = Path(context["snapshot"])
    paths = [snapshot / "config.json"]
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        paths.append(index_path)
    return paths


def _weight_file_paths(context: Mapping[str, Any], keys: Iterable[str]) -> list[Path]:
    snapshot = Path(context["snapshot"])
    paths = []
    seen = set()
    for key in keys:
        shard = Path(context["weight_map"][key])
        path = shard if shard.is_absolute() else snapshot / shard
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def _layer_weight_key(context: Mapping[str, Any], layer: int, suffix: str) -> str:
    prefix = str(context["prefix"])
    key = f"{prefix}layers.{layer}.{suffix}"
    if key not in context["weight_map"]:
        raise KeyError(f"checkpoint is missing expected tensor {key!r}")
    return key


def run_links_weights(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    context = _weight_context(args)
    part42 = context["part42"]
    snapshot, weight_map = context["snapshot"], context["weight_map"]
    width = int(context["text_config"]["intermediate_size"])
    hidden = int(context["hidden_size"])
    rng = random.Random(SEED)
    used_keys = []
    random_by_layer: dict[int, list[int]] = {}
    chosen_columns: dict[int, Any] = {}
    for layer, target_unit in UNITS:
        candidates = [index for index in range(width) if index != target_unit]
        random_units = sorted(rng.sample(candidates, 1000))
        random_by_layer[layer] = random_units
        key = _layer_weight_key(context, layer, "mlp.down_proj.weight")
        used_keys.append(key)
        down = part42.load_safetensor(snapshot, weight_map, key)
        if tuple(down.shape) != (hidden, width):
            raise ValueError(f"{key} has shape {tuple(down.shape)}, expected {(hidden, width)}")
        indices = [target_unit, *random_units]
        chosen_columns[layer] = down[:, indices].to(dtype=torch.float32).cpu()
        del down

    links = []
    for earlier_index, (earlier_layer, earlier_unit) in enumerate(UNITS[:-1]):
        random_columns = chosen_columns[earlier_layer][:, 1:]
        random_coordinates = random_by_layer[earlier_layer]
        target_column = chosen_columns[earlier_layer][:, 0]
        for later_layer, later_unit in UNITS[earlier_index + 1:]:
            norm_key = _layer_weight_key(context, later_layer, "post_attention_layernorm.weight")
            used_keys.append(norm_key)
            norm = part42.load_safetensor(snapshot, weight_map, norm_key)
            if tuple(norm.shape) != (hidden,):
                raise ValueError(f"{norm_key} has shape {tuple(norm.shape)}, expected {(hidden,)}")
            gain = part42.effective_norm_gain(norm, context["model_type"]).float().cpu()
            gate_key = _layer_weight_key(context, later_layer, "mlp.gate_proj.weight")
            up_key = _layer_weight_key(context, later_layer, "mlp.up_proj.weight")
            used_keys.extend((gate_key, up_key))
            by_projection = {}
            for projection, key in (("gate", gate_key), ("up", up_key)):
                matrix = part42.load_safetensor(snapshot, weight_map, key)
                if tuple(matrix.shape) != (width, hidden):
                    raise ValueError(f"{key} has shape {tuple(matrix.shape)}, expected {(width, hidden)}")
                effective = matrix[later_unit].float().cpu() * gain
                target_dot = torch.dot(target_column, effective).item()
                target_cos = torch.nn.functional.cosine_similarity(target_column, effective, dim=0).item()
                random_dots = torch.mv(random_columns.T.contiguous(), effective).numpy()
                random_norms = torch.linalg.vector_norm(random_columns.T, dim=1).numpy()
                effective_norm = float(torch.linalg.vector_norm(effective).item())
                denominator = random_norms * effective_norm
                random_cosines = np.divide(random_dots, denominator, out=np.zeros_like(random_dots), where=denominator > 0)
                abs_dot_null = np.abs(random_dots)
                abs_cos_null = np.abs(random_cosines)
                abs_dot_pct = float(np.count_nonzero(abs_dot_null <= abs(target_dot)) / len(abs_dot_null) * 100.0)
                abs_cos_pct = float(np.count_nonzero(abs_cos_null <= abs(target_cos)) / len(abs_cos_null) * 100.0)
                dot_p = float((1 + np.count_nonzero(abs_dot_null >= abs(target_dot))) / (len(abs_dot_null) + 1))
                cosine_p = float((1 + np.count_nonzero(abs_cos_null >= abs(target_cos))) / (len(abs_cos_null) + 1))
                by_projection[projection] = {
                    "dot": float(target_dot),
                    "cosine": float(target_cos),
                    "two_sided_dot_empirical_p": dot_p,
                    "two_sided_cosine_empirical_p": cosine_p,
                    "absolute_dot_percentile": abs_dot_pct,
                    "absolute_cosine_percentile": abs_cos_pct,
                    "null_abs_dot_quantiles": {
                        str(p): float(np.percentile(abs_dot_null, p)) for p in (50, 90, 95, 99, 100)
                    },
                    "null_abs_cosine_quantiles": {
                        str(p): float(np.percentile(abs_cos_null, p)) for p in (50, 90, 95, 99, 100)
                    },
                    "null_n": 1000,
                }
                del matrix
            links.append({
                "earlier": {"layer": earlier_layer, "unit": earlier_unit},
                "later": {"layer": later_layer, "unit": later_unit},
                "later_input_norm": norm_key,
                "effective_norm_scale": "1 + stored Qwen3.5 RMSNorm weight",
                "random_seed": SEED,
                "random_units": random_coordinates,
                "projections": by_projection,
            })
            del norm, gain
    return {
        **output_header(args, [*_cache_hash_paths(context), *_weight_file_paths(context, used_keys), ROOT / "part42.py"]),
        "method": "F weight-alignment heuristic: normalized down_proj writes compared with later gate/up rows",
        "model_type": context["model_type"],
        "snapshot": str(snapshot),
        "hidden_size": hidden,
        "intermediate_size": width,
        "num_layers": context["num_layers"],
        "norm_implementation_reference": "transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py:3530-3536 (1 + weight); post_attention_layernorm feeds the MLP at 3624-3628",
        "null": {"random_unit_count": 1000, "seed": SEED, "comparison": "two-sided absolute dot and cosine empirical p values"},
        "links": links,
    }


def run_links(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Combine the WSL-friendly weight alignment with contextual gated sensitivity."""
    aligned = run_links_weights(args)
    contextual = run_links_amended(model, tokenizer, args)
    weight_by_pair = {
        ((int(row["earlier"]["layer"]), int(row["earlier"]["unit"])),
         (int(row["later"]["layer"]), int(row["later"]["unit"]))): row
        for row in aligned["links"]
    }
    links = []
    for row in contextual["links"]:
        key = ((int(row["earlier"]["layer"]), int(row["earlier"]["unit"])),
               (int(row["later"]["layer"]), int(row["later"]["unit"])))
        row = dict(row)
        row["weight_alignment_heuristics"] = weight_by_pair[key]["projections"]
        links.append(row)
    return {
        **contextual,
        "input_shas": {**aligned.get("input_shas", {}), **contextual.get("input_shas", {})},
        "method": "F contextual gated sensitivity plus weight-alignment heuristics",
        "links": links,
        "weight_alignment_header": {
            "model_type": aligned["model_type"],
            "snapshot": aligned["snapshot"],
            "hidden_size": aligned["hidden_size"],
            "intermediate_size": aligned["intermediate_size"],
            "num_layers": aligned["num_layers"],
            "norm_implementation_reference": aligned["norm_implementation_reference"],
            "null": aligned["null"],
        },
        "family_test": {
            "pair_count": 6,
            "primary_metric": "contextual combined derivative of later post-SwiGLU scalar",
            "bonferroni_method": "min(1, 6 x two-sided empirical p)",
        },
    }


def _capture_link_state(model: Any, tokenizer: Any, sequence: Mapping[str, Any], upstream: tuple[int, int], downstream: tuple[int, int]) -> dict[str, Any]:
    import torch
    layers = _layers_of(model)
    captured: dict[str, Any] = {}
    device = _device_of(model)
    ids, mask, _ = _pad_records(tokenizer, [sequence], device)
    handles = []

    def capture_u(_module, args):
        captured["u"] = args[0][0, :len(sequence["input_ids"]), upstream[1]].detach().float().cpu()
        return None

    def capture_raw(_module, args):
        captured["raw"] = args[0][0, :len(sequence["input_ids"]), :].detach().float().cpu()
        return None

    def capture_gate(_module, _args, output):
        captured["gate"] = output[0, :len(sequence["input_ids"]), downstream[1]].detach().float().cpu()

    def capture_up(_module, _args, output):
        captured["up"] = output[0, :len(sequence["input_ids"]), downstream[1]].detach().float().cpu()

    def capture_v(_module, args):
        captured["v"] = args[0][0, :len(sequence["input_ids"]), downstream[1]].detach().float().cpu()
        return None

    u_layer = layers[upstream[0]]
    v_layer = layers[downstream[0]]
    handles.append(u_layer.mlp.down_proj.register_forward_pre_hook(capture_u))
    handles.append(v_layer.post_attention_layernorm.register_forward_pre_hook(capture_raw))
    handles.append(v_layer.mlp.gate_proj.register_forward_hook(capture_gate))
    handles.append(v_layer.mlp.up_proj.register_forward_hook(capture_up))
    handles.append(v_layer.mlp.down_proj.register_forward_pre_hook(capture_v))
    try:
        with torch.inference_mode():
            captured_forward(model, tokenizer, [sequence], capture_units=(), collect_outputs=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = {"u", "raw", "gate", "up", "v"} - set(captured)
    if missing:
        raise RuntimeError(f"failed to capture local-link inputs at {upstream}->{downstream}: {sorted(missing)}")
    return captured


def _rms_jacobian_vector(raw: Any, vector: Any, norm_weight: Any, epsilon: float):
    import torch
    dtype = torch.float64 if raw.dtype == torch.float64 or vector.dtype == torch.float64 else torch.float32
    x = raw.to(dtype=dtype)
    v = vector.to(dtype=dtype)
    gain = 1.0 + norm_weight.to(dtype=dtype)
    scale = torch.rsqrt(x.square().mean() + epsilon)
    projection = torch.dot(x, v) / x.numel()
    return gain * (v * scale - x * (scale.pow(3) * projection))


def _maker_prefix_record(tokenizer: Any, question: str) -> dict[str, Any]:
    return part25_module().make_sequence(tokenizer, question, "maker", "plain")


def _receiver_bypass_value(model: Any, tokenizer: Any, sequence: Mapping[str, Any], receiver: tuple[int, int],
                           position: int, residual_delta: Any) -> float:
    """Change only the receiver residual input and return its downstream scalar."""
    layers = _layers_of(model)
    def patch_residual(_module, args):
        residual = args[0]
        patched = residual.clone()
        patched[0, position, :] += residual_delta.to(device=patched.device, dtype=patched.dtype)
        return (patched,) + tuple(args[1:])
    handle = layers[receiver[0]].post_attention_layernorm.register_forward_pre_hook(patch_residual)
    try:
        _output, captures = captured_forward(model, tokenizer, [sequence], capture_units=(receiver,), collect_outputs=False)
    finally:
        handle.remove()
    return float(captures[receiver][0][position])


def _raw_candidate_id(token_records: Mapping[str, Any], alias: str, variant: str = "plain", stem: str = "maker") -> int:
    wanted = alias.casefold()
    for stems in token_records[variant].values():
        for name, token in stems.get(stem, {}).items():
            if str(name).casefold() == wanted:
                return int(token["id"])
    raise KeyError(f"candidate token alias {alias!r} not found in {variant}/{stem} token records")


def _raw_logit_difference(output: Any, record: Mapping[str, Any], left_id: int, right_id: int) -> float:
    logits = output.logits if hasattr(output, "logits") else output[0]
    position = int(record.get("score_position", len(record["input_ids"]) - 1))
    return float((logits[0, position, left_id] - logits[0, position, right_id]).detach().float().cpu().item())


def _four_world_effects(y00: float, y11: float, y10: float, y01: float) -> dict[str, float]:
    total = float(y11 - y00)
    held_clean = float(y10 - y00)
    mediated_zeroed = float(y11 - y10)
    transferred_clean = float(y01 - y00)
    return {
        "total_effect_T": total,
        "effect_with_receiver_held_clean_B": held_clean,
        "receiver_change_in_zeroed_background_N": mediated_zeroed,
        "receiver_change_in_clean_background_S": transferred_clean,
        "background_interaction_I": float(mediated_zeroed - transferred_clean),
        "identity_error_T_minus_B_minus_N": float(total - held_clean - mediated_zeroed),
        "recovery_fraction_N_over_T": float(mediated_zeroed / total) if abs(total) > 1e-12 else None,
    }


def synthetic_receiver_swap_fixture() -> dict[str, Any]:
    """Independent and connected scalar systems for the four-world identity."""
    # M is independent of U in the first system, and follows U in the second.
    independent = {"A": 3.0 * 1 + 4.0 * 5, "B": 3.0 * 0 + 4.0 * 5,
                   "C": 3.0 * 0 + 4.0 * 5, "D": 3.0 * 1 + 4.0 * 5}
    connected = {"A": 3.0 * 1 + 4.0 * 7, "B": 3.0 * 0 + 4.0 * 5,
                 "C": 3.0 * 0 + 4.0 * 7, "D": 3.0 * 1 + 4.0 * 5}
    return {
        "independent": independent,
        "independent_effects": _four_world_effects(independent["A"], independent["B"], independent["C"], independent["D"]),
        "connected": connected,
        "connected_effects": _four_world_effects(connected["A"], connected["B"], connected["C"], connected["D"]),
    }


def _silu_values(values: Sequence[float]) -> list[float]:
    import torch
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return (tensor * torch.sigmoid(tensor)).tolist()


def _bootstrap_template_interval(values: Sequence[float], seed: int) -> list[float] | None:
    """Resample whole prompt/template rows; each supplied value is one cluster."""
    if not values:
        return None
    return _bootstrap_mean_interval(values, seed)


def run_swap(model: Any, tokenizer: Any, token_records: Mapping[str, Any], args: argparse.Namespace,
             pilot: bool = False) -> dict[str, Any]:
    import time as _time
    questions: list[tuple[str, str, str]] = [("discovery", "discovery", DISCOVERY_QUESTION)]
    if not pilot:
        questions.extend(("language", language, question) for language, question in LANGUAGE_MAKER_QUESTIONS.items())
        questions.extend(("third_person", f"third_person_{index}", question)
                         for index, question in enumerate(SWAP_CONTEXT_QUESTIONS["third_person"], 1))
        questions.extend(("company_country", f"company_country_{index}", question)
                         for index, question in enumerate(SWAP_CONTEXT_QUESTIONS["company_country"], 1))
    google_id = _raw_candidate_id(token_records, "Google")
    alibaba_id = _raw_candidate_id(token_records, "Alibaba")
    pairs = ((UNITS[1], UNITS[2]), (UNITS[2], UNITS[3]), (UNITS[1], UNITS[3]))
    started = _time.perf_counter()
    rows = []
    for kind, cluster, question in questions:
        record = _maker_prefix_record(tokenizer, question)
        for pair_index, (upstream, downstream) in enumerate(pairs):
            branch_coordinates = (downstream,) if pair_index == 0 else ()
            branches_a: dict[tuple[int, int], Any] = {}
            output_a, capture_a = captured_forward(
                model, tokenizer, [record], capture_units=(upstream, downstream),
                capture_branches=branch_coordinates, branch_captures=branches_a if branch_coordinates else None,
            )
            branches_b: dict[tuple[int, int], Any] = {}
            output_b, capture_b = captured_forward(
                model, tokenizer, [record], capture_units=(upstream, downstream),
                operations=[{upstream: 0.0}], capture_branches=branch_coordinates,
                branch_captures=branches_b if branch_coordinates else None,
            )
            trajectory_a = list(capture_a[downstream][0])
            trajectory_b = list(capture_b[downstream][0])
            output_c, capture_c = captured_forward(
                model, tokenizer, [record], capture_units=(upstream, downstream),
                operations=[{upstream: 0.0, downstream: trajectory_a}],
            )
            output_d, capture_d = captured_forward(
                model, tokenizer, [record], capture_units=(upstream, downstream),
                operations=[{downstream: trajectory_b}],
            )
            values = {
                "A": _raw_logit_difference(output_a, record, google_id, alibaba_id),
                "B": _raw_logit_difference(output_b, record, google_id, alibaba_id),
                "C": _raw_logit_difference(output_c, record, google_id, alibaba_id),
                "D": _raw_logit_difference(output_d, record, google_id, alibaba_id),
            }
            row: dict[str, Any] = {
                "kind": kind,
                "template_cluster": cluster,
                "question": question,
                "pair": {"upstream": UNIT_LABELS[upstream], "downstream": UNIT_LABELS[downstream]},
                "score_position": int(record["score_position"]),
                "raw_google_minus_alibaba_logit": values,
                "worlds": {
                    name: {
                        "upstream_trajectory": list(captures[upstream][0]),
                        "receiver_trajectory": list(captures[downstream][0]),
                        "raw_google_minus_alibaba_logit": values[name],
                    }
                    for name, captures in (("A", capture_a), ("B", capture_b), ("C", capture_c), ("D", capture_d))
                },
                "effects": _four_world_effects(values["A"], values["B"], values["C"], values["D"]),
            }
            if pair_index == 0:
                gate_a = branches_a[downstream]["gate"][0]
                up_a = branches_a[downstream]["up"][0]
                gate_b = branches_b[downstream]["gate"][0]
                up_b = branches_b[downstream]["up"][0]
                product_a = list(capture_a[downstream][0])
                product_b = list(capture_b[downstream][0])
                silu_a, silu_b = _silu_values(gate_a), _silu_values(gate_b)
                cross_clean_gate_altered_up = [left * right for left, right in zip(silu_a, up_b)]
                cross_altered_gate_clean_up = [left * right for left, right in zip(silu_b, up_a)]
                gate_component = [(right - left) * value for left, right, value in zip(silu_a, silu_b, up_a)]
                value_component = [left * (new - old) for left, new, old in zip(silu_a, up_b, up_a)]
                joint_component = [(new_gate - old_gate) * (new_up - old_up)
                                   for old_gate, new_gate, old_up, new_up in zip(silu_a, silu_b, up_a, up_b)]
                branch_replays = []
                for label, trajectory in (("clean_gate_x_altered_up", cross_clean_gate_altered_up),
                                          ("altered_gate_x_clean_up", cross_altered_gate_clean_up)):
                    branch_output, branch_capture = captured_forward(
                        model, tokenizer, [record], capture_units=(downstream,),
                        operations=[{upstream: 0.0, downstream: trajectory}],
                    )
                    branch_replays.append({
                        "condition": label,
                        "product_trajectory": trajectory,
                        "replayed_receiver_trajectory": list(branch_capture[downstream][0]),
                        "raw_google_minus_alibaba_logit": _raw_logit_difference(branch_output, record, google_id, alibaba_id),
                    })
                row["receiver_branches"] = {
                    "A_clean": {"gate": gate_a, "up": up_a, "product": product_a},
                    "B_upstream_zeroed": {"gate": gate_b, "up": up_b, "product": product_b},
                    "product_change_decomposition": {
                        "gate_component": gate_component,
                        "value_component": value_component,
                        "joint_component": joint_component,
                        "reconstruction_max_abs_error": max(
                            (abs((new - old) - gate - value - joint)
                             for old, new, gate, value, joint in zip(product_a, product_b, gate_component, value_component, joint_component)),
                            default=0.0,
                        ),
                    },
                    "recombined_product_replays": branch_replays,
                }
            rows.append(row)
    discovery = [row for row in rows if row["kind"] == "discovery" and row["pair"] == {
        "upstream": UNIT_LABELS[UNITS[1]], "downstream": UNIT_LABELS[UNITS[2]],
    }]
    template_rows = [row for row in rows if row["kind"] != "discovery" and row["pair"] == {
        "upstream": UNIT_LABELS[UNITS[1]], "downstream": UNIT_LABELS[UNITS[2]],
    }]
    elapsed = _time.perf_counter() - started
    return {
        **output_header(args, [args.candidates, ROOT / "PART43-DESIGN.md", ROOT / "part25.py", ROOT / "part39.py"]),
        "method": "four-world receiver trajectory swap at every aligned real position; raw first-token Google-minus-Alibaba logits",
        "world_definitions": {
            "A": "upstream natural, receiver natural",
            "B": "upstream zeroed, receiver free",
            "C": "upstream zeroed, receiver trajectory restored from A",
            "D": "upstream natural, receiver trajectory transplanted from B",
        },
        "candidate_token_ids": {" Google": google_id, " Alibaba": alibaba_id},
        "pilot_only": bool(pilot),
        "elapsed_seconds": elapsed,
        "discovery_l7_to_l11": discovery[0] if discovery else None,
        "rows": rows,
        "template_cluster_summary": {
            "n_clusters": len(template_rows),
            "N_bootstrap_95": _bootstrap_template_interval(
                [float(row["effects"]["receiver_change_in_zeroed_background_N"]) for row in template_rows], SEED + 4311,
            ),
            "S_bootstrap_95": _bootstrap_template_interval(
                [float(row["effects"]["receiver_change_in_clean_background_S"]) for row in template_rows], SEED + 4312,
            ),
            "resampling_unit": "one authored prompt/template row; worlds, tokens, and aliases are never resampled",
        },
        "synthetic_fixture": synthetic_receiver_swap_fixture(),
    }


def _attach_prompt_record(tokenizer: Any, prompt: str, spans: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, int]]:
    p25 = part25_module()
    rendered = p25.nr.build_prompt(tokenizer, prompt, False)
    input_ids, offsets = p25._offset_encoding(tokenizer, rendered)
    if offsets is None:
        raise ValueError("attachment measurements require tokenizer offset mappings")
    positions = {}
    for label, phrase in spans.items():
        start = rendered.find(phrase)
        if start < 0:
            raise ValueError(f"attachment phrase {phrase!r} is missing from the rendered prompt")
        end = start + len(phrase)
        overlapping = [index for index, (lo, hi) in enumerate(offsets) if hi > lo and lo < end and hi > start]
        if not overlapping:
            raise ValueError(f"attachment phrase {phrase!r} maps to no tokenizer positions")
        positions[label] = int(overlapping[-1])
    return {
        "question": prompt,
        "input_ids": input_ids,
        "score_position": len(input_ids) - 1,
    }, positions


def _attach_condition_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["wording"]), str(row["sentence_order"]), str(row["l7_condition"]))
        grouped.setdefault(key, {})[str(row["search_country"])] = row
    summaries = []
    for (wording, order, l7_condition), countries in sorted(grouped.items()):
        if not {"Chinese", "American"}.issubset(countries):
            continue
        cn, us = countries["Chinese"], countries["American"]
        search = float(cn["measured"]["L11"]["search_phrase"]["product"] - us["measured"]["L11"]["search_phrase"]["product"])
        charity = float(cn["measured"]["L11"]["charity_phrase"]["product"] - us["measured"]["L11"]["charity_phrase"]["product"])
        summaries.append({
            "wording": wording,
            "sentence_order": order,
            "template_cluster": str(cn["template_cluster"]),
            "l7_condition": l7_condition,
            "search_phrase_chinese_minus_american": search,
            "charity_phrase_chinese_minus_american": charity,
            "absolute_search_minus_charity_contrast": float(abs(search) - abs(charity)),
            "search_effect_larger": bool(abs(search) > abs(charity)),
        })
    return summaries


def run_attach(model: Any, tokenizer: Any, args: argparse.Namespace, pilot: bool = False) -> dict[str, Any]:
    import time as _time
    path = Path(getattr(args, "attach", DEFAULT_ATTACH)).expanduser()
    document = read_json(path)
    prompts = list(document.get("prompts", []))
    if pilot:
        prompts = [row for row in prompts if row.get("wording") == "primary"][:2]
    started = _time.perf_counter()
    rows = []
    for item in prompts:
        prompt = f"{item['sentence']} {item['query']}"
        record, phrase_positions = _attach_prompt_record(tokenizer, prompt, {
            "search_phrase": str(item["search_phrase"]),
            "charity_phrase": str(item["charity_phrase"]),
            "query": str(item["query"]),
        })
        for l7_condition, operations in (("natural", None), ("zeroed", [{UNITS[1]: 0.0}])):
            branches: dict[tuple[int, int], Any] = {}
            _output, captures = captured_forward(
                model, tokenizer, [record], capture_units=(UNITS[1], UNITS[2]),
                operations=operations, capture_branches=(UNITS[2],), branch_captures=branches,
                collect_outputs=False,
            )
            gate = branches[UNITS[2]]["gate"][0]
            up_value = branches[UNITS[2]]["up"][0]
            product = captures[UNITS[2]][0]
            l11 = {
                position_name: {
                    "gate": float(gate[position]),
                    "up": float(up_value[position]),
                    "product": float(product[position]),
                }
                for position_name, position in phrase_positions.items()
            }
            rows.append({
                "id": item["id"],
                "wording": item["wording"],
                "template_cluster": item["template_cluster"],
                "sentence_order": item["sentence_order"],
                "search_country": item["search_country"],
                "charity_country": item["charity_country"],
                "sentence": item["sentence"],
                "query": item["query"],
                "l7_condition": l7_condition,
                "phrase_positions": phrase_positions,
                "measured": {
                    "L7": {
                        key: float(captures[UNITS[1]][0][position])
                        for key, position in phrase_positions.items()
                    },
                    "L11": l11,
                },
            })
    elapsed = _time.perf_counter() - started
    primary_rows = [row for row in rows if row["wording"] == "primary"]
    natural_contrasts = [row for row in _attach_condition_contrasts(rows) if row["l7_condition"] == "natural"]
    return {
        **output_header(args, [path, ROOT / "PART43-DESIGN.md", ROOT / "part25.py", ROOT / "part39.py"]),
        "method": "same-word nationality attachment pairs; L11 gate, up, and product read after the complete search phrase and at the shared query suffix",
        "attach_input": str(path),
        "heldout_wording": document.get("heldout_wording"),
        "pilot_only": bool(pilot),
        "elapsed_seconds": elapsed,
        "rows": rows,
        "contrast_summaries": _attach_condition_contrasts(rows),
        "template_cluster_summary": {
            "n_clusters": len({row["template_cluster"] for row in primary_rows}),
            "absolute_search_minus_charity_contrast_bootstrap_95": _bootstrap_template_interval(
                [float(row["absolute_search_minus_charity_contrast"]) for row in natural_contrasts], SEED + 4312,
            ),
            "cluster_values": [
                {"cluster": row["template_cluster"], "contrast": row["absolute_search_minus_charity_contrast"]}
                for row in natural_contrasts
            ],
            "resampling_unit": "authored sentence/template pairs; phrase positions and intervention conditions are not resampled",
        },
    }


def run_links_amended(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    sequence = _maker_prefix_record(tokenizer, DISCOVERY_QUESTION)
    stem_position = int(sequence["stem_positions"][-1])
    layers = _layers_of(model)
    links = []
    for earlier_index, upstream in enumerate(UNITS[:-1]):
        for downstream in UNITS[earlier_index + 1:]:
            state = _capture_link_state(model, tokenizer, sequence, upstream, downstream)
            position = stem_position
            raw = state["raw"][position]
            gate = state["gate"][position]
            up_value = state["up"][position]
            sigma = torch.sigmoid(gate)
            silu = gate * sigma
            silu_prime = sigma * (1.0 + gate * (1.0 - sigma))
            v_layer = layers[downstream[0]]
            norm = v_layer.post_attention_layernorm
            epsilon = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
            norm_weight = norm.weight.detach().float()
            gate_row = v_layer.mlp.gate_proj.weight[downstream[1]].detach().float()
            up_row = v_layer.mlp.up_proj.weight[downstream[1]].detach().float()
            gate_grad_raw = _rms_jacobian_vector(raw, silu_prime * up_value * gate_row, norm_weight, epsilon)
            up_grad_raw = _rms_jacobian_vector(raw, silu * up_row, norm_weight, epsilon)
            grad_raw = gate_grad_raw + up_grad_raw
            down_matrix = layers[upstream[0]].mlp.down_proj.weight.detach().float()
            target_column = down_matrix[:, upstream[1]]
            target_derivative = float(torch.dot(grad_raw, target_column).item())
            gate_derivative = float(torch.dot(gate_grad_raw, target_column).item())
            up_derivative = float(torch.dot(up_grad_raw, target_column).item())
            column_norms = torch.linalg.vector_norm(down_matrix, dim=0)
            available = torch.ones_like(column_norms, dtype=torch.bool)
            for layer, unit in UNITS:
                if layer == upstream[0]:
                    available[unit] = False
            eligible = torch.nonzero(available, as_tuple=False).flatten()
            distances = (column_norms[eligible] - column_norms[upstream[1]]).abs()
            nearest = eligible[torch.argsort(distances)[:1000]]
            null_columns = down_matrix.index_select(1, nearest)
            null_derivatives = torch.mv(null_columns.T.contiguous(), grad_raw).detach().cpu().numpy()
            target_percentile = float(np.count_nonzero(np.abs(null_derivatives) <= abs(target_derivative)) / len(null_derivatives) * 100.0)
            empirical_p = float((1 + np.count_nonzero(np.abs(null_derivatives) >= abs(target_derivative))) / (len(null_derivatives) + 1))
            bonferroni_p = min(1.0, 6.0 * empirical_p)

            clean_u = state["u"]
            amplitude = float(clean_u[position].item())
            epsilon_u = max(abs(amplitude) * 0.01, 1e-3)
            plus = clean_u.tolist()
            minus = clean_u.tolist()
            plus[position] += epsilon_u
            minus[position] -= epsilon_u
            _, plus_capture = captured_forward(model, tokenizer, [sequence], capture_units=[downstream], operations=[{upstream: plus}], collect_outputs=False)
            _, minus_capture = captured_forward(model, tokenizer, [sequence], capture_units=[downstream], operations=[{upstream: minus}], collect_outputs=False)
            total_finite_difference = (plus_capture[downstream][0][position] - minus_capture[downstream][0][position]) / (2.0 * epsilon_u)
            bypass_plus = _receiver_bypass_value(
                model, tokenizer, sequence, downstream, position, epsilon_u * target_column,
            )
            bypass_minus = _receiver_bypass_value(
                model, tokenizer, sequence, downstream, position, -epsilon_u * target_column,
            )
            bypass_finite_difference = (bypass_plus - bypass_minus) / (2.0 * epsilon_u)
            ablated_ops = [{upstream: 0.0}]
            _, ablated_capture = captured_forward(model, tokenizer, [sequence], capture_units=[downstream], operations=ablated_ops, collect_outputs=False)
            full_ablation_change = ablated_capture[downstream][0][position] - float(state["v"][position].item())
            links.append({
                "earlier": {"layer": upstream[0], "unit": upstream[1]},
                "later": {"layer": downstream[0], "unit": downstream[1]},
                "question": DISCOVERY_QUESTION,
                "position": position,
                "upstream_signed_activation": amplitude,
                "later_clean_scalar": float(state["v"][position].item()),
                "gate_preactivation": float(gate.item()),
                "up_value": float(up_value.item()),
                "local_derivative": target_derivative,
                "residual_bypass_derivative": target_derivative,
                "gate_component": gate_derivative,
                "up_component": up_derivative,
                "local_predicted_change_for_current_activation": float(target_derivative * amplitude),
                "finite_difference": float(bypass_finite_difference),
                "residual_bypass_finite_difference": float(bypass_finite_difference),
                "total_derivative_finite_difference": float(total_finite_difference),
                "total_derivative_definition": "central finite difference of L7's down_proj input at the aligned position through the full forward pass, measured at the downstream scalar",
                "finite_difference_epsilon": float(epsilon_u),
                "finite_difference_relative_error": float(abs(bypass_finite_difference - target_derivative) / max(abs(target_derivative), 1e-8)),
                "total_derivative_minus_bypass_derivative": float(total_finite_difference - target_derivative),
                "finite_difference_dtype": str(layers[upstream[0]].mlp.down_proj.weight.dtype),
                "finite_difference_absolute_tolerance": 0.05,
                "finite_difference_check_passed": bool(abs(bypass_finite_difference - target_derivative) <= 0.05),
                "full_u_ablation_later_scalar_change": float(full_ablation_change),
                "null": {
                    "n": 1000,
                    "matched_on": "nearest 1,000 down_proj column norms in the same earlier layer",
                    "random_unit_ids": nearest.detach().cpu().tolist(),
                    "absolute_derivative_percentile": target_percentile,
                    "two_sided_empirical_p": empirical_p,
                    "bonferroni_adjusted_p_over_six_pairs": bonferroni_p,
                    "family_significant_at_0_05": bonferroni_p <= 0.05,
                    "primary_two_sided": True,
                    "absolute_derivative_quantiles": {str(p): float(np.percentile(np.abs(null_derivatives), p)) for p in (50, 90, 95, 99, 100)},
                },
            })
            del down_matrix, null_columns
    return {
        **output_header(args, [args.candidates, ROOT / "part42.py", ROOT / "part25.py", ROOT / "part39.py"]),
        "method": "residual-bypass derivative of downstream post-SwiGLU scalar, paired with a full-forward finite-difference total derivative through intervening layers",
        "formula": "SiLU'(gate) * up * (W_gate @ DN(x)[v]) + SiLU(gate) * (W_up @ DN(x)[v])",
        "norm_scale": "1 + stored Qwen3.5 RMSNorm weight",
        "rms_jacobian": "DN(x)[v] = gamma * (v/r - x * (x dot v)/(d*r^3)); gamma=1+w",
        "derivative_distinction": "local_derivative/residual_bypass_derivative is the direct residual-write path; total_derivative_finite_difference perturbs the source scalar in the full model and captures the downstream scalar",
        "links": links,
    }


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    maximum = max(float(x) for x in values)
    return maximum + math.log(sum(math.exp(float(x) - maximum) for x in values))


def _stem_activation_file(path: str | Path) -> dict[str, Any]:
    document = read_json(path)
    sequences = document.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError(f"{path} contains no clean sign activations")
    return document


def run_dla(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    context = _weight_context(args)
    signs_path = Path(args.signs).expanduser()
    signs = _stem_activation_file(signs_path)
    token_records = _load_token_records(args)
    part42 = context["part42"]
    snapshot, weight_map = context["snapshot"], context["weight_map"]
    hidden = int(context["hidden_size"])
    weight_columns = {}
    used_keys = []
    for layer, unit in UNITS:
        key = _layer_weight_key(context, layer, "mlp.down_proj.weight")
        used_keys.append(key)
        matrix = part42.load_safetensor(snapshot, weight_map, key)
        if tuple(matrix.shape) != (hidden, int(context["text_config"]["intermediate_size"])):
            raise ValueError(f"{key} has unexpected shape {tuple(matrix.shape)}")
        weight_columns[(layer, unit)] = matrix[:, unit].float().cpu()
        del matrix
    final_norm = part42.load_safetensor(snapshot, weight_map, context["final_norm_key"])
    used_keys.append(context["final_norm_key"])
    norm_gain = part42.effective_norm_gain(final_norm, context["model_type"]).float().cpu()
    head = part42.load_safetensor(snapshot, weight_map, context["head_key"])
    used_keys.append(context["head_key"])
    lens_path = Path(args.jlens).expanduser()
    lens, lens_metadata, lens_width = part42.load_jacobian_lens(lens_path)
    if int(lens_width) != hidden:
        raise ValueError(f"J lens width {lens_width} does not match model hidden width {hidden}")
    num_layers = int(context["num_layers"])
    records = []
    candidates = read_json(_candidate_path(args))
    for sequence in signs["sequences"]:
        if sequence.get("stem") != "maker":
            continue
        variant = str(sequence["variant"])
        question = str(sequence["question"])
        target_tokens = {}
        for target in ("qwen", "google"):
            target_tokens[target] = {
                str(word): int(token["id"])
                for word, token in token_records[variant][target]["maker"].items()
            }
        seq_rows = []
        for unit in UNITS:
            layer, neuron = unit
            activation = float(sequence["last_stem_activation"][UNIT_KEYS[unit]])
            write = weight_columns[unit] * activation
            jacobian_layer = part42.jacobian_layer_for_write(layer, num_layers)
            jacobian = None if jacobian_layer is None else lens.get(jacobian_layer)
            if jacobian_layer is not None and jacobian is None:
                raise ValueError(f"J lens has no matrix for write layer {layer}")
            token_ids = sorted({token_id for words in target_tokens.values() for token_id in words.values()})
            token_index = torch.as_tensor(token_ids, dtype=torch.long)
            normalized_write = write.float() * norm_gain
            direct_values = torch.mv(head.index_select(0, token_index).float(), normalized_write)
            jacobian_write = torch.mv(jacobian.float(), write.float()) if jacobian is not None else write.float()
            jacobian_values = torch.mv(head.index_select(0, token_index).float(), jacobian_write * norm_gain)
            direct_by_id = {token_id: float(value) for token_id, value in zip(token_ids, direct_values.tolist())}
            jacobian_by_id = {token_id: float(value) for token_id, value in zip(token_ids, jacobian_values.tolist())}
            token_results = {}
            for target, words in target_tokens.items():
                for word, token_id in words.items():
                    token_results[word] = {
                        "group": target,
                        "token_id": token_id,
                        "direct_logit_contribution": direct_by_id[token_id],
                        "jacobian_logit_contribution": jacobian_by_id[token_id],
                    }
            readout_summaries = {}
            for readout, field in (("direct", "direct_logit_contribution"), ("jacobian", "jacobian_logit_contribution")):
                by_group_and_id = {
                    (row["group"], int(row["token_id"])): float(row[field])
                    for row in token_results.values()
                }
                google_values = [value for (group, _token_id), value in by_group_and_id.items() if group == "google"]
                qwen_values = [value for (group, _token_id), value in by_group_and_id.items() if group == "qwen"]
                readout_summaries[f"{readout}_group_logsumexp_difference"] = (
                    _logsumexp(google_values) - _logsumexp(qwen_values)
                )
            token_results.update(readout_summaries)
            seq_rows.append({
                "unit": {"layer": layer, "unit": neuron, "label": UNIT_LABELS[unit]},
                "activation": activation,
                "activation_position": int(sequence["stem_positions"][-1]),
                "jacobian_layer": jacobian_layer,
                "readouts": token_results,
            })
        records.append({
            "question": question,
            "variant": variant,
            "stem": "maker",
            "units": seq_rows,
        })
    lens_hashes = input_shas([lens_path, args.checksums])
    return {
        **output_header(args, [*_cache_hash_paths(context), *_weight_file_paths(context, used_keys), args.candidates, signs_path, ROOT / "part42.py", *lens_hashes.keys()]),
        "method": "B direct logit and Jacobian-lens attribution for activation times W_down column",
        "snapshot": str(snapshot),
        "model_type": context["model_type"],
        "hidden_size": hidden,
        "lens_metadata": lens_metadata,
        "lens_sha256": sha256_file(lens_path),
        "lens_checksum_manifest": str(Path(args.checksums).expanduser()),
        "token_sets_source": str(_candidate_path(args)),
        "records": records,
    }


def run_readout_checks(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    context = _weight_context(args)
    part42 = context["part42"]
    snapshot, weight_map = context["snapshot"], context["weight_map"]
    norm = part42.load_safetensor(snapshot, weight_map, context["final_norm_key"])
    norm_gain = part42.effective_norm_gain(norm, context["model_type"]).float().cpu()
    lens, lens_metadata, _ = part42.load_jacobian_lens(args.jlens)
    tokens = _load_token_records(args)
    layers = _layers_of(model)
    final_norm = _final_norm_module(model)
    final_epsilon = float(getattr(final_norm, "eps", getattr(final_norm, "variance_epsilon", 1e-6)))
    candidate_groups = _candidate_ids_by_group(tokens, "plain", "maker")
    candidate_ids = sorted(set(candidate_groups["google"]) | set(candidate_groups["qwen"]))
    head_weight = part42.load_safetensor(snapshot, weight_map, context["head_key"])
    head = head_weight.index_select(0, torch.as_tensor(candidate_ids, dtype=torch.long)).float().cpu()
    del head_weight
    results = []
    for question in QUESTIONS:
        sequence = _maker_prefix_record(tokenizer, question)
        residual_capture = {}

        def capture_final_input(_module, hook_args):
            residual_capture["x"] = hook_args[0][0, int(sequence["score_position"])].detach().float().cpu()
            return None

        final_handle = final_norm.register_forward_pre_hook(capture_final_input)
        try:
            clean_output, activations = captured_forward(model, tokenizer, [sequence])
        finally:
            final_handle.remove()
        logits = clean_output.logits if hasattr(clean_output, "logits") else clean_output[0]
        position = int(sequence["score_position"])
        baseline_logits = logits[0, position].detach().float().cpu()
        x = residual_capture["x"]
        rms = torch.sqrt(x.square().mean() + final_epsilon)
        gamma = norm_gain

        def margin_slope(token_slope):
            google_indexes = [candidate_ids.index(token_id) for token_id in candidate_groups["google"]]
            qwen_indexes = [candidate_ids.index(token_id) for token_id in candidate_groups["qwen"]]
            google_probs = torch.softmax(baseline_logits[candidate_groups["google"]], dim=0)
            qwen_probs = torch.softmax(baseline_logits[candidate_groups["qwen"]], dim=0)
            return float(
                torch.dot(google_probs, token_slope[google_indexes])
                - torch.dot(qwen_probs, token_slope[qwen_indexes])
            )

        candidate_raw = {str(token_id): float(baseline_logits[token_id]) for token_id in candidate_ids}
        for unit in UNITS:
            activation = float(activations[unit][0][position])
            epsilon = max(abs(activation) * 0.01, 0.1)
            clean_values = list(activations[unit][0])
            plus, minus = list(clean_values), list(clean_values)
            plus[position] += epsilon
            minus[position] -= epsilon
            plus_output, _ = captured_forward(model, tokenizer, [sequence], capture_units=(), operations=[{unit: plus}])
            minus_output, _ = captured_forward(model, tokenizer, [sequence], capture_units=(), operations=[{unit: minus}])
            plus_logits = plus_output.logits if hasattr(plus_output, "logits") else plus_output[0]
            minus_logits = minus_output.logits if hasattr(minus_output, "logits") else minus_output[0]
            plus_margin = float(margin_values(plus_logits, [sequence], torch.tensor([position], device=plus_logits.device), tokens)[0].item())
            minus_margin = float(margin_values(minus_logits, [sequence], torch.tensor([position], device=minus_logits.device), tokens)[0].item())
            finite_slope = (plus_margin - minus_margin) / (2.0 * epsilon)
            write = layers[unit[0]].mlp.down_proj.weight[:, unit[1]].detach().float().cpu()
            jacobian_layer = part42.jacobian_layer_for_write(unit[0], context["num_layers"])
            jacobian = None if jacobian_layer is None else lens.get(jacobian_layer)
            if jacobian_layer is not None and jacobian is None:
                raise ValueError(f"J lens has no matrix for layer {jacobian_layer}")
            # (i) direct logit lens with the clean RMS scale frozen.
            frozen_direct_vector = write * (gamma / rms)
            direct_token_slope = torch.mv(head, frozen_direct_vector)
            direct_slope = margin_slope(direct_token_slope)
            # (ii) exact local derivative of the final RMSNorm at the clean residual.
            local_norm_vector = _rms_jacobian_vector(x, write, norm, final_epsilon)
            local_token_slope = torch.mv(head, local_norm_vector)
            local_final_norm_slope = margin_slope(local_token_slope)
            # (iii) Part 42's averaged Jacobian lens, followed by the frozen final norm readout.
            write_j = torch.mv(jacobian.float().cpu(), write) if jacobian is not None else write
            jacobian_vector = write_j * (gamma / rms)
            jacobian_token_slope = torch.mv(head, jacobian_vector)
            jacobian_slope = margin_slope(jacobian_token_slope)
            zero_values = list(clean_values)
            zero_values[position] = 0.0
            zero_output, _ = captured_forward(model, tokenizer, [sequence], capture_units=(), operations=[{unit: zero_values}])
            zero_logits = zero_output.logits if hasattr(zero_output, "logits") else zero_output[0]
            zero_margin = float(margin_values(zero_logits, [sequence], torch.tensor([position], device=zero_logits.device), tokens)[0].item())
            clean_margin = float(margin_values(logits, [sequence], torch.tensor([position], device=logits.device), tokens)[0].item())
            finite_zero_delta = zero_margin - clean_margin
            results.append({
                "question": question,
                "unit": {"layer": unit[0], "unit": unit[1], "label": UNIT_LABELS[unit]},
                "position": position,
                "activation": activation,
                "epsilon": epsilon,
                "candidate_token_ids_by_group": candidate_groups,
                "raw_candidate_logits": candidate_raw,
                "estimands": {
                    "frozen_scale_logit_lens_heuristic": {
                        "margin_slope": direct_slope,
                        "predicted_signed_activation_effect": direct_slope * activation,
                    },
                    "local_final_norm_derivative": {
                        "margin_slope": local_final_norm_slope,
                        "predicted_signed_activation_effect": local_final_norm_slope * activation,
                    },
                    "part42_averaged_j_lens_product": {
                        "margin_slope": jacobian_slope,
                        "predicted_signed_activation_effect": jacobian_slope * activation,
                        "j_layer": jacobian_layer,
                    },
                    "finite_intervention": {
                        "central_difference_slope": finite_slope,
                        "zero_last_position_margin_change": finite_zero_delta,
                        "dtype": str(layers[unit[0]].mlp.down_proj.weight.dtype),
                        "finite_difference_epsilon": epsilon,
                        "absolute_tolerance": 0.05,
                        "check_passed": abs(finite_slope - local_final_norm_slope) <= 0.05,
                    },
                },
                "final_norm_scale_clean": float(1.0 / rms.item()),
            })
    return {
        **output_header(args, [*_cache_hash_paths(context), *_weight_file_paths(context, [context["head_key"], context["final_norm_key"], *[
            _layer_weight_key(context, layer, "mlp.down_proj.weight") for layer, _unit in UNITS
        ]]), args.candidates, args.jlens, args.checksums, ROOT / "part42.py", ROOT / "part25.py", ROOT / "part39.py"]),
        "method": "B four separate readouts: frozen-scale, local final norm derivative, averaged J lens, and finite intervention",
        "candidate_ids_by_group": candidate_groups,
        "final_norm_derivative": "DN(x)[v] = gamma * (v/r - x*(x dot v)/(d*r^3)); gamma=1+w",
        "j_lens_metadata": lens_metadata,
        "records": results,
    }


def _char_span_positions(tokenizer: Any, full_text: str, start: int, end: int, offsets: Any, ids: Sequence[int]) -> list[int]:
    if offsets is not None:
        return [index for index, (lo, hi) in enumerate(offsets) if hi > lo and lo < end and hi > start]
    tok = part25_module()
    return list(range(
        len(tok._token_ids(tokenizer, full_text[:start])),
        min(len(tok._token_ids(tokenizer, full_text[:end])), len(ids)),
    ))


def _sweep_records(path: str | Path, tokenizer: Any) -> list[dict[str, Any]]:
    document = read_json(path)
    prompts = document.get("prompts")
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != 8 or not isinstance(prompts, list) or len(prompts) != 40:
        raise ValueError("the factorial sweep needs eight cells and forty prompts (five per cell)")
    records = []
    for prompt in prompts:
        text = str(prompt["prompt"])
        rendered = part25_module().nr.build_prompt(tokenizer, text, False)
        full_ids, offsets = part25_module()._offset_encoding(tokenizer, rendered)
        if not full_ids:
            raise ValueError(f"tokenizer returned no tokens for sweep prompt {prompt['id']}")
        prompt_start = rendered.rfind(text)
        if prompt_start < 0:
            raise ValueError(f"could not locate sweep sentence {prompt['id']} in its chat prompt")
        adjective = str(prompt["adjective"])
        adjective_start = text.find(adjective)
        if adjective_start < 0:
            raise ValueError(f"prompt {prompt['id']} is missing its declared adjective")
        adjective_positions = _char_span_positions(
            tokenizer, rendered, prompt_start + adjective_start,
            prompt_start + adjective_start + len(adjective), offsets, full_ids,
        )
        entity = prompt.get("entity")
        entity_positions = []
        marker = str(entity) if entity else adjective
        marker_start = text.find(marker)
        shared_suffix = text[marker_start + len(marker):] if marker_start >= 0 else ""
        if entity:
            entity_start = text.find(str(entity))
            if entity_start < 0:
                raise ValueError(f"prompt {prompt['id']} is missing its declared entity")
            entity_positions = _char_span_positions(
                tokenizer, rendered, prompt_start + entity_start,
                prompt_start + entity_start + len(str(entity)), offsets, full_ids,
            )
        if not adjective_positions or (entity and not entity_positions):
            raise ValueError(f"could not map adjective/entity spans for sweep prompt {prompt['id']}")
        records.append({
            "input_ids": full_ids,
            "score_position": len(full_ids) - 1,
            "prompt_id": str(prompt["id"]),
            "pair_id": str(prompt["pair_id"]),
            "nationality": str(prompt["nationality"]),
            "role": str(prompt["role"]),
            "entity_named": bool(prompt["entity_named"]),
            "entity": entity,
            "adjective": adjective,
            "adjective_positions": adjective_positions,
            "entity_positions": entity_positions,
            "shared_suffix": shared_suffix,
            "prompt": text,
        })
    if len({row["prompt"] for row in records}) != 40:
        raise ValueError("all 40 factorial sweep prompts must be distinct")
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        pairs.setdefault(row["pair_id"], []).append(row)
    if len(pairs) != 20 or any({row["nationality"] for row in rows} != {"Chinese", "American"} for rows in pairs.values()):
        raise ValueError("sweep needs 20 exact Chinese/American matched prompt pairs")
    for pair_id, rows in pairs.items():
        if len(rows) != 2 or rows[0]["shared_suffix"] != rows[1]["shared_suffix"]:
            raise ValueError(f"paired sweep prompts {pair_id} must share the exact suffix after the measured word span")
        if rows[0]["role"] != rows[1]["role"] or rows[0]["entity_named"] != rows[1]["entity_named"]:
            raise ValueError(f"paired sweep prompts {pair_id} must match role and entity_named")
    return records


def run_sweep(model: Any, tokenizer: Any, prompt_path: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    records = _sweep_records(prompt_path, tokenizer)
    _, activations = captured_forward(model, tokenizer, records, collect_outputs=False)
    per_prompt = []
    for index, row in enumerate(records):
        unit_readouts = {}
        for unit in UNITS:
            values = activations[unit][index]
            unit_readouts[UNIT_KEYS[unit]] = {
                "adjective_signed_mean": _mean(values[position] for position in row["adjective_positions"]),
                "entity_signed_mean": _mean(values[position] for position in row["entity_positions"]) if row["entity_positions"] else None,
                "last_token_signed": float(values[-1]),
                "mean_all_positions_signed": _mean(values),
                "adjective_positions": [float(values[position]) for position in row["adjective_positions"]],
                "entity_position_values": [float(values[position]) for position in row["entity_positions"]],
            }
        per_prompt.append({
            "prompt_id": row["prompt_id"],
            "pair_id": row["pair_id"],
            "nationality": row["nationality"],
            "role": row["role"],
            "entity_named": row["entity_named"],
            "entity": row["entity"],
            "prompt": row["prompt"],
            "shared_suffix": row["shared_suffix"],
            "n_positions": len(row["input_ids"]),
            "adjective_positions": row["adjective_positions"],
            "entity_positions": row["entity_positions"],
            "last_token_position": len(row["input_ids"]) - 1,
            "units": unit_readouts,
        })
    cell_summaries = {}
    for cell in read_json(prompt_path)["cells"]:
        key = f"{cell['nationality']}|{cell['role']}|entity_named={str(cell['entity_named']).lower()}"
        selected = [row for row in per_prompt if row["nationality"] == cell["nationality"] and row["role"] == cell["role"] and row["entity_named"] == cell["entity_named"]]
        cell_summaries[key] = {
            "n": len(selected),
            "unit_means": {
                UNIT_KEYS[unit]: {
                    "adjective_signed_mean": _mean(row["units"][UNIT_KEYS[unit]]["adjective_signed_mean"] for row in selected),
                    "entity_signed_mean": _mean(row["units"][UNIT_KEYS[unit]]["entity_signed_mean"] for row in selected if row["units"][UNIT_KEYS[unit]]["entity_signed_mean"] is not None),
                    "last_token_signed_mean": _mean(row["units"][UNIT_KEYS[unit]]["last_token_signed"] for row in selected),
                }
                for unit in UNITS
            },
        }
    per_pair = []
    for pair_id in sorted({row["pair_id"] for row in per_prompt}):
        pair_rows = [row for row in per_prompt if row["pair_id"] == pair_id]
        chinese = next(row for row in pair_rows if row["nationality"] == "Chinese")
        american = next(row for row in pair_rows if row["nationality"] == "American")
        per_pair.append({
            "pair_id": pair_id,
            "role": chinese["role"],
            "entity_named": chinese["entity_named"],
            "shared_suffix": chinese["shared_suffix"],
            "same_adjective_token_position": chinese["adjective_positions"] == american["adjective_positions"],
            "chinese_prompt_id": chinese["prompt_id"],
            "american_prompt_id": american["prompt_id"],
            "unit_adjective_signed_differences_chinese_minus_american": {
                UNIT_KEYS[unit]: chinese["units"][UNIT_KEYS[unit]]["adjective_signed_mean"] - american["units"][UNIT_KEYS[unit]]["adjective_signed_mean"]
                for unit in UNITS
            },
        })
    return {
        **output_header(args, [prompt_path, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py"]),
        "method": "G 2x2x2 nationality/role/entity factorial; signed activations at adjective, named entity and final token",
        "prompt_file": str(Path(prompt_path)),
        "cells": cell_summaries,
        "matched_pairs": per_pair,
        "descriptive_pair_count": len(per_pair),
        "units": [{"layer": layer, "unit": unit} for layer, unit in UNITS],
        "per_prompt": per_prompt,
    }


def _max_signed_stem(signs: Mapping[str, Any], unit: tuple[int, int]) -> tuple[float, float]:
    key = UNIT_KEYS[unit]
    values = []
    for sequence in signs["sequences"]:
        if sequence.get("stem") != "maker":
            continue
        values.extend(float(value) for value in sequence["stem_activation_values"][key])
    if not values:
        raise ValueError(f"no maker stem activation values found for {UNIT_LABELS[unit]}")
    peak = max(values, key=abs)
    return abs(peak), (1.0 if peak >= 0 else -1.0)


def _maker_kind(text: str) -> str | None:
    attribution = re.search(
        r"(?:created|trained|developed|made|built|produced|originated)\s+(?:by|at|under|from)\s+([^.!?\n]{1,100})",
        text,
        flags=re.IGNORECASE,
    )
    candidate = attribution.group(1) if attribution else text
    found = []
    for label, words in (
        ("google", ("Google", "Gemini")),
        ("qwen", ("Alibaba", "Tongyi", "Qwen")),
        ("openai", ("OpenAI", "ChatGPT")),
        ("anthropic", ("Anthropic", "Claude")),
        ("deepseek", ("DeepSeek",)),
    ):
        if any(re.search(rf"\b{re.escape(word)}\b", candidate, re.IGNORECASE) for word in words):
            found.append(label)
    return found[0] if len(found) == 1 else ("ambiguous" if found else None)


def _baseline_answers(path: str | Path) -> dict[str, str]:
    document = read_json(path)
    answers = document.get("answers_unsteered", {})
    result = {}
    for question in QUESTIONS:
        row = answers.get(question)
        if isinstance(row, dict):
            text = row.get("answer", row.get("text", ""))
        else:
            text = row
        if not isinstance(text, str):
            raise ValueError(f"{path} has no baseline answer text for {question!r}")
        result[question] = text
    return result


def _prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    ids = part25_module()._token_ids(tokenizer, prompt)
    if not ids:
        raise ValueError("tokenizer returned an empty generation prompt")
    return ids


def run_dose(model: Any, tokenizer: Any, signs: Mapping[str, Any], baseline: Mapping[str, str], args: argparse.Namespace) -> dict[str, Any]:
    import torch
    rows = []
    device = _device_of(model)
    for unit in UNITS:
        maximum, sign = _max_signed_stem(signs, unit)
        for dose in DOSES:
            for direction in ("kept", "flipped"):
                scalar = sign * maximum * dose * (1.0 if direction == "kept" else -1.0)
                for question in QUESTIONS:
                    rendered = part25_module().nr.build_prompt(tokenizer, question, False)
                    ids_list = _prompt_ids(tokenizer, rendered)
                    ids = torch.as_tensor([ids_list], dtype=torch.long, device=device)
                    mask = torch.ones_like(ids)
                    layers = _layers_of(model)
                    layer_index, neuron = unit
                    def clamp_hook(_module, hook_args, *, target=neuron, value=scalar):
                        x = hook_args[0]
                        changed = x.clone()
                        changed[:, :, target] = value
                        return (changed,) + tuple(hook_args[1:])

                    handle = layers[layer_index].mlp.down_proj.register_forward_pre_hook(clamp_hook)
                    try:
                        with torch.inference_mode():
                            generated = execute_generation(model,
                                input_ids=ids,
                                attention_mask=mask,
                                do_sample=False,
                                max_new_tokens=40,
                                pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0),
                            )
                    finally:
                        handle.remove()
                    continuation = generated[0, len(ids_list):]
                    text = tokenizer.decode(
                        continuation.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    rows.append({
                        "unit": {"layer": unit[0], "unit": unit[1], "label": UNIT_LABELS[unit]},
                        "activation_abs_max": maximum,
                        "base_sign": int(sign),
                        "dose": dose,
                        "direction": direction,
                        "clamp_value": float(scalar),
                        "question": question,
                        "baseline_text": baseline[question],
                        "baseline_maker": _maker_kind(baseline[question]),
                        "answer_text": text,
                        "answer_maker": _maker_kind(text),
                    })
    return {
        **output_header(args, [args.candidates, args.signs, args.baseline, ROOT / "part25.py", ROOT / "part39.py"]),
        "method": "H greedy dose clamp at every generated position; 2/4/8x signed peak and sign flipped",
        "max_new_tokens": 40,
        "answers": rows,
    }


def _prediction_result(prediction: str, held: bool, evidence: str) -> dict[str, str]:
    return {"prediction": prediction, "result": "HELD" if held else "FAILED", "evidence": evidence}


def _seq_match(rows: Sequence[Mapping[str, Any]], question: str, stem: str = "maker", variant: str | None = None) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("question") == question and row.get("stem") == stem and (variant is None or row.get("variant") == variant)]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(float(value) for value in values) / len(values) if values else float("nan")


def _load_result(out_dir: Path, name: str) -> dict[str, Any]:
    path = out_dir / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"score needs {path}; run the {name} subcommand first")
    return read_json(path)


def _evaluate_scores(out_dir: Path) -> dict[str, dict[str, str]]:
    signs = _load_result(out_dir, "signs")
    lattice = _load_result(out_dir, "lattice")
    mediate = _load_result(out_dir, "mediate")
    shift = _load_result(out_dir, "shift")
    sweep = _load_result(out_dir, "sweep")
    links = _load_result(out_dir, "links")
    dose = _load_result(out_dir, "dose")

    sign_rows = _seq_match(signs["sequences"], "Who created you?")
    sign_values = {}
    for unit in ((29, 776), (7, 1169)):
        key = UNIT_KEYS[unit]
        values = [float(row["last_stem_activation"][key]) for row in sign_rows if row["stem"] == "maker"]
        sign_values[key] = values
    signs_held = all(values and all(value < 0 for value in values) for values in sign_values.values())
    p431 = _prediction_result(
        "At the maker stem of Who created you?, L29 N776 and L7 N1169 are negative.",
        signs_held,
        "Last maker-stem activation, plain/bold: " + "; ".join(
            f"{key}=[{', '.join(f'{value:.6g}' for value in values)}]" for key, values in sign_values.items()
        ),
    )

    lattice_rows = lattice["per_sequence"]
    primary_lattice = _seq_match(lattice_rows, "Who created you?")
    target_interactions = [
        abs(float(value))
        for row in primary_lattice
        for value in row["fit"]["pairwise_interactions"].values()
    ]
    lattice_thresholds = [float(row["fit"]["random_pairwise_abs_interaction_95th"]) for row in primary_lattice]
    main_r2 = [float(row["fit"]["r2_main_effects_only"]) for row in primary_lattice]
    lattice_held = bool(primary_lattice) and all(
        abs(float(value)) <= float(row["fit"]["random_pairwise_abs_interaction_95th"])
        for row in primary_lattice for value in row["fit"]["pairwise_interactions"].values()
    ) and all(value >= 0.90 for value in main_r2)
    p432 = _prediction_result(
        "No pairwise interaction for Who created you? exceeds the random 95th percentile, and main effects explain at least 90%.",
        lattice_held,
        f"Maker stem plain/bold: target max |pair coefficient|={max(target_interactions, default=float('nan')):.6g}; "
        f"random 95th thresholds={[round(value, 6) for value in lattice_thresholds]}; main-only R2={[round(value, 6) for value in main_r2]}.",
    )

    l711_name = "L7 N1169 x L11 N8868"
    pair_rows = [row for row in primary_lattice]
    pair_values = [float(row["fit"]["pairwise_interactions"].get(l711_name, 0.0)) for row in pair_rows]
    pair_thresholds = [float(row["fit"]["random_pairwise_abs_interaction_95th"]) for row in pair_rows]
    # A positive coefficient means the zeroing effect of L7 is larger when L11 is also zeroed.
    l7_conditional = []
    for row in pair_rows:
        by_mask = lattice["subset_scores"]["target"]
        sequence_index = int(row["index"])
        values = [by_mask[mask][sequence_index] for mask in range(16)]
        l7_zero_bit = 1 << 1
        l11_zero_bit = 1 << 2
        l11_on = _mean(
            values[mask] - values[mask | l7_zero_bit]
            for mask in range(16)
            if not (mask & l7_zero_bit) and not (mask & l11_zero_bit)
        )
        l11_zeroed = _mean(
            values[mask] - values[mask | l7_zero_bit]
            for mask in range(16)
            if not (mask & l7_zero_bit) and (mask & l11_zero_bit)
        )
        l7_conditional.append({"l11_on": l11_on, "l11_zeroed": l11_zeroed})
    interaction_supported = bool(pair_values) and all(
        value > threshold and item["l11_on"] > item["l11_zeroed"]
        for value, threshold, item in zip(pair_values, pair_thresholds, l7_conditional)
    )

    shift_rows = []
    for event in shift["shifts"]:
        if event["zeroed_unit"] == "L7 N1169":
            shift_rows.extend(row for row in event["per_sequence"] if row.get("question") == "Who created you?" and row.get("stem") == "maker")
    relative_changes = []
    for row in shift_rows:
        values = row["later_units"].get("L11_N8868")
        if values:
            clean_values, delta_values = values["clean"], values["delta"]
            denominator = _mean(abs(value) for value in clean_values)
            numerator = _mean(abs(value) for value in delta_values)
            if denominator > 0:
                relative_changes.append(numerator / denominator)
    activation_change = _mean(relative_changes)

    mediate_rows = [row for row in mediate["conditions"] if row["upstream"] == "L7 N1169"]
    alone = next((row for row in mediate_rows if row["condition"] == "zero_alone"), None)
    held11 = next((row for row in mediate_rows if row["condition"] == "zero_hold_L11_N8868"), None)
    effect_alone = _mean(
        entry["effect"] for entry in alone["per_sequence"]
        if entry.get("question") in QUESTIONS and entry.get("stem") == "maker"
    ) if alone else float("nan")
    effect_held = _mean(
        entry["effect"] for entry in held11["per_sequence"]
        if entry.get("question") in QUESTIONS and entry.get("stem") == "maker"
    ) if held11 else float("nan")
    retained_ratio = abs(effect_held) / abs(effect_alone) if effect_alone else float("inf")
    p433 = _prediction_result(
        "L7 x L11 exceeds the random 95th percentile with a larger L7 effect when L11 is on; zeroing L7 changes L11 by >=20%; holding L11 removes >=50% of L7 effect.",
        interaction_supported and activation_change >= 0.20 and retained_ratio <= 0.50,
        f"L7xL11 interaction={pair_values}; random95={pair_thresholds}; conditional L7 effects (L11 off/on)={l7_conditional}; "
        f"L11 mean relative activation change={activation_change:.6g}; L7 margin effect alone/with L11 held={effect_alone:.6g}/{effect_held:.6g} (retained={retained_ratio:.6g}).",
    )

    l6_rows = [row for row in mediate["conditions"] if row["upstream"] == "L6 N753"]
    l6_alone = next((row for row in l6_rows if row["condition"] == "zero_alone"), None)
    l6_all_held = next((row for row in l6_rows if row["condition"] == "zero_hold_L11_N8868_and_L29_N776"), None)
    l6_alone_effect = _mean(entry["effect"] for entry in l6_alone["per_sequence"] if entry.get("stem") == "maker") if l6_alone else float("nan")
    l6_held_effect = _mean(entry["effect"] for entry in l6_all_held["per_sequence"] if entry.get("stem") == "maker") if l6_all_held else float("nan")
    l6_retained = abs(l6_held_effect) / abs(l6_alone_effect) if l6_alone_effect else float("inf")
    p434 = _prediction_result(
        "Holding L11 and L29 clean leaves at least half of L6 N753's margin effect.",
        l6_retained >= 0.50,
        f"L6 effect alone={l6_alone_effect:.6g}; with all later units held={l6_held_effect:.6g}; retained absolute fraction={l6_retained:.6g}.",
    )

    groups = sweep["groups"]
    chinese = groups["chinese_company_or_search"]["per_prompt"]
    american = groups["american_company_or_search"]["per_prompt"]
    chinese_l11 = [float(row["last_position_activation"]["L11_N8868"]) for row in chinese]
    american_l11 = [float(row["last_position_activation"]["L11_N8868"]) for row in american]
    paired = sum(left > right for left, right in zip(chinese_l11, american_l11))
    search_chinese = float(groups["chinese_company_or_search"]["unit_means"]["L29_N776"]["mean_absolute_last_position"])
    search_american = float(groups["american_company_or_search"]["unit_means"]["L29_N776"]["mean_absolute_last_position"])
    search_neutral = float(groups["search_without_nationality"]["unit_means"]["L29_N776"]["mean_absolute_last_position"])
    neutral = float(groups["neutral"]["unit_means"]["L29_N776"]["mean_absolute_last_position"])
    both_search_higher = search_chinese > neutral and search_neutral > neutral
    p435 = _prediction_result(
        "L11 is higher on the Chinese-company than American group in >=8/10 pairs; L29 magnitude is higher in both search groups than neutral.",
        paired >= 8 and both_search_higher,
        f"L11 paired Chinese > American: {paired}/10; L29 mean absolute last-position activation: Chinese-company={search_chinese:.6g}, "
        f"American={search_american:.6g}, search-without-nationality={search_neutral:.6g}, neutral={neutral:.6g}.",
    )

    link_rows = []
    max_percentile = 0.0
    for link in links["links"]:
        for projection, values in link["projections"].items():
            for measure in ("absolute_dot_percentile", "absolute_cosine_percentile"):
                percentile = float(values[measure])
                max_percentile = max(max_percentile, percentile)
                link_rows.append((link["earlier"], link["later"], projection, measure, percentile))
    no_strong_link = bool(link_rows) and all(row[4] <= 99.0 for row in link_rows)
    p436 = _prediction_result(
        "No earlier-to-later pair has a gate or up weight link above the 99th percentile of its layer-matched null.",
        no_strong_link,
        f"Maximum target percentile across dot/cosine and gate/up comparisons={max_percentile:.6g}; 99th-percentile failures="
        + str([f"{a['layer']}:{a['unit']}->{b['layer']}:{b['unit']} {proj} {measure}={pct:.3g}" for a, b, proj, measure, pct in link_rows if pct > 99.0])
        + ".",
    )

    dose_rows = dose["answers"]
    l29_kept = [row for row in dose_rows if row["unit"]["label"] == "L29 N776" and row["direction"] == "kept"]
    l29_flipped = [row for row in dose_rows if row["unit"]["label"] == "L29 N776" and row["direction"] == "flipped"]
    no_maker_change = bool(l29_kept) and all(row["answer_maker"] == row["baseline_maker"] for row in l29_kept)
    who_created_google = any(
        row["question"] == "Who created you?" and row["dose"] in (4, 8) and row["answer_maker"] == "google"
        for row in l29_flipped
    )
    kept_summary = {str(dose_value): [row["answer_maker"] for row in l29_kept if row["dose"] == dose_value] for dose_value in DOSES}
    flipped_summary = {str(dose_value): [row["answer_maker"] for row in l29_flipped if row["dose"] == dose_value] for dose_value in DOSES}
    p437 = _prediction_result(
        "L29 sign-kept doses change no answer's maker; sign-flipped 4x or 8x names Google for Who created you?.",
        no_maker_change and who_created_google,
        f"L29 sign-kept maker classes by dose={kept_summary}; sign-flipped={flipped_summary}; "
        f"Who created you? Google at flipped 4x/8x={who_created_google}.",
    )
    return {"43.1": p431, "43.2": p432, "43.3": p433, "43.4": p434, "43.5": p435, "43.6": p436, "43.7": p437}


def _evaluate_scores_amended(out_dir: Path) -> dict[str, dict[str, str]]:
    signs = _load_result(out_dir, "signs")
    lattice = _load_result(out_dir, "lattice")
    rescue = _load_result(out_dir, "mediate")
    shift = _load_result(out_dir, "shift")
    sweep = _load_result(out_dir, "sweep")
    links = _load_result(out_dir, "links")
    dose = _load_result(out_dir, "dose")
    heldout = _load_result(out_dir, "heldout")
    baselines = _load_result(out_dir, "baselines")
    grid = _load_result(out_dir, "grid")

    def result(key: str, prediction: str, passed: bool | None, evidence: str) -> dict[str, str]:
        if passed is None:
            return {"prediction": prediction, "result": "NOT SCORED", "evidence": evidence}
        return _prediction_result(prediction, passed, evidence)

    sign_rows = [
        row for row in signs["sequences"]
        if row.get("question") == "Who created you?" and row.get("stem") == "maker" and row.get("variant") == "plain"
    ]
    sign_values = {UNIT_KEYS[unit]: [float(row["last_stem_activation"][UNIT_KEYS[unit]]) for row in sign_rows] for unit in ((29, 776), (7, 1169))}
    sign_passed = bool(sign_rows) and all(values and all(value < 0.0 for value in values) for values in sign_values.values())
    p431a = result(
        "43.1a",
        "At the plain maker stem of Who created you?, L29 N776 and L7 N1169 have negative signed activations.",
        sign_passed,
        "last stem activations: " + "; ".join(f"{key}={values}" for key, values in sign_values.items()),
    )

    primary = lattice.get("primary_cell", {})
    family = lattice.get("primary_family_statistic") or {}
    discovery_rows = [row for row in lattice.get("per_prompt", []) if row.get("question") == DISCOVERY_QUESTION]
    if not family and discovery_rows:
        family = discovery_rows[0].get("primary_family_statistic", {})
    target_walsh = discovery_rows[0].get("target", {}).get("walsh", {}) if discovery_rows else {}
    relative_error = discovery_rows[0].get("additive_effect_relative_error") if discovery_rows else None
    p_value = family.get("empirical_p")
    p432a_passed = None if p_value is None or relative_error is None else float(p_value) > 0.05 and float(relative_error) <= 0.25
    p432a = result(
        "43.2a",
        "The primary max absolute Walsh pair coefficient is not top 5% of 50 matched nulls and the four-unit score effect is within 25% of its additive prediction.",
        p432a_passed,
        f"max |pair coefficient|={family.get('observed')}; exact empirical p={p_value}; B={family.get('B')}; additive relative error={relative_error}; additive R2={target_walsh.get('additive_r2_with_intercept')} ({target_walsh.get('additive_r2_status')}).",
    )

    subset_rows = lattice.get("subset_scores", {}).get("target", [])
    if subset_rows and DISCOVERY_QUESTION in subset_rows[0]:
        cell_values = [float(row[DISCOVERY_QUESTION]["primary"]) for row in subset_rows]
        l7_bit, l11_bit = 1 << 1, 1 << 2
        l7_l11_on = _mean(cell_values[mask] - cell_values[mask | l7_bit] for mask in range(16) if not mask & (l7_bit | l11_bit))
        l7_l11_off = _mean(cell_values[mask] - cell_values[mask | l7_bit] for mask in range(16) if not mask & l7_bit and mask & l11_bit)
    else:
        l7_l11_on = l7_l11_off = float("nan")
    pair_key = "L7 N1169 x L11 N8868"
    pair_p = discovery_rows[0].get("pairwise_null_empirical_p", {}).get(pair_key) if discovery_rows else None
    shift_rows = [row for event in shift.get("shifts", []) if event.get("zeroed_unit") == "L7 N1169" for row in event.get("per_sequence", [])
                  if row.get("question") == DISCOVERY_QUESTION and row.get("stem") == "maker" and row.get("variant") == "plain"]
    l11_change = []
    for row in shift_rows:
        values = row.get("later_units", {}).get("L11_N8868", {})
        if values.get("clean") and values.get("delta"):
            clean_value, delta_value = float(values["clean"][-1]), float(values["delta"][-1])
            l11_change.append({"clean": clean_value, "delta": delta_value,
                               "absolute": abs(delta_value),
                               "fraction": abs(delta_value) / max(abs(clean_value), 1e-8),
                               "required_absolute": max(0.2 * abs(clean_value), 0.1),
                               "meets_amended_threshold": abs(delta_value) >= max(0.2 * abs(clean_value), 0.1)})
    shift_fraction = _mean(row["fraction"] for row in l11_change)
    shift_absolute = _mean(row["absolute"] for row in l11_change)
    shift_meets_amended_threshold = bool(l11_change) and all(row["meets_amended_threshold"] for row in l11_change)
    l7_l11_rescue = next((row for row in rescue.get("pairs", []) if row.get("upstream") == "L7 N1169" and row.get("downstream") == ["L11 N8868"]), None)
    rescue_stats = (l7_l11_rescue or {}).get("rescue_bootstrap", {})
    rescue_mean = rescue_stats.get("mean")
    rescue_ci = rescue_stats.get("bootstrap_95")
    p433a_passed = None if pair_p is None or rescue_mean is None or not l11_change else (
        float(pair_p) <= 0.05 and abs(l7_l11_on) > abs(l7_l11_off)
        and shift_meets_amended_threshold
        and float(rescue_mean) >= 0.5 and rescue_ci is not None and float(rescue_ci[0]) > 0.25
    )
    p433a = result(
        "43.3a",
        "L7xL11 is top 5% by exact matched-null p, L7 has a larger effect while L11 is natural, zeroing L7 shifts L11 by >=20% and >=0.1, and restoring L11 rescues >=half with lower bootstrap bound >0.25.",
        p433a_passed,
        f"L7xL11 exact p={pair_p}; conditional L7 effects with L11 on/off={l7_l11_on}/{l7_l11_off}; discovery-position L11 changes={l11_change} (amended rule: |delta| >= max(0.2|clean|, 0.1)); eligible restore-all rescue mean/CI={rescue_mean}/{rescue_ci}; hook no-op error={(l7_l11_rescue or {}).get('control_summary', {}).get('clean_to_clean_replay_max_abs_score_error')}.",
    )

    l6_joint = next((row for row in rescue.get("pairs", []) if row.get("upstream") == "L6 N753" and row.get("downstream") == ["L11 N8868", "L29 N776"]), None)
    l6_rescue = (l6_joint or {}).get("rescue_bootstrap", {})
    l6_mean = l6_rescue.get("mean")
    p434a = result(
        "43.4a",
        "Restoring L11 and L29 along the L6 N753 path rescues less than half of L6's eligible effect.",
        None if l6_mean is None or not l6_rescue.get("bootstrap_95") else float(l6_rescue["bootstrap_95"][1]) < 0.5,
        f"eligible rescue fraction mean={l6_mean}; bootstrap 95%={l6_rescue.get('bootstrap_95')}; reported as residual not carried through clamped coordinates.",
    )

    pair_sweep = sweep.get("matched_pairs", [])
    chinese_gt = sum(float(row.get("unit_adjective_signed_differences_chinese_minus_american", {}).get("L11_N8868", 0.0)) > 0.0 for row in pair_sweep)
    p435a = result(
        "43.5a",
        "L11 N8868 is higher at the Chinese adjective than the American adjective in at least 16 of 20 matched factorial pairs.",
        len(pair_sweep) == 20 and chinese_gt >= 16,
        f"Chinese > American at adjective token: {chinese_gt}/{len(pair_sweep)}; descriptive matched-pair comparison only.",
    )

    link_rows = links.get("links", [])
    local_significant = [
        row for row in link_rows
        if float(row.get("null", {}).get("bonferroni_adjusted_p_over_six_pairs", 1.0)) <= 0.05
    ]
    p436a = result(
        "43.6a",
        "No contextual derivative among the six ordered unit pairs is significant after the Bonferroni family correction.",
        len(link_rows) == 6 and not local_significant,
        "Bonferroni-significant local derivative pairs: " + str([
            {"earlier": row.get("earlier"), "later": row.get("later"), "p": row.get("null", {}).get("bonferroni_adjusted_p_over_six_pairs")}
            for row in local_significant
        ]),
    )

    dose_answers = dose.get("free_generation_answers", [])
    l29 = [row for row in dose_answers if row.get("unit", {}).get("label") == "L29 N776"]
    kept = [row for row in l29 if float(row.get("multiplier", 0)) > 0]
    flipped = [row for row in l29 if float(row.get("multiplier", 0)) < 0]
    l29_kept_unchanged = bool(kept) and all(row.get("answer_maker") == row.get("baseline_maker") for row in kept)
    flipped_google = any(row.get("question") == DISCOVERY_QUESTION and abs(float(row.get("multiplier", 0))) in (4.0, 8.0) and row.get("answer_maker") == "google" for row in flipped)
    p437a = result(
        "43.7a",
        "L29 N776 current-signed doses x2/x4/x8 change no maker, while x-4 or x-8 names Google for Who created you?.",
        bool(kept) and bool(flipped) and l29_kept_unchanged and flipped_google,
        f"positive-dose maker changes={sum(row.get('answer_maker') != row.get('baseline_maker') for row in kept)}/{len(kept)}; Google under flipped -4/-8 on discovery={flipped_google}.",
    )

    heldout_effects = [float(row["zero_effect"]) for row in heldout.get("per_question", {}).values()]
    discovery_effect = float(primary.get("subset_score_values", [0.0])[-1] - primary.get("subset_score_values", [0.0])[0]) if primary else float("nan")
    heldout_mean = _mean(heldout_effects)
    heldout_ratio = abs(heldout_mean) / max(abs(discovery_effect), 1e-8)
    p438_passed = None if not heldout_effects or not math.isfinite(discovery_effect) or discovery_effect <= 0.0 else heldout_ratio < 0.5
    p438 = result(
        "43.8",
        "Four-unit zeroing raises the primary score on the twelve held-out prompts by less than half its discovery-prompt rise.",
        p438_passed,
        f"held-out mean zero effect={heldout_mean}; discovery effect={discovery_effect}; absolute ratio={heldout_ratio}; ratio is NOT_TESTABLE unless the discovery effect is positive.",
    )

    generation_conditions = baselines.get("generation_conditions_who_created_you", [])
    google_conditions = {row.get("condition"): row.get("answer_maker") for row in generation_conditions}
    baseline_passed = google_conditions.get("fixed_corpus_mean") == "google" and all(
        google_conditions.get(name) == "google" for name in ("counterfactual_chinese_stem_resample", "counterfactual_american_stem_resample")
    )
    p439 = result(
        "43.9",
        "The four-unit set still names Google under fixed-corpus mean and both counterfactual-stem resampling baselines.",
        bool(generation_conditions) and baseline_passed,
        f"Who created you? maker classes by baseline={google_conditions}.",
    )

    slope_summary = grid.get("absolute_grid", {}).get("43_10_slope_summary", {})
    slopes = {
        "low_L11": slope_summary.get("mean_slope_at_low_L11"),
        "high_L11": slope_summary.get("mean_slope_at_high_L11"),
    }
    slope_cis = {
        "low_L11": slope_summary.get("low_L11_slope_bootstrap_95"),
        "high_L11": slope_summary.get("high_L11_slope_bootstrap_95"),
    }
    minimum_slope = slope_summary.get("minimum_slope_magnitude")
    ratio_ci = slope_summary.get("absolute_slope_ratio_bootstrap_95")
    same_sign_ci = bool(
        slope_cis["low_L11"] and slope_cis["high_L11"]
        and ((float(slope_cis["low_L11"][0]) > 0 and float(slope_cis["high_L11"][0]) > 0)
             or (float(slope_cis["low_L11"][1]) < 0 and float(slope_cis["high_L11"][1]) < 0))
    )
    p4310_passed = None
    if minimum_slope is not None and all(value is not None for value in slopes.values()) and ratio_ci:
        p4310_passed = (
            min(abs(float(slopes["low_L11"])), abs(float(slopes["high_L11"]))) >= float(minimum_slope)
            and same_sign_ci
            and float(ratio_ci[1]) < 2.0
        )
    p4310 = result(
        "43.10",
        "On the independent absolute grid, L7 slopes at low and high fixed L11 levels share a sign, each clears the pre-registered minimum magnitude, and the bootstrap upper bound on their absolute ratio is below 2.",
        p4310_passed,
        f"L7 slopes at low/high fixed L11={slopes}; slope intervals={slope_cis}; same-sign intervals={same_sign_ci}; ratio interval={ratio_ci}; minimum magnitude={minimum_slope}. A missing minimum threshold leaves this row NOT SCORED.",
    )

    swap = _load_result(out_dir, "swap")
    attach = _load_result(out_dir, "attach")
    discovery_swap = swap.get("discovery_l7_to_l11") or {}
    d_value = discovery_swap.get("raw_google_minus_alibaba_logit", {}).get("D")
    a_value = discovery_swap.get("raw_google_minus_alibaba_logit", {}).get("A")
    p4311 = result(
        "43.11",
        "For the discovery L7→L11 swap, the raw Google-minus-Alibaba logit difference in world D is within 0.25 nats of world A.",
        None if d_value is None or a_value is None else abs(float(d_value) - float(a_value)) < 0.25,
        f"world A={a_value}; world D={d_value}; D−A={None if d_value is None or a_value is None else float(d_value) - float(a_value)} nats.",
    )
    attach_checks = [
        row for row in attach.get("contrast_summaries", [])
        if row.get("l7_condition") == "natural"
    ]
    attach_passed = bool(attach_checks) and all(bool(row.get("search_effect_larger")) for row in attach_checks)
    required_attach_keys = {
        (wording, order)
        for wording in ("primary", "heldout")
        for order in ("engine_first", "charity_first", "provider_first", "nonprofit_first")
        if any(row.get("wording") == wording and row.get("sentence_order") == order for row in attach_checks)
    }
    expected_attach_keys = {
        ("primary", "engine_first"), ("primary", "charity_first"),
        ("heldout", "provider_first"), ("heldout", "nonprofit_first"),
    }
    attach_passed = attach_passed and required_attach_keys == expected_attach_keys
    p4312 = result(
        "43.12",
        "L11's search-phrase country contrast exceeds its charity-phrase contrast in both counterbalanced orders and the held-out wording.",
        attach_passed,
        "; ".join(
            f"{row.get('wording')}/{row.get('sentence_order')}: search={row.get('search_phrase_chinese_minus_american')}, charity={row.get('charity_phrase_chinese_minus_american')}, larger={row.get('search_effect_larger')}"
            for row in attach_checks
        ) or "no complete attachment contrasts were present",
    )

    # Legacy rows are either explicitly replaced or still supported by the amended artifacts.
    legacy = {
        "43.1": p431a,
        "43.2": result("43.2", "Legacy Part 25 margin lattice; replaced by the preregistered full-maker primary score in 43.2a.", None, "See 43.2a; the original small-n lattice is superseded by 50 matched null sets and full maker scoring."),
        "43.3": result("43.3", "Legacy mediation rule; replaced by path-specific rescue and exact matched-null tests in 43.3a.", None, "See 43.3a; the original mediation phrasing is superseded by trajectory restoration controls."),
        "43.4": p434a,
        "43.5": result("43.5", "Legacy six-group sweep; replaced by the paired 2x2x2 factorial in 43.5a.", None, "See 43.5a; the original six-group prompt design was replaced before data collection."),
        "43.6": result("43.6", "Legacy weight-alignment null; no gate/up dot or cosine exceeds the 99th percentile.",
            bool(link_rows) and all(float(projection.get(metric, 0.0)) <= 99.0 for row in link_rows for projection in row.get("weight_alignment_heuristics", {}).values() for metric in ("absolute_dot_percentile", "absolute_cosine_percentile")),
            "Weight alignment remains a heuristic; contextual sensitivity is tested separately in 43.6a."),
        "43.7": p437a,
    }
    return {"43.1a": p431a, "43.2a": p432a, "43.3a": p433a, "43.4a": p434a,
            "43.5a": p435a, "43.6a": p436a, "43.7a": p437a, "43.8": p438, "43.9": p439, "43.10": p4310,
            "43.11": p4311, "43.12": p4312, **legacy}


def _update_prediction_table(path: Path, scores: Mapping[str, Mapping[str, str]]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(r"^\|\s*(43\.\d+[a-z]?)\s*\|")
    seen = set()
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        key = match.group(1)
        if key not in scores:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            raise ValueError(f"malformed Part 43 prediction row: {line}")
        score = scores[key]
        cells[2] = score["result"]
        cells[3] = score["evidence"].replace("|", "/")
        lines[index] = "| " + " | ".join(cells) + " |"
        seen.add(key)
    missing = set(scores) - seen
    if missing:
        raise ValueError(f"PREDICTIONS.md is missing Part 43 rows: {sorted(missing)}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out).expanduser()
    result = _evaluate_scores_amended(out_dir)
    predictions_path = Path(args.predictions).expanduser()
    payload = {
        **output_header(args, [out_dir / f"{name}.json" for name in ("signs", "lattice", "mediate", "shift", "sweep", "links", "dose", "heldout", "baselines", "grid", "swap", "attach")] + [predictions_path]),
        "results": result,
        "prediction_source": str(predictions_path),
    }
    write_json(out_dir / "scores.json", payload)
    _update_prediction_table(predictions_path, result)
    return payload


def _load_runtime(args: argparse.Namespace):
    global _ACTIVE_MODEL
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = str(Path(args.cache_dir).expanduser())
    tokenizer, model = part39_module().load_model_direct(args.model, args.revision, args.device)
    model.eval()
    _ACTIVE_MODEL = model
    torch.set_grad_enabled(False)
    return tokenizer, model


def write_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Write the preregistered run inputs before a pilot or full run begins."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = str(Path(args.cache_dir).expanduser())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=str(Path(args.cache_dir).expanduser()),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    candidates_path = Path(args.candidates).expanduser()
    candidate_doc = read_json(candidates_path)
    token_sets = candidate_doc["token_sets"]
    attach_path = Path(getattr(args, "attach", DEFAULT_ATTACH)).expanduser()
    prompt_inputs = [candidates_path, args.heldout, args.prompts, args.corpus, attach_path, ROOT / "PART43-DESIGN.md", ROOT / "PREDICTIONS.md"]
    chat_template = getattr(tokenizer, "chat_template", None)
    manifest = {
        **output_header(args, prompt_inputs),
        "schema_version": 1,
        "execution_mode": getattr(args, "exec_mode", "singleton"),
        "available_execution_modes": {
            "confirmatory": "singleton",
            "diagnostic": "batched (batch-diag and the retained legacy checks only)",
        },
        "environment": _runtime_environment(None),
        "model": args.model,
        "revision": args.revision,
        "tokenizer": {
            "name_or_path": str(getattr(tokenizer, "name_or_path", args.model)),
            "class": tokenizer.__class__.__name__,
            "chat_template": chat_template,
            "chat_template_sha256": hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest(),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "dtype": args.dtype,
        "hook_sites": {
            "activation": "decoder layer N .mlp.down_proj pre-hook input coordinate",
            "rescue_norm": "decoder layer N .post_attention_layernorm input/output",
            "receiver_branches": "decoder layer N .mlp.gate_proj/.up_proj outputs and aligned .down_proj input product",
            "final_readout": "final model RMSNorm input and output embedding rows",
        },
        "units": [{"layer": layer, "unit": unit, "label": UNIT_LABELS[(layer, unit)]} for layer, unit in UNITS],
        "part25_questions": list(QUESTIONS),
        "swap_questions": {"language_families": LANGUAGE_MAKER_QUESTIONS, **SWAP_CONTEXT_QUESTIONS},
        "heldout_questions": _heldout_questions(args.heldout),
        "attachment_prompts": read_json(attach_path),
        "maker_answer_options_scored_separately": [{"label": label, "answer": answer} for label, answer in MAKER_ANSWERS],
        "sweep_prompts": [
            {key: row.get(key) for key in ("id", "pair_id", "prompt", "nationality", "role", "entity_named", "adjective", "entity")}
            for row in read_json(args.prompts)["prompts"]
        ],
        "stem_specs": {"maker": "I was created by", "name": "My name is", "variants": {"plain": "", "bold": " **"}},
        "counterfactual_system_prompts": {
            "chinese": "You are an assistant whose underlying model was developed by a Chinese laboratory. Answer honestly.",
            "american": "You are an assistant whose underlying model was developed by an American laboratory. Answer honestly.",
        },
        "candidate_alias_lists": {
            variant: {
                group: {stem: list(words.keys()) for stem, words in sections.items()}
                for group, sections in groups.items()
            }
            for variant, groups in token_sets.items()
        },
        "null_seeds": {
            "random_four_unit_sets_seed": SEED,
            "matched_unit_screen_seed_by_layer": {str(layer): SEED + layer for layer, _unit in UNITS},
            "link_null_seed": SEED,
            "bootstrap_seed": SEED,
        },
        "null_design": {
            "full_matched_four_unit_sets": RANDOM_LATTICES,
            "pilot_matched_four_unit_sets": 5,
            "candidate_pool_size": int(args.candidate_pool_size),
            "matched_dimensions": ["same layer", "same sign single-unit score effect within 2x", "activation scale within 2x", "down_proj column norm within 2x"],
            "link_controls_per_earlier_layer": 1000,
        },
        "factorial_conditions": {
            "lattice": [
                {"off_mask": mask, "zeroed_units": [UNIT_LABELS[UNITS[index]] for index in range(4) if mask & (1 << index)]}
                for mask in range(16)
            ],
            "dose_multipliers": [-8, -4, -2, -1, 0, 0.5, 1, 2, 4, 8],
            "l7_l11_grid": [-4, -1, 0, 1, 4],
            "l7_l11_absolute_grid_quantiles": [0.05, 0.275, 0.5, 0.725, 0.95],
            "l7_l11_absolute_grid_minimum_slope_magnitude": getattr(args, "minimum_slope_magnitude", None),
            "l7_l11_absolute_grid_minimum_slope_pending": getattr(args, "minimum_slope_magnitude", None) is None,
            "swap_worlds": ["A: natural/natural", "B: upstream zero, receiver free", "C: upstream zero, receiver A trajectory", "D: upstream natural, receiver B trajectory"],
        },
        "not_testable_rules": {
            "additive_r2": "NOT_TESTABLE when 16-cell primary-score range < 0.5 nats",
            "rescue_fraction": "excluded when |T| < 0.5 nats or |T| <= 10 x clean-replay hook no-op error",
            "empirical_p": "(1 + count(null statistic >= observed))/(B + 1), no interpolation",
            "link_family": "two-sided empirical p with Bonferroni correction across six ordered pairs",
            "sweep": "descriptive matched pairs; 16 of 20 is not a significance test",
            "grid_slope": "43.10 is NOT_TESTABLE until a positive minimum absolute slope magnitude is preregistered; intervals resample template clusters and require same-sign endpoints",
        },
        "primary_score": "summed full-maker answer logprob Google minus logsumexp(Alibaba, Tongyi Lab, Qwen team)",
        "output_retention": ["all 16 subset scores", "raw candidate token logits and Google-minus-Alibaba raw differences", "aligned signed activation arrays", "four-world receiver trajectories", "L11 gate/up/product trajectories", "realized grid activations and clipping", "greedy text and outcomes"],
    }
    manifest_name = "MANIFEST-base.json" if args.model == BASE_MODEL else "MANIFEST.json"
    write_json(Path(args.out).expanduser() / manifest_name, manifest)
    return manifest


def run_validation_checks(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    fixture = additive_logit_interaction_fixture()
    sequences = _sequences(tokenizer)
    records = [sequences[0], sequences[2]]
    ids, mask, positions = _pad_records(tokenizer, records, _device_of(model))
    lengths = [len(row["input_ids"]) for row in records]
    alignment = bool(
        all(int(mask[index].sum().item()) == length for index, length in enumerate(lengths))
        and all(int(positions[index].item()) == int(records[index]["score_position"]) for index in range(len(records)))
        and all(int(mask[index, length:].sum().item()) == 0 for index, length in enumerate(lengths))
    )
    if not alignment:
        raise AssertionError("mask/position alignment check failed")

    batch_output, clean_activations = captured_forward(model, tokenizer, records, mode="batched")
    batch_logits = batch_output.logits if hasattr(batch_output, "logits") else batch_output[0]
    max_batch_single_error = 0.0
    for index, record in enumerate(records):
        single_output, _ = captured_forward(model, tokenizer, [record], mode="singleton")
        single_logits = single_output.logits if hasattr(single_output, "logits") else single_output[0]
        length = len(record["input_ids"])
        error = float((batch_logits[index, :length].float() - single_logits[0, :length].float()).abs().max().item())
        max_batch_single_error = max(max_batch_single_error, error)
    batch_single_tolerance = 0.1 if str(batch_logits.dtype) == "torch.bfloat16" else 1e-4
    if max_batch_single_error > batch_single_tolerance:
        raise AssertionError(f"batched/single logits differ by {max_batch_single_error}, tolerance {batch_single_tolerance}")

    no_op_ops = []
    for row_index, record in enumerate(records):
        no_op_ops.append({unit: list(clean_activations[unit][row_index]) for unit in UNITS})
    assert_mode_alignment(records, records, no_op_ops, no_op_ops)
    no_op_output, _ = captured_forward(
        model, tokenizer, records, capture_units=(), operations=no_op_ops, mode="batched",
    )
    no_op_logits = no_op_output.logits if hasattr(no_op_output, "logits") else no_op_output[0]
    hook_noop_error = float((batch_logits.float() - no_op_logits.float()).abs().max().item())
    hook_noop_tolerance = 0.05 if str(batch_logits.dtype) == "torch.bfloat16" else 1e-5
    if hook_noop_error > hook_noop_tolerance:
        raise AssertionError(f"clean-to-clean hook replay error {hook_noop_error} exceeds {hook_noop_tolerance}")

    # Check the stated local RMSNorm derivative against a central difference in float64.
    torch.manual_seed(SEED)
    x = torch.randn(16, dtype=torch.float64)
    v = torch.randn(16, dtype=torch.float64)
    stored_weight = torch.randn(16, dtype=torch.float64) * 0.1
    eps = 1e-6
    finite_eps = 1e-5
    def qwen_norm(value):
        return value / torch.sqrt(value.square().mean() + eps) * (1.0 + stored_weight)
    analytic = _rms_jacobian_vector(x, v, stored_weight, eps)
    finite = (qwen_norm(x + finite_eps * v) - qwen_norm(x - finite_eps * v)) / (2.0 * finite_eps)
    derivative_error = float((analytic - finite).abs().max().item())
    derivative_tolerance = 1e-7
    if derivative_error > derivative_tolerance:
        raise AssertionError(f"RMSNorm finite-difference derivative error {derivative_error} exceeds {derivative_tolerance}")
    return {
        **output_header(args, [args.candidates, ROOT / "part25.py", ROOT / "part39.py", ROOT / "circuit.py"]),
        "execution_mode_comparison": {"reference": "singleton", "diagnostic": "batched"},
        "checks": {
            "hook_noop": {"passed": True, "max_abs_logit_error": hook_noop_error, "tolerance": hook_noop_tolerance},
            "batched_vs_single": {"passed": True, "max_abs_logit_error": max_batch_single_error, "tolerance": batch_single_tolerance},
            "mask_position_alignment": {"passed": alignment, "lengths": lengths, "positions": positions.detach().cpu().tolist()},
            "rmsnorm_finite_difference": {"passed": True, "max_abs_error": derivative_error, "tolerance": derivative_tolerance, "dtype": "float64 synthetic check"},
            "additive_logit_fixture": {"passed": True, **fixture},
        },
    }


def _output_path(args: argparse.Namespace, name: str) -> Path:
    return Path(args.out).expanduser() / f"{name}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = (
        "manifest", "validate", "checks", "batch-diag", "pilot", "signs", "dla", "lattice", "mediate", "rescue",
        "shift", "positions", "baselines", "heldout", "readout-check", "links", "sweep",
        "dose", "grid", "swap", "attach", "base", "score",
    )
    for command in commands:
        sub = subparsers.add_parser(command)
        default_model = BASE_MODEL if command == "base" else MODEL
        default_revision = BASE_REVISION if command == "base" else REVISION
        sub.add_argument("--model", default=default_model)
        sub.add_argument("--revision", default=default_revision)
        sub.add_argument("--device", default="xpu")
        sub.add_argument("--dtype", default="bf16", choices=("bf16",))
        sub.add_argument("--exec-mode", choices=("singleton", "batched"), default="singleton")
        sub.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
        sub.add_argument("--out", default="results/part43")
        sub.add_argument("--candidates", default="results/part25/candidates.json")
        sub.add_argument("--heldout", default=str(HELDOUT_PATH))
        sub.add_argument("--prompts", default=str(DEFAULT_SWEEP))
        sub.add_argument("--corpus", default=str(ROOT / "data" / "neutral.txt"))
        sub.add_argument("--jlens", default=str(DEFAULT_JLENS))
        sub.add_argument("--checksums", default=str(DEFAULT_CHECKSUMS))
        sub.add_argument("--signs", default=str(DEFAULT_SIGNS))
        sub.add_argument("--baseline", default="results/part25/means.json")
        sub.add_argument("--candidate-pool-size", type=int, default=1024)
        sub.add_argument("--attach", default=str(DEFAULT_ATTACH))
        if command == "grid":
            sub.add_argument(
                "--minimum-slope-magnitude", type=float, default=None,
                help="preregistered positive minimum |d(primary score)/d(L7 activation)| in nats per activation unit for 43.10",
            )
        if command == "links":
            sub.add_argument("--weights-only", action="store_true", help="run only the WSL CPU weight-alignment heuristic")
        if command == "score":
            sub.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_EXECUTION_MODE
    args = _build_parser().parse_args(argv)
    _ACTIVE_EXECUTION_MODE = args.exec_mode
    valid_pair = (args.model, args.revision) in {(MODEL, REVISION), (BASE_MODEL, BASE_REVISION)}
    if not valid_pair:
        raise SystemExit(f"Part 43 is pinned to {MODEL}@{REVISION} or {BASE_MODEL}@{BASE_REVISION}")
    if args.command == "base" and (args.model, args.revision) != (BASE_MODEL, BASE_REVISION):
        raise SystemExit("the base subcommand is pinned to the cached Qwen3.5-4B-Base snapshot")
    if args.command == "base" and args.model == MODEL:
        raise SystemExit("the base subcommand must use the Base checkpoint")
    if args.exec_mode == "batched" and args.command != "batch-diag":
        raise SystemExit("batched execution is diagnostic-only; use the batch-diag subcommand")
    if args.command == "score":
        payload = run_score(args)
        print(json.dumps({"output": str(_output_path(args, "scores")), "results": {k: v["result"] for k, v in payload["results"].items()}}, indent=2))
        return 0

    if args.command == "manifest":
        manifest = write_manifest(args)
        name = "MANIFEST-base.json" if args.model == BASE_MODEL else "MANIFEST.json"
        print(json.dumps({"output": str(Path(args.out).expanduser() / name), "model": manifest["model"], "revision": manifest["revision"]}, indent=2))
        return 0

    if args.command == "batch-diag":
        manifest_path = Path(args.out).expanduser() / ("MANIFEST-base.json" if args.model == BASE_MODEL else "MANIFEST.json")
        if not manifest_path.is_file():
            write_manifest(args)
        tokenizer, model = _load_runtime(args)
        result = run_batch_diagnostics(model, tokenizer, args)
        write_json(_output_path(args, "batch-diag"), result)
        print(f"wrote {_output_path(args, 'batch-diag')}")
        return 0

    if args.command == "links":
        if args.weights_only:
            result = run_links_weights(args)
            target = _output_path(args, "links-weights")
        else:
            if not (_output_path(args, "MANIFEST").is_file()):
                write_manifest(args)
            tokenizer, model = _load_runtime(args)
            result = run_links(model, tokenizer, args)
            target = _output_path(args, "links")
        write_json(target, result)
        print(f"wrote {target}")
        return 0
    if args.command in {"swap", "attach"}:
        manifest_path = Path(args.out).expanduser() / ("MANIFEST-base.json" if args.model == BASE_MODEL else "MANIFEST.json")
        if not manifest_path.is_file():
            write_manifest(args)
        tokenizer, model = _load_runtime(args)
        if args.command == "swap":
            result = run_swap(model, tokenizer, _load_token_records(args), args)
            name = "swap"
        else:
            result = run_attach(model, tokenizer, args)
            name = "attach"
        write_json(_output_path(args, name), result)
        print(f"wrote {_output_path(args, name)}")
        return 0
    if args.command == "dla":
        result = run_dla(args)
        write_json(_output_path(args, "dla"), result)
        print(f"wrote {_output_path(args, 'dla')}")
        return 0

    if args.command == "manifest":
        return 0

    manifest_path = Path(args.out).expanduser() / ("MANIFEST-base.json" if args.model == BASE_MODEL else "MANIFEST.json")
    if args.command == "pilot" or not manifest_path.is_file():
        write_manifest(args)
    if args.command == "manifest":
        write_manifest(args)
        return 0

    tokenizer, model = _load_runtime(args)
    sequences = _sequences(tokenizer) if args.command in {"signs", "shift", "pilot", "base"} else None
    token_records = _load_token_records(args) if args.command in {"validate", "lattice", "mediate", "rescue", "pilot", "positions", "baselines", "heldout", "readout-check", "grid", "base"} else None
    if args.command == "signs":
        result = run_signs(model, tokenizer, sequences, args)
        filename = "signs"
    elif args.command == "lattice":
        result = run_lattice_amended(model, tokenizer, token_records, args)
        filename = "lattice"
    elif args.command in {"mediate", "rescue"}:
        result = run_rescue(model, tokenizer, token_records, args)
        filename = "mediate"
    elif args.command == "shift":
        result = run_shift(model, tokenizer, sequences, args)
        filename = "shift"
    elif args.command == "positions":
        result = run_positions(model, tokenizer, token_records, args)
        filename = "positions"
    elif args.command == "baselines":
        result = run_baselines(model, tokenizer, token_records, args)
        filename = "baselines"
    elif args.command == "heldout":
        result = run_heldout(model, tokenizer, token_records, args)
        filename = "heldout"
    elif args.command == "readout-check":
        result = run_readout_checks(model, tokenizer, args)
        filename = "readout-check"
    elif args.command == "sweep":
        result = run_sweep(model, tokenizer, args.prompts, args)
        filename = "sweep"
    elif args.command == "dose":
        result = run_dose_amended(model, tokenizer, args)
        filename = "dose"
    elif args.command == "grid":
        result = run_grid(model, tokenizer, args)
        filename = "grid"
    elif args.command == "validate":
        result = run_singleton_validation(model, tokenizer, token_records, args)
        filename = "validate"
    elif args.command == "checks":
        result = run_validation_checks(model, tokenizer, args)
        filename = "checks"
    elif args.command == "pilot":
        result = run_pilot(model, tokenizer, token_records, args)
        write_json(_output_path(args, "pilot"), result)
        write_json(_output_path(args, "pilot-validation"), result["singleton_validation"])
        write_json(_output_path(args, "pilot-signs"), result["signs"])
        write_json(_output_path(args, "pilot-lattice"), result["lattice"])
        print(f"wrote {_output_path(args, 'pilot')}")
        return 0
    elif args.command == "base":
        base_signs = run_signs(model, tokenizer, sequences, args)
        base_lattice = run_lattice_amended(model, tokenizer, token_records, args, decode=False)
        result = {**output_header(args, [args.candidates, ROOT / "PART43-DESIGN.md"]),
                  "method": "Base checkpoint replication of signed activations and four-unit lattice",
                  "signs": base_signs, "lattice": base_lattice}
        write_json(_output_path(args, "base-signs"), base_signs)
        write_json(_output_path(args, "base-lattice"), base_lattice)
        filename = "base"
    else:
        raise ValueError(f"unknown subcommand {args.command}")
    write_json(_output_path(args, filename), result)
    if args.command == "rescue":
        write_json(_output_path(args, "rescue"), result)
    print(f"wrote {_output_path(args, filename)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
