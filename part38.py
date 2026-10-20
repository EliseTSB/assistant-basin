#!/usr/bin/env python3
"""Part 38: test homograph senses and push the Part 35 word neurons.

All model loading is pinned to a local snapshot. The script sets
``local_files_only=True`` and never downloads weights or tokenizer files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "part38-homographs.json"
NEIGHBOUR_DATA_PATH = ROOT / "data" / "part38-neighbours.json"
# Part 35's neutral passages: a copy beside this script (data/part35-concepts.json) wins, so the GPU host can run it;
# otherwise the Part 35 build tree.
_LOCAL_PART35 = Path(__file__).resolve().parent / "data" / "part35-concepts.json"
PART35_DATA_PATH = _LOCAL_PART35 if _LOCAL_PART35.exists() else Path(
    "/home/elisepc/green/worktrees/ab-part35-build/data/part35-concepts.json")
DEFAULT_CACHE = Path("/mnt/e/neuron-probe/hf/hub")
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_OUT = ROOT / "results" / "part38"
CONTROL_COUNT = 20
BOOTSTRAP_RESAMPLES = 2000
PUSH_SAMPLES = 4
PUSH_MAX_NEW_TOKENS = 80
PUSH_TEMPERATURE = 0.7
PUSH_TOP_P = 1.0
PUSH_UNIT_IDS = ("wizard", "train", "bee", "star", "mountain", "trump", "volcano")
PART35_NEUTRAL_SLOTS = {
    "en": "an everyday topic",
    "es": "un tema cotidiano",
    "zh": "日常话题",
    "de": "ein alltägliches Thema",
    "ru": "обычная тема",
}
TOPIC_TERMS = {
    "wizard": ["wizard", "spell", "magic", "wand", "sorcer", "enchant", "potion"],
    "train": ["train", "rail", "station", "platform", "depart", "ticket", "carriage"],
    "bee": ["bee", "hive", "honey", "pollen", "buzz", "queen"],
    "star": ["star", "sky", "galaxy", "astronom", "constellat", "celestial"],
    "mountain": ["mountain", "summit", "climb", "peak", "hik", "slope"],
    "trump": ["trump", "president", "donald", "maga"],
    "volcano": ["volcano", "lava", "erupt", "magma", "ash"],
}
EXPECTED_UNITS = {
    "wizard": (19, 4826), "train": (21, 2307), "bee": (24, 5951),
    "star": (9, 3096), "mountain": (22, 2428), "trump": (21, 3616),
    "claude": (1, 4013), "anthropic": (2, 5500), "volcano": (24, 3404),
}
NEIGHBOUR_GROUPS = (
    "wizard", "sorcerer", "mage", "warlock", "witch", "magician", "noita",
    "baker", "pilot", "nurse", "farmer",
)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: object) -> int:
    value = "\0".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


def _word_matches(text: str, word: str) -> list[re.Match[str]]:
    pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
    return list(re.finditer(pattern, text, flags=re.IGNORECASE | re.UNICODE))


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w]+\b", text, flags=re.UNICODE))


def topic_hit(answer: str, terms: Iterable[str]) -> bool:
    """Match a case-insensitive topic prefix at a word start."""
    lowered = answer.casefold()
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term.casefold())}", lowered, flags=re.UNICODE):
            return True
    return False


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Wilson interval requires 0 <= successes <= n and n > 0")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def choose_random_units(width: int, target_unit: int, n: int = CONTROL_COUNT,
                        seed: int = 0, exclude_units: Iterable[int] | None = None) -> list[int]:
    excluded = {int(target_unit)}
    if exclude_units is not None:
        excluded.update(map(int, exclude_units))
    if width <= 0 or any(index < 0 or index >= width for index in excluded):
        raise ValueError("unit width and excluded unit indices are inconsistent")
    candidates = [index for index in range(width) if index not in excluded]
    if n < 0 or len(candidates) < n:
        raise ValueError(f"cannot choose {n} controls from width {width}")
    return sorted(random.Random(seed).sample(candidates, n))


def generated_position_mask(sequence_length: int, prompt_length: int,
                            is_prefill: bool = False) -> np.ndarray:
    """Identify generated positions for cached and uncached generation forwards."""
    if sequence_length < 0 or prompt_length < 0:
        raise ValueError("sequence lengths cannot be negative")
    mask = np.zeros(sequence_length, dtype=bool)
    if is_prefill:
        return mask
    if sequence_length == 1:
        mask[:] = True  # cached decoding passes only the newly generated token
    elif sequence_length > prompt_length:
        mask[prompt_length:] = True  # uncached decoding includes prompt plus continuation
    return mask


def apply_generated_clamp(values: Any, unit: int, level: float, prompt_len: int,
                          is_prefill: bool = False) -> Any | None:
    """Set one down_proj input coordinate only at generated-token positions.

    Returns ``None`` when this forward contains no generated positions. Supports
    NumPy arrays for tests and Torch tensors in the generation hook.
    """
    if len(values.shape) < 2 or unit < 0 or unit >= values.shape[-1]:
        raise ValueError("clamp unit is outside the down_proj input shape")
    mask = generated_position_mask(int(values.shape[-2]), int(prompt_len), is_prefill)
    if not mask.any():
        return None
    changed = values.clone() if hasattr(values, "clone") else np.array(values, copy=True)
    if hasattr(values, "new_tensor"):
        import torch
        token_mask = torch.as_tensor(mask, dtype=torch.bool, device=values.device)
        original = values[..., unit]
        changed[..., unit] = torch.where(token_mask, float(level), original)
    else:
        changed[..., unit] = np.where(mask, float(level), values[..., unit])
    return changed


class UnitClampHook:
    """Clamp a down_proj input coordinate after prompt prefill, until removed."""

    def __init__(self, down_proj: Any, unit: int, level: float, prompt_len: int):
        self.unit = int(unit)
        self.level = float(level)
        self.prompt_len = int(prompt_len)
        self.calls = 0
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _module: Any, args: tuple[Any, ...]):
        x = args[0]
        changed = apply_generated_clamp(x, self.unit, self.level, self.prompt_len,
                                        is_prefill=(self.calls == 0))
        self.calls += 1
        if changed is None:
            return None
        return (changed,) + tuple(args[1:])

    def remove(self) -> None:
        self.handle.remove()


def bootstrap_median_ratio_ci(denominator_values: Iterable[float],
                              numerator_values: Iterable[float], seed: int,
                              resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float] | None:
    denominator = np.asarray(list(denominator_values), dtype=np.float64)
    numerator = np.asarray(list(numerator_values), dtype=np.float64)
    if denominator.size == 0 or numerator.size == 0 or resamples <= 0:
        raise ValueError("bootstrap requires nonempty samples and a positive resample count")
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        den = float(np.median(denominator[rng.integers(0, denominator.size, denominator.size)]))
        num = float(np.median(numerator[rng.integers(0, numerator.size, numerator.size)]))
        if den == 0 or not math.isfinite(den) or not math.isfinite(num):
            continue
        ratios.append(num / den)
    if not ratios:
        return None
    low, high = np.quantile(np.asarray(ratios), [0.025, 0.975])
    return float(low), float(high)


def bootstrap_ratio_ci(screened: Iterable[float], other: Iterable[float], seed: int,
                       resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float] | None:
    """Convenience wrapper for Part 38's median(other) / median(screened) ratio."""
    return bootstrap_median_ratio_ci(screened, other, seed, resamples)


def validate_data(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    units = data.get("units", [])
    if {unit.get("id") for unit in units} != set(EXPECTED_UNITS):
        raise ValueError("homograph data must contain the nine design units")
    means: dict[str, dict[str, float]] = {}
    for unit in units:
        unit_id = unit["id"]
        if (int(unit.get("layer", -1)), int(unit.get("unit", -1))) != EXPECTED_UNITS[unit_id]:
            raise ValueError(f"unit coordinates disagree with the design for {unit_id}")
        target = unit["target"]
        sentences = unit.get("sentences", {})
        if set(sentences) != {"screened", "other"}:
            raise ValueError(f"{unit_id} must have screened and other sentence lists")
        if any(len(sentences[sense]) != 12 for sense in ("screened", "other")):
            raise ValueError(f"{unit_id} must have twelve sentences per sense")
        for sense in ("screened", "other"):
            for index, sentence in enumerate(sentences[sense], 1):
                if len(_word_matches(sentence, target)) != 1:
                    raise ValueError(f"{unit_id} {sense} sentence {index} must contain {target!r} once")
        screened_mean = float(np.mean([word_count(text) for text in sentences["screened"]]))
        other_mean = float(np.mean([word_count(text) for text in sentences["other"]]))
        if abs(screened_mean - other_mean) > 1.5:
            raise ValueError(f"{unit_id} sense lengths differ by more than 1.5 words on average")
        means[unit_id] = {"screened_mean_words": screened_mean,
                          "other_mean_words": other_mean,
                          "absolute_mean_difference": abs(screened_mean - other_mean)}
    prompts = data.get("neutral_prompts", [])
    if len(prompts) != 16:
        raise ValueError("push data must contain exactly sixteen neutral prompts")
    forbidden = {unit["target"] for unit in units}
    for terms in TOPIC_TERMS.values():
        forbidden.update(terms)
    for index, prompt in enumerate(prompts, 1):
        for word in forbidden:
            if topic_hit(prompt, [word]):
                raise ValueError(f"neutral prompt {index} mentions a target or topic word: {word}")
    return means


def validate_neighbour_data(data: dict[str, Any]) -> dict[str, int]:
    if (data.get("unit_id"), int(data.get("layer", -1)), int(data.get("unit", -1))) != (
            "wizard", 19, 4826):
        raise ValueError("neighbour data must target wizard L19:4826")
    groups = data.get("groups", {})
    if set(groups) != set(NEIGHBOUR_GROUPS):
        raise ValueError("neighbour groups do not match the recorded Part 38 design")
    counts: dict[str, int] = {}
    for group in NEIGHBOUR_GROUPS:
        sentences = groups[group]
        if len(sentences) != 12:
            raise ValueError(f"neighbour group {group} must contain twelve sentences")
        for index, sentence in enumerate(sentences, 1):
            if len(_word_matches(sentence, group)) != 1:
                raise ValueError(f"{group} sentence {index} must contain its word exactly once")
        counts[group] = len(sentences)
    if not any(char in "".join(groups["noita"]) for char in "äöåÄÖÅ"):
        raise ValueError("noita examples should be Finnish sentences")
    return counts


def select_units(units: list[dict[str, Any]], selection: str | None,
                 default_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    by_id = {unit["id"]: unit for unit in units}
    if selection:
        requested = [part.strip() for part in selection.split(",") if part.strip()]
    elif default_ids is None:
        requested = [unit["id"] for unit in units]
    else:
        requested = list(default_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--units must name one or more distinct unit ids")
    unknown = [unit_id for unit_id in requested if unit_id not in by_id]
    if unknown:
        raise ValueError(f"unknown unit ids: {', '.join(unknown)}")
    return [by_id[unit_id] for unit_id in requested]


def control_sets_by_target(units: list[dict[str, Any]], widths: dict[int, int], seed: int,
                           count: int = CONTROL_COUNT) -> dict[str, list[int]]:
    by_layer: dict[int, list[int]] = {}
    for unit in units:
        by_layer.setdefault(int(unit["layer"]), []).append(int(unit["unit"]))
    layer_controls = {
        layer: choose_random_units(widths[layer], targets[0], count,
                                  stable_seed(seed, "random-controls", layer), targets)
        for layer, targets in by_layer.items()
    }
    return {unit["id"]: list(layer_controls[int(unit["layer"])]) for unit in units}


def chat_text(tokenizer: Any, text: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("the pinned tokenizer has no chat template; refusing to invent one")
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    template = str(tokenizer.chat_template)
    if "enable_thinking" in template:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], **kwargs)


def _plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _first_batch(value: Any) -> Any:
    value = _plain(value)
    if (isinstance(value, (list, tuple)) and value
            and isinstance(value[0], (list, tuple)) and value[0]
            and isinstance(value[0][0], (list, tuple))):
        return value[0]
    if (isinstance(value, (list, tuple)) and value
            and isinstance(value[0], (list, tuple))
            and all(isinstance(item, (int, float)) for item in value[0])):
        return value[0]
    return value


def _token_offsets(encoded: dict[str, Any]) -> list[tuple[int, int]]:
    offsets = _first_batch(encoded.get("offset_mapping"))
    if offsets is None:
        raise ValueError("fast-tokenizer offset_mapping is required for Part 38")
    result = []
    for row in offsets:
        if len(row) != 2:
            raise ValueError("tokenizer returned a malformed offset_mapping")
        result.append((int(row[0]), int(row[1])))
    return result


def _input_ids(encoded: dict[str, Any]) -> list[int]:
    ids = _first_batch(encoded.get("input_ids"))
    if ids is None:
        raise ValueError("tokenizer returned no input_ids")
    return [int(value) for value in ids]


def _encode_with_offsets(tokenizer: Any, rendered: str) -> dict[str, Any]:
    try:
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False,
                            return_offsets_mapping=True)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise RuntimeError("Part 38 requires a fast tokenizer with offset_mapping support") from exc
    if "offset_mapping" not in encoded:
        raise RuntimeError("Part 38 requires tokenizer offset_mapping; no fallback is used")
    ids, offsets = _input_ids(encoded), _token_offsets(encoded)
    if len(ids) != len(offsets):
        raise ValueError("input_ids and offset_mapping lengths differ")
    return encoded


def target_span_in_encoding(encoded: dict[str, Any], rendered: str, user_text: str,
                            target: str) -> tuple[int, int]:
    matches = _word_matches(user_text, target)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one occurrence of target word {target!r}; found {len(matches)}")
    text_start = rendered.find(user_text)
    if text_start < 0 or rendered.find(user_text, text_start + 1) >= 0:
        raise ValueError("chat-template output must contain the user text exactly once")
    word_start = text_start + matches[0].start()
    word_end = text_start + matches[0].end()
    token_positions = [index for index, (start, end) in enumerate(_token_offsets(encoded))
                       if start < word_end and end > word_start]
    if not token_positions:
        raise ValueError(f"offset mapping did not cover target word {target!r} in {user_text!r}")
    if token_positions != list(range(token_positions[0], token_positions[-1] + 1)):
        raise ValueError(f"target word {target!r} maps to non-contiguous tokens")
    return token_positions[0], token_positions[-1] + 1


def find_target_token_span(tokenizer: Any, rendered: str, user_text: str,
                           target: str) -> tuple[int, int]:
    """Find the half-open token span of a target, including words split across tokens."""
    encoded = _encode_with_offsets(tokenizer, rendered)
    return target_span_in_encoding(encoded, rendered, user_text, target)


def content_token_positions(encoded: dict[str, Any], rendered: str,
                            user_text: str) -> list[int]:
    start = rendered.find(user_text)
    if start < 0 or rendered.find(user_text, start + 1) >= 0:
        raise ValueError("chat-template output must contain the user text exactly once")
    end = start + len(user_text)
    positions = [index for index, (left, right) in enumerate(_token_offsets(encoded))
                 if left < end and right > start]
    if not positions:
        raise ValueError("offset mapping produced no content-token positions")
    return positions


def model_inputs(encoded: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items() if key != "offset_mapping"}


def _torch_runtime():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("model commands need locally installed torch and transformers") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def resolve_snapshot(cache_dir: str | Path, model_id: str, revision: str) -> Path:
    if model_id != MODEL_ID or revision != MODEL_REVISION:
        raise ValueError(f"Part 38 is fixed to {MODEL_ID}@{MODEL_REVISION}")
    model_dir = model_id.replace("/", "--")
    snapshot = Path(cache_dir) / f"models--{model_dir}" / "snapshots" / revision
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"pinned local model snapshot is missing: {snapshot}")
    return snapshot


def load_model(args: argparse.Namespace):
    torch, AutoModelForCausalLM, AutoTokenizer = _torch_runtime()
    snapshot = resolve_snapshot(args.cache_dir, args.model, args.revision)
    if args.device == "xpu":
        if not getattr(torch, "xpu", None) or not torch.xpu.is_available():
            raise RuntimeError("--device xpu requested but no XPU is available")
    if args.threads and args.device == "cpu":
        torch.set_num_threads(args.threads)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot), torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
    except (ValueError, KeyError, TypeError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            str(snapshot), torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
    model.to(args.device).eval()
    torch.set_grad_enabled(False)
    layers = find_text_layers(model, torch)
    return torch, tokenizer, model, layers, snapshot


def find_text_layers(model: Any, torch: Any) -> Any:
    best = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module):
            if getattr(getattr(module[0], "mlp", None), "down_proj", None) is None:
                continue
            key = ("language" in name.lower(), len(module))
            if best is None or key > best[0]:
                best = (key, module)
    if best is None:
        raise RuntimeError("could not locate decoder blocks with mlp.down_proj")
    return best[1]


def layer_widths(layers: Any, units: list[dict[str, Any]]) -> dict[int, int]:
    widths: dict[int, int] = {}
    for unit in units:
        layer_index = int(unit["layer"])
        if not 0 <= layer_index < len(layers):
            raise ValueError(f"layer {layer_index} is outside the loaded model")
        width = int(layers[layer_index].mlp.down_proj.in_features)
        unit_index = int(unit["unit"])
        if not 0 <= unit_index < width:
            raise ValueError(f"unit {unit_index} is outside layer {layer_index}'s width {width}")
        widths[layer_index] = width
    return widths


def capture_layer_maxima(torch: Any, tokenizer: Any, model: Any, layer: Any,
                         layer_index: int, units: Iterable[int], text: str,
                         device: str, positions: Iterable[int], encoded: dict[str, Any] | None = None) -> dict[int, float]:
    rendered = chat_text(tokenizer, text)
    if encoded is None:
        encoded = _encode_with_offsets(tokenizer, rendered)
    positions = list(map(int, positions))
    ids = _input_ids(encoded)
    if not positions or min(positions) < 0 or max(positions) >= len(ids):
        raise ValueError("activation positions are outside the tokenized user message")
    unit_list = sorted(set(map(int, units)))
    captured: dict[int, float] = {}

    def hook(_module: Any, args: tuple[Any, ...]):
        x = args[0]
        if x.shape[0] != 1 or x.shape[1] != len(ids):
            raise ValueError(f"layer {layer_index} activation shape does not match the tokenized passage")
        unit_index = torch.as_tensor(unit_list, dtype=torch.long, device=x.device)
        pos_index = torch.as_tensor(positions, dtype=torch.long, device=x.device)
        values = x.detach().float()[0].index_select(0, pos_index).index_select(1, unit_index)
        maxima = values.amax(dim=0).cpu().tolist()
        captured.update({unit: float(value) for unit, value in zip(unit_list, maxima)})

    handle = layer.mlp.down_proj.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(**model_inputs(encoded, device), use_cache=False)
    finally:
        handle.remove()
    if set(captured) != set(unit_list):
        raise RuntimeError(f"activation hook at layer {layer_index} did not capture every selected unit")
    return captured


def capture_multi_layer_maxima(torch: Any, tokenizer: Any, model: Any, layers: Any,
                               units_by_layer: dict[int, Iterable[int]], text: str,
                               device: str) -> dict[int, dict[int, float]]:
    rendered = chat_text(tokenizer, text)
    encoded = _encode_with_offsets(tokenizer, rendered)
    positions = content_token_positions(encoded, rendered, text)
    ids = _input_ids(encoded)
    captured: dict[int, dict[int, float]] = {}
    handles = []

    for layer_index, units in units_by_layer.items():
        unit_list = sorted(set(map(int, units)))

        def hook(_module: Any, args: tuple[Any, ...], *, target_layer: int = layer_index,
                 selected: list[int] = unit_list):
            x = args[0]
            if x.shape[0] != 1 or x.shape[1] != len(ids):
                raise ValueError(f"layer {target_layer} activation shape does not match the tokenized passage")
            unit_index = torch.as_tensor(selected, dtype=torch.long, device=x.device)
            pos_index = torch.as_tensor(positions, dtype=torch.long, device=x.device)
            values = x.detach().float()[0].index_select(0, pos_index).index_select(1, unit_index)
            maxima = values.amax(dim=0).cpu().tolist()
            captured[target_layer] = {unit: float(value) for unit, value in zip(selected, maxima)}

        handles.append(layers[layer_index].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            model(**model_inputs(encoded, device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(units_by_layer):
        raise RuntimeError("natural-max hooks did not capture all requested layers")
    return captured


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def prepare_jsonl(path: str | Path, header: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(path)
    expected = {"record_type": "header", **header}
    if rows:
        if rows[0] != expected:
            raise ValueError(f"existing output header does not match this run: {path}")
    else:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(expected, ensure_ascii=False, sort_keys=True) + "\n")
        rows = [expected]
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def write_json_atomic(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp, path)


def completed_keys(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["row_key"]) for row in rows if row.get("record_type") == "result"
            and row.get("row_key") is not None}


def selected_runtime_units(data: dict[str, Any], selection: str | None,
                           default_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    return select_units(data["units"], selection, default_ids)


def load_part35_neutral_passages(path: str | Path = PART35_DATA_PATH) -> list[dict[str, str]]:
    source = read_json(path)
    passages = []
    for frame in source.get("frame_bank", []):
        language = frame["language"]
        if language not in PART35_NEUTRAL_SLOTS:
            raise ValueError(f"no Part 35 neutral slot is defined for {language}")
        text = frame["template"].format(slot=PART35_NEUTRAL_SLOTS[language])
        passages.append({"id": frame["id"], "language": language, "text": text})
    if len(passages) != 36:
        raise ValueError(f"Part 35 neutral frame bank must have 36 passages, got {len(passages)}")
    return passages


def _model_header(args: argparse.Namespace, command: str) -> dict[str, Any]:
    return {"schema_version": 1, "command": command, "model": args.model,
            "revision": args.revision, "device": args.device, "dtype": args.dtype,
            "seed": args.seed}


def homograph_summary(rows: list[dict[str, Any]], units: list[dict[str, Any]],
                      controls: dict[str, list[int]], seed: int) -> dict[str, Any]:
    results = []
    data_rows = [row for row in rows if row.get("record_type") == "result"]
    for unit in units:
        unit_id = unit["id"]
        unit_rows = [row for row in data_rows if row["unit_id"] == unit_id]
        by_sense = {sense: [row for row in unit_rows if row["sense"] == sense]
                    for sense in ("screened", "other")}
        screened = [float(row["target_activation"]) for row in by_sense["screened"]]
        other = [float(row["target_activation"]) for row in by_sense["other"]]
        med_screened = float(np.median(screened)) if screened else None
        med_other = float(np.median(other)) if other else None
        ratio = (med_other / med_screened if med_screened not in (None, 0) and med_other is not None
                 else None)
        ci = (bootstrap_ratio_ci(screened, other,
                                 stable_seed(seed, "homograph-bootstrap", unit_id))
              if screened and other else None)
        random_ratios = []
        for random_unit in controls[unit_id]:
            random_screened = [float(row["random_activations"][str(random_unit)])
                               for row in by_sense["screened"]]
            random_other = [float(row["random_activations"][str(random_unit)])
                            for row in by_sense["other"]]
            den = float(np.median(random_screened)) if random_screened else None
            num = float(np.median(random_other)) if random_other else None
            if den not in (None, 0) and num is not None:
                random_ratios.append({"unit": int(random_unit), "ratio": num / den})
        random_values = [row["ratio"] for row in random_ratios]
        results.append({
            "unit_id": unit_id, "layer": int(unit["layer"]), "unit": int(unit["unit"]),
            "target": unit["target"], "screened_sense": unit["screened_sense"],
            "other_sense": unit["other_sense"],
            "counts": {sense: len(by_sense[sense]) for sense in by_sense},
            "median_screened": med_screened, "median_other": med_other,
            "other_over_screened_ratio": ratio,
            "ratio_bootstrap_95": list(ci) if ci is not None else None,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "random_unit_ratios": random_ratios,
            "random_ratio_median_p5_p95": (
                {"median": float(np.median(random_values)),
                 "p5": float(np.quantile(random_values, .05)),
                 "p95": float(np.quantile(random_values, .95))}
                if random_values else None),
        })
    return {"schema_version": 1, "summary_type": "homograph", "seed": int(seed),
            "units": results}


def run_homograph(args: argparse.Namespace) -> dict[str, Any]:
    data = read_json(args.data)
    means = validate_data(data)
    units = selected_runtime_units(data, args.units)
    torch, tokenizer, model, layers, _snapshot = load_model(args)
    widths = layer_widths(layers, units)
    controls = control_sets_by_target(units, widths, args.seed)
    header = {**_model_header(args, "homograph"), "data_sha256": sha256_file(args.data),
              "target_units": [{"id": unit["id"], "layer": unit["layer"], "unit": unit["unit"]}
                              for unit in units],
              "random_units_by_target": controls,
              "activation_measure": "maximum down_proj input coordinate over target word tokens"}
    out_path = args.out / "homograph.jsonl"
    rows = prepare_jsonl(out_path, header)
    done = completed_keys(rows)
    for unit in units:
        target_unit = int(unit["unit"])
        selected_units = [target_unit, *controls[unit["id"]]]
        for sense in ("screened", "other"):
            sentences = unit["sentences"][sense]
            if args.limit is not None:
                sentences = sentences[:args.limit]
            for sentence_index, sentence in enumerate(sentences, 1):
                row_key = f"{unit['id']}:{sense}:{sentence_index:02d}"
                if row_key in done:
                    continue
                rendered = chat_text(tokenizer, sentence)
                encoded = _encode_with_offsets(tokenizer, rendered)
                token_start, token_end = target_span_in_encoding(
                    encoded, rendered, sentence, unit["target"])
                span = list(range(token_start, token_end))
                values = capture_layer_maxima(torch, tokenizer, model, layers[int(unit["layer"])],
                                               int(unit["layer"]), selected_units, sentence,
                                               args.device, span, encoded)
                result = {"record_type": "result", "row_key": row_key,
                          "unit_id": unit["id"], "layer": int(unit["layer"]),
                          "unit": target_unit, "target": unit["target"], "sense": sense,
                          "sense_label": unit["screened_sense"] if sense == "screened" else unit["other_sense"],
                          "sentence_id": f"{sense}-{sentence_index:02d}", "text": sentence,
                          "target_token_span": [token_start, token_end],
                          "target_activation": values[target_unit],
                          "random_activations": {str(index): values[index]
                                                 for index in controls[unit["id"]]}}
                append_jsonl(out_path, result)
                done.add(row_key)
    rows = read_jsonl(out_path)
    summary = homograph_summary(rows, units, controls, args.seed)
    summary["data_mean_word_counts"] = {unit_id: means[unit_id] for unit_id in means
                                         if unit_id in {unit["id"] for unit in units}}
    summary["model"] = args.model
    summary["revision"] = args.revision
    write_json_atomic(summary, args.out / "homograph-summary.json")
    print(f"homograph rows: {sum(row.get('record_type') == 'result' for row in rows)}")
    return summary


def neighbour_summary(rows: list[dict[str, Any]], controls: list[int], seed: int) -> dict[str, Any]:
    data_rows = [row for row in rows if row.get("record_type") == "result"]
    wizard_rows = [row for row in data_rows if row["group"] == "wizard"]
    wizard_values = [float(row["target_activation"]) for row in wizard_rows]
    wizard_median = float(np.median(wizard_values)) if wizard_values else None
    summaries = []
    for group in NEIGHBOUR_GROUPS:
        group_rows = [row for row in data_rows if row["group"] == group]
        values = [float(row["target_activation"]) for row in group_rows]
        median = float(np.median(values)) if values else None
        ratio = (median / wizard_median if median is not None and wizard_median not in (None, 0)
                 else None)
        if group == "wizard" and wizard_median is not None:
            ratio_ci = [1.0, 1.0]
        else:
            ci = (bootstrap_median_ratio_ci(wizard_values, values,
                                            stable_seed(seed, "neighbour-bootstrap", group))
                  if wizard_values and values else None)
            ratio_ci = list(ci) if ci is not None else None
        random_ratios = []
        for random_unit in controls:
            random_wizard = [float(row["random_activations"][str(random_unit)])
                             for row in wizard_rows]
            random_group = [float(row["random_activations"][str(random_unit)])
                            for row in group_rows]
            den = float(np.median(random_wizard)) if random_wizard else None
            num = float(np.median(random_group)) if random_group else None
            if den not in (None, 0) and num is not None:
                random_ratios.append({"unit": int(random_unit), "ratio_to_wizard": num / den})
        null_values = [row["ratio_to_wizard"] for row in random_ratios]
        summaries.append({
            "word": group, "count": len(values), "median_activation": median,
            "share_of_wizard_median": ratio, "bootstrap_95": ratio_ci,
            "random_unit_ratios": random_ratios,
            "random_ratio_median_p5_p95": (
                {"median": float(np.median(null_values)),
                 "p5": float(np.quantile(null_values, .05)),
                 "p95": float(np.quantile(null_values, .95))}
                if null_values else None),
        })
    return {"schema_version": 1, "summary_type": "neighbours", "seed": int(seed),
            "unit_id": "wizard", "layer": 19, "unit": 4826,
            "wizard_median": wizard_median, "groups": summaries}


def run_neighbours(args: argparse.Namespace) -> dict[str, Any]:
    data = read_json(args.data)
    validate_neighbour_data(data)
    if args.units and args.units.strip() not in ("wizard",):
        raise ValueError("the neighbours assay is fixed to the wizard unit")
    wizard = {"id": "wizard", "layer": 19, "unit": 4826}
    torch, tokenizer, model, layers, _snapshot = load_model(args)
    controls = control_sets_by_target([wizard], {19: int(layers[19].mlp.down_proj.in_features)}, args.seed)["wizard"]
    header = {**_model_header(args, "neighbours"), "data_sha256": sha256_file(args.data),
              "target_unit": {"id": "wizard", "layer": 19, "unit": 4826},
              "random_units": controls,
              "activation_measure": "maximum down_proj input coordinate over target word tokens"}
    out_path = args.out / "neighbours.jsonl"
    rows = prepare_jsonl(out_path, header)
    done = completed_keys(rows)
    for group in NEIGHBOUR_GROUPS:
        sentences = data["groups"][group]
        if args.limit is not None:
            sentences = sentences[:args.limit]
        for sentence_index, sentence in enumerate(sentences, 1):
            row_key = f"{group}:{sentence_index:02d}"
            if row_key in done:
                continue
            rendered = chat_text(tokenizer, sentence)
            encoded = _encode_with_offsets(tokenizer, rendered)
            token_start, token_end = target_span_in_encoding(encoded, rendered, sentence, group)
            values = capture_layer_maxima(torch, tokenizer, model, layers[19], 19,
                                           [4826, *controls], sentence, args.device,
                                           range(token_start, token_end), encoded)
            result = {"record_type": "result", "row_key": row_key, "unit_id": "wizard",
                      "layer": 19, "unit": 4826, "group": group, "target": group,
                      "sentence_id": f"{group}-{sentence_index:02d}", "text": sentence,
                      "target_token_span": [token_start, token_end],
                      "target_activation": values[4826],
                      "random_activations": {str(index): values[index] for index in controls}}
            append_jsonl(out_path, result)
            done.add(row_key)
    rows = read_jsonl(out_path)
    summary = neighbour_summary(rows, controls, args.seed)
    summary["model"] = args.model
    summary["revision"] = args.revision
    write_json_atomic(summary, args.out / "neighbours-summary.json")
    print(f"neighbour rows: {sum(row.get('record_type') == 'result' for row in rows)}")
    return summary


def natural_max_summary(rows: list[dict[str, Any]], target_units: list[dict[str, Any]],
                        controls: dict[str, list[int]], expected_passages: list[dict[str, str]],
                        metadata: dict[str, Any]) -> dict[str, Any]:
    data_rows = [row for row in rows if row.get("record_type") == "result"]
    by_target: dict[str, Any] = {}
    all_complete = True
    for unit in target_units:
        unit_id = unit["id"]
        refs = {"target": [int(unit["unit"])],
                "random": [int(index) for index in controls[unit_id]]}
        values: dict[str, Any] = {"target": None, "random_units": {}}
        for role, unit_indices in refs.items():
            if role == "target":
                indexed = [
                    row for row in data_rows
                    if row["target_id"] == unit_id and row["unit_role"] == "target"
                    and int(row["unit_index"]) == int(unit["unit"])
                ]
                passage_ids = {row["passage_id"] for row in indexed}
                count = len(passage_ids)
                if count < len(expected_passages):
                    all_complete = False
                values["target"] = {
                    "unit": int(unit["unit"]),
                    "natural_max": max((float(row["activation_max"]) for row in indexed), default=None),
                    "passage_count": count,
                }
            else:
                for unit_index in unit_indices:
                    indexed = [
                        row for row in data_rows
                        if row["target_id"] == unit_id and row["unit_role"] == "random"
                        and int(row["unit_index"]) == int(unit_index)
                    ]
                    count = len({row["passage_id"] for row in indexed})
                    if count < len(expected_passages):
                        all_complete = False
                    values["random_units"][str(unit_index)] = {
                        "unit": int(unit_index),
                        "natural_max": max((float(row["activation_max"]) for row in indexed), default=None),
                        "passage_count": count,
                    }
        by_target[unit_id] = values
    return {"record_type": "summary", "schema_version": 1,
            "summary_type": "natural_max", "metadata": metadata,
            "expected_passages": len(expected_passages),
            "passage_ids": [row["id"] for row in expected_passages],
            "complete": all_complete, "values_by_target": by_target}


def run_natural_max(args: argparse.Namespace) -> dict[str, Any]:
    data = read_json(args.data)
    validate_data(data)
    units = selected_runtime_units(data, args.units, PUSH_UNIT_IDS)
    passages = load_part35_neutral_passages(args.part35_data)
    torch, tokenizer, model, layers, _snapshot = load_model(args)
    widths = layer_widths(layers, units)
    controls = control_sets_by_target(units, widths, args.seed)
    units_by_layer: dict[int, set[int]] = {}
    for unit in units:
        layer_index = int(unit["layer"])
        units_by_layer.setdefault(layer_index, set()).add(int(unit["unit"]))
        units_by_layer[layer_index].update(controls[unit["id"]])
    header = {**_model_header(args, "natural-max"), "data_sha256": sha256_file(args.data),
              "part35_data_sha256": sha256_file(args.part35_data),
              "target_units": [{"id": unit["id"], "layer": unit["layer"], "unit": unit["unit"]}
                              for unit in units],
              "random_units_by_target": controls,
              "passage_ids": [row["id"] for row in passages],
              "activation_measure": "maximum down_proj input coordinate over Part 35 content tokens"}
    out_path = args.out / "natural-max.jsonl"
    rows = prepare_jsonl(out_path, header)
    done = completed_keys(rows)
    selected_passages = passages[:args.limit] if args.limit is not None else passages
    logical_refs = []
    for unit in units:
        for role, indices in (("target", [int(unit["unit"])]),
                              ("random", controls[unit["id"]])):
            for index in indices:
                logical_refs.append((unit["id"], role, int(index), int(unit["layer"])))
    for passage in selected_passages:
        missing = [ref for ref in logical_refs
                   if f"{ref[0]}:{ref[1]}:{ref[2]}:{passage['id']}" not in done]
        if not missing:
            continue
        captured = capture_multi_layer_maxima(torch, tokenizer, model, layers,
                                              units_by_layer, passage["text"], args.device)
        for target_id, role, unit_index, layer_index in missing:
            row_key = f"{target_id}:{role}:{unit_index}:{passage['id']}"
            result = {"record_type": "result", "row_key": row_key,
                      "target_id": target_id, "unit_role": role, "layer": layer_index,
                      "unit_index": unit_index, "passage_id": passage["id"],
                      "language": passage["language"], "text": passage["text"],
                      "activation_max": captured[layer_index][unit_index]}
            append_jsonl(out_path, result)
            done.add(row_key)
    rows = read_jsonl(out_path)
    metadata = {"model": args.model, "revision": args.revision, "seed": int(args.seed),
                "data_sha256": sha256_file(args.data),
                "part35_data_sha256": sha256_file(args.part35_data),
                "target_units": [unit["id"] for unit in units],
                "random_units_by_target": controls}
    summary = natural_max_summary(rows, units, controls, passages, metadata)
    write_json_atomic(summary, args.out / "natural-max-summary.json")
    print(f"natural-max passages: {len(selected_passages)} / {len(passages)}")
    return summary


def load_natural_max(path: str | Path, args: argparse.Namespace,
                     units: list[dict[str, Any]], controls: dict[str, list[int]],
                     allow_incomplete: bool) -> dict[str, Any]:
    summary = read_json(path)
    metadata = summary.get("metadata", {})
    if metadata.get("model") != args.model or metadata.get("revision") != args.revision:
        raise ValueError("natural-max summary model or revision does not match this push")
    if int(metadata.get("seed", -1)) != int(args.seed):
        raise ValueError("natural-max summary seed does not match this push")
    for unit in units:
        unit_id = unit["id"]
        if unit_id not in summary.get("values_by_target", {}):
            raise ValueError(f"natural-max summary is missing {unit_id}")
        if metadata.get("random_units_by_target", {}).get(unit_id) != controls[unit_id]:
            raise ValueError(f"natural-max random units do not match for {unit_id}")
        values = summary["values_by_target"][unit_id]
        if not values.get("target") or values["target"].get("natural_max") is None:
            raise ValueError(f"natural maximum is missing for {unit_id}")
        if int(values["target"].get("passage_count", 0)) < int(summary.get("expected_passages", 0)):
            if not allow_incomplete:
                raise ValueError(f"natural maximum is incomplete for {unit_id}; rerun natural-max without --limit")
        for random_unit in controls[unit_id]:
            control = values.get("random_units", {}).get(str(random_unit))
            if not control or control.get("natural_max") is None:
                raise ValueError(f"natural maximum is missing for {unit_id} random unit {random_unit}")
            if int(control.get("passage_count", 0)) < int(summary.get("expected_passages", 0)):
                if not allow_incomplete:
                    raise ValueError(f"random natural maximum is incomplete for {unit_id}:{random_unit}")
    return summary


def _jsonl_result_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("record_type") == "result"]


def _rate(rows: list[dict[str, Any]], topic_terms: list[str]) -> dict[str, Any]:
    hits = sum(topic_hit(str(row.get("answer", "")), topic_terms) for row in rows)
    n = len(rows)
    return {"hits": hits, "n": n, "rate": hits / n if n else None,
            "wilson95": list(wilson_interval(hits, n)) if n else None}


def score_push_rows(rows: Iterable[dict[str, Any]], units: list[dict[str, Any]],
                    k_values: Iterable[int]) -> dict[str, Any]:
    results = _jsonl_result_rows(rows)
    summaries = []
    for unit in units:
        unit_id = unit["id"]
        terms = TOPIC_TERMS[unit_id]
        no_clamp_rows = [row for row in results
                         if row["target_id"] == unit_id and row["arm"] == "no_clamp"]
        no_clamp = _rate(no_clamp_rows, terms)
        for k in k_values:
            target_rows = [row for row in results
                           if row["target_id"] == unit_id and row["arm"] == "target"
                           and int(row["k"]) == int(k)]
            target_rate = _rate(target_rows, terms)
            random_rows = [row for row in results
                           if row["target_id"] == unit_id and row["arm"] == "random"
                           and int(row["k"]) == int(k)]
            by_random: dict[int, list[dict[str, Any]]] = {}
            for row in random_rows:
                by_random.setdefault(int(row["clamped_unit"]), []).append(row)
            random_rates = [{"unit": random_unit, **_rate(random_rows_for_unit, terms)}
                            for random_unit, random_rows_for_unit in sorted(by_random.items())]
            rates = [row["rate"] for row in random_rates if row["rate"] is not None]
            summaries.append({
                "unit_id": unit_id, "layer": int(unit["layer"]), "unit": int(unit["unit"]),
                "k": int(k), "topic_terms": terms, "target": target_rate,
                "random_unit_rates": random_rates,
                "random_rate_median": float(np.median(rates)) if rates else None,
                "random_rate_p95": float(np.quantile(rates, .95)) if rates else None,
                "no_clamp": no_clamp,
            })
    return {"schema_version": 1, "summary_type": "push_topic_rates", "rows": summaries}


def _generate_answer(torch: Any, tokenizer: Any, model: Any, layer: Any,
                     prompt: str, device: str, seed: int,
                     unit: int | None = None, level: float | None = None) -> str:
    rendered = chat_text(tokenizer, prompt)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    model_encoded = {key: value.to(device) if hasattr(value, "to") else value
                     for key, value in encoded.items()}
    prompt_length = int(model_encoded["input_ids"].shape[-1])
    torch.manual_seed(int(seed))
    if device == "xpu":
        torch.xpu.manual_seed_all(int(seed))
    hook = None
    if unit is not None:
        if level is None:
            raise ValueError("a clamp level is required when a unit is selected")
        hook = UnitClampHook(layer.mlp.down_proj, unit, level, prompt_length)
    try:
        with torch.inference_mode():
            output = model.generate(
                **model_encoded, do_sample=True, temperature=PUSH_TEMPERATURE,
                top_p=PUSH_TOP_P, max_new_tokens=PUSH_MAX_NEW_TOKENS,
                use_cache=True, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
    finally:
        if hook is not None:
            hook.remove()
    answer_ids = output[0, prompt_length:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def run_push(args: argparse.Namespace) -> dict[str, Any]:
    data = read_json(args.data)
    validate_data(data)
    units = selected_runtime_units(data, args.units, PUSH_UNIT_IDS)
    unsupported = [unit["id"] for unit in units if unit["id"] not in TOPIC_TERMS]
    if unsupported:
        raise ValueError(f"the design has no fixed topic list for: {', '.join(unsupported)}")
    k_values = list(dict.fromkeys(args.k))
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("--k must provide one or more positive multipliers")
    torch, tokenizer, model, layers, _snapshot = load_model(args)
    widths = layer_widths(layers, units)
    controls = control_sets_by_target(units, widths, args.seed)
    natural_path = args.natural_max or (args.out / "natural-max-summary.json")
    natural = load_natural_max(natural_path, args, units, controls,
                               allow_incomplete=(args.limit is not None))
    prompts = data["neutral_prompts"]
    prompt_samples = [(prompt_index, sample_index, prompt)
                      for prompt_index, prompt in enumerate(prompts, 1)
                      for sample_index in range(1, PUSH_SAMPLES + 1)]
    if args.limit is not None:
        prompt_samples = prompt_samples[:args.limit]
    header = {
        **_model_header(args, "push"), "data_sha256": sha256_file(args.data),
        "natural_max_sha256": sha256_file(natural_path),
        "target_units": [{"id": unit["id"], "layer": unit["layer"], "unit": unit["unit"]}
                        for unit in units],
        "random_units_by_target": controls, "k_values": k_values,
        "prompt_sha256": hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(),
        "samples_per_prompt": PUSH_SAMPLES,
        "prompt_sample_count": len(prompts) * PUSH_SAMPLES,
        "generation": {"temperature": PUSH_TEMPERATURE, "top_p": PUSH_TOP_P,
                       "max_new_tokens": PUSH_MAX_NEW_TOKENS},
    }
    out_path = args.out / "push.jsonl"
    rows = prepare_jsonl(out_path, header)
    done = completed_keys(rows)
    for unit in units:
        unit_id = unit["id"]
        layer = layers[int(unit["layer"])]
        unit_natural = natural["values_by_target"][unit_id]
        for prompt_index, sample_index, prompt in prompt_samples:
            prompt_id = f"p{prompt_index:02d}"
            gen_seed = stable_seed(args.seed, "push-sample", prompt_id, sample_index)
            baseline_key = f"{unit_id}:no_clamp:none:{prompt_id}:{sample_index}"
            if baseline_key not in done:
                answer = _generate_answer(torch, tokenizer, model, layer, prompt, args.device,
                                          gen_seed)
                append_jsonl(out_path, {
                    "record_type": "result", "row_key": baseline_key,
                    "target_id": unit_id, "prompt_id": prompt_id,
                    "sample": sample_index, "seed": gen_seed, "arm": "no_clamp",
                    "unit_role": None, "clamped_unit": None, "k": None,
                    "natural_max": None, "clamp_level": None,
                    "prompt": prompt, "answer": answer,
                })
                done.add(baseline_key)
            for k in k_values:
                for arm, unit_role, unit_index in (
                        ("target", "target", int(unit["unit"])),
                        *(('random', 'random', int(index)) for index in controls[unit_id])):
                    key = f"{unit_id}:{arm}:{unit_index}:k{k}:{prompt_id}:{sample_index}"
                    if key in done:
                        continue
                    record = (unit_natural["target"] if unit_role == "target"
                              else unit_natural["random_units"][str(unit_index)])
                    natural_max = float(record["natural_max"])
                    level = float(k * natural_max)
                    answer = _generate_answer(torch, tokenizer, model, layer, prompt, args.device,
                                              gen_seed, unit_index, level)
                    append_jsonl(out_path, {
                        "record_type": "result", "row_key": key,
                        "target_id": unit_id, "prompt_id": prompt_id,
                        "sample": sample_index, "seed": gen_seed,
                        "arm": arm, "unit_role": unit_role,
                        "clamped_unit": unit_index, "k": int(k),
                        "natural_max": natural_max, "clamp_level": level,
                        "prompt": prompt, "answer": answer,
                    })
                    done.add(key)
    rows = read_jsonl(out_path)
    scores = score_push_rows(rows, units, k_values)
    scores.update({"model": args.model, "revision": args.revision,
                   "seed": int(args.seed), "k_values": k_values,
                   "natural_max_path": str(natural_path)})
    write_json_atomic(scores, args.out / "push-scores.json")
    print(f"push rows: {sum(row.get('record_type') == 'result' for row in rows)}")
    return scores


def write_jsonl_atomic(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(temp, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp, path)


def run_blind_pack(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input or (args.out / "push.jsonl")
    rows = read_jsonl(input_path)
    if not rows or rows[0].get("record_type") != "header":
        raise ValueError(f"push JSONL has no header row: {input_path}")
    answers = _jsonl_result_rows(rows)
    if args.units:
        selected = {part.strip() for part in args.units.split(",") if part.strip()}
        answers = [row for row in answers if row.get("target_id") in selected]
    shuffled = list(answers)
    random.Random(args.seed).shuffle(shuffled)
    if args.limit is not None:
        shuffled = shuffled[:args.limit]
    pack_header = {"record_type": "header", "schema_version": 1,
                   "blind_pack": True, "seed": int(args.seed),
                   "answer_count": len(shuffled)}
    pack_rows = [pack_header]
    key_rows = []
    for index, row in enumerate(shuffled, 1):
        blind_id = f"blind-{index:05d}"
        pack_rows.append({"record_type": "answer", "blind_id": blind_id,
                          "text": str(row.get("answer", ""))})
        key_rows.append({"blind_id": blind_id, "source_row_key": row.get("row_key"),
                         "unit_id": row.get("target_id"), "arm": row.get("arm"),
                         "unit_role": row.get("unit_role"),
                         "clamped_unit": row.get("clamped_unit"), "k": row.get("k"),
                         "prompt_id": row.get("prompt_id"), "sample": row.get("sample")})
    pack_path = args.out / "blind-pack.jsonl"
    key_path = args.key_out or (args.out / "blind-pack-key.json")
    write_jsonl_atomic(pack_rows, pack_path)
    write_json_atomic({"schema_version": 1, "seed": int(args.seed),
                       "source": str(input_path), "answers": key_rows}, key_path)
    print(f"blind pack answers: {len(shuffled)}")
    return {"pack": str(pack_path), "key": str(key_path), "answer_count": len(shuffled)}


def _add_common_model_arguments(parser: argparse.ArgumentParser,
                                data_default: Path = DATA_PATH) -> None:
    parser.add_argument("--model", default=MODEL_ID, choices=(MODEL_ID,))
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3801)
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke-run limit per unit, passage list, or prompt/sample list")
    parser.add_argument("--units", default=None,
                        help="comma-separated unit ids; defaults depend on the subcommand")
    parser.add_argument("--data", type=Path, default=data_default)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory; JSONL outputs resume by row key")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("homograph", "neighbours", "natural-max", "push"):
        default_data = NEIGHBOUR_DATA_PATH if command == "neighbours" else DATA_PATH
        p = sub.add_parser(command)
        _add_common_model_arguments(p, default_data)
        if command == "natural-max":
            p.add_argument("--part35-data", type=Path, default=PART35_DATA_PATH)
        if command == "push":
            p.add_argument("--k", type=int, nargs="+", default=[4, 8])
            p.add_argument("--natural-max", type=Path, default=None,
                           help="natural-max-summary.json; defaults under --out")
    p = sub.add_parser("blind-pack")
    p.add_argument("--input", type=Path, default=None,
                   help="push JSONL; defaults to OUT/push.jsonl")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--key-out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=3801)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--units", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.command == "blind-pack":
        run_blind_pack(args)
        return 0
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.revision != MODEL_REVISION:
        parser.error(f"Part 38 requires the pinned revision {MODEL_REVISION}")
    try:
        if args.command == "homograph":
            run_homograph(args)
        elif args.command == "neighbours":
            run_neighbours(args)
        elif args.command == "natural-max":
            run_natural_max(args)
        elif args.command == "push":
            run_push(args)
        else:
            parser.error(f"unknown command: {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"part38: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
