#!/usr/bin/env python3
"""Part 39: measure Qwen3.5-4B layer 6 unit 753 and its random controls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

import circuit
import neuron_report as nr
import part26
import part25
import part27


CHAT_MODEL = part27.CHAT_MODEL
CHAT_REVISION = part27.CHAT_REVISION
BASE_MODEL = part27.BASE_MODEL
BASE_REVISION = part27.BASE_REVISION
TARGET = {"layer": 6, "unit": 753}
READ_CONTROL_COUNT = 20
GENERATION_CONTROL_COUNT = 10
BOOTSTRAP_RESAMPLES = 2000
SCREEN_SHUFFLES = 200
SCREEN_SEED = 3901
SCREEN_LAYER_COUNT = 32
SCREEN_WIDTH = 9216
SCREEN_CATEGORY_IDS = ("help_request", "second_person_statement", "ai_self_description",
                       "human_self_description", "third_person_ai", "neutral_prose")
LOGIT_IDENTITY_RE = re.compile(r"\b(ai|assistant|model|chatbot|claude|gpt)\b|人工智能|助手", re.I)
BATTERY_PATH = Path(__file__).resolve().parent / "data" / "part39-battery.json"
PERSONA_PATH = Path(__file__).resolve().parent / "data" / "part39-persona.json"
AI_CLAIM_RE = re.compile(
    r"\b(AI|artificial intelligence|language model|assistant|chatbot)\b", re.I
)
MAKER_NAMES = sorted(
    {name for target in part25.TARGET_WORDS.values() for name in target["maker"]},
    key=lambda value: (-len(value), value.casefold()),
)
MAKER_RE = re.compile(r"\b(?:" + "|".join(re.escape(name) for name in MAKER_NAMES) + r")\b", re.I)
WORD_RE = re.compile(r"\b[\w]+\b", re.UNICODE)


def _read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_atomic(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                     dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_read_arrays(path: Path, sentence_max: np.ndarray, assistant_marker: np.ndarray,
                      metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                sentence_max=sentence_max.astype(np.float16, copy=False),
                assistant_marker_value=assistant_marker.astype(np.float16, copy=False),
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_read_arrays(path: Path, expected_metadata: dict,
                      expected_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, dict] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            sentence_max = np.asarray(archive["sentence_max"], dtype=np.float16)
            assistant_marker = np.asarray(archive["assistant_marker_value"], dtype=np.float16)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path.name} is malformed; use a fresh --out ({exc})") from exc
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise SystemExit(f"{path.name} provenance mismatch ({key}); use a fresh --out")
    if sentence_max.shape != expected_shape or assistant_marker.shape != expected_shape:
        raise SystemExit(f"{path.name} has shape {sentence_max.shape}; expected {expected_shape}")
    return sentence_max, assistant_marker, metadata


def _atomic_write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                     dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        return []
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed JSONL at {source}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row at {source}:{line_no} is not an object")
                rows.append(row)
    return rows


class JsonlSink:
    """Append keyed rows, compacting any old duplicate keys when resuming."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        old_rows = read_jsonl(self.path)
        keyed: dict[str, dict] = {}
        for row in old_rows:
            key = row.get("row_key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"existing JSONL row in {self.path} has no row_key")
            keyed[key] = row
        self.rows_by_key = keyed
        if len(old_rows) != len(keyed):
            _atomic_write_jsonl(keyed.values(), self.path)

    @property
    def seen(self) -> set[str]:
        return set(self.rows_by_key)

    def append(self, row: dict) -> bool:
        key = row.get("row_key")
        if not isinstance(key, str) or not key:
            raise ValueError("JSONL row needs a nonempty row_key")
        if key in self.rows_by_key:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
        self.rows_by_key[key] = row
        return True

    def rows(self) -> list[dict]:
        return list(self.rows_by_key.values())


def validate_battery(data: dict) -> list[dict]:
    categories = data.get("categories")
    if not isinstance(categories, list) or len(categories) != 6:
        raise ValueError("battery must contain exactly six categories")
    ids = set()
    for category in categories:
        category_id = category.get("id")
        sentences = category.get("sentences")
        if not isinstance(category_id, str) or category_id in ids:
            raise ValueError("battery category ids must be unique strings")
        ids.add(category_id)
        if not isinstance(sentences, list) or len(sentences) != 20:
            raise ValueError(f"battery category {category_id!r} must contain 20 sentences")
        sentence_ids = set()
        for record in sentences:
            if not isinstance(record.get("id"), str) or not isinstance(record.get("text"), str):
                raise ValueError(f"battery category {category_id!r} has a malformed sentence")
            if record["id"] in sentence_ids:
                raise ValueError(f"battery category {category_id!r} repeats a sentence id")
            sentence_ids.add(record["id"])
    return categories


def _flat_ids(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _flat_offsets(values: Any) -> list[tuple[int, int]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list) and values[0] and isinstance(values[0][0], (list, tuple)):
        values = values[0]
    return [(int(pair[0]), int(pair[1])) for pair in values]


def _encode_with_offsets(tok, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    try:
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = _flat_ids(encoded["input_ids"])
        offsets = _flat_offsets(encoded["offset_mapping"])
        if len(ids) == len(offsets):
            return ids, offsets
    except (TypeError, ValueError, NotImplementedError, KeyError, AttributeError):
        pass
    encoded = tok(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        ids = _flat_ids(encoded["input_ids"])
    else:
        ids = _flat_ids(encoded)
    return ids, None


def _token_span(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [index for index, (lo, hi) in enumerate(offsets) if hi > lo and lo < end and hi > start]


def _fallback_span(tok, rendered: str, start: int, end: int) -> list[int]:
    before = tok(rendered[:start], add_special_tokens=False)
    through = tok(rendered[:end], add_special_tokens=False)
    before_ids = _flat_ids(before["input_ids"] if isinstance(before, dict) else before)
    through_ids = _flat_ids(through["input_ids"] if isinstance(through, dict) else through)
    return list(range(len(before_ids), len(through_ids)))


def _render_template(tok, messages: list[dict]) -> str:
    if not getattr(tok, "chat_template", None):
        raise ValueError("tokenizer has no chat_template; Part 39 requires the Qwen chat template")
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_plain_sample(tok, sentence: str) -> dict:
    ids, offsets = _encode_with_offsets(tok, sentence)
    if not ids:
        raise ValueError("tokenizer returned no ids for a battery sentence")
    positions = _token_span(offsets, 0, len(sentence)) if offsets is not None else list(range(len(ids)))
    if not positions:
        raise ValueError("could not map plain sentence tokens")
    return {"input_ids": ids, "content_positions": positions, "assistant_marker_index": None}


def prepare_template_sample(tok, sentence: str) -> dict:
    """Return template tokens and the content / assistant-role token positions."""
    rendered = _render_template(tok, [{"role": "user", "content": sentence}])
    content_start = rendered.rfind(sentence)
    marker_start = rendered.rfind("assistant")
    if content_start < 0 or marker_start < 0 or marker_start < content_start + len(sentence):
        raise ValueError("could not locate user content or final assistant role marker in chat template")
    ids, offsets = _encode_with_offsets(tok, rendered)
    if not ids:
        raise ValueError("tokenizer returned no ids for the templated sentence")
    if offsets is None:
        content_positions = _fallback_span(tok, rendered, content_start, content_start + len(sentence))
        marker_positions = _fallback_span(tok, rendered, marker_start, marker_start + len("assistant"))
    else:
        content_positions = _token_span(offsets, content_start, content_start + len(sentence))
        marker_positions = _token_span(offsets, marker_start, marker_start + len("assistant"))
    if not content_positions or not marker_positions:
        raise ValueError("could not map sentence or assistant marker to template tokens")
    # The role name may split into multiple tokens; the first one is the role marker.
    return {"input_ids": ids, "content_positions": content_positions,
            "assistant_marker_index": marker_positions[0], "rendered": rendered}


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _padded_batch(tok, samples: list[dict], device: torch.device):
    max_length = max(len(sample["input_ids"]) for sample in samples)
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    padding_side = getattr(tok, "padding_side", "right")
    ids = torch.full((len(samples), max_length), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(samples), max_length), dtype=torch.long)
    shifts = []
    for index, sample in enumerate(samples):
        row = torch.tensor(sample["input_ids"], dtype=torch.long)
        shift = max_length - len(row) if padding_side == "left" else 0
        ids[index, shift:shift + len(row)] = row
        mask[index, shift:shift + len(row)] = 1
        shifts.append(shift)
    return ids.to(device), mask.to(device), shifts


def _capture_all_layers(model, layers, tok, samples: list[dict]):
    """Reduce each layer to per-row maxima and marker vectors inside its hook."""
    ids, mask, shifts = _padded_batch(tok, samples, _model_device(model))
    maxima: dict[int, np.ndarray] = {}
    markers: dict[int, np.ndarray] = {}
    handles = []

    for layer_index, layer in enumerate(layers):
        def capture(_module, args, *, target_layer=layer_index):
            values = args[0]
            row_maxima = []
            row_markers = []
            for row_index, sample in enumerate(samples):
                shift = shifts[row_index]
                positions = torch.as_tensor(
                    [int(position) + shift for position in sample["content_positions"]],
                    dtype=torch.long, device=values.device)
                content = values[row_index].index_select(0, positions).float()
                row_maxima.append(content.amax(dim=0))
                marker_index = sample.get("assistant_marker_index")
                if marker_index is None:
                    row_markers.append(torch.full((values.shape[-1],), float("nan"),
                                                  dtype=torch.float32, device=values.device))
                else:
                    row_markers.append(values[row_index, int(marker_index) + shift].float())
            maxima[target_layer] = torch.stack(row_maxima).to(torch.float16).cpu().numpy()
            markers[target_layer] = torch.stack(row_markers).to(torch.float16).cpu().numpy()
            return None

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    expected = set(range(len(layers)))
    if set(maxima) != expected or set(markers) != expected:
        missing = sorted(expected - set(maxima))
        raise RuntimeError(f"not every down_proj ran during the read pass; missing layers {missing}")
    return maxima, markers, shifts


def _control_units(layers, count: int) -> list[dict]:
    controls = part27.control_sets(layers, [TARGET], count)
    output = []
    for row in controls:
        neuron = row["neurons"][0]
        output.append({"unit": int(neuron.get("unit", neuron.get("index"))),
                       "control_seed": int(row["seed"])})
    return output


def _read_key(run_seed: int, model_kind: str, model_id: str, revision: str,
              category_id: str, sentence_id: str, wrapping: str, unit_role: str,
              unit_index: int, control_seed: int | None) -> str:
    parts = ("read", str(run_seed), model_kind, model_id, revision, category_id,
             sentence_id, wrapping, unit_role,
             str(control_seed) if control_seed is not None else "target", f"6:{unit_index}")
    return "|".join(parts)


def _read_records(samples: list[dict], model_kind: str, model_id: str, revision: str,
                  run_seed: int, captured: list[list[dict]]) -> list[dict]:
    output = []
    for sample, sample_results in zip(samples, captured, strict=True):
        for result in sample_results:
            unit_index = int(result["unit"])
            row = {
                "record_type": "read",
                "run_seed": int(run_seed),
                "model_kind": model_kind,
                "model": model_id,
                "revision": revision,
                "layer": 6,
                "unit": unit_index,
                "unit_role": result["unit_role"],
                "control_seed": result["control_seed"],
                "category": sample["category"],
                "sentence_id": sample["sentence_id"],
                "wrapping": sample["wrapping"],
                "sentence_max": result["sentence_max"],
                "assistant_marker_value": result["assistant_marker_value"],
            }
            row["row_key"] = _read_key(run_seed, model_kind, model_id, revision,
                                       row["category"], row["sentence_id"], row["wrapping"],
                                       row["unit_role"], unit_index, row["control_seed"])
            output.append(row)
    return output


def _bootstrap_median(values: list[float], seed: int,
                      resamples: int = BOOTSTRAP_RESAMPLES) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    values_array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values_array), size=(resamples, len(values_array)))
    medians = np.median(values_array[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return [float(low), float(high)]


def _stable_group_seed(base_seed: int, key: tuple) -> int:
    digest = hashlib.sha256("|".join(map(str, key)).encode("utf-8")).digest()
    return int(base_seed) + int.from_bytes(digest[:4], "big")


def summarize_reads(rows: list[dict], run_seed: int, categories: list[dict]) -> dict:
    selected = [row for row in rows if int(row.get("run_seed", -1)) == int(run_seed)]
    groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        base = (row["model_kind"], row["model"], row["revision"], row["wrapping"],
                row["unit_role"], row.get("control_seed"), int(row["unit"]), row["category"])
        groups[base]["sentence_max"].append(float(row["sentence_max"]))
        if row.get("assistant_marker_value") is not None:
            groups[base]["assistant_marker_value"].append(float(row["assistant_marker_value"]))
    summaries = []
    for key, measurements in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        model_kind, model_id, revision, wrapping, unit_role, control_seed, unit_index, category = key
        for measure, values in sorted(measurements.items()):
            ci = _bootstrap_median(values, _stable_group_seed(run_seed, key + (measure,)))
            summaries.append({
                "model_kind": model_kind, "model": model_id, "revision": revision,
                "wrapping": wrapping, "unit_role": unit_role, "control_seed": control_seed,
                "layer": 6, "unit": unit_index, "category": category,
                "measure": measure, "n": len(values), "median": float(np.median(values)),
                "bootstrap_95": ci,
            })
    return {
        "schema_version": 1,
        "tool": "part39.py (Part 39, Qwen3.5-4B L6 N753)",
        "run_seed": int(run_seed),
        "bootstrap": {"method": "sentence-resampled percentile bootstrap of the median",
                      "resamples": BOOTSTRAP_RESAMPLES, "confidence": 0.95},
        "categories": [{"id": category["id"], "label": category["label"]} for category in categories],
        "rows": summaries,
    }



def load_model_direct(model_id: str, revision: str, device: str):
    """Part 39's loader: stream the bf16 weights straight to the device.

    The shared loader (neuron_report.load_model) builds the whole model on the host before .to(device); for
    Qwen3.5-4B that peaks near 10 GB of Windows RAM and the A770 job's RAM guard stopped Part 39 three times on
    2026-10-20. low_cpu_mem_usage with a device_map places each tensor on the device as it is read.
    """
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(revision=revision, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": device})
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    torch.set_grad_enabled(False)
    return tok, model

def _empty_cache(device: str) -> None:
    gc.collect()
    if device.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _read_units(controls: list[dict]) -> list[dict]:
    units = [{"unit": TARGET["unit"], "unit_role": "target", "control_seed": None}]
    units.extend({"unit": int(row["unit"]), "unit_role": "control",
                  "control_seed": int(row["control_seed"])} for row in controls)
    return units


def _append_read_for_sample(sink: JsonlSink, sample: dict, model_kind: str,
                            model_id: str, revision: str, run_seed: int,
                            units: list[dict], sentence_max: np.ndarray,
                            assistant_marker: np.ndarray) -> None:
    layer_max = sentence_max[sample["sentence_index"], sample["wrapping_index"], 6]
    marker_values = assistant_marker[sample["sentence_index"], sample["wrapping_index"], 6]
    has_marker = sample["wrapping"] == "template"
    read_group = [{
        "unit": int(unit["unit"]), "unit_role": unit["unit_role"],
        "control_seed": unit.get("control_seed"),
        "sentence_max": float(layer_max[int(unit["unit"])]),
        "assistant_marker_value": float(marker_values[int(unit["unit"])]) if has_marker else None,
    } for unit in units]
    for row in _read_records([sample], model_kind, model_id, revision, run_seed, [read_group]):
        sink.append(row)


def load_battery(path) -> tuple[list[dict], list[str], list[tuple[str, str, str]]]:
    """Read the battery file as stored (a list of {id, label, sentences: [{id, text}]}) and check its order."""
    categories = validate_battery(_read_json(path))
    category_ids = [category["id"] for category in categories]
    if category_ids != list(SCREEN_CATEGORY_IDS):
        raise ValueError("battery category order does not match Part 39 sections A and D")
    sentence_records = [(category["id"], record["id"], record["text"])
                        for category in categories for record in category["sentences"]]
    return categories, category_ids, sentence_records


def load_persona(path) -> tuple[str, list[dict]]:
    persona = _read_json(path)
    system_prompt = str(persona["system_prompt"])
    questions = persona.get("questions")
    if not isinstance(questions, list) or len(questions) != 12:
        raise ValueError("persona data must contain exactly 12 everyday questions")
    return system_prompt, questions


def run_reads(args) -> dict:
    categories, category_ids, sentence_records = load_battery(args.battery)
    sink = JsonlSink(args.out / "reads.jsonl")
    sentence_ids = [row[1] for row in sentence_records]
    descriptor_samples = []
    for sentence_index, (category_id, sentence_id, text) in enumerate(sentence_records):
        for wrapping_index, wrapping in enumerate(("plain", "template")):
            descriptor_samples.append({"category": category_id, "sentence_id": sentence_id,
                                       "text": text, "sentence_index": sentence_index,
                                       "wrapping": wrapping, "wrapping_index": wrapping_index})
    model_specs = [("chat", args.model, args.revision),
                   ("base", args.base_model, args.base_revision)]
    for model_kind, model_id, revision in model_specs:
        arrays_path = args.out / f"reads-{model_kind}.npz"
        expected_metadata = {"model_kind": model_kind, "model": model_id,
                             "revision": revision, "run_seed": int(args.seed),
                             "battery_sha256": _sha256_file(args.battery),
                             "categories": category_ids, "sentence_ids": sentence_ids,
                             "wrappings": ["plain", "template"]}
        expected_shape = (len(sentence_records), 2, SCREEN_LAYER_COUNT, SCREEN_WIDTH)
        cached = _load_read_arrays(arrays_path, expected_metadata, expected_shape)
        if cached is not None:
            sentence_max, assistant_marker, metadata = cached
            controls = metadata.get("controls")
            if not isinstance(controls, list) or len(controls) != READ_CONTROL_COUNT:
                raise SystemExit(f"{arrays_path.name} has malformed control metadata; use a fresh --out")
            units = _read_units(controls)
            for sample in descriptor_samples:
                _append_read_for_sample(sink, sample, model_kind, model_id, revision,
                                        args.seed, units, sentence_max, assistant_marker)
            del sentence_max, assistant_marker, metadata, controls, units
            continue

        tok = model = layers = _norm = None
        try:
            tok, model = load_model_direct(model_id, revision, args.device)
            model.requires_grad_(False)
            _layer_name, layers, _norm = nr.find_text_layers(model)
            layer_widths = [int(layer.mlp.down_proj.in_features) for layer in layers]
            if len(layers) != SCREEN_LAYER_COUNT or any(width != SCREEN_WIDTH for width in layer_widths):
                raise RuntimeError(
                    f"screen expects {SCREEN_LAYER_COUNT} layers x {SCREEN_WIDTH} units; "
                    f"found {len(layers)} layers with widths {sorted(set(layer_widths))}"
                )
            if len(layers) <= TARGET["layer"] or layer_widths[TARGET["layer"]] <= TARGET["unit"]:
                raise RuntimeError("model does not contain target unit layer 6 / unit 753")
            controls = _control_units(layers, READ_CONTROL_COUNT)
            units = _read_units(controls)
            samples = []
            for descriptor in descriptor_samples:
                sample = (prepare_plain_sample(tok, descriptor["text"])
                          if descriptor["wrapping"] == "plain"
                          else prepare_template_sample(tok, descriptor["text"]))
                samples.append({**descriptor, **sample})
            sentence_max = np.empty(expected_shape, dtype=np.float16)
            assistant_marker = np.full(expected_shape, np.nan, dtype=np.float16)
            for start in range(0, len(samples), args.batch_size):
                batch = samples[start:start + args.batch_size]
                max_by_layer, marker_by_layer, _shifts = _capture_all_layers(model, layers, tok, batch)
                for batch_index, sample in enumerate(batch):
                    sentence_index = sample["sentence_index"]
                    wrapping_index = sample["wrapping_index"]
                    for layer_index in range(len(layers)):
                        sentence_max[sentence_index, wrapping_index, layer_index] = max_by_layer[layer_index][batch_index]
                        assistant_marker[sentence_index, wrapping_index, layer_index] = marker_by_layer[layer_index][batch_index]
                    _append_read_for_sample(sink, sample, model_kind, model_id, revision,
                                            args.seed, units, sentence_max, assistant_marker)
                del max_by_layer, marker_by_layer, batch
            metadata = {**expected_metadata, "controls": controls,
                        "shape": list(expected_shape), "dtype": "float16"}
            _save_read_arrays(arrays_path, sentence_max, assistant_marker, metadata)
            del controls, units, samples, sentence_max, assistant_marker, metadata
        finally:
            if layers is not None:
                del layers
            if _norm is not None:
                del _norm
            if model is not None:
                del model
            if tok is not None:
                del tok
            _empty_cache(args.device)
    summary = summarize_reads(sink.rows(), args.seed, categories)
    summary["battery_sha256"] = _sha256_file(args.battery)
    summary["models"] = {"chat": {"model": args.model, "revision": args.revision},
                         "base": {"model": args.base_model, "revision": args.base_revision}}
    _write_json_atomic(summary, args.out / "reads-summary.json")
    return summary


def _unit_mad(values: np.ndarray, block_size: int = 8192) -> np.ndarray:
    result = np.empty(values.shape[1], dtype=np.float32)
    for start in range(0, values.shape[1], block_size):
        end = min(values.shape[1], start + block_size)
        block = values[:, start:end].astype(np.float32)
        median = np.median(block, axis=0)
        result[start:end] = np.median(np.abs(block - median), axis=0)
    return result


def _normalized_contrast(values: np.ndarray, positive: np.ndarray, negative: np.ndarray,
                         mad: np.ndarray, block_size: int = 8192) -> np.ndarray:
    result = np.empty(values.shape[1], dtype=np.float32)
    denominator = np.maximum(mad, 0.01)
    for start in range(0, values.shape[1], block_size):
        end = min(values.shape[1], start + block_size)
        block = values[:, start:end].astype(np.float32)
        positive_median = np.median(block[positive], axis=0)
        negative_median = np.median(block[negative], axis=0)
        result[start:end] = (positive_median - negative_median) / denominator[start:end]
    return result


def _top_rank_mask(scores: np.ndarray, top_count: int) -> np.ndarray:
    count = min(max(int(top_count), 1), len(scores))
    order = np.argsort(scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[-count:]] = True
    return mask


def compute_screen_statistics(plain_values: np.ndarray, marker_values: np.ndarray,
                              category_labels: np.ndarray, seed: int = SCREEN_SEED,
                              shuffles: int = SCREEN_SHUFFLES) -> dict:
    """Compute the preregistered contrasts, shuffle max-statistic, and split-half ranks."""
    values = np.asarray(plain_values, dtype=np.float16)
    markers = np.asarray(marker_values, dtype=np.float16)
    labels = np.asarray(category_labels, dtype=np.int8)
    if values.ndim != 2 or values.shape[0] != 120 or markers.shape != values.shape:
        raise ValueError("screen inputs must have shape [120 sentences, units]")
    if labels.shape != (120,) or set(labels.tolist()) != set(range(6)):
        raise ValueError("screen requires six balanced sentence category labels")
    if any(int(np.sum(labels == category)) != 20 for category in range(6)):
        raise ValueError("screen requires 20 sentences in each category")
    if shuffles < 1:
        raise ValueError("screen requires at least one category-label shuffle")

    unit_count = values.shape[1]
    mad = _unit_mad(values)
    denominator = np.maximum(mad, 0.01)
    addressee = _normalized_contrast(values, np.isin(labels, (0, 2)),
                                     np.isin(labels, (1, 3, 4, 5)), mad)
    request_only = _normalized_contrast(values, labels == 0, labels == 1, mad)
    ai_self_only = _normalized_contrast(values, labels == 2, labels == 3, mad)
    marker_median = np.empty(unit_count, dtype=np.float32)
    for start in range(0, unit_count, 8192):
        end = min(unit_count, start + 8192)
        marker_median[start:end] = np.median(markers[:, start:end].astype(np.float32), axis=0)

    half_scores = []
    half_masks = []
    top_count = max(1, int(np.ceil(unit_count * 0.01)))
    for first_sentence in (0, 10):
        indices = np.asarray([
            category * 20 + sentence_offset
            for category in range(6)
            for sentence_offset in range(first_sentence, first_sentence + 10)
        ], dtype=np.int64)
        half_values = values[indices]
        half_labels = labels[indices]
        half_mad = _unit_mad(half_values)
        half_score = _normalized_contrast(
            half_values, np.isin(half_labels, (0, 2)), np.isin(half_labels, (1, 3, 4, 5)), half_mad)
        half_scores.append(half_score)
        half_masks.append(_top_rank_mask(half_score, top_count))

    rng = np.random.default_rng(int(seed))
    null_maximum = np.empty(int(shuffles), dtype=np.float32)
    for shuffle_index in range(int(shuffles)):
        shuffled_labels = rng.permutation(labels)
        positive = np.isin(shuffled_labels, (0, 2))
        negative = np.isin(shuffled_labels, (1, 3, 4, 5))
        maximum = -np.inf
        for start in range(0, unit_count, 8192):
            end = min(unit_count, start + 8192)
            block = values[:, start:end].astype(np.float32)
            positive_median = np.median(block[positive], axis=0)
            negative_median = np.median(block[negative], axis=0)
            scores = ((positive_median - negative_median) / denominator[start:end])
            maximum = max(maximum, float(np.max(scores)))
        null_maximum[shuffle_index] = maximum

    null_threshold = float(np.quantile(null_maximum, 0.99))
    pass_mask = ((addressee > null_threshold) & half_masks[0] & half_masks[1])
    return {
        "addressee_contrast": addressee,
        "request_only_contrast": request_only,
        "ai_self_only_contrast": ai_self_only,
        "marker_median": marker_median,
        "mad": mad,
        "half1_addressee_contrast": half_scores[0],
        "half2_addressee_contrast": half_scores[1],
        "half1_top1_percent": half_masks[0],
        "half2_top1_percent": half_masks[1],
        "null_maximum": null_maximum,
        "null_threshold_99": null_threshold,
        "top1_percent_count": top_count,
        "pass_mask": pass_mask,
    }


def _screen_logit_lens(args, model_id: str, revision: str,
                       passing_units: list[int]) -> dict[int, dict]:
    if not passing_units:
        return {}
    tok = model = layers = norm = None
    output = {}
    try:
        tok, model = load_model_direct(model_id, revision, args.device)
        model.requires_grad_(False)
        _layer_name, layers, norm = nr.find_text_layers(model)
        by_layer: dict[int, list[int]] = defaultdict(list)
        for flat_index in passing_units:
            layer_index, unit_index = divmod(int(flat_index), SCREEN_WIDTH)
            by_layer[layer_index].append(unit_index)
        for layer_index, units in sorted(by_layer.items()):
            down_proj = layers[layer_index].mlp.down_proj
            top_ids = part26.logit_lens_top_ids(tok, model, norm, down_proj, units, top=30,
                                                device=args.device, max_gpu_gb=args.max_gpu_gb)
            bottom_ids = part26.logit_lens_top_ids(
                tok, model, norm, SimpleNamespace(weight=-down_proj.weight), units, top=30,
                device=args.device, max_gpu_gb=args.max_gpu_gb)
            for offset, unit_index in enumerate(units):
                top = [int(value) for value in top_ids[offset].tolist()]
                bottom = [int(value) for value in bottom_ids[offset].tolist()]
                top_decoded = [tok.decode([value]) for value in top]
                bottom_decoded = [tok.decode([value]) for value in bottom]
                output[layer_index * SCREEN_WIDTH + unit_index] = {
                    "top_ids": top,
                    "top_tokens": [part26.normalize_write_token(value) for value in top_decoded],
                    "bottom_ids": bottom,
                    "bottom_tokens": [part26.normalize_write_token(value) for value in bottom_decoded],
                }
        return output
    finally:
        if layers is not None:
            del layers
        if norm is not None:
            del norm
        if model is not None:
            del model
        if tok is not None:
            del tok
        _empty_cache(args.device)


def _model_file_slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", model_id.casefold()).strip("-")


def _screen_one_model(args, model_kind: str, model_id: str, revision: str) -> dict:
    arrays_path = args.out / f"reads-{model_kind}.npz"
    screen_path = args.out / f"screen-{_model_file_slug(model_id)}.json"
    if not arrays_path.is_file():
        raise SystemExit(f"{arrays_path.name} is missing; run reads first")
    expected_metadata = {"model_kind": model_kind, "model": model_id,
                         "revision": revision, "run_seed": int(args.seed),
                         "battery_sha256": _sha256_file(args.battery),
                         "categories": list(SCREEN_CATEGORY_IDS),
                         "wrappings": ["plain", "template"]}
    with np.load(arrays_path, allow_pickle=False) as archive:
        try:
            metadata = json.loads(str(archive["metadata"].item()))
            sentence_max = np.asarray(archive["sentence_max"], dtype=np.float16)
            assistant_marker = np.asarray(archive["assistant_marker_value"], dtype=np.float16)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{arrays_path.name} is malformed; run reads again ({exc})") from exc
    expected_shape = (120, 2, SCREEN_LAYER_COUNT, SCREEN_WIDTH)
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise SystemExit(f"{arrays_path.name} provenance mismatch ({key}); rerun reads with matching options")
    if sentence_max.shape != expected_shape or assistant_marker.shape != expected_shape:
        raise SystemExit(f"{arrays_path.name} shape mismatch; expected {expected_shape}")

    labels = np.repeat(np.arange(6, dtype=np.int8), 20)
    plain_values = sentence_max[:, 0].reshape(120, -1)
    marker_values = assistant_marker[:, 1].reshape(120, -1)
    stats = compute_screen_statistics(plain_values, marker_values, labels,
                                      seed=SCREEN_SEED, shuffles=SCREEN_SHUFFLES)
    passing_indices = np.flatnonzero(stats["pass_mask"]).astype(int).tolist()
    target_index = TARGET["layer"] * SCREEN_WIDTH + TARGET["unit"]
    target_passes = bool(stats["pass_mask"][target_index])
    screen_broken = model_kind == "chat" and not target_passes
    lens = _screen_logit_lens(args, model_id, revision, passing_indices)
    passing_rows = []
    for flat_index in passing_indices:
        layer_index, unit_index = divmod(flat_index, SCREEN_WIDTH)
        row = {
            "layer": layer_index, "unit": unit_index,
            "addressee_contrast": float(stats["addressee_contrast"][flat_index]),
            "request_only_contrast": float(stats["request_only_contrast"][flat_index]),
            "ai_self_only_contrast": float(stats["ai_self_only_contrast"][flat_index]),
            "marker_median": float(stats["marker_median"][flat_index]),
            "mad": float(stats["mad"][flat_index]),
            "half1_addressee_contrast": float(stats["half1_addressee_contrast"][flat_index]),
            "half2_addressee_contrast": float(stats["half2_addressee_contrast"][flat_index]),
        }
        if flat_index in lens:
            row["logit_lens"] = lens[flat_index]
        passing_rows.append(row)
    out = {
        "schema_version": 1,
        "tool": "part39.py (Part 39 section D screen)",
        "model_kind": model_kind, "model": model_id, "revision": revision,
        "run_seed": int(args.seed), "read_arrays": arrays_path.name,
        "battery_sha256": expected_metadata["battery_sha256"],
        "read_array_shape": list(expected_shape), "dtype": "float16",
        "screen_broken": screen_broken,
        "target": {"layer": TARGET["layer"], "unit": TARGET["unit"],
                   "passes": target_passes,
                   "addressee_contrast": float(stats["addressee_contrast"][target_index]),
                   "request_only_contrast": float(stats["request_only_contrast"][target_index]),
                   "ai_self_only_contrast": float(stats["ai_self_only_contrast"][target_index]),
                   "marker_median": float(stats["marker_median"][target_index])},
        "method": {
            "addressee_contrast": "(median of category 1 and 3 plain sentence maxima - median of categories 2, 4, 5, and 6) / MAD across all 120 plain maxima, with MAD floor 0.01",
            "request_only_contrast": "(category 1 median - category 2 median) / same all-sentence MAD",
            "ai_self_only_contrast": "(category 3 median - category 4 median) / same all-sentence MAD",
            "marker_value": "median of templated assistant-marker values over 120 sentences",
            "null": "maximum addressee contrast over all units after each of 200 category-label shuffles preserving six groups of 20; pass requires observed value above the 99th percentile",
            "split_half": "addressee contrast is in the exact top 1% of units in each sentence half (sentences 1-10 and 11-20 within every category)",
            "logit_lens": "Part 26 / Part 27 down_proj-column lens; top and bottom 30 tokens",
        },
        "null": {"seed": SCREEN_SEED, "shuffles": SCREEN_SHUFFLES,
                 "threshold_99": float(stats["null_threshold_99"]),
                 "maximum_statistics": [float(value) for value in stats["null_maximum"]]},
        "split_half": {"top1_percent_units_per_half": stats["top1_percent_count"]},
        "passing_unit_count": len(passing_rows),
        "passing_units": passing_rows,
    }
    _write_json_atomic(out, screen_path)
    del sentence_max, assistant_marker, plain_values, marker_values, stats, lens
    return out


def run_screen(args) -> dict:
    chat = _screen_one_model(args, "chat", args.model, args.revision)
    base = _screen_one_model(args, "base", args.base_model, args.base_revision)
    return {"chat": chat, "base": base}


def broken_answer(text: str) -> bool:
    words = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    if len(words) < 3:
        return True
    fourgrams = Counter(tuple(words[index:index + 4]) for index in range(len(words) - 3))
    return any(count >= 3 for count in fourgrams.values())


def score_answer(text: str) -> dict:
    is_broken = broken_answer(text)
    makers = list(dict.fromkeys(match.group(0) for match in MAKER_RE.finditer(text)))
    return {"broken": is_broken, "makers_named": makers,
            "maker_named": bool(makers),
            "ai_claim": bool(not is_broken and AI_CLAIM_RE.search(text))}


def _zero_key(args, condition: str, question_id: str, sampling: str,
              sample_index: int, control_seed: int | None = None) -> str:
    parts = ("zero", str(args.seed), args.model, args.revision, condition,
             str(control_seed) if control_seed is not None else "target",
             question_id, sampling, str(sample_index), str(args.max_new_tokens))
    return "|".join(parts)


def _push_key(args, condition: str, question_id: str, sample_index: int,
              k: int | None = None, control_seed: int | None = None) -> str:
    parts = ("push", str(args.seed), args.model, args.revision, condition,
             str(k) if k is not None else "none",
             str(control_seed) if control_seed is not None else "none",
             question_id, str(sample_index), str(args.max_new_tokens))
    return "|".join(parts)


def _generate_text(tok, model, layers, prompt: str, operations: dict | None,
                   seed: int, max_new_tokens: int, do_sample: bool) -> str:
    encoded = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"].to(_model_device(model))
    mask = encoded.get("attention_mask", torch.ones_like(ids)).to(ids.device)
    handles = part25.install_intervention_hooks(layers, [operations]) if operations else []
    try:
        torch.manual_seed(int(seed))
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(int(seed))
        kwargs = {"input_ids": ids, "attention_mask": mask,
                  "max_new_tokens": int(max_new_tokens),
                  "do_sample": bool(do_sample),
                  "pad_token_id": getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None),
                  "eos_token_id": getattr(tok, "eos_token_id", None)}
        if do_sample:
            kwargs.update({"temperature": 0.7, "top_p": 0.95})
        with torch.inference_mode():
            generated = model.generate(**kwargs)
        return tok.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        for handle in handles:
            handle.remove()


def _generation_row(args, experiment: str, condition: str, question: dict,
                    sample_index: int, sampling: str, seed: int, answer: str,
                    operations: dict | None, k: int | None = None,
                    control_seed: int | None = None) -> dict:
    scored = score_answer(answer)
    if experiment == "zero":
        key = _zero_key(args, condition, question["id"], sampling, sample_index, control_seed)
    else:
        key = _push_key(args, condition, question["id"], sample_index, k, control_seed)
    return {
        "row_key": key, "record_type": "generation", "experiment": experiment,
        "run_seed": int(args.seed), "model": args.model, "revision": args.revision,
        "condition": condition, "control_seed": control_seed, "layer": 6,
        "unit": TARGET["unit"] if condition != "control" else None,
        "clamped_unit": None if operations is None else int(next(iter(operations))[1]),
        "k": k, "question_id": question["id"], "question": question["question"],
        "sampling": sampling, "sample_index": int(sample_index), "seed": int(seed),
        "max_new_tokens": int(args.max_new_tokens), "answer": answer, **scored,
    }


def _load_chat(args):
    tok, model = load_model_direct(args.model, args.revision, args.device)
    model.requires_grad_(False)
    _layer_name, layers, _norm = nr.find_text_layers(model)
    if len(layers) <= TARGET["layer"]:
        raise RuntimeError(f"model has no layer {TARGET['layer']}")
    width = int(layers[TARGET["layer"]].mlp.down_proj.in_features)
    if width <= TARGET["unit"]:
        raise RuntimeError(f"layer 6 has only {width} down_proj input units; need unit 753")
    return tok, model, layers


def _sample_seed(base: int, *parts: int) -> int:
    value = int(base)
    for part in parts:
        value = (value * 1_000_003 + int(part) + 97) % (2**31 - 1)
    return value


def run_zero(args) -> list[dict]:
    prompts = list(circuit.ID_PROMPTS)
    questions = [{"id": f"identity_{index + 1}", "question": question}
                 for index, question in enumerate(prompts)]
    sink = JsonlSink(args.out / "zero.jsonl")
    tok = model = layers = None
    try:
        tok, model, layers = _load_chat(args)
        controls = _control_units(layers, GENERATION_CONTROL_COUNT)
        target_pair = [(TARGET["layer"], TARGET["unit"])]
        zero_target = part25._operations(target_pair, "zero")
        jobs = []
        for question_index, question in enumerate(questions):
            for condition, operations in (("baseline", None), ("zeroed", zero_target)):
                jobs.append((condition, question, "greedy", 0,
                             _sample_seed(args.seed, question_index, 0), operations, None))
                for sample_index in range(args.samples):
                    jobs.append((condition, question, "sample", sample_index,
                                 _sample_seed(args.seed, question_index, sample_index + 1),
                                 operations, None))
            for control in controls:
                unit_pair = [(TARGET["layer"], int(control["unit"]))]
                operations = part25._operations(unit_pair, "zero")
                for sample_index in range(2):
                    jobs.append(("control", question, "sample", sample_index,
                                 _sample_seed(args.seed, int(control["control_seed"]),
                                              question_index, sample_index + 31),
                                 operations, int(control["control_seed"])))
        for condition, question, sampling, sample_index, seed, operations, control_seed in jobs:
            key = _zero_key(args, condition, question["id"], sampling, sample_index, control_seed)
            if key in sink.seen:
                continue
            prompt = nr.build_prompt(tok, question["question"], False)
            answer = _generate_text(tok, model, layers, prompt, operations, seed,
                                    args.max_new_tokens, do_sample=(sampling == "sample"))
            sink.append(_generation_row(args, "zero", condition, question, sample_index,
                                        sampling, seed, answer, operations,
                                        control_seed=control_seed))
        return sink.rows()
    finally:
        if layers is not None:
            del layers
        if model is not None:
            del model
        if tok is not None:
            del tok
        _empty_cache(args.device)


def _mean_chat_assistant_value(reads_path: Path, args) -> float:
    path = reads_path / "reads.jsonl"
    rows = [row for row in read_jsonl(path)
            if row.get("record_type") == "read" and row.get("model_kind") == "chat"
            and row.get("model") == args.model and row.get("revision") == args.revision
            and int(row.get("run_seed", -1)) == int(args.seed)
            and row.get("wrapping") == "template" and row.get("unit_role") == "target"
            and row.get("layer") == TARGET["layer"] and row.get("unit") == TARGET["unit"]]
    unique = {(row["category"], row["sentence_id"]): row for row in rows
              if row.get("assistant_marker_value") is not None}
    if len(unique) != 120:
        raise RuntimeError(
            f"push needs completed reads first: found {len(unique)}/120 chat-model assistant-marker values; "
            "run part39.py reads with the same --model, --revision, --out, and --seed"
        )
    return float(np.mean([float(row["assistant_marker_value"]) for row in unique.values()]))


def run_push(args) -> list[dict]:
    system_prompt, questions = load_persona(args.persona)
    mean_value = _mean_chat_assistant_value(args.out, args)
    sink = JsonlSink(args.out / "push.jsonl")
    tok = model = layers = None
    try:
        tok, model, layers = _load_chat(args)
        controls = _control_units(layers, GENERATION_CONTROL_COUNT)
        jobs = []
        for question_index, question in enumerate(questions):
            for sample_index in range(args.samples):
                seed = _sample_seed(args.seed, question_index, sample_index + 1)
                jobs.append(("unclamped", question, sample_index, seed, None, None, None))
                for k in (1, 2):
                    level = float(k * mean_value)
                    operations = {(TARGET["layer"], TARGET["unit"]): level}
                    jobs.append(("target", question, sample_index, seed, operations, k, None))
            for control in controls:
                operations = {(TARGET["layer"], int(control["unit"])): float(2 * mean_value)}
                control_seed = int(control["control_seed"])
                seed = _sample_seed(args.seed, control_seed + 100, question_index + 1)
                jobs.append(("control", question, 0, seed, operations, 2, control_seed))
        for condition, question, sample_index, seed, operations, k, control_seed in jobs:
            key = _push_key(args, condition, question["id"], sample_index, k, control_seed)
            if key in sink.seen:
                continue
            prompt = nr.build_prompt(tok, question["question"], False, system_prompt=system_prompt)
            answer = _generate_text(tok, model, layers, prompt, operations, seed,
                                    args.max_new_tokens, do_sample=True)
            sink.append(_generation_row(args, "push", condition, question, sample_index,
                                        "sample", seed, answer, operations, k=k,
                                        control_seed=control_seed))
        return sink.rows()
    finally:
        if layers is not None:
            del layers
        if model is not None:
            del model
        if tok is not None:
            del tok
        _empty_cache(args.device)


def _summary_value(summary: dict, *, model_kind: str, model: str, revision: str,
                   wrapping: str, category: str, measure: str) -> dict:
    matches = [row for row in summary.get("rows", [])
               if row.get("model_kind") == model_kind and row.get("model") == model
               and row.get("revision") == revision and row.get("wrapping") == wrapping
               and row.get("unit_role") == "target" and row.get("unit") == TARGET["unit"]
               and row.get("category") == category and row.get("measure") == measure]
    if len(matches) != 1:
        raise ValueError(f"reads-summary is missing a unique {model_kind}/{wrapping}/{category}/{measure} target row")
    if matches[0].get("n") != 20:
        raise ValueError(f"reads-summary row {model_kind}/{wrapping}/{category}/{measure} has n={matches[0].get('n')}; expected 20")
    return matches[0]


def _selected_generation_rows(path: Path, experiment: str, args) -> list[dict]:
    return [row for row in read_jsonl(path)
            if row.get("experiment") == experiment and row.get("model") == args.model
            and row.get("revision") == args.revision and int(row.get("run_seed", -1)) == args.seed
            and int(row.get("max_new_tokens", -1)) == args.max_new_tokens]


def _overall_assistant_median(path: Path, *, model_kind: str, model: str,
                              revision: str, run_seed: int) -> float:
    rows = [row for row in read_jsonl(path)
            if row.get("record_type") == "read" and row.get("model_kind") == model_kind
            and row.get("model") == model and row.get("revision") == revision
            and int(row.get("run_seed", -1)) == int(run_seed)
            and row.get("unit_role") == "target" and row.get("unit") == TARGET["unit"]
            and row.get("wrapping") == "template" and row.get("assistant_marker_value") is not None]
    by_sentence = {(row["category"], row["sentence_id"]): float(row["assistant_marker_value"])
                   for row in rows}
    if len(by_sentence) != 120:
        raise ValueError(f"reads.jsonl has {len(by_sentence)}/120 {model_kind} assistant-marker values")
    return float(np.median(list(by_sentence.values())))


def _require_count(rows: list[dict], condition: str, question_ids: list[str],
                   count_per_question: int, sampling: str | None = None,
                   control_seed: int | None = None) -> list[dict]:
    selected = [row for row in rows if row.get("condition") == condition
                and (sampling is None or row.get("sampling") == sampling)
                and (control_seed is None or row.get("control_seed") == control_seed)]
    counts = Counter(row.get("question_id") for row in selected)
    missing = {question_id: counts.get(question_id, 0) for question_id in question_ids
               if counts.get(question_id, 0) != count_per_question}
    if missing:
        raise ValueError(f"incomplete {condition} rows: expected {count_per_question} per question; got {missing}")
    return selected


def _share(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot score an empty answer set")
    claims = sum(bool(row.get("ai_claim")) and not bool(row.get("broken")) for row in rows)
    broken = sum(bool(row.get("broken")) for row in rows)
    return {"claims": claims, "n": len(rows), "broken": broken,
            "share": claims / len(rows)}


def score_results(args) -> dict:
    categories = validate_battery(_read_json(args.battery))
    summary_path = args.out / "reads-summary.json"
    if not summary_path.is_file():
        raise SystemExit("reads-summary.json is missing; run reads first")
    summary = _read_json(summary_path)
    if (summary.get("battery_sha256") != _sha256_file(args.battery)
            or int(summary.get("run_seed", -1)) != int(args.seed)):
        raise SystemExit("reads-summary provenance mismatch; rerun reads with matching options")
    category_ids = [category["id"] for category in categories]
    chat_plain = {category: _summary_value(summary, model_kind="chat", model=args.model,
                                           revision=args.revision, wrapping="plain",
                                           category=category, measure="sentence_max")
                  for category in category_ids}
    chat_template = {category: _summary_value(summary, model_kind="chat", model=args.model,
                                              revision=args.revision, wrapping="template",
                                              category=category, measure="assistant_marker_value")
                     for category in category_ids}
    base_plain = {category: _summary_value(summary, model_kind="base", model=args.base_model,
                                           revision=args.base_revision, wrapping="plain",
                                           category=category, measure="sentence_max")
                  for category in category_ids}
    base_template = {category: _summary_value(summary, model_kind="base", model=args.base_model,
                                              revision=args.base_revision, wrapping="template",
                                              category=category, measure="assistant_marker_value")
                     for category in category_ids}

    zero_rows = _selected_generation_rows(args.out / "zero.jsonl", "zero", args)
    question_ids = [f"identity_{index}" for index in range(1, 7)]
    zero_base_greedy = _require_count(zero_rows, "baseline", question_ids, 1, "greedy")
    zero_target_greedy = _require_count(zero_rows, "zeroed", question_ids, 1, "greedy")
    zero_base_samples = _require_count(zero_rows, "baseline", question_ids, args.samples, "sample")
    zero_target_samples = _require_count(zero_rows, "zeroed", question_ids, args.samples, "sample")
    maker_changes = []
    baseline_by_question = {row["question_id"]: row for row in zero_base_greedy}
    for row in zero_target_greedy:
        baseline_names = sorted(name.casefold() for name in baseline_by_question[row["question_id"]]["makers_named"])
        zero_names = sorted(name.casefold() for name in row["makers_named"])
        if baseline_names != zero_names:
            maker_changes.append({"question_id": row["question_id"],
                                  "baseline": baseline_by_question[row["question_id"]]["makers_named"],
                                  "zeroed": row["makers_named"]})
    zero_base_share = _share(zero_base_samples)
    zero_target_share = _share(zero_target_samples)
    zero_delta = zero_target_share["share"] - zero_base_share["share"]

    push_rows = _selected_generation_rows(args.out / "push.jsonl", "push", args)
    persona = _read_json(args.persona)
    push_question_ids = [question["id"] for question in persona["questions"]]
    push_unclamped = _share(_require_count(push_rows, "unclamped", push_question_ids,
                                          args.samples, "sample"))
    push_k1 = _share(_require_count([row for row in push_rows if row.get("k") == 1],
                                    "target", push_question_ids, args.samples, "sample"))
    push_k2 = _share(_require_count([row for row in push_rows if row.get("k") == 2],
                                    "target", push_question_ids, args.samples, "sample"))
    control_shares = {}
    control_seeds = sorted({int(row["control_seed"]) for row in push_rows
                            if row.get("condition") == "control" and row.get("control_seed") is not None})
    if len(control_seeds) != GENERATION_CONTROL_COUNT:
        raise ValueError(f"push has {len(control_seeds)} random controls; expected {GENERATION_CONTROL_COUNT}")
    for control_seed in control_seeds:
        control_rows = _require_count(push_rows, "control", push_question_ids, 1, "sample", control_seed)
        control_shares[str(control_seed)] = _share(control_rows)
    push_delta_k1 = push_k1["share"] - push_unclamped["share"]
    push_delta_k2 = push_k2["share"] - push_unclamped["share"]
    exceeds_every_control = all(push_k2["share"] > item["share"] for item in control_shares.values())

    medians_chat = {category: float(chat_plain[category]["median"]) for category in category_ids}
    medians_base = {category: float(base_plain[category]["median"]) for category in category_ids}
    chat_assistant = {category: float(chat_template[category]["median"]) for category in category_ids}
    base_overall = _overall_assistant_median(args.out / "reads.jsonl", model_kind="base",
                                             model=args.base_model, revision=args.base_revision,
                                             run_seed=args.seed)
    chat_overall = _overall_assistant_median(args.out / "reads.jsonl", model_kind="chat",
                                             model=args.model, revision=args.revision,
                                             run_seed=args.seed)
    category_ids_set = set(category_ids)
    if not {"help_request", "second_person_statement", "ai_self_description",
            "third_person_ai", "neutral_prose"}.issubset(category_ids_set):
        raise ValueError("battery category ids do not match the Part 39 preregistered categories")
    ratios_391 = {}
    for numerator in ("help_request", "ai_self_description"):
        for denominator, minimum in (("second_person_statement", 2),
                                     ("third_person_ai", 4), ("neutral_prose", 4)):
            den = medians_chat[denominator]
            ratio = medians_chat[numerator] / den if den != 0 else None
            ratios_391[f"{numerator}/{denominator}"] = {
                "ratio": ratio, "minimum": minimum,
                "passes": ratio is not None and ratio >= minimum,
            }
    base_order_checks = {
        f"{high}>{low}": medians_base[high] > medians_base[low]
        for high in ("help_request", "ai_self_description")
        for low in ("third_person_ai", "neutral_prose")
    }
    screen_paths = {"chat": args.out / f"screen-{_model_file_slug(args.model)}.json",
                    "base": args.out / f"screen-{_model_file_slug(args.base_model)}.json"}
    if any(not path.is_file() for path in screen_paths.values()):
        raise SystemExit("a model-specific screen JSON is missing; run screen first")
    screen_chat = _read_json(screen_paths["chat"])
    screen_base = _read_json(screen_paths["base"])
    for kind, screen_obj, expected_model, expected_revision in (
            ("chat", screen_chat, args.model, args.revision),
            ("base", screen_base, args.base_model, args.base_revision)):
        if (screen_obj.get("model_kind") != kind or screen_obj.get("model") != expected_model
                or screen_obj.get("revision") != expected_revision
                or screen_obj.get("battery_sha256") != _sha256_file(args.battery)
                or int(screen_obj.get("run_seed", -1)) != int(args.seed)):
            raise SystemExit(f"screen-{_model_file_slug(expected_model)}.json provenance mismatch; rerun screen with matching options")
    screen_is_broken = bool(screen_chat.get("screen_broken")) or not bool(screen_chat.get("target", {}).get("passes"))
    if screen_is_broken:
        screen_overview = {"status": "broken",
                           "chat_target_passes": bool(screen_chat.get("target", {}).get("passes")),
                           "interpretation": "section D results were not evaluated because L6 N753 failed the chat screen gate"}
        screen_predictions = {
            prediction_id: {"authored_claim": claim, "scorable": False, "supported": None,
                            "reason": "L6 N753 did not pass the chat-model screen gate; section D results are invalid."}
            for prediction_id, claim in (
                ("39.7", "Claude: at least three other chat-model units pass the section D screen."),
                ("39.8", "Claude: at least half of chat-model passing units sit in layers 0 to 12."),
                ("39.9", "Claude: another passing unit has an AI or assistant term in its logit-lens top 30."),
                ("39.10", "Claude: the base screen passes at least half as many units as the chat screen."),
            )
        }
    else:
        chat_passes = screen_chat.get("passing_units", [])
        base_passes = screen_base.get("passing_units", [])
        screen_overview = {
            "status": "valid", "chat_target_passes": True,
            "chat_passing_unit_count": len(chat_passes),
            "base_passing_unit_count": len(base_passes),
        }
        target_key = (TARGET["layer"], TARGET["unit"])
        if not any((int(row["layer"]), int(row["unit"])) == target_key for row in chat_passes):
            raise ValueError("chat screen says L6 N753 passed but omits it from passing_units")
        chat_other = [row for row in chat_passes
                      if (int(row["layer"]), int(row["unit"])) != target_key]
        early_count = sum(int(row["layer"]) <= 12 for row in chat_passes)
        early_share = early_count / len(chat_passes) if chat_passes else 0.0
        identity_top_matches = [row for row in chat_other
                                if any(LOGIT_IDENTITY_RE.search(token)
                                       for token in row.get("logit_lens", {}).get("top_tokens", []))]
        screen_predictions = {
            "39.7": {"authored_claim": "Claude: at least three other chat-model units pass the section D screen.",
                     "other_passing_unit_count": len(chat_other),
                     "supported": len(chat_other) >= 3},
            "39.8": {"authored_claim": "Claude: at least half of chat-model passing units sit in layers 0 to 12.",
                     "passing_unit_count": len(chat_passes), "early_layer_count": early_count,
                     "early_layer_share": early_share,
                     "supported": early_share >= 0.5},
            "39.9": {"authored_claim": "Claude: another passing unit has an AI or assistant term in its logit-lens top 30.",
                     "matching_passing_units": [
                         {"layer": int(row["layer"]), "unit": int(row["unit"]),
                          "matching_tokens": [token for token in row.get("logit_lens", {}).get("top_tokens", [])
                                              if LOGIT_IDENTITY_RE.search(token)]}
                         for row in identity_top_matches
                     ],
                     "supported": bool(identity_top_matches)},
            "39.10": {"authored_claim": "Claude: the base screen passes at least half as many units as the chat screen.",
                      "chat_passing_unit_count": len(chat_passes),
                      "base_passing_unit_count": len(base_passes),
                      "supported": len(base_passes) >= 0.5 * len(chat_passes)},
        }
    predictions = {
        "39.1": {"authored_claim": "Claude: chat plain medians for categories 1 and 3 are >=2x category 2 and >=4x categories 5 and 6.",
                 "category_medians": medians_chat, "checks": ratios_391,
                 "supported": all(row["passes"] for row in ratios_391.values())},
        "39.2": {"authored_claim": "Claude: the chat-template assistant-marker median exceeds 2.0 in every category.",
                 "assistant_marker_medians": chat_assistant,
                 "checks": {category: value > 2.0 for category, value in chat_assistant.items()},
                 "supported": all(value > 2.0 for value in chat_assistant.values())},
        "39.3": {"authored_claim": "Claude: the base has the same plain-read order and an assistant-marker median below half the chat model.",
                 "base_plain_medians": medians_base, "base_order_checks": base_order_checks,
                 "assistant_marker_overall_median": {"base": base_overall, "chat": chat_overall,
                                                      "base_over_chat": base_overall / chat_overall if chat_overall else None},
                 "supported": all(base_order_checks.values()) and chat_overall != 0
                              and base_overall < 0.5 * chat_overall},
        "39.4": {"authored_claim": "Claude: zeroing changes the maker in none of six greedy answers and moves sampled AI-claim share by under 10 points.",
                 "greedy_maker_changes": maker_changes,
                 "baseline_sampled_ai_share": zero_base_share,
                 "zeroed_sampled_ai_share": zero_target_share,
                 "share_change": zero_delta,
                 "supported": not maker_changes and abs(zero_delta) < 0.10},
        "39.5": {"authored_claim": "Claude: k=2 raises Sam-persona AI-claim share by >=20 points and beats each random unit.",
                 "unclamped_share": push_unclamped, "k2_share": push_k2,
                 "share_change": push_delta_k2, "random_control_shares": control_shares,
                 "exceeds_every_random_control": exceeds_every_control,
                 "supported": push_delta_k2 >= 0.20 and exceeds_every_control},
        "39.6": {"authored_claim": "Claude: k=1 raises Sam-persona AI-claim share by under 10 points.",
                 "unclamped_share": push_unclamped, "k1_share": push_k1,
                 "share_change": push_delta_k1,
                 "supported": push_delta_k1 < 0.10},
    }
    predictions.update(screen_predictions)
    out = {
        "schema_version": 1,
        "tool": "part39.py (Part 39 score)",
        "run_seed": int(args.seed),
        "models": {"chat": {"model": args.model, "revision": args.revision},
                   "base": {"model": args.base_model, "revision": args.base_revision}},
        "scoring": {"ai_claim_regex": AI_CLAIM_RE.pattern,
                    "maker_names": MAKER_NAMES,
                    "broken_answer_rule": "fewer than 3 lexical words or any repeated normalized 4-gram appearing 3+ times",
                    "broken_answers_count_as_ai_claim": False,
                    "screen_identity_top_token_regex": LOGIT_IDENTITY_RE.pattern},
        "read_measurements": {"39.1": medians_chat, "39.2": chat_assistant,
                              "39.3_base_plain": medians_base,
                              "39.3_assistant_marker_overall": {"base": base_overall,
                                                                  "chat": chat_overall}},
        "zero": {"greedy_maker_changes": maker_changes,
                 "baseline_sampled_ai_share": zero_base_share,
                 "zeroed_sampled_ai_share": zero_target_share,
                 "share_change": zero_delta},
        "push": {"unclamped_share": push_unclamped, "k1_share": push_k1,
                 "k2_share": push_k2, "k1_share_change": push_delta_k1,
                 "k2_share_change": push_delta_k2, "random_control_shares": control_shares},
        "screen": screen_overview,
        "predictions": predictions,
    }
    _write_json_atomic(out, args.out / "score.json")
    return out


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=CHAT_MODEL)
    parser.add_argument("--revision", default=CHAT_REVISION)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--max-gpu-gb", type=float, default=14.5)
    parser.add_argument("--out", type=Path, default=Path("results/part39"))
    parser.add_argument("--seed", type=int, default=3901)
    parser.add_argument("--max-new-tokens", type=int, default=60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    reads = commands.add_parser("reads", help="measure target and seeded controls on the battery")
    _add_common_args(reads)
    reads.add_argument("--batch-size", type=int, default=4)
    reads.add_argument("--battery", type=Path, default=BATTERY_PATH)

    screen = commands.add_parser("screen", help="screen every down_proj unit using the reads arrays")
    _add_common_args(screen)
    screen.add_argument("--battery", type=Path, default=BATTERY_PATH)

    zero = commands.add_parser("zero", help="sample six identity questions with and without zeroing")
    _add_common_args(zero)
    zero.add_argument("--samples", type=int, default=8)

    push = commands.add_parser("push", help="clamp the target under Sam and sample controls")
    _add_common_args(push)
    push.add_argument("--samples", type=int, default=8)
    push.add_argument("--persona", type=Path, default=PERSONA_PATH)

    score = commands.add_parser("score", help="score reads, zero, and push outputs")
    _add_common_args(score)
    score.add_argument("--samples", type=int, default=8)
    score.add_argument("--battery", type=Path, default=BATTERY_PATH)
    score.add_argument("--persona", type=Path, default=PERSONA_PATH)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "reads":
        if args.batch_size < 1:
            raise SystemExit("--batch-size must be positive")
        run_reads(args)
    elif args.command == "screen":
        if args.max_gpu_gb <= 0:
            raise SystemExit("--max-gpu-gb must be positive")
        run_screen(args)
    elif args.command == "zero":
        if args.samples < 1 or args.max_new_tokens < 1:
            raise SystemExit("--samples and --max-new-tokens must be positive")
        run_zero(args)
    elif args.command == "push":
        if args.samples < 1 or args.max_new_tokens < 1:
            raise SystemExit("--samples and --max-new-tokens must be positive")
        run_push(args)
    elif args.command == "score":
        if args.samples < 1 or args.max_new_tokens < 1:
            raise SystemExit("--samples and --max-new-tokens must be positive")
        score_results(args)


if __name__ == "__main__":
    main()
