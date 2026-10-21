#!/usr/bin/env python3
"""Part 42: compare logit-lens and Jacobian-lens readouts of MLP writes.

This tool reads only the selected MLP down-projection tensors, final norm,
unembedding tensor, and the pre-fitted Jacobian lens. It never loads a model
or runs a forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parent
CONCEPTS_PATH = ROOT / "data" / "part35-concepts.json"
RESULTS_ROOT = ROOT / "results" / "part42"
SCREEN_DEFAULT = Path(
    "/mnt/c/Users/alexa/neuron-probe-gpu/results/part35/"
    "qwen-qwen3-1-7b/screen-floor-median.jsonl"
)
PART39_UNITS_DEFAULT = Path(
    "/home/elisepc/assistant-basin/results/part39/screen-auc-chat.json"
)
SEED = 4201
RANDOM_UNIT_COUNT = 20
READOUT_COUNT = 30
LM_HEAD_CHUNK_ROWS = 16384

MODEL_SLUGS = {
    "Qwen/Qwen3-1.7B": "qwen3-1.7b",
    "Qwen/Qwen3.5-4B": "qwen3.5-4b",
}
MODEL_CACHE_NAMES = {
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen/Qwen3.5-4B": "Qwen3.5-4B",
}
MODEL_LENS_PATHS = {
    "Qwen/Qwen3-1.7B": Path(
        "/home/elisepc/neuron-probe/jlens/qwen3-1.7b/jlens/"
        "Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"
    ),
    "Qwen/Qwen3.5-4B": Path(
        "/home/elisepc/neuron-probe/jlens/qwen3.5-4b/jlens/"
        "Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt"
    ),
}
MODEL_DEFAULT_SNAPSHOT_ROOT = Path("/mnt/e/neuron-probe/hf/hub")
LENS_CHECKSUMS_DEFAULT = Path("/home/elisepc/neuron-probe/jlens/SHA256SUMS")
FOUR_B_TARGETS = (
    (6, 753, "L6 N753", "ai_assistant"),
    (7, 1169, "L7 N1169", "china"),
    (11, 8868, "L11 N8868", "china"),
    (29, 776, "L29 N776", "google"),
    (22, 2742, "L22 N2742", "alien"),
)
FOUR_B_UNIT_LIST = tuple((layer, unit) for layer, unit, _, _ in FOUR_B_TARGETS)
MEANING_WORDS = {
    "wizard": ["spell", "magic", "wand", "sorcer", "witch", "potion", "enchant"],
    "L6 N753": [
        "ai", "assistant", "model", "chatbot", "人工智能", "助手", "claude", "gpt"
    ],
}
PART25_WORDS = {
    "L7 N1169": ["china", "chinese", "中国", "beijing"],
    "L11 N8868": ["china", "chinese", "中国", "beijing"],
    "L29 N776": ["google"],
}
LAST_QUARTER_FIRST_LAYER = {
    "Qwen/Qwen3-1.7B": 21,
    "Qwen/Qwen3.5-4B": 24,
}


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(row)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(destination)


def pick_rank1_units(
    screen_rows: Iterable[Mapping[str, Any]],
    concept_ids: Iterable[str],
    model_id: str = "Qwen/Qwen3-1.7B",
) -> dict[str, dict[str, int]]:
    """Match Part 40's screen/rank-1 picker and validate its requested IDs."""
    requested = [str(value) for value in concept_ids]
    wanted = set(requested)
    selected: dict[str, dict[str, int]] = {}
    for row in screen_rows:
        if row.get("stage") != "screen" or int(row.get("rank", 999999)) != 1:
            continue
        row_model = row.get("model_id")
        if row_model is not None and row_model != model_id:
            continue
        concept_id = str(row.get("concept_id", ""))
        if concept_id not in wanted:
            continue
        if concept_id in selected:
            raise ValueError(f"screen has more than one rank-1 row for concept {concept_id!r}")
        if "layer" not in row or "unit" not in row:
            raise ValueError(f"rank-1 screen row for {concept_id!r} is missing layer or unit")
        selected[concept_id] = {"layer": int(row["layer"]), "unit": int(row["unit"])}
    missing = [concept_id for concept_id in requested if concept_id not in selected]
    if missing:
        raise ValueError("screen is missing rank-1 units for: " + ", ".join(missing))
    return {concept_id: selected[concept_id] for concept_id in requested}


def targets_for_model(
    model: str,
    screen_path: str | Path = SCREEN_DEFAULT,
    concepts_path: str | Path = CONCEPTS_PATH,
) -> list[dict[str, Any]]:
    if model == "Qwen/Qwen3.5-4B":
        return [
            {
                "id": label.replace(" ", "-"),
                "label": label,
                "concept": concept,
                "role": "target",
                "layer": layer,
                "unit": unit,
            }
            for layer, unit, label, concept in FOUR_B_TARGETS
        ]
    if model != "Qwen/Qwen3-1.7B":
        raise ValueError(f"unsupported model: {model}")
    concepts = read_json(concepts_path).get("concepts", [])
    concept_ids = [str(row["id"]) for row in concepts]
    if len(concept_ids) != 43:
        raise ValueError(f"expected 43 Part 35 concepts, found {len(concept_ids)}")
    selected = pick_rank1_units(read_jsonl(screen_path), concept_ids, model)
    targets = [
        {
            "id": concept_id,
            "label": concept_id,
            "concept": concept_id,
            "role": "part35_rank1",
            **selected[concept_id],
        }
        for concept_id in concept_ids
    ]
    targets.append(
        {
            "id": "wizard-L19-N4826",
            "label": "wizard",
            "concept": "wizard",
            "role": "wizard",
            "layer": 19,
            "unit": 4826,
        }
    )
    return targets


def choose_random_units(
    targets: Iterable[Mapping[str, Any]],
    layer_widths: Mapping[int, int],
    count: int = RANDOM_UNIT_COUNT,
    seed: int = SEED,
    excluded_coordinates: Iterable[tuple[int, int]] = (),
) -> list[dict[str, Any]]:
    """Choose unique coordinates from target layers, equally sampling layers."""
    target_rows = list(targets)
    layers = sorted({int(row["layer"]) for row in target_rows})
    if not layers:
        raise ValueError("cannot choose random units without target layers")
    occupied = {
        (int(row["layer"]), int(row["unit"]))
        for row in target_rows
    }
    occupied.update((int(layer), int(unit)) for layer, unit in excluded_coordinates)
    selected: set[tuple[int, int]] = set()
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    for index in range(count):
        available_layers = [
            layer for layer in layers
            if int(layer_widths[layer]) > 0
            and any((layer, unit) not in occupied and (layer, unit) not in selected
                    for unit in range(int(layer_widths[layer])))
        ]
        if not available_layers:
            raise ValueError("not enough non-target coordinates for random units")
        layer = rng.choice(available_layers)
        width = int(layer_widths[layer])
        while True:
            unit = rng.randrange(width)
            coordinate = (layer, unit)
            if coordinate not in occupied and coordinate not in selected:
                selected.add(coordinate)
                break
        output.append(
            {
                "id": f"random-{index + 1:02d}",
                "label": f"random L{layer} N{unit}",
                "concept": None,
                "role": "random",
                "layer": layer,
                "unit": unit,
                "seed": seed,
            }
        )
    return output


def load_part39_units(path: str | Path = PART39_UNITS_DEFAULT) -> list[dict[str, Any]]:
    rows = read_json(Path(path).expanduser()).get("passing_units")
    if not isinstance(rows, list) or len(rows) != 211:
        found = len(rows) if isinstance(rows, list) else type(rows).__name__
        raise ValueError(f"Part 39 source must contain 211 passing_units, found {found}")
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or "layer" not in row or "unit" not in row:
            raise ValueError(f"Part 39 passing_units row {index} needs layer and unit")
        layer, unit = int(row["layer"]), int(row["unit"])
        coordinate = (layer, unit)
        if coordinate in seen:
            raise ValueError(f"duplicate Part 39 passing unit L{layer} N{unit}")
        seen.add(coordinate)
        output.append(
            {
                "id": f"part39-{index:03d}-L{layer}-N{unit}",
                "label": f"Part 39 L{layer} N{unit}",
                "concept": None,
                "role": "part39_passing",
                "layer": layer,
                "unit": unit,
            }
        )
    return output


def resolve_snapshot(model: str, override: str | Path | None = None) -> Path:
    if override is not None:
        snapshot = Path(override).expanduser()
        if not (snapshot / "config.json").is_file():
            raise FileNotFoundError(f"snapshot has no config.json: {snapshot}")
        return snapshot
    cache_name = MODEL_CACHE_NAMES[model]
    snapshots = MODEL_DEFAULT_SNAPSHOT_ROOT / f"models--Qwen--{cache_name}" / "snapshots"
    candidates = sorted(path for path in snapshots.iterdir() if path.is_dir()) if snapshots.is_dir() else []
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one cached snapshot under {snapshots}, found {len(candidates)}; "
            "pass --snapshot explicitly"
        )
    if not (candidates[0] / "config.json").is_file():
        raise FileNotFoundError(f"snapshot has no config.json: {candidates[0]}")
    return candidates[0]


def config_dimensions(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], int, int, str]:
    """Return text config, hidden width, layer count, and architecture model_type."""
    text_config = config.get("text_config", config)
    if not isinstance(text_config, Mapping):
        raise ValueError("config.json text_config must be an object")
    hidden_size = int(text_config["hidden_size"])
    num_layers = int(text_config["num_hidden_layers"])
    model_type = str(text_config.get("model_type", config.get("model_type", "")))
    if hidden_size <= 0 or num_layers <= 0 or not model_type:
        raise ValueError("config.json needs positive hidden_size/layer count and model_type")
    return text_config, hidden_size, num_layers, model_type


def load_weight_map(snapshot: str | Path) -> dict[str, str]:
    """Read safetensors metadata only, without loading checkpoint tensors."""
    from safetensors import safe_open

    snapshot_path = Path(snapshot)
    index_path = snapshot_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid safetensors index: {index_path}")
        return {str(key): str(value) for key, value in weight_map.items()}
    shards = sorted(snapshot_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found under {snapshot_path}")
    weight_map: dict[str, str] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise ValueError(f"duplicate tensor key across safetensors shards: {key}")
                weight_map[key] = shard.name
    return weight_map


def find_language_model_prefix(
    weight_map_or_index: Mapping[str, Any],
    require_language_model: bool = False,
) -> str:
    """Find the prefix before layers.*.mlp.down_proj.weight from index keys."""
    weight_map = weight_map_or_index.get("weight_map", weight_map_or_index)
    if not isinstance(weight_map, Mapping):
        raise ValueError("weight index must be a mapping")
    prefixes: set[str] = set()
    for key in weight_map:
        key_text = str(key)
        match = re.fullmatch(r"(.*?)(?:layers)\.\d+\.mlp\.down_proj\.weight", key_text)
        if match:
            prefixes.add(match.group(1))
    if require_language_model:
        prefixes = {prefix for prefix in prefixes if "language_model." in prefix}
    if len(prefixes) != 1:
        requirement = " language-model" if require_language_model else ""
        raise ValueError(
            f"expected one{requirement} MLP down-projection prefix in safetensors index, "
            f"found {sorted(prefixes)}"
        )
    return next(iter(prefixes))


def _shard_path(snapshot: str | Path, weight_map: Mapping[str, str], key: str) -> Path:
    try:
        shard = weight_map[key]
    except KeyError as exc:
        raise KeyError(f"checkpoint is missing required tensor {key!r}") from exc
    path = Path(shard)
    return path if path.is_absolute() else Path(snapshot) / path


def safetensor_shape(
    snapshot: str | Path,
    weight_map: Mapping[str, str],
    key: str,
) -> tuple[int, ...]:
    from safetensors import safe_open

    path = _shard_path(snapshot, weight_map, key)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return tuple(int(value) for value in handle.get_slice(key).get_shape())


def load_safetensor(
    snapshot: str | Path,
    weight_map: Mapping[str, str],
    key: str,
) -> Any:
    from safetensors import safe_open

    path = _shard_path(snapshot, weight_map, key)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def select_unembedding_key(
    weight_map: Mapping[str, str],
    prefix: str,
    tied_embeddings: bool,
) -> str:
    keys = set(weight_map)
    if tied_embeddings:
        candidates = [
            key for key in keys
            if key.endswith("embed_tokens.weight")
            and (key.startswith(prefix) or ".language_model." in key)
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected one tied embedding tensor, found {sorted(candidates)}")
        return candidates[0]
    if "lm_head.weight" in keys:
        return "lm_head.weight"
    candidates = [key for key in keys if key.endswith(".lm_head.weight")]
    if len(candidates) != 1:
        raise ValueError(f"expected one LM head tensor, found {sorted(candidates)}")
    return candidates[0]


def effective_norm_gain(norm_weight: Any, model_type: str) -> Any:
    """Use the installed Qwen RMSNorm convention for the selected architecture."""
    if model_type == "qwen3":
        return norm_weight
    if model_type in {"qwen3_5", "qwen3.5", "qwen3_5_text"}:
        return 1.0 + norm_weight
    raise ValueError(f"unsupported final RMSNorm convention for model_type={model_type!r}")


def load_jacobian_lens(path: str | Path) -> tuple[dict[int, Any], Any, int]:
    import torch

    lens = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(lens, Mapping):
        raise ValueError("Jacobian lens file must contain a dictionary")
    if lens.get("d_model") is None:
        raise ValueError("Jacobian lens is missing its d_model field")
    if lens.get("n_prompts") is None:
        raise ValueError("Jacobian lens is missing its n_prompts field")
    if lens.get("source_layers") is None:
        raise ValueError("Jacobian lens is missing its source_layers field")
    declared_d_model = int(lens["d_model"])
    n_prompts = int(lens["n_prompts"])
    if n_prompts <= 0:
        raise ValueError(f"Jacobian lens n_prompts must be positive, found {n_prompts}")
    source_layers = sorted({int(layer) for layer in lens["source_layers"]})
    jacobians = lens.get("J")
    if not isinstance(jacobians, Mapping):
        raise ValueError("Jacobian lens must contain J: {layer: tensor}")
    normalized: dict[int, Any] = {}
    for layer, tensor in jacobians.items():
        normalized[int(layer)] = tensor
    if sorted(normalized) != source_layers:
        raise ValueError(
            "Jacobian lens source_layers do not match its J keys: "
            f"source_layers={source_layers}, J={sorted(normalized)}"
        )
    metadata = {
        "n_prompts": n_prompts,
        "source_layers": source_layers,
        "d_model": declared_d_model,
    }
    return normalized, metadata, declared_d_model


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lens_provenance(
    lens_path: str | Path,
    lens_metadata: Mapping[str, Any],
    checksums_path: str | Path = LENS_CHECKSUMS_DEFAULT,
) -> dict[str, Any]:
    """Verify the downloaded lens and preserve its file and fit provenance."""
    lens_file = Path(lens_path).expanduser().resolve()
    checksums_file = Path(checksums_path).expanduser().resolve()
    if not checksums_file.is_file():
        raise FileNotFoundError(f"Jacobian lens checksum manifest is missing: {checksums_file}")
    if not lens_file.is_file():
        raise FileNotFoundError(f"Jacobian lens file is missing: {lens_file}")

    try:
        relative_lens = lens_file.relative_to(checksums_file.parent)
        checksum_entry = relative_lens.as_posix()
    except ValueError:
        raise ValueError(
            f"lens {lens_file} is outside checksum-manifest root {checksums_file.parent}"
        ) from None
    expected_sha256: str | None = None
    for line_number, line in enumerate(checksums_file.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.strip().split(maxsplit=1)
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"invalid checksum line {line_number} in {checksums_file}")
        entry_name = fields[1].lstrip("*")
        if entry_name == checksum_entry:
            expected_sha256 = fields[0].lower()
            break
    if expected_sha256 is None:
        raise ValueError(f"lens {checksum_entry!r} is not listed in {checksums_file}")
    actual_sha256 = _sha256_file(lens_file)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Jacobian lens checksum mismatch for {lens_file}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    config_file = lens_file.parent / "config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"Jacobian lens fit config is missing: {config_file}")
    config_text = config_file.read_text(encoding="utf-8")
    return {
        "lens_file": str(lens_file),
        "lens_sha256": actual_sha256,
        "checksum_manifest": str(checksums_file),
        "checksum_manifest_sha256": _sha256_file(checksums_file),
        "checksum_entry": checksum_entry,
        "expected_sha256": expected_sha256,
        "checksum_verified": True,
        "n_prompts": int(lens_metadata["n_prompts"]),
        "source_layers": [int(layer) for layer in lens_metadata["source_layers"]],
        "d_model": int(lens_metadata["d_model"]),
        "config_yaml_file": str(config_file),
        "config_yaml_sha256": _sha256_file(config_file),
        "config_yaml": config_text,
    }


def jacobian_layer_for_write(layer: int, num_layers: int) -> int | None:
    """Map a block-output MLP write to its output-stream J; final output is identity."""
    layer = int(layer)
    num_layers = int(num_layers)
    if num_layers <= 0 or not 0 <= layer < num_layers:
        raise ValueError(f"layer {layer} is outside model range 0..{num_layers - 1}")
    return None if layer == num_layers - 1 else layer


def readout_logits(
    torch: Any,
    lm_head: Any,
    write: Any,
    norm_gain: Any,
    jacobian: Any | None = None,
) -> Any:
    vector = write.float()
    if jacobian is not None:
        vector = torch.mv(jacobian.float(), vector)
    vector = norm_gain.float() * vector
    logits = torch.empty(int(lm_head.shape[0]), dtype=torch.float32, device=vector.device)
    for start in range(0, int(lm_head.shape[0]), LM_HEAD_CHUNK_ROWS):
        end = min(start + LM_HEAD_CHUNK_ROWS, int(lm_head.shape[0]))
        # Keep the checkpoint's head in its stored dtype and only upcast this
        # bounded slice. The result and all reduction math remain float32.
        head_chunk = lm_head[start:end].to(dtype=torch.float32)
        logits[start:end] = torch.mv(head_chunk, vector)
    return logits


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _top_bottom(torch: Any, logits: Any, tokenizer: Any) -> dict[str, list[dict[str, Any]]]:
    count = min(READOUT_COUNT, int(logits.numel()))
    if count == 0:
        raise ValueError("readout returned an empty vocabulary vector")
    top_values, top_ids = torch.topk(logits, count)
    bottom_values, bottom_ids = torch.topk(logits, count, largest=False)

    def rows(values: Any, ids: Any) -> list[dict[str, Any]]:
        return [
            {
                "token_id": int(token_id),
                "decoded": _decode_token(tokenizer, int(token_id)),
                "logit": float(value),
            }
            for value, token_id in zip(values.tolist(), ids.tolist())
        ]

    return {"top": rows(top_values, top_ids), "bottom": rows(bottom_values, bottom_ids)}


def _load_tokenizer(snapshot: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )


def _layer_tensor_key(prefix: str, layer: int) -> str:
    return f"{prefix}layers.{layer}.mlp.down_proj.weight"


def _validate_lens_layers(
    torch: Any,
    jacobians: Mapping[int, Any],
    layers: Iterable[int],
    hidden_size: int,
) -> None:
    missing = sorted(set(int(layer) for layer in layers) - set(jacobians))
    if missing:
        raise ValueError(f"Jacobian lens J is missing target layers: {missing}")
    for layer in sorted(set(int(value) for value in layers)):
        matrix = jacobians[layer]
        if not torch.is_tensor(matrix) or tuple(matrix.shape) != (hidden_size, hidden_size):
            shape = tuple(matrix.shape) if hasattr(matrix, "shape") else type(matrix).__name__
            raise ValueError(
                f"J[{layer}] has shape {shape}, expected ({hidden_size}, {hidden_size})"
            )


def run_read(args: argparse.Namespace) -> Path:
    import torch

    model = args.model
    snapshot = resolve_snapshot(model, args.snapshot)
    config = read_json(snapshot / "config.json")
    text_config, hidden_size, num_layers, model_type = config_dimensions(config)
    multimodal = model == "Qwen/Qwen3.5-4B"
    weight_map = load_weight_map(snapshot)
    prefix = find_language_model_prefix(weight_map, require_language_model=multimodal)
    targets = targets_for_model(model, args.screen)
    target_layers = {int(unit["layer"]) for unit in targets}
    part39_units = (
        load_part39_units(args.part39_units)
        if model == "Qwen/Qwen3.5-4B"
        else []
    )
    for unit in part39_units:
        if not 0 <= int(unit["layer"]) < num_layers:
            raise ValueError(
                f"Part 39 unit {unit['id']} layer {unit['layer']} is outside 0..{num_layers - 1}"
            )
    for unit in targets:
        if not 0 <= int(unit["layer"]) < num_layers:
            raise ValueError(f"unit {unit['id']} layer {unit['layer']} is outside 0..{num_layers - 1}")

    layer_widths: dict[int, int] = {}
    needed_layers = target_layers | {int(unit["layer"]) for unit in part39_units}
    for layer in sorted(needed_layers):
        down_key = _layer_tensor_key(prefix, layer)
        shape = safetensor_shape(snapshot, weight_map, down_key)
        if len(shape) != 2 or shape[0] != hidden_size:
            raise ValueError(
                f"{down_key} has shape {shape}, expected [{hidden_size}, intermediate_size]"
            )
        layer_widths[layer] = int(shape[1])
    for unit in targets:
        if not 0 <= int(unit["unit"]) < layer_widths[int(unit["layer"])]:
            raise ValueError(f"unit {unit['id']} index {unit['unit']} is outside its layer width")
    random_units = choose_random_units(
        targets,
        layer_widths,
        excluded_coordinates=(
            (int(unit["layer"]), int(unit["unit"])) for unit in part39_units
        ),
    )
    units = targets + random_units + part39_units
    unit_layers = {int(unit["layer"]) for unit in units}
    j_layers = {
        mapped_layer
        for layer in unit_layers
        if (mapped_layer := jacobian_layer_for_write(layer, num_layers)) is not None
    }

    lens_path = Path(args.lens).expanduser() if args.lens else MODEL_LENS_PATHS[model]
    jacobians, lens_metadata, lens_d_model = load_jacobian_lens(lens_path)
    provenance = build_lens_provenance(
        lens_path, lens_metadata, getattr(args, "lens_checksums", LENS_CHECKSUMS_DEFAULT)
    )
    print("Jacobian lens provenance:", json.dumps(provenance, ensure_ascii=False, default=str))
    if lens_d_model != hidden_size:
        raise ValueError(
            f"Jacobian lens d_model={lens_d_model}, model config hidden_size={hidden_size}"
        )
    _validate_lens_layers(torch, jacobians, j_layers, hidden_size)

    norm_key = f"{prefix}norm.weight"
    norm_weight = load_safetensor(snapshot, weight_map, norm_key).float()
    if tuple(norm_weight.shape) != (hidden_size,):
        raise ValueError(f"{norm_key} has shape {tuple(norm_weight.shape)}, expected ({hidden_size},)")
    norm_gain = effective_norm_gain(norm_weight, model_type)

    tied = bool(text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    unembedding_key = select_unembedding_key(weight_map, prefix, tied)
    lm_head = load_safetensor(snapshot, weight_map, unembedding_key)
    if lm_head.ndim != 2 or int(lm_head.shape[1]) != hidden_size:
        raise ValueError(
            f"{unembedding_key} has shape {tuple(lm_head.shape)}, expected [vocab, {hidden_size}]"
        )
    if int(lm_head.shape[0]) < READOUT_COUNT:
        raise ValueError(f"vocabulary has fewer than {READOUT_COUNT} tokens")

    tokenizer = _load_tokenizer(snapshot)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for unit in units:
        grouped.setdefault(int(unit["layer"]), []).append(unit)

    read_units: list[dict[str, Any]] = []
    for layer in sorted(grouped):
        down_key = _layer_tensor_key(prefix, layer)
        down_weight = load_safetensor(snapshot, weight_map, down_key)
        if tuple(down_weight.shape) != (hidden_size, layer_widths[layer]):
            raise ValueError(f"{down_key} changed shape after metadata inspection")
        j_layer = jacobian_layer_for_write(layer, num_layers)
        j_matrix = jacobians[j_layer].float() if j_layer is not None else None
        for unit in grouped[layer]:
            write = down_weight[:, int(unit["unit"])].float()
            logit_logits = readout_logits(torch, lm_head, write, norm_gain)
            jacobian_logits = (
                logit_logits
                if j_matrix is None
                else readout_logits(torch, lm_head, write, norm_gain, jacobian=j_matrix)
            )
            row = {
                **unit,
                "j_layer": j_layer,
                "j_identity": j_matrix is None,
                "readouts": {
                    "logit": _top_bottom(torch, logit_logits, tokenizer),
                    "jacobian": _top_bottom(torch, jacobian_logits, tokenizer),
                },
            }
            read_units.append(row)
            del write, logit_logits, jacobian_logits
        del down_weight, j_matrix

    document = {
        "schema_version": 1,
        "model": model,
        "model_slug": MODEL_SLUGS[model],
        "snapshot": str(snapshot),
        "lens": str(lens_path),
        "lens_provenance": provenance,
        "jacobian_alignment": (
            "ActivationRecorder captures block outputs; an MLP down_proj write in block l "
            "uses J[l]. The final block has no fitted J and uses identity."
        ),
        "lens_layers": sorted(jacobians),
        "layer_prefix": prefix,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "model_type": model_type,
        "norm_gain_rule": (
            "stored weight directly" if model_type == "qwen3"
            else "(1 + stored weight)"
        ),
        "unembedding_tensor": unembedding_key,
        "random_seed": SEED,
        "random_unit_count": len(random_units),
        "part39_passing_count": len(part39_units),
        "units": read_units,
    }
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else RESULTS_ROOT / MODEL_SLUGS[model] / "read.json"
    )
    write_json(output_path, document)
    print(f"Wrote {output_path}")
    return output_path


def normalize_match_text(value: str) -> str:
    # Keep letters and digits only (CJK included): " _magic", "/google" and ".Google" all read as the word.
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def concept_word_matches(token: str, word: str, minimum_characters: int = 4) -> bool:
    """A token names a word when it IS the word, or when one is a prefix of the other.

    Prefix in both directions keeps inflections ("spells", "wizards") and word pieces (" chines")
    while refusing a word buried inside another: the first build matched by substring and so read
    "Application" as naming "cat", "And" as "candle" and "knowledge" as "owl" (Claude, 2026-10-20).
    A partial match needs `minimum_characters` on its shorter side; exact matches of two or more
    characters always count, so "AI" and "中国" still match.
    """
    token_text = normalize_match_text(token)
    word_text = normalize_match_text(word)
    if len(token_text) >= 2 and token_text == word_text:
        return True
    if min(len(token_text), len(word_text)) < minimum_characters:
        return False
    return token_text.startswith(word_text) or word_text.startswith(token_text)


def matched_words(tokens: Iterable[Mapping[str, Any]], words: Iterable[str]) -> list[str]:
    decoded = [str(token.get("decoded", "")) for token in tokens]
    return [
        word for word in words
        if any(concept_word_matches(token, word) for token in decoded)
    ]


def concept_matches(tokens: Iterable[Mapping[str, Any]], concepts: Iterable[str]) -> list[str]:
    decoded = [str(token.get("decoded", "")) for token in tokens]
    return [
        concept for concept in concepts
        if any(concept_word_matches(token, concept) for token in decoded)
    ]


def _top_tokens(unit: Mapping[str, Any], readout: str) -> list[dict[str, Any]]:
    return list(unit["readouts"][readout]["top"])


def _unit_by_id(document: Mapping[str, Any], unit_id: str) -> dict[str, Any] | None:
    return next((unit for unit in document.get("units", []) if unit.get("id") == unit_id), None)


def _overlap_count(unit: Mapping[str, Any]) -> int:
    logit_ids = {int(token["token_id"]) for token in _top_tokens(unit, "logit")}
    jacobian_ids = {int(token["token_id"]) for token in _top_tokens(unit, "jacobian")}
    return len(logit_ids & jacobian_ids)


def _late_overlap_summary(document: Mapping[str, Any], threshold: int) -> dict[str, Any]:
    units = [
        unit for unit in document.get("units", [])
        if int(unit["layer"]) >= threshold
        and unit.get("role") != "part39_passing"
        and not unit.get("j_identity", False)
    ]
    per_unit = {str(unit["id"]): _overlap_count(unit) for unit in units}
    by_role: dict[str, list[int]] = {}
    for unit in units:
        by_role.setdefault(str(unit.get("role", "unknown")), []).append(per_unit[str(unit["id"])])
    return {
        "first_layer": threshold,
        "unit_count": len(units),
        "median_overlap": median(per_unit.values()) if per_unit else None,
        "per_unit": per_unit,
        "by_role_median": {
            role: median(values) for role, values in sorted(by_role.items())
        },
    }


def score_documents(
    documents: Mapping[str, Mapping[str, Any]],
    part35_concepts: Iterable[str],
) -> dict[str, Any]:
    concept_list = [str(value) for value in part35_concepts]
    scores: dict[str, Any] = {}
    unit_overlaps = {
        model: {
            str(unit["id"]): _overlap_count(unit)
            for unit in document.get("units", [])
        }
        for model, document in documents.items()
    }

    one_seven = documents.get("Qwen/Qwen3-1.7B")
    four_b = documents.get("Qwen/Qwen3.5-4B")

    early: dict[str, Any] = {"first_layer": 0, "last_layer": 6}
    if one_seven is not None:
        eligible = [
            unit for unit in one_seven.get("units", [])
            if unit.get("role") == "part35_rank1"
            and 0 <= int(unit["layer"]) <= 6
            and not unit.get("j_identity", False)
        ]
        per_readout: dict[str, list[str]] = {}
        counts: dict[str, int] = {}
        for readout in ("logit", "jacobian"):
            named = [
                unit for unit in eligible
                if concept_matches(_top_tokens(unit, readout), [str(unit["concept"])])
            ]
            per_readout[readout] = [str(unit["id"]) for unit in named]
            counts[readout] = len(named)
        early.update(
            unit_count=len(eligible),
            named_unit_ids=per_readout,
            named_unit_counts=counts,
            holds=counts["jacobian"] >= 2 * counts["logit"],
            status=(
                "supported" if counts["jacobian"] >= 2 * counts["logit"]
                else "not_supported"
            ),
        )
    else:
        early.update(unit_count=0, status="not_run", holds=None)
    scores["42.1"] = early

    late_by_model: dict[str, Any] = {}
    for model, document in documents.items():
        threshold = LAST_QUARTER_FIRST_LAYER[model]
        late_by_model[model] = _late_overlap_summary(document, threshold)
    complete_models = set(late_by_model) >= set(MODEL_SLUGS)
    late_holds = (
        all(
            late_by_model[model]["median_overlap"] is not None
            and late_by_model[model]["median_overlap"] >= 10
            for model in MODEL_SLUGS
        )
        if complete_models else None
    )
    scores["42.2"] = {
        "minimum_median_overlap": 10,
        "per_model": late_by_model,
        "holds": late_holds,
        "status": (
            "supported" if late_holds is True
            else "not_supported" if late_holds is False
            else "not_run"
        ),
    }

    meaning: dict[str, Any] = {}
    if one_seven is not None:
        wizard = _unit_by_id(one_seven, "wizard-L19-N4826")
        if wizard is not None:
            words = MEANING_WORDS["wizard"]
            found = matched_words(_top_tokens(wizard, "jacobian"), words)
            meaning["wizard"] = {
                "unit_id": wizard["id"],
                "words": words,
                "matched_words": found,
                "holds": len(found) >= 2,
            }
    if four_b is not None:
        unit = _unit_by_id(four_b, "L6-N753")
        if unit is not None:
            words = MEANING_WORDS["L6 N753"]
            found = matched_words(_top_tokens(unit, "jacobian"), words)
            meaning["L6 N753"] = {
                "unit_id": unit["id"],
                "words": words,
                "matched_words": found,
                "holds": len(found) >= 2,
            }
    meaning_complete = set(meaning) == {"wizard", "L6 N753"}
    meaning_holds = (
        all(item["holds"] for item in meaning.values()) if meaning_complete else None
    )
    scores["42.3"] = {
        "minimum_meaning_words": 2,
        "units": meaning,
        "holds": meaning_holds,
        "status": (
            "supported" if meaning_holds is True
            else "not_supported" if meaning_holds is False
            else "not_run"
        ),
    }

    part25_results: dict[str, Any] = {}
    if four_b is not None:
        for unit_id, words in PART25_WORDS.items():
            unit = _unit_by_id(four_b, unit_id.replace(" ", "-"))
            if unit is None:
                continue
            all_tokens = (
                _top_tokens(unit, "jacobian")
                + list(unit["readouts"]["jacobian"]["bottom"])
            )
            found = matched_words(all_tokens, words)
            part25_results[unit_id] = {
                "unit_id": unit["id"],
                "words": words,
                "matched_words": found,
                "holds": bool(found),
            }
    part25_complete = set(part25_results) == set(PART25_WORDS)
    part25_holds = (
        all(item["holds"] for item in part25_results.values())
        if part25_complete else None
    )
    scores["42.4"] = {
        "units": part25_results,
        "holds": part25_holds,
        "status": (
            "supported" if part25_holds is True
            else "not_supported" if part25_holds is False
            else "not_run"
        ),
    }

    random_results: dict[str, Any] = {}
    if one_seven is not None:
        random_units = [
            unit for unit in one_seven.get("units", [])
            if unit.get("role") == "random"
        ]
        per_readout: dict[str, Any] = {}
        for readout in ("logit", "jacobian"):
            by_unit = {
                str(unit["id"]): concept_matches(_top_tokens(unit, readout), concept_list)
                for unit in random_units
            }
            matching_units = [unit_id for unit_id, matches in by_unit.items() if matches]
            concept_counts = {
                concept: sum(concept in matches for matches in by_unit.values())
                for concept in concept_list
            }
            per_readout[readout] = {
                "units_naming_any_concept": matching_units,
                "unit_count_naming_any_concept": len(matching_units),
                "per_unit_concepts": by_unit,
                "concept_unit_counts": concept_counts,
            }
        random_complete = len(random_units) == RANDOM_UNIT_COUNT
        random_holds = (
            all(
                all(
                    count <= 1
                    for count in per_readout[readout]["concept_unit_counts"].values()
                )
                for readout in ("logit", "jacobian")
            )
            if random_complete else None
        )
        random_results = {
            "random_unit_count": len(random_units),
            "required_random_unit_count": RANDOM_UNIT_COUNT,
            "concept_count": len(concept_list),
            "per_readout": per_readout,
            "holds": random_holds,
            "status": (
                "supported" if random_holds is True
                else "not_supported" if random_holds is False
                else "not_run"
            ),
        }
    else:
        random_results = {"status": "not_run", "holds": None}
    scores["42.5"] = random_results

    part39_results: dict[str, Any] = {"status": "not_run", "holds": None}
    if four_b is not None:
        part39_units = [
            unit for unit in four_b.get("units", [])
            if unit.get("role") == "part39_passing"
        ]
        excluding = (6, 753)
        evaluated = [
            unit for unit in part39_units
            if (int(unit["layer"]), int(unit["unit"])) != excluding
        ]
        hits = {
            str(unit["id"]): matched_words(
                _top_tokens(unit, "logit"), MEANING_WORDS["L6 N753"]
            )
            for unit in evaluated
        }
        qualifying = {unit_id: words for unit_id, words in hits.items() if words}
        p39_logit_result = {
            "units_with_meaning_words": qualifying,
            "unit_count_with_meaning_words": len(qualifying),
            "any_match": bool(qualifying),
        }
        if part39_units:
            any_match = p39_logit_result["any_match"]
            part39_results = {
                "source_unit_count": len(part39_units),
                "evaluated_unit_count": len(evaluated),
                "passing_units": [
                    {
                        "id": str(unit["id"]),
                        "layer": int(unit["layer"]),
                        "unit": int(unit["unit"]),
                    }
                    for unit in part39_units
                ],
                "excluded_coordinate": {"layer": excluding[0], "unit": excluding[1]},
                "meaning_words": MEANING_WORDS["L6 N753"],
                "readout": "logit",
                "logit_lens": p39_logit_result,
                "holds": any_match,
                "status": "supported" if any_match else "not_supported",
            }
        else:
            part39_results = {"source_unit_count": 0, "status": "not_run", "holds": None}

    return {
        "schema_version": 1,
        "predictions": scores,
        "unit_top30_overlap_counts": unit_overlaps,
        "p39_9": part39_results,
    }


def run_score(args: argparse.Namespace) -> Path:
    model = args.model
    documents: dict[str, Mapping[str, Any]] = {}
    for model_id, slug in MODEL_SLUGS.items():
        default_path = RESULTS_ROOT / slug / "read.json"
        read_path = Path(args.read).expanduser() if args.read and model_id == model else default_path
        if read_path.is_file():
            document = read_json(read_path)
            if document.get("model") != model_id:
                raise ValueError(f"{read_path} contains {document.get('model')!r}, expected {model_id!r}")
            documents[model_id] = document
    if model not in documents:
        raise FileNotFoundError(
            f"no read.json for {model}; run the read command first or pass --read"
        )
    concepts = read_json(args.concepts).get("concepts", [])
    concept_ids = [str(row["id"]) for row in concepts]
    if len(concept_ids) != 43:
        raise ValueError(f"expected 43 Part 35 concepts, found {len(concept_ids)}")
    result = score_documents(documents, concept_ids)
    result["models_scored"] = sorted(documents)
    result["current_model"] = model
    result["source_reads"] = {
        model_id: str(
            Path(args.read).expanduser()
            if args.read and model_id == model
            else RESULTS_ROOT / MODEL_SLUGS[model_id] / "read.json"
        )
        for model_id in documents
    }
    output = (
        Path(args.output).expanduser()
        if args.output
        else RESULTS_ROOT / MODEL_SLUGS[model] / "score-42.json"
    )
    write_json(output, result)
    print(f"Wrote {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    read_parser = commands.add_parser("read", help="read selected units with both lens methods")
    read_parser.add_argument("--model", choices=MODEL_SLUGS, required=True)
    read_parser.add_argument("--snapshot", help="cached snapshot directory; auto-detect one snapshot by default")
    read_parser.add_argument("--lens", help="pre-fitted Jacobian lens .pt file")
    read_parser.add_argument(
        "--lens-checksums",
        default=str(LENS_CHECKSUMS_DEFAULT),
        help="SHA256SUMS manifest used to verify the selected Jacobian lens",
    )
    read_parser.add_argument("--screen", default=str(SCREEN_DEFAULT), help="Part 35 screen JSONL for 1.7B")
    read_parser.add_argument(
        "--part39-units",
        default=str(PART39_UNITS_DEFAULT),
        help="Part 39 passing-unit JSON used as the separate extra 4B read set",
    )
    read_parser.add_argument("--output", help="override output read.json path")
    read_parser.set_defaults(func=run_read)

    score_parser = commands.add_parser("score", help="score read.json outputs against predictions 42.1-42.5")
    score_parser.add_argument("--model", choices=MODEL_SLUGS, required=True)
    score_parser.add_argument("--read", help="override this model's input read.json")
    score_parser.add_argument("--concepts", default=str(CONCEPTS_PATH))
    score_parser.add_argument("--output", help="override output score-42.json path")
    score_parser.set_defaults(func=run_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
