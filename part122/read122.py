#!/usr/bin/env python3
"""Read Part 122 persona passages at each unit's down_proj input channels."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for candidate in (HERE, ROOT, ROOT / "part119", ROOT / "part92"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import common92 as common  # noqa: E402
import rows119  # noqa: E402
import write_passages  # noqa: E402

MODEL_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
DEFAULT_MODEL = (r"E:\neuron-probe\hf\hub\models--allenai--OLMo-2-0425-1B-Instruct\snapshots"
                 + "\\" + MODEL_REVISION)
DEFAULT_OUT = "results/part122"
RANDOM_SEED = 12202
SHUFFLE_SEED = 12203
GATE2_LAYER = 2
GATE2_NEURON = 2466
GATE2_PASSAGES = "part95/passages95.json"
FRAMES = ("T", "F", "C")


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_model(snapshot_dir: str | Path, device: str):
    """Load the pinned local model as bf16, directly onto non-CPU devices."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = str(Path(snapshot_dir).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, torch_dtype=torch.bfloat16, local_files_only=True
        ).to(device)
    else:
        import transformers.modeling_utils as modeling_utils
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, torch_dtype=torch.bfloat16, local_files_only=True,
            device_map={"": device},
        )
    model.requires_grad_(False)
    model.eval()
    return tokenizer, model


def _flat_ids(value) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (tuple, list)) and len(value) == 1 and isinstance(value[0], (tuple, list)):
        value = value[0]
    return [int(token) for token in value]


def _encode(tokenizer, text: str, **kwargs) -> dict:
    encoded = tokenizer(text, **kwargs)
    if not isinstance(encoded, dict) and not hasattr(encoded, "__getitem__"):
        raise ValueError("tokenizer returned no encoding")
    return encoded


def _find_subsequence(sequence: list[int], target: list[int]) -> tuple[int, int] | None:
    if not target:
        return None
    for start in range(len(sequence) - len(target) + 1):
        if sequence[start:start + len(target)] == target:
            return start, start + len(target)
    return None


def prepare_passage(tokenizer, passage: str, frame: str) -> dict:
    """Return full input ids and the exact token positions belonging to passage text."""
    if frame in ("T", "F"):
        bare = _flat_ids(_encode(tokenizer, passage, add_special_tokens=False)["input_ids"])
        full = _flat_ids(_encode(tokenizer, passage, add_special_tokens=True)["input_ids"])
        bounds = _find_subsequence(full, bare)
        if bounds is None:
            raise ValueError("could not locate plain passage tokens after special-token wrapping")
        start, end = bounds
    elif frame == "C":
        messages = [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": passage},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        start_char = rendered.rfind(passage)
        if start_char < 0:
            raise ValueError("chat template did not preserve the assistant passage text")
        end_char = start_char + len(passage)
        try:
            encoded = _encode(tokenizer, rendered, add_special_tokens=False,
                              return_offsets_mapping=True)
            full = _flat_ids(encoded["input_ids"])
            offsets = encoded["offset_mapping"]
            if hasattr(offsets, "tolist"):
                offsets = offsets.tolist()
            if offsets and isinstance(offsets[0], (tuple, list)) and len(offsets[0]) == 1 and \
                    isinstance(offsets[0][0], (tuple, list)):
                offsets = offsets[0]
            selected = [index for index, (left, right) in enumerate(offsets)
                        if int(left) >= start_char and int(right) <= end_char and int(right) > int(left)]
            if not selected:
                raise ValueError("offset mapping contained no assistant passage tokens")
            start, end = selected[0], selected[-1] + 1
            if selected != list(range(start, end)):
                raise ValueError("assistant passage token offsets are not contiguous")
        except (TypeError, KeyError, NotImplementedError, ValueError):
            full = _flat_ids(_encode(tokenizer, rendered, add_special_tokens=False)["input_ids"])
            prefix = _flat_ids(_encode(tokenizer, rendered[:start_char], add_special_tokens=False)["input_ids"])
            through = _flat_ids(_encode(tokenizer, rendered[:end_char], add_special_tokens=False)["input_ids"])
            start, end = len(prefix), len(through)
            if not 0 <= start < end <= len(full):
                raise ValueError("could not locate assistant passage token span in chat input")
    else:
        raise ValueError("frame must be T, F, or C")
    if not 0 <= start < end <= len(full):
        raise ValueError("passage token span is empty or outside the input")
    mask = [start <= index < end for index in range(len(full))]
    return {"input_ids": full, "token_mask": mask, "token_count": end - start,
            "token_start": start, "token_end": end}


def load_passages(path: str | Path, units: list[dict]) -> dict[str, dict]:
    source = repo_path(path)
    with source.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("units"), list):
        raise ValueError("passages file must contain a units array")
    found = {}
    for row in data["units"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("each passage unit needs a string id")
        unit_id = row["id"]
        if unit_id in found:
            raise ValueError(f"duplicate passages for unit {unit_id}")
        found[unit_id] = row
    by_id = {unit["id"]: unit for unit in units}
    for unit_id, unit in by_id.items():
        passage_unit = found.get(unit_id)
        if passage_unit is None:
            raise ValueError(f"passages are missing unit {unit_id}")
        if passage_unit.get("persona") != unit["persona"] or passage_unit.get("desc") != unit["desc"]:
            raise ValueError(f"passage persona data is stale for unit {unit_id}")
        payload = {frame: passage_unit.get(frame) for frame in write_passages.PERSONA_FIELDS}
        write_passages.validate_payload(payload, unit["persona"])
    return found


def build_read_set(units: list[dict], layers) -> dict:
    """Assign three seeded random same-layer channels and a non-self passage shuffle."""
    focal_by_layer: dict[int, set[int]] = {}
    for unit in units:
        focal_by_layer.setdefault(int(unit["layer"]), set()).add(int(unit["neuron"]))
    unit_channels = {}
    for unit in units:
        unit_id = str(unit["id"])
        layer_no = int(unit["layer"])
        width = int(layers[layer_no].mlp.down_proj.weight.shape[1])
        neuron = int(unit["neuron"])
        if not 0 <= neuron < width:
            raise ValueError(f"unit {unit_id} channel {neuron} is outside layer {layer_no} width {width}")
        available = [channel for channel in range(width) if channel not in focal_by_layer[layer_no]]
        if len(available) < 3:
            raise ValueError(f"layer {layer_no} lacks three random channels outside the focal-unit set")
        seed = int.from_bytes(hashlib.sha256(f"{RANDOM_SEED}:{unit_id}".encode()).digest()[:8], "big")
        rng = random.Random(seed)
        random_channels = rng.sample(available, 3)
        unit_channels[unit_id] = {
            "layer": layer_no, "neuron": neuron,
            "random": [{"channel": channel, "seed": seed, "random_id": index}
                       for index, channel in enumerate(random_channels)],
        }

    ids = sorted(unit_channels)
    if len(ids) < 2:
        partners = {unit_id: unit_id for unit_id in ids}
    else:
        rng = random.Random(SHUFFLE_SEED)
        shuffled = ids[:]
        for _ in range(100):
            rng.shuffle(shuffled)
            if all(source != target for source, target in zip(ids, shuffled)):
                break
        else:
            shuffled = ids[1:] + ids[:1]
        partners = dict(zip(ids, shuffled))
    return {"random_seed": RANDOM_SEED, "shuffle_seed": SHUFFLE_SEED,
            "units": unit_channels, "shuffled_partner": partners}


def _layer_descriptors(read_set: dict) -> dict[int, list[dict]]:
    by_layer: dict[int, list[dict]] = {}
    for unit_id, config in read_set["units"].items():
        layer_no = int(config["layer"])
        by_layer.setdefault(layer_no, []).append(
            {"unit_id": unit_id, "type": "unit", "random_id": None,
             "channel": int(config["neuron"])}
        )
        for row in config["random"]:
            by_layer[layer_no].append(
                {"unit_id": unit_id, "type": "random", "random_id": int(row["random_id"]),
                 "channel": int(row["channel"])}
            )
    return by_layer


def _pack(tokenizer, prepared: list[dict], device: str):
    import torch

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("tokenizer needs a pad or EOS token")
    width = max(len(row["input_ids"]) for row in prepared)
    input_ids = []
    attention = []
    token_masks = []
    for row in prepared:
        missing = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [int(pad_id)] * missing)
        attention.append([1] * len(row["input_ids"]) + [0] * missing)
        token_masks.append(row["token_mask"] + [False] * missing)
    return (torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(attention, dtype=torch.long, device=device),
            torch.tensor(token_masks, dtype=torch.bool, device=device))


def capture_batch(model, tokenizer, layers, read_set: dict, passages: list[dict],
                  frame: str, device: str) -> list[dict]:
    """Forward one batch and summarize every selected channel over passage tokens only."""
    import torch

    prepared = [prepare_passage(tokenizer, row["text"], frame) for row in passages]
    input_ids, attention, masks = _pack(tokenizer, prepared, device)
    descriptors = _layer_descriptors(read_set)
    values = [dict() for _ in passages]
    handles = []

    def hook_for(layer_no: int, rows: list[dict]):
        def hook(_module, inputs):
            activation = inputs[0].detach().float()
            channels = sorted({int(row["channel"]) for row in rows})
            if channels and max(channels) >= activation.shape[-1]:
                raise ValueError(f"read channel is outside layer {layer_no} down_proj input width")
            selected = activation[:, :, channels]
            valid = masks.to(device=activation.device)
            masked = selected.masked_fill(~valid.unsqueeze(-1), float("-inf"))
            maxima = masked.max(dim=1).values
            means = (selected * valid.unsqueeze(-1)).sum(dim=1) / \
                    valid.sum(dim=1).clamp_min(1).unsqueeze(-1)
            position = {channel: index for index, channel in enumerate(channels)}
            for descriptor in rows:
                channel_pos = position[int(descriptor["channel"])]
                key = descriptor["unit_id"]
                # Fill every batch row; this hook executes once for the whole batch.
                for row_index in range(len(passages)):
                    stats_row = {"channel": int(descriptor["channel"]),
                                 "max": float(maxima[row_index, channel_pos].item()),
                                 "mean": float(means[row_index, channel_pos].item())}
                    slot = values[row_index].setdefault(key, {"random": []})
                    if descriptor["type"] == "unit":
                        slot["unit"] = stats_row
                    else:
                        slot["random"].append({"random_id": descriptor["random_id"], **stats_row})
            return None
        return hook

    try:
        for layer_no, rows in descriptors.items():
            handles.append(layers[layer_no].mlp.down_proj.register_forward_pre_hook(
                hook_for(layer_no, rows)))
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    finally:
        for handle in reversed(handles):
            handle.remove()

    output = []
    for row_index, context in enumerate(passages):
        for entry in values[row_index].values():
            entry["random"].sort(key=lambda item: item["random_id"])
        output.append({
            "passage_id": context["passage_id"], "frame": frame,
            "source_unit_id": context["source_unit_id"], "pair_index": context["pair_index"],
            "side": context["side"], "text": context["text"],
            "token_count": prepared[row_index]["token_count"],
            "token_start": prepared[row_index]["token_start"],
            "token_end": prepared[row_index]["token_end"], "values": values[row_index],
        })
    return output


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _existing_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    found = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                path.write_text("".join(lines[:index]), encoding="utf-8")
                break
            raise ValueError(f"invalid activation JSONL at line {index + 1}") from exc
        key = str(row.get("passage_id", ""))
        if not key:
            raise ValueError(f"activation row {index + 1} has no passage_id")
        if key in found:
            raise ValueError(f"duplicate activation passage_id {key}")
        found[key] = row
    return found


def _write_json(path: Path, value: dict, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        with path.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous != value:
            raise ValueError(f"existing {path.name} does not match this run configuration")
        return
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _passage_contexts(units: list[dict], passage_units: dict[str, dict], frame: str) -> list[dict]:
    contexts = []
    for unit in units:
        unit_id = unit["id"]
        source = passage_units[unit_id]
        source_field = "third_person" if frame == "T" else "first_person"
        for pair_index, pair in enumerate(source[source_field]):
            pair_id = f"{frame}:{unit_id}:{pair_index}"
            for side in ("persona", "ordinary"):
                contexts.append({
                    "passage_id": f"{pair_id}:{side}", "source_unit_id": unit_id,
                    "pair_index": pair_index, "side": side, "text": pair[side],
                })
    return contexts


def _load_gate2_passages(path: str | Path) -> list[list[str]]:
    with repo_path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    pairs = data.get("pairs") if isinstance(data, dict) else None
    if not isinstance(pairs, list) or len(pairs) != 16:
        raise ValueError("Part 95 gate passages must contain exactly 16 pairs")
    result = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(text, str) for text in pair):
            raise ValueError(f"Part 95 gate pair {index} is malformed")
        result.append(pair)
    return result


def run_gate2(tokenizer, model, layers, out_dir: Path, passage_path: str | Path,
              device: str, *, layer_no: int | None = None, channel: int | None = None) -> dict:
    import torch

    layer_no = GATE2_LAYER if layer_no is None else int(layer_no)
    channel = GATE2_NEURON if channel is None else int(channel)
    if layer_no >= len(layers) or channel >= int(layers[layer_no].mlp.down_proj.weight.shape[1]):
        raise ValueError("Part 95 gate channel is outside the loaded model")
    pair_texts = _load_gate2_passages(passage_path)
    contexts = []
    for pair_index, (space_text, ordinary_text) in enumerate(pair_texts):
        for side, text in (("space", space_text), ("ordinary", ordinary_text)):
            contexts.append({"passage_id": f"gate2:{pair_index}:{side}", "source_unit_id": "gate2",
                             "pair_index": pair_index, "side": side, "text": text})
    prepared = [prepare_passage(tokenizer, row["text"], "T") for row in contexts]
    input_ids, attention, masks = _pack(tokenizer, prepared, device)
    captured = []

    def hook(_module, inputs):
        activation = inputs[0].detach().float()
        selected = activation[:, :, channel]
        valid = masks.to(device=activation.device)
        maxima = selected.masked_fill(~valid, float("-inf")).max(dim=1).values
        means = (selected * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        captured.extend({"max": float(maxima[index].item()), "mean": float(means[index].item())}
                        for index in range(len(contexts)))

    handle = layers[layer_no].mlp.down_proj.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    finally:
        handle.remove()
    rows = []
    for pair_index in range(16):
        space_index, ordinary_index = pair_index * 2, pair_index * 2 + 1
        rows.append({"pair_index": pair_index, "space": captured[space_index],
                     "ordinary": captured[ordinary_index],
                     "space_token_count": prepared[space_index]["token_count"],
                     "ordinary_token_count": prepared[ordinary_index]["token_count"]})
    gate = {"layer": layer_no, "neuron": channel, "pairs": rows,
            "source": str(passage_path), "model_revision": MODEL_REVISION}
    _write_json(out_dir / "gate2.json", gate)
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--passages", default=str(HERE / "passages122.json"))
    parser.add_argument("--gate2-passages", default=GATE2_PASSAGES)
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds-haiku", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    parser.add_argument("--batch", type=int, default=8, help="matched pairs per forward pass")
    parser.add_argument("--unit-limit", type=int, help="smoke only: read sources for the first N units")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.unit_limit is not None and args.unit_limit < 1:
        parser.error("--unit-limit must be positive")

    units = rows119.load_units(args.neurons, args.kinds_haiku, args.kinds_ds)
    selected = units if args.unit_limit is None else units[:args.unit_limit]
    if args.unit_limit is not None and len(selected) != args.unit_limit:
        parser.error("--unit-limit exceeds available units")
    passage_units = load_passages(args.passages, selected)
    passages_digest = hashlib.sha256(repo_path(args.passages).read_bytes()).hexdigest()
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    activation_path = out_dir / "activations.jsonl"
    meta_path = out_dir / "run_meta.json"
    if activation_path.exists() and not meta_path.exists():
        parser.error("existing activations lack run_meta.json; use a fresh output directory")

    tokenizer, model = load_model(args.model, args.device)
    parameter = next(model.parameters(), None)
    if parameter is None or parameter.dtype != __import__("torch").bfloat16:
        parser.error("loaded model parameters must remain bf16")
    layers = common.get_layers(model)
    read_set = build_read_set(units, layers)
    _write_json(out_dir / "read_set.json", read_set, refuse_existing=True)
    run_meta = {"model": str(args.model), "model_revision": MODEL_REVISION,
                "parameter_dtype": str(parameter.dtype), "passages_sha256": passages_digest,
                "unit_ids": [unit["id"] for unit in selected],
                "random_seed": RANDOM_SEED, "shuffle_seed": SHUFFLE_SEED}
    _write_json(meta_path, run_meta, refuse_existing=True)

    existing = _existing_rows(activation_path)
    for frame in FRAMES:
        before_frame = len(existing)
        contexts = _passage_contexts(selected, passage_units, frame)
        pending = [row for row in contexts if row["passage_id"] not in existing]
        # Keep each matched pair together in a batch. The last chunk may have fewer pairs.
        pairs = {}
        for row in pending:
            key = (row["source_unit_id"], row["pair_index"])
            pairs.setdefault(key, {})[row["side"]] = row
        ordered_pairs = []
        for unit in selected:
            for pair_index in range(8):
                rows = pairs.get((unit["id"], pair_index))
                if not rows:
                    continue
                missing = [side for side in ("persona", "ordinary") if side not in rows]
                if missing:
                    # Resume only absent passages, preserving the existing partner row.
                    ordered_pairs.extend([[rows[side]] for side in rows])
                else:
                    ordered_pairs.append([rows["persona"], rows["ordinary"]])
        for begin in range(0, len(ordered_pairs), args.batch):
            batch_contexts = [row for pair in ordered_pairs[begin:begin + args.batch] for row in pair]
            batch_rows = capture_batch(model, tokenizer, layers, read_set, batch_contexts, frame, args.device)
            _append_rows(activation_path, batch_rows)
            existing.update((row["passage_id"], row) for row in batch_rows)
        print(f"frame {frame}: wrote {len(existing) - before_frame} of {len(contexts)} passages", flush=True)

    run_gate2(tokenizer, model, layers, out_dir, args.gate2_passages, args.device)
    print(json.dumps({"activations": str(activation_path), "passages": len(existing),
                      "units": len(selected), "gate2": str(out_dir / "gate2.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
