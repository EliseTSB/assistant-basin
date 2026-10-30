#!/usr/bin/env python3
"""Capture Part 124 phrase-slot maxima at focal and random channels."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for candidate in (HERE, ROOT, ROOT / "part122", ROOT / "part119", ROOT / "part92", ROOT / "part85"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import passages124  # noqa: E402

DEFAULT_MODEL = (r"E:\neuron-probe\hf\hub\models--allenai--OLMo-2-0425-1B-Instruct\snapshots"
                 + "\\" + passages124.MODEL_REVISION)
DEFAULT_OUT = "results/part124"
DEFAULT_PASSAGES = "part124/passages124.json"
RANDOM_POOL_SEED = 12402
RANDOM_CONTROL_SEED = 12403
FRAMES = ("T", "F", "C")


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path):
    with repo_path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: dict, *, refuse_mismatch: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_mismatch and path.exists():
        with path.open(encoding="utf-8") as stream:
            old = json.load(stream)
        if old != value:
            raise ValueError(f"existing {path.name} does not match this run")
        return
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def build_read_set(units: list[dict], layers, random_count: int = 1000) -> dict:
    """Make a reproducible random pool at every used layer and three controls per unit."""
    if random_count < 3:
        raise ValueError("random channel pool must contain at least three channels")
    focal_by_layer: dict[int, set[int]] = {}
    for unit in units:
        focal_by_layer.setdefault(int(unit["layer"]), set()).add(int(unit["neuron"]))
    pools: dict[str, list[int]] = {}
    widths = {}
    for layer_no in sorted(focal_by_layer):
        if not 0 <= layer_no < len(layers):
            raise ValueError(f"unit layer {layer_no} is outside the loaded model")
        width = int(layers[layer_no].mlp.down_proj.weight.shape[1])
        widths[str(layer_no)] = width
        focal = focal_by_layer[layer_no]
        if any(channel < 0 or channel >= width for channel in focal):
            raise ValueError(f"a focal channel is outside layer {layer_no} width {width}")
        available = [channel for channel in range(width) if channel not in focal]
        if len(available) < random_count:
            raise ValueError(
                f"layer {layer_no} has {len(available)} channels outside focal channels; "
                f"need {random_count}"
            )
        seed = int.from_bytes(hashlib.sha256(
            f"{RANDOM_POOL_SEED}:{layer_no}".encode("utf-8")).digest()[:8], "big")
        pools[str(layer_no)] = random.Random(seed).sample(available, random_count)

    unit_map = {}
    for unit in units:
        unit_id = str(unit["id"])
        layer_no = int(unit["layer"])
        pool = pools[str(layer_no)]
        if len(pool) < 3:
            raise ValueError(f"layer {layer_no} random pool cannot supply three controls")
        seed = int.from_bytes(hashlib.sha256(
            f"{RANDOM_CONTROL_SEED}:{unit_id}".encode("utf-8")).digest()[:8], "big")
        channels = random.Random(seed).sample(pool, 3)
        unit_map[unit_id] = {
            "layer": layer_no,
            "channel": int(unit["neuron"]),
            "target_phrase": str(unit["persona"]),
            "random_controls": [
                {"channel": channel, "target_phrase": str(unit["persona"]), "control_index": index}
                for index, channel in enumerate(channels)
            ],
        }
    return {
        "random_pool_seed": RANDOM_POOL_SEED,
        "random_control_seed": RANDOM_CONTROL_SEED,
        "random_count_per_layer": random_count,
        "layer_widths": widths,
        "random_pools": pools,
        "units": unit_map,
    }


def build_contexts(manifest: dict) -> list[dict]:
    units = manifest.get("phrases")
    templates = manifest.get("templates")
    if not isinstance(units, list) or not isinstance(templates, dict):
        raise ValueError("passage manifest needs phrases and templates")
    contexts = []
    for frame in FRAMES:
        template_key = "T" if frame == "T" else "F"
        rows = templates.get(template_key)
        if not isinstance(rows, list) or len(rows) != 8:
            raise ValueError(f"frame {frame} must have eight templates")
        for template_index, template in enumerate(rows):
            for unit in units:
                for side, phrase_key in (("persona", "persona"), ("neutral", "neutral")):
                    phrase = unit[phrase_key]
                    contexts.append({
                        "frame": frame,
                        "template_index": template_index,
                        "template": str(template),
                        "unit_id": str(unit["id"]),
                        "kind": str(unit["kind"]),
                        "persona": str(unit["persona"]),
                        "phrase": str(phrase),
                        "side": side,
                        "text": template.replace("{slot}", str(phrase)),
                    })
    return contexts


def context_key(row: dict) -> tuple[str, int, str, str]:
    return (str(row["frame"]), int(row["template_index"]),
            str(row["unit_id"]), str(row["side"]))


def _existing_rows(path: Path) -> dict[tuple[str, int, str, str], dict]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    found = {}
    truncate_at = None
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                print(f"skipping torn final JSONL line: {path}:{index + 1}", file=sys.stderr, flush=True)
                truncate_at = index
                break
            raise ValueError(f"invalid activation JSONL at line {index + 1}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"activation line {index + 1} is not an object")
        key = context_key(row)
        if key in found:
            raise ValueError(f"duplicate activation row {key}")
        found[key] = row
    if truncate_at is not None:
        with path.open("w", encoding="utf-8") as stream:
            stream.writelines(lines[:truncate_at])
    elif lines and not lines[-1].endswith("\n"):
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    return found


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _read122_module():
    import read122

    return read122


def _prepare_batch(tokenizer, contexts: list[dict]) -> list[dict]:
    prepared = []
    for row in contexts:
        item = passages124.prepare_slot(
            tokenizer, row["frame"], str(row["template"]), row["phrase"]
        )
        prepared.append({**item, "token_mask": item["slot_mask"],
                         "post_token_mask": item["post_slot_mask"]})
    return prepared


def _channel_groups(units: list[dict], read_set: dict) -> dict[int, dict[str, list]]:
    by_layer: dict[int, dict[str, list]] = {}
    for unit in units:
        layer_no = int(unit["layer"])
        group = by_layer.setdefault(layer_no, {"focal": [], "random": []})
        group["focal"].append({"unit_id": str(unit["id"]), "channel": int(unit["neuron"])})
        for channel in read_set["units"][str(unit["id"])]["random_controls"]:
            group["random"].append({"unit_id": str(unit["id"]),
                                    "channel": int(channel["channel"]),
                                    "control_index": int(channel["control_index"])})
    for layer_no, group in by_layer.items():
        group["pool"] = [int(channel) for channel in read_set["random_pools"][str(layer_no)]]
    return by_layer


def capture_batch(model, tokenizer, layers, units: list[dict], read_set: dict,
                  contexts: list[dict], device: str, *, return_logits: bool = False):
    """Capture one batch; each layer's selected maxima transfer to CPU once in its hook."""
    import torch

    prepared = _prepare_batch(tokenizer, contexts)
    read122 = _read122_module()
    input_ids, attention, masks = read122._pack(tokenizer, prepared, device)
    width = input_ids.shape[1]
    post_masks = torch.tensor(
        [row["post_token_mask"] + [False] * (width - len(row["post_token_mask"]))
         for row in prepared],
        dtype=torch.bool,
        device=device,
    )
    groups = _channel_groups(units, read_set)
    row_values = [dict() for _ in contexts]
    handles = []

    def hook_for(layer_no: int, group: dict):
        channels = sorted(set([int(row["channel"]) for row in group["focal"]] + group["pool"]))
        position = {channel: index for index, channel in enumerate(channels)}

        def hook(_module, inputs):
            activation = inputs[0].detach().float()
            if channels and max(channels) >= activation.shape[-1]:
                raise ValueError(f"selected channel is outside layer {layer_no} down_proj input width")
            selected = activation[:, :, channels]
            valid = masks.to(device=activation.device)
            masked = selected.masked_fill(~valid.unsqueeze(-1), float("-inf"))
            maxima = masked.max(dim=1).values
            slot_counts = valid.sum(dim=1).clamp_min(1).unsqueeze(-1)
            means = selected.masked_fill(~valid.unsqueeze(-1), 0.0).sum(dim=1) / slot_counts
            post_valid = post_masks.to(device=activation.device)
            post_masked = selected.masked_fill(~post_valid.unsqueeze(-1), float("-inf"))
            post_maxima = post_masked.max(dim=1).values
            # One transfer for the full layer hook. Never call .item() per channel or row here.
            cpu_maxima, cpu_means, cpu_post_maxima = torch.stack(
                (maxima, means, post_maxima), dim=0
            ).detach().cpu().unbind(0)
            values = cpu_maxima.tolist()
            mean_values = cpu_means.tolist()
            post_values = cpu_post_maxima.tolist()
            focal_rows = group["focal"]
            pool = group["pool"]
            for row_index, context in enumerate(contexts):
                focal = {
                    str(layer_no): {
                        str(row["unit_id"]): float(values[row_index][position[int(row["channel"])]])
                        for row in focal_rows
                    }
                }
                row_values[row_index].setdefault("focal", {}).update(focal)
                focal_mean = {
                    str(layer_no): {
                        str(row["unit_id"]): float(
                            mean_values[row_index][position[int(row["channel"])]]
                        )
                        for row in focal_rows
                    }
                }
                row_values[row_index].setdefault("focal_mean", {}).update(focal_mean)
                focal_post_max = {
                    str(layer_no): {
                        str(row["unit_id"]): float(
                            post_values[row_index][position[int(row["channel"])]]
                        )
                        for row in focal_rows
                    }
                }
                row_values[row_index].setdefault("focal_post_max", {}).update(focal_post_max)
                if context["side"] == "persona":
                    random_values = [float(values[row_index][position[channel]]) for channel in pool]
                    row_values[row_index].setdefault("random_max", {})[str(layer_no)] = random_values

        return hook

    try:
        for layer_no, group in groups.items():
            handles.append(layers[layer_no].mlp.down_proj.register_forward_pre_hook(
                hook_for(layer_no, group)))
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    finally:
        for handle in reversed(handles):
            handle.remove()

    rows = []
    for index, context in enumerate(contexts):
        if set(row_values[index].get("focal", {})) != {str(layer) for layer in groups}:
            raise ValueError("a hook did not return every focal layer activation")
        row = {
            "frame": context["frame"],
            "template_index": int(context["template_index"]),
            "unit_id": context["unit_id"],
            "kind": context["kind"],
            "persona": context["persona"],
            "phrase": context["phrase"],
            "side": context["side"],
            "slot_token_count": prepared[index]["slot_token_count"],
            "focal": row_values[index]["focal"],
            "focal_mean": row_values[index]["focal_mean"],
            "focal_post_max": row_values[index]["focal_post_max"],
        }
        if context["side"] == "persona":
            row["random_max"] = row_values[index]["random_max"]
        rows.append(row)
    logits = output.logits.detach().cpu() if return_logits else None
    return rows, logits


def do_nothing_check(model, tokenizer, layers, units: list[dict], read_set: dict,
                     contexts: list[dict], device: str) -> dict:
    import torch

    check_contexts = contexts[:min(len(contexts), 16)]
    prepared = _prepare_batch(tokenizer, check_contexts)
    read122 = _read122_module()
    input_ids, attention, _masks = read122._pack(tokenizer, prepared, device)
    with torch.inference_mode():
        baseline_output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
        baseline_logits = baseline_output.logits.detach().cpu()
    _ignored_rows, hooked_logits = capture_batch(
        model, tokenizer, layers, units, read_set, check_contexts, device, return_logits=True
    )
    equal = bool(torch.equal(baseline_logits, hooked_logits))
    return {
        "passed": equal,
        "identical_logits": equal,
        "input_shape": list(input_ids.shape),
        "logit_shape": list(baseline_logits.shape),
        "read_only_hooks": True,
    }


def _run_read122_loader(model_path: str, device: str):
    read122 = _read122_module()
    tokenizer, model = read122.load_model(model_path, device)
    return tokenizer, model, read122.common.get_layers(model)


def main(argv: list[str] | None = None, *, tokenizer=None, model=None, layers=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--passages", default=DEFAULT_PASSAGES)
    parser.add_argument("--batch", type=int, default=8, help="matched persona/neutral pairs per forward")
    parser.add_argument("--random-count", type=int, default=1000,
                        help="instrument random channels to capture at each used layer")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.random_count < 3:
        parser.error("--random-count must be at least three")

    manifest = read_json(args.passages)
    if manifest.get("model_revision") != passages124.MODEL_REVISION:
        parser.error("passage manifest has a different model revision")
    construction = manifest.get("construction_check", {})
    if not isinstance(construction, dict) or construction.get("passed") is not True:
        parser.error("passage construction check did not pass")
    units = manifest.get("phrases")
    if not isinstance(units, list) or not units:
        parser.error("passage manifest has no phrase units")

    if tokenizer is None or model is None or layers is None:
        if tokenizer is not None or model is not None or layers is not None:
            parser.error("inject tokenizer, model and layers together")
        tokenizer, model, layers = _run_read122_loader(args.model, args.device)
    else:
        model.eval()
    import torch

    parameter = next(model.parameters(), None)
    if parameter is None or parameter.dtype != torch.bfloat16:
        parser.error("loaded model parameters must remain bf16")
    if not hasattr(tokenizer, "pad_token_id"):
        parser.error("tokenizer must define a pad_token_id")

    read_set = build_read_set(units, layers, args.random_count)
    contexts = build_contexts(manifest)
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    read_set_path = out_dir / "read_set.json"
    _write_json(read_set_path, read_set, refuse_mismatch=True)
    activation_path = out_dir / "activations.jsonl"
    meta_path = out_dir / "run_meta.json"
    if activation_path.exists() and not meta_path.exists():
        parser.error("existing activations lack run_meta.json; use a fresh output directory")

    passages_path = repo_path(args.passages)
    static_meta = {
        "model": str(args.model),
        "model_revision": passages124.MODEL_REVISION,
        "parameter_dtype": str(parameter.dtype),
        "device": args.device,
        "passages_sha256": hashlib.sha256(passages_path.read_bytes()).hexdigest(),
        "unit_ids": [str(unit["id"]) for unit in units],
        "random_count_per_layer": args.random_count,
        "random_pool_seed": RANDOM_POOL_SEED,
        "random_control_seed": RANDOM_CONTROL_SEED,
        "batch_pairs": args.batch,
    }
    previous_meta = read_json(meta_path) if meta_path.exists() else None
    if previous_meta is not None:
        previous_static = {key: previous_meta.get(key) for key in static_meta}
        if previous_static != static_meta:
            parser.error("existing run metadata does not match this run configuration")

    do_nothing = do_nothing_check(model, tokenizer, layers, units, read_set, contexts, args.device)
    run_meta = {**static_meta, "do_nothing": do_nothing}
    _write_json(meta_path, run_meta)

    existing = _existing_rows(activation_path)
    expected_keys = {context_key(row) for row in contexts}
    unknown = set(existing) - expected_keys
    if unknown:
        raise ValueError(f"activation file contains rows outside this manifest: {sorted(unknown)[:3]}")
    pending = [row for row in contexts if context_key(row) not in existing]
    chunk_size = args.batch * 2
    for start in range(0, len(pending), chunk_size):
        contexts_batch = pending[start:start + chunk_size]
        rows, _logits = capture_batch(
            model, tokenizer, layers, units, read_set, contexts_batch, args.device
        )
        _append_rows(activation_path, rows)
        existing.update((context_key(row), row) for row in rows)
        print(f"wrote {len(existing)} of {len(contexts)} activation rows", flush=True)

    if set(existing) != expected_keys:
        raise ValueError(f"missing {len(expected_keys - set(existing))} activation rows after capture")
    print(json.dumps({"activations": str(activation_path), "rows": len(existing),
                      "units": len(units), "layers": sorted(read_set["random_pools"], key=int),
                      "do_nothing": do_nothing}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
