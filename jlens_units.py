"""Read MLP down-projection columns through logit and Jacobian lenses.

Examples:
  python jlens_units.py --model-name qwen3-1.7b --units 22:1610 19:4826 --out results/part57/qwen3-1.7b/read.json
  python jlens_units.py --model-name olmo2-1b --out results/part57/olmo2-1b/read.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import part42


OLMO_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_ALIASES = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "olmo2-1b": OLMO_MODEL,
    "Qwen/Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    OLMO_MODEL: OLMO_MODEL,
}
OLMO_CONTROL_SEED = 5701
OLMO_WIDTH = 8192
OLMO_POSITIVE_CONTROL = (14, 3027)


def canonical_model_name(name: str) -> str:
    try:
        return MODEL_ALIASES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported model name: {name}") from exc


def olmo_default_units(seed: int = OLMO_CONTROL_SEED) -> list[dict[str, Any]]:
    """Target plus ten last-quarter controls, including Part 26's positive unit."""
    rng = random.Random(seed)
    candidates = [unit for unit in range(OLMO_WIDTH) if unit != OLMO_POSITIVE_CONTROL[1]]
    random_units = rng.sample(candidates, 9)
    output = [
        {
            "id": "olmo-target-L10-N6978",
            "label": "OLMo L10 N6978",
            "role": "target",
            "layer": 10,
            "unit": 6978,
        },
        {
            "id": "part26-positive-L14-N3027",
            "label": "Part 26 positive control L14 N3027",
            "role": "known_positive_control",
            "layer": OLMO_POSITIVE_CONTROL[0],
            "unit": OLMO_POSITIVE_CONTROL[1],
        },
    ]
    output.extend(
        {
            "id": f"random-control-{index:02d}-L14-N{unit}",
            "label": f"random control L14 N{unit}",
            "role": "random_control",
            "layer": 14,
            "unit": unit,
            "seed": seed,
        }
        for index, unit in enumerate(random_units, 1)
    )
    return output


def parse_unit_specs(specs: Iterable[str]) -> list[dict[str, Any]]:
    output = []
    for spec in specs:
        try:
            layer_text, unit_text = spec.split(":", 1)
            layer, unit = int(layer_text), int(unit_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"unit must use LAYER:UNIT syntax, found {spec!r}") from exc
        if layer < 0 or unit < 0:
            raise ValueError(f"layer and unit must be non-negative, found {spec!r}")
        output.append(
            {
                "id": f"L{layer}-N{unit}",
                "label": f"L{layer} N{unit}",
                "role": "requested",
                "layer": layer,
                "unit": unit,
            }
        )
    return output


def read_units(
    model_name: str,
    units: Iterable[Mapping[str, Any]],
    *,
    snapshot: str | Path | None = None,
    lens_path: str | Path | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Load selected weights and emit Part 42-compatible readouts."""
    import torch

    model = canonical_model_name(model_name)
    snapshot_path = part42.resolve_snapshot(model, snapshot)
    config = json.loads((snapshot_path / "config.json").read_text(encoding="utf-8"))
    text_config, hidden, num_layers, model_type = part42.config_dimensions(config)
    weight_map = part42.load_weight_map(snapshot_path)
    prefix = part42.find_language_model_prefix(
        weight_map, require_language_model=(model == "Qwen/Qwen3.5-4B")
    )
    chosen_lens = Path(lens_path) if lens_path else part42.MODEL_LENS_PATHS[model]
    jacobians, lens_meta, d_model = part42.load_jacobian_lens(chosen_lens)
    if d_model != hidden:
        raise ValueError(f"lens d_model={d_model} does not match model hidden size {hidden}")

    norm_key = f"{prefix}norm.weight"
    gain = part42.effective_norm_gain(
        part42.load_safetensor(snapshot_path, weight_map, norm_key).float(),
        model_type,
    )
    tied = bool(
        text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False))
    )
    head = part42.load_safetensor(
        snapshot_path,
        weight_map,
        part42.select_unembedding_key(weight_map, prefix, tied),
    )
    if tokenizer is None:
        tokenizer = part42._load_tokenizer(snapshot_path)

    output_units = []
    for unit_spec in units:
        layer, unit = int(unit_spec["layer"]), int(unit_spec["unit"])
        if not 0 <= layer < num_layers:
            raise ValueError(f"layer {layer} is outside model range 0..{num_layers - 1}")
        down = part42.load_safetensor(
            snapshot_path, weight_map, part42._layer_tensor_key(prefix, layer)
        )
        if not 0 <= unit < int(down.shape[1]):
            raise ValueError(
                f"unit {unit} is outside layer {layer} width 0..{down.shape[1] - 1}"
            )
        jac_layer = part42.jacobian_layer_for_write(layer, num_layers)
        jacobian = jacobians.get(jac_layer) if jac_layer is not None else None
        if jac_layer is not None and jacobian is None:
            raise ValueError(
                f"lens does not contain J[{jac_layer}] for requested layer {layer}"
            )
        write = down[:, unit].float()
        row = dict(unit_spec)
        row["readouts"] = {
            "logit_lens": part42._top_bottom(
                torch, part42.readout_logits(torch, head, write, gain), tokenizer
            ),
            "j_lens": part42._top_bottom(
                torch,
                part42.readout_logits(torch, head, write, gain, jacobian),
                tokenizer,
            ),
        }
        output_units.append(row)

    return {
        "schema_version": 1,
        "model": model,
        "model_slug": part42.MODEL_SLUGS.get(model, "olmo2-1b"),
        "snapshot": str(snapshot_path),
        "lens": str(chosen_lens),
        "lens_metadata": lens_meta,
        "jacobian_alignment": (
            "Block output write at layer l uses J[l]; the final block uses identity."
        ),
        "units": output_units,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model_name",
        default="qwen3-1.7b",
        help="qwen3-1.7b, qwen3.5-4b, olmo2-1b, or a full supported model id",
    )
    parser.add_argument("--units", nargs="+")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--lens", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = canonical_model_name(args.model_name)
    if args.units:
        units = parse_unit_specs(args.units)
    elif model == OLMO_MODEL:
        units = olmo_default_units()
    else:
        parser.error("--units is required for Qwen reads")
    result = read_units(
        model,
        units,
        snapshot=args.snapshot,
        lens_path=args.lens,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in result["units"]:
        for readout in ("logit_lens", "j_lens"):
            top = [item["decoded"] for item in row["readouts"][readout]["top"][:20]]
            bottom = [item["decoded"] for item in row["readouts"][readout]["bottom"][:20]]
            print(row["id"], readout, "TOP:", top)
            print(row["id"], readout, "BOT:", bottom)


if __name__ == "__main__":
    main()
