#!/usr/bin/env python3
"""Collect targeted Part 64 windows through the pinned atlas model hooks.

This module deliberately imports atlas_collect_fast2's document tokenization,
batching, forward hooks, and marked-window decoder. It narrows each layer to the
Part 60 dev units and increases only the per-unit reservoirs to Part 64's 3x
candidate counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import atlas_collect_fast2 as atlas
from part64 import collect64


PILOT_CONTROL_SEED = 5601
PILOT_CONTROL_POOL_SIZE = 100
FRESH_COUNTS = {"top": 36, "random": 18, "weak": 9}


class ModelCollectionError(RuntimeError):
    pass


def _validate_units(units: dict[str, dict[str, Any]]) -> dict[int, dict[str, list[int]]]:
    """Return per-layer neuron and control ids, checking canonical unit keys."""
    if not units:
        raise ModelCollectionError("no target units were supplied")
    selected: dict[int, dict[str, list[int]]] = {}
    seen: set[tuple[int, str, int]] = set()
    for key, row in units.items():
        try:
            layer = int(row["layer"])
            kind = str(row["type"])
            record = row["record"]
            if kind == "neuron":
                index = int(record["unit"])
                expected_key = f"L{layer}:u{index}"
                category = "neuron"
            elif kind == "control":
                index = int(record["control_id"])
                expected_key = f"L{layer}:c{index}"
                category = "control"
            else:
                raise ModelCollectionError(f"{key}: unsupported unit type {kind!r}")
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelCollectionError(f"{key}: malformed Part 60 unit record") from exc
        if key != expected_key:
            raise ModelCollectionError(f"unit key {key!r} does not match its Part 60 record {expected_key!r}")
        if not 0 <= layer < atlas.EXPECTED_LAYERS:
            raise ModelCollectionError(f"{key}: layer is outside the pinned model")
        if category == "neuron" and not 0 <= index < atlas.EXPECTED_WIDTH:
            raise ModelCollectionError(f"{key}: neuron index is outside the pinned MLP width")
        if category == "control" and not 0 <= index < PILOT_CONTROL_POOL_SIZE:
            raise ModelCollectionError(
                f"{key}: control id must be in Part 56's 0..{PILOT_CONTROL_POOL_SIZE - 1} pool"
            )
        identity = (layer, category, index)
        if identity in seen:
            raise ModelCollectionError(f"duplicate selected unit coordinate: {key}")
        seen.add(identity)
        bucket = selected.setdefault(layer, {"neuron": [], "control": []})
        bucket[category].append(index)
    for categories in selected.values():
        for kind in categories:
            categories[kind].sort()
    return selected


def _state_pair(layer: int, selected: dict[str, list[int]], seed: int,
                width: int) -> tuple[Any, Any]:
    neuron_ids = selected["neuron"]
    control_ids = selected["control"]
    neuron_state = atlas.UnitState(
        len(neuron_ids), len(neuron_ids), seed, layer, "neuron", unit_ids=neuron_ids
    )
    control_state = atlas.UnitState(
        len(control_ids), len(control_ids), seed, layer, "control",
        input_width=width, unit_ids=control_ids,
    )
    if control_ids:
        # The pilot used atlas seed 5601 and a 100-direction pool per layer.
        # Generate the canonical rows, then evaluate only selected control ids.
        all_directions = atlas.generate_controls(
            PILOT_CONTROL_POOL_SIZE, width, PILOT_CONTROL_SEED, layer
        )
        control_state.directions = all_directions[control_ids].copy()
    for state in (neuron_state, control_state):
        state.random = atlas.WindowReservoir(
            state.width, FRESH_COUNTS["random"], seed, layer, f"{state.kind}-random"
        )
        state.weak = atlas.WindowReservoir(
            state.width, FRESH_COUNTS["weak"], seed, layer, f"{state.kind}-weak"
        )
    return neuron_state, control_state


def collect_fresh(
    units: dict[str, dict[str, Any]],
    *,
    data: Path,
    torch: Any,
    tokenizer: Any,
    base_model: Any,
    layers: Any,
    device: str = "cpu",
    docs_limit: int | None = None,
    batch_size: int = 2,
    seed: int = collect64.SEED,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Collect exactly the requested unit coordinates and return collect64 rows.

    `docs_limit` bounds the scan to the first N documents. The production CLI
    passes `collect64.PILOT_DOCS` (Amendment 5: the Part 56 pilot's 1,000
    documents) after verifying the full registered corpus file; tests also use
    it for the 200-doc head fixture.
    """
    if batch_size <= 0:
        raise ModelCollectionError("batch size must be positive")
    selected = _validate_units(units)
    docs = atlas._read_docs(Path(data), tokenizer, docs_limit)
    batches = atlas.make_batches(docs, batch_size)
    total_positions = sum(len(row["ids"]) for row in docs)
    if total_positions <= 0:
        raise ModelCollectionError("corpus contains no valid token positions")

    previous_top_k = atlas.TOP_K
    atlas.TOP_K = FRESH_COUNTS["top"]
    try:
        states = {
            layer: _state_pair(layer, indices, seed, atlas.EXPECTED_WIDTH)
            for layer, indices in selected.items()
        }
        atlas._forward_batches(
            torch, base_model, layers, states, batches, device,
            int(tokenizer.pad_token_id), pass_number=1,
        )
        atlas._forward_batches(
            torch, base_model, layers, states, batches, device,
            int(tokenizer.pad_token_id), pass_number=2,
        )

        key_by_coord: dict[tuple[int, str, int], str] = {}
        for key, row in units.items():
            record = row["record"]
            if row["type"] == "neuron":
                key_by_coord[(int(row["layer"]), "neuron", int(record["unit"]))] = key
            else:
                key_by_coord[(int(row["layer"]), "control", int(record["control_id"]))] = key

        fresh: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for layer, (neuron_state, control_state) in states.items():
            for state in (neuron_state, control_state):
                rows = atlas._record_rows(layer, state, docs, tokenizer, total_positions)
                for row in rows:
                    if state.kind == "neuron":
                        coordinate = int(row["unit"])
                    else:
                        coordinate = int(state.unit_ids[int(row["control_id"])])
                    key = key_by_coord[(layer, state.kind, coordinate)]
                    strata: dict[str, list[dict[str, Any]]] = {}
                    for category in ("top", "random", "weak"):
                        candidates = row[category]
                        if len(candidates) != FRESH_COUNTS[category]:
                            raise ModelCollectionError(
                                f"{key} has {len(candidates)}/{FRESH_COUNTS[category]} "
                                f"fresh {category} candidates on {len(docs)} documents"
                            )
                        strata[category] = [
                            {"doc": item["doc"], "pos": item["pos"],
                             "activation": item["act"], "window": item["window"]}
                            for item in candidates
                        ]
                    fresh[key] = strata
        if set(fresh) != set(units):
            raise ModelCollectionError("model traversal did not return every requested target unit")
        return fresh
    finally:
        atlas.TOP_K = previous_top_k


def load_pinned_model(cache_dir: Path, device: str):
    """Load Part 38's pinned Qwen snapshot using atlas's registered model path."""
    if device not in ("cpu", "cuda", "xpu"):
        raise ModelCollectionError(f"unsupported model device: {device}")
    dtype = "bf16" if device in ("cuda", "xpu") else "fp32"
    args = SimpleNamespace(
        cache_dir=str(cache_dir), model=atlas.MODEL_ID, revision=atlas.MODEL_REVISION,
        device=device, dtype=dtype, threads=0,
    )
    torch, tokenizer, model, model_layers, snapshot = atlas.part38.load_model(args)
    if len(model_layers) != atlas.EXPECTED_LAYERS:
        raise ModelCollectionError(
            f"pinned model exposes {len(model_layers)} layers; expected {atlas.EXPECTED_LAYERS}"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise ModelCollectionError("pinned causal LM has no base .model module")
    return torch, tokenizer, model, base_model, model_layers, snapshot


def write_fresh(path: Path, fresh: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    """Write one JSONL row per selected unit in collect64.py's input schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for key in sorted(fresh):
            stream.write(json.dumps({"unit_key": key, "strata": fresh[key]}, ensure_ascii=False) + "\n")
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=collect64.CORPUS)
    parser.add_argument("--out", type=Path, required=True,
                        help="fresh JSONL consumed by collect64.py --fresh")
    parser.add_argument("--cache-dir", type=Path, default=atlas.part38.DEFAULT_CACHE)
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)

    corpus_meta = collect64.verify_corpus(args.data)
    units = collect64.unit_index()
    if len(units) != 500:
        raise ModelCollectionError(f"Part 60 dev split contains {len(units)} units, expected 500")
    torch, tokenizer, model, base_model, model_layers, snapshot = load_pinned_model(
        args.cache_dir, args.device
    )
    fresh = collect_fresh(
        units, data=args.data, torch=torch, tokenizer=tokenizer,
        base_model=base_model, layers=model_layers, device=args.device,
        docs_limit=collect64.PILOT_DOCS,  # Amendment 5: the pilot's 1,000 documents
        batch_size=args.batch_size,
    )
    write_fresh(args.out, fresh)
    print(json.dumps({
        "unit_count": len(fresh), "corpus": corpus_meta, "docs": collect64.PILOT_DOCS,
        "model": atlas.MODEL_ID, "revision": atlas.MODEL_REVISION,
        "snapshot": str(snapshot), "fresh_counts": FRESH_COUNTS,
        "seed": collect64.SEED, "pilot_control_seed": PILOT_CONTROL_SEED,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
