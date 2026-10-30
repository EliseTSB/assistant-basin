#!/usr/bin/env python3
"""Part 123 Stage A: sample suffixes, fit mean-contrast vectors, and score overlap."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "part92", ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common92 as common92  # noqa: E402 -- shared Part 92 model and residual-space helpers.

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
QUERY_ROOTS = (
    "What should I pack for a five-day trip to the mountains?",
    "How far in advance should I arrange transportation for a long journey?",
    "Can you suggest an easy recipe for a warm vegetable soup?",
    "How do I know when bread is fully baked?",
)
TEMPLATE_PAIRS = (
    'The requested persona is "{persona}".',
    'Please speak as "{persona}".',
    'For this reply, take the role "{persona}".',
    'Your identity for this answer is "{persona}".',
)
SUFFIX_LENGTHS = (0, 1, 3, 6)
SHUFFLE_SEED = 12301
SHUFFLES = 1000


def _read_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def selected_units(neurons: list[dict], kinds: dict) -> list[dict]:
    kind_map = kinds.get("kinds", kinds)
    units = []
    for source in neurons:
        kind = str(kind_map.get(str(source["id"]), ""))
        if kind in ("slot", "concept"):
            units.append({**source, "kind": kind})
    units.sort(key=lambda row: str(row["id"]))
    return units


def _ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _chat_ids(tokenizer, system: str | None, question: str) -> list[int]:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    return _ids(tokenizer, text)


def _set_seed(torch, device: str, seed: int) -> None:
    torch.manual_seed(int(seed))
    if str(device).startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed(int(seed))


def sample_suffixes(model, tokenizer, device: str) -> dict[str, dict]:
    """Make one shared unsteered continuation per query root, retrying short or refused output."""
    import torch

    refused = ("i cannot help", "i can't help", "i am unable to help", "i'm unable to help",
               "i cannot provide", "i can't provide", "i'm sorry, but i cannot")
    results = {}
    for query_index, question in enumerate(QUERY_ROOTS):
        prompt = _chat_ids(tokenizer, None, question)
        good = None
        last_text = ""
        for attempt in range(3):
            seed = 7 if attempt == 0 else 7 + 1_000_003 * (query_index + 1) + attempt
            _set_seed(torch, device, seed)
            input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids, do_sample=True, temperature=0.7, max_new_tokens=64,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                )
            new_ids = [int(token) for token in generated[0, len(prompt):].tolist()]
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and int(eos) in new_ids:
                new_ids = new_ids[:new_ids.index(int(eos))]
            last_text = str(tokenizer.decode(new_ids, skip_special_tokens=True)).strip()
            if len(new_ids) >= 6 and last_text and not any(term in last_text.lower() for term in refused):
                good = new_ids
                break
        if good is None:
            raise RuntimeError(f"no usable six-token suffix for query {query_index}: {last_text[:120]!r}")
        results[str(query_index)] = {
            "question": question,
            "seed": 7 if attempt == 0 else 7 + 1_000_003 * (query_index + 1) + attempt,
            "retry_count": attempt,
            "continuation_text": last_text,
            "continuation_ids": good,
            "suffixes": {str(length): good[:length] for length in SUFFIX_LENGTHS},
        }
    return results


def _padded_batch(torch, tokenizer, sequences: list[list[int]], device: str):
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    width = max(map(len, sequences))
    ids = [[int(pad_id)] * (width - len(row)) + row for row in sequences]
    mask = [[0] * (width - len(row)) + [1] * len(row) for row in sequences]
    return (torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(mask, dtype=torch.long, device=device))


def capture_last_states(model, tokenizer, layer, specs: list[dict], batch_size: int, device: str):
    """Capture only each prompt's last-token state, transferring once per hook call."""
    import torch

    sequences = [spec["input_ids"] for spec in specs]
    states = []

    def save_last(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        states.append(hidden[:, -1, :].detach().to(device="cpu"))

    handle = layer.register_forward_hook(save_last)
    try:
        with torch.inference_mode():
            for start in range(0, len(specs), batch_size):
                ids, mask = _padded_batch(torch, tokenizer, sequences[start:start + batch_size], device)
                model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        handle.remove()
    if len(states) != math.ceil(len(specs) / batch_size):
        raise RuntimeError("fit hook did not capture one last-token batch per model call")
    return torch.cat(states, dim=0).float()


def prompt_specs(tokenizer, units: list[dict], suffixes: dict[str, dict]) -> dict[int, list[dict]]:
    """Build 128 paired prompts per unit; each of 16 groups has four suffix siblings."""
    by_layer: dict[int, list[dict]] = {}
    for unit in units:
        layer_no = int(unit["layer"])
        specs = by_layer.setdefault(layer_no, [])
        for query_index, question in enumerate(QUERY_ROOTS):
            suffix_map = suffixes[str(query_index)]["suffixes"]
            for template_index, template in enumerate(TEMPLATE_PAIRS):
                group = query_index * len(TEMPLATE_PAIRS) + template_index
                for suffix_index, suffix_length in enumerate(SUFFIX_LENGTHS):
                    suffix = [int(token) for token in suffix_map[str(suffix_length)]]
                    for side, persona in (("positive", str(unit["persona"])),
                                          ("negative", "a generic helpful assistant")):
                        system = template.replace("{persona}", persona)
                        ids = _chat_ids(tokenizer, system, question) + suffix
                        specs.append({"unit_id": str(unit["id"]), "group": group,
                                      "suffix_index": suffix_index, "side": side,
                                      "input_ids": ids})
    return by_layer


def fit_unit_states(states, specs: list[dict], unit_id: str):
    """Return a unit vector and grouped positive-minus-negative states for null shuffles."""
    import torch

    positive = [[None] * len(SUFFIX_LENGTHS) for _ in range(16)]
    negative = [[None] * len(SUFFIX_LENGTHS) for _ in range(16)]
    for state, spec in zip(states, specs):
        if spec["unit_id"] != unit_id:
            continue
        target = positive if spec["side"] == "positive" else negative
        target[int(spec["group"])][int(spec["suffix_index"])] = state
    if any(item is None for group in positive + negative for item in group):
        raise ValueError(f"incomplete fit pairs for {unit_id}")
    pos = torch.stack([torch.stack(group) for group in positive])
    neg = torch.stack([torch.stack(group) for group in negative])
    group_diffs = pos - neg
    raw = group_diffs.mean(dim=(0, 1))
    norm = torch.linalg.vector_norm(raw)
    if float(norm) == 0.0:
        raise ValueError(f"zero mean-contrast vector for {unit_id}")
    return raw / norm, group_diffs


def overlap_row(vector, group_diffs, unit: dict, weight, shuffle_count: int = SHUFFLES,
                seed: int = SHUFFLE_SEED) -> tuple[dict, list[float]]:
    """Score the matched channel, layer rank, and a grouped label-shuffle cosine null."""
    import torch

    weight_cpu = weight.detach().to(device="cpu", dtype=torch.float32)
    vector = vector.detach().to(device="cpu", dtype=torch.float32)
    if weight_cpu.ndim != 2 or weight_cpu.shape[0] != vector.numel():
        raise ValueError("down_proj must map its channels into the fitted residual width")
    channel = int(unit["neuron"])
    if not 0 <= channel < weight_cpu.shape[1]:
        raise ValueError(f"matching channel is outside layer {unit['layer']}")
    columns = weight_cpu
    column_norms = torch.linalg.vector_norm(columns, dim=0)
    cosine_all = (vector @ columns) / (torch.linalg.vector_norm(vector) * column_norms.clamp_min(1e-20))
    matched = float(cosine_all[channel])
    channel_ids = torch.arange(int(columns.shape[1]))
    tied_before = (cosine_all == cosine_all[channel]) & (channel_ids < channel)
    rank = 1 + int((cosine_all > cosine_all[channel]).sum()) + int(tied_before.sum())
    group_means = group_diffs.sum(dim=1) / len(SUFFIX_LENGTHS)
    rng = random.Random(int(seed))
    signs = []
    for _ in range(int(shuffle_count)):
        reversed_groups = set(rng.sample(range(16), 8))
        signs.append([-1.0 if i in reversed_groups else 1.0 for i in range(16)])
    signed = torch.tensor(signs, dtype=torch.float32)
    shuffled = (signed @ group_means) / 16.0
    matched_column = columns[:, channel]
    denom = torch.linalg.vector_norm(shuffled, dim=1) * torch.linalg.vector_norm(matched_column)
    null_cosines = (shuffled @ matched_column) / denom.clamp_min(1e-20)
    null_values = [float(value) for value in null_cosines.tolist()]
    ordered = sorted(null_values)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None
    channels = int(columns.shape[1])
    top_decile = rank <= math.ceil(0.10 * channels)
    row = {
        "unit_id": str(unit["id"]), "persona": str(unit["persona"]), "kind": str(unit["kind"]),
        "layer": int(unit["layer"]), "channel": channel, "channel_count": channels,
        "signed_cosine": matched, "rank": rank, "top_decile_rank_limit": math.ceil(0.10 * channels),
        "top_decile": top_decile, "shuffle_seed": int(seed), "shuffle_count": len(null_values),
        "shuffle_p95_signed_cosine": p95,
        "above_shuffle_p95": bool(p95 is not None and matched > p95),
    }
    return row, null_values


def main(argv: list[str] | None = None, *, model_loader=None, validate_inventory: bool = True) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local pinned OLMo 2 1B Instruct snapshot")
    parser.add_argument("--neurons", default="results/part92/step2/neurons.json")
    parser.add_argument("--kinds", default="results/part92/step2/kinds.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffles", type=int, default=SHUFFLES)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.shuffles < 1:
        parser.error("--batch-size and --shuffles must be positive")

    units = selected_units(_read_json(args.neurons), _read_json(args.kinds))
    counts = {kind: sum(unit["kind"] == kind for unit in units) for kind in ("slot", "concept")}
    if validate_inventory and (len(units) != 39 or counts != {"slot": 21, "concept": 18}):
        raise ValueError(f"expected 39 selected Part 92 units (21 slot, 18 concept), found {len(units)} {counts}")

    loader = model_loader or common92.load_model
    tokenizer, model = loader(args.model, args.device)
    model.eval()
    layers = common92.get_layers(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    suffixes = sample_suffixes(model, tokenizer, args.device)
    (out / "suffixes.json").write_text(json.dumps(suffixes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    specs_by_layer = prompt_specs(tokenizer, units, suffixes)
    vectors = {}
    overlap_rows = []
    null_rows = []
    with torch.inference_mode():
        for layer_no in sorted(specs_by_layer):
            specs = specs_by_layer[layer_no]
            states = capture_last_states(model, tokenizer, layers[layer_no], specs, args.batch_size, args.device)
            layer_units = [unit for unit in units if int(unit["layer"]) == layer_no]
            weight = layers[layer_no].mlp.down_proj.weight
            for unit in layer_units:
                uid = str(unit["id"])
                vector, group_diffs = fit_unit_states(states, specs, uid)
                vectors[uid] = vector.contiguous()
                overlap, null_values = overlap_row(vector, group_diffs, unit, weight, args.shuffles)
                overlap_rows.append(overlap)
                null_rows.append({"unit_id": uid, "cosines": null_values})
            print(f"layer {layer_no}: fitted {len(layer_units)} units", flush=True)

    torch.save({"model_id": MODEL_ID, "revision": REVISION, "vectors": vectors}, out / "vectors.pt")
    _jsonl(out / "overlap.jsonl", overlap_rows)
    _jsonl(out / "shuffled-null.jsonl", null_rows)
    metadata = {
        "stage": "A", "model": MODEL_ID, "revision": REVISION, "units": len(units),
        "slot_units": counts["slot"], "concept_units": counts["concept"],
        "suffix_samples": len(suffixes), "fit_prompts": len(units) * 4 * 4 * 4 * 2,
        "fit_states_stored": "last token only", "shuffle_seed": SHUFFLE_SEED,
        "shuffles_per_unit": args.shuffles, "batch_size": args.batch_size,
        "overlap_rows": len(overlap_rows), "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (out / "fit-meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
