#!/usr/bin/env python3
"""Measure two forced apology-token probabilities under the Part 139 phase hook."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART139 = ROOT / "part139"
for candidate in (str(PART139), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import run139  # noqa: E402

QUESTION = "Who are you?"
ARMS = ("noop", "prompt-only", "answer-only", "full")
READOUTS = {
    "im_after_i": {"prefix": "I", "target": "'m"},
    "sorry_after_im": {"prefix": "I'm", "target": " sorry"},
}


def _one_token(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"registered target {text!r} must tokenize to one token, got {ids}")
    return int(ids[0])


def _prefix_ids(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"forced prefix {text!r} produced no tokens")
    return [int(token) for token in ids]


def probability(model, tokenizer, layer, vector, arm, prefix_text, target_id, device):
    """Prefill prompt, then teacher-force prefix tokens through the selected phase."""
    import torch

    prompt_ids = run139.c.prompt_ids(tokenizer, QUESTION, "chat")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    hook_vector = None if vector is None else vector.reshape(1, -1)
    handle = run139._position_hook(layer, hook_vector, arm) if hook_vector is not None else None
    try:
        with torch.inference_mode():
            output = model(input_ids=ids, use_cache=True)
            cache = output.past_key_values
            logits = output.logits
            for token in _prefix_ids(tokenizer, prefix_text):
                token_ids = torch.tensor([[token]], dtype=torch.long, device=device)
                output = model(input_ids=token_ids, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                logits = output.logits
            probability_value = torch.softmax(logits[0, -1].float(), dim=-1)[target_id]
            return float(probability_value.item())
    finally:
        if handle is not None:
            handle.remove()


def _read_seen(path):
    return run139._seen(path)


def _append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def _load_direct(snapshot, device):
    """Load bf16 weights directly onto the requested device, without a CPU staging copy."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(snapshot).expanduser()), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(snapshot).expanduser()), torch_dtype=torch.bfloat16,
        local_files_only=True, device_map=device,
    )
    model.requires_grad_(False)
    model.eval()
    return tokenizer, model


def run(args):
    import torch

    units_data = json.loads(Path(args.units).read_text(encoding="utf-8"))
    units = units_data["sets"]["R"] + units_data["sets"]["S"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.jsonl"
    seen = _read_seen(path)

    tokenizer, model = _load_direct(args.model, args.device)
    target_ids = {name: _one_token(tokenizer, spec["target"]) for name, spec in READOUTS.items()}
    layers = run139.c.get_layers(model)
    r_l = {
        layer_no: json.loads((ROOT / f"results/part92/screen-instruct/layer_{layer_no}.meta.json").read_text(encoding="utf-8"))["r_L"]
        for layer_no in (4, 5)
    }
    args.r_l = r_l

    plain_layer = layers[4]
    for readout, spec in READOUTS.items():
        key = f"{readout}|plain"
        if key not in seen:
            value = probability(model, tokenizer, plain_layer, None, "full", spec["prefix"], target_ids[readout], args.device)
            _append(path, {"key": key, "readout": readout, "arm": "plain", "set": "plain", "unit": "plain", "unit_id": "plain", "layer": None, "probability": value, "target": spec["target"], "forced_prefix": spec["prefix"], "question": QUESTION})
            seen.add(key)

    for unit in units:
        layer_no = int(unit["layer"])
        layer = layers[layer_no]
        weight = layer.mlp.down_proj.weight
        vector = run139._unit_vectors([unit], layer_no, weight, args.r_l[layer_no])[0]
        noop_vector = torch.zeros_like(vector)
        for arm in ARMS:
            for readout, spec in READOUTS.items():
                key = f"{readout}|{arm}|{unit['id']}"
                if key in seen:
                    continue
                applied_vector = noop_vector if arm == "noop" else vector
                value = probability(model, tokenizer, layer, applied_vector, arm, spec["prefix"], target_ids[readout], args.device)
                row = {
                    "key": key, "readout": readout, "arm": arm,
                    "set": unit["set"], "unit": unit["unit"], "index": unit["index"],
                    "unit_id": unit["id"], "layer": layer_no,
                    "probability": value, "target": spec["target"],
                    "forced_prefix": spec["prefix"], "question": QUESTION,
                    "dose": 0.8, "r_L": args.r_l[layer_no],
                }
                _append(path, row)
                seen.add(key)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", required=True)
    parser.add_argument("--units", default=str(HERE / "../part139/units139.json"))
    parser.add_argument("--out", default="results/part143")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
