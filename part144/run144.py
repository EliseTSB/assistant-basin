#!/usr/bin/env python3
"""Measure Part 144's forced apology-token probabilities across six checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for folder in (ROOT, ROOT / "part139", ROOT / "part92"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import run139  # noqa: E402
import common92 as c  # noqa: E402

QUESTION = "Who are you?"
CELLS = ("base-plain", "base-chat", "sft-chat", "dpo-chat", "instruct-chat", "instruct-plain")
CHECKPOINTS = {
    "base-plain": "pretrain_4T", "base-chat": "pretrain_4T", "sft-chat": "sft",
    "dpo-chat": "dpo", "instruct-chat": "instruct", "instruct-plain": "instruct",
}
ARMS = ("full", "prompt-only", "answer-only")
SETS = ("R", "S", "F")


def _one_token(tok, value):
    ids = tok(value, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{value!r} must be exactly one token, got {ids}")
    return int(ids[0])


def _top5(logits, tok):
    import torch
    probs = torch.softmax(logits.float(), dim=-1)
    vals, ids = torch.topk(probs, k=min(5, probs.numel()))
    return [[tok.decode([int(i)], skip_special_tokens=False), float(p)] for p, i in zip(vals.tolist(), ids.tolist())]


def sequence(model, tok, layer, vector, arm, frame, device):
    """Read the three nested probabilities while one fresh hook spans the sequence."""
    import torch
    first_text = "I" if frame == "chat" else " I"
    first_id = _one_token(tok, first_text)
    m_id = _one_token(tok, "'m")
    sorry_id = _one_token(tok, " sorry")
    im_ids = [int(x) for x in tok("I'm" if frame == "chat" else " I'm", add_special_tokens=False)["input_ids"]]
    if im_ids != [first_id, m_id]:
        raise ValueError(f"forced tokens do not match tokenizer for {frame}: {im_ids} != {[first_id, m_id]}")
    ids = torch.tensor([c.prompt_ids(tok, QUESTION, frame)], dtype=torch.long, device=device)
    handle = run139._position_hook(layer, vector.reshape(1, -1), arm) if vector is not None else None
    try:
        with torch.inference_mode():
            output = model(input_ids=ids, use_cache=True)
            cache = output.past_key_values
            logits_first = output.logits[0, -1].float()
            p_first = float(torch.softmax(logits_first, -1)[first_id].item())
            output = model(input_ids=torch.tensor([[first_id]], dtype=torch.long, device=device), past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits_m = output.logits[0, -1].float()
            p_m = float(torch.softmax(logits_m, -1)[m_id].item())
            output = model(input_ids=torch.tensor([[m_id]], dtype=torch.long, device=device), past_key_values=cache, use_cache=True)
            logits_sorry = output.logits[0, -1].float()
            p_sorry = float(torch.softmax(logits_sorry, -1)[sorry_id].item())
            return {"p_first": p_first, "p_m": p_m, "p_sorry": p_sorry,
                    "top5_first": _top5(logits_first, tok), "top5_m": _top5(logits_m, tok),
                    "top5_sorry": _top5(logits_sorry, tok),
                    "token_checks": {"first": first_text, "m": "'m", "sorry": " sorry", "im_ids": im_ids}}
    finally:
        if handle is not None:
            handle.remove()


def _read_seen(path):
    if not path.exists():
        return set()
    seen = set()
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    offset = 0
    for n, line in enumerate(lines):
        start = offset
        offset += len(line)
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if n == len(lines) - 1:
                with path.open("r+b") as stream:
                    stream.truncate(start)
                break
            raise
        seen.add(row["key"])
    return seen


def _append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def _load_direct(snapshot, device):
    """Load bf16 weights directly onto the requested device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path(snapshot).expanduser()), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(Path(snapshot).expanduser()), torch_dtype=torch.bfloat16,
                                                   local_files_only=True, device_map=device)
    model.requires_grad_(False)
    model.eval()
    return tok, model


def run(args):
    import torch
    if args.cell not in CELLS:
        raise ValueError(f"unknown cell {args.cell}")
    cell = args.cell
    frame = "plain" if cell.endswith("plain") else "chat"
    data = json.loads(Path(args.units).read_text(encoding="utf-8"))
    sets = data["sets"]
    units_by_set = {"R": sets["R"], "S": sets["S"], "F": data["stage_f"]["units"]}
    path_out = Path(args.out)
    path_out.mkdir(parents=True, exist_ok=True)
    path = path_out / (f"pilot-{cell}.jsonl" if args.pilot else f"{cell}.jsonl")
    seen = _read_seen(path)
    start = time.monotonic()
    tok, model = _load_direct(args.model, args.device)
    if args.tokenizer:
        # Amendment 1: Base ships no chat template. Its vocabulary and added tokens equal Instruct's, so Instruct's tokenizer renders the frame.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(Path(args.tokenizer).expanduser()), local_files_only=True)
    token_ids = [_one_token(tok, "I" if frame == "chat" else " I"), _one_token(tok, "'m"), _one_token(tok, " sorry")]
    if [int(x) for x in tok("I'm" if frame == "chat" else " I'm", add_special_tokens=False)["input_ids"]] != token_ids[:2]:
        raise ValueError("tokenization check failed for forced I'm prefix")
    layers = c.get_layers(model)
    if cell == "instruct-chat":
        r_l = {l: json.loads((ROOT / f"results/part92/screen-instruct/layer_{l}.meta.json").read_text(encoding="utf-8"))["r_L"] for l in (4, 5)}
    else:
        med = c.residual_medians(model, tok, frame, args.device)
        r_l = {l: med[l] for l in (4, 5)}
    ordered = []
    for set_name in SETS:
        selected = units_by_set[set_name]
        if args.pilot:
            selected = selected[:12] if set_name == "R" else []
        ordered.extend((set_name, unit) for unit in selected)
    # The plain row is one unpushed baseline per cell.
    plain_key = f"{cell}|plain"
    if plain_key not in seen:
        vals = sequence(model, tok, layers[4], None, "full", frame, args.device)
        _append(path, {"key": plain_key, "cell": cell, "frame": frame, "checkpoint": CHECKPOINTS[cell], "arm": "plain", "set": "plain", "unit": "plain", "unit_id": "plain", "index": None, "layer": None, "dose": 0.8, "r_L": None, "question": QUESTION, **vals})
        seen.add(plain_key)
    for pos, (set_name, unit) in enumerate(ordered):
        layer_no = int(unit["layer"])
        weight = layers[layer_no].mlp.down_proj.weight
        seed = 139100 + layer_no if set_name == "F" else None
        vector = run139._unit_vectors([unit], layer_no, weight, r_l[layer_no], alpha=0.8, random_seed=seed)[0]
        arms = list(ARMS)
        if pos < 12:
            arms.append("noop")
        for arm in arms:
            key = f"{cell}|{set_name}|{unit['id']}|{arm}"
            if key in seen:
                continue
            applied = torch.zeros_like(vector) if arm == "noop" else vector
            vals = sequence(model, tok, layers[layer_no], applied, arm, frame, args.device)
            row = {"key": key, "cell": cell, "frame": frame, "checkpoint": CHECKPOINTS[cell], "arm": arm,
                   "set": set_name, "unit": unit["unit"], "index": unit["index"], "unit_id": unit["id"],
                   "layer": layer_no, "dose": 0.8, "r_L": r_l[layer_no], "question": QUESTION, **vals}
            _append(path, row)
            seen.add(key)
    elapsed = time.monotonic() - start
    timing = path_out / f"timing-{cell}.json"
    timing.write_text(json.dumps({"cell": cell, "rows": len(seen), "seconds": elapsed}, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--model", required=True)
    ap.add_argument("--units", default=str(HERE / "../part139/units139.json"))
    ap.add_argument("--out", default="results/part144")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--tokenizer", help="snapshot whose tokenizer renders the prompt; base-chat only")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
