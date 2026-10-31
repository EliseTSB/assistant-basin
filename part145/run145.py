#!/usr/bin/env python3
"""Store full next-token distributions for Part 145."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # Help and isolated import work before the numerical runtime is loaded.
    np = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for folder in (ROOT, ROOT / "part144"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import run144  # noqa: E402

CELLS = run144.CELLS
CHECKPOINTS = run144.CHECKPOINTS
QUESTIONS = ("Who are you?", "Tell me about yourself.")


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, value.astype(np.float32, copy=False), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sequence(model, tok, layer, vector, question, frame, device):
    """Run prompt, forced first token and forced 'm' under one hook."""
    import torch

    first_text = "I" if frame == "chat" else " I"
    first_id = run144._one_token(tok, first_text)
    m_id = run144._one_token(tok, "'m")
    sorry_id = run144._one_token(tok, " sorry")
    forced = "I'm" if frame == "chat" else " I'm"
    if [int(x) for x in tok(forced, add_special_tokens=False)["input_ids"]] != [first_id, m_id]:
        raise ValueError(f"forced tokens do not match tokenizer for {frame}")
    import common92 as c

    prompt = torch.tensor([c.prompt_ids(tok, question, frame)], dtype=torch.long, device=device)
    handle = run144.run139._position_hook(layer, vector.reshape(1, -1), "full") if vector is not None else None
    try:
        with torch.inference_mode():
            out = model(input_ids=prompt, use_cache=True)
            cache = out.past_key_values
            out = model(input_ids=torch.tensor([[first_id]], device=device), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            out = model(input_ids=torch.tensor([[m_id]], device=device), past_key_values=cache, use_cache=True)
            logits = out.logits[0, -1].float().detach().cpu()
            log_probs = torch.log_softmax(logits, dim=-1).numpy().astype(np.float32, copy=False)
    finally:
        if handle is not None:
            handle.remove()
    if not np.isfinite(log_probs).all():
        raise FloatingPointError("non-finite log-probability; Gate 3 is a hard stop")
    return log_probs, {"sorry": sorry_id, "first": first_id, "m": m_id}


def _complete(matrix_path, index_path, units):
    if not matrix_path.is_file() or not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        arr = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        expected = [u["id"] for u in units]
        return (arr.ndim == 2 and arr.shape[0] == len(expected)
                and arr.shape[1] == int(index.get("vocab_size", -1))
                and index.get("unit_ids") == expected)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def run(args):
    if np is None:
        raise RuntimeError("NumPy is required to run Part 145")
    import torch
    import common92 as c

    if args.cell not in CELLS:
        raise ValueError(f"unknown cell {args.cell}")
    cell = args.cell
    frame = "plain" if cell.endswith("plain") else "chat"
    data = json.loads(Path(args.units).read_text(encoding="utf-8"))
    units = data["stage_f"]["units"]
    if args.limit is not None:
        units = units[:args.limit]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tok, model = run144._load_direct(args.model, args.device)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(Path(args.tokenizer).expanduser()), local_files_only=True)
    layers = c.get_layers(model)
    if cell == "instruct-chat":
        r_l = {layer: json.loads((ROOT / f"results/part92/screen-instruct/layer_{layer}.meta.json").read_text(encoding="utf-8"))["r_L"] for layer in (4, 5)}
    else:
        r_l = c.residual_medians(model, tok, frame, args.device)
    sorry_id = run144._one_token(tok, " sorry")
    first_text = "I" if frame == "chat" else " I"
    first_id = run144._one_token(tok, first_text)
    m_id = run144._one_token(tok, "'m")
    if [int(x) for x in tok("I'm" if frame == "chat" else " I'm", add_special_tokens=False)["input_ids"]] != [first_id, m_id]:
        raise ValueError("tokenization check failed for forced I'm prefix")
    started = time.monotonic()
    for qi, question in enumerate(QUESTIONS, 1):
        matrix_path = outdir / f"matrix-{cell}-q{qi}.npy"
        index_path = outdir / f"index-{cell}-q{qi}.json"
        if _complete(matrix_path, index_path, units):
            continue
        plain_path = outdir / f"plain-{cell}-q{qi}.npy"
        plain, _ = sequence(model, tok, layers[4], None, question, frame, args.device)
        if not np.isfinite(plain).all():
            raise FloatingPointError("non-finite plain log-probability; Gate 3 is a hard stop")
        _atomic_npy(plain_path, plain[None, :])
        top_id = int(np.argmax(plain))
        rows = np.empty((len(units), plain.size), dtype=np.float32)
        noops = []
        metadata = []
        by_layer = {}
        for layer_no in (4, 5):
            layer_units = [unit for unit in units if int(unit["layer"]) == layer_no]
            if layer_units:
                vectors = run144.run139._unit_vectors(layer_units, layer_no,
                    layers[layer_no].mlp.down_proj.weight, r_l[layer_no], alpha=0.8,
                    random_seed=139100 + layer_no)
                by_layer.update({unit["id"]: vector for unit, vector in zip(layer_units, vectors)})
        for pos, unit in enumerate(units):
            layer_no = int(unit["layer"])
            vector = by_layer[unit["id"]]
            rows[pos], _ = sequence(model, tok, layers[layer_no], vector, question, frame, args.device)
            metadata.append({"id": unit["id"], "unit": unit["unit"], "layer": layer_no})
            if pos < 12:
                noop, _ = sequence(model, tok, layers[layer_no], torch.zeros_like(vector), question, frame, args.device)
                noops.append({"unit_id": unit["id"], "unit": unit["unit"], "layer": layer_no,
                              "logp_sorry": float(noop[sorry_id]), "logp_plain_top": float(noop[top_id]),
                              "plain_top_id": top_id})
        if not np.isfinite(rows).all():
            raise FloatingPointError("non-finite matrix row; Gate 3 is a hard stop")
        # Include the eligible token strings for the scorer's fixed word set.
        eligible = np.flatnonzero(np.exp(plain.astype(np.float64)) >= 1e-5)
        tokens = [str(x) for x in tok.convert_ids_to_tokens(eligible.tolist())]
        sidecar = {"cell": cell, "question": question, "q": qi, "unit_ids": [u["id"] for u in units],
                   "units": metadata, "vocab_size": int(plain.size), "sorry_id": sorry_id,
                   "plain_top_id": top_id, "eligible_ids": eligible.tolist(), "eligible_tokens": tokens}
        noop_path = outdir / f"noop-{cell}-q{qi}.jsonl"
        tmp_noop = noop_path.with_suffix(noop_path.suffix + ".tmp")
        tmp_noop.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in noops), encoding="utf-8")
        os.replace(tmp_noop, noop_path)
        _atomic_npy(matrix_path, rows)
        tmp = index_path.with_suffix(index_path.suffix + ".tmp")
        tmp.write_text(json.dumps(sidecar, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, index_path)
    (outdir / f"timing-{cell}.json").write_text(
        json.dumps({"cell": cell, "seconds": time.monotonic() - started}, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--model", required=True)
    ap.add_argument("--units", default=str(ROOT / "part139/units139.json"))
    ap.add_argument("--out", default="results/part145")
    ap.add_argument("--tokenizer", help="tokenizer snapshot for rendering base-chat")
    ap.add_argument("--limit", type=int, help="limit Set F units for tests")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
