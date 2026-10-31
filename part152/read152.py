#!/usr/bin/env python3
"""Collect per-document activation maxima and minima for the Part 152 lens units."""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_TOKENS, KEEP, BEFORE, AFTER = 256, 20, 30, 12
CONTROL = (22, 2742)


def unit_name(layer, neuron):
    return f"L{layer}N{neuron}"


def read_units(lens_path):
    lens = json.loads(Path(lens_path).read_text(encoding="utf-8"))
    units = sorted((tuple(map(int, key.split(":"))) for key in lens), key=lambda x: x)
    if CONTROL not in units:
        units.append(CONTROL)
        units.sort()
    return units


def pile_documents(path, start, count):
    """Read a bounded slice while retaining each document's absolute line index."""
    selected = []
    with open(path, encoding="utf-8") as fh:
        for absolute, line in enumerate(fh):
            if absolute < start:
                continue
            if len(selected) >= count:
                break
            if line.strip():
                row = json.loads(line)
                selected.append({"i": absolute, "text": row["text"]})
    return selected


def _decode_windows(tok, all_ids, specs):
    """Decode all context segments in batches; specs contain (unit, side, act, doc, pos)."""
    out = {name: {"top": [], "bottom": []} for name in all_ids["_units"]}
    ids = all_ids
    batch_size = 4096
    for start in range(0, len(specs), batch_size):
        part = specs[start:start + batch_size]
        before_ids, mark_ids, after_ids = [], [], []
        for _name, _side, _act, doc, pos in part:
            row = ids[doc]
            before_ids.append(row[max(0, pos - BEFORE):pos])
            mark_ids.append(row[pos:pos + 1])
            after_ids.append(row[pos + 1:pos + 1 + AFTER])
        before = tok.batch_decode(before_ids, skip_special_tokens=True)
        marks = tok.batch_decode(mark_ids, skip_special_tokens=True)
        after = tok.batch_decode(after_ids, skip_special_tokens=True)
        for spec, b, m, a in zip(part, before, marks, after):
            name, side, act, doc, pos = spec
            out[name][side].append({"act": float(act), "doc": int(doc), "pos": int(pos),
                                    "before": b, "mark": m, "after": a})
    for unit in out.values():
        for side in ("top", "bottom"):
            unit[side].sort(key=lambda e: e["act"], reverse=(side == "top"))
    return out


def collect(model, tok, downs, docs, units, batch_size, device, torch, log=print):
    """Capture only requested columns and maintain fixed-size top and bottom arrays."""
    by_layer = {}
    for layer, neuron in units:
        by_layer.setdefault(layer, []).append(neuron)
    capture = {}
    handles = []
    for layer, neurons in by_layer.items():
        columns = np.asarray(neurons, dtype=np.int64)

        def grab(_module, args, layer=layer, columns=columns):
            capture[layer] = args[0][:, :, columns].float().cpu().numpy()
        handles.append(downs[layer].register_forward_pre_hook(grab))

    names = [unit_name(*u) for u in units]
    index = {u: names.index(unit_name(*u)) for u in units}
    top_v = np.full((len(units), KEEP), -np.inf, dtype=np.float32)
    top_d = np.full((len(units), KEEP), -1, dtype=np.int64)
    top_p = np.full((len(units), KEEP), -1, dtype=np.int32)
    bot_v = np.full((len(units), KEEP), np.inf, dtype=np.float32)
    bot_d = np.full((len(units), KEEP), -1, dtype=np.int64)
    bot_p = np.full((len(units), KEEP), -1, dtype=np.int32)
    all_ids = {"_units": names}
    tokens = 0
    t0 = time.time()
    try:
        for offset in range(0, len(docs), batch_size):
            batch = docs[offset:offset + batch_size]
            enc = tok([d["text"] for d in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_TOKENS)
            mask = enc["attention_mask"].numpy().astype(bool)
            ids = enc["input_ids"]
            with torch.inference_mode():
                model(input_ids=ids.to(device), attention_mask=enc["attention_mask"].to(device), use_cache=False)
            tokens += int(mask.sum())
            for row, doc in enumerate(batch):
                all_ids[doc["i"]] = ids[row][enc["attention_mask"][row].bool()].to("cpu").numpy().astype(np.int32)

            for layer, neurons in by_layer.items():
                vals = capture[layer]
                valid = mask[:, :, None]
                hi_grid = np.where(valid, vals, -np.inf)
                lo_grid = np.where(valid, vals, np.inf)
                hi_pos = np.argmax(hi_grid, axis=1)
                lo_pos = np.argmin(lo_grid, axis=1)
                hi_val = np.max(hi_grid, axis=1)
                lo_val = np.min(lo_grid, axis=1)
                unit_indices = np.asarray([index[(layer, n)] for n in neurons])
                for vals_now, pos_now, heaps_v, heaps_d, heaps_p, high in (
                        (hi_val, hi_pos, top_v, top_d, top_p, True),
                        (lo_val, lo_pos, bot_v, bot_d, bot_p, False)):
                    threshold = heaps_v[unit_indices, -1]
                    eligible = vals_now > threshold[None, :] if high else vals_now < threshold[None, :]
                    rows, cols = np.nonzero(eligible)
                    if not rows.size:
                        continue
                    # Sparse candidate updates avoid work for units no document can improve.
                    for col in np.unique(cols):
                        unit_idx = unit_indices[col]
                        chosen = rows[cols == col]
                        current_v = heaps_v[unit_idx]
                        current_d = heaps_d[unit_idx]
                        current_p = heaps_p[unit_idx]
                        candidate_v = vals_now[chosen, col]
                        candidate_d = np.asarray([batch[r]["i"] for r in chosen], dtype=np.int64)
                        candidate_p = pos_now[chosen, col].astype(np.int32)
                        merged_v = np.concatenate((current_v, candidate_v))
                        merged_d = np.concatenate((current_d, candidate_d))
                        merged_p = np.concatenate((current_p, candidate_p))
                        order = np.argsort(merged_v, kind="stable")
                        keep = order[-KEEP:] if high else order[:KEEP]
                        # Store strongest first for both arrays; bottom means numerically smallest.
                        if high:
                            keep = keep[::-1]
                        else:
                            keep = keep
                        heaps_v[unit_idx] = merged_v[keep]
                        heaps_d[unit_idx] = merged_d[keep]
                        heaps_p[unit_idx] = merged_p[keep]
            if ((offset + len(batch)) // 500 > offset // 500) or offset + len(batch) == len(docs):
                log(f"{offset + len(batch)} documents, {tokens} tokens, {time.time() - t0:.0f}s")
    finally:
        for handle in handles:
            handle.remove()

    specs = []
    for idx, name in enumerate(names):
        for j in range(KEEP):
            if top_d[idx, j] >= 0:
                specs.append((name, "top", top_v[idx, j], int(top_d[idx, j]), int(top_p[idx, j])))
            if bot_d[idx, j] >= 0:
                specs.append((name, "bottom", bot_v[idx, j], int(bot_d[idx, j]), int(bot_p[idx, j])))
    result = _decode_windows(tok, all_ids, specs)
    return result, tokens


def run(a, torch=None, nr=None):
    if torch is None:
        import torch
    if nr is None:
        import neuron_report as nr
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = pile_documents(a.pile, a.start, a.docs)
    units = read_units(a.lens)
    t0 = time.time()
    tok, model = nr.load_model(MODEL_ID, REVISION, a.device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    _, layers, _ = nr.find_text_layers(model)
    downs = {layer: layers[layer].mlp.down_proj for layer in sorted({l for l, _ in units})}
    result, tokens = collect(model, tok, downs, docs, units, a.batch_size, a.device, torch,
                            log=lambda s: print(s, flush=True))
    meta = {"model": MODEL_ID, "revision": REVISION, "docs": len(docs), "tokens": tokens,
            "seconds": round(time.time() - t0, 1), "start": a.start, "batch_size": a.batch_size,
            "max_tokens": MAX_TOKENS, "units": result}
    with gzip.open(out / "windows.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")
    print(json.dumps({k: meta[k] for k in ("docs", "tokens", "seconds")}), flush=True)


def main(argv=None, torch=None, nr=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=10000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--lens", default=str(HERE / "lens.json"))
    ap.add_argument("--units-extra", default="none")
    run(ap.parse_args(argv), torch=torch, nr=nr)


if __name__ == "__main__":
    main()
