#!/usr/bin/env python3
"""Read Part 126's head writes through the Part 111 Jacobian lens."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAYERS = (11, 15)
HEADS = 16
COPIES = ("A", "U")
COPY_INDEX = {name: i for i, name in enumerate(COPIES)}
ROLE_REGION = 0
REGION_NAMES = ("role", "template", "body", "question", "options", "other")
TOP_K = 20
PREFIX = "model.language_model."


def _write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class StubLensTokenizer:
    def decode(self, ids, **kwargs):
        names = [" I", " you", " assistant", " me", " my", " we", " your", " answer", " copy"]
        return "".join(names[int(i)] if 0 <= int(i) < len(names) else f" tok{int(i)}" for i in ids)


def _stub_inputs(hidden: int):
    torch.manual_seed(12611)
    vocab = 96
    embed = torch.randn(vocab, hidden, dtype=torch.float32)
    gain = torch.ones(hidden, dtype=torch.float32)
    js = torch.eye(hidden, dtype=torch.float32).repeat(31, 1, 1)
    projections = {}
    for layer in LAYERS:
        projections[layer] = torch.randn(hidden, HEADS * 2, dtype=torch.float32) / (hidden ** 0.5)
    return embed, gain, js, projections, StubLensTokenizer()


def _tokens_for(tok, top_ids, low_ids):
    top = [tok.decode([int(i)], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in top_ids]
    low = [tok.decode([int(i)], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in low_ids]
    return {"top20_written_tokens": top, "top20_suppressed_tokens": low}


def run(args):
    t0 = time.time()
    from part126.reads126 import _jsonl_rows
    from part111 import jlens111

    raw_path = Path(args.activations)
    run_path = Path(args.run_json or raw_path.parent / "run.json")
    rows_path = Path(args.rows or raw_path.parent / "rows.jsonl")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(raw_path, allow_pickle=False) as z:
        region_mass = z["region_mass"].astype(np.float32)
        region_values = z["region_values"].astype(np.float32)
        body_values = z["random_body_values"].astype(np.float32)
        row_indexes = z["row_indexes"].astype(np.int32)
        layer_ids = z["layer_ids"].astype(np.int32).tolist()
        region_names = [str(x) for x in z["region_names"].tolist()]
    rows = _jsonl_rows(rows_path)
    torn_tail_skipped = False
    if len(rows) != len(row_indexes):
        indexes = [int(row["row_index"]) for row in rows]
        if len(rows) == len(row_indexes) - 1 and row_indexes[:len(rows)].tolist() == indexes:
            print("LOG: skipped the activation row for the torn final JSONL line", file=sys.stderr)
            torn_tail_skipped = True
            region_mass = region_mass[:len(rows)]
            region_values = region_values[:len(rows)]
            body_values = body_values[:len(rows)]
            row_indexes = row_indexes[:len(rows)]
        else:
            raise ValueError(f"rows JSONL has {len(rows)} records but activation tensors have {len(row_indexes)} rows")
    if layer_ids != list(LAYERS) or region_names != list(REGION_NAMES):
        raise ValueError(f"activation schema mismatch: layers={layer_ids}, regions={region_names}")
    if [int(row["row_index"]) for row in rows] != row_indexes.tolist():
        raise ValueError("rows JSONL order does not match the activation tensor row indexes")
    if region_values.ndim != 5 or region_values.shape[1:4] != (2, HEADS, len(REGION_NAMES)):
        raise ValueError(f"unexpected region vector tensor shape {region_values.shape}")
    hidden = int(region_values.shape[-1])
    if region_mass.shape != (len(rows), 2, HEADS, len(REGION_NAMES)):
        raise ValueError(f"unexpected region mass tensor shape {region_mass.shape}")
    if body_values.shape != (len(rows), 2, HEADS, hidden):
        raise ValueError(f"unexpected body control tensor shape {body_values.shape}")

    if args.stub:
        embed, gain, lens_j, o_projections, tok = _stub_inputs(hidden)
        shape_check = {
            "passed": bool(lens_j.shape == (31, hidden, hidden)),
            "lens_shape": list(lens_j.shape), "expected_shape": [31, hidden, hidden],
            "hidden_size": hidden, "snapshot_layers": 32,
            "source": "synthetic end-to-end fixture", "selected_layers": list(LAYERS),
        }
    else:
        jlens111.SNAP = Path(args.snapshot or jlens111.SNAP)
        jlens111.LENS = Path(args.lens or jlens111.LENS)
        snap = jlens111.SNAP
        config = json.loads((snap / "config.json").read_text(encoding="utf-8"))
        text_config = config.get("text_config", config)
        snapshot_hidden = int(text_config["hidden_size"])
        snapshot_layers = int(text_config["num_hidden_layers"])
        weights = jlens111.Weights(snap)
        lens_file = torch.load(str(jlens111.LENS), map_location="cpu", weights_only=True)
        # "J" holds one matrix per layer, as a list or a dict, not one stacked tensor. Part 111 indexes it the same way.
        lens_j = lens_file.get("J")
        matrices = list(lens_j.values()) if isinstance(lens_j, dict) else lens_j
        j_shape = [len(matrices), *matrices[0].shape] if matrices is not None and len(matrices) else None
        shape_ok = bool(
            j_shape == [31, snapshot_hidden, snapshot_hidden]
            and all(0 <= layer < len(matrices) for layer in LAYERS)
        )
        shape_check = {
            "passed": shape_ok, "lens_shape": j_shape,
            "expected_shape": [31, snapshot_hidden, snapshot_hidden],
            "captured_head_dim": hidden, "snapshot_hidden_size": snapshot_hidden,
            "snapshot_layers": snapshot_layers, "selected_layers": list(LAYERS),
            "lens_path": str(jlens111.LENS), "snapshot": str(snap),
        }
        if not shape_ok:
            result = {"gates": {"lens_shape": shape_check}, "readouts": {}, "elapsed_seconds": time.time() - t0}
            _write_json(out / "lens126.json", result)
            print(json.dumps(shape_check, ensure_ascii=False))
            return
        hidden = snapshot_hidden  # what was captured is one head's width; a write lands in the model's hidden size
        gain = 1.0 + weights.get(PREFIX + "norm.weight").float().cpu()
        embed = weights.get(PREFIX + "embed_tokens.weight").float().cpu()
        tok = jlens111.load_tok()
        o_projections = {
            layer: weights.get(f"{PREFIX}layers.{layer}.self_attn.o_proj.weight").float().cpu()
            for layer in LAYERS
        }
        if int(embed.shape[1]) != hidden or int(gain.shape[0]) != hidden:
            shape_check["passed"] = False
            shape_check["unembedding_shape"] = list(embed.shape)
            shape_check["norm_shape"] = list(gain.shape)
            result = {"gates": {"lens_shape": shape_check}, "readouts": {}, "elapsed_seconds": time.time() - t0}
            _write_json(out / "lens126.json", result)
            return
        for layer, proj in o_projections.items():
            if tuple(proj.shape) != (hidden, HEADS * region_values.shape[-1]):
                shape_check["passed"] = False
                shape_check["o_proj_shapes"] = {str(k): list(v.shape) for k, v in o_projections.items()}
                result = {"gates": {"lens_shape": shape_check}, "readouts": {}, "elapsed_seconds": time.time() - t0}
                _write_json(out / "lens126.json", result)
                return

    shape_check["passed"] = bool(shape_check.get("passed"))
    if not shape_check["passed"]:
        result = {"gates": {"lens_shape": shape_check}, "readouts": {}, "elapsed_seconds": time.time() - t0}
        _write_json(out / "lens126.json", result)
        return
    if int(embed.shape[1]) != hidden or tuple(gain.shape) != (hidden,):
        raise ValueError("lens unembedding or final norm does not match the captured hidden size")
    if int(embed.shape[0]) < TOP_K:
        raise ValueError(f"vocabulary has fewer than {TOP_K} tokens")

    copy_rows = {
        copy: np.asarray([i for i, row in enumerate(rows) if row["rendering"] == copy], dtype=np.int64)
        for copy in COPIES
    }
    for copy, indexes in copy_rows.items():
        if len(indexes) == 0:
            raise ValueError(f"no rows for copy {copy}")

    # Residual-space means are saved separately so score126.py can score predictions from raw vectors.
    role_writes = np.zeros((2, HEADS, 2, hidden), dtype=np.float32)
    region_writes = np.zeros_like(role_writes)
    body_control_writes = np.zeros_like(role_writes)
    readouts = {}
    for layer_pos, layer in enumerate(LAYERS):
        projection = o_projections[layer].float()
        if tuple(projection.shape) != (hidden, HEADS * region_values.shape[-1]):
            shape_check["passed"] = False
            shape_check["o_proj_shapes"] = {str(k): list(v.shape) for k, v in o_projections.items()}
            result = {"gates": {"lens_shape": shape_check}, "readouts": {}, "elapsed_seconds": time.time() - t0}
            _write_json(out / "lens126.json", result)
            return
        dim = int(region_values.shape[-1])
        for copy, copy_pos in COPY_INDEX.items():
            indexes = copy_rows[copy]
            mean_mass = region_mass[indexes, layer_pos].mean(axis=0)  # [heads, regions]
            for head in range(HEADS):
                top_region_i = int(np.argmax(mean_mass[head]))
                head_block = projection[:, head * dim:(head + 1) * dim]
                role_in = region_values[indexes, layer_pos, head, ROLE_REGION, :].mean(axis=0)
                region_in = region_values[indexes, layer_pos, head, top_region_i, :].mean(axis=0)
                body_in = body_values[indexes, layer_pos, head, :].mean(axis=0)
                role_write = head_block @ torch.as_tensor(role_in, dtype=torch.float32)
                region_write = head_block @ torch.as_tensor(region_in, dtype=torch.float32)
                body_write = head_block @ torch.as_tensor(body_in, dtype=torch.float32)
                role_writes[layer_pos, head, copy_pos] = role_write.numpy()
                region_writes[layer_pos, head, copy_pos] = region_write.numpy()
                body_control_writes[layer_pos, head, copy_pos] = body_write.numpy()
                jmat = lens_j[layer].float().cpu() if not args.stub else lens_j[layer].float()
                vectors = torch.stack([role_write, region_write, body_write], dim=1)
                lens_vectors = gain.reshape(-1, 1) * (jmat @ vectors)
                best_ids, low_ids = jlens111.top_both(embed, lens_vectors, k=TOP_K)
                readouts[f"L{layer}h{head}"] = readouts.get(f"L{layer}h{head}", {})
                readouts[f"L{layer}h{head}"][copy] = {
                    "role_token": _tokens_for(tok, best_ids[:, 0].tolist(), low_ids[:, 0].tolist()),
                    "most_attended_region": {
                        "name": REGION_NAMES[top_region_i], "mean_attention_mass": float(mean_mass[head, top_region_i]),
                        **_tokens_for(tok, best_ids[:, 1].tolist(), low_ids[:, 1].tolist()),
                    },
                    "random_body_token_control": _tokens_for(tok, best_ids[:, 2].tolist(), low_ids[:, 2].tolist()),
                }

    vectors_path = out / "lens126-vectors.npz"
    tmp_vectors = out / "lens126-vectors.tmp.npz"
    np.savez_compressed(tmp_vectors, role_writes=role_writes, region_writes=region_writes,
                        body_control_writes=body_control_writes, layer_ids=np.asarray(LAYERS, dtype=np.int16),
                        copy_names=np.asarray(COPIES, dtype="U1"))
    os.replace(tmp_vectors, vectors_path)
    result = {
        "design": "PART126-DESIGN.md", "gates": {"lens_shape": shape_check},
        "formula": "unembedding @ ((1 + final_norm_weight) * (J_layer @ write))",
        "write_definition": "mean gated value through this head's o_proj input-column block, attention set to one",
        "top_k": TOP_K, "region_names": list(REGION_NAMES), "copies": list(COPIES),
        "random_body_token_control": "same head and prompt; one seeded uniform body token; no sign flip",
        "vectors_file": vectors_path.name, "readouts": readouts,
        "torn_final_jsonl_line_skipped": torn_tail_skipped,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    _write_json(out / "lens126.json", result)
    print(f"read {2 * HEADS} heads for both copies; lens gate {shape_check['passed']}; wrote {out / 'lens126.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--run-json", default=None)
    parser.add_argument("--rows", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--lens", default=None)
    parser.add_argument("--stub", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
