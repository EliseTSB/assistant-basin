#!/usr/bin/env python3
"""headweights182.py -- Part 182 Amendment 1, item 2 (Trap 2): how far each attention head's weights moved from the
mid-training end (donor) to Instruct (host), read from the two checkpoints' weights alone.

  python3 -I part182/headweights182.py --host DIR --donor DIR --layers 0,1,2,3,14 --out PATH

DIR is a local Hugging Face snapshot directory (config.json plus *.safetensors, sharded or not). Tensors are read one
at a time with ``safetensors.safe_open``; the model is never loaded. Cosines are computed in float64 on the CPU.

For each listed layer L and each head h of the host's ``num_attention_heads``:
  q_row_cos   cosine between the donor's and the host's q_proj rows h*head_dim:(h+1)*head_dim
  k_row_cos   the same for the k_proj rows of the key-value head that h uses: with grouped-query attention the
              key-value head of h is h // (num_attention_heads // num_key_value_heads); with one key-value head per
              query head it is h itself
  v_row_cos   the same for the v_proj rows of that key-value head
  o_col_cos   cosine between the donor's and the host's o_proj columns h*head_dim:(h+1)*head_dim
Per layer (whole vectors):
  q_norm_cos  cosine between the donor's and the host's q_norm.weight (num_attention_heads*head_dim)
  k_norm_cos  the same for k_norm.weight
Once:
  embed_cos   cosine between the donor's and the host's embed_tokens.weight, flattened

head_dim is read from config.json when present, else hidden_size // num_attention_heads (as the OLMo 2 config does).

Output JSON (written to --out, also printed):
{
  "host": "<path as given>", "donor": "<path as given>",
  "host_config": {"num_attention_heads": .., "num_key_value_heads": .., "head_dim": .., "hidden_size": ..},
  "donor_config": {...},            # the same four fields, so a shape mismatch is visible
  "tensor_names": {"host": {...}, "donor": {...}},
      # {"embed": name, "layers": {"<L>": {"q_proj": name, "k_proj": name, "v_proj": name, "o_proj": name,
      #   "q_norm": name, "k_norm": name}}}
  "embed_cos": float,
  "layers": {"<L>": {"q_norm_cos": float, "k_norm_cos": float,
                     "heads": {"<h>": {"q_row_cos": float, "k_row_cos": float, "v_row_cos": float,
                                       "o_col_cos": float}}}}
}
A cosine whose vector has zero norm in either checkpoint is null, never 0.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import torch

DEFAULT_LAYERS = "0,1,2,3,14"


# ---------------------------------------------------------------------------------------------------------------------
# Reading a snapshot's tensors one at a time.

class Snapshot:
    """A local HF snapshot directory: config.json plus one or more *.safetensors files. ``read(name)`` returns one
    tensor in float64, read from disk on each call; nothing else is held."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        if not self.path.is_dir():
            raise FileNotFoundError(f"snapshot directory not found: {self.path}")
        config_path = self.path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"no config.json in {self.path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        files = sorted(self.path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no *.safetensors in {self.path}")
        self.files = files
        self._index = None

    def _build_index(self):
        from safetensors import safe_open
        index = {}
        for file in self.files:
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in index:
                        raise ValueError(f"tensor {name} appears in both {index[name]} and {file.name}")
                    index[name] = file
        self._index = index

    def names(self):
        if self._index is None:
            self._build_index()
        return self._index

    def find(self, pattern):
        """The one tensor name matching ``pattern`` (fnmatch). Fails loudly on none or several."""
        matches = sorted(n for n in self.names() if fnmatch.fnmatchcase(n, pattern))
        if len(matches) != 1:
            raise KeyError(f"{self.path}: pattern {pattern!r} matched {len(matches)} tensors {matches[:6]}")
        return matches[0]

    def read(self, name):
        from safetensors import safe_open
        file = self.names()[name]
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).to(torch.float64)

    # -- names ---------------------------------------------------------------------------------------------------------
    def embed_name(self):
        return self.find("*embed_tokens.weight")

    def layer_names(self, layer_no):
        base = f"*layers.{int(layer_no)}.self_attn."
        return {key: self.find(base + suffix) for key, suffix in (
            ("q_proj", "q_proj.weight"), ("k_proj", "k_proj.weight"), ("v_proj", "v_proj.weight"),
            ("o_proj", "o_proj.weight"), ("q_norm", "q_norm.weight"), ("k_norm", "k_norm.weight"))}

    def geometry(self):
        """(num_attention_heads, num_key_value_heads, head_dim, hidden_size) from config.json."""
        cfg = self.config
        n_heads = int(cfg["num_attention_heads"])
        n_kv = int(cfg.get("num_key_value_heads") or n_heads)
        hidden = int(cfg["hidden_size"])
        head_dim = int(cfg.get("head_dim") or hidden // n_heads)
        return n_heads, n_kv, head_dim, hidden


def cosine(a: torch.Tensor, b: torch.Tensor):
    """Cosine of two flattened tensors in float64; None when either has zero norm."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} against {tuple(b.shape)}")
    a, b = a.reshape(-1), b.reshape(-1)
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if float(na) == 0.0 or float(nb) == 0.0:
        return None
    return float(torch.dot(a, b) / (na * nb))


def kv_head_for(head, n_heads, n_kv):
    """The key-value head that query head ``head`` uses (GQA: contiguous groups, as repeat_kv assumes)."""
    if n_heads % n_kv:
        raise ValueError(f"{n_heads} query heads do not divide into {n_kv} key-value heads")
    return head // (n_heads // n_kv)


def head_weights(host_dir, donor_dir, layers) -> dict:
    """The full record described in the module docstring."""
    host, donor = Snapshot(host_dir), Snapshot(donor_dir)
    h_heads, h_kv, h_dim, h_hidden = host.geometry()
    d_heads, d_kv, d_dim, d_hidden = donor.geometry()
    if (h_heads, h_kv, h_dim) != (d_heads, d_kv, d_dim):
        raise ValueError(f"attention geometry differs: host {(h_heads, h_kv, h_dim)}, donor {(d_heads, d_kv, d_dim)}")
    record = {
        "host": str(host_dir), "donor": str(donor_dir),
        "host_config": {"num_attention_heads": h_heads, "num_key_value_heads": h_kv, "head_dim": h_dim,
                        "hidden_size": h_hidden},
        "donor_config": {"num_attention_heads": d_heads, "num_key_value_heads": d_kv, "head_dim": d_dim,
                         "hidden_size": d_hidden},
        "tensor_names": {"host": {}, "donor": {}}, "embed_cos": None, "layers": {},
    }
    embed = (host.embed_name(), donor.embed_name())
    record["tensor_names"]["host"]["embed"] = embed[0]
    record["tensor_names"]["donor"]["embed"] = embed[1]
    record["embed_cos"] = cosine(host.read(embed[0]), donor.read(embed[1]))
    record["tensor_names"]["host"]["layers"] = {}
    record["tensor_names"]["donor"]["layers"] = {}
    for layer_no in sorted({int(l) for l in layers}):
        h_names, d_names = host.layer_names(layer_no), donor.layer_names(layer_no)
        record["tensor_names"]["host"]["layers"][str(layer_no)] = h_names
        record["tensor_names"]["donor"]["layers"][str(layer_no)] = d_names
        hq, dq = host.read(h_names["q_proj"]), donor.read(d_names["q_proj"])
        hk, dk = host.read(h_names["k_proj"]), donor.read(d_names["k_proj"])
        hv, dv = host.read(h_names["v_proj"]), donor.read(d_names["v_proj"])
        ho, do = host.read(h_names["o_proj"]), donor.read(d_names["o_proj"])
        for key, tensor in (("q_proj", hq), ("k_proj", hk), ("v_proj", hv), ("o_proj", ho)):
            if tensor.shape[0] != h_heads * h_dim:
                raise ValueError(f"layer {layer_no} {key} has {tensor.shape[0]} rows, not {h_heads} x {h_dim}")
        heads = {}
        for head in range(h_heads):
            rows = slice(head * h_dim, (head + 1) * h_dim)
            kv = kv_head_for(head, h_heads, h_kv)
            kv_rows = slice(kv * h_dim, (kv + 1) * h_dim)
            heads[str(head)] = {
                "q_row_cos": cosine(hq[rows], dq[rows]),
                "k_row_cos": cosine(hk[kv_rows], dk[kv_rows]),
                "v_row_cos": cosine(hv[kv_rows], dv[kv_rows]),
                "o_col_cos": cosine(ho[:, rows], do[:, rows]),
            }
        record["layers"][str(layer_no)] = {
            "q_norm_cos": cosine(host.read(h_names["q_norm"]), donor.read(d_names["q_norm"])),
            "k_norm_cos": cosine(host.read(h_names["k_norm"]), donor.read(d_names["k_norm"])),
            "heads": heads,
        }
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="Instruct snapshot directory")
    ap.add_argument("--donor", required=True, help="mid-training-end snapshot directory")
    ap.add_argument("--layers", default=DEFAULT_LAYERS, help="comma-separated decoder layer indices")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    record = head_weights(args.host, args.donor, layers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "tensor_names"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
