#!/usr/bin/env python3
"""Range-read the registered Part 85 weights and compute the direct-readout measures."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoints import BY_KEY, resolve_checkpoint


LAYER = 10
UNIT = 6978
CONTROL_UNIT = 7884
OFFWORLD_FORMS = (
    " alien", " aliens", " extraterrestrial", " planet", " planets", " galaxy", " space",
    " spaceship", " starship", " Mars", " cosmos", " universe", " interstellar", " outer",
    " orbit", " asteroid",
)
RANDOM_SEED = 85085
RANDOM_SET_COUNT = 1000
DTYPES = {
    "F64": (np.dtype("<f8"), 8),
    "F32": (np.dtype("<f4"), 4),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
    "I64": (np.dtype("<i8"), 8),
    "I32": (np.dtype("<i4"), 4),
    "I16": (np.dtype("<i2"), 2),
    "I8": (np.dtype("i1"), 1),
    "U8": (np.dtype("u1"), 1),
    "BOOL": (np.dtype("u1"), 1),
}


def decode_tensor(raw: bytes, dtype: str, shape) -> np.ndarray:
    if dtype not in DTYPES:
        raise ValueError(f"unsupported safetensors dtype {dtype}")
    base, _ = DTYPES[dtype]
    arr = np.frombuffer(raw, dtype=base).reshape(tuple(shape))
    if dtype == "BF16":
        bits = arr.astype(np.uint32) << 16
        return bits.view(np.float32)
    return arr.astype(np.float32, copy=False) if dtype.startswith("F") else arr


class SafetensorsRangeReader:
    """Read safetensors tensors locally or through HTTP byte ranges."""

    def __init__(self, source: str | Path, repo_id: str | None = None, revision: str | None = None,
                 limit_rate: int | None = None):
        self.local = Path(source).expanduser() if repo_id is None else None
        self.repo_id = repo_id
        self.revision = revision
        self.limit_rate = limit_rate
        self._headers = {}
        self._weight_map = None
        self._rate_wait_until = 0.0
        if self.local is not None:
            index_path = self.local / "model.safetensors.index.json"
            if not index_path.exists():
                raise FileNotFoundError(index_path)
            self._weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        else:
            if not repo_id or not revision:
                raise ValueError("remote reads require repo_id and revision")
            self._weight_map = json.loads(self._url_bytes(self._url("model.safetensors.index.json")))['weight_map']

    def _url(self, filename: str) -> str:
        from huggingface_hub import hf_hub_url
        return hf_hub_url(self.repo_id, filename, revision=self.revision)

    def _url_bytes(self, url: str, byte_range: tuple[int, int] | None = None) -> bytes:
        if self.limit_rate:
            delay = self._rate_wait_until - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        started = time.monotonic()
        headers = {"User-Agent": "Part85/1.0"}
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=180) as response:
            data = response.read()
            status = getattr(response, "status", response.getcode())
            content_range = response.headers.get("Content-Range")
        if self.limit_rate:
            self._rate_wait_until = started + len(data) / self.limit_rate
        if byte_range is not None:
            expected = byte_range[1] - byte_range[0] + 1
            if status != 206 or len(data) != expected:
                raise IOError(f"range request {byte_range} returned HTTP {status} with {len(data)} bytes")
            if content_range and not content_range.startswith(f"bytes {byte_range[0]}-{byte_range[1]}/"):
                raise IOError(f"unexpected Content-Range {content_range!r} for {byte_range}")
        return data

    def _read_file(self, filename: str, start: int, end: int) -> bytes:
        if self.local is not None:
            path = self.local / filename
            with path.open("rb") as stream:
                stream.seek(start)
                data = stream.read(end - start + 1)
            if len(data) != end - start + 1:
                raise IOError(f"short local read from {path}: got {len(data)} bytes")
            return data
        return self._url_bytes(self._url(filename), (start, end))

    def _header(self, filename: str) -> dict:
        if filename not in self._headers:
            prefix = self._read_file(filename, 0, 7)
            header_len = struct.unpack("<Q", prefix)[0]
            header = json.loads(self._read_file(filename, 8, 7 + header_len))
            self._headers[filename] = {k: v for k, v in header.items() if k != "__metadata__"}
        return self._headers[filename]

    def info(self, name: str) -> dict:
        if name not in self._weight_map:
            raise KeyError(f"tensor {name!r} is not in the safetensors index")
        filename = self._weight_map[name]
        try:
            tensor = self._header(filename)[name]
        except KeyError as exc:
            raise KeyError(f"index maps {name} to {filename}, but its header does not") from exc
        return {**tensor, "filename": filename}

    def read_slice(self, name: str, start_element: int, stop_element: int) -> np.ndarray:
        """Read a contiguous flattened element interval; stop is exclusive."""
        info = self.info(name)
        count = math.prod(info["shape"])
        if not (0 <= start_element <= stop_element <= count):
            raise IndexError(f"invalid slice [{start_element}:{stop_element}] for {name} with {count} elements")
        _, size = DTYPES[info["dtype"]]
        base = 8 + self._header_length(info["filename"])
        start = base + info["data_offsets"][0] + start_element * size
        end = base + info["data_offsets"][0] + stop_element * size - 1
        raw = b"" if stop_element == start_element else self._read_file(info["filename"], start, end)
        return decode_tensor(raw, info["dtype"], (stop_element - start_element,))

    def _header_length(self, filename: str) -> int:
        if filename not in self._headers:
            self._header(filename)
        # The safetensors header starts at byte 8; cache the serialized size separately.
        cache_key = ("__header_length__", filename)
        if cache_key not in self._headers:
            self._headers[cache_key] = struct.unpack("<Q", self._read_file(filename, 0, 7))[0]
        return self._headers[cache_key]

    def read_tensor(self, name: str) -> np.ndarray:
        info = self.info(name)
        return self.read_slice(name, 0, math.prod(info["shape"])).reshape(info["shape"])

    def read_row(self, name: str, row: int) -> np.ndarray:
        info = self.info(name)
        if len(info["shape"]) != 2:
            raise ValueError(f"{name} is not a matrix")
        rows, cols = info["shape"]
        if not 0 <= row < rows:
            raise IndexError(row)
        return self.read_slice(name, row * cols, (row + 1) * cols)

    def read_column(self, name: str, column: int) -> np.ndarray:
        """Read a matrix column; safetensors stores rows contiguously, so fetch the tensor."""
        tensor = self.read_tensor(name)
        if tensor.ndim != 2 or not 0 <= column < tensor.shape[1]:
            raise IndexError(f"column {column} is invalid for {name} shape {tensor.shape}")
        return tensor[:, column].copy()

    def read_rows(self, name: str, rows) -> np.ndarray:
        """Read selected rows with coalesced requests when adjacent rows are requested."""
        info = self.info(name)
        n_rows, width = info["shape"]
        requested = [int(i) for i in rows]
        if any(i < 0 or i >= n_rows for i in requested):
            raise IndexError(f"row outside [0, {n_rows})")
        out = np.empty((len(requested), width), dtype=np.float32)
        order = sorted(set(requested))
        chunk_rows = 256
        chunks = sorted({row // chunk_rows for row in order})
        cache = {}
        for chunk in chunks:
            start_row = chunk * chunk_rows
            stop_row = min(n_rows, start_row + chunk_rows)
            values = self.read_slice(name, start_row * width, stop_row * width).reshape(stop_row - start_row, width)
            for row in order:
                if start_row <= row < stop_row:
                    cache[row] = values[row - start_row]
        for i, row in enumerate(requested):
            out[i] = cache[row]
        return out


def tensor_names(reader: SafetensorsRangeReader) -> dict[str, str]:
    names = set(reader._weight_map)

    def choose(candidates):
        for name in candidates:
            if name in names:
                return name
        raise KeyError(f"none of these tensors were found: {candidates}")

    layer = f"model.layers.{LAYER}.mlp"
    return {
        "down": choose([layer + ".down_proj.weight", f"layers.{LAYER}.mlp.down_proj.weight"]),
        "gate": choose([layer + ".gate_proj.weight", f"layers.{LAYER}.mlp.gate_proj.weight"]),
        "up": choose([layer + ".up_proj.weight", f"layers.{LAYER}.mlp.up_proj.weight"]),
        "norm": choose(["model.norm.weight", "norm.weight", "transformer.norm_f.weight"]),
        "embedding": choose(["lm_head.weight", "model.embed_tokens.weight", "transformer.wte.weight"]),
    }


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else None


def token_sets(tokenizer, count=RANDOM_SET_COUNT, seed=RANDOM_SEED):
    off_ids = []
    unique_forms = []
    for form in OFFWORLD_FORMS:
        ids = tokenizer(form, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"off-world form tokenized to no tokens: {form!r}")
        token_id = int(ids[0])
        if token_id not in off_ids:
            off_ids.append(token_id)
            unique_forms.append(form)
    # Character length means the visible token spelling, with tokenizer boundary markers removed.
    def visible_length(i):
        token = str(tokenizer.convert_ids_to_tokens(int(i)))
        for prefix in ("Ġ", "▁"):
            if token.startswith(prefix):
                token = token[len(prefix):]
        if token.endswith("</w>"):
            token = token[:-4]
        return len(token)
    lengths = [visible_length(i) for i in off_ids]
    lo, hi = min(lengths), max(lengths)
    vocab = len(tokenizer)
    offworld_id_set = set(off_ids)
    special_ids = set(tokenizer.all_special_ids)
    candidates = [i for i in range(vocab) if i not in offworld_id_set
                  and i not in special_ids and lo <= visible_length(i) <= hi]
    if len(candidates) < len(off_ids):
        raise ValueError("not enough ordinary vocabulary tokens in the registered character-length range")
    rng = random.Random(seed)
    random_sets = [rng.sample(candidates, len(off_ids)) for _ in range(count)]
    return off_ids, random_sets, {"character_length_range": [lo, hi], "offworld_forms": unique_forms,
                                  "registered_offworld_forms": list(OFFWORLD_FORMS),
                                  "offworld_ids": off_ids, "random_seed": seed,
                                  "random_set_count": count, "random_set_size": len(off_ids)}


def analyze_local_or_remote(source, repo_id=None, revision=None, tokenizer=None, reference=None,
                            limit_rate=None, random_set_count=RANDOM_SET_COUNT):
    reader = SafetensorsRangeReader(source, repo_id=repo_id, revision=revision, limit_rate=limit_rate)
    names = tensor_names(reader)
    # Safetensors matrices are row major. Column 6978 is strided, so read the complete down projection.
    down = reader.read_column(names["down"], UNIT)
    gate = reader.read_row(names["gate"], UNIT)
    up = reader.read_row(names["up"], UNIT)
    norm = reader.read_tensor(names["norm"]).reshape(-1).astype(np.float32)
    if tokenizer is None:
        from transformers import AutoTokenizer
        if repo_id:
            tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(Path(source)))
    off_ids, rand_sets, token_meta = token_sets(tokenizer, count=random_set_count)
    all_ids = sorted(set(off_ids).union(*(set(x) for x in rand_sets)))
    # Embedding rows are fetched in row order, then used only for these registered token IDs.
    emb_name = names["embedding"]
    emb_info = reader.info(emb_name)
    if len(emb_info["shape"]) != 2 or emb_info["shape"][1] != down.shape[0]:
        raise ValueError(f"output embedding shape {emb_info['shape']} does not match down column width {down.shape[0]}")
    embeddings = reader.read_rows(emb_name, all_ids)
    row_map = {idx: embeddings[i] for i, idx in enumerate(all_ids)}
    gain = norm
    logits = {idx: float(np.dot(row_map[idx].astype(np.float64),
                                down.astype(np.float64) * gain.astype(np.float64))) for idx in all_ids}
    off_mean = float(np.mean([logits[i] for i in off_ids]))
    random_means = np.asarray([np.mean([logits[i] for i in rows]) for rows in rand_sets], dtype=np.float64)
    delta = off_mean - random_means
    std = float(np.std(random_means, ddof=1)) if len(random_means) > 1 else 0.0
    z_score = float((off_mean - float(np.mean(random_means))) / std) if std else None
    result = {
        "measures": {
            "direct_readout": {"offworld_mean_logit": off_mean, "random_set_mean_logit": float(np.mean(random_means)),
                               "random_set_sd": std, "offworld_minus_random_mean": float(np.mean(delta)),
                               "z_score": z_score},
            "cosines_with_instruct": None,
        },
        "token_sets": token_meta,
        "tensor_names": names,
        "weight_shapes": {"down_column": list(down.shape), "gate_row": list(gate.shape),
                          "up_row": list(up.shape), "final_norm": list(norm.shape)},
    }
    vectors = {"down_column_6978": down.tolist(), "gate_row_6978": gate.tolist(), "up_row_6978": up.tolist()}
    if reference is not None:
        ref = reference.get("reference_vectors") or reference.get("vectors")
        if not ref:
            raise ValueError("reference JSON lacks reference_vectors; run the Instruct checkpoint first")
        result["measures"]["cosines_with_instruct"] = {
            "down_proj_column_6978": cosine(down, ref["down_column_6978"]),
            "gate_proj_row_6978": cosine(gate, ref["gate_row_6978"]),
            "up_proj_row_6978": cosine(up, ref["up_row_6978"]),
        }
    result["reference_vectors"] = vectors
    return result


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def run(args):
    cp = ({**asdict(BY_KEY[args.checkpoint]), "commit": "local"}
          if args.local else resolve_checkpoint(args.checkpoint))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8")) if args.reference else None
    if cp["key"] == "instruct" and reference is not None:
        raise ValueError("Instruct is the reference checkpoint and must be run without --reference")
    if cp["key"] != "instruct" and reference is None:
        raise ValueError("run Instruct first and pass its step0.json with --reference")
    if reference is not None and reference.get("checkpoint", {}).get("key") != "instruct":
        raise ValueError("--reference must be the Instruct checkpoint's step0.json")
    tokenizer = None
    if args.local:
        source = args.local
        repo_id = None
        revision = None
    else:
        source = "remote"
        repo_id = cp["repo_id"]
        revision = cp["commit"]
    result = analyze_local_or_remote(source, repo_id=repo_id, revision=revision, tokenizer=tokenizer,
                                     reference=reference, limit_rate=args.limit_rate,
                                     random_set_count=RANDOM_SET_COUNT)
    if cp["key"] == "instruct":
        result["measures"]["cosines_with_instruct"] = {
            "down_proj_column_6978": 1.0, "gate_proj_row_6978": 1.0, "up_proj_row_6978": 1.0,
        }
    result.update({"checkpoint": {k: cp[k] for k in ("key", "stage", "repo_id", "branch", "commit_prefix", "commit")},
                   "dtype_source": "safetensors", "limit_rate_bytes_per_second": args.limit_rate})
    out = Path(args.out) if args.out else Path("results/part85") / cp["key"] / "step0.json"
    atomic_json(out, result)
    print(f"wrote {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, choices=tuple(BY_KEY))
    parser.add_argument("--local", help="local directory containing the safetensors index and shards")
    parser.add_argument("--reference", help="Instruct checkpoint's step0.json; required except for Instruct")
    parser.add_argument("--out")
    parser.add_argument("--limit-rate", type=int, help="throttle range reads to this many bytes per second")
    args = parser.parse_args(argv)
    if args.limit_rate is not None and args.limit_rate <= 0:
        parser.error("--limit-rate must be a positive number of bytes per second")
    run(args)


if __name__ == "__main__":
    main()
