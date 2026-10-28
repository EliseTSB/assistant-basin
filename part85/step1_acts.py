#!/usr/bin/env python3
"""Measure unit activations on fixed WikiText sequences and the registered passage pairs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoints import BY_KEY, resolve_checkpoint

UNIT = 6978
CONTROL_UNIT = 7884
SEQUENCE_COUNT = 200
SEQUENCE_LENGTH = 128
PASSAGES_PATH = Path(__file__).with_name("passages.json")
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-v1"
WIKITEXT_REVISION = "f776294184f13b8ff2337b3841cf9269a6216d1e"


def stable_ids_sha256(token_ids) -> str:
    ids = np.asarray(token_ids, dtype="<i4")
    return hashlib.sha256(ids.tobytes(order="C")).hexdigest()


def create_neutral_sequences(tokenizer, count=SEQUENCE_COUNT, length=SEQUENCE_LENGTH):
    """Build fixed contiguous token blocks from WikiText-103's test split."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install datasets to build the registered WikiText-103 token sequences") from exc
    dataset = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split="test", revision=WIKITEXT_REVISION)
    stream = []
    eos = tokenizer.eos_token_id
    for row in dataset:
        text = row.get("text", "").strip()
        if not text:
            continue
        stream.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if eos is not None:
            stream.append(int(eos))
        if len(stream) >= count * length:
            break
    needed = count * length
    if len(stream) < needed:
        raise RuntimeError(f"WikiText-103 yielded {len(stream)} tokens; need {needed}")
    ids = np.asarray(stream[:needed], dtype=np.int32).reshape(count, length)
    return ids


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
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


def load_or_create_token_ids(path: Path, tokenizer, checkpoint_key: str):
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        ids = np.asarray(raw["token_ids"], dtype=np.int32)
        if ids.shape != (SEQUENCE_COUNT, SEQUENCE_LENGTH):
            raise ValueError(f"neutral token file has shape {ids.shape}, expected {(SEQUENCE_COUNT, SEQUENCE_LENGTH)}")
        actual = stable_ids_sha256(ids)
        if raw.get("sha256") != actual:
            raise ValueError("neutral token file sha256 does not match its token ids")
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= len(tokenizer)):
            raise ValueError("neutral token file contains ids outside this checkpoint tokenizer vocabulary")
        return ids, actual
    if checkpoint_key != "instruct":
        raise FileNotFoundError(f"run Instruct first to create shared sequence ids: {path}")
    ids = create_neutral_sequences(tokenizer)
    digest = stable_ids_sha256(ids)
    save_json(path, {"source": "wikitext-103-v1 test", "shape": list(ids.shape), "sha256": digest,
                     "token_ids": ids.tolist()})
    return ids, digest


def find_text_layers(model):
    import torch
    best = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) and hasattr(getattr(module[0], "mlp", None), "down_proj"):
            key = ("language" in name, len(module))
            if best is None or key > best[0]:
                best = (key, name, module)
    if best is None:
        raise RuntimeError("model does not expose a ModuleList with mlp.down_proj")
    return best[2]


def activation_vector(model, down_proj, ids, device, units=(UNIT, CONTROL_UNIT)):
    captured = {}

    def hook(_module, inputs):
        x = inputs[0]
        captured["values"] = x.detach()[..., list(units)].float().cpu().numpy()[0]

    handle = down_proj.register_forward_pre_hook(hook)
    try:
        import torch
        with torch.inference_mode():
            model(input_ids=torch.as_tensor(ids, dtype=torch.long, device=device).reshape(1, -1), use_cache=False)
    finally:
        handle.remove()
    if "values" not in captured:
        raise RuntimeError("down_proj activation hook did not fire")
    return captured["values"]


def run_activations(model, down_proj, token_ids, device):
    rows = []
    for ids in token_ids:
        rows.append(activation_vector(model, down_proj, ids, device))
    return np.stack(rows, axis=0)


def passage_activation_means(model, tokenizer, down_proj, passages, device):
    off_means, ordinary_means = [], []
    for pair in passages:
        for kind, field in (("offworld", "offworld"), ("ordinary", "ordinary")):
            text = pair[field]
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"empty tokenization for passage {pair['id']} {field}")
            values = activation_vector(model, down_proj, ids, device, (UNIT,))[:, 0]
            (off_means if kind == "offworld" else ordinary_means).append(float(values.mean()))
    return np.asarray(off_means, dtype=np.float64), np.asarray(ordinary_means, dtype=np.float64)


def _correlation(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def collect_acts(checkpoint, tokenizer, model, sequence_ids, passages, device):
    layers = find_text_layers(model)
    down = layers[10].mlp.down_proj
    if down.in_features <= CONTROL_UNIT:
        raise RuntimeError(f"down_proj has only {down.in_features} input units")
    acts = run_activations(model, down, sequence_ids, device)
    off, ordinary = passage_activation_means(model, tokenizer, down, passages, device)
    return acts, off, ordinary


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cp = resolve_checkpoint(args.checkpoint)
    snapshot_dir = args.snapshot_dir
    if snapshot_dir is None:
        from huggingface_hub import snapshot_download
        snapshot_dir = snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    model = AutoModelForCausalLM.from_pretrained(snapshot_dir, dtype=torch.bfloat16)
    model.to(args.device).eval().requires_grad_(False)
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()})
    if not parameter_dtypes or any(dtype != "torch.bfloat16" for dtype in parameter_dtypes):
        raise RuntimeError(f"step 1 requires bf16 parameters; loaded {parameter_dtypes}")
    torch.set_grad_enabled(False)
    token_ids, ids_sha = load_or_create_token_ids(Path(args.sequence_ids), tokenizer, cp["key"])
    passages = json.loads(Path(args.passages).read_text(encoding="utf-8"))
    if len(passages) != 24:
        raise ValueError(f"expected 24 passage pairs, found {len(passages)}")
    acts, off, ordinary = collect_acts(cp, tokenizer, model, token_ids, passages, args.device)
    ref_path = Path(args.reference_acts) if args.reference_acts else Path(args.results_root) / "instruct" / "step1.json"
    if cp["key"] == "instruct":
        reference = acts
    else:
        if not ref_path.exists():
            raise FileNotFoundError(f"run Instruct step 1 first; missing {ref_path}")
        ref_payload = json.loads(ref_path.read_text(encoding="utf-8"))
        if ref_payload.get("checkpoint", {}).get("key") != "instruct":
            raise ValueError("activation reference must be the Instruct checkpoint's step1.json")
        if ref_payload.get("neutral_token_ids_sha256") != ids_sha:
            raise ValueError("Instruct activation reference uses different neutral token ids")
        reference = np.asarray(ref_payload["neutral_activations"], dtype=np.float32)
        if reference.shape != acts.shape:
            raise ValueError(f"reference activation shape {reference.shape} differs from current {acts.shape}")
    flat = acts.reshape(-1, 2)
    ref_flat = reference.reshape(-1, 2)
    means = {str(unit): float(flat[:, col].max()) for col, unit in enumerate((UNIT, CONTROL_UNIT))}
    result = {
        "checkpoint": {k: cp[k] for k in ("key", "stage", "repo_id", "branch", "commit_prefix", "commit")},
        "layer": 10,
        "units": {"target": UNIT, "control": CONTROL_UNIT},
        "dtype": str(next(model.parameters()).dtype),
        "parameter_dtypes": parameter_dtypes,
        "device": args.device,
        "neutral_source": f"{WIKITEXT_REPO}/{WIKITEXT_CONFIG} test",
        "dataset_revision": WIKITEXT_REVISION,
        "library_versions": _library_versions(),
        "neutral_sequence_count": SEQUENCE_COUNT,
        "neutral_sequence_length": SEQUENCE_LENGTH,
        "neutral_token_ids_sha256": ids_sha,
        "neutral_activation_maxima": means,
        "neutral_correlations_with_instruct": {
            "target": _correlation(flat[:, 0], ref_flat[:, 0]),
            "control": _correlation(flat[:, 1], ref_flat[:, 1]),
        },
        "neutral_activations": acts.tolist(),
        "passages": {
            "count_each": len(off),
            "offworld_mean": float(off.mean()),
            "ordinary_mean": float(ordinary.mean()),
            "ordinary_sd": float(ordinary.std(ddof=1)),
            "offworld_per_passage_mean": off.tolist(),
            "ordinary_per_passage_mean": ordinary.tolist(),
        },
    }
    out = Path(args.out) if args.out else Path(args.results_root) / cp["key"] / "step1.json"
    save_json(out, result)
    print(f"wrote {out}; sequence sha256 {ids_sha}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, choices=tuple(BY_KEY))
    parser.add_argument("--snapshot-dir", help="already-downloaded pinned checkpoint snapshot directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-root", default="results/part85")
    parser.add_argument("--sequence-ids", default="results/part85/neutral-token-ids.json")
    parser.add_argument("--passages", default=str(PASSAGES_PATH))
    parser.add_argument("--reference-acts")
    parser.add_argument("--out")
    run(parser.parse_args(argv))


def _library_versions():
    out = {}
    for package in ("torch", "transformers", "huggingface-hub", "safetensors", "numpy", "datasets"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


if __name__ == "__main__":
    main()
