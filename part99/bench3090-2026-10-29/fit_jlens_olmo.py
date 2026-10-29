#!/usr/bin/env python3
"""Fit a Part 57 OLMo Jacobian lens from pinned Wikitext parquet rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "third_party" / "jlens-part57"))

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
MAX_SEQ_LEN = 128
DEFAULT_SEED = 5701
SMOKE_SOURCE_LAYERS = (10, 14)
PART42_OUTPUT = REPO_ROOT / "results" / "part57" / "olmo2-1b" / "jacobian_lens.pt"


def default_corpus() -> Path:
    if os.name == "nt":
        return Path("E:/neuron-probe/corpus/wikitext103-train0.parquet")
    return Path("/mnt/e/neuron-probe/corpus/wikitext103-train0.parquet")


def default_cache() -> Path:
    if os.name == "nt":
        return Path("E:/neuron-probe/hf/hub")
    return Path("/mnt/e/neuron-probe/hf/hub")


def available_ram_gb() -> float:
    if os.name == "nt":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return float(status.ullAvailPhys) / (1024**3)

    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return float(available_pages * page_size) / (1024**3)


def guard_system_memory(device: str, minimum_gb: float, reserve_gb: float = 0.0) -> None:
    if device != "xpu":
        return
    available = available_ram_gb()
    required = minimum_gb + reserve_gb
    if available < required:
        raise RuntimeError(
            f"XPU memory guard: {available:.2f} GB system RAM available; "
            f"need at least {required:.2f} GB"
        )


def set_bos_behavior(tokenizer: Any) -> None:
    if getattr(tokenizer, "bos_token_id", None) is not None and hasattr(
        tokenizer, "add_bos_token"
    ):
        tokenizer.add_bos_token = True


def select_prompts(
    corpus_path: Path,
    tokenizer: Any,
    count: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    import pyarrow.parquet as pq

    if count <= 0:
        raise ValueError("prompt count must be positive")
    table = pq.read_table(corpus_path, columns=["text"])
    rows = table.column("text").to_pylist()
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    prompts: list[str] = []
    row_indices: list[int] = []
    examined = 0
    for row_index in order:
        raw = rows[row_index]
        if not isinstance(raw, str) or not raw.strip():
            continue
        encoded = tokenizer(raw, truncation=True, max_length=MAX_SEQ_LEN)
        if len(encoded["input_ids"]) != MAX_SEQ_LEN:
            continue
        prompts.append(raw)
        row_indices.append(row_index)
        if len(prompts) == count:
            return prompts, row_indices
        examined += 1
        if examined % 10000 == 0:
            print(
                f"selected {len(prompts)}/{count}  scanned {examined} nonempty rows",
                flush=True,
            )
    raise ValueError(
        f"found only {len(prompts)} rows with at least {MAX_SEQ_LEN} tokens; "
        f"needed {count}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_lens_save(lens: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    lens.save(str(temporary))
    temporary.replace(path)


def parse_layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        layers = sorted({int(item) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--source-layers must be comma-separated integers") from exc
    if not layers:
        raise argparse.ArgumentTypeError("--source-layers cannot be empty")
    return layers


def fit_incrementally(
    model: Any,
    prompts: list[str],
    *,
    source_layers: list[int],
    target_layer: int,
    dim_batch: int,
    device: str,
    minimum_free_ram_gb: float,
    checkpoint_every: int,
    output_path: Path,
    metadata: dict[str, Any],
) -> Any:
    import jlens

    combined = None
    started = time.perf_counter()
    partial_path = output_path.with_name(output_path.stem + ".partial.pt")
    metadata_path = output_path.with_suffix(".metadata.json")
    for index, prompt in enumerate(prompts, 1):
        guard_system_memory(device, minimum_free_ram_gb)
        prompt_started = time.perf_counter()
        piece = jlens.fit(
            model,
            [prompt],
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=MAX_SEQ_LEN,
            checkpoint_every=None,
            resume=False,
        )
        combined = piece if combined is None else jlens.JacobianLens.merge([combined, piece])
        elapsed = time.perf_counter() - started
        metadata["prompts_fitted"] = index
        metadata["elapsed_seconds"] = elapsed
        print(
            f"fit {index}/{len(prompts)}  prompt_seconds={time.perf_counter()-prompt_started:.1f} "
            f"elapsed_seconds={elapsed:.1f} free_ram_gb={available_ram_gb():.2f}",
            flush=True,
        )
        guard_system_memory(device, minimum_free_ram_gb)
        if checkpoint_every > 0 and index % checkpoint_every == 0:
            atomic_lens_save(combined, partial_path)
            atomic_json(metadata_path, metadata)

    if combined is None:
        raise RuntimeError("jlens.fit returned no fitted prompts")
    atomic_lens_save(combined, output_path)
    if partial_path.exists():
        partial_path.unlink()
    metadata["prompts_fitted"] = combined.n_prompts
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["lens_path"] = str(output_path)
    atomic_json(metadata_path, metadata)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=default_corpus())
    parser.add_argument("--cache-dir", type=Path, default=default_cache())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "xpu", "cuda"), default="xpu" if os.name == "nt" else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="fit 10 prompts at L10 and L14 only")
    parser.add_argument("--dry-run", action="store_true", help="select and list rows without loading OLMo")
    parser.add_argument("--source-layers", help="comma-separated source layers; default is every non-final layer")
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--minimum-free-ram-gb", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Wikitext parquet does not exist: {args.data}")
    if args.dim_batch <= 0 or args.threads <= 0 or args.checkpoint_every < 0:
        parser.error("dim-batch/threads must be positive and checkpoint-every non-negative")
    n_prompts = 10 if args.smoke else args.n_prompts
    source_layers = (
        list(SMOKE_SOURCE_LAYERS)
        if args.smoke and args.source_layers is None
        else parse_layers(args.source_layers)
    )
    dtype_name = "bfloat16" if args.dtype == "bf16" else "float32"
    output_path = args.out or (
        PART42_OUTPUT.with_name("smoke-10x2.pt") if args.smoke else PART42_OUTPUT
    )
    if output_path.exists() and not args.force and not args.dry_run:
        parser.error(f"output already exists (pass --force to replace): {output_path}")
    torch_dtype_name = dtype_name

    import jlens
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    if args.device == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            parser.error("torch.xpu is unavailable; use the Windows XPU Python environment")
        guard_system_memory("xpu", args.minimum_free_ram_gb, reserve_gb=2.5)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    set_bos_behavior(tokenizer)
    prompts, row_indices = select_prompts(args.data, tokenizer, n_prompts, args.seed)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "corpus": str(args.data),
                    "corpus_sha256": sha256_file(args.data),
                    "selection_method": "seeded shuffle of parquet rows; keep first rows tokenizing to exactly 128 tokens",
                    "seed": args.seed,
                    "n_prompts": len(prompts),
                    "max_seq_len": MAX_SEQ_LEN,
                    "row_indices": row_indices,
                },
                indent=2,
            )
        )
        return

    transformers_version = __import__("transformers").__version__
    dtype_argument = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
        trust_remote_code=False,
        **{dtype_argument: torch.bfloat16 if torch_dtype_name == "bfloat16" else torch.float32},
    )
    wrapped = jlens.from_hf(model, tokenizer, device=args.device)
    n_layers = wrapped.n_layers
    target_layer = args.target_layer if args.target_layer is not None else n_layers - 1
    if source_layers is None:
        source_layers = list(range(target_layer))
    if any(layer < 0 or layer >= target_layer for layer in source_layers):
        parser.error(
            f"source layers must be in 0..{target_layer - 1}; got {source_layers}"
        )
    if args.smoke and (source_layers != list(SMOKE_SOURCE_LAYERS) or target_layer != n_layers - 1):
        parser.error(f"smoke requires source layers {SMOKE_SOURCE_LAYERS} and the final target layer")

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": args.device,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "jlens_source": "third_party/jlens-part57 (Apache-2.0; explicit device patch)",
        "fit_strategy": "one prompt per jlens.fit call, merged with JacobianLens.merge",
        "corpus": str(args.data),
        "corpus_sha256": sha256_file(args.data),
        "seed": args.seed,
        "row_indices": row_indices,
        "n_prompts_requested": n_prompts,
        "max_seq_len": MAX_SEQ_LEN,
        "source_layers": source_layers,
        "target_layer": target_layer,
        "dim_batch": args.dim_batch,
        "minimum_free_ram_gb": args.minimum_free_ram_gb,
        "prompts_fitted": 0,
        "output_path": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"fit start model={MODEL_ID} revision={MODEL_REVISION} device={args.device} "
        f"dtype={args.dtype} rows={row_indices} source_layers={source_layers} "
        f"target_layer={target_layer} dim_batch={args.dim_batch}",
        flush=True,
    )
    fit_incrementally(
        wrapped,
        prompts,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=args.dim_batch,
        device=args.device,
        minimum_free_ram_gb=args.minimum_free_ram_gb,
        checkpoint_every=args.checkpoint_every,
        output_path=output_path,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
