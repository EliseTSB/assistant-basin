"""Generate Part 162 local-model answers on the Arc; appendable and resumable."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

MODELS = (
    ("pythia", "EleutherAI/pythia-1.4b", "models--EleutherAI--pythia-1.4b"),
    ("olmo", "allenai/OLMo-2-0425-1B-Instruct", "models--allenai--OLMo-2-0425-1B-Instruct"),
    ("qwen", "Qwen/Qwen3.5-4B", "models--Qwen--Qwen3.5-4B"),
)
DEFAULT_MODELS_ROOT = Path("E:/neuron-probe/hf/hub")
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "prompts162.json"
DEFAULT_OUT = Path("results/part162/answers.jsonl")
MAX_NEW_TOKENS = 200


def build_pythia_prompt(prompt: str) -> str:
    return f"Question: {prompt}\nAnswer:"


def build_chat_prompt(tokenizer: Any, prompt: str, family: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if family == "qwen":
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    if family == "olmo":
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise ValueError(f"unknown chat family {family!r}")


def encode_prompt(tokenizer: Any, prompt: str, family: str) -> dict[str, Any]:
    rendered = build_pythia_prompt(prompt) if family == "pythia" else build_chat_prompt(tokenizer, prompt, family)
    # Rendering first avoids transformers 5.20 returning an encoding from apply_chat_template.
    # A rendered chat template already holds its own special tokens.
    return tokenizer(rendered, add_special_tokens=(family == "pythia"), truncation=False)


def find_snapshot(models_root: Path, folder: str) -> Path:
    snapshots = models_root / folder / "snapshots"
    found = sorted(p for p in snapshots.iterdir() if p.is_dir()) if snapshots.is_dir() else []
    if not found:
        raise FileNotFoundError(f"no snapshot directories under {snapshots}")
    return found[-1]


def read_prompts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("prompts must be a JSON array")
    for row in rows:
        if not isinstance(row.get("prompt_id"), str) or not isinstance(row.get("prompt"), str):
            raise ValueError("each prompt needs prompt_id and prompt strings")
    return rows


def written_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("family") and row.get("prompt_id"):
                        keys.add((row["family"], row["prompt_id"]))
    return keys


def _load_model(snapshot: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # No device_map: transformers' allocator warm-up dies on the A770 (gotcha); this matches neuron_report.load_model.
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, dtype=torch.bfloat16)
    model.to(device).eval()
    torch.set_grad_enabled(False)
    return torch, tokenizer, model


def _row_completion(tokenizer: Any, ids: list[int]) -> tuple[str, int]:
    eos = tokenizer.eos_token_id
    if eos is not None and eos in ids:
        ids = ids[:ids.index(eos) + 1]
    return tokenizer.decode(ids, skip_special_tokens=True), len(ids)


def generate_family(family: str, model_name: str, folder: str, prompts: list[dict[str, str]],
                    args: argparse.Namespace, stop_after: int | None = None) -> int:
    import torch

    out_path = Path(args.out)
    already = written_keys(out_path)
    pending = [p for p in prompts if (family, p["prompt_id"]) not in already]
    if stop_after is not None:
        pending = pending[:stop_after]
    if not pending:
        return 0
    snapshot = find_snapshot(Path(args.models_root), folder)
    torch, tokenizer, model = _load_model(snapshot, args.device)
    encoded = [(p, encode_prompt(tokenizer, p["prompt"], family)) for p in pending]
    encoded.sort(key=lambda item: len(item[1]["input_ids"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    start = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as output:
        for offset in range(0, len(encoded), args.batch_size):
            batch = encoded[offset:offset + args.batch_size]
            inputs = tokenizer.pad([x[1] for x in batch], padding=True, return_tensors="pt")
            # OLMo's tokenizer also returns token_type_ids, which generate refuses.
            inputs = {name: inputs[name].to(args.device) for name in ("input_ids", "attention_mask")}
            input_width = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for index, (prompt, _) in enumerate(batch):
                ids = generated[index, input_width:].tolist()
                if len(set(ids)) <= 2:
                    # An A770 run can return all-NaN logits with no error (gotcha); those decode as one token repeated.
                    raise RuntimeError(f"{family} {prompt['prompt_id']}: degenerate output, {len(set(ids))} distinct tokens")
                answer, token_count = _row_completion(tokenizer, ids)
                row = {"family": family, "model": model_name, "prompt_id": prompt["prompt_id"],
                       "answer": answer, "token_count": token_count}
                output.write(json.dumps(row, ensure_ascii=True) + "\n")
                written += 1
            output.flush()
            if family == "pythia" and offset == 2 * args.batch_size:
                rate = written / max(time.perf_counter() - start, 1e-9)
                print(f"pythia first 3 batches: {rate:.3f} answers/s", flush=True)
                if rate < 0.5:
                    raise RuntimeError(f"Pythia throughput {rate:.3f} answers/s is below 0.5")
    del model, tokenizer
    gc.collect()
    if args.device.startswith("xpu") and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--pilot", action="store_true", help="run 8 Pythia answers and exit")
    parser.add_argument("--family", choices=[m[0] for m in MODELS], help="one model, in its own process")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
    except ImportError as exc:
        print(f"psutil is required for the Windows free-RAM guard: {exc}", file=sys.stderr)
        return 2
    print(f"Windows free RAM: {free_gb:.2f} GB")
    if free_gb < 10:
        print("free RAM is below 10 GB; exiting", file=sys.stderr)
        return 2
    prompts = read_prompts(args.prompts)
    if args.pilot:
        n = generate_family("pythia", MODELS[0][1], MODELS[0][2], prompts, args, stop_after=8)
        print(f"pilot wrote {n} Pythia answers")
        return 0
    total = 0
    for family, model_name, folder in MODELS:
        if args.family and family != args.family:
            continue
        total += generate_family(family, model_name, folder, prompts, args)
        gc.collect()
    print(f"wrote {total} answer rows")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
