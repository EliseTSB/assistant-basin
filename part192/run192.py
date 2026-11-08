#!/usr/bin/env python3
"""Run Part 39 with an eager attention loader and a checked per-invocation load record.

Use the same subcommands and options as part39.py. If --out is omitted, outputs go to
results/part192. Before scoring, run `run192.py --check-record results/part192/load.json`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUESTED_ATTENTION = "eager"
# score loads no model, so it is not wrapped: run part39.py score directly (steps.txt).
RERUN_COMMANDS = {"reads", "screen", "zero", "push"}


def eager_load_model_direct(model_id: str, revision: str, device: str):
    """Part 39's direct loader with one required change: request and record eager attention."""
    import torch
    import neuron_report as nr
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = {
        "revision": revision,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
        "attn_implementation": REQUESTED_ATTENTION,
    }

    modeling_utils = getattr(transformers, "modeling_utils", None)
    if modeling_utils is None:
        import importlib
        modeling_utils = importlib.import_module("transformers.modeling_utils")
    if hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *args, **kw: None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    torch.set_grad_enabled(False)

    # Record what the config says was set, never what was asked for: a config without the attribute records
    # None, and the gate then refuses the run.
    config_value = getattr(getattr(model, "config", None), "_attn_implementation", None)
    nr.LAST_LOAD = {
        "model": model_id,
        "revision": revision,
        "device": device,
        "requested_attn_implementation": REQUESTED_ATTENTION,
        "attn_implementation": config_value if isinstance(config_value, str) and config_value else None,
    }
    return tok, model


def check_kernels(kernels: list[str | None]) -> tuple[bool, str]:
    """A record is valid only when it has at least one model load and every load says eager."""
    if not kernels:
        return False, "no model load recorded"
    missing = [index for index, kernel in enumerate(kernels) if not kernel]
    if missing:
        return False, f"missing attention kernel in model load(s): {missing!r}"
    bad = [(index, kernel) for index, kernel in enumerate(kernels) if kernel != REQUESTED_ATTENTION]
    if bad:
        return False, f"non-eager model load(s) recorded: {bad!r}"
    return True, ""


def _kernels_from_record(record: dict) -> list[str | None]:
    kernels: list[str | None] = []
    for invocation in record.get("loads", []):
        model_loads = invocation.get("model_loads") if isinstance(invocation, dict) else None
        if not isinstance(model_loads, list) or not model_loads:
            kernels.append(None)
            continue
        for loaded in model_loads:
            kernels.append(loaded.get("attn_implementation") if isinstance(loaded, dict) else None)
    return kernels


def check_record(path: str | Path, expect: list[str] | None = None) -> int:
    """Host-side gate. Return 0 only when the record is present, every load is eager, and every expected stage
    (for example reads, screen, zero, push) has at least one recorded entry with a model load."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"PART192 RECORD FAILED: cannot read load record {path}: {exc}. The device stages write "
              "load.json into the job folder's results\\part192; copy that whole folder back first.", flush=True)
        return 3
    if not isinstance(record, dict):
        print("PART192 RECORD FAILED: load record is not an object", flush=True)
        return 3
    ok, why = check_kernels(_kernels_from_record(record))
    if ok and expect:
        seen = {entry.get("command") for entry in record.get("loads", [])
                if isinstance(entry, dict) and entry.get("model_loads")}
        missing = [command for command in expect if command not in seen]
        if missing:
            ok, why = False, f"no recorded entry for stage(s) {missing!r}"
    print("PART192 record ok: all recorded model loads are eager" +
          (f", stages present: {', '.join(expect)}" if expect else "") if ok else
          f"PART192 RECORD FAILED: {why}", flush=True)
    return 0 if ok else 3


def _row_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                key = row.get("row_key")
                if isinstance(key, str) and key:
                    keys.add(key)
    return keys


def _load_record(path: Path) -> dict:
    if not path.exists():
        return {"part192": {"reruns": "Part 39", "requested_attention": REQUESTED_ATTENTION},
                "loads": []}
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("loads"), list):
        raise SystemExit(f"{path} is malformed; preserve it and use a fresh --out")
    return record


def main(argv: list[str] | None = None, *, part39_module=None, loader=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--check-record":
        expect = None
        if len(args) == 4 and args[2] == "--expect":
            expect = [command for command in args[3].split(",") if command]
        elif len(args) != 2:
            print("usage: run192.py --check-record <load.json> [--expect reads,screen,zero,push]", file=sys.stderr)
            return 2
        return check_record(args[1], expect)
    if not args or args[0] not in RERUN_COMMANDS:
        print("usage: run192.py {reads,screen,zero,push} [Part 39 options]", file=sys.stderr)
        return 2

    if part39_module is None:
        import part39 as part39_module
    import neuron_report as nr

    # Amendment 1: `screen --screen-model chat|base` runs one model's half of Part 39's screen per process. In one
    # process the second model's load tripped the job runner's 2 GB RAM guard twice (2026-11-08 00:44 and 01:10).
    screen_model = None
    if "--screen-model" in args:
        at = args.index("--screen-model")
        if at + 1 >= len(args) or args[at + 1] not in {"chat", "base"}:
            print("--screen-model needs chat or base", file=sys.stderr)
            return 2
        screen_model = args[at + 1]
        args = args[:at] + args[at + 2:]

    # Keep the original Part 39 parser and runner. The wrapper only supplies its eager loader and records writes.
    if "--out" not in args:
        args.extend(["--out", "results/part192"])
    parsed = part39_module.build_parser().parse_args(args)
    if parsed.command not in RERUN_COMMANDS:
        return 2
    if screen_model is not None and parsed.command != "screen":
        print("--screen-model applies only to screen", file=sys.stderr)
        return 2
    recorded_command = parsed.command if screen_model is None else f"screen:{screen_model}"

    actual_loader = eager_load_model_direct if loader is None else loader
    model_loads: list[dict] = []
    written_outputs: list[str] = []

    def recorded_loader(model_id: str, revision: str, device: str):
        tok, model = actual_loader(model_id, revision, device)
        load = getattr(nr, "LAST_LOAD", None)
        model_loads.append(dict(load) if isinstance(load, dict) else None)
        return tok, model

    original_loader = part39_module.load_model_direct
    original_writer = part39_module._write_json_atomic

    def tracked_writer(value, path):
        result = original_writer(value, path)
        written_outputs.append(str(path))
        return result

    row_path = parsed.out / {"reads": "reads.jsonl", "zero": "zero.jsonl",
                             "push": "push.jsonl"}.get(parsed.command, "")
    before_keys = _row_keys(row_path) if parsed.command in {"reads", "zero", "push"} else set()
    original_run_screen = getattr(part39_module, "run_screen", None)

    def one_model_screen(screen_args):
        if screen_model == "chat":
            return {"chat": part39_module._screen_one_model(screen_args, "chat", screen_args.model,
                                                            screen_args.revision)}
        return {"base": part39_module._screen_one_model(screen_args, "base", screen_args.base_model,
                                                        screen_args.base_revision)}

    part39_module.load_model_direct = recorded_loader
    part39_module._write_json_atomic = tracked_writer
    if screen_model is not None:
        part39_module.run_screen = one_model_screen
    nr.LAST_LOAD = None
    try:
        part39_module.main(args)
    finally:
        part39_module.load_model_direct = original_loader
        part39_module._write_json_atomic = original_writer
        if original_run_screen is not None:
            part39_module.run_screen = original_run_screen

    rows_added = len(_row_keys(row_path) - before_keys) if parsed.command in {"reads", "zero", "push"} else 0
    if model_loads and (rows_added or written_outputs):
        record_path = parsed.out / "load.json"
        record = _load_record(record_path)
        record.setdefault("loads", []).append({
            "command": recorded_command,
            "model_loads": model_loads,
            "argv": args,
            "rows_added": rows_added,
            "outputs_written": written_outputs,
        })
        record_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = record_path.with_name(record_path.name + ".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, record_path)

    gate_path = parsed.out / "load.json"
    try:
        record = _load_record(gate_path)
    except (OSError, ValueError, SystemExit) as exc:
        print(f"PART192 GATE FAILED: {exc}", flush=True)
        return 3
    kernels = _kernels_from_record(record)
    ok, why = check_kernels(kernels)
    if not ok:
        print(f"PART192 GATE FAILED: {why}", flush=True)
        return 3
    print(f"PART192 gate passed: {len(kernels)} recorded model load(s), all eager", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
