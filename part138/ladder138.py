#!/usr/bin/env python3
"""Measure OLMo 3 7B generation on the Windows A770, stopping on memory spill."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

MODEL_ID = "allenai/Olmo-3-7B-Instruct"
WINDOWS_MODEL_DIR = r"E:\neuron-probe\hf\hub\models--allenai--Olmo-3-7B-Instruct\snapshots"
QUESTIONS = [
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the largest planet in our solar system?",
    "How many sides does a hexagon have?",
    "What gas do plants absorb from the air?",
    "Who wrote Pride and Prejudice?",
    "What is 17 multiplied by 19?",
    "Which ocean lies between Africa and Australia?",
    "At what temperature does water freeze in Celsius?",
    "What is the chemical symbol for gold?",
    "Name the process by which plants make food using sunlight.",
    "How many minutes are in three hours?",
    "Which planet is known as the Red Planet?",
    "What is the opposite of an even number?",
    "Who painted the Mona Lisa?",
    "What is the square root of 144?",
]
BATCH_SIZES = (2, 4, 8, 16, 32)
# Amendment 2026-10-31: the spill stop skipped the repeats, so a rerun names its ladder and output.
if "--batches" in sys.argv:
    BATCH_SIZES = tuple(int(x) for x in sys.argv[sys.argv.index("--batches") + 1].split(","))
MAX_NEW_TOKENS = 64
SHARED_SPILL_BYTES = 1_000_000_000
MIN_AVAILABLE_RAM_MB = 3_000
OUTPUT = Path("results/part138/ladder.json")
if "--output" in sys.argv:
    OUTPUT = Path(sys.argv[sys.argv.index("--output") + 1])


def parse_counter_json(raw: str | dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """Normalize Get-Counter JSON (including PowerShell's singleton-object case)."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(obj, dict):
        obj = [obj]
    parsed = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        path = str(item.get("Path", item.get("path", "")))
        value = item.get("CookedValue", item.get("cookedValue", item.get("Value", item.get("value"))))
        try:
            cooked = float(value)
        except (TypeError, ValueError):
            continue
        parsed.append({"path": path, "value": cooked})
    return parsed


def adapter_usage(samples: list[dict[str, Any]]) -> dict[str, float | None]:
    """Extract peak dedicated/shared bytes across GPU adapter instances."""
    found: dict[str, list[float]] = {"dedicated": [], "shared": []}
    for sample in samples:
        path = sample["path"]
        if "DEDICATED USAGE" in path.upper():
            found["dedicated"].append(sample["value"])
        elif "SHARED USAGE" in path.upper():
            found["shared"].append(sample["value"])
    return {key: max(values) if values else None for key, values in found.items()}


def should_stop_for_spill(start_shared: float | None, current_shared: float | None,
                          limit: float = SHARED_SPILL_BYTES) -> bool:
    return start_shared is not None and current_shared is not None and current_shared - start_shared > limit


def should_stop(row: dict[str, Any]) -> bool:
    return bool(row.get("generation_error") or row.get("spill"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Replace the JSON atomically, keeping all completed rows readable after a freeze."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)


class WindowsCounters:
    """Sample RAM and all adapter usage counters from a Windows PowerShell process."""
    SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$c = @('\Memory\Available MBytes', '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage')
$s = Get-Counter -Counter $c -MaxSamples 1
$items = foreach ($x in $s.CounterSamples) { [pscustomobject]@{Path=$x.Path; CookedValue=$x.CookedValue} }
ConvertTo-Json -InputObject @($items) -Compress
"""

    def sample(self) -> dict[str, Any]:
        exe = "powershell.exe" if os.name == "nt" else "powershell"
        completed = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", self.SCRIPT],
                                   capture_output=True, text=True, timeout=20, check=True)
        counters = parse_counter_json(completed.stdout)
        ram = next((x["value"] for x in counters if "AVAILABLE MBYTES" in x["path"].upper()), None)
        adapter = adapter_usage(counters)
        return {"available_ram_mb": ram, **adapter, "sampled_at": time.time()}


class CounterMonitor:
    def __init__(self, sampler: WindowsCounters, interval: float = 2.0):
        self.sampler, self.interval = sampler, interval
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, name="part138-counters", daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.sampler.sample()
                self.samples.append(item)
                if item["available_ram_mb"] is not None and item["available_ram_mb"] < MIN_AVAILABLE_RAM_MB:
                    self.stop_event.set()
            except Exception as exc:  # retain the failure in the eventual record
                self.error = f"{type(exc).__name__}: {exc}"
                self.stop_event.set()
                break
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=25)

    @property
    def low_ram(self) -> bool:
        return any(s.get("available_ram_mb") is not None and
                   s["available_ram_mb"] < MIN_AVAILABLE_RAM_MB for s in self.samples)

    def maxima(self) -> dict[str, float | None]:
        return {key: max((s[key] for s in self.samples if s.get(key) is not None), default=None)
                for key in ("dedicated", "shared")}


def load_model(device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import modeling_utils

    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    # Resolve the local snapshot without consulting the hub or downloading weights.
    snapshots = Path(WINDOWS_MODEL_DIR)
    dirs = sorted(p for p in snapshots.iterdir() if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No local model snapshot under {snapshots}")
    snapshot = dirs[-1]
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    arch = (config.get("architectures") or [""])[0]
    model_cls = AutoModelForCausalLM
    if "ConditionalGeneration" in arch or "ImageTextToText" in arch:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = model_cls.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16,
                                      device_map={"": device}, local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    torch.set_grad_enabled(False)
    return tok, model, config


def encode_batch(tokenizer, questions: list[str], device: str):
    prompts = []
    for question in questions:
        if getattr(tokenizer, "chat_template", None):
            prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                                         tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(question)
    return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)


def windows_ram_mb(samples: list[dict[str, Any]]) -> float | None:
    return next((s["available_ram_mb"] for s in reversed(samples)
                 if s.get("available_ram_mb") is not None), None)


def record_stop(payload: dict[str, Any], status: str, reason: str,
                path: str | Path = OUTPUT, *, monitor_error: str | None = None) -> None:
    """Persist a terminal ladder status before control leaves the ladder."""
    payload.update(status=status, stop_reason=reason)
    if monitor_error is not None:
        payload["monitor.error"] = monitor_error
    write_json(path, payload)


def preflight_counters(sampler: WindowsCounters, monitor: CounterMonitor,
                       payload: dict[str, Any], path: str | Path = OUTPUT) -> dict[str, Any] | None:
    """Read the required initial sample, recording a terminal status on failure."""
    try:
        initial = sampler.sample()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        monitor.error = error
        record_stop(payload, "stopped_counter_error", "GPU/RAM counter sample failed before model load",
                    path, monitor_error=error)
        return None
    monitor.samples.append(initial)
    return initial


def measure_batch(tokenizer, model, device: str, batch_size: int, monitor: CounterMonitor,
                  start_shared: float | None, first_answers: dict[int, list[str]] | None = None) -> dict[str, Any]:
    import torch
    questions = QUESTIONS[:]
    if batch_size > len(questions):
        questions = (questions * ((batch_size + len(questions) - 1) // len(questions)))[:batch_size]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    error = None
    answers: list[str] = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(questions), batch_size):
                if monitor.low_ram:
                    break
                qs = questions[offset:offset + batch_size]
                inputs = encode_batch(tokenizer, qs, device)
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
                prompt_width = inputs["input_ids"].shape[1]
                answers.extend(tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = max(time.perf_counter() - started, 1e-9)
    samples = list(monitor.samples)
    now_shared = monitor.maxima()["shared"]
    spill = should_stop_for_spill(start_shared, now_shared)
    xpu = getattr(torch, "xpu", None)
    allocated = reserved = peak = None
    if xpu is not None:
        try:
            allocated = int(xpu.memory_allocated(device))
            reserved = int(xpu.memory_reserved(device))
            peak = int(xpu.max_memory_allocated(device))
        except (AttributeError, RuntimeError):
            pass
    france_index = questions.index("What is the capital of France?")
    france_ok = len(answers) > france_index and "paris" in answers[france_index].casefold()
    if first_answers is not None and error is None and len(answers) == len(questions):
        expected = first_answers.get(batch_size)
        if expected is not None and answers != expected:
            error = "greedy answers differed from the first pass at this batch size"
    row = {"batch_size": batch_size, "answers": answers, "answers_per_s": len(answers) / elapsed,
           "elapsed_s": elapsed, "windows_ram_available_mb": windows_ram_mb(samples),
           "gpu_dedicated_usage_bytes_max": monitor.maxima()["dedicated"],
           "gpu_shared_usage_bytes_max": now_shared, "gpu_shared_usage_start_bytes": start_shared,
           "torch_xpu_memory_allocated_bytes": allocated, "torch_xpu_memory_reserved_bytes": reserved,
           "torch_xpu_max_memory_allocated_bytes": peak, "france_contains_paris": france_ok,
           "spill": spill, "generation_error": error, "low_ram_stop": monitor.low_ram}
    return row


def run() -> int:
    if os.name != "nt":
        raise RuntimeError("This ladder must run on Windows through tools/launch_arc.py")
    import torch
    device = "xpu"
    if not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is not available")
    payload: dict[str, Any] = {"model_id": MODEL_ID, "snapshot_root": WINDOWS_MODEL_DIR,
                               "batch_sizes": list(BATCH_SIZES), "max_new_tokens": MAX_NEW_TOKENS,
                               "questions": QUESTIONS, "rows": [], "repeats": [], "status": "starting"}
    # Ensure a valid incremental artifact exists even if loading freezes or fails.
    write_json(OUTPUT, payload)
    sampler = WindowsCounters()
    monitor = CounterMonitor(sampler)
    try:
        # A successful counter read is a prerequisite for loading or generating.
        initial = preflight_counters(sampler, monitor, payload)
        if initial is None:
            return 0
        if initial["available_ram_mb"] is not None and initial["available_ram_mb"] < MIN_AVAILABLE_RAM_MB:
            record_stop(payload, "stopped_low_ram", "Windows RAM below 3000 MB before model load")
            return 0
        start_shared = initial.get("shared")
        monitor.start()
        tokenizer, model, config = load_model(device)
        if monitor.error:
            record_stop(payload, "stopped_counter_error", "GPU/RAM counter sample failed after startup",
                        monitor_error=monitor.error)
            return 0
        payload.update(config={k: config.get(k) for k in ("architectures", "torch_dtype", "model_type")},
                       status="running")
        write_json(OUTPUT, payload)
        first_answers: dict[int, list[str]] = {}
        largest_safe = None
        for batch_size in BATCH_SIZES:
            if monitor.low_ram:
                record_stop(payload, "stopped_low_ram", "Windows RAM dropped below 3000 MB")
                break
            if monitor.error:
                record_stop(payload, "stopped_counter_error", "GPU/RAM counter sample failed",
                            monitor_error=monitor.error)
                break
            torch.xpu.reset_peak_memory_stats(device)
            row = measure_batch(tokenizer, model, device, batch_size, monitor, start_shared)
            if monitor.error:
                payload["rows"].append(row)
                record_stop(payload, "stopped_counter_error", "GPU/RAM counter sample failed during generation",
                            monitor_error=monitor.error)
                break
            if monitor.low_ram:
                record_stop(payload, "stopped_low_ram", "Windows RAM dropped below 3000 MB")
                break
            payload["rows"].append(row)
            first_answers[batch_size] = row["answers"]
            print(json.dumps({k: row[k] for k in ("batch_size", "answers_per_s", "windows_ram_available_mb",
                                                   "gpu_dedicated_usage_bytes_max", "gpu_shared_usage_bytes_max",
                                                   "torch_xpu_memory_allocated_bytes", "torch_xpu_memory_reserved_bytes",
                                                   "torch_xpu_max_memory_allocated_bytes")}), flush=True)
            write_json(OUTPUT, payload)
            france_index = QUESTIONS.index("What is the capital of France?")
            france_answer = row["answers"][france_index] if len(row["answers"]) > france_index else ""
            if "paris" not in france_answer.casefold():
                record_stop(payload, "stopped_gate", "France capital answer did not contain Paris")
                return 0
            if should_stop(row):
                record_stop(payload, "stopped_spill" if row["spill"] else "stopped_generation_error",
                            "shared memory exceeded 1 GB over baseline" if row["spill"] else row["generation_error"])
                break
            largest_safe = batch_size
            torch.xpu.empty_cache()
        if payload["status"] == "running" and largest_safe is not None and not monitor.low_ram:
            for repeat_number in range(1, 4):
                if monitor.low_ram:
                    record_stop(payload, "stopped_low_ram", "Windows RAM dropped below 3000 MB")
                    break
                if monitor.error:
                    record_stop(payload, "stopped_counter_error", "GPU/RAM counter sample failed",
                                monitor_error=monitor.error)
                    break
                torch.xpu.empty_cache()
                torch.xpu.reset_peak_memory_stats(device)
                row = measure_batch(tokenizer, model, device, largest_safe, monitor, start_shared, first_answers)
                if monitor.error:
                    payload["repeats"].append({**row, "repeat": repeat_number})
                    record_stop(payload, "stopped_counter_error", "GPU/RAM counter sample failed during generation",
                                monitor_error=monitor.error)
                    break
                if monitor.low_ram:
                    record_stop(payload, "stopped_low_ram", "Windows RAM dropped below 3000 MB")
                    break
                row["repeat"] = repeat_number
                payload["repeats"].append(row)
                print(json.dumps({"repeat": repeat_number, "batch_size": largest_safe,
                                  "answers_per_s": row["answers_per_s"], "spill": row["spill"],
                                  "generation_error": row["generation_error"]}), flush=True)
                write_json(OUTPUT, payload)
                if should_stop(row):
                    record_stop(payload, "repeat_stopped", "repeat spill or generation failure")
                    break
        if payload["status"] == "running" and monitor.error:
            record_stop(payload, "stopped_counter_error", "GPU/RAM counter sample failed",
                        monitor_error=monitor.error)
        if payload["status"] == "running":
            payload["status"] = "complete"
        write_json(OUTPUT, payload)
        return 0
    except Exception as exc:
        payload.update(status="error", error=f"{type(exc).__name__}: {exc}")
        write_json(OUTPUT, payload)
        raise
    finally:
        monitor.stop()


if __name__ == "__main__":
    raise SystemExit(run())
