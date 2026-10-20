#!/usr/bin/env python3
"""Collect Part 29 answers through OpenCode Go, CLIProxy, or local HF models."""

from __future__ import annotations

import argparse
import concurrent.futures
import email.utils
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


GO_BASE_URL = "https://opencode.ai/zen/go"
CLIPROXY_CHAT_URL = "http://127.0.0.1:8317/v1/chat/completions"
DEFAULT_PROMPTS = "data/part29-prompts.json"
DEFAULT_OUTPUT_DIR = "results/part29"
# 300 answer tokens is the design, but reasoning models spend the whole budget thinking (host smoke 2026-10-19:
# DeepSeek V4 Pro, GLM 5.3 and Kimi K3 returned empty text at 300). The cap leaves room for reasoning; the
# classifier reads only the first ANSWER_WORDS words of the final answer (part29_classify.py).
MAX_TOKENS = 4000
HF_MAX_NEW = 600  # local CPU generation: about 300 answer tokens plus room; --hf-max-new overrides
TEMPERATURE = 0.7
MAX_TRIES = 3
MAX_WORKERS = 4

# Each entry is a fixed, reviewable model ID and protocol. Eight IDs are the
# classifier's teacher families; DeepSeek V4.1 Flash and MiMo V2.6 Flash are
# additional student models.
MODEL_TABLE: dict[str, dict[str, str]] = {
    "deepseek-v4-pro": {"provider": "opencode-go-chat", "protocol": "chat"},
    "deepseek-v4.1-flash": {"provider": "opencode-go-chat", "protocol": "chat"},
    "glm-5.3": {"provider": "opencode-go-chat", "protocol": "chat"},
    "kimi-k3": {"provider": "opencode-go-chat", "protocol": "chat"},
    "qwen3.8-max": {"provider": "opencode-go-chat", "protocol": "chat"},
    "minimax-m3": {"provider": "opencode-go-chat", "protocol": "chat"},
    "gemini-4-argon-contributor": {"provider": "opencode-go-google", "protocol": "google"},  # Go refuses it on chat completions (ModelProtocolUnsupported)
    "mimo-v2.6-flash": {"provider": "opencode-go-chat", "protocol": "chat"},
    "claude-haiku-5-5": {"provider": "opencode-go-messages", "protocol": "messages"},
    "gpt-6-luna": {"provider": "cliproxy", "protocol": "chat"},
}


def answer_after_think(text: str) -> str:
    """Return the content after the last close marker, or all text if absent."""
    marker = "</think>"
    if marker not in text:
        return text
    return text.rsplit(marker, 1)[1].strip()


def prompt_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON object with a prompts array")
    out = []
    for fallback_index, row in enumerate(records):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
            raise ValueError(f"invalid prompt record at position {fallback_index}")
        out.append({"index": int(row.get("index", fallback_index)), "prompt": row["prompt"]})
    return out


def output_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_.") or "model"


def existing_indices(path: Path) -> set[int]:
    indices: set[int] = set()
    if not path.exists():
        return indices
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                indices.add(int(row["index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Ignore a malformed prior line; successful records remain
                # usable and the missing index will be retried on this run.
                continue
    return indices


def retry_delay(headers: Any, attempt: int) -> float:
    value = headers.get("Retry-After") if headers is not None else None
    if value:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                when = email.utils.parsedate_to_datetime(value).timestamp()
                return max(0.0, when - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    return float(2**attempt)


def _open(req: urllib.request.Request, timeout: int = 600):
    # Always disable inherited proxy settings for localhost so CLIProxy calls
    # cannot accidentally leave the machine through an HTTPS proxy.
    if req.full_url.startswith("http://127.0.0.1") or req.full_url.startswith("http://localhost"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _extract_openai_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    return ""


def parse_response(protocol: str, payload: dict[str, Any]) -> dict[str, Any]:
    if protocol == "messages":
        blocks = payload.get("content") or []
        text = "\n".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        reasoning_parts = [
            str(block.get("thinking", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "thinking" and block.get("thinking")
        ]
        result = {
            "text": text,
            "finish_reason": payload.get("stop_reason"),
            "usage": payload.get("usage") or {},
        }
        if reasoning_parts:
            result["reasoning"] = "\n".join(reasoning_parts)
        return result

    if protocol == "google":
        cands = payload.get("candidates") or []
        if not cands or not isinstance(cands[0], dict):
            raise ValueError("generateContent has no candidates")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(str(pt.get("text", "")) for pt in parts if isinstance(pt, dict) and not pt.get("thought"))
        thoughts = "".join(str(pt.get("text", "")) for pt in parts if isinstance(pt, dict) and pt.get("thought"))
        result = {"text": text, "finish_reason": cands[0].get("finishReason"), "usage": payload.get("usageMetadata") or {}}
        if thoughts:
            result["reasoning"] = thoughts
        return result

    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("chat completion has no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    result = {
        "text": _extract_openai_text(message.get("content")),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage") or {},
    }
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        result["reasoning"] = reasoning
    return result


def request_api_answer(model_id: str, prompt: str) -> dict[str, Any]:
    """Request one answer from the fixed API table, retrying rate/server errors."""
    spec = MODEL_TABLE[model_id]
    api_key = os.environ.get("OPENCODE_GO_API_KEY", "")
    if spec["protocol"] != "chat" or spec["provider"] != "cliproxy":
        if not api_key:
            raise RuntimeError("OPENCODE_GO_API_KEY is required for OpenCode Go models")

    if spec["provider"] == "cliproxy":
        url = CLIPROXY_CHAT_URL
        headers = {"content-type": "application/json"}
    elif spec["protocol"] == "messages":
        url = f"{GO_BASE_URL}/v1/messages"
        headers = {
            "content-type": "application/json",
            "x-api-key": api_key,  # the auth recognition-probe.py proved on Go's messages route (10-18)
            "anthropic-version": "2023-06-01",
            "x-opencode-session": "part29",
            "user-agent": "neuron-probe/0.1 (journal-claude)",
        }
    elif spec["protocol"] == "google":
        url = f"https://opencode.ai/inference/go/v1/models/{model_id}:generateContent"
        headers = {
            "content-type": "application/json",
            "x-goog-api-key": api_key,
            "x-opencode-session": "part29",
            "user-agent": "neuron-probe/0.1 (journal-claude)",
        }
    else:
        url = f"{GO_BASE_URL}/v1/chat/completions"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
            "x-opencode-session": "part29",
            "user-agent": "neuron-probe/0.1 (journal-claude)",
        }

    if spec["protocol"] == "google":
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": MAX_TOKENS},
        }
    else:
        body = {
            "model": model_id,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
    if model_id.startswith("claude-"):
        body.pop("temperature", None)  # Haiku 5.5 on Go: "`temperature` is deprecated for this model" (400, 2026-10-19)
    started = time.monotonic()
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(MAX_TRIES):
        req = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
        try:
            with _open(req) as response:
                payload = json.loads(response.read().decode("utf-8"))
            parsed = parse_response(spec["protocol"], payload)
            parsed["wall_time"] = round(time.monotonic() - started, 3)
            return parsed
        except urllib.error.HTTPError as exc:
            last_error = exc
            if (exc.code != 429 and not 500 <= exc.code < 600) or attempt + 1 >= MAX_TRIES:
                raise
            time.sleep(retry_delay(exc.headers, attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("request failed without a response")


def collect_api_model(
    model_id: str,
    records: list[dict[str, Any]],
    output: Path,
    workers: int = MAX_WORKERS,
) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    present = existing_indices(output)
    todo = [row for row in records if row["index"] not in present]
    if not todo:
        print(f"{model_id}: already complete ({len(present)} records)", flush=True)
        return 0, 0
    provider = MODEL_TABLE[model_id]["provider"]
    write_lock = threading.Lock()
    successes = 0
    failures = 0

    def work(row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        result = request_api_answer(model_id, row["prompt"])
        record: dict[str, Any] = {
            "index": row["index"],
            "model": model_id,
            "provider": provider,
            "text": result.get("text", ""),
            "answer": answer_after_think(result.get("text", "")),  # MiniMax M3 puts <think> inside the text
            "finish_reason": result.get("finish_reason"),
            "usage": result.get("usage") or {},
            "wall_time": result.get("wall_time"),
        }
        for optional in ("reasoning", "served_model"):
            if result.get(optional) is not None:
                record[optional] = result[optional]
        return row["index"], record

    with output.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
            future_map = {pool.submit(work, row): row for row in todo}
            for future in concurrent.futures.as_completed(future_map):
                row = future_map[future]
                try:
                    _index, result = future.result()
                    with write_lock:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
                    successes += 1
                except Exception as exc:  # keep failed indices absent for a resumable rerun
                    failures += 1
                    print(
                        f"{model_id} index {row['index']} failed: {str(exc)[:240]}",
                        file=sys.stderr,
                        flush=True,
                    )
    print(f"{model_id}: wrote {successes}; failed {failures}", flush=True)
    return successes, failures


def load_hf_model(spec: str, device_name: str):
    try:
        repo, revision = spec.rsplit("@", 1)
    except ValueError as exc:
        raise ValueError("--hf must be written as <repo>@<revision>") from exc
    if not repo or not revision:
        raise ValueError("--hf must include both a repository and revision")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("--device xpu requested, but torch.xpu is not available")
        device = "xpu"
    else:
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision).to(device)
    model.eval()
    return repo, revision, tokenizer, model, torch, device


def collect_hf_model(
    spec: str,
    records: list[dict[str, Any]],
    output: Path,
    device_name: str = "cpu",
) -> tuple[int, int]:
    try:
        repo, revision = spec.rsplit("@", 1)
    except ValueError as exc:
        raise ValueError("--hf must be written as <repo>@<revision>") from exc
    if not repo or not revision:
        raise ValueError("--hf must include both a repository and revision")
    model_id = f"{repo}@{revision}"
    provider = "huggingface"
    output.parent.mkdir(parents=True, exist_ok=True)
    present = existing_indices(output)
    todo = [row for row in records if row["index"] not in present]
    if not todo:
        print(f"{model_id}: already complete ({len(present)} records)", flush=True)
        return 0, 0
    _repo, _revision, tokenizer, model, torch, device = load_hf_model(spec, device_name)

    successes = failures = 0
    with output.open("a", encoding="utf-8") as handle:
        for row in todo:
            started = time.monotonic()
            try:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=False,  # Qwen3/3.5 templates think by default; others ignore the kwarg
                )
                if hasattr(rendered, "items"):
                    encoded = dict(rendered.items())
                    prompt_length = int(encoded["input_ids"].shape[-1])
                else:
                    encoded = {"input_ids": rendered}
                    prompt_length = int(rendered.shape[-1])
                encoded = {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        max_new_tokens=HF_MAX_NEW,
                    )
                new_tokens = generated[0, prompt_length:]
                full_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                eos = tokenizer.eos_token_id
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
                finish_reason = "stop" if len(new_tokens) and int(new_tokens[-1]) in eos_ids else "length"
                token_count = int(new_tokens.shape[-1])
                result = {
                    "index": row["index"],
                    "model": model_id,
                    "provider": provider,
                    "text": full_text,
                    "answer": answer_after_think(full_text),
                    "finish_reason": finish_reason,
                    "usage": {
                        "prompt_tokens": prompt_length,
                        "completion_tokens": token_count,
                        "total_tokens": prompt_length + token_count,
                    },
                    "wall_time": round(time.monotonic() - started, 3),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                successes += 1
            except Exception as exc:
                failures += 1
                print(f"{model_id} index {row['index']} failed: {str(exc)[:240]}", file=sys.stderr, flush=True)
    print(f"{model_id}: wrote {successes}; failed {failures}", flush=True)
    return successes, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", action="append", choices=sorted(MODEL_TABLE), help="API model ID; repeat to select several (default: all)")
    parser.add_argument("--hf", action="append", default=[], metavar="REPO@REVISION", help="local Hugging Face model; repeatable")
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="per model, capped at four")
    parser.add_argument("--first-index", type=int, default=0, help="skip prompts below this index (the classifier scores students on 240-299)")
    parser.add_argument("--hf-max-new", type=int, default=None)
    args = parser.parse_args()
    global HF_MAX_NEW
    if args.hf_max_new:
        HF_MAX_NEW = args.hf_max_new
    records = [r for r in prompt_records(Path(args.prompts)) if r["index"] >= args.first_index]
    output_dir = Path(args.out_dir)
    models = args.model if args.model else ([] if args.hf else list(MODEL_TABLE))  # --hf alone runs no API model
    total_failures = 0
    for model_id in models:
        _, failures = collect_api_model(
            model_id,
            records,
            output_dir / f"answers-{output_slug(model_id)}.jsonl",
            workers=args.workers,
        )
        total_failures += failures
    for spec in args.hf:
        try:
            repo, revision = spec.rsplit("@", 1)
        except ValueError as exc:
            raise SystemExit("--hf must be written as <repo>@<revision>") from exc
        model_id = f"{repo}@{revision}"
        _, failures = collect_hf_model(
            spec,
            records,
            output_dir / f"answers-{output_slug(model_id)}.jsonl",
            device_name=args.device,
        )
        total_failures += failures
    if total_failures:
        raise SystemExit(f"collection finished with {total_failures} failed prompt(s); rerun to resume")


if __name__ == "__main__":
    main()
