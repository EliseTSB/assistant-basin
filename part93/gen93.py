"""Generate one answer per Part 93 prompt, with resumable family JSONL files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import ssl
import threading
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MAX_OUTPUT_TOKENS = 300
FAMILY_MODELS = {
    "claude": "claude-haiku-5-5",
    "gpt": "gpt-6-luna[1m]",
    "deepseek": "deepseek-v4.1-flash",
}


def _anthropic_key() -> str:
    key_file = Path.home() / ".anthropic-credits.env"
    with key_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == "ANTHROPIC_API_KEY":
                return value.strip().strip("\"'")
    raise RuntimeError(f"ANTHROPIC_API_KEY was not found in {key_file}")


def _go_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def build_request(
    family: str,
    prompt: str,
    *,
    go_base: str | None = None,
    gemini_model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Build the HTTP request description; kept separate for offline checks."""
    if family == "claude":
        key = api_key if api_key is not None else _anthropic_key()
        return {
            "url": "https://api.anthropic.com/v1/messages",
            "headers": {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            "body": {
                "model": FAMILY_MODELS["claude"],
                "max_tokens": MAX_OUTPUT_TOKENS,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": prompt}],
            },
            "no_proxy": False,
        }
    if family == "gpt":
        return {
            "url": "http://127.0.0.1:8317/v1/chat/completions",
            "headers": {"content-type": "application/json"},
            "body": {
                "model": FAMILY_MODELS["gpt"],
                "max_tokens": MAX_OUTPUT_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            },
            "no_proxy": True,
        }
    if family in {"deepseek", "gemini"}:
        if not go_base:
            raise ValueError("--go-base is required for the OpenCode Go families")
        model = "deepseek-v4.1-flash" if family == "deepseek" else gemini_model
        if not model:
            raise ValueError("--gemini-model is required for gemini")
        key = api_key if api_key is not None else os.environ.get("OPENCODE_GO_API_KEY", "")
        if not key:
            raise RuntimeError("OPENCODE_GO_API_KEY is not set")
        return {
            "url": _go_endpoint(go_base),
            "headers": {"Authorization": f"Bearer {key}", "content-type": "application/json"},
            "body": {
                "model": model,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            },
            "no_proxy": False,
        }
    raise ValueError(f"family has no HTTP route: {family}")


def post_json(spec: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(spec["body"]).encode("utf-8")
    request = urllib.request.Request(spec["url"], data=payload, headers=spec["headers"], method="POST")
    timeout = 120
    if spec.get("no_proxy"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        response_context = opener.open(request, timeout=timeout)
    else:
        cert_file = os.environ.get("SSL_CERT_FILE")
        context = ssl.create_default_context(cafile=cert_file or None)
        response_context = urllib.request.urlopen(request, timeout=timeout, context=context)
    with response_context as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_remote(family: str, data: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    if family == "claude":
        answer = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
    else:
        choices = data.get("choices") or []
        answer = choices[0].get("message", {}).get("content", "") if choices else ""
        if isinstance(answer, list):
            answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
    usage = data.get("usage")
    return str(answer), usage if isinstance(usage, dict) else None


def qwen_chat_inputs(tokenizer: Any, prompt: str) -> Any:
    """Apply Qwen's non-thinking chat template and tokenize without special tokens."""
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(rendered, add_special_tokens=False, return_tensors="pt")


class QwenRunner:
    def __init__(self, model_dir: str, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True).to(device)
        self.model.eval()
        self.lock = threading.Lock()

    def answer(self, prompt: str) -> tuple[str, dict[str, int]]:
        inputs = qwen_chat_inputs(self.tokenizer, prompt)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        with self.lock, self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_OUTPUT_TOKENS,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, input_length:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        return answer, {"prompt_tokens": input_length, "completion_tokens": int(generated.shape[-1])}


def load_prompts(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, list):
        raise ValueError("prompts93.json must contain a JSON array")
    for index, item in enumerate(prompts, 1):
        if item.get("id") != f"p{index:03d}" or not item.get("prompt") or not item.get("kind"):
            raise ValueError(f"invalid prompt record at index {index - 1}")
    return prompts[:limit] if limit is not None else prompts


def read_done_rows(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    latest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            latest[row.get("id", "")] = row
    for prompt_id, row in latest.items():
        if prompt_id and not row.get("error") and row.get("answer") is not None:
            done.add(prompt_id)
    return done


def _answer_one(family: str, prompt: dict[str, str], args: argparse.Namespace, qwen: QwenRunner | None) -> dict[str, Any]:
    model = args.gemini_model if family == "gemini" else FAMILY_MODELS.get(family, "Qwen/Qwen3.5-4B")
    try:
        if family == "qwen":
            if qwen is None:
                raise RuntimeError("Qwen runner was not initialized")
            answer, usage = qwen.answer(prompt["prompt"])
        else:
            key = _anthropic_key() if family == "claude" else None
            spec = build_request(
                family,
                prompt["prompt"],
                go_base=args.go_base,
                gemini_model=args.gemini_model,
                api_key=key,
            )
            data = post_json(spec)
            answer, usage = _extract_remote(family, data)
        return {
            "id": prompt["id"], "family": family, "model": model,
            "prompt_id": prompt["id"], "answer": answer, "usage": usage, "error": None,
        }
    except Exception as exc:
        return {
            "id": prompt["id"], "family": family, "model": model,
            "prompt_id": prompt["id"], "answer": None, "usage": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_family(args: argparse.Namespace) -> int:
    prompts = load_prompts(Path(args.prompts), args.limit)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{args.family}.jsonl"
    done = read_done_rows(output_path)
    pending = [prompt for prompt in prompts if prompt["id"] not in done]
    if args.family == "qwen" and pending:
        if not args.model:
            raise ValueError("--model DIR is required for qwen")
        device = args.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        qwen = QwenRunner(args.model, device)
    else:
        qwen = None

    if pending:
        with output_path.open("a", encoding="utf-8") as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = {
                    pool.submit(_answer_one, args.family, prompt, args, qwen): prompt
                    for prompt in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
    return len(pending)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--family", required=True, choices=("claude", "gpt", "deepseek", "gemini", "qwen"))
    result.add_argument("--out", required=True, help="directory for per-family JSONL")
    result.add_argument("--workers", type=int, default=1)
    result.add_argument("--limit", type=int, help="run only the first N prompts")
    result.add_argument("--prompts", default=str(ROOT / "prompts93.json"))
    result.add_argument("--go-base", help="OpenCode Go OpenAI-compatible base URL")
    result.add_argument("--gemini-model", help="OpenCode Go Gemini model id")
    result.add_argument("--model", help="local Qwen model directory")
    result.add_argument("--device", default="auto", help="Qwen device, for example cuda or cpu")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        written = run_family(args)
    except Exception as exc:
        parser().error(str(exc))
    print(f"{args.family}: wrote or retried {written} answer rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
