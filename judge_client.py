"""Shared HTTP transport for judge.py and task-specific blind judges."""

from __future__ import annotations

import json
import os
import urllib.request


def call_prompt(
    content: str,
    *,
    model: str,
    route: str = "ocgo",
    session: str = "neuron-judge",
    max_tokens: int = 16000,
    effort: str | None = None,
    reasoning_effort: str | None = None,
    no_thinking: bool = False,
    api_key: str | None = None,
    timeout: int = 600,
) -> dict:
    """Send one user prompt using judge.py's three supported routes."""
    if route not in {"ocgo", "ocgo-chat", "cliproxy"}:
        raise ValueError(f"unsupported judge route: {route}")

    key = api_key if api_key is not None else os.environ.get("OPENCODE_GO_API_KEY", "")
    if route == "cliproxy":
        key = "x"
    elif not key:
        raise RuntimeError("OPENCODE_GO_API_KEY not set")

    body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    if effort and route != "ocgo-chat":
        body["output_config"] = {"effort": effort}
    if no_thinking:
        body["thinking"] = {"type": "disabled"}

    if route == "ocgo-chat":
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        url = "https://opencode.ai/zen/go/v1/chat/completions"
        headers = {"content-type": "application/json", "authorization": f"Bearer {key}"}
    else:
        if route == "cliproxy":
            body["thinking"] = {"type": "adaptive"}
            url = "http://127.0.0.1:8317/v1/messages"
        else:
            url = "https://opencode.ai/zen/go/v1/messages"
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": key,
        }

    headers["user-agent"] = "neuron-probe/0.1 (journal-claude)"
    headers["x-opencode-session"] = session
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST", headers=headers
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        result = json.load(response)

    if route == "ocgo-chat":
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        usage = result.get("usage") or {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        stop = choice.get("finish_reason")
    else:
        text = "\n".join(
            item.get("text", "")
            for item in result.get("content", [])
            if item.get("type") == "text"
        )
        usage = result.get("usage") or {}
        reasoning = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
        if reasoning is None:
            chars = sum(
                len(item.get("thinking", ""))
                for item in result.get("content", [])
                if item.get("type") == "thinking"
            )
            reasoning = {"thinking_chars": chars} if chars else None
        stop = result.get("stop_reason")

    return {
        "text": text,
        "usage": usage,
        "reasoning_tokens": reasoning,
        "stop_reason": stop,
        "served": result.get("model"),
    }
