#!/usr/bin/env python3
"""Token-exact prompt, greedy decode, and cache helpers for Part 125."""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "part92"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common92  # noqa: E402

QUESTIONS = (
    "Who are you?",
    "Tell me about yourself.",
    "What is the capital of France?",
    "Describe your perfect day.",
    "Who are you?",
)
MAX_NEW_TOKENS = 120


def _ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _replace_once(sequence: list[int], needle: list[int], replacement: list[int]) -> list[int]:
    if not needle:
        raise ValueError("template placeholder tokenized to an empty sequence")
    matches = [index for index in range(len(sequence) - len(needle) + 1)
               if sequence[index:index + len(needle)] == needle]
    if len(matches) != 1:
        raise ValueError(f"expected one chat-template placeholder, found {len(matches)}")
    start = matches[0]
    return sequence[:start] + replacement + sequence[start + len(needle):]


def render_ids(tokenizer, questions: list[str], answers: list[dict], *,
               current_question: str | None = None, generation_prompt: bool = True) -> list[int]:
    """Render chat framing with the pinned template and splice stored answer IDs.

    Generated text is never decoded and tokenized again. The assistant message's
    template EOS is the generated EOS, or the explicit synthetic closure for a
    capped answer. Only its body IDs are spliced into the template placeholder.
    """
    if len(questions) != len(answers) and len(questions) != len(answers) + 1:
        raise ValueError("questions and completed answers must describe a consecutive transcript")
    if len(answers) > len(questions):
        raise ValueError("there cannot be more answers than questions")
    messages: list[dict[str, str]] = []
    placeholders: list[tuple[str, list[int], list[int]]] = []
    for index, question in enumerate(questions):
        messages.append({"role": "user", "content": str(question)})
        if index < len(answers):
            marker = f"P125TOKENLEDGERANSWER{index}X"
            messages.append({"role": "assistant", "content": marker})
            token_ids = [int(token) for token in answers[index]["token_ids"]]
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is None:
                raise ValueError("tokenizer must define eos_token_id for assistant turn closure")
            if not token_ids or token_ids[-1] != int(eos_id):
                raise ValueError(f"answer {index} is missing its EOS turn closure")
            placeholders.append((marker, _ids(tokenizer, marker), token_ids[:-1]))
    if current_question is not None:
        if len(questions) != len(answers):
            raise ValueError("current question is only valid after all listed answers")
        messages.append({"role": "user", "content": current_question})
    if not messages:
        raise ValueError("cannot render an empty transcript")

    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=bool(generation_prompt),
        tokenize=False,
        enable_thinking=False,
    )
    result = _ids(tokenizer, rendered)
    for _marker, needle, replacement in placeholders:
        result = _replace_once(result, needle, replacement)
    return result


def completed_ids(tokenizer, questions: list[str], answers: list[dict]) -> list[int]:
    """Template IDs for the completed transcript, including every EOS closure."""
    return render_ids(tokenizer, questions, answers, generation_prompt=False)


def next_prompt_ids(tokenizer, questions: list[str], answers: list[dict], question: str) -> list[int]:
    """Full template IDs through the current assistant generation prompt."""
    return render_ids(tokenizer, questions, answers, current_question=question,
                      generation_prompt=True)


def clone_cache(cache):
    """Clone a Transformers cache without aliasing its mutable tensors."""
    if cache is None:
        return None
    cloned = copy.deepcopy(cache)
    if cloned is cache:
        raise RuntimeError("cache clone returned the original object")
    return cloned


def cache_length(cache) -> int:
    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    if hasattr(cache, "token_ids"):
        return len(cache.token_ids)
    if hasattr(cache, "input_ids"):
        return int(cache.input_ids.shape[-1])
    raise TypeError(f"cannot read sequence length from cache {type(cache).__name__}")


def logit_hash(logits) -> str:
    """Hash one call's logits without changing their dtype or shape."""
    import torch

    value = logits.detach().contiguous().to(device="cpu")
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy().tobytes()
    else:
        raw = value.numpy().tobytes()
    prefix = f"{value.dtype}:{tuple(value.shape)}:".encode("utf-8")
    return hashlib.sha256(prefix + raw).hexdigest()


def run_segment(model, token_ids: list[int], cache, call_plan, *, device: str,
                hook=None, call_log: list | None = None, turn: int | None = None):
    """Run transcript IDs in planned chunks, optionally applying a hook per call.

    A plan is a sequence of chunk sizes or call-log rows. Call-log rows preserve
    their token IDs and planned hook state, so a paired path can replay the
    same calls with ``hook=None``.
    """
    import torch

    if not token_ids:
        raise ValueError("segment cannot have an empty token list")
    ids = list(map(int, token_ids))
    if not call_plan:
        raise ValueError("segment requires a non-empty call plan")

    plan = []
    for item in call_plan:
        if isinstance(item, dict):
            planned_ids = item.get("token_ids")
            count = len(planned_ids) if planned_ids is not None else int(item.get("token_count", 0))
            hook_requested = bool(item.get("hook_on", hook is not None))
            planned_shape = item.get("shape")
            if planned_shape is not None and planned_shape != [1, count]:
                raise ValueError("call plan shape does not match its token IDs")
            expected_before = item.get("cache_length_before")
            expected_after = item.get("cache_length_after")
        else:
            count = int(item)
            hook_requested = hook is not None
            planned_ids = None
            expected_before = expected_after = None
        if count < 1:
            raise ValueError("call plan chunks must have positive sizes")
        plan.append((count, hook_requested, planned_ids, expected_before, expected_after))
    if sum(count for count, _hook_on, _planned_ids, _before, _after in plan) != len(ids):
        raise ValueError("call plan chunk sizes do not cover the segment token IDs")

    offset = 0
    logits = None
    for count, hook_requested, planned_ids, expected_before, expected_after in plan:
        chunk = ids[offset:offset + count]
        offset += count
        if planned_ids is not None and list(map(int, planned_ids)) != chunk:
            raise AssertionError("call plan token IDs differ from the segment")
        hook_on = bool(hook is not None and hook_requested)
        hook_layer, push = hook if hook_on else (None, None)
        prior = cache_length(cache)
        if expected_before is not None and int(expected_before) != prior:
            raise AssertionError("call plan cache length before call does not match")
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        mask = torch.ones((1, prior + count), dtype=torch.long, device=device)
        positions = torch.arange(prior, prior + count, dtype=torch.long,
                                 device=device).unsqueeze(0)
        if hook_on:
            push = push.to(device=input_ids.device)
        handle = common92.add_row_hook(hook_layer, push) if hook_on else None
        try:
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    use_cache=True,
                    attention_mask=mask,
                    position_ids=positions,
                )
        finally:
            if handle is not None:
                handle.remove()
        updated = getattr(output, "past_key_values", None)
        if updated is None:
            raise RuntimeError("model did not return past_key_values")
        logits = output.logits[:, -1, :]
        cache = updated
        after = cache_length(cache)
        if expected_after is not None and int(expected_after) != after:
            raise AssertionError("call plan cache length after call does not match")
        if call_log is not None:
            row = {"token_ids": chunk, "shape": [1, count],
                   "cache_length_before": prior,
                   "cache_length_after": after,
                   "hook_on": hook_on,
                   "logit_hash": logit_hash(logits)}
            if turn is not None:
                row["turn"] = int(turn)
            call_log.append(row)
    return logits, cache


def call_forward(model, token_ids: list[int], cache, *, device: str,
                 hook_layer=None, push=None, call_log: list | None = None,
                 turn: int | None = None):
    """Compatibility wrapper for a one-call segment."""
    hook = (hook_layer, push) if hook_layer is not None and push is not None else None
    plan = [{"token_count": len(token_ids), "hook_on": hook is not None}]
    logits, cache = run_segment(model, token_ids, cache, plan, device=device,
                                hook=hook, call_log=call_log, turn=turn)
    return logits, cache, None


def feed_chunks(model, chunks: list[list[int]], cache, *, device: str,
                hook_layer=None, push=None, call_log: list | None = None,
                turn: int | None = None):
    if not chunks or any(not chunk for chunk in chunks):
        raise ValueError("at least one non-empty chunk is required")
    hook = (hook_layer, push) if hook_layer is not None and push is not None else None
    return run_segment(model, [token for chunk in chunks for token in chunk],
                       cache, [len(chunk) for chunk in chunks], device=device,
                       hook=hook, call_log=call_log, turn=turn)


def feed_one_by_one(model, token_ids: list[int], cache, *, device: str,
                    hook_layer=None, push=None, call_log: list | None = None,
                    turn: int | None = None):
    if not token_ids:
        raise ValueError("at least one token is required")
    hook = (hook_layer, push) if hook_layer is not None and push is not None else None
    return run_segment(model, list(map(int, token_ids)), cache,
                       [1] * len(token_ids), device=device, hook=hook,
                       call_log=call_log, turn=turn)


def greedy_answer(model, tokenizer, logits, cache, *, device: str,
                  max_new_tokens: int = MAX_NEW_TOKENS, hook_layer=None, push=None,
                  call_log: list | None = None,
                  hook=None, turn: int | None = None) -> tuple[dict, object]:
    """Greedy decode one token at a time and always cache the EOS closure."""
    import torch

    eos_value = getattr(tokenizer, "eos_token_id", None)
    if eos_value is None:
        raise ValueError("tokenizer has no EOS token")
    eos_id = int(eos_value)
    if hook is not None and (hook_layer is not None or push is not None):
        raise ValueError("pass either hook or hook_layer/push, not both")
    if hook is None and hook_layer is not None and push is not None:
        hook = (hook_layer, push)
    generated: list[int] = []
    ended = False
    for _ in range(int(max_new_tokens)):
        token = int(torch.argmax(logits[0].float()).item())
        generated.append(token)
        logits, cache = run_segment(
            model, [token], cache, [{"token_count": 1, "hook_on": hook is not None}],
            device=device, hook=hook, call_log=call_log, turn=turn,
        )
        if token == eos_id:
            ended = True
            break
    cap_hit = not ended
    stored_ids = list(generated)
    if cap_hit:
        stored_ids.append(eos_id)
        _logits, cache = run_segment(
            model, [eos_id], cache, [{"token_count": 1, "hook_on": hook is not None}],
            device=device, hook=hook, call_log=call_log, turn=turn,
        )
    body = stored_ids[:-1]
    answer_text = tokenizer.decode(body, skip_special_tokens=True).strip()
    return ({"answer": answer_text, "token_ids": stored_ids,
             "generated_token_ids": generated, "token_count": len(generated),
             "cap_hit": bool(cap_hit),
             "closure": "synthetic_eos" if cap_hit else "model_eos"}, cache)


def residual_medians(model, tokenizer, device: str) -> dict[str, float]:
    """Measure Part 92 residual medians with one CPU transfer per layer hook."""
    import torch

    ids = common92.prompt_ids(tokenizer, "Who are you?", "chat")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0)
    captured: dict[int, object] = {}
    layers = common92.get_layers(model)
    handles = []
    for layer_no, layer in enumerate(layers):
        def make_hook(index):
            def capture(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                value = hidden[0, 1:].float().norm(dim=-1).median().detach().to(device="cpu")
                captured[index] = value
            return capture
        handles.append(layer.register_forward_hook(make_hook(layer_no)))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask,
                  position_ids=position_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(layers):
        raise RuntimeError("residual median capture missed one or more decoder layers")
    return {str(index): float(captured[index]) for index in range(len(layers))}


def target_sentence(screen_answer: str) -> str:
    text = str(screen_answer).strip()
    if not text:
        raise ValueError("Part 92 screen_answer is empty")
    for index, char in enumerate(text):
        if char in ".!?\n":
            text = text[:index + 1].strip()
            break
    return text


def target_ids(tokenizer, screen_answer: str) -> list[int]:
    tokens = _ids(tokenizer, target_sentence(screen_answer))
    if len(tokens) > 32:
        tokens = tokens[:32]
    return tokens


def assert_template_ledger(tokenizer, questions: list[str], answers: list[dict], ledger: list[int]) -> None:
    expected = completed_ids(tokenizer, questions, answers)
    if [int(token) for token in ledger] != expected:
        raise AssertionError("token ledger differs from chat-template transcript IDs")


def assert_equal_call_shapes(left: list[dict], right: list[dict]) -> bool:
    if len(left) != len(right):
        raise AssertionError(f"twin call count mismatch: {len(left)} vs {len(right)}")
    for index, (a, b) in enumerate(zip(left, right)):
        fields = ("token_ids", "shape", "cache_length_before", "cache_length_after", "turn")
        if any(a.get(field) != b.get(field) for field in fields):
            raise AssertionError(f"twin call mismatch at call {index}")
    return True

