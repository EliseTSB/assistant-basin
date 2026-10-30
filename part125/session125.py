#!/usr/bin/env python3
"""Token-exact prompt, greedy decode, and cache helpers for Part 125."""
from __future__ import annotations

import copy
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


def call_forward(model, token_ids: list[int], cache, *, device: str,
                 hook_layer=None, push=None, call_log: list | None = None):
    """Run one causal forward call and return logits plus its updated cache."""
    import torch

    if not token_ids:
        raise ValueError("forward call cannot have an empty token chunk")
    input_ids = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=device)
    prior = cache_length(cache)
    mask = torch.ones((1, prior + len(token_ids)), dtype=torch.long, device=device)
    positions = torch.arange(prior, prior + len(token_ids), dtype=torch.long,
                             device=device).unsqueeze(0)
    handle = common92.add_row_hook(hook_layer, push) if push is not None else None
    handle_id = getattr(handle, "id", None) if handle is not None else None
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
    if call_log is not None:
        call_log.append({"token_ids": list(map(int, token_ids)), "shape": [1, len(token_ids)]})
    return logits, updated, handle_id


def feed_chunks(model, chunks: list[list[int]], cache, *, device: str,
                hook_layer=None, push=None, call_log: list | None = None):
    logits = None
    for chunk in chunks:
        logits, cache, _handle_id = call_forward(
            model, chunk, cache, device=device, hook_layer=hook_layer, push=push,
            call_log=call_log,
        )
    if logits is None:
        raise ValueError("at least one chunk is required")
    return logits, cache


def feed_one_by_one(model, token_ids: list[int], cache, *, device: str,
                    hook_layer=None, push=None, call_log: list | None = None):
    logits = None
    for token in token_ids:
        logits, cache, _handle_id = call_forward(
            model, [int(token)], cache, device=device, hook_layer=hook_layer, push=push,
            call_log=call_log,
        )
    if logits is None:
        raise ValueError("at least one token is required")
    return logits, cache


def greedy_answer(model, tokenizer, logits, cache, *, device: str,
                  max_new_tokens: int = MAX_NEW_TOKENS, hook_layer=None, push=None,
                  call_log: list | None = None) -> tuple[dict, object]:
    """Greedy decode one token at a time and always cache the EOS closure."""
    import torch

    eos_value = getattr(tokenizer, "eos_token_id", None)
    if eos_value is None:
        raise ValueError("tokenizer has no EOS token")
    eos_id = int(eos_value)
    generated: list[int] = []
    ended = False
    try:
        for _ in range(int(max_new_tokens)):
            token = int(torch.argmax(logits[0].float()).item())
            generated.append(token)
            logits, cache, _handle_id = call_forward(
                model, [token], cache, device=device, hook_layer=hook_layer, push=push,
                call_log=call_log,
            )
            if token == eos_id:
                ended = True
                break
        cap_hit = not ended
        stored_ids = list(generated)
        if cap_hit:
            stored_ids.append(eos_id)
            _logits, cache, _handle_id = call_forward(
                model, [eos_id], cache, device=device, hook_layer=hook_layer, push=push,
                call_log=call_log,
            )
        body = stored_ids[:-1]
        answer_text = tokenizer.decode(body, skip_special_tokens=True).strip()
        return ({"answer": answer_text, "token_ids": stored_ids,
                 "generated_token_ids": generated, "token_count": len(generated),
                 "cap_hit": bool(cap_hit),
                 "closure": "synthetic_eos" if cap_hit else "model_eos"}, cache)
    finally:
        # call_forward removes each temporary handle in its own finally block.
        pass


def teacher_force(model, tokenizer, cache, prefix_ids: list[int], target_ids: list[int], *,
                  device: str) -> tuple[float, object, object]:
    """Return target mean log probability and the next-token logits on a cache clone."""
    import torch

    clone = clone_cache(cache)
    logits, clone = feed_one_by_one(model, prefix_ids, clone, device=device)
    initial_logits = logits.detach().float()
    if not target_ids:
        raise ValueError("teacher-forced target is empty")
    values = []
    for token in target_ids:
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        values.append(log_probs[int(token)])
        logits, clone, _handle_id = call_forward(model, [int(token)], clone, device=device)
    mean_logp = float(torch.stack(values).mean().detach().to(device="cpu"))
    return mean_logp, initial_logits, clone


def residual_medians(model, tokenizer, device: str) -> dict[str, float]:
    """Measure Part 92 residual medians with one CPU transfer per layer hook."""
    import torch

    ids = common92.prompt_ids(tokenizer, "Who are you?", "chat")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
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
            model(input_ids=input_ids, use_cache=False)
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


def assert_equal_call_shapes(left: list[dict], right: list[dict]) -> None:
    if len(left) != len(right):
        raise AssertionError(f"twin call count mismatch: {len(left)} vs {len(right)}")
    for index, (a, b) in enumerate(zip(left, right)):
        if a["token_ids"] != b["token_ids"] or a["shape"] != b["shape"]:
            raise AssertionError(f"twin call mismatch at call {index}")

