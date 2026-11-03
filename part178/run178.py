#!/usr/bin/env python3
"""Part 178 cached batched generation with independent per-row ordered sampling. (Design: Method, Optimization)"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

# Keep the Arc Level Zero copy engine disabled before any deferred torch import.
os.environ["UR_L0_USE_COPY_ENGINE"] = "0"
os.environ["SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE"] = "0"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # python -I excludes the script directory
from part178 import problems178  # noqa: E402

MODEL_SMALL = "Qwen/Qwen3.5-0.8B"
MODEL_LARGE = "Qwen/Qwen3.5-4B"
REVISION_SMALL = "2fc06364715b967f1860aea9cf38778875588b17"
REVISION_LARGE = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
CAP_GENERATED = 2048
TEMPERATURE = 0.6
TOP_K = 20
TOP_P = 0.95
PRESENCE_PENALTY = 1.5
QWEN_END_IDS = (248044, 248046)
SET_SPECS = {
    "G": {"model": MODEL_LARGE, "revision": REVISION_LARGE, "thinking": True, "max_new_tokens": CAP_GENERATED},
    "R8": {"model": MODEL_SMALL, "revision": REVISION_SMALL, "thinking": True, "max_new_tokens": CAP_GENERATED},
    "N4": {"model": MODEL_LARGE, "revision": REVISION_LARGE, "thinking": False, "max_new_tokens": CAP_GENERATED},
    "N8": {"model": MODEL_SMALL, "revision": REVISION_SMALL, "thinking": False, "max_new_tokens": CAP_GENERATED},
}


class NonFiniteLogitsError(FloatingPointError):
    def __init__(self, row_indices):
        self.row_indices = list(map(int, row_indices))
        super().__init__(f"non-finite kept logits in batch row(s) {self.row_indices}")


def row_seed(set_name, problem_id):
    material = json.dumps([str(set_name), str(problem_id)], ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def _normalise_end_ids(end_token_ids):
    if isinstance(end_token_ids, int):
        return (int(end_token_ids),)
    return tuple(int(value) for value in (end_token_ids or ()))


def completion_reason(token_id, generated_count, max_new_tokens, end_token_ids):
    """Return the design's stop reason; a finished row is never sampled or forwarded again. (Gate 4)"""
    if int(token_id) in _normalise_end_ids(end_token_ids):
        return "end-of-turn"
    if int(generated_count) >= int(max_new_tokens):
        return "cap"
    return None


def ordered_distribution(logits, generated_ids, *, presence_penalty=PRESENCE_PENALTY,
                         temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P):
    """Apply penalty and top-k on the logits device; finish top-p and distribution on CPU. (Gate 4)"""
    import torch

    work = logits.detach().to(dtype=torch.float32)
    if work.ndim == 1:
        work = work.unsqueeze(0)
    if work.ndim != 2:
        raise ValueError("sampler expects [batch, vocabulary] logits")
    finite = torch.isfinite(work).all(dim=1)
    if not bool(finite.all()):
        raise NonFiniteLogitsError((~finite).nonzero(as_tuple=False).flatten().tolist())
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    rows = list(generated_ids)
    if len(rows) != work.shape[0]:
        raise ValueError("one generated-id list is required per logits row")
    seen_mask = torch.zeros_like(work, dtype=torch.bool, device=work.device)
    row_indices = []
    token_indices = []
    for row_index, ids in enumerate(rows):
        valid = [int(token_id) for token_id in ids if 0 <= int(token_id) < work.shape[1]]
        row_indices.extend([row_index] * len(valid))
        token_indices.extend(valid)
    if token_indices:
        seen_rows = torch.tensor(row_indices, dtype=torch.long, device=work.device)
        seen_tokens = torch.tensor(token_indices, dtype=torch.long, device=work.device)
        seen_mask[seen_rows, seen_tokens] = True
    top_indices, top_probabilities = _filtered_topk_from_seen_mask(
        work, seen_mask, presence_penalty, temperature, top_k, top_p)
    probabilities = torch.zeros((work.shape[0], work.shape[1]), dtype=torch.float32, device="cpu")
    probabilities.scatter_(1, top_indices, top_probabilities)
    return probabilities


def _filtered_topk_from_seen_mask(logits, seen_mask, presence_penalty, temperature, top_k, top_p):
    """Return CPU ids and probabilities for survivors after device-side penalty and top-k."""
    import torch

    work = logits.detach().to(dtype=torch.float32)
    if work.ndim == 1:
        work = work.unsqueeze(0)
    if work.ndim != 2 or tuple(seen_mask.shape) != tuple(work.shape) or seen_mask.device != work.device:
        raise ValueError("seen mask must match [batch, vocabulary] logits")
    finite = torch.isfinite(work).all(dim=1)
    if not bool(finite.all()):
        raise NonFiniteLogitsError((~finite).nonzero(as_tuple=False).flatten().tolist())
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    # The 20th-place tie may be broken differently by CPU and XPU topk implementations.
    # Positive temperature scaling preserves order, so top-k can precede the CPU temperature step.
    penalized = work - float(presence_penalty) * seen_mask.to(dtype=work.dtype)
    k = int(top_k)
    if k < 0:
        raise ValueError("top_k cannot be negative")
    k = min(k, int(work.shape[1]))
    if k:
        top_values, top_indices = torch.topk(penalized, k=k, dim=-1, sorted=True)
    else:
        # The registered run uses top-k 20; zero preserves the helper's unbounded-top-k API.
        top_values, top_indices = torch.sort(penalized, descending=True, dim=-1)
    # Only the (batch, k) survivors cross to CPU; temperature and top-p follow there.
    top_values = top_values.to(device="cpu", dtype=torch.float32) / float(temperature)
    top_indices = top_indices.to(device="cpu", dtype=torch.long)
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if float(top_p) < 1.0:
        top_probabilities = torch.softmax(top_values, dim=-1)
        cumulative = top_probabilities.cumsum(dim=-1)
        remove = cumulative - top_probabilities > float(top_p)
        remove[:, 0] = False
        top_values = top_values.masked_fill(remove, float("-inf"))
    top_probabilities = torch.softmax(top_values, dim=-1)
    if not bool(torch.isfinite(top_probabilities).all()):
        raise FloatingPointError("non-finite probabilities after ordered CPU sampling filters")
    # Dense CPU sampling accumulates in token-id order; sort survivors to preserve that draw order.
    vocabulary_order = torch.argsort(top_indices, dim=-1)
    top_indices = top_indices.gather(1, vocabulary_order)
    top_probabilities = top_probabilities.gather(1, vocabulary_order)
    return top_indices, top_probabilities


def _distribution_from_seen_mask(logits, seen_mask, presence_penalty, temperature, top_k, top_p):
    """Return dense CPU probabilities for fixtures and distribution comparisons."""
    import torch

    work = logits.detach().to(dtype=torch.float32)
    if work.ndim == 1:
        work = work.unsqueeze(0)
    indices, survivor_probabilities = _filtered_topk_from_seen_mask(
        work, seen_mask, presence_penalty, temperature, top_k, top_p)
    probabilities = torch.zeros((work.shape[0], work.shape[1]), dtype=torch.float32, device="cpu")
    probabilities.scatter_(1, indices, survivor_probabilities)
    return probabilities


def ordered_sampler(logits, generated_ids, generators, *, presence_penalty=PRESENCE_PENALTY,
                    temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P):
    """Draw one row-local uniform and invert each CPU CDF. (Gate 4)"""
    import torch

    probabilities = ordered_distribution(logits, generated_ids, presence_penalty=presence_penalty,
                                          temperature=temperature, top_k=top_k, top_p=top_p)
    if len(generators) != probabilities.shape[0]:
        raise ValueError("one CPU generator is required per row")
    uniforms = _draw_uniforms(generators, probabilities.shape[0])
    return _inverse_cdf_dense(probabilities, uniforms)


def _draw_uniforms(generators, count):
    import torch
    if len(generators) != int(count):
        raise ValueError("one CPU generator is required per row")
    return torch.stack([torch.rand((), generator=generator, device="cpu") for generator in generators])


def _inverse_cdf_dense(probabilities, uniforms):
    import torch
    cdf = probabilities.cumsum(dim=-1)
    return torch.searchsorted(cdf.contiguous(), uniforms.to(torch.float32).reshape(-1, 1), right=False).squeeze(1).clamp_max(probabilities.shape[1] - 1)


class OrderedCpuSampler:
    """Transformers logits processor with prompt-excluded seen masks and row-local RNG."""

    def __init__(self, generators, prompt_width, end_token_ids, pad_token_id,
                 *, presence_penalty=PRESENCE_PENALTY, temperature=TEMPERATURE,
                 top_k=TOP_K, top_p=TOP_P):
        self.generators = list(generators)
        self.prompt_width = int(prompt_width)
        self.end_token_ids = _normalise_end_ids(end_token_ids)
        self.pad_token_id = int(pad_token_id)
        self.presence_penalty = float(presence_penalty)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.seen_mask = None
        self.finished = None
        self.last_uniforms = None
        self.last_tokens = None

    def __call__(self, input_ids, scores):
        import torch

        logits = scores.detach().to(dtype=torch.float32)
        batch, vocab = logits.shape
        score_device = logits.device
        self.calls = getattr(self, "calls", 0) + 1
        if score_device.type == "xpu" and self.calls % 64 == 0:
            # diag178c, 2026-11-02: releasing cached blocks every 64 steps held reserved memory at 3.8 GB against 5.2 GB
            # at batch 64 on Qwen3.5-0.8B, with no measured speed cost. It changes no number.
            torch.xpu.empty_cache()
        if batch != len(self.generators):
            raise ValueError("generate changed the sampler batch size")
        if self.seen_mask is None:
            self.seen_mask = torch.zeros((batch, vocab), dtype=torch.bool, device=score_device)
            self.finished = torch.zeros(batch, dtype=torch.bool, device="cpu")
        elif tuple(self.seen_mask.shape) != (batch, vocab) or self.seen_mask.device != score_device:
            raise ValueError("generate changed the sampler vocabulary shape")
        active = ~self.finished
        if int(input_ids.shape[1]) > self.prompt_width:
            last_on_device = input_ids[:, -1].detach().to(device=score_device, dtype=torch.long)
            rows = torch.arange(batch, device=score_device)
            active_on_device = (~self.finished).to(device=score_device)
            valid = active_on_device & (last_on_device >= 0) & (last_on_device < vocab)
            self.seen_mask[rows[valid], last_on_device[valid]] = True
            if self.end_token_ids:
                end_ids = torch.tensor(self.end_token_ids, dtype=torch.long, device=score_device)
                ended = active_on_device & torch.isin(last_on_device, end_ids)
                self.finished |= ended.to(device="cpu")
        survivor_ids, probabilities = _filtered_topk_from_seen_mask(
            logits, self.seen_mask, self.presence_penalty, self.temperature, self.top_k, self.top_p)
        uniforms = torch.zeros(batch, dtype=torch.float32, device="cpu")
        active = ~self.finished
        active_indices = active.nonzero(as_tuple=False).flatten().tolist()
        if active_indices:
            active_generators = [self.generators[index] for index in active_indices]
            drawn = _draw_uniforms(active_generators, len(active_indices))
            uniforms[active_indices] = drawn
        chosen = torch.full((batch,), self.pad_token_id, dtype=torch.long, device="cpu")
        if active_indices:
            survivor_columns = _inverse_cdf_dense(probabilities[active_indices], uniforms[active_indices])
            chosen[active_indices] = survivor_ids[active_indices].gather(
                1, survivor_columns.reshape(-1, 1)).squeeze(1)
        self.last_uniforms = uniforms.clone()
        self.last_tokens = chosen.clone()
        result = torch.full_like(scores, float("-inf"))
        result[torch.arange(batch, device=scores.device), chosen.to(device=scores.device)] = 0
        return result


def gate4_sampler_fixtures():
    """Small deterministic Gate 4 checks reused by the scorer and the tests."""
    import torch

    result = {}
    base = torch.tensor([[3.0, 2.9, 2.8, 0.0]])
    prompt_only = ordered_distribution(base, [[]], presence_penalty=1.5, temperature=1.0,
                                       top_k=4, top_p=1.0)[0]
    result["prompt_only_not_penalized"] = bool(float(prompt_only[0]) > float(prompt_only[1]))
    once = ordered_distribution(base, [[0]], presence_penalty=1.5, temperature=1.0,
                                top_k=4, top_p=1.0)
    many = ordered_distribution(base, [[0, 0, 0, 0]], presence_penalty=1.5, temperature=1.0,
                                top_k=4, top_p=1.0)
    result["seen_once_equals_repeated"] = bool(torch.equal(once, many))
    alden = ordered_distribution(base, [[0]], presence_penalty=1.5, temperature=1.0,
                                 top_k=2, top_p=1.0)[0]
    result["penalty_before_top_k_keeps_A_B"] = bool(alden[1] > 0 and alden[2] > 0 and alden[0] == 0)
    # Temperature changes the top-p boundary; top-k removes id 3, then p=.60 keeps ids 0 and 1.
    ordered = ordered_distribution(torch.tensor([[4.0, 3.0, 2.0, 1.0]]), [[]],
                                   presence_penalty=0.0, temperature=2.0, top_k=3, top_p=.60)[0]
    result["temperature_top_k_top_p_order"] = bool(ordered[3] == 0 and ordered[0] > 0 and
                                                   ordered[1] > 0 and ordered[2] == 0)
    # A per-row generator is independent of whether another row drew on earlier steps.
    seed = 812
    one = torch.Generator(device="cpu").manual_seed(seed)
    mate = torch.Generator(device="cpu").manual_seed(19)
    serial = [int(torch.multinomial(torch.tensor([.2, .3, .5]), 1, generator=one).item()) for _ in range(5)]
    two = torch.Generator(device="cpu").manual_seed(seed)
    two_mate = torch.Generator(device="cpu").manual_seed(19)
    changed_batch = []
    for step in range(5):
        changed_batch.append(int(torch.multinomial(torch.tensor([.2, .3, .5]), 1, generator=two).item()))
        if step == 0:
            _ = torch.multinomial(torch.tensor([.7, .3]), 1, generator=two_mate)  # mate ends early
    result["row_draws_batch_mate_early_finish"] = serial == changed_batch
    result["row_ends_stop_drawing"] = completion_reason(6, 1, 12, 6) == "end-of-turn"
    return {"passed": all(result.values()), "fixtures": result}


def _token_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], (tuple, list)):
        ids = ids[0]
    return [int(value) for value in ids]


def _special_id(tokenizer, token):
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None:
        return None
    unknown = getattr(tokenizer, "unk_token_id", None)
    return None if value == unknown else int(value)


def _prompt_ids(tokenizer, problem, thinking):
    user_text = str(problem) + "\n" + INSTRUCTION
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": user_text}],
                                              tokenize=False, add_generation_prompt=True,
                                              enable_thinking=bool(thinking))
    ids = _token_ids(tokenizer, rendered)
    open_id, close_id = _special_id(tokenizer, "<think>"), _special_id(tokenizer, "</think>")
    open_positions = [i for i, value in enumerate(ids) if value == open_id] if open_id is not None else []
    close_positions = [i for i, value in enumerate(ids) if value == close_id] if close_id is not None else []
    if thinking:
        valid = False
        if open_positions and (not close_positions or open_positions[-1] > close_positions[-1]):
            suffix = ids[open_positions[-1] + 1:]
            # Qwen's thinking prompt is the open-tag token followed only by its newline.
            valid = _decode_ids(tokenizer, suffix) == "\n"
    else:
        valid = False
        if open_id is not None and close_id is not None and open_positions:
            last_open = open_positions[-1]
            closes_after_open = [position for position in close_positions if position > last_open]
            if len(closes_after_open) == 1:
                close_position = closes_after_open[0]
                between = _decode_ids(tokenizer, ids[last_open + 1:close_position])
                after = _decode_ids(tokenizer, ids[close_position + 1:])
                valid = between == "\n\n" and after == "\n\n"
    payload = {"prompt_ids": ids, "prompt_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
               "prompt_ids_len": len(ids), "prompt_tail_ids": ids[-16:],
               "think_token_id": open_id, "think_close_token_id": close_id,
               "prompt_mode_valid": bool(valid), "prompt_mode": "thinking" if thinking else "non-thinking"}
    if not valid:
        raise ValueError(f"chat template did not produce the registered {payload['prompt_mode']} think block on token ids")
    return payload


def _device_of(model):
    return next(model.parameters()).device


def _decode_ids(tokenizer, ids):
    try:
        return tokenizer.decode(list(map(int, ids)), skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(list(map(int, ids)))


def _generation_config_and_eos(model, tokenizer, pad_token_id):
    """Copy the effective model generation config and resolve all registered end ids."""
    config = copy.deepcopy(getattr(model, "generation_config", None))
    if config is None:
        from transformers import GenerationConfig
        config = GenerationConfig.from_model_config(model.config)
    eos_ids = list(_normalise_end_ids(getattr(config, "eos_token_id", None)))
    tokenizer_end = _special_id(tokenizer, "<|im_end|>")
    if tokenizer_end is None:
        tokenizer_end = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_end is not None:
        eos_ids.append(int(tokenizer_end))
    vocab = int(getattr(model.config, "vocab_size", 0) or
                getattr(getattr(model.config, "text_config", None), "vocab_size", 0) or 0)
    if vocab > max(QWEN_END_IDS):
        eos_ids.extend(QWEN_END_IDS)
    eos_ids = list(dict.fromkeys(eos_ids))
    if not eos_ids:
        raise ValueError("effective generation config has no end token ids")
    config.eos_token_id = eos_ids
    config.pad_token_id = int(pad_token_id)
    return config, eos_ids


def _padded_inputs(states, prompt_width, device):
    import torch

    pad_id = int(states[0]["pad_id"])
    input_rows, mask_rows = [], []
    for state in states:
        prompt = state["prompt_ids"]
        pad = int(prompt_width) - len(prompt)
        input_rows.append([pad_id] * pad + list(prompt))
        mask_rows.append([0] * pad + [1] * len(prompt))
    return {"input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long, device=device)}


def _generate_batch(model, states, prompt_width, device, max_new_tokens, *, sampler=True,
                    max_cap=None):
    """Use Transformers' own hybrid-cache aware generate loop with last-position logits."""
    import torch
    from transformers import LogitsProcessorList

    inputs = _padded_inputs(states, prompt_width, device)
    eos_ids = states[0]["end_ids"]
    pad_id = states[0]["pad_id"]
    effective_generation_config, eos_ids = _generation_config_and_eos(model, states[0]["tokenizer"], pad_id)
    effective_generation_config.do_sample = False
    effective_generation_config.max_new_tokens = int(max_new_tokens)
    model.generation_config = effective_generation_config
    for state in states:
        state["end_ids"] = eos_ids
    logits_processors = LogitsProcessorList()
    processor = None
    if sampler:
        processor = OrderedCpuSampler([state["generator"] for state in states], prompt_width,
                                      eos_ids, pad_id)
        logits_processors.append(processor)
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=eos_ids,
            pad_token_id=pad_id,
            logits_processor=logits_processors,
            logits_to_keep=1,
        )
    new_ids = sequences[:, int(prompt_width):].detach().to(device="cpu", dtype=torch.long).tolist()
    output_rows = []
    for state, sequence in zip(states, new_ids):
        kept = []
        ended = None
        for token_id in sequence:
            token_id = int(token_id)
            if ended is not None:
                break
            kept.append(token_id)
            if token_id in eos_ids:
                ended = token_id
        if max_cap is not None and len(kept) > int(max_cap):
            kept = kept[:int(max_cap)]
            ended = kept[-1] if kept and kept[-1] in eos_ids else None
        state["generated_ids"] = kept
        state["ended_token_id"] = ended
        state["finish_reason"] = "end-of-turn" if ended is not None else "cap"
        close_id = state.get("think_close_token_id")
        thinking_ids = kept
        if close_id is not None and int(close_id) in kept:
            thinking_ids = kept[:kept.index(int(close_id))]
        state["thinking_text_empty"] = bool(state["thinking"] and not _decode_ids(state["tokenizer"], thinking_ids).strip())
        output_rows.append(state)
    return output_rows


def load_direct(model_id, revision, device="xpu"):
    """Direct bf16 load, matching Part 115b's device_map loader. (Design: Optimization)"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    import transformers.modeling_utils as modeling_utils
    if hasattr(modeling_utils, "caching_allocator_warmup"):
        # Avoid the full-model temporary allocation that exceeds the A770 block limit.
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    kwargs = {"revision": revision, "dtype": torch.bfloat16, "device_map": {"": device}}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    model.requires_grad_(False)
    torch.set_grad_enabled(False)
    return tokenizer, model


def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    nonblank = [index for index, line in enumerate(lines) if line.strip()]
    final_record_line = nonblank[-1] if nonblank else -1
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index != final_record_line or any(item.strip() for item in lines[index + 1:]):
                raise ValueError(f"unparseable non-final JSONL line {index + 1} in {path}") from exc
            warnings.warn(f"dropping unparseable final JSONL line from {path}; resuming from complete rows",
                          RuntimeWarning)
            path.write_bytes(b"".join(lines[:index]))
            break
    return rows


def _write_batch(path, records):
    records = list(records)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = False
    if Path(path).is_file() and Path(path).stat().st_size:
        with Path(path).open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_separator = existing.read(1) not in (b"\n", b"\r")
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        if needs_separator:
            handle.write("\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _valid_completion(row, expected_hash):
    return (str(row.get("prompt_ids_sha256")) == expected_hash and
            row.get("finish_reason") in ("end-of-turn", "cap") and
            isinstance(row.get("generated_ids"), list))


def _model_text(tokenizer, ids, end_token_id):
    visible = list(ids)
    if visible and visible[-1] in _normalise_end_ids(end_token_id):
        visible = visible[:-1]
    return _decode_ids(tokenizer, visible)


def _make_record(state, tokenizer, set_name, spec, finish_reason, error=None):
    ids = list(map(int, state["generated_ids"]))
    row = {"part": 178, "set": set_name, "model": spec["model"], "revision": spec["revision"],
           "thinking": bool(spec["thinking"]), "problem_id": str(state["problem_id"]),
           "generated_ids": ids, "generated_text": _model_text(tokenizer, ids, state["end_ids"]),
           "finish_reason": str(finish_reason), "prompt_ids_sha256": state["prompt_ids_sha256"],
           "prompt_ids_len": int(state["prompt_ids_len"]), "prompt_tail_ids": state["prompt_tail_ids"],
           "think_token_id": state["think_token_id"], "think_close_token_id": state["think_close_token_id"],
           "end_token_ids": list(state["end_ids"]), "end_token_id": state.get("ended_token_id"),
           "prompt_mode_valid": bool(state["prompt_mode_valid"]),
           "generated_has_think_tag": bool(any(token_id is not None and int(token_id) in ids
                                                for token_id in (state["think_token_id"],
                                                                 state["think_close_token_id"]))),
           "thinking_text_empty": state.get("thinking_text_empty"),
           "max_new_tokens": int(spec["max_new_tokens"]), "temperature": TEMPERATURE,
           "top_k": TOP_K, "top_p": TOP_P, "presence_penalty": PRESENCE_PENALTY,
           "sampler_seed": row_seed(set_name, state["problem_id"])}
    if error:
        row["error"] = str(error)
    return row


def run_set(set_name, problem_rows, out_path, batch=8, *, tokenizer=None, model=None,
            loader=None, device="xpu", spec=None, max_new_tokens=None, progress=None):
    """Generate and append complete fixed batches through Transformers' hybrid cache."""
    import torch

    if set_name not in SET_SPECS and spec is None:
        raise ValueError(f"unknown set {set_name}")
    spec = dict(spec or SET_SPECS[set_name])
    if max_new_tokens is not None:
        spec["max_new_tokens"] = int(max_new_tokens)
    if int(batch) < 1:
        raise ValueError("batch size must be positive")
    if model is None or tokenizer is None:
        if loader is None:
            tokenizer, model = load_direct(spec["model"], spec["revision"], device)
        else:
            try:
                tokenizer, model = loader(spec["model"], spec["revision"], device)
            except TypeError:
                tokenizer, model = loader(device)
    model.eval()
    model.requires_grad_(False)
    device = _device_of(model)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    effective_generation_config, end_ids = _generation_config_and_eos(model, tokenizer, int(pad_id))
    rows = list(problem_rows)
    states = []
    for row in rows:
        problem_id = str(row.get("problem_id", row.get("id", "")))
        if not problem_id:
            raise ValueError("problem row has no problem_id")
        mode = _prompt_ids(tokenizer, str(row["problem"]), bool(spec["thinking"]))
        prompt_ids = mode.pop("prompt_ids")
        states.append({"problem_id": problem_id, "problem": str(row["problem"]), **mode,
                       "prompt_ids": prompt_ids, "generated_ids": [],
                       "generator": torch.Generator(device="cpu").manual_seed(row_seed(set_name, problem_id)),
                       "end_ids": list(end_ids), "pad_id": int(pad_id), "tokenizer": tokenizer,
                       "thinking": bool(spec["thinking"]), "ended_token_id": None})
    existing_rows = _read_jsonl(out_path)
    by_id = {}
    for row in existing_rows:
        if str(row.get("set")) != str(set_name):
            raise ValueError(f"output file {out_path} includes a row for another set")
        by_id[str(row.get("problem_id"))] = row
    for state in states:
        prior = by_id.get(state["problem_id"])
        if prior and prior.get("prompt_ids_sha256") not in (None, state["prompt_ids_sha256"]):
            raise ValueError(f"resume prompt fingerprint changed for {state['problem_id']}")
    queue = [state for state in states
             if not _valid_completion(by_id.get(state["problem_id"], {}), state["prompt_ids_sha256"])]
    complete_prior = [row for row in by_id.values() if row.get("finish_reason") in ("end-of-turn", "cap")]
    if not queue:
        no_generated_think = bool(spec["thinking"]) or all(not row.get("generated_has_think_tag")
                                                              for row in complete_prior)
        return {"set": set_name, "written": 0, "resumed": len(states), "tokens": 0,
                "seconds": 0.0, "tokens_per_second": None,
                "mode_validity": {"prompt_ids": all(row.get("prompt_mode_valid") for row in complete_prior),
                                  "no_nonthinking_think_tag": no_generated_think},
                "empty_thinking_share": _empty_thinking_share(complete_prior, spec["thinking"]),
                "peak_memory_bytes": _peak_memory(model)}
    prompt_width = max(len(state["prompt_ids"]) for state in states)
    try:
        if str(device).startswith("xpu"):
            torch.xpu.reset_peak_memory_stats()
    except Exception:
        pass
    started, total_tokens, finished_count = time.perf_counter(), 0, 0
    for offset in range(0, len(queue), int(batch)):
        batch_states = queue[offset:offset + int(batch)]
        generate_states = list(batch_states)
        if len(batch_states) == 1:
            # diag178d and diag178e, 2026-11-02: on the Arc, Qwen3.5 greedy tokens at batch 1 diverged from a CPU
            # float32 reference (0.8B) and from batches 2 to 16 (4B), which all agreed. A lone row therefore runs
            # beside a discarded companion with its own generator, so no row is generated at batch 1.
            import torch
            companion = dict(batch_states[0])
            companion["generator"] = torch.Generator(device="cpu").manual_seed(0)
            companion["companion"] = True
            generate_states.append(companion)
        _generate_batch(model, generate_states, prompt_width, device, spec["max_new_tokens"])
        records = [_make_record(state, tokenizer, set_name, spec, state["finish_reason"])
                   for state in batch_states]
        _write_batch(out_path, records)
        finished_count += len(records)
        total_tokens += sum(len(state["generated_ids"]) for state in batch_states)
        if progress is not None:
            progress({"finished": finished_count, "active": 0, "tokens": total_tokens,
                      "batches": offset // int(batch) + 1})
    seconds = max(time.perf_counter() - started, 1e-9)
    completed_rows = [row for row in _read_jsonl(out_path) if row.get("set") == set_name and
                      row.get("finish_reason") in ("end-of-turn", "cap")]
    no_generated_think = bool(spec["thinking"]) or all(not row.get("generated_has_think_tag") for row in completed_rows)
    return {"set": set_name, "written": finished_count, "resumed": len(states) - finished_count,
            "tokens": total_tokens, "seconds": seconds, "tokens_per_second": total_tokens / seconds,
            "mode_validity": {"prompt_ids": all(state["prompt_mode_valid"] for state in states),
                              "no_nonthinking_think_tag": no_generated_think},
            "empty_thinking_share": _empty_thinking_share(completed_rows, spec["thinking"]),
            "peak_memory_bytes": _peak_memory(model)}


def _empty_thinking_share(rows, thinking_mode):
    if not thinking_mode:
        return None
    rows = list(rows)
    if not rows:
        return None
    return sum(bool(row.get("thinking_text_empty")) for row in rows) / len(rows)


def _peak_memory(model):
    try:
        import torch
        if str(next(model.parameters()).device).startswith("xpu"):
            return int(torch.xpu.max_memory_allocated())
    except Exception:
        return None
    return None


def _pilot_memory(device):
    """Read the XPU peaks and current reserved bytes for one pilot measurement."""
    if not str(device).startswith("xpu"):
        return {"peak_memory_allocated_bytes": None, "memory_reserved_bytes": None,
                "peak_memory_reserved_bytes": None}
    try:
        import torch
        values = {}
        for field, getter in (("peak_memory_allocated_bytes", torch.xpu.max_memory_allocated),
                              ("memory_reserved_bytes", torch.xpu.memory_reserved),
                              ("peak_memory_reserved_bytes", getattr(torch.xpu, "max_memory_reserved", None))):
            try:
                values[field] = int(getter()) if getter is not None else None
            except Exception:
                values[field] = None
        return values
    except Exception:
        return {"peak_memory_allocated_bytes": None, "memory_reserved_bytes": None,
                "peak_memory_reserved_bytes": None}


def _xpu_free_bytes(device):
    if not str(device).startswith("xpu"):
        return None
    try:
        import torch
        free, _total = torch.xpu.mem_get_info()
        return int(free)
    except Exception:
        return None


def _load_problem_rows(path, pilot=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run python -I -B part178\\problems178.py --check first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["pilot"] if pilot else payload["problems"])


def _pilot_plan(value):
    plan = []
    for item in (part.strip() for part in str(value).split(",")):
        if not item:
            continue
        try:
            set_name, batch_value = item.split(":", 1)
            batch = int(batch_value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid pilot plan item {item!r}; expected SET:BATCH") from exc
        if set_name not in SET_SPECS or batch < 1:
            raise ValueError(f"invalid pilot plan item {item!r}")
        plan.append((set_name, batch))
    if not plan:
        raise ValueError("pilot plan must contain at least one SET:BATCH item")
    return plan


def _pilot_rows_for_batch(rows, batch_size):
    """Fill a pilot batch with copies whose repeat index changes each row seed."""
    rows = list(rows)
    if not rows:
        raise ValueError("pilot rows cannot be empty")
    expanded = []
    for index in range(int(batch_size)):
        source = rows[index % len(rows)]
        repeat_index = index // len(rows)
        row = dict(source)
        original_id = str(source.get("problem_id", source.get("id", "")))
        if not original_id:
            raise ValueError("pilot row has no problem_id")
        row["problem_id"] = f"{original_id}::repeat-{repeat_index}"
        expanded.append(row)
    return expanded


def _greedy_batch_check(model, tokenizer, states, prompt_width, device, pad_id):
    """Compare 32 greedy tokens for four rows alone and inside the batch of sixteen."""
    singles = []
    for state in states[:4]:
        single = dict(state)
        _generate_batch(model, [single], prompt_width, device, 32, sampler=False)
        singles.append(single["generated_ids"])
    batched = [dict(state) for state in states[:16]]
    _generate_batch(model, batched, prompt_width, device, 32, sampler=False)
    matching = 0
    for left, right_state in zip(singles, batched[:4]):
        right = right_state["generated_ids"]
        left = list(left[:32]) + [int(pad_id)] * max(0, 32 - len(left))
        right = list(right[:32]) + [int(pad_id)] * max(0, 32 - len(right))
        matching += sum(int(a == b) for a, b in zip(left, right))
    return {"rows": 4, "tokens_per_row": 32, "matching_token_count": matching,
            "total_token_count": 128, "matched": matching == 128}


def _clear_device_cache(device):
    if not str(device).startswith("xpu"):
        return
    try:
        import torch
        torch.xpu.empty_cache()
    except Exception:
        pass


def _run_pilot(args, loader, config):
    pilot_root = Path(config.get("pilot_root", ROOT / "results" / "part178" / "pilot"))
    pilot_output = Path(config.get("pilot_output", pilot_root / "pilot.jsonl"))
    pilot_output.parent.mkdir(parents=True, exist_ok=True)
    pilot_output.write_text("", encoding="utf-8")
    rows = list(config.get("problem_rows") or _load_problem_rows(args.problems, pilot=True))[:16]
    if len(rows) != 16:
        raise ValueError(f"pilot requires 16 held-out rows, found {len(rows)}")
    plan = _pilot_plan(getattr(args, "plan", "R8:64,N8:64,G:16,G:32,N4:32"))
    by_set = {}
    for set_name, batch_size in plan:
        by_set.setdefault(set_name, []).append(batch_size)
    results = []
    stop_required = False
    for set_name in by_set:
        spec = dict(SET_SPECS[set_name])
        if "max_new_tokens" in config:
            spec["max_new_tokens"] = int(config["max_new_tokens"])
        tokenizer, model = loader(spec["model"], spec["revision"], args.device) if loader else \
            load_direct(spec["model"], spec["revision"], args.device)
        states = []
        try:
            import torch
            if hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = "left"
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", 0)
            _generation_config, end_ids = _generation_config_and_eos(model, tokenizer, int(pad_id))
            for row in rows:
                problem_id = str(row.get("problem_id", row.get("id", "")))
                mode = _prompt_ids(tokenizer, str(row["problem"]), bool(spec["thinking"]))
                prompt_ids = mode.pop("prompt_ids")
                states.append({"problem_id": problem_id, "problem": str(row["problem"]), **mode,
                               "prompt_ids": prompt_ids,
                               "generator": torch.Generator(device="cpu").manual_seed(row_seed(set_name, problem_id)),
                               "end_ids": end_ids, "pad_id": int(pad_id), "tokenizer": tokenizer,
                               "thinking": bool(spec["thinking"]), "ended_token_id": None})
            prompt_width = max(len(state["prompt_ids"]) for state in states)
            for batch_size in by_set[set_name]:
                path = pilot_root / f"{set_name}-batch{batch_size}.jsonl"
                if str(args.device).startswith("xpu"):
                    torch.xpu.reset_peak_memory_stats()
                batch_rows = _pilot_rows_for_batch(rows, batch_size)
                run_result = run_set(set_name, batch_rows, path, batch_size, tokenizer=tokenizer,
                                     model=model, spec=spec, device=args.device)
                result = {"set": set_name, "batch": batch_size,
                          "tokens_per_second": run_result.get("tokens_per_second"),
                          **_pilot_memory(args.device),
                          "mode_validity": run_result.get("mode_validity"),
                          "empty_thinking_share": run_result.get("empty_thinking_share"),
                          "thinking_empty": bool(spec["thinking"] and
                                                 run_result.get("empty_thinking_share") == 1.0),
                          "batch_check": None}
                if set_name == "G" and batch_size == 16:
                    result["batch_check"] = _greedy_batch_check(
                        model, tokenizer, states, prompt_width, args.device, pad_id)
                results.append(result)
                line = json.dumps(result, ensure_ascii=True, allow_nan=False)
                _write_batch(pilot_output, [result])
                print(line, flush=True)
                if set_name == "G" and result["thinking_empty"]:
                    stop_required = True
                    break
            if stop_required:
                # Still release and verify the current model before returning.
                pass
        finally:
            free_before = _xpu_free_bytes(args.device)
            del states
            del model
            del tokenizer
            _clear_device_cache(args.device)
            free_after = _xpu_free_bytes(args.device)
            release = {"event": "model_release_check", "set": set_name,
                       "device_free_before_bytes": free_before,
                       "device_free_after_bytes": free_after,
                       "released_bytes": (free_after - free_before)
                       if free_before is not None and free_after is not None else None,
                       "supported": free_before is not None and free_after is not None}
            release["passed"] = bool(release["supported"] and free_after > free_before)
            if not release["supported"]:
                release["reason"] = "device memory check requires an XPU run"
            _write_batch(pilot_output, [release])
            if release["supported"] and not release["passed"]:
                raise RuntimeError(f"XPU model memory was not released after pilot set {set_name}")
        if stop_required:
            break
    return {"pilot": True, "stop_required": stop_required, "results": results}


def run(args, loader=None, config=None):
    """Public orchestration entry used by the CLI and the tiny-model integration fixture."""
    config = dict(config or {})
    if args.pilot:
        return _run_pilot(args, loader, config)
    set_name = args.set
    if set_name not in SET_SPECS:
        raise ValueError("--set must be G, R8, N4 or N8")
    spec = dict(SET_SPECS[set_name])
    if "max_new_tokens" in config:
        spec["max_new_tokens"] = int(config["max_new_tokens"])
    rows = config.get("problem_rows") or _load_problem_rows(args.problems, pilot=False)
    output_path = Path(args.out) / f"{set_name}.jsonl"
    return run_set(set_name, rows, output_path, int(args.batch), loader=loader, device=args.device,
                   spec=spec, max_new_tokens=spec["max_new_tokens"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=tuple(SET_SPECS))
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "part178")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--plan", default="R8:64,N8:64,G:16,G:32,N4:32",
                        help="pilot plan, for example R8:64,N8:64,G:16,G:32,N4:32")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--problems", type=Path, default=HERE / "problems178.json")
    args = parser.parse_args(argv)
    if not args.pilot and not args.set:
        parser.error("--set is required unless --pilot is used")
    result = run(args)
    if args.pilot:
        # Each summary line was printed and synced to pilot.jsonl as its set finished.
        pass
    else:
        print(json.dumps({key: result.get(key) for key in
                          ("set", "written", "resumed", "tokens", "tokens_per_second", "mode_validity")},
                         ensure_ascii=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
