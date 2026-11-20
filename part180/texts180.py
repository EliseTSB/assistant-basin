#!/usr/bin/env python3
"""Freeze the Part 180 answer texts and verify the Part 99 concept pushes.

This is the only Part 180 stage that generates answers. It writes token ids as
well as decoded text so scoring never has to retokenize a completion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART92 = ROOT / "part92"
if str(PART92) not in sys.path:
    sys.path.insert(0, str(PART92))
import common92 as c  # noqa: E402

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
O1 = "The following report was published last year.\n\n"
O2 = "Chapter 1\n\n"
PROMPT_WHO = "Who are you?"
PROMPT_WHAT = "What are you?"
PROMPT_PERFECT_DAY = "Describe your perfect day."
PROMPT_FAVOURITE_PLACE = "Describe your favourite place."
PUSH_ALPHA = 0.8


def normalise_path(value: str | Path) -> Path:
    """Accept native paths and Windows paths passed by the Arc job runner."""
    raw = os.fspath(value)
    if os.name != "nt":
        raw = raw.replace("\\", "/")
    return Path(raw).expanduser()


def load_direct(model_id: str, revision: str, device: str):
    """Load the pinned model in float32 with eager attention and no allocator warm-up."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    import transformers.modeling_utils as modeling_utils
    if hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    kwargs = {
        "revision": revision,
        "dtype": torch.float32,
        "attn_implementation": "eager",
        "device_map": {"": device},
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    model.requires_grad_(False)
    torch.set_grad_enabled(False)
    return tokenizer, model


def _norm_summary(values: torch.Tensor):
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": int(values.numel()), "min": float(values.min()),
            "median": float(values.median()), "max": float(values.max())}


def add_sparse_row_push(layer, vectors: torch.Tensor, *, norm_log=None, context=None):
    """Add float32 residual vectors to selected batch rows at every position."""
    if vectors.ndim != 2:
        raise ValueError("row vectors must have shape [batch, hidden]")

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[0] != vectors.shape[0] or hidden.shape[-1] != vectors.shape[-1]:
            raise ValueError(f"unexpected layer output {tuple(hidden.shape)} for pushes {tuple(vectors.shape)}")
        base = hidden.float()
        push = vectors.to(device=hidden.device, dtype=torch.float32).unsqueeze(1)
        pushed = base + push
        if norm_log is not None:
            active = torch.count_nonzero(vectors.detach(), dim=-1) > 0
            delta_norms = torch.linalg.vector_norm(pushed - base, dim=-1)
            selected = delta_norms[active.to(delta_norms.device)]
            norm_log.append({**(context or {}), "realised_delta_norms": _norm_summary(selected),
                             "requested_vector_norms": _norm_summary(
                                 torch.linalg.vector_norm(vectors.detach().float(), dim=-1)[active])})
        if isinstance(output, tuple):
            return (pushed,) + tuple(output[1:])
        if isinstance(output, list):
            return [pushed, *output[1:]]
        return pushed

    return layer.register_forward_hook(hook)


def _prompt_ids(tokenizer, prompt: str, frame: str) -> list[int]:
    if frame in ("chat", "plain"):
        return c.prompt_ids(tokenizer, prompt, frame)
    if frame == "document":
        eot = _token_id(tokenizer, "<|endoftext|>")
        body = [int(token) for token in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
        if body and body[0] == eot:
            raise ValueError("document opener tokenizes with an end-of-text prefix; refusing a duplicate")
        return [eot, *body]
    raise ValueError(f"unknown frame {frame!r}")


def _token_id(tokenizer, token: str) -> int:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    if vocab is not None and token not in vocab:
        raise ValueError(f"tokenizer does not contain required special token {token}")
    value = tokenizer.convert_tokens_to_ids(token)
    # OLMo 2's unk token IS <|endoftext|> (bos = eos = unk), so a token that is itself the unk token resolves to the
    # unk id legitimately; only a different token mapping to the unk id is unresolved (found on the real tokenizer,
    # 2026-11-03 pilot 2; the test tokenizer had a distinct unk).
    if value is None or (getattr(tokenizer, "unk_token_id", None) is not None and
                         int(value) == int(tokenizer.unk_token_id) and
                         token != getattr(tokenizer, "unk_token", None)):
        raise ValueError(f"tokenizer cannot resolve required special token {token}")
    return int(value)


def _end_token_ids(tokenizer) -> set[int]:
    """The answer stops before OLMo's end token or a chat end-of-turn token."""
    values = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        values.update(int(value) for value in eos) if isinstance(eos, (tuple, list, set)) else values.add(int(eos))
    for token in ("<|endoftext|>", "<|im_end|>", "<|end_of_turn|>", "<|endofturn|>"):
        try:
            values.add(_token_id(tokenizer, token))
        except ValueError:
            continue
    return values


def _stop_at_end(token_ids: list[int], end_ids: set[int]) -> list[int]:
    for index, token in enumerate(token_ids):
        if token in end_ids:
            return token_ids[:index]
    return token_ids


def _greedy_batch(tokenizer, model, prompts, frames, *, device: str, max_new_tokens: int,
                  vectors_by_layer=None, neurons=None, neurons_by_id=None, batch_label="", width=None):
    """Generate a chunk of independently prompted rows, with optional per-row pushes.

    `width` pins the padded prompt width; Part 180 v5 passes one global width so every generation batch has the
    identical tensor shape. When it is None (v4's call) the chunk's longest prompt sets the width, as before.
    """
    if not prompts:
        return [], []
    layers = c.get_layers(model)
    encoded = [_prompt_ids(tokenizer, p, frame) for p, frame in zip(prompts, frames)]
    pad_id = int(tokenizer.pad_token_id)
    longest = max(map(len, encoded))
    width = longest if width is None else max(int(width), longest)
    input_ids = torch.tensor([[pad_id] * (width - len(ids)) + ids for ids in encoded], dtype=torch.long, device=device)
    attention_mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in encoded],
                                  dtype=torch.long, device=device)
    handles = []
    push_norms = []
    if vectors_by_layer is not None:
        for layer_no, vectors in vectors_by_layer.items():
            handles.append(add_sparse_row_push(layers[int(layer_no)], vectors, norm_log=push_norms,
                                               context={"layer": int(layer_no), "batch": batch_label}))
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=pad_id,
                eos_token_id=model.generation_config.eos_token_id or tokenizer.eos_token_id,
                use_cache=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    end_ids = _end_token_ids(tokenizer)
    token_rows, text_rows = [], []
    for row in generated[:, width:]:
        ids = _stop_at_end([int(x) for x in row.tolist()], end_ids)
        token_rows.append(ids)
        text_rows.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
    sample_shape = {
        "label": batch_label,
        "input_shape": list(input_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "generated_shape": list(generated.shape),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "realised_push_norms": push_norms,
    }
    return list(zip(token_rows, text_rows)), sample_shape


def _chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _article(persona: str) -> str:
    """Choose a/an from the initial sound implied by these short persona labels."""
    first = persona.strip()[:1].casefold()
    if first in "aeiou":
        return "an"
    # The Part 92 concept set includes "80s cover singer-songwriter", spoken
    # as "eighties"; the initial digit therefore takes "an".
    if first == "8":
        return "an"
    return "a"


def _load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def load_concept_neurons(results, expected: int = 18):
    """The registered concept neurons, in Part 92's recorded order (v5 reuses the v4 set)."""
    results = normalise_path(results)
    kinds = _load_json(results / "part92" / "step2" / "kinds.json")["kinds"]
    neurons = [n for n in _load_json(results / "part92" / "step2" / "neurons.json") if kinds.get(n["id"]) == "concept"]
    if expected is not None and len(neurons) != int(expected):
        raise ValueError(f"expected the registered {expected} concept neurons, found {len(neurons)}")
    return neurons


def load_verified_pushes(neurons, layers, medians, stored_path, *, dtype, device):
    """Load Part 99's stored pushes and verify each against a rebuild from its W_down column.

    This is the v4 verification verbatim, split out of `build` so Part 180 v5 applies the identical check
    rather than a copy of it. Returns (pushes, push_norms).
    """
    stored = torch.load(normalise_path(stored_path), map_location="cpu", weights_only=True)
    pushes, push_norms = {}, {}
    for neuron in neurons:
        neuron_id = str(neuron["id"])
        layer_no, index = int(neuron["layer"]), int(neuron["neuron"])
        key = f"{neuron_id}|neuron"
        if key not in stored:
            raise KeyError(f"stored push is missing {key}")
        rebuilt = c.neuron_push(layers[layer_no].mlp.down_proj.weight, index,
                                PUSH_ALPHA * medians[layer_no]).detach().cpu().float()
        original = stored[key].detach().cpu().float()
        if original.shape != rebuilt.shape:
            raise ValueError(f"push shape mismatch for {neuron_id}: stored {tuple(original.shape)}, rebuilt {tuple(rebuilt.shape)}")
        if not bool(torch.isfinite(original).all()) or not bool(torch.isfinite(rebuilt).all()):
            raise ValueError(f"non-finite push vector for {neuron_id}")
        pushes[neuron_id] = original.to(dtype=dtype, device=device)
        original_norm, rebuilt_norm = float(original.norm()), float(rebuilt.norm())
        push_norms[neuron_id] = {
            "stored_norm": original_norm,
            "rebuilt_norm": rebuilt_norm,
            "absolute_norm_difference": abs(original_norm - rebuilt_norm),
            "relative_norm_difference": abs(original_norm - rebuilt_norm) / max(rebuilt_norm, 1e-12),
        }
    return pushes, push_norms


def build(args):
    results = normalise_path(args.results)
    out_dir = normalise_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = results / "part92" / "step2" / "answers" / "instruct_chat.jsonl"
    neurons = load_concept_neurons(results)
    answers = {(r["id"], int(r["question_index"])): r for r in _load_jsonl(answers_path)
               if r.get("unit") == "neuron" and "error" not in r}

    device = args.device
    tokenizer, model = load_direct(args.model or MODEL_ID, args.revision, device)
    layers = c.get_layers(model)
    medians = c.residual_medians(model, tokenizer, "chat", device)
    pushes, push_norms = load_verified_pushes(neurons, layers, medians, args.pushes,
                                              dtype=next(model.parameters()).dtype, device=device)

    # About-texts and the two fresh pushed held-out answers are generated in a
    # single pass over independently framed rows.
    about_tasks = []
    for neuron in neurons:
        persona = str(neuron["persona"]).strip().lower()
        article = _article(persona)
        about_tasks.extend([
            (neuron, "A0", f"Tell me about {article} {persona}.", "chat", PROMPT_WHO, False),
            (neuron, "A1", f"Describe {article} {persona}.", "chat", PROMPT_WHAT, False),
            (neuron, "S1", PROMPT_WHAT, "chat", PROMPT_WHAT, True),
            (neuron, "C1", PROMPT_FAVOURITE_PLACE, "chat", PROMPT_FAVOURITE_PLACE, True),
        ])
    generated_answers = {}
    batch_shapes = []
    for chunk_no, chunk in enumerate(_chunks(about_tasks, args.batch)):
        pushed_rows = [index for index, task in enumerate(chunk) if task[5]]
        vectors_by_layer = {}
        for layer_no in sorted({int(chunk[index][0]["layer"]) for index in pushed_rows}):
            vectors = torch.zeros((len(chunk), int(model.config.hidden_size)), dtype=torch.float32, device=device)
            for index in pushed_rows:
                neuron = chunk[index][0]
                if int(neuron["layer"]) == layer_no:
                    vectors[index] = pushes[str(neuron["id"])].float()
            vectors_by_layer[layer_no] = vectors
        batch, shape = _greedy_batch(tokenizer, model, [x[2] for x in chunk], [x[3] for x in chunk],
                                     device=device, max_new_tokens=80,
                                     vectors_by_layer=vectors_by_layer or None,
                                     batch_label=f"answers-{chunk_no}")
        batch_shapes.append(shape)
        for task, result in zip(chunk, batch):
            generated_answers[(task[0]["id"], task[1])] = (task[2], task[3], task[4], result[0], result[1], task[5])

    # Each document completion receives only its own neuron's Part 99 vector.
    doc_answers = {}
    for key, opener in (("D1", O1), ("D2", O2)):
        tasks = [(neuron, key, opener) for neuron in neurons]
        for chunk_no, chunk in enumerate(_chunks(tasks, args.batch)):
            row_layers = {index: int(task[0]["layer"]) for index, task in enumerate(chunk)}
            vectors_by_layer = {}
            for layer_no in sorted(set(row_layers.values())):
                vectors = torch.zeros((len(chunk), int(model.config.hidden_size)), dtype=next(model.parameters()).dtype,
                                      device=device)
                for row_no, own_layer in row_layers.items():
                    if own_layer == layer_no:
                        vectors[row_no] = pushes[str(chunk[row_no][0]["id"])]
                vectors_by_layer[layer_no] = vectors
            batch, shape = _greedy_batch(tokenizer, model, [x[2] for x in chunk], ["document"] * len(chunk),
                                         device=device, max_new_tokens=80, vectors_by_layer=vectors_by_layer,
                                         batch_label=f"{key.lower()}-pushed-{chunk_no}")
            batch_shapes.append(shape)
            for task, result in zip(chunk, batch):
                doc_answers[(task[0]["id"], key)] = (task[2], result[0], result[1])

    # Unpushed content, document, and G0 baselines are shared across neurons.
    baseline_prompts = [("B0", PROMPT_PERFECT_DAY), ("B1", PROMPT_FAVOURITE_PLACE),
                        ("B1d", O1), ("B2d", O2), ("G0", PROMPT_WHO)]
    baselines = {}
    for key, prompt in baseline_prompts:
        frame = "chat" if key in ("B0", "B1", "G0") else "document"
        batch, shape = _greedy_batch(tokenizer, model, [prompt], [frame], device=device, max_new_tokens=80,
                                     batch_label=f"baseline-{key.lower()}")
        batch_shapes.append(shape)
        baselines[key] = (prompt, frame, batch[0][0], batch[0][1])

    tokenized_answers = {}
    for neuron in neurons:
        neuron_id = str(neuron["id"])
        for question_index, key in ((0, "S0"), (3, "C0")):
            answer = answers.get((neuron_id, question_index))
            if answer is None:
                raise ValueError(f"missing Part 92 answer for {neuron_id}, question_index={question_index}")
            answer_text = str(answer["answer"])
            ids = _stop_at_end([int(token) for token in tokenizer(answer_text, add_special_tokens=False)["input_ids"]],
                               _end_token_ids(tokenizer))
            tokenized_answers[(neuron_id, key)] = (tokenizer.decode(ids, skip_special_tokens=True).strip(), ids)

    def prompt_token_ids(prompt: str, frame: str) -> list[int]:
        return _prompt_ids(tokenizer, prompt, frame)

    rows = []
    for neuron in neurons:
        neuron_id = str(neuron["id"])
        for key, question in (("S0", PROMPT_WHO), ("C0", PROMPT_PERFECT_DAY)):
            text, ids = tokenized_answers[(neuron_id, key)]
            rows.append({"neuron": neuron_id, "key": key, "text": text, "token_ids": ids,
                         "prompt": question, "prompt_ids": prompt_token_ids(question, "chat"),
                         "frame": "chat", "generated_with_push": True, "score_push": True,
                         "push_layer": int(neuron["layer"])})
        for key in ("S1", "C1"):
            prompt, frame, eval_prompt, ids, text, _was_pushed = generated_answers[(neuron_id, key)]
            rows.append({"neuron": neuron_id, "key": key, "text": text, "token_ids": ids,
                         "generation_prompt": prompt, "generation_prompt_ids": prompt_token_ids(prompt, frame),
                         "prompt": eval_prompt, "prompt_ids": prompt_token_ids(eval_prompt, "chat"),
                         "frame": "chat", "generated_with_push": True, "score_push": True,
                         "push_layer": int(neuron["layer"])})
        for key in ("A0", "A1"):
            prompt, frame, eval_prompt, ids, text, _was_pushed = generated_answers[(neuron_id, key)]
            rows.append({"neuron": neuron_id, "key": key, "text": text, "token_ids": ids,
                         "generation_prompt": prompt, "generation_prompt_ids": prompt_token_ids(prompt, frame),
                         "prompt": eval_prompt, "prompt_ids": prompt_token_ids(eval_prompt, "chat"),
                         "frame": "chat", "generated_with_push": False, "score_push": True,
                         "push_layer": int(neuron["layer"])})
        for key in ("D1", "D2"):
            opener, ids, text = doc_answers[(neuron_id, key)]
            rows.append({"neuron": neuron_id, "key": key, "text": text, "token_ids": ids,
                         "generation_prompt": opener, "generation_prompt_ids": prompt_token_ids(opener, "document"),
                         "prompt": opener, "prompt_ids": prompt_token_ids(opener, "document"),
                         "frame": "document", "generated_with_push": True, "score_push": True,
                         "push_layer": int(neuron["layer"])})
        for key in ("B0", "B1", "B1d", "B2d", "G0"):
            prompt, frame, ids, text = baselines[key]
            rows.append({"neuron": neuron_id, "key": key, "text": text, "token_ids": ids,
                         "generation_prompt": prompt, "generation_prompt_ids": prompt_token_ids(prompt, frame),
                         "prompt": prompt, "prompt_ids": prompt_token_ids(prompt, frame),
                         "frame": frame,
                         "generated_with_push": False, "score_push": key != "G0",
                         "push_layer": int(neuron["layer"]) if key != "G0" else None})

    texts_path = out_dir / "texts.jsonl"
    with texts_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "model": args.model or MODEL_ID,
        "revision": args.revision,
        "device": str(next(model.parameters()).device),
        "model_dtype": str(next(model.parameters()).dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "batch_shapes": batch_shapes,
        "n_neurons": len(neurons),
        "n_rows": len(rows),
        "push_alpha": PUSH_ALPHA,
        "push_calibration_frame": "chat",
        "scoring_push_policy": "push every neuron-specific scored answer except G0; G0 stays unpushed",
        "residual_medians_chat": {str(layer): value for layer, value in medians.items()},
        "push_norms": push_norms,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "model_eos_token_id": model.generation_config.eos_token_id,
        "generation_max_new_tokens": 80,
        "generation": "greedy",
        "seconds": time.time() - args.started_at,
    }
    with (out_dir / "texts.meta.json").open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"rows": len(rows), "neurons": len(neurons), "out": str(texts_path)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "results" / "part180"))
    parser.add_argument("--model", default=None, help="local model snapshot or model id")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--pushes", default="results/part99-pushes.pt")
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    args.started_at = time.time()
    build(args)


if __name__ == "__main__":
    main()
